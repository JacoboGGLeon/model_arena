"""Athena execution, immutable S3 publication, and catalog registration."""
from pathlib import Path
from datetime import date,datetime,timezone
from decimal import Decimal
from io import BytesIO
from urllib.parse import urlsplit
import base64,hashlib,json,time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import literal,sha
from .schemas import SCHEMAS,athena_type


class Athena:
 def __init__(self,config,session=None,client=None):
  self.config=config
  if client is None:
   import boto3
   session=session or boto3.Session(region_name=config.region)
   client=session.client('athena')
  self.client=client
 def execute(self,sql):
  qid=self.client.start_query_execution(QueryString=sql,QueryExecutionContext={'Catalog':self.config.catalog,'Database':self.config.database},WorkGroup=self.config.workgroup,ResultConfiguration={'OutputLocation':self.config.query_output})['QueryExecutionId']
  end=time.monotonic()+self.config.query_timeout
  while True:
   q=self.client.get_query_execution(QueryExecutionId=qid)['QueryExecution'];state=q['Status']['State']
   if state=='SUCCEEDED':return dict(query_id=qid,bytes_scanned=q.get('Statistics',{}).get('DataScannedInBytes',0),sql=sql)
   if state in ('FAILED','CANCELLED'):raise RuntimeError(qid+' · '+state+' · '+q['Status'].get('StateChangeReason',''))
   if time.monotonic()>=end:
    self.client.stop_query_execution(QueryExecutionId=qid)
    raise TimeoutError('Consulta cancelada por timeout: '+qid)
   time.sleep(.4)
 def page(self,handle,token=None,size=None):
  args=dict(QueryExecutionId=handle['query_id'],MaxResults=min(1000,size or self.config.page_size))
  if token:args['NextToken']=token
  response=self.client.get_query_results(**args);result=response['ResultSet']
  columns=result.get('ResultSetMetadata',{}).get('ColumnInfo',[])
  names=[c['Name'] for c in columns];rows=result.get('Rows',[])
  if not token and rows:rows=rows[1:]
  def convert(value,type_):
   if value is None:return None
   if type_ in ('bigint','integer','int','smallint','tinyint'):return int(value)
   if type_ in ('double','float','real'):return float(value)
   if type_=='boolean':return value.lower()=='true'
   if type_.startswith('decimal'):return Decimal(value)
   if type_=='date':return date.fromisoformat(value)
   return value
  records=[]
  for row in rows:
   cells=row['Data'];records.append({c['Name']:convert(cells[i].get('VarCharValue') if i<len(cells) else None,c['Type']) for i,c in enumerate(columns)})
  return pd.DataFrame(records,columns=names),response.get('NextToken')
 def all(self,sql,max_rows=100000):
  handle=self.execute(sql);frames=[];token=None;count=0
  while True:
   frame,token=self.page(handle,token,1000);count+=len(frame)
   if count>max_rows:raise ValueError('Demasiados resultados. Aplica filtros más específicos')
   frames.append(frame)
   if not token:break
  return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(),handle


def split_s3(uri):
 parsed=urlsplit(uri)
 if parsed.scheme!='s3' or not parsed.netloc:raise ValueError('URI S3 inválida')
 return parsed.netloc,parsed.path.lstrip('/').rstrip('/')

class Publisher:
 def __init__(self,config,session=None,s3=None,athena=None,glue=None):
  self.config=config
  if session is None and (s3 is None or glue is None):
   import boto3
   session=boto3.Session(region_name=config.region)
  self.s3=s3 or session.client('s3');self.glue=glue or session.client('glue');self.athena=athena or Athena(config,session)
  self.bucket,self.prefix=split_s3(config.s3_base)
 def put_once(self,key,body,digest):
  try:
   result=self.s3.put_object(Bucket=self.bucket,Key=key,Body=body,IfNoneMatch='*',Metadata={'sha256':digest},ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode())
  except Exception as exc:
   response=getattr(exc,'response',{});code=str(response.get('Error',{}).get('Code',''))
   if code not in ('PreconditionFailed','412','ConditionalRequestConflict','409'):raise
   head=self.s3.head_object(Bucket=self.bucket,Key=key)
   if head.get('Metadata',{}).get('sha256')!=digest:raise RuntimeError('S3 ya contiene una versión distinta: '+key) from exc
  head=self.s3.head_object(Bucket=self.bucket,Key=key)
  if head.get('Metadata',{}).get('sha256')!=digest:raise RuntimeError('No se verificó la escritura: '+key)
 def register(self,name,columns,partitioned=True):
  table=self.config.table(name);uri=self.config.s3_base.rstrip('/')+('/datasets/'+name+'/' if partitioned else '/control/publications/')
  expected=[(c['name'],c['type']) for c in columns]
  try:
   existing=self.glue.get_table(DatabaseName=self.config.database,Name=self.config.table_prefix+name)['Table']
  except self.glue.exceptions.EntityNotFoundException:existing=None
  if existing:
   actual=[(c['Name'],c['Type']) for c in existing['StorageDescriptor']['Columns']]
   keys=[(k['Name'],k['Type']) for k in existing.get('PartitionKeys',[])]
   wanted=[('run_id','string'),('publication_id','string'),('segment','string')] if partitioned else []
   if actual!=expected or keys!=wanted or existing['StorageDescriptor']['Location'].rstrip('/')!=uri.rstrip('/'):
    raise ValueError('Contrato o ubicación incompatible en '+table+'; no se reemplaza automáticamente')
  definitions=', '.join('`'+n+'` '+t for n,t in expected)
  partitions=' PARTITIONED BY (`run_id` string, `publication_id` string, `segment` string)' if partitioned else ''
  self.athena.execute(f'CREATE EXTERNAL TABLE IF NOT EXISTS {table} ({definitions}){partitions} STORED AS PARQUET LOCATION {literal(uri)}')
 def publish(self,prepared,progress=print):
  folder=Path(prepared);manifest=json.loads((folder/'publication.json').read_text())
  if manifest['status']!='PREPARED':raise ValueError('Preparación incompleta')
  if manifest['shards']!=self.config.shards:raise ValueError('Config shards distinta a la preparación')
  run=manifest['run_id'];pub=manifest['publication_id'];cfg=self.config
  # Validate every local byte before any AWS mutation.
  for item in manifest['files']+manifest.get('raw_deliveries',[]):
   if sha(folder/item['path'])!=item['sha256']:raise ValueError('Archivo local modificado: '+item['path'])
  publication_columns=[dict(name=f.name,type=athena_type(f.type)) for f in SCHEMAS['publications']]
  self.register('publications',publication_columns,False)
  for name,meta in manifest['datasets'].items():self.register(name,meta['columns'])
  for item in manifest['files']+manifest.get('raw_deliveries',[]):
   p=folder/item['path']
   key=self.prefix+'/'+item['path'] if item in manifest['files'] else self.prefix+'/deliveries/'+run+'/'+pub+'/'+item['path']
   with p.open('rb') as body:self.put_once(key,body,item['sha256'])
  for name,meta in manifest['datasets'].items():
   segments=sorted({i['segment'] for i in manifest['files'] if i['dataset']==name})
   if segments:
    parts=[]
    for segment in segments:
     uri=cfg.s3_base.rstrip('/')+f'/datasets/{name}/run_id={run}/publication_id={pub}/segment={segment}/'
     parts.append(f'PARTITION (run_id={literal(run)},publication_id={literal(pub)},segment={literal(segment)}) LOCATION {literal(uri)}')
    self.athena.execute(f'ALTER TABLE {cfg.table(name)} ADD IF NOT EXISTS '+ ' '.join(parts))
   frame,_=self.athena.all(f'SELECT count(*) AS rows_count FROM {cfg.table(name)} WHERE run_id={literal(run)} AND publication_id={literal(pub)}')
   if int(frame.iloc[0]['rows_count'])!=meta['rows']:raise RuntimeError('Conteo Athena distinto al preparado: '+name)
   progress(name+' · '+str(meta['rows'])+' filas verificadas')
  manifest_bytes=json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode()
  key=self.prefix+f'/control/manifests/{pub}.json';self.put_once(key,manifest_bytes,hashlib.sha256(manifest_bytes).hexdigest())
  commit_path=folder/'commit_record.json'
  published_at=json.loads(commit_path.read_text())['published_at'] if commit_path.exists() else datetime.now(timezone.utc).isoformat()
  commit_path.write_text(json.dumps(dict(publication_id=pub,published_at=published_at)))
  record=dict(run_id=run,publication_id=pub,published_at=published_at,source_signature=manifest['source_signature'],manifest_uri=f's3://{self.bucket}/{key}',protocol=manifest['protocol'],shards=manifest['shards'])
  commit_bytes=folder/'commit_record.parquet'
  if commit_bytes.exists():
   if pq.read_table(commit_bytes).to_pylist()!=[record]:raise ValueError('El registro local de publicación no corresponde a esta preparación')
   payload=commit_bytes.read_bytes()
  else:
   buffer=BytesIO();pq.write_table(pa.Table.from_pylist([record],schema=SCHEMAS['publications']),buffer,compression='zstd')
   payload=buffer.getvalue();commit_bytes.write_bytes(payload)
  # This is the commit: the monitor sees a run only after uploads/partitions/counts pass.
  self.put_once(self.prefix+f'/control/publications/{pub}.parquet',payload,hashlib.sha256(payload).hexdigest())
  found,_=self.athena.all(f'SELECT publication_id FROM {cfg.table("publications")} WHERE publication_id={literal(pub)}')
  if len(found)!=1:raise RuntimeError('No se pudo verificar el registro final de publicación')
  receipt=dict(status='PUBLISHED',run_id=run,publication_id=pub,manifest_uri=record['manifest_uri'])
  (folder/'published.json').write_text(json.dumps(receipt,indent=2));progress('Publicación completa: '+pub)
  return receipt
