from pathlib import Path
from dataclasses import dataclass,fields
import hashlib,json,re


def literal(value):
 if isinstance(value,bool):return 'true' if value else 'false'
 if isinstance(value,int):return str(value)
 if isinstance(value,float):
  import math
  if not math.isfinite(value):raise ValueError('Literal numérico no finito')
  return repr(value)
 return "'"+str(value).replace("'","''")+"'"
def identifier(value):
 if not re.fullmatch(r'[a-z][a-z0-9_]*',value):raise ValueError('Identificador SQL inválido: '+value)
 return value

def shard(account,count):return str(int(hashlib.sha256(account.encode()).hexdigest()[:16],16)%count)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def source_signature(paths):
 return hashlib.sha256(json.dumps([(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(set(paths))]).encode()).hexdigest()

@dataclass(frozen=True)
class Config:
 region:str='us-east-1'
 database:str='observatorio_cuentas'
 catalog:str='AwsDataCatalog'
 workgroup:str='sandbox'
 s3_base:str=''
 query_output:str=''
 table_prefix:str='obs_'
 shards:int=16
 page_size:int=50
 query_timeout:int=180
 delivery_load_date:str|None=None
 display_timezone:str='America/Mexico_City'
 initial_publication_id:str|None=None
 @classmethod
 def read(cls,path):
  data=json.loads(Path(path).read_text());obj=cls(**data)
  identifier(obj.database);identifier(obj.table_prefix+'inventory')
  if not obj.s3_base.startswith('s3://') or not obj.query_output.startswith('s3://'):raise ValueError('Configura las ubicaciones S3')
  if not 1<=obj.shards<=128 or not 1<=obj.page_size<=1000:raise ValueError('shards/page_size inválidos')
  return obj
 def table(self,name):return self.database+'.'+identifier(self.table_prefix+name)
