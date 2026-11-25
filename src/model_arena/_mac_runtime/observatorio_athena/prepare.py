"""One-time local preparation from a completed 0.11 run. No AWS and no training."""
from pathlib import Path
from datetime import date,datetime,timezone
from decimal import Decimal,ROUND_HALF_UP,InvalidOperation
import base64,hashlib,json,math,shutil,uuid
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from model_arena.disk import check_space
from model_arena.loss_store import detail_files,read_batches
from .common import sha,source_signature,shard
from .schemas import SCHEMAS,ENGINE2_COLUMNS,athena_type
from .historical_sigma import expanding_sigma
from .observed_quantiles import trailing_quantiles
from .alert_taxonomy import evaluate
from .mac_rules_reference import OriginalMacRules,OFFICIAL_COLUMNS,REFERENCE_SHA256


def day(v):return v if isinstance(v,date) and not isinstance(v,datetime) else date.fromisoformat(str(v)[:10])
def clean(v):
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

class Writer:
 def __init__(self,folder,run_id,pub_id,shards):
  self.folder=Path(folder);self.run_id=run_id;self.pub_id=pub_id;self.shards=shards;self.entries=[];self.seq={};self.buffers={};self.buffer_bytes={}
 def write(self,name,rows,segment='all'):
  if isinstance(rows,pl.DataFrame):rows=rows.to_dicts()
  if not rows:return
  rows=[{k:clean(v) for k,v in r.items()} for r in rows]
  schema=SCHEMAS[name];table=pa.Table.from_pylist(rows,schema=schema)
  key=(name,segment)
  self.buffers.setdefault(key,[]).append(table)
  self.buffer_bytes[key]=self.buffer_bytes.get(key,0)+table.nbytes
  if self.buffer_bytes[key]>=32*1024*1024:self.flush_key(key)
 def flush_key(self,key):
  name,segment=key
  if not self.buffers.get(key):return
  table=pa.concat_tables(self.buffers.pop(key));self.buffer_bytes.pop(key,None)
  index=self.seq.get(key,0);self.seq[key]=index+1
  relative=Path(name)/f'run_id={self.run_id}'/f'publication_id={self.pub_id}'/f'segment={segment}'/f'part-{index:05d}.parquet'
  p=self.folder/'datasets'/relative;p.parent.mkdir(parents=True,exist_ok=True)
  check_space(p.parent,extra_bytes=table.nbytes*2,stage='publicación '+name)
  pq.write_table(table,p,compression='zstd',row_group_size=65536)
  self.entries.append(dict(dataset=name,segment=segment,path=str(p.relative_to(self.folder)),rows=table.num_rows,sha256=sha(p)))
 def flush(self):
  for key in list(self.buffers):self.flush_key(key)

 def by_account(self,name,rows):
  groups={}
  for r in rows:groups.setdefault(shard(r['account_id'],self.shards),[]).append(r)
  for s,rr in groups.items():self.write(name,rr,s)


def inclusive_std(values,skip_first=False):
 out=[];n=0;mean=m2=0.
 for i,x in enumerate(values):
  if not(skip_first and i==0):
   n+=1;delta=x-mean;mean+=delta/n;m2+=delta*(x-mean)
  out.append(math.sqrt(max(0,m2/(n-1))) if n>1 else 0.)
 return out


def source_groups(source,scratch):
 # A disk-backed sorted intermediate gives complete series across Arrow batches.
 pl.scan_parquet(source).sort('account_id','date').sink_parquet(scratch,compression='zstd')
 pending=None
 for batch in pq.ParquetFile(scratch).iter_batches(batch_size=65536):
  chunk=pl.from_arrow(batch)
  if pending is not None:chunk=pl.concat([pending,chunk],how='vertical_relaxed')
  last=chunk['account_id'][-1]
  ready=chunk.filter(pl.col('account_id')!=last)
  pending=chunk.filter(pl.col('account_id')==last)
  for _,group in ready.group_by('account_id',maintain_order=True):yield group.sort('date')
 if pending is not None and pending.height:yield pending.sort('date')


def read_prediction(path,inventory):
 frame=pl.read_parquet(path)
 methods=[x for x in frame['method'].unique().to_list() if x!='Persistencia']
 if len(methods)!=1:raise ValueError(f'{path}: se esperaba un modelo más Persistencia')
 keys=['account_id','currency','date','target','fold']
 def family(method,prefix):
  f=frame.filter(pl.col('method')==method)
  if f.select(*keys,'component').is_duplicated().any():raise ValueError('Cuantiles duplicados: '+str(path))
  w=f.pivot(on='component',index=keys,values='forecast')
  if not {'Q2.5','Q50','Q97.5'}<=set(w.columns):raise ValueError('Cuantiles incompletos: '+str(path))
  w=w.rename({'Q2.5':prefix+'_q025','Q50':prefix+'_q50','Q97.5':prefix+'_q975'})
  info=f.filter(pl.col('component')=='Q50').select(*keys,'observed','scale','origin','input',*[c for c in ['candidate_model'] if c in f.columns])
  if prefix=='model':
   info=info.with_columns(pl.col('candidate_model').alias('candidate') if 'candidate_model' in info.columns else pl.lit(method).alias('candidate')).drop('candidate_model',strict=False)
  else:info=info.select(*keys)
  return w.join(info,on=keys,validate='1:1')
 model=family(methods[0],'model');base=family('Persistencia','baseline')
 if model.height!=base.height:raise ValueError('Baseline y modelo tienen distinta población: '+str(path))
 paired=model.join(base,on=keys,validate='1:1')
 if paired.height!=model.height:raise ValueError('Baseline y modelo no están emparejados: '+str(path))
 numbers=['scale','model_q025','model_q50','model_q975','baseline_q025','baseline_q50','baseline_q975']
 if paired.filter(~pl.all_horizontal(pl.col(c).is_not_null() & pl.col(c).is_finite() for c in numbers)).height:raise ValueError('Predicción no finita')
 if paired.filter((pl.col('scale')<=0)|(pl.col('model_q025')>pl.col('model_q50'))|(pl.col('model_q50')>pl.col('model_q975'))|(pl.col('baseline_q025')>pl.col('baseline_q50'))|(pl.col('baseline_q50')>pl.col('baseline_q975'))).height:raise ValueError('Escala o cuantiles inválidos')
 if set(paired['target'])!={'saldo','variacion'} or paired.group_by('account_id','date','fold').agg(pl.col('target').n_unique().alias('n')).filter(pl.col('n')!=2).height:raise ValueError('Saldo y variación deben compartir las mismas cuentas y fechas')
 return paired.join(inventory.select('account_id','budget','cohort','taxonomy','sign'),on='account_id',validate='m:1')


def metric_rows(paired,comparison,role):
 output=[]
 paired=paired.filter(pl.col('observed').is_not_null() & pl.col('observed').is_finite())
 for prefix,kind in [('model',role),('baseline','Baseline')]:
  for keys,g in paired.group_by('account_id','currency','cohort','target','fold','input','budget',maintain_order=True):
   aid,ccy,cohort,target,fold,inp,budget=keys
   y=g['observed'].to_numpy();p=g[prefix+'_q50'].to_numpy();e=np.abs(y-p);den=np.abs(y)+np.abs(p);nz=np.abs(y)>0
   scale=g['scale'].to_numpy();lo=g[prefix+'_q025'].to_numpy();hi=g[prefix+'_q975'].to_numpy()
   dates=sorted(str(d) for d in g['date']);population=hashlib.sha256(json.dumps([aid,target,fold,dates]).encode()).hexdigest()
   output.append(dict(account_id=aid,currency=ccy,cohort=cohort,taxonomy=g["taxonomy"][0],target=target,fold=fold,input=inp,budget=budget,
    comparison_id=comparison,candidate=comparison if prefix=='model' else 'Persistencia',role=kind,population_hash=population,n=len(y),mape_n=int(nz.sum()),
    ae_sum=float(e.sum()),se_sum=float(np.square(e).sum()),smape_sum=float(np.divide(200*e,den,out=np.zeros_like(e),where=den>0).sum()),
    mape_sum=float((100*e[nz]/np.abs(y[nz])).sum()),covered_sum=float(((y>=lo)&(y<=hi)).sum()),width_sum=float((hi-lo).sum()),
    ae_scaled_sum=float((e/scale).sum()),se_scaled_sum=float(np.square(e/scale).sum())))
 return output


def decimal(value,precision=24):
 if value is None or not math.isfinite(float(value)):return None
 try:q=Decimal(str(value)).quantize(Decimal('.00000001'),rounding=ROUND_HALF_UP)
 except InvalidOperation:return None
 return q if abs(q)<Decimal(10)**(precision-8) else None


def prepare(run,out,config,*,delivery_dirs=(),training_path=None,progress=print):
 run=Path(run).expanduser().resolve()
 root=run/'curvas' if (run/'curvas/resultados_x0.json').is_file() else run
 run=root.parent
 report=json.loads((root/'resultados_x0.json').read_text())
 if report.get('version')!='0.11.0':raise ValueError('Adaptador para corrida Arena 0.11.0; no convierte silenciosamente otro protocolo')
 source=run/'source/engine1.parquet';definitions=json.loads((root/'candidatos.json').read_text())
 calendar_manifest=run.parent/'calendar/provenance.json'
 calendar_provenance=json.loads(calendar_manifest.read_text()) if calendar_manifest.is_file() else None
 paths=[*([calendar_manifest] if calendar_manifest.is_file() else []),*detail_files(root/'loss_entities'),source,root/'resultados_x0.json',root/'inventory.parquet',root/'candidatos.json',*sorted((root/'predicciones').glob('*.parquet')),*sorted(root.glob('winners_*.parquet'))]
 if training_path:paths.append(Path(training_path).resolve())
 raw_roots=list(dict.fromkeys([p for p in [run/'engine2',run/'engine3',root/'engine2',root/'engine3',*[Path(p).expanduser().resolve() for p in delivery_dirs]] if p.is_dir()]))
 raw_files=[(i,p,base) for i,base in enumerate(raw_roots) for p in sorted(base.rglob('*')) if p.is_file()]
 mac3_metadata=[p for pattern in ('inventory.parquet','scores.parquet','candidatos.json','resultados_x0.json','winners_*.parquet','identidad.json','hpo_trials.parquet') for p in sorted(root.glob(pattern))]
 raw_files.extend((len(raw_roots),p,root) for p in mac3_metadata)
 paths.extend(p for _,p,_ in raw_files)
 signature=source_signature(paths)
 data_modules=['prepare.py','schemas.py','common.py','alert_taxonomy.py','historical_sigma.py','observed_quantiles.py','mac_rules_reference.py']
 code_hashes={name:sha(Path(__file__).parent/name) for name in data_modules}
 preparation_signature=hashlib.sha256(json.dumps(dict(source=signature,shards=config.shards,delivery_load_date=config.delivery_load_date,code=code_hashes),sort_keys=True).encode()).hexdigest()
 out=Path(out).expanduser().resolve()
 if out==run or out.is_relative_to(run):raise ValueError('La preparación debe escribirse fuera de la corrida original')
 manifest_path=out/'publication.json'
 if manifest_path.exists():
  old=json.loads(manifest_path.read_text())
  if old.get('preparation_signature')!=preparation_signature:raise ValueError('Cambió la corrida, configuración o código. Usa otra carpeta de preparación')
  if old.get('status')=='PREPARED':
   for item in old['files']+old.get('raw_deliveries',[]):
    if sha(out/item['path'])!=item['sha256']:raise ValueError('Preparación modificada: '+item['path'])
   return old
 if out.exists() and any(out.iterdir()):raise ValueError('La carpeta de preparación contiene archivos incompletos. Usa una carpeta nueva')
 out.mkdir(parents=True,exist_ok=True)
 pub=uuid.uuid4().hex;run_id=hashlib.sha256(str(run).encode()).hexdigest()[:12]+'_'+run.name
 writer=Writer(out,run_id,pub,config.shards)
 if calendar_provenance:
  cp=calendar_provenance
  writer.write('calendar_provenance',[dict(generator=cp['generator'],generator_version=cp.get('generator_version'),profile=cp['profile'],view=cp.get('view'),start_date=cp['start'],end_date=cp['end'],calendar_sha256=cp['calendar_sha256'],feature_count=cp['feature_count'],rows=cp['rows'],metadata_json=json.dumps(cp,ensure_ascii=False,sort_keys=True))])
 stored=pl.read_parquet(root/'inventory.parquet')
 frozen={r['account_id']:r for r in stored.to_dicts()}
 assignments={r['account_id']:r for r in pl.read_parquet(root/'winners_FUTURO.parquet').to_dicts()}
 load_date=day(config.delivery_load_date) if config.delivery_load_date else None
 if load_date is None and 'load_date' in pl.scan_parquet(source).collect_schema():
  last_load=pl.scan_parquet(source).select(pl.col('load_date').max()).collect().item()
  if last_load is not None:load_date=day(last_load)
 inv=[];context_writers=[]
 ctx_schema=pa.schema([('account_id',pa.string()),('date',pa.string()),('balance',pa.float64()),('variation',pa.float64()),('previous_balance',pa.float64()),('fx',pa.float64()),('std_balance',pa.float64()),('std_variation',pa.float64())])
 ctx_path=out/'_context.parquet';ctx_writer=pq.ParquetWriter(ctx_path,ctx_schema,compression='zstd')
 progress('Preparando historia, z-score causal e inventario')
 try:
  for g in source_groups(source,out/'_sorted_source.parquet'):
   if g.select('date').is_duplicated().any():raise ValueError('Fechas duplicadas en '+g['account_id'][0])
   aid=g['account_id'][0];row=g.row(-1,named=True);meta=frozen.get(aid,{});route=assignments.get(aid,{})
   if 'load_date' in g.columns and load_date is None:
    available=g['load_date'].drop_nulls()
    if len(available):load_date=day(available.max())
   if not {'sign','fx','balance','variation','currency','taxonomy'}<=set(g.columns):raise ValueError('Engine 1 debe conservar sign, fx, saldo, variación, divisa y taxonomía')
   balances=g['balance'].to_numpy();variations=g['variation'].to_numpy()
   if not np.isfinite(balances).all() or not np.isfinite(variations).all():raise ValueError('Historia no finita: '+aid)
   inv.append(dict(account_id=aid,currency=row['currency'],taxonomy=row['taxonomy'],cohort=meta.get('cohort','regla' if row['taxonomy']=='cuentas_cero' else 'sin_modelo'),sign=row['sign'],candidate=route.get('candidate'),selection_source=route.get('selection_source'),budget=meta.get('budget'),observations=g.height))
   history=[]
   for target,values in [('saldo',balances),('variacion',variations)]:
    ref=expanding_sigma(values);quantiles=trailing_quantiles(values,report['horizonte'])
    for i,d in enumerate(g['date']):history.append(dict(account_id=aid,currency=row['currency'],date=day(d),target=target,observed=float(values[i]),sigma_center=ref['sigma_center'][i],sigma_std=ref['sigma_std'][i],zscore=ref['zscore'][i],sigma_reference_n=int(ref['sigma_reference_n'][i]),zscore_status=str(ref['zscore_status'][i]),hist_q025=quantiles[i,0],hist_q50=quantiles[i,1],hist_q975=quantiles[i,2]))
   writer.by_account('history',history)
   stdb=inclusive_std(balances);stdv=inclusive_std(variations,True)
   ctx=[dict(account_id=aid,date=str(d)[:10],balance=float(balances[i]),variation=float(variations[i]),previous_balance=float(balances[i-1]) if i else 0.,fx=clean(g['fx'][i]),std_balance=stdb[i],std_variation=stdv[i]) for i,d in enumerate(g['date'])]
   ctx_writer.write_table(pa.Table.from_pylist(ctx,schema=ctx_schema))
 finally:ctx_writer.close()
 if load_date is None:raise ValueError('Indica delivery_load_date: Engine 1 no informa load_date para el layout contable')
 inventory=pl.DataFrame(inv,infer_schema_length=None)
 writer.write('inventory',inv)
 writer.write('candidates',[dict(candidate=d['id'],input=d['inputs'],selection_score=d.get('score'),parameters_json=json.dumps(d,ensure_ascii=False,sort_keys=True)) for d in definitions])
 if (root/'hpo_trials.parquet').is_file():
  writer.write('hpo_trials',pl.read_parquet(root/'hpo_trials.parquet'))
 cv={};population_hashes={}
 for index,d in enumerate(definitions):
  progress(f"Métricas pareadas {index+1}/{len(definitions)} · {d['id']}")
  candidate_paths=sorted((root/'predicciones').glob(d['id']+'_F[123].parquet'))
  if len(candidate_paths)!=3:raise ValueError('Faltan folds completos de '+d['id'])
  for path in candidate_paths:
   paired=read_prediction(path,inventory)
   fold=paired['fold'][0]
   population=paired.select('account_id','currency','date','target','observed').sort('account_id','date','target').write_csv()
   digest=hashlib.sha256(population.encode()).hexdigest()
   if fold in population_hashes and population_hashes[fold]!=digest:raise ValueError('Candidatos con distinta población u observados en '+fold)
   population_hashes[fold]=digest
   rows=metric_rows(paired,d['id'],'Modeling');writer.write('metrics',rows)
   for r in rows:
    if r['role']!='Modeling':continue
    key=(r['comparison_id'],r['account_id'],r['target']);stats=cv.setdefault(key,dict(n=0,mape_n=0,ae_sum=0.,se_sum=0.,smape_sum=0.,mape_sum=0.))
    for k in stats:stats[k]+=r[k]
 # Zero-shot accounts without per-series CV keep null validation metrics.
 records=[*report['ajustes'],report['futuro']]
 errors=[]
 for record in records:
  progress('Emisión operativa y contratos · '+record['fold'])
  paired=read_prediction(root/record['predicciones'],inventory)
  writer.write('metrics',metric_rows(paired,'operational','Winner'))
  dates=paired['date'].unique().to_list()
  context=pl.scan_parquet(ctx_path).filter(pl.col('date').is_in([str(d)[:10] for d in dates])).collect()
  paired=paired.with_columns(pl.col('date').cast(pl.String)).join(context,on=['account_id','date'],validate='m:1',how='left')
  daily=[];events=[];engine2=[];validation=[]
  for r in paired.to_dicts():
   candidate=r['candidate'];target=r['target'];aid=r['account_id'];ccy=r['currency'];fold=r['fold']
   daily.append(dict(**{k:r.get(k) for k in ['account_id','currency','target','fold','candidate','input','taxonomy','sign','budget','fx','observed','model_q025','model_q50','model_q975','baseline_q025','baseline_q50','baseline_q975']},date=day(r['date']),origin=day(r['origin'])))
   s=cv.get((candidate,aid,target),{});n=s.get('n',0);mn=s.get('mape_n',0)
   values=[s.get('mape_sum',0)/mn if mn else None,s.get('smape_sum',0)/n if n else None,s.get('ae_sum',0)/n if n else None,s.get('se_sum',0)/n if n else None,math.sqrt(s.get('se_sum',0)/n) if n else None]
   e2=dict(load_date=load_date,cutoff_date=day(r['date']),cross_key_id=aid+('_var' if target=='variacion' else ''),data_mach_learning_model_type=candidate,dmlm_gl_account_pntb_amount=decimal(r['model_q50']),dmlm_lower_interval_number=decimal(r['model_q025']),dmlm_upper_interval_number=decimal(r['model_q975']),dmlm_business_attributes_desc='',dmlm_business_attributes1_desc='',dmlm_bus_atrb_value_number=None)
   e2.update({k:decimal(v,10 if i<2 else 24) for i,(k,v) in enumerate(zip(ENGINE2_COLUMNS[7:12],values))});engine2.append(e2)
   if r['observed'] is None:continue
   if r['fx'] is None or not math.isfinite(r['fx']) or r['fx']<=0:raise ValueError('Falta FX válido para Engine 3: '+aid)
   for prefix,label in [('model','Winner'),('baseline','Baseline')]:
    e=evaluate(r['observed'],r[prefix+'_q025'],r[prefix+'_q975'],r['sign'],target,r['taxonomy'],r['fx'])
    if e['active']:events.append(dict(account_id=aid,currency=ccy,taxonomy=r['taxonomy'],cohort=r['cohort'],budget=r['budget'],date=day(r['date']),target=target,fold=fold,source=label,severity=e['severity'],reasons='|'.join(e['reasons']),has_alert=any(x!='ALERTA_SOBREGIRO' for x in e['reasons']),overdraft=e['overdraft'],observed=r['observed'],lower=e['lower'],upper=e['upper'],percentage=e['percentage']))
   if r['balance'] is None:raise ValueError('Predicción observada sin historia de Engine 1: '+aid)
   validation.append(dict(account_id=aid,cutoff_date=r['date'],evaluation_phase=fold,gl_account_id=aid[:-len(ccy)] if aid.endswith(ccy) else aid,currency_id=ccy,classification_account_type=r['sign'],gl_account_hist_behaviour_type=r['taxonomy'],saldo_del_dia=r['balance'],variacion_saldo=r['variation'],saldo_anterior=r['previous_balance'],saldo_del_dia_valorizado=r['balance']*r['fx'],saldo_del_dia_anterior_valorizado=r['previous_balance']*r['fx'],variacion_valorizada=r['variation']*r['fx'],tipo_cambio_final=r['fx'],stddev_saldo_del_dia=r['std_balance'],stddev_variacion_saldo=r['std_variation'],final_model=candidate,cross_key_id=aid+('_var' if target=='variacion' else ''),forecast=r['model_q50'],lower_interval=r['model_q025'],upper_interval=r['model_q975']))
  # Accounting-zero rules remain independent of learned forecasting eligibility.
  zeros=context.join(inventory.select('account_id','currency','taxonomy','cohort','sign','budget'),on='account_id',validate='m:1').filter(pl.col('taxonomy')=='cuentas_cero')
  for z in zeros.to_dicts():
   if z['fx'] is None or not math.isfinite(z['fx']) or z['fx']<=0:raise ValueError('Falta FX válido para regla cero: '+z['account_id'])
   e=evaluate(z['balance'],None,None,z['sign'],'saldo',z['taxonomy'],z['fx'])
   if e['active']:events.append(dict(account_id=z['account_id'],currency=z['currency'],taxonomy=z['taxonomy'],cohort=z['cohort'],budget=z['budget'],date=day(z['date']),target='saldo',fold=record['fold'],source='Regla contable',severity=e['severity'],reasons='REGLA_CERO',has_alert=True,overdraft=False,observed=z['balance'],lower=None,upper=None,percentage=None))
   validation.append(dict(account_id=z['account_id'],cutoff_date=z['date'],evaluation_phase=record['fold'],gl_account_id=z['account_id'][:-len(z['currency'])] if z['account_id'].endswith(z['currency']) else z['account_id'],currency_id=z['currency'],classification_account_type=z['sign'],gl_account_hist_behaviour_type=z['taxonomy'],saldo_del_dia=z['balance'],variacion_saldo=z['variation'],saldo_anterior=z['previous_balance'],saldo_del_dia_valorizado=z['balance']*z['fx'],saldo_del_dia_anterior_valorizado=z['previous_balance']*z['fx'],variacion_valorizada=z['variation']*z['fx'],tipo_cambio_final=z['fx'],stddev_saldo_del_dia=z['std_balance'],stddev_variacion_saldo=z['std_variation'],final_model='',cross_key_id=z['account_id'],forecast=None,lower_interval=None,upper_interval=None))
  writer.by_account('daily',daily);writer.by_account('events',events);writer.write('engine2',engine2,record['fold'])
  if validation:
   official=OriginalMacRules().run_partitioned(pl.DataFrame(validation,infer_schema_length=None),context=('evaluation_phase','cutoff_date'),atypical_as_trainable=True).official.select(OFFICIAL_COLUMNS).with_columns(pl.col('cutoff_date','load_date').str.to_date())
   writer.write('engine3',official,record['fold'])
 if training_path:
  train=pl.read_parquet(training_path) if str(training_path).endswith('.parquet') else pl.read_csv(training_path)
  expected=SCHEMAS['training'].names
  if not set(expected)<=set(train.columns):raise ValueError('training_path requiere '+str(expected))
  writer.write('training',train.select(expected))
  for target in ('saldo','variacion'):
   col='train_loss_'+target
   if col in train.columns:
    writer.write('training_targets',train.select('candidate','fold','epoch',pl.lit(target).alias('target'),pl.col(col).alias('train_loss')))
 if training_path:
  for split in ['train','validation']:
   for target in ['saldo','variacion']:
    col='fixed_'+split+'_'+target
    if col in train.columns:
     writer.write('training_learning',train.filter(pl.col(col).is_not_null()).select('candidate','fold','epoch',pl.lit(target).alias('target'),pl.lit(split).alias('split'),pl.col(col).alias('loss')))
 for detail in detail_files(root/'loss_entities'):
  for batch in read_batches(detail):
   writer.write('training_entities',batch)
 raw=[]
 for i,p,base in raw_files:
  rel=Path('deliveries_original')/f'{i:02d}_{base.name}'/p.relative_to(base);dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  try:
   dest.hardlink_to(p)
  except OSError:
   check_space(dest.parent,extra_bytes=p.stat().st_size,stage='copia de entrega')
   shutil.copyfile(p,dest)
  item=dict(path=str(rel),source=str(p),sha256=sha(p),bytes=p.stat().st_size)
  if p.suffix=='.parquet':
   sc=pq.read_schema(p);item.update(columns=sc.names,arrow_schema_b64=base64.b64encode(sc.serialize().to_pybytes()).decode())
  raw.append(item)
 if source_signature(paths)!=signature:raise ValueError('La corrida cambió durante la preparación; no se permite publicarla')
 writer.flush()
 # Small summary table: the interactive monitor never scans per-account CV rows.
 metric_paths=[out/x['path'] for x in writer.entries if x['dataset']=='metrics']
 if metric_paths:
  lazy=pl.scan_parquet(metric_paths)
  keys=['currency','cohort','taxonomy','target','input','comparison_id','candidate','role','budget']
  moments=['n','mape_n','ae_sum','se_sum','smape_sum','mape_sum','covered_sum','width_sum','ae_scaled_sum','se_scaled_sum']
  for selected_fold in ['F1','F2','F3','ALL']:
   block=lazy if selected_fold=='ALL' else lazy.filter(pl.col('fold')==selected_fold)
   summary=block.group_by(keys).agg(pl.col('account_id').n_unique().alias('accounts'),*[pl.col(k).sum() for k in moments]).with_columns(pl.lit(selected_fold).alias('fold')).collect(engine='streaming')
   writer.write('arena_summary',summary)
 writer.flush()
 manifest=dict(status='PREPARED',version='0.12.1',run_id=run_id,run_label=run.name,publication_id=pub,source_signature=signature,preparation_signature=preparation_signature,code_hashes=code_hashes,shards=config.shards,files=writer.entries,raw_deliveries=raw,
  calendar_provenance=calendar_provenance,
  datasets={name:dict(columns=[dict(name=f.name,type=athena_type(f.type)) for f in sc],rows=sum(x['rows'] for x in writer.entries if x['dataset']==name)) for name,sc in SCHEMAS.items() if name!='publications'},
  protocol='F1–F3 usados en selección; no Test independiente. Layout Engine2 preservado, selección MAC3 explícita. Engine3 port literal MAC2 separado de diagnóstico MAC3.',
  population_hashes=population_hashes,engine3_reference_sha256=REFERENCE_SHA256,training_available=bool(training_path),delivery_load_date=str(load_date),created_at=datetime.now(timezone.utc).isoformat())
 manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 for name in ['_context.parquet','_sorted_source.parquet']:(out/name).unlink(missing_ok=True)
 progress('Preparación completa: '+str(manifest_path))
 return manifest
