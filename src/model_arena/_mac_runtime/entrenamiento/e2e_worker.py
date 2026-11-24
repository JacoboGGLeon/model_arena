"""Ingesta original Engine 1, aislada de los monitores y entrenadores anteriores."""
from pathlib import Path
import hashlib,json,sys,time
from model_arena.disk import check_space
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'visor'),*(str(p) for p in (ROOT/'libraries').glob('*/src'))]
from workspace import dump,canonical_history

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def execute(run,config,stage,device='cuda',resume=True):
 if stage!='ingesta':raise ValueError('Este adaptador solamente ingiere Engine 1')
 import polars as pl
 import numpy as np
 from observatorio_cuentas.calendar import read_calendar
 run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(Path(config).read_text())
 files=[*ROOT.glob('*.py'),*ROOT.rglob('libraries/**/*.py'),ROOT/'visor/workspace.py']
 identity=dict(config=cfg,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},calendar_sha256=sha(ROOT/'data/calendar_authoritative.csv'))
 if cfg.get('engine1_path'):identity['input_sha256']=sha(Path(cfg['engine1_path']).expanduser().resolve())
 lock=run/'configuration.json'
 if lock.exists():
  if json.loads(lock.read_text())!=identity:raise ValueError('Cambió el código, configuración o Engine1. Crea una corrida nueva.')
 else:dump(lock,identity)
 receipt=run/'receipts/ingesta.json'
 if resume and receipt.exists():
  rec=json.loads(receipt.read_text())
  if rec['configuration_sha256']!=sha(lock):raise ValueError('Identidad de ingesta distinta')
  for name,digest in rec['outputs'].items():
   if not (run/name).is_file() or sha(run/name)!=digest:raise ValueError('Ingesta modificada o incompleta: '+name)
  print('Ingesta reutilizada · hashes verificados',flush=True);return
 start=time.time();cal=read_calendar(ROOT/'data/calendar_authoritative.csv')
 if cfg.get('engine1_path'):
  source=Path(cfg['engine1_path']).expanduser().resolve()
  frame,_=canonical_history(source,cfg.get('snapshot'))
  dump(run/'source/file_origin.json',dict(path=str(source),sha256=sha(source)))
  axis=sorted(frame['date'].unique().to_list())
  dump(run/'source/account_plan.json',dict(status='FILE_INPUT_NO_PLAN_QUERY'))
 else:
  from observatorio_cuentas.aws import AwsAdapter,AwsSettings
  adapter=AwsAdapter(AwsSettings(**cfg['aws']),run)
  adapter.preflight();frame=adapter.ingest(cal);adapter.account_plan();axis=adapter.operational_dates(cal)
 cal.take(axis);cal.after(axis[-1],cfg['model']['horizon'])
 if not np.isfinite(frame.select('balance','variation','fx').to_numpy()).all() or frame.filter(pl.col('fx')<=0).height:raise ValueError('Importe, variación o FX inválido')
 frame=frame.sort('account_id','date');(run/'source').mkdir(exist_ok=True)
 check_space(run,extra_bytes=frame.estimated_size()*2,stage='guardar Engine 1')
 frame.write_parquet(run/'source/engine1.parquet',compression='zstd')
 dump(run/'source/operational_axis.json',dict(dates=axis,source='authoritative calendar intersect Engine1 operational dates before selection'))
 outputs=[p for p in (run/'source').glob('*') if p.is_file()]
 dump(receipt,dict(stage=stage,status='PASS',seconds=time.time()-start,configuration_sha256=sha(lock),outputs={str(p.relative_to(run)):sha(p) for p in outputs}))
 print('Ingesta PASS ·',frame.height,'filas',flush=True)
