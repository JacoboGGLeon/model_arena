"""Entrenar -> preparar -> publicar -> abrir, con identidad y reanudación explícitas."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,signal,subprocess,time,uuid
ROOT=Path(__file__).resolve().parent


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)


def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


def verify():return module('_launcher_flujo_0121',ROOT/'abrir_monitor.py').verify(ROOT)


def plan(*,new=False,device='cuda',training_config=None,publication_config=None,run_dir=None):
 verify()
 if device not in ('cuda','cpu'):raise ValueError('DEVICE debe ser cuda o cpu')
 train=Path(training_config or ROOT/'config/entrenamiento_ada.json').resolve()
 publication=Path(publication_config or ROOT/'config/ada.json').resolve()
 signatures={'training':sha(train),'publication':sha(publication)}
 latest=ROOT/'sesiones/ultima.json'
 if run_dir:
  directory=Path(run_dir).expanduser().resolve()
 elif not new and latest.exists() and (Path(json.loads(latest.read_text())['session'])/'plan.json').is_file():
  directory=Path(json.loads(latest.read_text())['session'])
 else:
  directory=ROOT/'sesiones'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
 file=directory/'plan.json'
 if file.exists():
  previous=json.loads(file.read_text())
  if previous['config_signatures']!=signatures or previous['device']!=device:raise ValueError('La sesión anterior tiene otra configuración/dispositivo. Usa NUEVA_CORRIDA=True.')
  for name,digest in previous['snapshot_hashes'].items():
   if sha(directory/name)!=digest:raise ValueError('Snapshot de configuración modificado: '+name)
  result=previous
 else:
  directory.mkdir(parents=True,exist_ok=True)
  train_data=json.loads(train.read_text())
  if train_data.get('requested_accounts')!=0:raise ValueError('El entrenador vigente requiere requested_accounts=0: universo elegible completo')
  if not train_data.get('engine1_path') and train_data['aws'].get('account_limit')!=0:raise ValueError('Para universo completo account_limit debe ser 0')
  write(directory/'entrenamiento.json',train_data)
  write(directory/'publicacion.json',json.loads(publication.read_text()))
  result={'session':str(directory),'run':str(directory/'entrenada'/directory.name),'prepared':str(directory/'preparada'),'training_config':str(directory/'entrenamiento.json'),'publication_config':str(directory/'publicacion.json'),'device':device,'config_signatures':signatures,'snapshot_hashes':{name:sha(directory/name) for name in ('entrenamiento.json','publicacion.json')}}
  write(file,result)
 write(latest,{'session':str(directory)})
 print('Sesión:',directory,'\nDispositivo:',device,'\nCorrida:',result['run'])
 return result


def bootstrap(device='cuda',provided=None):
 verify()
 worker=module('_runtime_flujo_0121',ROOT/'scripts/runtime_bootstrap.py')
 python,env,report=worker.ensure_runtime(ROOT,device=device,provided=provided)
 return str(python),env


def command(args,log,env,*,cwd=ROOT):
 """Mantiene el log y termina el grupo de procesos al interrumpir el notebook."""
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 import re
 def show(value):
  if value:print(re.sub(r'(https?://)[^\s/@]+:[^\s/@]*@',r'\1<redacted>@',value),end='',flush=True)
 offset=log.stat().st_size if log.exists() else 0
 with log.open('a',buffering=1) as output:
  process=subprocess.Popen(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  last=time.monotonic()
  try:
   while True:
    with log.open() as stream:stream.seek(offset);show(stream.read());offset=stream.tell()
    if process.poll() is not None:
     with log.open() as stream:stream.seek(offset);show(stream.read())
     if process.returncode:raise RuntimeError('Etapa fallida. Log: '+str(log))
     return
    if time.monotonic()-last>15:print('En curso ·',log,flush=True);last=time.monotonic()
    time.sleep(.3)
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def check_training(plan):
 session=Path(plan['session']);receipt=session/'entrenamiento_completo.json'
 if not receipt.exists():return False
 saved=json.loads(receipt.read_text())
 if saved['build']!=sha(ROOT/'BUILD_MONITOR.json'):raise ValueError('La corrida corresponde a otro código')
 for name,digest in saved['files'].items():
  path=Path(plan['run'])/name
  if not path.is_file() or sha(path)!=digest:raise ValueError('Corrida modificada o incompleta: '+name)
 return True


def train(plan,python,env):
 verify();session=Path(plan['session'])
 if check_training(plan):print('Entrenamiento completo reutilizado · hashes verificados');return
 command([python,'-u',ROOT/'entrenamiento/entrenar_curvas.py','--run',plan['run'],'--config',plan['training_config'],'--device',plan['device']],session/'logs/entrenamiento.log',env)
 run=Path(plan['run']);report=run/'curvas/resultados_x0.json'
 status=json.loads((run/'estado_curvas.json').read_text())
 if status['estado']!='COMPLETO' or not report.is_file() or not (run/'curvas/training_history.parquet').is_file():raise RuntimeError('El entrenamiento no produjo todos los artefactos')
 files={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.suffix not in ('.lock','.tmp')}
 write(session/'entrenamiento_completo.json',{'build':sha(ROOT/'BUILD_MONITOR.json'),'files':files,'status':'TRAINED'})
 print('Entrenamiento completo · checkpoints, predicciones y pérdidas guardados')


def prepare(plan,python,env,*,delivery_dirs=()):
 verify()
 if not check_training(plan):raise ValueError('Completa primero el entrenamiento')
 session=Path(plan['session']);root=Path(plan['run'])/'curvas'
 prepared=Path(plan['prepared'])
 if prepared.exists() and any(prepared.iterdir()) and not (prepared/'publication.json').is_file():
  archived=prepared.with_name(prepared.name+'_incompleta_'+uuid.uuid4().hex[:8])
  prepared.rename(archived)
  print('Preparación interrumpida conservada en',archived,'· se inicia otra preparación')
 args=[python,'-u',ROOT/'publicar.py','--run',plan['run'],'--out',plan['prepared'],'--config',plan['publication_config'],'--training-history',root/'training_history.parquet','--delivery-dir',root/'modelos','--prepare-only']
 for directory in delivery_dirs:args+=['--delivery-dir',Path(directory).expanduser().resolve()]
 command(args,session/'logs/preparacion.log',env)
 manifest=json.loads((Path(plan['prepared'])/'publication.json').read_text())
 write(session/'estado.json',{'status':'PREPARED','publication_id':manifest['publication_id']})
 print('Preparación completa · contratos Engine 2/3 y MAC3; modelos archivados')
 return manifest


def publish(plan,python,env):
 verify()
 if not check_training(plan):raise ValueError('La corrida no es válida')
 session=Path(plan['session']);prepared=Path(plan['prepared'])
 if not (prepared/'publication.json').is_file():raise ValueError('Completa primero la preparación')
 program="from observatorio_athena.common import Config;from observatorio_athena.aws import Publisher;import sys;Publisher(Config.read(sys.argv[1])).publish(sys.argv[2])"
 command([python,'-u','-c',program,plan['publication_config'],prepared],session/'logs/publicacion.log',env)
 receipt=json.loads((prepared/'published.json').read_text())
 manifest=json.loads((prepared/'publication.json').read_text())
 if receipt['status']!='PUBLISHED' or receipt['publication_id']!=manifest['publication_id']:raise RuntimeError('No se verificó la publicación de esta corrida')
 write(session/'estado.json',receipt)
 print('Publicación verificada en S3/Athena ·',receipt['publication_id'])
 return receipt


def open_monitor(plan,python=None):
 verify();session=Path(plan['session']);prepared=Path(plan['prepared'])
 receipt=json.loads((prepared/'published.json').read_text())
 manifest=json.loads((prepared/'publication.json').read_text())
 if receipt['status']!='PUBLISHED' or receipt['publication_id']!=manifest['publication_id']:raise ValueError('Falta publicar esta corrida')
 cfg=json.loads(Path(plan['publication_config']).read_text());cfg['initial_publication_id']=receipt['publication_id']
 target=session/'monitor.json';write(target,cfg)
 launcher=module('_abrir_flujo_0121',ROOT/'abrir_monitor.py')
 return launcher.launch(config=target,python=python)
