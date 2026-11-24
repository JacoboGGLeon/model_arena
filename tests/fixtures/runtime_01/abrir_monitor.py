"""Only launches the exact native Streamlit/Athena app shipped here."""
from pathlib import Path
import importlib.metadata,importlib.util,json,os,socket,subprocess,sys,time,urllib.request,hashlib
ROOT=Path(__file__).resolve().parent
BUILD='OBS-ENTRENAR-PUBLICAR-MONITOR-0121'
_WORKER_ENVS={}


def verify(root):
 root=Path(root).resolve();manifest=json.loads((root/'BUILD_MONITOR.json').read_text())
 if manifest.get('build_id')!=BUILD:raise ValueError('La carpeta no corresponde al monitor Streamlit Athena 0.12.1')
 for name,digest in manifest['files'].items():
  path=root/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Archivo de otra entrega: '+str(path))
 return root


def find_bundle(cwd=None,explicit=None):
 if explicit:return verify(explicit)
 cwd=Path(cwd or Path.cwd()).resolve();found=[]
 for base in [cwd,*list(cwd.parents)[:3]]:
  for root in [base,base/'Observatorio_Entrenar_Publicar_Monitor_0121']:
   if (root/'BUILD_MONITOR.json').is_file():
    try:checked=verify(root)
    except (ValueError,KeyError,FileNotFoundError):continue
    if checked not in found:found.append(checked)
 if len(found)!=1:raise ValueError('Indica BUNDLE_PATH: se encontraron '+str(len(found))+' copias verificadas')
 return found[0]


def worker_env(python):
 cached=_WORKER_ENVS.get(str(python))
 if cached is not None:return dict(cached)
 path=ROOT/'scripts/runtime_bootstrap.py'
 spec=importlib.util.spec_from_file_location('_obs_athena_runtime_environment',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.native_library_environment(module.worker_environment(),prefixes=(Path(python).parent.parent,sys.prefix,sys.base_prefix,'/opt/conda'))


def runtime(root,explicit=None):
 candidates=[Path(explicit).expanduser()] if explicit else [Path(sys.executable),*sorted(Path(root).glob('.venv*/bin/python')),*sorted(Path(root).parent.glob('*/.venv*/bin/python'))]
 failures=[]
 for path in dict.fromkeys(candidates):
  if not path.is_file():continue
  native=worker_env(path);variants=[native]
  # A monitor may reuse packages installed at user level in an existing runtime.
  # Test the inherited environment as well; accept it only after real imports pass.
  inherited=dict(os.environ)
  inherited['LD_LIBRARY_PATH']=native.get('LD_LIBRARY_PATH','')
  if inherited!=native:variants.append(inherited)
  for env in variants:
   checked=subprocess.run([str(path),'-c','import streamlit,polars,pyarrow,pandas,numpy,plotly,boto3'],capture_output=True,text=True,timeout=30,env=env)
   if checked.returncode==0:
    _WORKER_ENVS[str(path)]=env
    return str(path)
  failures.append(str(path)+' · '+checked.stderr[-300:])
 raise RuntimeError('Indica WORKER_PYTHON con las dependencias del visor.\n'+'\n'.join(failures))


def monitor_runtime(root,provided=None):
 if provided:return runtime(root,provided)
 local=Path(root)/'.venv-monitor'/ 'bin/python'
 for candidate in (local,Path(sys.executable)):
  if candidate.is_file():
   try:return runtime(root,str(candidate))
   except RuntimeError:pass
 import venv
 print('Preparando entorno privado del monitor; no requiere GPU ni entrenamientos anteriores.',flush=True)
 venv.EnvBuilder(with_pip=True,system_site_packages=True).create(local.parent.parent)
 subprocess.run([str(local),'-m','pip','install','-r',str(Path(root)/'requirements.txt')],check=True,env=worker_env(local))
 return runtime(root,str(local))


def launch(config=None,python=None):
 verify(ROOT);python=runtime(ROOT,python)
 config=Path(config or ROOT/'config/ada.json').resolve()
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 prefix=os.environ.get('JUPYTERHUB_SERVICE_PREFIX','/jupyterlab/default/').rstrip('/')+f'/proxy/absolute/{port}'
 logpath=ROOT/f'monitor_athena_{port}.log'
 with logpath.open('w') as log:
  proc=subprocess.Popen([python,'-m','streamlit','run',str(ROOT/'app.py'),'--server.address','127.0.0.1','--server.port',str(port),'--server.baseUrlPath',prefix.lstrip('/'),'--server.headless','true','--browser.gatherUsageStats','false','--theme.base','light'],cwd=ROOT,env=dict(worker_env(python),OBS_CONFIG=str(config)),stdout=log,stderr=subprocess.STDOUT)
 for _ in range(100):
  if proc.poll() is not None:raise RuntimeError(logpath.read_text()[-4000:])
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}{prefix}/_stcore/health',timeout=1) as response:
    if response.status==200:break
  except OSError:time.sleep(.25)
 else:
  proc.terminate();raise TimeoutError(logpath.read_text()[-4000:])
 proc.health_url=f'http://127.0.0.1:{port}{prefix}/_stcore/health';proc.logpath=logpath
 from IPython.display import display,HTML
 display(HTML(f'<a href="{prefix}/" target="_blank" rel="noopener" style="display:inline-block;background:#072146;color:white;padding:16px 28px;border-radius:10px;text-decoration:none;font:600 16px Arial;box-shadow:0 4px 14px #07214630;border-left:5px solid #02a5a5">Abrir Observatorio ↗</a>'))
 print('Build:',BUILD,'· Carpeta:',ROOT,'· PID:',proc.pid)
 print('La salud HTTP confirma el servidor; la interfaz requiere conexión del navegador.')
 return proc
