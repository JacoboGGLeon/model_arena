"""ADA worker bootstrap: validate actual imports/compute, never trust pip metadata.

No AWS calls. Never edits Conda or saved model/run artifacts.
Only the selected private worker may receive dependencies and the bundled source libraries.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv

SCHEMA = 'ada-runtime-bootstrap.v2'
PROBE = r'''
import json,sys
from pathlib import Path
import torch
x=torch.tensor([1.,2.,3.],requires_grad=True)
y=torch.mm(x.reshape(1,3),x.reshape(3,1)).sum();y.backward()
assert y.item()==14. and x.grad.tolist()==[2.,4.,6.]
requested=sys.argv[1];selected='cpu';gpu_error=None
if requested!='cpu':
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        z=torch.tensor([1.,2.,3.],device='cuda',requires_grad=True)
        (z*z).sum().backward();torch.cuda.synchronize()
        assert z.grad.cpu().tolist()==[2.,4.,6.]
        selected='cuda'
    except Exception as exc:
        gpu_error=type(exc).__name__+': '+str(exc)
        if requested=='cuda':raise
print(json.dumps({'status':'PASS','prefix':sys.prefix,'base_prefix':sys.base_prefix,'torch':torch.__version__,'torch_origin':torch.__file__,'cpu_forward':y.item(),'cpu_backward':True,'device':selected,'cuda_error':gpu_error}))
'''


def redact(value):
    return re.sub(r'(https?://)[^\s/@]+:[^\s/@]*@',r'\1<redacted>@',str(value))


def worker_environment(environ=None):
    result=dict(os.environ if environ is None else environ)
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP','PIP_TARGET','PIP_PREFIX','PIP_USER'):
        result.pop(key,None)
    result['PYTHONNOUSERSITE']='1'
    return result


def native_library_environment(env,prefixes=None):
    result=dict(env)
    paths=[Path(p)/'lib' for p in (prefixes or (sys.prefix,sys.base_prefix,'/opt/conda'))]
    dirs=list(dict.fromkeys(str(p) for p in paths if p.is_dir()))
    previous=result.get('LD_LIBRARY_PATH','')
    result['LD_LIBRARY_PATH']=os.pathsep.join(dirs+([previous] if previous else []))
    return result


def probe_python(python,env,device='auto'):
    try:
        p=subprocess.run([str(python),'-I','-c',PROBE,device],env=env,text=True,capture_output=True,timeout=120)
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'status':'FAIL','error':redact(exc)}
    if p.returncode:
        return {'status':'FAIL','error':redact((p.stderr or p.stdout)[-5000:])}
    try:return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError):return {'status':'FAIL','error':'Probe did not return valid JSON'}


def is_private_worker(python,report):
    home=Path(python).absolute().parent.parent
    return (home/'pyvenv.cfg').is_file() and report.get('status')=='PASS' and Path(report['prefix']).resolve()==home.resolve() and Path(report['prefix']).resolve()!=Path(report['base_prefix']).resolve()


def install(python,arguments,env,log_path):
    p=subprocess.run([str(python),'-I','-m','pip','install','--disable-pip-version-check','--no-cache-dir',*arguments],env=env,text=True,capture_output=True,timeout=1800)
    text=redact(p.stdout+'\n'+p.stderr)
    with Path(log_path).open('a',encoding='utf-8') as stream:stream.write(text+'\n')
    if p.returncode:
        raise RuntimeError('Installation failed. Inspect '+str(log_path)+'\n'+text[-5000:])


WORKER_NAMES = (
    '.venv-librerias-gpu-base-v1', '.venv-librerias-base',
    '.venv-librerias-kernel-v1', '.venv-librerias',
    '.venv-librerias-aislado-cu124-v1', '.venv-librerias-aislado-cpu-v1',
)


def resolve_worker(bundle, device, provided=None, bases=None):
    """Reuse compute like 0.3.3; never installs Torch or falls back from CUDA."""
    bundle = Path(bundle).resolve()
    requested = 'cuda' if device == 'auto' else device
    if requested not in ('cpu', 'cuda'):
        raise ValueError('Invalid requested device')
    env = worker_environment()
    attempts = []
    roots = [bundle] + sorted(p for p in bundle.parent.glob('Observatorio*') if p.is_dir() and p != bundle)
    candidates = ([Path(provided).absolute()] if provided else []) + [root/name/'bin/python' for root in roots for name in WORKER_NAMES]

    def check(python, kind, private=False):
        prefix = python.absolute().parent.parent
        for candidate_env in (env, native_library_environment(env, (prefix, sys.prefix, sys.base_prefix, '/opt/conda'))):
            report = probe_python(python, candidate_env, requested)
            attempts.append({'python':str(python), 'kind':kind, **report})
            if report.get('status') == 'PASS' and report.get('device') == requested:
                if not private or is_private_worker(python, report):
                    return candidate_env, report
        return None

    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file() or not (candidate.parent.parent/'pyvenv.cfg').is_file():
            continue
        print('Comprobando worker:', candidate, flush=True)
        checked = check(candidate, 'existing-worker', private=True)
        if checked:
            return candidate, *checked, attempts

    interpreters = list(dict.fromkeys(Path(p).absolute() for p in (bases if bases is not None else tuple(([str(provided)] if provided else [])+[sys.executable, getattr(sys, '_base_executable', sys.executable), '/opt/conda/bin/python']))))
    for base in interpreters:
        if not base.is_file():
            continue
        print('Comprobando Python base:', base, flush=True)
        checked = check(base, 'base-python')
        if not checked:
            continue
        # Distinct overlay per interpreter; never rewrite an old or failed environment.
        import hashlib
        key = hashlib.sha256(str(base).encode()).hexdigest()[:10]
        home = bundle / ('.venv-librerias-gpu-base-v1' if requested == 'cuda' else '.venv-librerias-cpu-base-v1')
        if home.exists():
            home = bundle / (home.name + '-' + key)
        if not (home/'bin/python').is_file():
            print('Creando worker privado heredando Torch de:', base, flush=True)
            try:
                subprocess.run([str(base), '-I', '-m', 'venv', '--system-site-packages', str(home)], env=checked[0], check=True, capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as exc:
                attempts.append({'python':str(base), 'kind':'overlay-create', 'status':'FAIL', 'error':redact(str(exc))})
                continue
        candidate = (home/'bin/python').absolute()
        overlay = check(candidate, 'inherited-overlay', private=True)
        if overlay:
            return candidate, *overlay, attempts
    failure = {'status':'FAIL', 'device':requested, 'attempts':attempts, 'torch_downloaded':False}
    (bundle/'runtime_bootstrap_diagnostic.json').write_text(json.dumps(failure, indent=2))
    raise RuntimeError('No hay un Python con '+requested+' funcional. No se descargó Torch ni se cambió de dispositivo. Ver runtime_bootstrap_diagnostic.json; indica un worker válido mediante WORKER_PYTHON.')


def ensure_runtime(bundle,device='cuda',aws=True,provided=None):
    bundle=Path(bundle).resolve()
    python,env,report,attempts=resolve_worker(bundle,device,provided)
    logs=bundle/'runtime_bootstrap_logs';logs.mkdir(exist_ok=True)
    install(python,['optuna>=4,<5','numpy>=1.26,<3','polars>=1.30,<2','plotly>=6,<7','boto3>=1.35,<2','streamlit>=1.52,<2','pandas>=2.1,<4','pyarrow>=14','Pillow>=10'],env,logs/'dependencies.log')
    install(python,['pydantic>=2,<3'],env,logs/'project.log')
    final=probe_python(python,env,device)
    if final.get('status')!='PASS':raise RuntimeError('Runtime inválido: '+redact(final.get('error','')))
    source=bundle/'entrenamiento'
    paths=[str(source),str(source/'visor'),*[str(p) for p in (source/'libraries').glob('*/src')]]
    check_code='import sys,json;sys.path[:0]='+repr(paths)+";import arena_estudio,streamlit,pyarrow,boto3,pandas,plotly;import agentic_forecasting;print(json.dumps({'source':agentic_forecasting.__file__}))"
    checked=subprocess.run([str(python),'-I','-c',check_code],env=env,text=True,capture_output=True,timeout=120)
    if checked.returncode:raise RuntimeError('El worker no puede importar el entrenador/monitor: '+redact(checked.stderr[-5000:]))
    origin=json.loads(checked.stdout.strip().splitlines()[-1])['source']
    if not Path(origin).resolve().is_relative_to(source):raise RuntimeError('El worker importó una librería de otra entrega')
    result={'python':str(python),'validation':final,'attempts':attempts,'library_policy':'current bundle source, never old installed wheels','source':origin}
    (bundle/'runtime_bootstrap_report.json').write_text(json.dumps(result,indent=2))
    return python,env,result
