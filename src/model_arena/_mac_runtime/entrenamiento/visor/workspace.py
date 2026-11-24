"""Read-only engine outputs -> isolated, indexed monitoring workspace (no Torch)."""
from pathlib import Path
import csv, hashlib, itertools, json, math, os, shutil
import numpy as np
import polars as pl

TAXONOMIES={'cuentas_entrenamiento','cuentas_atipicos','cuentas_cero'}
SIGNS={'positiva','negativa','ambos_signos','cuenta_cero'}

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));temp.replace(path)

def read(path):return json.loads(Path(path).read_text())
def key(aid):return hashlib.sha256(aid.encode()).hexdigest()[:24]
def load_frame(path):
    p=Path(path)
    return pl.read_parquet(p) if p.suffix!='.csv' else pl.read_csv(p,infer_schema=False)

def canonical_history(path,snapshot=None):
    frame=load_frame(path)
    legacy='cross_key_id' in frame.columns
    mapping={'cross_key_id':'account_id','cutoff_date':'date','gl_mthly_ab_mxn_amount':'balance',
             'currency_id':'currency','gl_account_hist_behaviour_type':'taxonomy','classification_account_type':'sign',
             'prev_day_bal_cbl_vary_amount':'variation'} if legacy else {}
    frame=frame.rename({a:b for a,b in mapping.items() if a in frame.columns})
    required={'account_id','date','balance','currency','taxonomy','sign'}
    if required-set(frame.columns):raise ValueError('Faltan columnas de Engine 1: '+str(sorted(required-set(frame.columns))))
    if 'load_date' in frame.columns:
        frame=frame.with_columns(pl.col('load_date').cast(pl.String))
        loads=frame['load_date'].unique().drop_nulls().to_list()
        if len(loads)>1 and snapshot is None:raise ValueError('Hay varias cargas de Engine 1; indica SNAPSHOT explícitamente')
        if snapshot is not None:
            frame=frame.filter(pl.col('load_date')==str(snapshot))
            if frame.is_empty():raise ValueError('SNAPSHOT no existe')
    frame=frame.with_columns([pl.col(c).cast(pl.String) for c in ('account_id','date','currency','taxonomy','sign')]+[pl.col('balance').cast(pl.Float64)])
    if frame.is_empty() or frame.select(pl.any_horizontal([pl.col(c).is_null() for c in required]).any()).item():raise ValueError('Engine 1 vacío o con campos obligatorios nulos')
    if not np.isfinite(frame['balance'].to_numpy()).all():raise ValueError('Saldos no finitos')
    if set(frame['taxonomy'])-TAXONOMIES or set(frame['sign'])-SIGNS:raise ValueError('Taxonomía o signo no reconocido; no se reclasifica automáticamente')
    if frame.select(pl.struct('account_id','date').n_unique()).item()!=frame.height:raise ValueError('Duplicados cuenta/fecha en Engine 1; no se eliminan silenciosamente')
    for _,g in frame.group_by('account_id'):
        if any(g[c].n_unique()!=1 for c in ('currency','taxonomy','sign')):raise ValueError('Cambios de identidad/clasificación dentro de una carga')
    frame=frame.sort(['account_id','date'])
    delta=(pl.col('balance')-pl.col('balance').shift(1).over('account_id')).fill_null(0)
    if 'variation' not in frame.columns:frame=frame.with_columns(delta.alias('variation'))
    else:frame=frame.with_columns(pl.col('variation').cast(pl.Float64))
    # Engine 1 lacks FX. Never silently assign 1 to a foreign currency.
    if 'fx' not in frame.columns:frame=frame.with_columns(pl.when(pl.col('currency')=='MXP').then(1.).otherwise(None).alias('fx'))
    else:frame=frame.with_columns(pl.col('fx').cast(pl.Float64))
    return frame,legacy

def entropy(values):
    a=np.asarray(values,float)
    if len(a)<3:return 0.
    w=np.lib.stride_tricks.sliding_window_view(a,3);orders=np.argsort(w,axis=1,kind='stable');weights=np.var(w,axis=1)
    if not weights.sum():return 0.
    probs=np.array([weights[np.all(orders==p,axis=1)].sum() for p in itertools.permutations(range(3))])/weights.sum()
    p=probs[probs>0];return float(-np.sum(p*np.log(p))/np.log(6))

def dynamic_z(values,min_periods=5):
    """Compatibility entry point for the recovered expanding reference."""
    from historical_sigma import expanding_sigma
    return expanding_sigma(values,minimum=min_periods)['zscore']

def account_history(root,aid):
    if (Path(root)/'resultados_x0.json').is_file():
        return pl.scan_parquet(Path(root).parent/'source/engine1.parquet').filter(pl.col('account_id')==aid).sort('date').collect()
    return pl.read_parquet(Path(root)/'history'/f'{key(aid)}.parquet')

def prepare(source_run,destination,engine1=None,snapshot=None):
    root=Path(destination).resolve();source=Path(source_run).resolve() if source_run else None
    path=Path(engine1).resolve() if engine1 else source/'universe.parquet' if source else None
    if path is None or not path.exists():raise FileNotFoundError('Indica ENGINE1_PATH o una corrida con universe.parquet')
    if root.exists() and (root/'manifest.json').exists():
        m=read(root/'manifest.json')
        if m['source_run']!=(str(source) if source else None) or m['engine1_path']!=str(path):raise ValueError('Usa otro WORKSPACE para otra fuente')
        if m.get('snapshot')!=snapshot or m['engine1_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Cambió la salida de Engine 1 o el snapshot; crea otro WORKSPACE')
        print('Índice existente reutilizado:',root,flush=True);return root
    if root.exists() and any(root.iterdir()):raise FileExistsError('WORKSPACE incompleto: usa otra carpeta; no se sobreescribe')
    if source and (root==source or source.is_relative_to(root)):raise ValueError('WORKSPACE debe estar separado de la corrida')
    root.mkdir(parents=True,exist_ok=True)
    frame,legacy=canonical_history(path,snapshot)
    meta={r['id']:r for r in read(source/'series.json')} if source and (source/'series.json').exists() else {}
    spec=read(source/'arena/experiment.json').get('spec',{}) if source and (source/'arena/experiment.json').exists() else {}
    H=spec.get('horizon',20);folds=spec.get('cv_folds',3);validation=spec.get('validation',20)
    groups={aid[0]:g for aid,g in frame.group_by('account_id',maintain_order=True)}
    values={aid:entropy(g['balance'].to_numpy()[:max(0,g.height-2*H-folds*validation)]) for aid,g in groups.items()}
    pool=[v for aid,v in values.items() if groups[aid]['taxonomy'][0]=='cuentas_entrenamiento' and groups[aid].height-2*H-folds*validation>=20 and meta.get(aid,{}).get('cohort','seen')=='seen']
    edges=np.quantile(pool,np.arange(1,10)/10).tolist() if pool else []
    inventory=[];(root/'history').mkdir()
    for aid,g in groups.items():
        prefix=g.height-2*H-folds*validation
        budget=meta.get(aid,{}).get('budget')
        budget_source='corrida_importada' if budget is not None else 'deciles_en_prefix_train' if edges and prefix>=20 else 'no_disponible'
        if budget is None and edges and prefix>=20:budget=int(np.searchsorted(edges,values[aid],side='right'))+1
        g.write_parquet(root/'history'/f'{key(aid)}.parquet')
        d=g['date'].to_list();expected=sum((__import__('datetime').date.fromisoformat(d[i])-__import__('datetime').date.fromisoformat(d[i-1])).days>4 for i in range(1,len(d)))
        inventory.append(dict(account_id=aid,taxonomy=g['taxonomy'][0],currency=g['currency'][0],sign=g['sign'][0],budget=budget,budget_source=budget_source,cohort=meta.get(aid,{}).get('cohort','sin_modelado'),observations=g.height,date_min=d[0],date_max=d[-1],zeros=float((g['balance']==0).mean()),entropy=values[aid],mean=float(g['balance'].mean()),std=float(g['balance'].std() or 0),long_gaps=expected,last_balance=float(g['balance'][-1])))
    pl.DataFrame(inventory).write_parquet(root/'inventory.parquet')
    dump(root/'budget_policy.json',dict(edges=edges,history_prefix='N-2H-cv_folds*validation',fit_population='entrenables elegibles, seen si consta',preserve_existing_budgets=True,notice='Budgets importados mantienen su procedencia; no se certifica su causalidad si falta evidencia original.'))
    if source:
        for name in ['arena/winners.json','arena/budget_winners.json','arena/experiment.json']:
            if (source/name).exists():
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
        if (source/'alertas_MAC.csv').exists():shutil.copy2(source/'alertas_MAC.csv',root/'imported_alertas_MAC.csv')
    dump(root/'manifest.json',dict(version='0.6.7',source_run=str(source) if source else None,engine1_path=str(path),engine1_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),engine1_contract='MAC2 columnas originales' if legacy else 'adaptador universe.parquet de corrida previa; origen Engine1 no certificado',snapshot=snapshot,accounts=len(inventory),observations=frame.height,horizon=H,cv_folds=folds,validation=validation,model_history_matches=None,monitor_only=True))
    print('Monitor 1 listo:',len(inventory),'cuentas. Sin entrenamiento.',flush=True)
    return root
