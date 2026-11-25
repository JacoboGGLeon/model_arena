"""Ingesta Engine 1, HPO en tres folds y pronóstico de veinte pasos."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'visor'), *(str(p) for p in (ROOT/'libraries').glob('*/src'))]

import numpy as np
import polars as pl
import torch
from agentic_forecasting import ExperimentSpec, Series
from agentic_forecasting.data import weighted_entropy
from observatorio_cuentas.calendar import read_calendar
from observatorio_cuentas.sampling import sample_universe
from observatorio_cuentas.universe import AccountingUniverse
from e2e_worker import execute, sha
from curvas_contrato import COMPONENTS, KIND, RESULT, TARGETS
from curvas_metricas import aggregate, curve_score, REFERENCE
from curvas_modelo import CurveTFT, fit_curves, predict_curves


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)


def spec_for(config):
    return ExperimentSpec(
        **config['model'],
        inputs=('X0',) if 'input_specs' in config['curvas'] else tuple(config['curvas']['inputs']),
        encoder_designs=tuple(config['curvas'].get('encoders',['independent'])),
        architectures=('tft_compact',), decoders=('direct',),
        approaches=('mixed',), cv_folds=3, test=config['model']['horizon'],
    )


def prepare(run, config, spec):
    axis = read(run/'source/operational_axis.json')['dates']
    if config.get('observations'):
        axis = axis[-config['observations']:]
    initial = len(axis)-3*spec.horizon
    if initial < max(spec.window,spec.horizon)+spec.horizon:
        raise ValueError('El historial no alcanza para entrenar y conservar F1, F2 y F3')
    frame = pl.read_parquet(run/'source/engine1.parquet').filter(pl.col('date').is_in(axis))
    counts = frame.group_by('taxonomy').agg(pl.col('account_id').n_unique().alias('candidatas'))
    candidates = frame.filter(pl.col('taxonomy')=='cuentas_entrenamiento').sort('account_id','date')
    matches = [aid for (aid,),g in candidates.group_by('account_id')
               if g['date'].to_list() == axis]
    if len(matches) < config['requested_accounts']:
        report = dict(solicitadas=config['requested_accounts'],aptas=len(matches),
                      cuentas_por_taxonomia=counts.to_dicts(),limite_athena=config['aws']['account_limit'])
        write(run/'muestra_insuficiente.json',report)
        raise ValueError(f"Hay {len(matches)} series entrenables completas y se requieren "
                         f"{config['requested_accounts']}. Consulta muestra_insuficiente.json y amplía el pool de Athena.")
    complete = candidates.filter(pl.col('account_id').is_in(matches))
    universe = AccountingUniverse(complete, [], '', 'Engine 1')
    sampling_spec = SimpleNamespace(minimum_history=len(axis), cv_folds=3,
                                    validation=spec.horizon, evaluation_span=0, seed=spec.seed)
    sampled = sample_universe(universe, sampling_spec, config['requested_accounts'], True, axis)
    groups = sorted(universe.history.group_by('account_id'),key=lambda g:g[0][0])
    series = []
    for (aid,), rows in groups:
        values = rows.select('balance','variation').to_numpy()
        budget = min(10, int(weighted_entropy(values[:initial,0])*10)+1)
        series.append(Series(aid,axis,values,rows['currency'][0],budget,'seen'))
    if len(series) != config['requested_accounts']:
        raise ValueError("El total entrenable no coincide con la muestra solicitada")
    return axis, series, {'solicitadas':config['requested_accounts'],'entrenadas':len(series),
                          'cuentas_por_taxonomia_en_ingesta':counts.to_dicts(),
                          'sin_eje_completo':candidates['account_id'].n_unique()-len(matches),
                          'seleccion':sampled}


def make_frame(model, reference, observed, series, dates, candidate, fold, origin, scales):
    n,h = len(series),len(dates)
    targets = (candidate.target,) if getattr(candidate, 'target', None) else TARGETS
    tile = len(targets)*len(COMPONENTS)
    obs = np.repeat(observed[..., None], len(COMPONENTS), axis=-1).ravel().tolist() if observed is not None else None
    data = pl.DataFrame({
        'account_id':np.repeat([s.id for s in series],h*tile).tolist(),
        'currency':np.repeat([s.currency for s in series],h*tile).tolist(),
        'date':np.tile(np.repeat(dates,tile),n).tolist(),
        'h':np.tile(np.repeat(np.arange(1,h+1),tile),n),
        'target':np.tile(np.repeat(targets,len(COMPONENTS)),n*h).tolist(),
        'component':np.tile(COMPONENTS,n*h*len(targets)).tolist(),
        'observed':obs if obs is not None else [None]*(n*h*tile),
        'scale':np.broadcast_to(scales[:,None,:,None],(n,h,len(targets),len(COMPONENTS))).ravel(),
    },schema_overrides={'observed':pl.Float64})
    return pl.concat([
        data.with_columns(pl.lit(candidate.id).alias('method'),pl.Series('forecast',model.ravel())),
        data.with_columns(pl.lit(REFERENCE).alias('method'),pl.Series('forecast',reference.ravel())),
    ]).with_columns(
        pl.lit(fold).alias('fold'),pl.lit(origin).alias('origin'),
        pl.lit(candidate.inputs).alias('input'),pl.lit(candidate.encoder_design).alias('encoder'),
    )


def train_job(root, series, dates, end, spec, candidate, fold, device, calendar, config):
    label = f'{candidate.id}_{fold}'
    output = root/'predicciones'/f'{label}.parquet'
    output.parent.mkdir(exist_ok=True)
    h = spec.horizon
    if output.is_file():
        frame = pl.read_parquet(output)
        if frame.height != 2*len(series)*h*len(TARGETS)*len(COMPONENTS):
            raise ValueError(f'Tamaño inesperado en {output}')
        print(f'Reutilizado · {label}',flush=True)
        return frame,output
    known = [Series(s.id, dates[:end+h],s.values[:end],s.currency,s.budget,s.cohort) for s in series]
    print(f'Ajustando {label} con {len(known)} series',flush=True)
    model = fit_curves(known,end,spec,candidate,device,calendar,config['curvas'])
    prediction,reference = predict_curves(model,known,end,spec,candidate,device,calendar)
    if fold == 'FUTURO':
        observed = None
    else:
        observed = np.stack([s.values[end:end+h] for s in series])
    history = np.stack([s.values[end-spec.window:end] for s in series])
    scales = np.maximum(np.maximum(np.median(np.abs(history),axis=1),
                         np.quantile(history,.75,axis=1)-np.quantile(history,.25,axis=1)),1.)
    frame = make_frame(prediction,reference,observed,series,dates[end:end+h],candidate,fold,dates[end-1],scales)
    temp = output.with_suffix('.tmp')
    frame.write_parquet(temp,compression='zstd')
    temp.replace(output)
    return frame,output


def run_study(run, config_path, device):
    config = read(config_path)
    if not isinstance(config['requested_accounts'],int) or config['requested_accounts']<1:
        raise ValueError('requested_accounts debe ser un entero positivo')
    spec = spec_for(config)
    torch.set_num_threads(spec.cpu_threads)
    device = torch.device(device)
    root = run/'curvas'
    root.mkdir(exist_ok=True)
    identity = dict(config=config,device=str(device),
                    codigo={f:sha(ROOT/f) for f in ('curvas_modelo.py','curvas_metricas.py',
                                                    'curvas_contrato.py','entrenar_curvas.py')},
                    calendario=sha(ROOT/'data/calendar_authoritative.csv'))
    if (root/'identidad.json').is_file() and read(root/'identidad.json')!=identity:
        raise ValueError('El diseño de la corrida cambió. Usa una carpeta nueva.')
    write(root/'identidad.json',identity)
    execute(run,config_path,'ingesta',str(device))
    axis,series,sample = prepare(run,config,spec)
    calendar = read_calendar(ROOT/'data/calendar_authoritative.csv')
    dates = axis+calendar.after(axis[-1],spec.horizon)
    calendar.take(dates)
    folds = [('F1',len(axis)-3*spec.horizon),('F2',len(axis)-2*spec.horizon),
             ('F3',len(axis)-spec.horizon)]
    candidates = [SimpleNamespace(inputs=inp, encoder_design=enc, depth=d, width=w,
                                  id=f"{inp}_{enc}_L{d}_H{w}")
                  for inp in spec.inputs for enc in spec.encoder_designs
                  for d in config["curvas"]["depths"] for w in config["curvas"]["widths"]]
    records = []
    fit_count = len(folds)*len(candidates)+1
    for fold,end in folds:
        for candidate in candidates:
            frame,path = train_job(root,series,dates,end,spec,candidate,fold,device,calendar,config)
            metrics = aggregate(frame)
            records.append(dict(fold=fold,input=candidate.inputs,encoder=candidate.encoder_design,
                                parametros_entrenables=sum(p.numel() for p in CurveTFT(spec,candidate,config['curvas']['dropout']).parameters() if p.requires_grad),
                                modelo=candidate.id,capas=candidate.depth,ancho=candidate.width,series_entrenadas=len(series),origen=dates[end-1],
                                desde=dates[end],hasta=dates[end+spec.horizon-1],
                                predicciones=str(path.relative_to(root)),**metrics))
            write(run/'estado_curvas.json',dict(estado='EN_CURSO',ajustes=len(records),total=fit_count))
    ranking = sorted(
        [dict(modelo=c.id,input=c.inputs,encoder=c.encoder_design,
              score_f1_f2_f3=sum(r['score'] for r in records if r['modelo']==c.id)/3)
         for c in candidates], key=lambda row:(row['score_f1_f2_f3'],row['modelo'])
    )
    best = next(c for c in candidates if c.id == ranking[0]['modelo'])
    write(root/'seleccion_f1_f2_f3.json',dict(elegido=best.id,ranking=ranking))
    _,future = train_job(root,series,dates,len(axis),spec,best,'FUTURO',device,calendar,config)
    manifest = dict(kind=KIND,run_id=run.name,accounts=len(series),horizonte=spec.horizon,
                    cuantiles_ventana=spec.horizon,reporte=RESULT,
                    source_kind='Engine 1 exportación' if config.get('engine1_path') else 'Athena',
                    engine1_path=config.get('engine1_path'))
    report = dict(
        version='0.10.1',corrida=run.name,muestra=sample,horizonte=spec.horizon,
        diseno=dict(modelo='TFT compacto global directo, dos cabezas de tres cuantiles',
                    inputs=list(spec.inputs),encoders=list(spec.encoder_designs),
                    parametros=config['model'],perdida=config['curvas'],
                    metrica='Media F1-F2-F3 de pinball normalizado',
                    rango_score='[0,infinito); menor es mejor',
                    cuantiles='Q2.5, Q50, Q97.5 predictivos por fecha',
                    referencia='Cuantiles empíricos de la historia al corte',
                    observados='Entrenamiento y evaluación cronológicos en F1–F3 retrospectivos',
                    sha256_engine1=sha(run/'source/engine1.parquet'),sha256_calendario=identity['calendario']),
        seleccion=dict(f1_f2_f3=read(root/'seleccion_f1_f2_f3.json'),mejor_tft=best.id,
                       referencia_recomendada_en_esta_prueba=best.id),
        ajustes=records,
        futuro=dict(estado='ESPERANDO_OBSERVADOS',modelo=best.id,origen=axis[-1],
                    fechas=dates[-spec.horizon:],predicciones=str(future.relative_to(root))),
    )
    write(root/RESULT,report)
    write(root/'manifest.json',manifest)
    write(ROOT/'active_curvas.json',dict(run=str(root)))
    write(run/'estado_curvas.json',dict(estado='COMPLETO',ajustes=fit_count,total=fit_count,reporte=str(root/RESULT)))
    print('Informe final',root/RESULT,flush=True)
    return root/RESULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    with (run/'curvas.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(run/'estado_curvas.json',dict(estado='EN_CURSO',ajustes=0))
        try:
            from arena_estudio import run_study as arena_run
            arena_run(run,args.config.resolve(),args.device)
        except BaseException as error:
            last = read(run/'estado_curvas.json')
            write(run/'estado_curvas.json',{**last,'estado':'ERROR','detalle':str(error)})
            raise


if __name__=='__main__':
    main()
