"""Temporal competition, persisted candidates, and winner-only inference."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import pickle
import numpy as np
import optuna
import polars as pl
import torch
from entrenar_curvas import (ROOT, read, write, spec_for, execute, sha, make_frame)
from agentic_forecasting import Series
from agentic_forecasting.data import weighted_entropy
from model_arena.winner_selection import assignments
from curvas_modelo import CurveTFT, fit_curves, predict_curves
from curvas_metricas import REFERENCE
from curvas_contrato import KIND, RESULT
from observatorio_cuentas.calendar import read_calendar


def population(run, config, spec):
    axis = read(run/'source/operational_axis.json')['dates']
    axis = axis[-config['observations']:] if config.get('observations') else axis
    initial = len(axis)-3*spec.horizon
    if initial < spec.window+spec.horizon:
        raise ValueError('Historia insuficiente para tres folds')
    data = pl.read_parquet(run/'source/engine1.parquet').filter(pl.col('date').is_in(axis)).sort('account_id','date')
    seen, atypical, inventory, exclusions = [], [], [], []
    for (aid,), group in data.group_by('account_id', maintain_order=True):
        tax = group['taxonomy'][0]
        if tax not in ('cuentas_entrenamiento','cuentas_atipicos'):
            continue
        values = group.select('balance','variation').to_numpy()
        if group['date'].to_list() != axis or not np.isfinite(values).all():
            exclusions.append(dict(account_id=aid,taxonomy=tax,reason='EJE_INCOMPLETO_O_NO_FINITO'))
            continue
        budget = min(10,int(weighted_entropy(values[:initial,0])*10)+1)
        cohort = 'seen' if tax == 'cuentas_entrenamiento' else 'zero_shot'
        item = Series(aid,axis,values,group['currency'][0],budget,cohort)
        (seen if cohort == 'seen' else atypical).append(item)
        inventory.append(dict(account_id=aid,currency=item.currency,budget=budget,cohort=cohort,taxonomy=tax))
    write(run/'elegibilidad.json',dict(exclusiones=exclusions,entrenables=len(seen),atipicas=len(atypical)))
    if not seen:
        raise ValueError('Sin entrenables elegibles, revisar elegibilidad.json')
    return axis, seen, atypical, pl.DataFrame(inventory)


def candidate_for(trial, inp, cfg):
    return SimpleNamespace(id=trial.user_attrs.get('resume_candidate_id', f'{inp}_T{trial.number:04d}'),inputs=inp,
        encoder_design=trial.suggest_categorical('encoder',cfg['encoders']),
        depth=trial.suggest_categorical('encoder_depth',cfg['depths']),
        decoder_depth=trial.suggest_categorical('decoder_depth',cfg['decoder_depths']),
        width=trial.suggest_categorical('width',cfg['widths']),
        activation=trial.suggest_categorical('activation',cfg['activations']),
        lr=trial.suggest_float('lr',*cfg['lr_range'],log=True),
        weight_decay=trial.suggest_float('weight_decay',*cfg['wd_range'],log=True))


def loss_rows(frame, candidate, inventory):
    error = (pl.col('observed')-pl.col('forecast'))/pl.col('scale')
    tau = pl.col('component').replace_strict({'Q2.5':.025,'Q50':.5,'Q97.5':.975})
    return (frame.filter(pl.col('method') != REFERENCE)
        .with_columns(pl.max_horizontal(tau*error,(tau-1)*error).alias('pinball'))
        .group_by('account_id','currency','target','fold').agg(pl.col('pinball').mean().alias('score'))
        .join(inventory.select('account_id','budget','cohort'),on='account_id',validate='m:1')
        .with_columns(pl.lit(candidate.id).alias('candidate'),pl.lit(candidate.inputs).alias('input')))


def persist_history(root, candidate, fold, rows):
    directory = root/'loss';directory.mkdir(exist_ok=True)
    path = directory/f'{candidate}_{fold}.csv';temp=path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['candidate','fold','epoch','train_loss','validation_loss','train_loss_saldo','train_loss_variacion','fixed_train_saldo','fixed_train_variacion','fixed_validation_saldo','fixed_validation_variacion'])
        writer.writeheader()
        writer.writerows(dict(candidate=candidate,fold=fold,**row) for row in rows)
    temp.replace(path)


def combine_history(root):
    frames=[pl.read_csv(path,schema_overrides={k:pl.Float64 for k in ['validation_loss','fixed_train_saldo','fixed_train_variacion','fixed_validation_saldo','fixed_validation_variacion']}) for path in sorted((root/'loss').glob('*.csv'))]
    if not frames:raise ValueError('No se generaron curvas de entrenamiento')
    combined=pl.concat(frames).sort('candidate','fold','epoch')
    path=root/'training_history.parquet';temp=path.with_suffix('.tmp')
    combined.write_parquet(temp,compression='zstd');temp.replace(path)
    return path


def fit_saved(root, seen, dates, end, spec, candidate, fold, device, calendar, cfg):
    if fold=='FUTURO':spec=replace(spec,epochs=int(cfg.get('final_epochs',spec.epochs)))
    if spec.epochs<1:raise ValueError('Épocas debe ser positivo')
    path = root/'modelos'/f'{candidate.id}_{fold}.pt'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        saved = torch.load(path,map_location=device,weights_only=False)
        model = CurveTFT(spec,candidate,cfg['dropout']).to(device)
        model.load_state_dict(saved['state_dict'])
        history = saved.get('training_history')
        if not history:
            raise ValueError('El checkpoint no contiene curvas de pérdida; usa una corrida de esta entrega')
        persist_history(root, candidate.id, fold, history)
        return model
    known = [Series(s.id,dates,s.values[:end],s.currency,s.budget,s.cohort) for s in seen]
    history = []
    def on_epoch(row):
        history.append(row)
        persist_history(root, candidate.id, fold, history)
    print(f'Ajustando {candidate.id} · {fold} · {len(known)} cuentas', flush=True)
    from monitor_learning import build_monitor
    monitor=build_monitor(known,seen,end,spec,candidate,device,calendar,fold,cfg,detail_path=root/"loss_entities"/f"{candidate.id}_{fold}.csv")
    model = fit_curves(known,end,spec,candidate,device,calendar,cfg,on_epoch=on_epoch,monitor=monitor)
    temp = path.with_suffix('.tmp')
    torch.save(dict(state_dict=model.state_dict(),candidate=vars(candidate),spec=asdict(spec),
                    training_history=history,loss_monitor=monitor.selection,end=end,fold=fold,scaling='asinh(history/scale); forecast*scale',version='0.11.0',training_account_ids=[s.id for s in seen]),temp)
    temp.replace(path)
    return model


def emit(model, series, dates, end, spec, candidate, fold, device, calendar):
    known = [Series(s.id,dates,s.values[:end],s.currency,s.budget,s.cohort) for s in series]
    forecast, baseline = predict_curves(model,known,end,spec,candidate,device,calendar)
    observed = None if fold == 'FUTURO' else np.stack([s.values[end:end+spec.horizon] for s in series])
    history = np.stack([s.values[end-spec.window:end] for s in series])
    scales = np.maximum(np.maximum(np.median(np.abs(history),axis=1),np.quantile(history,.75,axis=1)-np.quantile(history,.25,axis=1)),1.)
    return make_frame(forecast,baseline,observed,series,dates[end:end+spec.horizon],candidate,fold,dates[end-1],scales)



def run_study(run, config_path, device, *, stage="all"):
    config = read(config_path)
    if config["requested_accounts"] != 0:
        raise ValueError("Esta versión entrena el universo elegible completo. Usa requested_accounts=0")
    spec = spec_for(config)
    torch.set_num_threads(spec.cpu_threads)
    device = torch.device(device)
    root = run/'curvas';root.mkdir(exist_ok=True)
    identity = dict(config=config,codigo={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'arena_estudio.py',ROOT/'curvas_modelo.py',*list((ROOT/'libraries').rglob('*.py'))]})
    if (root/'identidad.json').exists() and read(root/'identidad.json')!=identity:
        raise ValueError('Usa una corrida nueva, cambió configuración o código')
    write(root/'identidad.json',identity)
    execute(run,config_path,'ingesta',str(device))
    axis, seen, atypical, inventory = population(run,config,spec)
    inventory.write_parquet(root/'inventory.parquet')
    calendar = read_calendar(ROOT/'data/calendar_authoritative.csv')
    dates = axis+calendar.after(axis[-1],spec.horizon)
    calendar.take(dates)
    folds = [(f'F{i+1}',len(axis)-(3-i)*spec.horizon) for i in range(3)]
    cfg=config['curvas'];candidates=[];score_frames=[];trials=[]
    (root/'predicciones').mkdir(exist_ok=True)
    for inp in cfg['inputs']:
        sampler_path=root/f'{inp}_sampler.pkl'
        sampler=pickle.loads(sampler_path.read_bytes()) if sampler_path.exists() else optuna.samplers.TPESampler(seed=spec.seed+cfg['inputs'].index(inp))
        study=optuna.create_study(study_name=inp,storage='sqlite:///'+str(root/'hpo.sqlite3'),load_if_exists=True,direction='minimize',sampler=sampler,pruner=optuna.pruners.NopPruner())
        def objective(trial):
            candidate=candidate_for(trial,inp,cfg)
            trial.set_user_attr('candidate_id',candidate.id)
            write(root/f'{candidate.id}.json',vars(candidate))
            scores=[]
            for fold,end in folds:
                path=root/'predicciones'/f'{candidate.id}_{fold}.parquet'
                if path.exists():
                    frame=pl.read_parquet(path)
                else:
                    model=fit_saved(root,seen,dates,end,spec,candidate,fold,device,calendar,cfg)
                    frame=emit(model,seen,dates,end,spec,candidate,fold,device,calendar)
                    temp=path.with_suffix('.tmp');frame.write_parquet(temp);temp.replace(path)
                scores.append(loss_rows(frame,candidate,inventory))
            table=pl.concat(scores);table.write_parquet(root/f'{candidate.id}_scores.parquet')
            return table['score'].mean()
        # Interrupted trials can be retried with their exact parameters and paths.
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
            study.enqueue_trial(trial.params,user_attrs={'resume_candidate_id':trial.user_attrs.get('candidate_id',f'{inp}_T{trial.number:04d}')})
        remaining=cfg['trials']-len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
        def persist_sampler(study, trial):
            temp=sampler_path.with_suffix('.tmp')
            temp.write_bytes(pickle.dumps(study.sampler));temp.replace(sampler_path)
        if remaining>0:study.optimize(objective,n_trials=remaining,callbacks=[persist_sampler])
        for trial in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            candidate_id=trial.user_attrs.get('candidate_id',f'{inp}_T{trial.number:04d}')
            candidate=SimpleNamespace(**read(root/f'{candidate_id}.json'))
            candidates.append(candidate)
            score_frames.append(pl.read_parquet(root/f'{candidate.id}_scores.parquet'))
            trials.append(dict(**vars(candidate),score=trial.value))
        write(run/'estado_curvas.json',dict(estado='HPO',input=inp,trials_completos=len(trials)))
    scores=pl.concat(score_frames);scores.write_parquet(root/'scores.parquet')
    write(root/'candidatos.json',trials)
    for name, groups in [('finalists_budget.parquet',['budget','input']),('finalists_total.parquet',['input'])]:
        finalists=scores.group_by(*groups,'candidate').agg(pl.col('score').mean()).sort(*groups,'score','candidate').unique(groups,keep='first')
        finalists.write_parquet(root/name)
    if stage == "tune":
        combine_history(root)
        return root
    records=[];prior=[]
    for fold,end in [*folds,('FUTURO',len(axis))]:
        chosen=assignments(scores,inventory,candidates,prior)
        chosen.write_parquet(root/f'winners_{fold}.parquet')
        pieces=[]
        for candidate in candidates:
            ids=set(chosen.filter(pl.col('candidate')==candidate.id)['account_id'])
            if not ids:continue
            model=fit_saved(root,seen,dates,end,spec,candidate,fold,device,calendar,cfg)
            group=[s for s in [*seen,*atypical] if s.id in ids]
            frame=emit(model,group,dates,end,spec,candidate,fold,device,calendar)
            pieces.append(frame.with_columns(pl.col('method').alias('candidate_model'),
                pl.when(pl.col('method')==REFERENCE).then(pl.lit(REFERENCE)).otherwise(pl.lit('Winner')).alias('method')))
        output=root/'predicciones'/f'Winner_{fold}.parquet'
        pl.concat(pieces).write_parquet(output)
        records.append(dict(modelo='Winner',fold=fold,origen=dates[end-1],desde=dates[end],hasta=dates[end+spec.horizon-1],predicciones=str(output.relative_to(root))))
        prior.append(fold)
    training_history=combine_history(root)
    report=dict(version='0.11.0',training_history=str(training_history.relative_to(root)),loss_function='pinball normalizado; tres cuantiles y dos objetivos',validation_loss_available=False,corrida=run.name,horizonte=spec.horizon,
        muestra=dict(entrenadas=len(seen),atipicas_zero_shot=len(atypical)),ajustes=records[:-1],futuro=records[-1],
        seleccion=dict(mejor_tft='Winner',referencia_recomendada_en_esta_prueba='Winner'),config=config)
    write(root/RESULT,report)
    write(root/'manifest.json',dict(kind=KIND,run_id=run.name,accounts=inventory.height,horizonte=spec.horizon))
    write(ROOT/'active_curvas.json',dict(run=str(root)))
    write(run/'estado_curvas.json',dict(estado='COMPLETO',reporte=str(root/RESULT)))
    return root/RESULT
