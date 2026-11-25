"""Temporal competition, persisted candidates, and winner-only inference."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import numpy as np
import optuna
import polars as pl
import torch
from model_arena.disk import check_space, DiskSpaceError
from model_arena.hpo_resume import enqueue_interrupted
from model_arena.pruning import make_pruner, report_epoch, trial_records, enqueue_encoder_schedule
from model_arena.loss_store import write_parquet
from entrenar_curvas import (ROOT, read, write, spec_for, execute, sha, make_frame)
from agentic_forecasting import Series
from agentic_forecasting.data import weighted_entropy
from model_arena.winner_selection import assignments
from model_arena.models.engine import temporal_folds
from model_arena.input_spec import study_plan
from model_arena.scoring import pinball_error
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
        budget = 1 if config['curvas'].get('policies', {}).get('segmenter') == 'none' else min(10,int(weighted_entropy(values[:initial,0])*10)+1)
        cohort = 'seen' if tax == 'cuentas_entrenamiento' else 'zero_shot'
        item = Series(aid,axis,values,group['currency'][0],budget,cohort)
        (seen if cohort == 'seen' else atypical).append(item)
        inventory.append(dict(account_id=aid,currency=item.currency,budget=budget,cohort=cohort,taxonomy=tax))
    write(run/'elegibilidad.json',dict(exclusiones=exclusions,entrenables=len(seen),atipicas=len(atypical)))
    if not seen:
        raise ValueError('Sin entrenables elegibles, revisar elegibilidad.json')
    return axis, seen, atypical, pl.DataFrame(inventory)


def candidate_for(trial, inp, cfg):
    family=cfg.get('family','encoder_decoder')
    if family != 'encoder_decoder':
        from model_arena.models.complete import suggest_parameters
        from model_arena.contracts import SearchSpec
        space=SearchSpec(width=cfg['widths'],depth=cfg['depths'],decoder_depth=cfg['decoder_depths'],
            lr_range=cfg['lr_range'],wd_range=cfg['wd_range'],dropout=cfg['dropout'])
        params=suggest_parameters(family,trial,space)
        return SimpleNamespace(id=trial.user_attrs.get('resume_candidate_id',f'{inp}_{family}_T{trial.number:04d}'),
            inputs=inp,family=family,family_params=params,encoder_kind=family,encoder_design='independent',
            lr=params['lr'],weight_decay=params['weight_decay'])
    return SimpleNamespace(id=trial.user_attrs.get('resume_candidate_id', f'{inp}_T{trial.number:04d}'),inputs=inp,family='encoder_decoder',
        encoder_kind=trial.suggest_categorical('encoder_kind',cfg['encoder_kinds']) if 'encoder_kinds' in cfg else 'tft_compact',
        encoder_design='single' if cfg.get('target') else trial.suggest_categorical('encoder',cfg['encoders']),
        depth=trial.suggest_categorical('encoder_depth',cfg['depths']),
        decoder_depth=trial.suggest_categorical('decoder_depth',cfg['decoder_depths']),
        width=trial.suggest_categorical('width',cfg['widths']),
        activation=trial.suggest_categorical('activation',cfg['activations']),
        lr=trial.suggest_float('lr',*cfg['lr_range'],log=True),
        weight_decay=trial.suggest_float('weight_decay',*cfg['wd_range'],log=True))


def loss_rows(frame, candidate, inventory):
    error = (pl.col('observed')-pl.col('forecast'))/pl.col('scale')
    tau = pl.col('component').replace_strict({'Q2.5':.025,'Q50':.5,'Q97.5':.975})
    frame = frame.filter(pl.col('method') != REFERENCE)
    if getattr(candidate, 'selection_metric', None) == 'scaled_mae':
        frame = frame.filter(pl.col('component') == 'Q50')
        loss = error.abs()
    else:
        loss = pinball_error(error, tau, maximum=pl.max_horizontal)
    return (frame.with_columns(loss.alias('score'))
        .group_by('account_id','currency','target','fold').agg(pl.col('score').mean())
        .join(inventory.select('account_id','budget','cohort'),on='account_id',validate='m:1')
        .with_columns(pl.lit(candidate.id).alias('candidate'),pl.lit(candidate.inputs).alias('input'),
            pl.lit(getattr(candidate,'family','encoder_decoder')).alias('family')))


def persist_history(root, candidate, fold, rows):
    frame = pl.DataFrame([dict(candidate=candidate,fold=fold,**row) for row in rows],
        schema_overrides={name:pl.Float64 for name in ["validation_loss","fixed_train_saldo",
        "fixed_train_variacion","fixed_validation_saldo","fixed_validation_variacion"]})
    write_parquet(frame, root/'loss'/f'{candidate}_{fold}.parquet')


def combine_history(root):
    files = sorted((root/'loss').glob('*.parquet'))
    stems = {p.stem for p in files}
    frames = [pl.read_parquet(path) for path in files]
    frames += [pl.read_csv(path,schema_overrides={k:pl.Float64 for k in
        ['validation_loss','fixed_train_saldo','fixed_train_variacion','fixed_validation_saldo','fixed_validation_variacion']})
        for path in sorted((root/'loss').glob('*.csv')) if path.stem not in stems]
    if not frames:raise ValueError('No se generaron curvas de entrenamiento')
    combined=pl.concat(frames,how='diagonal_relaxed').sort('candidate','fold','epoch')
    path=root/'training_history.parquet'
    write_parquet(combined,path)
    return path


def fit_saved(root, seen, dates, end, spec, candidate, fold, device, calendar, cfg, *, trial=None, fold_index=0):
    if fold=='FUTURO':spec=replace(spec,epochs=int(cfg.get('final_epochs',spec.epochs)))
    if spec.epochs<1:raise ValueError('Épocas debe ser positivo')
    path = root/'modelos'/f'{candidate.id}_{fold}.pt'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        saved = torch.load(path,map_location=device,weights_only=False)
        model = CurveTFT(spec,candidate,cfg['dropout']).to(device)
        if hasattr(model,'external') and saved.get('upstream_provenance') != model.external.provenance:
            raise ValueError('Cambió la implementación upstream del checkpoint')
        if cfg.get('calendar_sha256') and (
            saved.get('calendar_sha256') != calendar.source_sha256
            or saved.get('candidate', {}).get('input_features') != getattr(candidate, 'input_features', None)
        ):
            raise ValueError('Checkpoint incompatible con el esquema de calendario del candidato')
        state = saved['state_dict']
        if not hasattr(model, 'external') and state and not next(iter(state)).startswith('network.'):
            state = {'network.' + key: value for key, value in state.items()}
        model.load_state_dict(state)
        history = saved.get('training_history')
        if not history:
            raise ValueError('El checkpoint no contiene curvas de pérdida; usa una corrida de esta entrega')
        persist_history(root, candidate.id, fold, history)
        return model
    check_space(root, stage=f"ajuste {candidate.id} {fold}")
    known = [Series(s.id,dates,s.values[:end],s.currency,s.budget,s.cohort) for s in seen]
    history = []
    def on_epoch(row):
        history.append(row)
        persist_history(root, candidate.id, fold, history)
        if trial is not None:
            value = float(row['fixed_validation_'+candidate.target]) if getattr(candidate,'target',None) else float(np.mean([row['fixed_validation_saldo'],row['fixed_validation_variacion']]))
            report_epoch(trial,value,epoch=row['epoch'],fold_index=fold_index,
                epochs_per_fold=spec.epochs,fold=fold)
    print(f'Ajustando {candidate.id} · {fold} · {len(known)} cuentas', flush=True)
    from monitor_learning import build_monitor
    monitor=build_monitor(known,seen,end,spec,candidate,device,calendar,fold,cfg,detail_path=root/"loss_entities"/f"{candidate.id}_{fold}")
    model = fit_curves(known,end,spec,candidate,device,calendar,cfg,on_epoch=on_epoch,monitor=monitor)
    temp = path.with_suffix('.tmp')
    check_space(root, extra_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())*2, stage="checkpoint")
    torch.save(dict(upstream_provenance=model.external.provenance if hasattr(model,'external') else None,state_dict=model.state_dict(),candidate=vars(candidate),spec=asdict(spec),calendar_sha256=calendar.source_sha256,
                    training_history=history,loss_monitor=monitor.selection,end=end,fold=fold,scaling='asinh(history/scale); forecast*scale',version='0.11.0',training_account_ids=[s.id for s in seen]),temp)
    temp.replace(path)
    return model


def emit(model, series, dates, end, spec, candidate, fold, device, calendar):
    known = [Series(s.id,dates,s.values[:end],s.currency,s.budget,s.cohort) for s in series]
    forecast, baseline = predict_curves(model,known,end,spec,candidate,device,calendar)
    observed = None if fold == 'FUTURO' else np.stack([s.values[end:end+spec.horizon] for s in series])
    history = np.stack([s.values[end-spec.window:end] for s in series])
    scales = np.maximum(np.maximum(np.median(np.abs(history),axis=1),np.quantile(history,.75,axis=1)-np.quantile(history,.25,axis=1)),1.)
    if getattr(candidate,'target',None):
        index = ('saldo','variacion').index(candidate.target)
        scales = scales[:,index:index+1]
        if observed is not None:observed = observed[...,index:index+1]
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
    write_parquet(inventory,root/'inventory.parquet')
    calendar = read_calendar(ROOT/'data/calendar_authoritative.csv')
    dates = axis+calendar.after(axis[-1],spec.horizon)
    calendar.take(dates)
    folds = temporal_folds(len(axis), window=spec.window, horizon=spec.horizon, count=3, step=spec.horizon)
    cfg=config['curvas'];candidates=[];score_frames=[];trials=[];trial_audit=[]
    (root/'predicciones').mkdir(exist_ok=True)
    for task in study_plan(cfg, spec.epochs):
        inp,family,budget,target = (task[k] for k in ('input','family','trials','objective'))
        cfg={**config['curvas'],'family':family,'trials':budget,'target':target}
        study_key=inp if 'families' not in cfg else inp+'_'+family
        if target:study_key += '_'+target
        sampler_path=root/f'{study_key}_sampler.pkl'
        sampler=pickle.loads(sampler_path.read_bytes()) if sampler_path.exists() else optuna.samplers.TPESampler(seed=spec.seed+cfg['inputs'].index(inp))
        pruner=make_pruner(cfg.get('pruner'),epochs_per_fold=spec.epochs)
        study=optuna.create_study(study_name=study_key,storage='sqlite:///'+str(root/'hpo.sqlite3'),load_if_exists=True,direction='minimize',sampler=sampler,pruner=pruner)
        if family=='encoder_decoder' and 'encoder_kinds' in cfg:
            enqueue_encoder_schedule(study,cfg['encoder_kinds'],cfg['trials'])
        def objective(trial):
            check_space(root, stage=f'trial {inp} {trial.number}')
            candidate=candidate_for(trial,inp,cfg)
            if 'input_specs' in cfg:
                candidate.input_features = cfg['input_specs'][inp]
                candidate.transform = cfg['policies']['transform']
                candidate.selection_metric = cfg['policies']['selection_metric']
            if 'families' in cfg and family=='encoder_decoder' and 'resume_candidate_id' not in trial.user_attrs:
                candidate.id=f'{inp}_{family}_T{trial.number:04d}'
            if target:
                candidate.target=target
                candidate.encoder_design='single'
                if 'resume_candidate_id' not in trial.user_attrs:candidate.id=study_key+f'_T{trial.number:04d}'
                trial.set_user_attr('target',target)
            trial.set_user_attr('family',family)
            trial.set_user_attr('reference_group',candidate.encoder_kind)
            trial.set_user_attr('candidate_id',candidate.id)
            write(root/f'{candidate.id}.json',vars(candidate))
            scores=[]
            for fold_index,(fold,end) in enumerate(folds):
                path=root/'predicciones'/f'{candidate.id}_{fold}.parquet'
                if path.exists():
                    frame=pl.read_parquet(path)
                else:
                    model=fit_saved(root,seen,dates,end,spec,candidate,fold,device,calendar,cfg,trial=trial,fold_index=fold_index)
                    frame=emit(model,seen,dates,end,spec,candidate,fold,device,calendar)
                    write_parquet(frame,path)
                scores.append(loss_rows(frame,candidate,inventory))
            table=pl.concat(scores);write_parquet(table,root/f'{candidate.id}_scores.parquet')
            return table['score'].mean()
        unguarded_objective=objective
        def objective(trial, implementation=unguarded_objective):
            try:
                return implementation(trial)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                if isinstance(exc, DiskSpaceError) or getattr(exc, "errno", None) == 28:
                    trial.set_user_attr("retry_after_disk", True)
                import traceback
                traceback.print_exc()
                raise
        enqueue_interrupted(study, study_key)
        remaining=cfg['trials']-len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.PRUNED)))
        def persist_sampler(study, trial):
            temp=sampler_path.with_suffix('.tmp')
            temp.write_bytes(pickle.dumps(study.sampler));temp.replace(sampler_path)
            audit=pl.DataFrame(trial_audit+trial_records(study,inp),schema_overrides={
                'objective_value':pl.Float64,'last_intermediate_value':pl.Float64,
                'pruned_fold':pl.String,'pruned_epoch':pl.Int64})
            write_parquet(audit,root/'hpo_trials.parquet')
        if remaining>0:study.optimize(objective,n_trials=remaining,callbacks=[persist_sampler])
        trial_audit.extend(trial_records(study,inp))
        if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            raise ValueError('Sin candidato completo para '+inp+'; revisa hpo_trials.parquet')
        for trial in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            candidate_id=trial.user_attrs.get('candidate_id',f'{inp}_T{trial.number:04d}')
            candidate=SimpleNamespace(**read(root/f'{candidate_id}.json'))
            candidates.append(candidate)
            score_frames.append(pl.read_parquet(root/f'{candidate.id}_scores.parquet'))
            trials.append(dict(**vars(candidate),score=trial.value))
        write(run/'estado_curvas.json',dict(estado='HPO',input=inp,trials_completos=len(trials)))
    write_parquet(pl.DataFrame(trial_audit,schema_overrides={
        'objective_value':pl.Float64,'last_intermediate_value':pl.Float64,
        'pruned_fold':pl.String,'pruned_epoch':pl.Int64}),root/'hpo_trials.parquet')
    scores=pl.concat(score_frames,how='diagonal_relaxed')
    if 'family' in scores.columns:scores=scores.with_columns(pl.col('family').fill_null('encoder_decoder'))
    write_parquet(scores,root/'scores.parquet')
    write(root/'candidatos.json',trials)
    family_dims=(['target'] if config['curvas'].get('objectives') else [])+(['family'] if 'family' in scores.columns else [])
    for name, groups in [('finalists_budget.parquet',['budget',*family_dims,'input']),('finalists_total.parquet',[*family_dims,'input'])]:
        finalists=scores.group_by(*groups,'candidate').agg(pl.col('score').mean()).sort(*groups,'score','candidate').unique(groups,keep='first')
        write_parquet(finalists,root/name)
    if stage == "tune":
        combine_history(root)
        return root
    records=[];prior=[]
    for fold,end in [*folds,('FUTURO',len(axis))]:
        chosen=assignments(scores,inventory,candidates,prior, policy=cfg.get("policies"))
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
