"""Evaluación pareada y selección cronológica de las ocho curvas."""

import polars as pl
from curvas_contrato import COMPONENTS, TARGETS

REFERENCE = 'Persistencia'


def paired(frame):
    by_day = [
        'input','encoder','fold','origin','date','h','account_id',
        'currency','target','component',
    ]
    evaluated = frame.filter(pl.col('observed').is_not_null())
    original = evaluated.filter(pl.col('method') != REFERENCE)
    reference = evaluated.filter(pl.col('method') == REFERENCE).select(
        *by_day, pl.col('forecast').alias('reference')
    )
    points = original.join(reference, on=by_day, validate='1:1')
    if points.height != original.height or points.height != reference.height:
        raise ValueError('Predicciones y referencia incompletas')
    daily = points.with_columns(
        (pl.col('forecast')-pl.col('observed')).abs().alias('error'),
        (pl.col('reference')-pl.col('observed')).abs().alias('error_ref'),
        (pl.col('forecast')-pl.col('observed')).alias('signed_error'),
    )
    account = daily.group_by(
        'input','encoder','fold','account_id','currency','target','component'
    ).agg(
        mae=pl.col('error').mean(), mae_ref=pl.col('error_ref').mean(),
        sesgo=pl.col('signed_error').mean(), fechas=pl.len(),
    ).with_columns(
        (pl.col('mae')-pl.col('mae_ref')).alias('delta'),
        ((pl.col('mae')-pl.col('mae_ref'))/(pl.col('mae')+pl.col('mae_ref'))).fill_nan(0.).alias('delta_simetrico'),
    )
    return account, daily


def aggregate(frame):
    account, daily = paired(frame)
    totals = account.group_by('target','component').agg(
        cuentas=pl.len(),
        ganan=(pl.col('delta') < -1e-6).sum(),
        empatan=(pl.col('delta').abs() <= 1e-6).sum(),
        persistencia_exacta=(pl.col('mae_ref') == 0).sum(),
        cambio_simetrico_medio=pl.col('delta_simetrico').mean(),
        mediana_cambio_mae_pct=(
            100*pl.col('delta')/pl.when(pl.col('mae_ref')>0).then(pl.col('mae_ref'))
        ).median(),
    ).sort('target','component')
    by_currency = account.group_by('currency','target','component').agg(
        cuentas=pl.len(),ganan=(pl.col('delta') < -1e-6).sum(),
        mediana_mae_modelo=pl.col('mae').median(),mediana_mae_persistencia=pl.col('mae_ref').median(),
        mediana_delta=pl.col('delta').median(),p90_delta=pl.col('delta').quantile(.9,interpolation='linear'),
        mediana_sesgo=pl.col('sesgo').median(),
    ).sort('currency','target','component')
    by_step = daily.with_columns(
        ((pl.col('error')-pl.col('error_ref'))/(pl.col('error')+pl.col('error_ref'))).fill_nan(0.).alias('delta'),
    ).group_by('h','target','component').agg(
        cuentas=pl.len(),ganan=(pl.col('error') < pl.col('error_ref') - 1e-6).sum(),
        cambio_simetrico_medio=pl.col('delta').mean(),
    ).sort('h','target','component')
    diagnostics = []
    for target in TARGETS:
        subset = frame.filter(pl.col('target') == target)
        for method in subset['method'].unique().sort():
            wide = subset.filter(pl.col('method') == method).pivot(
                on='component', index=['account_id','date'], values='forecast')
            actual = subset.filter((pl.col('method') == method) & (pl.col('component') == 'Q50')).select(
                'account_id','date',pl.col('observed').alias('y'))
            wide = wide.join(actual,on=['account_id','date']).sort('account_id','date')
            loss = {}
            for name,q in [('Q2.5',.025),('Q50',.5),('Q97.5',.975)]:
                e = pl.col('y')-pl.col(name)
                loss[name] = pl.max_horizontal(q*e,(q-1)*e).mean()
            per = wide.group_by('account_id').agg(
                rmse_q50=((pl.col('y')-pl.col('Q50'))**2).mean().sqrt(),
                coverage=pl.col('y').is_between(pl.col('Q2.5'),pl.col('Q97.5')).mean(),
                width=(pl.col('Q97.5')-pl.col('Q2.5')).mean(), **loss).sort('account_id')
            diagnostics.append(dict(target=target,method=method,**per.drop('account_id').mean().row(0,named=True)))
    return {**selection_loss(frame),'diagnostico_predictivo':diagnostics,'por_curva':totals.to_dicts(),'por_divisa':by_currency.to_dicts(),'por_paso':by_step.to_dicts()}


def curve_score(evaluation):
    return evaluation['score']


def selection_loss(frame):
    """Score por observado real, normalizado con historia anterior al corte."""
    import numpy as np
    components = {}
    for target in TARGETS:
        rows = frame.filter((pl.col('target') == target) & (pl.col('method') != REFERENCE))
        losses = []
        for name, q in [('Q2.5', .025), ('Q50', .5), ('Q97.5', .975)]:
            error = (pl.col('observed') - pl.col('forecast')) / pl.col('scale')
            item = rows.filter(pl.col('component') == name)
            losses.append(item.select(pl.max_horizontal(q*error, (q-1)*error).mean()).item())
        pinball = float(np.mean(losses))
        components[target] = dict(pinball_normalizado=pinball, loss=pinball)
    return dict(score=float(np.mean([v['loss'] for v in components.values()])), loss_components=components)
