"""Controles y series del monitor sobre los artefactos emitidos por 0.9.5."""
import numpy as np
import plotly.graph_objects as go
import polars as pl

FOLDS = ('fold_1', 'fold_2', 'fold_3')
BLOCK_NAMES = {'Todos':'Todos', 'Tres folds':'Tres folds', 'train_backtest':'Backtesting de train',
               'fold_1':'Fold 1', 'fold_2':'Fold 2', 'fold_3':'Fold 3', 'future':'Horizonte futuro'}

def blocks(selection, available):
    if selection == 'Todos':
        return tuple(available)
    if selection == 'Tres folds':
        return tuple(block for block in FOLDS if block in available)
    return (selection,)

def streaks(path, selection, cutoff, graded):
    from collections import defaultdict
    active = defaultdict(set)
    for event in graded:
        active[event['account_id'], event['target']].add((event['date'], event['block']))
    query = pl.scan_parquet(path).filter(pl.col('block').is_in(selection))
    if cutoff != 'Todas':
        query = query.filter(pl.col('date') <= cutoff)
    observations = query.select('account_id','target','date','block','observed').sort('date').collect()
    result = {}
    for key, frame in observations.partition_by(['account_id','target'], as_dict=True).items():
        days = frame.select('date','block','observed').rows()
        while days and not valid(days[-1][-1]):
            days.pop()
        if days:
            n = 0
            for date, block, value in reversed(days):
                if not valid(value) or (date,block) not in active.get(key, ()):
                    break
                n += 1
            result[key] = n
    return result

def valid(value):
    return value is not None and np.isfinite(value)

FAMILIES = {
    'Observado': ('#235a97', 'rgba(49,98,174,.10)', 'circle', 'solid'),
    'Modelo': ('#007a71', 'rgba(0,134,119,.13)', 'square', 'dash'),
    'Baseline': ('#ba650f', 'rgba(207,125,35,.11)', 'x', 'dot'),
}
def columns(family, model):
    prefix = {'Observado':'hist', 'Modelo':model, 'Baseline':'baseline'}[family]
    return {k: prefix+'_'+v for k,v in [('Q2.5','q025'),('Q50','q50'),('Q97.5','q975')]} | {
        'Puntual': 'observed' if family=='Observado' else prefix+'_point'}

def traces(fig, frame, row, transform, family, model, detail, shaded, legends, model_name):
    color, fill, symbol, dash = FAMILIES[family]
    cols = columns(family, model)
    display = model_name if family == 'Modelo' else family
    dates = frame['date'].to_numpy()
    if shaded:
        low, high = [frame[cols[q]].to_numpy() for q in ('Q2.5','Q97.5')]
        indices = np.flatnonzero(np.isfinite(low) & np.isfinite(high) & (low <= high))
        for run in np.split(indices, np.flatnonzero(np.diff(indices)!=1)+1):
            if not run.size:
                continue
            fig.add_trace(go.Scatter(x=np.r_[dates[run],dates[run[::-1]]],
                          y=transform(np.r_[high[run],low[run[::-1]]]), fill='toself',fillcolor=fill,
                          mode='lines',line_width=0,showlegend=False,hoverinfo='skip'),row=row,col=1)
    if detail:
        series = ('Q2.5','Q97.5','Q50','Puntual')
    elif shaded:
        series = ('Q2.5','Q97.5','Puntual')
    else:
        series = ('Puntual',)
    if family != 'Observado' and cols['Puntual'] not in frame.columns and 'Q50' not in series:
        series = (*series, 'Q50')
    for q in series:
        if cols[q] not in frame.columns:
            continue
        y = frame[cols[q]].to_numpy()
        if not np.isfinite(y).any():
            continue
        label = display+' · '+({'Puntual':'yₜ real' if family=='Observado' else 'ŷₜ puntual'}.get(q,q))
        point = q=='Puntual'
        fig.add_trace(go.Scatter(x=dates,y=transform(y),customdata=y,name=label,
                      legendgroup=label,showlegend=label not in legends,
                      line=dict(color=color,width=2.2 if point else 1.1,
                                dash=dash if point else {'Q2.5':'dot','Q50':'dashdot','Q97.5':'longdash'}[q]),
                      mode='lines+markers' if point else 'lines',marker=dict(symbol=symbol,size=4),
                      hovertemplate=label+' · %{customdata:,.2f}<extra></extra>'),row=row,col=1)
        legends.add(label)
