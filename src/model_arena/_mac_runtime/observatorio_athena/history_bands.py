"""Descriptive bands for the observed account history in Athena."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .observed_quantiles import trailing_quantiles

VIEWS = ('Percentiles', 'MAD', 'Superposición')
WINDOW = 20
CONTEXT = 60
MAD_SCALE = 1.4826
MAD_MULTIPLIER = 2
FOLDS = ('F1', 'F2', 'F3')
OBSERVED_COLOR = '#16395a'
METHODS = {
    'pct': ('Percentiles Q2.5–Q97.5', '#8b579e', 'rgba(139,87,158,.22)'),
    'mad': ('Envolvente MAD', '#5495bc', 'rgba(84,149,188,.24)'),
}


def intervals(history, window=WINDOW):
    frame = history.loc[history['target'].eq('saldo'), ['date', 'observed']].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
    frame['observed'] = pd.to_numeric(frame['observed'], errors='coerce').replace(
        [np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=['date']).sort_values('date').drop_duplicates('date')
    observed = frame['observed'].to_numpy(dtype=float)
    frame[['pct_low', 'pct_mid', 'pct_high']] = trailing_quantiles(observed, window)
    radius = np.full(len(frame), np.nan)
    for index in range(window, len(frame)):
        prior_changes = np.diff(observed[index - window:index])
        if np.isfinite(prior_changes).all():
            median_change = np.median(prior_changes)
            mad = np.median(np.abs(prior_changes - median_change))
            radius[index] = MAD_MULTIPLIER * MAD_SCALE * mad
    frame['mad_low'] = observed - radius
    frame['mad_high'] = observed + radius
    frame['overlap_low'] = frame[['pct_low', 'mad_low']].max(axis=1, skipna=False)
    frame['overlap_high'] = frame[['pct_high', 'mad_high']].min(axis=1, skipna=False)
    frame.loc[frame['overlap_low'] > frame['overlap_high'], ['overlap_low', 'overlap_high']] = np.nan
    return frame.set_index('date')


def fold_points(daily):
    """Read the first published date and saldo Q50 of each available fold."""
    if daily.empty:
        return pd.DataFrame()
    points = daily.loc[daily['fold'].isin(FOLDS) & daily['target'].eq('saldo')].copy()
    points['date'] = pd.to_datetime(points['date'], errors='coerce').dt.normalize()
    points = points.dropna(subset=['date']).sort_values(['date', 'origin', 'candidate'])
    points = points.drop_duplicates('fold').set_index('date')
    points['model_q50'] = pd.to_numeric(points['model_q50'], errors='coerce').replace(
        [np.inf, -np.inf], np.nan)
    return points


def _area(fig, x, lower, upper, name, color, outline):
    lower, upper = np.asarray(lower), np.asarray(upper)
    valid = np.isfinite(lower) & np.isfinite(upper)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    dates, bounds = [], []
    for start, end in zip(edges[::2], edges[1::2]):
        dates.extend([*x[start:end], *x[start:end][::-1], None])
        bounds.extend([*upper[start:end], *lower[start:end][::-1], None])
    if dates:
        fig.add_scatter(x=dates, y=bounds, mode='lines', fill='toself',
                        fillcolor=color, line=dict(color=outline, width=1),
                        name=name, hoverinfo='skip')


def history_bands_figure(frame, points, view, *, full_history=False, show_estimates=False):
    if view not in VIEWS:
        raise ValueError(f'Vista de historia no reconocida: {view}')
    fig = go.Figure()
    x = frame.index
    methods = ('pct', 'mad') if view == 'Superposición' else (('pct',) if view == 'Percentiles' else ('mad',))
    for prefix in methods:
        label, outline, color = METHODS[prefix]
        _area(fig, x, frame[f'{prefix}_low'], frame[f'{prefix}_high'], label, color, outline)
    if view == 'Superposición':
        _area(fig, x, frame['overlap_low'], frame['overlap_high'], 'Intersección',
              'rgba(9,120,111,.74)', '#087a78')
    if view == 'Percentiles':
        fig.add_scatter(x=x, y=frame['pct_mid'], name='Q50 móvil histórico', mode='lines',
                        line=dict(color=METHODS['pct'][1], dash='dash', width=2),
                        connectgaps=False,
                        hovertemplate='%{y:,.2f}<extra>Q50 móvil histórico</extra>')
    fig.add_scatter(x=x, y=frame['observed'], name='Saldo observado', mode='lines',
                    line=dict(color=OBSERVED_COLOR, width=3), connectgaps=False,
                    hovertemplate='%{x|%d/%m/%Y}<br>%{y:,.2f}<extra>Saldo observado</extra>')
    shown_start = x.min()
    estimates = pd.DataFrame()
    if not points.empty:
        for date, fold in points['fold'].items():
            fig.add_vline(x=date.isoformat(), line_dash='dot', line_color='#526b83')
            fig.add_annotation(x=date.isoformat(), y=1, yref='paper', yanchor='bottom',
                               text=f'{fold} · {date:%d/%m}', showarrow=False)
        if show_estimates:
            estimates = points.dropna(subset=['model_q50'])
        if not estimates.empty:
            fig.add_scatter(x=estimates.index, y=estimates['model_q50'], mode='markers',
                            name='Q50 publicado por fold',
                            customdata=estimates[['fold', 'candidate']].to_numpy(),
                            marker=dict(symbol='diamond', size=11, color='#334e68',
                                        line=dict(color='white', width=1)),
                            hovertemplate=('%{x|%d/%m/%Y} · %{customdata[0]}<br>Q50: %{y:,.2f}'
                                           '<br>Modelo: %{customdata[1]}<extra></extra>'))
        if len(x) and not full_history:
            f1 = points.loc[points['fold'].eq('F1')]
            first_fold = f1.index.min() if not f1.empty else points.index.min()
            shown_start = x[max(0, x.searchsorted(first_fold) - CONTEXT)]
            fig.update_xaxes(range=[shown_start, max(x.max(), points.index.max())])
    values = frame.loc[shown_start:, ['observed', 'pct_low', 'pct_high', 'mad_low', 'mad_high']].to_numpy()
    if not estimates.empty:
        values = np.r_[values.ravel(), estimates['model_q50'].to_numpy()]
    finite = values[np.isfinite(values)]
    if finite.size:
        low, high = finite.min(), finite.max()
        pad = (high - low) * .06 or max(abs(low) * .04, 1.)
        fig.update_yaxes(range=[low - pad, high + pad])
    fig.update_layout(template='plotly_white', height=530, hovermode='closest',
                      margin=dict(l=65, r=30, t=42, b=70),
                      legend=dict(orientation='h', y=-.17),
                      xaxis_title='Corte contable', yaxis_title='Saldo en moneda de la cuenta')
    return fig
