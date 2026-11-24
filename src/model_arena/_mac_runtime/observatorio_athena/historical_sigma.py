"""Descriptive expanding reference recovered from the pre-0.3 MAC source.

Strictly earlier finite observations, population sigma, minimum 5. Not a
rolling-window statistic or a calibrated predictive/contable interval.
"""
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SIGMA_BANDS = ((-3, -2, '2–3σ', '#dba6ad', .24),
               (-2, -1, '1–2σ', '#edc06c', .26),
               (-1, 1, '±1σ', '#57ba9a', .26),
               (1, 2, '1–2σ', '#edc06c', .26),
               (2, 3, '2–3σ', '#dba6ad', .24))


def expanding_sigma(values, minimum=5):
    if minimum < 2:
        raise ValueError('minimum must be at least 2')
    values = np.asarray(values, dtype=float)
    center = np.full(len(values), np.nan)
    sigma = np.full(len(values), np.nan)
    z = np.full(len(values), np.nan)
    counts = np.zeros(len(values), dtype=int)
    n, mean, m2 = 0, 0., 0.
    for i, value in enumerate(values):
        counts[i] = n
        spread = math.sqrt(max(0., m2/n)) if n else 0.
        if n >= minimum and spread > 1e-12:
            center[i], sigma[i] = mean, spread
            if np.isfinite(value):
                z[i] = (value-mean)/spread
        if np.isfinite(value):
            n += 1
            delta = value-mean
            mean += delta/n
            m2 += delta*(value-mean)
    result = dict(sigma_center=center, sigma_std=sigma,
                  sigma_reference_n=counts, zscore=z)
    result['zscore_status'] = np.where(counts < minimum, 'historia insuficiente',
                                     np.where(np.isfinite(sigma), 'disponible', 'referencia constante'))
    for k in (1, 2, 3):
        result[f'sigma_lower_{k}'] = center-k*sigma
        result[f'sigma_upper_{k}'] = center+k*sigma
    return result


def history_figure(frame, show_bands=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=.14, subplot_titles=['Saldo', 'Variación'])
    rows = []
    dates = frame['date'].to_list()
    for row, column, target, color in [(1, 'balance', 'saldo', '#1464a5'),
                                       (2, 'variation', 'variacion', '#02a5a5')]:
        values = frame[column].to_numpy()
        ref = expanding_sigma(values)
        if show_bands:
            seen = set()
            for a, b, label, shade, alpha in SIGMA_BANDS:
                lo = ref['sigma_center']+a*ref['sigma_std']
                hi = ref['sigma_center']+b*ref['sigma_std']
                fill = 'rgba('+','.join(str(int(shade[i:i+2], 16)) for i in (1, 3, 5))+f',{alpha})'
                fig.add_trace(go.Scatter(x=dates, y=lo, mode='lines', line=dict(width=0),
                                        connectgaps=False, hoverinfo='skip', showlegend=False,
                                        legendgroup='sigma_'+label), row=row, col=1)
                fig.add_trace(go.Scatter(x=dates, y=hi, mode='lines', line=dict(width=0),
                                        fill='tonexty', fillcolor=fill, connectgaps=False,
                                        hoverinfo='skip', name='Referencia '+label,
                                        legendgroup='sigma_'+label, showlegend=row == 1 and label not in seen), row=row, col=1)
                seen.add(label)
        text = [f"Z causal: {z:.3f} · n={n}" if np.isfinite(z) else f"Z: N/D · {s} · n={n}"
                for z, n, s in zip(ref['zscore'], ref['sigma_reference_n'], ref['zscore_status'])]
        fig.add_trace(go.Scatter(x=dates, y=values, name='Observado', legendgroup='observado',
                                showlegend=row == 1, line=dict(color=color, width=1.5),
                                text=text, hovertemplate='%{x}<br>Observado: %{y:,.2f}<br>%{text}<extra></extra>'), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ref['sigma_center'], name='Media acumulada anterior',
                                legendgroup='media', showlegend=row == 1, line=dict(color='#8091a5', dash='dot', width=1),
                                hovertemplate='%{x}<br>Media anterior: %{y:,.2f}<extra></extra>'), row=row, col=1)
        for i, day in enumerate(dates):
            rows.append(dict(date=day, target=target, observed=float(values[i]),
                             **{k: v[i].item() if hasattr(v[i], 'item') else v[i] for k, v in ref.items()}))
    fig.update_layout(height=680, template='plotly_white', hovermode='x unified',
                      margin=dict(l=65, r=25, t=45, b=80), legend=dict(orientation='h', y=-.14))
    return fig, rows
