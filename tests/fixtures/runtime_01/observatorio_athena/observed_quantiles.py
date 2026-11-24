"""Empirical descriptive quantiles of strictly prior observations."""
import numpy as np

QUANTILES=(.025,.5,.975)


def trailing_quantiles(values,window=20):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or window<2:raise ValueError('Invalid historical quantile window')
    out=np.full((len(values),3),np.nan)
    for i in range(window,len(values)):
        prior=values[i-window:i]
        if np.isfinite(prior).all():out[i]=np.quantile(prior,QUANTILES)
    return out
