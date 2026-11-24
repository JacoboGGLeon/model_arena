"""Percentage errors with additive sufficient statistics; no epsilon bias."""
import numpy as np

def percentage_metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.shape!=p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Expected aligned, finite, nonempty targets and predictions")
    error=np.abs(y-p);den=np.abs(y)+np.abs(p);mask=y!=0
    ratios=np.divide(error,np.abs(y),out=np.zeros_like(error),where=mask)
    symmetric=np.divide(2*error,den,out=np.zeros_like(error),where=den!=0)
    result={"absolute_error_sum":float(error.sum()),"absolute_observed_sum":float(np.abs(y).sum()),"ape_sum":float(ratios.sum()),"mape_count":int(mask.sum()),"smape_sum":float(symmetric.sum()),"percentage_count":int(y.size)}
    result.update(mape=100*result['ape_sum']/result['mape_count'] if result['mape_count'] else None,smape=100*result['smape_sum']/result['percentage_count'],wmape=100*result['absolute_error_sum']/result['absolute_observed_sum'] if result['absolute_observed_sum'] else None)
    return result
