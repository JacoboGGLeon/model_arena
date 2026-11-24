"""Pure presentation calculations on Athena aggregate moments."""
import numpy as np
import pandas as pd


def score(frame,metric,target):
 f=frame.copy()
 if target!='Ambos':f=f[f.target==target].copy()
 keys=['role','candidate','comparison_id','input','budget','currency']
 if target=='Ambos':
  # A joint result requires both targets; never silently score only one.
  f=f.groupby(keys,dropna=False).filter(lambda g:set(g.target)=={'saldo','variacion'})
 if f.empty:return f
 moments=['n','ae_sum','se_sum','smape_sum','covered_sum','width_sum','ae_scaled_sum','se_scaled_sum']
 f=f.groupby(keys,dropna=False)[moments+['accounts']].sum().reset_index()
 if target=='Ambos':f['accounts']=f['accounts']/2
 n=f['n'].replace(0,np.nan)
 if metric=='SMAPE':f['value']=f.smape_sum/n
 elif metric=='MAE':f['value']=(f.ae_scaled_sum if target=='Ambos' else f.ae_sum)/n
 elif metric=='RMSE':f['value']=np.sqrt((f.se_scaled_sum if target=='Ambos' else f.se_sum)/n)
 elif metric=='Coverage':f['value']=100*f.covered_sum/n
 else:raise ValueError(metric)
 f['width']=f.width_sum/n
 f['rank']=(f['value']-95).abs() if metric=='Coverage' else f['value']
 return f


def comparisons(frame,metric,target,mode='Mejor por input',ranking='Cada budget'):
 values=score(frame,metric,target)
 if values.empty:return values
 role='Modeling' if mode=='Mejor por input' else 'Winner'
 model=values[values.role==role].copy()
 if mode=='Mejor por input':
  if ranking=='Universo filtrado':
   total=score(frame.assign(budget=0),metric,target)
   chosen=total[total.role=='Modeling'].sort_values(['rank','candidate'],kind='stable').drop_duplicates(['input','currency'])
   model=model.merge(chosen[['input','currency','candidate']],on=['input','currency','candidate'],validate='many_to_one')
  else:model=model.sort_values(['rank','candidate'],kind='stable').drop_duplicates(['input','budget','currency'])
 baseline=values[values.role=='Baseline'].copy()
 keys=['comparison_id','input','budget','currency']
 base=baseline[keys+['value','rank','n','accounts','width']].rename(columns={k:'baseline_'+k for k in ['value','rank','n','accounts','width']})
 table=model.merge(base,on=keys,validate='one_to_one')
 if ((table.n!=table.baseline_n)|(table.accounts!=table.baseline_accounts)).any():raise ValueError('Poblaciones distintas en comparación')
 table['difference']=table['value']-table.baseline_value
 table['improvement']=table.baseline_rank-table['rank']
 table['better']=np.where(np.isclose(table.improvement,0,rtol=1e-9,atol=1e-9),'Empate',np.where(table.improvement>0,'Winner','Baseline'))
 return table
