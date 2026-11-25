"""Independent objectives keep independent finalists and unseen routing."""
from types import SimpleNamespace
import polars as pl
from model_arena.winner_selection import assignments


def test_opposite_target_winners_and_zero_shot():
    rows=[]
    candidates=[]
    for target in ('saldo','variacion'):
        for family in ('dlinear','encoder_decoder'):
            key=target+'_'+family
            candidates.append(SimpleNamespace(id=key,target=target))
            value=1 if (family=='dlinear')==(target=='saldo') else 9
            rows.append(dict(account_id='seen',currency='USD',budget=1,target=target,
                             family=family,input='X0',candidate=key,score=value,fold='F1'))
    inventory=pl.DataFrame([dict(account_id=k,currency='USD',budget=1,cohort=cohort)
                           for k,cohort in [('seen','seen'),('unseen','zero_shot')]])
    chosen=assignments(pl.DataFrame(rows),inventory,candidates,['F1'])
    assert chosen.height==4
    assert chosen.filter(pl.col('target')=='saldo')['candidate'].to_list()==['saldo_dlinear']*2
    assert chosen.filter(pl.col('target')=='variacion')['candidate'].to_list()==['variacion_encoder_decoder']*2
    assert chosen.filter(pl.col('account_id')=='unseen')['selection_source'].to_list()==['DIVISA_BUDGET']*2


def test_initial_target_routes_never_mix_objectives():
    candidates=[SimpleNamespace(id=t,target=t) for t in ('saldo','variacion')]
    inventory=pl.DataFrame([dict(account_id='A',budget=1,currency='USD',cohort='seen')])
    scores=pl.DataFrame({'fold':['F1','F1'],'target':['saldo','variacion']})
    chosen=assignments(scores,inventory,candidates,[])
    assert chosen['candidate'].to_list()==chosen['target'].to_list()


def test_single_target_history_and_network_isolation():
    """Perturbing the other objective cannot change this candidate's tensors."""
    import subprocess
    import sys
    code = r'''
from importlib.resources import files
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
root=Path(str(files('model_arena')/'_mac_runtime/entrenamiento'))
sys.path[:0]=[str(p) for p in (root/'libraries').glob('*/src')]
from agentic_forecasting import Series
from agentic_forecasting.curve_models import CurveTFT,make_batch
spec=SimpleNamespace(window=6,horizon=3,calendar_features=243)
a=np.column_stack((np.arange(12)+100,np.arange(12)-5)).astype(float)
for target,index in [('saldo',0),('variacion',1)]:
    candidate=SimpleNamespace(target=target,inputs='X0',family='encoder_decoder',width=8,depth=1,decoder_depth=1,activation='GELU',encoder_kind='gru',encoder_design='single')
    changed=a.copy();changed[:,1-index]*=1e9
    frames=[]
    for values in [a,changed]:
        series=[Series('A',['2025-01-01']*12,values,'USD',1,'seen')]
        features,_,truth=make_batch(series,[(0,6)],spec,candidate,torch.device('cpu'),None,True)
        frames.append((features,truth))
    assert torch.equal(frames[0][0]['history_y'],frames[1][0]['history_y'])
    assert torch.equal(frames[0][1],frames[1][1])
    model=CurveTFT(spec,candidate,0)
    assert len(model.encoders)==len(model.quantiles)==1
    assert model(frames[0][0]).shape==(1,3,1,3)
'''
    subprocess.run([sys.executable,'-c',code],check=True)
