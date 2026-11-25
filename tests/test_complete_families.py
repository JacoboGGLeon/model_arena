import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from model_arena import ModelArena, ModelSpec, SearchSpec, PruningSpec
from model_arena.models.complete import CompleteNetwork,CompleteParameters

@pytest.mark.parametrize('family',['tide','nhits','nlinear','dlinear','nbeats'])
def test_upstream_forward_checkpoint_and_target_gradients(family,tmp_path):
    architecture={'hidden_size':8,'temporal_decoder_dim':8,'decoder_output_dim':8} if family=='tide' else {'mlp_units':[[8,8]]*3} if family in ('nhits','nbeats') else {}
    n=CompleteNetwork(family=family,targets=2,window=6,h=3,historical=0,future=0,parameters=CompleteParameters(architecture=architecture))
    n.eval();x=torch.randn(3,6,2)*.1
    y=n(x,None,None)
    assert y.shape==(3,3,2,3) and torch.isfinite(y).all()
    assert (y[...,0]<=y[...,1]).all() and (y[...,1]<=y[...,2]).all()
    y.square().mean().backward()
    for target in n.models:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in target.parameters())
    state=copy.deepcopy(n.state_dict());n.load_state_dict(state)
    torch.testing.assert_close(y,n(x,None,None))

@pytest.mark.parametrize('family',['tide','nhits'])
def test_calendar_covariates_are_consumed(family):
    a={'hidden_size':8,'temporal_decoder_dim':8,'decoder_output_dim':8,'dropout':0.} if family=='tide' else {'mlp_units':[[8,8]]*3}
    n=CompleteNetwork(family=family,targets=2,window=6,h=3,historical=2,future=2,parameters=CompleteParameters(architecture=a));n.eval()
    history=torch.randn(2,6,2)*.1;past=torch.zeros(2,6,2);future=torch.zeros(2,3,2)
    y=n(history,past,future)
    assert not torch.allclose(y,n(history,past+1,future))
    assert not torch.allclose(y,n(history,past,future+1))

@pytest.mark.parametrize('family',['nlinear','dlinear','nbeats'])
def test_linear_family_rejects_calendar(family):
    with pytest.raises(ValueError,match='X0 only'):
        CompleteNetwork(family=family,targets=2,window=6,h=3,historical=1,future=0,parameters=CompleteParameters())

@pytest.mark.train
def test_four_families_hpo_select_save_zero_shot(panel,spec,tmp_path):
    df,static=panel
    models=[ModelSpec(alias=k,kind=k,trials=1) for k in ['tide','nhits','nlinear','dlinear','nbeats']]
    search=SearchSpec(n_trials=1,width=[8],depth=[1],decoder_depth=[1],dropout=0.,pruner=PruningSpec(kind='patience'))
    arena=ModelArena(spec.model_copy(update={'models':models,'search':search}))
    study=arena.tune(df=df,static_df=static,epochs=2)
    curves=arena.results.learning_curves()
    assert set(curves.input)=={m.alias for m in models}
    assert set(curves.target)=={'level','change'} and set(curves.fold)=={'F1','F2','F3'}
    selection=arena.select(study)
    arena.fit(df=df,static_df=static,selection=selection,epochs=2)
    before=arena.predict()
    assert 'new' in set(before.entity)
    artifact=arena.save(tmp_path/'saved')
    loaded=ModelArena.load(artifact)
    after=loaded.predict(df=df,static_df=static)
    np.testing.assert_allclose(before.value,after.value,rtol=1e-6)


def test_mac_keeps_family_finalists_before_series_selection():
    from types import SimpleNamespace
    import polars as pl
    from model_arena.winner_selection import assignments
    rows=[]
    for candidate,family,values in [('A','tide',[.1,.9]),('B','nhits',[.9,.1])]:
        for account,value in zip(['one','two'],values):
            rows.append(dict(candidate=candidate,family=family,input='X0',budget=1,
                account_id=account,currency='USD',fold='F1',score=value))
    inventory=pl.DataFrame([dict(account_id=a,budget=1,currency='USD',cohort='seen') for a in ['one','two']])
    chosen=assignments(pl.DataFrame(rows),inventory,[SimpleNamespace(id=k) for k in ['A','B']],['F1'])
    assert chosen['candidate'].to_list()==['A','B']
