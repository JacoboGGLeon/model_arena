"""Encoder interchangeability with joint targets and ordered quantile heads."""
import json
from pathlib import Path

import optuna
import pytest

from model_arena.pruning import enqueue_encoder_schedule, make_pruner


@pytest.mark.train
@pytest.mark.parametrize('kind', ['mlp', 'gru', 'tft_compact'])
@pytest.mark.parametrize('design', ['shared', 'independent'])
def test_encoder_forward_backward_and_checkpoint(kind, design, tmp_path):
    import torch
    from model_arena.models.tft import Network, TFTParameters
    torch.manual_seed(42)
    params=TFTParameters(encoder_kind=kind,encoder_design=design,width=8,depth=2,dropout=0.)
    model=Network(2,3,5,5,params,window=6)
    history=torch.randn(4,6,2)
    past=torch.randn(4,6,5)
    future=torch.randn(4,3,5)
    out=model(history,past,future)
    assert out.shape==(4,3,2,3)
    assert torch.isfinite(out).all()
    assert (out[...,0]<=out[...,1]).all() and (out[...,1]<=out[...,2]).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path=tmp_path/'weights.pt';torch.save(model.state_dict(),path)
    restored=Network(2,3,5,5,params,window=6)
    restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(out,restored(history,past,future))
    if kind != 'tft_compact':
        assert not any(isinstance(m,(torch.nn.LSTM,torch.nn.MultiheadAttention)) for m in model.modules())


def test_balanced_allocation_and_reference_per_encoder():
    study=optuna.create_study(pruner=make_pruner({'kind':'patience'},epochs_per_fold=15))
    kinds=['mlp','gru','tft_compact']
    enqueue_encoder_schedule(study,kinds,25)
    queued=study.get_trials()
    assert len(queued)==25
    assigned=[t.system_attrs['fixed_params']['encoder_kind'] for t in queued]
    assert [assigned.count(k) for k in kinds]==[9,8,8]
    enqueue_encoder_schedule(study,kinds,25)
    assert len(study.trials)==25
    for i in range(3):
        trial=study.ask()
        kind=trial.suggest_categorical('encoder_kind',kinds)
        trial.set_user_attr('reference_group',kind)
        for epoch in range(15):
            trial.report(1.,epoch)
            assert not trial.should_prune()
        study.tell(trial,1.)
    trial=study.ask();kind=trial.suggest_categorical('encoder_kind',kinds)
    trial.set_user_attr('reference_group',kind)
    for epoch in range(5):trial.report(1.,epoch)
    assert trial.should_prune()


@pytest.mark.train
def test_generic_hpo_all_encoders_and_zero_shot(panel, spec):
    from model_arena import ModelArena,ModelSpec,SearchSpec,PruningSpec
    from model_arena.storage import read_table
    df,static=panel
    search=SearchSpec(n_trials=6,encoder_kind=['mlp','gru','tft_compact'],width=[8],depth=[1],
        decoder_depth=[1],activation=['GELU'],encoder_design=['independent'],
        pruner=PruningSpec(kind='patience',min_delta=1e20))
    arena=ModelArena(spec.model_copy(update={'models':[ModelSpec(alias='X3',kind='encoder_decoder',
        historical_features=['day'],future_features=['day'])],'search':search}))
    study=arena.tune(df=df,static_df=static,epochs=6)
    trials=read_table(Path(study.path)/'trials.json')
    assert trials.state.tolist()==['COMPLETE']*3+['PRUNED']*3
    candidates=json.loads((Path(study.path)/'candidates.json').read_text())
    assert {c['model']['params']['encoder_kind'] for c in candidates}=={'mlp','gru','tft_compact'}
    selected=arena.select(study)
    arena.fit(df=df,static_df=static,selection=selected,epochs=2)
    import pandas as pd
    dates=pd.date_range(df.date.max()+pd.Timedelta(days=1),periods=spec.h,freq='D')
    future=pd.DataFrame([dict(entity=entity,date=day,day=day.dayofweek/6)
                         for entity in static.entity for day in dates])
    predictions=arena.predict(futr_df=future)
    assert 'new' in set(predictions.entity)


def test_frozen_mac_backend_matches_public_encoder_contract():
    import model_arena
    root=Path(model_arena.__file__).parent
    backend=root/'_mac_runtime/entrenamiento/libraries/agentic_forecasting/src/agentic_forecasting/encoder_backends.py'
    assert backend.read_bytes()==(root/'encoders.py').read_bytes()
