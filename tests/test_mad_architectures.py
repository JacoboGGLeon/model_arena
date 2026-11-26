import optuna
import pytest
import torch
from torch import nn

from model_arena.contracts import SearchSpec
from model_arena.input_spec import study_plan
from model_arena.models.complete import FAMILIES, CompleteNetwork, CompleteParameters, suggest_parameters
from model_arena.models.tft import Network, TFTParameters


@pytest.mark.parametrize('family', FAMILIES)
def test_mad_activation_is_applied_to_point_forecast(family):
    knobs = {
        'activation': 'GELU', 'lr': 0.001, 'weight_decay': 0.0001,
        'hidden_size': 32, 'blocks_per_stack': 1, 'pooling': 2, 'shared_weights': True,
        'num_encoder_layers': 2, 'num_decoder_layers': 1, 'decoder_output_dim': 8,
        'temporal_decoder_dim': 32, 'moving_avg_window': 20,
    }
    trial = optuna.trial.FixedTrial(knobs)
    search = SearchSpec(activation=['ReLU', 'GELU', 'Tanh'], dropout=.1)
    params = CompleteParameters.model_validate(suggest_parameters(family, trial, search, mac_point=True))
    features = 2 if family in ('nhits', 'tide') else 0
    net = CompleteNetwork(family=family, targets=1, window=20, h=4, historical=features,
                          future=features, parameters=params, point=True)
    pred = net(torch.randn(3, 20, 1), torch.randn(3, 20, features), torch.randn(3, 4, features))
    assert pred.shape == (3, 4, 1, 1)
    pred.square().mean().backward()
    assert torch.isfinite(pred).all()
    assert any(isinstance(module, nn.GELU) for module in net.modules())
    assert any(isinstance(module, nn.Dropout) and module.p == .1 for module in net.modules())
    assert 'dropout' not in trial.params


@pytest.mark.parametrize('encoder', ['mlp', 'gru', 'tft_compact'])
def test_encoder_decoder_one_output_and_calendar(encoder):
    params = TFTParameters(encoder_kind=encoder, width=8, activation='GELU')
    net = Network(1, 3, 2, 2, params, window=5, point=True)
    net.eval()
    x, past, future = torch.randn(2, 5, 1), torch.randn(2, 5, 2), torch.randn(2, 3, 2)
    prediction = net(x, past, future)
    assert prediction.shape == (2, 3, 1, 1)
    assert not torch.allclose(prediction, net(x, past + 1, future))
    assert not torch.allclose(prediction, net(x, past, future + 1))
    prediction.mean().backward()


def test_ada_mad_plan_has_six_slots_per_input_and_shared_encoders():
    config = dict(
        inputs=['X0', 'X1', 'X2', 'X3'], objectives=['saldo', 'variacion'],
        families=dict.fromkeys([*FAMILIES, 'encoder_decoder'], 24),
        input_search=dict(mode='family_objective', startup_trials_per_input=6, tpe_startup_trials=6),
    )
    tasks = study_plan(config, 15)
    assert len(tasks) == 14 and sum(t['trials'] for t in tasks) == 336
    for task in tasks:
        assert task['tpe_startup_trials'] == 6
        if task['family'] in ('nhits', 'tide', 'encoder_decoder'):
            assert len(task['inputs']) == 4 and task['startup_trials_per_input'] == 6
            assert task['trials'] == len(task['inputs']) * task['startup_trials_per_input']
        else:
            assert task['input'] == 'X0'
