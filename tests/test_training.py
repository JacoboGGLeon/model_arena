import copy
import numpy as np
import pytest
from model_arena import ModelArena, ModelSpec
from model_arena.storage import read_table

pytestmark = pytest.mark.train


@pytest.fixture
def neural(spec):
    return spec.model_copy(update={"models": [ModelSpec(alias="X0", params=dict(width=8, depth=1,
                                     decoder_depth=1, dropout=0., lr=.001))]})


def test_real_neural_curves_persistence_and_inference(panel, neural, tmp_path):
    import torch
    df, static = panel
    arena = ModelArena(neural)
    study = arena.cross_validation(df=df, static_df=static)
    curves = arena.results.learning_curves()
    assert set(curves.fold) == {"F1", "F2", "F3"}
    assert set(curves.split) == {"train_fixed", "validation"}
    assert set(curves.target) == {"level", "change"}
    assert set(curves.epoch) == {1, 2}
    assert "new" not in set(curves[curves.split == "train_fixed"].entity)
    assert "new" in set(curves[curves.split == "validation"].entity)
    segment = arena.results.learning_curves(scope="segment")
    assert set(segment.segment) == {1}
    selection = arena.select(study)
    arena.fit(df=df, static_df=static, selection=selection, epochs=3)
    final = arena.results.learning_curves()
    assert set(final.fold) == {"FUTURO"}
    assert set(final.split) == {"train_fixed"}
    state = copy.deepcopy(arena._models[next(iter(arena._models))].network.state_dict())
    predictions = arena.predict()
    for name, tensor in arena._models[next(iter(arena._models))].network.state_dict().items():
        assert torch.equal(tensor, state[name])
    artifact = arena.save(tmp_path / "neural")
    other = ModelArena.load(artifact)
    actual = other.predict(df=df, static_df=static)
    np.testing.assert_allclose(predictions.value, actual.value, rtol=1e-6)
    q = predictions[predictions.role == "Winner"].pivot_table(index=["entity", "ds", "target"],
                                                              columns="quantile", values="value")
    assert (q[.025] <= q[.5]).all() and (q[.5] <= q[.975]).all()


def test_hpo_persists_all_candidates_and_nop_pruner(panel, neural):
    df, static = panel
    search = neural.search.model_copy(update={"width": [8], "depth": [1], "decoder_depth": [1],
                                                "encoder_design": ["shared"], "activation": ["GELU"]})
    arena = ModelArena(neural.model_copy(update={"search": search}))
    study = arena.tune(df=df, static_df=static, n_trials=2, epochs=1)
    from pathlib import Path
    scores = read_table(Path(study.path)/"scores.json")
    assert scores.candidate.nunique() == 2
    assert set(scores.fold) == {"F1", "F2", "F3"}
    assert arena._models == {}


def test_monitor_evaluation_does_not_change_training_weights(panel, neural):
    import torch
    df, static = panel
    data = ModelArena(neural).prepare(df=df, static_df=static)
    from model_arena.models.tft import TFTCompact
    model = TFTCompact(neural.models[0], neural)
    state = copy.deepcopy(model.network.state_dict())
    rng = torch.random.get_rng_state().clone()
    model._losses(data, [("alpha", 10)])
    assert torch.equal(torch.random.get_rng_state(), rng)
    for key, value in model.network.state_dict().items():
        assert torch.equal(value, state[key])

