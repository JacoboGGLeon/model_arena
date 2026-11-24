import sys
import subprocess
import json
import numpy as np
import pandas as pd
import pytest

from model_arena import ModelArena, ModelSpec
from model_arena.results import Results
from model_arena.storage import read_json, read_table


def test_monitor_import_is_independent_from_training():
    result = subprocess.run([sys.executable, "-c", "import sys; from model_arena.results import Results; "
                             "assert 'torch' not in sys.modules; assert 'optuna' not in sys.modules; "
                             "assert 'streamlit' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fixed_fit_predict_save_load_publish_end_to_end(panel, spec, tmp_path):
    df, static = panel
    arena = ModelArena(spec)
    assert arena.fit(df=df, static_df=static) is arena
    before = arena.predict()
    assert len(before) == 3*2*2*3*2
    assert set(before.role) == {"Baseline", "Winner"}
    artifact = arena.save(tmp_path / "artifact")
    restored = ModelArena.load(artifact)
    after = restored.predict(df=df, static_df=static)
    pd.testing.assert_frame_equal(before, after)
    release = arena.postprocess(before)
    publication = arena.publish(release, destination=tmp_path / "published")
    again = arena.publish(release, destination=tmp_path / "published")
    assert publication == again
    results = Results.open(tmp_path / "published")
    assert len(results.table("forecasts")) == len(before)
    assert set(results.table("winners").entity) == {"alpha", "beta", "new"}
    assert not (tmp_path / "runs" / "latest.json").exists()


def test_corrupt_artifact_fails_before_loading(panel, spec, tmp_path):
    df, static = panel
    artifact = ModelArena(spec).fit(df=df, static_df=static).save(tmp_path / "saved")
    from pathlib import Path
    (Path(artifact.path)/"C0000.weights").write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        ModelArena.load(artifact)


def test_cv_does_not_replace_operational_model(panel, spec):
    df, static = panel
    arena = ModelArena(spec).fit(df=df, static_df=static)
    before = arena.predict()
    weights = dict(arena._models)
    study = arena.cross_validation(df=df, static_df=static)
    assert weights == arena._models
    assert set(read_table(__import__('pathlib').Path(study.path)/"scores.json").fold) == {"F1", "F2", "F3"}
    # The study is separate; inference keeps the operational assignment table.
    assert hasattr(arena, "_fitted_ref")
    pd.testing.assert_frame_equal(before, arena.predict())


def test_zero_shot_selection_uses_only_training_evidence(panel, spec):
    df, static = panel
    arena = ModelArena(spec)
    study = arena.cross_validation(df=df, static_df=static)
    selection = arena.select(study)
    winners = read_table(__import__('pathlib').Path(selection.path)/"winners.json")
    row = winners[winners.entity == "new"].iloc[0]
    assert row.population == "zero_shot"
    assert row.evidence_origin == "GLOBAL"
    provenance = read_json(__import__('pathlib').Path(study.path)/"C0000_F1.provenance.json")
    assert set(provenance["training_ids"]) == {"alpha", "beta"}


@pytest.mark.parametrize("change", ["duplicate", "gap", "nonfinite", "missing_metadata", "no_training"])
def test_invalid_panel_is_rejected(panel, spec, change):
    df, static = panel
    if change == "duplicate":
        df = pd.concat([df, df.iloc[[0]]])
    elif change == "gap":
        df = df.iloc[1:]
    elif change == "nonfinite":
        df.loc[0, "level"] = np.inf
    elif change == "missing_metadata":
        static = static.iloc[1:]
    elif change == "no_training":
        static["population"] = "zero_shot"
    with pytest.raises(ValueError):
        ModelArena(spec).prepare(df=df, static_df=static)


def test_predict_requires_fit_and_horizon_compatibility(panel, spec):
    df, static = panel
    arena = ModelArena(spec)
    with pytest.raises(ValueError, match="fit"):
        arena.predict()
    arena.fit(df=df, static_df=static)
    with pytest.raises(ValueError, match="Horizon"):
        arena.predict(h=99)


def test_time_leak_final_weights_cannot_backtest(panel, spec):
    df, static = panel
    arena = ModelArena(spec).fit(df=df, static_df=static)
    old = df[df.date < df.date.max()]
    with pytest.raises(ValueError, match="historical backtest"):
        arena.predict(df=old, static_df=static)


def test_future_covariate_coverage_is_validated(panel, spec):
    df, static = panel
    configured = spec.model_copy(update={"models": [ModelSpec(alias="future", kind="historical_quantile",
                                                              future_features=["day"])]})
    arena = ModelArena(configured).fit(df=df, static_df=static)
    with pytest.raises(ValueError, match="covariates"):
        arena.predict()


def test_schema_and_predictions_are_domain_agnostic(panel, spec):
    df, static = panel
    assert "saldo" not in json.dumps(spec.model_dump())
    output = ModelArena(spec).fit(df=df, static_df=static).predict()
    assert set(output.target) == {"level", "change"}

