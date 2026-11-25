import json
from pathlib import Path
import pandas as pd
import pytest
from calendar_studio import CalendarStudio, example_holidays
from model_arena.calendar import prepare_snapshot, to_features, verify_snapshot


def test_artifact_projection_and_freeze(tmp_path):
    source = CalendarStudio.from_profile("mac_legacy_v1").generate(
        start="2026-01-01", end="2026-12-31", holidays=example_holidays()
    )
    with pytest.raises(ValueError, match="243"):
        prepare_snapshot(calendar=source)
    view = source.view("mac_legacy_v1")
    snapshot = prepare_snapshot(calendar=view)
    assert "2026-11-02" not in snapshot.frame.fecha.tolist()
    manifest = snapshot.save(tmp_path)
    from model_arena.storage import digest

    data = verify_snapshot(tmp_path, digest(manifest))
    import calendar_studio

    assert data["generator_version"] == calendar_studio.__version__
    assert data["feature_count"] == 243
    assert "2026-11-02" in data["source_metadata"]["holidays"]
    assert (tmp_path / "calendar/features.parquet").is_file()
    frozen = pd.read_parquet(tmp_path / "calendar/features.parquet")
    pd.testing.assert_frame_equal(snapshot.frame, frozen)
    (tmp_path / "calendar/source.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path, digest(manifest))


def test_artifact_directory(tmp_path):
    artifact = (
        CalendarStudio.from_profile("mac_legacy_v1")
        .generate(start="2026-01-01", end="2026-03-31", holidays=[])
        .view("mac_legacy_v1")
    )
    artifact.save(tmp_path / "input")
    snapshot = prepare_snapshot(calendar_path=tmp_path / "input")
    assert snapshot.provenance["view"] == "mac_legacy_v1"
    assert "source_manifest.json" in snapshot.source_files


def test_generic_features_are_not_mac_specific():
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "sin": [0.25, -0.3]})
    out = to_features(frame, date_col="date", time_col="ds")
    assert out.columns.tolist() == ["ds", "sin"]
    assert out.sin.tolist() == [0.25, -0.3]
    frame.loc[1, "date"] = "2026-01-01"
    with pytest.raises(ValueError, match="unique"):
        to_features(frame, date_col="date")


def test_csv_bytes_remain_identical(tmp_path):
    f = pd.DataFrame({"fecha": ["2026-01-01"], **{f"f{i}": [0] for i in range(243)}})
    path = tmp_path / "input.csv"
    f.to_csv(path, index=False)
    snapshot = prepare_snapshot(calendar_path=path)
    assert snapshot.csv_bytes == path.read_bytes()
    with pytest.raises(ValueError, match="exactly one"):
        prepare_snapshot(calendar=f, calendar_path=path)


def test_invalid_mac_tables():
    f = pd.DataFrame({"fecha": ["2026-01-01"], **{f"f{i}": [0] for i in range(243)}})
    f.loc[0, "f0"] = 2
    with pytest.raises(ValueError, match="binary"):
        prepare_snapshot(calendar=f)
    with pytest.raises(ValueError):
        prepare_snapshot(calendar=pd.DataFrame())


def test_mac_runtime_rejects_wrong_feature_configuration_before_creating_project(tmp_path):
    from model_arena.profiles.mac import MacProject

    training = {
        "requested_accounts": 0,
        "aws": {"account_limit": 0},
        "model": {"calendar_features": 244},
        "curvas": {},
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="243"):
        MacProject.create(
            tmp_path / "project",
            training_config=path,
            publication_config=tmp_path / "unused.json",
            calendar=pd.DataFrame(),
            device="cpu",
        )
    assert not (tmp_path / "project").exists()
