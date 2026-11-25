"""Full MAC migration profile on synthetic data, without AWS or corporate data."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_arena.profiles.mac import MacProject
from model_arena.storage import digest


@pytest.mark.train
@pytest.mark.parametrize("calendar_kind", ["csv", "parquet", "artifact", "families"])
def test_mac_lifecycle_preserves_outputs_and_frozen_inference(tmp_path, monkeypatch, calendar_kind):
    # Subprocesses inherit the CPU torch installation used by this test session.
    import torch
    import model_arena
    import optuna

    paths = [
        str(Path(model_arena.__file__).resolve().parent.parent),
        str(Path(torch.__file__).parents[1]),
        str(Path(optuna.__file__).parents[1]),
    ]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(dict.fromkeys(paths)))
    dates = pd.date_range("2025-01-01", periods=90).strftime("%Y-%m-%d").tolist()
    calendar = pd.DataFrame({"fecha": dates, **{f"f{i:03d}": [0] * 90 for i in range(243)}})
    calendar.to_csv(tmp_path / "calendar.csv", index=False)
    calendar_kwargs = {"calendar_path": tmp_path / "calendar.csv"}
    if calendar_kind == "parquet":
        calendar.to_parquet(tmp_path / "calendar.parquet", index=False)
        calendar_kwargs = {"calendar_path": tmp_path / "calendar.parquet"}
    elif calendar_kind == "artifact":
        from calendar_studio import CalendarStudio

        artifact = (
            CalendarStudio.from_profile("mac_legacy_v1")
            .generate(start="2025-01-01", end="2025-12-31", holidays=[])
            .view("mac_legacy_v1")
        )
        dates = artifact.to_pandas().fecha.tolist()
        calendar_kwargs = {"calendar": artifact}
    rows = []
    for i, (entity, taxonomy) in enumerate(
        [
            ("A", "cuentas_entrenamiento"),
            ("B", "cuentas_entrenamiento"),
            ("C", "cuentas_atipicos"),
            ("Z", "cuentas_cero"),
        ]
    ):
        previous = 0.0
        for j, date in enumerate(dates[:60]):
            amount = float(100 + i * 10 + j + np.sin(j)) if entity != "Z" else 0.0
            rows.append(
                dict(
                    account_id=entity,
                    date=date,
                    currency="USD",
                    taxonomy=taxonomy,
                    sign="positiva" if entity != "Z" else "cuenta_cero",
                    load_date=dates[59],
                    balance=amount,
                    variation=amount - previous,
                    fx=1.0,
                )
            )
            previous = amount
    pd.DataFrame(rows).to_parquet(tmp_path / "engine1.parquet", index=False)
    cfg = dict(
        requested_accounts=0,
        observations=60,
        engine1_path=str(tmp_path / "engine1.parquet"),
        aws=dict(account_limit=0),
        model=dict(
            epochs=1,
            samples_per_epoch=4,
            batch_size=2,
            cpu_threads=1,
            calendar_features=243,
            horizon=4,
            window=4,
            validation=4,
            learning_rate=0.001,
            seed=42,
        ),
        policy=dict(min_accounts=3, min_gain=0.05),
        curvas=dict(
            inputs=["X0", "X1", "X2", "X3"],
            encoders=["shared"],
            depths=[1],
            widths=[8],
            decoder_depths=[1],
            activations=["GELU"],
            dropout=0.0,
            trials=1,
            lr_range=[0.001, 0.002],
            wd_range=[1e-6, 1e-5],
            final_epochs=75,
            loss_monitor_accounts=2,
        ),
    )
    if calendar_kind == "csv":
        # Real training: force a plateau threshold to exercise pruning and resume.
        cfg["model"]["epochs"] = 6
        cfg["curvas"].update(trials=6, encoder_kinds=["mlp", "gru", "tft_compact"],
                             encoders=["independent"],
                             pruner=dict(kind="patience", patience=4, min_delta=1e20))
    if calendar_kind == "families":
        cfg['curvas'].update(families=dict(encoder_decoder=1,tide=1,nhits=1,nlinear=1,dlinear=1,nbeats=1))
    (tmp_path / "training.json").write_text(json.dumps(cfg))
    (tmp_path / "publication.json").write_text(
        json.dumps(dict(s3_base="s3://example/objects/", query_output="s3://example/results/", shards=2))
    )
    project = MacProject.create(
        tmp_path / "project",
        training_config=tmp_path / "training.json",
        publication_config=tmp_path / "publication.json",
        **calendar_kwargs,
        device="cpu",
    )
    assert project.calendar_provenance()["feature_count"] == 243
    project.prepare().tune()
    import polars as pl

    audit_path = project.path / "run/curvas/hpo_trials.parquet"
    audit = pl.read_parquet(audit_path)
    if calendar_kind == "csv":
        assert audit.filter(pl.col("state") == "COMPLETE").height == 12
        pruned = audit.filter(pl.col("state") == "PRUNED")
        assert pruned.height == 12 and set(pruned["pruned_epoch"]) == {5}
        for candidate in pruned["candidate"]:
            history = pl.read_parquet(project.path / f"run/curvas/loss/{candidate}_F1.parquet")
            assert history["epoch"].to_list() == [1, 2, 3, 4, 5]
            assert not (project.path / f"run/curvas/modelos/{candidate}_F1.pt").exists()
        project.tune()
        assert pl.read_parquet(audit_path).equals(audit)
    project.select().fit()
    weights = list((project.path / "run/curvas/modelos").glob("*_FUTURO.pt"))
    assert weights
    before = {str(p): digest(p) for p in weights}
    predictions = project.predict()
    frame = pl.read_parquet(predictions)
    assert set(frame["account_id"]) == {"A", "B", "C"}
    project.postprocess()
    assert before == {str(p): digest(p) for p in weights}
    manifest = json.loads((project.path / "release/publication.json").read_text())
    assert len(manifest["datasets"]["engine2"]["columns"]) == 15
    assert len(manifest["datasets"]["engine3"]["columns"]) == 22
    assert manifest["datasets"]["training_learning"]["rows"] > 0
    assert manifest["datasets"]["training_entities"]["rows"] > 0
    assert manifest["datasets"]["calendar_provenance"]["rows"] == 1
    assert manifest["datasets"]["hpo_trials"]["rows"] == audit.height
    assert manifest["calendar_provenance"]["feature_count"] == 243
    assert manifest["calendar_provenance"]["generator"] == (
        "calendar_studio" if calendar_kind == "artifact" else "external_table"
    )
    assert (project.path / "calendar/features.parquet").is_file()
    parts = [
        project.path / "release" / entry["path"]
        for entry in manifest["files"]
        if entry["dataset"] == "training_entities"
    ]
    detailed = pl.concat([pl.read_parquet(p) for p in parts])
    assert set(detailed["account_id"]) == {"A", "B"}
    assert set(detailed.filter(pl.col("fold") == "FUTURO")["split"]) == {"train"}
    assert set(detailed["budget"]) <= set(range(1, 11))
    assert not (project.path / "release/published.json").exists()
    assignments = [
        pl.read_parquet(project.path / f"run/curvas/winners_{fold}.parquet")
        for fold in ["F1", "F2", "F3", "FUTURO"]
    ]
    assert all(item["candidate"].to_list() == assignments[0]["candidate"].to_list() for item in assignments)
    assert all(torch.load(p, weights_only=False)["spec"]["epochs"] == 75 for p in weights)
