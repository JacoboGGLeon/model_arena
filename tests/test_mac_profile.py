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
@pytest.mark.parametrize("calendar_kind", ["csv", "parquet", "artifact", "families", "dates_only", "pooled", "mad", "group_residual", "group_tree"])
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
    width = {"parquet": 3, "dates_only": 0}.get(calendar_kind, 243)
    calendar = pd.DataFrame({"fecha": dates, **{f"f{i:03d}": [0] * 90 for i in range(width)}})
    if calendar_kind == "parquet":
        calendar["f002"] = np.linspace(-1., 1., len(dates))
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
    if calendar_kind == "parquet":
        cfg["model"].pop("calendar_features")
        cfg["curvas"].update(
            inputs=["X0", "custom"],
            input_specs={
                "X0": {"historical": [], "future": []},
                "custom": {"historical": ["f002", "f000"], "future": ["f001"]},
            },
            policies={"segmenter": "none", "fallback": ["global"],
                      "transform": "linear_history", "selection_metric": "scaled_mae"},
            final_epochs=2,
        )
    if calendar_kind == "dates_only":
        cfg["model"].pop("calendar_features")
        cfg["curvas"].update(inputs=["X0"], final_epochs=2)
    if calendar_kind == "families":
        cfg['curvas'].update(families=dict(encoder_decoder=1,tide=1,nhits=1,nlinear=1,dlinear=1,nbeats=1))
    if calendar_kind == "mad":
        cfg["curvas"].update(
            families={"nlinear": 1, "dlinear": 1, "rlinear": 1, "nbeats": 1,
                      "nhits": 4, "tide": 4, "encoder_decoder": 4},
            final_epochs=2, activations=["GELU"], encoder_kinds=["gru"],
            input_search={"mode": "family_objective", "startup_trials_per_input": 1},
            interval={"method": "mad", "window": 4},
            policies={"selection_metric": "scaled_mae"},
            pruner={"kind": "patience", "patience": 3},
        )
    if calendar_kind == "pooled":
        del cfg["curvas"]["trials"]
        cfg["curvas"].update(
            families={"tide": 5, "nhits": 5},
            input_search={"mode": "family_objective", "startup_trials_per_input": 1},
            final_epochs=2,
        )
    group = None
    if calendar_kind.startswith("group_"):
        from model_arena import RunSpec

        cfg["model"]["epochs"] = 2
        cfg["curvas"].update(
            final_epochs=2, encoder_kinds=["gru"],
            interval={"method": "mad", "window": 4},
            policies={"selection_metric": "scaled_mae"},
        )
        tree = calendar_kind == "group_tree"
        group = RunSpec.model_validate({
            "name": calendar_kind,
            "models": [{"model": "lightgbm" if tree else "encoder_decoder", "inputs": ["X0", "X3"]}],
            "objectives": {"saldo": 2, "variacion": 2},
            "tree_search": {"threads": 1, "min_data_in_leaf": [2]} if tree else None,
            "residuals": {} if tree else {
                "saldo": {"baseline": "last"}, "variacion": {"baseline": "learned", "widths": [4]},
            },
        })
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
        run_spec=group,
    )
    assert project.calendar_provenance()["feature_count"] == width
    if group is not None:
        plan = project.plan()
        assert plan["trials_by_objective"] == {"saldo": 2, "variacion": 2}
        assert plan["hpo_fits"] == 12
    if calendar_kind == "mad":
        plan = project.plan()
        assert len(plan["studies"]) == 14 and plan["hpo_fits"] == 96
    if calendar_kind == "parquet":
        assert project.plan()["final_epochs"] == 2
        assert project.plan()["hpo_fits"] == 12
        assert project.plan()["studies"][-1]["future"] == ["f001"]
    if calendar_kind == "pooled":
        plan = project.plan()
        assert len(plan["studies"]) == 4
        assert plan["candidate_configurations"] == 20
        assert plan["hpo_fits"] == 60
    project.prepare().tune()
    import polars as pl

    audit_path = project.path / "run/curvas/hpo_trials.parquet"
    audit = pl.read_parquet(audit_path)
    if calendar_kind == "csv":
        assert audit.filter(pl.col("state") == "COMPLETE").height == 24
        pruned = audit.filter(pl.col("state") == "PRUNED")
        assert pruned.height == 24 and set(pruned["pruned_epoch"]) == {5}
        for candidate in pruned["candidate"]:
            history = pl.read_parquet(project.path / f"run/curvas/loss/{candidate}_F1.parquet")
            assert history["epoch"].to_list() == [1, 2, 3, 4, 5]
            assert not (project.path / f"run/curvas/modelos/{candidate}_F1.pt").exists()
        project.tune()
        assert pl.read_parquet(audit_path).equals(audit)
    if calendar_kind == "pooled":
        assert audit.height == 20
        for family in ("tide", "nhits"):
            for target in ("saldo", "variacion"):
                study = optuna.load_study(
                    study_name=f"pooled_{family}_{target}",
                    storage=f"sqlite:///{project.path / 'run/curvas/hpo.sqlite3'}",
                )
                assert [trial.params["input"] for trial in study.trials[:4]] == ["X0", "X1", "X2", "X3"]
                assert len(study.trials) == 5
        for row in audit.to_dicts():
            assert row["candidate"].startswith(row["input"] + "_")
            assert json.loads(row["parameters_json"])["input"] == row["input"]
        project.tune()
        assert pl.read_parquet(audit_path).equals(audit)
    project.select().fit()
    weights = list((project.path / "run/curvas/modelos").glob("*_FUTURO.pt"))
    assert weights
    for checkpoint in weights:
        saved=torch.load(checkpoint,weights_only=False)
        target=saved['candidate']['target']
        other='variacion' if target=='saldo' else 'saldo'
        assert saved['candidate']['encoder_design']=='single'
        assert saved['training_history'][0]['train_loss_'+other] is None
        assert saved['training_history'][0]['fixed_train_'+other] is None
        if saved['candidate']['family']!='encoder_decoder':
            assert not any(key.startswith('external.networks.1.') for key in saved['state_dict'])
    chosen = pl.read_parquet(project.path / "run/curvas/winners_FUTURO.parquet")
    assert chosen.height == 6
    assert set(chosen["target"]) == {"saldo", "variacion"}
    assert chosen.unique(["account_id", "target"]).height == 6
    before = {str(p): digest(p) for p in weights}
    predictions = project.predict()
    frame = pl.read_parquet(predictions)
    assert set(frame["account_id"]) == {"A", "B", "C"}
    project.postprocess()
    assert before == {str(p): digest(p) for p in weights}
    manifest = json.loads((project.path / "release/publication.json").read_text())
    if calendar_kind == "mad":
        assert audit.height == 32
        assert manifest["interval"]["method"] == "mad"
        assert manifest["view"]["quantiles"] == []
        assert manifest["view"]["interval_method"] == "mad"
        variation = frame.filter(pl.col("target") == "variacion").pivot(
            on="component", index=["account_id", "date", "method"], values="forecast")
        np.testing.assert_allclose(variation["Q2.5"].to_numpy(), -variation["Q97.5"].to_numpy())
        assert manifest["datasets"]["events"]["rows"] > 0
    assert len(manifest["datasets"]["engine2"]["columns"]) == 15
    assert len(manifest["datasets"]["engine3"]["columns"]) == 22
    assert manifest["datasets"]["training_learning"]["rows"] > 0
    assert manifest["datasets"]["training_entities"]["rows"] > 0
    assert manifest["datasets"]["calendar_provenance"]["rows"] == 1
    assert manifest["datasets"]["hpo_trials"]["rows"] == audit.height
    assert manifest["calendar_provenance"]["feature_count"] == width
    assert [t["id"] for t in manifest["view"]["targets"]] == ["saldo", "variacion"]
    if calendar_kind == "pooled":
        assert manifest["input_search"]["mode"] == "family_objective"
    if calendar_kind == "parquet":
        assert manifest["policies"]["selection_metric"] == "scaled_mae"
        assert manifest["policies"]["fallback"] == ["global"]
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
    assert all(torch.load(p, weights_only=False)["spec"]["epochs"] == cfg["curvas"]["final_epochs"] for p in weights)
