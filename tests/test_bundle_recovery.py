import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

from model_arena.profiles.mac import MacProject
from model_arena.storage import digest
from model_arena.hpo_resume import enqueue_interrupted

BUNDLE = Path(__file__).resolve().parents[1]


def test_hpo_retry_keeps_candidate_identity_and_does_not_duplicate(tmp_path):
    import optuna

    study = optuna.create_study(storage="sqlite:///" + str(tmp_path / "hpo.sqlite3"))
    trial = study.ask()
    trial.suggest_float("lr", 1e-5, 1e-2)
    trial.set_user_attr("candidate_id", "X0_T0000")
    trial.set_user_attr("retry_after_disk", True)
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    enqueue_interrupted(study, "X0")
    enqueue_interrupted(study, "X0")
    waiting = study.get_trials(states=[optuna.trial.TrialState.WAITING])
    assert len(waiting) == 1
    retried = study.ask()
    assert retried.user_attrs["resume_candidate_id"] == "X0_T0000"
    assert retried.suggest_float("lr", 1e-5, 1e-2) == trial.params["lr"]
    retried.set_user_attr("candidate_id", "X0_T0000")
    study.tell(retried, 1.0)
    enqueue_interrupted(study, "X0")
    assert not study.get_trials(states=[optuna.trial.TrialState.WAITING])


def test_project_lock_blocks_concurrent_writer(tmp_path):
    (tmp_path / "project.json").write_text("{}")
    project = MacProject(tmp_path)
    with project.exclusive():
        with pytest.raises(RuntimeError, match="activa"):
            with MacProject(tmp_path).exclusive():
                pass


@pytest.mark.train
def test_upgrade_old_run_and_continue_without_refitting_completed_folds(tmp_path, monkeypatch):
    import torch
    import model_arena
    import optuna

    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [
                str(Path(model_arena.__file__).resolve().parent.parent),
                str(Path(torch.__file__).parents[1]),
                str(Path(optuna.__file__).parents[1]),
            ]
        ),
    )
    monkeypatch.syspath_prepend(str(BUNDLE))
    dates = pd.date_range("2025-01-01", periods=90).strftime("%Y-%m-%d").tolist()
    pd.DataFrame({"fecha": dates, **{f"f{i:03d}": [0] * 90 for i in range(243)}}).to_csv(
        tmp_path / "calendar.csv", index=False
    )
    rows = []
    for aid, tax in [
        ("A", "cuentas_entrenamiento"),
        ("B", "cuentas_entrenamiento"),
        ("C", "cuentas_atipicos"),
    ]:
        for i, date in enumerate(dates[:60]):
            rows.append(
                dict(
                    account_id=aid,
                    date=date,
                    currency="USD",
                    taxonomy=tax,
                    sign="positiva",
                    load_date=dates[59],
                    balance=float(100 + i),
                    variation=1.0,
                    fx=1.0,
                )
            )
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
            inputs=["X0"],
            encoders=["shared"],
            depths=[1],
            widths=[8],
            decoder_depths=[1],
            activations=["GELU"],
            dropout=0.0,
            trials=1,
            lr_range=[0.001, 0.002],
            wd_range=[1e-6, 1e-5],
        ),
    )
    (tmp_path / "training.json").write_text(json.dumps(cfg))
    (tmp_path / "publication.json").write_text(
        json.dumps(dict(s3_base="s3://example/objects/", query_output="s3://example/results/", shards=2))
    )
    project = MacProject.create(
        tmp_path / "project",
        training_config=tmp_path / "training.json",
        publication_config=tmp_path / "publication.json",
        calendar_path=tmp_path / "calendar.csv",
        device="cpu",
    )
    shutil.rmtree(project.path / "runtime")
    shutil.copytree(BUNDLE / "tests/fixtures/runtime_01", project.path / "runtime")
    calendar = project.path / "runtime/entrenamiento/data/calendar_authoritative.csv"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path / "calendar.csv", calendar)
    settings = json.loads((project.path / "project.json").read_text())
    settings["code_sha"] = digest(project.path / "runtime/BUILD_MONITOR.json")
    (project.path / "project.json").write_text(json.dumps(settings))
    project = MacProject(project.path)
    project.prepare().tune()
    curves = project.path / "run/curvas"
    weights = {p.name: digest(p) for p in (curves / "modelos").glob("*.pt")}
    predictions = {p.name: digest(p) for p in (curves / "predicciones").glob("*.parquet")}
    database_before = digest(curves / "hpo.sqlite3")
    assert list((curves / "loss_entities").glob("*.csv"))
    from model_arena.profiles.storage_upgrade import upgrade

    project = upgrade(project.path)
    assert database_before == digest(curves / "hpo.sqlite3")
    assert not list((curves / "loss_entities").glob("*.csv"))
    assert list((curves / "loss_entities").glob("*.parquet"))
    project.prepare().tune().select().fit().predict()
    project.postprocess()
    assert weights == {name: digest(curves / "modelos" / name) for name in weights}
    assert predictions == {name: digest(curves / "predicciones" / name) for name in predictions}
    manifest = json.loads((project.path / "release/publication.json").read_text())
    assert manifest["datasets"]["training_entities"]["rows"] > 0
    assert manifest["datasets"]["engine2"]["rows"] > 0
