from pathlib import Path

import polars as pl
import pytest

from model_arena.loss_store import write_parquet


@pytest.fixture
def local_curves(tmp_path):
    root = tmp_path / "run/curvas"
    for candidate in ["X0_T0000", "X0_T0001"]:
        for fold in ["F1", "F2", "F3"]:
            rows = []
            for epoch in [1, 2]:
                rows.append(
                    dict(
                        candidate=candidate,
                        fold=fold,
                        epoch=epoch,
                        fixed_train_saldo=1.0 / epoch,
                        fixed_train_variacion=2.0 / epoch,
                        fixed_validation_saldo=3.0 / epoch,
                        fixed_validation_variacion=4.0 / epoch,
                        train_loss_saldo=9.0 / epoch,
                        train_loss_variacion=8.0 / epoch,
                        validation_loss=None,
                    )
                )
                detail = []
                for account, budget, loss in [("A", 1, 1.0), ("B", 2, 3.0)]:
                    for target in ["saldo", "variacion"]:
                        for split in ["train", "validation"]:
                            detail.append(
                                dict(
                                    candidate=candidate,
                                    fold=fold,
                                    epoch=epoch,
                                    account_id=account,
                                    budget=budget,
                                    target=target,
                                    split=split,
                                    loss=loss,
                                    n_observations=12,
                                )
                            )
                write_parquet(
                    pl.DataFrame(detail),
                    root / "loss_entities" / f"{candidate}_{fold}" / f"epoch_{epoch:04d}.parquet",
                )
            write_parquet(pl.DataFrame(rows), root / "loss" / f"{candidate}_{fold}.parquet")
    write_parquet(pl.DataFrame({"account_id": ["A", "B"], "budget": [1, 2]}), root / "inventory.parquet")
    return tmp_path


def test_queries_restrict_fold_budget_and_account(local_curves, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from model_arena.profiles.loss_queries import curves, figure

    total = curves(local_curves)
    assert total.height == 48
    assert curves(local_curves, folds=["F1"]).height == 16
    budget = curves(local_curves, scope="Budget", value=2)
    account = curves(local_curves, scope="Cuenta", value="B")
    assert budget.equals(account)
    assert set(budget["loss"]) == {3.0}
    assert set(curves(local_curves, sampled=True)["split"]) == {"train"}
    chart = figure(total, "X0_T0000")
    assert len(chart.data) == 24
    assert all(trace.hovertemplate and "Época" in trace.hovertemplate for trace in chart.data)


def test_streamlit_local_dashboard_graph_and_table(local_curves, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import model_arena

    bundle = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(bundle))
    monkeypatch.setenv("OBS_LOSS_PROJECT", str(local_curves))
    app = AppTest.from_file(
        str(Path(model_arena.__file__).parent / "profiles/loss_app.py"), default_timeout=20
    ).run()
    assert not app.exception
    next(w for w in app.radio if w.label == "Desglose").set_value("Budget").run()
    assert not app.exception
    next(w for w in app.radio if w.label == "Presentación").set_value("Tabla").run()
    assert not app.exception
    assert len(app.dataframe) == 1
