import json

from model_arena import ArenaSpec, ModelArena
from model_arena.presentation import winner_rows
from model_arena.results import Results


def test_one_target_neural_workflow_with_units_and_no_budgets(panel, spec, tmp_path):
    df, _ = panel
    demand = df.rename(columns={"level": "demanda"}).drop(columns=["change", "day"])
    config = spec.model_dump()
    config.update(
        schema={"id_col": "entity", "time_col": "date", "target_cols": ["demanda"]},
        models=[{"alias": "red", "kind": "encoder_decoder",
                 "params": {"encoder_kind": "mlp", "width": 8, "depth": 1}}],
        target_labels={"demanda": "Demanda"},
        target_units={"demanda": "piezas"},
    )
    arena = ModelArena(ArenaSpec.model_validate(config))
    study = arena.cross_validation(df=demand)
    selection = arena.select(study)
    arena.fit(df=demand, selection=selection)
    forecast = arena.predict()
    assert set(forecast.target) == {"demanda"}
    result = arena.publish(arena.postprocess(forecast), destination=tmp_path / "published")
    published = Results.open(result)
    view = json.loads((published.path / "spec.json").read_text())["view"]
    assert view["targets"] == [{"id": "demanda", "label": "Demanda", "unit": "piezas"}]
    assert "budget" not in published.table("inventory")


def test_winner_rows_decodes_multiple_objectives_and_legacy_scalar():
    targets = ["saldo", "variacion"]
    rows = winner_rows("A", '{"saldo":"DLinear","variacion":"TFT"}',
                       '{"saldo":"SERIE","variacion":"DIVISA_BUDGET"}', targets)
    assert [(row["candidate"], row["selection_source"]) for row in rows] == [
        ("DLinear", "SERIE"), ("TFT", "DIVISA_BUDGET")
    ]
    assert [row["candidate"] for row in winner_rows("A", "X0_T0000", "SERIE", targets)] == [
        "X0_T0000", "X0_T0000"
    ]
