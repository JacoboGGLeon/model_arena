import numpy as np
import pandas as pd
import pytest
from model_arena import ArenaSpec, ModelSpec, PanelSchema


@pytest.fixture
def panel():
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    rows = []
    for i, entity in enumerate(["alpha", "beta", "new"]):
        for t, date in enumerate(dates):
            rows.append(dict(entity=entity, date=date, level=float(10+i+t/3+np.sin(t)),
                             change=float(np.sin(t)), day=float(date.dayofweek)/6))
    df = pd.DataFrame(rows)
    static = pd.DataFrame(dict(entity=["alpha", "beta", "new"],
                               population=["seen", "seen", "zero_shot"], segment=[1, 1, 1]))
    return df, static


@pytest.fixture
def spec(tmp_path):
    return ArenaSpec(schema=PanelSchema(id_col="entity", time_col="date", target_cols=["level", "change"],
                                       population_col="population", group_cols=["segment"]),
                     models=[ModelSpec(alias="history", kind="historical_quantile")],
                     freq="D", h=2, window=4, artifacts=str(tmp_path / "runs"),
                     training=dict(epochs=2, final_epochs=2, samples_per_epoch=8, batch_size=4,
                                   cpu_threads=1))

