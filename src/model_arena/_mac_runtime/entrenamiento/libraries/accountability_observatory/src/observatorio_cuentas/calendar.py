"""Read the frozen operational calendar and its numeric features."""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

ADA_SHA256 = "94ec45a2298d64a3bf54d7be1289ac9565745597f7a32f8eae0e77b52b474090"


def read_calendar(path, expected_sha256=None):
    from agentic_forecasting import CalendarTable

    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise ValueError("Authoritative calendar SHA-256 mismatch")
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    dates = frame["fecha"].to_list()
    names = [x for x in frame.columns if x != "fecha"]
    features = frame.select(names).to_numpy() if names else np.empty((len(dates), 0))
    return CalendarTable(dates, features, names, actual)
