"""Strict aligned panel validation, explicit eligibility and causal windows."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .storage import signature


@dataclass
class PreparedData:
    frame: pd.DataFrame
    metadata: pd.DataFrame
    axis: pd.DatetimeIndex
    ids: list[str]
    values: dict
    signature: str
    schema: object

    @property
    def training_ids(self):
        schema = self.schema
        if schema.population_col is None:
            return self.ids
        return self.metadata.loc[self.metadata[schema.population_col] == schema.training_label,
                                 schema.id_col].tolist()


def prepare_panel(df, static_df, spec):
    schema = spec.schema
    frame = df.copy(deep=True)
    required = [schema.id_col, schema.time_col, *schema.target_cols]
    features = sorted({f for m in spec.models for f in m.historical_features + m.future_features})
    missing = set(required + features) - set(frame)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame[required].isna().any().any():
        raise ValueError("Null identifiers, dates or targets")
    frame[schema.id_col] = frame[schema.id_col].astype(str)
    frame[schema.time_col] = pd.to_datetime(frame[schema.time_col], errors="raise")
    if frame.duplicated([schema.id_col, schema.time_col]).any():
        raise ValueError("Duplicate entity/date")
    if not np.isfinite(frame[schema.target_cols + features].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite targets or covariates")
    frame = frame.sort_values([schema.id_col, schema.time_col]).reset_index(drop=True)
    axis = pd.DatetimeIndex(sorted(frame[schema.time_col].unique()))
    if spec.calendar is not None:
        declared = pd.DatetimeIndex(pd.to_datetime(spec.calendar))
        if declared.has_duplicates or not declared.is_monotonic_increasing or not axis.isin(declared).all():
            raise ValueError("Invalid authoritative calendar")
        expected = declared[(declared >= axis[0]) & (declared <= axis[-1])]
    else:
        expected = pd.date_range(axis[0], axis[-1], freq=spec.freq)
    if not axis.equals(expected):
        raise ValueError("History has gaps or dates outside the declared calendar")
    groups = list(frame.groupby(schema.id_col, sort=True))
    if any(not pd.DatetimeIndex(group[schema.time_col]).equals(axis) for _, group in groups):
        raise ValueError("Each entity must have the same complete operational axis")
    metadata = (static_df.copy(deep=True) if static_df is not None
                else frame[[schema.id_col]].drop_duplicates())
    metadata[schema.id_col] = metadata[schema.id_col].astype(str)
    if metadata[schema.id_col].duplicated().any():
        raise ValueError("Static metadata must have exactly one row per entity")
    ids = [key for key, _ in groups]
    if set(ids) - set(metadata[schema.id_col]):
        raise ValueError("Static metadata missing entities")
    metadata = metadata[metadata[schema.id_col].isin(ids)].sort_values(schema.id_col).reset_index(drop=True)
    for column in [*schema.group_cols, *spec.selection.group_cols, schema.population_col]:
        if column is not None and (column not in metadata or metadata[column].isna().any()):
            raise ValueError("Missing metadata group/population: " + column)
    values = {key: group[schema.target_cols].to_numpy(dtype=np.float64) for key, group in groups}
    fingerprint = signature({"frame": pd.util.hash_pandas_object(frame, index=False).tolist(),
                             "metadata": pd.util.hash_pandas_object(metadata, index=False).tolist(),
                             "schema": schema.model_dump()})
    result = PreparedData(frame, metadata, axis, ids, values, fingerprint, schema)
    if not result.training_ids:
        raise ValueError("No training entities; zero-shot entities cannot train models")
    return result


def future_dates(data, spec, h):
    if spec.calendar is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(spec.calendar))
        dates = dates[dates > data.axis[-1]][:h]
        if len(dates) != h:
            raise ValueError("Authoritative calendar does not cover requested horizon")
        return dates
    return pd.date_range(data.axis[-1], periods=h + 1, freq=spec.freq)[1:]


def scale_for(history):
    return np.maximum(np.maximum(np.median(np.abs(history), axis=0),
                                  np.quantile(history, .75, axis=0)-np.quantile(history, .25, axis=0)), 1.)


def pinball(truth, prediction, quantiles):
    error = np.asarray(truth)[..., None] - np.asarray(prediction)
    q = np.asarray(quantiles)
    return np.maximum(q * error, (q - 1) * error)

