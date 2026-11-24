import calendar
import itertools
from datetime import date
from functools import lru_cache

import numpy as np
import torch

CALENDAR_FIELDS = (
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "monthday_sin",
    "monthday_cos",
    "month_end",
)


@lru_cache(maxsize=8192)
def calendar_vector(value):
    d = date.fromisoformat(str(value))
    return (
        np.sin(2 * np.pi * d.weekday() / 7),
        np.cos(2 * np.pi * d.weekday() / 7),
        np.sin(2 * np.pi * (d.month - 1) / 12),
        np.cos(2 * np.pi * (d.month - 1) / 12),
        np.sin(2 * np.pi * (d.day - 1) / 31),
        np.cos(2 * np.pi * (d.day - 1) / 31),
        float(d.day == calendar.monthrange(d.year, d.month)[1]),
    )


def entropy_rows(values):
    """Weighted permutation entropy, m=3, stable ties, vectorized over rows."""
    values = np.asarray(values)
    if values.shape[-1] < 3:
        return np.zeros(values.shape[:-1])
    windows = np.lib.stride_tricks.sliding_window_view(values, 3, axis=-1)
    patterns = np.argsort(windows, axis=-1, kind="stable")
    weights = np.var(windows, axis=-1)
    total = weights.sum(axis=-1)
    groups = []
    for pattern in itertools.permutations(range(3)):
        groups.append(np.sum(weights * np.all(patterns == pattern, axis=-1), axis=-1))
    p = np.stack(groups, axis=-1) / np.maximum(total[..., None], 1e-30)
    return -np.sum(
        np.where(p > 0, p * np.log(np.maximum(p, 1e-30)), 0), axis=-1
    ) / np.log(6)


def weighted_entropy(x):
    return float(entropy_rows(np.asarray(x)))


def correlations(x, lag):
    if x.shape[-1] <= lag:
        return np.zeros(x.shape[:-1])
    a = x[..., :-lag]
    b = x[..., lag:]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    return np.divide(
        np.sum(a * b, axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-24,
    )


def batch(
    series,
    refs,
    spec,
    candidate,
    device,
    currency_map,
    observed=True,
    calendar_table=None,
):
    history = np.stack([series[i].values[o - spec.window : o] for i, o in refs])
    scale = np.maximum(
        np.maximum(
            np.median(np.abs(history), axis=1),
            np.quantile(history, 0.75, axis=1) - np.quantile(history, 0.25, axis=1),
        ),
        1.0,
    )
    z = np.arcsinh(history / scale[:, None, :])
    result = {"history_y": z, "scales": scale, "last_balance": history[:, -1, 0]}
    if candidate.inputs != "X0":
        if calendar_table is None and spec.calendar_features != 7:
            raise ValueError("Authoritative calendar required by specification")
        dates_history = [series[i].dates[o - spec.window : o] for i, o in refs]
        dates_future = [series[i].dates[o : o + spec.horizon] for i, o in refs]
        encode = (
            calendar_table.take
            if calendar_table
            else lambda dates: [calendar_vector(d) for d in dates]
        )
        result.update(
            position_history=np.broadcast_to(
                np.linspace(-1, 0, spec.window)[None, :, None],
                (len(refs), spec.window, 1),
            ),
            position_future=np.broadcast_to(
                np.linspace(1 / spec.horizon, 1, spec.horizon)[None, :, None],
                (len(refs), spec.horizon, 1),
            ),
            history_calendar=np.array([encode(ds) for ds in dates_history]),
            future_calendar=np.array([encode(ds) for ds in dates_future]),
            series_age_history=np.array(
                [
                    (np.arange(o - spec.window + 1, o + 1) / max(o, 1))[:, None]
                    for _, o in refs
                ]
            ),
        )
        if candidate.inputs == "X8":
            v = history / scale[:, None, :]
            diff = np.diff(v, axis=1)
            zero = (history == 0).mean(axis=1)
            negative = (history < 0).mean(axis=1)
            absolute_diff = np.diff(history, axis=1, prepend=history[:, :1, :])
            local = np.stack(
                [
                    z.mean(axis=1),
                    z.std(axis=1),
                    zero,
                    negative,
                    np.abs(absolute_diff).mean(axis=1) / scale,
                ],
                axis=-1,
            ).reshape(len(refs), 10)
            median = np.median(np.abs(v), axis=1)
            iqr = np.quantile(v, 0.75, axis=1) - np.quantile(v, 0.25, axis=1)
            origins = np.array([o for _, o in refs])
            span = np.array(
                [
                    max(
                        (
                            date.fromisoformat(series[i].dates[o - 1])
                            - date.fromisoformat(series[i].dates[0])
                        ).days,
                        0,
                    )
                    for i, o in refs
                ]
            )
            age = np.broadcast_to(origins[:, None], (len(refs), 2))
            span2 = np.broadcast_to(span[:, None], age.shape)
            channel_first = np.swapaxes(v, 1, 2)
            a = correlations(channel_first, 5)
            b = correlations(channel_first, 21)
            numeric = np.stack(
                [
                    entropy_rows(channel_first),
                    np.log1p(age),
                    np.log1p(span2),
                    age / np.maximum(span2 + 1, 1),
                    np.log1p(median),
                    np.log1p(np.abs(iqr)),
                    zero,
                    negative,
                    diff.std(axis=1),
                    (v[:, -1] - v[:, 0])
                    / np.maximum((spec.window - 1) * np.maximum(median, 1e-6), 1e-6),
                    a,
                    b,
                    np.maximum(np.abs(a), np.abs(b)),
                ],
                axis=-1,
            ).reshape(len(refs), 26)
            result.update(
                local_context=local,
                static_numeric=numeric,
                currency_index=np.array(
                    [currency_map.get(series[i].currency, 0) for i, _ in refs]
                ),
                budget_index=np.array([series[i].budget for i, _ in refs]),
            )
    if candidate.inputs == "X1":
        result.pop("future_calendar", None)
    elif candidate.inputs == "X2":
        result.pop("history_calendar", None)
    for key in ("position_history", "position_future", "series_age_history"):
        result.pop(key, None)
    if observed:
        result["future_y"] = np.arcsinh(
            np.stack([series[i].values[o : o + spec.horizon] for i, o in refs])
            / scale[:, None, :]
        )
    return {
        k: torch.as_tensor(np.nan_to_num(v), dtype=torch.float32, device=device)
        for k, v in result.items()
    }
