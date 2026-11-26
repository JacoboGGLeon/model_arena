"""Causal MAD envelopes for the MAC point-forecast protocol."""

from typing import Literal

import numpy as np
from pydantic import Field

from .contracts import Contract

MAD_SCALE = 1.4826


class MacIntervalSpec(Contract):
    method: Literal["percentiles", "mad"] = "percentiles"
    window: int = Field(default=20, ge=3, strict=True)
    multiplier: float = Field(default=2.0, gt=0, allow_inf_nan=False)


def mad_envelopes(point, history, *, target, window, multiplier):
    """Pack the lower limit, point and upper limit for the legacy MAC layout."""
    if target not in ("saldo", "variacion"):
        raise ValueError("MAD forecasts require an explicit saldo or variacion target")
    if history.ndim != 2 or history.shape[1] < window or not np.isfinite(history[:, -window:]).all():
        raise ValueError("Insufficient finite balance history for the MAD envelope")
    changes = np.diff(history[:, -window:], axis=1)
    median = np.median(changes, axis=1, keepdims=True)
    radius = multiplier * MAD_SCALE * np.median(np.abs(changes - median), axis=1)
    point = point[..., 0, 0]
    if target == "saldo":
        center = np.concatenate((history[:, -1:], point[:, :-1]), axis=1)
    else:
        center = np.zeros_like(point)
    return np.stack((center - radius[:, None], point, center + radius[:, None]), axis=-1)[:, :, None, :]


def point_forecast(candidate):
    return getattr(candidate, "interval", {}).get("method") == "mad"
