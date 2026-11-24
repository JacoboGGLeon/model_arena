"""Known exogenous calendar table; dates never inferred from target values."""

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass
class CalendarTable:
    dates: list
    features: object
    names: list
    source_sha256: str

    def __post_init__(self):
        self.features = np.asarray(self.features, dtype=np.float32)
        if self.dates != sorted(set(self.dates)) or self.features.shape != (
            len(self.dates),
            len(self.names),
        ):
            raise ValueError("Calendar dates/features are not aligned and unique")
        for d in self.dates:
            date.fromisoformat(d)
        if not np.isfinite(self.features).all():
            raise ValueError("Calendar contains nonfinite features")
        self._index = {d: i for i, d in enumerate(self.dates)}

    def take(self, dates):
        try:
            return self.features[[self._index[d] for d in dates]]
        except KeyError as e:
            raise ValueError(
                f"Date absent from authoritative calendar: {e.args[0]}"
            ) from e

    def after(self, cutoff, horizon):
        dates = [d for d in self.dates if d > cutoff][:horizon]
        if len(dates) != horizon:
            raise ValueError("Authoritative calendar has insufficient future dates")
        return dates
