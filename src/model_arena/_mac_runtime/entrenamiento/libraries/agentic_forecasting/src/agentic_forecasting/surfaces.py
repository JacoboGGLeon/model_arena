"""Columnar forecast artifacts: small numeric arrays instead of millions of dicts."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ForecastSurface:
    metadata: dict
    point: object
    observed: object
    lower: object
    upper: object

    def save(self, prefix):
        prefix = Path(prefix)
        np.savez_compressed(
            str(prefix) + ".npz",
            point=np.asarray(self.point, dtype=np.float32),
            observed=np.asarray(self.observed, dtype=np.float32),
            lower=np.asarray(self.lower, dtype=np.float32),
            upper=np.asarray(self.upper, dtype=np.float32),
        )
        Path(str(prefix) + ".meta.json").write_text(
            json.dumps(self.metadata, allow_nan=False)
        )

    @classmethod
    def load(cls, prefix):
        prefix = Path(prefix)
        with np.load(str(prefix) + ".npz", allow_pickle=False) as a:
            return cls(
                json.loads(Path(str(prefix) + ".meta.json").read_text()),
                **{k: a[k] for k in ("point", "observed", "lower", "upper")},
            )

    def rows(self, account_ids=None):
        m = self.metadata
        for c, coverage in enumerate(m["coverages"]):
            for i, aid in enumerate(m["series_ids"]):
                if account_ids is not None and aid not in account_ids:
                    continue
                for h in range(self.point.shape[1]):
                    for t, target in enumerate(("saldo", "variacion")):
                        base = {
                            "series_id": aid,
                            "origin": m.get("origins_by_step", [[v]*self.point.shape[1] for v in m["origins"]])[i][h],
                            "target_date": m["target_dates"][i][h],
                            "horizon": 1 if m.get("origin_mode") == "rolling" else h + 1,
                            "position": h + 1,
                            "target": target,
                            "point": float(self.point[i, h, t]),
                            "observed": float(self.observed[i, h, t])
                            if m["phase"] in ("test", "horizon")
                            else None,
                            "coverage": coverage,
                            "candidate_id": m["candidate_id"],
                            "recipe": m["recipe"],
                            "currency": m["currencies"][i],
                            "cohort": m["cohorts"][i],
                            "phase": m["phase"],
                        }
                        for r, role in enumerate(m["roles"]):
                            yield base | {
                                "role": role,
                                "method": (
                                    "conformal"
                                    if m["recipe"] == "standard"
                                    else "quantiles"
                                )
                                if r == 0
                                else "mc_dropout",
                                "lower": float(self.lower[r, c, i, h, t]),
                                "upper": float(self.upper[r, c, i, h, t]),
                            }
