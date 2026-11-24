"""Native accounting snapshot policy with exclusions; no model dependency."""

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

CLASSES = {"cuentas_entrenamiento", "cuentas_cero", "cuentas_atipicos"}


@dataclass
class AccountingUniverse:
    history: pl.DataFrame
    exclusions: list
    snapshot_load_date: str
    identity: str

    @classmethod
    def from_frame(cls, frame):
        required = {
            "account_id",
            "date",
            "balance",
            "currency",
            "taxonomy",
            "sign",
            "load_date",
            "fx",
        }
        if required - set(frame.columns):
            raise ValueError("Missing columns: " + str(required - set(frame.columns)))
        frame = frame.with_columns(
            pl.col("account_id").cast(pl.String).str.strip_chars(),
            pl.col("currency").cast(pl.String).str.strip_chars().str.to_uppercase(),
            pl.col("date").cast(pl.String),
            pl.col("load_date").cast(pl.String),
            pl.col("balance").cast(pl.Float64),
            pl.col("fx").cast(pl.Float64),
        )
        for row in (
            frame.select("account_id", "currency").unique().iter_rows(named=True)
        ):
            if (
                not row["currency"]
                or not row["account_id"].endswith(row["currency"])
                or len(row["account_id"]) <= len(row["currency"])
            ):
                raise ValueError(
                    "Account must retain a prefix and end with its full source currency code"
                )
        for key in ("date", "load_date"):
            for x in frame[key].unique():
                date.fromisoformat(x)
        if (
            frame.select(
                pl.any_horizontal(
                    pl.col("balance").is_null() | ~pl.col("balance").is_finite(),
                    pl.col("fx").is_null()
                    | ~pl.col("fx").is_finite()
                    | (pl.col("fx") <= 0),
                )
            )
            .to_series()
            .any()
        ):
            raise ValueError("Nonfinite amount or invalid FX")
        latest = frame["load_date"].max()
        meta = (
            frame.filter(pl.col("load_date") == latest)
            .select("account_id", "currency", "taxonomy", "sign")
            .unique()
        )
        ambiguous = meta.group_by("account_id").len().filter(pl.col("len") > 1)
        if ambiguous.height:
            raise ValueError("Ambiguous current Engine1 metadata")
        # Same-load conflicting balances never receive an arbitrary winner.
        conflicts = (
            frame.group_by(["account_id", "date", "load_date"])
            .agg(pl.col("balance").n_unique().alias("versions"))
            .filter(pl.col("versions") > 1)
        )
        if conflicts.height:
            raise ValueError("Conflicting balances at the same load")
        current = (
            frame.sort("load_date")
            .unique(["account_id", "date"], keep="last")
            .sort(["account_id", "date"])
        )
        # Join current snapshot metadata in native columnar storage. Never expand
        # millions of observations into Python dictionaries.
        snapshot = meta.rename({"currency": "__snapshot_currency",
                                "taxonomy": "__snapshot_taxonomy", "sign": "__snapshot_sign"})
        classified = current.join(snapshot, on="account_id", how="left")
        missing = classified.filter(pl.col("__snapshot_taxonomy").is_null())
        excluded = []
        if missing.height:
            present = set(meta["account_id"].to_list())
            for row in missing.group_by("account_id").agg(
                pl.len().alias("observations"), pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
            ).sort("account_id").iter_rows(named=True):
                row["reason"] = ("UNCLASSIFIED_IN_CURRENT_ENGINE1_SNAPSHOT" if row["account_id"] in present
                                 else "ABSENT_FROM_CURRENT_ENGINE1_SNAPSHOT")
                excluded.append(row)
        classified = classified.filter(pl.col("__snapshot_taxonomy").is_not_null())
        if not classified.height:
            raise ValueError("No accounts classified in current snapshot")
        valid_sign = pl.col("__snapshot_sign").is_in(["positiva", "negativa", "ambos_signos"]) | (
            (pl.col("__snapshot_taxonomy") == "cuentas_cero") & (pl.col("__snapshot_sign") == "cuenta_cero")
        )
        if classified.filter(~(pl.col("__snapshot_taxonomy").is_in(sorted(CLASSES)) & valid_sign).fill_null(False)).height:
            raise ValueError("Invalid Engine1 classification")
        currency_conflicts = classified.group_by("account_id").agg(
            pl.col("currency").n_unique().alias("variants"),
            (pl.col("currency") != pl.col("__snapshot_currency")).any().alias("mismatch"),
        ).filter((pl.col("variants") != 1) | pl.col("mismatch"))
        if currency_conflicts.height:
            raise ValueError("Currency mismatch")
        history = classified.with_columns(
            pl.col("__snapshot_taxonomy").alias("taxonomy"),
            pl.col("__snapshot_sign").alias("sign"),
        ).drop("__snapshot_currency", "__snapshot_taxonomy", "__snapshot_sign").sort(["account_id", "date"])
        # Preserve the exact canonical CSV SHA-256 without a giant CSV string
        # or a second file on the user's limited disk.
        class HashSink:
            def __init__(self): self.hasher = hashlib.sha256()
            def write(self, data):
                if isinstance(data, str): data = data.encode()
                self.hasher.update(data)
                return len(data)
            def flush(self): pass
        sink = HashSink()
        frame.sort(frame.columns).write_csv(sink)
        return cls(history, excluded, latest, sink.hasher.hexdigest())

    @classmethod
    def read_csv(cls, path):
        return cls.from_frame(pl.read_csv(path, infer_schema_length=10000))

    def to_arena(self, spec, calendar_table=None, operational_dates=None):
        from agentic_forecasting import Series
        from agentic_forecasting.data import weighted_entropy

        series = []
        eligibility = []
        minimum = spec.minimum_history
        observed = set(self.history["date"].unique().to_list())
        calendar = sorted(set(operational_dates) if operational_dates is not None else observed)
        if calendar_table:
            calendar = sorted(set(calendar) & set(calendar_table.dates))
        if not calendar or observed - set(calendar):
            raise ValueError("Source dates absent from operational calendar")
        position = {d: i for i, d in enumerate(calendar)}
        for (aid,), g in self.history.group_by("account_id", maintain_order=True):
            if g["taxonomy"][0] not in ("cuentas_entrenamiento", "cuentas_atipicos"):
                continue
            dates = g["date"].to_list()
            indices = [position[d] for d in dates]
            if (
                len(dates) < minimum
                or indices != list(range(indices[0], indices[-1] + 1))
                or dates[-1] != calendar[-1]
            ):
                eligibility.append(
                    {
                        "account_id": aid,
                        "reason": "INSUFFICIENT_COMMON_PROTOCOL_HISTORY"
                        if len(dates) < minimum
                        else "STALE_ACCOUNT_END_DATE"
                        if dates[-1] != calendar[-1]
                        else "INTERNAL_HISTORY_GAPS",
                        "observations": len(dates),
                        "required_observations": minimum,
                    }
                )
                continue
            balance = g["balance"].to_numpy()
            variation = np.diff(balance, prepend=balance[0])
            values = np.column_stack([balance, variation])
            prefix = (
                len(values)
                - spec.evaluation_span
                - (spec.calibration if "standard" in spec.recipes else 0)
                - spec.cv_folds*spec.validation
            )
            budget = min(10, int(weighted_entropy(balance[:prefix]) * 10) + 1)
            future = [
                (date.fromisoformat(dates[-1]) + timedelta(days=j)).isoformat()
                for j in range(1, spec.horizon + 1)
            ]
            if calendar_table:
                future = calendar_table.after(dates[-1], spec.horizon)
            series.append(Series(aid, dates + future, values, g["currency"][0], budget, "zero_shot" if g["taxonomy"][0] == "cuentas_atipicos" else "seen"))
        groups = {}
        for x in series:
            if x.cohort == "seen":
                groups.setdefault(x.budget, []).append(x)
        target = max(1, int(len(series) * 0.1)) if len(series) >= 5 else 0
        selected = 0
        for _, group in sorted(
            groups.items(), key=lambda pair: (-len(pair[1]), pair[0])
        ):
            ordered = sorted(
                group,
                key=lambda x: hashlib.sha256(
                    f"{spec.seed}:{x.id}".encode()
                ).hexdigest(),
            )
            for x in ordered[: min(max(0, len(group) - 1), max(0, target - selected))]:
                x.cohort = "zero_shot"
                selected += 1
        if not series:
            raise ValueError("No trainable accounts eligible for the common protocol")
        seen=[x for x in series if x.cohort=="seen"]
        entropies={x.id:weighted_entropy(x.values[:len(x.values)-spec.evaluation_span-spec.cv_folds*spec.validation,0]) for x in series}
        if seen:
            edges=np.quantile([entropies[x.id] for x in seen],np.arange(1,10)/10)
            for x in series:x.budget=int(np.searchsorted(edges,entropies[x.id],side="right"))+1
        return series, eligibility


def synthetic_accounts(accounts=8, observations=180, seed=42, dates=None):
    rng = np.random.default_rng(seed)
    rows = []
    first = date(2025, 1, 1)
    for i in range(accounts + 2):
        currency = "MXP" if i % 2 == 0 else "USD"
        aid = f"DEMO{i:03d}" + currency
        taxonomy = (
            "cuentas_entrenamiento"
            if i < accounts
            else "cuentas_cero"
            if i == accounts
            else "cuentas_atipicos"
        )
        x = np.arange(observations)
        balance = (
            1000
            + 50 * np.sin(x / (3 + i % 3))
            + np.cumsum(rng.normal(0, 3 + i % 11, observations))
        )
        if taxonomy == "cuentas_cero":
            balance[:] = 0
            balance[-1] = 600000
        if taxonomy == "cuentas_atipicos":
            balance = np.where(x % 25 == 0, 400.0, 0.0)
        for j in range(observations):
            rows.append(
                {
                    "account_id": aid,
                    "date": dates[j]
                    if dates is not None
                    else (first + timedelta(days=j)).isoformat(),
                    "balance": float(balance[j]),
                    "currency": currency,
                    "taxonomy": taxonomy,
                    "sign": "positiva",
                    "load_date": "2026-01-01",
                    "fx": 1.0 if currency == "MXP" else 18.0,
                }
            )
    # Generic absent snapshot case used only as explicitly synthetic data.
    rows.append(
        {
            "account_id": "ARCHIVEDXYZ1",
            "date": dates[0] if dates is not None else first.isoformat(),
            "balance": 5.0,
            "currency": "XYZ1",
            "taxonomy": "cuentas_atipicos",
            "sign": "positiva",
            "load_date": "2025-01-01",
            "fx": 2.0,
        }
    )
    return pl.DataFrame(rows)
