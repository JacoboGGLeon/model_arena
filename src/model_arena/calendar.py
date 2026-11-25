"""Calendar table adapter; accepts files or validated artifact-like providers.

The forecasting core has no dependency on a particular calendar rule engine.
MAC's fixed-feature contract is checked only at its integration boundary.
"""

from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
from importlib.metadata import version, PackageNotFoundError
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from .storage import write_json, digest


def to_features(calendar, *, date_col=None, time_col="ds", columns=None):
    """Return a detached exogenous table; do not regenerate or fit anything."""
    if isinstance(calendar, pd.DataFrame):
        frame = calendar.copy(deep=True)
    else:
        calendar.validate().raise_for_errors()
        frame = calendar.to_pandas()
        date_col = date_col or calendar.metadata.get("spec", {}).get("date_col")
    if not date_col or date_col not in frame:
        raise ValueError("Specify an existing date_col")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate calendar columns")
    names = list(columns) if columns is not None else [c for c in frame if c != date_col]
    if len(names) != len(set(names)) or date_col in names or time_col in names:
        raise ValueError("Calendar column selection/name collision")
    out = frame[[date_col, *names]].copy()
    dates = pd.to_datetime(out[date_col], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Calendar dates must be non-null local dates without time")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Calendar dates must be ordered and unique")
    values = out[names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Calendar features must be finite numeric values")
    out[date_col] = dates
    return out.rename(columns={date_col: time_col})


@dataclass
class CalendarSnapshot:
    csv_bytes: bytes
    frame: pd.DataFrame
    provenance: dict
    source_files: dict

    def save(self, project):
        folder = Path(project) / "calendar"
        folder.mkdir()
        files = {}
        for name, data in self.source_files.items():
            (folder / name).write_bytes(data)
            files[name] = digest(folder / name)
        self.frame.to_parquet(folder / "features.parquet", index=False, compression="zstd")
        files["features.parquet"] = digest(folder / "features.parquet")
        self.provenance["files"] = files
        write_json(folder / "provenance.json", self.provenance)
        return folder / "provenance.json"


def prepare_snapshot(*, calendar=None, calendar_path=None, n_features=243):
    if (calendar is None) == (calendar_path is None):
        raise ValueError("Pass exactly one of calendar or calendar_path")
    raw_csv = None
    source_files = {}
    metadata = {}
    generator = "external_table"
    generator_version = None
    if calendar_path is not None:
        path = Path(calendar_path).expanduser().resolve()
        if path.is_dir():
            try:
                from calendar_studio import CalendarArtifact
            except ImportError as exc:
                raise ImportError("Install calendar-studio to load its artifacts") from exc
            calendar = CalendarArtifact.load(path)
            source_files = {
                "source_manifest.json": (path / "manifest.json").read_bytes(),
                "source.parquet": (path / "calendar.parquet").read_bytes(),
            }
        elif path.suffix == ".csv":
            raw_csv = path.read_bytes()
            with path.open(encoding="utf-8-sig", newline="") as stream:
                names = next(csv.reader(stream))
            if len(names) != len(set(names)):
                raise ValueError("Duplicate CSV calendar columns")
            frame = pd.read_csv(path)
            source_files = {"source.csv": raw_csv}
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            source_files = {"source.parquet": path.read_bytes()}
        else:
            raise ValueError("Calendar input must be CSV, Parquet or an artifact directory")
    if isinstance(calendar, pd.DataFrame):
        frame = calendar.copy(deep=True)
        generator = "dataframe"
    elif calendar is not None:
        calendar.validate().raise_for_errors()
        frame = calendar.to_pandas()
        metadata = calendar.metadata
        generator = type(calendar).__module__.split(".")[0]
        try:
            generator_version = version(generator.replace("_", "-"))
        except PackageNotFoundError:
            generator_version = metadata.get("library_version")
    if frame.empty or not frame.columns.is_unique or frame.columns[0] != "fecha":
        raise ValueError("MAC calendar must start with a unique fecha column")
    names = [c for c in frame if c != "fecha"]
    if len(names) != n_features:
        raise ValueError(
            f'MAC expects {n_features} features, received {len(names)}. For Calendar Studio pass .view("mac_legacy_v1") explicitly.'
        )
    aligned = to_features(frame, date_col="fecha", time_col="fecha")
    if not aligned[names].isin([0, 1]).all().all():
        raise ValueError("MAC calendar features must be binary")
    if aligned.empty:
        raise ValueError("MAC calendar is empty")
    frame = aligned.copy()
    frame["fecha"] = frame.fecha.dt.strftime("%Y-%m-%d")
    canonical = raw_csv or frame.to_csv(index=False).encode()
    if not source_files:
        stream = BytesIO()
        frame.to_parquet(stream, index=False, compression="zstd")
        source_files = {"source.parquet": stream.getvalue()}
    provenance = dict(
        format_version=1,
        generator=generator,
        generator_version=generator_version,
        profile=metadata.get("spec", {}).get("profile", "external_table"),
        view=metadata.get("view"),
        start=frame.fecha.iloc[0],
        end=frame.fecha.iloc[-1],
        rows=len(frame),
        feature_count=len(names),
        features=names,
        calendar_sha256=hashlib.sha256(canonical).hexdigest(),
        source_metadata=metadata,
    )
    # Reject unserializable provider provenance before writing any project files.
    json.dumps(provenance, sort_keys=True, allow_nan=False)
    return CalendarSnapshot(canonical, frame, provenance, source_files)


def verify_snapshot(project, expected_manifest_sha):
    project = Path(project)
    folder = project / "calendar"
    manifest = folder / "provenance.json"
    if digest(manifest) != expected_manifest_sha:
        raise ValueError("Calendar provenance changed")
    data = json.loads(manifest.read_text())
    for name, expected in data["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError("Frozen calendar snapshot changed: " + name)
    return data
