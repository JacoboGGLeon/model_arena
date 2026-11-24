"""Explicit storage-only upgrade of an unpublished SDK 0.1 project.

Existing source data, checkpoints, predictions, trial database and training setup
are retained. No training, S3 upload or model selection is performed here.
"""

from pathlib import Path
from datetime import datetime, timezone
import shutil
import sqlite3
import os

from model_arena.profiles.mac import MacProject
from model_arena.storage import digest, read_json, write_json
from model_arena.loss_store import compact_csv
from model_arena.disk import check_space

ROOT = Path(__file__).resolve().parent


def _ensure_stopped(project):
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = process.read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if b"model_arena.profiles.mac_worker" in args and os.fsencode(str(project)) in args:
            raise RuntimeError("Detén el worker anterior antes de actualizar este proyecto")


def upgrade(project_path, *, remove_verified_csv=True):
    project = MacProject(project_path)
    _ensure_stopped(project.path)
    with project.exclusive():
        project.verify()
        receipt = project.path / "storage_upgrade.json"
        if receipt.exists() and read_json(receipt).get("status") == "COMPLETE":
            previous = read_json(receipt)
            if remove_verified_csv and not previous.get("csv_removed"):
                _remove_sources(previous["files"])
                previous["csv_removed"] = True
                write_json(receipt, previous)
            return project
        if (project.path / "release").exists():
            raise ValueError("Este upgrade sólo admite corridas sin release; conserva la entrega existente")
        baseline = read_json(ROOT / "baseline_01.json")
        runtime = project.path / "runtime"
        if read_json(runtime / "BUILD_MONITOR.json")["files"] != baseline["files"]:
            raise ValueError("El run no corresponde al runtime SDK 0.1 verificado")
        check_space(project.path, stage="actualizar almacenamiento")
        db = project.path / "run/curvas/hpo.sqlite3"
        if db.exists():
            with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=3) as connection:
                if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise ValueError("SQLite requiere recuperación antes del upgrade")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = project.path / "storage_backups" / stamp
        backup.mkdir(parents=True)
        shutil.copytree(runtime, backup / "runtime")
        metadata = [
            Path(name)
            for name in [
                "project.json",
                "run/configuration.json",
                "run/receipts/ingesta.json",
                "run/curvas/identidad.json",
            ]
            if (project.path / name).exists()
        ]
        for name in metadata:
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project.path / name, target)
        converted = []
        curves = project.path / "run/curvas"
        # All CSV originals remain until the new runtime and identities validate.
        for directory in (curves / "loss_entities", curves / "loss"):
            for csv in sorted(directory.glob("*.csv")):
                converted.append(compact_csv(csv, csv.with_suffix(".parquet")))
        try:
            source = ROOT.parent / "_mac_runtime"
            shutil.copytree(
                source, runtime, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            training = runtime / "entrenamiento"
            config_lock = project.path / "run/configuration.json"
            if config_lock.exists():
                value = read_json(config_lock)
                files = [
                    *training.glob("*.py"),
                    *training.rglob("libraries/**/*.py"),
                    training / "visor/workspace.py",
                ]
                value["code_sha256"] = {str(p.relative_to(training)): digest(p) for p in sorted(set(files))}
                write_json(config_lock, value)
                ingestion = project.path / "run/receipts/ingesta.json"
                if ingestion.exists():
                    value = read_json(ingestion)
                    value["configuration_sha256"] = digest(config_lock)
                    write_json(ingestion, value)
            identity = curves / "identidad.json"
            if identity.exists():
                value = read_json(identity)
                paths = [
                    training / "arena_estudio.py",
                    training / "curvas_modelo.py",
                    *list((training / "libraries").rglob("*.py")),
                ]
                value["codigo"] = {str(p.relative_to(training)): digest(p) for p in paths}
                write_json(identity, value)
            settings = read_json(project.path / "project.json")
            settings["code_sha"] = digest(runtime / "BUILD_MONITOR.json")
            write_json(project.path / "project.json", settings)
            upgraded = MacProject(project.path)
            upgraded.verify()
        except Exception:
            # Restore runtime and identities. Verified Parquets are harmless next to CSVs.
            shutil.rmtree(runtime)
            shutil.copytree(backup / "runtime", runtime)
            for name in metadata:
                shutil.copy2(backup / name, project.path / name)
            raise
        write_json(
            receipt,
            {
                "status": "COMPLETE",
                "build": "ADA-0132",
                "backup": str(backup),
                "files": converted,
                "csv_removed": False,
                "scope": "storage only; models, trials, inputs and transforms retained",
            },
        )
        if remove_verified_csv:
            _remove_sources(converted)
            value = read_json(receipt)
            value["csv_removed"] = True
            write_json(receipt, value)
        return upgraded


def _remove_sources(converted):
    for item in converted:
        source = Path(item["source"])
        if not source.exists():
            continue
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["source_bytes"], item["source_mtime_ns"]):
            raise ValueError("El CSV cambió después de verificarlo; se conserva: " + str(source))
        compressed = Path(item["destination"])
        if not compressed.is_file() or digest(compressed) != item["parquet_sha256"]:
            raise ValueError("Parquet ausente o modificado; se conserva el CSV: " + str(source))
        source.unlink()
