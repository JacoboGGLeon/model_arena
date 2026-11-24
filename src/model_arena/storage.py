"""Immutable publication snapshots and guarded, portable table storage."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import pandas as pd

from .contracts import Publication


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        frame.to_json(temp, orient="table", date_format="iso", index=False, double_precision=15)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_table(path):
    path = Path(path)
    return pd.read_json(path, orient="table") if path.exists() else pd.DataFrame()


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    for name, expected in manifest["files"].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file() or digest(file) != expected:
            raise ValueError("Artifact integrity failure: " + name)
    return manifest


def publish_directory(source, destination):
    """Commit a complete local snapshot before updating its discovery pointer."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError("Publication destination must be outside the release")
    files = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")}
    identity = signature(files)
    publication_id = identity[:24]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / publication_id
    if not target.exists():
        staging = Path(tempfile.mkdtemp(dir=destination, prefix=".publishing-"))
        try:
            for name in files:
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, output)
            write_json(staging / "manifest.json", {"schema_version": "1", "publication_id": publication_id,
                                                  "files": files})
            verify_manifest(staging)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    verify_manifest(target)
    write_json(destination / "latest.json", {"publication_id": publication_id})
    return Publication(publication_id=publication_id, path=str(target),
                       manifest_sha256=digest(target / "manifest.json"))

