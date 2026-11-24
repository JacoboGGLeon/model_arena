import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from model_arena.disk import check_space, DiskSpaceError
from model_arena.loss_store import write_parquet, read_batches, detail_files, compact_csv


def test_disk_guard_preserves_existing_files(tmp_path, monkeypatch):
    path = tmp_path / "fit.parquet"
    frame = pl.DataFrame({"loss": [1.25]})
    write_parquet(frame, path)
    before = path.read_bytes()
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type(usage)(usage.total, usage.total, 0))
    with pytest.raises(DiskSpaceError):
        write_parquet(pl.DataFrame({"loss": [9.0]}), path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_parquet_preserves_values_and_legacy_csv(tmp_path):
    frame = pl.DataFrame({"account_id": ["a", "b", "a"], "epoch": [1, 1, 2], "loss": [1e-12, 1e12, None]})
    write_parquet(frame, tmp_path / "X0_T0000_F1" / "epoch_0001.parquet")
    frame.write_csv(tmp_path / "X0_T0000_F1.csv")
    frame.write_csv(tmp_path / "X1_T0000_F1.csv")
    files = detail_files(tmp_path)
    assert len(files) == 2
    assert not any(p.name == "X0_T0000_F1.csv" for p in files)
    restored = pl.concat(read_batches(files[0]))
    assert restored.equals(frame)


def test_streaming_compaction_verified_before_commit(tmp_path):
    frame = pl.DataFrame(
        {"account_id": ["A", "B"] * 40000, "epoch": [1] * 80000, "loss": [1e-12, 1e12] * 40000}
    )
    source = tmp_path / "old.csv"
    frame.write_csv(source)
    result = compact_csv(source, tmp_path / "new.parquet")
    assert result["rows"] == frame.height
    assert source.exists()
    assert pl.read_parquet(tmp_path / "new.parquet").equals(frame)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_failure_keeps_last_epoch(tmp_path, monkeypatch):
    path = tmp_path / "epoch_0001.parquet"
    write_parquet(pl.DataFrame({"loss": [1.0]}), path)
    before = path.read_bytes()

    def broken(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", broken)
    with pytest.raises(OSError):
        write_parquet(pl.DataFrame({"loss": [2.0]}), path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_bad_reserve_rejected(tmp_path):
    (tmp_path / "operations.json").write_text(json.dumps({"min_free_gib": -1}))
    with pytest.raises(ValueError):
        check_space(tmp_path)
