"""Atomic compressed loss shards and bounded readers; Float64 is preserved."""

from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from .disk import check_space


def write_parquet(frame, path):
    path = Path(path)
    check_space(path.parent, extra_bytes=frame.estimated_size() * 2, stage=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.write_parquet(temp, compression="zstd", statistics=True)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def detail_files(root):
    """Read compressed shards; permit legacy CSV only when no new shard owns its fit."""
    root = Path(root)
    parquet = sorted(root.rglob("*.parquet"))
    owned = {p.parent.name for p in parquet if p.parent != root} | {p.stem for p in parquet}
    return [*parquet, *[p for p in sorted(root.glob("*.csv")) if p.stem not in owned]]


def read_batches(path, batch_size=65536):
    path = Path(path)
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield pl.from_arrow(batch)
    else:
        import csv

        with path.open(newline="") as stream:
            columns = next(csv.reader(stream))
        floats = {
            "loss",
            "train_loss",
            "validation_loss",
            "train_loss_saldo",
            "train_loss_variacion",
            "fixed_train_saldo",
            "fixed_train_variacion",
            "fixed_validation_saldo",
            "fixed_validation_variacion",
        }
        strings = {"account_id", "currency", "cohort", "taxonomy", "candidate", "fold", "target", "split"}
        import pyarrow as pa
        import pyarrow.csv as pacsv

        overrides = {name: pa.float64() for name in columns if name in floats}
        overrides.update({name: pa.string() for name in columns if name in strings})
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=1024 * 1024),
            convert_options=pacsv.ConvertOptions(column_types=overrides, strings_can_be_null=True),
        )
        for batch in reader:
            yield pl.from_arrow(batch)


def compact_csv(source, destination):
    """Stream a CSV into Parquet and verify row order and value fingerprints."""
    import hashlib
    import os

    source, destination = Path(source), Path(destination)
    check_space(destination.parent, stage="compactar " + source.name)
    temp = destination.with_suffix(".parquet.tmp")
    before = source.stat()
    digest, rows, writer = hashlib.sha256(), 0, None
    try:
        for frame in read_batches(source):
            digest.update(frame.hash_rows(seed=42).to_numpy().tobytes())
            rows += frame.height
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            check_space(destination.parent, extra_bytes=table.nbytes * 2, stage="compactación")
            writer.write_table(table)
        if writer is None:
            raise ValueError("CSV sin filas: " + str(source))
        writer.close()
        writer = None
        verified, count = hashlib.sha256(), 0
        for frame in _parquet_batches(temp):
            verified.update(frame.hash_rows(seed=42).to_numpy().tobytes())
            count += frame.height
        after = source.stat()
        if (
            count != rows
            or verified.digest() != digest.digest()
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("Verificación de compactación fallida: " + str(source))
        # Commit only after the compressed content has been read back and verified.
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        temp.replace(destination)
        return {
            "source": str(source),
            "destination": str(destination),
            "rows": rows,
            "source_bytes": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "parquet_bytes": destination.stat().st_size,
            "row_fingerprint": digest.hexdigest(),
            "parquet_sha256": _file_sha(destination),
        }
    finally:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)


def _parquet_batches(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        yield pl.from_arrow(batch)


def _file_sha(path):
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
