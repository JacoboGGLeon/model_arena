"""Export MAC rows with the published column order and decimal precision."""

import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq

from .common import sha
from .schemas import SCHEMAS


ENGINES = ("engine2", "engine3")
FOLDS = ("F1", "F2", "F3", "FUTURO")


def layout_schema(engine, fold):
    if engine not in ENGINES:
        raise ValueError(f"Salida MAC desconocida: {engine}")
    if fold not in FOLDS:
        raise ValueError(f"Tramo MAC desconocido: {fold}")
    return SCHEMAS[engine]


def write_published_csv(athena, repository, engine, fold, stream):
    """Write each Athena page directly as CSV, preserving its textual values."""
    names = layout_schema(engine, fold).names
    writer = csv.writer(stream)
    writer.writerow(names)
    count = 0
    for columns, rows in athena.raw_pages(repository.delivery(engine, fold), size=1000):
        if columns != names:
            raise ValueError("Las columnas devueltas por Athena difieren del layout MAC")
        writer.writerows(rows)
        count += len(rows)
    return count


def export_prepared_layout(release, destination, *, engine, fold):
    """Write an atomic CSV or typed Parquet from a prepared MAC release."""
    schema = layout_schema(engine, fold)
    release = Path(release).resolve()
    destination = Path(destination).expanduser().resolve()
    fmt = destination.suffix.lower()
    if fmt not in (".csv", ".parquet"):
        raise ValueError("El destino debe terminar en .csv o .parquet")
    manifest = json.loads((release / "publication.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PREPARED" or engine not in manifest.get("datasets", {}):
        raise ValueError(f"La entrega preparada no incluye {engine}")
    if destination.is_relative_to(release):
        raise ValueError("El destino debe quedar fuera de la entrega preparada")
    entries = [
        item for item in manifest["files"]
        if item["dataset"] == engine and item["segment"] == fold
    ]

    def batches():
        for item in entries:
            path = release / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError(f"Archivo de la entrega modificado: {item['path']}")
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(schema) or parquet.metadata.num_rows != item["rows"]:
                raise ValueError(f"Archivo incompatible con el layout MAC: {item['path']}")
            yield from parquet.iter_batches()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if fmt == ".parquet":
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                for batch in batches():
                    writer.write_batch(batch)
        else:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(schema.names)
                for batch in batches():
                    columns = [column.to_pylist() for column in batch.columns]
                    writer.writerows(zip(*columns))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
