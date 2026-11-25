import csv
from datetime import date
from decimal import Decimal
from io import StringIO
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_arena._mac_runtime.observatorio_athena.aws import Athena
from model_arena._mac_runtime.observatorio_athena.common import sha
from model_arena._mac_runtime.observatorio_athena.layout_export import write_published_csv
from model_arena._mac_runtime.observatorio_athena.repository import Repository
from model_arena._mac_runtime.observatorio_athena.schemas import SCHEMAS
from model_arena.profiles.mac import MacProject


def test_download_reads_more_than_100000_rows_with_one_header(monkeypatch):
    names = SCHEMAS["engine2"].names
    columns = [{"Name": name} for name in names]
    header = {"Data": [{"VarCharValue": name} for name in names]}
    row = {"Data": [{"VarCharValue": value} for value in
                    ["2026-01-01", "2026-01-02", "A_var", "winner", "1234567890123456.12345678"]]}

    class Client:
        def get_query_results(self, **params):
            page = int(params.get("NextToken", "0"))
            rows = [row] * 1000 if page else [header] + [row] * 999
            response = {"ResultSet": {"ResultSetMetadata": {"ColumnInfo": columns}, "Rows": rows}}
            if page < 100:
                response["NextToken"] = str(page + 1)
            return response

    config = SimpleNamespace(shards=2, table=lambda name: name)
    repo = Repository(config, "r", "p")
    athena = Athena(config, client=Client())
    queries = []

    def execute(sql):
        queries.append(sql)
        return {"query_id": "query"}

    monkeypatch.setattr(athena, "execute", execute)
    stream = StringIO()
    assert write_published_csv(athena, repo, "engine2", "FUTURO", stream) == 100999
    assert len(queries) == 1
    assert "segment='FUTURO'" in queries[0]
    assert "publication_id='p'" in queries[0]
    stream.seek(0)
    rows = csv.reader(stream)
    assert next(rows) == names
    assert all(r[2:6] == ["A_var", "winner", "1234567890123456.12345678", ""] for r in rows)


@pytest.mark.parametrize("engine", ["engine2", "engine3"])
def test_export_layout_preserves_schema_and_isolates_segment(tmp_path, engine):
    release = tmp_path / "release"
    release.mkdir()
    (tmp_path / "project.json").write_text("{}")
    schema = SCHEMAS[engine]
    record = {
        field.name: Decimal("12.00000001") if pa.types.is_decimal(field.type)
        else date(2026, 1, 2) if pa.types.is_date(field.type) else "A"
        for field in schema
    }
    table = pa.Table.from_pylist([record], schema=schema)
    entries = []
    for fold in ("F1", "F2"):
        part = release / f"{fold}.parquet"
        pq.write_table(table, part)
        entries.append(dict(dataset=engine, segment=fold, path=part.name, sha256=sha(part), rows=1))
    (release / "publication.json").write_text(json.dumps(
        dict(status="PREPARED", datasets={engine: {}}, files=entries)
    ))
    project = MacProject(tmp_path)
    parquet = project.export_layout(tmp_path / "export.parquet", engine=engine, fold="F1")
    assert pq.read_table(parquet).equals(table)
    path = project.export_layout(tmp_path / "export.csv", engine=engine, fold="F1")
    with path.open(newline="") as stream:
        assert list(csv.reader(stream)) == [schema.names, [str(record[name]) for name in schema.names]]
    empty = project.export_layout(tmp_path / "empty.parquet", engine=engine, fold="FUTURO")
    assert pq.read_table(empty).equals(table.slice(0, 0))
    (release / "F1.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="modificado"):
        project.export_layout(parquet, engine=engine, fold="F1")
    assert pq.read_table(parquet).equals(table)
