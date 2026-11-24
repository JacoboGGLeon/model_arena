"""Optional AWS adapters; credentials come exclusively from the boto3 session."""
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import pandas as pd

from .contracts import Publication
from .storage import digest, signature, verify_manifest, write_json


def s3_parts(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("An s3://bucket/prefix URI is required")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def publish_s3(source, destination, *, client=None):
    import boto3
    client = client or boto3.client("s3")
    source = Path(source).resolve()
    bucket, prefix = s3_parts(destination)
    files = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")}
    publication_id = signature(files)[:24]
    base = "/".join(x for x in [prefix, publication_id] if x)
    for name, expected in files.items():
        client.upload_file(str(source / name), bucket, base+"/"+name,
                           ExtraArgs={"Metadata": {"sha256": expected}})
        if client.head_object(Bucket=bucket, Key=base+"/"+name)["Metadata"].get("sha256") != expected:
            raise ValueError("S3 upload integrity metadata mismatch")
    manifest = dict(schema_version="1", publication_id=publication_id, files=files)
    payload = json.dumps(manifest, sort_keys=True).encode()
    client.put_object(Bucket=bucket, Key=base+"/manifest.json", Body=payload,
                      ContentType="application/json")
    client.put_object(Bucket=bucket, Key=(prefix+"/" if prefix else "")+"latest.json",
                      Body=json.dumps({"publication_id": publication_id}).encode(),
                      ContentType="application/json")
    import hashlib
    return Publication(publication_id=publication_id, path=f"s3://{bucket}/{base}",
                       manifest_sha256=hashlib.sha256(payload).hexdigest())


def materialize_publication(uri, destination, *, client=None):
    import boto3
    client = client or boto3.client("s3")
    bucket, prefix = s3_parts(uri)
    root = Path(destination).resolve()
    manifest = json.loads(client.get_object(Bucket=bucket, Key=prefix+"/manifest.json")["Body"].read())
    for name in manifest["files"]:
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Invalid publication path")
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, prefix+"/"+name, str(target))
    write_json(root / "manifest.json", manifest)
    verify_manifest(root)
    return root


def read_parquet(uri, *, client=None):
    if not str(uri).startswith("s3://"):
        return pd.read_parquet(uri)
    import boto3
    client = client or boto3.client("s3")
    bucket, key = s3_parts(uri)
    return pd.read_parquet(BytesIO(client.get_object(Bucket=bucket, Key=key)["Body"].read()))


class AthenaSource:
    """Execute an explicit read query with a finite deadline and persist its query id."""
    def __init__(self, *, database, workgroup, output_location, catalog="AwsDataCatalog",
                 region=None, timeout=1800, client=None):
        import boto3
        self.client = client or boto3.client("athena", region_name=region)
        self.database, self.workgroup, self.output = database, workgroup, output_location
        self.catalog, self.timeout, self.last_query_id = catalog, timeout, None

    def read(self, sql):
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("Provide a SELECT query")
        response = self.client.start_query_execution(
            QueryString=sql, QueryExecutionContext={"Database": self.database, "Catalog": self.catalog},
            WorkGroup=self.workgroup, ResultConfiguration={"OutputLocation": self.output})
        query_id = self.last_query_id = response["QueryExecutionId"]
        deadline = time.monotonic()+self.timeout
        while True:
            state = self.client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]["Status"]
            if state["State"] == "SUCCEEDED":
                break
            if state["State"] in ("FAILED", "CANCELLED"):
                raise RuntimeError("Athena query failed; inspect query id " + query_id)
            if time.monotonic() > deadline:
                self.client.stop_query_execution(QueryExecutionId=query_id)
                raise TimeoutError("Athena query deadline exceeded: " + query_id)
            time.sleep(1)
        rows, columns = [], None
        paginator = self.client.get_paginator("get_query_results")
        for page in paginator.paginate(QueryExecutionId=query_id):
            result = page["ResultSet"]
            if columns is None:
                columns = [c["Name"] for c in result["ResultSetMetadata"]["ColumnInfo"]]
                data = result["Rows"][1:]
            else:
                data = result["Rows"]
            rows.extend([[c.get("VarCharValue") for c in row["Data"]] for row in data])
        return pd.DataFrame(rows, columns=columns)

