"""Real Athena/S3 adapter. AWS credentials come only from boto3's role chain.

Engine1 snapshot, latest available history, and Engine3 S/M/entity74 FX relation
are migrated from r10; snapshot orphans are audited generically, never relabeled.
"""

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
from botocore.exceptions import BotoCoreError, ClientError

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def split_s3(uri):
    p = urlparse(uri)
    if (
        p.scheme != "s3"
        or bool(p.query)
        or bool(p.fragment)
        or not p.netloc
        or not p.path.strip("/")
        or any(c in uri for c in "'\"\\\n\r")
    ):
        raise ValueError("A non-root safe S3 prefix is required")
    return p.netloc, p.path.strip("/")


@dataclass(frozen=True, kw_only=True)
class AwsSettings:
    region: str = "us-east-1"
    athena_catalog: str = "AwsDataCatalog"
    athena_database: str
    athena_workgroup: str = "sandbox"
    athena_table: str
    variation_column: str
    id_column: str
    date_column: str
    load_column: str
    amount_column: str
    currency_column: str
    taxonomy_column: str
    sign_column: str
    fx_database: str
    fx_table: str
    storage_base_uri: str
    athena_results_uri: str
    run_prefix: str = "observatorio-librerias/020"
    account_limit: int = 0
    cutoff_date: str | None = None
    metadata_load_date: str | None = None
    amount_basis: str = "native"
    sign_normalized: bool = True
    query_timeout_seconds: int = 1800
    poll_seconds: float = 2.0
    classification_audit_enabled: bool = True
    plan_enabled: bool = True
    plan_database: str
    plan_table: str
    plan_company: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                name.endswith(("_column", "_table", "_database"))
                or name == "athena_catalog"
            ) and (not isinstance(value, str) or not IDENTIFIER.fullmatch(value)):
                raise ValueError("Unsafe SQL identifier: " + name)
        if not isinstance(self.account_limit, int) or isinstance(
            self.account_limit, bool
        ):
            raise TypeError("account_limit must be an integer")
        if any(
            not isinstance(v, bool)
            for v in (
                self.sign_normalized,
                self.plan_enabled,
                self.classification_audit_enabled,
            )
        ):
            raise ValueError("AWS boolean flags must be JSON booleans")
        split_s3(self.storage_base_uri)
        split_s3(self.athena_results_uri)
        if not re.fullmatch(r"[A-Za-z0-9/_-]+", self.run_prefix):
            raise ValueError("Unsafe run prefix")
        if (
            self.account_limit < 0
            or self.query_timeout_seconds < 1
            or self.poll_seconds <= 0
        ):
            raise ValueError("Invalid AWS limits")
        if self.amount_basis not in ("native", "mxp"):
            raise ValueError("amount_basis must be native or mxp")
        for value in (self.cutoff_date, self.metadata_load_date):
            if value:
                date.fromisoformat(value)
        if not self.plan_company.isdigit():
            raise ValueError("Unsafe PLANCO company")


class AwsAdapter:
    def __init__(self, settings, root, session=None):
        self.settings = settings
        self.root = Path(root)
        self.evidence = self.root / "source"
        self.evidence.mkdir(parents=True, exist_ok=True)
        identity = self.evidence / "run_identity.json"
        if not identity.exists():
            identity.write_text(json.dumps({"run_id": "obs-" + uuid.uuid4().hex}))
        self.run_id = json.loads(identity.read_text())["run_id"]
        if not re.fullmatch(r"obs-[a-f0-9]{32}", self.run_id):
            raise ValueError("Invalid persisted run identity")
        if session is None:
            import boto3

            session = boto3.Session(region_name=settings.region)
        self.session = session
        from botocore.config import Config

        options = Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 4, "mode": "standard"},
        )
        self.athena = session.client("athena", config=options)
        self.s3 = session.client("s3", config=options)

    def _save(self, name, value):
        p = self.evidence / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))

    @property
    def table(self):
        s = self.settings
        return f'"{s.athena_catalog}"."{s.athena_database}"."{s.athena_table}"'

    @property
    def fx_table(self):
        s = self.settings
        return f'"{s.athena_catalog}"."{s.fx_database}"."{s.fx_table}"'

    def query(self, sql, name):
        stamp = hashlib.sha256(sql.encode()).hexdigest()
        token = hashlib.sha256(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "name": name,
                    "sql": stamp,
                    "settings": asdict(self.settings),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        p = self.evidence / "queries" / f"{name}.json"
        p.parent.mkdir(exist_ok=True)
        if p.exists():
            saved = json.loads(p.read_text())
            if saved["sha256"] != stamp:
                raise ValueError("Query changed in an existing run")
            qid = saved["query_execution_id"]
        else:
            qid = self.athena.start_query_execution(
                QueryString=sql,
                ClientRequestToken=token,
                QueryExecutionContext={
                    "Catalog": self.settings.athena_catalog,
                    "Database": self.settings.athena_database,
                },
                ResultConfiguration={
                    "OutputLocation": self.settings.athena_results_uri.rstrip("/") + "/"
                },
                WorkGroup=self.settings.athena_workgroup,
            )["QueryExecutionId"]
            p.write_text(json.dumps({"query_execution_id": qid, "sha256": stamp}))
            p.with_suffix(".sql").write_text(sql)
        started = time.monotonic()
        last = None
        last_log = 0
        while True:
            execution = self.athena.get_query_execution(QueryExecutionId=qid)[
                "QueryExecution"
            ]
            state = execution["Status"]["State"]
            elapsed = time.monotonic() - started
            if state != last or elapsed - last_log >= 30:
                print(f"[Athena] {name} {qid} {state} {elapsed:.0f}s", flush=True)
                last = state
                last_log = elapsed
            if state == "SUCCEEDED":
                self._save(
                    f"queries/{name}_result.json",
                    {
                        "query_execution_id": qid,
                        "state": state,
                        "statistics": execution.get("Statistics", {}),
                        "effective_result_configuration": execution.get(
                            "ResultConfiguration", {}
                        ),
                    },
                )
                return qid
            if state in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Athena {name}: {execution['Status']}")
            if elapsed > self.settings.query_timeout_seconds:
                self.athena.stop_query_execution(QueryExecutionId=qid)
                raise TimeoutError(
                    "Owned Athena query timed out and was cancelled: " + qid
                )
            time.sleep(self.settings.poll_seconds)

    def records(self, sql, name):
        qid = self.query(sql, name)
        rows = []
        columns = None
        token = None
        while True:
            args = {"QueryExecutionId": qid, "MaxResults": 1000}
            if token:
                args["NextToken"] = token
            page = self.athena.get_query_results(**args)
            result = page["ResultSet"]
            columns = columns or [
                c["Name"] for c in result["ResultSetMetadata"]["ColumnInfo"]
            ]
            data = result.get("Rows", [])
            data = data if token else data[1:]
            rows.extend(
                dict(
                    zip(
                        columns, [v.get("VarCharValue") for v in r["Data"]], strict=True
                    )
                )
                for r in data
            )
            token = page.get("NextToken")
            if not token:
                return rows

    def preflight(self):
        identity = self.session.client("sts").get_caller_identity()
        bucket, prefix = split_s3(self.settings.storage_base_uri)
        key = f"{prefix}/{self.settings.run_prefix}/{self.run_id}/preflight/write_read.json"
        body = json.dumps(
            {"run_id": self.run_id, "purpose": "owned S3 write/read preflight"}
        ).encode()
        self.s3.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json"
        )
        response = self.s3.get_object(Bucket=bucket, Key=key)
        if response["Body"].read() != body:
            raise ValueError("S3 write/read preflight mismatch")
        receipt = {
            "status": "PASS",
            "account": identity["Account"],
            "arn": identity["Arn"],
            "s3_write_read": f"s3://{bucket}/{key}",
            "credentials": "boto3 role/provider chain; no keys stored",
        }
        self._save("aws_identity.json", receipt)
        return receipt

    def schema(self):
        required = {
            self.settings.id_column,
            self.settings.date_column,
            self.settings.load_column,
            self.settings.amount_column,
            self.settings.variation_column,
            self.settings.currency_column,
            self.settings.taxonomy_column,
            self.settings.sign_column,
        }
        fx_required = {
            "currency_value_date",
            "currency_id",
            "currency_quote_type",
            "currency_instrument_type",
            "entity_id",
            "fix_exchange_amount",
        }
        receipt = {}
        for label, table, expected in [
            ("history", self.table, required),
            ("fx", self.fx_table, fx_required),
        ]:
            qid = self.query(f"SELECT * FROM {table} LIMIT 0", f"schema_{label}")
            response = self.athena.get_query_results(QueryExecutionId=qid, MaxResults=1)
            columns = {
                c["Name"]: c["Type"]
                for c in response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
            }
            missing = expected - set(columns)
            if missing:
                raise ValueError(f"{label} schema missing columns: {sorted(missing)}")
            receipt[label] = {"query_execution_id": qid, "columns": columns}
        self._save("schema.json", receipt)
        return receipt

    def resolve(self, calendar_table, *, horizon=20):
        s = self.settings
        pin = (
            s.metadata_load_date
            or self.records(
                f"SELECT CAST(MAX({s.load_column}) AS VARCHAR) value FROM {self.table}",
                "metadata_partition",
            )[0]["value"]
        )
        date.fromisoformat(pin)
        limits = self.records(
            f"SELECT CAST(MIN(CAST({s.date_column} AS DATE)) AS VARCHAR) first_date,CAST(MAX(CAST({s.date_column} AS DATE)) AS VARCHAR) last_date FROM {self.table} WHERE {s.load_column}<=DATE '{pin}'",
            "bounds",
        )[0]
        first = limits["first_date"]
        cutoff = s.cutoff_date or limits["last_date"]
        if not first or not cutoff:
            raise ValueError("Empty bounded Athena history")
        date.fromisoformat(first)
        date.fromisoformat(cutoff)
        if first < calendar_table.dates[0] or cutoff not in calendar_table.dates:
            raise ValueError(
                "Source bounds are outside the authoritative calendar; update the calendar explicitly"
            )
        calendar_table.after(cutoff, horizon)
        self.pin = pin
        self.bounds = {"first_date": first, "cutoff": cutoff, "snapshot_load_date": pin}
        self._save("bounds.json", self.bounds)
        return self.bounds

    def operational_dates(self, calendar_table):
        """Full bounded source date axis, independent of candidate account limit."""
        s = self.settings
        rows = self.records(
            f"SELECT DISTINCT CAST(CAST({s.date_column} AS DATE) AS VARCHAR) operation_date "
            f"FROM {self.table} WHERE {s.load_column}<=DATE '{self.pin}' "
            f"AND CAST({s.date_column} AS DATE) BETWEEN DATE '{self.bounds['first_date']}' "
            f"AND DATE '{self.bounds['cutoff']}' ORDER BY operation_date",
            "operational_dates_v1",
        )
        observed = {r["operation_date"] for r in rows}
        axis = sorted(observed & set(calendar_table.dates))
        if not axis or axis[-1] != self.bounds["cutoff"]:
            raise ValueError("Operational calendar does not reach source cutoff")
        return axis

    def ctes(self):
        s = self.settings
        b = self.bounds
        cap = f"LIMIT {s.account_limit}" if s.account_limit else ""
        return f"""WITH bounded AS (
 SELECT TRIM(CAST({s.id_column} AS VARCHAR)) cross_key_id,CAST({s.date_column} AS DATE) cutoff_date,
 CAST({s.amount_column} AS DOUBLE) source_amount,{s.amount_column} source_amount_exact,CAST({s.variation_column} AS DOUBLE) source_variation,
 CAST({s.load_column} AS DATE) source_load_date,UPPER(TRIM(CAST({s.currency_column} AS VARCHAR))) currency_id,
 {s.taxonomy_column} source_taxonomy,{s.sign_column} source_sign,
 ROW_NUMBER() OVER (PARTITION BY TRIM(CAST({s.id_column} AS VARCHAR)),CAST({s.date_column} AS DATE) ORDER BY {s.load_column} DESC) rn
 FROM {self.table} WHERE CAST({s.date_column} AS DATE) BETWEEN DATE '{b["first_date"]}' AND DATE '{b["cutoff"]}' AND {s.load_column}<=DATE '{self.pin}'),
 metadata_rows AS (SELECT DISTINCT TRIM(CAST({s.id_column} AS VARCHAR)) cross_key_id,
 UPPER(TRIM(CAST({s.currency_column} AS VARCHAR))) currency_id,{s.taxonomy_column} taxonomy,{s.sign_column} account_sign
 FROM {self.table} WHERE {s.load_column}=DATE '{self.pin}'),
 metadata AS (SELECT cross_key_id,COUNT(*) metadata_variants,MAX(currency_id) metadata_currency,MAX(taxonomy) taxonomy,MAX(account_sign) account_sign FROM metadata_rows GROUP BY cross_key_id),
 ranked_ids AS (SELECT cross_key_id,taxonomy,ROW_NUMBER() OVER (PARTITION BY taxonomy ORDER BY cross_key_id) sample_rank FROM metadata WHERE taxonomy IS NOT NULL AND cross_key_id IN (SELECT cross_key_id FROM bounded)),
 ids AS (SELECT cross_key_id FROM ranked_ids ORDER BY sample_rank,taxonomy,cross_key_id {cap}),
 history AS (SELECT b.*,m.metadata_variants,m.metadata_currency,m.taxonomy,m.account_sign FROM bounded b JOIN ids i ON b.cross_key_id=i.cross_key_id JOIN metadata m ON b.cross_key_id=m.cross_key_id WHERE b.rn=1),
 fx AS (SELECT CAST(currency_value_date AS DATE) cutoff_date,UPPER(TRIM(currency_id)) currency_id,COUNT(*) fx_matches,MAX(fix_exchange_amount) fx_value
 FROM {self.fx_table} WHERE currency_quote_type='S' AND currency_instrument_type='M' AND entity_id=74
 AND CAST(currency_value_date AS DATE) BETWEEN DATE '{b["first_date"]}' AND DATE '{b["cutoff"]}' GROUP BY 1,2),
 joined AS (SELECT h.*,f.fx_matches,CASE WHEN h.currency_id='MXP' THEN 1.0 WHEN f.fx_matches=1 THEN CAST(f.fx_value AS DOUBLE) ELSE NULL END fx
 FROM history h LEFT JOIN fx f ON h.cutoff_date=f.cutoff_date AND h.currency_id=f.currency_id)
"""

    def validate(self):
        c = self.ctes()
        orphan = self.records(
            c
            + """SELECT b.cross_key_id account_id,MAX(b.currency_id) currency,COUNT(*) observations,CAST(MIN(b.cutoff_date) AS VARCHAR) first_date,CAST(MAX(b.cutoff_date) AS VARCHAR) last_date,
 CASE WHEN MAX(m.cross_key_id) IS NULL THEN 'ABSENT_FROM_CURRENT_ENGINE1_SNAPSHOT' ELSE 'UNCLASSIFIED_IN_CURRENT_ENGINE1_SNAPSHOT' END reason
 FROM bounded b LEFT JOIN metadata m ON b.cross_key_id=m.cross_key_id WHERE b.rn=1 AND (m.cross_key_id IS NULL OR m.taxonomy IS NULL) GROUP BY b.cross_key_id""",
            "snapshot_exclusions",
        )
        self._save("snapshot_exclusions.json", orphan)
        counts = self.records(
            c
            + """SELECT COUNT(*) total_rows,
 COALESCE(SUM(CASE WHEN metadata_variants<>1 THEN 1 ELSE 0 END),0) ambiguous_metadata,
 COALESCE(SUM(CASE WHEN metadata_currency IS NULL OR metadata_currency<>currency_id OR currency_id IS NULL THEN 1 ELSE 0 END),0) currency_mismatch,
 COALESCE(SUM(CASE WHEN taxonomy NOT IN ('cuentas_entrenamiento','cuentas_cero','cuentas_atipicos') OR account_sign IS NULL OR NOT (account_sign IN ('positiva','negativa','ambos_signos') OR (taxonomy='cuentas_cero' AND account_sign='cuenta_cero')) THEN 1 ELSE 0 END),0) invalid_classification,
 COALESCE(SUM(CASE WHEN fx IS NULL OR fx<=0 OR NOT is_finite(fx) THEN 1 ELSE 0 END),0) invalid_fx,
 COALESCE(SUM(CASE WHEN currency_id<>'MXP' AND fx_matches>1 THEN 1 ELSE 0 END),0) duplicate_fx,
 COALESCE(SUM(CASE WHEN source_amount IS NULL OR NOT is_finite(source_amount) THEN 1 ELSE 0 END),0) invalid_amount,
 COALESCE(SUM(CASE WHEN source_variation IS NULL OR NOT is_finite(source_variation) THEN 1 ELSE 0 END),0) invalid_variation,
 COALESCE(SUM(CASE WHEN source_taxonomy<>taxonomy OR source_sign<>account_sign THEN 1 ELSE 0 END),0) historical_label_changes FROM joined""",
            "contracts_classification_v2",
        )[0]
        classification_distribution = self.records(
            c + """SELECT taxonomy,account_sign,COUNT(*) observations,
 COUNT(DISTINCT cross_key_id) accounts FROM joined GROUP BY taxonomy,account_sign
 ORDER BY taxonomy,account_sign""",
            "classification_distribution_v2",
        )
        self._save("classification_distribution.json", classification_distribution)
        counts = {k: int(v) for k, v in counts.items()}
        blocked = not counts["total_rows"] or any(
            v
            for k, v in counts.items()
            if k not in ("total_rows", "historical_label_changes")
        )
        conflicts = self.records(
            c
            + """SELECT b.cross_key_id account_id,CAST(b.cutoff_date AS VARCHAR) date,CAST(b.source_load_date AS VARCHAR) load_date FROM bounded b JOIN ids i ON b.cross_key_id=i.cross_key_id GROUP BY 1,2,3 HAVING COUNT(DISTINCT b.source_amount_exact)>1""",
            "same_load_conflicts",
        )
        receipt = {
            "status": "FAIL" if blocked or conflicts else "PASS",
            "counts": counts,
            "classification_contract": "mac-cuenta-cero-v2",
            "classification_distribution": classification_distribution,
            "same_load_conflicts": conflicts,
            "exclusions": orphan,
            "policy": "current Engine1 snapshot, available history; no historical taxonomy fallback",
            "fx_filters": {
                "currency_quote_type": "S",
                "currency_instrument_type": "M",
                "entity_id": 74,
                "MXP": 1,
            },
            "scope": "current snapshot taxonomy, not point-in-time historical classification",
        }
        self._save("contracts.json", receipt)
        if receipt["status"] != "PASS":
            raise ValueError(
                f"Athena data contracts failed: {counts}; conflicts={len(conflicts)}; "
                "inspect source/classification_distribution.json"
            )
        return receipt

    def ingest(self, calendar_table, *, horizon=20):
        if self.settings.amount_basis!="native" or not self.settings.sign_normalized:
            raise ValueError("Este perfil conserva importes y variación Engine1: requiere amount_basis=native y sign_normalized=true. Configura un adaptador explícito para otra convención.")
        self.schema()
        self.resolve(calendar_table, horizon=horizon)
        receipt = self.validate()
        self.audit_classifications()
        s = self.settings
        amount = (
            "source_amount"
            if s.amount_basis == "native"
            else "source_amount / NULLIF(fx,0)"
        )
        if not s.sign_normalized:
            amount = (
                f"CASE WHEN account_sign='negativa' THEN -1 ELSE 1 END * ({amount})"
            )
        sql = (
            self.ctes()
            + f"""SELECT cross_key_id account_id,CAST(cutoff_date AS VARCHAR) date,{amount} balance,currency_id currency,taxonomy,account_sign sign,CAST(DATE '{self.pin}' AS VARCHAR) load_date,fx,CAST(source_load_date AS VARCHAR) source_load_date,source_taxonomy,source_sign,source_variation variation FROM joined"""
        )
        bucket, prefix = split_s3(s.storage_base_uri)
        owned = f"{prefix}/{s.run_prefix}/{self.run_id}/source/unload/"
        uri = f"s3://{bucket}/{owned}"
        self.query(
            f"UNLOAD ({sql}) TO '{uri}' WITH (format='PARQUET',compression='SNAPPY')",
            "unload",
        )
        paths = []
        parts = self.evidence / "parts"
        parts.mkdir(exist_ok=True)
        for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=owned
        ):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith(("/", ".csv", ".metadata", ".manifest", ".json")):
                    continue
                local = parts / (hashlib.sha256(key.encode()).hexdigest() + ".parquet")
                self.s3.download_file(bucket, key, str(local))
                with local.open("rb") as f:
                    magic = f.read(4)
                if magic != b"PAR1":
                    local.unlink()
                    continue
                paths.append(local)
        if not paths:
            raise ValueError("Athena UNLOAD produced no Parquet data")
        frame = pl.scan_parquet(paths).collect().sort(["account_id", "date"])
        if set(frame["date"].unique().to_list()) - set(calendar_table.dates):
            raise ValueError("History contains dates missing from ADA calendar")
        if frame.height != receipt["counts"]["total_rows"]:
            raise ValueError(
                "UNLOAD row count differs from validated snapshot; source may have changed"
            )
        self._save(
            "manifest.json",
            {
                "rows": frame.height,
                "accounts": frame["account_id"].n_unique(),
                "snapshot": self.bounds,
                "calendar_sha256": calendar_table.source_sha256,
                "selection": "candidate pool interleaved by current taxonomy; final eligibility and sampling applied locally",
                "account_limit": s.account_limit,
                "amount_basis_input": s.amount_basis,
                "canonical_balance_basis": "native",
                "metadata_policy": receipt["policy"],
                "unload_uri": uri,
            },
        )
        return frame

    def audit_classifications(self):
        if not self.settings.classification_audit_enabled:
            self._save(
                "reclassification_events.json", {"status": "DISABLED_EXPLICITLY"}
            )
            return
        from .source_audit import classification_events

        s = self.settings
        b = self.bounds
        sql = (
            self.ctes()
            + f"""SELECT TRIM(CAST({s.id_column} AS VARCHAR)) cross_key_id,
 CAST({s.load_column} AS VARCHAR) load_date,UPPER(TRIM(CAST({s.currency_column} AS VARCHAR))) currency_id,
 {s.taxonomy_column} taxonomy,{s.sign_column} account_sign,COUNT(DISTINCT {s.date_column}) observations
 FROM {self.table} WHERE {s.load_column}<=DATE '{self.pin}'
 AND CAST({s.date_column} AS DATE) BETWEEN DATE '{b["first_date"]}' AND DATE '{b["cutoff"]}'
 AND TRIM(CAST({s.id_column} AS VARCHAR)) IN (SELECT cross_key_id FROM ids)
 GROUP BY 1,2,3,4,5 ORDER BY 1,2,3,4,5"""
        )
        snapshots = self.records(sql, "classification_snapshots")
        receipt = classification_events(snapshots)
        receipt["metadata_partition"] = self.pin
        self._save("reclassification_events.json", receipt)

    def account_plan(self):
        s = self.settings
        if not s.plan_enabled:
            result = {"status": "DISABLED_EXPLICITLY", "rows": []}
        else:
            table = f'"{s.athena_catalog}"."{s.plan_database}"."{s.plan_table}"'
            sql = f"""WITH latest AS (SELECT MAX(CAST(load_date AS VARCHAR)) load_date FROM {table}
 WHERE CAST(company_id AS VARCHAR)='{s.plan_company}'),ranked AS (
 SELECT TRIM(CAST(p.gl_account_id AS VARCHAR)) gl_account_id,
 TRIM(COALESCE(CAST(p.epigraph_desc AS VARCHAR),'')) epigraph_desc,
 TRIM(COALESCE(CAST(p.average_epigraph_desc AS VARCHAR),'')) average_epigraph_desc,
 CAST(p.load_date AS VARCHAR) account_plan_load_date,
 ROW_NUMBER() OVER (PARTITION BY p.company_id,TRIM(CAST(p.gl_account_id AS VARCHAR))
 ORDER BY COALESCE(CAST(p.epigraph_desc AS VARCHAR),'') DESC,COALESCE(CAST(p.average_epigraph_desc AS VARCHAR),'') DESC) rn
 FROM {table} p CROSS JOIN latest l WHERE CAST(p.company_id AS VARCHAR)='{s.plan_company}'
 AND CAST(p.load_date AS VARCHAR)=l.load_date AND p.gl_account_id IS NOT NULL AND TRIM(CAST(p.gl_account_id AS VARCHAR))<>'')
 SELECT gl_account_id,epigraph_desc,average_epigraph_desc,account_plan_load_date FROM ranked WHERE rn=1 ORDER BY gl_account_id"""
            try:
                result = {
                    "status": "AVAILABLE",
                    "rows": self.records(sql, "account_plan"),
                }
            except (
                BotoCoreError,
                ClientError,
                RuntimeError,
                TimeoutError,
                ValueError,
            ) as exc:
                # Descriptive labels are optional; balances/classification/FX stay strict.
                result = {
                    "status": "UNAVAILABLE",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                    "rows": [],
                }
                print(
                    "PLANCO unavailable; keeping the canonical account IDs. Inspect source/account_plan.json",
                    flush=True,
                )
        self._save("account_plan.json", result)
        return result

    def publish(self, stage):
        s = self.settings
        bucket, prefix = split_s3(s.storage_base_uri)
        base = f"{prefix}/{s.run_prefix}/{self.run_id}/artifacts/"
        state_path = self.root / "s3_upload_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        changed = 0
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path == state_path or path.suffix == ".tmp":
                continue
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Refusing to publish a file outside the owned run")
            rel = path.relative_to(self.root).as_posix()
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if state.get(rel) == sha:
                continue
            self.s3.upload_file(str(path), bucket, base + rel)
            state[rel] = sha
            changed += 1
        state_path.write_text(json.dumps(state, indent=2))
        self.s3.upload_file(str(state_path), bucket, base + state_path.name)
        receipt = {
            "stage": stage,
            "status": "COMPLETED",
            "uploaded_files": changed,
            "uri": f"s3://{bucket}/{base}",
            "run_id": self.run_id,
        }
        self.s3.put_object(
            Bucket=bucket,
            Key=base + f"commits/{stage}.json",
            Body=json.dumps(receipt).encode(),
            ContentType="application/json",
        )
        return receipt
