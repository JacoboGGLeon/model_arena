# ADR-0004 · Calendar providers and immutable run provenance

Status: implemented for release 0.1.2. Date: 2026-11-25 UTC.

## Context

Calendar Studio 0.1.0 is independently published and its project holiday fixture corrects 2026-11-02. Model Arena must consume generated calendars without coupling the generic forecasting API to those rules, and existing trained runs must preserve their original inputs.

## Decision

Add a table/provider adapter and explicit `calendar` argument alongside the backward-compatible `calendar_path`. Require exactly one. Accept CSV, Parquet, DataFrames and validated artifact-like providers. MAC enforces its binary-feature/date contract and requires explicit 243-feature projection. The generic date-feature helper accepts other finite numeric features without a fixed feature count.

Freeze raw source bytes where available, a model-input Parquet table, source metadata and hashes. Preserve the exact CSV runtime snapshot for compatibility. Existing projects without provenance remain supported; do not update their source hashes or regenerate calendars. Calendar Studio is included in the SDK's existing complete installation policy, but callers can supply other generators' tables.

Publish an additive calendar_provenance dataset and frozen calendar raw deliveries. Query provenance from Observatory only on demand, with legacy publication fallback. Do not alter existing publication columns or Engine 2/3 contracts.

## Alternatives and consequences

Automatically selecting/dropping features could change trained model inputs silently. Embedding generation inside inference would break reproducibility. Replacing historical snapshot files would invalidate model provenance. Regenerating calendars on every notebook execution would prevent reliable resumption.

New runs store small redundant CSV/Parquet copies deliberately for the existing runtime. Input holiday coverage is the caller's responsibility; the library does not infer complete official calendars. New calendar rules/holidays require a new run. Publication needs permission to create the additive dataset table.

## Validation and rollback

Test external CSV bytes, Parquet and Calendar Studio artifact lifecycle; verify full unprojected artifacts are rejected; detect snapshot mutation; test generic numeric features; exercise all MAC stages on CPU for all three formats, zero-shot behavior and unchanged Engine 2/3 layouts. Run the suite from the installed wheel as well as source. Existing runs and publications can be opened with their frozen runtime without conversion; reverting the package must not rewrite their data.
