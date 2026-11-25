# ADR 0009 · Schema-driven MAC profile and Observatory

## Context

MAC has an operational calendar, objective-specific studies, accounting validation and two published delivery schemas. Model Arena also supports forecasting panels from other domains. Both paths need to share forecasting behavior while retaining their input adapters and operating contracts.

## Decision

- Creation validates the operational calendar and freezes the source, ordered feature names, feature types, coverage and hash. Named input definitions project historical and future calendar columns separately. A legacy explicit feature count acts as an assertion. X0 accepts a calendar with dates only.
- The MAC profile declares saldo and variación. Each configured input, family and objective has a study with its trial budget. Family capabilities apply before tuning. The plan lists the maximum fold fits before pruning.
- MAC uses the same neural networks, epoch loop, quantile scoring, temporal folds, score-based ranking and batch prediction as the generic API. MAC owns population construction, per-objective Optuna orchestration, budget finalists, checkpoint persistence, zero-shot routing and accounting exports. The generic workflow accepts arbitrary target names and optional grouping.
- MAC freezes the `complete_finite` eligibility policy, `saldo_entropy` or `none` segmentation, `family_default` or `linear_history` transformation, and pinball or scaled Q50 MAE selection. The transformation choice applies to the Encoder–Decoder family. Zero-shot routing lists fallback steps ending in global evidence. Final fits default to 75 epochs. Optuna patience still monitors fixed validation pinball.
- Observatory reads labels, metrics and objective routing from the publication. Earlier MAC publications have compatibility defaults. The generic dashboard reads each target's label and unit. The accounting monitor shows each effective winner when both objectives are selected.
- The SDK exports the prepared layout using its Arrow schema. Observatory consumes raw Athena pages to produce the complete published CSV with its original column order and decimal precision.

## Validation and compatibility

Synthetic regression cases exercise the 243-column operational view, a three-column calendar with named subsets, X0 with no calendar features, screening incompatible input combinations, and a single-objective neural pipeline without budgets. A fixed-seed comparison checks numerical parity for MLP, GRU and compact TFT checkpoints with the 0.1.5 trainer. Engine schemas retain their ordered columns and published exchange formats. Existing frozen projects and legacy publications remain readable. GPU, S3 and Athena behavior in ADA require a separate live run.
