# ADR-0008: Independent objective studies and global routing

Status: Accepted for 0.1.5. Supersedes the MAC joint-objective policy in ADR-0007.

## Context
Saldo and variation have different dynamics. Combining their losses can prune a useful specialist and forces shared hyperparameters. Separate winners alone do not resolve that compromise.

## Decision
New MAC projects explicitly freeze `curvas.objectives=["saldo","variacion"]` and `selection_scope="entity_target"`. A study is identified by input, family and objective. Each candidate owns one target, one encoder and decoder (or one complete upstream network). History is sliced before the network; losses, reports and pruning contain only that objective. Cross-objective encoder sharing is removed from the MAC search. NBEATS block shared_weights remains a separate, legitimate parameter.

All candidates remain global across eligible training accounts. No account-specific weight fitting occurs. Finalists are retained by objective, family, input and budget. Selection uses F1–F3 and routes by account and objective. Atypical accounts are excluded from training and own-score selection; fallback uses seen segment evidence. Selected configurations are fit from scratch for 75 epochs and inference never optimizes weights. Historical predictions use fold-specific fits of the final selected configuration; F1–F3 are selection evidence, not an independent test.

Prediction/daily rows already include target. Inventory retains one row per account; candidate and selection_source contain an explicit JSON mapping by objective. Engine2/3 schemas stay unchanged. The alert monitor resolves winners per target.

## Alternatives and cost
Joint studies are cheaper in study count but share configurations/pruning decisions. Independent studies with the same per-study budget double trials: 750 candidates /2250 fold fits at 25, before pruning. Each fit has one objective; wall-time and disk ratios require ADA measurement. Trial budgets are configurable before creation and unrelated to entropy budgets.

## Compatibility and rollback
Existing project snapshots remain unchanged; absent objectives in legacy frozen training retain the joint path. New runs are required; no in-place conversion of winners or publications. Generic ModelArena remains agnostic and supports multi-target schemas; use separate instances with one target_col for independent objectives in other domains. Revert to 0.1.4 with a separate project to restore joint training.

## Validation
Synthetic MAC lifecycle tests cover tuning/pruning, independent target routes, global checkpoints, zero-shot accounts, prediction, release preparation and Engine2/3 contracts. Routing tests use opposing objective winners. CPU validation does not establish ADA GPU speed or production accuracy.
