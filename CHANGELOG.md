# Changelog

## 0.1.3

- Compare MLP, GRU and compact TFT encoders with the same joint-target quantile decoder.
- Add balanced, resume-safe family allocation within each input's trial budget.
- Retain a complete reference per encoder/input before patience pruning.
- Preserve TFT defaults, frozen historical projects and existing MAC output layouts.

## 0.1.2

- Accept Calendar Studio artifacts, artifact directories, external Parquet/CSV and DataFrame calendars through an explicit MAC adapter.
- Freeze calendar data, source metadata and hashes in each new project; preserve existing project snapshots.
- Publish calendar provenance as an additive dataset and expose an on-demand provenance view in Observatory.
- Generate calendars explicitly in the Slow System notebook from deployment holiday inputs.
- Add optional fold-local Optuna pruning with exact consecutive-epoch patience, fixed validation reports, partial-curve retention and a complete-reference safeguard.
- Configure new ADA deployments with 15 maximum HPO epochs, patience 4 and 15 COMPLETE/PRUNED attempts per input; keep 75 final epochs.
- Publish every trial state and parameters in an additive hpo_trials dataset; never rank pruned candidates as complete three-fold candidates.
- Preserve model mathematics, selection, alerting rules and Engine 2/3 schemas.

## 0.1.1

- Store per-account epoch losses and per-fit histories as atomic Zstandard Parquet, preserving Float64.
- Read detailed loss datasets in batches for publication; reuse immutable raw deliveries with hardlinks when possible.
- Add configurable disk reserve checks, one-writer project locks, and exact-parameter retry for disk-interrupted candidates.
- Package the local read-only loss dashboard with input/fold/budget/account filters, chart/table views and Parquet download.
- Provide explicit verified conversion for unpublished SDK 0.1 runs, retaining completed weights, predictions and HPO state.
- Keep model mathematics, scaling, winner selection and accounting alert rules unchanged.
- Update installed-SDK notebooks to 0.1.1 and test installed wheels in CI before publishing.

## 0.1

Initial complete SDK installation with generic forecasting APIs, MAC compatibility profile and Observatory.
