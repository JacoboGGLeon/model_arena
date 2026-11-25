# Model Arena

Explicit forecasting competition and observable learning histories.

Version **0.1.4**. Install the complete SDK with `python -m pip install "model-arena-sdk==0.1.4"`.
Source: https://github.com/JacoboGGLeon/model_arena · PyPI: https://pypi.org/project/model-arena-sdk/.

The import remains `model_arena`. Default dependencies cover training, AWS publication and Observatory; no extras are required for the ADA workflow. This release includes a generic forecasting core and a MAC migration adapter. The storage profile was exercised end-to-end in ADA, including S3/Athena publication; automated tests use synthetic data.

```python
from model_arena import ModelArena

arena = ModelArena.from_config('arena.yaml')
study = arena.tune(df=history, static_df=metadata)
selection = arena.select(study)
arena.fit(df=history, static_df=metadata, selection=selection, epochs=75)
forecasts = arena.predict(futr_df=known_future)
release = arena.postprocess(forecasts)
publication = arena.publish(release, destination='publications')
```

`fit` does not launch HPO or publish. `predict` never trains. Temporal selection folds are not an independent test. A global configuration can serve multiple entities; zero-shot entities are excluded from parameter fitting. Every target retains its own output head and learning history.

## ADA accounting workflow

`notebooks/10_ADA_Slow_System_End_to_End.ipynb` uses `model_arena.profiles.mac.MacProject` to preserve the established accounting engines and Athena dashboard during migration. `00_ADA_Fast_System_Observatory.ipynb` opens that monitor without training. Deployment configurations and the calendar are supplied separately, outside the public repository. See the implementation status for the explicit migration boundary.

## Installation

The default installation includes the complete forecasting and monitoring runtime. In ADA, retain the corporate pip/Artifactory configuration. Do not print `pip config list` or embed tokens in notebooks.

```bash
python -m pip install "model-arena-sdk==0.1.4"
```

In ADA, keep the existing Artifactory indexes and approved CUDA-compatible PyTorch installation. No package index or CUDA build is forced by the SDK. Default installation includes PyTorch and the dashboard dependencies; imports remain lazy and reading results does not initialize CUDA. Existing extras remain available for compatibility.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src/model_arena tests scripts --exclude _mac_runtime
python -m build
python -m twine check dist/*
```

See `docs/INSTALL_ADA.md`, `docs/RELEASE.md` and `docs/IMPLEMENTATION_STATUS.md`. Notebooks contain no credentials, internal endpoints or datasets. Corporate configuration is provided as a local file excluded from git.

## Storage and local loss monitor (0.1.1)

Loss details and summaries use atomic Zstandard Parquet with Float64 precision. A configurable disk reserve and project writer lock protect sequential runs. Trials stopped by the disk guard are retried with their original parameters and candidate identity. Completed checkpoints are reused; incomplete fits restart from scratch.

```python
from model_arena.profiles.loss_monitor import launch
monitor = launch("runs/my_project")
```

This read-only monitor filters fixed training/validation curves by input, fold, budget and account, and offers chart/table views and Parquet downloads. It does not require publication or GPU.

An explicit upgrade of an unpublished SDK 0.1 run is available through `model_arena.profiles.storage_upgrade.upgrade(path)`. It verifies the old runtime, backs up identity metadata and runtime, validates CSV-to-Parquet conversion before removing CSVs, and retains models/trials. Existing published releases do not need conversion. For a tested ADA bundle project, `MacProject(path)` can reuse its runtime snapshot unchanged.

The disk reserve is a point-in-time guard, not a guarantee of capacity for an entire run. The model mathematics, target scaling and accounting rules are unchanged.

## Calendar Studio integration (0.1.3)

The complete SDK installation includes Calendar Studio. Its use is explicit: Model Arena still accepts external CSV/Parquet calendars and the generic forecasting core does not prescribe a calendar rule engine.

```python
from calendar_studio import CalendarStudio, load_holidays
from model_arena.profiles.mac import MacProject

calendar = CalendarStudio.from_profile("mac_legacy_v1").generate(
    start="2022-01-01", end="2027-12-31",
    holidays=load_holidays("private/festivos.csv", date_col="fecha"),
).view("mac_legacy_v1")

project = MacProject.create(
    "runs/my_project", training_config="private/training.json",
    publication_config="private/publication.json", calendar=calendar, device="cuda",
)
print(project.calendar_provenance())
```

Pass exactly one of `calendar` (a validated artifact-like provider or a DataFrame) and `calendar_path` (CSV, Parquet or a Calendar Studio artifact directory). MAC requires `fecha` and exactly the configured binary feature count; the current MAC runtime uses 243 features. Full Calendar Studio outputs have 245 features and must be projected explicitly using `.view("mac_legacy_v1")`. No columns are silently discarded.

Each new project freezes source bytes/metadata and a Zstandard `calendar/features.parquet`, while retaining an exact CSV compatibility snapshot for its existing training runtime. Provenance records generator/version, profile/view, input rules and holidays, date coverage, ordered feature names and hashes. Project verification detects changed provenance or frozen input files. CSV input bytes are preserved exactly.

Historical projects without structured provenance remain supported and retain their original runtime and calendar snapshots. Installing a newer library does not regenerate their calendars. To use corrected holidays, create a new project. The supplied deployment holiday list includes 2026-11-02; review its coverage before extending to other years.

New publications add a separate `calendar_provenance` dataset and include frozen calendar sources in their raw deliveries. Observatory's provenance expander queries it on demand; older publications without that table/row remain usable. Existing Engine 2/3 layouts are unchanged.

For other projects, `model_arena.calendar.to_features(calendar, time_col="ds")` returns an independent date-feature table. DataFrames require an explicit `date_col`; continuous finite numeric calendar features are permitted outside the MAC adapter. Join/broadcast this table to the panel as required by your schema.

## Optional HPO patience (0.1.3)

The new ADA deployment uses 15 trials per input, at most 15 epochs per fold, and four consecutive epochs without improvement on fixed temporal validation. The first trial of each input completes all selection folds to retain a complete reference; subsequent stagnant trials are marked `PRUNED`. The patience counter resets for each fold. A pruned candidate is discarded as a whole; partial fold results do not enter winner selection. The final fit still runs 75 epochs from scratch, without pruning.

```python
from model_arena import PruningSpec, SearchSpec, TrainingSpec

training = TrainingSpec(epochs=15, final_epochs=75)
search = SearchSpec(n_trials=15, pruner=PruningSpec(
    kind="patience", patience=4, min_delta=0.0, min_completed_trials=1,
))
```

Supply these contracts in `ArenaSpec(training=training, search=search, ...)`. Generic HPO reports the mean validation pinball over declared targets and training entities. The MAC deployment uses `model.epochs=15` and `curvas.pruner={"kind":"patience","patience":4,"min_delta":0.0,"min_completed_trials":1}` in `training.json`; it reports the mean normalized validation pinball for saldo/variación on the fixed monitored training accounts. Configurations without an explicit pruning policy remain unpruned.

One step is one reported epoch. With the first best value at epoch 1, unchanged losses at epochs 2–5 prune at the end of epoch 5. Any strictly lower loss resets patience when `min_delta=0`; ties count as no improvement. Reports use distinct step ranges per fold, and the pruner compares values only inside the current fold. `FoldPatiencePruner` implements this exact policy through Optuna's `BasePruner`; it is not Optuna's `PatientPruner` wrapper or a cross-trial Median/Hyperband rule. No smoothing is applied to the reported curves.

The 15-attempt limit counts `COMPLETE` plus `PRUNED`. Disk-interrupted retries retain their original identity and parameters. Trial state, pruning fold/epoch and parameters are retained in Parquet; only complete trials have a final objective value. New publications add `hpo_trials`, accessible on demand in Observatory, while their partial epoch curves remain available. Reopening an old run does not modify its configuration or frozen runtime.

## Interchangeable encoders (0.1.3)

Use `ModelSpec(alias="X3", kind="encoder_decoder", historical_features=[...], future_features=[...])`
with `SearchSpec(n_trials=25, encoder_kind=["mlp", "gru", "tft_compact"], pruner=PruningSpec(kind="patience"))`.
`encoder_kind` selects architecture; `encoder_design` selects shared versus independent encoders across targets. Every target retains its own horizon/quantile head. The MAC adapter keeps saldo and variación.

MAC configuration uses `curvas.encoder_kinds=["mlp","gru","tft_compact"]` and `curvas.trials=25`.
This is **25 total trials per input**, balanced 9/8/8 in declared order. For 25 per encoder use 75 total trials per input. Each encoder/input retains its first complete reference before pruning. One TPE study per input samples the remaining hyperparameters across the balanced family allocation; these are not isolated per-encoder studies. All 4 inputs use at most 100 candidates / 300 fold fits at the 25-total setting. Failed disk retries retain identity and do not consume another completed/pruned candidate slot.

Existing projects keep frozen code/configuration and default to TFT when encoder_kind is absent. Use a new project to explore all three families. Stored candidate JSON and trial parameter JSON identify the architecture. No model quality or runtime advantage is claimed without a real ADA comparison.

## Complete forecasting families (0.1.4)

`ModelSpec(kind="tide" | "nhits" | "nlinear" | "dlinear" | "nbeats", alias=..., trials=25)` declares a complete model family, not a TFT encoder. Each target has an independent upstream network inside the candidate; selection combines target scores using the existing policy. The registered families use NeuralForecast **3.1.2**, with Lightning 2.5.x for compatibility with tested PyTorch 2.6. Architecture forwards are trained by Model Arena's AdamW/window/epoch loop, not a Lightning Trainer. Quantile outputs are monotonically rearranged. History uses linear window normalization for these families; Encoder–Decoder keeps its original transform.

NLinear/DLinear/NBEATS support **X0 only**. Supplying covariates raises an error. TiDE/NHITS support past and future covariates. The historical portion of the future-covariate channel is explicitly zero-masked; past calendar enters the separate historical channel. No observed future targets enter inference. Complete families have their own typed parameter container (`lr`, `weight_decay`, `architecture`); encoder-layer/shared flags are not applied to them.

```python
models = [
    ModelSpec(alias="ED_X0", kind="encoder_decoder", trials=25),
    ModelSpec(alias="TiDE_X0", kind="tide", trials=25),
    ModelSpec(alias="NHITS_X0", kind="nhits", trials=25),
    ModelSpec(alias="NLinear_X0", kind="nlinear", trials=25),
    ModelSpec(alias="DLinear_X0", kind="dlinear", trials=25),
    ModelSpec(alias="NBEATS_X0", kind="nbeats", trials=25),
]
```

Each `ModelSpec` has a separate TPE study and budget; `tune(n_trials=...)` explicitly overrides those budgets. Spaces tune TiDE dense/temporal widths and depths; NHITS block counts, pooling, MLP widths/activation; DLinear moving-average length; NBEATS block counts, dense widths, activation and shared weights; all families tune AdamW LR/weight decay. NLinear has no invented depth/activation dimensions.

MAC deployment opts in through `curvas.families={"encoder_decoder":25,"tide":25,"nhits":25,"nlinear":25,"dlinear":25,"nbeats":25}`. X0-only families run only on X0; all other configured inputs are retained for the other families. This means up to **375 candidates / 1125 fold fits** for X0–X3, before pruning; final fits remain 75 epochs. Missing `families` preserves the old Encoder–Decoder-only behavior. Capabilities and budgets are explicit, not inferred from entropy budgets. Checkpoints record upstream version and architecture-source hashes; changed upstream code is rejected when restoring. Create a new MAC project for this protocol.

NBEATS `shared_weights` controls reuse between blocks, not sharing between saldo and variación networks.
