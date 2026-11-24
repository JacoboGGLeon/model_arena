# Model Arena

Explicit forecasting competition and observable learning histories.

Version **0.1**. Install the complete SDK with `python -m pip install "model-arena-sdk==0.1"`.
Source: https://github.com/JacoboGGLeon/model_arena · PyPI: https://pypi.org/project/model-arena-sdk/.

The import remains `model_arena`. Default dependencies cover training, AWS publication and Observatory; no extras are required for the ADA workflow. This release includes a generic forecasting core and a MAC migration adapter. Local synthetic tests validate the engine layouts; live ADA validation remains pending.

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
python -m pip install "model-arena-sdk==0.1"
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
