"""Explicit HPO, selection, fit and inference lifecycle."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from .contracts import (ArenaSpec, ModelSpec, PanelSchema, RunRef, StudyResult, SelectionResult,
                        ArtifactRef)
from .data import PreparedData, prepare_panel, future_dates, pinball, scale_for
from .models.base import ModelRegistry
from .results import Results
from ._version import __version__
from .storage import (signature, write_json, read_json, write_table, read_table,
                      publish_directory, verify_manifest, digest)


def run_id(prefix):
    return prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


class ModelArena:
    def __init__(self, spec=None, *, models=None, schema=None, freq=None, calendar=None,
                 h=20, window=20, artifacts="runs", registry=None):
        if spec is None:
            spec = ArenaSpec(models=models, schema=schema or PanelSchema(), freq=freq,
                             calendar=calendar, h=h, window=window, artifacts=str(artifacts))
        self.spec = spec if isinstance(spec, ArenaSpec) else ArenaSpec.model_validate(spec)
        self.registry = registry or ModelRegistry()
        self._models = {}
        self._prepared = None
        self._selection = None
        self._study = None
        self._results = None
        self._cutoff = None

    @classmethod
    def from_config(cls, path, **kwargs):
        return cls(ArenaSpec.from_yaml(path), **kwargs)

    def describe(self):
        return self.spec.model_dump(mode="json")

    def plan(self, df=None, *, static_df=None):
        data = self.prepare(df=df, static_df=static_df) if df is not None else None
        return dict(inputs=[m.alias for m in self.spec.models],
                    candidate_configurations=len(self.spec.models)*self.spec.search.n_trials,
                    hpo_fits=len(self.spec.models)*self.spec.search.n_trials*self.spec.cv.n_windows,
                    hpo_epochs=self.spec.training.epochs, final_epochs=self.spec.training.final_epochs,
                    training_entities=len(data.training_ids) if data else None,
                    total_entities=len(data.ids) if data else None,
                    final_unique_models="resolved after selection", publishes=False)

    def prepare(self, prepared=None, *, df=None, static_df=None):
        if prepared is not None:
            if df is not None or static_df is not None:
                raise ValueError("Choose prepared data or DataFrames, not both")
            if not isinstance(prepared, PreparedData) or prepared.schema != self.spec.schema:
                raise ValueError("PreparedData with the same schema is required")
            return prepared
        if df is None:
            raise ValueError("Provide df or PreparedData")
        return prepare_panel(df, static_df, self.spec)

    def _new_run(self, prefix, data, extra=None):
        identity = signature({"spec": self.describe(), "data": data.signature, "extra": extra})
        path = Path(self.spec.artifacts).resolve() / run_id(prefix)
        path.mkdir(parents=True)
        write_json(path / "identity.json", {"signature": identity, "data": data.signature,
                                            "spec": self.describe(), "version": __version__})
        inventory = data.metadata.rename(columns={data.schema.id_col: "entity"})
        write_table(path / "inventory.json", inventory)
        write_json(path / "status.json", {"status": "RUNNING"})
        return RunRef(run_id=path.name, path=str(path), signature=identity)

    def _folds(self, data, cv):
        step = cv.step_size or self.spec.h
        first = len(data.axis)-self.spec.h-(cv.n_windows-1)*step
        if first < self.spec.window + self.spec.h:
            raise ValueError("Insufficient history for temporal protocol")
        return [(f"F{i+1}", first+i*step) for i in range(cv.n_windows)]

    def _scores(self, data, model, end, candidate, inp, fold):
        refs = [(key, end) for key in data.ids]
        predictions = model.predict(data, origins=refs)
        rows = []
        for key, prediction in zip(data.ids, predictions):
            truth = data.values[key][end:end+self.spec.h]
            history = data.values[key][end-self.spec.window:end]
            baseline = np.repeat(np.quantile(history, self.spec.quantiles, axis=0).T[None],
                                 self.spec.h, axis=0)
            scale = scale_for(history)
            model_loss = pinball(truth/scale, prediction/scale[None, :, None], self.spec.quantiles)
            baseline_loss = pinball(truth/scale, baseline/scale[None, :, None], self.spec.quantiles)
            for i, target in enumerate(self.spec.schema.target_cols):
                rows.append(dict(candidate=candidate, input=inp, fold=fold, entity=key, target=target,
                                 value=float(model_loss[:, i].mean()),
                                 baseline_value=float(baseline_loss[:, i].mean()),
                                 n_observations=self.spec.h*len(self.spec.quantiles)))
        return pd.DataFrame(rows)

    def _evaluate_candidate(self, data, model_spec, candidate, run, cv, epochs):
        tables = []
        root = Path(run.path)
        for fold, end in self._folds(data, cv):
            model = self.registry.create(model_spec, self.spec)
            history = []
            def record(rows):
                for row in rows:
                    history.append(dict(**row, candidate=candidate, input=model_spec.alias, fold=fold,
                                        phase="hpo", metric="pinball", scale="window_normalized",
                                        population_ref=data.signature, cutoff=str(data.axis[end-1])))
                write_table(root / "learning_curves" / f"{candidate}_{fold}.json", pd.DataFrame(history))
            model.fit(data, end=end, epochs=epochs, fold=fold, on_epoch=record)
            table = self._scores(data, model, end, candidate, model_spec.alias, fold)
            tables.append(table)
            # Fold weights remain available for causal inspection and future exports.
            model.save(root / f"{candidate}_{fold}.weights")
            write_json(root / f"{candidate}_{fold}.provenance.json",
                       {"candidate": candidate, "fold": fold, "cutoff": str(data.axis[end-1]),
                        "training_ids": data.training_ids, "data_signature": data.signature,
                        "monitor": getattr(model, "monitor", None)})
        return pd.concat(tables, ignore_index=True)

    def cross_validation(self, prepared=None, *, df=None, static_df=None, cv=None, epochs=None):
        data = self.prepare(prepared, df=df, static_df=static_df)
        cv = cv or self.spec.cv
        self._folds(data, cv)
        run = self._new_run("backtest", data, cv.model_dump())
        tables, candidates = [], []
        for i, model_spec in enumerate(self.spec.models):
            candidate = f"C{i:04d}"
            tables.append(self._evaluate_candidate(data, model_spec, candidate, run, cv,
                                                   epochs or self.spec.training.epochs))
            candidates.append(dict(candidate=candidate, model=model_spec.model_dump(mode="json")))
        write_table(Path(run.path) / "scores.json", pd.concat(tables, ignore_index=True))
        write_json(Path(run.path) / "candidates.json", candidates)
        write_json(Path(run.path) / "status.json", {"status": "COMPLETE"})
        self._study = StudyResult(**run.model_dump(), evidence_role=cv.evidence_role)
        self._results = Results(run.path)
        return self._study

    def tune(self, prepared=None, *, df=None, static_df=None, cv=None, n_trials=None, epochs=None):
        import optuna
        data = self.prepare(prepared, df=df, static_df=static_df)
        cv = cv or self.spec.cv
        self._folds(data, cv)
        run = self._new_run("study", data, cv.model_dump())
        root = Path(run.path)
        space, candidates, scores, trials = self.spec.search, [], [], []
        for i, base in enumerate(self.spec.models):
            if base.kind != "tft_compact":
                raise ValueError("HPO requires a declared search adapter; use cross_validation for fixed models")
            sampler = optuna.samplers.TPESampler(seed=self.spec.training.seed+i)
            study = optuna.create_study(direction="minimize", sampler=sampler,
                                        pruner=optuna.pruners.NopPruner(),
                                        storage="sqlite:///"+str(root / "hpo.sqlite3"), study_name=base.alias)
            def objective(trial):
                params = {key: trial.suggest_categorical(key, getattr(space, key))
                          for key in ("encoder_design", "depth", "decoder_depth", "width", "activation")}
                params.update(lr=trial.suggest_float("lr", *space.lr_range, log=True),
                              weight_decay=trial.suggest_float("weight_decay", *space.wd_range, log=True),
                              dropout=space.dropout)
                model_spec = base.model_copy(update={"params": {**base.params, **params}})
                candidate = f"M{i:02d}_T{trial.number:04d}"
                candidates.append(dict(candidate=candidate, model=model_spec.model_dump(mode="json")))
                write_json(root / "candidates.json", candidates)
                table = self._evaluate_candidate(data, model_spec, candidate, run, cv,
                                                  epochs or self.spec.training.epochs)
                scores.append(table)
                write_table(root / "scores.json", pd.concat(scores, ignore_index=True))
                value = float(table[table.entity.isin(data.training_ids)].value.mean())
                trials.append(dict(candidate=candidate, input=base.alias, value=value, state="COMPLETE"))
                write_table(root / "trials.json", pd.DataFrame(trials))
                return value
            study.optimize(objective, n_trials=n_trials or space.n_trials)
        write_json(root / "sampler.json", {"kind": "TPESampler", "optuna_version": optuna.__version__,
                                           "seed": self.spec.training.seed, "pruner": "NopPruner"})
        write_json(root / "status.json", {"status": "COMPLETE"})
        self._study = StudyResult(**run.model_dump(), evidence_role=cv.evidence_role)
        self._results = Results(root)
        return self._study

    def select(self, study, *, by=None, policy=None):
        root = Path(study.path)
        scores = read_table(root / "scores.json")
        inventory = read_table(root / "inventory.json")
        policy = policy or self.spec.selection
        by = by or self.spec.schema.id_col
        if by != self.spec.schema.id_col:
            raise ValueError("This release selects one winner per entity")
        identity = read_json(root / "identity.json")
        if identity["spec"] != self.describe():
            raise ValueError("Study belongs to a different resolved specification")
        population = self.spec.schema.population_col
        training_ids = set(inventory.entity if population is None else
                           inventory.loc[inventory[population] == self.spec.schema.training_label, "entity"])
        training = scores[scores.entity.isin(training_ids)]
        if training.empty:
            raise ValueError("No training evidence available for selection")
        # One finalist per input; deterministic tie-break by candidate identifier.
        finalists = (training.groupby(["input", "candidate"]).value.mean().reset_index()
                     .sort_values(["value", "candidate"]).drop_duplicates("input"))
        admissible = training[training.candidate.isin(finalists.candidate)]
        rows = []
        for item in inventory.to_dict("records"):
            key = item["entity"]
            evidence = admissible[admissible.entity == key] if key in training_ids else pd.DataFrame()
            origin = "SERIE"
            if evidence.empty:
                keys = list(policy.group_cols)
                peers = inventory[inventory.entity.isin(training_ids)]
                for name in keys:
                    peers = peers[peers[name] == item[name]]
                origin = "GROUP" if keys and len(peers) >= policy.min_accounts else "GLOBAL"
                evidence = admissible[admissible.entity.isin(peers.entity)] if origin == "GROUP" else admissible
            rank = evidence.groupby(["candidate", "input"]).agg(
                value=("value", "mean"), baseline_value=("baseline_value", "mean")).reset_index()
            winner = rank.sort_values(["value", "candidate"]).iloc[0]
            gain = (winner.baseline_value-winner.value)/max(abs(winner.baseline_value), 1e-12)
            rows.append(dict(entity=key, candidate=winner.candidate, input=winner.input,
                             reference="Winner" if gain >= policy.min_gain else "Baseline",
                             gain=float(gain), evidence_origin=origin,
                             population="seen" if key in training_ids else "zero_shot"))
        selection_path = root / "selection"
        write_table(selection_path / "winners.json", pd.DataFrame(rows))
        write_table(selection_path / "finalists.json", finalists)
        write_json(selection_path / "policy.json", policy.model_dump())
        self._selection = SelectionResult(run_id=run_id("selection"), path=str(selection_path),
                                          signature=study.signature, study_id=study.run_id)
        self._study = study
        return self._selection

    def fit(self, prepared=None, *, df=None, static_df=None, selection=None, epochs=None):
        data = self.prepare(prepared, df=df, static_df=static_df)
        if selection is not None:
            study_root = Path(selection.path).parent
            if read_json(study_root / "identity.json")["data"] != data.signature:
                raise ValueError("Selection and final fitting data differ")
            winners = read_table(Path(selection.path) / "winners.json")
            catalog = {x["candidate"]: ModelSpec.model_validate(x["model"])
                       for x in read_json(study_root / "candidates.json")}
            models = {key: catalog[key] for key in sorted(set(winners.candidate))}
        else:
            if len(self.spec.models) != 1:
                raise ValueError("Provide a selection or one explicit model for fit")
            models = {"C0000": self.spec.models[0]}
            winners = pd.DataFrame([dict(entity=key, candidate="C0000", input=self.spec.models[0].alias,
                                         reference="Winner", evidence_origin="EXPLICIT") for key in data.ids])
        run = self._new_run("fit", data)
        root, fitted = Path(run.path), {}
        write_table(root / "winners.json", winners)
        for candidate, model_spec in models.items():
            model = self.registry.create(model_spec, self.spec)
            history = []
            def record(rows):
                for row in rows:
                    history.append(dict(**row, candidate=candidate, input=model_spec.alias, fold="FUTURO",
                                        phase="final", metric="pinball", scale="window_normalized",
                                        population_ref=data.signature, cutoff=str(data.axis[-1])))
                write_table(root / "learning_curves" / f"{candidate}_FUTURO.json", pd.DataFrame(history))
            model.fit(data, end=len(data.axis), epochs=epochs or self.spec.training.final_epochs,
                      fold="FUTURO", on_epoch=record)
            fitted[candidate] = model
            model.save(root / f"{candidate}.weights")
        write_json(root / "status.json", {"status": "FITTED", "training_entities": len(data.training_ids),
                                           "total_entities": len(data.ids), "cutoff": str(data.axis[-1])})
        self._models, self._prepared, self._selection, self._cutoff = fitted, data, selection, data.axis[-1]
        self._fitted_ref, self._results = run, Results(root)
        return self

    def predict(self, *, df=None, static_df=None, futr_df=None, h=None, quantiles=None):
        if not self._models:
            raise ValueError("Call fit or load fitted state before predict")
        if h is not None and h != self.spec.h:
            raise ValueError("Horizon differs from fitted decoder capacity")
        if quantiles is not None and list(quantiles) != self.spec.quantiles:
            raise ValueError("Requested quantiles differ from fitted outputs")
        data = self.prepare(df=df, static_df=static_df) if df is not None else self._prepared
        if data is None:
            raise ValueError("Provide df and metadata after loading without history")
        if data.axis[-1] < self._cutoff:
            raise ValueError("Final fitted weights cannot create a causal historical backtest")
        winners = Results(self._fitted_ref.path).table("winners")
        if set(data.ids) - set(winners.entity):
            raise ValueError("New entities require an explicit transfer selection")
        dates = future_dates(data, self.spec, self.spec.h)
        required_features = {f for model in self._models.values() for f in model.model_spec.future_features}
        if required_features:
            if futr_df is None:
                raise ValueError("Future known covariates are required")
            future = futr_df.copy()
            future[self.spec.schema.id_col] = future[self.spec.schema.id_col].astype(str)
            future[self.spec.schema.time_col] = pd.to_datetime(future[self.spec.schema.time_col])
            expected = pd.MultiIndex.from_product([data.ids, dates])
            actual = pd.MultiIndex.from_frame(future[[self.spec.schema.id_col, self.spec.schema.time_col]])
            if actual.has_duplicates or set(actual) != set(expected):
                raise ValueError("Future covariates must cover every entity and forecast date exactly")
            if not np.isfinite(future[sorted(required_features)].to_numpy(dtype=float)).all():
                raise ValueError("Non-finite future covariates")
        else:
            future = None
        rows = []
        for candidate, model in self._models.items():
            ids = [key for key in winners.loc[winners.candidate == candidate, "entity"] if key in data.ids]
            if not ids:
                continue
            predictions = model.predict(data, origins=[(key, len(data.axis)) for key in ids], future=future)
            for key, prediction in zip(ids, predictions):
                baseline = np.repeat(np.quantile(data.values[key][-self.spec.window:],
                                                  self.spec.quantiles, axis=0).T[None], self.spec.h, axis=0)
                for role, values in [("Winner", prediction), ("Baseline", baseline)]:
                    for j, date in enumerate(dates):
                        for k, target in enumerate(self.spec.schema.target_cols):
                            for q, value in zip(self.spec.quantiles, values[j, k]):
                                rows.append(dict(entity=key, origin=data.axis[-1], ds=date, target=target,
                                                 quantile=q, value=float(value), role=role,
                                                 candidate=candidate, input=model.model_spec.alias))
        return pd.DataFrame(rows)

    def evaluate(self, forecasts, *, observed):
        keys = ["entity", "ds", "target"]
        if observed.duplicated(keys).any():
            raise ValueError("Duplicate observations")
        joined = forecasts.merge(observed[keys+["observed"]], on=keys, validate="m:1")
        error = joined.observed-joined.value
        joined["pinball"] = np.maximum(joined["quantile"]*error, (joined["quantile"]-1)*error)
        return joined.groupby(["entity", "target", "role"]).pinball.mean().reset_index()

    def postprocess(self, forecasts, *, context=None, postprocessors=()):
        if self._results is None:
            raise ValueError("Fitted results required")
        release = Path(self._fitted_ref.path) / run_id("release")
        release.mkdir()
        for name in ["winners", "inventory"]:
            write_table(release / (name+".json"), Results(self._fitted_ref.path).table(name))
        write_table(release / "forecasts.json", forecasts)
        if self._prepared is not None:
            write_table(release / "history.json", self._prepared.frame)
        curve_paths = [Path(self._fitted_ref.path) / "learning_curves"]
        if self._study:
            study = Path(self._study.path)
            curve_paths.insert(0, study / "learning_curves")
            for name in ["scores", "trials"]:
                if (study / (name+".json")).is_file():
                    shutil.copy2(study / (name+".json"), release / (name+".json"))
        for folder in curve_paths:
            if folder.exists():
                shutil.copytree(folder, release / "learning_curves", dirs_exist_ok=True)
        write_json(release / "spec.json", self.describe())
        for processor in postprocessors:
            processor.process(release, forecasts=forecasts, context=context)
        return ArtifactRef(path=str(release), signature=signature(self.describe()))

    def publish(self, release, *, destination):
        if str(destination).startswith("s3://"):
            from .aws import publish_s3
            return publish_s3(release.path, destination)
        return publish_directory(release.path, destination)

    def save(self, destination):
        if not self._models:
            raise ValueError("No fitted models")
        target = Path(destination)
        if target.exists():
            raise FileExistsError("Choose a new artifact destination")
        target.mkdir(parents=True)
        files = {}
        for candidate, model in self._models.items():
            model.save(target / f"{candidate}.weights")
            files[candidate] = model.model_spec.model_dump(mode="json")
        write_json(target / "fitted.json", {"spec": self.describe(), "models": files,
                                            "cutoff": str(self._cutoff), "format": "1"})
        write_table(target / "winners.json", Results(self._fitted_ref.path).table("winners"))
        write_table(target / "inventory.json", Results(self._fitted_ref.path).table("inventory"))
        checksums = {p.name: digest(p) for p in target.iterdir() if p.is_file()}
        write_json(target / "manifest.json", {"files": checksums})
        return ArtifactRef(path=str(target.resolve()), signature=signature(checksums))

    @classmethod
    def load(cls, reference, *, registry=None):
        root = Path(getattr(reference, "path", reference))
        verify_manifest(root)
        state = read_json(root / "fitted.json")
        arena = cls(ArenaSpec.model_validate(state["spec"]), registry=registry)
        for candidate, config in state["models"].items():
            model = arena.registry.create(ModelSpec.model_validate(config), arena.spec)
            model.restore(root / f"{candidate}.weights")
            arena._models[candidate] = model
        arena._results, arena._cutoff = Results(root), pd.Timestamp(state["cutoff"])
        arena._fitted_ref = RunRef(run_id=root.name, path=str(root), signature=signature(state))
        return arena

    @property
    def results(self):
        if self._results is None:
            raise ValueError("No results yet; execute a study or open Results from a publication")
        return self._results

    def status(self, reference=None):
        root = Path(reference.path if reference is not None else self._fitted_ref.path)
        return read_json(root / "status.json")

