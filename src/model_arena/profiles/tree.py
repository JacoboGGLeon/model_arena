"""Global tabular forecaster with one shared model across accounts and horizons."""

import numpy as np
import torch
from torch import nn

from .run_spec import TreeSearch


class HorizonBooster(nn.Module):
    provenance = {"library": "lightgbm", "layout": "global_horizon_v1"}

    def __init__(self, horizon, threads):
        super().__init__()
        self.horizon = horizon
        self.threads = threads
        self.booster = None
        self.rounds = None
        self.register_buffer("model_bytes", torch.empty(0, dtype=torch.uint8))

    def matrix(self, history, past, future):
        history = history.cpu().numpy().reshape(len(history), -1)
        windows = [history]
        if past is not None:
            past = past.cpu().numpy()
            windows.extend((past[:, -1], past.mean(axis=1)))
        repeated = np.repeat(np.concatenate(windows, axis=1), self.horizon, axis=0)
        horizon = np.tile(np.arange(1, self.horizon + 1, dtype=np.float32), len(history))[:, None]
        columns = [repeated, horizon]
        if future is not None:
            columns.append(future.cpu().numpy().reshape(len(repeated), -1))
        return np.concatenate(columns, axis=1)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        encoded = state.get(prefix + "model_bytes")
        if encoded is not None:
            self.model_bytes = torch.empty_like(encoded)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)

    def forward(self, history, past, future):
        if self.booster is None:
            from lightgbm import Booster

            self.booster = Booster(model_str=self.model_bytes.cpu().numpy().tobytes().decode())
        result = self.booster.predict(self.matrix(history, past, future), num_iteration=self.rounds,
                                      num_threads=self.threads)
        return torch.as_tensor(result, dtype=history.dtype, device=history.device).reshape(
            len(history), self.horizon, 1, 1
        )

    def freeze(self):
        encoded = self.booster.model_to_string(num_iteration=self.rounds).encode()
        self.model_bytes = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy())
        self.booster = None


class LightGBMDefinition:
    encoder_search = False
    device = "cpu"

    def supports(self, historical, future):
        pass

    def validate_study(self, study, curves, spec):
        if curves.get("interval", {}).get("method") != "mad" or spec.residuals:
            raise ValueError("LightGBM requires a direct point forecast")
        if spec.tree_search is None:
            raise ValueError("Declare tree_search for LightGBM")

    def suggest(self, trial, config, search, *, window):
        space = TreeSearch.model_validate(config["run_spec"]["tree_search"])
        return dict(
            encoder_kind="lightgbm",
            encoder_design="single",
            tree_params={
                "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
                "num_leaves": trial.suggest_categorical("num_leaves", space.num_leaves),
                "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", space.min_data_in_leaf),
                "lambda_l2": trial.suggest_float("lambda_l2", *space.lambda_l2, log=True),
            },
            tree_limits={"patience": space.patience, "threads": space.threads, "max_rows": space.max_rows},
        )

    def network(self, spec, candidate, dropout, *, targets, past, future, legacy_encoder):
        if targets != 1:
            raise ValueError("LightGBM needs an independent study for each target")
        return HorizonBooster(spec.horizon, candidate.tree_limits["threads"])

    def fit(self, *, series, end, spec, candidate, device, calendar, config, on_epoch,
            monitor, make_network, make_batch, **_):
        from lightgbm import Booster, Dataset

        model = make_network(spec, candidate, config["dropout"])
        tree = model.external
        rng = np.random.default_rng(spec.seed)
        first, last = max(spec.window, spec.horizon), end - spec.horizon
        if first > last:
            raise ValueError("Insufficient history for the tree model")
        # The outer fold remains untouched when choosing the number of trees.
        inner_cutoff = last if last - spec.horizon >= first and monitor and monitor.selection["validation_origin"] else None
        if inner_cutoff is not None:
            last = inner_cutoff - spec.horizon
        capacity = candidate.tree_limits["max_rows"] // spec.horizon
        if capacity < len(series):
            raise ValueError("Increase tree_search.max_rows to include each training series")
        windows = min(capacity, max(spec.samples_per_epoch, len(series)))
        indices = np.concatenate((np.arange(len(series)), rng.integers(len(series), size=windows - len(series))))
        refs = np.column_stack((indices, rng.integers(first, last + 1, size=windows)))
        rng.shuffle(refs)
        matrix = None
        labels = np.empty(windows * spec.horizon, dtype=np.float32)
        for offset in range(0, windows, spec.batch_size):
            features, _, truth = make_batch(series, refs[offset:offset + spec.batch_size], spec,
                                            candidate, device, calendar, True)
            rows = tree.matrix(features["history_y"], features.get("history_calendar"),
                               features.get("future_calendar"))
            if matrix is None:
                matrix = np.empty((len(labels), rows.shape[1]), dtype=np.float32)
            start = offset * spec.horizon
            matrix[start:start + len(rows)] = rows
            labels[start:start + len(rows)] = truth.numpy().reshape(-1)
        params = dict(objective="regression_l1", metric="l1", verbosity=-1, seed=spec.seed,
                      num_threads=tree.threads, force_col_wise=True, **candidate.tree_params)
        dataset = Dataset(matrix, label=labels, params=params)
        tree.booster = Booster(train_set=dataset, params=params)
        def early_stop_loss():
            error, count = 0.0, 0
            indices = [series_index for series_index, item in enumerate(series)
                       if item.id in monitor_ids]
            for offset in range(0, len(indices), spec.batch_size):
                refs = [(index, inner_cutoff) for index in indices[offset:offset + spec.batch_size]]
                features, _, truth = make_batch(series, refs, spec, candidate, device, calendar, True)
                residual = truth.unsqueeze(-1) - model(features)
                error += residual.abs().sum().item()
                count += residual.numel()
            return error / count

        monitor_ids = set(monitor.selection["account_ids"]) if inner_cutoff is not None else set()
        best, best_round, patience_count = np.inf, None, 0
        for step in range(1, spec.epochs + 1):
            finished = tree.booster.update()
            score = float(tree.booster.eval_train()[0][2])
            row = dict(epoch=step, train_loss=score, validation_loss=None,
                       train_loss_saldo=None, train_loss_variacion=None)
            row[f"train_loss_{candidate.target}"] = score
            if monitor is not None:
                row.update(monitor(model))
            if on_epoch is not None:
                on_epoch(row)
            if inner_cutoff is not None:
                validation = early_stop_loss()
                if validation < best:
                    best, best_round, patience_count = validation, tree.booster.current_iteration(), 0
                else:
                    patience_count += 1
            if finished or patience_count >= candidate.tree_limits["patience"]:
                break
        tree.rounds = best_round
        tree.freeze()
        return model
