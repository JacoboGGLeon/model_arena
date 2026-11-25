"""Optuna pruning with exact consecutive-epoch patience, scoped to each fold.

An equal value counts as no improvement. A pruned candidate is discarded as a
whole; its partial folds must never enter the complete-candidate ranking.
"""

import math

import optuna

from .contracts import PruningSpec


class FoldPatiencePruner(optuna.pruners.BasePruner):
    def __init__(self, *, epochs_per_fold, patience=4, min_delta=0.0, min_completed_trials=1):
        self.spec = PruningSpec(
            kind="patience", patience=patience, min_delta=min_delta,
            min_completed_trials=min_completed_trials,
        )
        if not isinstance(epochs_per_fold, int) or epochs_per_fold < 1:
            raise ValueError("epochs_per_fold must be a positive integer")
        self.epochs_per_fold = epochs_per_fold

    def prune(self, study, trial):
        if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) < self.spec.min_completed_trials:
            return False
        if not trial.intermediate_values:
            return False
        last = max(trial.intermediate_values)
        start = last // self.epochs_per_fold * self.epochs_per_fold
        values = [v for step, v in sorted(trial.intermediate_values.items()) if start <= step <= last]
        best, stale = None, 0
        minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Pruning requires finite validation losses")
            improved = best is None or (
                value < best-self.spec.min_delta if minimize else value > best+self.spec.min_delta
            )
            if improved:
                best, stale = value, 0
            else:
                stale += 1
        return stale >= self.spec.patience


def make_pruner(settings, *, epochs_per_fold):
    spec = settings if isinstance(settings, PruningSpec) else PruningSpec.model_validate(settings or {})
    if spec.kind == "none":
        return optuna.pruners.NopPruner()
    return FoldPatiencePruner(epochs_per_fold=epochs_per_fold, **spec.model_dump(exclude={"kind"}))


def report_epoch(trial, value, *, epoch, fold_index, epochs_per_fold, fold):
    """Report a fixed validation set after an epoch; never report final-fit data."""
    if not math.isfinite(value):
        raise ValueError("Pruning requires finite validation losses")
    trial.report(float(value), fold_index*epochs_per_fold+epoch-1)
    if trial.should_prune():
        trial.set_user_attr("pruned_fold", fold)
        trial.set_user_attr("pruned_epoch", epoch)
        trial.set_user_attr("pruned_reason", "validation_patience")
        raise optuna.TrialPruned(f"{fold}: validation did not improve; stopped at epoch {epoch}")


def trial_records(study, inp):
    """Full trial audit, with partial objectives kept separate from final scores."""
    import json

    records = []
    for trial in study.get_trials():
        attrs = trial.user_attrs
        records.append(dict(
            input=inp, candidate=attrs.get("candidate_id", attrs.get("resume_candidate_id", f"{inp}_T{trial.number:04d}")),
            trial_number=trial.number, state=trial.state.name,
            value=trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None,
            objective_value=trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None,
            last_intermediate_value=trial.intermediate_values[max(trial.intermediate_values)] if trial.intermediate_values else None,
            pruned_fold=attrs.get("pruned_fold"), pruned_epoch=attrs.get("pruned_epoch"),
            parameters_json=json.dumps(trial.params, sort_keys=True),
        ))
    return records
