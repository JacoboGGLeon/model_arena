import optuna
import pytest

from model_arena import PruningSpec
from model_arena.pruning import FoldPatiencePruner, make_pruner, report_epoch, trial_records


def study_with_reference(**kwargs):
    study = optuna.create_study(pruner=FoldPatiencePruner(epochs_per_fold=15, **kwargs))
    study.add_trial(optuna.trial.create_trial(value=1.0))
    return study, study.ask()


def test_four_consecutive_epochs_including_equal_plateau():
    study, trial = study_with_reference()
    for epoch in range(1, 5):
        report_epoch(trial, 1.0, epoch=epoch, fold_index=0, epochs_per_fold=15, fold="F1")
    with pytest.raises(optuna.TrialPruned, match="epoch 5"):
        report_epoch(trial, 1.0, epoch=5, fold_index=0, epochs_per_fold=15, fold="F1")
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    row = trial_records(study, "X0")[-1]
    assert row["state"] == "PRUNED"
    assert row["objective_value"] is None
    assert row["last_intermediate_value"] == 1.0
    assert row["pruned_fold"] == "F1" and row["pruned_epoch"] == 5


def test_improvement_resets_patience_and_fold_resets_scale():
    _, trial = study_with_reference()
    for epoch, value in enumerate([10., 10., 10., 9., 9., 9., 9.], 1):
        report_epoch(trial, value, epoch=epoch, fold_index=0, epochs_per_fold=15, fold="F1")
    # F2 values are much larger; they must not be compared with F1.
    for epoch in range(1, 5):
        report_epoch(trial, 1000., epoch=epoch, fold_index=1, epochs_per_fold=15, fold="F2")
    with pytest.raises(optuna.TrialPruned, match="F2"):
        report_epoch(trial, 1000., epoch=5, fold_index=1, epochs_per_fold=15, fold="F2")


def test_complete_reference_guard_and_disabled_pruner():
    study = optuna.create_study(pruner=make_pruner(dict(kind="patience"), epochs_per_fold=15))
    trial = study.ask()
    for step in range(15):
        trial.report(1., step)
        assert not trial.should_prune()
    study.tell(trial, 1.)
    assert isinstance(make_pruner(None, epochs_per_fold=15), optuna.pruners.NopPruner)


@pytest.mark.parametrize("direction,values", [("minimize", [10, 9.9, 9.8, 9.7, 9.6]),
                                               ("maximize", [10, 10.1, 10.2, 10.3, 10.4])])
def test_min_delta_and_direction(direction, values):
    study = optuna.create_study(direction=direction, pruner=FoldPatiencePruner(
        epochs_per_fold=15, min_delta=1.0))
    study.add_trial(optuna.trial.create_trial(value=10.))
    trial = study.ask()
    for i, value in enumerate(values):
        trial.report(value, i)
    assert trial.should_prune()


def test_settings_and_nonfinite_losses_rejected():
    for settings in [dict(patience=0), dict(min_delta=-1), dict(min_delta=float("nan")),
                     dict(kind="unknown"), dict(min_completed_trials=0)]:
        with pytest.raises(ValueError):
            PruningSpec(**settings)
    _, trial = study_with_reference()
    with pytest.raises(ValueError, match="finite"):
        report_epoch(trial, float("nan"), epoch=1, fold_index=0, epochs_per_fold=15, fold="F1")


def test_sqlite_reload_retains_fold_local_patience(tmp_path):
    storage = "sqlite:///"+str(tmp_path / "study.sqlite3")
    study = optuna.create_study(study_name="X0", storage=storage,
                               pruner=FoldPatiencePruner(epochs_per_fold=15))
    study.add_trial(optuna.trial.create_trial(value=1.))
    trial = study.ask()
    for step in range(4):
        trial.report(1., step)
    reloaded = optuna.load_study(study_name="X0", storage=storage,
                                pruner=FoldPatiencePruner(epochs_per_fold=15))
    active = optuna.trial.Trial(reloaded, trial._trial_id)
    with pytest.raises(optuna.TrialPruned):
        report_epoch(active, 1., epoch=5, fold_index=0, epochs_per_fold=15, fold="F1")
