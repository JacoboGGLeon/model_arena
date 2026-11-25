import optuna
import pytest

from model_arena.hpo_resume import enqueue_interrupted
from model_arena.input_spec import study_plan


@pytest.mark.parametrize(
    ("families", "count"),
    [
        ({"tide": 25}, 50),
        ({"nhits": 25}, 50),
        ({"tide": 25, "nhits": 25}, 100),
        (dict(tide=25, nhits=25, nlinear=25, dlinear=25, nbeats=25), 250),
    ],
)
def test_budget_is_shared_only_across_compatible_inputs(families, count):
    config = dict(
        inputs=["X0", "X1", "X2", "X3"], objectives=["saldo", "variacion"],
        families=families, input_search={"mode": "family_objective"},
    )
    tasks = study_plan(config, 15)
    assert len(tasks) == len(families) * 2
    assert sum(task["trials"] for task in tasks) == count
    for task in tasks:
        if task["family"] in {"tide", "nhits"}:
            assert task["inputs"] == config["inputs"]
            assert task["startup_trials_per_input"] == 4
        else:
            assert task["input"] == "X0"


def test_insufficient_shared_budget_fails_before_starting():
    config = dict(inputs=["X0", "X1"], families={"tide": 7}, input_search={"mode": "family_objective"})
    with pytest.raises(ValueError, match="requieren al menos 8"):
        study_plan(config, 15)
    assert len(study_plan({**config, "input_search": {"mode": "per_input"}}, 15)) == 2


def test_retry_preserves_startup_input_before_first_suggestion():
    study = optuna.create_study()
    study.enqueue_trial({"input": "X2"})
    study.ask()
    enqueue_interrupted(study, "pooled_tide_saldo", pooled=True)
    enqueue_interrupted(study, "pooled_tide_saldo", pooled=True)
    waiting = study.get_trials(states=(optuna.trial.TrialState.WAITING,))
    assert len(waiting) == 1
    retry = study.ask()
    assert retry.suggest_categorical("input", ["X0", "X2"]) == "X2"
    assert "resume_candidate_id" not in retry.user_attrs
