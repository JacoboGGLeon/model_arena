"""Retry interrupted fits with the same candidate identity, at most once queued."""

import optuna


def enqueue_interrupted(study, inp):
    all_trials = study.get_trials()
    State = optuna.trial.TrialState
    occupied = {
        t.user_attrs.get("resume_candidate_id", t.user_attrs.get("candidate_id"))
        for t in all_trials
        if t.state in (State.WAITING, State.COMPLETE, State.PRUNED)
    }
    for trial in reversed(all_trials):
        retry = trial.state == State.RUNNING or (
            trial.state == State.FAIL and trial.user_attrs.get("retry_after_disk", False)
        )
        if not retry:
            continue
        candidate = trial.user_attrs.get("candidate_id", f"{inp}_T{trial.number:04d}")
        if candidate in occupied:
            continue
        if trial.state == State.RUNNING:
            study.tell(trial.number, state=State.FAIL)
        study.enqueue_trial(trial.params, user_attrs={"resume_candidate_id": candidate})
        occupied.add(candidate)
