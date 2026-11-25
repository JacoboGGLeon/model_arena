"""Retry interrupted fits with fixed parameters and stable candidate identity."""

import optuna


def enqueue_interrupted(study, inp, *, pooled=False):
    all_trials = study.get_trials()
    State = optuna.trial.TrialState
    live = (State.WAITING, State.COMPLETE, State.PRUNED)
    occupied = {
        t.user_attrs.get("resume_candidate_id", t.user_attrs.get("candidate_id"))
        for t in all_trials if t.state in live
    }
    queued = {
        t.user_attrs.get("resume_trial_number") for t in all_trials if t.state in live
    }
    for trial in reversed(all_trials):
        retry = trial.state == State.RUNNING or (
            trial.state == State.FAIL and trial.user_attrs.get("retry_after_disk", False)
        )
        if not retry:
            continue
        number = trial.user_attrs.get("resume_trial_number", trial.number)
        candidate = trial.user_attrs.get("candidate_id", trial.user_attrs.get("resume_candidate_id"))
        if candidate is None and not pooled:
            candidate = f"{inp}_T{trial.number:04d}"
        if number in queued or (candidate is not None and candidate in occupied):
            continue
        if trial.state == State.RUNNING:
            study.tell(trial.number, state=State.FAIL)
        attrs = {"resume_trial_number": number}
        if candidate is not None:
            attrs["resume_candidate_id"] = candidate
            occupied.add(candidate)
        study.enqueue_trial({**trial.system_attrs.get("fixed_params", {}), **trial.params}, user_attrs=attrs)
        queued.add(number)
