# ADR 0010 · Shared input budget in MAC

## Decision

The ADA bundle selects `curvas.input_search.mode = "family_objective"`. The
library retains `per_input` for configurations that omit the option. The new
mode uses one study per family and objective. A trial with multiple eligible
inputs samples the alias as the categorical parameter `input`. Input features
come from the project's frozen calendar and capability checks.

Multi-input studies enqueue `startup_trials_per_input` trials per input
(default four). Startup slots are idempotent on resume. Pruning requires a
completed reference for the trial's input and encoder where applicable. An
interrupted trial gets its fixed and sampled parameters back, with the same
candidate ID once assigned. Single-input studies keep their existing strategy.

Ranking continues to use completed candidates and selects by account and target.
The audit stores the actual input per trial and the publication records the
configured search mode. Switching modes requires a new run identity.

## Budget

With 25 trials per family and objective, TiDE or NHITS alone gets 50 trials.
Both get 100. The five-family bundle gets 250 versus 550 in per-input mode.
These are planned search attempts, with three fold fits per unpruned trial.
Wall-clock speedup depends on pruning, models, instance and workloads.
