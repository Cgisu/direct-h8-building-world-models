# Strengthened Ridge-ARX Post-Outcome Robustness Analysis

## Scope

This is a post-outcome robustness analysis motivated by a visible limitation of
the original schedule-matched Ridge-ARX comparator: 14 of 15 fits selected the
largest available regularization value, alpha 100. It does not change the frozen
neural study, its gate, or the original eight-lag ARX result. Its conclusions are
descriptive sensitivity results.

## Development-only selection

For each of the same three cases and five schedule seeds, lag and Ridge alpha are
selected jointly from:

```text
lag:   4, 8, 16, 24, 40
alpha: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5
```

Every candidate reuses the exact 64,000 one-step source rows selected by the
parent training schedule. Changing lag changes only the causal history visible
at those source rows. It does not add source rows or alter their multiplicities.
The objective is the existing parent development-validation H8 standardized
affected-channel MAE, averaged equally across fault family and channel. Ties
choose the smaller lag and then the smaller alpha.

No locked-test trajectory is an admissible selection input. The selected model is
not refit after validation selection.

## Model and rollout

Each candidate is a multi-output linear Ridge-ARX model. Features contain causal
corrupted-observation, availability, `log1p(age)`, and previous-action lags;
current candidate action; and current plus next known context. Per-case FIT
scalers, schedule multiplicities, and equal fault-channel/family row weighting
are unchanged from the original comparator.

For early FIT sources that precede a candidate's full lag length, unavailable
pre-trajectory observation and action lags are left-padded with standardized
zeros and observation availability zero. This preserves the exact parent source
rows without introducing noncausal data. Validation and held-out anchors have
the complete 40-step history.

Evaluation is recursive. After the anchor, each prediction replaces the next
observation, with availability one and age zero. Future simulator observations
are never consumed. Candidate future actions and known contexts are supplied
under the same information contract as the original comparison.

## Source lock and held-out evaluation

Before this branch opens a held-out trajectory value, a readiness receipt binds
the code, fixed grids, selection rule, source-row identity, selected models,
development inputs, per-case scalers, parent schedules, transport metadata,
neural evaluation identity, output schema, and descriptive analysis contract.
The evaluation output is write-once, so the completed held-out pass cannot be
silently replaced.

An initial readiness namespace named a nonexistent `data/raw` directory. The
mistake was detected by a path-only preflight before an evaluation-attempt marker
or new held-out response access. That namespace is closed as terminal. The
replacement source lock corrects the path to `data/locked_transport_raw` and
recovers the 15 development-only selection runs byte-for-byte, with both source
and replacement identities recorded.

That replacement attempt then stopped after scoring the first case in memory:
the case-local evaluator incorrectly called a validator that required the full
three-case row count. It wrote no result CSV or completion, but it did access
first-case held-out values, so it is explicitly recorded as a failed partial
evaluation rather than described as outcome-blind or ignored. The final
replacement separates case-local and full-grid validation, binds the failed
attempt, and is the only completed held-out pass.

The same paired policies, windows, cases, seeds, silent fault families, anchors,
horizons, equal-weight aggregation, and hierarchical bootstrap used by the
original ARX comparison are reused. No confirmatory category is assigned.

## Permitted interpretation

The analysis asks whether the deterministic world-model advantage over the
original eight-lag Ridge-ARX survives a materially stronger validation-selected
linear baseline. It cannot support generic architectural superiority, observed
building generalization, physical-parameter recovery, closed-loop control,
planning, energy, comfort, or cost claims.
