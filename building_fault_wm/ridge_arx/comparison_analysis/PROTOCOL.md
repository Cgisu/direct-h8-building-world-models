# Frozen Neural-versus-ARX Comparison Analysis

## Boundary

This is a separately frozen, secondary analysis of two immutable evaluator outputs:

- the direct-H8 v3 evaluator (`gate_core.csv`); and
- the schedule-matched recursive Ridge-ARX evaluator (`arx_core.csv`).

It cannot modify either upstream study, gate, category, or artifact. The analysis is
frozen before either v5 trajectory output is inspected. Its source and contract are
kept in a subpackage so that preparing it cannot change the already-running ARX
training source lock.

The analysis runner requires a revision-pinned public GitHub Gist receipt whose
remote bytes match the metadata-only local contract and source snapshot. It validates
that receipt before opening either upstream result CSV.

## Exact pairing

Both inputs must pass their complete upstream schemas. After retaining the silent
fault families (`bias`, `drift`, and `stuck`), every row is paired across all four
arms by:

`case, policy, window_id, trajectory_day, scenario_seed, trajectory_seed, model_seed,
cell_id, fault_channel, family, sign, severity, onset, anchor, horizon`.

Duplicates, missing rows, changed trajectory identities, extra arms, negative errors,
non-finite values, incomplete windows, or input-hash drift are fatal.

## Weighting and hierarchy

The H8 score follows the frozen v3 hierarchy. It first averages repeated
anchor/onset rows within each `case / policy / window / model-seed / family /
fault-channel / sign / severity / arm` cell. It then gives equal weight, in order,
to sign, severity, fault channel, and family. This leaves one paired score per case,
policy, window, model seed, and arm.

The 10,000-draw PCG64 bootstrap uses seed `202608029`. Cases, model seeds, and windows
within case are resampled hierarchically. The same draw indices are used for both
arms and both action-dwell policies.

## Estimand and decisions

The only inferential comparison is the deterministic recurrent world model
(`deterministic_wm`) against the schedule-matched recursive Ridge-ARX model at H8 on
silent faults. For each policy:

`effect = 1 - MAE(deterministic_wm) / MAE(ARX)`.

Positive values favor the neural model. `new_4h` is the primary policy; `old_2h` is a
predeclared persistence control.

The inherited symmetric categories are:

- neural advantage: point effect at least `+10%`, 95% interval above zero, positive
  in every case and family, and positive in at least four of five seeds;
- ARX advantage: point effect at most `-10%`, 95% interval below zero, negative in
  every case and family, and negative in at least four of five seeds;
- practical equivalence: the 90% interval is wholly inside `[-5%, +5%]`, and every
  case and family point effect is within `[-10%, +10%]`;
- otherwise: inconclusive.

The 10% rule is a category threshold plus evidence of direction; it is not a
confidence bound proving an effect greater than 10%. The 90% interval rule is the
two-one-sided-test form of equivalence at the fixed 5% margin.

Persistence across dwell is reported only when both policies receive the same
non-inconclusive category. The paired `new_4h - old_2h` effect and interval are also
reported.

## Descriptive-only outputs

H1, H2, and H4 results are descriptive. Comparisons of ARX with `legacy` and
`ungated_h8` RSSM arms are descriptive. They cannot support superiority,
non-inferiority, equivalence, or architecture claims.

No output supports claims about physical parameter identification, observed
buildings, closed-loop control, planning, MPC, energy, cost, comfort, or generic
architectural superiority.
