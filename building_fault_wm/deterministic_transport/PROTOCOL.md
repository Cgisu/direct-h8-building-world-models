# Long-Horizon Supervision and Action-Persistence Transport in Building World Models

## Status

This document defines a prospective, additive extension to the completed direct-H8
study. The parent v2 result and publication artifact remain immutable. V3 neither
recalculates nor replaces the parent `FALLBACK_CONFIRM` verdict.

The extension answers two reviewer-facing questions:

1. Does the direct-H8 benefit within the frozen Dreamer-style RSSM transport to
   previously uncollected weather windows and a longer action-dwell policy?
2. Under matched source sequences, optimizer budget, rollout semantics, active
   parameter count, and future information, is stochastic latent state useful
   relative to a deterministic recurrent world model?

This is an applied world-model architecture and transport study. It is not a
complete Dreamer agent, an RL experiment, or a closed-loop control claim.

The deterministic comparator grid was trained on development data before the
external outcome lock. The executable training files remained byte-identical
after training. Response-blind protocol clarifications made before that lock
expanded the prior-identity certificate and post-lock reporting contract; the
pre-lock registry records both the training-time and final protocol hashes.
The final comparator grid is trained only after its complete 11-file shared
runtime dependency closure is hashed. The grid receipt records that manifest,
the trainer checks it before and after all 15 runs, and pre-lock, collection,
and evaluation validation require the live shared files to match the copied
bundle byte-for-byte. Before the first model is trained, an exclusive root-level
source lock binds the parent package, v3 training files, shared dependency
manifest, and frozen configuration; a partial grid cannot resume under different
bytes.

## Immutable parent

The parent is `artifacts/direct_h8_publication_v2`, canonical package digest:

```text
b758859c6cb99d34930452c36e3fd59b5abd0e7f56b19710fa2b1998b23760b8
```

Before and after every v3 stage, the complete parent package inventory, file sizes,
types, and SHA-256 values must be identical. No file in a v1 or v2 scientific
namespace may be edited.

The following update-400 RSSM checkpoints are reused byte-for-byte for all three
cases and model seeds `202608011..202608015`:

- `legacy_u0400.pt`
- `ungated_h8_u0400.pt`

No RSSM retraining, checkpoint reselection, or recalibration is permitted.

## Public simulator and cases

The data source is the public IBPSA Project 1 BOPTEST repository at commit
`0f8a467cb1823f005b6512937e9333c65e1e483e`, using the already pinned worker
image and the following public cases:

- `bestest_hydronic_heat_pump`
- `multizone_office_simple_air`
- `twozone_apartment_hydronic`

All observations remain simulator-generated. Faults are synthetic. The three cases
do not constitute an external building population.

## Prospective windows

The candidate pool for each case is the 28 entries labelled `fit` or `validation`
in the v2 full plan. Those entries were planned from public weather metadata but
were never collected. V2 collected only its 12 `locked_test` entries.

`V3_PLAN_SEED = 202607312`.

Selection is response-value blind:

1. Within each of the five frozen temperature strata, rank candidates by ascending
   SHA-256 of
   `v3-window:202607312:{case}:{stratum}:{day}:{source_role}:{parent_seed}` and
   select two.
2. Rank all remaining candidates by the analogous string prefixed `v3-extra`.
   Select its first entry and then the first entry from a different stratum.

This produces 12 windows per case, with at least two from every temperature
stratum. The old role is retained as `source_plan_role`; v3 assigns
`locked_transport`.

Before external freeze, a certificate must prove:

- none of the selected day intervals was collected in v2;
- every selected one-day-warmup plus two-day-score interval is disjoint from every
  collected v1 FIT, validation, and locked interval;
- every selected interval is disjoint from every v2 locked interval;
- no selected v3 identity occurs in a known raw manifest, receipt, or CSV;
- no trajectory values were read to select a window.

The certificate is built before any `data_v6` response destination exists. Its
prior-evidence scope is every local multicase namespace from `data` through
`data_v5` (including an explicit absence record for `data_v1`) plus raw CSVs,
manifests, receipts, and collection records in the byte-verified immutable v2
publication package. For every evidence file it records the byte count, SHA-256,
and normalized `(case, day, trajectory_seed)` identities extracted from that
file. The embedded inventory, unique identity set, unique seed set, and v2
locked-CSV subset are independently hashed and self-validated. Construction
fails if a selected v3 case/day, full identity, or identity seed occurs anywhere
in that prior evidence, or if the canonical v2 CSV identities differ from the
v2 locked plan. `build_plan_artifacts.py` writes the three plans and certificate
with exclusive creation and refuses an existing `data_v6` destination.

These are prospectively collected, response-value-unseen temporal holdouts. They
are not a new climate year, an external dataset, or an independently conceived
replication.

## Replacement after an entrypoint failure

The first externally frozen readiness digest
`e5b7a6dc690b88f01c9a343c74a0e2cca2b21cde6103569e273a11d16dc48433`
is terminal. Its pinned BOPTEST container failed while Python imported the v3
package because that image does not contain PyTorch. The worker module did not
load, no simulator was initialized, the staging tree remained empty, and zero
trajectory CSVs or response values were created. The attempt and failure markers,
log, original public freeze, and their hashes are retained.

The byte-bound closeout is
`artifacts/direct_h8_deterministic_transport_v3_failed_attempt_closeout_v1.json`
(SHA-256
`578e95a82204eb92f963ad8d51a5b1914b7fd6ed394a39ded89c8d9034cf4a24`).
The replacement uses the same still-response-unseen plan in a new `data_v7`
output namespace and a new state namespace. It runs the lightweight worker as a
direct script and requires that exact entrypoint to pass inside the pinned
container before a new external freeze. The failed readiness digest is never
retried.

## Paired action policies

Each selected window is initialized twice with identical case, start time, one-day
warmup, BOPTEST scenario seed, weather, price realization, and initial state.

`old_2h` uses 24 balanced two-hour blocks: eight blocks at each action level
`{-1, 0, 1}`, expanded to eight 15-minute steps per block.

`new_4h` uses 12 balanced four-hour blocks: four blocks at each action level,
expanded to sixteen 15-minute steps per block. Adjacent blocks cannot repeat.

Thus both branches contain exactly 64 steps at every action level. The intervention
changes dwell time and transition frequency, not the action alphabet or marginal
action counts.

Before advancing either branch, the collector must hash the complete initialized
state and exogenous forecast. The two hashes must match exactly. Both action arrays,
their transition matrices, dwell distributions, and SHA-256 values are frozen in
the v3 plans.

The paired `old_2h` branch is mandatory. Comparing a new four-hour trajectory with
an old trajectory from another day would confound policy with weather and initial
state.

## Deterministic recurrent world model

The comparator is a recurrent state-space predictor, not the old endpoint-only
GRU.

Filtering input at time `t` is:

```text
corrupted observation (4)
availability mask (4)
log1p sensor age (4)
previous action (1)
known context (5)
```

The model uses one `GRUCell(input=18, hidden=64)` and an observation decoder
`Linear(64,53) -> SiLU -> Linear(53,4)`. It has 19,789 trainable parameters. The
frozen RSSM has 19,784 active observation-dynamics parameters and 38,072 total
parameters including unused auxiliary heads.

`filter()` consumes observations. `imagine()` receives only future actions and
known future contexts; future observations cannot enter the API. During recursive
imagination, the model feeds back its own decoded current observation with zero age.
That value is a model prediction, never a future simulator measurement.

Training reuses each paired RSSM's frozen schedule byte-for-byte:

- 400 updates;
- four 48-step fault-augmented sequences per update;
- the same FIT variants, scalers, cell IDs, aligned starts, and model seeds;
- all 40 valid H8 source states per sequence, or 160 H8 endpoints per update;
- filtered clean-observation reconstruction Smooth-L1 plus direct-H8 Smooth-L1,
  both weight 1;
- `beta=1`, Adam at `3e-4`, default Adam parameters, zero weight decay;
- gradient norm clip 100;
- deterministic PyTorch algorithms.

Update 400 is fixed. Checkpoints at 100, 200, 300, and 400 are retained only to
show learning curves. There is no early stopping, validation selection, learning
rate sweep, or rerun selection. The architecture claim is limited to this frozen
training budget, not universal optimizer superiority.

## Evaluation

The byte-identical v2 fault specification is applied independently to both policy
branches. Primary errors use the affected sensor channel and v1 FIT scalers.

At evaluation, every model:

- filters the same 40 corrupted history steps;
- receives the same future actions and contexts;
- receives no future observations;
- recursively predicts H1, H2, H4, and H8;
- runs deterministically (`sample=False` for the RSSM).

Primary scope is standardized H8 MAE for silent persistent faults
`{bias, drift, stuck}`. Healthy, dropout, H1/H2/H4, persistence, alternate-action
sensitivity, within-dwell windows, and dwell-boundary-crossing windows are
prespecified diagnostics. A rollout is boundary-crossing when the action changes
between the step immediately before its first imagined action and that action, or
between any two actions inside the imagined block. Because every frozen fault
anchor starts a new `old_2h` action block, that policy has no within-dwell H8
stratum; structurally unavailable diagnostic cells are reported as absent and are
not imputed or replaced with newly selected anchors.

Raw temperature error in K and power error in W are reported separately and are
never averaged together. Parameter count, optimizer updates, training time,
inference time, peak memory, and serialized model size are reported.

## Estimands

For policy `p`:

```text
A_p = 1 - MAE(ungated_h8 RSSM, p) / MAE(deterministic world model, p)
D_p = 1 - MAE(ungated_h8 RSSM, p) / MAE(legacy RSSM, p)
```

Positive `A_p` favors the RSSM. Negative `A_p` favors the deterministic world
model. Positive `D_p` favors direct-H8 supervision.

`A_new_4h` and `D_new_4h` are primary. `A_old_2h`, `D_old_2h`, and paired
four-hour-minus-two-hour differences are transport controls.

Errors are averaged over anchors and onsets first, then sign, severity, fault
channel, family, whole window, model seed, and case with equal weight.

## Paired inference

Use 10,000 percentile bootstrap draws with `PCG64(202608029)`. Each draw resamples:

- three cases with replacement;
- five paired model-seed indices with replacement;
- twelve whole windows within each selected case with replacement.

The two policy branches of a window remain paired. Identical resample indices are
used for all models and contrasts. All fault cells and anchors remain inside their
whole-window cluster. Ratios are computed inside each draw.

The persisted core CSV and gate JSON must also pass a separate implementation in
`independent_verify.py`. That verifier cannot import the primary gate analysis or
bootstrap helpers. It independently validates the arm and paired-policy grids,
repeats the equal-weight reductions and fixed `PCG64` bootstrap, reconstructs both
estimands and dwell-persistence decisions, and fails if any JSON field differs.
Only floating values may use a documented absolute and relative comparison tolerance
of `1e-12` to accommodate algebraically equivalent reduction order.

## Symmetric decision categories

The practical dominance margin is 10%; the equivalence margin is 5%.

For each primary estimand:

- `POSITIVE_ADVANTAGE`: point at least `+0.10`, 95% CI lower bound above zero,
  positive in every case and silent-fault family, positive in at least four of
  five seeds.
- `NEGATIVE_ADVANTAGE`: point at most `-0.10`, 95% CI upper bound below zero,
  negative in every case and silent-fault family, negative in at least four of
  five seeds.
- `PRACTICAL_EQUIVALENCE`: the complete 90% CI is inside `[-0.05,+0.05]`, and
  every case and family point is inside `[-0.10,+0.10]`.
- `INCONCLUSIVE`: every other outcome.

For `A`, the directional labels are `RSSM_ADVANTAGE` and
`DETERMINISTIC_WM_ADVANTAGE`. For `D`, they are `H8_BENEFIT` and `H8_HARM`.

`PERSISTENT_ACROSS_DWELL` is added only when the same non-inconclusive category
holds under both policies. The two-hour branch cannot rescue a failed or
inconclusive four-hour primary result.

All categories, case effects, family effects, seed effects, intervals, and
diagnostics are published unchanged.

## Lock and failure rules

Before external freeze, implementation fixes and synthetic smoke tests are allowed;
v3 simulator response values must remain unopened. The external record binds the
protocol, source code, runtime, plans, action arrays, parent artifact, parent
schedules, scalers, RSSM checkpoints, deterministic-model checkpoints, fault
contract, metrics, bootstrap, and gate.

After the collection-attempt marker, models, candidates, metrics, thresholds,
weights, seeds, checkpoints, and code cannot change. Any logged collection
failure is terminal for that frozen readiness digest; there is no retry path.
A nonfinite prediction is a model failure, not retryable infrastructure.

Any semantic defect discovered after outcome access invalidates that v3 digest.
Previously opened windows cannot be reused under a replacement protocol.

## Claim boundary and venue

The strongest permitted claim is an exposure- and rollout-matched comparison of
two frozen world-model pipelines under persistent synthetic sensor faults and a
paired action-dwell shift in three public building simulators.

V3 does not establish intrinsic stochastic-versus-deterministic superiority across
all architectures, observed-building validity, or improved energy/comfort control.
The intended venue is an applied engineering-AI journal; acceptance and current
quartile cannot be guaranteed by an experiment.
