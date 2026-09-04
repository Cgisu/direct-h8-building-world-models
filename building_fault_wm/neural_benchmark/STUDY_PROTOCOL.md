# Multi-Case Sensor-Fault World-Model Study Protocol

Status: re-frozen after an outcome-blind repair to a void development attempt

## Void-attempt repair record

The first production development process ran under scientific-code manifest
`668f7b38214d9a67a61f3989a4d5c8bdb323e0f04e8cd17522a91af4b48e228a` and stopped
on 2026-07-22 at 08:37 Asia/Dubai after validation-based checkpoint and baseline
selection, but before consolidated baseline H8 evaluation, gate construction, or a
scientific decision. No validation score or error values were inspected. Structural
inspection showed that the GRU artifact correctly stored the amended 40-step history,
but its loader still expected the superseded literal value 8.

Before restarting, an outcome-blind source audit also found that RSSM evaluation used
the whole available trajectory prefix while the direct competence baselines used the
declared 40-step history. The corrected evaluator now gives every RSSM arm exactly the
same causal history window: observations `t-39..t`, prior actions `t-40..t-1`, and
known context `c[t-39..t]`. The versioned direct-H8 baseline contract receives those
same historical fields. Every arm then receives actions `a[t..t+7]` and known context
`c[t+1..t+8]` through the endpoint. This matches the maximum 40-step prefix among the
40 H8 source endpoints in each 48-step RSSM training sequence; the trained source
prefixes range from one through 40 steps.

The failed staging tree and hashes are retained in
`artifacts/development_v4_loader_incident_20260722.md`. None of its training,
validation, or baseline artifacts will be reused. The development screen is restarted
from the sealed public development corpus under a new complete scientific-code
manifest. The corpus, RSSM arms, seeds, RSSM training, losses, metrics, estimands,
thresholds, and gate logic are unchanged. The direct-H8 Ridge and GRU competence
baselines are refit under the versioned information-parity contract.

## Pre-analysis amendment record

The corpus split, public cases, trajectory plans, fault grid, RSSM arms, primary
estimand, and stop/go thresholds were fixed before FIT/validation collection. On
2026-07-22 at 03:15 Asia/Dubai (UTC+04:00), while deterministic FIT/validation
simulation was still running but before any development model fitting or validation
analysis, a source and protocol audit made these outcome-blind amendments:

- added crash-safe prelock preparation and one-shot confirmation transactions, source
  manifests, runtime receipts, and evidence recomputation;
- stated explicitly that both gates use the already configured 2,000 bootstrap draws;
- strengthened the direct-H8 Ridge/GRU history from 8 to 40 steps and raised GRU batch
  exposure from 4 to 160 endpoints per update to match the RSSM direct-H8 exposure;
- made any missing competence baseline return `INCOMPLETE` at development as well as
  confirmation.

At amendment time, no development checkpoint, baseline fit, validation prediction, or
gate result existed. The amendment was prompted by independent code/science audit, not
by an observed model outcome. This amended file is included in the scientific code
manifest and must not change after production development fitting begins.

## Question and claim boundary

The primary question is whether a causal pre-posterior reliability gate improves
eight-step, action-conditioned RSSM imagination when persistent building telemetry
faults corrupt the observations used to infer the current latent state.

The contribution is not a new RSSM family, a first sensor-fault method, or a first
building world model. It is a public, reproducible BOPTEST benchmark and controlled
comparison of where reliability intervention belongs inside an action-conditioned
latent world model. Any paper claim must be limited to the cases, actions, fault
families, horizons, and baselines evaluated here.

This is an open-loop world-model study, not a Dreamer or closed-loop control study.
There is no actor, value function, MPC optimizer, or control-benefit claim in the
primary experiment.

## Public source and cases

- Source: https://github.com/ibpsa/project1-boptest
- BOPTEST commit: `0f8a467cb1823f005b6512937e9333c65e1e483e`
- BOPTEST API: `1.0.0-dev`
- License: BOPTEST repository license at the pinned commit
- Cases:
  - `bestest_hydronic_heat_pump`, Brussels weather
  - `twozone_apartment_hydronic`, Milan weather
  - `multizone_office_simple_air`, Chicago weather

The clean corpus contains 40 non-overlapping 48-hour trajectories per case at a
15-minute control step. Weather-stratified whole-trajectory roles are 20 FIT, 8
validation, and 12 locked test. FIT and validation are generated first. Locked-test
trajectories must not be generated until implementation, hyperparameters, checkpoint
selection, fault grid, and evaluator hashes are frozen.

Each case is modeled separately. The four canonical observations are zone-temperature
aggregate, total HVAC electric power, and two case-specific auxiliary signals. The two
scored fault channels are aggregate zone temperature and aggregate HVAC power. They
are benchmark telemetry channels, not claims about individual physical sensors.
Known context contains outdoor temperature, global horizontal solar irradiance,
comfort lower and upper limits, and electricity price. The action is the normalized
supervisory setpoint perturbation. All scalers are fit on clean FIT trajectories for
one case only.

## World-model mechanism

The backbone is a compact recurrent state-space model with deterministic state `h_t`
and stochastic state `z_t`:

1. The transition prior advances `(h_(t-1), z_(t-1))` using previous action
   `a_(t-1)` and known context `c_t`.
2. The posterior filters current observation `y_t` into the latent belief.
3. Starting from that posterior, `imagine()` applies candidate future actions and
   future-known contexts without reading future observations.
4. The candidate gate computes a reliability probability from the stopped prior,
   prior prediction, current innovation, availability, and observation age before the
   posterior update. Health labels are training targets and are never inference inputs.
5. A direct H8 SmoothL1 loss trains the same observation-free endpoint used by the
   primary evaluation.

For state index `t`, H8 imagination uses actions `a_t ... a_(t+7)`, contexts
`c_(t+1) ... c_(t+8)`, and scores clean observation `y_(t+8)`. The source posterior is
not detached. A path is invalid if it crosses padding or an episode boundary.

## Fixed learned arms

All RSSM arms use byte-identical initialization per paired seed, the same FIT batches,
fault realizations, optimizer, update budget, and per-case scalers.

| Arm | Posterior measurement gate | Binary reliability CE | Direct H8 loss |
|---|---|---:|---:|
| `legacy` | bypass | 0 | 0 |
| `ungated_h8` | bypass | 0 | 1 |
| `aux_h8` | bypass | 0.25 | 1 |
| `gated_h8` | learned reliability | 0.25 | 1 |
| `huber_h8` | fixed causal innovation weight | 0 | 1 |

The gate target is binary: reliable versus faulty. Fault-family classification is a
secondary diagnostic, not the quantity controlling the posterior. The primary paired
comparison is `gated_h8` versus `ungated_h8`. The other RSSM contrasts isolate direct
rollout supervision, a trained-but-unused gate branch, and learned versus fixed robust
filtering. Because the gate features use a stopped prior, `aux_h8` is a negative
control: its RSSM core must remain bit-identical to `ungated_h8` under matched updates;
only its unused gate branch may differ.

## Competent non-RSSM baselines

- Causal persistence.
- Per-case standardized Ridge ARX with at least eight observation/action lags and
  validation-selected regularization.
- Direct-H8 Ridge using observations, availability, age, previous actions, and known
  context over the same 40-step causal history as RSSM evaluation, plus the same future
  actions and known context.
- A GRU using that same 40-step direct-H8 contract, the same update/checkpoint budget,
  and 160 endpoint examples per update, matching the four RSSM sequences times their 40
  valid H8 source states. It uses the same three development seeds and five confirmatory
  seeds as the RSSM arms.

All RSSM arms are evaluated from the same 40-step causal history supplied to the
direct-H8 Ridge and GRU. No evaluator consumes observations earlier than that window.

No world-model advantage may be claimed unless the RSSM backbone first beats
persistence and is competitive with the strongest validated ARX/GRU baseline. State
or one-step error is diagnostic only.

## Fault protocol

Persistent faults are applied only to the observations consumed by the filter; clean
simulator values remain targets. Families are healthy, bias, drift, stuck, and dropout.
Fault schedules, onsets, signs, severities, source hashes, and RNG seeds are
materialized before optimization. FIT, validation, and test use disjoint whole
trajectories and deterministic fault realizations. Scored H8 anchors require the fault
to remain active for the complete source-to-endpoint path.

Fault onset timing is also disjoint: FIT uses steps 32 and 96, validation uses 48 and
112, and locked test uses 64 and 128. This prevents a fixed-onset shortcut and makes
validation/test an onset-transfer evaluation.

Bias, drift, and stuck are the primary silent-fault families. Dropout is reported
separately because availability reveals it explicitly. Metrics are computed in both
raw physical units and FIT-standardized units.

## Development and checkpoint selection

Development may read FIT and validation only. Three paired model seeds are used for a
small feasibility gate. One common checkpoint update for all RSSM arms is selected on
validation data using the mean H8 score of `ungated_h8` and `gated_h8`; no arm selects
its own favorable update. Development results cannot support a paper claim.

Development returns `SCREEN_GO`, not a paper pass, only when all of these point checks
hold: at least 7.5% silent-fault improvement over `ungated_h8`; positive improvement in
all three cases and each silent family; at least two of three seeds positive; healthy
degradation at most 5%; both primary RSSMs beat persistence in every case; and gated
aggregate MAE is at most 1.15 times the strongest completed ARX/direct-Ridge/GRU
baseline. A development interval excluding zero is not required because three seeds
are a screening design. Otherwise the method returns `SCREEN_STOP`; missing evidence
returns `INCOMPLETE`.

The locked test is generated and opened exactly once after source, protocol,
implementation, model-config, scaler, fault-manifest, selected-update, checkpoint, and
evaluator hashes are recorded. The confirmatory experiment uses five paired seeds.

## Primary estimand and gate

For every valid cell, compute the H8 absolute error on the same channel that was
faulted. Standardize using FIT-only channel scale. Aggregate cells with equal weight
across case, silent fault family, sensor channel, sign, and severity. The paired raw
improvement is `MAE(ungated_h8) - MAE(gated_h8)`; relative improvement divides by the
ungated value. Use a paired hierarchical bootstrap over cases, seeds, and whole
trajectories. Report the point estimate and 95% interval.
Both the development screen and confirmatory gate use exactly 2,000 bootstrap draws
with the frozen bootstrap seed; the draw count is not changed after screening.

A positive primary result requires all of the following:

1. At least 10% lower equal-weight silent-fault H8 MAE.
2. The paired 95% interval for raw improvement is strictly above zero.
3. Positive point improvement in every BOPTEST case.
4. Positive point improvement for bias, drift, and stuck separately.
5. Positive point improvement at both frozen severities.
6. At least four of five paired seeds are positive.
7. Healthy H8 degradation is at most 5%, and its paired 95% upper bound is at most 5%.
8. `ungated_h8` and `gated_h8` each beat persistence in every case.
9. `gated_h8` improves over `aux_h8`, showing that intervention, not only auxiliary
   classifier training, matters.
10. `gated_h8` is no worse than `huber_h8` by more than 2%.
11. The predeclared `gated_h8` aggregate MAE is at most 1.05 times the strongest validated
    ARX/direct-Ridge/GRU baseline, the paired hierarchical 95% upper ratio is at most
    1.10, and the point ratio is at most 1.10 in every case.
12. On healthy cells, realized-action H8 MAE is lower than alternate-action H8 MAE,
    and mean action-induced prediction change is nonzero in every case.

As a predeclared action-use diagnostic, validation and locked evaluation also repeat
H8 imagination after replacing the eight realized future actions with the next
nonrepeating eight-step action block from the same trajectory. Evaluate this on
healthy cells only, equal-weighted across case, channel, trajectory, and seed, so the
diagnostic is not confounded by the reliability intervention. Report the mean
standardized prediction change and the change in H8 MAE. To describe the RSSM as
empirically action-conditioned, realized actions must have lower aggregate H8 MAE than
alternate actions and the mean prediction change must be nonzero in every case. This
diagnostic is not optimized or used for checkpoint selection.

The rule is conjunctive. A failed development method is stopped rather than tuned on
locked data.

Report raw-unit MAE separately for temperature and power, standardized MAE for pooled
comparisons, and a separate dropout table. Dropout is never pooled into the silent-
fault primary estimand because its availability flag reveals the fault directly.

## Publishability rule

The reliability-gate method is publishable only if the confirmatory primary gate is
met. A broader benchmark paper can remain viable after a method failure only if the
predeclared multi-case comparison produces a statistically stable, practically useful
cross-case finding across competent robust and non-robust baselines. One failed neural
model, one isolated positive cell, or a result dominated by an incompetent baseline is
not a paper.

Closed-loop comfort, energy, or cost claims require a separate post-confirmation
BOPTEST controller evaluation. Prediction results alone cannot support a control
benefit claim.

## Compute budget

The implementation targets CPU collection and a 12 GB GPU or Colab Pro for training.
The complete study must remain below 250 of the available 550 Colab compute units and
record simulator steps, optimizer updates, wall time, peak memory, and device for every
run.

## Closest-work positioning

The related-work audit must explicitly cover:

- Dreamer/RSSM observation filtering and latent imagination.
- Robust world models under visual distractors or corrupted observations.
- Trust/reliability-gated state-space models.
- SensorFault-Bench and other sensor-fault forecasting benchmarks.
- Sensor-fault-tolerant HVAC control and building fault detection/accommodation.
- BOPTEST surrogate modeling and model-predictive control.

The differentiator is the intersection of persistent telemetry faults, causal latent
belief intervention, action-conditioned observation-free imagination, and a public
multi-building BOPTEST comparison. Architecture-level novelty is not claimed.
