# Reviewer-Motivated Subspace State-Space Comparison

## Status and scope

This comparison was added after the original neural and Ridge-ARX results were
known, in response to supervisory review. It is a descriptive, post-outcome
comparison and cannot change either frozen neural category. Development-only
prototyping may inspect fitting and validation trajectories, but the model grid,
selection rule, analysis, code identity, and output contract must be bound
before the new comparator opens a transport response value.

## Model class

The comparator is a case-specific linear state-space model identified by a
multi-experiment past-output multivariable output-error subspace procedure. The
four standardized observations are the outputs. The input contains the
standardized action and five known next-step context variables. Each fitting
trajectory contributes its own block-Hankel columns, so no artificial state
transition is introduced between trajectories.

The model has the innovation form

```text
x[k+1] = A x[k] + B u[k] + w[k]
y[k+1] = C x[k] + D u[k] + v[k].
```

State and output residuals estimate the joint process/measurement covariance.
At evaluation, a Kalman observer filters the corrupted history through the
anchor, after which the model rolls forward without future observations.
Candidate innovation clipping is applied componentwise in standardized
innovation units and is selected only on development validation data.

## Development-only selection

For each simulator case, one deterministic model is selected from:

```text
block rows:       8, 12, 16
state order:      2, 4, 6, 8, 12, 16
innovation clip:  none, 3 sigma, 5 sigma
```

Every candidate uses all clean fitting-role trajectories for that case and the
same fitting-role scalers as the neural study. A candidate is inadmissible if
the state-transition spectral radius exceeds one or if any fitted quantity is
non-finite. Selection minimizes eight-step standardized affected-channel mean
absolute error on the existing development-validation fault grid, with equal
weight across fault family and channel. Ties choose fewer block rows, lower
state order, and then no clipping before the smaller finite clipping threshold.
The selected model is not refitted after validation.

The first development-selection namespace stopped before candidate scoring
because its shape guard expected 576 rather than the sealed 192-sample raw
trajectory length. It accessed fitting-role values only and produced no model,
validation score, readiness record, or held-out result. That namespace is
terminal; the replacement changes only the shape guard and retains the grid and
selection rule above.

The second development-only namespace was stopped during validation scoring
after profiling showed that identical time-varying Kalman gains were being
recomputed for every fault variant. It wrote no candidate score or model and
never opened transport responses. The replacement caches the mathematically
identical gain sequence once per candidate; no model, score, grid, or tie rule
changes.

The third development-only namespace completed candidate scoring, but the
pre-readiness audit found that finite clipping thresholds had been applied in
standardized-output units rather than the specified innovation-standard-
deviation units. That selection is invalid and terminal. It opened no transport
responses. The replacement corrects only this normalization and reruns the
unchanged candidate grid from development data.

The fourth development selection is numerically valid but was superseded before
readiness to strengthen the automated source-lock verifier and add an external-
implementation equivalence test. The replacement reruns the unchanged grid;
the prior namespace opened no transport response.

The fifth development selection reproduced the fourth selection byte-for-byte,
but its hardened verifier compared JSON lists with in-memory tuples directly
and rejected the otherwise matching configuration. The final replacement uses
canonical payload hashes for that check and again runs the unchanged grid. No
transport response had been opened.

The sixth development selection passed its hardened verifier, but the first
readiness attempt exposed a metadata-only dependency on the mutable live copy
of a historical, terminal collection-failure namespace. No readiness directory
was created and no trajectory number was interpreted. The final replacement
binds and reads the immutable, independently verified v6 publication package
instead; the model grid and development selection are unchanged.

The seventh selection used the immutable-package adapter, but an immediate
reproduction check found that the host's multithreaded linear-algebra reduction
could change a near-boundary candidate and the selected order. Its readiness is
terminal and no held-out evaluation was run. The final replacement fixes the
BLAS thread count at one, records the numerical-library versions, adds a
cross-thread-limit regression test, and reruns the unchanged development grid.

## Held-out evaluation

The selected model is evaluated on the existing hash-sealed transport
collection: three cases, 12 paired response-unseen windows per case, two-hour
and four-hour action dwell, the three silent fault families, and horizons one,
two, four, and eight. The comparator is deterministic; its prediction is paired
with each of the five neural seeds so that uncertainty from neural fitting,
case, and response-unseen window is retained without inventing comparator seed
variation.

The descriptive effects are

```text
1 - MAE(neural arm) / MAE(subspace state-space model),
```

so positive values favor the neural arm. Aggregation and the hierarchical
case/model-seed/window bootstrap match the existing secondary comparison.

## Evidence and interpretation

Before numerical held-out parsing, a readiness record binds this protocol,
implementation, configuration, selected model matrices, development source
identities, frozen scalers, the canonical v6 publication-package digest,
transport manifest, collection records, neural evaluation, output schema, and
analysis settings. The package verifier checks the read-only inventory and
file hashes before the record is written. The held-out namespace is write-once.
A separate auditor rechecks row identities and recomputes every reported
comparison from the raw per-row errors.

This comparison tests one linear time-invariant subspace model class. It does
not identify physical resistance or capacitance parameters and is not an RC or
physics-informed model. It does not establish controller, energy, comfort,
safety, economic, occupied-building, or general model-class superiority.
