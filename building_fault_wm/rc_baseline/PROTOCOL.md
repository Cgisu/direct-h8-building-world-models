# Reviewer-Requested Resistance-Capacitance Comparator

## Status and scope

This physically structured comparator was added after the original neural,
autoregressive, and subspace results were known. It is descriptive and cannot
change a frozen category. Model development may read only the existing fitting
and development-validation roles. The source, selected models, analysis
contract, and immutable transport-package identity must be written into a new
readiness record before any response-unseen value is opened.

## Model class

The thermal core is a case-specific resistance-capacitance (RC) network. The
candidate topologies are 1R1C and 2R2C. Their discrete heat balances are

```text
Tz[k+1] = Tz[k]
          + a_oa (Tout[k+1] - Tz[k])
          + a_zm (Tm[k] - Tz[k])
          + b_q Qhvac[k+1]
          + b_solar G[k+1]
          + b_constant

Tm[k+1] = Tm[k]
          + a_mz (Tz[k] - Tm[k]).
```

All active conductance-derived coefficients and the delivered-heat gain are
strictly positive. Zone and mass coefficients are constrained below 0.3. The
2R2C network contains the outdoor--zone and zone--mass resistances; the 1R1C
candidate omits the mass node. For the two hydronic cases, `Qhvac` is the
recorded delivered heating power. For the air-system office, it is the sensible
supply-air heat calculated from flow, supply temperature, and zone
temperature. The constant heat-flow term is signed and represents unresolved
time-invariant gains or losses. The remaining coefficients are converted to
positive capacities and resistances and persisted with the selected model.
The delivered-heat gain is bounded between $10^{-5}$ and 1 K per kW per
15-minute step, corresponding to a zone capacity between approximately
$9\times10^5$ and $9\times10^{10}$ J/K. Effective solar aperture is bounded
between 0 and 10,000 m$^2$, and the signed constant heat flow between -1,000
and 1,000 kW. These broad bounds exclude degenerate parameter ratios without
encoding a case-specific building size.
Each active conductance-derived discrete coefficient is at least $10^{-5}$,
and the 2R2C mass-to-zone capacity ratio is bounded between 1 and 100. These
limits prevent a nominally active path from collapsing to zero or the latent
mass from becoming an effectively infinite reservoir.

The frozen corpus does not expose the internal controller or plant states
needed to forecast electric power, water flow, or supply conditions directly.
A one-step ridge-regularized equipment map therefore predicts electric power
and the two case-specific auxiliary channels from the current observation,
candidate action, known next-step context, and heating/cooling degree. Those
predictions drive the RC balance. This empirical equipment map is disclosed as
part of the grey-box model; the temperature dynamics themselves remain
physics-structured.

## Development-only selection

Each case uses all 20 clean fitting-role trajectories. RC parameters minimize
robust multi-step zone-temperature simulation error. The equipment map uses the
same fitting-only scalers as the frozen neural study. Candidate selection uses
only the existing development-validation fault grid:

The unobserved mass node is initialized to the measured zone temperature at
the beginning of every trajectory in both fitting and evaluation; no
trajectory-specific latent initial state is fitted.

```text
RC topology:                 1R1C, 2R2C
equipment regularization:   ordinary least squares, then 1e-4 through 1e4 by decades
observer innovation clip:   none, 3 sigma, 5 sigma
```

This gives 60 candidates per case and 180 in total. Ordinary least squares is
an explicit terminal endpoint: if it is selected, there is no unresolved
smaller regularization strength. Selection minimizes
eight-step standardized affected-channel mean absolute error, equally weighted
over bias, drift, and stuck families and the two faulted channels. Ties prefer
1R1C, then the larger ridge strength, then no clipping before the smaller
finite clipping threshold. The selected model is not refitted.

Candidate validation uses three fixed worker processes and one linear-algebra
thread per worker. Ordered result collection preserves the declared grid and
ranking order; process scheduling cannot alter a score or tie rule.

Each bounded thermal fit uses nine deterministic, topology-aware starting
points. The fit with the smallest robust fitting objective is retained; ties
use fewer function evaluations and then the earlier declared start. This
guards against relying on a single local nonlinear solution without adding a
random-search degree of freedom.

The selected-model record flags any physical parameter within 0.1% of a
declared bound span. Boundary contact is reported as an identifiability
diagnostic, not silently treated as a well-estimated physical property.

## Development chronology before the source lock

All events in this section used fitting and development-validation roles only;
no response-unseen transport value was read. An initial prototype was stopped
because its nominal 2R2C network contained an unintended outdoor-to-mass path.
The corrected topology retained only outdoor-to-zone and zone-to-mass paths.
Subsequent development checks rejected unconstrained parameterizations that
collapsed active paths or produced unbounded capacity ratios. The final broad
physical bounds above were then fixed. A nine-value finite ridge grid was
tested in development; one case selected its smallest finite value on a nearly
flat validation surface. Ordinary least squares was therefore added as the
mathematically terminal endpoint before the final source lock. These diagnostic
iterations are superseded and cannot assign a confirmation category.

## Filtering and open-loop prediction

An extended Kalman observer filters the corrupted history through each forecast
anchor. Its state contains standardized zone temperature, latent mass
temperature, electric power, and the two auxiliary equipment variables.
Candidate innovation clipping is applied componentwise in standardized
innovation units. Open-loop prediction then receives only the planned actions
and known future context; no future observation is used.

## Held-out evaluation and analysis

The selected model is evaluated once on the immutable response-unseen
transport corpus: three cases, 12 paired windows per case, two- and four-hour
action dwell, the three silent fault families, and horizons one, two, four, and
eight. The deterministic RC result is duplicated across neural-seed identities
only for exact pairing; this creates no artificial RC-model variation.

For neural arm `m` and policy `p`, the descriptive effect is

```text
R[m,p] = 1 - MAE(m,p) / MAE(RC,p),
```

so positive values favor the neural arm. The existing equal-weight reduction
and 10,000-draw case/seed/window hierarchical bootstrap are reused. A separate
reader must reconstruct the paired result from the persisted neural and RC
rows before a sealed report can be created.
