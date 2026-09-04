# Health-aware RSSM core

This directory contains model, training, and planning infrastructure, not an
experimental result. It must not be used to claim that fault-aware world-model
control works before the preregistered BOPTEST feasibility gate and decision
experiments pass.

This code does not establish a novel building world model or a novel MPC
algorithm. The implemented candidate is a **health-supervised shared-latent
RSSM**: physical and sensor-health information may share one recurrent state,
and health is decoded by an auxiliary supervised head. It is not a factorized
physical-state/health-state architecture. The intended later question is
whether that health supervision improves control under telemetry faults. Any
building study must compare against strong physics-structured or
forced-response dynamics baselines when implementations are available, in
addition to learned baselines.

## Why this is a world model

The model has two distinct inference paths:

1. **Filtering:** `observe_step` turns a corrupted measurement, its availability
   mask, age, previous action, and current known context into a latent belief.
2. **Imagination:** `imagine` advances that belief under candidate future actions
   and fixed known future contexts. Its API accepts no future observations.

The latent prior predicts future observations, cost, constraint events,
continuation, and per-sensor health. A planner can therefore compare action
sequences inside the model. This action-conditioned open-loop transition is the
world-model component; a sequence imputer or health classifier alone would not
qualify.

The deterministic recurrent state stores history. The stochastic shared state
can encode uncertain physical and sensor-health information, but the code does
not assign separate latent coordinates or transition processes to them. During
training, the posterior sees the current corrupted observation. During
planning, only the prior is available:

```text
previous belief + action + known context -> recurrent state -> latent prior
                                               + current observation -> posterior
```

### Timing contract

The observed sequence at index `t` is aligned as follows:

```text
posterior_t = filter(previous_belief, a_(t-1), c_t, y_t)
```

Starting from `posterior_t`, imagined index zero is:

```text
prediction_(t+1) = transition(posterior_t, a_t, c_(t+1))
```

Context is known but not optimized. It can include weather, solar gain,
occupancy, tariff, time encoding, and comfort schedule. It must not include
future measured building state. `context_dim=0` keeps the context-free API
compatible; otherwise context tensors are required.

## Tensor contract

- Observation, mask, age: `[batch, observation_dim]`
- Action: `[batch, action_dim]`
- Context: `[batch, context_dim]`
- Deterministic state: `[batch, deterministic_dim]`
- Stochastic state: `[batch, stochastic_dim]`
- Sequence inputs and rollout outputs: time-major `[time, batch, ...]`
- Health logits: `[batch, sensor_dim, health_classes]`

Masks use `1` for observed and `0` for unavailable. Missing values are removed
from the value path. Finiteness is combined with availability, so a NaN or
infinite observation is encoded as unavailable even if its external mask is
one. Age must be finite and nonnegative and is log-transformed before encoding.

`BeliefController` consumes the filtered deterministic and stochastic state
without planning. Comparing it directly with `LatentMPC` is a useful broader
controller comparison:

- belief-only controller: `BeliefController(filtered_state)`;
- latent MPC: candidate actions scored through `imagine(...)`.

It does **not** isolate the value of imagined training data because the policy
optimization algorithms differ. The stricter ablation requires matched
actor/critic architecture, features, real-data updates, and update budget, with
the only difference being additional actor/critic updates from imagined
transitions. That matched real-only versus real-plus-imagined actor/critic is not
implemented in this directory.

The matched health-supervision ablation must instantiate the same class and
configuration, retain every decoder (including the health head), and keep the
optimizer and update budget fixed. The baseline sets `health_weight=0` and is
never given health labels; the candidate changes only that loss weight and label
access. At deployment both receive identical corrupted values, availability,
age, context, and MPC. This supports a narrow auxiliary-supervision claim if it
wins, not a latent-factorization claim.

## Training objective

`training.py` distinguishes causal filter inputs from supervised targets:

- inputs: previous actions, current known contexts, corrupted observations,
  availability, and age;
- targets: clean observations, cost, constraint events, continuation, and
  per-sensor health class;
- masks: a padding mask plus a separate mask for every target family.

The objective combines Gaussian NLL for clean observations and cost, binary
cross-entropy for constraints and continuation, class-weighted health
cross-entropy with an ignore label, and a free-nats balanced KL. The balanced KL
uses `KL(stopgrad(posterior) || prior)` to train latent dynamics and
`KL(posterior || stopgrad(prior))` to regularize the representation.

An optional latent overshooting term starts from a stopped posterior, rolls the
prior open loop with aligned future actions and contexts, and matches stopped
future posteriors at distances 2 through `overshooting_horizon`. A pair is used
only if every intervening sequence step is valid. Set `overshooting_weight > 0`
through `sequence_training_loss(...)`; `loss_from_rollout(...)` cannot compute
overshooting because it does not receive aligned action/context inputs.
Overshooting is disabled by default (`overshooting_weight=0`) and must be an
explicitly logged experimental choice.

`sequence_training_loss(...)` runs `observe(...)` and returns both the latent
rollout and all scalar components. Masked NaN targets are safe; a non-finite
target under an active supervision mask raises an error.

## Latent MPC

`planning.py` implements bounded cross-entropy-method MPC. It ranks action
sequences using discounted predicted cost, constraint probability, decoded cost
uncertainty, and decoded observation uncertainty. It returns the first action
for receding-horizon execution together with the winning sequence and score
breakdown.

The planner receives an `RSSMState` plus `[horizon, batch, context_dim]` known
future contexts and calls `imagine(...)`. CEM expands each batch member's
context path unchanged across candidates and samples only actions. It has no
future-observation argument and runs with gradients disabled, so candidate
ranking cannot depend on future measurements through the planning API.

## Scope and compute

The default dimensions are deliberately small (`128` deterministic, `32`
stochastic). The module depends only on PyTorch, supports CPU execution, and is
well inside a 12 GB GPU budget. No BOPTEST loader, policy claim, or experimental
result is included here.

Run the synthetic invariant tests from the repository root:

```bash
.venv/bin/python -m unittest discover \
  -s building_fault_wm/recurrent_models -p 'test*.py' -v
```

The tests verify tensor contracts, timing/impulse alignment, context-dependent
priors, missing-value validation, target masking, balanced and overshooting
gradient flow, fixed-context bounded MPC, and an engineered case where different
imagined action consequences produce different selected actions.
