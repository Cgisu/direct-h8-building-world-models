"""Masked sequence objective for the health-aware RSSM.

The model filters corrupted measurements, but its reconstruction target is the
clean signal supplied by a simulator or other supervised data source. Each
target family has a separate mask so unavailable labels are never silently
treated as zeros. The output at index ``t`` is aligned with the state after
``previous_actions[t]`` and the current corrupted observation have been read.
"""

from dataclasses import dataclass, replace
from math import log, pi
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import HealthAwareRSSM, RSSMRollout, RSSMState


@dataclass(frozen=True)
class RSSMSequenceInputs:
    """Time-major causal inputs used by the posterior filter.

    All tensors have shape ``[time, batch, features]``. At index ``t``,
    ``previous_actions[t]`` is ``a_(t-1)``, ``contexts[t]`` is known ``c_t``,
    and the corrupted observation is ``y_t``. The observations may contain
    NaNs; finiteness is combined with ``availability`` before encoding.
    """

    previous_actions: Tensor
    corrupted_observations: Tensor
    availability: Tensor
    age: Tensor
    contexts: Optional[Tensor] = None


@dataclass(frozen=True)
class RSSMSequenceTargets:
    """Supervised targets and explicit masks for one observed sequence.

    ``valid_steps`` has shape ``[time, batch]`` and excludes padding. The
    remaining masks match their target tensors and are intersected with
    ``valid_steps``. Health labels use integer class indices; labels equal to
    ``RSSMLossConfig.health_ignore_index`` are ignored even when the health
    mask is true.
    """

    clean_observations: Tensor
    clean_observation_mask: Tensor
    costs: Tensor
    cost_mask: Tensor
    constraints: Tensor
    constraint_mask: Tensor
    continuations: Tensor
    continuation_mask: Tensor
    health_labels: Tensor
    health_mask: Tensor
    valid_steps: Tensor


@dataclass(frozen=True)
class RSSMLossConfig:
    """Weights and numerical settings for the sequence objective.

    ``kl_balance`` weights the dynamics term ``KL(stopgrad(q) || p)``; the
    remaining weight is assigned to ``KL(q || stopgrad(p))``. Free nats are
    applied per time step after summing over stochastic dimensions.
    """

    observation_weight: float = 1.0
    cost_weight: float = 1.0
    constraint_weight: float = 1.0
    continuation_weight: float = 1.0
    health_weight: float = 1.0
    kl_weight: float = 1.0
    kl_balance: float = 0.8
    kl_free_nats: float = 1.0
    health_ignore_index: int = -100
    health_class_weights: Optional[Tensor] = None
    minimum_std: float = 1e-6
    overshooting_horizon: int = 3
    overshooting_weight: float = 0.0

    def __post_init__(self) -> None:
        weights = (
            self.observation_weight,
            self.cost_weight,
            self.constraint_weight,
            self.continuation_weight,
            self.health_weight,
            self.kl_weight,
            self.overshooting_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be nonnegative")
        if not 0.0 <= self.kl_balance <= 1.0:
            raise ValueError("kl_balance must be in [0, 1]")
        if self.kl_free_nats < 0:
            raise ValueError("kl_free_nats must be nonnegative")
        if self.minimum_std <= 0:
            raise ValueError("minimum_std must be positive")
        if self.overshooting_horizon < 2:
            raise ValueError("overshooting_horizon must be at least 2")
        if self.health_class_weights is not None:
            if self.health_class_weights.ndim != 1:
                raise ValueError("health_class_weights must be one-dimensional")
            if not torch.isfinite(self.health_class_weights).all():
                raise ValueError("health_class_weights must be finite")
            if (self.health_class_weights < 0).any():
                raise ValueError("health_class_weights must be nonnegative")


@dataclass(frozen=True)
class RSSMLoss:
    """Scalar objective, unweighted components, and supervision counts."""

    total: Tensor
    observation_nll: Tensor
    cost_nll: Tensor
    constraint_bce: Tensor
    continuation_bce: Tensor
    health_ce: Tensor
    kl: Tensor
    dynamics_kl: Tensor
    representation_kl: Tensor
    overshooting_kl: Tensor
    valid_counts: dict[str, int]

@dataclass(frozen=True)
class RSSMTrainingOutput:
    """Observed latent rollout together with its training objective."""

    rollout: RSSMRollout
    loss: RSSMLoss

def _as_mask(mask: Tensor, expected_shape: torch.Size, name: str) -> Tensor:
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    if mask.dtype == torch.bool:
        return mask
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"{name} must contain only zero/one or boolean values")
    return mask.bool()


def _require_same_device(reference: Tensor, tensors: dict[str, Tensor]) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")


def _require_finite_where(target: Tensor, mask: Tensor, name: str) -> None:
    if mask.any() and not torch.isfinite(target[mask]).all():
        raise ValueError(f"{name} contains a non-finite supervised target")

def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_float = mask.to(values.dtype)
    count = mask_float.sum()
    if int(count.detach()) == 0:
        return values.sum() * 0.0
    return (values * mask_float).sum() / count

def _gaussian_nll(
    mean: Tensor,
    std: Tensor,
    target: Tensor,
    mask: Tensor,
    minimum_std: float,
) -> Tensor:
    safe_std = std.clamp_min(minimum_std)
    # Masked targets may legitimately be NaN. Replacing them before arithmetic
    # prevents NaN * 0 from contaminating the reduction.
    safe_target = torch.where(mask, target, mean.detach())
    normalized_error = (safe_target - mean) / safe_std
    nll = 0.5 * normalized_error.square() + safe_std.log() + 0.5 * log(2.0 * pi)
    return _masked_mean(nll, mask)

def _diagonal_kl(
    posterior_mean: Tensor,
    posterior_std: Tensor,
    prior_mean: Tensor,
    prior_std: Tensor,
) -> Tensor:
    """Return ``KL(q || p)`` summed over the final latent dimension."""
    posterior_var = posterior_std.square()
    prior_var = prior_std.square()
    return 0.5 * (
        (posterior_var + (posterior_mean - prior_mean).square()) / prior_var
        - 1.0
        + 2.0 * (prior_std.log() - posterior_std.log())
    ).sum(dim=-1)


def loss_from_rollout(
    rollout: RSSMRollout,
    targets: RSSMSequenceTargets,
    config: Optional[RSSMLossConfig] = None,
) -> RSSMLoss:
    """Compute one-step heads and KL for an already observed rollout.

    Multi-step overshooting requires the model plus aligned action/context
    inputs; use ``sequence_training_loss`` when its weight is nonzero.
    """
    config = RSSMLossConfig() if config is None else config
    if config.overshooting_weight > 0:
        raise ValueError(
            "overshooting requires sequence_training_loss with aligned inputs"
        )
    if rollout.posterior_mean is None or rollout.posterior_std is None:
        raise ValueError("training loss requires a posterior rollout from observe()")

    time_batch = rollout.observation_mean.shape[:2]
    expected = {
        "clean_observations": rollout.observation_mean.shape,
        "costs": rollout.cost_mean.shape,
        "constraints": rollout.constraint_logits.shape,
        "continuations": rollout.continuation_logit.shape,
        "health_labels": rollout.health_logits.shape[:-1],
    }
    values = {
        "clean_observations": targets.clean_observations,
        "costs": targets.costs,
        "constraints": targets.constraints,
        "continuations": targets.continuations,
        "health_labels": targets.health_labels,
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}")

    _require_same_device(
        rollout.observation_mean,
        {
            **values,
            "clean_observation_mask": targets.clean_observation_mask,
            "cost_mask": targets.cost_mask,
            "constraint_mask": targets.constraint_mask,
            "continuation_mask": targets.continuation_mask,
            "health_mask": targets.health_mask,
            "valid_steps": targets.valid_steps,
        },
    )
    valid_steps = _as_mask(targets.valid_steps, time_batch, "valid_steps")
    observation_mask = _as_mask(
        targets.clean_observation_mask,
        targets.clean_observations.shape,
        "clean_observation_mask",
    ) & valid_steps.unsqueeze(-1)
    cost_mask = _as_mask(targets.cost_mask, targets.costs.shape, "cost_mask")
    cost_mask = cost_mask & valid_steps.unsqueeze(-1)
    constraint_mask = _as_mask(
        targets.constraint_mask, targets.constraints.shape, "constraint_mask"
    ) & valid_steps.unsqueeze(-1)
    continuation_mask = _as_mask(
        targets.continuation_mask,
        targets.continuations.shape,
        "continuation_mask",
    ) & valid_steps.unsqueeze(-1)
    health_mask = _as_mask(
        targets.health_mask, targets.health_labels.shape, "health_mask"
    ) & valid_steps.unsqueeze(-1)

    _require_finite_where(
        targets.clean_observations, observation_mask, "clean_observations"
    )
    _require_finite_where(targets.costs, cost_mask, "costs")
    _require_finite_where(targets.constraints, constraint_mask, "constraints")
    _require_finite_where(
        targets.continuations, continuation_mask, "continuations"
    )
    for name, target, mask in (
        ("constraints", targets.constraints, constraint_mask),
        ("continuations", targets.continuations, continuation_mask),
    ):
        if mask.any() and not ((target[mask] >= 0) & (target[mask] <= 1)).all():
            raise ValueError(f"{name} targets must be in [0, 1]")

    observation_nll = _gaussian_nll(
        rollout.observation_mean,
        rollout.observation_std,
        targets.clean_observations,
        observation_mask,
        config.minimum_std,
    )
    cost_nll = _gaussian_nll(
        rollout.cost_mean,
        rollout.cost_std,
        targets.costs,
        cost_mask,
        config.minimum_std,
    )
    constraint_values = targets.constraints.to(rollout.constraint_logits.dtype)
    constraint_target = torch.where(
        constraint_mask, constraint_values, torch.zeros_like(constraint_values)
    )
    constraint_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(
            rollout.constraint_logits, constraint_target, reduction="none"
        ),
        constraint_mask,
    )
    continuation_values = targets.continuations.to(
        rollout.continuation_logit.dtype
    )
    continuation_target = torch.where(
        continuation_mask,
        continuation_values,
        torch.zeros_like(continuation_values),
    )
    continuation_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(
            rollout.continuation_logit, continuation_target, reduction="none"
        ),
        continuation_mask,
    )

    if targets.health_labels.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("health_labels must use an integer dtype")
    health_mask = health_mask & (
        targets.health_labels != config.health_ignore_index
    )
    classes = rollout.health_logits.shape[-1]
    if health_mask.any():
        supervised_labels = targets.health_labels[health_mask]
        if not ((supervised_labels >= 0) & (supervised_labels < classes)).all():
            raise ValueError("supervised health label is outside the class range")
    safe_health_labels = torch.where(
        health_mask,
        targets.health_labels.long(),
        torch.full_like(targets.health_labels, config.health_ignore_index).long(),
    )
    class_weights = config.health_class_weights
    if class_weights is not None:
        if class_weights.numel() != classes:
            raise ValueError("health_class_weights length must match health classes")
        class_weights = class_weights.to(
            device=rollout.health_logits.device, dtype=rollout.health_logits.dtype
        )
    health_losses = F.cross_entropy(
        rollout.health_logits.reshape(-1, classes),
        safe_health_labels.reshape(-1),
        weight=class_weights,
        ignore_index=config.health_ignore_index,
        reduction="none",
    ).reshape_as(safe_health_labels)
    if class_weights is None:
        health_ce = _masked_mean(health_losses, health_mask)
    else:
        weight_indices = torch.where(
            health_mask, safe_health_labels, torch.zeros_like(safe_health_labels)
        )
        normalizer = torch.where(
            health_mask,
            class_weights[weight_indices],
            torch.zeros_like(health_losses),
        ).sum()
        health_ce = (
            health_losses.sum() / normalizer
            if float(normalizer.detach()) > 0
            else health_losses.sum() * 0.0
        )

    dynamics_kl_values = _diagonal_kl(
        rollout.posterior_mean.detach(),
        rollout.posterior_std.detach(),
        rollout.prior_mean,
        rollout.prior_std,
    )
    representation_kl_values = _diagonal_kl(
        rollout.posterior_mean,
        rollout.posterior_std,
        rollout.prior_mean.detach(),
        rollout.prior_std.detach(),
    )
    dynamics_kl = _masked_mean(
        dynamics_kl_values.clamp_min(config.kl_free_nats), valid_steps
    )
    representation_kl = _masked_mean(
        representation_kl_values.clamp_min(config.kl_free_nats), valid_steps
    )
    kl = (
        config.kl_balance * dynamics_kl
        + (1.0 - config.kl_balance) * representation_kl
    )

    total = (
        config.observation_weight * observation_nll
        + config.cost_weight * cost_nll
        + config.constraint_weight * constraint_bce
        + config.continuation_weight * continuation_bce
        + config.health_weight * health_ce
        + config.kl_weight * kl
    )
    counts = {
        "steps": int(valid_steps.sum().detach()),
        "observations": int(observation_mask.sum().detach()),
        "costs": int(cost_mask.sum().detach()),
        "constraints": int(constraint_mask.sum().detach()),
        "continuations": int(continuation_mask.sum().detach()),
        "health": int(health_mask.sum().detach()),
        "overshooting_pairs": 0,
    }
    overshooting_kl = rollout.prior_mean.sum() * 0.0
    return RSSMLoss(
        total=total,
        observation_nll=observation_nll,
        cost_nll=cost_nll,
        constraint_bce=constraint_bce,
        continuation_bce=continuation_bce,
        health_ce=health_ce,
        kl=kl,
        dynamics_kl=dynamics_kl,
        representation_kl=representation_kl,
        overshooting_kl=overshooting_kl,
        valid_counts=counts,
    )

def _overshooting_dynamics_loss(
    model: HealthAwareRSSM,
    rollout: RSSMRollout,
    inputs: RSSMSequenceInputs,
    valid_steps: Tensor,
    config: RSSMLossConfig,
) -> tuple[Tensor, int]:
    """Match multi-step open-loop priors to stopped future posteriors.

    From posterior ``t``, transition to target index ``j`` uses
    ``previous_actions[j] = a_(j-1)`` and ``contexts[j] = c_j``. Distances start
    at two because the ordinary rollout KL already trains the one-step prior.
    A pair is supervised only when every step from source through target is
    valid, preventing an overshoot from crossing padding or episode boundaries.
    """
    if rollout.posterior_mean is None or rollout.posterior_std is None:
        raise ValueError("overshooting requires posterior states")
    time_steps, batch_size = rollout.deterministic.shape[:2]
    contexts = model._prepare_context_sequence(
        inputs.contexts,
        time_steps,
        batch_size,
        rollout.deterministic,
        name="contexts",
    )
    total = rollout.prior_mean.sum() * 0.0
    pair_count = 0
    maximum_distance = min(config.overshooting_horizon, time_steps - 1)
    if maximum_distance < 2:
        return total, pair_count

    for source in range(time_steps - 2):
        state = RSSMState(
            deterministic=rollout.deterministic[source].detach(),
            stochastic=rollout.stochastic[source].detach(),
        )
        path_valid = valid_steps[source].clone()
        final_target = min(time_steps - 1, source + maximum_distance)
        for target in range(source + 1, final_target + 1):
            path_valid = path_valid & valid_steps[target]
            step = model.transition(
                state,
                inputs.previous_actions[target],
                contexts[target],
                sample=False,
            )
            state = step.state
            if target - source < 2:
                continue
            pair_kl = _diagonal_kl(
                rollout.posterior_mean[target].detach(),
                rollout.posterior_std[target].detach(),
                step.prior_mean,
                step.prior_std,
            ).clamp_min(config.kl_free_nats)
            pair_mask = path_valid.to(pair_kl.dtype)
            total = total + (pair_kl * pair_mask).sum()
            pair_count += int(path_valid.sum().detach())

    if pair_count == 0:
        return total * 0.0, pair_count
    return total / pair_count, pair_count

def sequence_training_loss(
    model: HealthAwareRSSM,
    inputs: RSSMSequenceInputs,
    targets: RSSMSequenceTargets,
    config: Optional[RSSMLossConfig] = None,
    *,
    start_state: Optional[RSSMState] = None,
    sample: bool = True,
) -> RSSMTrainingOutput:
    """Filter a causal sequence and compute all world-model losses."""
    config = RSSMLossConfig() if config is None else config
    rollout = model.observe(
        inputs.previous_actions,
        inputs.corrupted_observations,
        inputs.availability,
        inputs.age,
        inputs.contexts,
        start_state=start_state,
        sample=sample,
    )
    base_loss = loss_from_rollout(
        rollout,
        targets,
        replace(config, overshooting_weight=0.0),
    )
    valid_steps = _as_mask(
        targets.valid_steps,
        rollout.observation_mean.shape[:2],
        "valid_steps",
    )
    if config.overshooting_weight > 0:
        overshooting_kl, pair_count = _overshooting_dynamics_loss(
            model, rollout, inputs, valid_steps, config
        )
    else:
        overshooting_kl = rollout.prior_mean.sum() * 0.0
        pair_count = 0
    counts = dict(base_loss.valid_counts)
    counts["overshooting_pairs"] = pair_count
    combined_loss = replace(
        base_loss,
        total=(
            base_loss.total
            + config.overshooting_weight * overshooting_kl
        ),
        overshooting_kl=overshooting_kl,
        valid_counts=counts,
    )
    return RSSMTrainingOutput(
        rollout=rollout,
        loss=combined_loss,
    )
