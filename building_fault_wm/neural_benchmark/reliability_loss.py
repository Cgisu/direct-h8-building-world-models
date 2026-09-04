"""Rollout-aligned training loss for the reliability-gated RSSM."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from building_fault_wm.recurrent_models.model import RSSMState
from building_fault_wm.recurrent_models.training import (
    RSSMLoss,
    RSSMLossConfig,
    RSSMSequenceInputs,
    RSSMSequenceTargets,
    _overshooting_dynamics_loss,
    loss_from_rollout,
)

from .reliability_model import (
    GateMode,
    ReliabilityGatedRSSM,
    ReliabilityRollout,
)


@dataclass(frozen=True)
class ReliabilityLossConfig(RSSMLossConfig):
    """Ordinary RSSM settings plus direct H8 observation supervision."""

    direct_horizon: int = 8
    direct_horizon_weight: float = 1.0
    direct_horizon_beta: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.direct_horizon != 8:
            raise ValueError("the multi-case benchmark direct horizon must be H8")
        if self.direct_horizon_weight < 0:
            raise ValueError("direct_horizon_weight must be nonnegative")
        if self.direct_horizon_beta <= 0:
            raise ValueError("direct_horizon_beta must be positive")


@dataclass(frozen=True)
class ReliabilityLoss:
    """Combined loss with its unweighted rollout-aligned components."""

    total: Tensor
    ordinary: RSSMLoss
    direct_horizon_smooth_l1: Tensor
    latent_overshooting_kl: Tensor
    valid_counts: dict[str, int]


@dataclass(frozen=True)
class ReliabilityTrainingOutput:
    """Causal filtered rollout and its complete training objective."""

    rollout: ReliabilityRollout
    loss: ReliabilityLoss


def _as_mask(mask: Tensor, expected_shape: torch.Size, name: str) -> Tensor:
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    if mask.dtype == torch.bool:
        return mask
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"{name} must contain only zero/one or boolean values")
    return mask.bool()


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_value = mask.to(values.dtype)
    count = mask_value.sum()
    if int(count.detach()) == 0:
        return values.sum() * 0.0
    return (values * mask_value).sum() / count


def direct_h8_observation_loss(
    model: ReliabilityGatedRSSM,
    rollout: ReliabilityRollout,
    inputs: RSSMSequenceInputs,
    targets: RSSMSequenceTargets,
    config: ReliabilityLossConfig,
) -> tuple[Tensor, int, int]:
    """Supervise exactly the observation-free endpoint at ``t + 8``.

    A posterior at sequence index ``t`` consumes the current observation. Its
    first imagined transition consumes ``previous_actions[t + 1]`` and
    ``contexts[t + 1]``; eight transitions therefore target the clean
    observation at index ``t + 8``. No source state is detached.
    """
    rssm = rollout.rssm
    time_steps, batch_size = rssm.deterministic.shape[:2]
    horizon = config.direct_horizon
    if time_steps <= horizon:
        raise ValueError("a direct H8 loss requires at least nine sequence steps")
    source_count = time_steps - horizon

    expected_actions = (
        time_steps,
        batch_size,
        model.config.action_dim,
    )
    if inputs.previous_actions.shape != expected_actions:
        raise ValueError(f"previous_actions must have shape {expected_actions}")
    if inputs.previous_actions.device != rssm.deterministic.device:
        raise ValueError("previous_actions and rollout must share a device")

    valid_steps = _as_mask(
        targets.valid_steps,
        torch.Size((time_steps, batch_size)),
        "valid_steps",
    )
    clean_mask = _as_mask(
        targets.clean_observation_mask,
        targets.clean_observations.shape,
        "clean_observation_mask",
    )
    expected_observations = (
        time_steps,
        batch_size,
        model.config.observation_dim,
    )
    if targets.clean_observations.shape != expected_observations:
        raise ValueError(
            f"clean_observations must have shape {expected_observations}"
        )
    if any(
        value.device != rssm.deterministic.device
        for value in (
            targets.clean_observations,
            clean_mask,
            valid_steps,
        )
    ):
        raise ValueError("direct H8 targets and rollout must share a device")

    path_valid = torch.ones(
        source_count,
        batch_size,
        dtype=torch.bool,
        device=rssm.deterministic.device,
    )
    for offset in range(horizon + 1):
        path_valid = path_valid & valid_steps[offset : offset + source_count]

    start_state = RSSMState(
        deterministic=rssm.deterministic[:source_count].reshape(
            source_count * batch_size, -1
        ),
        stochastic=rssm.stochastic[:source_count].reshape(
            source_count * batch_size, -1
        ),
    )
    actions = torch.stack(
        [
            inputs.previous_actions[offset : offset + source_count]
            for offset in range(1, horizon + 1)
        ],
        dim=0,
    ).reshape(horizon, source_count * batch_size, model.config.action_dim)

    if model.config.context_dim == 0:
        if inputs.contexts is not None:
            expected_contexts = (time_steps, batch_size, 0)
            if inputs.contexts.shape != expected_contexts:
                raise ValueError(f"contexts must have shape {expected_contexts}")
        future_contexts: Optional[Tensor] = None
    else:
        if inputs.contexts is None:
            raise ValueError("contexts are required when context_dim > 0")
        expected_contexts = (
            time_steps,
            batch_size,
            model.config.context_dim,
        )
        if inputs.contexts.shape != expected_contexts:
            raise ValueError(f"contexts must have shape {expected_contexts}")
        if inputs.contexts.device != rssm.deterministic.device:
            raise ValueError("contexts and rollout must share a device")
        future_contexts = torch.stack(
            [
                inputs.contexts[offset : offset + source_count]
                for offset in range(1, horizon + 1)
            ],
            dim=0,
        ).reshape(
            horizon,
            source_count * batch_size,
            model.config.context_dim,
        )

    imagined = model.imagine(
        start_state,
        actions,
        future_contexts,
        sample=False,
    )
    predicted = imagined.observation_mean[-1].reshape(
        source_count, batch_size, model.config.observation_dim
    )
    target = targets.clean_observations[horizon:]
    target_mask = clean_mask[horizon:] & path_valid.unsqueeze(-1)
    if target_mask.any() and not torch.isfinite(target[target_mask]).all():
        raise ValueError("clean_observations contains a non-finite H8 target")
    safe_target = torch.where(target_mask, target, predicted.detach())
    element_loss = F.smooth_l1_loss(
        predicted,
        safe_target,
        reduction="none",
        beta=config.direct_horizon_beta,
    )
    loss = _masked_mean(element_loss, target_mask)
    return (
        loss,
        int(path_valid.sum().detach()),
        int(target_mask.sum().detach()),
    )


def reliability_sequence_training_loss(
    model: ReliabilityGatedRSSM,
    inputs: RSSMSequenceInputs,
    targets: RSSMSequenceTargets,
    config: Optional[ReliabilityLossConfig] = None,
    *,
    gate_mode: GateMode = "learned",
    start_state: Optional[RSSMState] = None,
    sample: bool = True,
) -> ReliabilityTrainingOutput:
    """Filter a sequence and compute ordinary, latent, and direct H8 losses."""
    config = ReliabilityLossConfig() if config is None else config
    rollout = model.filter(
        inputs.previous_actions,
        inputs.corrupted_observations,
        inputs.availability,
        inputs.age,
        inputs.contexts,
        start_state=start_state,
        gate_mode=gate_mode,
        sample=sample,
    )

    ordinary = loss_from_rollout(
        rollout.rssm,
        targets,
        replace(config, overshooting_weight=0.0),
    )
    valid_steps = _as_mask(
        targets.valid_steps,
        rollout.rssm.observation_mean.shape[:2],
        "valid_steps",
    )
    if config.overshooting_weight > 0:
        latent_overshooting, overshooting_pairs = _overshooting_dynamics_loss(
            model,
            rollout.rssm,
            inputs,
            valid_steps,
            config,
        )
    else:
        latent_overshooting = rollout.rssm.prior_mean.sum() * 0.0
        overshooting_pairs = 0

    if config.direct_horizon_weight > 0:
        direct_horizon, direct_paths, direct_targets = direct_h8_observation_loss(
            model,
            rollout,
            inputs,
            targets,
            config,
        )
    else:
        direct_horizon = rollout.rssm.observation_mean.sum() * 0.0
        direct_paths = 0
        direct_targets = 0

    counts = dict(ordinary.valid_counts)
    counts.update(
        {
            "overshooting_pairs": overshooting_pairs,
            "direct_h8_paths": direct_paths,
            "direct_h8_targets": direct_targets,
        }
    )
    total = (
        ordinary.total
        + config.overshooting_weight * latent_overshooting
        + config.direct_horizon_weight * direct_horizon
    )
    return ReliabilityTrainingOutput(
        rollout=rollout,
        loss=ReliabilityLoss(
            total=total,
            ordinary=ordinary,
            direct_horizon_smooth_l1=direct_horizon,
            latent_overshooting_kl=latent_overshooting,
            valid_counts=counts,
        ),
    )
