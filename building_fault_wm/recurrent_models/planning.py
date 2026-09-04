"""Bounded latent MPC for the health-aware RSSM.

The planner is gradient-free and accepts a filtered latent state plus known
future context. Every candidate is evaluated through
``HealthAwareRSSM.imagine``; there is no API for future observations, masks, or
sensor-health labels. CEM samples actions only and holds context fixed.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from .model import HealthAwareRSSM, RSSMState


ActionBound = Union[float, Sequence[float], Tensor]


@dataclass(frozen=True)
class LatentMPCConfig:
    """Cross-entropy-method search and risk-scoring parameters."""

    horizon: int = 12
    candidates: int = 256
    iterations: int = 4
    elite_fraction: float = 0.1
    discount: float = 0.99
    constraint_weight: float = 10.0
    cost_uncertainty_weight: float = 0.25
    observation_uncertainty_weight: float = 0.05
    smoothing: float = 0.1
    minimum_std_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.candidates <= 1 or self.iterations <= 0:
            raise ValueError("horizon/iterations must be positive and candidates > 1")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if min(
            self.constraint_weight,
            self.cost_uncertainty_weight,
            self.observation_uncertainty_weight,
        ) < 0:
            raise ValueError("risk weights must be nonnegative")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if self.minimum_std_fraction <= 0:
            raise ValueError("minimum_std_fraction must be positive")


@dataclass(frozen=True)
class LatentMPCPlan:
    """Best sequence and its discounted objective decomposition."""

    first_action: Tensor
    action_sequence: Tensor
    objective: Tensor
    predicted_cost: Tensor
    predicted_constraint_risk: Tensor
    predicted_uncertainty: Tensor


class LatentMPC:
    """Plan bounded actions by CEM search over open-loop latent rollouts.

    Batch planning is supported. ``start_state`` has batch size ``B`` and the
    returned first action has shape ``[B, action_dim]``.
    """

    def __init__(
        self,
        model: HealthAwareRSSM,
        action_low: ActionBound,
        action_high: ActionBound,
        config: Optional[LatentMPCConfig] = None,
    ) -> None:
        self.model = model
        self.config = LatentMPCConfig() if config is None else config
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32).flatten()
        action_dim = model.config.action_dim
        if self.action_low.numel() == 1:
            self.action_low = self.action_low.expand(action_dim).clone()
        if self.action_high.numel() == 1:
            self.action_high = self.action_high.expand(action_dim).clone()
        if self.action_low.shape != (action_dim,) or self.action_high.shape != (
            action_dim,
        ):
            raise ValueError("action bounds must be scalar or have action_dim entries")
        if not torch.isfinite(self.action_low).all() or not torch.isfinite(
            self.action_high
        ).all():
            raise ValueError("action bounds must be finite")
        if not (self.action_low < self.action_high).all():
            raise ValueError("every action_low entry must be below action_high")

    @staticmethod
    def _expand_state(state: RSSMState, candidates: int) -> RSSMState:
        batch = state.deterministic.shape[0]
        return RSSMState(
            deterministic=state.deterministic[:, None, :]
            .expand(batch, candidates, -1)
            .reshape(batch * candidates, -1),
            stochastic=state.stochastic[:, None, :]
            .expand(batch, candidates, -1)
            .reshape(batch * candidates, -1),
        )

    def _score(
        self,
        start_state: RSSMState,
        sequences: Tensor,
        future_contexts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Score ``[batch, candidates, horizon, action]`` sequences."""
        batch, candidates, horizon, action_dim = sequences.shape
        if horizon != self.config.horizon or action_dim != self.model.config.action_dim:
            raise ValueError("candidate sequence shape does not match planner config")
        expanded_state = self._expand_state(start_state, candidates)
        time_major_actions = sequences.permute(2, 0, 1, 3).reshape(
            horizon, batch * candidates, action_dim
        )
        # Known contexts vary by horizon step and environment batch, but every
        # candidate for the same environment sees the identical context path.
        expanded_contexts = future_contexts[:, :, None, :].expand(
            horizon, batch, candidates, self.model.config.context_dim
        ).reshape(horizon, batch * candidates, self.model.config.context_dim)
        rollout = self.model.imagine(
            expanded_state,
            time_major_actions,
            expanded_contexts,
            sample=False,
        )
        predicted_cost = rollout.cost_mean.squeeze(-1).reshape(
            horizon, batch, candidates
        )
        constraint_risk = torch.sigmoid(rollout.constraint_logits).sum(-1).reshape(
            horizon, batch, candidates
        )
        uncertainty = (
            self.config.cost_uncertainty_weight * rollout.cost_std.squeeze(-1)
            + self.config.observation_uncertainty_weight
            * rollout.observation_std.mean(-1)
        ).reshape(horizon, batch, candidates)
        discounts = torch.pow(
            torch.as_tensor(
                self.config.discount,
                device=sequences.device,
                dtype=sequences.dtype,
            ),
            torch.arange(horizon, device=sequences.device, dtype=sequences.dtype),
        ).view(horizon, 1, 1)
        discounted_cost = (discounts * predicted_cost).sum(0)
        discounted_constraint = (discounts * constraint_risk).sum(0)
        discounted_uncertainty = (discounts * uncertainty).sum(0)
        objective = (
            discounted_cost
            + self.config.constraint_weight * discounted_constraint
            + discounted_uncertainty
        )
        return (
            objective,
            discounted_cost,
            discounted_constraint,
            discounted_uncertainty,
        )

    @torch.no_grad()
    def plan(
        self,
        start_state: RSSMState,
        *,
        future_contexts: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> LatentMPCPlan:
        """Plan ``a_t`` with fixed contexts ``c_(t+1:t+horizon)``."""
        self.model._check_state(start_state)
        c = self.config
        batch = start_state.deterministic.shape[0]
        device = start_state.deterministic.device
        dtype = start_state.deterministic.dtype
        future_contexts = self.model._prepare_context_sequence(
            future_contexts,
            c.horizon,
            batch,
            start_state.deterministic,
            name="future_contexts",
        )
        low = self.action_low.to(device=device, dtype=dtype)
        high = self.action_high.to(device=device, dtype=dtype)
        action_dim = self.model.config.action_dim
        midpoint = (low + high) / 2.0
        half_range = (high - low) / 2.0
        mean = midpoint.view(1, 1, action_dim).expand(
            batch, c.horizon, action_dim
        ).clone()
        std = half_range.view(1, 1, action_dim).expand_as(mean).clone()
        minimum_std = c.minimum_std_fraction * (high - low)
        elite_count = max(1, int(round(c.candidates * c.elite_fraction)))

        best_sequence = mean.clone()
        best_score = torch.full((batch,), torch.inf, device=device, dtype=dtype)
        best_cost = torch.zeros_like(best_score)
        best_constraint = torch.zeros_like(best_score)
        best_uncertainty = torch.zeros_like(best_score)
        batch_indices = torch.arange(batch, device=device)

        for _ in range(c.iterations):
            random_count = c.candidates - 1
            noise = torch.randn(
                batch,
                random_count,
                c.horizon,
                action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            random_sequences = mean[:, None] + std[:, None] * noise
            random_sequences = torch.maximum(
                torch.minimum(random_sequences, high.view(1, 1, 1, -1)),
                low.view(1, 1, 1, -1),
            )
            # Including the current mean prevents an iteration from discarding
            # a good solution due only to finite random sampling.
            sequences = torch.cat([mean[:, None], random_sequences], dim=1)
            objective, cost, constraint, uncertainty = self._score(
                start_state, sequences, future_contexts
            )

            iteration_score, iteration_index = objective.min(dim=1)
            improved = iteration_score < best_score
            selected_sequence = sequences[batch_indices, iteration_index]
            selected_cost = cost[batch_indices, iteration_index]
            selected_constraint = constraint[batch_indices, iteration_index]
            selected_uncertainty = uncertainty[batch_indices, iteration_index]
            best_sequence = torch.where(
                improved[:, None, None], selected_sequence, best_sequence
            )
            best_score = torch.where(improved, iteration_score, best_score)
            best_cost = torch.where(improved, selected_cost, best_cost)
            best_constraint = torch.where(
                improved, selected_constraint, best_constraint
            )
            best_uncertainty = torch.where(
                improved, selected_uncertainty, best_uncertainty
            )

            elite_indices = objective.topk(
                elite_count, dim=1, largest=False
            ).indices
            elites = sequences.gather(
                1,
                elite_indices[:, :, None, None].expand(
                    batch, elite_count, c.horizon, action_dim
                ),
            )
            elite_mean = elites.mean(dim=1)
            elite_std = elites.std(dim=1, unbiased=False).clamp_min(
                minimum_std.view(1, 1, -1)
            )
            mean = c.smoothing * mean + (1.0 - c.smoothing) * elite_mean
            std = c.smoothing * std + (1.0 - c.smoothing) * elite_std

        return LatentMPCPlan(
            first_action=best_sequence[:, 0],
            action_sequence=best_sequence,
            objective=best_score,
            predicted_cost=best_cost,
            predicted_constraint_risk=best_constraint,
            predicted_uncertainty=best_uncertainty,
        )

    def act(
        self,
        start_state: RSSMState,
        *,
        future_contexts: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Convenience wrapper returning only the receding-horizon action."""
        return self.plan(
            start_state,
            future_contexts=future_contexts,
            generator=generator,
        ).first_action
