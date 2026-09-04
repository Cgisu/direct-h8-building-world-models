"""Compact health-aware recurrent state-space model.

The posterior filters corrupted measurements into a latent belief. The learned
prior advances that belief with optimized actions and separately supplied known
context, so it can simulate candidate control sequences without reading future
observations or optimizing exogenous inputs.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class RSSMConfig:
    """Model dimensions.

    ``sensor_dim`` may be smaller than ``observation_dim`` when only a subset of
    channels receive explicit health labels. ``constraint_dim`` is the number
    of binary constraint events predicted at every imagined step.
    ``context_dim`` contains known, non-optimized features such as weather,
    solar gain, occupancy, tariff, time encoding, and comfort schedule.
    """

    observation_dim: int
    action_dim: int
    sensor_dim: int
    constraint_dim: int
    deterministic_dim: int = 128
    stochastic_dim: int = 32
    embedding_dim: int = 128
    hidden_dim: int = 128
    health_classes: int = 5
    min_std: float = 0.1
    context_dim: int = 0

    def __post_init__(self) -> None:
        integer_dims = (
            self.observation_dim,
            self.action_dim,
            self.sensor_dim,
            self.constraint_dim,
            self.deterministic_dim,
            self.stochastic_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.health_classes,
        )
        if any(dim <= 0 for dim in integer_dims):
            raise ValueError("all RSSM dimensions must be positive")
        if self.context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if self.min_std <= 0:
            raise ValueError("min_std must be positive")


@dataclass(frozen=True)
class RSSMState:
    """Latent belief at one time step.

    Shapes are ``[batch, deterministic_dim]`` and
    ``[batch, stochastic_dim]`` respectively.
    """

    deterministic: Tensor
    stochastic: Tensor


@dataclass(frozen=True)
class RSSMPrediction:
    """Decoded predictions from one latent state."""

    observation_mean: Tensor
    observation_std: Tensor
    cost_mean: Tensor
    cost_std: Tensor
    constraint_logits: Tensor
    continuation_logit: Tensor
    health_logits: Tensor


@dataclass(frozen=True)
class RSSMStep:
    """One posterior or prior transition and its decoded predictions."""

    state: RSSMState
    prior_mean: Tensor
    prior_std: Tensor
    prediction: RSSMPrediction
    posterior_mean: Optional[Tensor] = None
    posterior_std: Optional[Tensor] = None


@dataclass(frozen=True)
class RSSMRollout:
    """Time-major tensors from filtering or open-loop imagination.

    Every tensor starts with ``[time, batch, ...]``. Posterior statistics are
    present for ``observe()`` and absent for ``imagine()``.
    """

    deterministic: Tensor
    stochastic: Tensor
    prior_mean: Tensor
    prior_std: Tensor
    observation_mean: Tensor
    observation_std: Tensor
    cost_mean: Tensor
    cost_std: Tensor
    constraint_logits: Tensor
    continuation_logit: Tensor
    health_logits: Tensor
    final_state: RSSMState
    posterior_mean: Optional[Tensor] = None
    posterior_std: Optional[Tensor] = None


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class HealthAwareRSSM(nn.Module):
    """Action-conditioned latent dynamics with fault-aware filtering.

    At observed time ``t``, ``observe_step`` filters ``y_t`` with previous
    action ``a_(t-1)`` and known context ``c_t``. Starting from that posterior,
    the first imagined step applies candidate ``a_t`` and known ``c_(t+1)`` to
    predict ``y_(t+1)`` and its interval outcome. Context is never an optimized
    action, and no future measurement enters an imagined transition.
    """

    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config
        c = config

        # Each channel contributes its masked value, availability, and age.
        self.observation_encoder = _mlp(
            3 * c.observation_dim, c.hidden_dim, c.embedding_dim
        )
        self.recurrent = nn.GRUCell(
            c.stochastic_dim + c.action_dim + c.context_dim,
            c.deterministic_dim,
        )
        # Direct action/context inputs make their conditioning explicit as well
        # as carrying them through the recurrent deterministic state.
        self.prior_net = _mlp(
            c.deterministic_dim + c.action_dim + c.context_dim,
            c.hidden_dim,
            2 * c.stochastic_dim,
        )
        self.posterior_net = _mlp(
            c.deterministic_dim
            + c.action_dim
            + c.context_dim
            + c.embedding_dim,
            c.hidden_dim,
            2 * c.stochastic_dim,
        )

        feature_dim = c.deterministic_dim + c.stochastic_dim
        self.observation_head = _mlp(
            feature_dim, c.hidden_dim, 2 * c.observation_dim
        )
        self.cost_head = _mlp(feature_dim, c.hidden_dim, 2)
        self.constraint_head = _mlp(
            feature_dim, c.hidden_dim, c.constraint_dim
        )
        self.continuation_head = _mlp(feature_dim, c.hidden_dim, 1)
        self.health_head = _mlp(
            feature_dim, c.hidden_dim, c.sensor_dim * c.health_classes
        )

    @property
    def feature_dim(self) -> int:
        return self.config.deterministic_dim + self.config.stochastic_dim

    def initial(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> RSSMState:
        """Return a zero belief for a new batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        return RSSMState(
            deterministic=torch.zeros(
                batch_size,
                self.config.deterministic_dim,
                device=device,
                dtype=dtype,
            ),
            stochastic=torch.zeros(
                batch_size,
                self.config.stochastic_dim,
                device=device,
                dtype=dtype,
            ),
        )

    def belief_features(self, state: RSSMState) -> Tensor:
        """Features for a belief-only controller that performs no imagination."""
        self._check_state(state)
        return torch.cat([state.deterministic, state.stochastic], dim=-1)

    def _distribution(self, statistics: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_std = statistics.chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.config.min_std
        return mean, std

    @staticmethod
    def _latent(mean: Tensor, std: Tensor, sample: bool) -> Tensor:
        if sample:
            return mean + std * torch.randn_like(std)
        return mean

    def _check_state(self, state: RSSMState) -> None:
        c = self.config
        if state.deterministic.ndim != 2 or state.stochastic.ndim != 2:
            raise ValueError("RSSM state tensors must have shape [batch, features]")
        if state.deterministic.shape[0] != state.stochastic.shape[0]:
            raise ValueError("deterministic and stochastic batch sizes differ")
        if state.deterministic.shape[-1] != c.deterministic_dim:
            raise ValueError("wrong deterministic state dimension")
        if state.stochastic.shape[-1] != c.stochastic_dim:
            raise ValueError("wrong stochastic state dimension")

    def _check_action(self, action: Tensor, batch_size: int) -> None:
        if action.ndim != 2:
            raise ValueError("action must have shape [batch, action_dim]")
        if action.shape != (batch_size, self.config.action_dim):
            raise ValueError("action shape does not match state batch and action_dim")

    def _prepare_context(
        self,
        context: Optional[Tensor],
        batch_size: int,
        reference: Tensor,
        *,
        name: str = "context",
    ) -> Tensor:
        """Validate known context and return it in the model feature dtype."""
        context_dim = self.config.context_dim
        if context is None:
            if context_dim != 0:
                raise ValueError(f"{name} is required when context_dim > 0")
            return reference.new_zeros((batch_size, 0))
        if context.ndim != 2 or context.shape != (batch_size, context_dim):
            raise ValueError(
                f"{name} must have shape [batch, context_dim]"
            )
        if context.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")
        if not torch.isfinite(context).all():
            raise ValueError(f"{name} must be finite")
        return context.to(reference.dtype)

    def transition(
        self,
        state: RSSMState,
        action: Tensor,
        context: Optional[Tensor] = None,
        *,
        sample: bool = True,
    ) -> RSSMStep:
        """Advance a belief with an action and known next-state context."""
        self._check_state(state)
        self._check_action(action, state.deterministic.shape[0])
        if action.device != state.deterministic.device:
            raise ValueError("action and state must share a device")
        if not torch.isfinite(action).all():
            raise ValueError("action must be finite")
        action = action.to(state.deterministic.dtype)
        context = self._prepare_context(
            context,
            state.deterministic.shape[0],
            state.deterministic,
        )
        recurrent_input = torch.cat([state.stochastic, action, context], dim=-1)
        deterministic = self.recurrent(recurrent_input, state.deterministic)
        prior_mean, prior_std = self._distribution(
            self.prior_net(
                torch.cat([deterministic, action, context], dim=-1)
            )
        )
        stochastic = self._latent(prior_mean, prior_std, sample)
        next_state = RSSMState(deterministic, stochastic)
        return RSSMStep(
            state=next_state,
            prior_mean=prior_mean,
            prior_std=prior_std,
            prediction=self.decode(next_state),
        )

    def _encode_observation(
        self, observation: Tensor, mask: Tensor, age: Tensor
    ) -> Tensor:
        c = self.config
        if observation.ndim != 2:
            raise ValueError("observation must have shape [batch, observation_dim]")
        expected = (observation.shape[0], c.observation_dim)
        if observation.shape != expected:
            raise ValueError("observation must have shape [batch, observation_dim]")
        if mask.shape != expected or age.shape != expected:
            raise ValueError("mask and age must match observation shape")
        if mask.device != observation.device or age.device != observation.device:
            raise ValueError("observation, mask, and age must share a device")
        if mask.dtype == torch.bool:
            availability = mask
        else:
            if not torch.isfinite(mask).all() or not (
                (mask == 0) | (mask == 1)
            ).all():
                raise ValueError("mask must contain only zero/one or boolean values")
            availability = mask.bool()
        if not torch.isfinite(age).all():
            raise ValueError("age must be finite")
        if (age < 0).any():
            raise ValueError("age must be nonnegative")
        # A non-finite observation is unavailable even when the external mask
        # says otherwise. This prevents an inconsistent NaN-as-available value
        # from entering either the value path or availability feature.
        effective_mask = availability & torch.isfinite(observation)
        mask_value = effective_mask.to(observation.dtype)
        age = age.to(observation.dtype)
        value = torch.where(
            effective_mask, observation, torch.zeros_like(observation)
        )
        encoded_input = torch.cat(
            [value, mask_value, torch.log1p(age)], dim=-1
        )
        return self.observation_encoder(encoded_input)

    def observe_step(
        self,
        state: RSSMState,
        previous_action: Tensor,
        observation: Tensor,
        mask: Tensor,
        age: Tensor,
        context: Optional[Tensor] = None,
        *,
        sample: bool = True,
    ) -> RSSMStep:
        """Filter ``y_t`` using previous action ``a_(t-1)`` and ``c_t``."""
        context = self._prepare_context(
            context,
            state.deterministic.shape[0],
            state.deterministic,
        )
        prior = self.transition(
            state, previous_action, context, sample=False
        )
        embedding = self._encode_observation(observation, mask, age)
        posterior_statistics = self.posterior_net(
            torch.cat(
                [
                    prior.state.deterministic,
                    previous_action,
                    context,
                    embedding,
                ],
                dim=-1,
            )
        )
        posterior_mean, posterior_std = self._distribution(posterior_statistics)
        stochastic = self._latent(posterior_mean, posterior_std, sample)
        posterior_state = RSSMState(prior.state.deterministic, stochastic)
        return RSSMStep(
            state=posterior_state,
            prior_mean=prior.prior_mean,
            prior_std=prior.prior_std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prediction=self.decode(posterior_state),
        )

    def decode(self, state: RSSMState) -> RSSMPrediction:
        """Predict measurable and decision-relevant quantities from a belief."""
        features = self.belief_features(state)
        observation_mean, observation_std = self._distribution(
            self.observation_head(features)
        )
        cost_mean, cost_std = self._distribution(self.cost_head(features))
        health = self.health_head(features).reshape(
            features.shape[0],
            self.config.sensor_dim,
            self.config.health_classes,
        )
        return RSSMPrediction(
            observation_mean=observation_mean,
            observation_std=observation_std,
            cost_mean=cost_mean,
            cost_std=cost_std,
            constraint_logits=self.constraint_head(features),
            continuation_logit=self.continuation_head(features),
            health_logits=health,
        )

    def _prepare_context_sequence(
        self,
        contexts: Optional[Tensor],
        time_steps: int,
        batch_size: int,
        reference: Tensor,
        *,
        name: str,
    ) -> Tensor:
        if contexts is None:
            if self.config.context_dim != 0:
                raise ValueError(f"{name} is required when context_dim > 0")
            return reference.new_zeros((time_steps, batch_size, 0))
        expected = (time_steps, batch_size, self.config.context_dim)
        if contexts.ndim != 3 or contexts.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if contexts.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")
        if not torch.isfinite(contexts).all():
            raise ValueError(f"{name} must be finite")
        return contexts.to(reference.dtype)

    def observe(
        self,
        previous_actions: Tensor,
        observations: Tensor,
        masks: Tensor,
        ages: Tensor,
        contexts: Optional[Tensor] = None,
        *,
        start_state: Optional[RSSMState] = None,
        sample: bool = True,
    ) -> RSSMRollout:
        """Filter ``y_t`` with aligned ``a_(t-1)`` and ``c_t`` sequences."""
        self._check_observation_sequence(previous_actions, observations, masks, ages)
        contexts = self._prepare_context_sequence(
            contexts,
            observations.shape[0],
            observations.shape[1],
            observations,
            name="contexts",
        )
        state = (
            self.initial(
                observations.shape[1],
                device=observations.device,
                dtype=observations.dtype,
            )
            if start_state is None
            else start_state
        )
        steps = []
        for time_index in range(observations.shape[0]):
            step = self.observe_step(
                state,
                previous_actions[time_index],
                observations[time_index],
                masks[time_index],
                ages[time_index],
                contexts[time_index],
                sample=sample,
            )
            steps.append(step)
            state = step.state
        return self._stack_steps(steps, include_posterior=True)

    def imagine(
        self,
        start_state: RSSMState,
        actions: Tensor,
        future_contexts: Optional[Tensor] = None,
        *,
        sample: bool = True,
    ) -> RSSMRollout:
        """Roll out ``[time, batch, action_dim]`` actions with no observations.

        Given posterior state ``t``, returned index zero applies ``a_t`` and
        ``c_(t+1)`` to predict ``y_(t+1)`` and its interval outcome. This API has
        no observation argument, preventing a planner from consuming future
        measurements. ``future_contexts`` are known inputs, not decisions.
        """
        self._check_state(start_state)
        if actions.ndim != 3:
            raise ValueError("actions must have shape [time, batch, action_dim]")
        if actions.shape[0] == 0:
            raise ValueError("imagination horizon must be positive")
        if actions.shape[1:] != (
            start_state.deterministic.shape[0],
            self.config.action_dim,
        ):
            raise ValueError("imagined actions do not match start state")
        future_contexts = self._prepare_context_sequence(
            future_contexts,
            actions.shape[0],
            actions.shape[1],
            actions,
            name="future_contexts",
        )
        state = start_state
        steps = []
        for action, context in zip(actions, future_contexts):
            step = self.transition(state, action, context, sample=sample)
            steps.append(step)
            state = step.state
        return self._stack_steps(steps, include_posterior=False)

    def _check_observation_sequence(
        self,
        actions: Tensor,
        observations: Tensor,
        masks: Tensor,
        ages: Tensor,
    ) -> None:
        c = self.config
        if observations.ndim != 3:
            raise ValueError(
                "observations must have shape [time, batch, observation_dim]"
            )
        if observations.shape[0] == 0:
            raise ValueError("observation sequence must be non-empty")
        if observations.shape[-1] != c.observation_dim:
            raise ValueError("wrong observation dimension")
        if masks.shape != observations.shape or ages.shape != observations.shape:
            raise ValueError("masks and ages must match observations")
        if actions.shape != observations.shape[:2] + (c.action_dim,):
            raise ValueError("previous actions must align with observations")

    @staticmethod
    def _stack_steps(steps: list[RSSMStep], include_posterior: bool) -> RSSMRollout:
        prediction = [step.prediction for step in steps]
        return RSSMRollout(
            deterministic=torch.stack([step.state.deterministic for step in steps]),
            stochastic=torch.stack([step.state.stochastic for step in steps]),
            prior_mean=torch.stack([step.prior_mean for step in steps]),
            prior_std=torch.stack([step.prior_std for step in steps]),
            posterior_mean=(
                torch.stack([step.posterior_mean for step in steps])
                if include_posterior
                else None
            ),
            posterior_std=(
                torch.stack([step.posterior_std for step in steps])
                if include_posterior
                else None
            ),
            observation_mean=torch.stack(
                [item.observation_mean for item in prediction]
            ),
            observation_std=torch.stack(
                [item.observation_std for item in prediction]
            ),
            cost_mean=torch.stack([item.cost_mean for item in prediction]),
            cost_std=torch.stack([item.cost_std for item in prediction]),
            constraint_logits=torch.stack(
                [item.constraint_logits for item in prediction]
            ),
            continuation_logit=torch.stack(
                [item.continuation_logit for item in prediction]
            ),
            health_logits=torch.stack([item.health_logits for item in prediction]),
            final_state=steps[-1].state,
        )

    @staticmethod
    def kl_divergence(step: RSSMStep) -> Tensor:
        """Elementwise ``KL(q(z|o) || p(z))`` for an observed step."""
        if step.posterior_mean is None or step.posterior_std is None:
            raise ValueError("KL requires a posterior step from observe_step")
        prior_var = step.prior_std.square()
        posterior_var = step.posterior_std.square()
        return 0.5 * (
            (posterior_var + (step.posterior_mean - step.prior_mean).square())
            / prior_var
            - 1.0
            + 2.0 * (step.prior_std.log() - step.posterior_std.log())
        ).sum(dim=-1)


class BeliefController(nn.Module):
    """Small no-planning actor for a broader controller comparison."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        if feature_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("controller dimensions must be positive")
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: RSSMState) -> Tensor:
        features = torch.cat([state.deterministic, state.stochastic], dim=-1)
        return self.net(features)
