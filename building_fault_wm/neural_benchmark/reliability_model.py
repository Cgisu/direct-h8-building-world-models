"""Causal reliability-gated filtering for the multi-case RSSM benchmark.

The learned gate sees only the current measurement innovation, availability,
age, and a stopped-gradient prior belief. It scales the measurement contribution
before the posterior encoder, so a zero-reliability measurement cannot alter
the belief and unit reliability recovers the original RSSM filter exactly.
Health labels are training targets for ``gate_health_logits``; they are not an
inference input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from building_fault_wm.recurrent_models.model import (
    HealthAwareRSSM,
    RSSMConfig,
    RSSMRollout,
    RSSMState,
    RSSMStep,
)


GateMode = Literal["bypass", "learned", "huber"]
GATE_MODES = ("bypass", "learned", "huber")


@dataclass(frozen=True)
class ReliabilityRSSMConfig(RSSMConfig):
    """RSSM dimensions plus the causal reliability-filter settings."""

    sensor_observation_indices: tuple[int, ...] = ()
    healthy_class_index: int = 0
    innovation_clip: float = 10.0
    innovation_scale_floor: float = 0.1
    huber_threshold: float = 3.0
    gate_healthy_init_probability: float = 0.99

    def __post_init__(self) -> None:
        super().__post_init__()
        indices = self.sensor_observation_indices
        if len(indices) != self.sensor_dim:
            raise ValueError(
                "sensor_observation_indices must contain one index per sensor"
            )
        if len(set(indices)) != len(indices):
            raise ValueError("sensor_observation_indices must be unique")
        if any(index < 0 or index >= self.observation_dim for index in indices):
            raise ValueError("a sensor observation index is outside observation_dim")
        if not 0 <= self.healthy_class_index < self.health_classes:
            raise ValueError("healthy_class_index is outside health_classes")
        if self.innovation_clip <= 0:
            raise ValueError("innovation_clip must be positive")
        if self.innovation_scale_floor <= 0:
            raise ValueError("innovation_scale_floor must be positive")
        if self.huber_threshold <= 0:
            raise ValueError("huber_threshold must be positive")
        if not 0.5 < self.gate_healthy_init_probability < 1.0:
            raise ValueError("gate_healthy_init_probability must be in (0.5, 1)")


@dataclass(frozen=True)
class ReliabilityStep:
    """One posterior step plus the pre-posterior gate outputs."""

    rssm_step: RSSMStep
    observation_reliability: Tensor
    gate_health_logits: Tensor


@dataclass(frozen=True)
class ReliabilityRollout:
    """A filtered RSSM rollout with time-major gate diagnostics."""

    rssm: RSSMRollout
    observation_reliability: Tensor
    gate_health_logits: Tensor


class ReliabilityGatedRSSM(HealthAwareRSSM):
    """Health-aware RSSM whose posterior consumes gated prior innovations.

    ``bypass`` admits every available measurement, ``learned`` uses the
    predicted probability of the healthy class on mapped sensor channels, and
    ``huber`` applies a fixed robust weight to the normalized innovation.
    Unmapped observation channels are admitted whenever they are available.
    """

    config: ReliabilityRSSMConfig

    def __init__(self, config: ReliabilityRSSMConfig):
        super().__init__(config)
        self.config = config

        # Gate input: stopped prior belief plus, per observation channel,
        # prior mean/log-std, raw and normalized innovation (signed/absolute),
        # availability, and age.
        gate_input_dim = self.feature_dim + 8 * config.observation_dim
        self.reliability_feature_net = nn.Sequential(
            nn.Linear(gate_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, self.feature_dim),
            nn.SiLU(),
        )
        self._initialize_healthy_gate()

    def _initialize_healthy_gate(self) -> None:
        """Start close to the ordinary filter without making the gate constant."""
        final = self.health_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("health_head must end with a linear layer")
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        probability = self.config.gate_healthy_init_probability
        healthy_bias = log(
            probability
            * (self.config.health_classes - 1)
            / (1.0 - probability)
        )
        with torch.no_grad():
            bias = final.bias.reshape(
                self.config.sensor_dim, self.config.health_classes
            )
            bias[:, self.config.healthy_class_index] = healthy_bias

    @staticmethod
    def _check_gate_mode(gate_mode: str) -> GateMode:
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}")
        return gate_mode  # type: ignore[return-value]

    def _validated_measurement_inputs(
        self,
        observation: Tensor,
        mask: Tensor,
        age: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected = (reference.shape[0], self.config.observation_dim)
        if observation.ndim != 2 or observation.shape != expected:
            raise ValueError(f"observation must have shape {expected}")
        if mask.shape != expected or age.shape != expected:
            raise ValueError("mask and age must match observation shape")
        if any(value.device != reference.device for value in (observation, mask, age)):
            raise ValueError("observation, mask, age, and state must share a device")
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
        effective = availability & torch.isfinite(observation)
        return (
            observation.to(reference.dtype),
            effective,
            age.to(reference.dtype),
        )

    def _gate_logits(
        self,
        prior_state: RSSMState,
        prior_observation_mean: Tensor,
        prior_observation_std: Tensor,
        residual: Tensor,
        normalized_innovation: Tensor,
        availability: Tensor,
        age: Tensor,
    ) -> Tensor:
        prior_features = self.belief_features(prior_state).detach()
        gate_input = torch.cat(
            [
                prior_features,
                prior_observation_mean.detach(),
                prior_observation_std.detach().log(),
                residual,
                residual.abs(),
                normalized_innovation,
                normalized_innovation.abs(),
                availability.to(residual.dtype),
                torch.log1p(age),
            ],
            dim=-1,
        )
        features = self.reliability_feature_net(gate_input)
        return self.health_head(features).reshape(
            residual.shape[0],
            self.config.sensor_dim,
            self.config.health_classes,
        )

    def _observation_reliability(
        self,
        gate_logits: Tensor,
        normalized_innovation: Tensor,
        availability: Tensor,
        gate_mode: GateMode,
    ) -> Tensor:
        reliability = availability.to(normalized_innovation.dtype)
        if gate_mode == "bypass":
            return reliability

        indices = torch.as_tensor(
            self.config.sensor_observation_indices,
            dtype=torch.long,
            device=availability.device,
        )
        mapped_available = availability.index_select(1, indices).to(reliability.dtype)
        if gate_mode == "learned":
            healthy_probability = gate_logits.softmax(dim=-1)[
                ..., self.config.healthy_class_index
            ]
            mapped_reliability = mapped_available * healthy_probability
        else:
            magnitude = normalized_innovation.index_select(1, indices).abs()
            robust_weight = (
                self.config.huber_threshold / magnitude.clamp_min(1e-12)
            ).clamp(max=1.0)
            mapped_reliability = mapped_available * robust_weight

        scatter_indices = indices.unsqueeze(0).expand(availability.shape[0], -1)
        return reliability.scatter(1, scatter_indices, mapped_reliability)

    def filter_step(
        self,
        state: RSSMState,
        previous_action: Tensor,
        observation: Tensor,
        mask: Tensor,
        age: Tensor,
        context: Optional[Tensor] = None,
        *,
        gate_mode: GateMode = "learned",
        sample: bool = True,
    ) -> ReliabilityStep:
        """Filter one current measurement without using health labels."""
        gate_mode = self._check_gate_mode(gate_mode)
        self._check_state(state)
        self._check_action(previous_action, state.deterministic.shape[0])
        if previous_action.device != state.deterministic.device:
            raise ValueError("previous_action and state must share a device")
        if not torch.isfinite(previous_action).all():
            raise ValueError("previous_action must be finite")
        previous_action = previous_action.to(state.deterministic.dtype)
        context = self._prepare_context(
            context,
            state.deterministic.shape[0],
            state.deterministic,
        )
        observation, effective_mask, age = self._validated_measurement_inputs(
            observation, mask, age, state.deterministic
        )

        prior = self.transition(state, previous_action, context, sample=False)
        prior_mean = prior.prediction.observation_mean
        prior_std = prior.prediction.observation_std
        safe_observation = torch.where(effective_mask, observation, prior_mean.detach())
        residual = safe_observation - prior_mean.detach()
        normalized = residual / prior_std.detach().clamp_min(
            self.config.innovation_scale_floor
        )
        gate_normalized = normalized.clamp(
            -self.config.innovation_clip, self.config.innovation_clip
        )
        gate_logits = self._gate_logits(
            prior.state,
            prior_mean,
            prior_std,
            residual,
            gate_normalized,
            effective_mask,
            age,
        )
        reliability = self._observation_reliability(
            gate_logits, normalized, effective_mask, gate_mode
        )

        gated_measurement = torch.where(
            effective_mask,
            prior_mean.detach() + reliability * residual,
            torch.zeros_like(observation),
        )
        embedding = self.observation_encoder(
            torch.cat([gated_measurement, reliability, torch.log1p(age)], dim=-1)
        )
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
        posterior_state = RSSMState(
            prior.state.deterministic,
            self._latent(posterior_mean, posterior_std, sample),
        )
        prediction = replace(
            self.decode(posterior_state), health_logits=gate_logits
        )
        step = RSSMStep(
            state=posterior_state,
            prior_mean=prior.prior_mean,
            prior_std=prior.prior_std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prediction=prediction,
        )
        return ReliabilityStep(
            rssm_step=step,
            observation_reliability=reliability,
            gate_health_logits=gate_logits,
        )

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
        gate_mode: GateMode = "learned",
    ) -> RSSMStep:
        """Compatibility API that defaults to the learned reliability gate."""
        return self.filter_step(
            state,
            previous_action,
            observation,
            mask,
            age,
            context,
            gate_mode=gate_mode,
            sample=sample,
        ).rssm_step

    def filter(
        self,
        previous_actions: Tensor,
        observations: Tensor,
        masks: Tensor,
        ages: Tensor,
        contexts: Optional[Tensor] = None,
        *,
        start_state: Optional[RSSMState] = None,
        gate_mode: GateMode = "learned",
        sample: bool = True,
    ) -> ReliabilityRollout:
        """Filter a time-major sequence and retain causal gate diagnostics."""
        gate_mode = self._check_gate_mode(gate_mode)
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
        steps: list[ReliabilityStep] = []
        for time_index in range(observations.shape[0]):
            result = self.filter_step(
                state,
                previous_actions[time_index],
                observations[time_index],
                masks[time_index],
                ages[time_index],
                contexts[time_index],
                gate_mode=gate_mode,
                sample=sample,
            )
            steps.append(result)
            state = result.rssm_step.state
        rssm = self._stack_steps(
            [result.rssm_step for result in steps], include_posterior=True
        )
        return ReliabilityRollout(
            rssm=rssm,
            observation_reliability=torch.stack(
                [result.observation_reliability for result in steps]
            ),
            gate_health_logits=torch.stack(
                [result.gate_health_logits for result in steps]
            ),
        )

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
        gate_mode: GateMode = "learned",
    ) -> RSSMRollout:
        """Standard RSSM API backed by the reliability-gated filter."""
        return self.filter(
            previous_actions,
            observations,
            masks,
            ages,
            contexts,
            start_state=start_state,
            gate_mode=gate_mode,
            sample=sample,
        ).rssm
