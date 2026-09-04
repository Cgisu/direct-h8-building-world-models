"""Causal deterministic recurrent world model for the v3 transport study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import (
    EXPECTED_ACTIVE_PARAMETERS,
    FROZEN_CONFIG,
    DeterministicTransportConfig,
)


@dataclass(frozen=True)
class DeterministicRollout:
    """Time-major hidden states and decoded clean-observation means."""

    hidden: Tensor
    observation_mean: Tensor
    final_hidden: Tensor


def active_parameter_count(module: nn.Module) -> int:
    """Count tensors that actively participate in gradient training."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


class DeterministicRecurrentWorldModel(nn.Module):
    """Deterministic filter with observation-free recursive imagination.

    Filtering at index ``t`` consumes corrupted ``y_t``, its availability and
    age, previous action ``a_(t-1)``, and known context ``c_t``. Given the
    filtered hidden state at ``t``, imagination index zero consumes only
    candidate ``a_t`` and known ``c_(t+1)``. The model recursively feeds back
    its own decoded observation; the imagination API has no future-observation
    argument.
    """

    def __init__(
        self,
        config: DeterministicTransportConfig = FROZEN_CONFIG,
    ) -> None:
        super().__init__()
        self.config = config
        self.recurrent_cell = nn.GRUCell(
            input_size=config.filter_input_dim,
            hidden_size=config.hidden_dim,
        )
        self.observation_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.decoder_hidden_dim, config.observation_dim),
        )
        count = active_parameter_count(self)
        if count != EXPECTED_ACTIVE_PARAMETERS:
            raise AssertionError(
                f"deterministic model has {count} active parameters, "
                f"expected {EXPECTED_ACTIVE_PARAMETERS}"
            )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.config.hidden_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def decode(self, hidden: Tensor) -> Tensor:
        self._check_hidden(hidden)
        return self.observation_decoder(hidden)

    def observe_step(
        self,
        hidden: Tensor,
        corrupted_observation: Tensor,
        availability: Tensor,
        age: Tensor,
        previous_action: Tensor,
        context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Filter one current measurement and decode its clean observation."""

        self._check_hidden(hidden)
        batch_size = hidden.shape[0]
        self._check_matrix(
            corrupted_observation,
            batch_size,
            self.config.observation_dim,
            "corrupted_observation",
            allow_nonfinite=True,
        )
        self._check_matrix(
            availability,
            batch_size,
            self.config.observation_dim,
            "availability",
            finite_binary=True,
        )
        self._check_matrix(
            age,
            batch_size,
            self.config.observation_dim,
            "age",
        )
        self._check_matrix(
            previous_action,
            batch_size,
            self.config.action_dim,
            "previous_action",
        )
        self._check_matrix(
            context,
            batch_size,
            self.config.context_dim,
            "context",
        )
        if (age < 0).any():
            raise ValueError("age must be nonnegative")

        available = availability.bool()
        if (available & ~torch.isfinite(corrupted_observation)).any():
            raise ValueError("an available corrupted observation is non-finite")
        safe_observation = torch.where(
            available & torch.isfinite(corrupted_observation),
            corrupted_observation,
            torch.zeros_like(corrupted_observation),
        )
        recurrent_input = torch.cat(
            [
                safe_observation.to(hidden.dtype),
                available.to(hidden.dtype),
                torch.log1p(age.to(hidden.dtype)),
                previous_action.to(hidden.dtype),
                context.to(hidden.dtype),
            ],
            dim=-1,
        )
        next_hidden = self.recurrent_cell(recurrent_input, hidden)
        return next_hidden, self.observation_decoder(next_hidden)

    def filter(
        self,
        corrupted_observations: Tensor,
        availability: Tensor,
        age: Tensor,
        previous_actions: Tensor,
        contexts: Tensor,
        *,
        start_hidden: Optional[Tensor] = None,
    ) -> DeterministicRollout:
        """Filter time-major observed inputs causally."""

        self._check_observed_sequence(
            corrupted_observations,
            availability,
            age,
            previous_actions,
            contexts,
        )
        hidden = (
            self.initial_hidden(
                corrupted_observations.shape[1],
                device=corrupted_observations.device,
            )
            if start_hidden is None
            else start_hidden
        )
        self._check_hidden(hidden, corrupted_observations.shape[1])
        hidden_steps: list[Tensor] = []
        observation_steps: list[Tensor] = []
        for values in zip(
            corrupted_observations,
            availability,
            age,
            previous_actions,
            contexts,
            strict=True,
        ):
            hidden, prediction = self.observe_step(hidden, *values)
            hidden_steps.append(hidden)
            observation_steps.append(prediction)
        return DeterministicRollout(
            hidden=torch.stack(hidden_steps),
            observation_mean=torch.stack(observation_steps),
            final_hidden=hidden,
        )

    def imagine(
        self,
        start_hidden: Tensor,
        future_actions: Tensor,
        future_contexts: Tensor,
    ) -> DeterministicRollout:
        """Roll out actions and contexts without accepting future observations.

        The decoded current observation is fed back autoregressively. It is a
        model prediction, never a future simulator measurement.
        """

        self._check_hidden(start_hidden)
        if future_actions.ndim != 3 or future_actions.shape[0] == 0:
            raise ValueError(
                "future_actions must have shape [time, batch, action_dim]"
            )
        expected_actions = (
            future_actions.shape[0],
            start_hidden.shape[0],
            self.config.action_dim,
        )
        if future_actions.shape != expected_actions:
            raise ValueError(f"future_actions must have shape {expected_actions}")
        expected_contexts = (
            future_actions.shape[0],
            start_hidden.shape[0],
            self.config.context_dim,
        )
        if future_contexts.shape != expected_contexts:
            raise ValueError(f"future_contexts must have shape {expected_contexts}")
        if any(
            value.device != start_hidden.device
            for value in (future_actions, future_contexts)
        ):
            raise ValueError("imagined inputs and start_hidden must share a device")
        if not torch.isfinite(future_actions).all():
            raise ValueError("future_actions must be finite")
        if not torch.isfinite(future_contexts).all():
            raise ValueError("future_contexts must be finite")

        hidden = start_hidden
        hidden_steps: list[Tensor] = []
        observation_steps: list[Tensor] = []
        available = torch.ones(
            start_hidden.shape[0],
            self.config.observation_dim,
            dtype=torch.bool,
            device=start_hidden.device,
        )
        age = torch.zeros(
            start_hidden.shape[0],
            self.config.observation_dim,
            dtype=start_hidden.dtype,
            device=start_hidden.device,
        )
        for action, context in zip(
            future_actions, future_contexts, strict=True
        ):
            autoregressive_observation = self.observation_decoder(hidden)
            hidden, prediction = self.observe_step(
                hidden,
                autoregressive_observation,
                available,
                age,
                action,
                context,
            )
            hidden_steps.append(hidden)
            observation_steps.append(prediction)
        return DeterministicRollout(
            hidden=torch.stack(hidden_steps),
            observation_mean=torch.stack(observation_steps),
            final_hidden=hidden,
        )

    def _check_hidden(self, hidden: Tensor, batch_size: int | None = None) -> None:
        expected_batch = (
            hidden.shape[0]
            if batch_size is None and hidden.ndim == 2
            else batch_size
        )
        expected = (expected_batch, self.config.hidden_dim)
        if hidden.ndim != 2 or hidden.shape != expected:
            raise ValueError(f"hidden must have shape {expected}")
        if not torch.isfinite(hidden).all():
            raise ValueError("hidden must be finite")

    @staticmethod
    def _check_matrix(
        value: Tensor,
        batch_size: int,
        width: int,
        name: str,
        *,
        allow_nonfinite: bool = False,
        finite_binary: bool = False,
    ) -> None:
        expected = (batch_size, width)
        if value.ndim != 2 or value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if finite_binary:
            if value.dtype != torch.bool and (
                not torch.isfinite(value).all()
                or not ((value == 0) | (value == 1)).all()
            ):
                raise ValueError(f"{name} must contain boolean or zero/one values")
        elif not allow_nonfinite and not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")

    def _check_observed_sequence(
        self,
        observations: Tensor,
        availability: Tensor,
        age: Tensor,
        actions: Tensor,
        contexts: Tensor,
    ) -> None:
        expected_observations = (
            self.config.sequence_length,
            self.config.batch_size,
            self.config.observation_dim,
        )
        if observations.shape != expected_observations:
            raise ValueError(
                f"corrupted_observations must have shape {expected_observations}"
            )
        if availability.shape != observations.shape or age.shape != observations.shape:
            raise ValueError("availability and age must match observations")
        expected_actions = observations.shape[:2] + (self.config.action_dim,)
        if actions.shape != expected_actions:
            raise ValueError(f"previous_actions must have shape {expected_actions}")
        expected_contexts = observations.shape[:2] + (self.config.context_dim,)
        if contexts.shape != expected_contexts:
            raise ValueError(f"contexts must have shape {expected_contexts}")
        if any(
            value.device != observations.device
            for value in (availability, age, actions, contexts)
        ):
            raise ValueError("all filtered inputs must share a device")
        parameter = next(self.parameters())
        if observations.device != parameter.device:
            raise ValueError("filtered inputs and model parameters must share a device")
