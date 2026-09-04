from __future__ import annotations

import inspect

import torch

from .config import (
    EXPECTED_ACTIVE_PARAMETERS,
    REFERENCE_RSSM_ACTIVE_PARAMETERS,
)
from .model import DeterministicRecurrentWorldModel, active_parameter_count


def _observed_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    observation = torch.randn(48, 4, 4)
    availability = torch.ones_like(observation, dtype=torch.bool)
    age = torch.zeros_like(observation)
    action = torch.randn(48, 4, 1)
    context = torch.randn(48, 4, 5)
    observation[10:14, 1, 0] = torch.nan
    availability[10:14, 1, 0] = False
    age[10:14, 1, 0] = torch.arange(1, 5)
    return observation, availability, age, action, context


def test_active_parameter_count_matches_rssm_within_five_parameters() -> None:
    model = DeterministicRecurrentWorldModel()
    assert active_parameter_count(model) == EXPECTED_ACTIVE_PARAMETERS == 19_789
    assert EXPECTED_ACTIVE_PARAMETERS - REFERENCE_RSSM_ACTIVE_PARAMETERS == 5
    first_decoder = model.observation_decoder[0]
    assert isinstance(first_decoder, torch.nn.Linear)
    assert first_decoder.out_features == 53


def test_filter_and_imagination_shapes_exclude_future_observations() -> None:
    model = DeterministicRecurrentWorldModel()
    filtered = model.filter(*_observed_inputs())
    assert filtered.hidden.shape == (48, 4, 64)
    assert filtered.observation_mean.shape == (48, 4, 4)
    assert filtered.final_hidden.shape == (4, 64)

    signature = inspect.signature(model.imagine)
    assert tuple(signature.parameters) == (
        "start_hidden",
        "future_actions",
        "future_contexts",
    )
    imagined = model.imagine(
        filtered.hidden[0],
        torch.zeros(8, 4, 1),
        torch.zeros(8, 4, 5),
    )
    assert imagined.hidden.shape == (8, 4, 64)
    assert imagined.observation_mean.shape == (8, 4, 4)


def test_imagination_is_sensitive_to_candidate_actions() -> None:
    torch.manual_seed(11)
    model = DeterministicRecurrentWorldModel()
    start = torch.randn(3, 64)
    contexts = torch.randn(8, 3, 5)
    low = model.imagine(start, -torch.ones(8, 3, 1), contexts)
    high = model.imagine(start, torch.ones(8, 3, 1), contexts)
    assert not torch.allclose(
        low.observation_mean[-1],
        high.observation_mean[-1],
        rtol=0.0,
        atol=1e-8,
    )
