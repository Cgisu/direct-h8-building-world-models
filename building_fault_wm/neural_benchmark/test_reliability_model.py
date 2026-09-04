"""Focused invariants for the causal reliability-gated RSSM."""

from __future__ import annotations

import inspect
import io
from types import MethodType

import torch
import torch.nn as nn

from building_fault_wm.recurrent_models.model import HealthAwareRSSM, RSSMConfig
from building_fault_wm.neural_benchmark.reliability_model import (
    ReliabilityGatedRSSM,
    ReliabilityRSSMConfig,
)


def make_model(seed: int = 17) -> ReliabilityGatedRSSM:
    torch.manual_seed(seed)
    return ReliabilityGatedRSSM(
        ReliabilityRSSMConfig(
            observation_dim=4,
            action_dim=2,
            sensor_dim=2,
            constraint_dim=1,
            deterministic_dim=9,
            stochastic_dim=5,
            embedding_dim=7,
            hidden_dim=13,
            health_classes=5,
            context_dim=3,
            sensor_observation_indices=(0, 2),
        )
    )


def step_inputs(model: ReliabilityGatedRSSM, batch: int = 3):
    config = model.config
    state = model.initial(batch)
    action = torch.randn(batch, config.action_dim)
    observation = torch.randn(batch, config.observation_dim)
    mask = torch.ones_like(observation, dtype=torch.bool)
    age = torch.zeros_like(observation)
    context = torch.randn(batch, config.context_dim)
    return state, action, observation, mask, age, context


def force_gate_class(model: ReliabilityGatedRSSM, class_index: int) -> None:
    def constant_logits(self, prior_state, prior_mean, prior_std, residual,
                        normalized, availability, age):
        del prior_state, prior_mean, prior_std, normalized, availability, age
        logits = residual.new_full(
            (residual.shape[0], self.config.sensor_dim, self.config.health_classes),
            -torch.inf,
        )
        logits[..., class_index] = 0.0
        return logits

    model._gate_logits = MethodType(constant_logits, model)


def test_bypass_is_exact_and_independent_of_gate_parameters():
    seed = 17
    model = make_model(seed)
    inputs = step_inputs(model)
    first = model.filter_step(*inputs, gate_mode="bypass", sample=False)
    assert torch.equal(
        first.observation_reliability,
        inputs[3].to(first.observation_reliability.dtype),
    )

    torch.manual_seed(seed)
    base = HealthAwareRSSM(
        RSSMConfig(
            observation_dim=4,
            action_dim=2,
            sensor_dim=2,
            constraint_dim=1,
            deterministic_dim=9,
            stochastic_dim=5,
            embedding_dim=7,
            hidden_dim=13,
            health_classes=5,
            context_dim=3,
        )
    )
    base_step = base.observe_step(*inputs, sample=False)
    torch.testing.assert_close(
        first.rssm_step.posterior_mean,
        base_step.posterior_mean,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        first.rssm_step.prediction.observation_mean,
        base_step.prediction.observation_mean,
        rtol=0.0, atol=0.0,
    )

    with torch.no_grad():
        for parameter in model.reliability_feature_net.parameters():
            parameter.fill_(123.0)
        for parameter in model.health_head.parameters():
            parameter.fill_(-77.0)
    second = model.filter_step(*inputs, gate_mode="bypass", sample=False)
    torch.testing.assert_close(
        first.rssm_step.posterior_mean, second.rssm_step.posterior_mean,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        first.rssm_step.prediction.observation_mean,
        second.rssm_step.prediction.observation_mean,
        rtol=0.0, atol=0.0,
    )


def test_forced_healthy_learned_gate_equals_bypass():
    model = make_model()
    inputs = step_inputs(model)
    bypass = model.filter_step(*inputs, gate_mode="bypass", sample=False)
    force_gate_class(model, model.config.healthy_class_index)
    learned = model.filter_step(*inputs, gate_mode="learned", sample=False)
    torch.testing.assert_close(
        learned.observation_reliability,
        bypass.observation_reliability,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        learned.rssm_step.posterior_mean,
        bypass.rssm_step.posterior_mean,
        rtol=0.0, atol=0.0,
    )


def test_forced_zero_reliability_blocks_mapped_measurement_value():
    model = make_model()
    inputs = list(step_inputs(model))
    force_gate_class(model, class_index=1)
    changed = inputs[2].clone()
    changed[:, 0] += 10_000.0
    changed[:, 2] -= 10_000.0

    first = model.filter_step(*inputs, gate_mode="learned", sample=False)
    second_inputs = [*inputs]
    second_inputs[2] = changed
    second = model.filter_step(*second_inputs, gate_mode="learned", sample=False)
    assert torch.equal(
        first.observation_reliability[:, [0, 2]],
        torch.zeros_like(first.observation_reliability[:, [0, 2]]),
    )
    torch.testing.assert_close(
        first.rssm_step.posterior_mean,
        second.rssm_step.posterior_mean,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        first.rssm_step.prediction.observation_mean,
        second.rssm_step.prediction.observation_mean,
        rtol=0.0, atol=0.0,
    )

    prior = model.transition(
        inputs[0], inputs[1], inputs[5], sample=False
    ).prediction.observation_mean
    captured = {}

    def capture_encoder(_module, arguments):
        captured["input"] = arguments[0].detach().clone()

    handle = model.observation_encoder.register_forward_pre_hook(capture_encoder)
    model.filter_step(*inputs, gate_mode="learned", sample=False)
    handle.remove()
    torch.testing.assert_close(
        captured["input"][:, [0, 2]],
        prior[:, [0, 2]],
        rtol=0.0,
        atol=0.0,
    )


def test_nan_is_equivalent_to_explicitly_missing_and_outputs_are_finite():
    model = make_model()
    inputs = list(step_inputs(model))
    implicit_observation = inputs[2].clone()
    implicit_observation[0, 0] = torch.nan
    implicit_inputs = [*inputs]
    implicit_inputs[2] = implicit_observation

    explicit_observation = inputs[2].clone()
    explicit_observation[0, 0] = 999_999.0
    explicit_mask = inputs[3].clone()
    explicit_mask[0, 0] = False
    explicit_inputs = [*inputs]
    explicit_inputs[2] = explicit_observation
    explicit_inputs[3] = explicit_mask

    implicit = model.filter_step(*implicit_inputs, sample=False)
    explicit = model.filter_step(*explicit_inputs, sample=False)
    torch.testing.assert_close(
        implicit.observation_reliability,
        explicit.observation_reliability,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        implicit.rssm_step.posterior_mean,
        explicit.rssm_step.posterior_mean,
        rtol=0.0, atol=0.0,
    )
    for value in (
        implicit.observation_reliability,
        implicit.gate_health_logits,
        implicit.rssm_step.posterior_mean,
        implicit.rssm_step.prediction.observation_mean,
    ):
        assert torch.isfinite(value).all()


def test_filter_and_imagination_api_have_no_label_or_future_observation_input():
    model = make_model()
    forbidden = ("label", "health", "clean", "future_observation")
    for method_name in ("filter_step", "observe_step", "filter", "observe", "imagine"):
        parameters = inspect.signature(getattr(model, method_name)).parameters
        assert not any(
            token in name for name in parameters for token in forbidden
        ), (method_name, tuple(parameters))

    state, action, observation, mask, age, context = step_inputs(model)
    posterior = model.filter_step(
        state, action, observation, mask, age, context, sample=False
    ).rssm_step.state

    class FailsIfCalled(nn.Module):
        def forward(self, value):
            raise AssertionError("imagination accessed the observation encoder")

    model.observation_encoder = FailsIfCalled()
    future_actions = torch.randn(4, state.deterministic.shape[0], model.config.action_dim)
    future_contexts = torch.randn(
        4, state.deterministic.shape[0], model.config.context_dim
    )
    imagined = model.imagine(
        posterior, future_actions, future_contexts, sample=False
    )
    assert imagined.observation_mean.shape == (4, 3, 4)


def test_learned_gate_and_core_receive_gradients():
    model = make_model()
    inputs = step_inputs(model)
    result = model.filter_step(*inputs, gate_mode="learned", sample=False)
    loss = (
        result.rssm_step.prediction.observation_mean.square().mean()
        + result.gate_health_logits.square().mean()
    )
    loss.backward()

    for module_name in (
        "reliability_feature_net",
        "health_head",
        "observation_encoder",
        "posterior_net",
        "observation_head",
    ):
        module = getattr(model, module_name)
        total = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert total > 0.0, module_name


def test_sequence_outputs_and_save_load_are_reproducible():
    model = make_model()
    time_steps, batch = 5, 2
    actions = torch.randn(time_steps, batch, model.config.action_dim)
    observations = torch.randn(time_steps, batch, model.config.observation_dim)
    masks = torch.ones_like(observations, dtype=torch.bool)
    ages = torch.zeros_like(observations)
    contexts = torch.randn(time_steps, batch, model.config.context_dim)
    first = model.filter(
        actions, observations, masks, ages, contexts, sample=False
    )
    assert first.observation_reliability.shape == (time_steps, batch, 4)
    assert first.gate_health_logits.shape == (time_steps, batch, 2, 5)
    torch.testing.assert_close(
        first.rssm.health_logits, first.gate_health_logits,
        rtol=0.0, atol=0.0,
    )

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model(seed=999)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    second = restored.filter(
        actions, observations, masks, ages, contexts, sample=False
    )
    for left, right in (
        (first.observation_reliability, second.observation_reliability),
        (first.gate_health_logits, second.gate_health_logits),
        (first.rssm.posterior_mean, second.rssm.posterior_mean),
        (first.rssm.observation_mean, second.rssm.observation_mean),
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_huber_gate_is_causal_bounded_and_only_changes_mapped_channels():
    model = make_model()
    inputs = list(step_inputs(model))
    extreme = inputs[2].clone()
    extreme[:, 0] += 1_000_000.0
    extreme[:, 2] -= 1_000_000.0
    inputs[2] = extreme
    result = model.filter_step(*inputs, gate_mode="huber", sample=False)
    assert ((result.observation_reliability >= 0.0) &
            (result.observation_reliability <= 1.0)).all()
    assert (result.observation_reliability[:, [0, 2]] < 1e-3).all()
    torch.testing.assert_close(
        result.observation_reliability[:, [1, 3]],
        inputs[3][:, [1, 3]].to(result.observation_reliability.dtype),
        rtol=0.0, atol=0.0,
    )
