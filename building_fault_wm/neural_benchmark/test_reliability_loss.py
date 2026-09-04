"""Alignment and gradient tests for the direct H8 RSSM objective."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from building_fault_wm.recurrent_models.training import (
    RSSMSequenceInputs,
    RSSMSequenceTargets,
)
from building_fault_wm.neural_benchmark.reliability_loss import (
    ReliabilityLossConfig,
    direct_h8_observation_loss,
    reliability_sequence_training_loss,
)
from building_fault_wm.neural_benchmark.reliability_model import (
    ReliabilityGatedRSSM,
    ReliabilityRSSMConfig,
)


class RecordingReliabilityRSSM(ReliabilityGatedRSSM):
    def __init__(self, config):
        super().__init__(config)
        self.imagination_calls = []

    def imagine(self, start_state, actions, future_contexts=None, *, sample=True):
        result = super().imagine(
            start_state,
            actions,
            future_contexts,
            sample=sample,
        )
        self.imagination_calls.append(
            {
                "start_state": start_state,
                "actions": actions.detach().clone(),
                "contexts": (
                    None
                    if future_contexts is None
                    else future_contexts.detach().clone()
                ),
                "sample": sample,
                "observation_mean": result.observation_mean.detach().clone(),
            }
        )
        return result


def make_model(*, recording: bool = False, seed: int = 29):
    torch.manual_seed(seed)
    model_class = RecordingReliabilityRSSM if recording else ReliabilityGatedRSSM
    return model_class(
        ReliabilityRSSMConfig(
            observation_dim=3,
            action_dim=2,
            sensor_dim=2,
            constraint_dim=1,
            deterministic_dim=8,
            stochastic_dim=4,
            embedding_dim=7,
            hidden_dim=11,
            health_classes=3,
            context_dim=2,
            sensor_observation_indices=(0, 2),
        )
    )


def make_batch(model, *, time_steps: int = 10, batch_size: int = 2):
    torch.manual_seed(101)
    config = model.config
    previous_actions = torch.arange(
        time_steps * batch_size * config.action_dim, dtype=torch.float32
    ).reshape(time_steps, batch_size, config.action_dim) / 10.0
    contexts = torch.arange(
        time_steps * batch_size * config.context_dim, dtype=torch.float32
    ).reshape(time_steps, batch_size, config.context_dim) / 20.0
    corrupted = torch.randn(time_steps, batch_size, config.observation_dim)
    availability = torch.ones_like(corrupted, dtype=torch.bool)
    age = torch.zeros_like(corrupted)
    clean = torch.randn(time_steps, batch_size, config.observation_dim)
    valid = torch.ones(time_steps, batch_size, dtype=torch.bool)
    health = torch.randint(
        0,
        config.health_classes,
        (time_steps, batch_size, config.sensor_dim),
    )
    inputs = RSSMSequenceInputs(
        previous_actions=previous_actions,
        corrupted_observations=corrupted,
        availability=availability,
        age=age,
        contexts=contexts,
    )
    targets = RSSMSequenceTargets(
        clean_observations=clean,
        clean_observation_mask=torch.ones_like(clean, dtype=torch.bool),
        costs=torch.zeros(time_steps, batch_size, 1),
        cost_mask=torch.zeros(time_steps, batch_size, 1, dtype=torch.bool),
        constraints=torch.zeros(time_steps, batch_size, 1),
        constraint_mask=torch.zeros(
            time_steps, batch_size, 1, dtype=torch.bool
        ),
        continuations=torch.zeros(time_steps, batch_size, 1),
        continuation_mask=torch.zeros(
            time_steps, batch_size, 1, dtype=torch.bool
        ),
        health_labels=health,
        health_mask=torch.ones_like(health, dtype=torch.bool),
        valid_steps=valid,
    )
    return inputs, targets


def loss_config(**changes):
    base = ReliabilityLossConfig(
        observation_weight=1.0,
        cost_weight=0.0,
        constraint_weight=0.0,
        continuation_weight=0.0,
        health_weight=0.25,
        kl_weight=0.1,
        kl_free_nats=0.0,
        overshooting_horizon=8,
        overshooting_weight=0.0,
        direct_horizon_weight=1.0,
    )
    return replace(base, **changes)


def test_h8_uses_exact_next_actions_contexts_and_endpoint_target():
    model = make_model(recording=True)
    inputs, targets = make_batch(model)
    output = reliability_sequence_training_loss(
        model,
        inputs,
        targets,
        loss_config(),
        gate_mode="learned",
        sample=False,
    )
    assert len(model.imagination_calls) == 1
    call = model.imagination_calls[0]
    source_count, batch_size = 2, 2
    expected_actions = torch.stack(
        [
            inputs.previous_actions[offset : offset + source_count]
            for offset in range(1, 9)
        ]
    ).reshape(8, source_count * batch_size, 2)
    expected_contexts = torch.stack(
        [inputs.contexts[offset : offset + source_count] for offset in range(1, 9)]
    ).reshape(8, source_count * batch_size, 2)
    torch.testing.assert_close(call["actions"], expected_actions, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        call["contexts"], expected_contexts, rtol=0.0, atol=0.0
    )
    assert call["sample"] is False
    assert call["start_state"].deterministic.requires_grad
    assert output.loss.valid_counts["direct_h8_paths"] == 4
    assert output.loss.valid_counts["direct_h8_targets"] == 12

    changed_early = replace(
        targets, clean_observations=targets.clean_observations.clone()
    )
    changed_early.clean_observations[7] += 100.0
    early_loss, _, _ = direct_h8_observation_loss(
        model, output.rollout, inputs, changed_early, loss_config()
    )
    torch.testing.assert_close(
        early_loss,
        output.loss.direct_horizon_smooth_l1,
        rtol=0.0,
        atol=0.0,
    )
    changed_endpoint = replace(
        targets, clean_observations=targets.clean_observations.clone()
    )
    changed_endpoint.clean_observations[8] += 100.0
    endpoint_loss, _, _ = direct_h8_observation_loss(
        model, output.rollout, inputs, changed_endpoint, loss_config()
    )
    assert not torch.equal(endpoint_loss, output.loss.direct_horizon_smooth_l1)


def test_path_mask_rejects_every_source_crossing_one_invalid_step():
    model = make_model()
    inputs, targets = make_batch(model)
    valid = targets.valid_steps.clone()
    valid[4, 0] = False
    targets = replace(targets, valid_steps=valid)
    output = reliability_sequence_training_loss(
        model, inputs, targets, loss_config(), sample=False
    )
    # Both batch-0 paths, 0..8 and 1..9, cross invalid index 4.
    assert output.loss.valid_counts["direct_h8_paths"] == 2
    assert output.loss.valid_counts["direct_h8_targets"] == 6


def test_targets_change_loss_but_cannot_change_filtered_or_imagined_predictions():
    model = make_model(recording=True)
    inputs, targets = make_batch(model)
    first = reliability_sequence_training_loss(
        model, inputs, targets, loss_config(), sample=False
    )
    changed_clean = targets.clean_observations.clone()
    changed_clean[8:] += 50.0
    changed_targets = replace(targets, clean_observations=changed_clean)
    second = reliability_sequence_training_loss(
        model, inputs, changed_targets, loss_config(), sample=False
    )
    assert not torch.equal(
        first.loss.direct_horizon_smooth_l1,
        second.loss.direct_horizon_smooth_l1,
    )
    for left, right in (
        (first.rollout.rssm.posterior_mean, second.rollout.rssm.posterior_mean),
        (first.rollout.rssm.observation_mean, second.rollout.rssm.observation_mean),
        (
            first.rollout.observation_reliability,
            second.rollout.observation_reliability,
        ),
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert len(model.imagination_calls) == 2
    torch.testing.assert_close(
        model.imagination_calls[0]["observation_mean"],
        model.imagination_calls[1]["observation_mean"],
        rtol=0.0,
        atol=0.0,
    )


def test_direct_h8_gradient_reaches_gate_filter_dynamics_and_decoder():
    model = make_model()
    inputs, targets = make_batch(model, time_steps=12)
    config = loss_config(
        observation_weight=0.0,
        health_weight=0.0,
        kl_weight=0.0,
        overshooting_weight=0.0,
    )
    output = reliability_sequence_training_loss(
        model,
        inputs,
        targets,
        config,
        gate_mode="learned",
        sample=False,
    )
    output.loss.total.backward()
    for module_name in (
        "reliability_feature_net",
        "health_head",
        "observation_encoder",
        "posterior_net",
        "recurrent",
        "prior_net",
        "observation_head",
    ):
        module = getattr(model, module_name)
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert gradient > 0.0, module_name


def test_generic_health_ce_is_computed_from_preposterior_gate_logits():
    model = make_model()
    inputs, targets = make_batch(model)
    output = reliability_sequence_training_loss(
        model,
        inputs,
        targets,
        loss_config(
            observation_weight=0.0,
            health_weight=1.0,
            kl_weight=0.0,
            direct_horizon_weight=0.0,
        ),
        sample=False,
    )
    torch.testing.assert_close(
        output.rollout.rssm.health_logits,
        output.rollout.gate_health_logits,
        rtol=0.0,
        atol=0.0,
    )
    expected = F.cross_entropy(
        output.rollout.gate_health_logits.reshape(-1, model.config.health_classes),
        targets.health_labels.reshape(-1),
    )
    torch.testing.assert_close(output.loss.ordinary.health_ce, expected)
    output.loss.total.backward()
    for module_name in ("reliability_feature_net", "health_head"):
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in getattr(model, module_name).parameters()
            if parameter.grad is not None
        )
        assert gradient > 0.0, module_name


def test_optional_latent_overshooting_is_finite_and_counted():
    model = make_model()
    inputs, targets = make_batch(model)
    output = reliability_sequence_training_loss(
        model,
        inputs,
        targets,
        loss_config(overshooting_weight=0.1),
        sample=False,
    )
    assert torch.isfinite(output.loss.latent_overshooting_kl)
    assert output.loss.valid_counts["overshooting_pairs"] > 0
    expected = (
        output.loss.ordinary.total
        + 0.1 * output.loss.latent_overshooting_kl
        + output.loss.direct_horizon_smooth_l1
    )
    torch.testing.assert_close(output.loss.total, expected)
