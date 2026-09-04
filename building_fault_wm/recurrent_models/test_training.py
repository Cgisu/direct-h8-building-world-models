"""Unit tests for the masked RSSM sequence objective."""

from dataclasses import replace
import unittest

import torch

from building_fault_wm.recurrent_models.model import HealthAwareRSSM, RSSMConfig
from building_fault_wm.recurrent_models.training import (
    RSSMLossConfig,
    RSSMSequenceInputs,
    RSSMSequenceTargets,
    loss_from_rollout,
    sequence_training_loss,
)


class RecordingHealthAwareRSSM(HealthAwareRSSM):
    """Record transition inputs to audit overshooting index alignment."""

    def __init__(self, config):
        super().__init__(config)
        self.transition_actions = []
        self.transition_contexts = []

    def transition(self, state, action, context=None, *, sample=True):
        self.transition_actions.append(action.detach().clone())
        self.transition_contexts.append(
            None if context is None else context.detach().clone()
        )
        return super().transition(
            state, action, context, sample=sample
        )


class RSSMTrainingLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.time = 5
        self.batch = 3
        self.config = RSSMConfig(
            observation_dim=4,
            action_dim=2,
            sensor_dim=3,
            constraint_dim=2,
            deterministic_dim=9,
            stochastic_dim=5,
            embedding_dim=7,
            hidden_dim=13,
            health_classes=4,
        )
        self.model = HealthAwareRSSM(self.config)
        previous_actions = torch.randn(self.time, self.batch, 2)
        corrupted = torch.randn(self.time, self.batch, 4)
        availability = torch.ones_like(corrupted, dtype=torch.bool)
        age = torch.zeros_like(corrupted)
        availability[2, 1, 0] = False
        corrupted[2, 1, 0] = torch.nan
        age[2, 1, 0] = 3.0
        self.inputs = RSSMSequenceInputs(
            previous_actions=previous_actions,
            corrupted_observations=corrupted,
            availability=availability,
            age=age,
        )

        clean = torch.randn(self.time, self.batch, 4)
        clean_mask = torch.ones_like(clean, dtype=torch.bool)
        clean_mask[0, 0, 1] = False
        clean[0, 0, 1] = torch.nan
        costs = torch.rand(self.time, self.batch, 1)
        cost_mask = torch.ones_like(costs, dtype=torch.bool)
        constraints = torch.randint(0, 2, (self.time, self.batch, 2))
        constraint_mask = torch.ones_like(constraints, dtype=torch.bool)
        continuations = torch.ones(self.time, self.batch, 1, dtype=torch.long)
        continuations[-1] = 0.0
        continuation_mask = torch.ones_like(continuations, dtype=torch.bool)
        health_labels = torch.randint(0, 4, (self.time, self.batch, 3))
        health_mask = torch.ones_like(health_labels, dtype=torch.bool)
        health_labels[0, 0, 0] = 255
        health_mask[0, 0, 0] = False
        valid_steps = torch.ones(self.time, self.batch, dtype=torch.bool)
        valid_steps[-1, 1] = False
        self.targets = RSSMSequenceTargets(
            clean_observations=clean,
            clean_observation_mask=clean_mask,
            costs=costs,
            cost_mask=cost_mask,
            constraints=constraints,
            constraint_mask=constraint_mask,
            continuations=continuations,
            continuation_mask=continuation_mask,
            health_labels=health_labels,
            health_mask=health_mask,
            valid_steps=valid_steps,
        )
        self.loss_config = RSSMLossConfig(
            kl_free_nats=0.0,
            kl_balance=0.6,
            health_ignore_index=255,
            health_class_weights=torch.tensor([1.0, 2.0, 1.5, 0.5]),
        )

    def test_loss_is_finite_and_gradients_reach_every_model_component(self) -> None:
        output = sequence_training_loss(
            self.model,
            self.inputs,
            self.targets,
            self.loss_config,
            sample=True,
        )
        self.assertTrue(torch.isfinite(output.loss.total))
        self.assertEqual(output.loss.valid_counts["steps"], 14)
        self.assertEqual(output.loss.valid_counts["observations"], 55)
        self.assertEqual(output.loss.valid_counts["health"], 41)
        output.loss.total.backward()

        for module_name in (
            "observation_encoder",
            "recurrent",
            "prior_net",
            "posterior_net",
            "observation_head",
            "cost_head",
            "constraint_head",
            "continuation_head",
            "health_head",
        ):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in getattr(self.model, module_name).parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, module_name)

    def test_masked_targets_cannot_change_the_loss(self) -> None:
        rollout = self.model.observe(
            self.inputs.previous_actions,
            self.inputs.corrupted_observations,
            self.inputs.availability,
            self.inputs.age,
            sample=False,
        )
        first = loss_from_rollout(rollout, self.targets, self.loss_config)
        changed_clean = self.targets.clean_observations.clone()
        changed_clean[0, 0, 1] = 1e20
        changed_health = self.targets.health_labels.clone()
        changed_health[0, 0, 0] = 999
        changed = replace(
            self.targets,
            clean_observations=changed_clean,
            health_labels=changed_health,
        )
        second = loss_from_rollout(rollout, changed, self.loss_config)
        torch.testing.assert_close(first.total, second.total)

    def test_supervised_invalid_health_label_is_rejected(self) -> None:
        rollout = self.model.observe(
            self.inputs.previous_actions,
            self.inputs.corrupted_observations,
            self.inputs.availability,
            self.inputs.age,
            sample=False,
        )
        labels = self.targets.health_labels.clone()
        labels[1, 0, 0] = 99
        with self.assertRaisesRegex(ValueError, "outside the class range"):
            loss_from_rollout(
                rollout,
                replace(self.targets, health_labels=labels),
                self.loss_config,
            )

    def test_multistep_overshooting_uses_valid_pairs_and_trains_dynamics(self) -> None:
        torch.manual_seed(77)
        model = RecordingHealthAwareRSSM(
            RSSMConfig(
                observation_dim=2,
                action_dim=1,
                sensor_dim=1,
                constraint_dim=1,
                deterministic_dim=6,
                stochastic_dim=4,
                embedding_dim=5,
                hidden_dim=8,
                health_classes=3,
                context_dim=1,
            )
        )
        time, batch = 5, 2
        inputs = RSSMSequenceInputs(
            previous_actions=torch.randn(time, batch, 1),
            corrupted_observations=torch.randn(time, batch, 2),
            availability=torch.ones(time, batch, 2, dtype=torch.bool),
            age=torch.zeros(time, batch, 2),
            contexts=torch.randn(time, batch, 1),
        )
        valid_steps = torch.ones(time, batch, dtype=torch.bool)
        valid_steps[-1, 1] = False
        targets = RSSMSequenceTargets(
            clean_observations=torch.randn(time, batch, 2),
            clean_observation_mask=torch.ones(
                time, batch, 2, dtype=torch.bool
            ),
            costs=torch.randn(time, batch, 1),
            cost_mask=torch.ones(time, batch, 1, dtype=torch.bool),
            constraints=torch.randint(0, 2, (time, batch, 1)),
            constraint_mask=torch.ones(time, batch, 1, dtype=torch.bool),
            continuations=torch.ones(time, batch, 1),
            continuation_mask=torch.ones(time, batch, 1, dtype=torch.bool),
            health_labels=torch.randint(0, 3, (time, batch, 1)),
            health_mask=torch.ones(time, batch, 1, dtype=torch.bool),
            valid_steps=valid_steps,
        )
        config = RSSMLossConfig(
            observation_weight=0.0,
            cost_weight=0.0,
            constraint_weight=0.0,
            continuation_weight=0.0,
            health_weight=0.0,
            kl_weight=0.0,
            kl_free_nats=0.0,
            overshooting_horizon=3,
            overshooting_weight=1.0,
        )
        output = sequence_training_loss(
            model, inputs, targets, config, sample=False
        )

        # Batch 0 contributes five distance-2/3 pairs. Batch 1 contributes
        # three because an overshoot may not cross its invalid final step.
        self.assertEqual(output.loss.valid_counts["overshooting_pairs"], 8)
        self.assertGreater(float(output.loss.overshooting_kl.detach()), 0.0)
        torch.testing.assert_close(
            output.loss.total, output.loss.overshooting_kl
        )
        # The final eight calls are overshooting transitions. From posterior
        # source 0/1/2 they must consume target indices
        # [1,2,3], [2,3,4], and [3,4], respectively.
        expected_indices = [1, 2, 3, 2, 3, 4, 3, 4]
        for recorded, index in zip(
            model.transition_actions[-8:], expected_indices
        ):
            torch.testing.assert_close(
                recorded, inputs.previous_actions[index]
            )
        for recorded, index in zip(
            model.transition_contexts[-8:], expected_indices
        ):
            torch.testing.assert_close(recorded, inputs.contexts[index])
        output.loss.total.backward()
        for module_name in ("recurrent", "prior_net"):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in getattr(model, module_name).parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, module_name)


if __name__ == "__main__":
    unittest.main()
