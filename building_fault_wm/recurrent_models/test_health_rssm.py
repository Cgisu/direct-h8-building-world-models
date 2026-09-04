"""Synthetic invariant tests for the health-aware RSSM core."""

import copy
import unittest

import torch
import torch.nn as nn

from building_fault_wm.recurrent_models.model import (
    BeliefController,
    HealthAwareRSSM,
    RSSMConfig,
)


class HealthAwareRSSMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch = 3
        self.config = RSSMConfig(
            observation_dim=5,
            action_dim=2,
            sensor_dim=4,
            constraint_dim=2,
            deterministic_dim=8,
            stochastic_dim=6,
            embedding_dim=7,
            hidden_dim=11,
            health_classes=5,
        )
        self.model = HealthAwareRSSM(self.config)
        self.state = self.model.initial(self.batch)
        self.action = torch.randn(self.batch, self.config.action_dim)
        self.observation = torch.randn(self.batch, self.config.observation_dim)
        self.mask = torch.ones_like(self.observation)
        self.age = torch.zeros_like(self.observation)

    def test_shapes_and_belief_only_controller(self) -> None:
        step = self.model.observe_step(
            self.state,
            self.action,
            self.observation,
            self.mask,
            self.age,
            sample=False,
        )
        self.assertEqual(step.state.deterministic.shape, (self.batch, 8))
        self.assertEqual(step.state.stochastic.shape, (self.batch, 6))
        self.assertEqual(step.prior_mean.shape, (self.batch, 6))
        self.assertEqual(step.posterior_mean.shape, (self.batch, 6))
        self.assertEqual(step.prediction.observation_mean.shape, (self.batch, 5))
        self.assertEqual(step.prediction.cost_mean.shape, (self.batch, 1))
        self.assertEqual(step.prediction.constraint_logits.shape, (self.batch, 2))
        self.assertEqual(step.prediction.continuation_logit.shape, (self.batch, 1))
        self.assertEqual(step.prediction.health_logits.shape, (self.batch, 4, 5))

        controller = BeliefController(self.model.feature_dim, self.config.action_dim, 9)
        controller_action = controller(step.state)
        self.assertEqual(controller_action.shape, (self.batch, 2))
        self.assertTrue(torch.all(controller_action.abs() <= 1.0))

    def test_actions_change_the_open_loop_prior(self) -> None:
        zero_action = torch.zeros(self.batch, self.config.action_dim)
        one_action = torch.ones(self.batch, self.config.action_dim)
        zero_step = self.model.transition(self.state, zero_action, sample=False)
        one_step = self.model.transition(self.state, one_action, sample=False)
        difference = (zero_step.prior_mean - one_step.prior_mean).abs().max()
        self.assertGreater(float(difference.detach()), 1e-6)

        differentiable_action = torch.zeros_like(zero_action, requires_grad=True)
        action_step = self.model.transition(
            self.state, differentiable_action, sample=False
        )
        action_step.prior_mean.sum().backward()
        self.assertGreater(float(differentiable_action.grad.abs().sum()), 0.0)

    def test_imagination_never_calls_observation_encoder(self) -> None:
        observed = self.model.observe_step(
            self.state,
            self.action,
            self.observation,
            self.mask,
            self.age,
            sample=False,
        )

        class FailsIfCalled(nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                raise AssertionError("open-loop imagination read an observation")

        self.model.observation_encoder = FailsIfCalled()
        actions = torch.randn(4, self.batch, self.config.action_dim)
        rollout = self.model.imagine(observed.state, actions, sample=False)
        self.assertEqual(rollout.deterministic.shape, (4, self.batch, 8))
        self.assertEqual(rollout.observation_mean.shape, (4, self.batch, 5))
        self.assertEqual(rollout.cost_mean.shape, (4, self.batch, 1))
        self.assertIsNone(rollout.posterior_mean)

    def test_gradients_reach_filter_dynamics_and_heads(self) -> None:
        observation = self.observation.clone().requires_grad_(True)
        action = self.action.clone().requires_grad_(True)
        step = self.model.observe_step(
            self.state,
            action,
            observation,
            self.mask,
            self.age,
            sample=True,
        )
        prediction = step.prediction
        loss = (
            prediction.observation_mean.square().mean()
            + prediction.observation_std.mean()
            + prediction.cost_mean.square().mean()
            + prediction.cost_std.mean()
            + prediction.constraint_logits.square().mean()
            + prediction.continuation_logit.square().mean()
            + prediction.health_logits.square().mean()
            + 0.1 * self.model.kl_divergence(step).mean()
        )
        loss.backward()

        self.assertGreater(float(observation.grad.abs().sum()), 0.0)
        self.assertGreater(float(action.grad.abs().sum()), 0.0)
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
            module = getattr(self.model, module_name)
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, module_name)

    def test_seeded_sampling_and_sequence_shapes_are_reproducible(self) -> None:
        model_copy = copy.deepcopy(self.model)
        torch.manual_seed(1234)
        first = self.model.observe_step(
            self.state,
            self.action,
            self.observation,
            self.mask,
            self.age,
            sample=True,
        )
        torch.manual_seed(1234)
        second = model_copy.observe_step(
            self.state,
            self.action,
            self.observation,
            self.mask,
            self.age,
            sample=True,
        )
        torch.testing.assert_close(first.state.stochastic, second.state.stochastic)

        horizon = 4
        actions = torch.randn(horizon, self.batch, self.config.action_dim)
        observations = torch.randn(horizon, self.batch, self.config.observation_dim)
        masks = torch.ones_like(observations)
        ages = torch.zeros_like(observations)
        rollout = self.model.observe(
            actions, observations, masks, ages, sample=False
        )
        self.assertEqual(rollout.posterior_mean.shape, (horizon, self.batch, 6))
        self.assertEqual(rollout.health_logits.shape, (horizon, self.batch, 4, 5))
        self.assertEqual(rollout.final_state.stochastic.shape, (self.batch, 6))

    def test_nonfinite_observation_is_missing_and_age_is_validated(self) -> None:
        nonfinite = self.observation.clone()
        nonfinite[0, 0] = torch.nan
        externally_available = self.mask.clone()
        explicitly_missing = self.mask.clone()
        explicitly_missing[0, 0] = 0.0
        arbitrary_missing_value = self.observation.clone()
        arbitrary_missing_value[0, 0] = 123.0

        implicit = self.model.observe_step(
            self.state,
            self.action,
            nonfinite,
            externally_available,
            self.age,
            sample=False,
        )
        explicit = self.model.observe_step(
            self.state,
            self.action,
            arbitrary_missing_value,
            explicitly_missing,
            self.age,
            sample=False,
        )
        torch.testing.assert_close(implicit.posterior_mean, explicit.posterior_mean)

        negative_age = self.age.clone()
        negative_age[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "age must be nonnegative"):
            self.model.observe_step(
                self.state,
                self.action,
                self.observation,
                self.mask,
                negative_age,
            )
        nonfinite_age = self.age.clone()
        nonfinite_age[0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "age must be finite"):
            self.model.observe_step(
                self.state,
                self.action,
                self.observation,
                self.mask,
                nonfinite_age,
            )

    def test_context_changes_prior_and_sequence_indices_have_no_shift(self) -> None:
        torch.manual_seed(31)
        config = RSSMConfig(
            observation_dim=3,
            action_dim=2,
            sensor_dim=2,
            constraint_dim=1,
            deterministic_dim=7,
            stochastic_dim=4,
            embedding_dim=6,
            hidden_dim=9,
            health_classes=3,
            context_dim=2,
        )
        model = HealthAwareRSSM(config)
        batch = 2
        state = model.initial(batch)
        action = torch.zeros(batch, 2)
        zero_context = torch.zeros(batch, 2)
        one_context = torch.ones(batch, 2)
        zero_step = model.transition(
            state, action, zero_context, sample=False
        )
        one_step = model.transition(
            state, action, one_context, sample=False
        )
        self.assertGreater(
            float(
                (zero_step.prior_mean - one_step.prior_mean)
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )
        with self.assertRaisesRegex(ValueError, "context is required"):
            model.transition(state, action)

        time = 4
        previous_actions = torch.zeros(time, batch, 2)
        contexts = torch.zeros(time, batch, 2)
        observations = torch.zeros(time, batch, 3)
        masks = torch.ones_like(observations)
        ages = torch.zeros_like(observations)
        baseline = model.observe(
            previous_actions,
            observations,
            masks,
            ages,
            contexts,
            sample=False,
        )

        # Observed index 2 represents posterior y_2 and therefore consumes
        # previous_actions[2] = a_1 together with contexts[2] = c_2.
        action_impulse = previous_actions.clone()
        action_impulse[2, :, 0] = 1.0
        action_changed = model.observe(
            action_impulse,
            observations,
            masks,
            ages,
            contexts,
            sample=False,
        )
        torch.testing.assert_close(
            baseline.prior_mean[:2], action_changed.prior_mean[:2]
        )
        self.assertGreater(
            float(
                (baseline.prior_mean[2] - action_changed.prior_mean[2])
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )
        context_impulse = contexts.clone()
        context_impulse[2, :, 0] = 1.0
        context_changed = model.observe(
            previous_actions,
            observations,
            masks,
            ages,
            context_impulse,
            sample=False,
        )
        torch.testing.assert_close(
            baseline.prior_mean[:2], context_changed.prior_mean[:2]
        )
        self.assertGreater(
            float(
                (baseline.prior_mean[2] - context_changed.prior_mean[2])
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )

        # From posterior t, imagined index 1 consumes a_(t+1), c_(t+2);
        # an impulse there cannot alter the index-0 prediction for t+1.
        future_actions = torch.zeros(3, batch, 2)
        future_contexts = torch.zeros(3, batch, 2)
        imagined = model.imagine(
            state, future_actions, future_contexts, sample=False
        )
        future_action_impulse = future_actions.clone()
        future_action_impulse[1, :, 0] = 1.0
        imagined_action_changed = model.imagine(
            state, future_action_impulse, future_contexts, sample=False
        )
        torch.testing.assert_close(
            imagined.prior_mean[0], imagined_action_changed.prior_mean[0]
        )
        self.assertGreater(
            float(
                (
                    imagined.prior_mean[1]
                    - imagined_action_changed.prior_mean[1]
                )
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )
        future_context_impulse = future_contexts.clone()
        future_context_impulse[1, :, 0] = 1.0
        imagined_context_changed = model.imagine(
            state, future_actions, future_context_impulse, sample=False
        )
        torch.testing.assert_close(
            imagined.prior_mean[0], imagined_context_changed.prior_mean[0]
        )
        self.assertGreater(
            float(
                (
                    imagined.prior_mean[1]
                    - imagined_context_changed.prior_mean[1]
                )
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
