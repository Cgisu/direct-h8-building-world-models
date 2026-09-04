"""Unit tests for observation-free latent MPC."""

from types import SimpleNamespace
import unittest

import torch

from building_fault_wm.recurrent_models.model import HealthAwareRSSM, RSSMConfig
from building_fault_wm.recurrent_models.planning import LatentMPC, LatentMPCConfig


class EngineeredActionWorldModel(HealthAwareRSSM):
    """RSSM subtype with a known imagined action optimum for planner tests."""

    def __init__(self, target_action: float):
        super().__init__(
            RSSMConfig(
                observation_dim=2,
                action_dim=1,
                sensor_dim=1,
                constraint_dim=1,
                deterministic_dim=3,
                stochastic_dim=2,
                embedding_dim=3,
                hidden_dim=4,
                health_classes=2,
            )
        )
        self.target_action = target_action
        self.imagination_calls = 0

    def imagine(
        self, start_state, actions, future_contexts=None, *, sample=True
    ):
        self.imagination_calls += 1
        del start_state, future_contexts, sample
        cost = (actions - self.target_action).square()
        batch = actions.shape[1]
        return SimpleNamespace(
            cost_mean=cost,
            cost_std=torch.full_like(cost, 0.05),
            constraint_logits=torch.full(
                (actions.shape[0], batch, 1), -12.0, device=actions.device
            ),
            observation_std=torch.full(
                (actions.shape[0], batch, 2), 0.1, device=actions.device
            ),
        )


class ContextTargetWorldModel(HealthAwareRSSM):
    """Known context sets the action optimum and records planner expansion."""

    def __init__(self):
        super().__init__(
            RSSMConfig(
                observation_dim=2,
                action_dim=1,
                sensor_dim=1,
                constraint_dim=1,
                deterministic_dim=3,
                stochastic_dim=2,
                embedding_dim=3,
                hidden_dim=4,
                health_classes=2,
                context_dim=1,
            )
        )
        self.context_calls = []

    def imagine(
        self, start_state, actions, future_contexts=None, *, sample=True
    ):
        del start_state, sample
        if future_contexts is None:
            raise AssertionError("planner omitted known future context")
        self.context_calls.append(future_contexts.detach().clone())
        cost = (actions - future_contexts).square()
        batch = actions.shape[1]
        return SimpleNamespace(
            cost_mean=cost,
            cost_std=torch.full_like(cost, 0.01),
            constraint_logits=torch.full(
                (actions.shape[0], batch, 1), -12.0, device=actions.device
            ),
            observation_std=torch.full(
                (actions.shape[0], batch, 2), 0.01, device=actions.device
            ),
        )


class LatentMPCTest(unittest.TestCase):
    def _plan(self, target: float):
        model = EngineeredActionWorldModel(target)
        planner = LatentMPC(
            model,
            action_low=-1.0,
            action_high=1.0,
            config=LatentMPCConfig(
                horizon=4,
                candidates=192,
                iterations=4,
                elite_fraction=0.1,
                constraint_weight=2.0,
                cost_uncertainty_weight=0.2,
                observation_uncertainty_weight=0.1,
            ),
        )
        generator = torch.Generator().manual_seed(2026)
        result = planner.plan(model.initial(2), generator=generator)
        return model, result

    def test_imagined_action_consequences_change_the_selected_action(self) -> None:
        negative_model, negative = self._plan(-0.7)
        positive_model, positive = self._plan(0.7)

        self.assertEqual(negative_model.imagination_calls, 4)
        self.assertEqual(positive_model.imagination_calls, 4)
        self.assertTrue(torch.all(negative.first_action < -0.45))
        self.assertTrue(torch.all(positive.first_action > 0.45))
        self.assertGreater(
            float((positive.first_action - negative.first_action).min()), 0.9
        )

    def test_plan_is_bounded_batched_and_has_finite_risk_breakdown(self) -> None:
        _, result = self._plan(0.3)
        self.assertEqual(result.first_action.shape, (2, 1))
        self.assertEqual(result.action_sequence.shape, (2, 4, 1))
        self.assertTrue(torch.all(result.action_sequence >= -1.0))
        self.assertTrue(torch.all(result.action_sequence <= 1.0))
        for value in (
            result.objective,
            result.predicted_cost,
            result.predicted_constraint_risk,
            result.predicted_uncertainty,
        ):
            self.assertEqual(value.shape, (2,))
            self.assertTrue(torch.isfinite(value).all())

    def test_cem_holds_each_batch_context_fixed_across_candidates(self) -> None:
        model = ContextTargetWorldModel()
        config = LatentMPCConfig(
            horizon=3,
            candidates=96,
            iterations=3,
            elite_fraction=0.1,
            constraint_weight=0.0,
            cost_uncertainty_weight=0.0,
            observation_uncertainty_weight=0.0,
        )
        planner = LatentMPC(model, -1.0, 1.0, config)
        contexts = torch.tensor(
            [[[-0.7], [0.7]], [[-0.6], [0.6]], [[-0.5], [0.5]]]
        )
        result = planner.plan(
            model.initial(2),
            future_contexts=contexts,
            generator=torch.Generator().manual_seed(44),
        )

        self.assertLess(float(result.first_action[0, 0]), -0.4)
        self.assertGreater(float(result.first_action[1, 0]), 0.4)
        self.assertEqual(len(model.context_calls), config.iterations)
        expected = contexts[:, :, None, :].expand(
            config.horizon, 2, config.candidates, 1
        )
        for expanded in model.context_calls:
            torch.testing.assert_close(
                expanded.reshape(config.horizon, 2, config.candidates, 1),
                expected,
            )

        with self.assertRaisesRegex(ValueError, "future_contexts is required"):
            planner.plan(model.initial(2))


if __name__ == "__main__":
    unittest.main()
