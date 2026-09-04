"""Dependency-light checks for the multi-case downstream design."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from building_fault_wm.downstream_control import experiment as v1

from . import protocol
from .controllers import PriceAwareRuleController


class ProtocolTests(unittest.TestCase):
    def test_development_and_final_days_are_disjoint(self) -> None:
        for case in protocol.CASES:
            development = {
                int(item["day"]) for item in protocol.load_development_windows(case)
            }
            final = {int(item["day"]) for item in protocol.load_final_windows(case)}
            self.assertEqual(len(development), 8)
            self.assertEqual(len(final), 12)
            self.assertTrue(development.isdisjoint(final))

    def test_operational_budgets_ignore_policy_outputs(self) -> None:
        rows = protocol.EPISODE_STEPS
        frame = pd.DataFrame(
            {
                "control_stage": [False] * protocol.HISTORY_STEPS
                + [True] * (rows - protocol.HISTORY_STEPS),
                "outcome_electricity_price": np.linspace(0.05, 0.30, rows),
                "outcome_comfort_lower_k": np.full(rows, 294.15),
                "outcome_comfort_upper_k": np.full(rows, 297.15),
                "outcome_hvac_electric_power_w": np.linspace(0.0, 1e6, rows),
                "outcome_zone_temperature_k": np.linspace(270.0, 330.0, rows),
            }
        )
        first = protocol.score_budgets(frame)
        frame["outcome_hvac_electric_power_w"] *= 100.0
        frame["outcome_zone_temperature_k"] *= 0.5
        self.assertEqual(first, protocol.score_budgets(frame))

    def test_scalar_score_has_fixed_equal_weights(self) -> None:
        self.assertAlmostEqual(
            protocol.scalar_score(3.0, 7.0, 1.0, 3.0, 2.0, 4.0),
            1.0,
        )

    def test_rule_direction_matches_case_actuation(self) -> None:
        original = v1.protocol
        try:
            parameters = protocol.RuleParameters(0.5, 0.5, 0.2)
            future = np.asarray(
                [[280.0, 0.0, 294.0, 297.0, 0.1]] * protocol.CONTROL_HORIZON
            )
            v1.protocol = SimpleNamespace(CASE="bestest_hydronic_heat_pump")
            heating = PriceAwareRuleController(parameters)
            heating.update(
                np.asarray([293.0, 1000.0, 0.0, 0.0]),
                0.0,
                future[0],
            )
            self.assertEqual(heating._selected_action(future), 1.0)

            v1.protocol = SimpleNamespace(CASE="multizone_office_simple_air")
            cooling = PriceAwareRuleController(parameters)
            cooling.update(
                np.asarray([298.0, 1000.0, 0.0, 0.0]),
                0.0,
                future[0],
            )
            self.assertEqual(cooling._selected_action(future), -1.0)
        finally:
            v1.protocol = original

    def test_case_inputs_exist_and_hash(self) -> None:
        for case in protocol.CASES:
            inputs = protocol.case_inputs(case)
            self.assertEqual(inputs["boptest_fmu"]["sha256"], protocol.sha256_file(protocol.fmu_path(case)))
            self.assertEqual(len(inputs["rssm_checkpoints"]), 10)
            self.assertEqual(len(inputs["deterministic_checkpoints"]), 5)


if __name__ == "__main__":
    unittest.main()
