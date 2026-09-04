"""Contract tests for RC readiness and held-out evaluation."""

from __future__ import annotations

import unittest

from .config import CASES, CONFIG, MODEL_SEEDS, POLICIES, SILENT_FAMILIES
from .evaluation import (
    DETAIL_COLUMNS,
    PHYSICAL_COLUMNS,
    evaluation_source_identity,
)
from .model import physical_diagnostics


class RcEvaluationContractTests(unittest.TestCase):
    def test_candidate_grid_and_bootstrap_are_fixed(self) -> None:
        per_case = (
            len(CONFIG.topologies)
            * len(CONFIG.equipment_ridge_alphas)
            * len(CONFIG.innovation_clip_sigmas)
        )
        self.assertEqual(per_case, 60)
        self.assertEqual(per_case * len(CASES), 180)
        self.assertEqual(CONFIG.bootstrap_draws, 10_000)

    def test_paired_evaluation_grid_identity(self) -> None:
        self.assertEqual(len(CASES), 3)
        self.assertEqual(len(POLICIES), 2)
        self.assertEqual(len(SILENT_FAMILIES), 3)
        self.assertEqual(len(MODEL_SEEDS), 5)
        self.assertIn("standardized_abs_error", DETAIL_COLUMNS)

    def test_physical_diagnostic_columns_are_complete(self) -> None:
        coefficients = {
            "a_oa": 0.02,
            "a_zm": 0.04,
            "a_mz": 0.01,
            "a_mo": 0.0,
            "hvac_gain_k_per_kw_step": 0.02,
            "solar_gain_k_per_w_m2_step": 0.0002,
            "constant_heat_flow_k_per_step": 0.01,
        }
        self.assertEqual(
            tuple(physical_diagnostics(coefficients)), PHYSICAL_COLUMNS
        )

    def test_readiness_source_identity_is_complete(self) -> None:
        identity = evaluation_source_identity()
        self.assertIn("audit.py", identity)
        self.assertIn("reproducibility.py", identity)
        self.assertIn("publication_corpus_upstream.py", identity)
        self.assertTrue(all(len(value) == 64 for value in identity.values()))


if __name__ == "__main__":
    unittest.main()
