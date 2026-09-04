"""Unit tests for the physically structured RC comparator."""

from __future__ import annotations

import unittest

import numpy as np

from building_fault_wm.ridge_arx_sensitivity import study as strong_arx
from .model import (
    RcModel,
    _thermal_coefficients,
    _thermal_start_points,
    parameter_boundary_diagnostics,
    physical_diagnostics,
    restore_model,
    thermal_step,
    transition,
    transition_jacobian,
)


class RcModelTests(unittest.TestCase):
    def _model(self, topology: str = "2r2c") -> RcModel:
        coefficients = np.zeros((14, 3))
        coefficients[0] = (0.1, 0.0, 0.1)
        coefficients[1] = (0.05, 0.02, -0.01)
        coefficients[2] = (0.7, 0.1, 0.0)
        coefficients[3] = (0.0, 0.7, 0.0)
        coefficients[4] = (0.0, 0.0, 0.7)
        return RcModel(
            case="bestest_hydronic_heat_pump",
            topology=topology,
            equipment_ridge_alpha=1.0,
            innovation_clip_sigma=3.0,
            thermal_coefficients={
                "a_oa": 0.02,
                "a_zm": 0.05 if topology == "2r2c" else 0.0,
                "a_mz": 0.04 if topology == "2r2c" else 0.0,
                "a_mo": 0.0,
                "hvac_gain_k_per_kw_step": 0.02,
                "solar_gain_k_per_w_m2_step": 0.0002,
                "constant_heat_flow_k_per_step": 0.01,
            },
            equipment_coefficients=coefficients,
            equipment_lower_standardized=np.full(3, -10.0),
            equipment_upper_standardized=np.full(3, 10.0),
            process_covariance=np.eye(5) * 0.01,
            measurement_covariance=np.eye(4) * 0.0004,
            fit_diagnostics={"fit_temperature_mae_k": 0.1},
        )

    def test_positive_coefficients_map_to_positive_rc_parameters(self) -> None:
        coefficients = _thermal_coefficients("2r2c", np.zeros(6))
        physical = physical_diagnostics(coefficients)
        self.assertGreater(physical["zone_capacity_j_per_k"], 0.0)
        self.assertGreater(physical["outdoor_resistance_k_per_w"], 0.0)
        self.assertGreater(physical["mass_capacity_j_per_k"], 0.0)
        self.assertLess(coefficients["a_oa"] + coefficients["a_zm"], 0.3)
        self.assertLess(coefficients["a_mz"], 0.3)
        self.assertGreaterEqual(coefficients["a_oa"], 1e-5)
        self.assertGreaterEqual(coefficients["a_zm"], 1e-5)
        self.assertEqual(coefficients["a_mo"], 0.0)
        self.assertIsNone(physical["mass_outdoor_resistance_k_per_w"])
        self.assertGreaterEqual(physical["mass_to_zone_capacity_ratio"], 1.0)
        self.assertLessEqual(physical["mass_to_zone_capacity_ratio"], 100.0)
        self.assertGreaterEqual(
            coefficients["hvac_gain_k_per_kw_step"], 1e-5
        )
        self.assertLessEqual(physical["effective_solar_aperture_m2"], 10_000.0)
        self.assertLessEqual(abs(physical["constant_heat_flow_w"]), 1_000_000.0)

    def test_positive_heat_increases_zone_temperature(self) -> None:
        model = self._model()
        no_heat, _ = thermal_step(
            model.topology,
            model.thermal_coefficients,
            295.0,
            295.0,
            295.0,
            0.0,
            0.0,
        )
        with_heat, _ = thermal_step(
            model.topology,
            model.thermal_coefficients,
            295.0,
            295.0,
            295.0,
            0.0,
            10.0,
        )
        self.assertGreater(with_heat, no_heat)

    def test_thermal_multistart_grid_is_fixed(self) -> None:
        self.assertEqual(len(_thermal_start_points("1r1c")), 9)
        self.assertEqual(len(_thermal_start_points("2r2c")), 9)
        self.assertTrue(
            all(
                np.isfinite(value).all()
                for topology in ("1r1c", "2r2c")
                for value in _thermal_start_points(topology)
            )
        )

    def test_boundary_diagnostics_distinguish_terminal_ols(self) -> None:
        model = self._model()
        diagnostics = parameter_boundary_diagnostics(
            RcModel(
                **{
                    **model.__dict__,
                    "equipment_ridge_alpha": 0.0,
                }
            )
        )
        self.assertEqual(
            diagnostics["equipment_regularization_endpoint"],
            "ordinary_least_squares",
        )

    def test_payload_roundtrip(self) -> None:
        original = self._model()
        restored = restore_model(original.payload())
        self.assertEqual(restored.case, original.case)
        self.assertEqual(restored.topology, original.topology)
        np.testing.assert_allclose(
            restored.equipment_coefficients, original.equipment_coefficients
        )

    def test_analytic_transition_jacobian_matches_finite_difference(self) -> None:
        model = self._model()
        scalers = strong_arx.load_frozen_scaler(model.case)
        state = np.asarray([-1.0, -0.5, 0.1, 0.1, 0.1])
        action = np.asarray([0.0])
        context = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
        analytic = transition_jacobian(model, state, action, context, scalers)
        base = transition(model, state, action, context, scalers)
        numerical = np.empty_like(analytic)
        epsilon = 1e-6
        for column in range(5):
            perturbed = state.copy()
            perturbed[column] += epsilon
            numerical[:, column] = (
                transition(model, perturbed, action, context, scalers) - base
            ) / epsilon
        np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
