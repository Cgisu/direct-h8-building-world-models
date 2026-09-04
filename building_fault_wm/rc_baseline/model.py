"""Physically structured RC thermal model with an empirical equipment map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.fault_data import FaultScalers
from .config import CONFIG


MODEL_SCHEMA = "reviewer-rc-grey-box-model-v1"
STATE_DIM = 5
OBSERVATION_DIM = 4
EQUIPMENT_DIM = 3


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _positive_pair(raw: np.ndarray, maximum_sum: float) -> tuple[float, float]:
    weights = np.exp(np.clip(raw, -30.0, 30.0))
    denominator = 1.0 + float(weights.sum())
    minimum = CONFIG.minimum_active_conductance_coefficient
    available = maximum_sum - 2.0 * minimum
    return tuple(
        float(minimum + available * value / denominator) for value in weights
    )


def _inverse(values: np.ndarray, mean: Sequence[float], scale: Sequence[float]) -> np.ndarray:
    return values * np.asarray(scale, dtype=float) + np.asarray(mean, dtype=float)


def _thermal_coefficients(topology: str, raw: np.ndarray) -> dict[str, float]:
    if topology == "1r1c":
        a_oa = float(
            CONFIG.minimum_active_conductance_coefficient
            + (
                CONFIG.maximum_zone_coefficient_sum
                - CONFIG.minimum_active_conductance_coefficient
            )
            * _sigmoid(raw[0])
        )
        offset = 1
        a_zm = a_mz = a_mo = 0.0
    elif topology == "2r2c":
        a_oa, a_zm = _positive_pair(
            raw[:2], CONFIG.maximum_zone_coefficient_sum
        )
        capacity_ratio = float(
            CONFIG.minimum_mass_to_zone_capacity_ratio
            + (
                CONFIG.maximum_mass_to_zone_capacity_ratio
                - CONFIG.minimum_mass_to_zone_capacity_ratio
            )
            * _sigmoid(raw[2])
        )
        a_mz = a_zm / capacity_ratio
        a_mo = 0.0
        offset = 3
    else:
        raise ValueError(f"unknown RC topology: {topology}")
    hvac_gain = float(
        CONFIG.minimum_hvac_gain_k_per_kw_step
        + (
            CONFIG.maximum_hvac_gain_k_per_kw_step
            - CONFIG.minimum_hvac_gain_k_per_kw_step
        )
        * _sigmoid(raw[offset])
    )
    solar_aperture = float(
        CONFIG.maximum_effective_solar_aperture_m2 * _sigmoid(raw[offset + 1])
    )
    constant_heat_flow_kw = float(
        CONFIG.maximum_constant_heat_flow_kw * np.tanh(raw[offset + 2])
    )
    solar_gain = hvac_gain * solar_aperture / 1000.0
    constant_heat_flow = hvac_gain * constant_heat_flow_kw
    return {
        "a_oa": a_oa,
        "a_zm": a_zm,
        "a_mz": a_mz,
        "a_mo": a_mo,
        "hvac_gain_k_per_kw_step": hvac_gain,
        "solar_gain_k_per_w_m2_step": solar_gain,
        "constant_heat_flow_k_per_step": constant_heat_flow,
    }


def _initial_raw(topology: str) -> np.ndarray:
    def log_weight(value: float, remainder: float) -> float:
        return float(
            np.log(
                (value - CONFIG.minimum_active_conductance_coefficient)
                / remainder
            )
        )

    if topology == "1r1c":
        base = [
            float(
                np.log(
                    (0.02 - CONFIG.minimum_active_conductance_coefficient)
                    / (CONFIG.maximum_zone_coefficient_sum - 0.02)
                )
            )
        ]
    elif topology == "2r2c":
        base = [
            log_weight(0.02, CONFIG.maximum_zone_coefficient_sum - 0.07),
            log_weight(0.05, CONFIG.maximum_zone_coefficient_sum - 0.07),
            float(
                np.log(
                    (10.0 - CONFIG.minimum_mass_to_zone_capacity_ratio)
                    / (CONFIG.maximum_mass_to_zone_capacity_ratio - 10.0)
                )
            ),
        ]
    else:
        raise ValueError(f"unknown RC topology: {topology}")
    base.extend(
        [
            float(
                np.log(
                    (0.02 - CONFIG.minimum_hvac_gain_k_per_kw_step)
                    / (CONFIG.maximum_hvac_gain_k_per_kw_step - 0.02)
                )
            ),
            float(
                np.log(
                    20.0
                    / (CONFIG.maximum_effective_solar_aperture_m2 - 20.0)
                )
            ),
            0.0,
        ]
    )
    return np.asarray(base, dtype=float)


def _thermal_start_points(topology: str) -> tuple[np.ndarray, ...]:
    """Return deterministic, topology-aware starts for the bounded fit."""

    base = _initial_raw(topology)
    offsets = [np.zeros_like(base)]
    if topology == "1r1c":
        for index, magnitude in ((0, 2.0), (1, 2.0), (2, 2.0), (3, 1.0)):
            for sign in (-1.0, 1.0):
                value = np.zeros_like(base)
                value[index] = sign * magnitude
                offsets.append(value)
    elif topology == "2r2c":
        for first, second in ((2.0, -2.0), (-2.0, 2.0)):
            value = np.zeros_like(base)
            value[0] = first
            value[1] = second
            offsets.append(value)
        for index in (2, 3, 4):
            for sign in (-1.0, 1.0):
                value = np.zeros_like(base)
                value[index] = sign * 2.0
                offsets.append(value)
    else:
        raise ValueError(f"unknown RC topology: {topology}")
    starts = tuple(base + offset for offset in offsets)
    if len(starts) != CONFIG.thermal_multistart_count:
        raise ValueError("RC thermal multistart grid changed")
    return starts


def thermal_input_kw(case: str, observation: np.ndarray, zone_temperature: float) -> float:
    """Return delivered sensible heat in kW, positive into the zone."""

    if case == "multizone_office_simple_air":
        flow_m3_s = max(float(observation[2]), 0.0)
        supply_temperature = float(observation[3])
        return 1.2 * 1.006 * flow_m3_s * (supply_temperature - zone_temperature)
    return float(observation[3]) / 1000.0


def physical_diagnostics(coefficients: Mapping[str, float]) -> dict[str, float | None]:
    """Map discrete coefficients to an interpretable RC parameterization."""

    gain = float(coefficients["hvac_gain_k_per_kw_step"])
    if gain <= 0.0:
        raise ValueError("RC HVAC gain is not positive")
    zone_capacity = CONFIG.step_seconds * 1000.0 / gain
    a_oa = float(coefficients["a_oa"])
    a_zm = float(coefficients["a_zm"])
    a_mz = float(coefficients["a_mz"])
    a_mo = float(coefficients["a_mo"])
    envelope_resistance = (
        CONFIG.step_seconds / (zone_capacity * a_oa) if a_oa > 0.0 else None
    )
    if a_zm > 0.0 and a_mz > 0.0:
        mass_capacity = zone_capacity * a_zm / a_mz
        mass_resistance = CONFIG.step_seconds / (zone_capacity * a_zm)
        mass_outdoor_resistance = None
    else:
        mass_capacity = mass_resistance = mass_outdoor_resistance = None
    return {
        "zone_capacity_j_per_k": zone_capacity,
        "outdoor_resistance_k_per_w": envelope_resistance,
        "mass_capacity_j_per_k": mass_capacity,
        "mass_to_zone_capacity_ratio": (
            mass_capacity / zone_capacity if mass_capacity is not None else None
        ),
        "zone_mass_resistance_k_per_w": mass_resistance,
        "mass_outdoor_resistance_k_per_w": mass_outdoor_resistance,
        "effective_solar_aperture_m2": float(
            coefficients["solar_gain_k_per_w_m2_step"] / gain * 1000.0
        ),
        "constant_heat_flow_w": float(
            coefficients["constant_heat_flow_k_per_step"] / gain * 1000.0
        ),
    }


def thermal_step(
    topology: str,
    coefficients: Mapping[str, float],
    zone_temperature: float,
    mass_temperature: float,
    outdoor_temperature: float,
    solar_irradiance: float,
    heat_kw: float,
) -> tuple[float, float]:
    zone_next = (
        zone_temperature
        + float(coefficients["a_oa"]) * (outdoor_temperature - zone_temperature)
        + float(coefficients["a_zm"]) * (mass_temperature - zone_temperature)
        + float(coefficients["hvac_gain_k_per_kw_step"]) * heat_kw
        + float(coefficients["solar_gain_k_per_w_m2_step"]) * solar_irradiance
        + float(coefficients["constant_heat_flow_k_per_step"])
    )
    if topology == "2r2c":
        mass_next = (
            mass_temperature
            + float(coefficients["a_mz"]) * (zone_temperature - mass_temperature)
            + float(coefficients["a_mo"]) * (outdoor_temperature - mass_temperature)
        )
    else:
        mass_next = zone_next
    return float(zone_next), float(mass_next)


def fit_thermal_model(
    case: str,
    topology: str,
    observations: Sequence[np.ndarray],
    contexts: Sequence[np.ndarray],
) -> tuple[dict[str, float], tuple[float, ...], dict[str, float | int]]:
    if len(observations) != 20 or len(contexts) != 20:
        raise ValueError("RC thermal fit requires all 20 fitting trajectories")
    observation_array = np.stack(observations)
    context_array = np.stack(contexts)

    def residual(raw: np.ndarray) -> np.ndarray:
        coefficients = _thermal_coefficients(topology, raw)
        zone = observation_array[:, 0, 0].copy()
        mass = zone.copy()
        values = []
        for step in range(observation_array.shape[1] - 1):
            target = observation_array[:, step + 1]
            if case == "multizone_office_simple_air":
                flow = np.maximum(target[:, 2], 0.0)
                heat = 1.2 * 1.006 * flow * (target[:, 3] - zone)
            else:
                heat = target[:, 3] / 1000.0
            outdoor = context_array[:, step + 1, 0]
            solar = context_array[:, step + 1, 1]
            zone_next = (
                zone
                + coefficients["a_oa"] * (outdoor - zone)
                + coefficients["a_zm"] * (mass - zone)
                + coefficients["hvac_gain_k_per_kw_step"] * heat
                + coefficients["solar_gain_k_per_w_m2_step"] * solar
                + coefficients["constant_heat_flow_k_per_step"]
            )
            if topology == "2r2c":
                mass_next = (
                    mass
                    + coefficients["a_mz"] * (zone - mass)
                    + coefficients["a_mo"] * (outdoor - mass)
                )
            else:
                mass_next = zone_next
            values.append(zone_next - target[:, 0])
            zone, mass = zone_next, mass_next
        return np.concatenate(values)

    results = []
    for start_index, raw0 in enumerate(_thermal_start_points(topology)):
        result = least_squares(
            residual,
            raw0,
            loss="soft_l1",
            f_scale=0.10,
            max_nfev=300,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        if np.isfinite(result.cost) and np.isfinite(result.x).all():
            results.append((float(result.cost), int(result.nfev), start_index, result))
    if len(results) != CONFIG.thermal_multistart_count:
        raise ValueError("RC thermal multistart produced an invalid fit")
    _, _, selected_start, result = min(
        results, key=lambda value: (value[0], value[1], value[2])
    )
    coefficients = _thermal_coefficients(topology, result.x)
    initial_offsets = tuple(0.0 for _ in observations)
    errors = residual(result.x)
    diagnostics = {
        "least_squares_success": int(bool(result.success)),
        "least_squares_status": int(result.status),
        "least_squares_function_evaluations": int(result.nfev),
        "least_squares_multistart_count": len(results),
        "least_squares_selected_start": selected_start,
        "least_squares_successful_starts": int(
            sum(bool(value[3].success) for value in results)
        ),
        "fit_temperature_mae_k": float(np.mean(np.abs(errors))),
        "fit_temperature_rmse_k": float(np.sqrt(np.mean(errors**2))),
    }
    return coefficients, initial_offsets, diagnostics


def equipment_features(
    case: str,
    observation_standardized: np.ndarray,
    action_standardized: np.ndarray,
    context_next_standardized: np.ndarray,
    scalers: FaultScalers,
) -> np.ndarray:
    observation = _inverse(
        observation_standardized,
        scalers.observation.mean,
        scalers.observation.scale,
    )
    action = _inverse(action_standardized, scalers.action.mean, scalers.action.scale)
    context = _inverse(
        context_next_standardized,
        scalers.context.mean,
        scalers.context.scale,
    )
    adapter = boptest.CASES[case]
    setpoint = adapter.base_setpoint_k + adapter.action_amplitude_k * float(action[0])
    zone = float(observation[0])
    temperature_scale = float(scalers.observation.scale[0])
    heat_degree = max(setpoint - zone, 0.0) / temperature_scale
    cool_degree = max(zone - setpoint, 0.0) / temperature_scale
    outdoor_delta = (float(context[0]) - zone) / temperature_scale
    return np.concatenate(
        [
            np.ones(1),
            np.asarray(observation_standardized, dtype=float),
            np.asarray(action_standardized, dtype=float),
            np.asarray(context_next_standardized, dtype=float),
            np.asarray([heat_degree, cool_degree, outdoor_delta]),
        ]
    )


def fit_equipment_map(
    case: str,
    alpha: float,
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    contexts: Sequence[np.ndarray],
    scalers: FaultScalers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    design = []
    targets = []
    observed_equipment = []
    for obs, action, context in zip(observations, actions, contexts):
        obs_z = scalers.observation.transform(obs)
        action_z = scalers.action.transform(action)
        context_z = scalers.context.transform(context)
        for step in range(len(obs) - 1):
            design.append(
                equipment_features(
                    case, obs_z[step], action_z[step], context_z[step + 1], scalers
                )
            )
            targets.append(obs_z[step + 1, 1:])
            observed_equipment.append(obs_z[step + 1, 1:])
    x = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float)
    penalty = np.eye(x.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    if alpha == 0.0:
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    residual = y - x @ coefficients
    observed = np.asarray(observed_equipment, dtype=float)
    span = observed.max(axis=0) - observed.min(axis=0)
    lower = observed.min(axis=0) - 0.05 * span
    upper = observed.max(axis=0) + 0.05 * span
    return coefficients, lower, upper, {
        "fit_equipment_standardized_mae": float(np.mean(np.abs(residual))),
        "fit_equipment_standardized_rmse": float(np.sqrt(np.mean(residual**2))),
    }


@dataclass(frozen=True)
class RcModel:
    case: str
    topology: str
    equipment_ridge_alpha: float
    innovation_clip_sigma: float
    thermal_coefficients: Mapping[str, float]
    equipment_coefficients: np.ndarray
    equipment_lower_standardized: np.ndarray
    equipment_upper_standardized: np.ndarray
    process_covariance: np.ndarray
    measurement_covariance: np.ndarray
    fit_diagnostics: Mapping[str, float | int]

    @property
    def active_coefficients(self) -> int:
        thermal = 4 if self.topology == "1r1c" else 6
        return int(thermal + self.equipment_coefficients.size)

    def payload(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "case": self.case,
            "topology": self.topology,
            "equipment_ridge_alpha": self.equipment_ridge_alpha,
            "innovation_clip_sigma": self.innovation_clip_sigma,
            "thermal_coefficients": dict(self.thermal_coefficients),
            "physical_diagnostics": physical_diagnostics(self.thermal_coefficients),
            "parameter_boundary_diagnostics": parameter_boundary_diagnostics(self),
            "equipment_coefficients": self.equipment_coefficients.tolist(),
            "equipment_lower_standardized": self.equipment_lower_standardized.tolist(),
            "equipment_upper_standardized": self.equipment_upper_standardized.tolist(),
            "process_covariance": self.process_covariance.tolist(),
            "measurement_covariance": self.measurement_covariance.tolist(),
            "fit_diagnostics": dict(self.fit_diagnostics),
            "active_coefficients": self.active_coefficients,
        }


def parameter_boundary_diagnostics(model: RcModel) -> dict[str, object]:
    """Flag selected parameters within 0.1% of a declared bound span."""

    coefficients = model.thermal_coefficients
    physical = physical_diagnostics(coefficients)
    tolerance = 0.001

    def near(value: float, lower: float, upper: float) -> bool:
        margin = tolerance * (upper - lower)
        return bool(value - lower <= margin or upper - value <= margin)

    ratio = physical["mass_to_zone_capacity_ratio"]
    aperture = float(physical["effective_solar_aperture_m2"])
    constant_kw = abs(float(physical["constant_heat_flow_w"])) / 1000.0
    flags: dict[str, bool | None] = {
        "outdoor_conductance_near_bound": near(
            float(coefficients["a_oa"]),
            CONFIG.minimum_active_conductance_coefficient,
            CONFIG.maximum_zone_coefficient_sum,
        ),
        "zone_coefficient_sum_near_upper_bound": bool(
            CONFIG.maximum_zone_coefficient_sum
            - float(coefficients["a_oa"] + coefficients["a_zm"])
            <= tolerance * CONFIG.maximum_zone_coefficient_sum
        ),
        "mass_capacity_ratio_near_bound": (
            near(
                float(ratio),
                CONFIG.minimum_mass_to_zone_capacity_ratio,
                CONFIG.maximum_mass_to_zone_capacity_ratio,
            )
            if ratio is not None
            else None
        ),
        "hvac_gain_near_bound": near(
            float(coefficients["hvac_gain_k_per_kw_step"]),
            CONFIG.minimum_hvac_gain_k_per_kw_step,
            CONFIG.maximum_hvac_gain_k_per_kw_step,
        ),
        "solar_aperture_near_bound": near(
            aperture, 0.0, CONFIG.maximum_effective_solar_aperture_m2
        ),
        "constant_heat_flow_near_bound": bool(
            CONFIG.maximum_constant_heat_flow_kw - constant_kw
            <= tolerance * (2.0 * CONFIG.maximum_constant_heat_flow_kw)
        ),
    }
    return {
        "boundary_relative_span_tolerance": tolerance,
        "equipment_regularization_endpoint": (
            "ordinary_least_squares"
            if model.equipment_ridge_alpha == 0.0
            else (
                "largest_finite_ridge"
                if model.equipment_ridge_alpha == max(CONFIG.equipment_ridge_alphas)
                else "interior"
            )
        ),
        **flags,
        "physical_boundary_flag_count": sum(value is True for value in flags.values()),
    }


def restore_model(payload: Mapping[str, object]) -> RcModel:
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("RC model schema changed")
    model = RcModel(
        case=str(payload["case"]),
        topology=str(payload["topology"]),
        equipment_ridge_alpha=float(payload["equipment_ridge_alpha"]),
        innovation_clip_sigma=float(payload["innovation_clip_sigma"]),
        thermal_coefficients={
            str(key): float(value)
            for key, value in dict(payload["thermal_coefficients"]).items()
        },
        equipment_coefficients=np.asarray(payload["equipment_coefficients"], dtype=float),
        equipment_lower_standardized=np.asarray(
            payload["equipment_lower_standardized"], dtype=float
        ),
        equipment_upper_standardized=np.asarray(
            payload["equipment_upper_standardized"], dtype=float
        ),
        process_covariance=np.asarray(payload["process_covariance"], dtype=float),
        measurement_covariance=np.asarray(payload["measurement_covariance"], dtype=float),
        fit_diagnostics=dict(payload["fit_diagnostics"]),
    )
    if model.case not in boptest.CASES or model.topology not in CONFIG.topologies:
        raise ValueError("RC model identity changed")
    if (
        model.equipment_ridge_alpha not in CONFIG.equipment_ridge_alphas
        or model.innovation_clip_sigma not in CONFIG.innovation_clip_sigmas
    ):
        raise ValueError("RC selected hyperparameter changed")
    if model.equipment_coefficients.shape != (14, EQUIPMENT_DIM):
        raise ValueError("RC equipment coefficient shape changed")
    if model.process_covariance.shape != (STATE_DIM, STATE_DIM):
        raise ValueError("RC process covariance shape changed")
    if model.measurement_covariance.shape != (OBSERVATION_DIM, OBSERVATION_DIM):
        raise ValueError("RC measurement covariance shape changed")
    if (
        model.equipment_lower_standardized.shape != (EQUIPMENT_DIM,)
        or model.equipment_upper_standardized.shape != (EQUIPMENT_DIM,)
        or np.any(
            model.equipment_lower_standardized
            >= model.equipment_upper_standardized
        )
    ):
        raise ValueError("RC equipment output bounds changed")
    arrays = (
        model.equipment_coefficients,
        model.equipment_lower_standardized,
        model.equipment_upper_standardized,
        model.process_covariance,
        model.measurement_covariance,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("RC model contains a non-finite value")
    expected_keys = {
        "a_oa",
        "a_zm",
        "a_mz",
        "a_mo",
        "hvac_gain_k_per_kw_step",
        "solar_gain_k_per_w_m2_step",
        "constant_heat_flow_k_per_step",
    }
    coefficients = model.thermal_coefficients
    if set(coefficients) != expected_keys or coefficients["a_mo"] != 0.0:
        raise ValueError("RC thermal network topology changed")
    if (
        coefficients["a_oa"] < CONFIG.minimum_active_conductance_coefficient
        or coefficients["a_oa"] + coefficients["a_zm"]
        > CONFIG.maximum_zone_coefficient_sum + 1e-12
        or coefficients["hvac_gain_k_per_kw_step"]
        < CONFIG.minimum_hvac_gain_k_per_kw_step
        or coefficients["hvac_gain_k_per_kw_step"]
        > CONFIG.maximum_hvac_gain_k_per_kw_step
    ):
        raise ValueError("RC thermal coefficients leave their physical bounds")
    if model.topology == "1r1c":
        if any(coefficients[name] != 0.0 for name in ("a_zm", "a_mz", "a_mo")):
            raise ValueError("1R1C payload contains a latent-mass path")
    elif (
        coefficients["a_zm"] < CONFIG.minimum_active_conductance_coefficient
        or coefficients["a_mz"] <= 0.0
    ):
        raise ValueError("2R2C payload contains an inactive mass path")
    physical = physical_diagnostics(coefficients)
    if payload.get("physical_diagnostics") != physical:
        raise ValueError("RC physical diagnostics changed")
    if payload.get("parameter_boundary_diagnostics") != parameter_boundary_diagnostics(model):
        raise ValueError("RC parameter-boundary diagnostics changed")
    if (
        not np.allclose(model.process_covariance, model.process_covariance.T)
        or np.linalg.eigvalsh(model.process_covariance).min() <= 0.0
        or not np.allclose(
            model.measurement_covariance,
            np.eye(OBSERVATION_DIM) * CONFIG.measurement_variance,
        )
    ):
        raise ValueError("RC covariance contract changed")
    return model


def state_from_observation(observation_standardized: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            observation_standardized[0],
            observation_standardized[0],
            observation_standardized[1],
            observation_standardized[2],
            observation_standardized[3],
        ],
        dtype=float,
    )


def observation_from_state(state: np.ndarray) -> np.ndarray:
    return np.asarray([state[0], state[2], state[3], state[4]], dtype=float)


def transition(
    model: RcModel,
    state: np.ndarray,
    action_standardized: np.ndarray,
    context_next_standardized: np.ndarray,
    scalers: FaultScalers,
) -> np.ndarray:
    observation_z = observation_from_state(state)
    features = equipment_features(
        model.case,
        observation_z,
        action_standardized,
        context_next_standardized,
        scalers,
    )
    equipment_z = np.clip(
        features @ model.equipment_coefficients,
        model.equipment_lower_standardized,
        model.equipment_upper_standardized,
    )
    observation_next_z = np.concatenate([[state[0]], equipment_z])
    observation_next = _inverse(
        observation_next_z, scalers.observation.mean, scalers.observation.scale
    )
    zone = float(
        _inverse(
            np.asarray([state[0]]),
            [scalers.observation.mean[0]],
            [scalers.observation.scale[0]],
        )[0]
    )
    mass = float(
        _inverse(
            np.asarray([state[1]]),
            [scalers.observation.mean[0]],
            [scalers.observation.scale[0]],
        )[0]
    )
    context_next = _inverse(
        context_next_standardized, scalers.context.mean, scalers.context.scale
    )
    heat_kw = thermal_input_kw(model.case, observation_next, zone)
    zone_next, mass_next = thermal_step(
        model.topology,
        model.thermal_coefficients,
        zone,
        mass,
        float(context_next[0]),
        float(context_next[1]),
        heat_kw,
    )
    temperature_mean = float(scalers.observation.mean[0])
    temperature_scale = float(scalers.observation.scale[0])
    return np.asarray(
        [
            (zone_next - temperature_mean) / temperature_scale,
            (mass_next - temperature_mean) / temperature_scale,
            *equipment_z,
        ],
        dtype=float,
    )


def transition_jacobian(
    model: RcModel,
    state: np.ndarray,
    action_standardized: np.ndarray,
    context_next_standardized: np.ndarray,
    scalers: FaultScalers,
) -> np.ndarray:
    observation_z = observation_from_state(state)
    features = equipment_features(
        model.case,
        observation_z,
        action_standardized,
        context_next_standardized,
        scalers,
    )
    feature_jacobian = np.zeros((len(features), STATE_DIM))
    feature_jacobian[1, 0] = 1.0
    feature_jacobian[2, 2] = 1.0
    feature_jacobian[3, 3] = 1.0
    feature_jacobian[4, 4] = 1.0
    observation = _inverse(
        observation_z, scalers.observation.mean, scalers.observation.scale
    )
    action = _inverse(action_standardized, scalers.action.mean, scalers.action.scale)
    adapter = boptest.CASES[model.case]
    setpoint = adapter.base_setpoint_k + adapter.action_amplitude_k * float(action[0])
    zone = float(observation[0])
    if setpoint > zone:
        feature_jacobian[11, 0] = -1.0
    if zone > setpoint:
        feature_jacobian[12, 0] = 1.0
    feature_jacobian[13, 0] = -1.0
    equipment_unclipped = features @ model.equipment_coefficients
    equipment_jacobian = model.equipment_coefficients.T @ feature_jacobian
    clipped = (equipment_unclipped <= model.equipment_lower_standardized) | (
        equipment_unclipped >= model.equipment_upper_standardized
    )
    equipment_jacobian[clipped] = 0.0
    equipment_z = np.clip(
        equipment_unclipped,
        model.equipment_lower_standardized,
        model.equipment_upper_standardized,
    )
    equipment_raw = _inverse(
        equipment_z,
        scalers.observation.mean[1:],
        scalers.observation.scale[1:],
    )
    heat_jacobian = np.zeros(STATE_DIM)
    if model.case == "multizone_office_simple_air":
        flow = max(float(equipment_raw[1]), 0.0)
        supply = float(equipment_raw[2])
        constant = 1.2 * 1.006
        if flow > 0.0:
            heat_jacobian += (
                constant
                * (supply - zone)
                * float(scalers.observation.scale[2])
                * equipment_jacobian[1]
            )
            heat_jacobian += (
                constant
                * flow
                * float(scalers.observation.scale[3])
                * equipment_jacobian[2]
            )
            heat_jacobian[0] -= (
                constant * flow * float(scalers.observation.scale[0])
            )
    else:
        heat_jacobian = (
            float(scalers.observation.scale[3])
            / 1000.0
            * equipment_jacobian[2]
        )
    temperature_scale = float(scalers.observation.scale[0])
    coefficients = model.thermal_coefficients
    zone_jacobian_raw = (
        float(coefficients["hvac_gain_k_per_kw_step"]) * heat_jacobian
    )
    zone_jacobian_raw[0] += (
        1.0 - float(coefficients["a_oa"]) - float(coefficients["a_zm"])
    ) * temperature_scale
    zone_jacobian_raw[1] += float(coefficients["a_zm"]) * temperature_scale
    jacobian = np.zeros((STATE_DIM, STATE_DIM))
    jacobian[0] = zone_jacobian_raw / temperature_scale
    if model.topology == "2r2c":
        jacobian[1, 0] = float(coefficients["a_mz"])
        jacobian[1, 1] = (
            1.0 - float(coefficients["a_mz"]) - float(coefficients["a_mo"])
        )
    else:
        jacobian[1] = jacobian[0]
    jacobian[2:] = equipment_jacobian
    return jacobian


def estimate_process_covariance(
    model: RcModel,
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    contexts: Sequence[np.ndarray],
    scalers: FaultScalers,
) -> np.ndarray:
    residuals = []
    for obs, action, context in zip(observations, actions, contexts):
        obs_z = scalers.observation.transform(obs)
        action_z = scalers.action.transform(action)
        context_z = scalers.context.transform(context)
        state = state_from_observation(obs_z[0])
        for step in range(len(obs) - 1):
            predicted = transition(model, state, action_z[step], context_z[step + 1], scalers)
            target = state_from_observation(obs_z[step + 1])
            target[1] = predicted[1]
            residuals.append(target - predicted)
            state = target
    variance = np.var(np.asarray(residuals, dtype=float), axis=0)
    variance[1] = max(float(variance[0]) * 0.05, 1e-6)
    variance = np.clip(variance, 1e-6, 1.0)
    return np.diag(variance)
