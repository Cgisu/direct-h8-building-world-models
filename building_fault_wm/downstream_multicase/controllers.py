"""Supervisory controllers used by the multi-case downstream study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from building_fault_wm.downstream_control import experiment as v1
from building_fault_wm.rc_baseline.model import (
    OBSERVATION_DIM,
    STATE_DIM,
    observation_from_state,
    restore_model,
    state_from_observation,
    transition,
    transition_jacobian,
)

from . import protocol


NEURAL_POLICIES = ("legacy_rssm", "direct_h8_rssm", "deterministic_wm")
MEASUREMENT = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


class RcSupervisoryController:
    """Use the selected RC model as a three-action supervisory selector."""

    def __init__(self, frozen: Mapping[str, object]):
        item = dict(frozen["inputs"])["rc_model"]
        if not isinstance(item, dict):
            raise ValueError("RC model receipt is invalid")
        path = protocol.PROJECT_ROOT / str(item["path"])
        if protocol.sha256_file(path) != str(item["sha256"]):
            raise ValueError("RC model hash differs")
        self.model = restore_model(_load_json(path))
        if self.model.case != v1.protocol.CASE:
            raise ValueError("RC model case differs from runtime case")
        scaler_item = dict(frozen["inputs"])["fit_scaler"]
        if not isinstance(scaler_item, dict):
            raise ValueError("scaler receipt is invalid")
        self.scaler = v1.load_scaler(
            v1.protocol.SCALER_PATH,
            v1.protocol.CASE,
            str(scaler_item["sha256"]),
        )
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    def update(
        self,
        visible_observation: np.ndarray,
        previous_action: float,
        current_context: np.ndarray,
    ) -> None:
        output = self.scaler.observation.transform(visible_observation)
        if self.state is None:
            self.state = state_from_observation(output)
            self.covariance = np.eye(STATE_DIM) * 0.1
            return
        assert self.covariance is not None
        action = self.scaler.action.transform(
            np.asarray([previous_action], dtype=float)
        )
        context = self.scaler.context.transform(current_context)
        jacobian = transition_jacobian(
            self.model, self.state, action, context, self.scaler
        )
        predicted = transition(
            self.model, self.state, action, context, self.scaler
        )
        predicted_covariance = (
            jacobian @ self.covariance @ jacobian.T
            + self.model.process_covariance
        )
        innovation_covariance = (
            MEASUREMENT @ predicted_covariance @ MEASUREMENT.T
            + self.model.measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance, MEASUREMENT @ predicted_covariance
        ).T
        residual = output - MEASUREMENT @ predicted
        if self.model.innovation_clip_sigma > 0.0:
            scales = np.sqrt(
                np.maximum(np.diag(innovation_covariance), 1e-12)
            )
            residual = np.clip(
                residual,
                -self.model.innovation_clip_sigma * scales,
                self.model.innovation_clip_sigma * scales,
            )
        self.state = predicted + gain @ residual
        identity = np.eye(STATE_DIM)
        correction = identity - gain @ MEASUREMENT
        self.covariance = (
            correction @ predicted_covariance @ correction.T
            + gain @ self.model.measurement_covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        if (
            not np.isfinite(self.state).all()
            or not np.isfinite(self.covariance).all()
        ):
            raise ValueError("RC online observer produced a non-finite state")

    def decide(
        self, future_contexts: np.ndarray
    ) -> tuple[float, list[dict[str, object]]]:
        if self.state is None or self.covariance is None:
            raise RuntimeError("RC controller has no filtered state")
        if future_contexts.shape != (protocol.CONTROL_HORIZON, 5):
            raise ValueError("RC future-context horizon changed")
        contexts = self.scaler.context.transform(future_contexts)
        rows: list[dict[str, object]] = []
        for level in protocol.ACTION_LEVELS:
            action = self.scaler.action.transform(
                np.asarray([level], dtype=float)
            )
            state = np.array(self.state, copy=True)
            predictions = []
            for context in contexts:
                state = transition(
                    self.model, state, action, context, self.scaler
                )
                observation = observation_from_state(state)
                predictions.append(
                    observation
                    * np.asarray(self.scaler.observation.scale, dtype=float)
                    + np.asarray(self.scaler.observation.mean, dtype=float)
                )
            prediction = np.asarray(predictions, dtype=float)
            if prediction.shape != (protocol.CONTROL_HORIZON, OBSERVATION_DIM):
                raise ValueError("RC rollout shape changed")
            metrics = v1.predicted_metrics(prediction, future_contexts)
            rows.append(
                {
                    "action_level": level,
                    **metrics,
                    "predicted_h8_zone_k": float(prediction[-1, 0]),
                    "predicted_h8_power_w": float(prediction[-1, 1]),
                }
            )
        selected = v1.select_candidate(rows)
        return float(selected["action_level"]), rows


class PriceAwareRuleController:
    """Price-aware hysteretic rule with development-selected parameters."""

    def __init__(self, parameters: protocol.RuleParameters):
        self.parameters = parameters
        self.visible_observation: np.ndarray | None = None
        self.current_context: np.ndarray | None = None

    def update(
        self,
        visible_observation: np.ndarray,
        previous_action: float,
        current_context: np.ndarray,
    ) -> None:
        del previous_action
        self.visible_observation = np.asarray(visible_observation, dtype=float)
        self.current_context = np.asarray(current_context, dtype=float)

    def _selected_action(self, future_contexts: np.ndarray) -> float:
        if self.visible_observation is None or self.current_context is None:
            raise RuntimeError("rule-based controller has no current observation")
        zone = float(self.visible_observation[0])
        price = float(np.mean(future_contexts[:, 4]))
        high_price = price >= self.parameters.price_threshold
        margin = self.parameters.margin_k
        if v1.protocol.CASE == "multizone_office_simple_air":
            upper_guard = float(
                min(self.current_context[3], np.min(future_contexts[:, 3]))
            ) - margin
            if zone >= upper_guard:
                return -1.0
            if high_price:
                return 1.0
            return 0.0
        lower_guard = float(
            max(self.current_context[2], np.max(future_contexts[:, 2]))
        ) + margin
        if zone <= lower_guard:
            return 1.0
        if high_price:
            return -1.0
        return 0.0

    def decide(
        self, future_contexts: np.ndarray
    ) -> tuple[float, list[dict[str, object]]]:
        if self.visible_observation is None:
            raise RuntimeError("rule-based controller has no current observation")
        selected = self._selected_action(future_contexts)
        persistence = np.repeat(
            self.visible_observation[None, :], protocol.CONTROL_HORIZON, axis=0
        )
        metrics = v1.predicted_metrics(persistence, future_contexts)
        rows = [
            {
                "action_level": level,
                **metrics,
                "predicted_h8_zone_k": float(persistence[-1, 0]),
                "predicted_h8_power_w": float(persistence[-1, 1]),
            }
            for level in protocol.ACTION_LEVELS
        ]
        return selected, rows


def controller_for(
    policy_name: str,
    frozen: Mapping[str, object],
    neural_controller_type: type,
    rule_parameters: protocol.RuleParameters,
) -> object:
    if policy_name in NEURAL_POLICIES:
        return neural_controller_type(policy_name, frozen)
    if policy_name == "rc_supervisory_selector":
        return RcSupervisoryController(frozen)
    if policy_name == "tuned_rule_based":
        return PriceAwareRuleController(rule_parameters)
    raise ValueError(f"unknown non-neutral policy: {policy_name}")
