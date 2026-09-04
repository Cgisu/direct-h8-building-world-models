"""Execute and seal the one-case downstream closed-loop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
import torch

from building_fault_wm.deterministic_transport import evaluate
from building_fault_wm.neural_benchmark.fault_data import (
    FaultScalers,
    ScaleStats,
)
from building_fault_wm.neural_benchmark import protocol as boptest

from . import protocol


ARM_TO_CHECKPOINT = {
    "legacy_rssm": "legacy",
    "direct_h8_rssm": "ungated_h8",
}
SUMMARY_ENDPOINTS = (
    "cost_tot",
    "tdis_tot",
    "control_cost_proxy",
    "control_discomfort_kh",
    "control_energy_kwh",
)
DECISION_COLUMNS = (
    "day",
    "window_id",
    "condition",
    "policy",
    "decision_id",
    "decision_step",
    "selected",
    "action_level",
    "predicted_discomfort_kh",
    "predicted_cost_proxy",
    "predicted_energy_kwh",
    "comfort_feasible",
    "predicted_h8_zone_k",
    "predicted_h8_power_w",
)


def scale_stats(payload: object, dimension: int, label: str) -> ScaleStats:
    if not isinstance(payload, dict) or set(payload) != {"mean", "scale"}:
        raise ValueError(f"{label} scaler fields are invalid")
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0.0).any()
    ):
        raise ValueError(f"{label} scaler values are invalid")
    return ScaleStats(
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
    )


def load_scaler(path: Path, case: str, expected_sha256: str) -> FaultScalers:
    if protocol.sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen FIT scaler hash differs for {case}")
    payload = json.loads(path.read_text(encoding="ascii"))
    if set(payload) != {"observation", "action", "context", "fit_source_sha256"}:
        raise ValueError(f"frozen FIT scaler fields differ for {case}")
    sources = payload["fit_source_sha256"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].startswith(f"{case}:fit:")
            or not isinstance(item[1], str)
            or len(item[1]) != 64
            or any(character not in "0123456789abcdef" for character in item[1])
            for item in sources
        )
    ):
        raise ValueError(f"frozen FIT scaler source grid differs for {case}")
    return FaultScalers(
        observation=scale_stats(payload["observation"], 4, "observation"),
        action=scale_stats(payload["action"], 1, "action"),
        context=scale_stats(payload["context"], 5, "context"),
        fit_source_sha256=tuple((str(a), str(b)) for a, b in sources),
    )


class BoptestClient:
    def __init__(self, base_url: str, timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def request(
        self,
        method: str,
        path: str,
        data: Mapping[str, Any] | None = None,
        *,
        idempotent: bool = True,
    ) -> Any:
        attempts = 3 if idempotent else 1
        error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.request(
                    method,
                    self.base_url + path,
                    json=None if data is None else dict(data),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                if not response.content:
                    return None
                if "application/json" not in response.headers.get("Content-Type", ""):
                    return response.text
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as caught:
                error = caught
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
        assert error is not None
        raise error

    def payload(
        self,
        method: str,
        path: str,
        data: Mapping[str, Any] | None = None,
        *,
        idempotent: bool = True,
    ) -> Any:
        response = self.request(method, path, data, idempotent=idempotent)
        if not isinstance(response, dict) or response.get("status") != 200:
            raise RuntimeError(f"unexpected BOPTEST response from {path}: {response!r}")
        return response["payload"]

    @contextmanager
    def selected_test(self) -> Iterator[str]:
        test_id: str | None = None
        try:
            selected = self.request(
                "POST", f"/testcases/{protocol.CASE}/select", {}, idempotent=False
            )
            if not isinstance(selected, dict) or not isinstance(selected.get("testid"), str):
                raise RuntimeError(f"testcase selection failed: {selected!r}")
            test_id = selected["testid"]
            yield test_id
        finally:
            if test_id is not None:
                self.request("PUT", f"/stop/{test_id}", {}, idempotent=True)


def canonical_sha256(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def canonical_observation(state: Mapping[str, object]) -> np.ndarray:
    adapter = boptest.CASES[protocol.CASE]
    if (
        adapter.base_setpoint_k != protocol.BASE_SETPOINT_K
        or adapter.action_amplitude_k != protocol.ACTION_AMPLITUDE_K
    ):
        raise ValueError("case action mapping differs from the downstream protocol")
    reductions = (
        (adapter.zone_keys, "mean"),
        (adapter.power_keys, "sum"),
        (adapter.auxiliary_1_keys, adapter.auxiliary_1_reduction),
        (adapter.auxiliary_2_keys, adapter.auxiliary_2_reduction),
    )
    values = []
    for keys, reduction in reductions:
        raw = np.asarray([float(state[key]) for key in keys], dtype=float)
        values.append(float(raw.mean() if reduction == "mean" else raw.sum()))
    result = np.asarray(values, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("simulator returned an invalid canonical observation")
    return result


def context_at(forecast: Mapping[str, Sequence[object]], index: int) -> np.ndarray:
    adapter = boptest.CASES[protocol.CASE]

    def mean(keys: Sequence[str]) -> float:
        values = np.asarray([float(forecast[key][index]) for key in keys], dtype=float)
        return float(values.mean())

    value = np.asarray(
        [
            mean(("TDryBul",)),
            mean(("HGloHor",)),
            mean(adapter.lower_forecast_keys),
            mean(adapter.upper_forecast_keys),
            mean((adapter.price_forecast_key,)),
        ],
        dtype=float,
    )
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError("forecast returned an invalid canonical context")
    return value


@dataclass
class FaultInjector:
    condition: str
    last_clean_zone: float | None = None

    def apply(self, step: int, observation: np.ndarray) -> np.ndarray:
        if self.condition not in protocol.CONDITIONS:
            raise ValueError(f"unknown condition: {self.condition}")
        visible = np.asarray(observation, dtype=float).copy()
        if step == protocol.FAULT_START - 1:
            self.last_clean_zone = float(observation[0])
        active = protocol.FAULT_START <= step < protocol.FAULT_STOP
        if not active or self.condition == "clean":
            return visible
        if self.condition in {"zone_bias_negative", "zone_bias_positive"}:
            sign = -1.0 if self.condition.endswith("negative") else 1.0
            visible[0] += sign * protocol.ZONE_BIAS_K
        elif self.condition in {"zone_drift_negative", "zone_drift_positive"}:
            sign = -1.0 if self.condition.endswith("negative") else 1.0
            visible[0] += sign * protocol.ZONE_DRIFT_K_PER_STEP * (
                step - protocol.FAULT_START + 1
            )
        elif self.condition == "zone_stuck":
            if self.last_clean_zone is None:
                raise RuntimeError("stuck fault has no pre-onset clean value")
            visible[0] = self.last_clean_zone
        return visible


def predicted_metrics(prediction: np.ndarray, future_contexts: np.ndarray) -> dict[str, float | bool]:
    if prediction.shape != (protocol.CONTROL_HORIZON, 4):
        raise ValueError("candidate prediction shape changed")
    temperature = prediction[:, 0]
    power_w = np.maximum(prediction[:, 1], 0.0)
    lower = future_contexts[:, 2]
    upper = future_contexts[:, 3]
    price = future_contexts[:, 4]
    violation = np.maximum(lower - temperature, 0.0) + np.maximum(
        temperature - upper, 0.0
    )
    duration_h = protocol.STEP_SECONDS / 3600.0
    return {
        "predicted_discomfort_kh": float(duration_h * violation.sum()),
        "predicted_cost_proxy": float(duration_h * np.sum(power_w / 1000.0 * price)),
        "predicted_energy_kwh": float(duration_h * np.sum(power_w / 1000.0)),
        "comfort_feasible": bool(np.all(violation <= 0.0)),
    }


def select_candidate(candidate_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if {float(row["action_level"]) for row in candidate_rows} != set(protocol.ACTION_LEVELS):
        raise ValueError("candidate action alphabet changed")

    def key(row: Mapping[str, object]) -> tuple[float, ...]:
        action = float(row["action_level"])
        cost = float(row["predicted_cost_proxy"])
        discomfort = float(row["predicted_discomfort_kh"])
        if bool(row["comfort_feasible"]):
            return (0.0, cost, abs(action), action)
        return (1.0, discomfort, cost, abs(action), action)

    return min(candidate_rows, key=key)


class EnsembleController:
    def __init__(self, policy: str, frozen: Mapping[str, object]):
        if policy not in protocol.MODEL_POLICIES:
            raise ValueError(f"not a model policy: {policy}")
        self.policy = policy
        self.scaler = load_scaler(
            protocol.SCALER_PATH,
            protocol.CASE,
            str(frozen["inputs"]["fit_scaler"]["sha256"]),  # type: ignore[index]
        )
        self.models: list[torch.nn.Module] = []
        self.states: list[object] = []
        for seed in protocol.MODEL_SEEDS:
            if policy in ARM_TO_CHECKPOINT:
                arm = ARM_TO_CHECKPOINT[policy]
                item = frozen["inputs"]["rssm_checkpoints"][f"seed{seed}/{arm}"]  # type: ignore[index]
                model = evaluate.load_frozen_v2_rssm(
                    protocol.PROJECT_ROOT / str(item["path"]),
                    case=protocol.CASE,
                    model_seed=seed,
                    arm=arm,
                    expected_file_sha256=str(item["sha256"]),
                    device="cpu",
                )
                state: object = model.initial(1, device="cpu", dtype=torch.float32)
            else:
                item = frozen["inputs"]["deterministic_checkpoints"][f"seed{seed}"]  # type: ignore[index]
                model = evaluate.load_deterministic_checkpoint(
                    protocol.PROJECT_ROOT / str(item["path"]),
                    model_seed=seed,
                    expected_file_sha256=str(item["sha256"]),
                    device="cpu",
                )
                state = model.initial_hidden(1, device="cpu", dtype=torch.float32)
            self.models.append(model)
            self.states.append(state)

    @torch.no_grad()
    def update(
        self,
        visible_observation: np.ndarray,
        previous_action: float,
        current_context: np.ndarray,
    ) -> None:
        observation = torch.as_tensor(
            self.scaler.observation.transform(visible_observation)[None, :],
            dtype=torch.float32,
        )
        action = torch.as_tensor(
            self.scaler.action.transform(np.asarray([[previous_action]], dtype=float)),
            dtype=torch.float32,
        )
        context = torch.as_tensor(
            self.scaler.context.transform(current_context)[None, :],
            dtype=torch.float32,
        )
        availability = torch.ones((1, 4), dtype=torch.bool)
        age = torch.zeros((1, 4), dtype=torch.float32)
        for index, model in enumerate(self.models):
            if self.policy in ARM_TO_CHECKPOINT:
                result = model.filter_step(  # type: ignore[attr-defined]
                    self.states[index],
                    action,
                    observation,
                    availability,
                    age,
                    context,
                    gate_mode="bypass",
                    sample=False,
                )
                self.states[index] = result.rssm_step.state
            else:
                hidden, _ = model.observe_step(  # type: ignore[attr-defined]
                    self.states[index],
                    observation,
                    availability,
                    age,
                    action,
                    context,
                )
                self.states[index] = hidden

    @torch.no_grad()
    def decide(self, future_contexts: np.ndarray) -> tuple[float, list[dict[str, object]]]:
        if future_contexts.shape != (protocol.CONTROL_HORIZON, 5):
            raise ValueError("future context horizon changed")
        scaled_contexts = torch.as_tensor(
            self.scaler.context.transform(future_contexts)[:, None, :],
            dtype=torch.float32,
        )
        rows: list[dict[str, object]] = []
        for level in protocol.ACTION_LEVELS:
            action_raw = np.full((protocol.CONTROL_HORIZON, 1), level, dtype=float)
            actions = torch.as_tensor(
                self.scaler.action.transform(action_raw)[:, None, :],
                dtype=torch.float32,
            )
            seed_predictions = []
            for model, state in zip(self.models, self.states, strict=True):
                if self.policy in ARM_TO_CHECKPOINT:
                    rollout = model.imagine(  # type: ignore[attr-defined]
                        state, actions, scaled_contexts, sample=False
                    )
                else:
                    rollout = model.imagine(state, actions, scaled_contexts)  # type: ignore[attr-defined]
                standardized = rollout.observation_mean[:, 0].detach().cpu().numpy()
                seed_predictions.append(
                    standardized * np.asarray(self.scaler.observation.scale)
                    + np.asarray(self.scaler.observation.mean)
                )
            prediction = np.median(np.stack(seed_predictions), axis=0)
            metrics = predicted_metrics(prediction, future_contexts)
            rows.append(
                {
                    "action_level": level,
                    **metrics,
                    "predicted_h8_zone_k": float(prediction[-1, 0]),
                    "predicted_h8_power_w": float(prediction[-1, 1]),
                }
            )
        selected = select_candidate(rows)
        return float(selected["action_level"]), rows


def run_episode(
    client: BoptestClient,
    frozen: Mapping[str, object],
    window: Mapping[str, object],
    condition: str,
    policy_name: str,
    *,
    steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if (
        steps < protocol.HISTORY_STEPS + protocol.CONTROL_HORIZON
        or steps > protocol.EPISODE_STEPS
        or (steps - protocol.HISTORY_STEPS) % protocol.ACTION_DWELL_STEPS != 0
    ):
        raise ValueError("episode step count cannot support the fixed controller")
    controller = (
        None
        if policy_name == "constant_zero"
        else EnsembleController(policy_name, frozen)
    )
    adapter = boptest.CASES[protocol.CASE]
    fault = FaultInjector(condition)
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    held_action = 0.0
    previous_action = 0.0
    with client.selected_test() as test_id:
        client.payload("PUT", f"/step/{test_id}", {"step": protocol.STEP_SECONDS})
        client.payload(
            "PUT",
            f"/scenario/{test_id}",
            {
                "electricity_price": "dynamic",
                "temperature_uncertainty": None,
                "solar_uncertainty": None,
                "seed": int(window["scenario_seed"]),
            },
        )
        state = client.payload(
            "PUT",
            f"/initialize/{test_id}",
            {
                "start_time": int(window["day"]) * 86_400,
                "warmup_period": protocol.WARMUP_SECONDS,
            },
        )
        forecast = client.payload(
            "PUT",
            f"/forecast/{test_id}",
            {
                "point_names": list(adapter.forecast_keys),
                "horizon": steps * protocol.STEP_SECONDS,
                "interval": protocol.STEP_SECONDS,
            },
        )
        expected_times = int(window["day"]) * 86_400 + np.arange(steps + 1) * protocol.STEP_SECONDS
        required_forecast_keys = ("time", *adapter.forecast_keys)
        if any(
            key not in forecast or len(forecast[key]) != steps + 1
            for key in required_forecast_keys
        ):
            raise ValueError("BOPTEST forecast does not cover the full episode")
        if not np.array_equal(
            np.asarray(forecast["time"], dtype=float), expected_times.astype(float)
        ):
            raise ValueError("BOPTEST forecast time grid differs from the protocol")
        if float(state["time"]) != float(expected_times[0]):
            raise ValueError("BOPTEST initialized at an unexpected time")
        initialized_state_sha256 = canonical_sha256(state)
        forecast_sha256 = canonical_sha256(forecast)
        for step in range(steps):
            true_observation = canonical_observation(state)
            current_context = context_at(forecast, step)
            visible_observation = fault.apply(step, true_observation)
            if controller is not None:
                controller.update(visible_observation, previous_action, current_context)
            decision_id = -1
            if step < protocol.HISTORY_STEPS:
                held_action = 0.0
            elif policy_name == "constant_zero":
                held_action = 0.0
            elif (step - protocol.HISTORY_STEPS) % protocol.ACTION_DWELL_STEPS == 0:
                future_contexts = np.stack(
                    [context_at(forecast, index) for index in range(step + 1, step + 1 + protocol.CONTROL_HORIZON)]
                )
                assert controller is not None
                held_action, candidates = controller.decide(future_contexts)
                decision_id = len(decisions) // len(protocol.ACTION_LEVELS)
                for candidate in candidates:
                    decisions.append(
                        {
                            "day": int(window["day"]),
                            "window_id": str(window["window_id"]),
                            "condition": condition,
                            "policy": policy_name,
                            "decision_id": decision_id,
                            "decision_step": step,
                            "selected": bool(float(candidate["action_level"]) == held_action),
                            **candidate,
                        }
                    )
            active = step >= protocol.HISTORY_STEPS
            setpoint, payload = boptest.action_payload(adapter, held_action)
            next_state = client.payload(
                "POST", f"/advance/{test_id}", payload, idempotent=False
            )
            if float(next_state["time"]) != float(expected_times[step + 1]):
                raise ValueError("BOPTEST advance left the fixed time grid")
            outcome_observation = canonical_observation(next_state)
            outcome_context = context_at(forecast, step + 1)
            outcome_lower, outcome_upper, outcome_price = outcome_context[2:]
            outcome_violation = max(
                outcome_lower - outcome_observation[0], 0.0
            ) + max(outcome_observation[0] - outcome_upper, 0.0)
            rows.append(
                {
                    "day": int(window["day"]),
                    "window_id": str(window["window_id"]),
                    "condition": condition,
                    "policy": policy_name,
                    "step": step,
                    "time_s": float(state["time"]),
                    "outcome_time_s": float(next_state["time"]),
                    "control_stage": active,
                    "true_zone_temperature_k": true_observation[0],
                    "visible_zone_temperature_k": visible_observation[0],
                    "true_hvac_electric_power_w": true_observation[1],
                    "visible_hvac_electric_power_w": visible_observation[1],
                    "true_auxiliary_1": true_observation[2],
                    "visible_auxiliary_1": visible_observation[2],
                    "true_auxiliary_2": true_observation[3],
                    "visible_auxiliary_2": visible_observation[3],
                    "current_outdoor_temperature_k": current_context[0],
                    "current_horizontal_solar_w_m2": current_context[1],
                    "current_comfort_lower_k": current_context[2],
                    "current_comfort_upper_k": current_context[3],
                    "current_electricity_price": current_context[4],
                    "outcome_zone_temperature_k": outcome_observation[0],
                    "outcome_hvac_electric_power_w": outcome_observation[1],
                    "outcome_auxiliary_1": outcome_observation[2],
                    "outcome_auxiliary_2": outcome_observation[3],
                    "outcome_outdoor_temperature_k": outcome_context[0],
                    "outcome_horizontal_solar_w_m2": outcome_context[1],
                    "outcome_comfort_lower_k": outcome_lower,
                    "outcome_comfort_upper_k": outcome_upper,
                    "outcome_electricity_price": outcome_price,
                    "normalized_action": held_action,
                    "setpoint_k": setpoint,
                    "outcome_discomfort_k": outcome_violation,
                    "decision_id": decision_id,
                }
            )
            state = next_state
            previous_action = held_action
        kpis = client.payload("GET", f"/kpi/{test_id}")
    frame = pd.DataFrame(rows)
    decision_frame = pd.DataFrame(decisions, columns=DECISION_COLUMNS)
    expected_decisions = (steps - protocol.HISTORY_STEPS) // protocol.ACTION_DWELL_STEPS
    if controller is not None:
        grouped = decision_frame.groupby("decision_id", sort=True)
        if (
            len(grouped) != expected_decisions
            or len(decision_frame) != expected_decisions * len(protocol.ACTION_LEVELS)
            or not (grouped.selected.sum() == 1).all()
        ):
            raise ValueError("model policy did not emit one selection per decision")
    elif not decision_frame.empty:
        raise ValueError("constant policy unexpectedly emitted model decisions")
    control = frame.loc[frame.control_stage]
    duration_h = protocol.STEP_SECONDS / 3600.0
    summary: dict[str, object] = {
        "day": int(window["day"]),
        "window_id": str(window["window_id"]),
        "temperature_stratum": int(window["temperature_stratum"]),
        "condition": condition,
        "policy": policy_name,
        "steps": len(frame),
        "control_steps": len(control),
        "decisions": expected_decisions if controller is not None else 0,
        "initialized_state_sha256": initialized_state_sha256,
        "forecast_sha256": forecast_sha256,
        "control_cost_proxy": float(
            duration_h
            * np.sum(
                np.maximum(control.outcome_hvac_electric_power_w.to_numpy(), 0.0)
                / 1000.0
                * control.outcome_electricity_price.to_numpy()
            )
        ),
        "control_discomfort_kh": float(duration_h * control.outcome_discomfort_k.sum()),
        "control_energy_kwh": float(
            duration_h
            * np.sum(
                np.maximum(control.outcome_hvac_electric_power_w.to_numpy(), 0.0)
                / 1000.0
            )
        ),
        "action_minus_fraction": float(np.mean(control.normalized_action == -1.0)),
        "action_zero_fraction": float(np.mean(control.normalized_action == 0.0)),
        "action_plus_fraction": float(np.mean(control.normalized_action == 1.0)),
        "action_changes": int(
            np.count_nonzero(
                np.diff(
                    np.concatenate(
                        ([0.0], control.normalized_action.to_numpy(dtype=float))
                    )
                )
            )
        ),
    }
    for key, value in dict(kpis).items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            summary[str(key)] = float(value)
    missing = [key for key in ("cost_tot", "tdis_tot") if key not in summary]
    if missing:
        raise ValueError(f"BOPTEST KPI payload is missing {missing}")
    return frame, decision_frame, summary


def paired_bootstrap(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap values are invalid")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def analyze(summaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"day", "condition", "policy", *SUMMARY_ENDPOINTS}
    if not required.issubset(summaries) or summaries.empty:
        raise ValueError("episode summaries are incomplete")
    aggregate_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    present_conditions = tuple(
        condition
        for condition in protocol.CONDITIONS
        if condition in set(summaries["condition"])
    )
    if set(summaries["condition"]) != set(present_conditions):
        raise ValueError("episode summaries contain an unknown condition")
    for condition in present_conditions:
        condition_rows = summaries.loc[summaries.condition == condition]
        for policy_name in protocol.POLICIES:
            selected = condition_rows.loc[condition_rows.policy == policy_name].set_index("day")
            if selected.empty or not selected.index.is_unique:
                raise ValueError("policy episodes are missing or duplicated")
            for endpoint in SUMMARY_ENDPOINTS:
                values = selected[endpoint].to_numpy(dtype=float)
                aggregate_rows.append(
                    {
                        "condition": condition,
                        "policy": policy_name,
                        "endpoint": endpoint,
                        "windows": len(values),
                        "median": float(np.median(values)),
                        "mean": float(np.mean(values)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
        day_grid = condition_rows.loc[
            condition_rows.policy == protocol.POLICIES[0], "day"
        ].tolist()
        for policy_name in protocol.POLICIES[1:]:
            compared_grid = condition_rows.loc[
                condition_rows.policy == policy_name, "day"
            ].tolist()
            if compared_grid != day_grid:
                raise ValueError("policy episodes do not share the same day grid")
        for candidate_name, reference_name in protocol.CONTRASTS:
            candidate = condition_rows.loc[
                condition_rows.policy == candidate_name
            ].set_index("day")
            reference = condition_rows.loc[
                condition_rows.policy == reference_name
            ].set_index("day")
            for endpoint in SUMMARY_ENDPOINTS:
                differences = candidate[endpoint].to_numpy(dtype=float) - reference[
                    endpoint
                ].to_numpy(dtype=float)
                low, high = paired_bootstrap(
                    differences,
                    int.from_bytes(
                        hashlib.sha256(
                            f"{condition}:{candidate_name}:{reference_name}:{endpoint}".encode(
                                "ascii"
                            )
                        ).digest()[:8],
                        "little",
                    ),
                )
                paired_rows.append(
                    {
                        "condition": condition,
                        "candidate": candidate_name,
                        "reference": reference_name,
                        "endpoint": endpoint,
                        "windows": len(differences),
                        "median_paired_difference": float(np.median(differences)),
                        "mean_paired_difference": float(np.mean(differences)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "improved_windows": int(np.count_nonzero(differences < 0.0)),
                        "tied_windows": int(np.count_nonzero(differences == 0.0)),
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(paired_rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="ascii")


def file_inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"report_manifest.json", "report_manifest.canonical.sha256"}:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": protocol.sha256_file(path),
                }
            )
    return rows


def seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )


def run(
    *,
    base_url: str,
    output: Path,
    pilot: bool,
) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite downstream output: {output}")
    frozen = protocol.protocol_payload() if pilot else protocol.validate_frozen_protocol()
    version = BoptestClient(base_url).payload("GET", "/version")
    if not isinstance(version, dict) or version.get("version") != protocol.BOPTEST_VERSION:
        raise ValueError("live BOPTEST version differs from the protocol")
    windows = (
        [{"day": protocol.PILOT_DAY, "window_id": f"pilot:day{protocol.PILOT_DAY:03d}", "scenario_seed": 90_000 + protocol.PILOT_DAY, "temperature_stratum": -1}]
        if pilot
        else list(frozen["windows"])
    )
    conditions = ("clean", "zone_bias_positive") if pilot else protocol.CONDITIONS
    steps = 96 if pilot else protocol.EPISODE_STEPS
    staging = output.parent / f".{output.name}.staging"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale downstream staging exists: {staging}")
    staging.mkdir(parents=True)
    started = time.time()
    summaries: list[dict[str, object]] = []
    identity_by_day: dict[int, tuple[str, str]] = {}
    client = BoptestClient(base_url)
    try:
        for window in windows:
            for condition in conditions:
                for policy_name in protocol.POLICIES:
                    print(
                        f"day={window['day']} condition={condition} policy={policy_name}",
                        flush=True,
                    )
                    frame, decisions, summary = run_episode(
                        client,
                        frozen,
                        window,
                        condition,
                        policy_name,
                        steps=steps,
                    )
                    identity = (
                        str(summary["initialized_state_sha256"]),
                        str(summary["forecast_sha256"]),
                    )
                    day = int(window["day"])
                    if day in identity_by_day and identity_by_day[day] != identity:
                        raise ValueError("paired downstream branches have different initialization or forecast")
                    identity_by_day[day] = identity
                    stem = f"day{day:03d}_{condition}_{policy_name}"
                    frame.to_csv(staging / f"{stem}_trajectory.csv", index=False, float_format="%.17g")
                    decisions.to_csv(staging / f"{stem}_decisions.csv", index=False, float_format="%.17g")
                    summaries.append(summary)
        summary_frame = pd.DataFrame(summaries).sort_values(
            ["day", "condition", "policy"], kind="stable"
        )
        summary_frame.to_csv(staging / "episode_summary.csv", index=False, float_format="%.17g")
        aggregate, paired = analyze(summary_frame)
        aggregate.to_csv(staging / "aggregate_summary.csv", index=False, float_format="%.17g")
        paired.to_csv(staging / "paired_effects.csv", index=False, float_format="%.17g")
        metadata = {
            "schema": "direct-h8-downstream-control-result-v1",
            "pilot": pilot,
            "protocol_file_sha256": protocol.sha256_file(protocol.PROTOCOL_PATH) if not pilot else None,
            "protocol_canonical_sha256": protocol.canonical_sha256(frozen),
            "base_url": base_url,
            "episodes": len(summary_frame),
            "wall_seconds": time.time() - started,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }
        write_json(staging / "run_metadata.json", metadata)
        manifest = {
            "schema": "direct-h8-downstream-control-report-manifest-v1",
            "files": file_inventory(staging),
        }
        manifest = {**manifest, "payload_sha256": protocol.canonical_sha256(manifest)}
        write_json(staging / "report_manifest.json", manifest)
        (staging / "report_manifest.canonical.sha256").write_text(
            protocol.canonical_sha256(manifest) + "\n", encoding="ascii"
        )
        staging.rename(output)
        if not pilot:
            seal_tree(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(run(base_url=args.base_url, output=args.output, pilot=args.pilot))


if __name__ == "__main__":
    main()
