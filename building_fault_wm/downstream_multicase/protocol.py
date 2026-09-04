"""Fixed design and input inventory for the multi-case downstream study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from building_fault_wm.neural_benchmark import protocol as boptest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

CASES = (
    "bestest_hydronic_heat_pump",
    "twozone_apartment_hydronic",
    "multizone_office_simple_air",
)
RESPONSE_UNSEEN_CASES = (
    "twozone_apartment_hydronic",
    "multizone_office_simple_air",
)
POLICIES = (
    "constant_zero",
    "legacy_rssm",
    "direct_h8_rssm",
    "deterministic_wm",
    "rc_supervisory_selector",
    "tuned_rule_based",
)
MODEL_POLICIES = POLICIES[1:]
CONDITIONS = (
    "clean",
    "zone_bias_negative",
    "zone_bias_positive",
    "zone_drift_negative",
    "zone_drift_positive",
    "zone_stuck",
)
MODEL_SEEDS = (202608011, 202608012, 202608013, 202608014, 202608015)
ACTION_LEVELS = (-1.0, 0.0, 1.0)
STEP_SECONDS = 900
PARALLEL_WORKERS = 8
WARMUP_SECONDS = 86_400
EPISODE_STEPS = 192
HISTORY_STEPS = 48
CONTROL_HORIZON = 8
ACTION_DWELL_STEPS = 16
FAULT_START = HISTORY_STEPS
FAULT_STOP = HISTORY_STEPS + 48
ZONE_BIAS_K = 2.0
ZONE_DRIFT_K_PER_STEP = 0.05

# The cost budget is the full control-period tariff exposure of a fixed
# 10 W/m2 electrical load. The discomfort budget is the forecast comfort-band
# half-width integrated over the same period. Neither scale uses policy output.
COST_REFERENCE_POWER_DENSITY_KW_M2 = 0.010
COST_WEIGHT = 0.5
DISCOMFORT_WEIGHT = 0.5

RULE_MARGIN_GRID_K = (0.0, 0.5, 1.0)
RULE_PRICE_QUANTILES = (0.25, 0.5, 0.75)
TUNING_ROLE = "validation"
PILOT_DAYS = {
    "bestest_hydronic_heat_pump": 13,
    "twozone_apartment_hydronic": 4,
    "multizone_office_simple_air": 4,
}

BOPTEST_ROOT = (Path.home() / "external/project1-boptest")
TESTCASE_ROOT = BOPTEST_ROOT / "testcases"
FINAL_PLAN_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data_v6/plans/full"
)
DEVELOPMENT_PLAN_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data_v4/plans/full"
)
DEVELOPMENT_RAW_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data_v4/development_raw"
)
SCALER_ROOT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_publication_v2/experiment/prelock_bundle/frozen/fit_scalers"
)
RSSM_ROOT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_publication_v2/experiment/prelock_bundle/frozen/checkpoints"
)
DETERMINISTIC_ROOT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_training_bound_v2"
)
RC_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_report_v3/models"
TUNING_ROOT = PROJECT_ROOT / "artifacts/direct_h8_downstream_multicase_tuning_v2"
TUNING_PATH = TUNING_ROOT / "tuning.json"
PROTOCOL_ROOT = PROJECT_ROOT / "artifacts/direct_h8_downstream_multicase_protocol_v2"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
PROTOCOL_DIGEST_PATH = PROTOCOL_ROOT / "protocol.canonical.sha256"
DEFAULT_PILOT_OUTPUT = PROJECT_ROOT / "artifacts/direct_h8_downstream_multicase_pilot_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/direct_h8_downstream_multicase_v2"

IMPLEMENTATION_PATHS = (
    HERE / "protocol.py",
    HERE / "controllers.py",
    HERE / "experiment.py",
    HERE / "freeze.py",
    HERE / "verify_report.py",
    PROJECT_ROOT / "building_fault_wm/downstream_control/experiment.py",
    PROJECT_ROOT / "building_fault_wm/rc_baseline/model.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/protocol.py",
)


@dataclass(frozen=True)
class RuleParameters:
    margin_k: float
    price_quantile: float
    price_threshold: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def final_plan_path(case: str) -> Path:
    return FINAL_PLAN_ROOT / f"{case}.json"


def development_plan_path(case: str) -> Path:
    return DEVELOPMENT_PLAN_ROOT / f"{case}.json"


def scaler_path(case: str) -> Path:
    return SCALER_ROOT / f"{case}.json"


def fmu_path(case: str) -> Path:
    return TESTCASE_ROOT / case / "models/wrapped.fmu"


def rc_model_path(case: str) -> Path:
    return RC_ROOT / f"{case}.json"


def rssm_checkpoint(case: str, seed: int, arm: str) -> Path:
    if seed not in MODEL_SEEDS or arm not in {"legacy", "ungated_h8"}:
        raise ValueError("unknown RSSM checkpoint identity")
    return RSSM_ROOT / case / f"seed{seed}/{arm}_u0400.pt"


def deterministic_checkpoint(case: str, seed: int) -> Path:
    if seed not in MODEL_SEEDS:
        raise ValueError("unknown deterministic checkpoint identity")
    return DETERMINISTIC_ROOT / case / f"seed{seed}/checkpoints/update_0400.pt"


def load_final_windows(case: str) -> list[dict[str, object]]:
    value = _read_json(final_plan_path(case))
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != 12:
        raise ValueError(f"{case} final plan must contain 12 windows")
    windows = []
    for entry in entries:
        windows.append(
            {
                "day": int(entry["day"]),
                "window_id": str(entry["window_id"]),
                "scenario_seed": int(entry["scenario_seed"]),
                "temperature_stratum": int(entry["temperature_stratum"]),
                "mean_outdoor_temperature_k": float(
                    entry["mean_outdoor_temperature_k"]
                ),
            }
        )
    if len({item["day"] for item in windows}) != 12:
        raise ValueError(f"{case} final plan contains duplicate days")
    return windows


def load_development_windows(case: str) -> list[dict[str, object]]:
    value = _read_json(development_plan_path(case))
    entries = [
        entry for entry in value.get("entries", []) if entry.get("role") == TUNING_ROLE
    ]
    if len(entries) != 8:
        raise ValueError(f"{case} development plan must contain eight validation windows")
    final_days = {int(item["day"]) for item in load_final_windows(case)}
    windows = [
        {
            "day": int(entry["day"]),
            "window_id": f"{case}:development:day{int(entry['day']):03d}",
            "scenario_seed": int(entry["trajectory_seed"]),
            "temperature_stratum": int(entry["temperature_stratum"]),
        }
        for entry in entries
    ]
    if final_days.intersection(int(item["day"]) for item in windows):
        raise ValueError(f"{case} development windows overlap final windows")
    return sorted(windows, key=lambda item: int(item["day"]))


def development_csvs(case: str) -> tuple[Path, ...]:
    paths = tuple(sorted((DEVELOPMENT_RAW_ROOT / case).glob("*validation*.csv")))
    expected_days = {int(item["day"]) for item in load_development_windows(case)}
    found_days: set[int] = set()
    for path in paths:
        with path.open(newline="", encoding="ascii") as stream:
            first = next(csv.DictReader(stream))
        found_days.add(int(first["day"]))
    if len(paths) != 8 or found_days != expected_days:
        raise ValueError(f"{case} validation CSV inventory differs from the plan")
    return paths


def price_thresholds(case: str) -> dict[float, float]:
    values: list[float] = []
    for path in development_csvs(case):
        with path.open(newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != EPISODE_STEPS:
            raise ValueError(f"development trajectory length changed: {path}")
        values.extend(
            float(row["next_electricity_price"])
            for row in rows[HISTORY_STEPS:]
        )
    array = np.asarray(values, dtype=float)
    if array.size != 8 * (EPISODE_STEPS - HISTORY_STEPS) or not np.isfinite(array).all():
        raise ValueError(f"{case} validation tariff grid is invalid")
    return {
        quantile: float(np.quantile(array, quantile, method="linear"))
        for quantile in RULE_PRICE_QUANTILES
    }


def rule_grid(case: str) -> tuple[RuleParameters, ...]:
    thresholds = price_thresholds(case)
    return tuple(
        RuleParameters(margin, quantile, thresholds[quantile])
        for margin in RULE_MARGIN_GRID_K
        for quantile in RULE_PRICE_QUANTILES
    )


def load_tuning() -> dict:
    if not TUNING_PATH.is_file():
        raise FileNotFoundError("rule-based controller tuning has not completed")
    value = _read_json(TUNING_PATH)
    if value.get("schema") != "direct-h8-multicase-rule-tuning-v2":
        raise ValueError("rule-based tuning schema differs")
    if value.get("payload_sha256") != canonical_sha256(
        {key: item for key, item in value.items() if key != "payload_sha256"}
    ):
        raise ValueError("rule-based tuning digest differs")
    expected_implementation = {
        str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256_file(path)}
        for path in IMPLEMENTATION_PATHS
    }
    if value.get("implementation") != expected_implementation:
        raise ValueError("rule-based tuning implementation receipt differs")
    if value.get("parallel_workers") != PARALLEL_WORKERS:
        raise ValueError("rule-based tuning worker count differs")
    rows_receipt = value.get("rows_file")
    if not isinstance(rows_receipt, dict) or rows_receipt.get("path") != "tuning_rows.csv":
        raise ValueError("rule-based tuning row receipt differs")
    rows_path = TUNING_ROOT / "tuning_rows.csv"
    if sha256_file(rows_path) != rows_receipt.get("sha256"):
        raise ValueError("rule-based tuning row hash differs")
    with rows_path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(CASES) * len(RULE_MARGIN_GRID_K) * len(RULE_PRICE_QUANTILES) * 8:
        raise ValueError("rule-based tuning row grid differs")
    cases_payload = value.get("cases")
    if not isinstance(cases_payload, dict) or set(cases_payload) != set(CASES):
        raise ValueError("rule-based tuning case grid differs")
    for case in CASES:
        case_payload = cases_payload[case]
        if not isinstance(case_payload, dict):
            raise ValueError(f"{case} tuning payload is invalid")
        grid = case_payload.get("grid")
        if not isinstance(grid, list) or len(grid) != 9:
            raise ValueError(f"{case} tuning candidate grid differs")
        recomputed = []
        for candidate in grid:
            if not isinstance(candidate, dict):
                raise ValueError(f"{case} tuning candidate is invalid")
            candidate_rows = [
                row
                for row in rows
                if row["case"] == case
                and row["candidate_id"] == str(candidate["candidate_id"])
            ]
            if len(candidate_rows) != 8:
                raise ValueError(f"{case} tuning candidate window grid differs")
            scores = []
            action_fractions = np.zeros(3, dtype=float)
            for row in candidate_rows:
                score = scalar_score(
                    float(row["cost_tot"]),
                    float(row["tdis_tot"]),
                    float(row["neutral_cost_tot"]),
                    float(row["neutral_tdis_tot"]),
                    float(row["cost_budget"]),
                    float(row["discomfort_budget"]),
                )
                if not math.isclose(
                    score,
                    float(row["operational_score"]),
                    rel_tol=1e-11,
                    abs_tol=1e-11,
                ):
                    raise ValueError(f"{case} tuning score does not re-derive")
                scores.append(score)
                action_fractions += np.asarray(
                    [
                        float(row["action_minus_fraction"]),
                        float(row["action_zero_fraction"]),
                        float(row["action_plus_fraction"]),
                    ]
                )
            action_fractions /= len(candidate_rows)
            checks = {
                "median_operational_score": statistics.median(scores),
                "mean_operational_score": statistics.fmean(scores),
                "action_minus_fraction": action_fractions[0],
                "action_zero_fraction": action_fractions[1],
                "action_plus_fraction": action_fractions[2],
            }
            for key, expected in checks.items():
                if not math.isclose(
                    float(candidate[key]), float(expected), rel_tol=1e-11, abs_tol=1e-11
                ):
                    raise ValueError(f"{case} tuning {key} does not re-derive")
            if int(candidate["improved_windows"]) != sum(score < 0.0 for score in scores):
                raise ValueError(f"{case} tuning win count does not re-derive")
            recomputed.append(candidate)
        selected = min(
            recomputed,
            key=lambda item: (
                float(item["median_operational_score"]),
                float(item["mean_operational_score"]),
                float(item["margin_k"]),
                float(item["price_quantile"]),
            ),
        )
        if case_payload.get("selected") != selected:
            raise ValueError(f"{case} tuning selection does not re-derive")
    if value.get("all_competence_gates_passed") is not True:
        raise ValueError("rule-based controller did not pass every competence gate")
    return value


def selected_rule_parameters(case: str, tuning: Mapping[str, object]) -> RuleParameters:
    selected = dict(dict(tuning["cases"])[case])["selected"]
    if not isinstance(selected, dict):
        raise ValueError(f"{case} selected rule parameters are invalid")
    value = RuleParameters(
        margin_k=float(selected["margin_k"]),
        price_quantile=float(selected["price_quantile"]),
        price_threshold=float(selected["price_threshold"]),
    )
    if value not in rule_grid(case):
        raise ValueError(f"{case} selected rule parameters leave the fixed grid")
    return value


def case_inputs(case: str) -> dict[str, object]:
    adapter = boptest.CASES[case]
    if sha256_file(fmu_path(case)) != adapter.fmu_sha256:
        raise ValueError(f"{case} FMU differs from the fixed adapter")
    return {
        "final_plan": {
            "path": str(final_plan_path(case).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(final_plan_path(case)),
        },
        "development_plan": {
            "path": str(development_plan_path(case).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(development_plan_path(case)),
        },
        "development_validation_trajectories": {
            str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256_file(path)}
            for path in development_csvs(case)
        },
        "fit_scaler": {
            "path": str(scaler_path(case).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(scaler_path(case)),
        },
        "rssm_checkpoints": {
            f"seed{seed}/{arm}": {
                "path": str(rssm_checkpoint(case, seed, arm).relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(rssm_checkpoint(case, seed, arm)),
            }
            for seed in MODEL_SEEDS
            for arm in ("legacy", "ungated_h8")
        },
        "deterministic_checkpoints": {
            f"seed{seed}": {
                "path": str(deterministic_checkpoint(case, seed).relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(deterministic_checkpoint(case, seed)),
            }
            for seed in MODEL_SEEDS
        },
        "rc_model": {
            "path": str(rc_model_path(case).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(rc_model_path(case)),
        },
        "boptest_fmu": {
            "path": f"project1-boptest/testcases/{case}/models/wrapped.fmu",
            "sha256": sha256_file(fmu_path(case)),
        },
    }


def protocol_payload() -> dict:
    tuning = load_tuning()
    cases: dict[str, object] = {}
    for case in CASES:
        adapter = boptest.CASES[case]
        rule = selected_rule_parameters(case, tuning)
        cases[case] = {
            "downstream_response_status": (
                "response_unseen" if case in RESPONSE_UNSEEN_CASES else "previously_evaluated"
            ),
            "windows": load_final_windows(case),
            "base_setpoint_k": adapter.base_setpoint_k,
            "action_amplitude_k": adapter.action_amplitude_k,
            "floor_area_m2": float(
                _read_json(TESTCASE_ROOT / case / "models/config.json")["area"]
            ),
            "rule_parameters": {
                "margin_k": rule.margin_k,
                "price_quantile": rule.price_quantile,
                "price_threshold": rule.price_threshold,
            },
            "inputs": case_inputs(case),
        }
    payload = {
        "schema": "direct-h8-downstream-multicase-protocol-v2",
        "status": "descriptive_multicase_downstream_evaluation",
        "boptest": {
            "commit": boptest.BOPTEST_COMMIT,
            "version": boptest.BOPTEST_API_VERSION,
            "worker_image_id": boptest.WORKER_IMAGE_ID,
        },
        "cases": cases,
        "response_unseen_cases": list(RESPONSE_UNSEEN_CASES),
        "step_seconds": STEP_SECONDS,
        "parallel_workers": PARALLEL_WORKERS,
        "warmup_seconds": WARMUP_SECONDS,
        "episode_steps": EPISODE_STEPS,
        "history_steps": HISTORY_STEPS,
        "control_horizon_steps": CONTROL_HORIZON,
        "action_dwell_steps": ACTION_DWELL_STEPS,
        "action_levels": list(ACTION_LEVELS),
        "model_seeds": list(MODEL_SEEDS),
        "policies": list(POLICIES),
        "conditions": list(CONDITIONS),
        "fault": {
            "channel": "canonical mean zone-temperature model input",
            "start_step": FAULT_START,
            "stop_step_exclusive": FAULT_STOP,
            "bias_k": ZONE_BIAS_K,
            "drift_k_per_step": ZONE_DRIFT_K_PER_STEP,
            "stuck_value": "last clean canonical value before onset",
            "simulator_state_and_kpis_unmodified": True,
        },
        "score": {
            "formula": "0.5*(C-C0)/B_C + 0.5*(D-D0)/B_D",
            "cost_weight": COST_WEIGHT,
            "discomfort_weight": DISCOMFORT_WEIGHT,
            "cost_budget": (
                "10 W/m2 reference electrical load multiplied by the nonnegative "
                "dynamic tariff and integrated over the control period"
            ),
            "cost_reference_power_density_kw_m2": COST_REFERENCE_POWER_DENSITY_KW_M2,
            "discomfort_budget": (
                "forecast comfort-band half-width integrated over the control period"
            ),
            "reference_policy": "constant_zero",
            "scale_independence": "budgets use forecasts and fixed constants only",
        },
        "controller_roles": {
            "rc_supervisory_selector": "physically structured mechanism comparator",
            "tuned_rule_based": (
                "conventional price-aware hysteretic reference selected on clean "
                "development-validation windows"
            ),
        },
        "analysis": {
            "case_specific": "primary descriptive reporting unit",
            "response_unseen_pair": "reported separately without population inference",
            "all_three_cases": "finite-panel descriptive summary",
            "interval_unit": "whole window within case",
            "forbidden_label": "resolved preference",
        },
        "claim_limit": (
            "Three deterministic simulator case studies, including two cases with "
            "response-unseen downstream outcomes; no building-population, deployment, "
            "or universal controller-superiority claim."
        ),
        "tuning_receipt": {
            "path": str(TUNING_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(TUNING_PATH),
            "canonical_sha256": str(tuning["payload_sha256"]),
        },
        "implementation": {
            str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256_file(path)}
            for path in IMPLEMENTATION_PATHS
        },
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def validate_frozen_protocol() -> dict:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_DIGEST_PATH.is_file():
        raise FileNotFoundError("multi-case downstream protocol has not been frozen")
    frozen = _read_json(PROTOCOL_PATH)
    current = protocol_payload()
    if canonical_bytes(frozen) != canonical_bytes(current):
        raise ValueError("current multi-case downstream protocol differs from frozen")
    if PROTOCOL_DIGEST_PATH.read_text(encoding="ascii").strip() != canonical_sha256(frozen):
        raise ValueError("multi-case downstream protocol digest differs")
    return frozen


def runtime_protocol(case: str, frozen_case: Mapping[str, object]) -> SimpleNamespace:
    adapter = boptest.CASES[case]
    return SimpleNamespace(
        PROJECT_ROOT=PROJECT_ROOT,
        CASE=case,
        STEP_SECONDS=STEP_SECONDS,
        WARMUP_SECONDS=WARMUP_SECONDS,
        EPISODE_STEPS=EPISODE_STEPS,
        HISTORY_STEPS=HISTORY_STEPS,
        CONTROL_HORIZON=CONTROL_HORIZON,
        ACTION_DWELL_STEPS=ACTION_DWELL_STEPS,
        ACTION_LEVELS=ACTION_LEVELS,
        BASE_SETPOINT_K=adapter.base_setpoint_k,
        ACTION_AMPLITUDE_K=adapter.action_amplitude_k,
        MODEL_SEEDS=MODEL_SEEDS,
        MODEL_POLICIES=MODEL_POLICIES,
        POLICIES=POLICIES,
        CONDITIONS=CONDITIONS,
        CONTRASTS=(),
        FAULT_START=FAULT_START,
        FAULT_STOP=FAULT_STOP,
        ZONE_BIAS_K=ZONE_BIAS_K,
        ZONE_DRIFT_K_PER_STEP=ZONE_DRIFT_K_PER_STEP,
        SCALER_PATH=scaler_path(case),
        sha256_file=sha256_file,
    )


def score_budgets(trajectory: object) -> tuple[float, float]:
    control = trajectory.loc[trajectory.control_stage]
    duration_h = STEP_SECONDS / 3600.0
    price = np.maximum(
        control.outcome_electricity_price.to_numpy(dtype=float), 0.0
    )
    lower = control.outcome_comfort_lower_k.to_numpy(dtype=float)
    upper = control.outcome_comfort_upper_k.to_numpy(dtype=float)
    cost_budget = float(
        COST_REFERENCE_POWER_DENSITY_KW_M2 * duration_h * price.sum()
    )
    discomfort_budget = float(duration_h * np.maximum((upper - lower) / 2.0, 0.0).sum())
    if not np.isfinite([cost_budget, discomfort_budget]).all():
        raise ValueError("operational score budgets are non-finite")
    if cost_budget <= 0.0 or discomfort_budget <= 0.0:
        raise ValueError("operational score budgets must be positive")
    return cost_budget, discomfort_budget


def scalar_score(
    cost: float,
    discomfort: float,
    neutral_cost: float,
    neutral_discomfort: float,
    cost_budget: float,
    discomfort_budget: float,
) -> float:
    values = np.asarray(
        [cost, discomfort, neutral_cost, neutral_discomfort, cost_budget, discomfort_budget],
        dtype=float,
    )
    if not np.isfinite(values).all() or cost_budget <= 0.0 or discomfort_budget <= 0.0:
        raise ValueError("invalid scalar-score inputs")
    return float(
        COST_WEIGHT * (cost - neutral_cost) / cost_budget
        + DISCOMFORT_WEIGHT * (discomfort - neutral_discomfort) / discomfort_budget
    )
