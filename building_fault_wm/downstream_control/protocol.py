"""Fixed protocol and immutable input inventory for the downstream evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE = "bestest_hydronic_heat_pump"
BOPTEST_COMMIT = "0f8a467cb1823f005b6512937e9333c65e1e483e"
BOPTEST_VERSION = "1.0.0-dev"
WORKER_IMAGE_ID = (
    "sha256:28b5ebfe981237c3d3c381519c7a5feab526ff45603e8825fe43a6293c61910f"
)
FMU_SHA256 = "674b9500c1c89fdbec64f2c053af8ddcd06afebe63bd7c1b7fc708692a11863e"

STEP_SECONDS = 900
EPISODE_STEPS = 192
WARMUP_SECONDS = 86_400
HISTORY_STEPS = 48
CONTROL_HORIZON = 8
ACTION_DWELL_STEPS = 16
ACTION_LEVELS = (-1.0, 0.0, 1.0)
BASE_SETPOINT_K = 294.15
ACTION_AMPLITUDE_K = 0.75
MODEL_SEEDS = (202608011, 202608012, 202608013, 202608014, 202608015)
MODEL_POLICIES = ("legacy_rssm", "direct_h8_rssm", "deterministic_wm")
POLICIES = ("constant_zero", *MODEL_POLICIES)
CONTRASTS = (
    ("legacy_rssm", "constant_zero"),
    ("direct_h8_rssm", "constant_zero"),
    ("deterministic_wm", "constant_zero"),
    ("direct_h8_rssm", "legacy_rssm"),
    ("deterministic_wm", "direct_h8_rssm"),
)
CONDITIONS = (
    "clean",
    "zone_bias_negative",
    "zone_bias_positive",
    "zone_drift_negative",
    "zone_drift_positive",
    "zone_stuck",
)
FAULT_START = HISTORY_STEPS
FAULT_STOP = HISTORY_STEPS + 48
ZONE_BIAS_K = 2.0
ZONE_DRIFT_K_PER_STEP = 0.05
PILOT_DAY = 31

PLAN_PATH = (
    PROJECT_ROOT
    / "building_fault_wm/neural_benchmark/data_v6/plans/full"
    / f"{CASE}.json"
)
SCALER_PATH = (
    PROJECT_ROOT
    / "artifacts/direct_h8_publication_v2/experiment/prelock_bundle/frozen"
    / f"fit_scalers/{CASE}.json"
)
RSSM_ROOT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_publication_v2/experiment/prelock_bundle/frozen/checkpoints"
    / CASE
)
DETERMINISTIC_ROOT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_deterministic_transport_v3_training_bound_v2"
    / CASE
)
FMU_PATH = (
    (Path.home() / "external/project1-boptest/testcases")
    / CASE
    / "models/wrapped.fmu"
)
PROTOCOL_ROOT = PROJECT_ROOT / "artifacts/direct_h8_downstream_control_protocol_v1"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
PROTOCOL_DIGEST_PATH = PROTOCOL_ROOT / "protocol.canonical.sha256"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/direct_h8_downstream_control_v1"
IMPLEMENTATION_PATHS = (
    PROJECT_ROOT / "building_fault_wm/downstream_control/experiment.py",
    PROJECT_ROOT / "building_fault_wm/deterministic_transport/evaluate.py",
    PROJECT_ROOT / "building_fault_wm/deterministic_transport/config.py",
    PROJECT_ROOT / "building_fault_wm/deterministic_transport/gate.py",
    PROJECT_ROOT / "building_fault_wm/deterministic_transport/model.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/protocol.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/fault_data.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/reliability_loss.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/reliability_model.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/study_config.py",
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/study_train.py",
    PROJECT_ROOT / "building_fault_wm/recurrent_models/model.py",
    PROJECT_ROOT / "building_fault_wm/recurrent_models/training.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def load_plan() -> dict:
    value = json.loads(PLAN_PATH.read_text(encoding="ascii"))
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != 12:
        raise ValueError("transport plan must contain all 12 response-unseen windows")
    days = tuple(int(entry["day"]) for entry in entries)
    if len(set(days)) != 12:
        raise ValueError("transport plan days are not unique")
    if PILOT_DAY in days:
        raise ValueError("development pilot day overlaps the final response-unseen windows")
    return value


def rssm_checkpoint(seed: int, arm: str) -> Path:
    if seed not in MODEL_SEEDS or arm not in {"legacy", "ungated_h8"}:
        raise ValueError("unknown RSSM checkpoint identity")
    return RSSM_ROOT / f"seed{seed}/{arm}_u0400.pt"


def deterministic_checkpoint(seed: int) -> Path:
    if seed not in MODEL_SEEDS:
        raise ValueError("unknown deterministic checkpoint identity")
    return DETERMINISTIC_ROOT / f"seed{seed}/checkpoints/update_0400.pt"


def protocol_payload() -> dict:
    plan = load_plan()
    if sha256_file(FMU_PATH) != FMU_SHA256:
        raise ValueError("local BOPTEST FMU differs from the fixed case")
    inputs = {
        "case_plan": {"path": str(PLAN_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_file(PLAN_PATH)},
        "fit_scaler": {"path": str(SCALER_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_file(SCALER_PATH)},
        "rssm_checkpoints": {
            f"seed{seed}/{arm}": {
                "path": str(rssm_checkpoint(seed, arm).relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(rssm_checkpoint(seed, arm)),
            }
            for seed in MODEL_SEEDS
            for arm in ("legacy", "ungated_h8")
        },
        "deterministic_checkpoints": {
            f"seed{seed}": {
                "path": str(deterministic_checkpoint(seed).relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(deterministic_checkpoint(seed)),
            }
            for seed in MODEL_SEEDS
        },
        "boptest_fmu": {
            "path": f"project1-boptest/testcases/{CASE}/models/wrapped.fmu",
            "sha256": sha256_file(FMU_PATH),
        },
        "implementation": {
            str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256_file(path)}
            for path in IMPLEMENTATION_PATHS
        },
    }
    windows = [
        {
            "day": int(entry["day"]),
            "window_id": str(entry["window_id"]),
            "scenario_seed": int(entry["scenario_seed"]),
            "temperature_stratum": int(entry["temperature_stratum"]),
            "mean_outdoor_temperature_k": float(entry["mean_outdoor_temperature_k"]),
        }
        for entry in plan["entries"]
    ]
    payload = {
        "schema": "direct-h8-downstream-control-protocol-v1",
        "status": "reviewer_motivated_post_primary_analysis_descriptive",
        "case": CASE,
        "boptest": {
            "commit": BOPTEST_COMMIT,
            "version": BOPTEST_VERSION,
            "worker_image_id": WORKER_IMAGE_ID,
        },
        "windows": windows,
        "step_seconds": STEP_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "episode_steps": EPISODE_STEPS,
        "common_history_steps": HISTORY_STEPS,
        "control_horizon_steps": CONTROL_HORIZON,
        "action_dwell_steps": ACTION_DWELL_STEPS,
        "action_levels": list(ACTION_LEVELS),
        "base_setpoint_k": BASE_SETPOINT_K,
        "action_amplitude_k": ACTION_AMPLITUDE_K,
        "model_seeds": list(MODEL_SEEDS),
        "policies": list(POLICIES),
        "paired_contrasts": [
            {"candidate": candidate, "reference": reference}
            for candidate, reference in CONTRASTS
        ],
        "conditions": list(CONDITIONS),
        "development_pilot": {
            "day": PILOT_DAY,
            "excluded_from_final_evaluation": True,
            "purpose": "verify the simulator interface and controller execution before protocol freeze",
        },
        "fault": {
            "channel": "zone_temperature_k",
            "start_step": FAULT_START,
            "stop_step_exclusive": FAULT_STOP,
            "bias_k": ZONE_BIAS_K,
            "drift_k_per_step": ZONE_DRIFT_K_PER_STEP,
            "bias_and_drift_signs": [-1, 1],
            "stuck_value": "last clean value before onset",
            "scope": "model observation only; simulator state and KPI calculation remain unmodified",
        },
        "controller": {
            "ensemble_reduction": "pointwise median across five fixed model seeds",
            "candidate_sequences": "each action level held constant for eight predicted steps",
            "selection": (
                "minimize predicted electricity cost among candidates with zero predicted "
                "comfort violation; if none are feasible, minimize predicted K h violation, "
                "then predicted electricity cost, absolute action, and signed action"
            ),
            "execution": "apply the selected level for 16 simulator steps before replanning",
            "rssm_filter_mode": "bypass",
            "rssm_sampling": False,
        },
        "endpoints": {
            "primary_descriptive": ["BOPTEST cost_tot", "BOPTEST tdis_tot"],
            "supporting": [
                "control-stage electricity-cost proxy",
                "control-stage temperature-discomfort K h",
                "control-stage electric energy kWh",
                "action selections and model disagreements",
            ],
            "reporting": (
                "paired per-window values, medians, and descriptive paired bootstrap intervals; "
                "negative candidate-minus-reference differences favor the candidate for each endpoint; "
                "no confirmation label"
            ),
        },
        "claim_limit": (
            "One deterministic simulator case and a discrete supervisory controller; "
            "not evidence of occupied-building safety, deployment benefit, or universal control superiority."
        ),
        "inputs": inputs,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def validate_frozen_protocol() -> dict:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_DIGEST_PATH.is_file():
        raise FileNotFoundError("downstream protocol has not been frozen")
    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))
    current = protocol_payload()
    if frozen != current:
        raise ValueError("current downstream protocol differs from the frozen protocol")
    recorded = PROTOCOL_DIGEST_PATH.read_text(encoding="ascii").strip()
    if recorded != canonical_sha256(frozen):
        raise ValueError("downstream protocol digest is invalid")
    return frozen
