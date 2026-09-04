"""Generate deterministic publication tables and figures from sealed v3 outputs.

The reporter is deliberately outcome agnostic.  It accepts only a byte-bound
completed evaluation tree, reconstructs the frozen gate from the persisted core
CSV, and renders the same artifact set for every possible gate category.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluate, gate, plan, run_evaluation
from .config import FROZEN_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = run_evaluation.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_report"
)
REPORT_SCHEMA = "direct-h8-deterministic-transport-report-v1"
REPORT_RECEIPT_NAME = "report_manifest.json"

SOURCE_FILES = {
    run_evaluation.CORE_NAME,
    run_evaluation.DETAIL_NAME,
    run_evaluation.DIAGNOSTIC_SUMMARY_NAME,
    run_evaluation.GATE_RESULT_NAME,
    run_evaluation.PROVENANCE_NAME,
    run_evaluation.COMPLETION_NAME,
}
TABLE_FILES = (
    "primary_estimands.csv",
    "stratum_effects.csv",
    "horizon_arm_mae.csv",
    "raw_unit_mae.csv",
    "boundary_diagnostics.csv",
    "action_sensitivity.csv",
    "fault_family_diagnostics.csv",
    "model_resources.csv",
    "paired_dwell_transport.csv",
)
MARKDOWN_TABLE_FILES = tuple(name.replace(".csv", ".md") for name in TABLE_FILES)
FIGURE_FILES = (
    "figure_primary_estimands.png",
    "figure_horizon_arm_mae.png",
    "figure_boundary_diagnostics.png",
)
RESULTS_NAME = "RESULTS.md"


@dataclass(frozen=True)
class VerifiedEvaluation:
    core: pd.DataFrame
    detailed: pd.DataFrame
    diagnostic_summary: pd.DataFrame
    gate_result: dict
    provenance: dict
    completion: dict
    completion_file_sha256: str


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"report input is not a plain file: {path}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="ascii"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {value}")
        ),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _artifact_inventory(root: Path, *, exclude: Iterable[str] = ()) -> list[dict]:
    excluded = set(exclude)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symbolic link is forbidden in artifact tree: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                rows.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": plan.sha256_file(path),
                    }
                )
        elif not path.is_dir():
            raise ValueError(f"non-regular artifact entry: {path}")
    return rows


def _require_finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} is negative or non-finite")
    return numeric


def _validate_completion(
    evaluation_dir: Path, expected_completion_sha256: str
) -> tuple[dict, str]:
    if not _valid_sha256(expected_completion_sha256):
        raise ValueError("expected_completion_sha256 is not a lowercase SHA-256")
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise ValueError("evaluation input is not a plain directory")
    completion_path = evaluation_dir / run_evaluation.COMPLETION_NAME
    actual_completion_sha256 = plan.sha256_file(completion_path)
    if actual_completion_sha256 != expected_completion_sha256:
        raise ValueError("evaluation completion receipt differs from expected SHA-256")
    completion = _strict_json(completion_path)
    expected_fields = {
        "schema",
        "study_kind",
        "prelock_registry_sha256",
        "readiness_sha256",
        "corpus_manifest_payload_sha256",
        "fault_manifest_sha256",
        "gate_result_sha256",
        "provenance_file_sha256",
        "artifact_inventory_excludes_completion",
        "artifact_inventory_sha256",
        "complete",
    }
    if set(completion) != expected_fields:
        raise ValueError("evaluation completion receipt fields changed")
    if (
        completion["schema"] != run_evaluation.COMPLETION_SCHEMA
        or completion["study_kind"] != "direct_h8_deterministic_transport_v3"
        or completion["complete"] is not True
    ):
        raise ValueError("evaluation completion identity is invalid")
    for field in (
        "prelock_registry_sha256",
        "readiness_sha256",
        "corpus_manifest_payload_sha256",
        "fault_manifest_sha256",
        "gate_result_sha256",
        "provenance_file_sha256",
        "artifact_inventory_sha256",
    ):
        if not _valid_sha256(completion[field]):
            raise ValueError(f"evaluation completion has invalid {field}")
    inventory = completion["artifact_inventory_excludes_completion"]
    if not isinstance(inventory, list):
        raise ValueError("evaluation completion inventory is not a list")
    actual_inventory = _artifact_inventory(
        evaluation_dir, exclude={run_evaluation.COMPLETION_NAME}
    )
    if (
        inventory != actual_inventory
        or completion["artifact_inventory_sha256"]
        != plan.canonical_sha256(actual_inventory)
    ):
        raise ValueError("evaluation artifact inventory differs from run receipt")
    paths = {
        item.get("path")
        for item in inventory
        if isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}
    }
    if SOURCE_FILES - paths - {run_evaluation.COMPLETION_NAME}:
        raise ValueError("evaluation inventory omits a required report source")
    return completion, actual_completion_sha256


def _validate_provenance(provenance: dict, core_rows: int) -> None:
    if (
        provenance.get("schema") != run_evaluation.PROVENANCE_SCHEMA
        or provenance.get("study_kind")
        != "direct_h8_deterministic_transport_v3"
    ):
        raise ValueError("evaluation provenance identity changed")
    contract = provenance.get("evaluation_contract")
    expected_contract = {
        "cases": list(gate.CASES),
        "model_seeds": list(gate.CONFIRMATION_SEEDS),
        "arms": list(gate.ARMS),
        "policies": list(gate.POLICIES),
        "horizons": list(evaluate.EVALUATION_HORIZONS),
        "history": evaluate.EVALUATION_HISTORY,
        "role": "locked_test",
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) != value for key, value in expected_contract.items()
    ):
        raise ValueError("evaluation provenance contract changed")
    if (
        contract.get("bootstrap_draws") != gate.BOOTSTRAP_DRAWS
        or contract.get("bootstrap_seed") != gate.BOOTSTRAP_SEED
        or not isinstance(contract.get("fault_spec"), dict)
    ):
        raise ValueError("evaluation provenance inference contract changed")
    counts = provenance.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("gate_core_rows") != core_rows
        or counts.get("detailed_rows") != core_rows
        or counts.get("evaluation_units")
        != len(gate.CASES) * len(gate.CONFIRMATION_SEEDS)
    ):
        raise ValueError("evaluation provenance row counts changed")
    policy = provenance.get("runtime_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("device") != "cpu"
        or policy.get("deterministic_algorithms") is not True
    ):
        raise ValueError("evaluation runtime policy changed")
    resources = provenance.get("model_resources")
    if not isinstance(resources, dict):
        raise ValueError("evaluation model resources are absent")
    expected_resource_fields = {
        "rssm_total_parameters",
        "rssm_active_observation_dynamics_parameters",
        "deterministic_total_parameters",
        "selected_checkpoint_bytes",
        "training",
    }
    if set(resources) != expected_resource_fields:
        raise ValueError("evaluation model-resource schema changed")
    if (
        resources["rssm_active_observation_dynamics_parameters"] != 19_784
        or resources["deterministic_total_parameters"] != 19_789
        or not isinstance(resources["rssm_total_parameters"], int)
        or resources["rssm_total_parameters"] <= 0
    ):
        raise ValueError("evaluation parameter counts changed")
    checkpoint_bytes = resources["selected_checkpoint_bytes"]
    expected_checkpoint_keys = {
        f"{case}/seed{seed}/{arm}"
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
        for arm in gate.ARMS
    }
    if (
        not isinstance(checkpoint_bytes, dict)
        or set(checkpoint_bytes) != expected_checkpoint_keys
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in checkpoint_bytes.values()
        )
    ):
        raise ValueError("selected-checkpoint size grid changed")
    _validate_training_resources(resources["training"])
    _validate_evaluation_timing(provenance.get("timing"))


def _validate_training_resources(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "updates_per_model",
        "deterministic",
        "rssm",
    }:
        raise ValueError("training-resource schema changed")
    if payload["updates_per_model"] != 400:
        raise ValueError("training update count changed")
    deterministic = payload["deterministic"]
    if not isinstance(deterministic, dict) or set(deterministic) != {
        "timing_source_file_sha256",
        "wall_seconds_by_case_seed",
        "available_count",
        "unavailable_count",
        "unavailable_reason",
    }:
        raise ValueError("deterministic training-timing schema changed")
    if not _valid_sha256(deterministic["timing_source_file_sha256"]):
        raise ValueError("deterministic training-timing source hash is invalid")
    timings = deterministic["wall_seconds_by_case_seed"]
    expected_units = {
        f"{case}/seed{seed}"
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
    }
    if not isinstance(timings, dict) or set(timings) != expected_units:
        raise ValueError("deterministic training-timing grid changed")
    available = 0
    for key, value in timings.items():
        if value is None:
            continue
        if _require_finite_nonnegative(value, f"training time {key}") == 0.0:
            raise ValueError("an available training time is zero")
        available += 1
    unavailable = len(timings) - available
    if (
        deterministic["available_count"] != available
        or deterministic["unavailable_count"] != unavailable
        or (
            unavailable > 0
            and not isinstance(deterministic["unavailable_reason"], str)
        )
    ):
        raise ValueError("deterministic training availability counts changed")
    rssm = payload["rssm"]
    if (
        not isinstance(rssm, dict)
        or set(rssm) != {"available", "reason"}
        or rssm["available"] is not False
        or not isinstance(rssm["reason"], str)
        or not rssm["reason"]
    ):
        raise ValueError("RSSM training-time availability statement changed")


def _validate_evaluation_timing(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "total_wall_seconds",
        "unit_timing",
        "peak_rss_kb_before_evaluation",
        "peak_rss_kb_after_evaluation",
    }:
        raise ValueError("evaluation timing schema changed")
    _require_finite_nonnegative(payload["total_wall_seconds"], "total runtime")
    before = _require_finite_nonnegative(
        payload["peak_rss_kb_before_evaluation"], "peak RSS before"
    )
    after = _require_finite_nonnegative(
        payload["peak_rss_kb_after_evaluation"], "peak RSS after"
    )
    if after < before:
        raise ValueError("peak RSS decreased inside one process")
    rows = payload["unit_timing"]
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError("evaluation unit-timing grid changed")
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "case",
            "model_seed",
            "model_load_seconds",
            "prediction_seconds_by_arm",
            "unit_wall_seconds",
            "core_rows",
        }:
            raise ValueError("evaluation unit-timing row schema changed")
        identities.add((row["case"], row["model_seed"]))
        for field in ("model_load_seconds", "unit_wall_seconds"):
            _require_finite_nonnegative(row[field], field)
        prediction = row["prediction_seconds_by_arm"]
        if not isinstance(prediction, dict) or set(prediction) != set(gate.ARMS):
            raise ValueError("per-arm prediction timing grid changed")
        for arm, value in prediction.items():
            _require_finite_nonnegative(value, f"{arm} prediction time")
        if (
            isinstance(row["core_rows"], bool)
            or not isinstance(row["core_rows"], int)
            or row["core_rows"] <= 0
        ):
            raise ValueError("evaluation unit core-row count is invalid")
    expected = {
        (case, seed)
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
    }
    if identities != expected:
        raise ValueError("evaluation unit-timing identities changed")


def load_verified_evaluation(
    evaluation_dir: Path,
    *,
    expected_completion_sha256: str,
) -> VerifiedEvaluation:
    """Load, reconstruct, and byte-verify all report inputs."""

    completion, completion_sha = _validate_completion(
        evaluation_dir, expected_completion_sha256
    )
    core = pd.read_csv(evaluation_dir / run_evaluation.CORE_NAME)
    detailed = pd.read_csv(evaluation_dir / run_evaluation.DETAIL_NAME)
    diagnostic_summary = pd.read_csv(
        evaluation_dir / run_evaluation.DIAGNOSTIC_SUMMARY_NAME
    )
    if tuple(core.columns) != gate.REQUIRED_COLUMNS:
        raise ValueError("persisted report core schema changed")
    if tuple(detailed.columns) != evaluate.DETAIL_COLUMNS:
        raise ValueError("persisted report detail schema changed")
    numeric = detailed.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if detailed.empty or not np.isfinite(numeric).all():
        raise ValueError("persisted report diagnostics are empty or non-finite")
    expected_units = {
        "zone_temperature_k": "K",
        "hvac_electric_power_w": "W",
    }
    observed_units = (
        detailed.loc[:, ["fault_channel", "raw_unit"]]
        .drop_duplicates()
        .set_index("fault_channel")["raw_unit"]
        .to_dict()
    )
    if observed_units != expected_units:
        raise ValueError("raw diagnostic units changed")
    expected_boundary_classes = {
        "old_2h": {True},
        "new_4h": {False, True},
    }
    for policy, expected in expected_boundary_classes.items():
        observed = set(
            detailed.loc[
                detailed["policy"] == policy, "boundary_crossing"
            ].astype(bool)
        )
        if observed != expected:
            raise ValueError(
                f"boundary diagnostics differ from frozen {policy} geometry"
            )
    reconstructed_diagnostics = run_evaluation._diagnostic_summary(detailed)
    pd.testing.assert_frame_equal(
        diagnostic_summary.reset_index(drop=True),
        reconstructed_diagnostics.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        core.reset_index(drop=True),
        detailed.loc[:, gate.REQUIRED_COLUMNS].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    gate.validate_input(core)
    recorded = _strict_json(evaluation_dir / run_evaluation.GATE_RESULT_NAME)
    reconstructed = gate.analyze_gate(
        core,
        bootstrap_draws=gate.BOOTSTRAP_DRAWS,
        bootstrap_seed=gate.BOOTSTRAP_SEED,
    )
    if (
        recorded != reconstructed
        or completion["gate_result_sha256"]
        != plan.canonical_sha256(reconstructed)
    ):
        raise ValueError("recorded gate does not reconstruct from persisted core CSV")
    provenance_path = evaluation_dir / run_evaluation.PROVENANCE_NAME
    provenance = _strict_json(provenance_path)
    if completion["provenance_file_sha256"] != plan.sha256_file(provenance_path):
        raise ValueError("evaluation provenance file hash changed")
    _validate_provenance(provenance, len(core))
    return VerifiedEvaluation(
        core=core,
        detailed=detailed,
        diagnostic_summary=diagnostic_summary,
        gate_result=recorded,
        provenance=provenance,
        completion=completion,
        completion_file_sha256=completion_sha,
    )


def _equal_weight_mean(
    frame: pd.DataFrame,
    *,
    value: str,
    retain: Sequence[str],
) -> pd.DataFrame:
    """Apply the frozen anchor-to-case equal-weight hierarchy."""

    retain = tuple(retain)
    if len(set(retain)) != len(retain):
        raise ValueError("report aggregation repeats a retained dimension")
    base = [
        "case",
        "policy",
        "window_id",
        "model_seed",
        "family",
        "fault_channel",
        "sign",
        "severity",
        "horizon",
        "arm",
    ]
    for dimension in retain:
        if dimension not in frame.columns:
            raise ValueError(f"report aggregation lacks {dimension}")
        if dimension not in base:
            base.append(dimension)
    result = frame.groupby(base, as_index=False, dropna=False)[value].mean()
    collapse = (
        "sign",
        "severity",
        "fault_channel",
        "family",
        "window_id",
        "model_seed",
        "case",
    )
    for dimension in collapse:
        if dimension in retain:
            continue
        grouping = [column for column in base if column != dimension]
        result = result.groupby(grouping, as_index=False, dropna=False)[
            value
        ].mean()
        base = grouping
    return result.sort_values(base, kind="stable").reset_index(drop=True)


def build_primary_estimands(result: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    results = result["results"]
    assert isinstance(results, dict)
    comparisons = {
        "A": "ungated_h8 RSSM vs deterministic_wm",
        "D": "ungated_h8 RSSM vs legacy RSSM",
    }
    for policy in gate.POLICIES:
        for estimand in ("A", "D"):
            values = results[policy][estimand]
            rows.append(
                {
                    "policy": policy,
                    "analysis_role": (
                        "primary" if policy == "new_4h" else "transport_control"
                    ),
                    "estimand": estimand,
                    "comparison": comparisons[estimand],
                    "positive_direction": (
                        "RSSM" if estimand == "A" else "direct_H8"
                    ),
                    "estimate_percent": 100.0 * values["point"],
                    "ci90_lower_percent": 100.0 * values["ci90_lower"],
                    "ci90_upper_percent": 100.0 * values["ci90_upper"],
                    "ci95_lower_percent": 100.0 * values["ci95_lower"],
                    "ci95_upper_percent": 100.0 * values["ci95_upper"],
                    "category": values["category"],
                    "positive_seed_count": values["positive_seed_count"],
                    "negative_seed_count": values["negative_seed_count"],
                }
            )
    return pd.DataFrame(rows)


def build_stratum_effects(result: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    results = result["results"]
    assert isinstance(results, dict)
    for policy in gate.POLICIES:
        for estimand in ("A", "D"):
            values = results[policy][estimand]
            for key, output_name in (
                ("by_case", "case"),
                ("by_family", "family"),
                ("by_seed", "model_seed"),
            ):
                for stratum, effect in sorted(values[key].items()):
                    rows.append(
                        {
                            "policy": policy,
                            "estimand": estimand,
                            "stratum_type": output_name,
                            "stratum": stratum,
                            "estimate_percent": 100.0 * effect,
                        }
                    )
    return pd.DataFrame(rows)


def build_horizon_arm_mae(detailed: pd.DataFrame) -> pd.DataFrame:
    selected = detailed.loc[detailed["family"].isin(gate.SILENT_FAMILIES)]
    result = _equal_weight_mean(
        selected,
        value="standardized_abs_error",
        retain=(),
    ).rename(
        columns={"standardized_abs_error": "standardized_mae"}
    )
    return result.loc[:, ["policy", "horizon", "arm", "standardized_mae"]]


def build_raw_unit_mae(detailed: pd.DataFrame) -> pd.DataFrame:
    selected = detailed.loc[detailed["family"].isin(gate.SILENT_FAMILIES)]
    result = _equal_weight_mean(
        selected,
        value="raw_abs_error",
        retain=("fault_channel", "raw_unit"),
    )
    return result.loc[
        :,
        [
            "policy",
            "horizon",
            "arm",
            "fault_channel",
            "raw_unit",
            "raw_abs_error",
        ],
    ].rename(columns={"raw_abs_error": "raw_mae"})


def build_boundary_diagnostics(detailed: pd.DataFrame) -> pd.DataFrame:
    selected = detailed.loc[detailed["family"].isin(gate.SILENT_FAMILIES)]
    means = _equal_weight_mean(
        selected,
        value="standardized_abs_error",
        retain=("boundary_crossing",),
    ).rename(columns={"standardized_abs_error": "standardized_mae"})
    counts = (
        selected.groupby(
            ["policy", "horizon", "arm", "boundary_crossing"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "diagnostic_rows"})
    )
    result = means.merge(
        counts,
        on=["policy", "horizon", "arm", "boundary_crossing"],
        validate="one_to_one",
    )
    result["boundary_class"] = np.where(
        result["boundary_crossing"], "boundary_crossing", "within_dwell"
    )
    return result.loc[
        :,
        [
            "policy",
            "horizon",
            "arm",
            "boundary_class",
            "standardized_mae",
            "diagnostic_rows",
        ],
    ]


def build_action_sensitivity(detailed: pd.DataFrame) -> pd.DataFrame:
    selected = detailed.loc[detailed["family"].isin(gate.SILENT_FAMILIES)]
    means = _equal_weight_mean(
        selected,
        value="action_prediction_change_standardized",
        retain=("boundary_crossing",),
    ).rename(
        columns={
            "action_prediction_change_standardized": (
                "alternate_action_prediction_change_standardized"
            )
        }
    )
    counts = (
        selected.groupby(
            ["policy", "horizon", "arm", "boundary_crossing"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "diagnostic_rows"})
    )
    result = means.merge(
        counts,
        on=["policy", "horizon", "arm", "boundary_crossing"],
        validate="one_to_one",
    )
    result["boundary_class"] = np.where(
        result["boundary_crossing"], "boundary_crossing", "within_dwell"
    )
    return result.loc[
        :,
        [
            "policy",
            "horizon",
            "arm",
            "boundary_class",
            "alternate_action_prediction_change_standardized",
            "diagnostic_rows",
        ],
    ]


def build_fault_family_diagnostics(detailed: pd.DataFrame) -> pd.DataFrame:
    result = _equal_weight_mean(
        detailed,
        value="standardized_abs_error",
        retain=("family",),
    ).rename(columns={"standardized_abs_error": "standardized_mae"})
    return result.loc[
        :,
        ["policy", "horizon", "arm", "family", "standardized_mae"],
    ]


def build_paired_dwell_transport(result: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    results = result["results"]
    transport = result["transport"]
    assert isinstance(results, dict) and isinstance(transport, dict)
    for estimand in ("A", "D"):
        values = transport[estimand]["new_4h_minus_old_2h"]
        rows.append(
            {
                "estimand": estimand,
                "old_2h_estimate_percent": (
                    100.0 * results["old_2h"][estimand]["point"]
                ),
                "new_4h_estimate_percent": (
                    100.0 * results["new_4h"][estimand]["point"]
                ),
                "new_minus_old_percent": 100.0 * values["point"],
                "ci90_lower_percent": 100.0 * values["ci90_lower"],
                "ci90_upper_percent": 100.0 * values["ci90_upper"],
                "ci95_lower_percent": 100.0 * values["ci95_lower"],
                "ci95_upper_percent": 100.0 * values["ci95_upper"],
                "old_2h_category": results["old_2h"][estimand]["category"],
                "new_4h_category": results["new_4h"][estimand]["category"],
                "persistent_across_dwell": transport[estimand][
                    "persistent_across_dwell"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_model_resources(
    provenance: Mapping[str, object], detailed: pd.DataFrame
) -> pd.DataFrame:
    resources = provenance["model_resources"]
    timing = provenance["timing"]
    assert isinstance(resources, dict) and isinstance(timing, dict)
    training = resources["training"]
    assert isinstance(training, dict)
    deterministic_training = training["deterministic"]
    assert isinstance(deterministic_training, dict)
    available_times = [
        float(value)
        for value in deterministic_training[
            "wall_seconds_by_case_seed"
        ].values()
        if value is not None
    ]
    prediction_seconds = {arm: 0.0 for arm in gate.ARMS}
    for row in timing["unit_timing"]:
        for arm in gate.ARMS:
            prediction_seconds[arm] += float(
                row["prediction_seconds_by_arm"][arm]
            )
    scored_rows = detailed.groupby("arm").size().to_dict()
    checkpoint_bytes = resources["selected_checkpoint_bytes"]
    rows = []
    for arm in gate.ARMS:
        sizes = [
            int(value)
            for key, value in checkpoint_bytes.items()
            if key.endswith(f"/{arm}")
        ]
        if len(sizes) != 15:
            raise ValueError(f"checkpoint-size grid is incomplete for {arm}")
        is_deterministic = arm == "deterministic_wm"
        if is_deterministic:
            training_seconds = sum(available_times)
            timed_models = deterministic_training["available_count"]
            untimed_models = deterministic_training["unavailable_count"]
            training_status = (
                "partial_recorded"
                if untimed_models
                else "complete_recorded"
            )
            parameters = resources["deterministic_total_parameters"]
            active_parameters = parameters
            direct_h8_endpoints = FROZEN_CONFIG.h8_endpoints_per_update
        else:
            training_seconds = np.nan
            timed_models = 0
            untimed_models = 15
            training_status = "not_recorded_in_parent"
            parameters = resources["rssm_total_parameters"]
            active_parameters = resources[
                "rssm_active_observation_dynamics_parameters"
            ]
            direct_h8_endpoints = (
                0
                if arm == "legacy"
                else FROZEN_CONFIG.h8_endpoints_per_update
            )
        seconds = prediction_seconds[arm]
        rows.append(
            {
                "arm": arm,
                "trainable_parameters": parameters,
                "active_observation_dynamics_parameters": active_parameters,
                "optimizer_updates_per_model": training["updates_per_model"],
                "source_steps_per_update": (
                    FROZEN_CONFIG.batch_size * FROZEN_CONFIG.sequence_length
                ),
                "direct_h8_endpoints_per_update": direct_h8_endpoints,
                "training_wall_seconds_available_sum": training_seconds,
                "training_timed_model_count": timed_models,
                "training_untimed_model_count": untimed_models,
                "training_time_status": training_status,
                "checkpoint_bytes_mean": float(np.mean(sizes)),
                "checkpoint_bytes_min": min(sizes),
                "checkpoint_bytes_max": max(sizes),
                "prediction_wall_seconds": seconds,
                "prediction_microseconds_per_diagnostic_row": (
                    1e6 * seconds / int(scored_rows[arm])
                ),
                "evaluation_total_wall_seconds": timing[
                    "total_wall_seconds"
                ],
                "evaluation_peak_rss_kb": timing[
                    "peak_rss_kb_after_evaluation"
                ],
            }
        )
    return pd.DataFrame(rows)


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "not recorded"
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _write_ascii(path: Path, content: str) -> None:
    path.write_bytes(content.encode("ascii"))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.write_bytes(
        frame.to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        ).encode("ascii")
    )


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "wm-buildings-experiment-v3"},
    )
    plt.close(figure)


def _plot_primary(frame: pd.DataFrame, path: Path) -> None:
    _figure_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    colors = {"old_2h": "#2878B5", "new_4h": "#D95F02"}
    for axis, estimand in zip(axes, ("A", "D"), strict=True):
        selected = frame.loc[frame["estimand"] == estimand].reset_index(drop=True)
        for index, row in selected.iterrows():
            point = row["estimate_percent"]
            low95 = max(0.0, point - row["ci95_lower_percent"])
            high95 = max(0.0, row["ci95_upper_percent"] - point)
            low90 = max(0.0, point - row["ci90_lower_percent"])
            high90 = max(0.0, row["ci90_upper_percent"] - point)
            axis.errorbar(
                point,
                index,
                xerr=np.asarray([[low95], [high95]]),
                fmt="o",
                color=colors[row["policy"]],
                capsize=3,
                linewidth=1,
            )
            axis.errorbar(
                point,
                index,
                xerr=np.asarray([[low90], [high90]]),
                fmt="none",
                color=colors[row["policy"]],
                linewidth=4,
                alpha=0.55,
            )
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.axvspan(-5.0, 5.0, color="#999999", alpha=0.12)
        axis.set_yticks(range(len(selected)), selected["policy"])
        axis.set_title(
            "Architecture contrast A" if estimand == "A" else "H8 supervision contrast D"
        )
        axis.set_xlabel("Relative effect (%)")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Action-dwell policy")
    figure.suptitle("Frozen primary and transport-control estimates", y=1.02)
    _save_figure(figure, path)


def _plot_horizon(frame: pd.DataFrame, path: Path) -> None:
    _figure_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    colors = {
        "legacy": "#777777",
        "ungated_h8": "#2878B5",
        "deterministic_wm": "#D95F02",
    }
    labels = {
        "legacy": "Legacy RSSM",
        "ungated_h8": "Direct-H8 RSSM",
        "deterministic_wm": "Deterministic WM",
    }
    for axis, policy in zip(axes, gate.POLICIES, strict=True):
        selected = frame.loc[frame["policy"] == policy]
        for arm in gate.ARMS:
            rows = selected.loc[selected["arm"] == arm].sort_values("horizon")
            axis.plot(
                rows["horizon"],
                rows["standardized_mae"],
                marker="o",
                linewidth=1.5,
                color=colors[arm],
                label=labels[arm],
            )
        axis.set_xticks(evaluate.EVALUATION_HORIZONS)
        axis.set_xlabel("Prediction horizon (15-minute steps)")
        axis.set_title(policy)
        axis.grid(color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Affected-channel standardized MAE")
    axes[1].legend(frameon=False)
    figure.suptitle("Long-horizon error under paired action dwell", y=1.02)
    _save_figure(figure, path)


def _plot_boundary(frame: pd.DataFrame, path: Path) -> None:
    _figure_style()
    selected = frame.loc[frame["horizon"] == 8].copy()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    colors = {"within_dwell": "#3C8D5A", "boundary_crossing": "#C44E52"}
    for axis, policy in zip(axes, gate.POLICIES, strict=True):
        policy_rows = selected.loc[selected["policy"] == policy]
        x = np.arange(len(gate.ARMS), dtype=float)
        width = 0.34
        available = [
            boundary_class
            for boundary_class in ("within_dwell", "boundary_crossing")
            if boundary_class in set(policy_rows["boundary_class"])
        ]
        if not available:
            raise ValueError(f"boundary diagnostic is empty for {policy}")
        offsets = (0.0,) if len(available) == 1 else (-0.5, 0.5)
        for offset, boundary_class in zip(offsets, available, strict=True):
            rows = (
                policy_rows.loc[
                    policy_rows["boundary_class"] == boundary_class
                ]
                .set_index("arm")
                .loc[list(gate.ARMS)]
            )
            axis.bar(
                x + offset * width,
                rows["standardized_mae"],
                width=width,
                color=colors[boundary_class],
                label=boundary_class.replace("_", " "),
            )
        axis.set_xticks(x, ("Legacy", "H8 RSSM", "Det. WM"))
        axis.set_xlabel("Model arm")
        axis.set_title(policy)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("H8 standardized MAE")
    axes[1].legend(frameon=False)
    figure.suptitle("Within-dwell and boundary-crossing diagnostics", y=1.02)
    _save_figure(figure, path)


def _percent_interval(values: Mapping[str, object], level: int = 95) -> str:
    return (
        f"{100.0 * float(values['point']):.2f}% "
        f"({level}% CI {100.0 * float(values[f'ci{level}_lower']):.2f}% to "
        f"{100.0 * float(values[f'ci{level}_upper']):.2f}%)"
    )


def _results_markdown(
    verified: VerifiedEvaluation,
    tables: Mapping[str, pd.DataFrame],
) -> str:
    result = verified.gate_result
    primary = result["results"]["new_4h"]
    control = result["results"]["old_2h"]
    transport = result["transport"]
    horizon = tables["horizon_arm_mae.csv"]
    h8 = horizon.loc[
        (horizon["policy"] == "new_4h") & (horizon["horizon"] == 8)
    ].set_index("arm")
    resources = tables["model_resources.csv"].set_index("arm")
    raw = tables["raw_unit_mae.csv"]
    raw_h8 = raw.loc[
        (raw["policy"] == "new_4h") & (raw["horizon"] == 8)
    ]
    raw_fragments = []
    for channel, unit in (
        ("zone_temperature_k", "K"),
        ("hvac_electric_power_w", "W"),
    ):
        rows = raw_h8.loc[raw_h8["fault_channel"] == channel].set_index("arm")
        values = ", ".join(
            f"{arm} {rows.loc[arm, 'raw_mae']:.3f} {unit}" for arm in gate.ARMS
        )
        raw_fragments.append(f"{channel}: {values}")
    families = tables["fault_family_diagnostics.csv"]
    family_h8 = families.loc[
        (families["policy"] == "new_4h")
        & (families["horizon"] == 8)
        & families["family"].isin(("healthy", "dropout"))
    ]
    family_fragments = []
    for family in ("healthy", "dropout"):
        rows = family_h8.loc[family_h8["family"] == family].set_index("arm")
        family_fragments.append(
            f"{family}: "
            + ", ".join(
                f"{arm} {rows.loc[arm, 'standardized_mae']:.4f}"
                for arm in gate.ARMS
            )
        )
    training_row = resources.loc["deterministic_wm"]
    lines = [
        "# Results",
        "",
        "## Frozen primary contrasts",
        "",
        (
            "For the primary `new_4h` policy, architecture estimand A was "
            f"{_percent_interval(primary['A'])}; the frozen category was "
            f"`{primary['A']['category']}`. Positive A favors the direct-H8 "
            "RSSM and negative A favors the deterministic recurrent world model."
        ),
        "",
        (
            "For the same policy, supervision estimand D was "
            f"{_percent_interval(primary['D'])}; the frozen category was "
            f"`{primary['D']['category']}`. Positive D favors direct-H8 "
            "supervision over the legacy RSSM objective."
        ),
        "",
        (
            "Under the paired `old_2h` transport-control policy, A was "
            f"{_percent_interval(control['A'])} "
            f"(`{control['A']['category']}`), and D was "
            f"{_percent_interval(control['D'])} "
            f"(`{control['D']['category']}`)."
        ),
        "",
        (
            "The paired `new_4h - old_2h` change was "
            f"{_percent_interval(transport['A']['new_4h_minus_old_2h'])} for A "
            f"and {_percent_interval(transport['D']['new_4h_minus_old_2h'])} "
            "for D. Persistence flags apply only when both dwell policies have "
            "the same non-inconclusive frozen category."
        ),
        "",
        "## Error and diagnostic results",
        "",
        (
            "Primary-policy H8 standardized MAE was "
            + ", ".join(
                f"{arm} {h8.loc[arm, 'standardized_mae']:.4f}"
                for arm in gate.ARMS
            )
            + ". H1/H2/H4/H8 values for both policies are reported in "
            "`horizon_arm_mae.csv`."
        ),
        "",
        (
            "Raw H8 errors are kept in separate physical units and are never "
            "averaged together. " + "; ".join(raw_fragments) + "."
        ),
        "",
        (
            "The prespecified primary-policy H8 healthy and dropout diagnostics "
            "were " + "; ".join(family_fragments) + ". All fault families and "
            "horizons are published in `fault_family_diagnostics.csv`."
        ),
        "",
        (
            "The boundary table separates forecasts contained within one action "
            "dwell from forecasts crossing an action transition. The action "
            "sensitivity table reports prediction change under a prespecified "
            "alternate action block; it is a response diagnostic, not evidence "
            "of closed-loop control quality or causal treatment benefit. Cells "
            "that cannot occur under the frozen action schedule and fault anchors "
            "are left structurally absent rather than imputed."
        ),
        "",
        "## Resources",
        "",
        (
            "The deterministic comparator has "
            f"{int(training_row['trainable_parameters']):,} trainable parameters "
            "and the RSSM has "
            f"{int(resources.loc['ungated_h8', 'active_observation_dynamics_parameters']):,} "
            "active observation-dynamics parameters. Every model used 400 "
            "optimizer updates. Recorded deterministic training wall time covers "
            f"{int(training_row['training_timed_model_count'])} of 15 models; "
            "unrecorded runs and parent RSSM training times remain explicitly "
            "missing rather than estimated."
        ),
        "",
        "## Scope and limits",
        "",
        (
            "These results characterize deterministic open-loop prediction in "
            "three public BOPTEST simulation cases under synthetic sensor faults, "
            "two matched action-dwell schedules, five fixed model seeds, and the "
            "frozen training budget. They do not establish intrinsic superiority "
            "of stochastic or deterministic world models, real-building "
            "generalization, fault prevalence, or closed-loop energy, comfort, "
            "safety, or financial benefit."
        ),
        "",
        "## Reproducibility",
        "",
        (
            "This report was generated only after reconstructing the frozen gate "
            "from the persisted core CSV and matching the detailed CSV row for "
            "row. Source `evaluation_complete.json` SHA-256: "
            f"`{verified.completion_file_sha256}`. Source artifact-inventory "
            f"digest: `{verified.completion['artifact_inventory_sha256']}`."
        ),
        "",
    ]
    return "\n".join(lines)


def _seal_tree(root: Path) -> None:
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


def generate_report(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    expected_completion_sha256: str,
) -> Path:
    """Verify a sealed evaluation and atomically publish fixed report outputs."""

    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite report output: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale report staging exists: {staging}")
    verified = load_verified_evaluation(
        evaluation_dir,
        expected_completion_sha256=expected_completion_sha256,
    )
    tables = {
        "primary_estimands.csv": build_primary_estimands(
            verified.gate_result
        ),
        "stratum_effects.csv": build_stratum_effects(verified.gate_result),
        "horizon_arm_mae.csv": build_horizon_arm_mae(verified.detailed),
        "raw_unit_mae.csv": build_raw_unit_mae(verified.detailed),
        "boundary_diagnostics.csv": build_boundary_diagnostics(
            verified.detailed
        ),
        "action_sensitivity.csv": build_action_sensitivity(
            verified.detailed
        ),
        "fault_family_diagnostics.csv": build_fault_family_diagnostics(
            verified.detailed
        ),
        "model_resources.csv": build_model_resources(
            verified.provenance, verified.detailed
        ),
        "paired_dwell_transport.csv": build_paired_dwell_transport(
            verified.gate_result
        ),
    }
    if tuple(tables) != TABLE_FILES:
        raise AssertionError("fixed report table inventory changed")

    staging.mkdir(parents=True)
    try:
        for name, frame in tables.items():
            _write_csv(staging / name, frame)
            _write_ascii(
                staging / name.replace(".csv", ".md"),
                _markdown_table(frame),
            )
        _plot_primary(
            tables["primary_estimands.csv"],
            staging / "figure_primary_estimands.png",
        )
        _plot_horizon(
            tables["horizon_arm_mae.csv"],
            staging / "figure_horizon_arm_mae.png",
        )
        _plot_boundary(
            tables["boundary_diagnostics.csv"],
            staging / "figure_boundary_diagnostics.png",
        )
        _write_ascii(
            staging / RESULTS_NAME,
            _results_markdown(verified, tables),
        )
        inventory = _artifact_inventory(staging)
        expected_names = {
            *TABLE_FILES,
            *MARKDOWN_TABLE_FILES,
            *FIGURE_FILES,
            RESULTS_NAME,
        }
        if {row["path"] for row in inventory} != expected_names:
            raise AssertionError("fixed report artifact inventory changed")
        receipt = {
            "schema": REPORT_SCHEMA,
            "study_kind": "direct_h8_deterministic_transport_v3",
            "source_evaluation_completion_file_sha256": (
                verified.completion_file_sha256
            ),
            "source_evaluation_artifact_inventory_sha256": (
                verified.completion["artifact_inventory_sha256"]
            ),
            "source_prelock_registry_sha256": verified.completion[
                "prelock_registry_sha256"
            ],
            "source_readiness_sha256": verified.completion[
                "readiness_sha256"
            ],
            "source_gate_result_sha256": verified.completion[
                "gate_result_sha256"
            ],
            "primary_architecture_category": verified.gate_result[
                "primary_architecture_category"
            ],
            "primary_supervision_category": verified.gate_result[
                "primary_supervision_category"
            ],
            "artifact_inventory_excludes_manifest": inventory,
            "artifact_inventory_sha256": plan.canonical_sha256(inventory),
            "complete": True,
        }
        _write_ascii(
            staging / REPORT_RECEIPT_NAME,
            json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        )
        _seal_tree(staging)
        staging.rename(output_dir)
    except BaseException:
        raise
    return output_dir


def verify_report(
    report_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> dict:
    """Verify a generated report without consulting mutable source paths."""

    if not _valid_sha256(expected_manifest_sha256):
        raise ValueError("expected report manifest hash is invalid")
    if report_dir.is_symlink() or not report_dir.is_dir():
        raise ValueError("report path is not a plain directory")
    manifest_path = report_dir / REPORT_RECEIPT_NAME
    if plan.sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("report manifest differs from expected SHA-256")
    manifest = _strict_json(manifest_path)
    expected_fields = {
        "schema",
        "study_kind",
        "source_evaluation_completion_file_sha256",
        "source_evaluation_artifact_inventory_sha256",
        "source_prelock_registry_sha256",
        "source_readiness_sha256",
        "source_gate_result_sha256",
        "primary_architecture_category",
        "primary_supervision_category",
        "artifact_inventory_excludes_manifest",
        "artifact_inventory_sha256",
        "complete",
    }
    if (
        set(manifest) != expected_fields
        or manifest["schema"] != REPORT_SCHEMA
        or manifest["study_kind"] != "direct_h8_deterministic_transport_v3"
        or manifest["complete"] is not True
    ):
        raise ValueError("report manifest schema or identity changed")
    for field in (
        "source_evaluation_completion_file_sha256",
        "source_evaluation_artifact_inventory_sha256",
        "source_prelock_registry_sha256",
        "source_readiness_sha256",
        "source_gate_result_sha256",
        "artifact_inventory_sha256",
    ):
        if not _valid_sha256(manifest[field]):
            raise ValueError(f"report manifest has invalid {field}")
    inventory = _artifact_inventory(
        report_dir, exclude={REPORT_RECEIPT_NAME}
    )
    if (
        inventory != manifest["artifact_inventory_excludes_manifest"]
        or plan.canonical_sha256(inventory)
        != manifest["artifact_inventory_sha256"]
    ):
        raise ValueError("report artifacts differ from manifest")
    expected_names = {
        *TABLE_FILES,
        *MARKDOWN_TABLE_FILES,
        *FIGURE_FILES,
        RESULTS_NAME,
    }
    if {row["path"] for row in inventory} != expected_names:
        raise ValueError("report artifact set changed")
    return {
        "schema": REPORT_SCHEMA,
        "verified": True,
        "primary_architecture_category": manifest[
            "primary_architecture_category"
        ],
        "primary_supervision_category": manifest[
            "primary_supervision_category"
        ],
        "artifact_inventory_sha256": manifest[
            "artifact_inventory_sha256"
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--evaluation", type=Path, default=DEFAULT_EVALUATION
    )
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--expected-completion-sha256", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--report", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "generate":
        output = generate_report(
            args.evaluation,
            args.output,
            expected_completion_sha256=args.expected_completion_sha256,
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "report_manifest_sha256": plan.sha256_file(
                        output / REPORT_RECEIPT_NAME
                    ),
                },
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                verify_report(
                    args.report,
                    expected_manifest_sha256=args.expected_manifest_sha256,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
