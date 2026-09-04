"""Readiness, held-out evaluation, and analysis for the RC comparator."""

from __future__ import annotations

import hashlib
import os
import platform
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from building_fault_wm.deterministic_transport import (
    evaluate as v3_evaluate,
    gate as v3_gate,
    run_evaluation as v3_run,
)
from building_fault_wm.neural_benchmark.fault_data import (
    FAULT_CHANNELS,
    FaultScalers,
    FaultVariant,
    TrajectoryKey,
    build_fault_manifest,
    iter_role_variants,
)
from building_fault_wm.ridge_arx_sensitivity import study as strong_arx
from building_fault_wm.ridge_arx import (
    evaluate as arx_evaluate,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    write_json_once,
    write_once,
)

from . import publication_corpus
from .config import CASES, CONFIG, MODEL_SEEDS, POLICIES, SILENT_FAMILIES
from .model import RcModel, physical_diagnostics, restore_model
from .reproducibility import DEFAULT_OUTPUT_ROOT as DEFAULT_REPRODUCTION_ROOT
from .study import (
    DEFAULT_TRAINING_ROOT,
    ARM,
    filtered_states,
    open_loop_prediction,
    verify_training,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
NEURAL_EVALUATION = publication_corpus.NEURAL_EVALUATION_ROOT
DEFAULT_READINESS_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_readiness_v1"
DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_evaluation_v1"

NEURAL_ARMS = ("deterministic_wm", "ungated_h8")
ALL_ARMS = (*NEURAL_ARMS, ARM)
CORE_COLUMNS = arx_evaluate.CORE_COLUMNS
PAIR_COLUMNS = tuple(
    column for column in CORE_COLUMNS if column not in {"arm", "standardized_abs_error"}
)
PHYSICAL_COLUMNS = (
    "zone_capacity_j_per_k",
    "outdoor_resistance_k_per_w",
    "mass_capacity_j_per_k",
    "mass_to_zone_capacity_ratio",
    "zone_mass_resistance_k_per_w",
    "mass_outdoor_resistance_k_per_w",
    "effective_solar_aperture_m2",
    "constant_heat_flow_w",
)
DETAIL_COLUMNS = (
    *CORE_COLUMNS,
    "selected_topology",
    "selected_equipment_ridge_alpha",
    "selected_innovation_clip_sigma",
    "active_coefficients",
    *PHYSICAL_COLUMNS,
    "fault_channel_index",
    "severity_unit",
    "target_index",
    "target_raw",
    "prediction_raw",
    "prediction_standardized",
    "raw_abs_error",
    "raw_unit",
    "boundary_crossing",
    "action_transition_count",
)

READINESS_SCHEMA = "reviewer-rc-readiness-v1"
EVALUATION_SCHEMA = "reviewer-rc-evaluation-v1"
COMPLETION_SCHEMA = "reviewer-rc-evaluation-completion-v1"


def _source_files() -> dict[str, Path]:
    upstream = PROJECT_ROOT / "building_fault_wm/subspace_baseline"
    return {
        "PROTOCOL.md": HERE / "PROTOCOL.md",
        "__init__.py": HERE / "__init__.py",
        "config.py": HERE / "config.py",
        "model.py": HERE / "model.py",
        "study.py": HERE / "study.py",
        "evaluation.py": HERE / "evaluation.py",
        "publication_corpus.py": HERE / "publication_corpus.py",
        "reproducibility.py": HERE / "reproducibility.py",
        "audit.py": HERE / "audit.py",
        "cli.py": HERE / "cli.py",
        "__main__.py": HERE / "__main__.py",
        "test_model.py": HERE / "test_model.py",
        "test_evaluation.py": HERE / "test_evaluation.py",
        "publication_corpus_upstream.py": upstream / "publication_corpus.py",
        "requirements.txt": PROJECT_ROOT / "requirements.txt",
    }


def evaluation_source_identity() -> dict[str, str]:
    return {
        name: sha256_file(path) for name, path in sorted(_source_files().items())
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    return write_once(path, frame.to_csv(index=False).encode("ascii"))


def _transport_binding() -> dict[str, object]:
    return publication_corpus.package_binding()


def prepare_readiness(
    training_root: Path = DEFAULT_TRAINING_ROOT,
    reproduction_root: Path = DEFAULT_REPRODUCTION_ROOT,
    output_root: Path = DEFAULT_READINESS_ROOT,
) -> Path:
    """Bind selected models and all analysis code before held-out access."""

    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite RC readiness: {output_root}")
    training = verify_training(training_root)
    reproduction_path = reproduction_root / "reproduction_receipt.json"
    reproduction = strict_json(reproduction_path)
    if (
        reproduction.get("complete") is not True
        or reproduction.get("heldout_values_accessed") is not False
        or reproduction.get("byte_identical_numerical_selection") is not True
        or reproduction.get("production_training_completion_payload_sha256")
        != canonical_sha256(training)
    ):
        raise ValueError("RC selection reproduction audit changed")
    binding = _transport_binding()
    readiness = {
        "schema": READINESS_SCHEMA,
        "scope": "post_outcome_supervisory_review_physical_comparator",
        "complete": True,
        "new_rc_heldout_values_accessed": False,
        "rerun_in_same_namespace_permitted": False,
        "config": CONFIG.to_dict(),
        "evaluation_source_sha256": evaluation_source_identity(),
        "training_complete_file_sha256": sha256_file(
            training_root / "training_complete.json"
        ),
        "training_complete_payload_sha256": canonical_sha256(training),
        "model_file_sha256_by_case": training["model_file_sha256_by_case"],
        "selection_reproduction_receipt_file_sha256": sha256_file(
            reproduction_path
        ),
        "selection_reproduction_receipt_payload_sha256": canonical_sha256(
            reproduction
        ),
        "input_identity": binding,
        "output_contract": {
            "arm": ARM,
            "core_columns": list(CORE_COLUMNS),
            "detail_columns": list(DETAIL_COLUMNS),
            "expected_core_rows": 207_360,
            "neural_arms": list(NEURAL_ARMS),
            "effect": "1 - neural_mae / rc_mae",
            "positive_favors": "neural_arm",
            "bootstrap_draws": CONFIG.bootstrap_draws,
            "bootstrap_seed": CONFIG.bootstrap_seed,
        },
    }
    path = write_json_once(output_root / "readiness.json", readiness)
    write_once(
        output_root / "readiness.canonical.sha256",
        (canonical_sha256(readiness) + "\n").encode("ascii"),
    )
    return path


def verify_readiness(
    training_root: Path = DEFAULT_TRAINING_ROOT,
    reproduction_root: Path = DEFAULT_REPRODUCTION_ROOT,
    output_root: Path = DEFAULT_READINESS_ROOT,
) -> dict:
    training = verify_training(training_root)
    reproduction_path = reproduction_root / "reproduction_receipt.json"
    reproduction = strict_json(reproduction_path)
    readiness = strict_json(output_root / "readiness.json")
    if (
        readiness.get("schema") != READINESS_SCHEMA
        or readiness.get("complete") is not True
        or readiness.get("new_rc_heldout_values_accessed") is not False
        or readiness.get("evaluation_source_sha256") != evaluation_source_identity()
        or readiness.get("training_complete_file_sha256")
        != sha256_file(training_root / "training_complete.json")
        or readiness.get("training_complete_payload_sha256")
        != canonical_sha256(training)
        or readiness.get("model_file_sha256_by_case")
        != training["model_file_sha256_by_case"]
        or reproduction.get("complete") is not True
        or reproduction.get("byte_identical_numerical_selection") is not True
        or reproduction.get("production_training_completion_payload_sha256")
        != canonical_sha256(training)
        or readiness.get("selection_reproduction_receipt_file_sha256")
        != sha256_file(reproduction_path)
        or readiness.get("selection_reproduction_receipt_payload_sha256")
        != canonical_sha256(reproduction)
        or readiness.get("input_identity") != _transport_binding()
    ):
        raise ValueError("RC readiness identity changed")
    recorded = (output_root / "readiness.canonical.sha256").read_text(
        encoding="ascii"
    ).strip()
    if recorded != canonical_sha256(readiness):
        raise ValueError("RC readiness canonical digest changed")
    return readiness


def evaluate_variants(
    model: RcModel,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    trajectory_metadata: Mapping[
        TrajectoryKey, v3_evaluate.PolicyTrajectoryMetadata
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    silent = tuple(
        variant for variant in variants if variant.cell.family in SILENT_FAMILIES
    )
    v3_evaluate._validate_policy_metadata(silent, trajectory_metadata)
    mean = np.asarray(scalers.observation.mean)
    scale = np.asarray(scalers.observation.scale)
    physical = physical_diagnostics(model.thermal_coefficients)
    rows = []
    with threadpool_limits(limits=CONFIG.blas_threads, user_api="blas"):
        for variant in silent:
            states, actions, contexts_next = filtered_states(model, variant, scalers)
            cell = variant.cell
            key = cell.trajectory
            metadata = trajectory_metadata[key]
            channel = FAULT_CHANNELS.index(cell.fault_channel)
            for anchor in cell.anchors:
                for horizon in CONFIG.horizons:
                    prediction = open_loop_prediction(
                        model,
                        states[anchor],
                        actions,
                        contexts_next,
                        scalers,
                        anchor,
                        horizon,
                    )
                    target_raw = variant.clean_observations[anchor + horizon]
                    target_standardized = scalers.observation.transform(
                        target_raw[None]
                    )[0]
                    prediction_raw = prediction * scale + mean
                    boundary, transitions = v3_evaluate.action_block_crosses_transition(
                        variant.actions, anchor, horizon
                    )
                    base = {
                        "case": key.case,
                        "policy": metadata.policy,
                        "window_id": metadata.window_id,
                        "trajectory_day": key.day,
                        "scenario_seed": metadata.scenario_seed,
                        "trajectory_seed": key.trajectory_seed,
                        "arm": ARM,
                        "cell_id": cell.cell_id,
                        "fault_channel": cell.fault_channel,
                        "family": cell.family,
                        "sign": cell.sign,
                        "severity": cell.severity,
                        "onset": cell.onset,
                        "anchor": anchor,
                        "horizon": horizon,
                        "standardized_abs_error": float(
                            abs(prediction[channel] - target_standardized[channel])
                        ),
                        "selected_topology": model.topology,
                        "selected_equipment_ridge_alpha": model.equipment_ridge_alpha,
                        "selected_innovation_clip_sigma": model.innovation_clip_sigma,
                        "active_coefficients": model.active_coefficients,
                        **physical,
                        "fault_channel_index": channel,
                        "severity_unit": cell.severity_unit,
                        "target_index": anchor + horizon,
                        "target_raw": float(target_raw[channel]),
                        "prediction_raw": float(prediction_raw[channel]),
                        "prediction_standardized": float(prediction[channel]),
                        "raw_abs_error": float(
                            abs(prediction_raw[channel] - target_raw[channel])
                        ),
                        "raw_unit": arx_evaluate.RAW_UNITS[cell.fault_channel],
                        "boundary_crossing": bool(boundary),
                        "action_transition_count": transitions,
                    }
                    for model_seed in MODEL_SEEDS:
                        rows.append({**base, "model_seed": model_seed})
    detailed = pd.DataFrame(rows, columns=DETAIL_COLUMNS).sort_values(
        [
            "case",
            "window_id",
            "policy",
            "model_seed",
            "fault_channel",
            "family",
            "sign",
            "severity",
            "onset",
            "anchor",
            "horizon",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return detailed.loc[:, CORE_COLUMNS].copy(), detailed


def _validate_core(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != CORE_COLUMNS or len(frame) != 207_360:
        raise ValueError("RC held-out core shape changed")
    if set(frame["arm"]) != {ARM} or set(frame["family"]) != set(SILENT_FAMILIES):
        raise ValueError("RC held-out categories changed")
    if set(frame["case"]) != set(CASES) or set(frame["policy"]) != set(POLICIES):
        raise ValueError("RC held-out case or policy grid changed")
    if set(frame["model_seed"].astype(int)) != set(MODEL_SEEDS):
        raise ValueError("RC held-out seed pairing changed")
    identities = [column for column in CORE_COLUMNS if column != "standardized_abs_error"]
    if frame.duplicated(identities).any():
        raise ValueError("RC held-out core contains duplicate rows")
    if (
        not np.isfinite(frame["standardized_abs_error"]).all()
        or (frame["standardized_abs_error"] < 0).any()
    ):
        raise ValueError("RC held-out errors are invalid")


def _pair(neural: pd.DataFrame, rc: pd.DataFrame) -> pd.DataFrame:
    neural = v3_gate.validate_input(neural)
    neural = neural.loc[
        neural["arm"].isin(NEURAL_ARMS)
        & neural["family"].isin(SILENT_FAMILIES)
    ].copy()
    _validate_core(rc)
    neural_keys = set(
        map(tuple, neural.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None))
    )
    rc_keys = set(
        map(tuple, rc.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None))
    )
    if neural_keys != rc_keys or len(rc_keys) != len(rc):
        raise ValueError("RC and neural rows are not exact pairs")
    combined = pd.concat([neural, rc], ignore_index=True)
    if combined.duplicated([*PAIR_COLUMNS, "arm"]).any():
        raise ValueError("RC comparison contains duplicate arm rows")
    paired = combined.pivot(
        index=list(PAIR_COLUMNS), columns="arm", values="standardized_abs_error"
    ).reset_index()
    paired.columns.name = None
    if set(ALL_ARMS) - set(paired.columns):
        raise ValueError("RC comparison pivot is missing an arm")
    return paired.sort_values(list(PAIR_COLUMNS), kind="stable").reset_index(drop=True)


def _equal_weight_scores(
    paired: pd.DataFrame, horizon: int, *, retain_family: bool = False
) -> pd.DataFrame:
    selected = paired.loc[paired["horizon"] == horizon].copy()
    retained = ("family",) if retain_family else ()
    group = [
        "case",
        "policy",
        "window_id",
        "model_seed",
        *retained,
        *(
            value
            for value in ("family", "fault_channel", "sign", "severity")
            if value not in retained
        ),
    ]
    result = selected.groupby(group, as_index=False, dropna=False)[list(ALL_ARMS)].mean()
    for dimension in ("sign", "severity", "fault_channel", "family"):
        if dimension in retained:
            continue
        group = [value for value in group if value != dimension]
        result = result.groupby(group, as_index=False, dropna=False)[list(ALL_ARMS)].mean()
    expected = len(CASES) * len(POLICIES) * 12 * len(MODEL_SEEDS)
    if retain_family:
        expected *= len(SILENT_FAMILIES)
    if len(result) != expected:
        raise ValueError("RC equal-weight score grid is incomplete")
    return result.sort_values(group, kind="stable").reset_index(drop=True)


def _bootstrap_plan(scores: pd.DataFrame) -> dict[str, object]:
    rng = np.random.Generator(np.random.PCG64(CONFIG.bootstrap_seed))
    windows = {
        case: tuple(sorted(set(scores.loc[scores["case"] == case, "window_id"])))
        for case in CASES
    }
    return {
        "cases": rng.integers(0, len(CASES), size=(CONFIG.bootstrap_draws, len(CASES))),
        "seeds": rng.integers(
            0, len(MODEL_SEEDS), size=(CONFIG.bootstrap_draws, len(MODEL_SEEDS))
        ),
        "windows": {
            case: rng.integers(0, len(values), size=(CONFIG.bootstrap_draws, len(values)))
            for case, values in windows.items()
        },
        "window_values": windows,
    }


def _bootstrap_means(
    scores: pd.DataFrame, policy: str, plan: Mapping[str, object]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    selected = scores.loc[scores["policy"] == policy]
    case_points = []
    case_draws = []
    for case in CASES:
        windows = plan["window_values"][case]
        index = pd.MultiIndex.from_product(
            [MODEL_SEEDS, windows], names=["model_seed", "window_id"]
        )
        rows = selected.loc[selected["case"] == case].set_index(
            ["model_seed", "window_id"]
        )
        if rows.index.has_duplicates or set(rows.index) != set(index):
            raise ValueError("RC bootstrap seed/window matrix is incomplete")
        matrix = rows.loc[index, list(ALL_ARMS)].to_numpy(dtype=float).reshape(
            len(MODEL_SEEDS), len(windows), len(ALL_ARMS)
        )
        sampled = matrix[
            plan["seeds"][:, :, None],
            plan["windows"][case][:, None, :],
            :,
        ]
        case_points.append(matrix.mean(axis=(0, 1)))
        case_draws.append(sampled.mean(axis=(1, 2)))
    point = np.stack(case_points).mean(axis=0)
    by_case = np.stack(case_draws, axis=1)
    draws = by_case[
        np.arange(CONFIG.bootstrap_draws)[:, None], plan["cases"], :
    ].mean(axis=1)
    return (
        {arm: float(point[i]) for i, arm in enumerate(ALL_ARMS)},
        {arm: draws[:, i] for i, arm in enumerate(ALL_ARMS)},
    )


def _summary(point: float, draws: np.ndarray) -> dict[str, float]:
    lower95, lower90, upper90, upper95 = np.quantile(
        draws, (0.025, 0.05, 0.95, 0.975)
    )
    return {
        "point": float(point),
        "ci95_lower": float(lower95),
        "ci95_upper": float(upper95),
        "ci90_lower": float(lower90),
        "ci90_upper": float(upper90),
    }


def analyze(
    neural: pd.DataFrame, rc: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    paired = _pair(neural, rc)
    scores = _equal_weight_scores(paired, 8)
    family_scores = _equal_weight_scores(paired, 8, retain_family=True)
    plan = _bootstrap_plan(scores)
    result: dict[str, object] = {
        "schema": "reviewer-rc-comparison-result-v1",
        "scope": "post_outcome_supervisory_review_physical_comparator",
        "confirmatory_category_assigned": False,
        "estimand": "1 - equal_weight_MAE(neural_arm) / equal_weight_MAE(RC)",
        "positive_favors": "neural_arm",
        "policy_results": {},
    }
    for policy in POLICIES:
        points, draws = _bootstrap_means(scores, policy, plan)
        comparisons = {}
        for neural_arm in NEURAL_ARMS:
            effect_draws = 1.0 - draws[neural_arm] / draws[ARM]
            point = 1.0 - points[neural_arm] / points[ARM]
            comparisons[neural_arm] = {
                **_summary(point, effect_draws),
                "mean_standardized_mae_by_arm": {
                    neural_arm: points[neural_arm],
                    ARM: points[ARM],
                },
                "by_case": {
                    str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                    for key, rows in scores.loc[
                        scores["policy"] == policy
                    ].groupby("case", sort=True)
                },
                "by_family": {
                    str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                    for key, rows in family_scores.loc[
                        family_scores["policy"] == policy
                    ].groupby("family", sort=True)
                },
                "by_seed": {
                    str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                    for key, rows in scores.loc[
                        scores["policy"] == policy
                    ].groupby("model_seed", sort=True)
                },
            }
        result["policy_results"][policy] = comparisons
    descriptive_rows = []
    for horizon in CONFIG.horizons:
        values = _equal_weight_scores(paired, horizon)
        for (policy, case), rows in values.groupby(["policy", "case"], sort=True):
            for arm in ALL_ARMS:
                descriptive_rows.append(
                    {
                        "horizon": horizon,
                        "policy": policy,
                        "case": case,
                        "arm": arm,
                        "mean_standardized_mae": float(rows[arm].mean()),
                    }
                )
    return result, paired, pd.DataFrame(descriptive_rows)


def run_evaluation(
    training_root: Path = DEFAULT_TRAINING_ROOT,
    reproduction_root: Path = DEFAULT_REPRODUCTION_ROOT,
    readiness_root: Path = DEFAULT_READINESS_ROOT,
    output_root: Path = DEFAULT_EVALUATION_ROOT,
) -> Path:
    """Perform the single write-once response-unseen RC evaluation."""

    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to rerun RC evaluation: {output_root}")
    readiness = verify_readiness(training_root, reproduction_root, readiness_root)
    output_root.mkdir(parents=True, exist_ok=False)
    attempt_path = write_json_once(
        output_root / "evaluation_attempt.json",
        {
            "schema": "reviewer-rc-evaluation-attempt-v1",
            "readiness_file_sha256": sha256_file(readiness_root / "readiness.json"),
            "readiness_payload_sha256": canonical_sha256(readiness),
            "heldout_values_accessed_before_attempt": False,
            "rerun_in_same_namespace_permitted": False,
        },
    )
    started = time.perf_counter()
    current_binding = _transport_binding()
    collection = publication_corpus.load_publication_collection()
    if collection.package_binding != current_binding:
        raise ValueError("publication package changed during RC evaluation")
    fault_manifest = build_fault_manifest(
        collection.index, strong_arx.load_frozen_fault_spec()
    )
    core_frames = []
    detail_frames = []
    selected_rows = []
    for case in CASES:
        model_path = training_root / case / "model.json"
        model = restore_model(strict_json(model_path))
        variants = tuple(
            iter_role_variants(
                collection.index,
                fault_manifest,
                "locked_test",
                cases=(case,),
                allow_locked_test=True,
            )
        )
        metadata = {
            key: value
            for key, value in collection.trajectory_metadata.items()
            if key.case == case
        }
        core, detailed = evaluate_variants(
            model, variants, strong_arx.load_frozen_scaler(case), metadata
        )
        core_frames.append(core)
        detail_frames.append(detailed)
        selected_rows.append(
            {
                "case": case,
                "selected_topology": model.topology,
                "selected_equipment_ridge_alpha": model.equipment_ridge_alpha,
                "selected_innovation_clip_sigma": model.innovation_clip_sigma,
                "active_coefficients": model.active_coefficients,
                **physical_diagnostics(model.thermal_coefficients),
                "model_file_sha256": sha256_file(model_path),
            }
        )
    core = pd.concat(core_frames, ignore_index=True)
    detailed = pd.concat(detail_frames, ignore_index=True)
    _validate_core(core)
    neural_path = NEURAL_EVALUATION / v3_run.CORE_NAME
    if sha256_file(neural_path) != readiness["input_identity"]["neural_gate_core_file_sha256"]:
        raise ValueError("neural input changed after RC readiness")
    neural = pd.read_csv(neural_path, float_precision="round_trip")
    result, paired, descriptive = analyze(neural, core)
    core_path = _write_frame(output_root / "rc_core.csv", core)
    detail_path = _write_frame(output_root / "rc_detailed_diagnostics.csv", detailed)
    paired_path = _write_frame(output_root / "paired_neural_rc_rows.csv", paired)
    descriptive_path = _write_frame(
        output_root / "descriptive_by_horizon.csv", descriptive
    )
    selected_path = _write_frame(
        output_root / "selected_hyperparameters.csv",
        pd.DataFrame(selected_rows).sort_values("case", kind="stable"),
    )
    result_path = write_json_once(output_root / "comparison_result.json", result)
    provenance_path = write_json_once(
        output_root / "evaluation_provenance.json",
        {
            "schema": EVALUATION_SCHEMA,
            "scope": "post_outcome_supervisory_review_physical_comparator",
            "single_completed_heldout_pass": True,
            "readiness_file_sha256": sha256_file(readiness_root / "readiness.json"),
            "readiness_payload_sha256": canonical_sha256(readiness),
            "publication_package_binding_payload_sha256": canonical_sha256(
                current_binding
            ),
            "transport_manifest_file_sha256": collection.manifest_file_sha256,
            "fault_manifest_sha256": fault_manifest.sha256,
            "neural_core_file_sha256": sha256_file(neural_path),
            "rows": len(core),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "wall_seconds": time.perf_counter() - started,
                "device": "cpu",
            },
        },
    )
    outputs = (
        attempt_path,
        core_path,
        detail_path,
        paired_path,
        descriptive_path,
        selected_path,
        result_path,
        provenance_path,
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "scope": "post_outcome_supervisory_review_physical_comparator",
        "complete": True,
        "single_completed_heldout_pass": True,
        "row_count": len(core),
        "result_payload_sha256": canonical_sha256(result),
        "file_sha256_by_name": {path.name: sha256_file(path) for path in outputs},
    }
    return write_json_once(output_root / "evaluation_complete.json", completion)


def verify_evaluation(output_root: Path = DEFAULT_EVALUATION_ROOT) -> dict:
    completion = strict_json(output_root / "evaluation_complete.json")
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("complete") is not True
        or completion.get("single_completed_heldout_pass") is not True
        or completion.get("row_count") != 207_360
    ):
        raise ValueError("RC evaluation completion changed")
    for name, expected in completion["file_sha256_by_name"].items():
        if sha256_file(output_root / name) != expected:
            raise ValueError(f"RC evaluation file changed: {name}")
    result = strict_json(output_root / "comparison_result.json")
    if canonical_sha256(result) != completion["result_payload_sha256"]:
        raise ValueError("RC result payload changed")
    return completion
