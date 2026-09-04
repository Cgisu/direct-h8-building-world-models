"""Fit, select, bind, and evaluate the reviewer-requested RC comparator."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from building_fault_wm.deterministic_transport import evaluate as v3_evaluate
from building_fault_wm.neural_benchmark.fault_data import (
    FAULT_CHANNELS,
    FaultScalers,
    FaultVariant,
    build_fault_manifest,
    fit_scalers,
    iter_role_variants,
    load_corpus_index,
    load_role_trajectories,
)
from building_fault_wm.ridge_arx_sensitivity import study as strong_arx
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    write_json_once,
)
from .config import CASES, CONFIG, SILENT_FAMILIES
from .model import (
    OBSERVATION_DIM,
    STATE_DIM,
    RcModel,
    estimate_process_covariance,
    fit_equipment_map,
    fit_thermal_model,
    observation_from_state,
    parameter_boundary_diagnostics,
    physical_diagnostics,
    restore_model,
    state_from_observation,
    transition,
    transition_jacobian,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEVELOPMENT_MANIFEST = strong_arx.DEVELOPMENT_MANIFEST
DEFAULT_TRAINING_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_training_v1"

TRAINING_SCHEMA = "reviewer-rc-selection-v1"
ARM = "reviewer_rc_grey_box"

_WORKER_VARIANTS: Sequence[FaultVariant] | None = None
_WORKER_SCALERS: FaultScalers | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _source_files() -> dict[str, Path]:
    return {
        "PROTOCOL.md": HERE / "PROTOCOL.md",
        "__init__.py": HERE / "__init__.py",
        "config.py": HERE / "config.py",
        "model.py": HERE / "model.py",
        "study.py": HERE / "study.py",
        "test_model.py": HERE / "test_model.py",
        "fault_data.py": PROJECT_ROOT
        / "building_fault_wm/neural_benchmark/fault_data.py",
        "benchmark_protocol.py": PROJECT_ROOT
        / "building_fault_wm/neural_benchmark/protocol.py",
        "requirements.txt": PROJECT_ROOT / "requirements.txt",
    }


def source_identity() -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in _source_files().items()}


def runtime_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
        "device": "cpu",
        "multiprocessing_start_method": "fork",
        "validation_processes": CONFIG.validation_processes,
        "blas_threads_per_process": CONFIG.blas_threads,
    }


def _development_data(
    case: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], FaultScalers]:
    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    trajectories = load_role_trajectories(index, "fit", cases=(case,))
    scalers = fit_scalers(trajectories)
    frozen = strong_arx.load_frozen_scaler(case)
    if asdict(scalers) != asdict(frozen):
        raise ValueError("RC FIT scalers differ from the frozen study scalers")
    if len(trajectories) != 20:
        raise ValueError("RC fitting trajectory count changed")
    observations = [np.asarray(item.observations, dtype=float) for item in trajectories]
    actions = [np.asarray(item.actions, dtype=float) for item in trajectories]
    contexts = [np.asarray(item.contexts, dtype=float) for item in trajectories]
    if any(value.shape != (192, 4) for value in observations):
        raise ValueError("RC fitting observation shape changed")
    return observations, actions, contexts, scalers


def fit_case_candidates(case: str) -> list[RcModel]:
    observations, actions, contexts, scalers = _development_data(case)
    candidates = []
    with threadpool_limits(limits=CONFIG.blas_threads, user_api="blas"):
        for topology in CONFIG.topologies:
            thermal, _, thermal_diagnostics = fit_thermal_model(
                case, topology, observations, contexts
            )
            physical = physical_diagnostics(thermal)
            if any(
                value is not None and not np.isfinite(float(value))
                for value in physical.values()
            ):
                raise ValueError("RC physical parameter conversion failed")
            for alpha in CONFIG.equipment_ridge_alphas:
                coefficients, lower, upper, equipment_diagnostics = fit_equipment_map(
                    case, alpha, observations, actions, contexts, scalers
                )
                provisional = RcModel(
                    case=case,
                    topology=topology,
                    equipment_ridge_alpha=alpha,
                    innovation_clip_sigma=0.0,
                    thermal_coefficients=thermal,
                    equipment_coefficients=coefficients,
                    equipment_lower_standardized=lower,
                    equipment_upper_standardized=upper,
                    process_covariance=np.eye(STATE_DIM) * 0.01,
                    measurement_covariance=np.eye(OBSERVATION_DIM)
                    * CONFIG.measurement_variance,
                    fit_diagnostics={**thermal_diagnostics, **equipment_diagnostics},
                )
                process = estimate_process_covariance(
                    provisional, observations, actions, contexts, scalers
                )
                for clip in CONFIG.innovation_clip_sigmas:
                    candidates.append(
                        replace(
                            provisional,
                            innovation_clip_sigma=clip,
                            process_covariance=process,
                        )
                    )
    expected = (
        len(CONFIG.topologies)
        * len(CONFIG.equipment_ridge_alphas)
        * len(CONFIG.innovation_clip_sigmas)
    )
    if len(candidates) != expected:
        raise ValueError("RC case candidate grid is incomplete")
    return candidates


def _variant_arrays(
    variant: FaultVariant, scalers: FaultScalers
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = scalers.observation.transform(variant.corrupted_observations)
    actions = scalers.action.transform(variant.actions[:-1])
    contexts_next = scalers.context.transform(variant.contexts[1:])
    if (
        outputs.shape != (192, 4)
        or actions.shape != (191, 1)
        or contexts_next.shape != (191, 5)
        or not np.isfinite(outputs).all()
    ):
        raise ValueError("RC silent-fault input shape changed")
    return outputs, actions, contexts_next


def filtered_states(
    model: RcModel, variant: FaultVariant, scalers: FaultScalers
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs, actions, contexts_next = _variant_arrays(variant, scalers)
    measurement = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    identity = np.eye(STATE_DIM)
    state = state_from_observation(outputs[0])
    covariance = np.eye(STATE_DIM) * 0.1
    states = np.empty((len(outputs), STATE_DIM))
    states[0] = state
    for step in range(len(actions)):
        jacobian = transition_jacobian(
            model, state, actions[step], contexts_next[step], scalers
        )
        predicted = transition(
            model, state, actions[step], contexts_next[step], scalers
        )
        predicted_covariance = (
            jacobian @ covariance @ jacobian.T + model.process_covariance
        )
        innovation_covariance = (
            measurement @ predicted_covariance @ measurement.T
            + model.measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement @ predicted_covariance,
        ).T
        residual = outputs[step + 1] - measurement @ predicted
        if model.innovation_clip_sigma > 0.0:
            scales = np.sqrt(np.maximum(np.diag(innovation_covariance), 1e-12))
            residual = np.clip(
                residual,
                -model.innovation_clip_sigma * scales,
                model.innovation_clip_sigma * scales,
            )
        state = predicted + gain @ residual
        correction = identity - gain @ measurement
        covariance = (
            correction @ predicted_covariance @ correction.T
            + gain @ model.measurement_covariance @ gain.T
        )
        covariance = (covariance + covariance.T) / 2.0
        states[step + 1] = state
    if not np.isfinite(states).all():
        raise ValueError("RC observer produced a non-finite state")
    return states, actions, contexts_next


def open_loop_prediction(
    model: RcModel,
    state: np.ndarray,
    actions: np.ndarray,
    contexts_next: np.ndarray,
    scalers: FaultScalers,
    anchor: int,
    horizon: int,
) -> np.ndarray:
    if horizon not in CONFIG.horizons or anchor + horizon > len(actions):
        raise ValueError("RC forecast leaves its fixed support")
    current = np.array(state, copy=True)
    for step in range(anchor, anchor + horizon):
        current = transition(model, current, actions[step], contexts_next[step], scalers)
    prediction = observation_from_state(current)
    if not np.isfinite(prediction).all():
        raise ValueError("RC open-loop prediction is invalid")
    return prediction


def validation_score(
    model: RcModel,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
) -> float:
    rows = []
    with threadpool_limits(limits=CONFIG.blas_threads, user_api="blas"):
        for variant in variants:
            if variant.cell.family not in SILENT_FAMILIES:
                continue
            states, actions, contexts_next = filtered_states(model, variant, scalers)
            channel = FAULT_CHANNELS.index(variant.cell.fault_channel)
            for anchor in variant.cell.anchors:
                prediction = open_loop_prediction(
                    model,
                    states[anchor],
                    actions,
                    contexts_next,
                    scalers,
                    anchor,
                    8,
                )
                target = scalers.observation.transform(
                    variant.clean_observations[anchor + 8 : anchor + 9]
                )[0]
                rows.append(
                    {
                        "family": variant.cell.family,
                        "fault_channel": variant.cell.fault_channel,
                        "error": abs(prediction[channel] - target[channel]),
                    }
                )
    cells = pd.DataFrame(rows).groupby(
        ["family", "fault_channel"], as_index=False, dropna=False
    )["error"].mean()
    if len(cells) != len(SILENT_FAMILIES) * len(FAULT_CHANNELS):
        raise ValueError("RC validation grid is incomplete")
    return float(cells["error"].mean())


def _initialize_validation_worker(
    variants: Sequence[FaultVariant], scalers: FaultScalers
) -> None:
    global _WORKER_VARIANTS, _WORKER_SCALERS
    _WORKER_VARIANTS = variants
    _WORKER_SCALERS = scalers


def _validation_worker(model: RcModel) -> float:
    if _WORKER_VARIANTS is None or _WORKER_SCALERS is None:
        raise RuntimeError("RC validation worker was not initialized")
    return validation_score(model, _WORKER_VARIANTS, _WORKER_SCALERS)


def score_case_candidates(
    candidates: Sequence[RcModel],
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
) -> list[float]:
    expected = (
        len(CONFIG.topologies)
        * len(CONFIG.equipment_ridge_alphas)
        * len(CONFIG.innovation_clip_sigmas)
    )
    if len(candidates) != expected:
        raise ValueError("RC validation candidate grid is incomplete")
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=CONFIG.validation_processes,
        initializer=_initialize_validation_worker,
        initargs=(variants, scalers),
    ) as pool:
        scores = pool.map(_validation_worker, candidates, chunksize=1)
    if len(scores) != len(candidates) or not np.isfinite(scores).all():
        raise ValueError("RC parallel validation returned an invalid score grid")
    return [float(value) for value in scores]


def run_development_selection(output_root: Path = DEFAULT_TRAINING_ROOT) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to replace RC selection: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    source_lock = {
        "schema": "reviewer-rc-development-source-lock-v1",
        "scope": "post_outcome_supervisory_review_physical_comparator",
        "heldout_values_accessed": False,
        "config": CONFIG.to_dict(),
        "source_sha256": source_identity(),
        "development_manifest_file_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "frozen_scaler_file_sha256_by_case": {
            case: sha256_file(strong_arx.SCALER_ROOT / f"{case}.json")
            for case in CASES
        },
        "runtime": runtime_identity(),
    }
    source_lock_path = write_json_once(
        output_root / "training_source_lock.json", source_lock
    )
    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    fault_manifest = build_fault_manifest(index, strong_arx.load_frozen_fault_spec())
    score_rows = []
    selected_rows = []
    model_hashes = {}
    case_receipt_hashes = {}
    for case in CASES:
        case_started = time.perf_counter()
        candidates = fit_case_candidates(case)
        variants = tuple(
            iter_role_variants(index, fault_manifest, "validation", cases=(case,))
        )
        scaler = strong_arx.load_frozen_scaler(case)
        case_scores = []
        scores = score_case_candidates(candidates, variants, scaler)
        for model, score in zip(candidates, scores):
            row = {
                "case": case,
                "topology": model.topology,
                "equipment_ridge_alpha": model.equipment_ridge_alpha,
                "innovation_clip_sigma": model.innovation_clip_sigma,
                "validation_h8_mae": score,
                "active_coefficients": model.active_coefficients,
            }
            score_rows.append(row)
            case_scores.append((score, model))
        topology_rank = {"1r1c": 0, "2r2c": 1}
        clip_rank = {0.0: 0, 3.0: 1, 5.0: 2}
        case_scores.sort(
            key=lambda item: (
                item[0],
                topology_rank[item[1].topology],
                -item[1].equipment_ridge_alpha,
                clip_rank[item[1].innovation_clip_sigma],
            )
        )
        selected_score, selected = case_scores[0]
        case_root = output_root / case
        case_root.mkdir()
        model_path = write_json_once(case_root / "model.json", selected.payload())
        model_hashes[case] = sha256_file(model_path)
        physical = physical_diagnostics(selected.thermal_coefficients)
        boundaries = parameter_boundary_diagnostics(selected)
        selected_rows.append(
            {
                "case": case,
                "selected_topology": selected.topology,
                "selected_equipment_ridge_alpha": selected.equipment_ridge_alpha,
                "selected_innovation_clip_sigma": selected.innovation_clip_sigma,
                "selected_validation_h8_mae": selected_score,
                "active_coefficients": selected.active_coefficients,
                **physical,
                **boundaries,
            }
        )
        receipt_path = write_json_once(
            case_root / "selection_receipt.json",
            {
                "schema": "reviewer-rc-case-selection-receipt-v1",
                "case": case,
                "complete": True,
                "heldout_values_accessed": False,
                "candidate_count": len(candidates),
                "selected_validation_h8_mae": selected_score,
                "selected_model_file_sha256": model_hashes[case],
                "wall_seconds": time.perf_counter() - case_started,
            },
        )
        case_receipt_hashes[case] = sha256_file(receipt_path)
        print(
            f"RC development case complete: {case} "
            f"({selected.topology}, alpha={selected.equipment_ridge_alpha:g}, "
            f"clip={selected.innovation_clip_sigma:g}, "
            f"validation H8 MAE={selected_score:.9f})",
            flush=True,
        )
    scores = pd.DataFrame(score_rows).sort_values(
        ["case", "validation_h8_mae", "topology", "equipment_ridge_alpha"],
        kind="stable",
    )
    selected = pd.DataFrame(selected_rows).sort_values("case", kind="stable")
    score_path = _write_frame(output_root / "selection_scores.csv", scores)
    selected_path = _write_frame(output_root / "selected_hyperparameters.csv", selected)
    completion = {
        "schema": TRAINING_SCHEMA,
        "complete": True,
        "heldout_values_accessed": False,
        "scope": "post_outcome_supervisory_review_physical_comparator",
        "config": CONFIG.to_dict(),
        "source_sha256": source_identity(),
        "training_source_lock_file_sha256": sha256_file(source_lock_path),
        "training_source_lock_payload_sha256": canonical_sha256(source_lock),
        "development_manifest_file_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "fault_manifest_sha256": fault_manifest.sha256,
        "model_file_sha256_by_case": model_hashes,
        "case_selection_receipt_sha256_by_case": case_receipt_hashes,
        "selection_scores_file_sha256": sha256_file(score_path),
        "selected_hyperparameters_file_sha256": sha256_file(selected_path),
        "candidate_count": len(scores),
        "selected_count": len(selected),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": "cpu",
            "blas_threads": CONFIG.blas_threads,
        },
    }
    return write_json_once(output_root / "training_complete.json", completion)


def verify_training(output_root: Path = DEFAULT_TRAINING_ROOT) -> dict:
    completion = strict_json(output_root / "training_complete.json")
    source_lock = strict_json(output_root / "training_source_lock.json")
    expected_candidates = (
        len(CASES)
        * len(CONFIG.topologies)
        * len(CONFIG.equipment_ridge_alphas)
        * len(CONFIG.innovation_clip_sigmas)
    )
    if (
        completion.get("schema") != TRAINING_SCHEMA
        or completion.get("complete") is not True
        or completion.get("heldout_values_accessed") is not False
        or canonical_sha256(completion.get("config"))
        != canonical_sha256(CONFIG.to_dict())
        or completion.get("source_sha256") != source_identity()
        or completion.get("candidate_count") != expected_candidates
        or completion.get("selected_count") != 3
    ):
        raise ValueError("RC selection identity changed")
    if (
        source_lock.get("schema") != "reviewer-rc-development-source-lock-v1"
        or source_lock.get("heldout_values_accessed") is not False
        or canonical_sha256(source_lock.get("config"))
        != canonical_sha256(CONFIG.to_dict())
        or source_lock.get("source_sha256") != source_identity()
        or source_lock.get("development_manifest_file_sha256")
        != sha256_file(DEVELOPMENT_MANIFEST)
        or source_lock.get("frozen_scaler_file_sha256_by_case")
        != {
            case: sha256_file(strong_arx.SCALER_ROOT / f"{case}.json")
            for case in CASES
        }
        or source_lock.get("runtime") != runtime_identity()
        or sha256_file(output_root / "training_source_lock.json")
        != completion.get("training_source_lock_file_sha256")
        or canonical_sha256(source_lock)
        != completion.get("training_source_lock_payload_sha256")
    ):
        raise ValueError("RC development source lock changed")
    for case in CASES:
        path = output_root / case / "model.json"
        if sha256_file(path) != completion["model_file_sha256_by_case"][case]:
            raise ValueError("RC selected model hash changed")
        restore_model(strict_json(path))
        receipt_path = output_root / case / "selection_receipt.json"
        if (
            sha256_file(receipt_path)
            != completion["case_selection_receipt_sha256_by_case"][case]
        ):
            raise ValueError("RC case selection receipt hash changed")
        receipt = strict_json(receipt_path)
        if (
            receipt.get("complete") is not True
            or receipt.get("heldout_values_accessed") is not False
            or receipt.get("candidate_count")
            != len(CONFIG.topologies)
            * len(CONFIG.equipment_ridge_alphas)
            * len(CONFIG.innovation_clip_sigmas)
        ):
            raise ValueError("RC case selection receipt changed")
    if (
        sha256_file(output_root / "selection_scores.csv")
        != completion["selection_scores_file_sha256"]
        or sha256_file(output_root / "selected_hyperparameters.csv")
        != completion["selected_hyperparameters_file_sha256"]
    ):
        raise ValueError("RC selection table hash changed")
    scores = pd.read_csv(output_root / "selection_scores.csv")
    selected = pd.read_csv(output_root / "selected_hyperparameters.csv")
    if len(scores) != expected_candidates or len(selected) != len(CASES):
        raise ValueError("RC selection table shape changed")
    return completion
