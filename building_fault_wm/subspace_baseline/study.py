"""Fit, select, bind, and evaluate the reviewer-requested subspace baseline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
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
    fit_scalers,
    iter_role_variants,
    load_corpus_index,
    load_role_trajectories,
)
from building_fault_wm.ridge_arx_sensitivity import (
    study as strong_arx,
)
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
from .config import CASES, CONFIG, MODEL_SEEDS, POLICIES, SILENT_FAMILIES
from . import publication_corpus


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEVELOPMENT_MANIFEST = strong_arx.DEVELOPMENT_MANIFEST
NEURAL_EVALUATION = publication_corpus.NEURAL_EVALUATION_ROOT

DEFAULT_TRAINING_ROOT = PROJECT_ROOT / "artifacts/reviewer_subspace_training_v8"
DEFAULT_READINESS_ROOT = PROJECT_ROOT / "artifacts/reviewer_subspace_readiness_v3"
DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "artifacts/reviewer_subspace_evaluation_v3"

ARM = "reviewer_subspace_state_space"
NEURAL_ARMS = ("deterministic_wm", "ungated_h8")
ALL_ARMS = (*NEURAL_ARMS, ARM)
CORE_COLUMNS = arx_evaluate.CORE_COLUMNS
PAIR_COLUMNS = tuple(
    column for column in CORE_COLUMNS if column not in {"arm", "standardized_abs_error"}
)
DETAIL_COLUMNS = (
    *CORE_COLUMNS,
    "selected_block_rows",
    "selected_state_order",
    "selected_innovation_clip_sigma",
    "spectral_radius",
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

TRAINING_SCHEMA = "reviewer-subspace-selection-v1"
MODEL_SCHEMA = "reviewer-subspace-model-v1"
READINESS_SCHEMA = "reviewer-subspace-readiness-v1"
EVALUATION_SCHEMA = "reviewer-subspace-evaluation-v1"
COMPLETION_SCHEMA = "reviewer-subspace-evaluation-completion-v1"


def _block_hankel(matrix: np.ndarray, rows: int) -> np.ndarray:
    if matrix.ndim != 2 or rows <= 0 or len(matrix) < 2 * rows:
        raise ValueError("subspace Hankel input is invalid")
    result = np.empty((rows * matrix.shape[1], len(matrix) - rows + 1))
    for column in range(result.shape[1]):
        result[:, column] = matrix[column : column + rows].reshape(-1)
    return result


def _positive_semidefinite(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


@dataclass(frozen=True)
class SubspaceModel:
    block_rows: int
    state_order: int
    innovation_clip_sigma: float
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    covariance: np.ndarray
    singular_values: np.ndarray

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.a))))

    @property
    def active_coefficients(self) -> int:
        return int(self.a.size + self.b.size + self.c.size + self.d.size)

    def payload(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "block_rows": self.block_rows,
            "state_order": self.state_order,
            "innovation_clip_sigma": self.innovation_clip_sigma,
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "d": self.d.tolist(),
            "covariance": self.covariance.tolist(),
            "singular_values": self.singular_values.tolist(),
            "spectral_radius": self.spectral_radius,
            "active_coefficients": self.active_coefficients,
        }


def restore_model(payload: Mapping[str, object]) -> SubspaceModel:
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("subspace model schema changed")
    order = int(payload["state_order"])
    model = SubspaceModel(
        block_rows=int(payload["block_rows"]),
        state_order=order,
        innovation_clip_sigma=float(payload["innovation_clip_sigma"]),
        a=np.asarray(payload["a"], dtype=float),
        b=np.asarray(payload["b"], dtype=float),
        c=np.asarray(payload["c"], dtype=float),
        d=np.asarray(payload["d"], dtype=float),
        covariance=np.asarray(payload["covariance"], dtype=float),
        singular_values=np.asarray(payload["singular_values"], dtype=float),
    )
    shapes = (
        model.a.shape == (order, order),
        model.b.shape == (order, CONFIG.input_dim),
        model.c.shape == (CONFIG.observation_dim, order),
        model.d.shape == (CONFIG.observation_dim, CONFIG.input_dim),
        model.covariance.shape
        == (order + CONFIG.observation_dim, order + CONFIG.observation_dim),
    )
    arrays = (model.a, model.b, model.c, model.d, model.covariance)
    if not all(shapes) or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("subspace model arrays are invalid")
    if not np.isclose(model.spectral_radius, float(payload["spectral_radius"])):
        raise ValueError("subspace model spectral radius changed")
    return model


def _standardized_sequences(case: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], FaultScalers]:
    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    trajectories = load_role_trajectories(index, "fit", cases=(case,))
    scalers = fit_scalers(trajectories)
    frozen = strong_arx.load_frozen_scaler(case)
    if asdict(scalers) != asdict(frozen):
        raise ValueError("subspace FIT scalers differ from the frozen study scalers")
    sequences = []
    for trajectory in trajectories:
        outputs = scalers.observation.transform(trajectory.observations[1:])
        inputs = np.concatenate(
            [
                scalers.action.transform(trajectory.actions[:-1]),
                scalers.context.transform(trajectory.contexts[1:]),
            ],
            axis=1,
        )
        expected_steps = len(trajectory.observations) - 1
        if (
            expected_steps != 191
            or outputs.shape != (expected_steps, CONFIG.observation_dim)
            or inputs.shape != (expected_steps, CONFIG.input_dim)
        ):
            raise ValueError("subspace fitting sequence shape changed")
        sequences.append((inputs, outputs))
    if len(sequences) != 20:
        raise ValueError("subspace fitting trajectory count changed")
    return sequences, scalers


def _identify_models_unbounded(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]], block_rows: int
) -> dict[int, SubspaceModel]:
    """Identify all candidate orders from one multi-experiment decomposition."""

    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    widths = []
    for inputs, outputs in sequences:
        u_hankel = _block_hankel(inputs, block_rows)
        y_hankel = _block_hankel(outputs, block_rows)
        matrices.append((u_hankel, y_hankel))
        widths.append(u_hankel.shape[1])
    groups = (
        np.concatenate([u[:, block_rows:] for u, _ in matrices], axis=1),
        np.concatenate([u[:, :-block_rows] for u, _ in matrices], axis=1),
        np.concatenate([y[:, :-block_rows] for _, y in matrices], axis=1),
        np.concatenate([y[:, block_rows:] for _, y in matrices], axis=1),
    )
    instrumental = np.concatenate(groups, axis=0)
    _, upper = np.linalg.qr(instrumental.T, mode="reduced")
    lower = upper.T
    input_rows = CONFIG.input_dim * block_rows
    output_rows = CONFIG.observation_dim * block_rows
    r32 = lower[-output_rows:, input_rows:-output_rows]
    r22 = lower[input_rows:-output_rows, input_rows:-output_rows]
    combined = np.concatenate(
        [np.concatenate([u, y], axis=0) for u, y in matrices], axis=1
    )
    observability = r32 @ np.linalg.pinv(r22) @ combined
    _, singular_values, right = np.linalg.svd(observability, full_matrices=False)
    candidates = {}
    for order in CONFIG.state_orders:
        if order > output_rows:
            continue
        state_all = np.sqrt(singular_values[:order, None]) * right[:order]
        regressors = []
        responses = []
        offset = 0
        for width, (inputs, outputs) in zip(widths, sequences):
            state_full = state_all[:, offset : offset + width]
            offset += width
            state = state_full[:, :-1]
            aligned_outputs = outputs[block_rows:].T
            aligned_inputs = inputs[block_rows:].T
            regressors.append(
                np.concatenate([state[:, :-1], aligned_inputs[:, :-1]], axis=0)
            )
            responses.append(
                np.concatenate([state[:, 1:], aligned_outputs[:, :-1]], axis=0)
            )
        regressor = np.concatenate(regressors, axis=1)
        response = np.concatenate(responses, axis=1)
        gram = regressor @ regressor.T
        coefficients = response @ regressor.T @ np.linalg.pinv(gram)
        residual = response - coefficients @ regressor
        covariance = _positive_semidefinite(
            residual @ residual.T / residual.shape[1]
        )
        candidates[order] = SubspaceModel(
            block_rows=block_rows,
            state_order=order,
            innovation_clip_sigma=0.0,
            a=coefficients[:order, :order],
            b=coefficients[:order, order:],
            c=coefficients[order:, :order],
            d=coefficients[order:, order:],
            covariance=covariance,
            singular_values=singular_values[: min(len(singular_values), 32)],
        )
    return candidates


def identify_models(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]], block_rows: int
) -> dict[int, SubspaceModel]:
    """Identify candidates with a fixed BLAS thread count for repeatability."""

    with threadpool_limits(limits=CONFIG.blas_threads, user_api="blas"):
        return _identify_models_unbounded(sequences, block_rows)


def _variant_arrays(
    variant: FaultVariant, scalers: FaultScalers
) -> tuple[np.ndarray, np.ndarray]:
    outputs = scalers.observation.transform(variant.corrupted_observations)
    inputs = np.concatenate(
        [
            scalers.action.transform(variant.actions[:-1]),
            scalers.context.transform(variant.contexts[1:]),
        ],
        axis=1,
    )
    if not np.isfinite(outputs).all() or not np.isfinite(inputs).all():
        raise ValueError("subspace silent-fault input contains a non-finite value")
    return outputs, inputs


def _observer_schedule(
    model: SubspaceModel, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    order = model.state_order
    q = model.covariance[:order, :order]
    s = model.covariance[:order, order:]
    r = model.covariance[order:, order:]
    covariance = np.eye(order)
    gains = np.empty((steps, order, CONFIG.observation_dim))
    innovation_scales = np.empty((steps, CONFIG.observation_dim))
    for step in range(steps):
        innovation = _positive_semidefinite(
            r + model.c @ covariance @ model.c.T
        )
        innovation_scales[step] = np.sqrt(np.diag(innovation))
        cross = s + model.a @ covariance @ model.c.T
        gain = cross @ np.linalg.pinv(innovation)
        gains[step] = gain
        covariance = _positive_semidefinite(
            model.a @ covariance @ model.a.T + q - gain @ cross.T
        )
    return gains, innovation_scales


def filtered_prediction_states(
    model: SubspaceModel,
    variant: FaultVariant,
    scalers: FaultScalers,
    gains: np.ndarray | None = None,
    innovation_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the prior state available at every causal forecast anchor."""

    outputs, inputs = _variant_arrays(variant, scalers)
    if gains is None or innovation_scales is None:
        gains, innovation_scales = _observer_schedule(model, len(inputs))
    if gains.shape != (len(inputs), model.state_order, CONFIG.observation_dim):
        raise ValueError("subspace observer gain grid changed")
    if innovation_scales.shape != (len(inputs), CONFIG.observation_dim):
        raise ValueError("subspace innovation-scale grid changed")
    states = np.empty((len(outputs), model.state_order))
    state = np.zeros(model.state_order)
    states[0] = state
    for step in range(len(inputs)):
        residual = outputs[step + 1] - model.c @ state - model.d @ inputs[step]
        if model.innovation_clip_sigma > 0.0:
            residual = np.clip(
                residual,
                -model.innovation_clip_sigma * innovation_scales[step],
                model.innovation_clip_sigma * innovation_scales[step],
            )
        state = model.a @ state + model.b @ inputs[step] + gains[step] @ residual
        states[step + 1] = state
    if not np.isfinite(states).all():
        raise ValueError("subspace observer produced a non-finite state")
    return states, inputs


def open_loop_prediction(
    model: SubspaceModel,
    state: np.ndarray,
    inputs: np.ndarray,
    anchor: int,
    horizon: int,
) -> np.ndarray:
    if horizon not in CONFIG.horizons or anchor + horizon > len(inputs):
        raise ValueError("subspace forecast leaves its fixed horizon support")
    current = np.array(state, copy=True)
    prediction = None
    for step in range(anchor, anchor + horizon):
        prediction = model.c @ current + model.d @ inputs[step]
        current = model.a @ current + model.b @ inputs[step]
    if prediction is None or not np.isfinite(prediction).all():
        raise ValueError("subspace open-loop prediction is invalid")
    return prediction


def validation_score(
    model: SubspaceModel,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
) -> float:
    rows = []
    steps = len(variants[0].corrupted_observations) - 1
    gains, innovation_scales = _observer_schedule(model, steps)
    for variant in variants:
        if variant.cell.family not in SILENT_FAMILIES:
            continue
        states, inputs = filtered_prediction_states(
            model,
            variant,
            scalers,
            gains=gains,
            innovation_scales=innovation_scales,
        )
        channel = FAULT_CHANNELS.index(variant.cell.fault_channel)
        for anchor in variant.cell.anchors:
            prediction = open_loop_prediction(
                model, states[anchor], inputs, anchor, 8
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
        raise ValueError("subspace validation grid is incomplete")
    return float(cells["error"].mean())


def _source_files() -> dict[str, Path]:
    return {
        "PROTOCOL.md": HERE / "PROTOCOL.md",
        "config.py": HERE / "config.py",
        "study.py": HERE / "study.py",
        "audit.py": HERE / "audit.py",
        "cli.py": HERE / "cli.py",
        "__init__.py": HERE / "__init__.py",
        "__main__.py": HERE / "__main__.py",
        "test_study.py": HERE / "test_study.py",
        "publication_corpus.py": HERE / "publication_corpus.py",
        "requirements.txt": (
            PROJECT_ROOT / "requirements.txt"
        ),
    }


def source_identity() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(_source_files().items())}


def runtime_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("nfoursid", "numpy", "scipy", "threadpoolctl")
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    return write_once(path, frame.to_csv(index=False).encode("ascii"))


def run_development_selection(
    output_root: Path = DEFAULT_TRAINING_ROOT,
) -> Path:
    """Fit and select the complete development-only case grid once."""

    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite subspace training: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    source_lock = {
        "schema": "reviewer-subspace-development-source-lock-v1",
        "scope": "post_outcome_supervisory_review",
        "heldout_values_accessed_by_this_branch": False,
        "config": CONFIG.to_dict(),
        "runtime_versions": runtime_versions(),
        "source_sha256": source_identity(),
        "development_manifest_file_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "frozen_scaler_file_sha256_by_case": {
            case: sha256_file(strong_arx.SCALER_ROOT / f"{case}.json")
            for case in CASES
        },
    }
    lock_path = write_json_once(output_root / "training_source_lock.json", source_lock)
    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    fault_manifest = build_fault_manifest(index, strong_arx.load_frozen_fault_spec())
    selected_rows = []
    candidate_rows = []
    model_paths = []
    for case in CASES:
        sequences, scalers = _standardized_sequences(case)
        validation = tuple(
            variant
            for variant in iter_role_variants(
                index, fault_manifest, "validation", cases=(case,)
            )
            if variant.cell.family in SILENT_FAMILIES
        )
        case_candidates = []
        for block_rows in CONFIG.block_rows:
            base_models = identify_models(sequences, block_rows)
            for order, base in sorted(base_models.items()):
                for clip in CONFIG.innovation_clip_sigmas:
                    model = SubspaceModel(
                        block_rows=base.block_rows,
                        state_order=base.state_order,
                        innovation_clip_sigma=clip,
                        a=base.a,
                        b=base.b,
                        c=base.c,
                        d=base.d,
                        covariance=base.covariance,
                        singular_values=base.singular_values,
                    )
                    stable = model.spectral_radius <= CONFIG.maximum_spectral_radius
                    score = validation_score(model, validation, scalers) if stable else None
                    row = {
                        "case": case,
                        "block_rows": block_rows,
                        "state_order": order,
                        "innovation_clip_sigma": clip,
                        "spectral_radius": model.spectral_radius,
                        "active_coefficients": model.active_coefficients,
                        "stable": stable,
                        "validation_h8_mae": score,
                    }
                    candidate_rows.append(row)
                    if stable and score is not None and np.isfinite(score):
                        case_candidates.append((float(score), block_rows, order, clip, model))
        if not case_candidates:
            raise ValueError(f"no stable subspace candidate for {case}")
        score, block_rows, order, clip, selected = min(
            case_candidates,
            key=lambda value: (value[0], value[1], value[2], value[3]),
        )
        model_path = write_json_once(output_root / case / "model.json", selected.payload())
        model_paths.append(model_path)
        selected_rows.append(
            {
                "case": case,
                "selected_block_rows": block_rows,
                "selected_state_order": order,
                "selected_innovation_clip_sigma": clip,
                "selected_validation_h8_mae": score,
                "spectral_radius": selected.spectral_radius,
                "active_coefficients": selected.active_coefficients,
                "model_file_sha256": sha256_file(model_path),
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["case", "block_rows", "state_order", "innovation_clip_sigma"],
        kind="stable",
    )
    selected = pd.DataFrame(selected_rows).sort_values("case", kind="stable")
    candidates_path = _write_frame(output_root / "selection_scores.csv", candidates)
    selected_path = _write_frame(output_root / "selected_hyperparameters.csv", selected)
    completion = {
        "schema": TRAINING_SCHEMA,
        "complete": True,
        "case_count": len(CASES),
        "candidate_count": len(candidates),
        "training_source_lock_file_sha256": sha256_file(lock_path),
        "selection_scores_file_sha256": sha256_file(candidates_path),
        "selected_hyperparameters_file_sha256": sha256_file(selected_path),
        "model_file_sha256_by_case": {
            case: sha256_file(output_root / case / "model.json") for case in CASES
        },
    }
    return write_json_once(output_root / "training_complete.json", completion)


def verify_training(output_root: Path = DEFAULT_TRAINING_ROOT) -> dict:
    completion = strict_json(output_root / "training_complete.json")
    source_lock = strict_json(output_root / "training_source_lock.json")
    if (
        completion.get("schema") != TRAINING_SCHEMA
        or completion.get("complete") is not True
        or completion.get("case_count") != len(CASES)
        or completion.get("candidate_count")
        != len(CASES)
        * len(CONFIG.block_rows)
        * len(CONFIG.state_orders)
        * len(CONFIG.innovation_clip_sigmas)
    ):
        raise ValueError("subspace training completion changed")
    if (
        source_lock.get("schema")
        != "reviewer-subspace-development-source-lock-v1"
        or source_lock.get("heldout_values_accessed_by_this_branch") is not False
        or canonical_sha256(source_lock.get("config"))
        != canonical_sha256(CONFIG.to_dict())
        or source_lock.get("runtime_versions") != runtime_versions()
        or source_lock.get("source_sha256") != source_identity()
        or source_lock.get("development_manifest_file_sha256")
        != sha256_file(DEVELOPMENT_MANIFEST)
        or sha256_file(output_root / "training_source_lock.json")
        != completion.get("training_source_lock_file_sha256")
    ):
        raise ValueError("subspace training source lock changed")
    expected = completion["model_file_sha256_by_case"]
    for case in CASES:
        path = output_root / case / "model.json"
        if sha256_file(path) != expected[case]:
            raise ValueError("subspace selected model hash changed")
        model = restore_model(strict_json(path))
        if model.spectral_radius > CONFIG.maximum_spectral_radius:
            raise ValueError("selected subspace model is unstable")
    return completion


def _transport_binding() -> dict:
    return publication_corpus.package_binding()


def prepare_readiness(
    training_root: Path = DEFAULT_TRAINING_ROOT,
    output_root: Path = DEFAULT_READINESS_ROOT,
) -> Path:
    """Bind the selected models and sealed inputs before held-out access."""

    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite subspace readiness: {output_root}")
    training = verify_training(training_root)
    binding = _transport_binding()
    readiness = {
        "schema": READINESS_SCHEMA,
        "scope": "post_outcome_supervisory_review",
        "complete": True,
        "new_subspace_heldout_values_accessed": False,
        "rerun_in_same_namespace_permitted": False,
        "config": CONFIG.to_dict(),
        "source_sha256": source_identity(),
        "training_complete_file_sha256": sha256_file(
            training_root / "training_complete.json"
        ),
        "training_complete_payload_sha256": canonical_sha256(training),
        "model_file_sha256_by_case": training["model_file_sha256_by_case"],
        "input_identity": binding,
        "output_contract": {
            "arm": ARM,
            "core_columns": list(CORE_COLUMNS),
            "detail_columns": list(DETAIL_COLUMNS),
            "expected_core_rows": 207_360,
            "neural_arms": list(NEURAL_ARMS),
            "effect": "1 - neural_mae / subspace_mae",
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
    output_root: Path = DEFAULT_READINESS_ROOT,
) -> dict:
    training = verify_training(training_root)
    readiness = strict_json(output_root / "readiness.json")
    current_binding = _transport_binding()
    if (
        readiness.get("schema") != READINESS_SCHEMA
        or readiness.get("complete") is not True
        or readiness.get("new_subspace_heldout_values_accessed") is not False
        or readiness.get("source_sha256") != source_identity()
        or readiness.get("training_complete_file_sha256")
        != sha256_file(training_root / "training_complete.json")
        or readiness.get("training_complete_payload_sha256")
        != canonical_sha256(training)
        or readiness.get("model_file_sha256_by_case")
        != training["model_file_sha256_by_case"]
        or readiness.get("input_identity") != current_binding
    ):
        raise ValueError("subspace readiness identity changed")
    recorded = (output_root / "readiness.canonical.sha256").read_text(
        encoding="ascii"
    ).strip()
    if recorded != canonical_sha256(readiness):
        raise ValueError("subspace readiness canonical digest changed")
    return readiness


def evaluate_variants(
    model: SubspaceModel,
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
    steps = len(silent[0].corrupted_observations) - 1
    gains, innovation_scales = _observer_schedule(model, steps)
    rows = []
    for variant in silent:
        states, inputs = filtered_prediction_states(
            model,
            variant,
            scalers,
            gains=gains,
            innovation_scales=innovation_scales,
        )
        cell = variant.cell
        key = cell.trajectory
        metadata = trajectory_metadata[key]
        channel = FAULT_CHANNELS.index(cell.fault_channel)
        for anchor in cell.anchors:
            for horizon in CONFIG.horizons:
                prediction = open_loop_prediction(
                    model, states[anchor], inputs, anchor, horizon
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
                    "selected_block_rows": model.block_rows,
                    "selected_state_order": model.state_order,
                    "selected_innovation_clip_sigma": model.innovation_clip_sigma,
                    "spectral_radius": model.spectral_radius,
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
    core = detailed.loc[:, CORE_COLUMNS].copy()
    return core, detailed


def _validate_core(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != CORE_COLUMNS or len(frame) != 207_360:
        raise ValueError("subspace held-out core shape changed")
    if set(frame["arm"]) != {ARM} or set(frame["family"]) != set(SILENT_FAMILIES):
        raise ValueError("subspace held-out core categories changed")
    if set(frame["case"]) != set(CASES) or set(frame["policy"]) != set(POLICIES):
        raise ValueError("subspace held-out case/policy grid changed")
    if set(frame["model_seed"].astype(int)) != set(MODEL_SEEDS):
        raise ValueError("subspace held-out seed pairing changed")
    identities = [column for column in CORE_COLUMNS if column != "standardized_abs_error"]
    if frame.duplicated(identities).any():
        raise ValueError("subspace held-out core contains duplicate rows")
    if (
        not np.isfinite(frame["standardized_abs_error"]).all()
        or (frame["standardized_abs_error"] < 0).any()
    ):
        raise ValueError("subspace held-out errors are invalid")


def _pair(neural: pd.DataFrame, subspace: pd.DataFrame) -> pd.DataFrame:
    neural = v3_gate.validate_input(neural)
    neural = neural.loc[
        neural["arm"].isin(NEURAL_ARMS)
        & neural["family"].isin(SILENT_FAMILIES)
    ].copy()
    _validate_core(subspace)
    neural_keys = set(
        map(tuple, neural.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None))
    )
    subspace_keys = set(
        map(tuple, subspace.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None))
    )
    if neural_keys != subspace_keys or len(neural_keys) != len(subspace):
        raise ValueError("subspace and neural rows are not exact pairs")
    combined = pd.concat([neural, subspace], ignore_index=True)
    if combined.duplicated([*PAIR_COLUMNS, "arm"]).any():
        raise ValueError("subspace comparison contains duplicate arm rows")
    paired = combined.pivot(
        index=list(PAIR_COLUMNS), columns="arm", values="standardized_abs_error"
    ).reset_index()
    paired.columns.name = None
    if set(ALL_ARMS) - set(paired.columns):
        raise ValueError("subspace comparison pivot is missing an arm")
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
        *(value for value in ("family", "fault_channel", "sign", "severity") if value not in retained),
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
        raise ValueError("subspace equal-weight score grid is incomplete")
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
            case: rng.integers(
                0, len(values), size=(CONFIG.bootstrap_draws, len(values))
            )
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
    for case_index, case in enumerate(CASES):
        windows = plan["window_values"][case]
        index = pd.MultiIndex.from_product(
            [MODEL_SEEDS, windows], names=["model_seed", "window_id"]
        )
        rows = selected.loc[selected["case"] == case].set_index(
            ["model_seed", "window_id"]
        )
        if rows.index.has_duplicates or set(rows.index) != set(index):
            raise ValueError("subspace bootstrap seed/window matrix is incomplete")
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


def analyze(neural: pd.DataFrame, subspace: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    paired = _pair(neural, subspace)
    scores = _equal_weight_scores(paired, 8)
    family_scores = _equal_weight_scores(paired, 8, retain_family=True)
    plan = _bootstrap_plan(scores)
    result: dict[str, object] = {
        "schema": "reviewer-subspace-comparison-result-v1",
        "scope": "post_outcome_supervisory_review",
        "confirmatory_category_assigned": False,
        "estimand": "1 - equal_weight_MAE(neural_arm) / equal_weight_MAE(subspace)",
        "positive_favors": "neural_arm",
        "policy_results": {},
    }
    for policy in POLICIES:
        points, draws = _bootstrap_means(scores, policy, plan)
        comparisons = {}
        for neural_arm in NEURAL_ARMS:
            effect_draws = 1.0 - draws[neural_arm] / draws[ARM]
            point = 1.0 - points[neural_arm] / points[ARM]
            by_case = {
                str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                for key, rows in scores.loc[scores["policy"] == policy].groupby(
                    "case", sort=True
                )
            }
            by_family = {
                str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                for key, rows in family_scores.loc[
                    family_scores["policy"] == policy
                ].groupby("family", sort=True)
            }
            by_seed = {
                str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                for key, rows in scores.loc[scores["policy"] == policy].groupby(
                    "model_seed", sort=True
                )
            }
            comparisons[neural_arm] = {
                **_summary(point, effect_draws),
                "mean_standardized_mae_by_arm": {
                    neural_arm: points[neural_arm],
                    ARM: points[ARM],
                },
                "by_case": by_case,
                "by_family": by_family,
                "by_seed": by_seed,
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
    readiness_root: Path = DEFAULT_READINESS_ROOT,
    output_root: Path = DEFAULT_EVALUATION_ROOT,
) -> Path:
    """Perform the single write-once response-unseen comparator evaluation."""

    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to rerun subspace evaluation: {output_root}")
    readiness = verify_readiness(training_root, readiness_root)
    output_root.mkdir(parents=True, exist_ok=False)
    attempt_path = write_json_once(
        output_root / "evaluation_attempt.json",
        {
            "schema": "reviewer-subspace-evaluation-attempt-v1",
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
        raise ValueError("publication package changed during evaluation")
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
                "selected_block_rows": model.block_rows,
                "selected_state_order": model.state_order,
                "selected_innovation_clip_sigma": model.innovation_clip_sigma,
                "spectral_radius": model.spectral_radius,
                "active_coefficients": model.active_coefficients,
                "model_file_sha256": sha256_file(model_path),
            }
        )
    core = pd.concat(core_frames, ignore_index=True)
    detailed = pd.concat(detail_frames, ignore_index=True)
    _validate_core(core)
    if (
        sha256_file(NEURAL_EVALUATION / v3_run.CORE_NAME)
        != readiness["input_identity"]["neural_gate_core_file_sha256"]
    ):
        raise ValueError("neural input changed after subspace readiness")
    neural = pd.read_csv(
        NEURAL_EVALUATION / v3_run.CORE_NAME, float_precision="round_trip"
    )
    result, paired, descriptive = analyze(neural, core)
    core_path = _write_frame(output_root / "subspace_core.csv", core)
    detail_path = _write_frame(
        output_root / "subspace_detailed_diagnostics.csv", detailed
    )
    paired_path = _write_frame(
        output_root / "paired_neural_subspace_rows.csv", paired
    )
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
            "scope": "post_outcome_supervisory_review",
            "single_completed_heldout_pass": True,
            "readiness_file_sha256": sha256_file(readiness_root / "readiness.json"),
            "readiness_payload_sha256": canonical_sha256(readiness),
            "publication_package_binding_payload_sha256": canonical_sha256(
                current_binding
            ),
            "transport_manifest_file_sha256": collection.manifest_file_sha256,
            "fault_manifest_sha256": fault_manifest.sha256,
            "neural_core_file_sha256": sha256_file(
                NEURAL_EVALUATION / v3_run.CORE_NAME
            ),
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
        "scope": "post_outcome_supervisory_review",
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
        raise ValueError("subspace evaluation completion changed")
    for name, expected in completion["file_sha256_by_name"].items():
        if sha256_file(output_root / name) != expected:
            raise ValueError(f"subspace evaluation file changed: {name}")
    result = strict_json(output_root / "comparison_result.json")
    if canonical_sha256(result) != completion["result_payload_sha256"]:
        raise ValueError("subspace result payload changed")
    return completion
