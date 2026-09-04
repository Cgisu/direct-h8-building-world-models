"""Train, lock, and evaluate a strengthened schedule-matched Ridge-ARX."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import sklearn
from sklearn.linear_model import Ridge

from building_fault_wm.deterministic_transport import (
    corpus as v3_corpus,
    evaluate as v3_evaluate,
    gate as v3_gate,
    run_evaluation as v3_run,
)
from building_fault_wm.transport_collection import (
    external_freeze as transport_external_freeze,
    runner as transport_runner,
)
from building_fault_wm.neural_benchmark.fault_data import (
    FAULT_CHANNELS,
    FaultScalers,
    FaultSpec,
    FaultVariant,
    ScaleStats,
    SequenceReference,
    TrajectoryKey,
    build_fault_manifest,
    iter_role_variants,
    load_corpus_index,
)
from building_fault_wm.neural_benchmark.study_train import (
    ScheduledBatch,
    canonical_payload_sha256,
    prepare_case_training_data,
)
from building_fault_wm.ridge_arx import (
    evaluate as original_evaluate,
)
from building_fault_wm.ridge_arx import (
    train as original_train,
)
from building_fault_wm.ridge_arx.comparison_analysis import (
    analysis as original_analysis,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)
from building_fault_wm.ridge_arx.lock import (
    bind_completed_transport_metadata,
)

from .config import CASES, CONFIG, MODEL_SEEDS, POLICIES, SILENT_FAMILIES


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PARENT_ROOT = PROJECT_ROOT / "artifacts/direct_h8_publication_v2"
DEVELOPMENT_MANIFEST = (
    PARENT_ROOT
    / "experiment/prelock_bundle/corpus/manifests/development_all_corpus_manifest.json"
)
SCHEDULE_ROOT = PARENT_ROOT / "experiment/prelock_bundle/frozen/schedules"
SCALER_ROOT = PARENT_ROOT / "experiment/prelock_bundle/frozen/fit_scalers"
FAULT_CONTRACT = (
    PARENT_ROOT / "experiment/prelock_bundle/frozen/frozen_fault_contract.json"
)
ORIGINAL_TRAINING_ROOT = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_training_v2"
)
ORIGINAL_ARX_EVALUATION = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_evaluation_v3"
)
NEURAL_EVALUATION = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_evaluation_v6"
)
TRANSPORT_PRELOCK_ROOT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_prelock_v4"
)
TRANSPORT_DATA_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data"
)
TRANSPORT_READINESS = (
    PROJECT_ROOT
    / "artifacts/direct_h8_transport_collection_freeze_v5/collection_readiness.json"
)
TRANSPORT_EXTERNAL_FREEZE = (
    PROJECT_ROOT
    / "artifacts/direct_h8_transport_collection_freeze_v5/external_freeze_receipt.json"
)
TRANSPORT_STATE_ROOT = (
    PROJECT_ROOT
    / "building_fault_wm/transport_collection"
    / ".direct_h8_transport_collection_state_v3"
)
TRANSPORT_MANIFEST = (
    TRANSPORT_DATA_ROOT / "manifests/locked_transport_corpus_manifest.json"
)
ORIGINAL_TRANSPORT_BINDING = (
    PROJECT_ROOT
    / "artifacts/schedule_matched_arx_transport_prelock_v2"
    / "bundle/transport_collection_binding.json"
)

TERMINAL_V2_TRAINING_ROOT = (
    PROJECT_ROOT / "artifacts/post_outcome_strong_arx_training_v2"
)
TERMINAL_V2_CLOSEOUT = (
    PROJECT_ROOT
    / "artifacts/post_outcome_strong_arx_v2_failed_partial_evaluation_closeout.json"
)
DEFAULT_TRAINING_ROOT = (
    PROJECT_ROOT / "artifacts/post_outcome_strong_arx_training_v3"
)
DEFAULT_READINESS_ROOT = (
    PROJECT_ROOT / "artifacts/post_outcome_strong_arx_readiness_v3"
)
DEFAULT_EVALUATION_ROOT = (
    PROJECT_ROOT / "artifacts/post_outcome_strong_arx_evaluation_v3"
)

TRAINING_LOCK_SCHEMA = "post-outcome-strong-arx-training-source-lock-v1"
MODEL_SCHEMA = "post-outcome-strong-ridge-arx-model-v1"
RUN_SCHEMA = "post-outcome-strong-ridge-arx-selection-run-v1"
GRID_SCHEMA = "post-outcome-strong-ridge-arx-selection-grid-v1"
READINESS_SCHEMA = "post-outcome-strong-ridge-arx-readiness-v1"
EVALUATION_SCHEMA = "post-outcome-strong-ridge-arx-evaluation-v1"
COMPLETION_SCHEMA = "post-outcome-strong-ridge-arx-completion-v1"
RESULT_SCHEMA = "post-outcome-strong-arx-neural-sensitivity-v1"
ARM = "post_outcome_strengthened_recursive_ridge_arx"

TRAINING_LOCK_NAME = "training_source_lock.json"
TRAINING_COMPLETE_NAME = "selection_grid_complete.json"
READINESS_NAME = "readiness.json"
READINESS_DIGEST_NAME = "readiness.canonical.sha256"
EVALUATION_ATTEMPT_NAME = "evaluation_attempt.json"
COMPLETION_NAME = "evaluation_complete.json"

CORE_COLUMNS = original_evaluate.CORE_COLUMNS
DETAIL_COLUMNS = (
    *CORE_COLUMNS,
    "selected_history",
    "selected_alpha",
    "fault_channel_index",
    "severity_unit",
    "history_start",
    "history_stop",
    "target_index",
    "target_raw",
    "prediction_raw",
    "prediction_standardized",
    "raw_abs_error",
    "raw_unit",
    "boundary_crossing",
    "action_transition_count",
)
RAW_UNITS = {
    "zone_temperature_k": "K",
    "hvac_electric_power_w": "W",
}


def source_manifest() -> dict[str, str]:
    """Hash every source and protocol file in this isolated namespace."""

    paths = {
        path.name: path
        for path in HERE.iterdir()
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    required = {
        "__init__.py",
        "config.py",
        "study.py",
        "audit.py",
        "report.py",
        "cli.py",
        "PROTOCOL.md",
        "test_study.py",
    }
    if set(paths) != required:
        raise ValueError("strengthened ARX source inventory changed")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    return write_once(
        path,
        frame.to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        ).encode("ascii"),
    )


def _config_payload() -> dict:
    return json.loads(json.dumps(CONFIG.to_dict()))


def _frozen_scaler_payload(case: str) -> dict:
    payload = strict_json(SCALER_ROOT / f"{case}.json")
    payload["fit_source_sha256"] = [
        tuple(item) for item in payload["fit_source_sha256"]
    ]
    return payload


def _scale_stats(payload: object, dimension: int, label: str) -> ScaleStats:
    if not isinstance(payload, dict) or set(payload) != {"mean", "scale"}:
        raise ValueError(f"{label} scaler fields changed")
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0.0).any()
    ):
        raise ValueError(f"{label} scaler values changed")
    return ScaleStats(tuple(mean), tuple(scale))


def load_frozen_scaler(case: str) -> FaultScalers:
    if case not in CASES:
        raise ValueError("scaler case is outside the strengthened grid")
    payload = strict_json(SCALER_ROOT / f"{case}.json")
    if set(payload) != {
        "observation",
        "action",
        "context",
        "fit_source_sha256",
    }:
        raise ValueError("frozen scaler fields changed")
    sources = payload["fit_source_sha256"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in sources
        )
    ):
        raise ValueError("frozen scaler source identity changed")
    return FaultScalers(
        observation=_scale_stats(payload["observation"], 4, "observation"),
        action=_scale_stats(payload["action"], 1, "action"),
        context=_scale_stats(payload["context"], 5, "context"),
        fit_source_sha256=tuple((str(a), str(b)) for a, b in sources),
    )


def load_frozen_fault_spec() -> FaultSpec:
    payload = strict_json(FAULT_CONTRACT)
    if (
        payload.get("schema")
        != "boptest-multicase-frozen-fault-contract-v1"
        or not isinstance(payload.get("spec"), dict)
    ):
        raise ValueError("frozen fault contract changed")
    spec = FaultSpec.from_dict(payload["spec"])
    if spec.evaluation_horizon != 8:
        raise ValueError("strengthened ARX requires the H8 fault contract")
    return spec


def _feature_from_standardized(
    standardized_observations: np.ndarray,
    availability: np.ndarray,
    age: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    source: int,
    scalers: FaultScalers,
    history: int,
) -> np.ndarray:
    """Construct one causal ARX feature without using a future observation."""

    if history not in CONFIG.histories:
        raise ValueError("ARX history is outside the strengthened grid")
    if source < 0 or source + 1 >= len(actions):
        raise ValueError("ARX source leaves its trajectory")
    start = source - history + 1
    available_start = max(0, start)
    available_count = source - available_start + 1
    observation = np.zeros((history, CONFIG.observation_dim), dtype=float)
    mask = np.zeros((history, CONFIG.observation_dim), dtype=float)
    log_age = np.zeros((history, CONFIG.observation_dim), dtype=float)
    observation[-available_count:] = standardized_observations[
        available_start : source + 1
    ]
    mask[-available_count:] = availability[
        available_start : source + 1
    ].astype(float)
    observation = np.where(mask.astype(bool), observation, 0.0)
    log_age[-available_count:] = np.log1p(
        age[available_start : source + 1]
    )
    action_start = max(0, source - history)
    action_count = source - action_start
    previous_action = np.zeros((history, CONFIG.action_dim), dtype=float)
    if action_count:
        previous_action[-action_count:] = scalers.action.transform(
            actions[action_start:source]
        )
    current_action = scalers.action.transform(actions[source : source + 1])
    known_context = scalers.context.transform(contexts[source : source + 2])
    features = np.concatenate(
        [
            observation.reshape(-1),
            mask.reshape(-1),
            log_age.reshape(-1),
            previous_action.reshape(-1),
            current_action.reshape(-1),
            known_context.reshape(-1),
        ]
    )
    if (
        features.shape != (CONFIG.feature_dim(history),)
        or not np.isfinite(features).all()
    ):
        raise ValueError("ARX feature vector differs from its fixed contract")
    return features


def one_step_features(
    variant: FaultVariant,
    source: int,
    scalers: FaultScalers,
    history: int,
) -> np.ndarray:
    standardized = np.zeros_like(variant.corrupted_observations, dtype=float)
    standardized[: source + 1] = scalers.observation.transform(
        variant.corrupted_observations[: source + 1]
    )
    return _feature_from_standardized(
        standardized,
        variant.availability,
        variant.age,
        variant.actions,
        variant.contexts,
        source,
        scalers,
        history,
    )


def recursive_prediction(
    model: Ridge,
    variant: FaultVariant,
    scalers: FaultScalers,
    anchor: int,
    horizon: int,
    history: int,
    *,
    candidate_actions: np.ndarray | None = None,
) -> np.ndarray:
    """Recursively predict without consuming any observation after the anchor."""

    if history not in CONFIG.histories or horizon not in CONFIG.horizons:
        raise ValueError("recursive prediction is outside the fixed grid")
    if anchor < history or anchor + horizon >= len(variant.clean_observations):
        raise ValueError("recursive prediction leaves its causal support")
    actions = np.array(variant.actions, copy=True)
    if candidate_actions is not None:
        candidate = np.asarray(candidate_actions, dtype=float)
        if candidate.shape != (horizon, CONFIG.action_dim):
            raise ValueError("candidate action block has the wrong shape")
        actions[anchor : anchor + horizon] = candidate
    standardized = np.zeros_like(variant.corrupted_observations, dtype=float)
    standardized[: anchor + 1] = scalers.observation.transform(
        variant.corrupted_observations[: anchor + 1]
    )
    availability = np.zeros_like(variant.availability, dtype=bool)
    availability[: anchor + 1] = variant.availability[: anchor + 1]
    age = np.zeros_like(variant.age, dtype=float)
    age[: anchor + 1] = variant.age[: anchor + 1]
    prediction = None
    for source in range(anchor, anchor + horizon):
        features = _feature_from_standardized(
            standardized,
            availability,
            age,
            actions,
            variant.contexts,
            source,
            scalers,
            history,
        )
        prediction = np.asarray(model.predict(features[None])[0], dtype=float)
        if prediction.shape != (CONFIG.observation_dim,) or not np.isfinite(
            prediction
        ).all():
            raise ValueError("recursive prediction produced an invalid value")
        standardized[source + 1] = prediction
        availability[source + 1] = True
        age[source + 1] = 0.0
    if prediction is None:
        raise AssertionError("recursive prediction performed no step")
    return prediction


def scheduled_dataset(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    schedule: Sequence[ScheduledBatch],
    history: int,
    *,
    require_complete: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, str], ...], str]:
    """Materialize the exact parent source rows with a candidate causal lag."""

    if history not in CONFIG.histories:
        raise ValueError("scheduled dataset history is outside the fixed grid")
    normalized = tuple(schedule)
    if require_complete and (
        len(normalized) != CONFIG.updates
        or tuple(item.update for item in normalized)
        != tuple(range(1, CONFIG.updates + 1))
    ):
        raise ValueError("selection requires the complete parent schedule")
    if not normalized or any(
        len(item.references) != CONFIG.batch_size for item in normalized
    ):
        raise ValueError("selection schedule has an incomplete batch")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[tuple[str, str]] = []
    semantic = hashlib.sha256()
    for scheduled in normalized:
        for batch_index, reference in enumerate(scheduled.references):
            if not 0 <= reference.variant_index < len(variants):
                raise ValueError("schedule variant index is invalid")
            variant = variants[reference.variant_index]
            first = reference.aligned_start + CONFIG.source_history
            stop = reference.aligned_start + CONFIG.sequence_length
            if stop - first != CONFIG.scheduled_sources_per_sequence:
                raise AssertionError("fixed source-row count changed")
            for source in range(first, stop):
                features.append(
                    one_step_features(variant, source, scalers, history)
                )
                targets.append(
                    scalers.observation.transform(
                        variant.clean_observations[source + 1 : source + 2]
                    )[0]
                )
                groups.append(
                    (variant.cell.fault_channel, variant.cell.family)
                )
                semantic.update(
                    (
                        f"{scheduled.update}:{batch_index}:"
                        f"{variant.cell.cell_id}:{source}\n"
                    ).encode("ascii")
                )
    x = np.stack(features)
    y = np.stack(targets)
    if (
        x.shape[1] != CONFIG.feature_dim(history)
        or y.shape[1] != CONFIG.observation_dim
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("scheduled selection dataset is invalid")
    if require_complete and len(x) != CONFIG.scheduled_rows_per_model:
        raise ValueError("scheduled selection row count changed")
    return x, y, tuple(groups), semantic.hexdigest()


def _ridge_from_state(
    alpha: float,
    coefficient: np.ndarray,
    intercept: np.ndarray,
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    model.coef_ = np.asarray(coefficient, dtype=np.float64)
    model.intercept_ = np.asarray(intercept, dtype=np.float64)
    model.n_features_in_ = int(model.coef_.shape[1])
    return model


def ridge_path(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alphas: Sequence[float] = CONFIG.alphas,
) -> dict[float, Ridge]:
    """Fit the exact weighted Ridge path from one shared centered Gram matrix."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if (
        x.ndim != 2
        or y.shape != (len(x), CONFIG.observation_dim)
        or weight.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(weight).all()
        or (weight <= 0).any()
    ):
        raise ValueError("weighted Ridge path inputs are invalid")
    alpha_grid = tuple(float(value) for value in alphas)
    if not alpha_grid or any(value <= 0 for value in alpha_grid):
        raise ValueError("weighted Ridge path alphas are invalid")
    x_mean = np.average(x, axis=0, weights=weight)
    y_mean = np.average(y, axis=0, weights=weight)
    x_centered = x - x_mean
    y_centered = y - y_mean
    root_weight = np.sqrt(weight)
    weighted_x = x_centered * root_weight[:, None]
    weighted_y = y_centered * root_weight[:, None]
    gram = weighted_x.T @ weighted_x
    rhs = weighted_x.T @ weighted_y
    identity = np.eye(x.shape[1], dtype=np.float64)
    models = {}
    for alpha in alpha_grid:
        coefficient = scipy.linalg.solve(
            gram + alpha * identity,
            rhs,
            assume_a="pos",
            check_finite=False,
        ).T
        intercept = y_mean - x_mean @ coefficient.T
        models[alpha] = _ridge_from_state(alpha, coefficient, intercept)
    return models


def validation_score(
    model: Ridge,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    history: int,
) -> float:
    """Parent development-validation H8 equal-family/channel objective."""

    rows = []
    for variant in variants:
        if variant.cell.trajectory.role != "validation":
            raise ValueError("selection received a non-validation trajectory")
        for anchor in variant.cell.anchors:
            prediction = recursive_prediction(
                model,
                variant,
                scalers,
                anchor,
                max(CONFIG.horizons),
                history,
            )
            target = scalers.observation.transform(
                variant.clean_observations[
                    anchor + max(CONFIG.horizons) :
                    anchor + max(CONFIG.horizons) + 1
                ]
            )[0]
            channel = FAULT_CHANNELS.index(variant.cell.fault_channel)
            rows.append(
                {
                    "family": variant.cell.family,
                    "fault_channel": variant.cell.fault_channel,
                    "error": abs(prediction[channel] - target[channel]),
                }
            )
    if not rows:
        raise ValueError("selection validation grid is empty")
    cells = pd.DataFrame(rows).groupby(
        ["family", "fault_channel"], as_index=False, dropna=False
    )["error"].mean()
    return float(cells["error"].mean())


def validate_selection_table(table: pd.DataFrame) -> None:
    """Reject any incomplete or duplicated lag/alpha candidate grid."""

    required = (
        "history",
        "alpha",
        "feature_dim",
        "active_coefficients",
        "validation_h8_mae",
        "selected",
    )
    expected = {
        (history, alpha)
        for history in CONFIG.histories
        for alpha in CONFIG.alphas
    }
    observed = set(
        zip(
            (int(value) for value in table.get("history", [])),
            (float(value) for value in table.get("alpha", [])),
        )
    )
    if (
        tuple(table.columns) != required
        or len(table) != len(expected)
        or observed != expected
        or table.duplicated(["history", "alpha"]).any()
        or int(table["selected"].astype(bool).sum()) != 1
        or not np.isfinite(
            table[
                [
                    "history",
                    "alpha",
                    "feature_dim",
                    "active_coefficients",
                    "validation_h8_mae",
                ]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("strengthened selection candidate grid is incomplete")
    for row in table.itertuples(index=False):
        if (
            int(row.feature_dim) != CONFIG.feature_dim(int(row.history))
            or int(row.active_coefficients)
            != CONFIG.active_coefficients(int(row.history))
        ):
            raise ValueError("strengthened selection candidate size changed")


def selection_boundary_flags(history: int, alpha: float) -> dict[str, bool]:
    if history not in CONFIG.histories or alpha not in CONFIG.alphas:
        raise ValueError("selected hyperparameter lies outside the fixed grid")
    return {
        "selected_history_at_grid_max": history == max(CONFIG.histories),
        "selected_alpha_at_grid_max": alpha == max(CONFIG.alphas),
    }


def _model_state_payload(model: Ridge) -> dict:
    body = {
        "alpha": float(model.alpha),
        "fit_intercept": True,
        "solver": "precomputed_weighted_cholesky",
        "n_features_in": int(model.n_features_in_),
        "coef": np.asarray(model.coef_, dtype=np.float64).tolist(),
        "intercept": np.asarray(model.intercept_, dtype=np.float64).tolist(),
    }
    return {**body, "state_sha256": canonical_sha256(body)}


def restore_model(payload: Mapping[str, object]) -> tuple[Ridge, int]:
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("strengthened ARX model schema changed")
    history = int(payload.get("selected_history", -1))
    if history not in CONFIG.histories:
        raise ValueError("strengthened ARX selected history changed")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("strengthened ARX state is missing")
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != canonical_sha256(body):
        raise ValueError("strengthened ARX state hash changed")
    coefficient = np.asarray(body["coef"], dtype=np.float64)
    intercept = np.asarray(body["intercept"], dtype=np.float64)
    if coefficient.shape != (
        CONFIG.observation_dim,
        CONFIG.feature_dim(history),
    ) or intercept.shape != (CONFIG.observation_dim,):
        raise ValueError("strengthened ARX coefficient shape changed")
    model = _ridge_from_state(
        float(body["alpha"]), coefficient, intercept
    )
    if int(body["n_features_in"]) != model.n_features_in_:
        raise ValueError("strengthened ARX input width changed")
    return model, history


def _selection_inputs(
    case: str,
) -> tuple[
    Sequence[FaultVariant],
    Sequence[FaultVariant],
    FaultScalers,
    object,
]:
    """Load only parent development FIT and validation data."""

    if case not in CASES:
        raise ValueError("selection case is outside the fixed grid")
    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    if index.collection_kind != "development":
        raise ValueError("selection input is not the development corpus")
    fault_manifest = build_fault_manifest(index)
    fit_variants, scalers = prepare_case_training_data(
        index, fault_manifest, case
    )
    validation_variants = tuple(
        iter_role_variants(
            index,
            fault_manifest,
            "validation",
            cases=(case,),
        )
    )
    if (
        any(item.cell.trajectory.role != "fit" for item in fit_variants)
        or any(
            item.cell.trajectory.role != "validation"
            for item in validation_variants
        )
        or canonical_payload_sha256(asdict(scalers))
        != canonical_payload_sha256(_frozen_scaler_payload(case))
    ):
        raise ValueError("selection development roles or scaler identity changed")
    return fit_variants, validation_variants, scalers, fault_manifest


def _old_exposure_digest(case: str, model_seed: int) -> str:
    receipt = strict_json(
        ORIGINAL_TRAINING_ROOT
        / case
        / f"seed{model_seed}"
        / "training_receipt.json"
    )
    value = receipt.get("scheduled_exposure_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("original ARX exposure digest changed")
    return value


def _training_lock_payload() -> dict:
    original_selected = [
        float(
            strict_json(
                ORIGINAL_TRAINING_ROOT
                / case
                / f"seed{seed}"
                / "training_receipt.json"
            )["selected_alpha"]
        )
        for case in CASES
        for seed in MODEL_SEEDS
    ]
    return {
        "schema": TRAINING_LOCK_SCHEMA,
        "scope": "post_outcome_robustness",
        "motivation": (
            "original alpha grid reached its maximum in "
            f"{sum(value == 100.0 for value in original_selected)}/15 fits"
        ),
        "original_max_alpha_selected_count": sum(
            value == 100.0 for value in original_selected
        ),
        "locked_test_values_admissible_for_selection": False,
        "selection_roles": ["fit", "validation"],
        "selection_rule": (
            "minimum_parent_development_validation_h8_equal_family_channel_mae;"
            "ties_smaller_history_then_smaller_alpha"
        ),
        "config": _config_payload(),
        "source_manifest": source_manifest(),
        "development_manifest_file_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "fault_contract_file_sha256": sha256_file(FAULT_CONTRACT),
        "schedule_file_sha256_by_case_seed": {
            f"{case}/seed{seed}": sha256_file(
                SCHEDULE_ROOT / case / f"seed{seed}.json"
            )
            for case in CASES
            for seed in MODEL_SEEDS
        },
        "fit_scaler_file_sha256_by_case": {
            case: sha256_file(SCALER_ROOT / f"{case}.json")
            for case in CASES
        },
        "original_exposure_sha256_by_case_seed": {
            f"{case}/seed{seed}": _old_exposure_digest(case, seed)
            for case in CASES
            for seed in MODEL_SEEDS
        },
        "selection_recovery": {
            "kind": "byte_identical_development_selection_recovery",
            "source_root": str(TERMINAL_V2_TRAINING_ROOT.resolve()),
            "source_grid_file_sha256": sha256_file(
                TERMINAL_V2_TRAINING_ROOT / TRAINING_COMPLETE_NAME
            ),
            "source_lock_file_sha256": sha256_file(
                TERMINAL_V2_TRAINING_ROOT / TRAINING_LOCK_NAME
            ),
            "reason": (
                "v2 evaluation stopped after first-case in-memory scoring when "
                "a case-local frame reached the full-grid validator"
            ),
        },
    }


def prepare_training_root(
    output_root: Path = DEFAULT_TRAINING_ROOT,
) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite strengthened training root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    write_json_once(output_root / TRAINING_LOCK_NAME, _training_lock_payload())
    return output_root


def _verify_training_lock(output_root: Path) -> dict:
    payload = strict_json(output_root / TRAINING_LOCK_NAME)
    if payload != _training_lock_payload():
        raise ValueError("strengthened training source lock changed")
    return payload


def recover_v2_selection_runs(
    output_root: Path = DEFAULT_TRAINING_ROOT,
) -> Path:
    """Recover byte-identical development selections after the v2 closeout."""

    _verify_training_lock(output_root)
    if not TERMINAL_V2_CLOSEOUT.is_file():
        raise ValueError("terminal v2 closeout is missing")
    closeout = strict_json(TERMINAL_V2_CLOSEOUT)
    if (
        closeout.get("schema")
        != "post-outcome-strong-arx-v2-failed-partial-evaluation-closeout-v1"
        or closeout.get("new_strengthened_arx_heldout_values_accessed")
        is not True
        or closeout.get("result_csv_written") is not False
        or closeout.get("rerun_under_same_namespace_permitted") is not False
    ):
        raise ValueError("terminal v2 closeout is invalid")
    for case in CASES:
        for seed in MODEL_SEEDS:
            source = TERMINAL_V2_TRAINING_ROOT / case / f"seed{seed}"
            destination = output_root / case / f"seed{seed}"
            if os.path.lexists(destination):
                raise FileExistsError(
                    f"refusing to overwrite recovered run: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, copy_function=shutil.copyfile)
            load_model_run(destination, case=case, model_seed=seed)
    return output_root


def fit_case_seed(
    *,
    case: str,
    model_seed: int,
    output_root: Path = DEFAULT_TRAINING_ROOT,
) -> Path:
    """Select lag and alpha from development data for one case/seed."""

    _verify_training_lock(output_root)
    if case not in CASES or model_seed not in MODEL_SEEDS:
        raise ValueError("selection identity is outside the fixed grid")
    output_dir = output_root / case / f"seed{model_seed}"
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite selection run: {output_dir}")
    started = time.perf_counter()
    fit_variants, validation_variants, scalers, _ = _selection_inputs(case)
    schedule_path = SCHEDULE_ROOT / case / f"seed{model_seed}.json"
    schedule = original_train.load_parent_schedule(
        schedule_path, fit_variants
    )
    counts: Counter[tuple[str, str]] | None = None
    exposure_sha256 = None
    rows = []
    candidates: dict[tuple[int, float], Ridge] = {}
    for history in CONFIG.histories:
        x, y, groups, candidate_exposure = scheduled_dataset(
            fit_variants,
            scalers,
            schedule,
            history,
        )
        if exposure_sha256 is None:
            exposure_sha256 = candidate_exposure
        if candidate_exposure != exposure_sha256:
            raise AssertionError("candidate histories use different source rows")
        if candidate_exposure != _old_exposure_digest(case, model_seed):
            raise ValueError("strengthened selection changed parent source rows")
        candidate_counts = Counter(groups)
        if counts is None:
            counts = candidate_counts
        if candidate_counts != counts:
            raise AssertionError("candidate histories changed source strata")
        weight = np.asarray(
            [1.0 / candidate_counts[group] for group in groups], dtype=float
        )
        weight *= len(weight) / weight.sum()
        path = ridge_path(x, y, weight)
        for alpha, model in path.items():
            score = validation_score(
                model, validation_variants, scalers, history
            )
            candidates[(history, alpha)] = model
            rows.append(
                {
                    "history": history,
                    "alpha": alpha,
                    "feature_dim": CONFIG.feature_dim(history),
                    "active_coefficients": CONFIG.active_coefficients(history),
                    "validation_h8_mae": score,
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["validation_h8_mae", "history", "alpha"], kind="stable"
    ).reset_index(drop=True)
    table["selected"] = False
    table.loc[0, "selected"] = True
    validate_selection_table(table)
    selected_history = int(table.loc[0, "history"])
    selected_alpha = float(table.loc[0, "alpha"])
    boundary = selection_boundary_flags(selected_history, selected_alpha)
    selected_model = candidates[(selected_history, selected_alpha)]
    state = _model_state_payload(selected_model)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_payload = {
        "schema": MODEL_SCHEMA,
        "scope": "post_outcome_robustness",
        "case": case,
        "model_seed": model_seed,
        "selected_history": selected_history,
        "selected_alpha": selected_alpha,
        "config": _config_payload(),
        "state": state,
    }
    model_path = write_json_once(output_dir / "model.json", model_payload)
    score_path = _write_frame(output_dir / "selection_scores.csv", table)
    receipt = {
        "schema": RUN_SCHEMA,
        "scope": "post_outcome_robustness",
        "case": case,
        "model_seed": model_seed,
        "selection_rule": (
            "minimum_parent_development_validation_h8_equal_family_channel_mae;"
            "ties_smaller_history_then_smaller_alpha"
        ),
        "locked_test_values_accessed": False,
        "candidate_count": len(table),
        "complete_history_grid": sorted(
            set(int(value) for value in table["history"])
        )
        == list(CONFIG.histories),
        "complete_alpha_grid": sorted(
            set(float(value) for value in table["alpha"])
        )
        == list(CONFIG.alphas),
        "selected_history": selected_history,
        "selected_alpha": selected_alpha,
        "selected_validation_h8_mae": float(
            table.loc[0, "validation_h8_mae"]
        ),
        **boundary,
        "training_rows": CONFIG.scheduled_rows_per_model,
        "scheduled_exposure_sha256": exposure_sha256,
        "original_scheduled_exposure_sha256": _old_exposure_digest(
            case, model_seed
        ),
        "schedule_file_sha256": sha256_file(schedule_path),
        "fit_scalers_sha256": canonical_payload_sha256(asdict(scalers)),
        "fit_variant_identity_sha256": canonical_sha256(
            [variant.cell.cell_id for variant in fit_variants]
        ),
        "validation_variant_identity_sha256": canonical_sha256(
            [variant.cell.cell_id for variant in validation_variants]
        ),
        "model_state_sha256": state["state_sha256"],
        "model_file_sha256": sha256_file(model_path),
        "score_table_file_sha256": sha256_file(score_path),
        "score_table_payload_sha256": canonical_sha256(
            table.to_dict(orient="records")
        ),
        "wall_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
        },
    }
    write_json_once(output_dir / "selection_receipt.json", receipt)
    return output_dir


def load_model_run(
    run_dir: Path,
    *,
    case: str,
    model_seed: int,
) -> tuple[Ridge, int, dict]:
    receipt = strict_json(run_dir / "selection_receipt.json")
    model_path = run_dir / "model.json"
    score_path = run_dir / "selection_scores.csv"
    if (
        receipt.get("schema") != RUN_SCHEMA
        or receipt.get("case") != case
        or receipt.get("model_seed") != model_seed
        or receipt.get("locked_test_values_accessed") is not False
        or receipt.get("candidate_count")
        != len(CONFIG.histories) * len(CONFIG.alphas)
        or receipt.get("complete_history_grid") is not True
        or receipt.get("complete_alpha_grid") is not True
        or receipt.get("model_file_sha256") != sha256_file(model_path)
        or receipt.get("score_table_file_sha256") != sha256_file(score_path)
    ):
        raise ValueError("strengthened selection receipt changed")
    table = pd.read_csv(score_path, float_precision="round_trip")
    validate_selection_table(table)
    if receipt.get("score_table_payload_sha256") != canonical_sha256(
        table.to_dict(orient="records")
    ):
        raise ValueError("strengthened selection grid changed")
    selected = table.loc[table["selected"].astype(bool)].iloc[0]
    payload = strict_json(model_path)
    model, history = restore_model(payload)
    if (
        payload.get("case") != case
        or payload.get("model_seed") != model_seed
        or history != int(selected["history"])
        or float(model.alpha) != float(selected["alpha"])
        or receipt.get("selected_history") != history
        or float(receipt.get("selected_alpha")) != float(model.alpha)
        or {
            key: receipt.get(key)
            for key in (
                "selected_history_at_grid_max",
                "selected_alpha_at_grid_max",
            )
        }
        != selection_boundary_flags(history, float(model.alpha))
        or receipt.get("scheduled_exposure_sha256")
        != receipt.get("original_scheduled_exposure_sha256")
    ):
        raise ValueError("selected strengthened model identity changed")
    return model, history, receipt


def finalize_training_grid(
    output_root: Path = DEFAULT_TRAINING_ROOT,
) -> Path:
    lock = _verify_training_lock(output_root)
    runs = []
    for case in CASES:
        for seed in MODEL_SEEDS:
            run_dir = output_root / case / f"seed{seed}"
            _, _, receipt = load_model_run(
                run_dir, case=case, model_seed=seed
            )
            runs.append(
                {
                    "case": case,
                    "model_seed": seed,
                    "selected_history": receipt["selected_history"],
                    "selected_alpha": receipt["selected_alpha"],
                    "selected_validation_h8_mae": receipt[
                        "selected_validation_h8_mae"
                    ],
                    "selected_history_at_grid_max": receipt[
                        "selected_history_at_grid_max"
                    ],
                    "selected_alpha_at_grid_max": receipt[
                        "selected_alpha_at_grid_max"
                    ],
                    "model_file_sha256": receipt["model_file_sha256"],
                    "selection_scores_file_sha256": receipt[
                        "score_table_file_sha256"
                    ],
                    "selection_receipt_file_sha256": sha256_file(
                        run_dir / "selection_receipt.json"
                    ),
                }
            )
    payload = {
        "schema": GRID_SCHEMA,
        "scope": "post_outcome_robustness",
        "complete_grid": True,
        "source_lock_file_sha256": sha256_file(
            output_root / TRAINING_LOCK_NAME
        ),
        "source_lock_payload_sha256": canonical_sha256(lock),
        "case_seed_count": len(runs),
        "candidate_count_per_case_seed": (
            len(CONFIG.histories) * len(CONFIG.alphas)
        ),
        "total_validation_candidates": (
            len(runs) * len(CONFIG.histories) * len(CONFIG.alphas)
        ),
        "selected_history_at_grid_max_count": sum(
            bool(row["selected_history_at_grid_max"]) for row in runs
        ),
        "selected_alpha_at_grid_max_count": sum(
            bool(row["selected_alpha_at_grid_max"]) for row in runs
        ),
        "runs": runs,
        "selection_recovery": {
            "recovered_from_terminal_v2": True,
            "terminal_v2_closeout_file_sha256": sha256_file(
                TERMINAL_V2_CLOSEOUT
            ),
            "terminal_v2_grid_file_sha256": sha256_file(
                TERMINAL_V2_TRAINING_ROOT / TRAINING_COMPLETE_NAME
            ),
        },
    }
    return write_json_once(output_root / TRAINING_COMPLETE_NAME, payload)


def verify_training_grid(
    output_root: Path = DEFAULT_TRAINING_ROOT,
) -> dict:
    lock = _verify_training_lock(output_root)
    receipt = strict_json(output_root / TRAINING_COMPLETE_NAME)
    if (
        receipt.get("schema") != GRID_SCHEMA
        or receipt.get("scope") != "post_outcome_robustness"
        or receipt.get("complete_grid") is not True
        or receipt.get("source_lock_file_sha256")
        != sha256_file(output_root / TRAINING_LOCK_NAME)
        or receipt.get("source_lock_payload_sha256")
        != canonical_sha256(lock)
        or receipt.get("case_seed_count") != 15
        or receipt.get("candidate_count_per_case_seed") != 50
        or receipt.get("total_validation_candidates") != 750
    ):
        raise ValueError("strengthened training-grid receipt changed")
    expected = {(case, seed) for case in CASES for seed in MODEL_SEEDS}
    rows = receipt.get("runs")
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected)
        or {
            (row.get("case"), row.get("model_seed"))
            for row in rows
            if isinstance(row, dict)
        }
        != expected
    ):
        raise ValueError("strengthened training grid is incomplete")
    for row in rows:
        case = str(row["case"])
        seed = int(row["model_seed"])
        run_dir = output_root / case / f"seed{seed}"
        _, _, run = load_model_run(run_dir, case=case, model_seed=seed)
        if (
            row.get("model_file_sha256") != run["model_file_sha256"]
            or row.get("selection_scores_file_sha256")
            != run["score_table_file_sha256"]
            or row.get("selection_receipt_file_sha256")
            != sha256_file(run_dir / "selection_receipt.json")
        ):
            raise ValueError("strengthened run binding changed")
    return receipt


def _input_identity() -> dict:
    """Bind metadata and completed upstream outputs without parsing result CSVs."""

    transport_binding = strict_json(ORIGINAL_TRANSPORT_BINDING)
    neural_completion = strict_json(NEURAL_EVALUATION / v3_run.COMPLETION_NAME)
    original_completion = strict_json(
        ORIGINAL_ARX_EVALUATION / "evaluation_complete.json"
    )
    return {
        "transport_binding_file_sha256": sha256_file(
            ORIGINAL_TRANSPORT_BINDING
        ),
        "transport_binding_payload_sha256": canonical_sha256(
            transport_binding
        ),
        "transport_manifest_file_sha256": sha256_file(TRANSPORT_MANIFEST),
        "transport_readiness_file_sha256": sha256_file(TRANSPORT_READINESS),
        "transport_external_freeze_file_sha256": sha256_file(
            TRANSPORT_EXTERNAL_FREEZE
        ),
        "neural_completion_file_sha256": sha256_file(
            NEURAL_EVALUATION / v3_run.COMPLETION_NAME
        ),
        "neural_completion_payload_sha256": canonical_sha256(
            neural_completion
        ),
        "neural_gate_core_file_sha256": sha256_file(
            NEURAL_EVALUATION / v3_run.CORE_NAME
        ),
        "original_arx_completion_file_sha256": sha256_file(
            ORIGINAL_ARX_EVALUATION / "evaluation_complete.json"
        ),
        "original_arx_completion_payload_sha256": canonical_sha256(
            original_completion
        ),
        "original_arx_core_file_sha256": sha256_file(
            ORIGINAL_ARX_EVALUATION / "arx_core.csv"
        ),
    }


def _output_contract() -> dict:
    return {
        "core_columns": list(CORE_COLUMNS),
        "detail_columns": list(DETAIL_COLUMNS),
        "arm": ARM,
        "cases": list(CASES),
        "policies": list(POLICIES),
        "model_seeds": list(MODEL_SEEDS),
        "families": list(SILENT_FAMILIES),
        "horizons": list(CONFIG.horizons),
        "windows_per_case": 12,
        "expected_core_rows": 207_360,
        "analysis": {
            "scope": "post_outcome_descriptive_sensitivity",
            "estimand": (
                "1 - equal_weight_MAE(deterministic_wm) / "
                "equal_weight_MAE(strengthened_ridge_arx)"
            ),
            "bootstrap_draws": CONFIG.bootstrap_draws,
            "bootstrap_seed": CONFIG.bootstrap_seed,
            "hierarchy": "case/model_seed/window_within_case",
            "paired_across_arms_and_policies": True,
            "confirmatory_category": False,
        },
    }


def prepare_readiness(
    *,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    output_root: Path = DEFAULT_READINESS_ROOT,
) -> Path:
    """Freeze code, selections, inputs, and outputs before new held-out use."""

    training = verify_training_grid(training_root)
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite strengthened readiness: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    training_inventory = tree_inventory(training_root)
    payload = {
        "schema": READINESS_SCHEMA,
        "scope": "post_outcome_robustness",
        "post_outcome": True,
        "new_strengthened_arx_heldout_values_accessed_before_receipt": False,
        "original_outcome_motivation_disclosed": (
            "14/15 original fits selected alpha=100, the old grid maximum"
        ),
        "config": _config_payload(),
        "selection_rule": (
            "minimum_parent_development_validation_h8_equal_family_channel_mae;"
            "ties_smaller_history_then_smaller_alpha"
        ),
        "source_manifest": source_manifest(),
        "training_grid_file_sha256": sha256_file(
            training_root / TRAINING_COMPLETE_NAME
        ),
        "training_grid_payload_sha256": canonical_sha256(training),
        "training_inventory": training_inventory,
        "training_inventory_sha256": canonical_sha256(training_inventory),
        "input_identity": _input_identity(),
        "output_contract": _output_contract(),
    }
    path = write_json_once(output_root / READINESS_NAME, payload)
    digest = canonical_sha256(payload)
    write_once(
        output_root / READINESS_DIGEST_NAME,
        f"{digest}\n".encode("ascii"),
    )
    verify_readiness(training_root=training_root, readiness_root=output_root)
    return path


def verify_readiness(
    *,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    readiness_root: Path = DEFAULT_READINESS_ROOT,
) -> dict:
    payload = strict_json(readiness_root / READINESS_NAME)
    digest = canonical_sha256(payload)
    if (
        payload.get("schema") != READINESS_SCHEMA
        or payload.get("scope") != "post_outcome_robustness"
        or payload.get("post_outcome") is not True
        or payload.get(
            "new_strengthened_arx_heldout_values_accessed_before_receipt"
        )
        is not False
        or payload.get("config") != _config_payload()
        or payload.get("source_manifest") != source_manifest()
        or payload.get("input_identity") != _input_identity()
        or payload.get("output_contract") != _output_contract()
        or (readiness_root / READINESS_DIGEST_NAME).read_text(
            encoding="ascii"
        )
        != f"{digest}\n"
    ):
        raise ValueError("strengthened ARX readiness identity changed")
    training = verify_training_grid(training_root)
    inventory = tree_inventory(training_root)
    if (
        payload.get("training_grid_file_sha256")
        != sha256_file(training_root / TRAINING_COMPLETE_NAME)
        or payload.get("training_grid_payload_sha256")
        != canonical_sha256(training)
        or payload.get("training_inventory") != inventory
        or payload.get("training_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise ValueError("strengthened readiness training binding changed")
    return payload


def _validate_core(
    frame: pd.DataFrame,
    *,
    expected_cases: Sequence[str] = CASES,
) -> None:
    expected_case_set = set(expected_cases)
    if (
        not expected_case_set
        or expected_case_set - set(CASES)
        or len(expected_case_set) != len(tuple(expected_cases))
    ):
        raise ValueError("strengthened ARX expected-case grid is invalid")
    if tuple(frame.columns) != CORE_COLUMNS:
        raise ValueError("strengthened ARX core columns changed")
    identity = [
        column for column in CORE_COLUMNS if column != "standardized_abs_error"
    ]
    if (
        frame.empty
        or frame.duplicated(identity).any()
        or (frame["standardized_abs_error"] < 0).any()
    ):
        raise ValueError("strengthened ARX core is empty or invalid")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("strengthened ARX core has non-finite values")
    if (
        set(frame["case"]) != expected_case_set
        or set(frame["policy"]) != set(POLICIES)
        or set(int(value) for value in frame["model_seed"])
        != set(MODEL_SEEDS)
        or set(frame["arm"]) != {ARM}
        or set(frame["family"]) != set(SILENT_FAMILIES)
        or set(int(value) for value in frame["horizon"])
        != set(CONFIG.horizons)
        or len(frame) != 69_120 * len(expected_case_set)
    ):
        raise ValueError("strengthened ARX held-out grid is incomplete")
    for case in expected_case_set:
        if frame.loc[frame["case"] == case, "window_id"].nunique() != 12:
            raise ValueError(f"strengthened ARX windows incomplete for {case}")


def evaluate_variants(
    models: Mapping[int, tuple[Ridge, int, float]],
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    trajectory_metadata: Mapping[
        TrajectoryKey, v3_evaluate.PolicyTrajectoryMetadata
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selected models on the exact original transport grid."""

    if set(models) != set(MODEL_SEEDS):
        raise ValueError("strengthened ARX model grid is incomplete")
    silent = tuple(
        variant
        for variant in variants
        if variant.cell.family in SILENT_FAMILIES
    )
    v3_evaluate._validate_policy_metadata(silent, trajectory_metadata)
    rows: list[dict[str, object]] = []
    mean = np.asarray(scalers.observation.mean)
    scale = np.asarray(scalers.observation.scale)
    for model_seed, (model, history, alpha) in sorted(models.items()):
        for variant in silent:
            cell = variant.cell
            key = cell.trajectory
            metadata = trajectory_metadata[key]
            channel = FAULT_CHANNELS.index(cell.fault_channel)
            for anchor in cell.anchors:
                for horizon in CONFIG.horizons:
                    if (
                        anchor < history
                        or anchor + horizon >= cell.stop
                        or anchor + horizon >= len(variant.clean_observations)
                    ):
                        raise ValueError(
                            "strengthened ARX endpoint leaves its contract"
                        )
                    prediction = recursive_prediction(
                        model,
                        variant,
                        scalers,
                        anchor,
                        horizon,
                        history,
                    )
                    target_raw = variant.clean_observations[anchor + horizon]
                    target_standardized = scalers.observation.transform(
                        target_raw[None]
                    )[0]
                    prediction_raw = prediction * scale + mean
                    boundary, transition_count = (
                        v3_evaluate.action_block_crosses_transition(
                            variant.actions, anchor, horizon
                        )
                    )
                    rows.append(
                        {
                            "case": key.case,
                            "policy": metadata.policy,
                            "window_id": metadata.window_id,
                            "trajectory_day": key.day,
                            "scenario_seed": metadata.scenario_seed,
                            "trajectory_seed": key.trajectory_seed,
                            "model_seed": model_seed,
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
                                abs(
                                    prediction[channel]
                                    - target_standardized[channel]
                                )
                            ),
                            "selected_history": history,
                            "selected_alpha": alpha,
                            "fault_channel_index": channel,
                            "severity_unit": cell.severity_unit,
                            "history_start": anchor - history + 1,
                            "history_stop": anchor,
                            "target_index": anchor + horizon,
                            "target_raw": float(target_raw[channel]),
                            "prediction_raw": float(prediction_raw[channel]),
                            "prediction_standardized": float(
                                prediction[channel]
                            ),
                            "raw_abs_error": float(
                                abs(
                                    prediction_raw[channel]
                                    - target_raw[channel]
                                )
                            ),
                            "raw_unit": RAW_UNITS[cell.fault_channel],
                            "boundary_crossing": bool(boundary),
                            "action_transition_count": transition_count,
                        }
                    )
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
    evaluated_cases = tuple(sorted(set(core["case"])))
    _validate_core(core, expected_cases=evaluated_cases)
    return core, detailed


def _analyze(
    neural: pd.DataFrame,
    strengthened: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Reuse the original exact pairing, weighting, and bootstrap machinery."""

    normalized = strengthened.copy()
    normalized["arm"] = original_evaluate.ARM
    paired = original_analysis.validate_and_pair(neural, normalized)
    scores = original_analysis.equal_weight_scores(paired, horizon=8)
    family_scores = original_analysis.equal_weight_scores(
        paired, horizon=8, retain_family=True
    )
    plan = original_analysis._bootstrap_plan(
        scores,
        draws=CONFIG.bootstrap_draws,
        seed=CONFIG.bootstrap_seed,
    )
    policy_results = {}
    policy_draws = {}
    for policy in POLICIES:
        points, draws = original_analysis._bootstrap_arm_means(
            scores, policy, plan
        )
        effect_draws = np.asarray(
            1.0
            - draws[original_analysis.PRIMARY_NEURAL_ARM]
            / draws[original_evaluate.ARM],
            dtype=float,
        )
        point = float(
            1.0
            - points[original_analysis.PRIMARY_NEURAL_ARM]
            / points[original_evaluate.ARM]
        )
        policy_draws[policy] = effect_draws
        by_case = original_analysis._stratum_effects(
            scores, policy, "case"
        )
        by_family = original_analysis._stratum_effects(
            family_scores, policy, "family"
        )
        by_seed = original_analysis._stratum_effects(
            scores, policy, "model_seed"
        )
        policy_results[policy] = {
            **original_analysis._summary(point, effect_draws),
            "inference_scope": "post_outcome_descriptive_sensitivity",
            "confirmatory_category": None,
            "mean_standardized_mae_by_arm": {
                "deterministic_wm": points["deterministic_wm"],
                ARM: points[original_evaluate.ARM],
            },
            "by_case": by_case,
            "by_family": by_family,
            "by_seed": by_seed,
            "positive_seed_count": sum(value > 0 for value in by_seed.values()),
            "negative_seed_count": sum(value < 0 for value in by_seed.values()),
        }
    difference = (
        policy_results["new_4h"]["point"]
        - policy_results["old_2h"]["point"]
    )
    transport = original_analysis._summary(
        float(difference),
        policy_draws["new_4h"] - policy_draws["old_2h"],
    )
    descriptive = original_analysis.descriptive_by_horizon(paired)
    descriptive["arm"] = descriptive["arm"].replace(
        {original_evaluate.ARM: ARM}
    )
    result = {
        "schema": RESULT_SCHEMA,
        "scope": "post_outcome_descriptive_sensitivity",
        "post_outcome": True,
        "confirmatory_category_assigned": False,
        "primary_horizon_steps": 8,
        "families": list(SILENT_FAMILIES),
        "cases": list(CASES),
        "model_seeds": list(MODEL_SEEDS),
        "estimand": (
            "1 - equal_weight_MAE(deterministic_wm) / "
            "equal_weight_MAE(strengthened_ridge_arx)"
        ),
        "positive_favors": "deterministic_wm",
        "bootstrap": {
            "draws": CONFIG.bootstrap_draws,
            "seed": CONFIG.bootstrap_seed,
            "generator": "numpy.random.PCG64",
            "hierarchy": "case/model_seed/window_within_case",
            "paired_across_arms_and_policies": True,
        },
        "policy_results": policy_results,
        "transport_new_4h_minus_old_2h": transport,
        "original_new_4h_effect": 0.4081540134340662,
        "claim_scope": {
            "permitted": (
                "sensitivity of the deterministic world-model comparison to "
                "a validation-selected strengthened linear Ridge-ARX"
            ),
            "forbidden": [
                "confirmatory reclassification",
                "generic architecture superiority",
                "physical parameter identification",
                "observed-building generalization",
                "closed-loop control",
                "planning or MPC",
                "energy, cost, or comfort benefit",
            ],
        },
    }
    return result, paired, descriptive


def _descriptive_summary(core: pd.DataFrame) -> pd.DataFrame:
    cell_keys = [
        "case",
        "policy",
        "window_id",
        "model_seed",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "horizon",
    ]
    cells = core.groupby(cell_keys, as_index=False, dropna=False)[
        "standardized_abs_error"
    ].mean()
    keys = ["case", "policy", "fault_channel", "family", "horizon"]
    summary = cells.groupby(keys, as_index=False, dropna=False)[
        "standardized_abs_error"
    ].agg(["mean", "std", "count"]).reset_index()
    return summary.rename(
        columns={
            "mean": "equal_cell_mean_standardized_mae",
            "std": "equal_cell_std_standardized_mae",
            "count": "equal_cell_count",
        }
    ).sort_values(keys, kind="stable").reset_index(drop=True)


def run_evaluation(
    *,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    readiness_root: Path = DEFAULT_READINESS_ROOT,
    output_root: Path = DEFAULT_EVALUATION_ROOT,
) -> Path:
    """Perform the single write-once held-out pass after readiness validates."""

    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to rerun strengthened held-out evaluation: {output_root}"
        )
    readiness = verify_readiness(
        training_root=training_root, readiness_root=readiness_root
    )
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema": "post-outcome-strong-arx-evaluation-attempt-v1",
        "readiness_file_sha256": sha256_file(
            readiness_root / READINESS_NAME
        ),
        "readiness_payload_sha256": canonical_sha256(readiness),
        "new_strengthened_arx_heldout_values_accessed_before_attempt": False,
        "rerun_permitted_in_same_namespace": False,
    }
    write_json_once(output_root / EVALUATION_ATTEMPT_NAME, attempt)

    current_binding = bind_completed_transport_metadata(
        transport_prelock_root=TRANSPORT_PRELOCK_ROOT,
        transport_live_data_root=TRANSPORT_DATA_ROOT,
        transport_readiness_path=TRANSPORT_READINESS,
        transport_external_freeze_receipt_path=TRANSPORT_EXTERNAL_FREEZE,
        transport_state_root=TRANSPORT_STATE_ROOT,
        transport_manifest_path=TRANSPORT_MANIFEST,
        live_transport_external_freeze=False,
    )
    if current_binding != strict_json(ORIGINAL_TRANSPORT_BINDING):
        raise ValueError("held-out transport metadata changed")
    readiness_document = strict_json(TRANSPORT_READINESS)
    expected_prelock = str(readiness_document["prelock_registry_sha256"])
    expected_readiness = str(readiness_document["readiness_sha256"])
    _, collection_readiness = transport_runner.load_bound_readiness(
        prelock_root=TRANSPORT_PRELOCK_ROOT,
        live_data_root=TRANSPORT_DATA_ROOT,
        readiness_path=TRANSPORT_READINESS,
        expected_prelock_sha256=expected_prelock,
        expected_readiness_sha256=expected_readiness,
    )
    external_freeze = (
        transport_external_freeze.validate_external_freeze_receipt(
            TRANSPORT_EXTERNAL_FREEZE,
            expected_prelock,
            expected_readiness,
            prelock_root=TRANSPORT_PRELOCK_ROOT,
            readiness_path=TRANSPORT_READINESS,
            live=False,
        )
    )
    collection = v3_corpus.load_transport_corpus_index(
        manifest_path=TRANSPORT_MANIFEST,
        raw_root=TRANSPORT_DATA_ROOT / "locked_transport_raw",
        readiness=collection_readiness,
        expected_prelock_sha256=expected_prelock,
        state_root=TRANSPORT_STATE_ROOT,
        external_freeze=external_freeze,
        external_freeze_receipt_path=TRANSPORT_EXTERNAL_FREEZE,
    )
    spec = load_frozen_fault_spec()
    fault_manifest = build_fault_manifest(collection.index, spec)
    core_frames = []
    detail_frames = []
    selected = []
    for case in CASES:
        variants = tuple(
            iter_role_variants(
                collection.index,
                fault_manifest,
                "locked_test",
                cases=(case,),
                allow_locked_test=True,
            )
        )
        models = {}
        for seed in MODEL_SEEDS:
            run_dir = training_root / case / f"seed{seed}"
            model, history, receipt = load_model_run(
                run_dir, case=case, model_seed=seed
            )
            alpha = float(receipt["selected_alpha"])
            models[seed] = (model, history, alpha)
            selected.append(
                {
                    "case": case,
                    "model_seed": seed,
                    "selected_history": history,
                    "selected_alpha": alpha,
                    "selected_validation_h8_mae": receipt[
                        "selected_validation_h8_mae"
                    ],
                    "selected_history_at_grid_max": receipt[
                        "selected_history_at_grid_max"
                    ],
                    "selected_alpha_at_grid_max": receipt[
                        "selected_alpha_at_grid_max"
                    ],
                }
            )
        metadata = {
            key: value
            for key, value in collection.trajectory_metadata.items()
            if key.case == case
        }
        core, detailed = evaluate_variants(
            models, variants, load_frozen_scaler(case), metadata
        )
        core_frames.append(core)
        detail_frames.append(detailed)
    core = pd.concat(core_frames, ignore_index=True)
    detailed = pd.concat(detail_frames, ignore_index=True)
    _validate_core(core)
    summary = _descriptive_summary(core)

    neural_completion = strict_json(
        NEURAL_EVALUATION / v3_run.COMPLETION_NAME
    )
    if (
        sha256_file(NEURAL_EVALUATION / v3_run.COMPLETION_NAME)
        != readiness["input_identity"]["neural_completion_file_sha256"]
        or sha256_file(NEURAL_EVALUATION / v3_run.CORE_NAME)
        != readiness["input_identity"]["neural_gate_core_file_sha256"]
    ):
        raise ValueError("neural comparison input changed after readiness")
    neural = pd.read_csv(
        NEURAL_EVALUATION / v3_run.CORE_NAME,
        float_precision="round_trip",
    )
    result, paired, descriptive = _analyze(neural, core)

    core_path = _write_frame(output_root / "strong_arx_core.csv", core)
    detail_path = _write_frame(
        output_root / "strong_arx_detailed_diagnostics.csv", detailed
    )
    summary_path = _write_frame(
        output_root / "strong_arx_descriptive_summary.csv", summary
    )
    paired_path = _write_frame(
        output_root / "paired_neural_strong_arx_rows.csv", paired
    )
    descriptive_path = _write_frame(
        output_root / "descriptive_by_horizon.csv", descriptive
    )
    selection_path = _write_frame(
        output_root / "selected_hyperparameters.csv",
        pd.DataFrame(selected).sort_values(
            ["case", "model_seed"], kind="stable"
        ),
    )
    result_path = write_json_once(
        output_root / "sensitivity_result.json", result
    )
    provenance = {
        "schema": EVALUATION_SCHEMA,
        "scope": "post_outcome_descriptive_sensitivity",
        "single_heldout_pass": True,
        "readiness_file_sha256": sha256_file(
            readiness_root / READINESS_NAME
        ),
        "readiness_payload_sha256": canonical_sha256(readiness),
        "evaluation_attempt_file_sha256": sha256_file(
            output_root / EVALUATION_ATTEMPT_NAME
        ),
        "transport_binding_payload_sha256": canonical_sha256(
            current_binding
        ),
        "transport_manifest_file_sha256": collection.manifest_file_sha256,
        "fault_manifest_sha256": fault_manifest.sha256,
        "neural_completion_payload_sha256": canonical_sha256(
            neural_completion
        ),
        "selection_grid_file_sha256": sha256_file(
            training_root / TRAINING_COMPLETE_NAME
        ),
        "rows": len(core),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "wall_seconds": time.perf_counter() - started,
            "device": "cpu",
        },
    }
    provenance_path = write_json_once(
        output_root / "evaluation_provenance.json", provenance
    )
    output_paths = (
        core_path,
        detail_path,
        summary_path,
        paired_path,
        descriptive_path,
        selection_path,
        result_path,
        provenance_path,
        output_root / EVALUATION_ATTEMPT_NAME,
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "scope": "post_outcome_descriptive_sensitivity",
        "complete": True,
        "single_heldout_pass": True,
        "row_count": len(core),
        "result_payload_sha256": canonical_sha256(result),
        "file_sha256_by_name": {
            path.name: sha256_file(path) for path in output_paths
        },
    }
    return write_json_once(output_root / COMPLETION_NAME, completion)


def verify_evaluation(
    output_root: Path = DEFAULT_EVALUATION_ROOT,
) -> dict:
    completion = strict_json(output_root / COMPLETION_NAME)
    hashes = completion.get("file_sha256_by_name")
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("scope") != "post_outcome_descriptive_sensitivity"
        or completion.get("complete") is not True
        or completion.get("single_heldout_pass") is not True
        or completion.get("row_count") != 207_360
        or not isinstance(hashes, dict)
    ):
        raise ValueError("strengthened evaluation completion changed")
    expected = set(hashes) | {COMPLETION_NAME}
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if expected != actual:
        raise ValueError("strengthened evaluation file set changed")
    for name, digest in hashes.items():
        if sha256_file(output_root / name) != digest:
            raise ValueError(f"strengthened evaluation changed: {name}")
    result = strict_json(output_root / "sensitivity_result.json")
    if completion.get("result_payload_sha256") != canonical_sha256(result):
        raise ValueError("strengthened sensitivity result changed")
    return completion
