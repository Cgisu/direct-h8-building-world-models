from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from sklearn.linear_model import Ridge

from .fault_data import FAULT_CHANNELS, FaultScalers, FaultVariant
from .runtime_provenance import (
    numerical_runtime_fingerprint,
    validate_numerical_runtime_fingerprint,
)
from .study_config import StudyConfig


RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

BASELINE_RESULT_COLUMNS = (
    "case",
    "role",
    "trajectory_day",
    "trajectory_seed",
    "model_seed",
    "arm",
    "update",
    "cell_id",
    "fault_channel",
    "fault_channel_index",
    "family",
    "sign",
    "severity",
    "severity_unit",
    "onset",
    "anchor",
    "horizon",
    "target_raw",
    "prediction_raw",
    "standardized_abs_error",
    "alternate_action_prediction_raw",
    "alternate_action_standardized_abs_error",
    "action_prediction_change_standardized",
    "persistence_prediction_raw",
    "persistence_standardized_abs_error",
    "source_reliability",
    "source_healthy_probability",
)


def _canonical_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(content).hexdigest()


BASELINE_PRODUCER_CODE_SCHEMA = "boptest-multicase-baseline-producer-code-v1"


def baseline_producer_code_manifest() -> dict:
    here = Path(__file__).resolve().parent
    payload = {
        "schema": BASELINE_PRODUCER_CODE_SCHEMA,
        "files": {
            "multicase_fault_benchmark/baselines.py": hashlib.sha256(
                (here / "baselines.py").read_bytes()
            ).hexdigest(),
            "multicase_fault_benchmark/fault_data.py": hashlib.sha256(
                (here / "fault_data.py").read_bytes()
            ).hexdigest(),
            "multicase_fault_benchmark/study_config.py": hashlib.sha256(
                (here / "study_config.py").read_bytes()
            ).hexdigest(),
            "multicase_fault_benchmark/runtime_provenance.py": hashlib.sha256(
                (here / "runtime_provenance.py").read_bytes()
            ).hexdigest(),
        },
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def _validate_baseline_producer_sha256(value: object) -> None:
    if value != baseline_producer_code_manifest()["sha256"]:
        raise ValueError(
            "baseline producer-code SHA-256 differs from the current implementation"
        )


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("ascii")).hexdigest()


def _tensor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _variant_identity_sha256(variants: Sequence[FaultVariant]) -> str:
    identities = [
        {
            "cell_id": variant.cell.cell_id,
            "trajectory": variant.cell.trajectory.text,
            "source_sha256": variant.cell.source_sha256,
            "anchors": list(variant.cell.anchors),
        }
        for variant in sorted(variants, key=lambda item: item.cell.cell_id)
    ]
    return _canonical_sha256(identities)


def _scalers_sha256(scalers: FaultScalers) -> str:
    payload = {
        "observation": {
            "mean": list(scalers.observation.mean),
            "scale": list(scalers.observation.scale),
        },
        "action": {
            "mean": list(scalers.action.mean),
            "scale": list(scalers.action.scale),
        },
        "context": {
            "mean": list(scalers.context.mean),
            "scale": list(scalers.context.scale),
        },
        "fit_source_sha256": dict(scalers.fit_source_sha256),
    }
    return _canonical_sha256(payload)


@dataclass(frozen=True)
class BaselineSelectionReceipt:
    """Semantic selection record for an outer immutable run receipt."""

    schema: str
    baseline: str
    case: str
    model_seed: int
    fit_role: str
    validation_role: str
    fit_variant_identity_sha256: str
    validation_variant_identity_sha256: str
    fit_scalers_sha256: str
    feature_contract_sha256: str
    training_config_sha256: str
    selection_metric: str
    candidate_grid: tuple[float | int, ...]
    selected_candidate: float | int
    selected_validation_score: float
    score_table_sha256: str
    training_updates: int
    batch_size: int
    schedule_sha256: str
    selected_model_state_sha256: str
    producer_code_sha256: str
    runtime_fingerprint: dict

    @property
    def payload(self) -> dict:
        return {
            "schema": self.schema,
            "baseline": self.baseline,
            "case": self.case,
            "model_seed": self.model_seed,
            "fit_role": self.fit_role,
            "validation_role": self.validation_role,
            "fit_variant_identity_sha256": self.fit_variant_identity_sha256,
            "validation_variant_identity_sha256": self.validation_variant_identity_sha256,
            "fit_scalers_sha256": self.fit_scalers_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "training_config_sha256": self.training_config_sha256,
            "selection_metric": self.selection_metric,
            "candidate_grid": list(self.candidate_grid),
            "selected_candidate": self.selected_candidate,
            "selected_validation_score": self.selected_validation_score,
            "score_table_sha256": self.score_table_sha256,
            "training_updates": self.training_updates,
            "batch_size": self.batch_size,
            "schedule_sha256": self.schedule_sha256,
            "selected_model_state_sha256": self.selected_model_state_sha256,
            "producer_code_sha256": self.producer_code_sha256,
            "runtime_fingerprint": self.runtime_fingerprint,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.payload)


def _validate_selection_roles(
    fit_variants: Sequence[FaultVariant],
    validation_variants: Sequence[FaultVariant],
) -> str:
    if not fit_variants or not validation_variants:
        raise ValueError("baseline selection requires nonempty FIT and validation variants")
    if {item.cell.trajectory.role for item in fit_variants} != {"fit"}:
        raise ValueError("baseline fitting may use FIT variants only")
    if {item.cell.trajectory.role for item in validation_variants} != {"validation"}:
        raise ValueError("baseline selection may use validation variants only")
    fit_cases = {item.cell.trajectory.case for item in fit_variants}
    validation_cases = {item.cell.trajectory.case for item in validation_variants}
    if len(fit_cases) != 1 or fit_cases != validation_cases:
        raise ValueError("baseline selection requires one matching case")
    case = next(iter(fit_cases))
    return case


def _validate_fit_scaler_identity(scalers: FaultScalers, case: str) -> None:
    source_items = tuple(scalers.fit_source_sha256)
    sources = dict(source_items)
    if not sources:
        raise ValueError("baseline scalers have no FIT source identities")
    if len(sources) != len(source_items):
        raise ValueError("baseline scalers contain duplicate FIT source identities")
    for identity, digest in sources.items():
        parts = identity.split(":")
        if len(parts) < 2 or parts[0] != case or parts[1] != "fit":
            raise ValueError("baseline scalers are not FIT-only for the requested case")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("baseline scaler FIT source SHA-256 is malformed")


def _selection_score(
    prediction: np.ndarray,
    target: np.ndarray,
    references: Sequence[tuple[int, int]],
    variants: Sequence[FaultVariant],
) -> float:
    rows = []
    for row_index, (variant_index, _) in enumerate(references):
        variant = variants[variant_index]
        channel = FAULT_CHANNELS.index(variant.cell.fault_channel)
        rows.append(
            {
                "case": variant.cell.trajectory.case,
                "family": variant.cell.family,
                "fault_channel": variant.cell.fault_channel,
                "error": abs(
                    float(prediction[row_index, channel])
                    - float(target[row_index, channel])
                ),
            }
        )
    cells = pd.DataFrame(rows).groupby(
        ["case", "family", "fault_channel"], as_index=False
    )["error"].mean()
    return float(cells["error"].mean())


@dataclass(frozen=True)
class DirectH8FeatureSpec:
    # Match the maximum causal prefix exposed before an H8 endpoint in RSSM training.
    history: int = 40
    horizon: int = 8

    def __post_init__(self) -> None:
        if self.history < self.horizon:
            raise ValueError("direct-H8 baseline history must cover the scored horizon")
        if self.horizon != 8:
            raise ValueError("the direct baseline is frozen to H8")

    @property
    def contract(self) -> dict:
        return {
            "schema": "boptest-causal-direct-h8-features-v2",
            "history": self.history,
            "horizon": self.horizon,
            "history_fields": [
                "corrupted_observation_or_zero",
                "availability",
                "log1p_age",
                "previous_action",
                "known_context",
            ],
            "future_fields": [
                "candidate_action_t_through_t_plus_7",
                "known_context_t_plus_1_through_t_plus_8",
            ],
            "future_observations": False,
        }


def direct_h8_features(
    variant: FaultVariant,
    anchor: int,
    scalers: FaultScalers,
    spec: DirectH8FeatureSpec | None = None,
) -> np.ndarray:
    """Causal direct-H8 features with the same action/context information as RSSM."""
    spec = DirectH8FeatureSpec() if spec is None else spec
    history_start = anchor - spec.history + 1
    if history_start < 1:
        raise ValueError("direct-H8 anchor lacks a complete previous-action history")
    if anchor + spec.horizon >= len(variant.clean_observations):
        raise ValueError("direct-H8 anchor leaves a whole trajectory")

    observation_history = scalers.observation.transform(
        variant.corrupted_observations[history_start : anchor + 1]
    )
    availability_history = variant.availability[history_start : anchor + 1].astype(float)
    age_history = np.log1p(variant.age[history_start : anchor + 1])
    observation_history = np.where(
        availability_history.astype(bool), observation_history, 0.0
    )
    previous_actions = scalers.action.transform(
        variant.actions[anchor - spec.history : anchor]
    )
    context_history = scalers.context.transform(
        variant.contexts[history_start : anchor + 1]
    )
    future_actions = scalers.action.transform(
        variant.actions[anchor : anchor + spec.horizon]
    )
    future_context = scalers.context.transform(
        variant.contexts[anchor + 1 : anchor + spec.horizon + 1]
    )
    features = np.concatenate(
        [
            observation_history.reshape(-1),
            availability_history.reshape(-1),
            age_history.reshape(-1),
            previous_actions.reshape(-1),
            context_history.reshape(-1),
            future_actions.reshape(-1),
            future_context.reshape(-1),
        ]
    )
    if not np.isfinite(features).all():
        raise ValueError("direct-H8 features contain a non-finite value")
    return features


def direct_h8_dataset(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    spec: DirectH8FeatureSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    spec = DirectH8FeatureSpec() if spec is None else spec
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    references: list[tuple[int, int]] = []
    for variant_index, variant in enumerate(variants):
        for anchor in variant.cell.anchors:
            features.append(direct_h8_features(variant, anchor, scalers, spec))
            targets.append(
                scalers.observation.transform(
                    variant.clean_observations[anchor + spec.horizon][None]
                )[0]
            )
            references.append((variant_index, anchor))
    if not features:
        raise ValueError("direct-H8 dataset is empty")
    x = np.stack(features)
    y = np.stack(targets)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("direct-H8 dataset contains non-finite values")
    return x, y, references


def fit_direct_h8_ridge(
    fit_variants: Sequence[FaultVariant],
    validation_variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    spec: DirectH8FeatureSpec | None = None,
) -> tuple[Ridge, pd.DataFrame, BaselineSelectionReceipt]:
    """Select regularization on whole validation trajectories, then refit on FIT only."""
    spec = DirectH8FeatureSpec() if spec is None else spec
    case = _validate_selection_roles(fit_variants, validation_variants)
    _validate_fit_scaler_identity(scalers, case)
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alpha grid must contain positive finite values")

    fit_x, fit_y, fit_references = direct_h8_dataset(
        fit_variants, scalers, spec
    )
    fit_groups = [
        (
            fit_variants[variant_index].cell.fault_channel,
            fit_variants[variant_index].cell.family,
        )
        for variant_index, _ in fit_references
    ]
    group_counts = Counter(fit_groups)
    sample_weight = np.asarray(
        [1.0 / group_counts[group] for group in fit_groups], dtype=float
    )
    sample_weight *= len(sample_weight) / sample_weight.sum()
    validation_x, validation_y, validation_references = direct_h8_dataset(
        validation_variants, scalers, spec
    )
    metadata = pd.DataFrame(
        [
            {
                "variant_index": variant_index,
                "anchor": anchor,
                "case": validation_variants[variant_index].cell.trajectory.case,
                "family": validation_variants[variant_index].cell.family,
                "fault_channel": validation_variants[variant_index].cell.fault_channel,
            }
            for variant_index, anchor in validation_references
        ]
    )
    scores = []
    models = {}
    for alpha in sorted(set(float(value) for value in alphas)):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(fit_x, fit_y, sample_weight=sample_weight)
        prediction = model.predict(validation_x)
        own_error = np.empty(len(prediction), dtype=float)
        for row_index, variant_index in enumerate(metadata["variant_index"]):
            channel = FAULT_CHANNELS.index(
                validation_variants[int(variant_index)].cell.fault_channel
            )
            own_error[row_index] = abs(
                prediction[row_index, channel] - validation_y[row_index, channel]
            )
        score_frame = metadata.assign(error=own_error)
        equal_cells = score_frame.groupby(
            ["case", "family", "fault_channel"], as_index=False
        )["error"].mean()
        score = float(equal_cells["error"].mean())
        scores.append({"alpha": alpha, "validation_score": score})
        models[alpha] = model
    score_table = pd.DataFrame(scores).sort_values(
        ["validation_score", "alpha"], kind="stable"
    ).reset_index(drop=True)
    score_table["selected"] = False
    score_table.loc[0, "selected"] = True
    selected_alpha = float(score_table.loc[0, "alpha"])
    selected_model = models[selected_alpha]
    schedule_payload = [
        {
            "cell_id": fit_variants[variant_index].cell.cell_id,
            "anchor": anchor,
        }
        for variant_index, anchor in fit_references
    ]
    receipt = BaselineSelectionReceipt(
        schema="boptest-baseline-selection-receipt-v1",
        baseline="direct_h8_ridge",
        case=case,
        model_seed=0,
        fit_role="fit",
        validation_role="validation",
        fit_variant_identity_sha256=_variant_identity_sha256(fit_variants),
        validation_variant_identity_sha256=_variant_identity_sha256(
            validation_variants
        ),
        fit_scalers_sha256=_scalers_sha256(scalers),
        feature_contract_sha256=_canonical_sha256(spec.contract),
        training_config_sha256=_canonical_sha256(
            {
                "estimator": "sklearn.linear_model.Ridge",
                "estimator_defaults_except_alpha": {
                    key: value
                    for key, value in Ridge(fit_intercept=True)
                    .get_params(deep=False)
                    .items()
                    if key != "alpha"
                },
                "alpha_grid": sorted(set(float(value) for value in alphas)),
                "target": "FIT-standardized clean H8 endpoint",
                "weighting": "equal sensor/family stratum total weight",
                "sklearn_version": sklearn.__version__,
            }
        ),
        selection_metric="equal_mean_case_family_fault_channel_standardized_H8_MAE",
        candidate_grid=tuple(sorted(set(float(value) for value in alphas))),
        selected_candidate=selected_alpha,
        selected_validation_score=float(score_table.loc[0, "validation_score"]),
        score_table_sha256=_frame_sha256(score_table),
        training_updates=1,
        batch_size=len(fit_x),
        schedule_sha256=_canonical_sha256(schedule_payload),
        selected_model_state_sha256=_ridge_state_sha256(selected_model),
        producer_code_sha256=baseline_producer_code_manifest()["sha256"],
        runtime_fingerprint=numerical_runtime_fingerprint(
            "cpu", include_sklearn=True
        ),
    )
    return selected_model, score_table, receipt


def evaluate_direct_h8_ridge(
    model: Ridge,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    receipt: BaselineSelectionReceipt,
    *,
    role: str,
    spec: DirectH8FeatureSpec | None = None,
) -> pd.DataFrame:
    _validate_baseline_producer_sha256(receipt.producer_code_sha256)
    validate_numerical_runtime_fingerprint(
        receipt.runtime_fingerprint, include_sklearn=True
    )
    spec = DirectH8FeatureSpec() if spec is None else spec
    if not variants or {variant.cell.trajectory.role for variant in variants} != {role}:
        raise ValueError("Ridge evaluation variants differ from the requested role")
    cases = {variant.cell.trajectory.case for variant in variants}
    if cases != {receipt.case} or receipt.baseline != "direct_h8_ridge":
        raise ValueError("direct Ridge evaluation identity differs from its receipt")
    _validate_fit_scaler_identity(scalers, receipt.case)
    if receipt.fit_scalers_sha256 != _scalers_sha256(scalers):
        raise ValueError("direct Ridge scalers differ from its selection receipt")
    if receipt.selected_model_state_sha256 != _ridge_state_sha256(model):
        raise ValueError("direct Ridge model differs from its selection receipt")
    if receipt.feature_contract_sha256 != _canonical_sha256(spec.contract):
        raise ValueError("direct Ridge feature contract differs from its receipt")
    x, target, references = direct_h8_dataset(variants, scalers, spec)
    prediction = model.predict(x)
    alternate_x = _direct_alternate_features(
        variants, references, scalers, spec
    )
    alternate_prediction = model.predict(alternate_x)
    return _baseline_result_frame(
        prediction,
        alternate_prediction,
        target,
        references,
        variants,
        scalers,
        role=role,
        arm="direct_h8_ridge",
        model_seed=0,
        update=0,
        horizon=spec.horizon,
    )


@dataclass(frozen=True)
class ARXFeatureSpec:
    history: int = 8
    horizon: int = 8

    def __post_init__(self) -> None:
        if self.history < 8:
            raise ValueError("ARX requires at least eight observation/action lags")
        if self.horizon != 8:
            raise ValueError("ARX evaluation is frozen to H8")

    @property
    def contract(self) -> dict:
        return {
            "schema": "boptest-causal-standardized-arx-features-v1",
            "history": self.history,
            "horizon": self.horizon,
            "history_fields": [
                "corrupted_observation_or_zero",
                "availability",
                "log1p_age",
                "previous_action",
            ],
            "current_fields": ["candidate_action", "context_t", "context_t_plus_1"],
            "future_observations": False,
            "rollout": "iterative_predictions_replace_unobserved_future_measurements",
        }


def _arx_features_from_standardized(
    observations: np.ndarray,
    availability: np.ndarray,
    age: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    source: int,
    scalers: FaultScalers,
    spec: ARXFeatureSpec,
) -> np.ndarray:
    history_start = source - spec.history + 1
    if history_start < 0 or source + 1 >= len(observations):
        raise ValueError("ARX source lacks causal history or a one-step target")
    observation_history = observations[history_start : source + 1]
    availability_history = availability[history_start : source + 1].astype(float)
    age_history = np.log1p(age[history_start : source + 1])
    observation_history = np.where(
        availability_history.astype(bool), observation_history, 0.0
    )
    previous_actions = scalers.action.transform(
        actions[source - spec.history : source]
    )
    current_action = scalers.action.transform(actions[source : source + 1])
    known_context = scalers.context.transform(contexts[source : source + 2])
    features = np.concatenate(
        [
            observation_history.reshape(-1),
            availability_history.reshape(-1),
            age_history.reshape(-1),
            previous_actions.reshape(-1),
            current_action.reshape(-1),
            known_context.reshape(-1),
        ]
    )
    if not np.isfinite(features).all():
        raise ValueError("ARX features contain a non-finite value")
    return features


def arx_one_step_features(
    variant: FaultVariant,
    source: int,
    scalers: FaultScalers,
    spec: ARXFeatureSpec | None = None,
) -> np.ndarray:
    """Return standardized causal features for predicting clean ``source + 1``."""
    spec = ARXFeatureSpec() if spec is None else spec
    observations = np.zeros_like(variant.corrupted_observations, dtype=float)
    observations[: source + 1] = scalers.observation.transform(
        variant.corrupted_observations[: source + 1]
    )
    return _arx_features_from_standardized(
        observations,
        variant.availability,
        variant.age,
        variant.actions,
        variant.contexts,
        source,
        scalers,
        spec,
    )


def arx_one_step_dataset(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    spec: ARXFeatureSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Build one-step examples only inside each predeclared fault interval."""
    spec = ARXFeatureSpec() if spec is None else spec
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    references: list[tuple[int, int]] = []
    for variant_index, variant in enumerate(variants):
        first_source = max(spec.history - 1, variant.cell.onset)
        last_source = variant.cell.stop - 2
        if first_source > last_source:
            raise ValueError("ARX fault interval has no valid one-step examples")
        for source in range(first_source, last_source + 1):
            features.append(arx_one_step_features(variant, source, scalers, spec))
            targets.append(
                scalers.observation.transform(
                    variant.clean_observations[source + 1 : source + 2]
                )[0]
            )
            references.append((variant_index, source))
    if not features:
        raise ValueError("ARX one-step dataset is empty")
    x = np.stack(features)
    y = np.stack(targets)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ARX one-step dataset contains a non-finite value")
    return x, y, references


def _alternate_action_block(
    variant: FaultVariant, anchor: int, horizon: int
) -> np.ndarray:
    realized = variant.actions[anchor : anchor + horizon]
    if len(realized) != horizon:
        raise ValueError("realized H8 action block leaves a whole trajectory")
    for shift in range(horizon, len(variant.actions) - anchor - horizon + 1, horizon):
        candidate = variant.actions[anchor + shift : anchor + shift + horizon]
        if not np.array_equal(candidate, realized):
            return candidate
    raise ValueError("trajectory has no nonrepeating alternate H8 action block")


def iterative_arx_h8_prediction(
    model: Ridge,
    variant: FaultVariant,
    anchor: int,
    scalers: FaultScalers,
    spec: ARXFeatureSpec | None = None,
    *,
    candidate_actions: np.ndarray | None = None,
) -> np.ndarray:
    """Roll ARX to H8 without consuming observations after ``anchor``."""
    spec = ARXFeatureSpec() if spec is None else spec
    if anchor < spec.history - 1 or anchor + spec.horizon >= len(
        variant.clean_observations
    ):
        raise ValueError("ARX H8 anchor lacks history or leaves the trajectory")
    actions = np.array(variant.actions, copy=True)
    if candidate_actions is not None:
        candidate_actions = np.asarray(candidate_actions, dtype=float)
        expected_shape = (spec.horizon, variant.actions.shape[1])
        if candidate_actions.shape != expected_shape:
            raise ValueError("candidate ARX action block has the wrong shape")
        actions[anchor : anchor + spec.horizon] = candidate_actions
    observations = np.zeros_like(variant.corrupted_observations, dtype=float)
    observations[: anchor + 1] = scalers.observation.transform(
        variant.corrupted_observations[: anchor + 1]
    )
    availability = np.zeros_like(variant.availability, dtype=bool)
    availability[: anchor + 1] = variant.availability[: anchor + 1]
    age = np.zeros_like(variant.age, dtype=float)
    age[: anchor + 1] = variant.age[: anchor + 1]
    prediction = None
    for source in range(anchor, anchor + spec.horizon):
        features = _arx_features_from_standardized(
            observations,
            availability,
            age,
            actions,
            variant.contexts,
            source,
            scalers,
            spec,
        )
        prediction = np.asarray(model.predict(features[None])[0], dtype=float)
        if prediction.shape != (variant.clean_observations.shape[1],) or not np.isfinite(
            prediction
        ).all():
            raise ValueError("ARX rollout produced an invalid prediction")
        observations[source + 1] = prediction
        availability[source + 1] = True
        age[source + 1] = 0.0
    if prediction is None:
        raise AssertionError("ARX H8 rollout performed no step")
    return prediction


def _arx_h8_predictions(
    model: Ridge,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    spec: ARXFeatureSpec,
    *,
    alternate: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    references: list[tuple[int, int]] = []
    for variant_index, variant in enumerate(variants):
        for anchor in variant.cell.anchors:
            candidate = (
                _alternate_action_block(variant, anchor, spec.horizon)
                if alternate
                else None
            )
            predictions.append(
                iterative_arx_h8_prediction(
                    model,
                    variant,
                    anchor,
                    scalers,
                    spec,
                    candidate_actions=candidate,
                )
            )
            targets.append(
                scalers.observation.transform(
                    variant.clean_observations[
                        anchor + spec.horizon : anchor + spec.horizon + 1
                    ]
                )[0]
            )
            references.append((variant_index, anchor))
    if not predictions:
        raise ValueError("ARX H8 evaluation dataset is empty")
    return np.stack(predictions), np.stack(targets), references


def _ridge_state_sha256(model: Ridge) -> str:
    payload = {
        "alpha": float(model.alpha),
        "fit_intercept": bool(model.fit_intercept),
        "coef_shape": list(np.asarray(model.coef_).shape),
        "coef_sha256": hashlib.sha256(
            np.asarray(model.coef_, dtype=np.float64).tobytes()
        ).hexdigest(),
        "intercept_sha256": hashlib.sha256(
            np.asarray(model.intercept_, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    return _canonical_sha256(payload)


def fit_arx_ridge(
    fit_variants: Sequence[FaultVariant],
    validation_variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    spec: ARXFeatureSpec | None = None,
) -> tuple[Ridge, pd.DataFrame, BaselineSelectionReceipt]:
    """Fit on FIT one-step targets and select alpha by validation H8 error."""
    spec = ARXFeatureSpec() if spec is None else spec
    case = _validate_selection_roles(fit_variants, validation_variants)
    _validate_fit_scaler_identity(scalers, case)
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("ARX alpha grid must contain positive finite values")
    fit_x, fit_y, fit_references = arx_one_step_dataset(
        fit_variants, scalers, spec
    )
    fit_groups = [
        (
            fit_variants[variant_index].cell.fault_channel,
            fit_variants[variant_index].cell.family,
        )
        for variant_index, _ in fit_references
    ]
    group_counts = Counter(fit_groups)
    sample_weight = np.asarray(
        [1.0 / group_counts[group] for group in fit_groups], dtype=float
    )
    sample_weight *= len(sample_weight) / sample_weight.sum()
    models: dict[float, Ridge] = {}
    scores: list[dict] = []
    for alpha in sorted(set(float(value) for value in alphas)):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(fit_x, fit_y, sample_weight=sample_weight)
        prediction, target, references = _arx_h8_predictions(
            model, validation_variants, scalers, spec
        )
        score = _selection_score(
            prediction, target, references, validation_variants
        )
        models[alpha] = model
        scores.append({"alpha": alpha, "validation_score": score})
    score_table = pd.DataFrame(scores).sort_values(
        ["validation_score", "alpha"], kind="stable"
    ).reset_index(drop=True)
    score_table["selected"] = False
    score_table.loc[0, "selected"] = True
    selected_alpha = float(score_table.loc[0, "alpha"])
    selected_model = models[selected_alpha]
    schedule_payload = [
        {
            "cell_id": fit_variants[variant_index].cell.cell_id,
            "source": source,
        }
        for variant_index, source in fit_references
    ]
    receipt = BaselineSelectionReceipt(
        schema="boptest-baseline-selection-receipt-v1",
        baseline="ridge_arx",
        case=case,
        model_seed=0,
        fit_role="fit",
        validation_role="validation",
        fit_variant_identity_sha256=_variant_identity_sha256(fit_variants),
        validation_variant_identity_sha256=_variant_identity_sha256(
            validation_variants
        ),
        fit_scalers_sha256=_scalers_sha256(scalers),
        feature_contract_sha256=_canonical_sha256(spec.contract),
        training_config_sha256=_canonical_sha256(
            {
                "estimator": "sklearn.linear_model.Ridge",
                "estimator_defaults_except_alpha": {
                    key: value
                    for key, value in Ridge(fit_intercept=True)
                    .get_params(deep=False)
                    .items()
                    if key != "alpha"
                },
                "alpha_grid": sorted(set(float(value) for value in alphas)),
                "target": "FIT-standardized clean one-step observation",
                "weighting": "equal sensor/family stratum total weight",
                "sklearn_version": sklearn.__version__,
            }
        ),
        selection_metric="equal_mean_case_family_fault_channel_standardized_H8_MAE",
        candidate_grid=tuple(sorted(set(float(value) for value in alphas))),
        selected_candidate=selected_alpha,
        selected_validation_score=float(score_table.loc[0, "validation_score"]),
        score_table_sha256=_frame_sha256(score_table),
        training_updates=1,
        batch_size=len(fit_x),
        schedule_sha256=_canonical_sha256(schedule_payload),
        selected_model_state_sha256=_ridge_state_sha256(selected_model),
        producer_code_sha256=baseline_producer_code_manifest()["sha256"],
        runtime_fingerprint=numerical_runtime_fingerprint(
            "cpu", include_sklearn=True
        ),
    )
    return selected_model, score_table, receipt


class DirectH8GRU(nn.Module):
    """Small deterministic GRU over exactly the direct-Ridge information set."""

    def __init__(
        self,
        *,
        observation_dim: int = 4,
        action_dim: int = 1,
        context_dim: int = 5,
        hidden_dim: int = 64,
        spec: DirectH8FeatureSpec | None = None,
    ) -> None:
        super().__init__()
        self.spec = DirectH8FeatureSpec() if spec is None else spec
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        history_input_dim = 3 * observation_dim + action_dim + context_dim
        future_input_dim = (
            self.spec.horizon * action_dim
            + self.spec.horizon * context_dim
        )
        self.history_gru = nn.GRU(
            input_size=history_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(hidden_dim + future_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, observation_dim),
        )

    @property
    def expected_feature_dim(self) -> int:
        return (
            self.spec.history * self.observation_dim * 3
            + self.spec.history * self.action_dim
            + self.spec.history * self.context_dim
            + self.spec.horizon * self.action_dim
            + self.spec.horizon * self.context_dim
        )

    @property
    def feature_contract(self) -> dict:
        return {
            "schema": "boptest-direct-h8-gru-features-v2",
            "base_feature_schema": "direct_h8_features",
            "history": self.spec.history,
            "horizon": self.spec.horizon,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "future_observations": False,
            "information_equality": "byte-identical input vector to direct-H8 Ridge",
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.expected_feature_dim:
            raise ValueError("direct-H8 GRU feature tensor has the wrong shape")
        batch = features.shape[0]
        cursor = 0

        def take(width: int) -> torch.Tensor:
            nonlocal cursor
            value = features[:, cursor : cursor + width]
            cursor += width
            return value

        observation = take(self.spec.history * self.observation_dim).reshape(
            batch, self.spec.history, self.observation_dim
        )
        availability = take(self.spec.history * self.observation_dim).reshape(
            batch, self.spec.history, self.observation_dim
        )
        age = take(self.spec.history * self.observation_dim).reshape(
            batch, self.spec.history, self.observation_dim
        )
        previous_action = take(self.spec.history * self.action_dim).reshape(
            batch, self.spec.history, self.action_dim
        )
        context_history = take(self.spec.history * self.context_dim).reshape(
            batch, self.spec.history, self.context_dim
        )
        future_action = take(self.spec.horizon * self.action_dim)
        future_context = take(self.spec.horizon * self.context_dim)
        if cursor != features.shape[1]:
            raise AssertionError("direct-H8 GRU feature layout did not consume its input")
        history = torch.cat(
            [observation, availability, age, previous_action, context_history], dim=-1
        )
        _, hidden = self.history_gru(history)
        endpoint_input = torch.cat(
            [hidden[-1], future_action, future_context], dim=-1
        )
        return self.endpoint_head(endpoint_input)


@dataclass(frozen=True)
class GRUFitResult:
    model: DirectH8GRU
    score_table: pd.DataFrame
    training_log: pd.DataFrame
    receipt: BaselineSelectionReceipt


def _stable_seed(*parts: object) -> int:
    content = ":".join(str(part) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "little") & 0x7FFFFFFF


def _make_gru_schedule(
    variants: Sequence[FaultVariant],
    references: Sequence[tuple[int, int]],
    config: StudyConfig,
    *,
    case: str,
    model_seed: int,
) -> tuple[np.ndarray, str]:
    groups: dict[tuple[str, str], list[int]] = {}
    for row_index, (variant_index, _) in enumerate(references):
        variant = variants[variant_index]
        groups.setdefault(
            (variant.cell.fault_channel, variant.cell.family), []
        ).append(row_index)
    if not groups:
        raise ValueError("GRU schedule has no FIT groups")
    group_keys = tuple(sorted(groups))
    rng = np.random.Generator(
        np.random.PCG64(
            _stable_seed(config.schedule_seed, case, model_seed, "direct_h8_gru")
        )
    )
    queue: list[tuple[str, str]] = []
    schedule = np.empty((config.updates, config.gru_batch_size), dtype=np.int64)
    semantic_rows = []
    for update_index in range(config.updates):
        for batch_index in range(config.gru_batch_size):
            if not queue:
                queue = [group_keys[index] for index in rng.permutation(len(group_keys))]
            group = queue.pop()
            candidates = groups[group]
            row_index = candidates[int(rng.integers(len(candidates)))]
            schedule[update_index, batch_index] = row_index
            variant_index, anchor = references[row_index]
            semantic_rows.append(
                {
                    "update": update_index + 1,
                    "batch_index": batch_index,
                    "cell_id": variants[variant_index].cell.cell_id,
                    "anchor": anchor,
                }
            )
    return schedule, _canonical_sha256(semantic_rows)


def _predict_gru(
    model: DirectH8GRU,
    features: np.ndarray,
    *,
    device: torch.device | str,
    batch_size: int = 4096,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("GRU prediction batch size must be positive")
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(batch).cpu().numpy())
    if not outputs:
        raise ValueError("GRU prediction dataset is empty")
    prediction = np.concatenate(outputs)
    if not np.isfinite(prediction).all():
        raise ValueError("GRU produced a non-finite prediction")
    return prediction


def fit_direct_h8_gru(
    fit_variants: Sequence[FaultVariant],
    validation_variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    config: StudyConfig,
    *,
    model_seed: int,
    spec: DirectH8FeatureSpec | None = None,
    device: torch.device | str = "cpu",
) -> GRUFitResult:
    """Train on FIT only and select one configured checkpoint on validation."""
    spec = DirectH8FeatureSpec() if spec is None else spec
    case = _validate_selection_roles(fit_variants, validation_variants)
    _validate_fit_scaler_identity(scalers, case)
    if model_seed not in config.confirmatory_seeds:
        raise ValueError("GRU model seed is outside the frozen three/five-seed set")
    if spec.horizon != config.direct_horizon:
        raise ValueError("GRU feature and study horizons differ")
    fit_x, fit_y, fit_references = direct_h8_dataset(
        fit_variants, scalers, spec
    )
    validation_x, validation_y, validation_references = direct_h8_dataset(
        validation_variants, scalers, spec
    )
    schedule, schedule_sha256 = _make_gru_schedule(
        fit_variants,
        fit_references,
        config,
        case=case,
        model_seed=model_seed,
    )
    torch.manual_seed(model_seed)
    model = DirectH8GRU(
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        spec=spec,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
    score_rows: list[dict] = []
    log_rows: list[dict] = []
    try:
        for update in range(1, config.updates + 1):
            row_indices = schedule[update - 1]
            batch_x = torch.as_tensor(
                fit_x[row_indices], dtype=torch.float32, device=device
            )
            batch_y = torch.as_tensor(
                fit_y[row_indices], dtype=torch.float32, device=device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, batch_y, beta=config.direct_horizon_beta
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("GRU direct-H8 loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("GRU direct-H8 gradient is non-finite")
            optimizer.step()
            log_rows.append(
                {
                    "update": update,
                    "fit_smooth_l1": float(loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                }
            )
            if update in config.validation_checkpoints:
                checkpoint_states[update] = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                validation_prediction = _predict_gru(
                    model, validation_x, device=device
                )
                score_rows.append(
                    {
                        "update": update,
                        "validation_score": _selection_score(
                            validation_prediction,
                            validation_y,
                            validation_references,
                            validation_variants,
                        ),
                        "model_state_sha256": _tensor_state_sha256(
                            checkpoint_states[update]
                        ),
                    }
                )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    if set(checkpoint_states) != set(config.validation_checkpoints):
        raise AssertionError("GRU did not materialize the frozen checkpoint grid")
    score_table = pd.DataFrame(score_rows).sort_values(
        ["validation_score", "update"], kind="stable"
    ).reset_index(drop=True)
    score_table["selected"] = False
    score_table.loc[0, "selected"] = True
    selected_update = int(score_table.loc[0, "update"])
    model.load_state_dict(checkpoint_states[selected_update], strict=True)
    training_log = pd.DataFrame(log_rows)
    receipt = BaselineSelectionReceipt(
        schema="boptest-baseline-selection-receipt-v1",
        baseline="deterministic_gru",
        case=case,
        model_seed=model_seed,
        fit_role="fit",
        validation_role="validation",
        fit_variant_identity_sha256=_variant_identity_sha256(fit_variants),
        validation_variant_identity_sha256=_variant_identity_sha256(
            validation_variants
        ),
        fit_scalers_sha256=_scalers_sha256(scalers),
        feature_contract_sha256=_canonical_sha256(model.feature_contract),
        training_config_sha256=_canonical_sha256(
            {
                "study_config": config.to_dict(),
                "optimizer": "torch.optim.Adam",
                "optimizer_parameters": {
                    "learning_rate": config.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                },
                "loss": "SmoothL1_mean",
                "direct_horizon_beta": config.direct_horizon_beta,
                "gradient_clip": config.gradient_clip,
                "selection_checkpoints": list(config.validation_checkpoints),
                "device": str(device),
                "torch_version": torch.__version__,
            }
        ),
        selection_metric="equal_mean_case_family_fault_channel_standardized_H8_MAE",
        candidate_grid=tuple(config.validation_checkpoints),
        selected_candidate=selected_update,
        selected_validation_score=float(score_table.loc[0, "validation_score"]),
        score_table_sha256=_frame_sha256(score_table),
        training_updates=config.updates,
        batch_size=config.gru_batch_size,
        schedule_sha256=schedule_sha256,
        selected_model_state_sha256=_tensor_state_sha256(model.state_dict()),
        producer_code_sha256=baseline_producer_code_manifest()["sha256"],
        runtime_fingerprint=numerical_runtime_fingerprint(
            device, include_sklearn=True
        ),
    )
    return GRUFitResult(model, score_table, training_log, receipt)


def _direct_alternate_features(
    variants: Sequence[FaultVariant],
    references: Sequence[tuple[int, int]],
    scalers: FaultScalers,
    spec: DirectH8FeatureSpec,
) -> np.ndarray:
    features = []
    for variant_index, anchor in references:
        variant = variants[variant_index]
        actions = np.array(variant.actions, copy=True)
        actions[anchor : anchor + spec.horizon] = _alternate_action_block(
            variant, anchor, spec.horizon
        )
        features.append(
            direct_h8_features(replace(variant, actions=actions), anchor, scalers, spec)
        )
    if not features:
        raise ValueError("alternate direct-H8 feature dataset is empty")
    return np.stack(features)


def _baseline_result_frame(
    prediction: np.ndarray,
    alternate_prediction: np.ndarray,
    target: np.ndarray,
    references: Sequence[tuple[int, int]],
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    *,
    role: str,
    arm: str,
    model_seed: int,
    update: int,
    horizon: int,
) -> pd.DataFrame:
    if prediction.shape != target.shape or alternate_prediction.shape != target.shape:
        raise ValueError("baseline prediction and target shapes differ")
    if len(prediction) != len(references):
        raise ValueError("baseline predictions and references differ")
    observation_mean = np.asarray(scalers.observation.mean)
    observation_scale = np.asarray(scalers.observation.scale)
    rows = []
    for row_index, (variant_index, anchor) in enumerate(references):
        variant = variants[variant_index]
        channel = FAULT_CHANNELS.index(variant.cell.fault_channel)
        available = np.flatnonzero(variant.availability[: anchor + 1, channel])
        if not len(available):
            raise ValueError("persistence has no available history")
        persistence_raw = float(
            variant.corrupted_observations[available[-1], channel]
        )
        persistence_standardized = (
            persistence_raw - observation_mean[channel]
        ) / observation_scale[channel]
        target_raw = np.asarray(
            variant.clean_observations[anchor + horizon], dtype=float
        )
        expected_target = (target_raw - observation_mean) / observation_scale
        if not np.allclose(
            target[row_index], expected_target, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("baseline target is not the exact clean H8 endpoint")
        prediction_raw = (
            prediction[row_index] * observation_scale + observation_mean
        )
        alternate_raw = (
            alternate_prediction[row_index] * observation_scale + observation_mean
        )
        row = {
            "case": variant.cell.trajectory.case,
            "role": role,
            "trajectory_day": variant.cell.trajectory.day,
            "trajectory_seed": variant.cell.trajectory.trajectory_seed,
            "model_seed": model_seed,
            "arm": arm,
            "update": update,
            "cell_id": variant.cell.cell_id,
            "fault_channel": variant.cell.fault_channel,
            "fault_channel_index": channel,
            "family": variant.cell.family,
            "sign": variant.cell.sign,
            "severity": variant.cell.severity,
            "severity_unit": variant.cell.severity_unit,
            "onset": variant.cell.onset,
            "anchor": anchor,
            "horizon": horizon,
            "target_raw": float(target_raw[channel]),
            "prediction_raw": float(prediction_raw[channel]),
            "standardized_abs_error": float(
                abs(prediction[row_index, channel] - target[row_index, channel])
            ),
            "alternate_action_prediction_raw": float(alternate_raw[channel]),
            "alternate_action_standardized_abs_error": float(
                abs(
                    alternate_prediction[row_index, channel]
                    - target[row_index, channel]
                )
            ),
            "action_prediction_change_standardized": float(
                abs(
                    prediction[row_index, channel]
                    - alternate_prediction[row_index, channel]
                )
            ),
            "persistence_prediction_raw": persistence_raw,
            "persistence_standardized_abs_error": float(
                abs(persistence_standardized - target[row_index, channel])
            ),
            # Reliability outputs are not defined for these baselines.
            "source_reliability": -1.0,
            "source_healthy_probability": -1.0,
        }
        for observation_index in range(target.shape[1]):
            row[f"target_standardized_{observation_index}"] = float(
                target[row_index, observation_index]
            )
            row[f"prediction_standardized_{observation_index}"] = float(
                prediction[row_index, observation_index]
            )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        [
            "trajectory_day",
            "fault_channel",
            "family",
            "sign",
            "severity",
            "onset",
            "anchor",
        ],
        kind="stable",
    ).reset_index(drop=True)
    required = set(BASELINE_RESULT_COLUMNS) | {
        f"{prefix}_standardized_{index}"
        for prefix in ("target", "prediction")
        for index in range(target.shape[1])
    }
    if set(frame.columns) != required:
        raise AssertionError("baseline result does not match the evaluator row schema")
    if not len(frame) or not np.isfinite(
        frame.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("baseline evaluation produced invalid results")
    return frame


def evaluate_arx_h8(
    model: Ridge,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    receipt: BaselineSelectionReceipt,
    *,
    role: str,
    spec: ARXFeatureSpec | None = None,
) -> pd.DataFrame:
    _validate_baseline_producer_sha256(receipt.producer_code_sha256)
    validate_numerical_runtime_fingerprint(
        receipt.runtime_fingerprint, include_sklearn=True
    )
    spec = ARXFeatureSpec() if spec is None else spec
    if not variants or {variant.cell.trajectory.role for variant in variants} != {role}:
        raise ValueError("ARX evaluation variants differ from the requested role")
    cases = {variant.cell.trajectory.case for variant in variants}
    if cases != {receipt.case} or receipt.baseline != "ridge_arx":
        raise ValueError("ARX evaluation identity differs from its selection receipt")
    _validate_fit_scaler_identity(scalers, receipt.case)
    if receipt.fit_scalers_sha256 != _scalers_sha256(scalers):
        raise ValueError("ARX scalers differ from its selection receipt")
    if receipt.selected_model_state_sha256 != _ridge_state_sha256(model):
        raise ValueError("ARX model differs from its selection receipt")
    if receipt.feature_contract_sha256 != _canonical_sha256(spec.contract):
        raise ValueError("ARX feature contract differs from its selection receipt")
    prediction, target, references = _arx_h8_predictions(
        model, variants, scalers, spec
    )
    alternate_prediction, alternate_target, alternate_references = (
        _arx_h8_predictions(model, variants, scalers, spec, alternate=True)
    )
    if references != alternate_references or not np.array_equal(
        target, alternate_target
    ):
        raise AssertionError("ARX alternate-action evaluation changed identities")
    return _baseline_result_frame(
        prediction,
        alternate_prediction,
        target,
        references,
        variants,
        scalers,
        role=role,
        arm="ridge_arx",
        model_seed=0,
        update=0,
        horizon=spec.horizon,
    )


def evaluate_direct_h8_gru(
    model: DirectH8GRU,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    receipt: BaselineSelectionReceipt,
    *,
    role: str,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    _validate_baseline_producer_sha256(receipt.producer_code_sha256)
    validate_numerical_runtime_fingerprint(
        receipt.runtime_fingerprint, include_sklearn=True
    )
    if not variants or {variant.cell.trajectory.role for variant in variants} != {role}:
        raise ValueError("GRU evaluation variants differ from the requested role")
    cases = {variant.cell.trajectory.case for variant in variants}
    if cases != {receipt.case} or receipt.baseline != "deterministic_gru":
        raise ValueError("GRU evaluation identity differs from its selection receipt")
    _validate_fit_scaler_identity(scalers, receipt.case)
    if receipt.fit_scalers_sha256 != _scalers_sha256(scalers):
        raise ValueError("GRU scalers differ from its selection receipt")
    if receipt.selected_model_state_sha256 != _tensor_state_sha256(model.state_dict()):
        raise ValueError("GRU model differs from its selection receipt")
    if receipt.feature_contract_sha256 != _canonical_sha256(model.feature_contract):
        raise ValueError("GRU feature contract differs from its selection receipt")
    features, target, references = direct_h8_dataset(
        variants, scalers, model.spec
    )
    alternate_features = _direct_alternate_features(
        variants, references, scalers, model.spec
    )
    model = model.to(device)
    prediction = _predict_gru(model, features, device=device)
    alternate_prediction = _predict_gru(
        model, alternate_features, device=device
    )
    return _baseline_result_frame(
        prediction,
        alternate_prediction,
        target,
        references,
        variants,
        scalers,
        role=role,
        arm="deterministic_gru",
        model_seed=receipt.model_seed,
        update=int(receipt.selected_candidate),
        horizon=model.spec.horizon,
    )
