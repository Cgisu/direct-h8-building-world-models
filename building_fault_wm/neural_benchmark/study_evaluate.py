from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .fault_data import (
    FAMILIES,
    FAULT_CHANNELS,
    FaultSpec,
    FaultScalers,
    FaultVariant,
    SequenceReference,
    fault_cell_signatures,
    materialize_rssm_batch,
)
from .protocol import CASES
from .reliability_model import ReliabilityGatedRSSM
from .runtime_provenance import validate_numerical_runtime_fingerprint
from .study_config import ArmName, StudyConfig
from .study_train import (
    core_tensor_state_sha256,
    tensor_state_sha256,
    validate_rssm_producer_code_manifest,
)


def _scale(values: np.ndarray, mean: tuple[float, ...], scale: tuple[float, ...]) -> np.ndarray:
    return (values - np.asarray(mean)) / np.asarray(scale)


def _unscale(values: np.ndarray, mean: tuple[float, ...], scale: tuple[float, ...]) -> np.ndarray:
    return values * np.asarray(scale) + np.asarray(mean)


def _last_available(
    variant: FaultVariant, anchor: int, channel_index: int
) -> float:
    available = np.flatnonzero(variant.availability[: anchor + 1, channel_index])
    if not len(available):
        raise ValueError("persistence has no available history for a scored channel")
    return float(variant.corrupted_observations[available[-1], channel_index])


def _alternate_action_block(
    variant: FaultVariant, anchor: int, config: StudyConfig
) -> np.ndarray:
    """Return the first later H8 block that differs from the realized block."""
    horizon = config.direct_horizon
    realized = variant.actions[anchor : anchor + horizon]
    if len(realized) != horizon:
        raise ValueError("realized action diagnostic block leaves the trajectory")
    for shift in range(
        config.action_diagnostic_shift,
        len(variant.actions) - anchor - horizon + 1,
        config.action_diagnostic_shift,
    ):
        candidate = variant.actions[
            anchor + shift : anchor + shift + horizon
        ]
        if not np.array_equal(candidate, realized):
            return candidate
    raise ValueError("trajectory has no nonrepeating alternate H8 action block")


@torch.no_grad()
def evaluate_model_h8(
    model: ReliabilityGatedRSSM,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    config: StudyConfig,
    *,
    arm: ArmName,
    case: str,
    model_seed: int,
    update: int,
    role: str,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """Evaluate causal H8 imagination at every frozen fault-cell anchor."""
    if not variants:
        raise ValueError("evaluation requires at least one fault variant")
    if {variant.cell.trajectory.case for variant in variants} != {case}:
        raise ValueError("evaluation variants must belong to exactly the requested case")
    if {variant.cell.trajectory.role for variant in variants} != {role}:
        raise ValueError("evaluation variants differ from the requested role")
    if model_seed not in config.confirmatory_seeds:
        raise ValueError("evaluation model seed is outside the frozen seed set")
    if update not in config.validation_checkpoints:
        raise ValueError("evaluation update is outside the frozen checkpoint grid")

    model.eval()
    arm_config = config.arm_config(arm)
    by_anchor: dict[int, list[int]] = defaultdict(list)
    for variant_index, variant in enumerate(variants):
        for anchor in variant.cell.anchors:
            if anchor <= 0 or anchor + config.direct_horizon >= len(
                variant.clean_observations
            ):
                raise ValueError("evaluation anchor leaves a whole trajectory")
            if anchor + config.direct_horizon >= variant.cell.stop:
                raise ValueError("evaluation H8 endpoint leaves the active fault")
            by_anchor[anchor].append(variant_index)

    rows: list[dict] = []
    observation_mean = np.asarray(scalers.observation.mean)
    observation_scale = np.asarray(scalers.observation.scale)
    evaluation_history = config.sequence_length - config.direct_horizon
    if evaluation_history != 40:
        raise ValueError("RSSM evaluation history differs from the frozen direct-H8 contract")
    for anchor, variant_indices in sorted(by_anchor.items()):
        if anchor < evaluation_history:
            raise ValueError("evaluation anchor lacks the frozen causal history")
        selected = [variants[index] for index in variant_indices]
        references = tuple(
            SequenceReference(index, anchor - evaluation_history)
            for index in range(len(selected))
        )
        history = materialize_rssm_batch(
            selected,
            scalers,
            references,
            length=evaluation_history,
        )
        filtered = model.filter(
            torch.as_tensor(
                history.previous_actions, dtype=torch.float32, device=device
            ),
            torch.as_tensor(
                history.corrupted_observations, dtype=torch.float32, device=device
            ),
            torch.as_tensor(history.availability, dtype=torch.bool, device=device),
            torch.as_tensor(history.age, dtype=torch.float32, device=device),
            torch.as_tensor(history.contexts, dtype=torch.float32, device=device),
            gate_mode=arm_config.gate_mode,
            sample=False,
        )
        future_actions = np.stack(
            [
                scalers.action.transform(
                    variant.actions[anchor : anchor + config.direct_horizon]
                )
                for variant in selected
            ],
            axis=1,
        )
        future_contexts = np.stack(
            [
                scalers.context.transform(
                    variant.contexts[anchor + 1 : anchor + config.direct_horizon + 1]
                )
                for variant in selected
            ],
            axis=1,
        )
        imagined = model.imagine(
            filtered.rssm.final_state,
            torch.as_tensor(future_actions, dtype=torch.float32, device=device),
            torch.as_tensor(future_contexts, dtype=torch.float32, device=device),
            sample=False,
        )
        alternate_actions = np.stack(
            [
                scalers.action.transform(
                    _alternate_action_block(variant, anchor, config)
                )
                for variant in selected
            ],
            axis=1,
        )
        alternate_imagined = model.imagine(
            filtered.rssm.final_state,
            torch.as_tensor(alternate_actions, dtype=torch.float32, device=device),
            torch.as_tensor(future_contexts, dtype=torch.float32, device=device),
            sample=False,
        )
        predicted_standardized = imagined.observation_mean[-1].cpu().numpy()
        alternate_predicted_standardized = (
            alternate_imagined.observation_mean[-1].cpu().numpy()
        )
        predicted_raw = _unscale(
            predicted_standardized,
            scalers.observation.mean,
            scalers.observation.scale,
        )
        target_raw = np.stack(
            [
                variant.clean_observations[anchor + config.direct_horizon]
                for variant in selected
            ]
        )
        target_standardized = _scale(
            target_raw,
            scalers.observation.mean,
            scalers.observation.scale,
        )
        reliability = filtered.observation_reliability[-1].cpu().numpy()
        healthy_probability = filtered.gate_health_logits[-1].softmax(dim=-1)[
            ..., model.config.healthy_class_index
        ].cpu().numpy()

        for batch_index, variant in enumerate(selected):
            channel_index = FAULT_CHANNELS.index(variant.cell.fault_channel)
            persistence_raw = _last_available(variant, anchor, channel_index)
            persistence_standardized = (
                persistence_raw - observation_mean[channel_index]
            ) / observation_scale[channel_index]
            row = {
                "case": case,
                "role": role,
                "trajectory_day": variant.cell.trajectory.day,
                "trajectory_seed": variant.cell.trajectory.trajectory_seed,
                "model_seed": model_seed,
                "arm": arm,
                "update": update,
                "cell_id": variant.cell.cell_id,
                "fault_channel": variant.cell.fault_channel,
                "fault_channel_index": channel_index,
                "family": variant.cell.family,
                "sign": variant.cell.sign,
                "severity": variant.cell.severity,
                "severity_unit": variant.cell.severity_unit,
                "onset": variant.cell.onset,
                "anchor": anchor,
                "horizon": config.direct_horizon,
                "target_raw": float(target_raw[batch_index, channel_index]),
                "prediction_raw": float(predicted_raw[batch_index, channel_index]),
                "standardized_abs_error": float(
                    abs(
                        predicted_standardized[batch_index, channel_index]
                        - target_standardized[batch_index, channel_index]
                    )
                ),
                "alternate_action_prediction_raw": float(
                    alternate_predicted_standardized[batch_index, channel_index]
                    * observation_scale[channel_index]
                    + observation_mean[channel_index]
                ),
                "alternate_action_standardized_abs_error": float(
                    abs(
                        alternate_predicted_standardized[batch_index, channel_index]
                        - target_standardized[batch_index, channel_index]
                    )
                ),
                "action_prediction_change_standardized": float(
                    abs(
                        predicted_standardized[batch_index, channel_index]
                        - alternate_predicted_standardized[batch_index, channel_index]
                    )
                ),
                "persistence_prediction_raw": persistence_raw,
                "persistence_standardized_abs_error": float(
                    abs(
                        persistence_standardized
                        - target_standardized[batch_index, channel_index]
                    )
                ),
                "source_reliability": float(
                    reliability[batch_index, channel_index]
                ),
                "source_healthy_probability": float(
                    healthy_probability[batch_index, channel_index]
                ),
            }
            for observation_index in range(model.config.observation_dim):
                row[f"target_standardized_{observation_index}"] = float(
                    target_standardized[batch_index, observation_index]
                )
                row[f"prediction_standardized_{observation_index}"] = float(
                    predicted_standardized[batch_index, observation_index]
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
    if not len(frame) or not np.isfinite(
        frame.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("evaluation produced an empty or non-finite result table")
    return frame


def load_model_checkpoint(
    path: Path,
    config: StudyConfig,
    *,
    case: str,
    model_seed: int,
    arm: ArmName,
    update: int,
    expected_checkpoint_sha256: str,
    expected_provenance: Mapping[str, object],
    device: torch.device | str = "cpu",
) -> ReliabilityGatedRSSM:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("checkpoint file SHA-256 differs from the frozen receipt")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "schema": "boptest-reliability-rssm-checkpoint-v2",
        "case": case,
        "model_seed": model_seed,
        "arm": arm,
        "update": update,
        "config": config.to_dict(),
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint identity/configuration differs from evaluation request")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint has no training provenance")
    validate_rssm_producer_code_manifest(provenance.get("producer_code"))
    validate_numerical_runtime_fingerprint(
        provenance.get("runtime"), include_sklearn=False
    )
    if provenance != dict(expected_provenance):
        raise ValueError("checkpoint training provenance differs from evaluation receipt")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no model state dictionary")
    if checkpoint.get("model_state_sha256") != tensor_state_sha256(state_dict):
        raise ValueError("checkpoint model-state SHA-256 is invalid")
    if checkpoint.get("core_state_sha256") != core_tensor_state_sha256(state_dict):
        raise ValueError("checkpoint RSSM-core SHA-256 is invalid")
    model = ReliabilityGatedRSSM(config.model_config()).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def result_sha256(frame: pd.DataFrame) -> str:
    content = frame.to_csv(index=False).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def _validation_fault_signatures(
    config: StudyConfig,
) -> set[tuple[str, str, int, float, str, int, int, int]]:
    spec = FaultSpec()
    if spec.evaluation_horizon != config.direct_horizon:
        raise ValueError("fault grid and study horizon differ")
    return fault_cell_signatures(spec, "validation")


def validation_selection_scores(
    frame: pd.DataFrame, config: StudyConfig
) -> pd.DataFrame:
    """Return equal-case/family/channel scores for common checkpoint selection."""
    required_columns = {
        "update",
        "arm",
        "case",
        "role",
        "model_seed",
        "cell_id",
        "anchor",
        "family",
        "fault_channel",
        "sign",
        "severity",
        "severity_unit",
        "onset",
        "horizon",
        "trajectory_day",
        "trajectory_seed",
        "target_raw",
        "persistence_prediction_raw",
        "persistence_standardized_abs_error",
        "standardized_abs_error",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"checkpoint selection rows are missing columns: {missing}")
    required_arms = {"ungated_h8", "gated_h8"}
    if set(frame["arm"]) != required_arms:
        raise ValueError("checkpoint selection accepts only ungated_h8 and gated_h8 rows")
    if set(frame["role"]) != {"validation"}:
        raise ValueError("checkpoint selection may use validation rows only")
    if set(frame["case"]) != set(CASES):
        raise ValueError("checkpoint selection requires every frozen BOPTEST case")
    if set(frame["model_seed"]) != set(config.development_seeds):
        raise ValueError("checkpoint selection requires every development model seed")
    if set(frame["update"]) != set(config.validation_checkpoints):
        raise ValueError("checkpoint selection requires the complete frozen update grid")
    if set(frame["family"]) != set(FAMILIES):
        raise ValueError("checkpoint selection requires every frozen fault family")
    if set(frame["fault_channel"]) != set(FAULT_CHANNELS):
        raise ValueError("checkpoint selection requires both frozen fault channels")

    model_independent_identity = [
        "case",
        "trajectory_day",
        "trajectory_seed",
        "anchor",
        "family",
        "fault_channel",
        "sign",
        "severity",
        "severity_unit",
        "onset",
        "horizon",
    ]
    cell_id_counts = frame.groupby(
        model_independent_identity, dropna=False
    )["cell_id"].nunique(dropna=False)
    if not len(cell_id_counts) or (cell_id_counts != 1).any():
        raise ValueError("checkpoint selection cell IDs differ across seeds or updates")
    invariant_columns = [
        "target_raw",
        "persistence_prediction_raw",
        "persistence_standardized_abs_error",
    ]
    invariant_counts = frame.groupby(
        model_independent_identity, dropna=False
    )[invariant_columns].nunique(dropna=False)
    if not len(invariant_counts) or (invariant_counts != 1).any().any():
        raise ValueError(
            "checkpoint selection target/persistence values differ across seeds or updates"
        )

    identity = [
        "case",
        "trajectory_day",
        "trajectory_seed",
        "model_seed",
        "cell_id",
        "anchor",
        "family",
        "fault_channel",
        "sign",
        "severity",
        "severity_unit",
        "onset",
        "horizon",
        "target_raw",
        "persistence_prediction_raw",
        "persistence_standardized_abs_error",
    ]
    keyed = [*identity, "update", "arm"]
    if frame.duplicated(keyed).any():
        raise ValueError("checkpoint selection contains duplicate evaluation identities")

    trajectory_counts = (
        frame.loc[:, ["case", "trajectory_day", "trajectory_seed"]]
        .drop_duplicates()
        .groupby("case")
        .size()
    )
    if set(trajectory_counts.index) != set(CASES) or not (
        trajectory_counts == 8
    ).all():
        raise ValueError("checkpoint selection requires eight validation trajectories per case")
    expected_signatures = _validation_fault_signatures(config)
    signature_columns = [
        "fault_channel",
        "family",
        "sign",
        "severity",
        "severity_unit",
        "onset",
        "anchor",
        "horizon",
    ]
    grid_groups = [
        "case",
        "trajectory_day",
        "trajectory_seed",
        "model_seed",
        "update",
        "arm",
    ]
    for _, group in frame.groupby(grid_groups, sort=False):
        signatures = set(group.loc[:, signature_columns].itertuples(index=False, name=None))
        if signatures != expected_signatures or len(group) != len(expected_signatures):
            raise ValueError("checkpoint selection differs from the frozen validation fault grid")

    expected_pairs = len(required_arms) * len(config.validation_checkpoints)
    coverage = frame.groupby(identity, dropna=False).size()
    if not len(coverage) or not (coverage == expected_pairs).all():
        raise ValueError("checkpoint selection has a cherry-picked or mismatched identity grid")
    if len(frame) != len(coverage) * expected_pairs:
        raise ValueError("checkpoint selection row count differs from its complete grid")
    cells = (
        frame.groupby(
            ["update", "arm", "case", "family", "fault_channel"],
            as_index=False,
        )["standardized_abs_error"]
        .mean()
    )
    arm_scores = (
        cells.groupby(["update", "arm"], as_index=False)[
            "standardized_abs_error"
        ]
        .mean()
        .rename(columns={"standardized_abs_error": "arm_score"})
    )
    scores = (
        arm_scores.groupby("update", as_index=False)["arm_score"]
        .mean()
        .rename(columns={"arm_score": "common_score"})
        .sort_values(["common_score", "update"], kind="stable")
        .reset_index(drop=True)
    )
    scores["selected"] = False
    scores.loc[0, "selected"] = True
    return scores
