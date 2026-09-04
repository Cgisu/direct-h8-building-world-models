"""Causal paired-policy evaluation for the v3 transport study."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch

from building_fault_wm.neural_benchmark.fault_data import (
    FAULT_CHANNELS,
    FaultScalers,
    FaultVariant,
    SequenceReference,
    TrajectoryKey,
    materialize_rssm_batch,
)
from building_fault_wm.neural_benchmark.reliability_model import (
    ReliabilityGatedRSSM,
)
from building_fault_wm.neural_benchmark.study_config import StudyConfig
from building_fault_wm.neural_benchmark.study_train import (
    core_tensor_state_sha256,
    tensor_state_sha256,
)

from .config import FROZEN_CONFIG, DeterministicTransportConfig
from .gate import ARMS, POLICIES, REQUIRED_COLUMNS
from .model import DeterministicRecurrentWorldModel


EVALUATION_HORIZONS = (1, 2, 4, 8)
EVALUATION_HISTORY = 40
SELECTED_UPDATE = 400
RSSM_ARMS = ("legacy", "ungated_h8")
RAW_UNITS = {
    "zone_temperature_k": "K",
    "hvac_electric_power_w": "W",
}


@dataclass(frozen=True)
class PolicyTrajectoryMetadata:
    """Policy-independent and policy-specific identity for one trajectory."""

    policy: str
    window_id: str
    scenario_seed: int

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if not isinstance(self.window_id, str) or not self.window_id:
            raise ValueError("window_id must be a nonempty string")
        if (
            isinstance(self.scenario_seed, bool)
            or not isinstance(self.scenario_seed, int)
            or self.scenario_seed < 0
        ):
            raise ValueError("scenario_seed must be a nonnegative integer")


@dataclass(frozen=True)
class EvaluationFrames:
    """Gate-ready rows and their richer, row-aligned diagnostics."""

    core: pd.DataFrame
    detailed: pd.DataFrame


DETAIL_COLUMNS = (
    *REQUIRED_COLUMNS,
    "update",
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
    "alternate_action_prediction_raw",
    "action_prediction_change_standardized",
)


def _valid_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_v2_rssm(
    path: Path,
    *,
    case: str,
    model_seed: int,
    arm: str,
    expected_file_sha256: str,
    config: StudyConfig | None = None,
    device: torch.device | str = "cpu",
) -> ReliabilityGatedRSSM:
    """Load a byte-bound update-400 v2 RSSM checkpoint.

    The caller supplies the checkpoint digest from the immutable parent
    inventory. Full-file verification binds the embedded provenance as well as
    the model state.
    """

    config = StudyConfig() if config is None else config
    if arm not in RSSM_ARMS:
        raise ValueError(f"v3 accepts only the frozen RSSM arms {RSSM_ARMS}")
    if model_seed not in config.confirmatory_seeds:
        raise ValueError("model_seed is outside the inherited five-seed set")
    if not _valid_sha256(expected_file_sha256):
        raise ValueError("expected_file_sha256 is not a lowercase SHA-256")
    if _file_sha256(path) != expected_file_sha256:
        raise ValueError("v2 RSSM checkpoint differs from the frozen digest")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "schema": "boptest-reliability-rssm-checkpoint-v2",
        "case": case,
        "model_seed": model_seed,
        "arm": arm,
        "update": SELECTED_UPDATE,
        "config": config.to_dict(),
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("v2 RSSM checkpoint identity/configuration changed")
    if not isinstance(checkpoint.get("provenance"), dict):
        raise ValueError("v2 RSSM checkpoint has no training provenance")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("v2 RSSM checkpoint has no model state")
    if checkpoint.get("model_state_sha256") != tensor_state_sha256(state):
        raise ValueError("v2 RSSM checkpoint model-state hash is invalid")
    if checkpoint.get("core_state_sha256") != core_tensor_state_sha256(state):
        raise ValueError("v2 RSSM checkpoint core-state hash is invalid")
    model = ReliabilityGatedRSSM(config.model_config()).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_deterministic_checkpoint(
    path: Path,
    *,
    model_seed: int,
    expected_file_sha256: str,
    config: DeterministicTransportConfig = FROZEN_CONFIG,
    device: torch.device | str = "cpu",
) -> DeterministicRecurrentWorldModel:
    """Load a byte-bound, fixed-update deterministic v3 checkpoint."""

    if model_seed not in config.paired_model_seeds:
        raise ValueError("model_seed is outside the paired five-seed set")
    if not _valid_sha256(expected_file_sha256):
        raise ValueError("expected_file_sha256 is not a lowercase SHA-256")
    if _file_sha256(path) != expected_file_sha256:
        raise ValueError("deterministic checkpoint differs from the frozen digest")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "schema": "boptest-deterministic-transport-checkpoint-v1",
        "update": SELECTED_UPDATE,
        "model_seed": model_seed,
        "config": config.to_dict(),
        "selected": True,
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("deterministic checkpoint identity/configuration changed")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("deterministic checkpoint has no model state")
    if checkpoint.get("model_state_sha256") != tensor_state_sha256(state):
        raise ValueError("deterministic checkpoint model-state hash is invalid")
    model = DeterministicRecurrentWorldModel(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def metadata_from_case_plan(
    plan: Mapping[str, object],
) -> dict[TrajectoryKey, PolicyTrajectoryMetadata]:
    """Build the explicit trajectory-to-policy map from a validated v3 plan."""

    from .plan import validate_case_plan

    validate_case_plan(plan)
    case = str(plan["case"])
    entries = plan["entries"]
    assert isinstance(entries, list)
    result: dict[TrajectoryKey, PolicyTrajectoryMetadata] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("v3 plan contains a non-object entry")
        policies = entry.get("policies")
        if not isinstance(policies, dict) or set(policies) != set(POLICIES):
            raise ValueError("v3 plan entry has an incomplete policy map")
        for policy in POLICIES:
            branch = policies[policy]
            if not isinstance(branch, dict):
                raise ValueError("v3 plan policy branch is not an object")
            key = TrajectoryKey(
                case=case,
                role="locked_test",
                day=int(entry["day"]),
                trajectory_seed=int(branch["trajectory_seed"]),
            )
            if key in result:
                raise ValueError("v3 plan repeats a trajectory identity")
            result[key] = PolicyTrajectoryMetadata(
                policy=policy,
                window_id=str(entry["window_id"]),
                scenario_seed=int(entry["scenario_seed"]),
            )
    return result


def action_block_crosses_transition(
    actions: np.ndarray, anchor: int, horizon: int
) -> tuple[bool, int]:
    """Classify transitions entering or occurring within the rollout block."""

    if actions.ndim != 2 or actions.shape[1] != 1:
        raise ValueError("actions must have shape [time, 1]")
    if (
        isinstance(anchor, bool)
        or not isinstance(anchor, int)
        or anchor < 0
        or horizon not in EVALUATION_HORIZONS
    ):
        raise ValueError("anchor/horizon are outside the evaluation contract")
    start = max(anchor - 1, 0)
    block = actions[start : anchor + horizon]
    expected = horizon + int(anchor > 0)
    if len(block) != expected or not np.isfinite(block).all():
        raise ValueError("action block leaves the finite trajectory")
    transitions = int(np.count_nonzero(np.diff(block[:, 0]) != 0.0))
    return transitions > 0, transitions


def _alternate_action_block(
    variant: FaultVariant,
    anchor: int,
    horizon: int,
    *,
    shift: int,
) -> np.ndarray:
    realized = variant.actions[anchor : anchor + horizon]
    for offset in range(shift, len(variant.actions) - anchor - horizon + 1, shift):
        candidate = variant.actions[
            anchor + offset : anchor + offset + horizon
        ]
        if not np.array_equal(candidate, realized):
            return candidate
    raise ValueError("trajectory has no later, different action diagnostic block")


def _normalize_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(horizons)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(value not in EVALUATION_HORIZONS for value in normalized)
    ):
        raise ValueError(
            f"horizons must be unique values drawn from {EVALUATION_HORIZONS}"
        )
    return normalized


def _variant_signature(variant: FaultVariant) -> tuple[object, ...]:
    cell = variant.cell
    return (
        cell.fault_channel,
        cell.family,
        cell.sign,
        cell.severity,
        cell.severity_unit,
        cell.onset,
        cell.stop,
        cell.anchors,
    )


def _validate_policy_metadata(
    variants: Sequence[FaultVariant],
    metadata: Mapping[TrajectoryKey, PolicyTrajectoryMetadata],
) -> None:
    trajectory_keys = {variant.cell.trajectory for variant in variants}
    if not variants or set(metadata) != trajectory_keys:
        raise ValueError(
            "policy metadata must map exactly the evaluated trajectory identities"
        )
    by_window: dict[
        tuple[str, str], dict[str, tuple[TrajectoryKey, set[tuple[object, ...]]]]
    ] = defaultdict(dict)
    for key in trajectory_keys:
        item = metadata[key]
        if key.day <= 0:
            raise ValueError("trajectory day must be positive")
        variants_for_key = [
            variant for variant in variants if variant.cell.trajectory == key
        ]
        slot = (key.case, item.window_id)
        if item.policy in by_window[slot]:
            raise ValueError("a window has more than one trajectory for a policy")
        by_window[slot][item.policy] = (
            key,
            {_variant_signature(variant) for variant in variants_for_key},
        )
    for (_, window_id), branches in by_window.items():
        if set(branches) != set(POLICIES):
            raise ValueError(f"window {window_id} is missing a paired policy")
        keys = [branches[policy][0] for policy in POLICIES]
        branch_metadata = [metadata[key] for key in keys]
        if len({key.day for key in keys}) != 1:
            raise ValueError("paired policies differ in trajectory day")
        if len({item.scenario_seed for item in branch_metadata}) != 1:
            raise ValueError("paired policies differ in scenario seed")
        if len({key.trajectory_seed for key in keys}) != len(POLICIES):
            raise ValueError("paired policies must use distinct trajectory seeds")
        if branches[POLICIES[0]][1] != branches[POLICIES[1]][1]:
            raise ValueError("paired policies have different fault-cell grids")


def _validate_models(
    models: Mapping[str, torch.nn.Module],
    *,
    model_seed: int,
    rssm_config: StudyConfig,
) -> None:
    if set(models) != set(ARMS):
        raise ValueError(f"evaluation models must contain exactly {ARMS}")
    if model_seed not in rssm_config.confirmatory_seeds:
        raise ValueError("model_seed is outside the inherited five-seed set")
    if FROZEN_CONFIG.paired_model_seeds != rssm_config.confirmatory_seeds:
        raise ValueError("RSSM and deterministic seed grids differ")
    if models["legacy"] is models["ungated_h8"]:
        raise ValueError("legacy and ungated_h8 must be distinct checkpoint objects")
    if not all(
        isinstance(models[arm], ReliabilityGatedRSSM) for arm in RSSM_ARMS
    ):
        raise TypeError("legacy and ungated_h8 must be ReliabilityGatedRSSM models")
    if not isinstance(
        models["deterministic_wm"], DeterministicRecurrentWorldModel
    ):
        raise TypeError(
            "deterministic_wm must be a DeterministicRecurrentWorldModel"
        )
    if any(
        models[arm].config != rssm_config.model_config()  # type: ignore[attr-defined]
        for arm in RSSM_ARMS
    ):
        raise ValueError("an RSSM model configuration differs from the frozen study")
    if models["deterministic_wm"].config != FROZEN_CONFIG:  # type: ignore[attr-defined]
        raise ValueError("the deterministic model configuration is not frozen v3")


def _predict(
    model: torch.nn.Module,
    arm: str,
    history,
    future_actions: np.ndarray,
    future_contexts: np.ndarray,
    alternate_actions: np.ndarray,
    *,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = {
        "previous_actions": torch.as_tensor(
            history.previous_actions, dtype=torch.float32, device=device
        ),
        "observations": torch.as_tensor(
            history.corrupted_observations, dtype=torch.float32, device=device
        ),
        "availability": torch.as_tensor(
            history.availability, dtype=torch.bool, device=device
        ),
        "age": torch.as_tensor(
            history.age, dtype=torch.float32, device=device
        ),
        "contexts": torch.as_tensor(
            history.contexts, dtype=torch.float32, device=device
        ),
        "future_actions": torch.as_tensor(
            future_actions, dtype=torch.float32, device=device
        ),
        "future_contexts": torch.as_tensor(
            future_contexts, dtype=torch.float32, device=device
        ),
        "alternate_actions": torch.as_tensor(
            alternate_actions, dtype=torch.float32, device=device
        ),
    }
    model.eval()
    if arm in RSSM_ARMS:
        assert isinstance(model, ReliabilityGatedRSSM)
        filtered = model.filter(
            tensors["previous_actions"],
            tensors["observations"],
            tensors["availability"],
            tensors["age"],
            tensors["contexts"],
            gate_mode="bypass",
            sample=False,
        )
        actual = model.imagine(
            filtered.rssm.final_state,
            tensors["future_actions"],
            tensors["future_contexts"],
            sample=False,
        )
        alternate = model.imagine(
            filtered.rssm.final_state,
            tensors["alternate_actions"],
            tensors["future_contexts"],
            sample=False,
        )
    else:
        assert isinstance(model, DeterministicRecurrentWorldModel)
        hidden = model.initial_hidden(
            tensors["observations"].shape[1],
            device=device,
            dtype=tensors["observations"].dtype,
        )
        for values in zip(
            tensors["observations"],
            tensors["availability"],
            tensors["age"],
            tensors["previous_actions"],
            tensors["contexts"],
            strict=True,
        ):
            hidden, _ = model.observe_step(hidden, *values)
        actual = model.imagine(
            hidden,
            tensors["future_actions"],
            tensors["future_contexts"],
        )
        alternate = model.imagine(
            hidden,
            tensors["alternate_actions"],
            tensors["future_contexts"],
        )
    return (
        actual.observation_mean[-1].detach().cpu().numpy(),
        alternate.observation_mean[-1].detach().cpu().numpy(),
    )


def assert_paired_rows(core: pd.DataFrame) -> None:
    """Assert exact model-arm rows and paired action-policy branches."""

    if tuple(core.columns) != REQUIRED_COLUMNS:
        raise ValueError("evaluation core columns differ from gate.REQUIRED_COLUMNS")
    if core.empty or core.duplicated(
        [column for column in REQUIRED_COLUMNS if column != "standardized_abs_error"]
    ).any():
        raise ValueError("evaluation core rows are empty or duplicated")
    arm_identity = [
        column
        for column in REQUIRED_COLUMNS
        if column not in {"arm", "standardized_abs_error"}
    ]
    arm_counts = core.groupby(arm_identity, dropna=False)["arm"].agg(
        lambda values: frozenset(values)
    )
    if not len(arm_counts) or any(value != frozenset(ARMS) for value in arm_counts):
        raise ValueError("model arms do not share exactly the same row identities")

    policy_identity = [
        "case",
        "window_id",
        "trajectory_day",
        "scenario_seed",
        "model_seed",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "anchor",
        "horizon",
    ]
    branches = core.loc[
        :, [*policy_identity, "policy", "trajectory_seed", "cell_id"]
    ].drop_duplicates()
    grouped = branches.groupby(policy_identity, dropna=False, sort=False)
    for _, rows in grouped:
        if set(rows["policy"]) != set(POLICIES) or len(rows) != len(POLICIES):
            raise ValueError(
                "paired policy rows differ beyond policy-specific trajectory identity"
            )
        if rows["trajectory_seed"].nunique() != len(POLICIES):
            raise ValueError("paired policy rows reuse a trajectory seed")


@torch.no_grad()
def evaluate_transport_models(
    models: Mapping[str, torch.nn.Module],
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    trajectory_metadata: Mapping[
        TrajectoryKey, PolicyTrajectoryMetadata
    ],
    *,
    model_seed: int,
    horizons: Sequence[int] = EVALUATION_HORIZONS,
    rssm_config: StudyConfig | None = None,
    device: torch.device | str = "cpu",
    prediction_seconds_by_arm: MutableMapping[str, float] | None = None,
) -> EvaluationFrames:
    """Evaluate all three arms from identical causal histories and futures."""

    rssm_config = StudyConfig() if rssm_config is None else rssm_config
    normalized_horizons = _normalize_horizons(horizons)
    _validate_models(models, model_seed=model_seed, rssm_config=rssm_config)
    _validate_policy_metadata(variants, trajectory_metadata)
    if rssm_config.sequence_length - rssm_config.direct_horizon != EVALUATION_HISTORY:
        raise ValueError("RSSM config no longer implies the exact 40-step history")

    by_anchor: dict[int, list[int]] = defaultdict(list)
    for index, variant in enumerate(variants):
        if variant.cell.fault_channel not in FAULT_CHANNELS:
            raise ValueError("variant fault channel is outside the frozen contract")
        for anchor in variant.cell.anchors:
            if anchor < EVALUATION_HISTORY:
                raise ValueError("evaluation anchor lacks a 40-step causal history")
            if any(
                anchor + horizon >= variant.cell.stop
                or anchor + horizon >= len(variant.clean_observations)
                for horizon in normalized_horizons
            ):
                raise ValueError("evaluation endpoint leaves the active fault/trajectory")
            by_anchor[anchor].append(index)

    rows: list[dict[str, object]] = []
    observation_mean = np.asarray(scalers.observation.mean)
    observation_scale = np.asarray(scalers.observation.scale)
    for horizon in normalized_horizons:
        for anchor, indices in sorted(by_anchor.items()):
            selected = [variants[index] for index in indices]
            references = tuple(
                SequenceReference(index, anchor - EVALUATION_HISTORY)
                for index in range(len(selected))
            )
            history = materialize_rssm_batch(
                selected,
                scalers,
                references,
                length=EVALUATION_HISTORY,
            )
            expected_rows = np.arange(
                anchor - EVALUATION_HISTORY + 1, anchor + 1
            )
            if not all(
                np.array_equal(history.source_rows[:, index], expected_rows)
                for index in range(len(selected))
            ):
                raise AssertionError("materialized history is not the exact causal window")
            future_actions = np.stack(
                [
                    scalers.action.transform(
                        variant.actions[anchor : anchor + horizon]
                    )
                    for variant in selected
                ],
                axis=1,
            )
            future_contexts = np.stack(
                [
                    scalers.context.transform(
                        variant.contexts[anchor + 1 : anchor + horizon + 1]
                    )
                    for variant in selected
                ],
                axis=1,
            )
            alternate_actions = np.stack(
                [
                    scalers.action.transform(
                        _alternate_action_block(
                            variant,
                            anchor,
                            horizon,
                            shift=rssm_config.action_diagnostic_shift,
                        )
                    )
                    for variant in selected
                ],
                axis=1,
            )
            target_raw = np.stack(
                [
                    variant.clean_observations[anchor + horizon]
                    for variant in selected
                ]
            )
            target_standardized = scalers.observation.transform(target_raw)

            for arm in ARMS:
                started = time.perf_counter()
                predicted, alternate_predicted = _predict(
                    models[arm],
                    arm,
                    history,
                    future_actions,
                    future_contexts,
                    alternate_actions,
                    device=device,
                )
                if prediction_seconds_by_arm is not None:
                    prediction_seconds_by_arm[arm] = (
                        float(prediction_seconds_by_arm.get(arm, 0.0))
                        + time.perf_counter()
                        - started
                    )
                predicted_raw = (
                    predicted * observation_scale + observation_mean
                )
                alternate_raw = (
                    alternate_predicted * observation_scale + observation_mean
                )
                for batch_index, variant in enumerate(selected):
                    cell = variant.cell
                    key = cell.trajectory
                    metadata = trajectory_metadata[key]
                    channel_index = FAULT_CHANNELS.index(cell.fault_channel)
                    boundary, transition_count = action_block_crosses_transition(
                        variant.actions, anchor, horizon
                    )
                    raw_error = abs(
                        predicted_raw[batch_index, channel_index]
                        - target_raw[batch_index, channel_index]
                    )
                    row = {
                        "case": key.case,
                        "policy": metadata.policy,
                        "window_id": metadata.window_id,
                        "trajectory_day": key.day,
                        "scenario_seed": metadata.scenario_seed,
                        "trajectory_seed": key.trajectory_seed,
                        "model_seed": model_seed,
                        "arm": arm,
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
                                predicted[batch_index, channel_index]
                                - target_standardized[
                                    batch_index, channel_index
                                ]
                            )
                        ),
                        "update": SELECTED_UPDATE,
                        "fault_channel_index": channel_index,
                        "severity_unit": cell.severity_unit,
                        "history_start": anchor - EVALUATION_HISTORY + 1,
                        "history_stop": anchor,
                        "target_index": anchor + horizon,
                        "target_raw": float(
                            target_raw[batch_index, channel_index]
                        ),
                        "prediction_raw": float(
                            predicted_raw[batch_index, channel_index]
                        ),
                        "prediction_standardized": float(
                            predicted[batch_index, channel_index]
                        ),
                        "raw_abs_error": float(raw_error),
                        "raw_unit": RAW_UNITS[cell.fault_channel],
                        "boundary_crossing": bool(boundary),
                        "action_transition_count": transition_count,
                        "alternate_action_prediction_raw": float(
                            alternate_raw[batch_index, channel_index]
                        ),
                        "action_prediction_change_standardized": float(
                            abs(
                                predicted[batch_index, channel_index]
                                - alternate_predicted[
                                    batch_index, channel_index
                                ]
                            )
                        ),
                    }
                    rows.append(row)

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
            "arm",
        ],
        kind="stable",
    ).reset_index(drop=True)
    if detailed.empty:
        raise ValueError("evaluation produced no rows")
    numeric = detailed.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("evaluation produced non-finite numeric diagnostics")
    core = detailed.loc[:, REQUIRED_COLUMNS].copy()
    assert_paired_rows(core)
    return EvaluationFrames(core=core, detailed=detailed)
