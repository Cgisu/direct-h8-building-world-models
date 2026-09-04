"""Exact-schedule training for the deterministic v3 world model."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from building_fault_wm.recurrent_models.training import (
    RSSMSequenceInputs,
    RSSMSequenceTargets,
)
from building_fault_wm.neural_benchmark.fault_data import (
    FaultScalers,
    FaultVariant,
    SequenceReference,
)
from building_fault_wm.neural_benchmark.study_train import (
    ScheduledBatch,
    materialize_torch_batch,
    tensor_state_sha256,
)

from .config import (
    FROZEN_CONFIG,
    PARENT_SCHEDULE_SCHEMA,
    DeterministicTransportConfig,
)
from .model import DeterministicRecurrentWorldModel, DeterministicRollout


@dataclass(frozen=True)
class ImaginationBatch:
    start_hidden: Tensor
    future_actions: Tensor
    future_contexts: Tensor
    targets: Tensor
    source_count: int
    path_count: int


@dataclass(frozen=True)
class DeterministicLoss:
    total: Tensor
    one_step_smooth_l1: Tensor
    direct_h8_smooth_l1: Tensor
    one_step_paths: int
    direct_h8_paths: int
    direct_h8_targets: int


@dataclass(frozen=True)
class TrainingLogRow:
    update: int
    total: float
    one_step_smooth_l1: float
    direct_h8_smooth_l1: float
    gradient_norm: float
    one_step_paths: int
    direct_h8_paths: int


@dataclass
class DeterministicFitResult:
    model: DeterministicRecurrentWorldModel
    optimizer_state_dict: dict[str, object]
    log: tuple[TrainingLogRow, ...]
    checkpoint_states: dict[int, dict[str, Tensor]]
    selected_update: int
    schedule_sha256: str


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _schedule_body(payload: Mapping[str, object]) -> dict[str, object]:
    if set(payload) != {"schema", "updates", "sha256"}:
        raise ValueError("parent schedule fields differ from the frozen schema")
    body = {"schema": payload["schema"], "updates": payload["updates"]}
    if payload["schema"] != PARENT_SCHEDULE_SCHEMA:
        raise ValueError("parent schedule schema mismatch")
    if payload["sha256"] != _canonical_sha256(body):
        raise ValueError("parent schedule SHA-256 mismatch")
    return body


def map_parent_schedule(
    payload: Mapping[str, object],
    variants: Sequence[FaultVariant],
    config: DeterministicTransportConfig = FROZEN_CONFIG,
) -> tuple[ScheduledBatch, ...]:
    """Map persisted ``cell_id`` references back to current variant indices."""

    body = _schedule_body(payload)
    by_cell_id: dict[str, int] = {}
    for variant_index, variant in enumerate(variants):
        cell_id = variant.cell.cell_id
        if cell_id in by_cell_id:
            raise ValueError(f"duplicate variant cell_id: {cell_id}")
        by_cell_id[cell_id] = variant_index
    if not by_cell_id:
        raise ValueError("cannot map a parent schedule without variants")

    raw_updates = body["updates"]
    if not isinstance(raw_updates, list) or len(raw_updates) != config.updates:
        raise ValueError("parent schedule does not contain exactly 400 updates")
    mapped: list[ScheduledBatch] = []
    for expected_update, raw in enumerate(raw_updates, start=1):
        if not isinstance(raw, dict) or set(raw) != {
            "update",
            "latent_seed",
            "references",
        }:
            raise ValueError("parent schedule update fields are invalid")
        update = raw["update"]
        latent_seed = raw["latent_seed"]
        references = raw["references"]
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update != expected_update
        ):
            raise ValueError("parent schedule update identities are not sequential")
        if (
            isinstance(latent_seed, bool)
            or not isinstance(latent_seed, int)
            or latent_seed < 0
        ):
            raise ValueError("parent schedule latent_seed is invalid")
        if not isinstance(references, list) or len(references) != config.batch_size:
            raise ValueError("parent schedule batch size changed")
        mapped_references: list[SequenceReference] = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {
                "cell_id",
                "aligned_start",
            }:
                raise ValueError("parent schedule reference fields are invalid")
            cell_id = reference["cell_id"]
            aligned_start = reference["aligned_start"]
            if not isinstance(cell_id, str) or cell_id not in by_cell_id:
                raise ValueError(f"parent schedule cell_id is unavailable: {cell_id}")
            if (
                isinstance(aligned_start, bool)
                or not isinstance(aligned_start, int)
                or aligned_start < 0
            ):
                raise ValueError("parent schedule aligned_start is invalid")
            variant = variants[by_cell_id[cell_id]]
            if variant.cell.trajectory.role != "fit":
                raise ValueError(
                    "parent training schedule references a non-FIT variant"
                )
            if aligned_start != variant.cell.onset - 17:
                raise ValueError(
                    "parent schedule aligned_start differs from the frozen onset rule"
                )
            if aligned_start + config.sequence_length >= len(
                variant.clean_observations
            ):
                raise ValueError("parent schedule reference crosses a trajectory")
            mapped_references.append(
                SequenceReference(by_cell_id[cell_id], aligned_start)
            )
        mapped.append(
            ScheduledBatch(
                update=update,
                latent_seed=latent_seed,
                references=tuple(mapped_references),
            )
        )
    return tuple(mapped)


def load_parent_schedule(
    path: Path,
    variants: Sequence[FaultVariant],
    config: DeterministicTransportConfig = FROZEN_CONFIG,
) -> tuple[ScheduledBatch, ...]:
    """Load and map one hash-validated persisted parent RSSM schedule."""

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key in parent schedule: {key}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load parent schedule: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("parent schedule must be a JSON object")
    return map_parent_schedule(payload, variants, config)


def aligned_imagination_batch(
    filtered_hidden: Tensor,
    inputs: RSSMSequenceInputs,
    targets: RSSMSequenceTargets,
    horizon: int,
) -> ImaginationBatch:
    """Align posterior ``t`` with actions/contexts ``t+1...t+h`` and target ``t+h``."""

    if filtered_hidden.ndim != 3:
        raise ValueError("filtered_hidden must have shape [time, batch, hidden]")
    time_steps, batch_size = filtered_hidden.shape[:2]
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 0 < horizon < time_steps
    ):
        raise ValueError("horizon must be an integer inside the sequence")
    if inputs.previous_actions.ndim != 3:
        raise ValueError("previous_actions must be time-major")
    if inputs.previous_actions.shape[:2] != (time_steps, batch_size):
        raise ValueError("previous_actions and filtered_hidden are misaligned")
    if inputs.contexts is None or inputs.contexts.ndim != 3:
        raise ValueError("contexts are required and must be time-major")
    if inputs.contexts.shape[:2] != (time_steps, batch_size):
        raise ValueError("contexts and filtered_hidden are misaligned")
    if targets.clean_observations.shape[:2] != (time_steps, batch_size):
        raise ValueError("clean targets and filtered_hidden are misaligned")

    source_count = time_steps - horizon
    start_hidden = filtered_hidden[:source_count].reshape(
        source_count * batch_size, filtered_hidden.shape[-1]
    )
    future_actions = torch.stack(
        [
            inputs.previous_actions[offset : offset + source_count]
            for offset in range(1, horizon + 1)
        ],
        dim=0,
    ).reshape(
        horizon,
        source_count * batch_size,
        inputs.previous_actions.shape[-1],
    )
    future_contexts = torch.stack(
        [
            inputs.contexts[offset : offset + source_count]
            for offset in range(1, horizon + 1)
        ],
        dim=0,
    ).reshape(
        horizon,
        source_count * batch_size,
        inputs.contexts.shape[-1],
    )
    clean_targets = targets.clean_observations[horizon:].reshape(
        source_count * batch_size,
        targets.clean_observations.shape[-1],
    )
    return ImaginationBatch(
        start_hidden=start_hidden,
        future_actions=future_actions,
        future_contexts=future_contexts,
        targets=clean_targets,
        source_count=source_count,
        path_count=source_count * batch_size,
    )


def deterministic_transport_loss(
    model: DeterministicRecurrentWorldModel,
    inputs: RSSMSequenceInputs,
    targets: RSSMSequenceTargets,
    config: DeterministicTransportConfig = FROZEN_CONFIG,
) -> tuple[DeterministicRollout, DeterministicLoss]:
    """Compute equal-weight one-step and direct-H8 observation losses."""

    rollout = model.filter(
        inputs.corrupted_observations,
        inputs.availability,
        inputs.age,
        inputs.previous_actions,
        inputs.contexts,
    )
    h8 = aligned_imagination_batch(
        rollout.hidden, inputs, targets, horizon=config.direct_horizon
    )
    h8_prediction = model.imagine(
        h8.start_hidden,
        h8.future_actions,
        h8.future_contexts,
    ).observation_mean[-1]
    if (
        not torch.isfinite(targets.clean_observations).all()
        or not torch.isfinite(h8.targets).all()
    ):
        raise ValueError("clean observation targets must be finite")
    one_step_loss = F.smooth_l1_loss(
        rollout.observation_mean,
        targets.clean_observations,
        beta=config.smooth_l1_beta,
    )
    h8_loss = F.smooth_l1_loss(
        h8_prediction,
        h8.targets,
        beta=config.smooth_l1_beta,
    )
    total = (
        config.observation_weight * one_step_loss
        + config.direct_h8_weight * h8_loss
    )
    return rollout, DeterministicLoss(
        total=total,
        one_step_smooth_l1=one_step_loss,
        direct_h8_smooth_l1=h8_loss,
        one_step_paths=int(targets.clean_observations.shape[0] * targets.clean_observations.shape[1]),
        direct_h8_paths=h8.path_count,
        direct_h8_targets=h8.targets.numel(),
    )


def _clone_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _validate_schedule_prefix(
    schedule: Sequence[ScheduledBatch],
    config: DeterministicTransportConfig,
    update_count: int,
) -> tuple[ScheduledBatch, ...]:
    normalized = tuple(schedule)
    if not 0 < update_count <= config.updates:
        raise ValueError("update_count must lie inside the frozen update grid")
    if len(normalized) < update_count:
        raise ValueError("schedule is shorter than the requested update prefix")
    prefix = normalized[:update_count]
    if tuple(item.update for item in prefix) != tuple(range(1, update_count + 1)):
        raise ValueError("schedule update identities are incomplete")
    if any(len(item.references) != config.batch_size for item in prefix):
        raise ValueError("schedule batch size changed")
    return prefix


def _fit_schedule_prefix(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    schedule: Sequence[ScheduledBatch],
    *,
    model_seed: int,
    update_count: int,
    checkpoint_updates: Sequence[int],
    config: DeterministicTransportConfig = FROZEN_CONFIG,
    device: torch.device | str = "cpu",
) -> DeterministicFitResult:
    """Run a deterministic prefix; the public protocol calls the full 400."""

    if model_seed not in config.paired_model_seeds:
        raise ValueError("model_seed is outside the paired five-seed set")
    prefix = _validate_schedule_prefix(schedule, config, update_count)
    checkpoints = tuple(checkpoint_updates)
    if (
        tuple(sorted(set(checkpoints))) != checkpoints
        or any(not 0 < update <= update_count for update in checkpoints)
    ):
        raise ValueError("checkpoint updates must be unique, sorted, and in-range")
    torch.manual_seed(model_seed)
    model = DeterministicRecurrentWorldModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    log: list[TrainingLogRow] = []
    checkpoint_states: dict[int, dict[str, Tensor]] = {}
    try:
        for scheduled in prefix:
            inputs, targets = materialize_torch_batch(
                variants,
                scalers,
                scheduled,
                config,  # type: ignore[arg-type]
                device=device,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, loss = deterministic_transport_loss(model, inputs, targets, config)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("deterministic transport loss is non-finite")
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    "deterministic transport gradient is non-finite"
                )
            optimizer.step()
            log.append(
                TrainingLogRow(
                    update=scheduled.update,
                    total=float(loss.total.detach()),
                    one_step_smooth_l1=float(loss.one_step_smooth_l1.detach()),
                    direct_h8_smooth_l1=float(
                        loss.direct_h8_smooth_l1.detach()
                    ),
                    gradient_norm=float(gradient_norm.detach()),
                    one_step_paths=loss.one_step_paths,
                    direct_h8_paths=loss.direct_h8_paths,
                )
            )
            if scheduled.update in checkpoints:
                checkpoint_states[scheduled.update] = _clone_state_dict(model)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    if set(checkpoint_states) != set(checkpoints):
        raise AssertionError("requested deterministic checkpoints were not created")
    schedule_updates = [
        {
            "update": item.update,
            "latent_seed": item.latent_seed,
            "references": [
                {
                    "cell_id": variants[reference.variant_index].cell.cell_id,
                    "aligned_start": reference.aligned_start,
                }
                for reference in item.references
            ],
        }
        for item in prefix
    ]
    schedule_payload = {
        "schema": PARENT_SCHEDULE_SCHEMA,
        "updates": schedule_updates,
    }
    return DeterministicFitResult(
        model=model,
        optimizer_state_dict=optimizer.state_dict(),
        log=tuple(log),
        checkpoint_states=checkpoint_states,
        selected_update=update_count,
        schedule_sha256=_canonical_sha256(schedule_payload),
    )


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _persist_result(
    result: DeterministicFitResult,
    output_dir: Path,
    *,
    model_seed: int,
    config: DeterministicTransportConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes: dict[str, str] = {}
    for update, state in sorted(result.checkpoint_states.items()):
        path = output_dir / "checkpoints" / f"update_{update:04d}.pt"
        _atomic_torch_save(
            path,
            {
                "schema": "boptest-deterministic-transport-checkpoint-v1",
                "update": update,
                "model_seed": model_seed,
                "config": config.to_dict(),
                "model_state_dict": state,
                "model_state_sha256": tensor_state_sha256(state),
                "selected": update == config.updates,
            },
        )
        checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "boptest-deterministic-transport-training-v1",
        "model_seed": model_seed,
        "updates": config.updates,
        "checkpoint_updates": list(config.checkpoint_updates),
        "selected_update": result.selected_update,
        "selection_rule": "fixed_final_update_no_validation_selection",
        "schedule_sha256": result.schedule_sha256,
        "final_model_state_sha256": tensor_state_sha256(
            result.model.state_dict()
        ),
        "checkpoint_file_sha256": checkpoint_hashes,
        "config": config.to_dict(),
        "training_log": [asdict(row) for row in result.log],
    }
    content = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("ascii")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_dir, prefix=".training_receipt.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_dir / "training_receipt.json")


def train_fixed_400(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    schedule: Sequence[ScheduledBatch],
    *,
    model_seed: int,
    output_dir: Path | None = None,
    config: DeterministicTransportConfig = FROZEN_CONFIG,
    device: torch.device | str = "cpu",
) -> DeterministicFitResult:
    """Train all 400 inherited updates and select only update 400."""

    if config != FROZEN_CONFIG:
        raise ValueError("the public fixed trainer requires the frozen v3 config")
    if len(schedule) != config.updates:
        raise ValueError("the fixed trainer requires all 400 scheduled updates")
    result = _fit_schedule_prefix(
        variants,
        scalers,
        schedule,
        model_seed=model_seed,
        update_count=config.updates,
        checkpoint_updates=config.checkpoint_updates,
        config=config,
        device=device,
    )
    if result.selected_update != 400 or set(result.checkpoint_states) != {
        100,
        200,
        300,
        400,
    }:
        raise AssertionError("fixed update-400 selection contract changed")
    if output_dir is not None:
        _persist_result(
            result,
            output_dir,
            model_seed=model_seed,
            config=config,
        )
    return result
