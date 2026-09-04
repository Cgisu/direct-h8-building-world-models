from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from building_fault_wm.neural_benchmark.fault_data import (
    FAMILIES,
    FAULT_CHANNELS,
    FaultCell,
    FaultScalers,
    FaultVariant,
    ScaleStats,
    TrajectoryKey,
)
from building_fault_wm.neural_benchmark.study_train import (
    ScheduledBatch,
    materialize_torch_batch,
    tensor_state_sha256,
)

from .config import FROZEN_CONFIG, PARENT_SCHEDULE_SCHEMA
from . import train_grid
from .model import DeterministicRecurrentWorldModel
from .train import (
    _fit_schedule_prefix,
    aligned_imagination_batch,
    deterministic_transport_loss,
    load_parent_schedule,
    map_parent_schedule,
)


def _variants() -> list[FaultVariant]:
    time_steps = 128
    time = np.arange(time_steps, dtype=float)
    clean = np.stack(
        [
            np.sin(time / 13.0),
            np.cos(time / 17.0),
            time / time_steps,
            (time % 9.0) / 9.0,
        ],
        axis=1,
    )
    actions = ((time.astype(int) % 3) - 1).reshape(-1, 1).astype(float)
    contexts = np.stack(
        [np.sin(time / (index + 7.0)) for index in range(5)], axis=1
    )
    variants: list[FaultVariant] = []
    for channel_index, channel in enumerate(FAULT_CHANNELS):
        for family_index, family in enumerate(FAMILIES):
            corrupted = clean.copy()
            availability = np.ones_like(clean, dtype=bool)
            age = np.zeros_like(clean)
            labels = np.zeros((time_steps, len(FAULT_CHANNELS)), dtype=np.int64)
            if family != "healthy":
                corrupted[32:80, channel_index] += 0.25 + 0.1 * family_index
                labels[32:80, channel_index] = family_index
            if family == "dropout":
                corrupted[32:80, channel_index] = np.nan
                availability[32:80, channel_index] = False
                age[32:80, channel_index] = np.arange(1, 49)
            cell_id = f"case:{channel}:{family}"
            variants.append(
                FaultVariant(
                    cell=FaultCell(
                        cell_id=cell_id,
                        trajectory=TrajectoryKey("case", "fit", 1, 101),
                        source_sha256="a" * 64,
                        fault_channel=channel,
                        family=family,
                        sign=0,
                        severity=0.0,
                        severity_unit="none",
                        onset=32,
                        stop=80,
                        anchors=(40, 48, 56, 64),
                    ),
                    source=Path("synthetic.csv"),
                    clean_observations=clean,
                    corrupted_observations=corrupted,
                    actions=actions,
                    contexts=contexts,
                    availability=availability,
                    age=age,
                    health_labels=labels,
                )
            )
    return variants


def _scalers() -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "a" * 64),),
    )


def _schedule_payload(variants: list[FaultVariant]) -> dict[str, object]:
    updates = []
    for update in range(1, 401):
        indices = tuple((update + offset) % len(variants) for offset in range(4))
        updates.append(
            {
                "update": update,
                "latent_seed": 9000 + update,
                "references": [
                    {
                        "cell_id": variants[index].cell.cell_id,
                        "aligned_start": 15,
                    }
                    for index in indices
                ],
            }
        )
    body = {"schema": PARENT_SCHEDULE_SCHEMA, "updates": updates}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


def test_parent_schedule_mapping_preserves_cell_and_alignment(
    tmp_path: Path,
) -> None:
    variants = _variants()
    payload = _schedule_payload(variants)
    mapped = map_parent_schedule(payload, variants)
    first_raw = payload["updates"][0]  # type: ignore[index]
    first_mapped = mapped[0]
    assert first_mapped.update == 1
    assert first_mapped.latent_seed == 9001
    assert len(first_mapped.references) == 4
    for raw, reference in zip(
        first_raw["references"],  # type: ignore[index]
        first_mapped.references,
        strict=True,
    ):
        assert variants[reference.variant_index].cell.cell_id == raw["cell_id"]
        assert reference.aligned_start == raw["aligned_start"] == 15

    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    assert load_parent_schedule(path, variants) == mapped


def test_h8_alignment_and_endpoint_count_are_exact() -> None:
    variants = _variants()
    scheduled = map_parent_schedule(_schedule_payload(variants), variants)[0]
    inputs, targets = materialize_torch_batch(
        variants, _scalers(), scheduled, FROZEN_CONFIG
    )
    hidden = torch.arange(48 * 4 * 64, dtype=torch.float32).reshape(48, 4, 64)
    aligned = aligned_imagination_batch(hidden, inputs, targets, horizon=8)
    assert aligned.source_count == 40
    assert aligned.path_count == 160
    assert aligned.start_hidden.shape == (160, 64)
    assert aligned.future_actions.shape == (8, 160, 1)
    assert aligned.future_contexts.shape == (8, 160, 5)
    assert aligned.targets.shape == (160, 4)

    torch.testing.assert_close(aligned.start_hidden[0], hidden[0, 0])
    torch.testing.assert_close(
        aligned.future_actions[:, 0],
        inputs.previous_actions[1:9, 0],
    )
    torch.testing.assert_close(
        aligned.future_contexts[:, 0],
        inputs.contexts[1:9, 0],
    )
    torch.testing.assert_close(aligned.targets[0], targets.clean_observations[8, 0])

    model = DeterministicRecurrentWorldModel()
    _, loss = deterministic_transport_loss(
        model, inputs, targets, FROZEN_CONFIG
    )
    assert loss.one_step_paths == 48 * 4
    assert loss.direct_h8_paths == 40 * 4
    assert loss.direct_h8_targets == 40 * 4 * 4
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


def test_one_update_training_replays_bit_identically() -> None:
    variants = _variants()
    schedule = map_parent_schedule(_schedule_payload(variants), variants)
    seed = FROZEN_CONFIG.paired_model_seeds[0]
    left = _fit_schedule_prefix(
        variants,
        _scalers(),
        schedule,
        model_seed=seed,
        update_count=1,
        checkpoint_updates=(1,),
    )
    right = _fit_schedule_prefix(
        variants,
        _scalers(),
        schedule,
        model_seed=seed,
        update_count=1,
        checkpoint_updates=(1,),
    )
    assert left.log == right.log
    assert tensor_state_sha256(left.model.state_dict()) == tensor_state_sha256(
        right.model.state_dict()
    )
    assert tensor_state_sha256(left.checkpoint_states[1]) == tensor_state_sha256(
        right.checkpoint_states[1]
    )
    assert left.schedule_sha256 == right.schedule_sha256


def test_tampered_or_unavailable_parent_schedule_is_rejected() -> None:
    variants = _variants()
    payload = _schedule_payload(variants)
    payload["updates"][0]["references"][0]["aligned_start"] = 16  # type: ignore[index]
    try:
        map_parent_schedule(payload, variants)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("tampered schedule was accepted")


def test_training_source_lock_rejects_mixed_source_resume(
    tmp_path: Path,
) -> None:
    output = tmp_path / "training"
    output.mkdir()
    payload = {
        "schema": train_grid.SOURCE_LOCK_SCHEMA,
        "parent_package": {"canonical_digest": "a" * 64},
        "v3_training_source_manifest": {"train.py": "b" * 64},
        "shared_runtime_source_manifest": {"sha256": "c" * 64},
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
    }
    path = train_grid._bind_training_source(output, payload)
    assert json.loads(path.read_text(encoding="ascii")) == payload
    assert train_grid._bind_training_source(output, payload) == path
    changed = {
        **payload,
        "v3_training_source_manifest": {"train.py": "d" * 64},
    }
    with pytest.raises(ValueError, match="mixed-source resume"):
        train_grid._bind_training_source(output, changed)

    legacy = tmp_path / "legacy-training"
    (legacy / "case").mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing retroactive binding"):
        train_grid._bind_training_source(legacy, payload)
