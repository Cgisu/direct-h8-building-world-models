from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .fault_data import (
    FAMILIES,
    FAULT_CHANNELS,
    FaultCell,
    FaultScalers,
    FaultVariant,
    ScaleStats,
    TrajectoryKey,
)
from .study_config import ARMS, StudyConfig
from .reliability_loss import reliability_sequence_training_loss
from .runtime_provenance import _canonical_sha256, numerical_runtime_fingerprint
from . import study_train as study_train_module
from .study_train import (
    canonical_payload_sha256,
    core_tensor_state_sha256,
    create_matched_models,
    make_training_schedule,
    materialize_torch_batch,
    rssm_producer_code_manifest,
    tensor_state_sha256,
    training_provenance,
    validate_rssm_producer_code_manifest,
)


def make_variants() -> list[FaultVariant]:
    variants = []
    time_steps = 192
    observations = np.arange(time_steps * 4, dtype=float).reshape(time_steps, 4)
    actions = (np.arange(time_steps, dtype=float) % 3 - 1).reshape(-1, 1)
    contexts = np.arange(time_steps * 5, dtype=float).reshape(time_steps, 5)
    for channel_index, channel in enumerate(FAULT_CHANNELS):
        for family_index, family in enumerate(FAMILIES):
            cell = FaultCell(
                cell_id=f"case:{channel}:{family}",
                trajectory=TrajectoryKey("case", "fit", 10, 100),
                source_sha256="a" * 64,
                fault_channel=channel,
                family=family,
                sign=0,
                severity=0.0,
                severity_unit="none",
                onset=48,
                stop=96,
                anchors=(56, 64, 72, 80),
            )
            corrupted = observations.copy()
            availability = np.ones_like(corrupted, dtype=bool)
            age = np.zeros_like(corrupted)
            labels = np.zeros((time_steps, 2), dtype=np.int64)
            if family != "healthy":
                corrupted[48:96, channel_index] += 100.0 + family_index
                labels[48:96, channel_index] = family_index
            if family == "dropout":
                corrupted[48:96, channel_index] = np.nan
                availability[48:96, channel_index] = False
                age[48:96, channel_index] = np.arange(1, 49)
            variants.append(
                FaultVariant(
                    cell=cell,
                    source=None,
                    clean_observations=observations,
                    corrupted_observations=corrupted,
                    actions=actions,
                    contexts=contexts,
                    availability=availability,
                    age=age,
                    health_labels=labels,
                )
            )
    return variants


def unit_scalers() -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "a" * 64),),
    )


def test_matched_initialization_is_exact_and_storage_is_independent():
    config = StudyConfig()
    models = create_matched_models(config, config.development_seeds[0])
    assert len({tensor_state_sha256(model.state_dict()) for model in models.values()}) == 1
    first, second = models[ARMS[0]], models[ARMS[1]]
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert left.data_ptr() != right.data_ptr()
    with torch.no_grad():
        next(first.parameters()).add_(1.0)
    assert tensor_state_sha256(first.state_dict()) != tensor_state_sha256(second.state_dict())


def test_schedule_is_deterministic_balanced_and_shared_across_arms():
    variants = make_variants()
    config = StudyConfig(
        updates=10,
        checkpoint_every=10,
        validation_checkpoints=(10,),
    )
    first = make_training_schedule(
        variants, config, case="case", model_seed=config.development_seeds[0]
    )
    second = make_training_schedule(
        variants, config, case="case", model_seed=config.development_seeds[0]
    )
    assert first == second
    selected = Counter(
        (
            variants[reference.variant_index].cell.fault_channel,
            variants[reference.variant_index].cell.family,
        )
        for batch in first
        for reference in batch.references
    )
    assert max(selected.values()) - min(selected.values()) == 0
    assert set(selected) == {
        (channel, family) for channel in FAULT_CHANNELS for family in FAMILIES
    }


def test_materialized_training_batch_is_aligned_and_binary():
    variants = make_variants()
    config = StudyConfig(
        updates=10,
        checkpoint_every=10,
        validation_checkpoints=(10,),
    )
    scheduled = make_training_schedule(
        variants, config, case="case", model_seed=config.development_seeds[0]
    )[0]
    inputs, targets = materialize_torch_batch(
        variants, unit_scalers(), scheduled, config
    )
    assert inputs.previous_actions.shape == (48, 4, 1)
    assert inputs.corrupted_observations.shape == (48, 4, 4)
    assert inputs.contexts.shape == (48, 4, 5)
    assert targets.health_labels.shape == (48, 4, 2)
    assert set(torch.unique(targets.health_labels).tolist()) <= {0, 1}
    assert targets.valid_steps.all()
    for batch_index, reference in enumerate(scheduled.references):
        variant = variants[reference.variant_index]
        np.testing.assert_array_equal(
            inputs.previous_actions[:, batch_index].numpy(),
            variant.actions[reference.aligned_start : reference.aligned_start + 48],
        )
        np.testing.assert_array_equal(
            targets.clean_observations[:, batch_index].numpy(),
            variant.clean_observations[
                reference.aligned_start + 1 : reference.aligned_start + 49
            ],
        )


def test_canonical_payload_hash_is_order_independent_and_value_sensitive():
    assert canonical_payload_sha256({"b": 2, "a": [1, 3]}) == (
        canonical_payload_sha256({"a": [1, 3], "b": 2})
    )


def test_rssm_producer_manifest_binds_all_tensor_producer_files():
    manifest = rssm_producer_code_manifest()
    assert set(manifest["files"]) == {
        "multicase_fault_benchmark/study_train.py",
        "multicase_fault_benchmark/reliability_model.py",
        "multicase_fault_benchmark/reliability_loss.py",
        "multicase_fault_benchmark/fault_data.py",
        "multicase_fault_benchmark/study_config.py",
        "multicase_fault_benchmark/runtime_provenance.py",
        "health_rssm/model.py",
        "health_rssm/training.py",
    }
    validate_rssm_producer_code_manifest(manifest)
    changed = {
        **manifest,
        "files": {**manifest["files"], "health_rssm/model.py": "0" * 64},
    }
    with np.testing.assert_raises_regex(ValueError, "producer-code manifest"):
        validate_rssm_producer_code_manifest(changed)
    assert canonical_payload_sha256({"a": [1, 3], "b": 2}) != (
        canonical_payload_sha256({"a": [1, 4], "b": 2})
    )


def test_recorded_foreign_cuda_runtime_does_not_touch_current_cuda(monkeypatch):
    runtime = numerical_runtime_fingerprint("cpu", include_sklearn=False)
    runtime.update(
        {
            "torch_cuda_version": "12.8",
            "cudnn_version": 91002,
            "device": {
                "type": "cuda",
                "index": 7,
                "name": "Foreign GPU",
                "capability": [9, 0],
            },
        }
    )
    runtime["sha256"] = _canonical_sha256(
        {key: value for key, value in runtime.items() if key != "sha256"}
    )

    def fail_if_current_runtime_is_read(*args, **kwargs):
        raise AssertionError("recorded runtime must not resolve current CUDA")

    monkeypatch.setattr(
        study_train_module,
        "numerical_runtime_fingerprint",
        fail_if_current_runtime_is_read,
    )
    provenance = training_provenance(
        SimpleNamespace(manifest_sha256="a" * 64),
        SimpleNamespace(sha256="b" * 64),
        unit_scalers(),
        StudyConfig(),
        {"sha256": "c" * 64},
        recorded_runtime=runtime,
    )
    assert provenance["runtime"] == runtime
    with pytest.raises(ValueError, match="exactly one"):
        training_provenance(
            SimpleNamespace(manifest_sha256="a" * 64),
            SimpleNamespace(sha256="b" * 64),
            unit_scalers(),
            StudyConfig(),
            {"sha256": "c" * 64},
            device="cpu",
            recorded_runtime=runtime,
        )


def test_auxiliary_negative_control_cannot_change_the_rssm_core():
    variants = make_variants()
    config = StudyConfig(
        updates=10,
        checkpoint_every=10,
        validation_checkpoints=(10,),
    )
    scheduled = make_training_schedule(
        variants, config, case="case", model_seed=config.development_seeds[0]
    )[0]
    inputs, targets = materialize_torch_batch(
        variants, unit_scalers(), scheduled, config
    )
    models = create_matched_models(
        config,
        config.development_seeds[0],
        arms=("ungated_h8", "aux_h8"),
    )
    optimizers = {
        arm: torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for arm, model in models.items()
    }
    for arm in ("ungated_h8", "aux_h8"):
        torch.manual_seed(scheduled.latent_seed)
        arm_config = config.arm_config(arm)
        output = reliability_sequence_training_loss(
            models[arm],
            inputs,
            targets,
            config.loss_config(arm),
            gate_mode=arm_config.gate_mode,
            sample=True,
        )
        output.loss.total.backward()
        optimizers[arm].step()

    left = models["ungated_h8"].state_dict()
    right = models["aux_h8"].state_dict()
    gate_prefixes = ("reliability_feature_net.", "health_head.")
    core_names = [name for name in left if not name.startswith(gate_prefixes)]
    gate_names = [name for name in left if name.startswith(gate_prefixes)]
    assert core_names and gate_names
    assert all(torch.equal(left[name], right[name]) for name in core_names)
    assert any(not torch.equal(left[name], right[name]) for name in gate_names)
    assert core_tensor_state_sha256(left) == core_tensor_state_sha256(right)
