from __future__ import annotations

import hashlib
import copy

import numpy as np
import pandas as pd
import torch

from .fault_data import (
    FaultCell,
    FaultScalers,
    FaultVariant,
    ScaleStats,
    TrajectoryKey,
)
from .reliability_model import ReliabilityGatedRSSM
from .study_config import StudyConfig
from .study_evaluate import (
    _alternate_action_block,
    _validation_fault_signatures,
    evaluate_model_h8,
    load_model_checkpoint,
    validation_selection_scores,
)
from .fault_data import FAMILIES, FAULT_CHANNELS
from .protocol import CASES
from .runtime_provenance import numerical_runtime_fingerprint
from .study_train import (
    core_tensor_state_sha256,
    rssm_producer_code_manifest,
    tensor_state_sha256,
)


class RecordingModel(ReliabilityGatedRSSM):
    def __init__(self, config):
        super().__init__(config)
        self.filtered_inputs = None
        self.imagined_inputs = []

    def filter(self, previous_actions, observations, masks, ages, contexts=None, **kwargs):
        self.filtered_inputs = (
            previous_actions.detach().clone(),
            observations.detach().clone(),
            contexts.detach().clone(),
        )
        return super().filter(
            previous_actions, observations, masks, ages, contexts, **kwargs
        )

    def imagine(self, start_state, actions, future_contexts=None, **kwargs):
        self.imagined_inputs.append(
            (actions.detach().clone(), future_contexts.detach().clone())
        )
        return super().imagine(start_state, actions, future_contexts, **kwargs)


def test_evaluator_uses_previous_action_history_and_exact_future_h8_slice():
    config = StudyConfig(
        updates=100,
        checkpoint_every=100,
        validation_checkpoints=(100,),
    )
    model = RecordingModel(config.model_config())
    steps = 192
    clean = np.arange(steps * 4, dtype=float).reshape(steps, 4) / 100.0
    corrupted = clean.copy()
    corrupted[48:96, 0] += 1.0
    actions = (np.arange(steps) % 3 - 1).astype(float).reshape(-1, 1)
    contexts = np.arange(steps * 5, dtype=float).reshape(steps, 5) / 50.0
    variant = FaultVariant(
        cell=FaultCell(
            cell_id="case:validation:zone:bias",
            trajectory=TrajectoryKey("case", "validation", 10, 100),
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family="bias",
            sign=1,
            severity=1.0,
            severity_unit="K",
            onset=48,
            stop=96,
            anchors=(56,),
        ),
        source=None,
        clean_observations=clean,
        corrupted_observations=corrupted,
        actions=actions,
        contexts=contexts,
        availability=np.ones_like(clean, dtype=bool),
        age=np.zeros_like(clean),
        health_labels=np.zeros((steps, 2), dtype=np.int64),
    )
    scalers = FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "b" * 64),),
    )
    frame = evaluate_model_h8(
        model,
        [variant],
        scalers,
        config,
        arm="gated_h8",
        case="case",
        model_seed=config.development_seeds[0],
        update=100,
        role="validation",
    )
    assert len(frame) == 1 and frame.loc[0, "anchor"] == 56
    filtered_actions, filtered_observations, filtered_contexts = model.filtered_inputs
    assert len(filtered_actions) == 40
    np.testing.assert_array_equal(filtered_actions[:, 0].numpy(), actions[16:56])
    np.testing.assert_allclose(
        filtered_observations[:, 0].numpy(),
        corrupted[17:57],
        rtol=0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        filtered_contexts[:, 0].numpy(), contexts[17:57], rtol=0.0, atol=5e-7
    )
    assert len(model.imagined_inputs) == 2
    imagined_actions, imagined_contexts = model.imagined_inputs[0]
    np.testing.assert_array_equal(imagined_actions[:, 0].numpy(), actions[56:64])
    np.testing.assert_allclose(
        imagined_contexts[:, 0].numpy(), contexts[57:65], rtol=0.0, atol=5e-7
    )
    alternate_actions, alternate_contexts = model.imagined_inputs[1]
    np.testing.assert_array_equal(alternate_actions[:, 0].numpy(), actions[64:72])
    np.testing.assert_array_equal(
        alternate_contexts[:, 0].numpy(), imagined_contexts[:, 0].numpy()
    )
    assert frame.loc[0, "target_raw"] == clean[64, 0]
    assert np.isfinite(frame.loc[0, "action_prediction_change_standardized"])

    reference_prediction = frame.loc[0, "prediction_raw"]
    pre_window_variant = copy.deepcopy(variant)
    pre_window_variant.corrupted_observations[:17] += 1_000_000.0
    pre_window_variant.actions[:16] += 1_000_000.0
    pre_window_variant.contexts[:17] += 1_000_000.0
    repeated = evaluate_model_h8(
        model,
        [pre_window_variant],
        scalers,
        config,
        arm="gated_h8",
        case="case",
        model_seed=config.development_seeds[0],
        update=100,
        role="validation",
    )
    assert repeated.loc[0, "prediction_raw"] == reference_prediction


def test_alternate_action_diagnostic_skips_an_identical_immediate_block():
    config = StudyConfig()
    variant = _test_variant_for_actions()
    actions = variant.actions.copy()
    actions[56:64] = -1.0
    actions[64:72] = -1.0
    actions[72:80] = 1.0
    variant = FaultVariant(**{**variant.__dict__, "actions": actions})
    np.testing.assert_array_equal(
        _alternate_action_block(variant, 56, config), actions[72:80]
    )


def test_evaluator_uses_the_same_frozen_window_at_a_late_anchor():
    config = StudyConfig(
        updates=100,
        checkpoint_every=100,
        validation_checkpoints=(100,),
    )
    model = RecordingModel(config.model_config())
    steps = 192
    clean = np.arange(steps * 4, dtype=float).reshape(steps, 4) / 100.0
    corrupted = clean.copy()
    corrupted[48:176, 0] += 1.0
    actions = (np.arange(steps) % 3 - 1).astype(float).reshape(-1, 1)
    contexts = np.arange(steps * 5, dtype=float).reshape(steps, 5) / 50.0
    variant = FaultVariant(
        cell=FaultCell(
            cell_id="case:validation:late-anchor",
            trajectory=TrajectoryKey("case", "validation", 10, 100),
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family="bias",
            sign=1,
            severity=1.0,
            severity_unit="K",
            onset=48,
            stop=176,
            anchors=(144,),
        ),
        source=None,
        clean_observations=clean,
        corrupted_observations=corrupted,
        actions=actions,
        contexts=contexts,
        availability=np.ones_like(clean, dtype=bool),
        age=np.zeros_like(clean),
        health_labels=np.zeros((steps, 2), dtype=np.int64),
    )
    scalers = FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "b" * 64),),
    )

    frame = evaluate_model_h8(
        model,
        [variant],
        scalers,
        config,
        arm="gated_h8",
        case="case",
        model_seed=config.development_seeds[0],
        update=100,
        role="validation",
    )

    filtered_actions, filtered_observations, filtered_contexts = model.filtered_inputs
    np.testing.assert_array_equal(filtered_actions[:, 0].numpy(), actions[104:144])
    np.testing.assert_allclose(
        filtered_observations[:, 0].numpy(), corrupted[105:145], rtol=0.0, atol=3e-7
    )
    np.testing.assert_allclose(
        filtered_contexts[:, 0].numpy(), contexts[105:145], rtol=0.0, atol=2e-6
    )
    imagined_actions, imagined_contexts = model.imagined_inputs[0]
    np.testing.assert_array_equal(imagined_actions[:, 0].numpy(), actions[144:152])
    np.testing.assert_allclose(
        imagined_contexts[:, 0].numpy(), contexts[145:153], rtol=0.0, atol=2e-6
    )
    assert frame.loc[0, "target_raw"] == clean[152, 0]


def _test_variant_for_actions() -> FaultVariant:
    steps = 192
    clean = np.zeros((steps, 4), dtype=float)
    return FaultVariant(
        cell=FaultCell(
            cell_id="case:validation:actions",
            trajectory=TrajectoryKey("case", "validation", 10, 100),
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family="healthy",
            sign=0,
            severity=0.0,
            severity_unit="none",
            onset=48,
            stop=96,
            anchors=(56,),
        ),
        source=None,
        clean_observations=clean,
        corrupted_observations=clean.copy(),
        actions=np.zeros((steps, 1), dtype=float),
        contexts=np.zeros((steps, 5), dtype=float),
        availability=np.ones_like(clean, dtype=bool),
        age=np.zeros_like(clean),
        health_labels=np.zeros((steps, 2), dtype=np.int64),
    )


def test_checkpoint_loader_requires_file_training_and_model_provenance(tmp_path):
    config = StudyConfig(
        updates=100,
        checkpoint_every=100,
        validation_checkpoints=(100,),
    )
    model = ReliabilityGatedRSSM(config.model_config())
    provenance = {
        "corpus_manifest_sha256": "a" * 64,
        "fault_manifest_sha256": "b" * 64,
        "fit_scalers_sha256": "c" * 64,
        "fit_source_sha256": "d" * 64,
        "fit_source_sha256_by_trajectory": {"case:fit": "e" * 64},
        "training_schedule_sha256": "f" * 64,
        "config_sha256": "1" * 64,
        "producer_code": rssm_producer_code_manifest(),
        "runtime": numerical_runtime_fingerprint(
            "cpu", include_sklearn=False
        ),
    }
    payload = {
        "schema": "boptest-reliability-rssm-checkpoint-v2",
        "case": "case",
        "model_seed": config.development_seeds[0],
        "arm": "gated_h8",
        "update": 100,
        "config": config.to_dict(),
        "provenance": provenance,
        "model_state_sha256": tensor_state_sha256(model.state_dict()),
        "core_state_sha256": core_tensor_state_sha256(model.state_dict()),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_model_checkpoint(
        path,
        config,
        case="case",
        model_seed=config.development_seeds[0],
        arm="gated_h8",
        update=100,
        expected_checkpoint_sha256=file_sha256,
        expected_provenance=provenance,
    )
    assert tensor_state_sha256(loaded.state_dict()) == payload["model_state_sha256"]

    with np.testing.assert_raises_regex(ValueError, "file SHA-256"):
        load_model_checkpoint(
            path,
            config,
            case="case",
            model_seed=config.development_seeds[0],
            arm="gated_h8",
            update=100,
            expected_checkpoint_sha256="0" * 64,
            expected_provenance=provenance,
        )
    changed = {**provenance, "fit_scalers_sha256": "0" * 64}
    with np.testing.assert_raises_regex(ValueError, "training provenance"):
        load_model_checkpoint(
            path,
            config,
            case="case",
            model_seed=config.development_seeds[0],
            arm="gated_h8",
            update=100,
            expected_checkpoint_sha256=file_sha256,
            expected_provenance=changed,
        )

    runtime_tampered = copy.deepcopy(payload)
    runtime_provenance = runtime_tampered["provenance"]
    runtime_provenance["runtime"]["python_version"] = "0.0.0"
    torch.save(runtime_tampered, path)
    runtime_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.testing.assert_raises_regex(ValueError, "numerical runtime fingerprint"):
        load_model_checkpoint(
            path,
            config,
            case="case",
            model_seed=config.development_seeds[0],
            arm="gated_h8",
            update=100,
            expected_checkpoint_sha256=runtime_file_sha256,
            expected_provenance=runtime_provenance,
        )

    producer_tampered = copy.deepcopy(payload)
    producer_provenance = producer_tampered["provenance"]
    first_file = next(iter(producer_provenance["producer_code"]["files"]))
    producer_provenance["producer_code"]["files"][first_file] = "0" * 64
    torch.save(producer_tampered, path)
    producer_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.testing.assert_raises_regex(ValueError, "producer-code manifest"):
        load_model_checkpoint(
            path,
            config,
            case="case",
            model_seed=config.development_seeds[0],
            arm="gated_h8",
            update=100,
            expected_checkpoint_sha256=producer_file_sha256,
            expected_provenance=producer_provenance,
        )

    torch.save(payload, path)
    payload["model_state_sha256"] = "0" * 64
    torch.save(payload, path)
    changed_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.testing.assert_raises_regex(ValueError, "model-state SHA-256"):
        load_model_checkpoint(
            path,
            config,
            case="case",
            model_seed=config.development_seeds[0],
            arm="gated_h8",
            update=100,
            expected_checkpoint_sha256=changed_file_sha256,
            expected_provenance=provenance,
        )


def test_validation_selection_requires_the_complete_frozen_grid():
    config = StudyConfig()
    rows = []
    signatures = _validation_fault_signatures(config)
    for update in config.validation_checkpoints:
        for arm in ("ungated_h8", "gated_h8"):
            for case in CASES:
                for model_seed in config.development_seeds:
                    for trajectory_index in range(8):
                        day = 10 + trajectory_index
                        trajectory_seed = 1000 + trajectory_index
                        for (
                            channel,
                            family,
                            sign,
                            severity,
                            severity_unit,
                            onset,
                            anchor,
                            horizon,
                        ) in signatures:
                            error = float(update) / 1000.0
                            if arm == "gated_h8":
                                error -= 0.01
                            rows.append(
                                {
                                    "update": update,
                                    "arm": arm,
                                    "case": case,
                                    "role": "validation",
                                    "model_seed": model_seed,
                                    "trajectory_day": day,
                                    "trajectory_seed": trajectory_seed,
                                    "cell_id": (
                                        f"{case}:{day}:{family}:{channel}:{sign}:"
                                        f"{severity}:{onset}"
                                    ),
                                    "anchor": anchor,
                                    "family": family,
                                    "fault_channel": channel,
                                    "sign": sign,
                                    "severity": severity,
                                    "severity_unit": severity_unit,
                                    "onset": onset,
                                    "horizon": horizon,
                                    "target_raw": 1.0,
                                    "persistence_prediction_raw": 0.0,
                                    "persistence_standardized_abs_error": 1.0,
                                    "standardized_abs_error": error,
                                }
                            )
    frame = pd.DataFrame(rows)
    scores = validation_selection_scores(frame, config)
    assert set(scores["update"]) == set(config.validation_checkpoints)
    assert scores["selected"].sum() == 1

    missing_update = frame[frame["update"] != config.validation_checkpoints[-1]]
    with np.testing.assert_raises_regex(ValueError, "complete frozen update grid"):
        validation_selection_scores(missing_update, config)
    missing_identity = frame.drop(index=frame.index[0])
    with np.testing.assert_raises_regex(ValueError, "frozen validation fault grid"):
        validation_selection_scores(missing_identity, config)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        validation_selection_scores(duplicate, config)
    inconsistent = frame.copy()
    inconsistent.loc[inconsistent.index[0], "target_raw"] = 2.0
    with np.testing.assert_raises_regex(ValueError, "target/persistence"):
        validation_selection_scores(inconsistent, config)
    relabeled = frame.copy()
    relabeled.loc[relabeled.index[0], "cell_id"] += ":changed"
    with np.testing.assert_raises_regex(ValueError, "cell IDs"):
        validation_selection_scores(relabeled, config)
