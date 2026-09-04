from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from .baselines import (
    BASELINE_RESULT_COLUMNS,
    ARXFeatureSpec,
    baseline_producer_code_manifest,
    DirectH8GRU,
    DirectH8FeatureSpec,
    arx_one_step_features,
    direct_h8_dataset,
    direct_h8_features,
    evaluate_arx_h8,
    evaluate_direct_h8_ridge,
    evaluate_direct_h8_gru,
    fit_arx_ridge,
    fit_direct_h8_gru,
    fit_direct_h8_ridge,
    iterative_arx_h8_prediction,
)
from .fault_data import (
    FaultCell,
    FaultScalers,
    FaultVariant,
    ScaleStats,
    TrajectoryKey,
)
from .study_config import StudyConfig


def make_variant(role: str, day: int, offset: float = 0.0) -> FaultVariant:
    steps = 192
    clean = np.arange(steps * 4, dtype=float).reshape(steps, 4) / 10.0 + offset
    corrupted = clean.copy()
    corrupted[48:96, 0] += 1.0
    actions = (np.arange(steps) % 3 - 1).astype(float).reshape(-1, 1)
    contexts = np.arange(steps * 5, dtype=float).reshape(steps, 5) / 20.0
    return FaultVariant(
        cell=FaultCell(
            cell_id=f"case:{role}:{day}",
            trajectory=TrajectoryKey("case", role, day, 100 + day),
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family="bias",
            sign=1,
            severity=1.0,
            severity_unit="K",
            onset=48,
            stop=96,
            anchors=(56, 64),
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


def scalers() -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "a" * 64),),
    )


def nonidentity_scalers() -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((10.0, 20.0, 30.0, 40.0), (2.0, 4.0, 5.0, 10.0)),
        action=ScaleStats((0.5,), (2.0,)),
        context=ScaleStats(
            (1.0, 2.0, 3.0, 4.0, 5.0),
            (2.0, 4.0, 5.0, 8.0, 10.0),
        ),
        fit_source_sha256=(("case:fit", "a" * 64),),
    )


def test_direct_features_use_exact_causal_history_and_future_action_context():
    variant = make_variant("fit", 10)
    features = direct_h8_features(variant, 56, scalers())
    expected = np.concatenate(
        [
            variant.corrupted_observations[17:57].reshape(-1),
            np.ones((40, 4)).reshape(-1),
            np.zeros((40, 4)).reshape(-1),
            variant.actions[16:56].reshape(-1),
            variant.contexts[17:57].reshape(-1),
            variant.actions[56:64].reshape(-1),
            variant.contexts[57:65].reshape(-1),
        ]
    )
    np.testing.assert_array_equal(features, expected)
    changed_future_observation = replace(
        variant, corrupted_observations=variant.corrupted_observations.copy()
    )
    changed_future_observation.corrupted_observations[57:] += 1_000_000.0
    np.testing.assert_array_equal(
        direct_h8_features(changed_future_observation, 56, scalers()), features
    )
    changed_pre_window = replace(
        variant,
        corrupted_observations=variant.corrupted_observations.copy(),
        actions=variant.actions.copy(),
        contexts=variant.contexts.copy(),
    )
    changed_pre_window.corrupted_observations[:17] += 1_000_000.0
    changed_pre_window.actions[:16] += 1_000_000.0
    changed_pre_window.contexts[:17] += 1_000_000.0
    np.testing.assert_array_equal(
        direct_h8_features(changed_pre_window, 56, scalers()), features
    )

    changed_context_history = replace(variant, contexts=variant.contexts.copy())
    changed_context_history.contexts[17] += 1.0
    assert not np.array_equal(
        direct_h8_features(changed_context_history, 56, scalers()), features
    )


def test_dataset_target_is_exact_h8_endpoint_and_ridge_selects_validation_only():
    fit = [make_variant("fit", 10), make_variant("fit", 11, 1.0)]
    validation = [make_variant("validation", 20, 0.5)]
    x, y, references = direct_h8_dataset(fit, scalers())
    assert x.shape[0] == y.shape[0] == len(references) == 4
    first_variant, first_anchor = references[0]
    np.testing.assert_array_equal(
        y[0], fit[first_variant].clean_observations[first_anchor + 8]
    )
    model, scores, receipt = fit_direct_h8_ridge(
        fit, validation, scalers(), alphas=(0.01, 1.0)
    )
    assert model.alpha in {0.01, 1.0}
    assert scores["selected"].sum() == 1
    assert set(scores["alpha"]) == {0.01, 1.0}
    assert receipt.baseline == "direct_h8_ridge"
    frame = evaluate_direct_h8_ridge(
        model, validation, scalers(), receipt, role="validation"
    )
    expected_columns = set(BASELINE_RESULT_COLUMNS) | {
        f"{prefix}_standardized_{index}"
        for prefix in ("target", "prediction")
        for index in range(4)
    }
    assert set(frame.columns) == expected_columns
    assert set(frame["arm"]) == {"direct_h8_ridge"}
    assert set(frame["model_seed"]) == {0}
    wrong_scalers = replace(
        scalers(), observation=ScaleStats((100.0,) * 4, (2.0,) * 4)
    )
    with pytest.raises(ValueError, match="scalers"):
        evaluate_direct_h8_ridge(
            model,
            validation,
            wrong_scalers,
            receipt,
            role="validation",
        )
    producer = baseline_producer_code_manifest()
    assert set(producer["files"]) == {
        "multicase_fault_benchmark/baselines.py",
        "multicase_fault_benchmark/fault_data.py",
        "multicase_fault_benchmark/study_config.py",
        "multicase_fault_benchmark/runtime_provenance.py",
    }
    assert receipt.producer_code_sha256 == producer["sha256"]
    with pytest.raises(ValueError, match="producer-code"):
        evaluate_direct_h8_ridge(
            model,
            validation,
            scalers(),
            replace(receipt, producer_code_sha256="0" * 64),
            role="validation",
        )
    changed_runtime = {
        **receipt.runtime_fingerprint,
        "python_version": "0.0.0",
    }
    with pytest.raises(ValueError, match="numerical runtime fingerprint"):
        evaluate_direct_h8_ridge(
            model,
            validation,
            scalers(),
            replace(receipt, runtime_fingerprint=changed_runtime),
            role="validation",
        )


def test_arx_features_apply_fit_standardization_and_roll_exactly_eight_steps():
    variant = make_variant("fit", 10)
    scale = nonidentity_scalers()
    features = arx_one_step_features(variant, 56, scale)
    expected = np.concatenate(
        [
            scale.observation.transform(
                variant.corrupted_observations[49:57]
            ).reshape(-1),
            np.ones((8, 4)).reshape(-1),
            np.zeros((8, 4)).reshape(-1),
            scale.action.transform(variant.actions[48:56]).reshape(-1),
            scale.action.transform(variant.actions[56:57]).reshape(-1),
            scale.context.transform(variant.contexts[56:58]).reshape(-1),
        ]
    )
    np.testing.assert_allclose(features, expected, rtol=0.0, atol=0.0)

    class CountingModel:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def predict(self, value: np.ndarray) -> np.ndarray:
            self.calls.append(value.copy())
            return np.full((len(value), 4), float(len(self.calls)))

    model = CountingModel()
    final = iterative_arx_h8_prediction(
        model, variant, 56, scalers(), ARXFeatureSpec()
    )
    assert len(model.calls) == 8
    np.testing.assert_array_equal(final, np.full(4, 8.0))
    second_observation_history = model.calls[1][0, : 8 * 4].reshape(8, 4)
    np.testing.assert_array_equal(second_observation_history[-1], np.ones(4))


def test_arx_is_standardized_causal_iterative_h8_and_emits_evaluator_schema():
    fit = [make_variant("fit", 10), make_variant("fit", 11, 1.0)]
    validation = [make_variant("validation", 20, 0.5)]
    spec = ARXFeatureSpec(history=8, horizon=8)
    features = arx_one_step_features(fit[0], 56, scalers(), spec)
    expected = np.concatenate(
        [
            fit[0].corrupted_observations[49:57].reshape(-1),
            np.ones((8, 4)).reshape(-1),
            np.zeros((8, 4)).reshape(-1),
            fit[0].actions[48:56].reshape(-1),
            fit[0].actions[56:57].reshape(-1),
            fit[0].contexts[56:58].reshape(-1),
        ]
    )
    np.testing.assert_array_equal(features, expected)

    model, scores, receipt = fit_arx_ridge(
        fit, validation, scalers(), alphas=(0.01, 1.0), spec=spec
    )
    assert scores["selected"].sum() == 1
    assert receipt.baseline == "ridge_arx"
    assert receipt.fit_role == "fit" and receipt.validation_role == "validation"
    assert len(receipt.sha256) == 64

    anchor = 56
    original = iterative_arx_h8_prediction(
        model, validation[0], anchor, scalers(), spec
    )
    corrupted = validation[0].corrupted_observations.copy()
    corrupted[anchor + 1 :] += 1_000_000.0
    future_changed = replace(validation[0], corrupted_observations=corrupted)
    changed = iterative_arx_h8_prediction(
        model, future_changed, anchor, scalers(), spec
    )
    np.testing.assert_allclose(original, changed, rtol=0.0, atol=0.0)

    frame = evaluate_arx_h8(
        model, validation, scalers(), receipt, role="validation", spec=spec
    )
    expected_columns = set(BASELINE_RESULT_COLUMNS) | {
        f"{prefix}_standardized_{index}"
        for prefix in ("target", "prediction")
        for index in range(4)
    }
    assert set(frame.columns) == expected_columns
    assert set(frame["arm"]) == {"ridge_arx"}
    assert set(frame["model_seed"]) == {0}
    assert set(frame["update"]) == {0}
    assert np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
    wrong_scalers = replace(
        scalers(), observation=ScaleStats((100.0,) * 4, (2.0,) * 4)
    )
    with pytest.raises(ValueError, match="scalers"):
        evaluate_arx_h8(
            model,
            validation,
            wrong_scalers,
            receipt,
            role="validation",
            spec=spec,
        )


def test_arx_and_gru_selection_reject_non_fit_or_non_validation_roles():
    fit = [make_variant("fit", 10)]
    validation = [make_variant("validation", 20)]
    locked = [make_variant("locked_test", 30)]
    with pytest.raises(ValueError, match="validation"):
        fit_arx_ridge(fit, locked, scalers(), alphas=(1.0,))
    with pytest.raises(ValueError, match="FIT"):
        fit_arx_ridge(locked, validation, scalers(), alphas=(1.0,))
    mismatched_validation = [
        replace(
            validation[0],
            cell=replace(
                validation[0].cell,
                trajectory=TrajectoryKey("other", "validation", 20, 120),
            ),
        )
    ]
    with pytest.raises(ValueError, match="matching case"):
        fit_direct_h8_ridge(
            fit, mismatched_validation, scalers(), alphas=(1.0,)
        )

    config = replace(
        StudyConfig(),
        hidden_dim=8,
        batch_size=2,
        gru_batch_size=80,
        updates=2,
        checkpoint_every=1,
        validation_checkpoints=(1, 2),
    )
    with pytest.raises(ValueError, match="validation"):
        fit_direct_h8_gru(
            fit,
            locked,
            scalers(),
            config,
            model_seed=config.development_seeds[0],
        )
    with pytest.raises(ValueError, match="seed"):
        fit_direct_h8_gru(
            fit, validation, scalers(), config, model_seed=123
        )


def test_gru_uses_direct_feature_contract_is_deterministic_and_validation_selected():
    fit = [make_variant("fit", 10), make_variant("fit", 11, 1.0)]
    validation_variant = make_variant("validation", 20, 0.5)
    validation_variant = replace(
        validation_variant,
        cell=replace(validation_variant.cell, anchors=(56,)),
    )
    validation = [validation_variant]
    config = replace(
        StudyConfig(),
        hidden_dim=8,
        batch_size=2,
        gru_batch_size=80,
        updates=4,
        checkpoint_every=2,
        validation_checkpoints=(2, 4),
    )
    spec = DirectH8FeatureSpec()
    feature = direct_h8_features(fit[0], 56, scalers(), spec)
    shape_model = DirectH8GRU(hidden_dim=config.hidden_dim, spec=spec)
    assert feature.shape == (shape_model.expected_feature_dim,)
    assert shape_model(torch.as_tensor(feature[None], dtype=torch.float32)).shape == (
        1,
        4,
    )

    first = fit_direct_h8_gru(
        fit,
        validation,
        scalers(),
        config,
        model_seed=config.development_seeds[0],
        spec=spec,
    )
    second = fit_direct_h8_gru(
        fit,
        validation,
        scalers(),
        config,
        model_seed=config.development_seeds[0],
        spec=spec,
    )
    assert first.receipt.selected_model_state_sha256 == (
        second.receipt.selected_model_state_sha256
    )
    assert first.receipt.schedule_sha256 == second.receipt.schedule_sha256
    assert first.receipt.candidate_grid == config.validation_checkpoints
    assert first.receipt.selected_candidate in config.validation_checkpoints
    assert first.score_table.equals(second.score_table)
    assert len(first.training_log) == config.updates
    assert first.score_table["selected"].sum() == 1

    original = evaluate_direct_h8_gru(
        first.model,
        validation,
        scalers(),
        first.receipt,
        role="validation",
    )
    corrupted = validation_variant.corrupted_observations.copy()
    corrupted[57:] += 1_000_000.0
    changed_validation = [
        replace(validation_variant, corrupted_observations=corrupted)
    ]
    future_changed = evaluate_direct_h8_gru(
        first.model,
        changed_validation,
        scalers(),
        first.receipt,
        role="validation",
    )
    np.testing.assert_allclose(
        original["prediction_raw"],
        future_changed["prediction_raw"],
        rtol=0.0,
        atol=0.0,
    )
    assert set(original["arm"]) == {"deterministic_gru"}
    assert set(original["model_seed"]) == {config.development_seeds[0]}
    assert set(original["update"]) == {first.receipt.selected_candidate}
    expected_columns = set(BASELINE_RESULT_COLUMNS) | {
        f"{prefix}_standardized_{index}"
        for prefix in ("target", "prediction")
        for index in range(4)
    }
    assert set(original.columns) == expected_columns
    wrong_scalers = replace(
        scalers(), observation=ScaleStats((100.0,) * 4, (2.0,) * 4)
    )
    with pytest.raises(ValueError, match="scalers"):
        evaluate_direct_h8_gru(
            first.model,
            validation,
            wrong_scalers,
            first.receipt,
            role="validation",
        )
