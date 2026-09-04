from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from building_fault_wm.neural_benchmark.fault_data import (
    FaultCell,
    FaultScalers,
    FaultVariant,
    ScaleStats,
    SequenceReference,
    TrajectoryKey,
)
from building_fault_wm.neural_benchmark.study_train import ScheduledBatch

from .config import CONFIG
from .study import (
    _feature_from_standardized,
    recursive_prediction,
    ridge_path,
    scheduled_dataset,
    selection_boundary_flags,
    _validate_core,
    validate_selection_table,
    validation_score,
)


def _scalers() -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "a" * 64),),
    )


def _variant(
    *,
    role: str = "validation",
    post_anchor_shift: float = 0.0,
) -> FaultVariant:
    steps = 112
    time = np.arange(steps, dtype=float)
    clean = np.stack(
        [
            0.03 * time,
            np.sin(time / 7.0),
            np.cos(time / 9.0),
            (time % 11.0) / 11.0,
        ],
        axis=1,
    )
    corrupted = clean.copy()
    corrupted[32:88, 0] += 0.5
    corrupted[41:, :] += post_anchor_shift
    actions = ((time.astype(int) // 5) % 3 - 1).reshape(-1, 1).astype(float)
    contexts = np.stack(
        [np.sin(time / (index + 5.0)) for index in range(5)], axis=1
    )
    return FaultVariant(
        cell=FaultCell(
            cell_id=f"case:{role}:zone_temperature_k:bias",
            trajectory=TrajectoryKey("case", role, 10, 101),
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family="bias",
            sign=1,
            severity=0.5,
            severity_unit="K",
            onset=32,
            stop=88,
            anchors=(40,),
        ),
        source=Path("synthetic.csv"),
        clean_observations=clean,
        corrupted_observations=corrupted,
        actions=actions,
        contexts=contexts,
        availability=np.ones_like(clean, dtype=bool),
        age=np.zeros_like(clean),
        health_labels=np.zeros_like(clean, dtype=np.int64),
    )


def _action_model(history: int = 8) -> Ridge:
    model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    model.coef_ = np.zeros(
        (CONFIG.observation_dim, CONFIG.feature_dim(history)), dtype=float
    )
    current_action_index = history * (
        3 * CONFIG.observation_dim + CONFIG.action_dim
    )
    model.coef_[0, current_action_index] = 1.0
    model.intercept_ = np.zeros(CONFIG.observation_dim)
    model.n_features_in_ = CONFIG.feature_dim(history)
    return model


def _complete_table() -> pd.DataFrame:
    rows = []
    for history in CONFIG.histories:
        for alpha in CONFIG.alphas:
            rows.append(
                {
                    "history": history,
                    "alpha": alpha,
                    "feature_dim": CONFIG.feature_dim(history),
                    "active_coefficients": CONFIG.active_coefficients(history),
                    "validation_h8_mae": history + alpha,
                    "selected": history == 4 and alpha == 1e-4,
                }
            )
    return pd.DataFrame(rows)


def test_selection_rejects_locked_test_trajectory() -> None:
    with pytest.raises(ValueError, match="non-validation"):
        validation_score(
            _action_model(),
            [_variant(role="locked_test")],
            _scalers(),
            8,
        )


def test_candidate_grid_must_be_complete_and_unique() -> None:
    table = _complete_table()
    validate_selection_table(table)
    with pytest.raises(ValueError, match="incomplete"):
        validate_selection_table(table.iloc[:-1].copy())
    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="incomplete"):
        validate_selection_table(duplicated)


def test_every_lag_reuses_identical_scheduled_source_rows() -> None:
    variant = _variant()
    schedule = (
        ScheduledBatch(
            update=1,
            latent_seed=7,
            references=tuple(SequenceReference(0, 47) for _ in range(4)),
        ),
    )
    x4, y4, groups4, digest4 = scheduled_dataset(
        [variant],
        _scalers(),
        schedule,
        4,
        require_complete=False,
    )
    x40, y40, groups40, digest40 = scheduled_dataset(
        [variant],
        _scalers(),
        schedule,
        40,
        require_complete=False,
    )
    assert x4.shape == (160, CONFIG.feature_dim(4))
    assert x40.shape == (160, CONFIG.feature_dim(40))
    np.testing.assert_array_equal(y4, y40)
    assert groups4 == groups40
    assert digest4 == digest40


def test_causal_lag_alignment_uses_only_source_and_earlier_values() -> None:
    source = 50
    history = 4
    observations = np.arange(120 * 4, dtype=float).reshape(120, 4)
    availability = np.ones_like(observations, dtype=bool)
    age = np.zeros_like(observations)
    actions = np.arange(120, dtype=float).reshape(-1, 1)
    contexts = np.arange(120 * 5, dtype=float).reshape(120, 5)
    features = _feature_from_standardized(
        observations,
        availability,
        age,
        actions,
        contexts,
        source,
        _scalers(),
        history,
    )
    np.testing.assert_array_equal(
        features[: history * 4],
        observations[source - history + 1 : source + 1].reshape(-1),
    )
    action_start = history * 12
    np.testing.assert_array_equal(
        features[action_start : action_start + history],
        actions[source - history : source, 0],
    )
    assert observations[source + 1, 0] not in features[: history * 4]


def test_early_h40_source_is_causally_left_padded() -> None:
    source = 23
    history = 40
    observations = np.arange(120 * 4, dtype=float).reshape(120, 4)
    availability = np.ones_like(observations, dtype=bool)
    features = _feature_from_standardized(
        observations,
        availability,
        np.zeros_like(observations),
        np.arange(120, dtype=float).reshape(-1, 1),
        np.zeros((120, 5), dtype=float),
        source,
        _scalers(),
        history,
    )
    missing_observation_steps = history - (source + 1)
    observation_width = history * 4
    mask_start = observation_width
    assert not features[: missing_observation_steps * 4].any()
    assert not features[
        mask_start : mask_start + missing_observation_steps * 4
    ].any()
    np.testing.assert_array_equal(
        features[
            missing_observation_steps * 4 : observation_width
        ],
        observations[: source + 1].reshape(-1),
    )


def test_recursive_rollout_never_consumes_future_observation() -> None:
    original = _variant()
    changed = _variant(post_anchor_shift=10_000.0)
    actions = np.full((8, 1), 0.25)
    left = recursive_prediction(
        _action_model(),
        original,
        _scalers(),
        40,
        8,
        8,
        candidate_actions=actions,
    )
    right = recursive_prediction(
        _action_model(),
        changed,
        _scalers(),
        40,
        8,
        8,
        candidate_actions=actions,
    )
    np.testing.assert_array_equal(left, right)


def test_boundary_flags_report_each_maximum_independently() -> None:
    assert selection_boundary_flags(40, 1e5) == {
        "selected_history_at_grid_max": True,
        "selected_alpha_at_grid_max": True,
    }
    assert selection_boundary_flags(24, 1e5) == {
        "selected_history_at_grid_max": False,
        "selected_alpha_at_grid_max": True,
    }
    with pytest.raises(ValueError, match="outside"):
        selection_boundary_flags(32, 1e5)


def test_precomputed_weighted_ridge_path_matches_sklearn() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 17))
    y = rng.normal(size=(120, 4))
    weight = rng.uniform(0.2, 2.0, size=120)
    for alpha, candidate in ridge_path(
        x, y, weight, alphas=(1e-4, 1.0, 1e5)
    ).items():
        reference = Ridge(
            alpha=alpha, fit_intercept=True, solver="cholesky"
        ).fit(x, y, sample_weight=weight)
        np.testing.assert_allclose(
            candidate.predict(x),
            reference.predict(x),
            rtol=2e-9,
            atol=2e-9,
        )


def test_case_local_core_uses_case_local_expected_row_count() -> None:
    root = Path(__file__).resolve().parents[2]
    frame = pd.read_csv(
        root
        / "artifacts/schedule_matched_arx_transport_evaluation_v3/arx_core.csv",
        float_precision="round_trip",
    )
    case = "bestest_hydronic_heat_pump"
    local = frame.loc[frame["case"] == case].copy()
    local["arm"] = "post_outcome_strengthened_recursive_ridge_arx"
    _validate_core(local, expected_cases=(case,))
    with pytest.raises(ValueError, match="incomplete"):
        _validate_core(local)
