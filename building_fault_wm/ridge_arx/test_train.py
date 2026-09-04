from __future__ import annotations

import json
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

from .config import FROZEN_CONFIG
from .train import (
    MODEL_SCHEMA,
    RUN_SCHEMA,
    _ridge_state_payload,
    load_model_run,
    recursive_prediction,
    restore_model,
    scheduled_one_step_dataset,
)
from .io import canonical_sha256, sha256_file, write_csv_once, write_json_once


def synthetic_variant(
    *,
    policy_seed: int = 101,
    family: str = "bias",
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
    availability = np.ones_like(clean, dtype=bool)
    age = np.zeros_like(clean)
    labels = np.zeros_like(clean, dtype=np.int64)
    return FaultVariant(
        cell=FaultCell(
            cell_id=f"case:seed{policy_seed}:zone_temperature_k:{family}",
            trajectory=TrajectoryKey("case", "locked_test", 10, policy_seed),
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family=family,
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
        availability=availability,
        age=age,
        health_labels=labels,
    )


def identity_scalers() -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("case:fit", "a" * 64),),
    )


def action_sensitive_model() -> Ridge:
    model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    model.coef_ = np.zeros(
        (FROZEN_CONFIG.observation_dim, FROZEN_CONFIG.feature_dim),
        dtype=float,
    )
    current_action_index = (
        FROZEN_CONFIG.history
        * (3 * FROZEN_CONFIG.observation_dim + FROZEN_CONFIG.action_dim)
    )
    model.coef_[0, current_action_index] = 1.0
    model.intercept_ = np.zeros(FROZEN_CONFIG.observation_dim)
    model.n_features_in_ = FROZEN_CONFIG.feature_dim
    return model


def test_schedule_materializes_exact_sequence_contained_sources() -> None:
    variant = synthetic_variant()
    schedule = (
        ScheduledBatch(
            update=1,
            latent_seed=9,
            references=tuple(SequenceReference(0, 15) for _ in range(4)),
        ),
    )
    x, y, groups, digest = scheduled_one_step_dataset(
        [variant],
        identity_scalers(),
        schedule,
        require_complete=False,
    )
    assert x.shape == (160, 115)
    assert y.shape == (160, 4)
    assert len(groups) == 160
    assert len(digest) == 64
    assert FROZEN_CONFIG.scheduled_rows_per_model == 64_000
    assert FROZEN_CONFIG.active_coefficients == 464


def test_recursive_rollout_never_consumes_post_anchor_observations() -> None:
    model = action_sensitive_model()
    original = synthetic_variant()
    changed_future = synthetic_variant(post_anchor_shift=10_000.0)
    actions = np.full((8, 1), 0.25)
    left = recursive_prediction(
        model,
        original,
        identity_scalers(),
        anchor=40,
        horizon=8,
        candidate_actions=actions,
    )
    right = recursive_prediction(
        model,
        changed_future,
        identity_scalers(),
        anchor=40,
        horizon=8,
        candidate_actions=actions,
    )
    np.testing.assert_array_equal(left, right)
    changed_actions = recursive_prediction(
        model,
        original,
        identity_scalers(),
        anchor=40,
        horizon=8,
        candidate_actions=np.full((8, 1), -0.5),
    )
    assert left[0] != changed_actions[0]
    with pytest.raises(ValueError, match="history"):
        recursive_prediction(
            model,
            original,
            identity_scalers(),
            anchor=7,
            horizon=1,
        )


def test_json_model_round_trip_and_tamper_rejection() -> None:
    model = action_sensitive_model()
    state = _ridge_state_payload(model)
    payload = {
        "schema": MODEL_SCHEMA,
        "case": "bestest_hydronic_heat_pump",
        "model_seed": FROZEN_CONFIG.model_seeds[0],
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
        "state": state,
    }
    restored = restore_model(payload)
    features = np.arange(FROZEN_CONFIG.feature_dim, dtype=float)[None]
    np.testing.assert_array_equal(model.predict(features), restored.predict(features))

    tampered = json.loads(json.dumps(payload))
    tampered["state"]["coef"][0][0] = 1.0
    with pytest.raises(ValueError, match="state hash"):
        restore_model(tampered)


def test_model_run_uses_round_trip_csv_float_parsing(tmp_path: Path) -> None:
    model = action_sensitive_model()
    model.alpha = 100.0
    state = _ridge_state_payload(model)
    case = "bestest_hydronic_heat_pump"
    seed = FROZEN_CONFIG.model_seeds[0]
    model_payload = {
        "schema": MODEL_SCHEMA,
        "case": case,
        "model_seed": seed,
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
        "state": state,
    }
    model_path = write_json_once(tmp_path / "model.json", model_payload)
    table = pd.DataFrame(
        [
            {
                "alpha": 100.0,
                "validation_h8_mae": 0.5783742564472306,
                "selected": True,
            },
            {
                "alpha": 1.0,
                "validation_h8_mae": 1.1511464987107503,
                "selected": False,
            },
            *[
                {
                    "alpha": alpha,
                    "validation_h8_mae": 2.0 + index,
                    "selected": False,
                }
                for index, alpha in enumerate(
                    value
                    for value in FROZEN_CONFIG.alphas
                    if value not in {100.0, 1.0}
                )
            ],
        ]
    )
    score_path = write_csv_once(tmp_path / "validation_scores.csv", table)
    receipt = {
        "schema": RUN_SCHEMA,
        "case": case,
        "model_seed": seed,
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
        "selected_alpha": 100.0,
        "selected_validation_h8_mae": 0.5783742564472306,
        "training_rows": FROZEN_CONFIG.scheduled_rows_per_model,
        "feature_dim": FROZEN_CONFIG.feature_dim,
        "target_dim": FROZEN_CONFIG.observation_dim,
        "active_coefficients": FROZEN_CONFIG.active_coefficients,
        "model_state_sha256": state["state_sha256"],
        "model_file_sha256": sha256_file(model_path),
        "score_table_file_sha256": sha256_file(score_path),
        "score_table_payload_sha256": canonical_sha256(
            table.to_dict(orient="records")
        ),
    }
    write_json_once(tmp_path / "training_receipt.json", receipt)
    restored, loaded = load_model_run(tmp_path, case=case, model_seed=seed)
    assert loaded["score_table_payload_sha256"] == receipt[
        "score_table_payload_sha256"
    ]
    assert float(restored.alpha) == 100.0
