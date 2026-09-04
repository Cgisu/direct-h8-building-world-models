from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from building_fault_wm.neural_benchmark.fault_data import (
    FaultCell,
    FaultScalers,
    FaultVariant,
    ScaleStats,
    TrajectoryKey,
)
from building_fault_wm.neural_benchmark.reliability_model import (
    ReliabilityGatedRSSM,
)
from building_fault_wm.neural_benchmark.study_config import StudyConfig

from .config import FROZEN_CONFIG
from .evaluate import (
    EVALUATION_HORIZONS,
    PolicyTrajectoryMetadata,
    action_block_crosses_transition,
    assert_paired_rows,
    evaluate_transport_models,
    load_deterministic_checkpoint,
    load_frozen_v2_rssm,
)
from .gate import ARMS, REQUIRED_COLUMNS
from .model import DeterministicRecurrentWorldModel
from building_fault_wm.neural_benchmark.study_train import (
    core_tensor_state_sha256,
    tensor_state_sha256,
)


CASE = "bestest_hydronic_heat_pump"
MODEL_SEED = 202608011
ANCHOR = 56


class RecordingRSSM(ReliabilityGatedRSSM):
    def __init__(self, config):
        super().__init__(config)
        self.filter_calls = []
        self.imagine_calls = []

    def filter(
        self,
        previous_actions,
        observations,
        masks,
        ages,
        contexts=None,
        **kwargs,
    ):
        self.filter_calls.append(
            tuple(
                value.detach().clone()
                for value in (
                    previous_actions,
                    observations,
                    masks,
                    ages,
                    contexts,
                )
            )
        )
        return super().filter(
            previous_actions,
            observations,
            masks,
            ages,
            contexts,
            **kwargs,
        )

    def imagine(self, state, actions, future_contexts=None, **kwargs):
        self.imagine_calls.append(
            (
                actions.detach().clone(),
                future_contexts.detach().clone(),
                dict(kwargs),
            )
        )
        return super().imagine(state, actions, future_contexts, **kwargs)


class RecordingDeterministic(DeterministicRecurrentWorldModel):
    def __init__(self):
        super().__init__(FROZEN_CONFIG)
        self.observe_calls = []
        self.imagine_calls = []

    def observe_step(
        self,
        hidden,
        observations,
        availability,
        age,
        previous_actions,
        contexts,
    ):
        self.observe_calls.append(
            tuple(
                value.detach().clone()
                for value in (
                    observations,
                    availability,
                    age,
                    previous_actions,
                    contexts,
                )
            )
        )
        return super().observe_step(
            hidden,
            observations,
            availability,
            age,
            previous_actions,
            contexts,
        )

    def imagine(self, hidden, actions, contexts):
        self.imagine_calls.append(
            (actions.detach().clone(), contexts.detach().clone())
        )
        return super().imagine(hidden, actions, contexts)


def _action_series(policy: str) -> np.ndarray:
    steps = 192
    dwell = 8 if policy == "old_2h" else 16
    levels = np.asarray((-1.0, 0.0, 1.0, 0.0))
    return np.resize(np.repeat(levels, dwell), steps).reshape(-1, 1)


def _variant(policy: str, trajectory_seed: int) -> FaultVariant:
    steps = 192
    time = np.arange(steps, dtype=float)
    clean = np.stack(
        [
            290.0 + time / 100.0,
            1000.0 + 4.0 * time,
            np.sin(time / 11.0),
            np.cos(time / 13.0),
        ],
        axis=1,
    )
    corrupted = clean.copy()
    corrupted[48:96, 0] += 1.0
    contexts = np.stack(
        [time / (index + 20.0) for index in range(5)], axis=1
    )
    key = TrajectoryKey(
        CASE,
        "locked_test",
        42,
        trajectory_seed,
    )
    labels = np.zeros((steps, 2), dtype=np.int64)
    labels[48:96, 0] = 1
    return FaultVariant(
        cell=FaultCell(
            cell_id=f"{key.text}:{policy}:zone:bias",
            trajectory=key,
            source_sha256="a" * 64,
            fault_channel="zone_temperature_k",
            family="bias",
            sign=1,
            severity=1.0,
            severity_unit="K",
            onset=48,
            stop=96,
            anchors=(ANCHOR,),
        ),
        source=Path(f"{policy}.csv"),
        clean_observations=clean,
        corrupted_observations=corrupted,
        actions=_action_series(policy),
        contexts=contexts,
        availability=np.ones_like(clean, dtype=bool),
        age=np.zeros_like(clean),
        health_labels=labels,
    )


def _inputs(recording: bool = False):
    variants = [_variant("old_2h", 101), _variant("new_4h", 102)]
    metadata = {
        variants[0].cell.trajectory: PolicyTrajectoryMetadata(
            "old_2h", f"{CASE}:day042", 7001
        ),
        variants[1].cell.trajectory: PolicyTrajectoryMetadata(
            "new_4h", f"{CASE}:day042", 7001
        ),
    }
    config = StudyConfig()
    torch.manual_seed(31)
    if recording:
        models = {
            "legacy": RecordingRSSM(config.model_config()),
            "ungated_h8": RecordingRSSM(config.model_config()),
            "deterministic_wm": RecordingDeterministic(),
        }
    else:
        models = {
            "legacy": ReliabilityGatedRSSM(config.model_config()),
            "ungated_h8": ReliabilityGatedRSSM(config.model_config()),
            "deterministic_wm": DeterministicRecurrentWorldModel(),
        }
    scalers = FaultScalers(
        observation=ScaleStats((290.0, 1000.0, 0.0, 0.0), (2.0, 500.0, 1.0, 1.0)),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=(("fit", "b" * 64),),
    )
    return models, variants, metadata, scalers


@pytest.mark.parametrize("horizon", EVALUATION_HORIZONS)
def test_exact_history_and_future_alignment(horizon: int) -> None:
    models, variants, metadata, scalers = _inputs(recording=True)
    result = evaluate_transport_models(
        models,
        variants,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(horizon,),
    )
    assert tuple(result.core.columns) == REQUIRED_COLUMNS
    assert set(result.core["arm"]) == set(ARMS)
    assert set(result.core["horizon"]) == {horizon}
    assert set(result.detailed["history_start"]) == {17}
    assert set(result.detailed["history_stop"]) == {56}
    assert set(result.detailed["target_index"]) == {56 + horizon}

    rssm = models["legacy"]
    assert isinstance(rssm, RecordingRSSM)
    previous, observations, _, _, contexts = rssm.filter_calls[0]
    np.testing.assert_array_equal(
        previous[:, 0].numpy(), variants[0].actions[16:56]
    )
    np.testing.assert_allclose(
        observations[:, 0].numpy(),
        scalers.observation.transform(
            variants[0].corrupted_observations[17:57]
        ),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        contexts[:, 0].numpy(), variants[0].contexts[17:57], atol=2e-6
    )
    actions, future_contexts, kwargs = rssm.imagine_calls[0]
    np.testing.assert_array_equal(
        actions[:, 0].numpy(), variants[0].actions[56 : 56 + horizon]
    )
    np.testing.assert_allclose(
        future_contexts[:, 0].numpy(),
        variants[0].contexts[57 : 57 + horizon],
        atol=2e-6,
    )
    assert kwargs["sample"] is False

    deterministic = models["deterministic_wm"]
    assert isinstance(deterministic, RecordingDeterministic)
    assert len(deterministic.observe_calls) == 40 + 2 * horizon
    np.testing.assert_allclose(
        deterministic.observe_calls[0][0][0].numpy(),
        scalers.observation.transform(
            variants[0].corrupted_observations[17:18]
        )[0],
        atol=2e-5,
    )
    assert deterministic.imagine_calls[0][0].shape[0] == horizon


def test_future_corrupted_observations_are_never_read() -> None:
    models, variants, metadata, scalers = _inputs()
    reference = evaluate_transport_models(
        models,
        variants,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(8,),
    )
    changed = []
    for variant in variants:
        corrupted = variant.corrupted_observations.copy()
        corrupted[ANCHOR + 1 :] += 1_000_000.0
        changed.append(replace(variant, corrupted_observations=corrupted))
    repeated = evaluate_transport_models(
        models,
        changed,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(8,),
    )
    pd.testing.assert_frame_equal(reference.core, repeated.core, check_exact=True)
    pd.testing.assert_series_equal(
        reference.detailed["prediction_raw"],
        repeated.detailed["prediction_raw"],
        check_exact=True,
    )


def test_predictions_are_sensitive_to_future_actions() -> None:
    models, variants, metadata, scalers = _inputs()
    reference = evaluate_transport_models(
        models,
        variants,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(8,),
    )
    changed = []
    for variant in variants:
        actions = variant.actions.copy()
        actions[ANCHOR : ANCHOR + 8] = 0.75
        changed.append(replace(variant, actions=actions))
    repeated = evaluate_transport_models(
        models,
        changed,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(8,),
    )
    merged = reference.detailed.merge(
        repeated.detailed,
        on=[
            column
            for column in REQUIRED_COLUMNS
            if column not in {"standardized_abs_error"}
        ],
        suffixes=("_before", "_after"),
    )
    by_arm = merged.groupby("arm").apply(
        lambda rows: np.max(
            np.abs(
                rows["prediction_standardized_before"]
                - rows["prediction_standardized_after"]
            )
        ),
        include_groups=False,
    )
    assert (by_arm > 1e-8).all()


def test_core_rows_are_exactly_arm_and_policy_paired() -> None:
    models, variants, metadata, scalers = _inputs()
    result = evaluate_transport_models(
        models,
        variants,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(1, 8),
    )
    assert tuple(result.core.columns) == REQUIRED_COLUMNS
    assert len(result.core) == 2 * 2 * len(ARMS)
    assert_paired_rows(result.core)
    with pytest.raises(ValueError, match="model arms"):
        assert_paired_rows(result.core.iloc[:-1].copy())


def test_deterministic_replay_is_bit_exact() -> None:
    models, variants, metadata, scalers = _inputs()
    first = evaluate_transport_models(
        models,
        variants,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
    )
    second = evaluate_transport_models(
        models,
        variants,
        scalers,
        metadata,
        model_seed=MODEL_SEED,
    )
    pd.testing.assert_frame_equal(first.core, second.core, check_exact=True)
    pd.testing.assert_frame_equal(first.detailed, second.detailed, check_exact=True)


def test_boundary_crossing_counts_entry_and_internal_dwell_transitions() -> None:
    actions = np.asarray([0.0] * 4 + [1.0] * 4 + [-1.0] * 4).reshape(-1, 1)
    assert action_block_crosses_transition(actions, 0, 4) == (False, 0)
    assert action_block_crosses_transition(actions, 3, 2) == (True, 1)
    assert action_block_crosses_transition(actions, 4, 4) == (True, 1)
    assert action_block_crosses_transition(actions, 3, 8) == (True, 2)


def test_detailed_rows_keep_raw_units_separate() -> None:
    models, variants, metadata, scalers = _inputs()
    power_variants = []
    for variant in variants:
        cell = replace(
            variant.cell,
            cell_id=f"{variant.cell.cell_id}:power",
            fault_channel="hvac_electric_power_w",
            severity=250.0,
            severity_unit="W",
        )
        corrupted = variant.clean_observations.copy()
        corrupted[48:96, 1] += 250.0
        power_variants.append(
            replace(
                variant,
                cell=cell,
                corrupted_observations=corrupted,
            )
        )
    result = evaluate_transport_models(
        models,
        [*variants, *power_variants],
        scalers,
        metadata,
        model_seed=MODEL_SEED,
        horizons=(8,),
    )
    assert set(result.detailed["raw_unit"]) == {"K", "W"}
    np.testing.assert_allclose(
        result.detailed["raw_abs_error"],
        (
            result.detailed["prediction_raw"]
            - result.detailed["target_raw"]
        ).abs(),
        rtol=0.0,
        atol=0.0,
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_loaders_bind_identity_state_and_file_hash(tmp_path: Path) -> None:
    config = StudyConfig()
    rssm = ReliabilityGatedRSSM(config.model_config())
    rssm_state = rssm.state_dict()
    rssm_path = tmp_path / "legacy_u0400.pt"
    torch.save(
        {
            "schema": "boptest-reliability-rssm-checkpoint-v2",
            "case": CASE,
            "model_seed": MODEL_SEED,
            "arm": "legacy",
            "update": 400,
            "config": config.to_dict(),
            "provenance": {"bound_by": "full_file_sha256"},
            "model_state_sha256": tensor_state_sha256(rssm_state),
            "core_state_sha256": core_tensor_state_sha256(rssm_state),
            "model_state_dict": rssm_state,
        },
        rssm_path,
    )
    loaded_rssm = load_frozen_v2_rssm(
        rssm_path,
        case=CASE,
        model_seed=MODEL_SEED,
        arm="legacy",
        expected_file_sha256=_sha256(rssm_path),
    )
    assert tensor_state_sha256(loaded_rssm.state_dict()) == tensor_state_sha256(
        rssm_state
    )

    deterministic = DeterministicRecurrentWorldModel()
    deterministic_state = deterministic.state_dict()
    deterministic_path = tmp_path / "update_0400.pt"
    torch.save(
        {
            "schema": "boptest-deterministic-transport-checkpoint-v1",
            "update": 400,
            "model_seed": MODEL_SEED,
            "config": FROZEN_CONFIG.to_dict(),
            "model_state_dict": deterministic_state,
            "model_state_sha256": tensor_state_sha256(deterministic_state),
            "selected": True,
        },
        deterministic_path,
    )
    loaded_deterministic = load_deterministic_checkpoint(
        deterministic_path,
        model_seed=MODEL_SEED,
        expected_file_sha256=_sha256(deterministic_path),
    )
    assert tensor_state_sha256(
        loaded_deterministic.state_dict()
    ) == tensor_state_sha256(deterministic_state)

    with pytest.raises(ValueError, match="frozen digest"):
        load_frozen_v2_rssm(
            rssm_path,
            case=CASE,
            model_seed=MODEL_SEED,
            arm="legacy",
            expected_file_sha256="0" * 64,
        )
