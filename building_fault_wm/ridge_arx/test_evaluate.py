from __future__ import annotations

from dataclasses import replace

import numpy as np

from building_fault_wm.deterministic_transport.evaluate import (
    PolicyTrajectoryMetadata,
)
from building_fault_wm.neural_benchmark.fault_data import TrajectoryKey

from .config import FROZEN_CONFIG
from .evaluate import ARM, CORE_COLUMNS, evaluate_variants, summarize_secondary
from .test_train import (
    action_sensitive_model,
    identity_scalers,
    synthetic_variant,
)


def paired_variants(post_anchor_shift: float = 0.0):
    old = synthetic_variant(
        policy_seed=101, post_anchor_shift=post_anchor_shift
    )
    new = synthetic_variant(
        policy_seed=202, post_anchor_shift=post_anchor_shift
    )
    new = replace(
        new,
        cell=replace(
            new.cell,
            cell_id=old.cell.cell_id.replace("seed101", "seed202"),
        ),
    )
    metadata = {
        TrajectoryKey("case", "locked_test", 10, 101): PolicyTrajectoryMetadata(
            "old_2h", "window-1", 303
        ),
        TrajectoryKey("case", "locked_test", 10, 202): PolicyTrajectoryMetadata(
            "new_4h", "window-1", 303
        ),
    }
    return (old, new), metadata


def test_secondary_evaluation_is_paired_and_causal() -> None:
    variants, metadata = paired_variants()
    models = {
        seed: action_sensitive_model() for seed in FROZEN_CONFIG.model_seeds
    }
    core, detailed = evaluate_variants(
        models,
        variants,
        identity_scalers(),
        metadata,
        horizons=(1,),
    )
    assert tuple(core.columns) == CORE_COLUMNS
    assert len(core) == 10
    assert len(detailed) == 10
    assert set(core["arm"]) == {ARM}
    assert set(core["policy"]) == {"old_2h", "new_4h"}

    shifted, shifted_metadata = paired_variants(post_anchor_shift=99_999.0)
    shifted_core, _ = evaluate_variants(
        models,
        shifted,
        identity_scalers(),
        shifted_metadata,
        horizons=(1,),
    )
    np.testing.assert_array_equal(
        core["standardized_abs_error"].to_numpy(),
        shifted_core["standardized_abs_error"].to_numpy(),
    )


def test_secondary_summary_equal_weights_cells() -> None:
    variants, metadata = paired_variants()
    models = {
        seed: action_sensitive_model() for seed in FROZEN_CONFIG.model_seeds
    }
    core, _ = evaluate_variants(
        models,
        variants,
        identity_scalers(),
        metadata,
        horizons=(1,),
    )
    summary = summarize_secondary(core)
    assert len(summary) == 2
    assert set(summary["policy"]) == {"old_2h", "new_4h"}
    assert (summary["equal_cell_count"] == 5).all()
