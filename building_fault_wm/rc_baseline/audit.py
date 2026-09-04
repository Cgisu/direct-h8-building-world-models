"""Independent numerical audit for the reviewer RC comparison."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from building_fault_wm.deterministic_transport import (
    gate as v3_gate,
    run_evaluation as v3_run,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    write_json_once,
)

from .config import CASES, CONFIG, MODEL_SEEDS, POLICIES, SILENT_FAMILIES
from .evaluation import (
    ALL_ARMS,
    ARM,
    CORE_COLUMNS,
    DEFAULT_EVALUATION_ROOT,
    NEURAL_ARMS,
    NEURAL_EVALUATION,
    PAIR_COLUMNS,
    verify_evaluation,
)


DEFAULT_AUDIT_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts/reviewer_rc_audit_v1"
)


def _scores(paired: pd.DataFrame, retain_family: bool = False) -> pd.DataFrame:
    selected = paired.loc[paired["horizon"] == 8].copy()
    retained = ("family",) if retain_family else ()
    group = [
        "case",
        "policy",
        "window_id",
        "model_seed",
        *retained,
        *(
            value
            for value in ("family", "fault_channel", "sign", "severity")
            if value not in retained
        ),
    ]
    result = selected.groupby(group, as_index=False, dropna=False)[list(ALL_ARMS)].mean()
    for dimension in ("sign", "severity", "fault_channel", "family"):
        if dimension in retained:
            continue
        group = [value for value in group if value != dimension]
        result = result.groupby(group, as_index=False, dropna=False)[list(ALL_ARMS)].mean()
    expected = len(CASES) * len(POLICIES) * 12 * len(MODEL_SEEDS)
    if retain_family:
        expected *= len(SILENT_FAMILIES)
    if len(result) != expected:
        raise ValueError("RC audit equal-weight score grid is incomplete")
    return result.sort_values(group, kind="stable").reset_index(drop=True)


def _pair(neural: pd.DataFrame, comparator: pd.DataFrame) -> pd.DataFrame:
    if tuple(comparator.columns) != CORE_COLUMNS or len(comparator) != 207_360:
        raise ValueError("RC audit comparator core changed")
    neural = v3_gate.validate_input(neural)
    neural = neural.loc[
        neural["arm"].isin(NEURAL_ARMS)
        & neural["family"].isin(SILENT_FAMILIES)
    ].copy()
    left = set(
        map(tuple, neural[list(PAIR_COLUMNS)].itertuples(index=False, name=None))
    )
    right = set(
        map(
            tuple,
            comparator[list(PAIR_COLUMNS)].itertuples(index=False, name=None),
        )
    )
    if left != right or len(right) != len(comparator):
        raise ValueError("RC audit inputs are not exact row pairs")
    combined = pd.concat([neural, comparator], ignore_index=True)
    if combined.duplicated([*PAIR_COLUMNS, "arm"]).any():
        raise ValueError("RC audit comparison has duplicate arm rows")
    paired = combined.pivot(
        index=list(PAIR_COLUMNS), columns="arm", values="standardized_abs_error"
    ).reset_index()
    paired.columns.name = None
    return paired


def _draws(scores: pd.DataFrame, policy: str, rng_data: dict) -> tuple[dict, dict]:
    selected = scores.loc[scores["policy"] == policy]
    points = []
    draws = []
    for case in CASES:
        windows = rng_data["window_values"][case]
        index = pd.MultiIndex.from_product(
            [MODEL_SEEDS, windows], names=["model_seed", "window_id"]
        )
        rows = selected.loc[selected["case"] == case].set_index(
            ["model_seed", "window_id"]
        )
        if rows.index.has_duplicates or set(rows.index) != set(index):
            raise ValueError("RC audit seed/window matrix is incomplete")
        matrix = rows.loc[index, list(ALL_ARMS)].to_numpy(dtype=float).reshape(
            len(MODEL_SEEDS), len(windows), len(ALL_ARMS)
        )
        sampled = matrix[
            rng_data["seed_indices"][:, :, None],
            rng_data["window_indices"][case][:, None, :],
            :,
        ]
        points.append(matrix.mean(axis=(0, 1)))
        draws.append(sampled.mean(axis=(1, 2)))
    point = np.stack(points).mean(axis=0)
    by_case = np.stack(draws, axis=1)
    sample = by_case[
        np.arange(CONFIG.bootstrap_draws)[:, None], rng_data["case_indices"], :
    ].mean(axis=1)
    return (
        {arm: float(point[i]) for i, arm in enumerate(ALL_ARMS)},
        {arm: sample[:, i] for i, arm in enumerate(ALL_ARMS)},
    )


def _interval(point: float, draws: np.ndarray) -> dict[str, float]:
    lower95, lower90, upper90, upper95 = np.quantile(
        draws, (0.025, 0.05, 0.95, 0.975)
    )
    return {
        "point": float(point),
        "ci95_lower": float(lower95),
        "ci95_upper": float(upper95),
        "ci90_lower": float(lower90),
        "ci90_upper": float(upper90),
    }


def recompute(neural: pd.DataFrame, comparator: pd.DataFrame) -> dict:
    paired = _pair(neural, comparator)
    scores = _scores(paired)
    family_scores = _scores(paired, retain_family=True)
    rng = np.random.Generator(np.random.PCG64(CONFIG.bootstrap_seed))
    windows = {
        case: tuple(sorted(set(scores.loc[scores["case"] == case, "window_id"])))
        for case in CASES
    }
    rng_data = {
        "case_indices": rng.integers(
            0, len(CASES), size=(CONFIG.bootstrap_draws, len(CASES))
        ),
        "seed_indices": rng.integers(
            0, len(MODEL_SEEDS), size=(CONFIG.bootstrap_draws, len(MODEL_SEEDS))
        ),
        "window_indices": {
            case: rng.integers(0, len(values), size=(CONFIG.bootstrap_draws, len(values)))
            for case, values in windows.items()
        },
        "window_values": windows,
    }
    result = {
        "schema": "reviewer-rc-comparison-result-v1",
        "scope": "post_outcome_supervisory_review_physical_comparator",
        "confirmatory_category_assigned": False,
        "estimand": "1 - equal_weight_MAE(neural_arm) / equal_weight_MAE(RC)",
        "positive_favors": "neural_arm",
        "policy_results": {},
    }
    for policy in POLICIES:
        points, sampled = _draws(scores, policy, rng_data)
        comparisons = {}
        for neural_arm in NEURAL_ARMS:
            effect_draws = 1.0 - sampled[neural_arm] / sampled[ARM]
            point = 1.0 - points[neural_arm] / points[ARM]
            comparisons[neural_arm] = {
                **_interval(point, effect_draws),
                "mean_standardized_mae_by_arm": {
                    neural_arm: points[neural_arm],
                    ARM: points[ARM],
                },
                "by_case": {
                    str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                    for key, rows in scores.loc[
                        scores["policy"] == policy
                    ].groupby("case", sort=True)
                },
                "by_family": {
                    str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                    for key, rows in family_scores.loc[
                        family_scores["policy"] == policy
                    ].groupby("family", sort=True)
                },
                "by_seed": {
                    str(key): float(1.0 - rows[neural_arm].mean() / rows[ARM].mean())
                    for key, rows in scores.loc[
                        scores["policy"] == policy
                    ].groupby("model_seed", sort=True)
                },
            }
        result["policy_results"][policy] = comparisons
    return result


def run_audit(
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    output_root: Path = DEFAULT_AUDIT_ROOT,
) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite RC audit: {output_root}")
    completion = verify_evaluation(evaluation_root)
    neural_path = NEURAL_EVALUATION / v3_run.CORE_NAME
    comparator_path = evaluation_root / "rc_core.csv"
    neural = pd.read_csv(neural_path, float_precision="round_trip")
    comparator = pd.read_csv(comparator_path, float_precision="round_trip")
    recomputed = recompute(neural, comparator)
    recorded = strict_json(evaluation_root / "comparison_result.json")
    if canonical_sha256(recomputed) != canonical_sha256(recorded):
        raise ValueError("independent RC result differs from recorded result")
    receipt = {
        "schema": "reviewer-rc-standalone-audit-v1",
        "complete": True,
        "evaluation_completion_file_sha256": sha256_file(
            evaluation_root / "evaluation_complete.json"
        ),
        "evaluation_completion_payload_sha256": canonical_sha256(completion),
        "neural_core_file_sha256": sha256_file(neural_path),
        "rc_core_file_sha256": sha256_file(comparator_path),
        "recomputed_result_payload_sha256": canonical_sha256(recomputed),
    }
    return write_json_once(output_root / "audit_receipt.json", receipt)
