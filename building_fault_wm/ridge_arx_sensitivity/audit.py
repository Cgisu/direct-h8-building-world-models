"""Standalone numerical audit for the strengthened ARX sensitivity result."""

from __future__ import annotations

import json
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
    write_json_once,
    write_once,
)

from .config import CASES, CONFIG, MODEL_SEEDS, POLICIES, SILENT_FAMILIES
from .study import (
    ARM,
    CORE_COLUMNS,
    DEFAULT_EVALUATION_ROOT,
    NEURAL_EVALUATION,
    verify_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_ROOT = (
    PROJECT_ROOT / "artifacts/post_outcome_strong_arx_audit_v3"
)
AUDIT_SCHEMA = "post-outcome-strong-arx-standalone-audit-v1"
PAIR_COLUMNS = tuple(
    column
    for column in CORE_COLUMNS
    if column not in {"arm", "standardized_abs_error"}
)
WEIGHTING_DIMENSIONS = ("family", "fault_channel", "sign", "severity")
BASE_CLUSTER = ("case", "policy", "window_id", "model_seed")


def _pair(neural: pd.DataFrame, arx: pd.DataFrame) -> pd.DataFrame:
    if tuple(arx.columns) != CORE_COLUMNS:
        raise ValueError("audit ARX columns changed")
    neural = neural.loc[
        (neural["arm"] == "deterministic_wm")
        & neural["family"].isin(SILENT_FAMILIES)
    ].copy()
    if tuple(neural.columns) != CORE_COLUMNS:
        raise ValueError("audit neural columns changed")
    left = neural.rename(
        columns={"standardized_abs_error": "deterministic_wm"}
    ).drop(columns=["arm"])
    right = arx.rename(
        columns={"standardized_abs_error": ARM}
    ).drop(columns=["arm"])
    if (
        left.duplicated(list(PAIR_COLUMNS)).any()
        or right.duplicated(list(PAIR_COLUMNS)).any()
        or set(map(tuple, left[list(PAIR_COLUMNS)].itertuples(index=False, name=None)))
        != set(
            map(
                tuple,
                right[list(PAIR_COLUMNS)].itertuples(index=False, name=None),
            )
        )
    ):
        raise ValueError("audit inputs are not exact row pairs")
    paired = left.merge(
        right,
        on=list(PAIR_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    return paired.sort_values(list(PAIR_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _scores(
    paired: pd.DataFrame,
    *,
    retain_family: bool = False,
) -> pd.DataFrame:
    selected = paired.loc[paired["horizon"] == 8].copy()
    retained = ("family",) if retain_family else ()
    group = [
        *BASE_CLUSTER,
        *retained,
        *(
            value
            for value in WEIGHTING_DIMENSIONS
            if value not in retained
        ),
    ]
    arms = ["deterministic_wm", ARM]
    result = selected.groupby(group, as_index=False, dropna=False)[arms].mean()
    for dimension in ("sign", "severity", "fault_channel", "family"):
        if dimension in retained:
            continue
        group = [value for value in group if value != dimension]
        result = result.groupby(group, as_index=False, dropna=False)[arms].mean()
    expected = 3 * 2 * 12 * 5 * (3 if retain_family else 1)
    if len(result) != expected:
        raise ValueError("audit equal-weight grid is incomplete")
    return result.sort_values(group, kind="stable").reset_index(drop=True)


def _bootstrap(
    scores: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, np.ndarray]],
]:
    rng = np.random.Generator(np.random.PCG64(CONFIG.bootstrap_seed))
    sampled_cases = rng.integers(
        0, len(CASES), size=(CONFIG.bootstrap_draws, len(CASES))
    )
    sampled_seeds = rng.integers(
        0, len(MODEL_SEEDS), size=(CONFIG.bootstrap_draws, len(MODEL_SEEDS))
    )
    windows = {
        case: tuple(
            sorted(set(scores.loc[scores["case"] == case, "window_id"]))
        )
        for case in CASES
    }
    sampled_windows = {
        case: rng.integers(
            0,
            len(values),
            size=(CONFIG.bootstrap_draws, len(values)),
        )
        for case, values in windows.items()
    }
    points = {}
    draws = {}
    for policy in POLICIES:
        case_points = []
        case_draws = []
        selected = scores.loc[scores["policy"] == policy]
        for case in CASES:
            index = pd.MultiIndex.from_product(
                [MODEL_SEEDS, windows[case]],
                names=["model_seed", "window_id"],
            )
            rows = selected.loc[selected["case"] == case].set_index(
                ["model_seed", "window_id"]
            )
            if rows.index.has_duplicates or set(rows.index) != set(index):
                raise ValueError("audit bootstrap matrix is incomplete")
            matrix = rows.loc[index, ["deterministic_wm", ARM]].to_numpy(
                dtype=float
            ).reshape(len(MODEL_SEEDS), len(windows[case]), 2)
            sampled = matrix[
                sampled_seeds[:, :, None],
                sampled_windows[case][:, None, :],
                :,
            ]
            case_points.append(matrix.mean(axis=(0, 1)))
            case_draws.append(sampled.mean(axis=(1, 2)))
        point = np.stack(case_points).mean(axis=0)
        by_case = np.stack(case_draws, axis=1)
        sampled = by_case[
            np.arange(CONFIG.bootstrap_draws)[:, None],
            sampled_cases,
            :,
        ].mean(axis=1)
        points[policy] = {
            "deterministic_wm": float(point[0]),
            ARM: float(point[1]),
        }
        draws[policy] = {
            "deterministic_wm": sampled[:, 0],
            ARM: sampled[:, 1],
        }
    return points, draws


def _effect(frame: pd.DataFrame) -> float:
    return float(
        1.0 - frame["deterministic_wm"].mean() / frame[ARM].mean()
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


def recompute(
    neural: pd.DataFrame,
    arx: pd.DataFrame,
) -> dict:
    """Recompute every headline from raw per-row inputs."""

    paired = _pair(neural, arx)
    scores = _scores(paired)
    family_scores = _scores(paired, retain_family=True)
    points, draws = _bootstrap(scores)
    policies = {}
    effect_draws = {}
    for policy in POLICIES:
        values = (
            1.0
            - draws[policy]["deterministic_wm"]
            / draws[policy][ARM]
        )
        point = (
            1.0
            - points[policy]["deterministic_wm"]
            / points[policy][ARM]
        )
        effect_draws[policy] = values
        policies[policy] = {
            **_interval(float(point), values),
            "mean_standardized_mae_by_arm": points[policy],
            "by_case": {
                str(key): _effect(rows)
                for key, rows in scores.loc[
                    scores["policy"] == policy
                ].groupby("case", sort=True)
            },
            "by_family": {
                str(key): _effect(rows)
                for key, rows in family_scores.loc[
                    family_scores["policy"] == policy
                ].groupby("family", sort=True)
            },
            "by_seed": {
                str(key): _effect(rows)
                for key, rows in scores.loc[
                    scores["policy"] == policy
                ].groupby("model_seed", sort=True)
            },
        }
    difference = policies["new_4h"]["point"] - policies["old_2h"]["point"]
    return {
        "policy_results": policies,
        "transport_new_4h_minus_old_2h": _interval(
            float(difference),
            effect_draws["new_4h"] - effect_draws["old_2h"],
        ),
        "paired_row_count": len(paired),
        "equal_weight_cluster_count": len(scores),
    }


def _flatten(prefix: str, value: object, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        output[prefix] = float(value)


def run_audit(
    *,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    output_root: Path = DEFAULT_AUDIT_ROOT,
) -> Path:
    completion = verify_evaluation(evaluation_root)
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite audit: {output_root}")
    neural_path = NEURAL_EVALUATION / v3_run.CORE_NAME
    arx_path = evaluation_root / "strong_arx_core.csv"
    neural = pd.read_csv(neural_path, float_precision="round_trip")
    arx = pd.read_csv(arx_path, float_precision="round_trip")
    recomputed = recompute(neural, arx)
    reported = json.loads(
        (evaluation_root / "sensitivity_result.json").read_text(
            encoding="ascii"
        )
    )
    expected = {
        "policy_results": {
            policy: {
                key: reported["policy_results"][policy][key]
                for key in (
                    "point",
                    "ci95_lower",
                    "ci95_upper",
                    "ci90_lower",
                    "ci90_upper",
                    "mean_standardized_mae_by_arm",
                    "by_case",
                    "by_family",
                    "by_seed",
                )
            }
            for policy in POLICIES
        },
        "transport_new_4h_minus_old_2h": reported[
            "transport_new_4h_minus_old_2h"
        ],
    }
    observed_values: dict[str, float] = {}
    expected_values: dict[str, float] = {}
    _flatten("", recomputed, observed_values)
    _flatten("", expected, expected_values)
    shared = sorted(set(observed_values) & set(expected_values))
    differences = {
        key: abs(observed_values[key] - expected_values[key])
        for key in shared
    }
    if set(expected_values) - set(observed_values):
        raise ValueError("audit did not recompute every reported numeric field")
    maximum = max(differences.values(), default=0.0)
    if maximum > 1e-12:
        raise ValueError(f"standalone audit mismatch: {maximum}")
    output_root.mkdir(parents=True, exist_ok=False)
    recomputed_path = write_json_once(
        output_root / "recomputed_headlines.json", recomputed
    )
    comparison_path = write_once(
        output_root / "numeric_comparison.csv",
        pd.DataFrame(
            [
                {
                    "field": key,
                    "reported": expected_values[key],
                    "recomputed": observed_values[key],
                    "absolute_difference": differences[key],
                }
                for key in shared
            ]
        )
        .to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        )
        .encode("ascii"),
    )
    receipt = {
        "schema": AUDIT_SCHEMA,
        "scope": "post_outcome_robustness",
        "standalone_aggregation_implementation": True,
        "evaluation_completion_file_sha256": sha256_file(
            evaluation_root / "evaluation_complete.json"
        ),
        "evaluation_completion_payload_sha256": canonical_sha256(completion),
        "neural_raw_core_file_sha256": sha256_file(neural_path),
        "strong_arx_raw_core_file_sha256": sha256_file(arx_path),
        "recomputed_numeric_field_count": len(shared),
        "maximum_absolute_difference": maximum,
        "tolerance": 1e-12,
        "exact_within_tolerance": True,
        "recomputed_headlines_file_sha256": sha256_file(recomputed_path),
        "numeric_comparison_file_sha256": sha256_file(comparison_path),
    }
    receipt_path = write_json_once(output_root / "audit_receipt.json", receipt)
    complete = {
        "schema": "post-outcome-strong-arx-audit-completion-v1",
        "complete": True,
        "file_sha256_by_name": {
            path.name: sha256_file(path)
            for path in (recomputed_path, comparison_path, receipt_path)
        },
    }
    return write_json_once(output_root / "audit_complete.json", complete)
