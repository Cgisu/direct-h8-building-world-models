"""Independent reconstruction of the frozen v3 statistical gate.

This module intentionally does not import :mod:`gate`.  Constants, validation,
equal-weight aggregation, resampling, effect calculation, and classification
are reimplemented here so a defect in the primary gate cannot automatically
propagate into its verifier.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CASES = (
    "bestest_hydronic_heat_pump",
    "multizone_office_simple_air",
    "twozone_apartment_hydronic",
)
POLICIES = ("old_2h", "new_4h")
ARMS = ("legacy", "ungated_h8", "deterministic_wm")
SILENT_FAMILIES = ("bias", "drift", "stuck")
MODEL_SEEDS = (
    202608011,
    202608012,
    202608013,
    202608014,
    202608015,
)
WINDOWS_PER_CASE = 12
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202608029
DOMINANCE_MARGIN = 0.10
EQUIVALENCE_MARGIN = 0.05
STRATUM_EQUIVALENCE_LIMIT = 0.10

REQUIRED_COLUMNS = (
    "case",
    "policy",
    "window_id",
    "trajectory_day",
    "scenario_seed",
    "trajectory_seed",
    "model_seed",
    "arm",
    "cell_id",
    "fault_channel",
    "family",
    "sign",
    "severity",
    "onset",
    "anchor",
    "horizon",
    "standardized_abs_error",
)
IDENTITY_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS if column != "standardized_abs_error"
)
CLUSTER_COLUMNS = ("case", "policy", "window_id", "model_seed")

# Reordered, explicitly hierarchical means can differ in their last binary
# digits from the primary gate's algebraically equivalent grouped means.
FLOAT_ABS_TOLERANCE = 1e-12
FLOAT_REL_TOLERANCE = 1e-12


def _strict_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"independent verifier input is not a plain file: {path}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {path}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON from {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"independent verifier expected a JSON object: {path}")
    return payload


def _require_integer_column(frame: pd.DataFrame, column: str) -> None:
    values = frame[column].to_numpy()
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} is not numeric") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column} is not finite and integer-valued")


def validate_core(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the frozen cases, arms, seeds, windows, and exact pairings."""

    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError("independent gate input columns differ from the frozen schema")
    if frame.empty or frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("independent gate input is empty or duplicates a row identity")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("independent gate input contains non-finite numeric values")
    if (frame["standardized_abs_error"] < 0.0).any():
        raise ValueError("independent gate input contains negative absolute error")
    for column in (
        "trajectory_day",
        "scenario_seed",
        "trajectory_seed",
        "model_seed",
        "sign",
        "onset",
        "anchor",
        "horizon",
    ):
        _require_integer_column(frame, column)
    if set(frame["case"]) != set(CASES):
        raise ValueError("independent gate case grid is incomplete")
    if set(frame["policy"]) != set(POLICIES):
        raise ValueError("independent gate policy grid is incomplete")
    if set(frame["arm"]) != set(ARMS):
        raise ValueError("independent gate arm grid is incomplete")
    if set(int(value) for value in frame["model_seed"]) != set(MODEL_SEEDS):
        raise ValueError("independent gate model-seed grid is incomplete")
    if not set(SILENT_FAMILIES).issubset(set(frame["family"])):
        raise ValueError("independent gate silent-family grid is incomplete")

    for case in CASES:
        case_rows = frame.loc[frame["case"] == case]
        windows = tuple(sorted(set(case_rows["window_id"])))
        if len(windows) != WINDOWS_PER_CASE:
            raise ValueError(f"independent window grid is incomplete for {case}")
        for window_id in windows:
            rows = case_rows.loc[case_rows["window_id"] == window_id]
            if set(rows["policy"]) != set(POLICIES):
                raise ValueError("an independent window is missing its paired policy")
            branch_identity = rows.loc[
                :,
                [
                    "policy",
                    "trajectory_day",
                    "scenario_seed",
                    "trajectory_seed",
                ],
            ].drop_duplicates()
            if len(branch_identity) != len(POLICIES):
                raise ValueError("policy branch identity changes within a window")
            if (
                branch_identity["trajectory_day"].nunique() != 1
                or branch_identity["scenario_seed"].nunique() != 1
                or branch_identity["trajectory_seed"].nunique() != len(POLICIES)
            ):
                raise ValueError("paired policies do not share day/scenario identity")

    arm_identity = [
        column
        for column in IDENTITY_COLUMNS
        if column not in {"arm", "trajectory_seed"}
    ]
    arm_groups = frame.groupby(
        arm_identity, sort=False, dropna=False
    ).agg(
        row_count=("arm", "size"),
        arm_count=("arm", "nunique"),
        trajectory_seed_count=("trajectory_seed", "nunique"),
    )
    if (
        (arm_groups["row_count"] != len(ARMS)).any()
        or (arm_groups["arm_count"] != len(ARMS)).any()
        or (arm_groups["trajectory_seed_count"] != 1).any()
    ):
        raise ValueError("independent model-arm rows are not exactly paired")

    policy_identity = [
        "case",
        "window_id",
        "trajectory_day",
        "scenario_seed",
        "model_seed",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "anchor",
        "horizon",
        "arm",
    ]
    branches = frame.loc[
        :,
        [*policy_identity, "policy", "trajectory_seed", "cell_id"],
    ].drop_duplicates()
    policy_groups = branches.groupby(
        policy_identity, sort=False, dropna=False
    ).agg(
        row_count=("policy", "size"),
        policy_count=("policy", "nunique"),
        trajectory_seed_count=("trajectory_seed", "nunique"),
    )
    if (
        (policy_groups["row_count"] != len(POLICIES)).any()
        or (policy_groups["policy_count"] != len(POLICIES)).any()
    ):
        raise ValueError("independent policy branches are not exactly paired")
    if (policy_groups["trajectory_seed_count"] != len(POLICIES)).any():
        raise ValueError("paired policy branches reuse a trajectory seed")
    return frame.copy()


def _equal_weight_scores(
    frame: pd.DataFrame, *, retain_family: bool
) -> pd.DataFrame:
    """Reduce errors in the protocol order, independently of the primary gate."""

    selected = frame.loc[
        (frame["horizon"] == 8) & frame["family"].isin(SILENT_FAMILIES)
    ].copy()
    keys = [
        *CLUSTER_COLUMNS,
        "arm",
        "family",
        "fault_channel",
        "sign",
        "severity",
        "onset",
    ]
    # First average anchors inside each onset.  Every frozen fault onset has an
    # equal anchor grid; checking that balance prevents an accidental row-count
    # weighting from masquerading as the prespecified onset mean.
    anchor_counts = selected.groupby(keys, sort=True, dropna=False)["anchor"].nunique()
    if anchor_counts.empty or anchor_counts.nunique() != 1:
        raise ValueError("primary fault onsets do not share an equal anchor grid")
    current = (
        selected.groupby(keys, as_index=False, sort=True, dropna=False)[
            "standardized_abs_error"
        ]
        .mean()
        .rename(columns={"standardized_abs_error": "error"})
    )
    dimensions = ["onset", "sign", "severity", "fault_channel"]
    if not retain_family:
        dimensions.append("family")
    for dimension in dimensions:
        remaining = [
            column
            for column in current.columns
            if column not in {dimension, "error"}
        ]
        current = current.groupby(
            remaining, as_index=False, sort=True, dropna=False
        )["error"].mean()
    index = [*CLUSTER_COLUMNS]
    if retain_family:
        index.append("family")
    pivoted = current.pivot(index=index, columns="arm", values="error").reset_index()
    pivoted.columns.name = None
    if set(pivoted.columns) != set(index) | set(ARMS):
        raise ValueError("independent score cube is missing or adds a model arm")
    return pivoted.sort_values(index, kind="stable").reset_index(drop=True)


def _effect(values: Mapping[str, float | np.ndarray], estimand: str):
    if estimand == "A":
        numerator = values["ungated_h8"]
        denominator = values["deterministic_wm"]
    elif estimand == "D":
        numerator = values["ungated_h8"]
        denominator = values["legacy"]
    else:
        raise ValueError(f"unknown independent estimand: {estimand}")
    denominator_array = np.asarray(denominator)
    if (
        not np.isfinite(denominator_array).all()
        or (denominator_array <= 0.0).any()
    ):
        raise ValueError("independent relative-effect denominator is not positive")
    return 1.0 - numerator / denominator


def _bootstrap_indices(
    scores: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    windows = {
        case: tuple(sorted(set(scores.loc[scores["case"] == case, "window_id"])))
        for case in CASES
    }
    if any(len(values) != WINDOWS_PER_CASE for values in windows.values()):
        raise ValueError("independent bootstrap window grid is incomplete")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    case_indices = rng.integers(
        0, len(CASES), size=(BOOTSTRAP_DRAWS, len(CASES))
    )
    seed_indices = rng.integers(
        0, len(MODEL_SEEDS), size=(BOOTSTRAP_DRAWS, len(MODEL_SEEDS))
    )
    window_indices = {
        case: rng.integers(
            0,
            len(case_windows),
            size=(BOOTSTRAP_DRAWS, len(case_windows)),
        )
        for case, case_windows in windows.items()
    }
    return case_indices, seed_indices, window_indices, windows


def _arm_statistics(
    scores: pd.DataFrame,
    policy: str,
    case_indices: np.ndarray,
    seed_indices: np.ndarray,
    window_indices: Mapping[str, np.ndarray],
    windows: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    selected = scores.loc[scores["policy"] == policy]
    case_points: list[np.ndarray] = []
    case_resamples: list[np.ndarray] = []
    for case in CASES:
        case_windows = windows[case]
        matrix = np.empty(
            (len(MODEL_SEEDS), len(case_windows), len(ARMS)), dtype=np.float64
        )
        case_rows = selected.loc[selected["case"] == case]
        for seed_index, model_seed in enumerate(MODEL_SEEDS):
            for window_index, window_id in enumerate(case_windows):
                rows = case_rows.loc[
                    (case_rows["model_seed"] == model_seed)
                    & (case_rows["window_id"] == window_id)
                ]
                if len(rows) != 1:
                    raise ValueError(
                        "independent seed/window score matrix is incomplete"
                    )
                matrix[seed_index, window_index] = rows.loc[:, ARMS].to_numpy(
                    dtype=np.float64
                )[0]
        sampled = matrix[
            seed_indices[:, :, None],
            window_indices[case][:, None, :],
            :,
        ]
        case_points.append(matrix.mean(axis=(0, 1)))
        case_resamples.append(sampled.mean(axis=(1, 2)))
    point = np.asarray(case_points).mean(axis=0)
    by_case = np.stack(case_resamples, axis=1)
    sampled_cases = by_case[
        np.arange(BOOTSTRAP_DRAWS)[:, None],
        case_indices,
        :,
    ].mean(axis=1)
    return (
        {arm: float(point[index]) for index, arm in enumerate(ARMS)},
        {arm: sampled_cases[:, index] for index, arm in enumerate(ARMS)},
    )


def _summary(point: float, draws: np.ndarray) -> dict[str, float]:
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


def _stratum_effects(
    scores: pd.DataFrame, policy: str, estimand: str, column: str
) -> dict[str, float]:
    selected = scores.loc[scores["policy"] == policy]
    result: dict[str, float] = {}
    for identity, rows in selected.groupby(column, sort=True, dropna=False):
        means = {arm: float(rows[arm].mean()) for arm in ARMS}
        result[str(identity)] = float(_effect(means, estimand))
    return result


def _classify(
    estimand: str,
    summary: Mapping[str, float],
    by_case: Mapping[str, float],
    by_family: Mapping[str, float],
    by_seed: Mapping[str, float],
) -> str:
    positive = (
        summary["point"] >= DOMINANCE_MARGIN
        and summary["ci95_lower"] > 0.0
        and all(value > 0.0 for value in by_case.values())
        and all(value > 0.0 for value in by_family.values())
        and sum(value > 0.0 for value in by_seed.values()) >= 4
    )
    negative = (
        summary["point"] <= -DOMINANCE_MARGIN
        and summary["ci95_upper"] < 0.0
        and all(value < 0.0 for value in by_case.values())
        and all(value < 0.0 for value in by_family.values())
        and sum(value < 0.0 for value in by_seed.values()) >= 4
    )
    equivalent = (
        summary["ci90_lower"] >= -EQUIVALENCE_MARGIN
        and summary["ci90_upper"] <= EQUIVALENCE_MARGIN
        and all(abs(value) <= STRATUM_EQUIVALENCE_LIMIT for value in by_case.values())
        and all(
            abs(value) <= STRATUM_EQUIVALENCE_LIMIT
            for value in by_family.values()
        )
    )
    if positive:
        return "RSSM_ADVANTAGE" if estimand == "A" else "H8_BENEFIT"
    if negative:
        return "DETERMINISTIC_WM_ADVANTAGE" if estimand == "A" else "H8_HARM"
    if equivalent:
        return "PRACTICAL_EQUIVALENCE"
    return "INCONCLUSIVE"


def reconstruct_gate(frame: pd.DataFrame) -> dict:
    """Reconstruct the full frozen result without calling the primary gate."""

    validated = validate_core(frame)
    scores = _equal_weight_scores(validated, retain_family=False)
    family_scores = _equal_weight_scores(validated, retain_family=True)
    case_indices, seed_indices, window_indices, windows = _bootstrap_indices(scores)

    arm_points: dict[str, dict[str, float]] = {}
    arm_draws: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, dict[str, dict]] = {}
    for policy in POLICIES:
        arm_points[policy], arm_draws[policy] = _arm_statistics(
            scores,
            policy,
            case_indices,
            seed_indices,
            window_indices,
            windows,
        )
        results[policy] = {}
        for estimand in ("A", "D"):
            point = float(_effect(arm_points[policy], estimand))
            draws = np.asarray(_effect(arm_draws[policy], estimand), dtype=np.float64)
            summary = _summary(point, draws)
            by_case = _stratum_effects(scores, policy, estimand, "case")
            by_family = _stratum_effects(
                family_scores, policy, estimand, "family"
            )
            by_seed = _stratum_effects(scores, policy, estimand, "model_seed")
            results[policy][estimand] = {
                **summary,
                "category": _classify(
                    estimand, summary, by_case, by_family, by_seed
                ),
                "by_case": by_case,
                "by_family": by_family,
                "by_seed": by_seed,
                "positive_seed_count": sum(value > 0.0 for value in by_seed.values()),
                "negative_seed_count": sum(value < 0.0 for value in by_seed.values()),
            }

    transport: dict[str, dict] = {}
    for estimand in ("A", "D"):
        old_draws = np.asarray(
            _effect(arm_draws["old_2h"], estimand), dtype=np.float64
        )
        new_draws = np.asarray(
            _effect(arm_draws["new_4h"], estimand), dtype=np.float64
        )
        old_point = float(_effect(arm_points["old_2h"], estimand))
        new_point = float(_effect(arm_points["new_4h"], estimand))
        old_category = results["old_2h"][estimand]["category"]
        new_category = results["new_4h"][estimand]["category"]
        transport[estimand] = {
            "new_4h_minus_old_2h": _summary(
                new_point - old_point, new_draws - old_draws
            ),
            "persistent_across_dwell": (
                new_category == old_category and new_category != "INCONCLUSIVE"
            ),
        }

    return {
        "schema": "direct-h8-deterministic-transport-gate-v1",
        "primary_policy": "new_4h",
        "scope": {
            "horizon": 8,
            "families": list(SILENT_FAMILIES),
            "error": "affected-channel standardized absolute error",
            "weighting": (
                "anchors/onsets then sign/severity/channel/family/window/"
                "model_seed/case equally"
            ),
        },
        "margins": {
            "dominance": DOMINANCE_MARGIN,
            "equivalence": EQUIVALENCE_MARGIN,
            "case_family_equivalence": STRATUM_EQUIVALENCE_LIMIT,
        },
        "bootstrap": {
            "method": "paired crossed-case-seed whole-window percentile bootstrap",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "results": results,
        "transport": transport,
        "primary_architecture_category": results["new_4h"]["A"]["category"],
        "primary_supervision_category": results["new_4h"]["D"]["category"],
    }


def assert_semantically_equal(
    recorded: object,
    reconstructed: object,
    *,
    path: str = "$",
) -> None:
    """Require identical JSON structure and close reordered floating arithmetic."""

    if isinstance(recorded, bool) or isinstance(reconstructed, bool):
        if type(recorded) is not bool or type(reconstructed) is not bool:
            raise ValueError(f"independent gate type mismatch at {path}")
        if recorded != reconstructed:
            raise ValueError(f"independent gate boolean mismatch at {path}")
        return
    if isinstance(recorded, dict) or isinstance(reconstructed, dict):
        if not isinstance(recorded, dict) or not isinstance(reconstructed, dict):
            raise ValueError(f"independent gate type mismatch at {path}")
        if set(recorded) != set(reconstructed):
            raise ValueError(f"independent gate object keys mismatch at {path}")
        for key in sorted(recorded):
            assert_semantically_equal(
                recorded[key], reconstructed[key], path=f"{path}.{key}"
            )
        return
    if isinstance(recorded, list) or isinstance(reconstructed, list):
        if not isinstance(recorded, list) or not isinstance(reconstructed, list):
            raise ValueError(f"independent gate type mismatch at {path}")
        if len(recorded) != len(reconstructed):
            raise ValueError(f"independent gate list length mismatch at {path}")
        pairs = zip(recorded, reconstructed, strict=True)
        for index, (left, right) in enumerate(pairs):
            assert_semantically_equal(left, right, path=f"{path}[{index}]")
        return
    if isinstance(recorded, float) or isinstance(reconstructed, float):
        if (
            isinstance(recorded, bool)
            or isinstance(reconstructed, bool)
            or not isinstance(recorded, (int, float))
            or not isinstance(reconstructed, (int, float))
            or not math.isfinite(float(recorded))
            or not math.isfinite(float(reconstructed))
            or not math.isclose(
                float(recorded),
                float(reconstructed),
                rel_tol=FLOAT_REL_TOLERANCE,
                abs_tol=FLOAT_ABS_TOLERANCE,
            )
        ):
            raise ValueError(f"independent gate numeric mismatch at {path}")
        return
    if type(recorded) is not type(reconstructed) or recorded != reconstructed:
        raise ValueError(f"independent gate value mismatch at {path}")


def verify_files(core_csv: Path, gate_result_json: Path) -> dict:
    """Verify a persisted gate result against a separately parsed core CSV."""

    if core_csv.is_symlink() or not core_csv.is_file():
        raise ValueError(
            f"independent verifier input is not a plain file: {core_csv}"
        )
    core = pd.read_csv(core_csv)
    reconstructed = reconstruct_gate(core)
    recorded = _strict_json(gate_result_json)
    assert_semantically_equal(recorded, reconstructed)
    return {
        "schema": "direct-h8-deterministic-transport-independent-verification-v1",
        "verified": True,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "float_absolute_tolerance": FLOAT_ABS_TOLERANCE,
        "float_relative_tolerance": FLOAT_REL_TOLERANCE,
        "primary_architecture_category": reconstructed[
            "primary_architecture_category"
        ],
        "primary_supervision_category": reconstructed[
            "primary_supervision_category"
        ],
        "architecture_persistent_across_dwell": reconstructed["transport"]["A"][
            "persistent_across_dwell"
        ],
        "supervision_persistent_across_dwell": reconstructed["transport"]["D"][
            "persistent_across_dwell"
        ],
        "rows": len(core),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    receipt = verify_files(args.core.resolve(), args.gate_result.resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
