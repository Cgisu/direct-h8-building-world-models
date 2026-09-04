"""Paired hierarchical inference for the deterministic transport extension."""

from __future__ import annotations

from dataclasses import dataclass
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
CONFIRMATION_SEEDS = (
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
ROW_ID_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS if column != "standardized_abs_error"
)
BASE_CLUSTER = ("case", "policy", "window_id", "model_seed")


@dataclass(frozen=True)
class BootstrapPlan:
    cases: tuple[str, ...]
    seeds: tuple[int, ...]
    windows: dict[str, tuple[str, ...]]
    sampled_cases: np.ndarray
    sampled_seeds: np.ndarray
    sampled_windows: dict[str, np.ndarray]


def _safe_ratio(
    numerator: float | np.ndarray, denominator: float | np.ndarray
) -> float | np.ndarray:
    values = np.asarray(denominator)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("relative-effect denominator must be positive and finite")
    return numerator / denominator


def _effect(values: Mapping[str, float | np.ndarray], estimand: str):
    if estimand == "A":
        return 1.0 - _safe_ratio(
            values["ungated_h8"], values["deterministic_wm"]
        )
    if estimand == "D":
        return 1.0 - _safe_ratio(values["ungated_h8"], values["legacy"])
    raise ValueError(f"unknown v3 estimand: {estimand}")


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError("v3 gate input columns differ from the frozen schema")
    if frame.empty or frame.duplicated(list(ROW_ID_COLUMNS)).any():
        raise ValueError("v3 gate input is empty or has duplicate row identities")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("v3 gate input contains non-finite values")
    if (frame["standardized_abs_error"] < 0.0).any():
        raise ValueError("v3 gate input contains a negative absolute error")
    if set(frame["case"]) != set(CASES):
        raise ValueError("v3 gate case grid is incomplete")
    if set(frame["policy"]) != set(POLICIES):
        raise ValueError("v3 gate policy grid is incomplete")
    if set(frame["arm"]) != set(ARMS):
        raise ValueError("v3 gate arm grid is incomplete")
    if set(int(value) for value in frame["model_seed"]) != set(CONFIRMATION_SEEDS):
        raise ValueError("v3 gate seed grid is incomplete")
    if not set(SILENT_FAMILIES).issubset(set(frame["family"])):
        raise ValueError("v3 gate silent-fault grid is incomplete")
    for case in CASES:
        windows = set(frame.loc[frame["case"] == case, "window_id"])
        if len(windows) != WINDOWS_PER_CASE:
            raise ValueError(f"v3 gate window grid is incomplete for {case}")
        for window in windows:
            policies = set(
                frame.loc[
                    (frame["case"] == case) & (frame["window_id"] == window),
                    "policy",
                ]
            )
            if policies != set(POLICIES):
                raise ValueError("a v3 window is missing its paired policy branch")

    arm_identity = [
        column
        for column in ROW_ID_COLUMNS
        if column not in {"arm", "trajectory_seed"}
    ]
    grouped = frame.groupby(arm_identity, dropna=False, sort=False)
    if any(set(rows["arm"]) != set(ARMS) for _, rows in grouped):
        raise ValueError("v3 model rows are not exactly paired")
    return frame.copy()


def _pivot_scores(
    frame: pd.DataFrame, *, retain: Sequence[str] = ()
) -> pd.DataFrame:
    selected = frame.loc[
        (frame["horizon"] == 8) & frame["family"].isin(SILENT_FAMILIES)
    ].copy()
    retain = tuple(retain)
    if len(set(retain)) != len(retain) or set(retain) - {"family"}:
        raise ValueError("v3 score strata may retain only the fault family")
    weighting_dimensions = ("family", "fault_channel", "sign", "severity")
    first_group = [
        *BASE_CLUSTER,
        *retain,
        *(
            dimension
            for dimension in weighting_dimensions
            if dimension not in retain
        ),
        "arm",
    ]
    averaged = selected.groupby(first_group, as_index=False, dropna=False)[
        "standardized_abs_error"
    ].mean()
    pivot_index = [column for column in first_group if column != "arm"]
    pivoted = averaged.pivot(
        index=pivot_index,
        columns="arm",
        values="standardized_abs_error",
    ).reset_index()
    pivoted.columns.name = None
    if set(ARMS) - set(pivoted.columns):
        raise ValueError("v3 score pivot is missing a model arm")

    result = pivoted
    group = [
        *BASE_CLUSTER,
        *retain,
        *(
            dimension
            for dimension in weighting_dimensions
            if dimension not in retain
        ),
    ]
    for dimension in ("sign", "severity", "fault_channel", "family"):
        if dimension in retain:
            continue
        remaining = [column for column in group if column != dimension]
        result = result.groupby(remaining, as_index=False, dropna=False)[
            list(ARMS)
        ].mean()
        group = remaining
    return result.sort_values(group, kind="stable").reset_index(drop=True)


def _bootstrap_plan(
    scores: pd.DataFrame, *, draws: int, seed: int
) -> BootstrapPlan:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    cases = tuple(CASES)
    seeds = tuple(CONFIRMATION_SEEDS)
    windows = {}
    for case in cases:
        identities = tuple(
            sorted(set(scores.loc[scores["case"] == case, "window_id"]))
        )
        if len(identities) != WINDOWS_PER_CASE:
            raise ValueError("bootstrap window grid is incomplete")
        windows[case] = identities
    rng = np.random.Generator(np.random.PCG64(seed))
    return BootstrapPlan(
        cases=cases,
        seeds=seeds,
        windows=windows,
        sampled_cases=rng.integers(0, len(cases), size=(draws, len(cases))),
        sampled_seeds=rng.integers(0, len(seeds), size=(draws, len(seeds))),
        sampled_windows={
            case: rng.integers(
                0,
                len(case_windows),
                size=(draws, len(case_windows)),
            )
            for case, case_windows in windows.items()
        },
    )


def _bootstrap_arm_means(
    scores: pd.DataFrame, policy: str, plan: BootstrapPlan
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    selected = scores.loc[scores["policy"] == policy]
    case_points = []
    case_draws = []
    for case in plan.cases:
        windows = plan.windows[case]
        index = pd.MultiIndex.from_product(
            [plan.seeds, windows], names=["model_seed", "window_id"]
        )
        rows = selected.loc[selected["case"] == case].set_index(
            ["model_seed", "window_id"]
        )
        if rows.index.has_duplicates or set(rows.index) != set(index):
            raise ValueError("bootstrap seed/window matrix is incomplete")
        matrix = rows.loc[index, list(ARMS)].to_numpy(dtype=float).reshape(
            len(plan.seeds), len(windows), len(ARMS)
        )
        sampled = matrix[
            plan.sampled_seeds[:, :, None],
            plan.sampled_windows[case][:, None, :],
            :,
        ]
        case_points.append(matrix.mean(axis=(0, 1)))
        case_draws.append(sampled.mean(axis=(1, 2)))
    point = np.stack(case_points).mean(axis=0)
    by_case_draw = np.stack(case_draws, axis=1)
    selected_cases = by_case_draw[
        np.arange(len(plan.sampled_cases))[:, None],
        plan.sampled_cases,
        :,
    ].mean(axis=1)
    return (
        {arm: float(point[index]) for index, arm in enumerate(ARMS)},
        {arm: selected_cases[:, index] for index, arm in enumerate(ARMS)},
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
    result = {}
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
        and all(abs(value) <= STRATUM_EQUIVALENCE_LIMIT for value in by_family.values())
    )
    if positive:
        return "RSSM_ADVANTAGE" if estimand == "A" else "H8_BENEFIT"
    if negative:
        return "DETERMINISTIC_WM_ADVANTAGE" if estimand == "A" else "H8_HARM"
    if equivalent:
        return "PRACTICAL_EQUIVALENCE"
    return "INCONCLUSIVE"


def analyze_gate(
    frame: pd.DataFrame,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Return all frozen v3 estimates and symmetric decision categories."""

    frame = validate_input(frame)
    scores = _pivot_scores(frame)
    family_scores = _pivot_scores(frame, retain=("family",))
    plan = _bootstrap_plan(scores, draws=bootstrap_draws, seed=bootstrap_seed)

    arm_points = {}
    arm_draws = {}
    results: dict[str, dict[str, dict]] = {}
    for policy in POLICIES:
        arm_points[policy], arm_draws[policy] = _bootstrap_arm_means(
            scores, policy, plan
        )
        results[policy] = {}
        for estimand in ("A", "D"):
            point = float(_effect(arm_points[policy], estimand))
            draws = np.asarray(_effect(arm_draws[policy], estimand), dtype=float)
            summary = _summary(point, draws)
            by_case = _stratum_effects(scores, policy, estimand, "case")
            by_family = _stratum_effects(
                family_scores, policy, estimand, "family"
            )
            by_seed = _stratum_effects(scores, policy, estimand, "model_seed")
            category = _classify(
                estimand, summary, by_case, by_family, by_seed
            )
            results[policy][estimand] = {
                **summary,
                "category": category,
                "by_case": by_case,
                "by_family": by_family,
                "by_seed": by_seed,
                "positive_seed_count": sum(value > 0.0 for value in by_seed.values()),
                "negative_seed_count": sum(value < 0.0 for value in by_seed.values()),
            }

    transport = {}
    for estimand in ("A", "D"):
        old_draw = np.asarray(
            _effect(arm_draws["old_2h"], estimand), dtype=float
        )
        new_draw = np.asarray(
            _effect(arm_draws["new_4h"], estimand), dtype=float
        )
        old_point = float(_effect(arm_points["old_2h"], estimand))
        new_point = float(_effect(arm_points["new_4h"], estimand))
        category = results["new_4h"][estimand]["category"]
        old_category = results["old_2h"][estimand]["category"]
        transport[estimand] = {
            "new_4h_minus_old_2h": _summary(
                new_point - old_point, new_draw - old_draw
            ),
            "persistent_across_dwell": (
                category == old_category and category != "INCONCLUSIVE"
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
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
        },
        "results": results,
        "transport": transport,
        "primary_architecture_category": results["new_4h"]["A"]["category"],
        "primary_supervision_category": results["new_4h"]["D"]["category"],
    }
