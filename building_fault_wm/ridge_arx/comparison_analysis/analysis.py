"""Fail-closed paired inference for frozen neural and Ridge-ARX outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from building_fault_wm.deterministic_transport import (
    gate as v3_gate,
    run_evaluation as v3_run,
)
from building_fault_wm.ridge_arx import (
    evaluate as arx_evaluate,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)


SCHEMA = "schedule-matched-arx-neural-comparison-v1"
COMPLETION_SCHEMA = "schedule-matched-arx-neural-comparison-completion-v1"
PRIMARY_NEURAL_ARM = "deterministic_wm"
ARX_ARM = arx_evaluate.ARM
ALL_ARMS = (*v3_gate.ARMS, ARX_ARM)
PRIMARY_POLICY = "new_4h"
CONTROL_POLICY = "old_2h"
HORIZON = 8
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202608029
DOMINANCE_MARGIN = 0.10
EQUIVALENCE_MARGIN = 0.05
STRATUM_EQUIVALENCE_LIMIT = 0.10
HORIZONS = (1, 2, 4, 8)
BASE_CLUSTER = ("case", "policy", "window_id", "model_seed")
WEIGHTING_DIMENSIONS = ("family", "fault_channel", "sign", "severity")
PAIR_COLUMNS = tuple(
    column
    for column in v3_gate.REQUIRED_COLUMNS
    if column not in {"arm", "standardized_abs_error"}
)
OUTPUT_FILES = (
    "paired_rows.csv",
    "equal_weight_h8_scores.csv",
    "descriptive_by_horizon.csv",
    "comparison_result.json",
    "analysis_provenance.json",
)


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
    values = np.asarray(denominator, dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("ARX comparison denominator must be positive and finite")
    return numerator / denominator


def _effect(
    neural: float | np.ndarray, arx: float | np.ndarray
) -> float | np.ndarray:
    return 1.0 - _safe_ratio(neural, arx)


def validate_and_pair(
    v3_frame: pd.DataFrame, arx_frame: pd.DataFrame
) -> pd.DataFrame:
    """Return exact row pairs after both complete upstream schemas validate."""

    v3 = v3_gate.validate_input(v3_frame)
    arx_evaluate._validate_secondary_rows(arx_frame)
    arx = arx_frame.copy()
    if (arx["standardized_abs_error"] < 0.0).any():
        raise ValueError("ARX comparison input contains a negative absolute error")
    v3 = v3.loc[
        v3["family"].isin(v3_gate.SILENT_FAMILIES)
    ].copy()
    if set(arx["family"]) != set(v3_gate.SILENT_FAMILIES):
        raise ValueError("ARX comparison silent-family grid changed")

    v3_keys = set(map(tuple, v3.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None)))
    arx_keys = set(
        map(tuple, arx.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None))
    )
    if len(v3_keys) != len(v3) // len(v3_gate.ARMS):
        raise ValueError("v3 comparison rows are duplicated within an arm")
    if len(arx_keys) != len(arx):
        raise ValueError("ARX comparison rows are duplicated")
    if v3_keys != arx_keys:
        missing = len(v3_keys - arx_keys)
        extra = len(arx_keys - v3_keys)
        raise ValueError(
            f"neural/ARX row identities are not exact pairs: missing={missing}, extra={extra}"
        )

    selected = pd.concat([v3, arx], ignore_index=True)
    if selected.duplicated([*PAIR_COLUMNS, "arm"]).any():
        raise ValueError("comparison contains a duplicate arm-row identity")
    arm_sets = selected.groupby(
        list(PAIR_COLUMNS), dropna=False, sort=False
    )["arm"].agg(lambda values: frozenset(values))
    expected_arms = frozenset(ALL_ARMS)
    if any(arms != expected_arms for arms in arm_sets):
        raise ValueError("comparison row does not contain every exact model arm")
    paired = selected.pivot(
        index=list(PAIR_COLUMNS),
        columns="arm",
        values="standardized_abs_error",
    ).reset_index()
    paired.columns.name = None
    if set(ALL_ARMS) - set(paired.columns):
        raise ValueError("comparison pivot is missing a model arm")
    return paired.sort_values(list(PAIR_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def equal_weight_scores(
    paired: pd.DataFrame,
    *,
    horizon: int,
    retain_family: bool = False,
) -> pd.DataFrame:
    """Apply the frozen v3 equal-stratum hierarchy to all four paired arms."""

    selected = paired.loc[paired["horizon"] == horizon].copy()
    if selected.empty:
        raise ValueError(f"comparison has no rows for H{horizon}")
    retained: tuple[str, ...] = ("family",) if retain_family else ()
    first_group = [
        *BASE_CLUSTER,
        *retained,
        *(
            dimension
            for dimension in WEIGHTING_DIMENSIONS
            if dimension not in retained
        ),
    ]
    result = selected.groupby(first_group, as_index=False, dropna=False)[
        list(ALL_ARMS)
    ].mean()
    group = list(first_group)
    for dimension in ("sign", "severity", "fault_channel", "family"):
        if dimension in retained:
            continue
        remaining = [column for column in group if column != dimension]
        result = result.groupby(remaining, as_index=False, dropna=False)[
            list(ALL_ARMS)
        ].mean()
        group = remaining
    expected_rows = (
        len(v3_gate.CASES)
        * len(v3_gate.POLICIES)
        * v3_gate.WINDOWS_PER_CASE
        * len(v3_gate.CONFIRMATION_SEEDS)
        * (len(v3_gate.SILENT_FAMILIES) if retain_family else 1)
    )
    if len(result) != expected_rows:
        raise ValueError("equal-weight comparison score grid is incomplete")
    return result.sort_values(group, kind="stable").reset_index(drop=True)


def _bootstrap_plan(
    scores: pd.DataFrame, *, draws: int, seed: int
) -> BootstrapPlan:
    if draws <= 0:
        raise ValueError("comparison bootstrap draws must be positive")
    cases = tuple(v3_gate.CASES)
    seeds = tuple(v3_gate.CONFIRMATION_SEEDS)
    windows = {
        case: tuple(
            sorted(set(scores.loc[scores["case"] == case, "window_id"]))
        )
        for case in cases
    }
    if any(len(values) != v3_gate.WINDOWS_PER_CASE for values in windows.values()):
        raise ValueError("comparison bootstrap window grid is incomplete")
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
            raise ValueError("comparison bootstrap seed/window matrix is incomplete")
        matrix = rows.loc[index, list(ALL_ARMS)].to_numpy(dtype=float).reshape(
            len(plan.seeds), len(windows), len(ALL_ARMS)
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
        {arm: float(point[index]) for index, arm in enumerate(ALL_ARMS)},
        {arm: selected_cases[:, index] for index, arm in enumerate(ALL_ARMS)},
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
    scores: pd.DataFrame, policy: str, column: str
) -> dict[str, float]:
    result = {}
    for identity, rows in scores.loc[
        scores["policy"] == policy
    ].groupby(column, sort=True, dropna=False):
        result[str(identity)] = float(
            _effect(
                float(rows[PRIMARY_NEURAL_ARM].mean()),
                float(rows[ARX_ARM].mean()),
            )
        )
    return result


def _classify(
    summary: Mapping[str, float],
    by_case: Mapping[str, float],
    by_family: Mapping[str, float],
    by_seed: Mapping[str, float],
) -> str:
    neural_advantage = (
        summary["point"] >= DOMINANCE_MARGIN
        and summary["ci95_lower"] > 0.0
        and all(value > 0.0 for value in by_case.values())
        and all(value > 0.0 for value in by_family.values())
        and sum(value > 0.0 for value in by_seed.values()) >= 4
    )
    arx_advantage = (
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
    if neural_advantage:
        return "DETERMINISTIC_WM_ADVANTAGE_OVER_ARX"
    if arx_advantage:
        return "ARX_ADVANTAGE_OVER_DETERMINISTIC_WM"
    if equivalent:
        return "PRACTICAL_EQUIVALENCE"
    return "INCONCLUSIVE"


def descriptive_by_horizon(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        scores = equal_weight_scores(paired, horizon=horizon)
        for (case, policy), block in scores.groupby(
            ["case", "policy"], sort=True, dropna=False
        ):
            for arm in ALL_ARMS:
                rows.append(
                    {
                        "horizon": horizon,
                        "case": case,
                        "policy": policy,
                        "arm": arm,
                        "mean_standardized_mae": float(block[arm].mean()),
                        "std_across_seed_window_clusters": float(
                            block[arm].std(ddof=1)
                        ),
                        "clusters": len(block),
                        "claim_scope": (
                            "inferential_primary"
                            if horizon == HORIZON
                            and arm in {PRIMARY_NEURAL_ARM, ARX_ARM}
                            else "descriptive_only"
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "case", "policy", "arm"], kind="stable"
    ).reset_index(drop=True)


def analyze_frames(
    v3_frame: pd.DataFrame,
    arx_frame: pd.DataFrame,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze already loaded frames under the frozen secondary contract."""

    paired = validate_and_pair(v3_frame, arx_frame)
    scores = equal_weight_scores(paired, horizon=HORIZON)
    family_scores = equal_weight_scores(
        paired, horizon=HORIZON, retain_family=True
    )
    plan = _bootstrap_plan(scores, draws=bootstrap_draws, seed=bootstrap_seed)
    arm_points: dict[str, dict[str, float]] = {}
    arm_draws: dict[str, dict[str, np.ndarray]] = {}
    policies = {}
    for policy in v3_gate.POLICIES:
        arm_points[policy], arm_draws[policy] = _bootstrap_arm_means(
            scores, policy, plan
        )
        draws = np.asarray(
            _effect(
                arm_draws[policy][PRIMARY_NEURAL_ARM],
                arm_draws[policy][ARX_ARM],
            ),
            dtype=float,
        )
        point = float(
            _effect(
                arm_points[policy][PRIMARY_NEURAL_ARM],
                arm_points[policy][ARX_ARM],
            )
        )
        summary = _summary(point, draws)
        by_case = _stratum_effects(scores, policy, "case")
        by_family = _stratum_effects(family_scores, policy, "family")
        by_seed = _stratum_effects(scores, policy, "model_seed")
        policies[policy] = {
            **summary,
            "category": _classify(summary, by_case, by_family, by_seed),
            "mean_standardized_mae_by_arm": arm_points[policy],
            "by_case": by_case,
            "by_family": by_family,
            "by_seed": by_seed,
            "positive_seed_count": sum(value > 0.0 for value in by_seed.values()),
            "negative_seed_count": sum(value < 0.0 for value in by_seed.values()),
        }

    new_draw = np.asarray(
        _effect(
            arm_draws[PRIMARY_POLICY][PRIMARY_NEURAL_ARM],
            arm_draws[PRIMARY_POLICY][ARX_ARM],
        ),
        dtype=float,
    )
    old_draw = np.asarray(
        _effect(
            arm_draws[CONTROL_POLICY][PRIMARY_NEURAL_ARM],
            arm_draws[CONTROL_POLICY][ARX_ARM],
        ),
        dtype=float,
    )
    transport_point = (
        policies[PRIMARY_POLICY]["point"] - policies[CONTROL_POLICY]["point"]
    )
    exploratory = {
        policy: {
            arm: float(
                _effect(arm_points[policy][arm], arm_points[policy][ARX_ARM])
            )
            for arm in ("legacy", "ungated_h8")
        }
        for policy in v3_gate.POLICIES
    }
    result = {
        "schema": SCHEMA,
        "secondary_only": True,
        "primary_policy": PRIMARY_POLICY,
        "control_policy": CONTROL_POLICY,
        "primary_horizon": HORIZON,
        "families": list(v3_gate.SILENT_FAMILIES),
        "primary_estimand": (
            "1 - equal_weight_MAE(deterministic_wm) / "
            "equal_weight_MAE(schedule_matched_recursive_ridge_arx)"
        ),
        "positive_favors": PRIMARY_NEURAL_ARM,
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "generator": "numpy.random.PCG64",
            "hierarchy": "case/model_seed/window_within_case",
            "paired_across_arms_and_policies": True,
        },
        "margins": {
            "dominance_point_threshold": DOMINANCE_MARGIN,
            "equivalence_tost_margin": EQUIVALENCE_MARGIN,
            "case_family_equivalence_limit": STRATUM_EQUIVALENCE_LIMIT,
        },
        "policy_results": policies,
        "transport": {
            "new_4h_minus_old_2h": _summary(
                float(transport_point), new_draw - old_draw
            ),
            "persistent_across_dwell": (
                policies[PRIMARY_POLICY]["category"]
                == policies[CONTROL_POLICY]["category"]
                and policies[PRIMARY_POLICY]["category"] != "INCONCLUSIVE"
            ),
        },
        "descriptive_only_rssm_relative_effects": exploratory,
        "claim_scope": {
            "inferential": (
                "H8 deterministic_wm versus ARX on silent faults; new_4h "
                "primary and old_2h persistence control"
            ),
            "descriptive_only": (
                "H1/H2/H4 and legacy/ungated_h8 RSSM comparisons"
            ),
            "forbidden": [
                "generic architecture superiority",
                "physical parameter identification",
                "observed-building generalization",
                "closed-loop control",
                "planning or MPC",
                "energy, cost, or comfort benefit",
            ],
        },
    }
    return result, paired, scores, descriptive_by_horizon(paired)


def _verify_v3_output(root: Path) -> tuple[Path, dict]:
    completion_path = root / v3_run.COMPLETION_NAME
    completion = strict_json(completion_path)
    completion_fields = {
        "schema",
        "study_kind",
        "prelock_registry_sha256",
        "readiness_sha256",
        "corpus_manifest_payload_sha256",
        "fault_manifest_sha256",
        "gate_result_sha256",
        "provenance_file_sha256",
        "artifact_inventory_excludes_completion",
        "artifact_inventory_sha256",
        "complete",
    }
    expected_files = {
        v3_run.FAULT_MANIFEST_NAME,
        v3_run.CORE_NAME,
        v3_run.DETAIL_NAME,
        v3_run.DIAGNOSTIC_SUMMARY_NAME,
        v3_run.GATE_RESULT_NAME,
        v3_run.PROVENANCE_NAME,
    }
    if (
        set(completion) != completion_fields
        or completion.get("schema") != v3_run.COMPLETION_SCHEMA
        or completion.get("study_kind")
        != "direct_h8_deterministic_transport_v3"
        or completion.get("complete") is not True
    ):
        raise ValueError("v3 comparison input is not a complete evaluation")
    inventory = completion.get("artifact_inventory_excludes_completion")
    if (
        not isinstance(inventory, list)
        or {row.get("path") for row in inventory if isinstance(row, dict)}
        != expected_files
        or completion.get("artifact_inventory_sha256")
        != canonical_sha256(inventory)
        or tree_inventory(root, exclude={v3_run.COMPLETION_NAME}) != inventory
    ):
        raise ValueError("v3 comparison input inventory changed")
    core = root / v3_run.CORE_NAME
    rows = [row for row in inventory if row.get("path") == v3_run.CORE_NAME]
    if len(rows) != 1 or rows[0].get("sha256") != sha256_file(core):
        raise ValueError("v3 gate-core hash changed")
    gate_result = strict_json(root / v3_run.GATE_RESULT_NAME)
    if completion.get("gate_result_sha256") != canonical_sha256(gate_result):
        raise ValueError("v3 gate-result hash changed")
    if completion.get("provenance_file_sha256") != sha256_file(
        root / v3_run.PROVENANCE_NAME
    ):
        raise ValueError("v3 provenance hash changed")
    return core, completion


def _verify_arx_output(root: Path) -> tuple[Path, dict]:
    completion_path = root / "evaluation_complete.json"
    completion = strict_json(completion_path)
    hashes = completion.get("file_sha256_by_name")
    if (
        set(completion)
        != {
            "schema",
            "secondary_only",
            "cannot_modify_v2_or_v3_gate",
            "row_count",
            "file_sha256_by_name",
        }
        or completion.get("schema") != arx_evaluate.COMPLETION_SCHEMA
        or completion.get("secondary_only") is not True
        or completion.get("cannot_modify_v2_or_v3_gate") is not True
        or not isinstance(hashes, dict)
        or set(hashes) != {
            "arx_core.csv",
            "arx_detailed_diagnostics.csv",
            "arx_descriptive_summary.csv",
            "evaluation_provenance.json",
        }
    ):
        raise ValueError("ARX comparison input is not a complete evaluation")
    expected_files = {*hashes, "evaluation_complete.json"}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("ARX comparison input file set changed")
    for name, digest in hashes.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"ARX comparison input hash changed: {name}")
    return root / "arx_core.csv", completion


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    return write_once(
        path,
        frame.to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        ).encode("ascii"),
    )


def run_bound_analysis(
    *,
    v3_output_root: Path,
    arx_output_root: Path,
    freeze_bundle_root: Path,
    public_freeze_receipt_path: Path,
    output_root: Path,
    live_public_freeze: bool = True,
) -> Path:
    """Verify the prospective freeze and upstream hashes before reading values."""

    from .freeze import verify_local_freeze_bundle
    from .public_freeze import validate_public_freeze_receipt

    freeze_registry = verify_local_freeze_bundle(freeze_bundle_root)
    public_freeze = validate_public_freeze_receipt(
        public_freeze_receipt_path,
        freeze_bundle_root,
        live=live_public_freeze,
    )
    v3_core, v3_completion = _verify_v3_output(v3_output_root)
    arx_core, arx_completion = _verify_arx_output(arx_output_root)
    # No result CSV is opened before the prospective analysis freeze validates.
    v3_frame = pd.read_csv(v3_core, float_precision="round_trip")
    arx_frame = pd.read_csv(arx_core, float_precision="round_trip")
    if arx_completion.get("row_count") != len(arx_frame):
        raise ValueError("ARX completion row count changed")
    result, paired, scores, descriptive = analyze_frames(v3_frame, arx_frame)
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite comparison analysis: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    paired_path = _write_frame(output_root / OUTPUT_FILES[0], paired)
    scores_path = _write_frame(output_root / OUTPUT_FILES[1], scores)
    descriptive_path = _write_frame(output_root / OUTPUT_FILES[2], descriptive)
    result_path = write_json_once(output_root / OUTPUT_FILES[3], result)
    provenance = {
        "schema": "schedule-matched-arx-neural-comparison-provenance-v1",
        "analysis_freeze_registry_sha256": canonical_sha256(freeze_registry),
        "public_freeze_receipt_file_sha256": sha256_file(
            public_freeze_receipt_path
        ),
        "public_freeze_revision": public_freeze["revision"],
        "v3_completion_file_sha256": sha256_file(
            v3_output_root / v3_run.COMPLETION_NAME
        ),
        "v3_core_file_sha256": sha256_file(v3_core),
        "v3_completion_payload_sha256": canonical_sha256(v3_completion),
        "arx_completion_file_sha256": sha256_file(
            arx_output_root / "evaluation_complete.json"
        ),
        "arx_core_file_sha256": sha256_file(arx_core),
        "arx_completion_payload_sha256": canonical_sha256(arx_completion),
        "result_payload_sha256": canonical_sha256(result),
        "input_values_opened_only_after_freeze_validation": True,
    }
    provenance_path = write_json_once(
        output_root / OUTPUT_FILES[4], provenance
    )
    hashes = {
        path.name: sha256_file(path)
        for path in (
            paired_path,
            scores_path,
            descriptive_path,
            result_path,
            provenance_path,
        )
    }
    completion = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "secondary_only": True,
        "file_sha256_by_name": hashes,
        "result_payload_sha256": canonical_sha256(result),
    }
    return write_json_once(output_root / "analysis_complete.json", completion)
