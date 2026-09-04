"""Tune, execute, analyze, and seal the multi-case downstream study."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from building_fault_wm.downstream_control import experiment as v1
from building_fault_wm.neural_benchmark import protocol as boptest

from . import protocol
from .controllers import controller_for


TUNING_SCHEMA = "direct-h8-multicase-rule-tuning-v2"
RESULT_SCHEMA = "direct-h8-downstream-multicase-result-v2"
MANIFEST_SCHEMA = "direct-h8-downstream-multicase-manifest-v2"
SUMMARY_ENDPOINTS = (
    "operational_score",
    "cost_tot",
    "tdis_tot",
    "control_cost_proxy",
    "control_discomfort_kh",
    "control_energy_kwh",
)
FINAL_CONTRASTS = (
    ("legacy_rssm", "constant_zero"),
    ("direct_h8_rssm", "constant_zero"),
    ("deterministic_wm", "constant_zero"),
    ("rc_supervisory_selector", "constant_zero"),
    ("tuned_rule_based", "constant_zero"),
    ("direct_h8_rssm", "legacy_rssm"),
    ("deterministic_wm", "direct_h8_rssm"),
    ("rc_supervisory_selector", "deterministic_wm"),
    ("tuned_rule_based", "deterministic_wm"),
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )


def file_inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    excluded = {"report_manifest.json", "report_manifest.canonical.sha256"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": protocol.sha256_file(path),
                }
            )
    return rows


def seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )


def server_version(base_url: str) -> str:
    value = v1.BoptestClient(base_url).payload("GET", "/version")
    if not isinstance(value, dict):
        raise ValueError("BOPTEST version payload is invalid")
    version = str(value.get("version"))
    if version != boptest.BOPTEST_API_VERSION:
        raise ValueError("live BOPTEST version differs from the fixed version")
    return version


@contextmanager
def case_runtime(
    case: str,
    frozen_case: Mapping[str, object],
    rule_parameters: protocol.RuleParameters,
) -> Iterator[None]:
    original_protocol = v1.protocol
    original_controller = v1.EnsembleController
    runtime = protocol.runtime_protocol(case, frozen_case)

    def factory(policy_name: str, frozen: Mapping[str, object]) -> object:
        return controller_for(
            policy_name,
            frozen,
            original_controller,
            rule_parameters,
        )

    v1.protocol = runtime
    v1.EnsembleController = factory  # type: ignore[assignment]
    try:
        yield
    finally:
        v1.EnsembleController = original_controller
        v1.protocol = original_protocol


def run_episode(
    *,
    base_url: str,
    case: str,
    frozen_case: Mapping[str, object],
    rule_parameters: protocol.RuleParameters,
    window: Mapping[str, object],
    condition: str,
    policy_name: str,
    steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    with case_runtime(case, frozen_case, rule_parameters):
        trajectory, decisions, summary = v1.run_episode(
            v1.BoptestClient(base_url),
            frozen_case,
            window,
            condition,
            policy_name,
            steps=steps,
        )
    cost_budget, discomfort_budget = protocol.score_budgets(trajectory)
    summary = {
        "case": case,
        **summary,
        "cost_budget": cost_budget,
        "discomfort_budget": discomfort_budget,
    }
    trajectory.insert(0, "case", case)
    decisions.insert(0, "case", case)
    return trajectory, decisions, summary


def _worker_episode(
    task: tuple[
        str,
        str,
        Mapping[str, object],
        protocol.RuleParameters,
        Mapping[str, object],
        str,
        str,
        int,
    ]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    (
        base_url,
        case,
        frozen_case,
        rule_parameters,
        window,
        condition,
        policy_name,
        steps,
    ) = task
    torch.set_num_threads(1)
    return run_episode(
        base_url=base_url,
        case=case,
        frozen_case=frozen_case,
        rule_parameters=rule_parameters,
        window=window,
        condition=condition,
        policy_name=policy_name,
        steps=steps,
    )


def process_pool(workers: int) -> concurrent.futures.ProcessPoolExecutor:
    if workers < 1 or workers > protocol.PARALLEL_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {protocol.PARALLEL_WORKERS}"
        )
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _candidate_id(parameters: protocol.RuleParameters) -> str:
    margin = str(parameters.margin_k).replace(".", "p")
    quantile = str(parameters.price_quantile).replace(".", "p")
    return f"margin_{margin}_price_q{quantile}"


def _empty_case_inputs(case: str) -> dict[str, object]:
    return {"inputs": protocol.case_inputs(case)}


def tune_rule_controller(
    *,
    base_url: str,
    output: Path = protocol.TUNING_ROOT,
    workers: int = protocol.PARALLEL_WORKERS,
) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite tuning output: {output}")
    staging = output.parent / f".{output.name}.staging"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale tuning staging exists: {staging}")
    staging.mkdir(parents=True)
    started = time.time()
    version = server_version(base_url)
    implementation = {
        str(path.relative_to(protocol.PROJECT_ROOT)): {
            "sha256": protocol.sha256_file(path)
        }
        for path in protocol.IMPLEMENTATION_PATHS
    }
    case_payloads: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    try:
        with process_pool(workers) as executor:
            for case in protocol.CASES:
                print(f"tuning case={case}", flush=True)
                frozen_case = _empty_case_inputs(case)
                windows = protocol.load_development_windows(case)
                grid = protocol.rule_grid(case)
                neutral: dict[int, dict[str, object]] = {}
                identities: dict[int, tuple[str, str]] = {}
                neutral_rule = grid[0]
                neutral_tasks = [
                    (
                        base_url,
                        case,
                        frozen_case,
                        neutral_rule,
                        window,
                        "clean",
                        "constant_zero",
                        protocol.EPISODE_STEPS,
                    )
                    for window in windows
                ]
                for _, _, summary in executor.map(
                    _worker_episode, neutral_tasks, chunksize=1
                ):
                    day = int(summary["day"])
                    neutral[day] = summary
                    identities[day] = (
                        str(summary["initialized_state_sha256"]),
                        str(summary["forecast_sha256"]),
                    )
                if set(neutral) != {int(window["day"]) for window in windows}:
                    raise ValueError("neutral tuning window grid is incomplete")
                candidate_tasks = [
                    (
                        base_url,
                        case,
                        frozen_case,
                        parameters,
                        window,
                        "clean",
                        "tuned_rule_based",
                        protocol.EPISODE_STEPS,
                    )
                    for parameters in grid
                    for window in windows
                ]
                candidate_results = executor.map(
                    _worker_episode, candidate_tasks, chunksize=1
                )
                results_by_candidate: dict[
                    str, list[tuple[protocol.RuleParameters, dict[str, object]]]
                ] = {_candidate_id(parameters): [] for parameters in grid}
                for task, (_, _, summary) in zip(
                    candidate_tasks, candidate_results, strict=True
                ):
                    parameters = task[3]
                    results_by_candidate[_candidate_id(parameters)].append(
                        (parameters, summary)
                    )
                candidate_summaries = []
                for parameters in grid:
                    candidate_id = _candidate_id(parameters)
                    scores = []
                    action_counts = np.zeros(3, dtype=float)
                    results = results_by_candidate[candidate_id]
                    if len(results) != len(windows):
                        raise ValueError("tuning candidate window grid is incomplete")
                    for _, summary in results:
                        day = int(summary["day"])
                        identity = (
                            str(summary["initialized_state_sha256"]),
                            str(summary["forecast_sha256"]),
                        )
                        if identity != identities[day]:
                            raise ValueError(
                                "tuning branches differ in initialization or forecast"
                            )
                        reference = neutral[day]
                        score = protocol.scalar_score(
                            float(summary["cost_tot"]),
                            float(summary["tdis_tot"]),
                            float(reference["cost_tot"]),
                            float(reference["tdis_tot"]),
                            float(reference["cost_budget"]),
                            float(reference["discomfort_budget"]),
                        )
                        scores.append(score)
                        action_counts += np.asarray(
                            [
                                float(summary["action_minus_fraction"]),
                                float(summary["action_zero_fraction"]),
                                float(summary["action_plus_fraction"]),
                            ]
                        )
                        all_rows.append(
                            {
                                "case": case,
                                "candidate_id": candidate_id,
                                "margin_k": parameters.margin_k,
                                "price_quantile": parameters.price_quantile,
                                "price_threshold": parameters.price_threshold,
                                "day": day,
                                "operational_score": score,
                                "cost_tot": float(summary["cost_tot"]),
                                "tdis_tot": float(summary["tdis_tot"]),
                                "neutral_cost_tot": float(reference["cost_tot"]),
                                "neutral_tdis_tot": float(reference["tdis_tot"]),
                                "cost_budget": float(reference["cost_budget"]),
                                "discomfort_budget": float(
                                    reference["discomfort_budget"]
                                ),
                                "action_minus_fraction": float(
                                    summary["action_minus_fraction"]
                                ),
                                "action_zero_fraction": float(
                                    summary["action_zero_fraction"]
                                ),
                                "action_plus_fraction": float(
                                    summary["action_plus_fraction"]
                                ),
                            }
                        )
                    score_array = np.asarray(scores, dtype=float)
                    mean_action = action_counts / len(windows)
                    candidate_summaries.append(
                        {
                            "candidate_id": candidate_id,
                            "margin_k": parameters.margin_k,
                            "price_quantile": parameters.price_quantile,
                            "price_threshold": parameters.price_threshold,
                            "windows": len(windows),
                            "median_operational_score": float(
                                np.median(score_array)
                            ),
                            "mean_operational_score": float(np.mean(score_array)),
                            "improved_windows": int(
                                np.count_nonzero(score_array < 0.0)
                            ),
                            "action_minus_fraction": float(mean_action[0]),
                            "action_zero_fraction": float(mean_action[1]),
                            "action_plus_fraction": float(mean_action[2]),
                        }
                    )
                selected = min(
                    candidate_summaries,
                    key=lambda item: (
                        float(item["median_operational_score"]),
                        float(item["mean_operational_score"]),
                        float(item["margin_k"]),
                        float(item["price_quantile"]),
                    ),
                )
                action_fractions = np.asarray(
                    [
                        selected["action_minus_fraction"],
                        selected["action_zero_fraction"],
                        selected["action_plus_fraction"],
                    ],
                    dtype=float,
                )
                competence = {
                    "finite": bool(
                        np.isfinite(
                            [
                                selected["median_operational_score"],
                                selected["mean_operational_score"],
                                *action_fractions,
                            ]
                        ).all()
                    ),
                    "median_score_not_above_neutral": bool(
                        float(selected["median_operational_score"]) <= 0.0
                    ),
                    "wins_at_least_half_of_windows": bool(
                        int(selected["improved_windows"]) >= len(windows) // 2
                    ),
                    "uses_at_least_two_action_levels": bool(
                        np.count_nonzero(action_fractions > 0.0) >= 2
                    ),
                    "not_more_than_95_percent_one_action": bool(
                        float(action_fractions.max()) < 0.95
                    ),
                }
                competence["passed"] = all(competence.values())
                case_payloads[case] = {
                    "development_windows": windows,
                    "price_thresholds": {
                        str(key): value
                        for key, value in protocol.price_thresholds(case).items()
                    },
                    "grid": candidate_summaries,
                    "selected": selected,
                    "competence_gate": competence,
                    "inputs": {
                        str(path.relative_to(protocol.PROJECT_ROOT)): {
                            "sha256": protocol.sha256_file(path)
                        }
                        for path in protocol.development_csvs(case)
                    },
                }
        all_passed = all(
            bool(dict(value)["competence_gate"]["passed"])
            for value in case_payloads.values()
        )
        frame = pd.DataFrame(all_rows).sort_values(
            ["case", "candidate_id", "day"], kind="stable"
        )
        frame.to_csv(
            staging / "tuning_rows.csv", index=False, float_format="%.17g"
        )
        current_implementation = {
            str(path.relative_to(protocol.PROJECT_ROOT)): {
                "sha256": protocol.sha256_file(path)
            }
            for path in protocol.IMPLEMENTATION_PATHS
        }
        if current_implementation != implementation:
            raise ValueError("implementation changed during rule-based tuning")
        payload = {
            "schema": TUNING_SCHEMA,
            "selection": (
                "minimum median operational score, then mean score, margin, "
                "and price quantile"
            ),
            "outcome_scope": "clean development-validation windows only",
            "final_window_responses_used": False,
            "rule_grid": {
                "margins_k": list(protocol.RULE_MARGIN_GRID_K),
                "price_quantiles": list(protocol.RULE_PRICE_QUANTILES),
            },
            "score": {
                "cost_reference_power_density_kw_m2": (
                    protocol.COST_REFERENCE_POWER_DENSITY_KW_M2
                ),
                "cost_weight": protocol.COST_WEIGHT,
                "discomfort_weight": protocol.DISCOMFORT_WEIGHT,
            },
            "boptest_version": version,
            "parallel_workers": workers,
            "implementation": implementation,
            "cases": case_payloads,
            "all_competence_gates_passed": all_passed,
            "wall_seconds": time.time() - started,
            "rows_file": {
                "path": "tuning_rows.csv",
                "sha256": protocol.sha256_file(staging / "tuning_rows.csv"),
            },
        }
        payload = {**payload, "payload_sha256": protocol.canonical_sha256(payload)}
        write_json(staging / "tuning.json", payload)
        staging.rename(output)
        seal_tree(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def paired_bootstrap(
    values: np.ndarray, seed_key: str, draws: int = 10_000
) -> tuple[float, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap values are invalid")
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("ascii")).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def attach_operational_scores(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "case",
        "day",
        "condition",
        "policy",
        "cost_tot",
        "tdis_tot",
        "cost_budget",
        "discomfort_budget",
    }
    if not required.issubset(frame):
        raise ValueError("episode summary lacks operational-score inputs")
    result = frame.copy()
    reference = result.loc[
        result.policy == "constant_zero",
        [
            "case",
            "day",
            "condition",
            "cost_tot",
            "tdis_tot",
            "cost_budget",
            "discomfort_budget",
        ],
    ].rename(
        columns={
            "cost_tot": "neutral_cost_tot",
            "tdis_tot": "neutral_tdis_tot",
            "cost_budget": "neutral_cost_budget",
            "discomfort_budget": "neutral_discomfort_budget",
        }
    )
    if reference.duplicated(["case", "day", "condition"]).any():
        raise ValueError("neutral branch is duplicated")
    result = result.merge(
        reference, on=["case", "day", "condition"], how="left", validate="many_to_one"
    )
    if result[["neutral_cost_tot", "neutral_tdis_tot"]].isna().any().any():
        raise ValueError("neutral branch is missing")
    if not np.allclose(result.cost_budget, result.neutral_cost_budget) or not np.allclose(
        result.discomfort_budget, result.neutral_discomfort_budget
    ):
        raise ValueError("operational budgets differ across paired policies")
    result["operational_score"] = [
        protocol.scalar_score(*values)
        for values in zip(
            result.cost_tot,
            result.tdis_tot,
            result.neutral_cost_tot,
            result.neutral_tdis_tot,
            result.neutral_cost_budget,
            result.neutral_discomfort_budget,
            strict=True,
        )
    ]
    return result


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for case in protocol.CASES:
        case_rows = frame.loc[frame.case == case]
        for condition in protocol.CONDITIONS:
            condition_rows = case_rows.loc[case_rows.condition == condition]
            for policy_name in protocol.POLICIES:
                selected = condition_rows.loc[condition_rows.policy == policy_name]
                if len(selected) != 12:
                    raise ValueError("final case-condition-policy grid is incomplete")
                for endpoint in SUMMARY_ENDPOINTS:
                    values = selected[endpoint].to_numpy(dtype=float)
                    aggregate_rows.append(
                        {
                            "case": case,
                            "condition": condition,
                            "policy": policy_name,
                            "endpoint": endpoint,
                            "windows": len(values),
                            "median": float(np.median(values)),
                            "mean": float(np.mean(values)),
                            "minimum": float(np.min(values)),
                            "maximum": float(np.max(values)),
                        }
                    )
            for candidate_name, reference_name in FINAL_CONTRASTS:
                candidate = condition_rows.loc[
                    condition_rows.policy == candidate_name
                ].set_index("day")
                reference = condition_rows.loc[
                    condition_rows.policy == reference_name
                ].set_index("day")
                if not candidate.index.equals(reference.index):
                    raise ValueError("paired final window grids differ")
                for endpoint in SUMMARY_ENDPOINTS:
                    differences = candidate[endpoint].to_numpy(dtype=float) - reference[
                        endpoint
                    ].to_numpy(dtype=float)
                    low, high = paired_bootstrap(
                        differences,
                        f"{case}:{condition}:{candidate_name}:{reference_name}:{endpoint}",
                    )
                    paired_rows.append(
                        {
                            "case": case,
                            "condition": condition,
                            "candidate": candidate_name,
                            "reference": reference_name,
                            "endpoint": endpoint,
                            "windows": len(differences),
                            "median_paired_difference": float(np.median(differences)),
                            "mean_paired_difference": float(np.mean(differences)),
                            "ci95_low": low,
                            "ci95_high": high,
                            "improved_windows": int(np.count_nonzero(differences < 0.0)),
                            "tied_windows": int(np.count_nonzero(differences == 0.0)),
                        }
                    )
    aggregate = pd.DataFrame(aggregate_rows)
    paired = pd.DataFrame(paired_rows)
    for scope, cases in (
        ("response_unseen_pair", protocol.RESPONSE_UNSEEN_CASES),
        ("all_three_cases", protocol.CASES),
    ):
        for condition in protocol.CONDITIONS:
            for policy_name in protocol.POLICIES:
                selected = aggregate.loc[
                    aggregate.case.isin(cases)
                    & (aggregate.condition == condition)
                    & (aggregate.policy == policy_name)
                    & (aggregate.endpoint == "operational_score")
                ]
                if len(selected) != len(cases):
                    raise ValueError("finite-panel case summary is incomplete")
                values = selected.set_index("case").loc[list(cases), "median"].to_numpy(
                    dtype=float
                )
                panel_rows.append(
                    {
                        "scope": scope,
                        "condition": condition,
                        "policy": policy_name,
                        "cases": len(cases),
                        "case_median_scores": json.dumps(
                            {case: float(value) for case, value in zip(cases, values, strict=True)},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "mean_of_case_medians": float(np.mean(values)),
                        "median_of_case_medians": float(np.median(values)),
                        "population_inference": False,
                    }
                )
    return aggregate, paired, pd.DataFrame(panel_rows)


def run_study(
    *,
    base_url: str,
    output: Path,
    pilot: bool,
    workers: int = protocol.PARALLEL_WORKERS,
) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite downstream output: {output}")
    frozen = protocol.validate_frozen_protocol()
    server_version(base_url)
    staging = output.parent / f".{output.name}.staging"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale downstream staging exists: {staging}")
    staging.mkdir(parents=True)
    started = time.time()
    summaries: list[dict[str, object]] = []
    identities: dict[tuple[str, int], tuple[str, str]] = {}
    try:
        tasks: list[
            tuple[
                str,
                str,
                Mapping[str, object],
                protocol.RuleParameters,
                Mapping[str, object],
                str,
                str,
                int,
            ]
        ] = []
        for case in protocol.CASES:
            frozen_case = dict(dict(frozen["cases"])[case])
            rule = protocol.RuleParameters(**dict(frozen_case["rule_parameters"]))
            if pilot:
                day = protocol.PILOT_DAYS[case]
                windows = [
                    {
                        "day": day,
                        "window_id": f"{case}:pilot:day{day:03d}",
                        "scenario_seed": boptest.stable_seed(
                            "downstream-multicase-pilot-v2", case, day
                        ),
                        "temperature_stratum": -1,
                    }
                ]
                conditions: Sequence[str] = ("clean", "zone_bias_positive")
                steps = 96
            else:
                windows = list(frozen_case["windows"])
                conditions = protocol.CONDITIONS
                steps = protocol.EPISODE_STEPS
            for window in windows:
                for condition in conditions:
                    for policy_name in protocol.POLICIES:
                        tasks.append(
                            (
                                base_url,
                                case,
                                frozen_case,
                                rule,
                                window,
                                condition,
                                policy_name,
                                steps,
                            )
                        )
        with process_pool(workers) as executor:
            results = executor.map(_worker_episode, tasks, chunksize=1)
            for index, (task, result) in enumerate(
                zip(tasks, results, strict=True), start=1
            ):
                _, case, _, _, window, condition, policy_name, _ = task
                trajectory, decisions, summary = result
                key = (case, int(window["day"]))
                identity = (
                    str(summary["initialized_state_sha256"]),
                    str(summary["forecast_sha256"]),
                )
                if key in identities and identities[key] != identity:
                    raise ValueError(
                        "paired branches differ in initialization or forecast"
                    )
                identities[key] = identity
                stem = (
                    f"{case}_day{int(window['day']):03d}_{condition}_{policy_name}"
                )
                trajectory.to_csv(
                    staging / f"{stem}_trajectory.csv",
                    index=False,
                    float_format="%.17g",
                )
                decisions.to_csv(
                    staging / f"{stem}_decisions.csv",
                    index=False,
                    float_format="%.17g",
                )
                summaries.append(summary)
                print(
                    f"completed={index}/{len(tasks)} case={case} day={window['day']} "
                    f"condition={condition} policy={policy_name}",
                    flush=True,
                )
        summary_frame = attach_operational_scores(
            pd.DataFrame(summaries).sort_values(
                ["case", "day", "condition", "policy"], kind="stable"
            )
        )
        summary_frame.to_csv(
            staging / "episode_summary.csv", index=False, float_format="%.17g"
        )
        if not pilot:
            aggregate, paired, panel = analyze(summary_frame)
            aggregate.to_csv(
                staging / "aggregate_summary.csv", index=False, float_format="%.17g"
            )
            paired.to_csv(
                staging / "paired_effects.csv", index=False, float_format="%.17g"
            )
            panel.to_csv(
                staging / "finite_panel_summary.csv", index=False, float_format="%.17g"
            )
        metadata = {
            "schema": RESULT_SCHEMA,
            "pilot": pilot,
            "protocol_file_sha256": protocol.sha256_file(protocol.PROTOCOL_PATH),
            "protocol_canonical_sha256": protocol.canonical_sha256(frozen),
            "episodes": len(summary_frame),
            "cases": list(protocol.CASES),
            "response_unseen_cases": list(protocol.RESPONSE_UNSEEN_CASES),
            "wall_seconds": time.time() - started,
            "device": "cpu",
            "parallel_workers": workers,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }
        write_json(staging / "run_metadata.json", metadata)
        manifest = {"schema": MANIFEST_SCHEMA, "files": file_inventory(staging)}
        manifest = {**manifest, "payload_sha256": protocol.canonical_sha256(manifest)}
        write_json(staging / "report_manifest.json", manifest)
        (staging / "report_manifest.canonical.sha256").write_text(
            protocol.canonical_sha256(manifest) + "\n", encoding="ascii"
        )
        staging.rename(output)
        seal_tree(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tune", "pilot", "run"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers", type=int, default=protocol.PARALLEL_WORKERS
    )
    args = parser.parse_args()
    if args.mode == "tune":
        output = args.output or protocol.TUNING_ROOT
        print(
            tune_rule_controller(
                base_url=args.base_url, output=output, workers=args.workers
            )
        )
    else:
        output = args.output or (
            protocol.DEFAULT_PILOT_OUTPUT if args.mode == "pilot" else protocol.DEFAULT_OUTPUT
        )
        print(
            run_study(
                base_url=args.base_url,
                output=output,
                pilot=args.mode == "pilot",
                workers=args.workers,
            )
        )


if __name__ == "__main__":
    main()
