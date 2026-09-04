from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from . import evaluate, gate, plan, report, run_evaluation


def _synthetic_detail() -> pd.DataFrame:
    rows = []
    arm_error = {
        "legacy": 1.0,
        "ungated_h8": 0.75,
        "deterministic_wm": 1.0,
    }
    for case_index, case in enumerate(gate.CASES):
        for window_index in range(gate.WINDOWS_PER_CASE):
            window_id = f"{case}:w{window_index:02d}"
            day = 10 + 9 * window_index
            scenario_seed = 9000 + 100 * case_index + window_index
            for policy_index, policy in enumerate(gate.POLICIES):
                boundary = policy == "old_2h" or bool(window_index % 2)
                trajectory_seed = (
                    20_000 + 1000 * case_index + 2 * window_index + policy_index
                )
                policy_multiplier = 1.0 if policy == "new_4h" else 1.04
                for model_seed in gate.CONFIRMATION_SEEDS:
                    for family_index, family in enumerate(
                        ("healthy", *gate.SILENT_FAMILIES, "dropout")
                    ):
                        for channel_index, (channel, unit, raw_scale) in enumerate(
                            (
                                ("zone_temperature_k", "K", 2.0),
                                ("hvac_electric_power_w", "W", 500.0),
                            )
                        ):
                            cell_id = (
                                f"{case}:{window_id}:{policy}:{family}:{channel}"
                            )
                            for horizon in evaluate.EVALUATION_HORIZONS:
                                horizon_multiplier = 0.4 + 0.6 * horizon / 8.0
                                for arm_index, arm in enumerate(gate.ARMS):
                                    error = (
                                        arm_error[arm]
                                        * policy_multiplier
                                        * horizon_multiplier
                                        * (1.0 + 0.01 * family_index)
                                    )
                                    target_raw = (
                                        295.0 if channel_index == 0 else 4000.0
                                    )
                                    raw_error = error * raw_scale
                                    rows.append(
                                        {
                                            "case": case,
                                            "policy": policy,
                                            "window_id": window_id,
                                            "trajectory_day": day,
                                            "scenario_seed": scenario_seed,
                                            "trajectory_seed": trajectory_seed,
                                            "model_seed": model_seed,
                                            "arm": arm,
                                            "cell_id": cell_id,
                                            "fault_channel": channel,
                                            "family": family,
                                            "sign": 1,
                                            "severity": float(channel_index + 1),
                                            "onset": 48,
                                            "anchor": 64,
                                            "horizon": horizon,
                                            "standardized_abs_error": error,
                                            "update": 400,
                                            "fault_channel_index": channel_index,
                                            "severity_unit": unit,
                                            "history_start": 25,
                                            "history_stop": 64,
                                            "target_index": 64 + horizon,
                                            "target_raw": target_raw,
                                            "prediction_raw": (
                                                target_raw + raw_error
                                            ),
                                            "prediction_standardized": error,
                                            "raw_abs_error": raw_error,
                                            "raw_unit": unit,
                                            "boundary_crossing": boundary,
                                            "action_transition_count": (
                                                1 if boundary else 0
                                            ),
                                            "alternate_action_prediction_raw": (
                                                target_raw
                                                + raw_error
                                                + raw_scale
                                                * (0.05 + 0.01 * arm_index)
                                            ),
                                            "action_prediction_change_standardized": (
                                                0.05 + 0.01 * arm_index
                                            ),
                                        }
                                    )
    return pd.DataFrame(rows, columns=evaluate.DETAIL_COLUMNS)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != run_evaluation.COMPLETION_NAME:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": plan.sha256_file(path),
                }
            )
    return rows


def test_boundary_plot_allows_a_structurally_absent_policy_class(
    tmp_path: Path,
) -> None:
    frame = report.build_boundary_diagnostics(_synthetic_detail())
    frame = frame.loc[
        ~(
            (frame["policy"] == "old_2h")
            & (frame["boundary_class"] == "within_dwell")
        )
    ]
    output = tmp_path / "boundary.png"
    report._plot_boundary(frame, output)
    assert output.is_file() and output.stat().st_size > 0


def _seal_completion(root: Path, gate_result: dict) -> str:
    inventory = _inventory(root)
    completion = {
        "schema": run_evaluation.COMPLETION_SCHEMA,
        "study_kind": "direct_h8_deterministic_transport_v3",
        "prelock_registry_sha256": "1" * 64,
        "readiness_sha256": "2" * 64,
        "corpus_manifest_payload_sha256": "3" * 64,
        "fault_manifest_sha256": "4" * 64,
        "gate_result_sha256": plan.canonical_sha256(gate_result),
        "provenance_file_sha256": plan.sha256_file(
            root / run_evaluation.PROVENANCE_NAME
        ),
        "artifact_inventory_excludes_completion": inventory,
        "artifact_inventory_sha256": plan.canonical_sha256(inventory),
        "complete": True,
    }
    _write_json(root / run_evaluation.COMPLETION_NAME, completion)
    return plan.sha256_file(root / run_evaluation.COMPLETION_NAME)


def _build_evaluation(root: Path) -> str:
    root.mkdir()
    detail = _synthetic_detail()
    core = detail.loc[:, gate.REQUIRED_COLUMNS]
    core.to_csv(
        root / run_evaluation.CORE_NAME,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    detail.to_csv(
        root / run_evaluation.DETAIL_NAME,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    persisted_core = pd.read_csv(root / run_evaluation.CORE_NAME).loc[
        :, gate.REQUIRED_COLUMNS
    ]
    gate_result = gate.analyze_gate(persisted_core)
    _write_json(root / run_evaluation.GATE_RESULT_NAME, gate_result)
    run_evaluation._diagnostic_summary(detail).to_csv(
        root / run_evaluation.DIAGNOSTIC_SUMMARY_NAME,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    _write_json(
        root / run_evaluation.FAULT_MANIFEST_NAME,
        {"schema": "synthetic-report-test-fixture"},
    )

    checkpoint_bytes = {
        f"{case}/seed{seed}/{arm}": 100_000 + arm_index
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
        for arm_index, arm in enumerate(gate.ARMS)
    }
    training_times = {
        f"{case}/seed{seed}": (
            None
            if (case, seed)
            in {
                (gate.CASES[0], gate.CONFIRMATION_SEEDS[0]),
                (gate.CASES[0], gate.CONFIRMATION_SEEDS[1]),
            }
            else 31.0
        )
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
    }
    rows_per_unit = len(core) // (
        len(gate.CASES) * len(gate.CONFIRMATION_SEEDS)
    )
    unit_timing = [
        {
            "case": case,
            "model_seed": seed,
            "model_load_seconds": 0.01,
            "prediction_seconds_by_arm": {
                arm: 0.1 + 0.01 * arm_index
                for arm_index, arm in enumerate(gate.ARMS)
            },
            "unit_wall_seconds": 0.5,
            "core_rows": rows_per_unit,
        }
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
    ]
    provenance = {
        "schema": run_evaluation.PROVENANCE_SCHEMA,
        "study_kind": "direct_h8_deterministic_transport_v3",
        "evaluation_started_at_utc": "2026-08-01T00:00:00Z",
        "evaluation_completed_at_utc": "2026-08-01T00:01:00Z",
        "input_hashes": {},
        "external_freeze": {},
        "runtime": {},
        "runtime_policy": {
            "device": "cpu",
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "deterministic_algorithms": True,
        },
        "evaluation_contract": {
            "cases": list(gate.CASES),
            "model_seeds": list(gate.CONFIRMATION_SEEDS),
            "arms": list(gate.ARMS),
            "policies": list(gate.POLICIES),
            "horizons": list(evaluate.EVALUATION_HORIZONS),
            "history": evaluate.EVALUATION_HISTORY,
            "role": "locked_test",
            "fault_spec": {},
            "bootstrap_draws": gate.BOOTSTRAP_DRAWS,
            "bootstrap_seed": gate.BOOTSTRAP_SEED,
        },
        "counts": {
            "clean_trajectories": 72,
            "fault_variants": 72 * 44,
            "gate_core_rows": len(core),
            "detailed_rows": len(detail),
            "diagnostic_summary_rows": 1,
            "evaluation_units": 15,
        },
        "model_resources": {
            "rssm_total_parameters": 38_072,
            "rssm_active_observation_dynamics_parameters": 19_784,
            "deterministic_total_parameters": 19_789,
            "selected_checkpoint_bytes": checkpoint_bytes,
            "training": {
                "updates_per_model": 400,
                "deterministic": {
                    "timing_source_file_sha256": "5" * 64,
                    "wall_seconds_by_case_seed": training_times,
                    "available_count": 13,
                    "unavailable_count": 2,
                    "unavailable_reason": (
                        "two runs predated the complete grid receipt"
                    ),
                },
                "rssm": {
                    "available": False,
                    "reason": (
                        "parent checkpoint provenance has no training wall time"
                    ),
                },
            },
        },
        "timing": {
            "total_wall_seconds": 60.0,
            "unit_timing": unit_timing,
            "peak_rss_kb_before_evaluation": 100_000,
            "peak_rss_kb_after_evaluation": 200_000,
        },
    }
    _write_json(root / run_evaluation.PROVENANCE_NAME, provenance)
    return _seal_completion(root, gate_result)


@pytest.fixture(scope="module")
def sealed_evaluation(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("report-input") / "evaluation"
    return root, _build_evaluation(root)


def test_generate_and_verify_fixed_publication_artifacts(
    tmp_path: Path, sealed_evaluation: tuple[Path, str]
) -> None:
    source, completion_sha = sealed_evaluation
    output = tmp_path / "report"
    report.generate_report(
        source,
        output,
        expected_completion_sha256=completion_sha,
    )
    manifest_sha = plan.sha256_file(output / report.REPORT_RECEIPT_NAME)
    verified = report.verify_report(
        output, expected_manifest_sha256=manifest_sha
    )
    assert verified["verified"] is True
    expected = {
        *report.TABLE_FILES,
        *report.MARKDOWN_TABLE_FILES,
        *report.FIGURE_FILES,
        report.RESULTS_NAME,
        report.REPORT_RECEIPT_NAME,
    }
    assert {path.name for path in output.iterdir()} == expected
    for figure_name in report.FIGURE_FILES:
        assert (output / figure_name).stat().st_size > 5_000

    primary = pd.read_csv(output / "primary_estimands.csv")
    assert len(primary) == 4
    assert set(primary["policy"]) == set(gate.POLICIES)
    assert set(primary["estimand"]) == {"A", "D"}
    assert set(primary["category"]) == {"RSSM_ADVANTAGE", "H8_BENEFIT"}
    raw = pd.read_csv(output / "raw_unit_mae.csv")
    assert set(raw["raw_unit"]) == {"K", "W"}
    assert len(raw) == 2 * 4 * 3 * 2
    horizon = pd.read_csv(output / "horizon_arm_mae.csv")
    assert len(horizon) == 2 * 4 * 3
    families = pd.read_csv(output / "fault_family_diagnostics.csv")
    assert set(families["family"]) == {
        "healthy",
        "bias",
        "drift",
        "stuck",
        "dropout",
    }
    resources = pd.read_csv(output / "model_resources.csv")
    deterministic = resources.loc[
        resources["arm"] == "deterministic_wm"
    ].iloc[0]
    assert deterministic["training_timed_model_count"] == 13
    assert deterministic["training_untimed_model_count"] == 2
    rssm = resources.loc[resources["arm"] == "legacy"].iloc[0]
    assert pd.isna(rssm["training_wall_seconds_available_sum"])

    results = (output / report.RESULTS_NAME).read_text(encoding="ascii")
    assert "RSSM_ADVANTAGE" in results
    assert "H8_BENEFIT" in results
    assert "not establish intrinsic superiority" in results
    assert "closed-loop" in results


def test_report_is_byte_deterministic(
    tmp_path: Path, sealed_evaluation: tuple[Path, str]
) -> None:
    source, completion_sha = sealed_evaluation
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        report.generate_report(
            source,
            output,
            expected_completion_sha256=completion_sha,
        )
    first = {
        path.name: plan.sha256_file(path)
        for path in outputs[0].iterdir()
        if path.is_file()
    }
    second = {
        path.name: plan.sha256_file(path)
        for path in outputs[1].iterdir()
        if path.is_file()
    }
    assert first == second


def test_wrong_completion_hash_fails_closed(
    tmp_path: Path, sealed_evaluation: tuple[Path, str]
) -> None:
    source, _ = sealed_evaluation
    with pytest.raises(ValueError, match="completion receipt differs"):
        report.generate_report(
            source,
            tmp_path / "report",
            expected_completion_sha256="f" * 64,
        )


def test_semantically_changed_gate_fails_even_with_resealed_receipt(
    tmp_path: Path, sealed_evaluation: tuple[Path, str]
) -> None:
    source, _ = sealed_evaluation
    changed = tmp_path / "changed-gate"
    shutil.copytree(source, changed)
    gate_path = changed / run_evaluation.GATE_RESULT_NAME
    payload = json.loads(gate_path.read_text(encoding="ascii"))
    payload["results"]["new_4h"]["A"]["point"] += 0.01
    _write_json(gate_path, payload)
    completion_sha = _seal_completion(changed, payload)
    with pytest.raises(ValueError, match="does not reconstruct"):
        report.generate_report(
            changed,
            tmp_path / "report",
            expected_completion_sha256=completion_sha,
        )


def test_changed_detail_schema_fails_even_with_resealed_receipt(
    tmp_path: Path, sealed_evaluation: tuple[Path, str]
) -> None:
    source, _ = sealed_evaluation
    changed = tmp_path / "changed-detail"
    shutil.copytree(source, changed)
    detail_path = changed / run_evaluation.DETAIL_NAME
    detail = pd.read_csv(detail_path).rename(
        columns={"raw_abs_error": "raw_error"}
    )
    detail.to_csv(detail_path, index=False, lineterminator="\n")
    gate_result = json.loads(
        (changed / run_evaluation.GATE_RESULT_NAME).read_text(encoding="ascii")
    )
    completion_sha = _seal_completion(changed, gate_result)
    with pytest.raises(ValueError, match="detail schema changed"):
        report.generate_report(
            changed,
            tmp_path / "report",
            expected_completion_sha256=completion_sha,
        )
