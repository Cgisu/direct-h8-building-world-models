from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from building_fault_wm.deterministic_transport import gate as v3_gate
from building_fault_wm.ridge_arx import (
    evaluate as arx_evaluate,
)

from .analysis import (
    ARX_ARM,
    _verify_arx_output,
    analyze_frames,
    run_bound_analysis,
    validate_and_pair,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    tree_inventory,
)
from building_fault_wm.deterministic_transport import (
    run_evaluation as v3_run,
)
from . import freeze, public_freeze


def synthetic_inputs(
    *,
    deterministic_error: float = 0.8,
    arx_error: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arm_errors = {
        "legacy": 1.1,
        "ungated_h8": 0.9,
        "deterministic_wm": deterministic_error,
        ARX_ARM: arx_error,
    }
    v3_rows = []
    arx_rows = []
    for case_index, case in enumerate(v3_gate.CASES):
        for window_index in range(v3_gate.WINDOWS_PER_CASE):
            window = f"{case}-window-{window_index:02d}"
            scenario_seed = 10_000 + 100 * case_index + window_index
            for policy_index, policy in enumerate(v3_gate.POLICIES):
                trajectory_seed = scenario_seed * 10 + policy_index
                for model_seed in v3_gate.CONFIRMATION_SEEDS:
                    for family in v3_gate.SILENT_FAMILIES:
                        for channel in (
                            "zone_temperature_k",
                            "hvac_electric_power_w",
                        ):
                            for horizon in (1, 2, 4, 8):
                                identity = {
                                    "case": case,
                                    "policy": policy,
                                    "window_id": window,
                                    "trajectory_day": 10 + window_index,
                                    "scenario_seed": scenario_seed,
                                    "trajectory_seed": trajectory_seed,
                                    "model_seed": model_seed,
                                    "cell_id": (
                                        f"{case}:{trajectory_seed}:{family}:{channel}"
                                    ),
                                    "fault_channel": channel,
                                    "family": family,
                                    "sign": 1,
                                    "severity": 1.0,
                                    "onset": 32,
                                    "anchor": 40,
                                    "horizon": horizon,
                                }
                                for arm in v3_gate.ARMS:
                                    v3_rows.append(
                                        {
                                            **identity,
                                            "arm": arm,
                                            "standardized_abs_error": (
                                                arm_errors[arm]
                                            ),
                                        }
                                    )
                                arx_rows.append(
                                    {
                                        **identity,
                                        "arm": ARX_ARM,
                                        "standardized_abs_error": (
                                            arm_errors[ARX_ARM]
                                        ),
                                    }
                                )
    return (
        pd.DataFrame(v3_rows, columns=v3_gate.REQUIRED_COLUMNS),
        pd.DataFrame(arx_rows, columns=arx_evaluate.CORE_COLUMNS),
    )


def test_primary_neural_advantage_and_persistence_are_classified() -> None:
    v3, arx = synthetic_inputs(deterministic_error=0.8, arx_error=1.0)
    result, paired, scores, descriptive = analyze_frames(
        v3, arx, bootstrap_draws=64
    )
    assert len(paired) == len(arx)
    assert len(scores) == 360
    assert not descriptive.empty
    for policy in v3_gate.POLICIES:
        assert result["policy_results"][policy]["point"] == pytest.approx(0.2)
        assert (
            result["policy_results"][policy]["category"]
            == "DETERMINISTIC_WM_ADVANTAGE_OVER_ARX"
        )
    assert result["transport"]["persistent_across_dwell"] is True


def test_practical_equivalence_uses_the_five_percent_interval() -> None:
    v3, arx = synthetic_inputs(deterministic_error=0.98, arx_error=1.0)
    result, *_ = analyze_frames(v3, arx, bootstrap_draws=64)
    assert (
        result["policy_results"]["new_4h"]["category"]
        == "PRACTICAL_EQUIVALENCE"
    )


def test_duplicate_neural_row_is_rejected() -> None:
    v3, arx = synthetic_inputs()
    duplicated = pd.concat([v3, v3.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_and_pair(duplicated, arx)


def test_missing_arx_cell_is_rejected() -> None:
    v3, arx = synthetic_inputs()
    with pytest.raises(ValueError, match="paired policies|not exact pairs"):
        validate_and_pair(v3, arx.iloc[:-1].copy())


def test_changed_trajectory_identity_is_rejected() -> None:
    v3, arx = synthetic_inputs()
    changed = arx.copy()
    changed.loc[0, "trajectory_seed"] += 999
    with pytest.raises(ValueError, match="not exact pairs"):
        validate_and_pair(v3, changed)


def test_upstream_arx_hash_drift_is_rejected(tmp_path: Path) -> None:
    names = (
        "arx_core.csv",
        "arx_detailed_diagnostics.csv",
        "arx_descriptive_summary.csv",
        "evaluation_provenance.json",
    )
    for name in names:
        (tmp_path / name).write_text(f"{name}\n", encoding="ascii")
    completion = {
        "schema": arx_evaluate.COMPLETION_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "row_count": 1,
        "file_sha256_by_name": {
            name: sha256_file(tmp_path / name) for name in names
        },
    }
    (tmp_path / "evaluation_complete.json").write_text(
        json.dumps(completion) + "\n", encoding="ascii"
    )
    _verify_arx_output(tmp_path)
    (tmp_path / "arx_core.csv").write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="hash changed"):
        _verify_arx_output(tmp_path)


def test_bound_analysis_runs_only_from_exact_completed_inputs(
    tmp_path: Path,
) -> None:
    v3, arx = synthetic_inputs()
    freeze_root = freeze.prepare_local_freeze_bundle(tmp_path / "freeze")
    gist = "c" * 32
    revision = "d" * 40
    owner = "owner"
    receipt = {
        "schema": public_freeze.SCHEMA,
        "provider": public_freeze.PROVIDER,
        "public": True,
        "prelock_registry_sha256": public_freeze._prelock_digest(freeze_root),
        "gist_id": gist,
        "revision": revision,
        "owner_login": owner,
        "provider_created_at_utc": "2026-08-01T00:00:00Z",
        "provider_updated_at_utc": "2026-08-01T00:01:00Z",
        "revision_committed_at_utc": "2026-08-01T00:00:30Z",
        "revision_api_url": f"https://api.github.com/gists/{gist}/{revision}",
        "revision_html_url": (
            f"https://gist.github.com/{owner}/{gist}/{revision}"
        ),
        "file_sha256_by_name": public_freeze.expected_file_hashes(freeze_root),
    }
    receipt_path = tmp_path / "public_freeze_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="ascii")

    v3_root = tmp_path / "v3"
    v3_root.mkdir()
    v3.to_csv(
        v3_root / v3_run.CORE_NAME,
        index=False,
        float_format="%.17g",
    )
    (v3_root / v3_run.FAULT_MANIFEST_NAME).write_text(
        "{}\n", encoding="ascii"
    )
    (v3_root / v3_run.DETAIL_NAME).write_text("detail\n", encoding="ascii")
    (v3_root / v3_run.DIAGNOSTIC_SUMMARY_NAME).write_text(
        "summary\n", encoding="ascii"
    )
    gate_result = {"schema": "synthetic-gate"}
    (v3_root / v3_run.GATE_RESULT_NAME).write_text(
        json.dumps(gate_result) + "\n", encoding="ascii"
    )
    (v3_root / v3_run.PROVENANCE_NAME).write_text(
        "{}\n", encoding="ascii"
    )
    inventory = tree_inventory(v3_root)
    v3_completion = {
        "schema": v3_run.COMPLETION_SCHEMA,
        "study_kind": "direct_h8_deterministic_transport_v3",
        "prelock_registry_sha256": "a" * 64,
        "readiness_sha256": "b" * 64,
        "corpus_manifest_payload_sha256": "c" * 64,
        "fault_manifest_sha256": "d" * 64,
        "gate_result_sha256": canonical_sha256(gate_result),
        "provenance_file_sha256": sha256_file(
            v3_root / v3_run.PROVENANCE_NAME
        ),
        "artifact_inventory_excludes_completion": inventory,
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "complete": True,
    }
    (v3_root / v3_run.COMPLETION_NAME).write_text(
        json.dumps(v3_completion) + "\n", encoding="ascii"
    )

    arx_root = tmp_path / "arx"
    arx_root.mkdir()
    arx.to_csv(
        arx_root / "arx_core.csv", index=False, float_format="%.17g"
    )
    for name in (
        "arx_detailed_diagnostics.csv",
        "arx_descriptive_summary.csv",
        "evaluation_provenance.json",
    ):
        (arx_root / name).write_text(f"{name}\n", encoding="ascii")
    arx_hashes = {
        name: sha256_file(arx_root / name)
        for name in (
            "arx_core.csv",
            "arx_detailed_diagnostics.csv",
            "arx_descriptive_summary.csv",
            "evaluation_provenance.json",
        )
    }
    arx_completion = {
        "schema": arx_evaluate.COMPLETION_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "row_count": len(arx),
        "file_sha256_by_name": arx_hashes,
    }
    (arx_root / "evaluation_complete.json").write_text(
        json.dumps(arx_completion) + "\n", encoding="ascii"
    )
    complete = run_bound_analysis(
        v3_output_root=v3_root,
        arx_output_root=arx_root,
        freeze_bundle_root=freeze_root,
        public_freeze_receipt_path=receipt_path,
        output_root=tmp_path / "result",
        live_public_freeze=False,
    )
    payload = json.loads(complete.read_text(encoding="ascii"))
    assert payload["complete"] is True
