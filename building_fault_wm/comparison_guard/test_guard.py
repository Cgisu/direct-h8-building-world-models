from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from building_fault_wm.deterministic_transport import (
    run_evaluation as v3_run,
)
from building_fault_wm.ridge_arx import (
    evaluate as arx_evaluate,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    tree_inventory,
    write_json_once,
)

from . import freeze as identity_freeze
from . import guard
from . import public_freeze as identity_public_freeze


def base_contract() -> dict:
    hashes = {
        "prelock_registry_sha256": guard.EXPECTED_TRANSPORT_PRELOCK_SHA256,
        "readiness_sha256": guard.EXPECTED_TRANSPORT_READINESS_SHA256,
        "external_freeze_receipt_file_sha256": "a" * 64,
        "corpus_manifest_file_sha256": "b" * 64,
        "corpus_manifest_payload_sha256": "c" * 64,
        "collection_attempt_file_sha256": "d" * 64,
        "collection_completion_file_sha256": "e" * 64,
    }
    return {
        "schema": guard.SCHEMA,
        "outcome_values_accessed": False,
        "transport": {
            "prelock_registry_sha256": (
                guard.EXPECTED_TRANSPORT_PRELOCK_SHA256
            ),
            "readiness_sha256": guard.EXPECTED_TRANSPORT_READINESS_SHA256,
        },
        "neural_evaluation": {
            "completion_schema": v3_run.COMPLETION_SCHEMA,
            "provenance_schema": v3_run.PROVENANCE_SCHEMA,
            "study_kind": "direct_h8_deterministic_transport_v3",
            "expected_input_hashes": hashes,
            "external_freeze_revision": "1" * 40,
            "evaluation_contract": {"synthetic": "fixed"},
        },
        "arx": {
            "prelock_registry_sha256": "2" * 64,
            "external_freeze_receipt_file_sha256": "3" * 64,
            "external_freeze_revision": "4" * 40,
            "transport_binding_payload_sha256": "5" * 64,
            "transport_manifest_file_sha256": "b" * 64,
            "config": {"synthetic": "fixed"},
            "model_file_sha256_by_case_seed": {"case/seed1": "6" * 64},
        },
        "comparison": {
            "prelock_registry_sha256": (
                guard.EXPECTED_COMPARISON_PRELOCK_SHA256
            )
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="ascii")


def write_v3_output(
    root: Path,
    contract: dict,
    *,
    prelock: str | None = None,
    readiness: str | None = None,
) -> None:
    root.mkdir()
    expected = contract["neural_evaluation"]
    inputs = dict(expected["expected_input_hashes"])
    inputs["prelock_registry_sha256"] = (
        prelock or guard.EXPECTED_TRANSPORT_PRELOCK_SHA256
    )
    inputs["readiness_sha256"] = (
        readiness or guard.EXPECTED_TRANSPORT_READINESS_SHA256
    )
    (root / v3_run.CORE_NAME).write_text(
        "this fixture must never be parsed\n", encoding="ascii"
    )
    _write_json(root / v3_run.FAULT_MANIFEST_NAME, {"synthetic": True})
    (root / v3_run.DETAIL_NAME).write_text("detail\n", encoding="ascii")
    (root / v3_run.DIAGNOSTIC_SUMMARY_NAME).write_text(
        "summary\n", encoding="ascii"
    )
    gate_result = {"schema": "synthetic-gate"}
    _write_json(root / v3_run.GATE_RESULT_NAME, gate_result)
    provenance = {
        "schema": expected["provenance_schema"],
        "study_kind": expected["study_kind"],
        "input_hashes": inputs,
        "external_freeze": {"revision": expected["external_freeze_revision"]},
        "evaluation_contract": expected["evaluation_contract"],
    }
    _write_json(root / v3_run.PROVENANCE_NAME, provenance)
    inventory = tree_inventory(root)
    completion = {
        "schema": expected["completion_schema"],
        "study_kind": expected["study_kind"],
        "prelock_registry_sha256": inputs["prelock_registry_sha256"],
        "readiness_sha256": inputs["readiness_sha256"],
        "corpus_manifest_payload_sha256": inputs[
            "corpus_manifest_payload_sha256"
        ],
        "fault_manifest_sha256": "7" * 64,
        "gate_result_sha256": canonical_sha256(gate_result),
        "provenance_file_sha256": sha256_file(
            root / v3_run.PROVENANCE_NAME
        ),
        "artifact_inventory_excludes_completion": inventory,
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "complete": True,
    }
    _write_json(root / v3_run.COMPLETION_NAME, completion)


def write_arx_output(
    root: Path, contract: dict, *, prelock: str | None = None
) -> None:
    root.mkdir()
    expected = contract["arx"]
    (root / "arx_core.csv").write_text(
        "this fixture must never be parsed\n", encoding="ascii"
    )
    (root / "arx_detailed_diagnostics.csv").write_text(
        "detail\n", encoding="ascii"
    )
    (root / "arx_descriptive_summary.csv").write_text(
        "summary\n", encoding="ascii"
    )
    provenance = {
        "schema": arx_evaluate.OUTPUT_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "config": expected["config"],
        "prelock_registry_sha256": (
            prelock or expected["prelock_registry_sha256"]
        ),
        "addendum_external_freeze_receipt_sha256": expected[
            "external_freeze_receipt_file_sha256"
        ],
        "addendum_external_freeze_revision": expected[
            "external_freeze_revision"
        ],
        "transport_collection_binding_sha256": expected[
            "transport_binding_payload_sha256"
        ],
        "transport_manifest_file_sha256": expected[
            "transport_manifest_file_sha256"
        ],
        "model_file_sha256_by_case_seed": expected[
            "model_file_sha256_by_case_seed"
        ],
        "rows": 123,
    }
    _write_json(root / "evaluation_provenance.json", provenance)
    names = (
        "arx_core.csv",
        "arx_detailed_diagnostics.csv",
        "arx_descriptive_summary.csv",
        "evaluation_provenance.json",
    )
    completion = {
        "schema": arx_evaluate.COMPLETION_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "row_count": 123,
        "file_sha256_by_name": {
            name: sha256_file(root / name) for name in names
        },
    }
    _write_json(root / "evaluation_complete.json", completion)


def test_intended_completed_outputs_validate_without_csv_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = base_contract()
    v3_root = tmp_path / "v3"
    arx_root = tmp_path / "arx"
    write_v3_output(v3_root, contract)
    write_arx_output(arx_root, contract)

    def forbidden(*args, **kwargs):
        raise AssertionError("result CSV was opened during identity validation")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    guard.verify_v3_evaluation_metadata(v3_root, contract)
    guard.verify_arx_evaluation_metadata(arx_root, contract)


def test_wrong_but_internally_consistent_prior_v3_is_rejected(
    tmp_path: Path,
) -> None:
    contract = base_contract()
    root = tmp_path / "prior-v3"
    write_v3_output(
        root,
        contract,
        prelock="8" * 64,
        readiness="9" * 64,
    )
    with pytest.raises(ValueError, match="different study instance"):
        guard.verify_v3_evaluation_metadata(root, contract)


def test_wrong_but_internally_consistent_prior_arx_is_rejected(
    tmp_path: Path,
) -> None:
    contract = base_contract()
    root = tmp_path / "prior-arx"
    write_arx_output(root, contract, prelock="8" * 64)
    with pytest.raises(ValueError, match="different study instance"):
        guard.verify_arx_evaluation_metadata(root, contract)


def test_changed_core_hash_is_rejected_before_csv_read(tmp_path: Path) -> None:
    contract = base_contract()
    root = tmp_path / "arx"
    write_arx_output(root, contract)
    (root / "arx_core.csv").write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="hash changed"):
        guard.verify_arx_evaluation_metadata(root, contract)


def _dummy_paths(tmp_path: Path) -> guard.BindingPaths:
    return guard.BindingPaths(
        transport_prelock_root=tmp_path / "transport-prelock",
        transport_live_data_root=tmp_path / "transport-live",
        transport_readiness_path=tmp_path / "readiness.json",
        transport_external_freeze_receipt_path=tmp_path / "transport-receipt.json",
        transport_state_root=tmp_path / "state",
        transport_manifest_path=tmp_path / "manifest.json",
        arx_prelock_root=tmp_path / "arx-prelock",
        arx_external_freeze_receipt_path=tmp_path / "arx-receipt.json",
        arx_training_root=tmp_path / "training",
        comparison_prelock_root=tmp_path / "comparison-prelock",
        comparison_public_freeze_receipt_path=tmp_path
        / "comparison-receipt.json",
    )


def _patch_guard_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict,
) -> tuple[Path, Path]:
    prelock = tmp_path / "identity-prelock"
    prelock.mkdir()
    (prelock / "identity_guard_prelock.canonical.sha256").write_text(
        "a" * 64 + "\n", encoding="ascii"
    )
    receipt = tmp_path / "identity-receipt.json"
    receipt.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        identity_freeze,
        "load_verified_binding_contract",
        lambda root: contract,
    )
    monkeypatch.setattr(
        identity_public_freeze,
        "validate_public_freeze_receipt",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        guard,
        "build_binding_contract",
        lambda *args, **kwargs: contract,
    )
    return prelock, receipt


def test_guarded_run_is_atomic_sealed_and_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = base_contract()
    v3_root = tmp_path / "v3"
    arx_root = tmp_path / "arx"
    write_v3_output(v3_root, contract)
    write_arx_output(arx_root, contract)
    prelock, receipt = _patch_guard_preconditions(
        tmp_path, monkeypatch, contract
    )
    calls = []

    def fake_comparison(**kwargs):
        calls.append(kwargs)
        output = kwargs["output_root"]
        output.mkdir()
        return write_json_once(
            output / "analysis_complete.json",
            {"schema": "synthetic-comparison", "complete": True},
        )

    monkeypatch.setattr(
        guard.frozen_comparison, "run_bound_analysis", fake_comparison
    )
    output = tmp_path / "guarded"
    complete = guard.run_guarded_analysis(
        paths=_dummy_paths(tmp_path),
        identity_prelock_root=prelock,
        identity_public_freeze_receipt_path=receipt,
        v3_output_root=v3_root,
        arx_output_root=arx_root,
        output_root=output,
        live_external_freezes=False,
    )
    assert complete.is_file()
    assert len(calls) == 1
    provenance = json.loads(
        (output / "identity_guard_provenance.json").read_text(encoding="ascii")
    )
    assert (
        provenance["identity_wrapper_prelock_registry_sha256"] == "a" * 64
    )
    assert provenance[
        "identity_wrapper_public_freeze_receipt_file_sha256"
    ] == sha256_file(receipt)
    assert provenance[
        "neural_evaluation_completion_file_sha256"
    ] == sha256_file(v3_root / v3_run.COMPLETION_NAME)
    assert provenance[
        "arx_evaluation_completion_file_sha256"
    ] == sha256_file(arx_root / "evaluation_complete.json")
    assert (
        provenance["v5_readiness_sha256"]
        == guard.EXPECTED_TRANSPORT_READINESS_SHA256
    )
    completion = json.loads(complete.read_text(encoding="ascii"))
    assert completion["complete"] is True
    assert completion["artifact_inventory_sha256"] == canonical_sha256(
        completion["artifact_inventory_excludes_completion"]
    )
    assert output.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in output.rglob("*"))
    with pytest.raises(FileExistsError, match="overwrite"):
        guard.run_guarded_analysis(
            paths=_dummy_paths(tmp_path),
            identity_prelock_root=prelock,
            identity_public_freeze_receipt_path=receipt,
            v3_output_root=v3_root,
            arx_output_root=arx_root,
            output_root=output,
            live_external_freezes=False,
        )


def test_wrong_prior_output_never_reaches_frozen_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = base_contract()
    v3_root = tmp_path / "prior-v3"
    arx_root = tmp_path / "arx"
    write_v3_output(
        v3_root,
        contract,
        prelock="8" * 64,
        readiness="9" * 64,
    )
    write_arx_output(arx_root, contract)
    prelock, receipt = _patch_guard_preconditions(
        tmp_path, monkeypatch, contract
    )

    def forbidden(**kwargs):
        raise AssertionError("frozen comparison was called on a prior output")

    monkeypatch.setattr(
        guard.frozen_comparison, "run_bound_analysis", forbidden
    )
    with pytest.raises(ValueError, match="different study instance"):
        guard.run_guarded_analysis(
            paths=_dummy_paths(tmp_path),
            identity_prelock_root=prelock,
            identity_public_freeze_receipt_path=receipt,
            v3_output_root=v3_root,
            arx_output_root=arx_root,
            output_root=tmp_path / "must-not-exist",
            live_external_freezes=False,
        )
    assert not (tmp_path / "must-not-exist").exists()
