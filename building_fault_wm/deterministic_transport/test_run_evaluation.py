from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.fault_data import (
    CorpusIndex,
    FaultCell,
    FaultManifest,
    FaultSpec,
    TrajectoryKey,
    build_fault_manifest,
)

from . import collect, corpus, evaluate, gate, plan
from .run_evaluation import (
    COMPLETION_NAME,
    COMPLETION_SCHEMA,
    CORE_NAME,
    DETAIL_NAME,
    DIAGNOSTIC_SUMMARY_NAME,
    FAULT_MANIFEST_NAME,
    GATE_RESULT_NAME,
    PROVENANCE_NAME,
    PROVENANCE_SCHEMA,
    FrozenAssets,
    _artifact_inventory,
    _input_hash_payload,
    _load_fault_spec,
    _write_csv,
    _write_json,
    verify_evaluation_output,
)
from .test_gate import _frame


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def test_fault_spec_is_loaded_from_byte_bound_parent_manifest(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent_selected"
    manifest_path = parent / "evidence/locked_fault_manifest.json"
    contract_path = (
        parent
        / "experiment/prelock_bundle/frozen/frozen_fault_contract.json"
    )
    spec = FaultSpec(zone_bias_severities=(1.25, 2.5))
    key = TrajectoryKey(
        "bestest_hydronic_heat_pump",
        "locked_test",
        7,
        101,
    )
    manifest = FaultManifest(
        schema="boptest-multicase-fault-manifest-v2",
        corpus_manifest_sha256="a" * 64,
        source_sha256=((key.text, "b" * 64),),
        spec=spec,
        cells=(
            FaultCell(
                cell_id=f"{key.text}:fixture",
                trajectory=key,
                source_sha256="b" * 64,
                fault_channel="zone_temperature_k",
                family="healthy",
                sign=0,
                severity=0.0,
                severity_unit="none",
                onset=64,
                stop=112,
                anchors=(72, 80, 88, 96),
            ),
        ),
    )
    manifest.write(manifest_path)
    _json(
        contract_path,
        {
            "schema": "boptest-multicase-frozen-fault-contract-v1",
            "development_corpus_manifest_sha256": "c" * 64,
            "spec": asdict(spec),
            "signatures_by_role": {},
            "signature_sha256_by_role": {},
        },
    )
    reused = {
        "evidence/locked_fault_manifest.json": plan.sha256_file(manifest_path),
        (
            "experiment/prelock_bundle/frozen/"
            "frozen_fault_contract.json"
        ): plan.sha256_file(contract_path),
    }
    observed, manifest_sha, contract_sha = _load_fault_spec(parent, reused)
    assert observed == spec
    assert observed != FaultSpec()
    assert manifest_sha == reused["evidence/locked_fault_manifest.json"]
    assert contract_sha == reused[
        "experiment/prelock_bundle/frozen/frozen_fault_contract.json"
    ]


def _state_fixture(
    root: Path,
    manifest_path: Path,
    receipt_path: Path,
    *,
    prelock_sha: str,
    readiness_sha: str,
    freeze_time: str,
    attempt_time: str,
) -> tuple[collect.Readiness, dict]:
    manifest = {
        "manifest_sha256": "d" * 64,
        "manifest": {},
    }
    _json(manifest_path, manifest)
    collection_code = {
        "plan.py": "1" * 64,
        "worker_collect.py": "2" * 64,
        "collect.py": "3" * 64,
    }
    readiness = collect.Readiness(
        plans={},
        plan_paths=(),
        certificate={},
        expected_certificate_sha256="4" * 64,
        collection_code_sha256=collection_code,
        report={
            "readiness_sha256": readiness_sha,
            "plan_sha256_by_case": {},
            "protocol_sha256": "8" * 64,
        },
    )
    freeze = {
        "gist_id": "abc123",
        "revision": "5" * 40,
        "revision_committed_at_utc": freeze_time,
    }
    _json(receipt_path, freeze)
    state = root / readiness_sha
    state.mkdir(parents=True)
    attempt = {
        "schema": collect.ATTEMPT_SCHEMA,
        "stage": "locked_transport_collection",
        "started_at_utc": attempt_time,
        "certificate_sha256": "4" * 64,
        "readiness_sha256": readiness_sha,
        "prelock_registry_sha256": prelock_sha,
        "collection_code_sha256": collection_code,
        "plan_sha256_by_case": {},
        "protocol_sha256": "8" * 64,
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "worker_boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "locked_response_values_accessed": False,
        "external_freeze_receipt_sha256": plan.sha256_file(receipt_path),
        "external_freeze_gist_id": freeze["gist_id"],
        "external_freeze_revision": freeze["revision"],
        "external_freeze_revision_committed_at_utc": freeze_time,
    }
    attempt_path = state / collect.ATTEMPT_MARKER
    _json(attempt_path, attempt)
    completion = {
        "schema": collect.COMPLETION_SCHEMA,
        "stage": "locked_transport_collection_complete",
        "completed_at_utc": "2026-07-23T10:03:00Z",
        "certificate_sha256": "4" * 64,
        "readiness_sha256": readiness_sha,
        "prelock_registry_sha256": prelock_sha,
        "collection_code_sha256": collection_code,
        "attempt_marker_sha256": plan.sha256_file(attempt_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": plan.sha256_file(manifest_path),
        "manifest_payload_sha256": "d" * 64,
        "locked_response_values_accessed_after_attempt": True,
    }
    _json(state / collect.COMPLETION_MARKER, completion)
    return readiness, freeze


def test_collection_state_requires_freeze_before_attempt(tmp_path: Path) -> None:
    prelock_sha = "6" * 64
    readiness_sha = "7" * 64
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "freeze.json"
    state_root = tmp_path / "state"
    readiness, freeze = _state_fixture(
        state_root,
        manifest_path,
        receipt_path,
        prelock_sha=prelock_sha,
        readiness_sha=readiness_sha,
        freeze_time="2026-07-23T10:00:00Z",
        attempt_time="2026-07-23T10:01:00Z",
    )
    attempt, completion, _, _ = corpus.validate_collection_completion(
        state_root=state_root,
        readiness=readiness,
        manifest_path=manifest_path,
        expected_prelock_sha256=prelock_sha,
        external_freeze=freeze,
        external_freeze_receipt_path=receipt_path,
    )
    assert attempt["readiness_sha256"] == readiness_sha
    assert completion["locked_response_values_accessed_after_attempt"] is True

    late_root = tmp_path / "late"
    late_manifest = late_root / "manifest.json"
    late_receipt = late_root / "freeze.json"
    late_readiness, late_freeze = _state_fixture(
        late_root / "state",
        late_manifest,
        late_receipt,
        prelock_sha=prelock_sha,
        readiness_sha=readiness_sha,
        freeze_time="2026-07-23T10:02:00Z",
        attempt_time="2026-07-23T10:01:00Z",
    )
    with pytest.raises(ValueError, match="after collection began"):
        corpus.validate_collection_completion(
            state_root=late_root / "state",
            readiness=late_readiness,
            manifest_path=late_manifest,
            expected_prelock_sha256=prelock_sha,
            external_freeze=late_freeze,
            external_freeze_receipt_path=late_receipt,
        )


def test_verify_reconstructs_gate_and_checks_output_hashes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    core = _frame(ungated=0.8, legacy=1.0, deterministic=1.0)
    detail = core.copy()
    defaults = {
        "update": 400,
        "fault_channel_index": 0,
        "severity_unit": "K",
        "history_start": 33,
        "history_stop": 72,
        "target_index": 80,
        "target_raw": 290.0,
        "prediction_raw": 291.0,
        "prediction_standardized": 1.0,
        "raw_abs_error": 1.0,
        "raw_unit": "K",
        "boundary_crossing": False,
        "action_transition_count": 0,
        "alternate_action_prediction_raw": 291.1,
        "action_prediction_change_standardized": 0.1,
    }
    for name, value in defaults.items():
        detail[name] = value
    detail = detail.loc[:, evaluate.DETAIL_COLUMNS]
    _write_csv(output / CORE_NAME, core)
    _write_csv(output / DETAIL_NAME, detail)
    _write_csv(
        output / DIAGNOSTIC_SUMMARY_NAME,
        pd.DataFrame({"synthetic": [1]}),
    )

    index = CorpusIndex(
        root=tmp_path,
        manifest_path=tmp_path / "synthetic_manifest.json",
        manifest_sha256="8" * 64,
        collection_kind="locked_test",
        prelock_registry_sha256="9" * 64,
        allowed_roles=("locked_test",),
        records=(),
        plan_sha256_by_case=(),
    )
    fault_manifest = build_fault_manifest(index, FaultSpec())
    _write_json(output / FAULT_MANIFEST_NAME, fault_manifest.payload())
    result = gate.analyze_gate(core)
    _write_json(output / GATE_RESULT_NAME, result)

    receipt = tmp_path / "freeze_receipt.json"
    receipt.write_text("{}\n", encoding="ascii")
    validated = SimpleNamespace(
        index=index,
        manifest_file_sha256="a" * 64,
        attempt_file_sha256="b" * 64,
        completion_file_sha256="c" * 64,
    )
    assets = FrozenAssets(
        fault_spec=FaultSpec(),
        fault_source_file_sha256="d" * 64,
        fault_contract_file_sha256="e" * 64,
        scalers={},
        scaler_file_sha256_by_case={},
        rssm_checkpoint={},
        deterministic_checkpoint={},
        deterministic_receipt_sha256={},
        deterministic_training_wall_seconds={},
        training_grid_file_sha256="f" * 64,
    )
    input_hashes = _input_hash_payload(
        prelock_sha256="9" * 64,
        readiness_sha256="7" * 64,
        freeze_receipt_path=receipt,
        validated=validated,
        assets=assets,
    )
    _write_json(
        output / PROVENANCE_NAME,
        {
            "schema": PROVENANCE_SCHEMA,
            "input_hashes": input_hashes,
        },
    )
    inventory = _artifact_inventory(output)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "study_kind": "direct_h8_deterministic_transport_v3",
        "prelock_registry_sha256": "9" * 64,
        "readiness_sha256": "7" * 64,
        "corpus_manifest_payload_sha256": index.manifest_sha256,
        "fault_manifest_sha256": fault_manifest.sha256,
        "gate_result_sha256": plan.canonical_sha256(result),
        "provenance_file_sha256": plan.sha256_file(
            output / PROVENANCE_NAME
        ),
        "artifact_inventory_excludes_completion": inventory,
        "artifact_inventory_sha256": plan.canonical_sha256(inventory),
        "complete": True,
    }
    _write_json(output / COMPLETION_NAME, completion)
    verified = verify_evaluation_output(
        output,
        expected_prelock_sha256="9" * 64,
        expected_readiness_sha256="7" * 64,
        validated=validated,
        assets=assets,
        external_freeze_receipt_path=receipt,
    )
    assert verified["verified"] is True
    assert verified["primary_architecture_category"] == "RSSM_ADVANTAGE"

    with (output / CORE_NAME).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="inventory"):
        verify_evaluation_output(
            output,
            expected_prelock_sha256="9" * 64,
            expected_readiness_sha256="7" * 64,
            validated=validated,
            assets=assets,
            external_freeze_receipt_path=receipt,
        )
