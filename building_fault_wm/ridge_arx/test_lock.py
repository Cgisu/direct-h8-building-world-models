from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from building_fault_wm.deterministic_transport import (
    corpus as v3_corpus,
    plan as v3_plan,
)
from building_fault_wm.neural_benchmark import protocol as boptest

from . import external_freeze, lock
from .io import sha256_file


def _v3_manifest() -> dict:
    return {
        "schema": "direct-h8-transport-corpus-manifest-v1",
        "study_kind": "paired_schedule_transport",
        "collection_kind": "paired_locked_transport",
        "output_role": "locked_test",
        "row_schema": "row",
        "fields": [],
        "plan_sha256_by_case": {case: "a" * 64 for case in v3_corpus.EXPECTED_CASES},
        "certificate_sha256": "b" * 64,
        "certificate_file_sha256": "c" * 64,
        "readiness_sha256": "d" * 64,
        "collection_code_sha256": {"collect.py": "e" * 64},
        "source_sha256_by_case": {
            case: {"wrapped_fmu": "f" * 64, "weather_csv": "1" * 64}
            for case in v3_corpus.EXPECTED_CASES
        },
        "worker_image_id": "image",
        "worker_boptest_version": "version",
        "boptest_commit": "2" * 40,
        "counts": {
            "cases": len(v3_corpus.EXPECTED_CASES),
            "windows": v3_corpus.EXPECTED_WINDOWS,
            "branches": v3_corpus.EXPECTED_BRANCHES,
            "rows_per_branch": boptest.TRAJECTORY_STEPS,
        },
        "files": [
            {"path": f"branch-{index}.csv", "sha256": "3" * 64}
            for index in range(v3_corpus.EXPECTED_BRANCHES)
        ],
        "worker_receipts": [],
    }


def test_manifest_metadata_binding_never_needs_raw_files(tmp_path: Path) -> None:
    manifest = _v3_manifest()
    payload_sha = v3_plan.canonical_sha256(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"manifest_sha256": payload_sha, "manifest": manifest}) + "\n",
        encoding="ascii",
    )
    completion = {
        "manifest_payload_sha256": payload_sha,
        "manifest_file_sha256": sha256_file(path),
    }
    metadata = lock._manifest_metadata(path, completion)
    assert metadata["manifest_payload_sha256"] == payload_sha
    assert metadata["counts"]["branches"] == v3_corpus.EXPECTED_BRANCHES

    completion["manifest_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not bind"):
        lock._manifest_metadata(path, completion)


def test_v5_metadata_adapter_rejects_terminal_data_v7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_path = tmp_path / "collection_readiness.json"
    freeze_path = tmp_path / "external_freeze_receipt.json"
    manifest_path = tmp_path / "manifest.json"
    for path in (freeze_path, manifest_path):
        path.write_text("{}\n", encoding="ascii")
    readiness = {
        "schema": lock.transport_runner.READINESS_SCHEMA,
        "replacement_kind": "ownership_corrected_full_recollection",
        "prelock_registry_sha256": (
            lock.transport_runner.ORIGINAL_PRELOCK_SHA256
        ),
        "readiness_sha256": "a" * 64,
        "namespaces": {
            "data": "data",
            "state": "state_v3",
            "freeze": "freeze_v5",
        },
        "full_recollection_required": True,
        "data_v7_raw_reuse_permitted": False,
        "operational_code_sha256": {"runner.py": "b" * 64},
        "frozen_evaluation_source_sha256": {"corpus.py": "c" * 64},
    }
    readiness_path.write_text(json.dumps(readiness), encoding="ascii")
    monkeypatch.setattr(
        lock.transport_runner,
        "validate_namespace_separation",
        lambda **_: None,
    )
    ready = SimpleNamespace(
        expected_certificate_sha256="d" * 64,
        collection_code_sha256={"collect.py": "e" * 64},
    )
    monkeypatch.setattr(
        lock.transport_runner,
        "load_bound_readiness",
        lambda **_: ({}, ready),
    )
    freeze = {
        "provider": "github-gist",
        "gist_id": "f" * 32,
        "revision": "1" * 40,
        "revision_committed_at_utc": "2026-08-01T00:00:00Z",
    }
    monkeypatch.setattr(
        lock.transport_external_freeze,
        "validate_external_freeze_receipt",
        lambda *_, **__: freeze,
    )
    attempt = {
        "locked_response_values_accessed": False,
        "started_at_utc": "2026-08-01T00:01:00Z",
    }
    completion = {
        "locked_response_values_accessed_after_attempt": True,
        "completed_at_utc": "2026-08-01T00:02:00Z",
    }
    monkeypatch.setattr(
        lock.v3_corpus,
        "validate_collection_completion",
        lambda **_: (attempt, completion, "2" * 64, "3" * 64),
    )
    monkeypatch.setattr(
        lock,
        "_manifest_metadata",
        lambda *_, **__: {"counts": {"branches": 72}},
    )
    binding = lock.bind_completed_transport_metadata(
        transport_prelock_root=tmp_path / "prelock",
        transport_live_data_root=tmp_path / "data",
        transport_readiness_path=readiness_path,
        transport_external_freeze_receipt_path=freeze_path,
        transport_state_root=tmp_path / "state_v3",
        transport_manifest_path=manifest_path,
    )
    assert binding["metadata_adapter"]["data_namespace"] == "data"
    assert binding["metadata_adapter"]["terminal_data_v7_rejected"] is True

    readiness["namespaces"]["data"] = "data_v7"
    readiness_path.write_text(json.dumps(readiness), encoding="ascii")
    with pytest.raises(ValueError, match="canonical v5"):
        lock.bind_completed_transport_metadata(
            transport_prelock_root=tmp_path / "prelock",
            transport_live_data_root=tmp_path / "data",
            transport_readiness_path=readiness_path,
            transport_external_freeze_receipt_path=freeze_path,
            transport_state_root=tmp_path / "state_v3",
            transport_manifest_path=manifest_path,
        )


def test_prelock_bundle_is_write_once_and_inventory_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = {
        "schema": lock.TRAINING_BINDING_SCHEMA,
        "training_root": str((tmp_path / "training").resolve()),
        "verification": {"runs": 15},
        "training_source_manifest": {},
    }
    parent = {
        "schema": lock.PARENT_BINDING_SCHEMA,
        "parent_package_digest": "a" * 64,
    }
    v3 = {
        "schema": lock.TRANSPORT_BINDING_SCHEMA,
        "validation_scope": "metadata_only_no_trajectory_csv_or_response_value_opened",
        "paths": {},
        "file_sha256_by_role": {},
    }
    monkeypatch.setattr(lock, "_training_binding", lambda _: training)
    monkeypatch.setattr(lock, "parent_asset_binding", lambda: parent)
    monkeypatch.setattr(
        lock, "bind_completed_transport_metadata", lambda **_: v3
    )
    original_verify = lock.verify_prelock
    monkeypatch.setattr(lock, "verify_prelock", lambda *_, **__: {})
    output = tmp_path / "prelock"
    lock.prepare_prelock(
        output_root=output,
        training_root=tmp_path / "training",
        transport_prelock_root=tmp_path / "transport-prelock",
        transport_live_data_root=tmp_path / "data",
        transport_readiness_path=tmp_path / "readiness.json",
        transport_external_freeze_receipt_path=tmp_path / "freeze.json",
        transport_state_root=tmp_path / "state_v3",
        transport_manifest_path=tmp_path / "manifest.json",
    )
    registry = original_verify(output, verify_live_assets=False)
    assert registry["locked_response_values_accessed_while_preparing"] is False
    assert len(lock.external_freeze_file_paths.__name__) > 0
    with pytest.raises(FileExistsError):
        lock.prepare_prelock(
            output_root=output,
            training_root=tmp_path / "training",
            transport_prelock_root=tmp_path,
            transport_live_data_root=tmp_path / "data",
            transport_readiness_path=tmp_path / "r",
            transport_external_freeze_receipt_path=tmp_path / "f",
            transport_state_root=tmp_path / "state_v3",
            transport_manifest_path=tmp_path / "m",
        )


def test_external_freeze_receipt_is_exact_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = {"PROTOCOL.md": "a" * 64}
    monkeypatch.setattr(external_freeze, "expected_file_hashes", lambda _: hashes)
    monkeypatch.setattr(external_freeze, "_prelock_digest", lambda _: "b" * 64)
    gist = "c" * 32
    revision = "d" * 40
    owner = "owner"
    receipt = {
        "schema": external_freeze.SCHEMA,
        "provider": external_freeze.PROVIDER,
        "public": True,
        "prelock_registry_sha256": "b" * 64,
        "gist_id": gist,
        "revision": revision,
        "owner_login": owner,
        "provider_created_at_utc": "2026-08-01T00:00:00Z",
        "provider_updated_at_utc": "2026-08-01T00:01:00Z",
        "revision_committed_at_utc": "2026-08-01T00:00:30Z",
        "revision_api_url": f"https://api.github.com/gists/{gist}/{revision}",
        "revision_html_url": f"https://gist.github.com/{owner}/{gist}/{revision}",
        "file_sha256_by_name": hashes,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="ascii")
    assert external_freeze.validate_external_freeze_receipt(
        path, tmp_path, live=False
    ) == receipt
    receipt["public"] = False
    path.write_text(json.dumps(receipt), encoding="ascii")
    with pytest.raises(ValueError, match="identity"):
        external_freeze.validate_external_freeze_receipt(
            path, tmp_path, live=False
        )


def test_create_public_freeze_pins_writes_once_and_live_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen\n", encoding="ascii")
    gist = "c" * 32
    revision = "d" * 40
    owner = "owner"
    receipt = {
        "schema": external_freeze.SCHEMA,
        "provider": external_freeze.PROVIDER,
        "public": True,
        "prelock_registry_sha256": "a" * 64,
        "gist_id": gist,
        "revision": revision,
        "owner_login": owner,
        "provider_created_at_utc": "2026-08-01T00:00:00Z",
        "provider_updated_at_utc": "2026-08-01T00:01:00Z",
        "revision_committed_at_utc": "2026-08-01T00:00:30Z",
        "revision_api_url": f"https://api.github.com/gists/{gist}/{revision}",
        "revision_html_url": f"https://gist.github.com/{owner}/{gist}/{revision}",
        "file_sha256_by_name": {"source.txt": sha256_file(source)},
    }
    monkeypatch.setattr(
        external_freeze,
        "external_freeze_file_paths",
        lambda _: {"source.txt": source},
    )
    monkeypatch.setattr(
        external_freeze.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"https://gist.github.com/{owner}/{gist}\n"
        ),
    )
    monkeypatch.setattr(
        external_freeze,
        "_fetch",
        lambda _: external_freeze.ProviderResponse(
            payload={
                "public": True,
                "owner": {"login": owner},
                "history": [{"version": revision}],
            },
            provider_verified_at_utc="2026-08-01T00:02:00Z",
        ),
    )
    monkeypatch.setattr(
        external_freeze,
        "build_external_freeze_receipt",
        lambda *args, **kwargs: receipt,
    )
    monkeypatch.setattr(
        external_freeze,
        "validate_external_freeze_receipt",
        lambda *args, **kwargs: {
            **receipt,
            "provider_verified_at_utc": "2026-08-01T00:02:00Z",
        },
    )
    path = tmp_path / "receipt.json"
    created = external_freeze.create_public_freeze(
        prelock_root=tmp_path, receipt_path=path
    )
    assert created["revision"] == revision
    assert json.loads(path.read_text(encoding="ascii")) == receipt
    with pytest.raises(FileExistsError):
        external_freeze.create_public_freeze(
            prelock_root=tmp_path, receipt_path=path
        )
