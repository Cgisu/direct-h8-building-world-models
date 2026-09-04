from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from . import collect, locked_state
from . import study_locked_collection as locked


DIGEST = "a" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="ascii")


def _freeze_receipt(digest: str = DIGEST) -> dict:
    return {
        "schema": locked.EXTERNAL_FREEZE_SCHEMA,
        "prelock_registry_sha256": digest,
        "timestamp_utc": "2020-01-01T00:00:00Z",
        "provider": "OpenTimestamps calendar",
        "public_https_url": "https://example.org/public/prelock-proof",
    }


@pytest.fixture
def formal_layout(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    output = tmp_path / "package" / "data_v4"
    prelock = tmp_path / "artifacts" / "prelock_v4"
    state_root = tmp_path / "package" / ".locked_study_state_v1"
    registry = prelock / "prelock_registry.json"
    bundle = prelock / "bundle"
    digest_record = prelock / "prelock_registry.canonical.sha256"
    manifest_root = output / collect.MANIFEST_SUBDIR
    manifest = manifest_root / "locked_test_all_corpus_manifest.json"
    raw = output / collect.LOCKED_RAW_SUBDIR
    bundle.mkdir(parents=True)
    _write_json(registry, {"test": "prelock"})
    digest_record.write_text(f"{DIGEST}\n", encoding="ascii")
    monkeypatch.setattr(locked, "CANONICAL_OUTPUT", output)
    monkeypatch.setattr(locked, "CANONICAL_LOCKED_RAW", raw)
    monkeypatch.setattr(locked, "CANONICAL_MANIFEST_ROOT", manifest_root)
    monkeypatch.setattr(locked, "CANONICAL_LOCKED_MANIFEST", manifest)
    monkeypatch.setattr(locked, "CANONICAL_PRELOCK_ROOT", prelock)
    monkeypatch.setattr(locked, "CANONICAL_PRELOCK_REGISTRY", registry)
    monkeypatch.setattr(locked, "CANONICAL_PRELOCK_BUNDLE", bundle)
    monkeypatch.setattr(locked, "CANONICAL_PRELOCK_DIGEST", digest_record)
    monkeypatch.setattr(locked, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collect, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(locked_state, "STATE_ROOT", state_root)
    monkeypatch.setattr(
        locked,
        "validate_prelock_bundle",
        lambda registry_path, artifact_root, config, digest: {"validated": digest},
    )
    return {
        "output": output,
        "prelock": prelock,
        "state_root": state_root,
        "registry": registry,
        "manifest": manifest,
        "raw": raw,
    }


def _publish_fake_locked_manifest(layout: dict[str, Path], digest: str = DIGEST) -> None:
    layout["raw"].mkdir(parents=True)
    manifest = {
        "collection_kind": "locked_test",
        "plan_mode": "full",
        "allowed_roles": ["locked_test"],
        "selected_cases": sorted(locked.CASES),
        "prelock_registry_sha256": digest,
        "counts": {
            "cases": len(locked.CASES),
            "trajectories": 36,
            "rows": 6912,
            "roles": {"locked_test": 36},
        },
    }
    _write_json(
        layout["manifest"],
        {"manifest_sha256": locked.canonical_sha256(manifest), "manifest": manifest},
    )


def test_external_freeze_receipt_contract_rejects_malformed_evidence(tmp_path: Path):
    path = tmp_path / "receipt.json"
    invalid = [
        {**_freeze_receipt(), "extra": True},
        {**_freeze_receipt(), "prelock_registry_sha256": "b" * 64},
        {**_freeze_receipt(), "timestamp_utc": "2026-07-22 04:00:00"},
        {**_freeze_receipt(), "timestamp_utc": "2026-02-30T04:00:00Z"},
        {**_freeze_receipt(), "provider": ""},
        {**_freeze_receipt(), "public_https_url": "file:///tmp/proof"},
        {**_freeze_receipt(), "public_https_url": "https://localhost/proof"},
        {**_freeze_receipt(), "public_https_url": "https://127.0.0.1/proof"},
    ]
    for payload in invalid:
        _write_json(path, payload)
        with pytest.raises(ValueError):
            locked.validate_external_freeze_receipt(path, DIGEST)


def test_formal_collection_marks_attempt_before_invocation_and_completes(
    monkeypatch, formal_layout: dict[str, Path], tmp_path: Path
):
    receipt = tmp_path / "external.json"
    _write_json(receipt, _freeze_receipt())
    observed: list[list[str]] = []

    def fake_run(command, *, check, cwd):
        state_dir = formal_layout["state_root"] / DIGEST
        assert (state_dir / locked.COLLECTION_ATTEMPT_MARKER).is_file()
        assert not formal_layout["manifest"].exists()
        assert check is True
        assert cwd == tmp_path
        observed.append(command)
        _publish_fake_locked_manifest(formal_layout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(locked.subprocess, "run", fake_run)
    result = locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    assert result == formal_layout["manifest"]
    assert len(observed) == 1
    command = observed[0]
    assert "--output-dir" in command
    assert command[command.index("--output-dir") + 1] == str(formal_layout["output"])
    assert command.count("--case") == len(locked.CASES)
    assert "--dry-run" not in command

    state_dir = formal_layout["state_root"] / DIGEST
    attempt = locked.load_strict_json(state_dir / locked.COLLECTION_ATTEMPT_MARKER)
    completion = locked.load_strict_json(state_dir / locked.COLLECTION_COMPLETION_MARKER)
    assert set(attempt) == locked.ATTEMPT_FIELDS
    assert set(completion) == locked.COMPLETION_FIELDS
    assert completion["locked_values_accessed_after_attempt"] is True
    assert completion["prelock_registry_sha256"] == DIGEST
    assert completion["external_freeze_receipt_sha256"] == attempt[
        "external_freeze_receipt_sha256"
    ]
    assert locked.validate_locked_collection_completion(DIGEST) == completion

    with pytest.raises(FileExistsError, match="already has"):
        locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    assert len(observed) == 1


def test_failure_is_immutable_and_prevents_retry(
    monkeypatch, formal_layout: dict[str, Path], tmp_path: Path
):
    receipt = tmp_path / "external.json"
    _write_json(receipt, _freeze_receipt())
    calls = 0

    def fail_after_attempt(command, *, check, cwd):
        nonlocal calls
        calls += 1
        assert (
            formal_layout["state_root"]
            / DIGEST
            / locked.COLLECTION_ATTEMPT_MARKER
        ).is_file()
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(locked.subprocess, "run", fail_after_attempt)
    with pytest.raises(subprocess.CalledProcessError):
        locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    state_dir = formal_layout["state_root"] / DIGEST
    failure = locked.load_strict_json(state_dir / locked.COLLECTION_FAILURE_MARKER)
    assert failure["collector_return_code"] == 17
    assert not (state_dir / locked.COLLECTION_COMPLETION_MARKER).exists()
    with pytest.raises(FileExistsError, match="already has"):
        locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    assert calls == 1


def test_invalid_freeze_fails_before_state_or_collector(
    monkeypatch, formal_layout: dict[str, Path], tmp_path: Path
):
    receipt = tmp_path / "external.json"
    _write_json(receipt, {**_freeze_receipt(), "public_https_url": "http://example.org"})
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("collector must not be invoked")

    monkeypatch.setattr(locked.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="public HTTPS"):
        locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    assert not formal_layout["state_root"].exists()
    assert called is False


def test_future_freeze_timestamp_fails_before_state(
    monkeypatch, formal_layout: dict[str, Path], tmp_path: Path
):
    receipt = tmp_path / "external.json"
    _write_json(receipt, {**_freeze_receipt(), "timestamp_utc": "2021-01-01T00:00:00Z"})
    monkeypatch.setattr(locked, "_utc_now", lambda: "2020-01-01T00:00:00Z")

    def forbidden(*args, **kwargs):
        raise AssertionError("collector must not be invoked")

    monkeypatch.setattr(locked.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="later than"):
        locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    assert not formal_layout["state_root"].exists()


def test_completion_validator_detects_manifest_change(
    monkeypatch, formal_layout: dict[str, Path], tmp_path: Path
):
    receipt = tmp_path / "external.json"
    _write_json(receipt, _freeze_receipt())

    def fake_run(command, *, check, cwd):
        _publish_fake_locked_manifest(formal_layout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(locked.subprocess, "run", fake_run)
    locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    formal_layout["manifest"].write_text("{}\n", encoding="ascii")
    with pytest.raises(ValueError, match="wrapper fields"):
        locked.validate_locked_collection_completion(DIGEST)


def test_completion_validator_enforces_attempt_paths_and_chronology(
    monkeypatch, formal_layout: dict[str, Path], tmp_path: Path
):
    receipt = tmp_path / "external.json"
    _write_json(receipt, _freeze_receipt())

    def fake_run(command, *, check, cwd):
        _publish_fake_locked_manifest(formal_layout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(locked.subprocess, "run", fake_run)
    locked.run_locked_collection(DIGEST, receipt, locked.CONFIRMATION_TOKEN)
    state_dir = formal_layout["state_root"] / DIGEST
    attempt_path = state_dir / locked.COLLECTION_ATTEMPT_MARKER
    attempt = locked.load_strict_json(attempt_path)
    attempt["output_root"] = "/tmp/alternate"
    attempt_path.chmod(0o644)
    _write_json(attempt_path, attempt)
    with pytest.raises(ValueError, match="attempt output_root"):
        locked.validate_locked_collection_completion(DIGEST)

    attempt["output_root"] = str(formal_layout["output"])
    _write_json(attempt_path, attempt)
    completion_path = state_dir / locked.COLLECTION_COMPLETION_MARKER
    completion = locked.load_strict_json(completion_path)
    completion["completed_at_utc"] = "2019-12-31T23:59:59Z"
    completion_path.chmod(0o644)
    _write_json(completion_path, completion)
    with pytest.raises(ValueError, match="chronology"):
        locked.validate_locked_collection_completion(DIGEST)


def test_manifest_evidence_requires_exact_locked_corpus_size(
    formal_layout: dict[str, Path]
):
    _publish_fake_locked_manifest(formal_layout)
    wrapper = locked.load_strict_json(formal_layout["manifest"])
    wrapper["manifest"]["counts"]["trajectories"] = 35
    wrapper["manifest"]["counts"]["roles"] = {"locked_test": 35}
    wrapper["manifest_sha256"] = locked.canonical_sha256(wrapper["manifest"])
    _write_json(formal_layout["manifest"], wrapper)
    with pytest.raises(ValueError, match="counts"):
        locked._manifest_evidence(DIGEST)


def test_cli_has_no_output_or_prelock_path_overrides():
    parser = locked.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--expected-prelock-sha256",
                DIGEST,
                "--external-freeze-receipt",
                "receipt.json",
                "--confirm-locked-collection",
                locked.CONFIRMATION_TOKEN,
                "--output-dir",
                "/tmp/alternate",
            ]
        )
