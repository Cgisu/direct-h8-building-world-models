from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import external_freeze, runner


def _remote_fixture(
    files: dict[str, bytes],
    *,
    gist_id: str = "abc123",
    revision: str = "a" * 40,
    owner: str = "tester",
    created: str = "2026-07-23T01:00:00Z",
    committed: str = "2026-07-23T01:01:00Z",
    updated: str = "2026-07-23T01:02:00Z",
    provider: str = "2026-07-23T01:03:00Z",
) -> external_freeze.GitHubResponse:
    api_url, _ = external_freeze._urls(gist_id, revision, owner)
    return external_freeze.GitHubResponse(
        payload={
            "id": gist_id,
            "public": True,
            "owner": {"login": owner},
            "html_url": f"https://gist.github.com/{owner}/{gist_id}",
            "created_at": created,
            "updated_at": updated,
            "history": [
                {
                    "version": revision,
                    "committed_at": committed,
                    "url": api_url,
                    "user": {"login": owner},
                }
            ],
            "files": {
                name: {
                    "filename": name,
                    "truncated": False,
                    "content": content.decode("utf-8"),
                    "size": len(content),
                }
                for name, content in sorted(files.items())
            },
        },
        provider_date_utc=provider,
    )


def test_freeze_contract_includes_closeout_and_all_operational_sources() -> None:
    assert {
        "PROTOCOL_ADDENDUM.md",
        "runner.py",
        "external_freeze.py",
        "evaluation_adapter.py",
        "v4_closeout.py",
        "v4_terminal_failure_closeout.json",
        "v3_paired_collection_attempt.json",
        "v3_paired_collection_failure.json",
    } <= external_freeze.FREEZE_FILENAMES


def test_build_and_live_verify_custom_freeze(
    response_blind_readiness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = response_blind_readiness["readiness_path"]
    files = external_freeze.read_freeze_files(
        runner.PRELOCK_ROOT, readiness_path
    )
    remote = _remote_fixture(files)
    monkeypatch.setattr(external_freeze, "_fetch_json", lambda _url: remote)
    receipt = external_freeze.build_external_freeze_receipt(
        "abc123",
        "a" * 40,
        "tester",
        prelock_root=runner.PRELOCK_ROOT,
        readiness_path=readiness_path,
    )
    terminal = runner.terminal_v4_failure_binding()["binding"]
    code = runner.operational_code_hashes()
    assert receipt["protocol_addendum_sha256"] == code["PROTOCOL_ADDENDUM.md"]
    assert receipt["runner_sha256"] == code["runner.py"]
    assert receipt["evaluation_adapter_sha256"] == code["evaluation_adapter.py"]
    assert receipt["external_freeze_sha256"] == code["external_freeze.py"]
    assert (
        receipt["terminal_v4_closeout_file_sha256"]
        == terminal["terminal_closeout_file_sha256"]
    )
    assert (
        receipt["terminal_v4_collection_log_sha256"]
        == terminal["collection_log_sha256"]
    )
    assert (
        receipt["terminal_v4_raw_inventory_canonical_sha256"]
        == terminal["incomplete_raw_inventory_canonical_sha256"]
    )

    receipt_path = tmp_path / "external_freeze_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    verified = external_freeze.validate_external_freeze_receipt(
        receipt_path,
        runner.ORIGINAL_PRELOCK_SHA256,
        response_blind_readiness["readiness"].report["readiness_sha256"],
        prelock_root=runner.PRELOCK_ROOT,
        readiness_path=readiness_path,
        live=True,
    )
    assert verified["provider_verified_at_utc"] == remote.provider_date_utc


def test_remote_validation_rejects_bad_chronology_and_hash(
    response_blind_readiness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = response_blind_readiness["readiness_path"]
    files = external_freeze.read_freeze_files(
        runner.PRELOCK_ROOT, readiness_path
    )
    good = _remote_fixture(files)
    monkeypatch.setattr(external_freeze, "_fetch_json", lambda _url: good)
    receipt = external_freeze.build_external_freeze_receipt(
        "abc123",
        "a" * 40,
        "tester",
        prelock_root=runner.PRELOCK_ROOT,
        readiness_path=readiness_path,
    )

    bad_chronology = _remote_fixture(
        files,
        committed="2026-07-23T01:04:00Z",
    )
    with pytest.raises(ValueError, match="chronology"):
        external_freeze._validate_remote(bad_chronology, receipt, files)

    bad_files = dict(files)
    bad_files["runner.py"] = bad_files["runner.py"] + b"\n"
    bad_content = _remote_fixture(bad_files)
    with pytest.raises(ValueError, match="byte-for-byte|hash"):
        external_freeze._validate_remote(bad_content, receipt, files)


def test_readiness_tamper_fails_before_network(
    response_blind_readiness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = json.loads(
        response_blind_readiness["readiness_path"].read_text(encoding="ascii")
    )
    report["operational_code_sha256"]["runner.py"] = "0" * 64
    tampered = tmp_path / "collection_readiness.json"
    tampered.write_text(json.dumps(report) + "\n", encoding="ascii")
    called = False

    def should_not_fetch(_url):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(external_freeze, "_fetch_json", should_not_fetch)
    with pytest.raises(ValueError, match="outcome-blind lock"):
        external_freeze.build_external_freeze_receipt(
            "abc123",
            "a" * 40,
            "tester",
            prelock_root=runner.PRELOCK_ROOT,
            readiness_path=tampered,
        )
    assert called is False

