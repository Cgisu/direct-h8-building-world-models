from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from building_fault_wm.deterministic_transport import (
    collect as frozen_collect,
)
from building_fault_wm.deterministic_transport import (
    plan as frozen_plan,
)
from building_fault_wm.deterministic_transport import (
    corpus as frozen_corpus,
)

from . import runner, v4_closeout


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): frozen_plan.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _fake_freeze() -> dict:
    return {
        "gist_id": "abc123",
        "revision": "a" * 40,
        "revision_committed_at_utc": "2026-07-23T01:00:00Z",
        "provider_verified_at_utc": "2026-07-23T01:01:00Z",
    }


def test_package_imports_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from building_fault_wm.transport_collection "
                "import runner, external_freeze, evaluation_adapter, v4_closeout; "
                "print(runner.READINESS_SCHEMA)"
            ),
        ],
        cwd=runner.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == runner.READINESS_SCHEMA


def test_namespaces_are_separate_and_legacy_data_is_rejected(
    tmp_path: Path,
) -> None:
    runner.validate_namespace_separation(
        data_root=tmp_path / "data",
        state_root=tmp_path / "state_v3",
        freeze_root=tmp_path / "freeze_v5",
    )
    with pytest.raises(ValueError, match="data_v7"):
        runner.validate_namespace_separation(
            data_root=runner.LEGACY_DATA_ROOT,
            state_root=tmp_path / "state_v3",
            freeze_root=tmp_path / "freeze_v5",
        )
    with pytest.raises(ValueError, match="data_v7"):
        runner.validate_namespace_separation(
            data_root=tmp_path / "data",
            state_root=runner.LEGACY_DATA_ROOT / "state_v3",
            freeze_root=tmp_path / "freeze_v5",
        )
    with pytest.raises(ValueError, match="data_v7"):
        runner.validate_namespace_separation(
            data_root=tmp_path / "data",
            state_root=tmp_path / "state_v3",
            freeze_root=runner.LEGACY_DATA_ROOT.parent,
        )
    with pytest.raises(ValueError, match="overlap"):
        runner.validate_namespace_separation(
            data_root=tmp_path / "data",
            state_root=tmp_path / "data/state_v3",
            freeze_root=tmp_path / "freeze_v5",
        )


def test_worker_command_is_frozen_command_plus_one_success_chown(
    response_blind_readiness,
    tmp_path: Path,
) -> None:
    readiness = response_blind_readiness["readiness"]
    plan_path = readiness.plan_paths[0]
    kwargs = {
        "plan_root": tmp_path / "plans/full",
        "certificate_path": tmp_path / "certificate.json",
        "testcase_root": tmp_path / "boptest/testcases",
    }
    frozen = frozen_collect._docker_command(
        plan_path, tmp_path / "out", readiness, **kwargs
    )
    corrected = runner.docker_worker_command(
        plan_path, tmp_path / "out", readiness, **kwargs
    )
    assert corrected[:-1] == frozen[:-1]
    assert corrected[-1] == (
        frozen[-1]
        + f" && chown -R {runner.EXPECTED_UID}:{runner.EXPECTED_GID} /out"
    )
    assert corrected[-1].count("chown -R") == 1

    marker = tmp_path / "chown_would_have_run"
    subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"false && printf ran > {marker}",
        ],
        check=False,
    )
    assert not marker.exists()


def test_isolated_ownership_probe_checks_identity_and_cleans(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_container(command, **_kwargs):
        commands.append(command)
        mount = command[command.index("-v") + 1]
        probe = Path(mount.removesuffix(":/out"))
        nested = probe / "nested"
        nested.mkdir()
        (nested / "probe").write_bytes(b"")

    report = runner.validate_isolated_ownership_probe(
        probe_parent=tmp_path,
        command_runner=fake_container,
    )
    assert len(commands) == 1
    assert report["output_uid"] == runner.EXPECTED_UID
    assert report["output_gid"] == runner.EXPECTED_GID
    assert report["host_write_delete_validated"] is True
    assert not tuple(tmp_path.glob(".direct_h8_v5_ownership_probe.*"))


def test_ownership_probe_contract_digest_is_path_independent(
    tmp_path: Path,
) -> None:
    left = runner.ownership_probe_command(tmp_path / "left")
    right = runner.ownership_probe_command(tmp_path / "right")
    assert left != right
    digest = runner.ownership_probe_contract_sha256()
    assert len(digest) == 64
    assert str(tmp_path) not in digest


def test_terminal_v4_closeout_is_value_blind_and_fully_bound() -> None:
    closeout = v4_closeout.validate_closeout(
        validate_live_filesystem_hashes=True
    )
    payload = closeout["closeout"]
    raw = payload["incomplete_raw_tree"]
    assert payload["outcome_values_parsed"] is False
    assert payload["trajectory_csv_parsing_performed"] is False
    assert raw["trajectory_values_parsed"] is False
    assert raw["files"] == 75
    assert raw["directories"] == 5
    assert (
        sum(
            (row["uid"], row["gid"]) == (0, 0)
            for row in raw["inventory"]
        )
        == 79
    )
    assert (
        sum(
            (row["uid"], row["gid"]) == (runner.EXPECTED_UID, runner.EXPECTED_GID)
            for row in raw["inventory"]
        )
        == 1
    )
    assert payload["collection_log"]["bytes"] == 23_516_102

    binding = runner.terminal_v4_failure_binding()
    bound = binding["binding"]
    assert bound["terminal_closeout_file_sha256"] == frozen_plan.sha256_file(
        v4_closeout.OUTPUT
    )
    assert (
        bound["terminal_closeout_payload_sha256"]
        == closeout["closeout_sha256"]
    )
    assert (
        bound["collection_log_sha256"]
        == payload["collection_log"]["sha256"]
    )
    assert (
        bound["incomplete_raw_inventory_canonical_sha256"]
        == raw["inventory_canonical_sha256"]
    )
    assert bound["retry_permitted_under_same_readiness_digest"] is False
    assert bound["data_v7_raw_reuse_permitted"] is False


def test_terminal_v4_digest_is_rejected_by_v5_entry_points(
    response_blind_readiness,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="terminal v4"):
        runner.load_bound_readiness(
            prelock_root=runner.PRELOCK_ROOT,
            live_data_root=response_blind_readiness["data_root"],
            readiness_path=response_blind_readiness["readiness_path"],
            expected_prelock_sha256=runner.ORIGINAL_PRELOCK_SHA256,
            expected_readiness_sha256=runner.TERMINAL_V4_READINESS_SHA256,
        )
    with pytest.raises(ValueError, match="terminal v4"):
        runner.run_collection(
            expected_readiness_sha256=runner.TERMINAL_V4_READINESS_SHA256,
            confirmation=runner.CONFIRMATION_TOKEN,
            data_root=tmp_path / "data",
            state_root=tmp_path / "state_v3",
            freeze_root=tmp_path / "freeze_v5",
        )
    report = dict(response_blind_readiness["readiness"].report)
    report["readiness_sha256"] = runner.TERMINAL_V4_READINESS_SHA256
    with pytest.raises(ValueError, match="self-verify"):
        runner.write_readiness_report(tmp_path / "readiness.json", report)


def test_staged_data_contains_only_frozen_plans_and_certificate(
    response_blind_readiness,
) -> None:
    data_root = response_blind_readiness["data_root"]
    actual = {
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
    }
    expected = {
        "plans",
        "plans/full",
        "disjointness_certificate.json",
        *{
            f"plans/full/{case}.json"
            for case in sorted(runner.boptest.CASES)
        },
    }
    assert actual == expected
    assert not (data_root / "locked_transport_raw").exists()
    assert not (data_root / "manifests").exists()
    assert all(
        frozen_plan.sha256_file(
            data_root / "plans/full" / f"{case}.json"
        )
        == frozen_plan.sha256_file(
            runner.PRELOCK_ROOT / "bundle/plans/full" / f"{case}.json"
        )
        for case in runner.boptest.CASES
    )


def test_readiness_self_hash_and_every_operational_hash_is_load_bearing(
    response_blind_readiness,
) -> None:
    report = response_blind_readiness["readiness"].report
    unsigned = {
        key: value for key, value in report.items() if key != "readiness_sha256"
    }
    assert frozen_plan.canonical_sha256(unsigned) == report["readiness_sha256"]
    assert report["readiness_sha256"] != runner.TERMINAL_V4_READINESS_SHA256
    assert report["prelock_registry_sha256"] == runner.ORIGINAL_PRELOCK_SHA256
    assert report["operational_code_sha256"] == runner.operational_code_hashes()
    for name in sorted(report["operational_code_sha256"]):
        changed = copy.deepcopy(unsigned)
        changed["operational_code_sha256"][name] = "f" * 64
        assert frozen_plan.canonical_sha256(changed) != report["readiness_sha256"]


def test_readiness_rejects_extra_pre_attempt_data(
    response_blind_readiness,
) -> None:
    data_root = response_blind_readiness["data_root"]
    extra = data_root / "unexpected"
    extra.write_text("not outcome data\n", encoding="ascii")
    try:
        with pytest.raises(ValueError, match="only frozen plans"):
            runner.validate_staged_plan_assets(
                data_root=data_root,
                prelock_root=runner.PRELOCK_ROOT,
                require_plan_only=True,
            )
    finally:
        extra.unlink()


def test_preflight_failure_consumes_digest_and_retry_fails_closed(
    response_blind_readiness,
    tmp_path: Path,
) -> None:
    readiness = response_blind_readiness["readiness"]
    data_root = tmp_path / "data"
    data_root.mkdir()
    raw = data_root / runner.RAW_RELATIVE
    raw.mkdir()
    state_root = tmp_path / "state_v3"
    freeze_root = tmp_path / "freeze_v5"
    freeze_root.mkdir()
    readiness_path = freeze_root / "collection_readiness.json"
    readiness_path.write_bytes(
        response_blind_readiness["readiness_path"].read_bytes()
    )
    receipt = freeze_root / "external_freeze_receipt.json"
    receipt.write_text("{}\n", encoding="ascii")

    def load(**_kwargs):
        return {}, readiness

    with pytest.raises(FileExistsError, match="destination exists"):
        runner.run_collection(
            expected_readiness_sha256=readiness.report["readiness_sha256"],
            confirmation=runner.CONFIRMATION_TOKEN,
            data_root=data_root,
            state_root=state_root,
            freeze_root=freeze_root,
            readiness_path=readiness_path,
            external_freeze_receipt_path=receipt,
            readiness_loader=load,
            freeze_validator=lambda *_args, **_kwargs: _fake_freeze(),
        )
    digest_state = state_root / readiness.report["readiness_sha256"]
    attempt = json.loads((digest_state / runner.ATTEMPT_MARKER).read_text())
    failure = json.loads((digest_state / runner.FAILURE_MARKER).read_text())
    assert attempt["locked_response_values_accessed"] is False
    assert failure["simulator_process_started"] is False
    assert failure["retry_permitted_under_same_readiness_digest"] is False

    with pytest.raises(FileExistsError, match="terminal"):
        runner.run_collection(
            expected_readiness_sha256=readiness.report["readiness_sha256"],
            confirmation=runner.CONFIRMATION_TOKEN,
            data_root=data_root,
            state_root=state_root,
            freeze_root=freeze_root,
            readiness_path=readiness_path,
            external_freeze_receipt_path=receipt,
            readiness_loader=load,
            freeze_validator=lambda *_args, **_kwargs: _fake_freeze(),
        )


def test_worker_failure_writes_terminal_marker_without_publishing(
    response_blind_readiness,
    tmp_path: Path,
) -> None:
    readiness = response_blind_readiness["readiness"]
    data_root = tmp_path / "data"
    data_root.mkdir()
    state_root = tmp_path / "state_v3"
    freeze_root = tmp_path / "freeze_v5"
    freeze_root.mkdir()
    readiness_path = freeze_root / "collection_readiness.json"
    readiness_path.write_bytes(
        response_blind_readiness["readiness_path"].read_bytes()
    )
    receipt = freeze_root / "external_freeze_receipt.json"
    receipt.write_text("{}\n", encoding="ascii")

    calls: list[list[str]] = []

    def fail_worker(command, **_kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(17, command)

    with pytest.raises(subprocess.CalledProcessError):
        runner.run_collection(
            expected_readiness_sha256=readiness.report["readiness_sha256"],
            confirmation=runner.CONFIRMATION_TOKEN,
            data_root=data_root,
            state_root=state_root,
            freeze_root=freeze_root,
            readiness_path=readiness_path,
            external_freeze_receipt_path=receipt,
            readiness_loader=lambda **_kwargs: ({}, readiness),
            freeze_validator=lambda *_args, **_kwargs: _fake_freeze(),
            command_runner=fail_worker,
        )
    digest_state = state_root / readiness.report["readiness_sha256"]
    failure = json.loads((digest_state / runner.FAILURE_MARKER).read_text())
    assert len(calls) == 1
    assert calls[0][-1].endswith(
        f"&& chown -R {runner.EXPECTED_UID}:{runner.EXPECTED_GID} /out"
    )
    assert failure["simulator_process_started"] is True
    assert failure["retry_permitted_under_same_readiness_digest"] is False
    assert not (data_root / runner.MANIFEST_RELATIVE).exists()


def test_attempt_completion_and_manifest_remain_frozen_loader_compatible(
    response_blind_readiness,
    tmp_path: Path,
) -> None:
    readiness = response_blind_readiness["readiness"]
    digest = readiness.report["readiness_sha256"]
    state_root = tmp_path / "state_v3"
    state_dir = state_root / digest
    state_dir.mkdir(parents=True)
    receipt = tmp_path / "external_freeze_receipt.json"
    receipt.write_text("{}\n", encoding="ascii")
    manifest_path = tmp_path / "locked_transport_corpus_manifest.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    freeze = _fake_freeze()
    attempt = runner._attempt_payload(
        readiness,
        freeze,
        external_freeze_receipt_path=receipt,
        commands=[],
        plan_root=tmp_path / "plans/full",
        certificate_path=tmp_path / "certificate.json",
        raw_root=tmp_path / "raw",
        manifest_path=manifest_path,
        staging=tmp_path / "staging",
    )
    attempt_path = state_dir / runner.ATTEMPT_MARKER
    attempt_path.write_bytes(frozen_plan.canonical_bytes(attempt))
    completion = runner._completion_payload(
        readiness=readiness,
        attempt_path=attempt_path,
        manifest_path=manifest_path,
        manifest={"manifest_sha256": "d" * 64},
    )
    completion_path = state_dir / runner.COMPLETION_MARKER
    completion_path.write_bytes(frozen_plan.canonical_bytes(completion))

    loaded_attempt, loaded_completion, _, _ = (
        frozen_corpus.validate_collection_completion(
            state_root=state_root,
            readiness=readiness,
            manifest_path=manifest_path,
            expected_prelock_sha256=runner.ORIGINAL_PRELOCK_SHA256,
            external_freeze=freeze,
            external_freeze_receipt_path=receipt,
        )
    )
    assert loaded_attempt["schema"] == frozen_collect.ATTEMPT_SCHEMA
    assert loaded_completion["schema"] == frozen_collect.COMPLETION_SCHEMA
    manifest = frozen_collect.build_manifest(
        readiness, {"files": [], "worker_receipts": []}
    )
    assert manifest["manifest"]["schema"] == frozen_collect.CORPUS_MANIFEST_SCHEMA
    assert set(manifest["manifest"]) == frozen_corpus.MANIFEST_FIELDS


def test_success_path_revalidates_shared_runtime_after_all_workers(
    response_blind_readiness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = response_blind_readiness["readiness"]
    data_root = tmp_path / "data"
    data_root.mkdir()
    state_root = tmp_path / "state_v3"
    freeze_root = tmp_path / "freeze_v5"
    freeze_root.mkdir()
    readiness_path = freeze_root / "collection_readiness.json"
    readiness_path.write_bytes(
        response_blind_readiness["readiness_path"].read_bytes()
    )
    receipt = freeze_root / "external_freeze_receipt.json"
    receipt.write_text("{}\n", encoding="ascii")
    shared_calls: list[dict] = []

    monkeypatch.setattr(runner, "_validate_output_ownership", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_validate_scientific_prelock",
        lambda _root: {"synthetic": True},
    )

    def validate_shared(**_kwargs):
        shared_calls.append(_kwargs)
        return readiness.report["live_shared_runtime_validation"]

    monkeypatch.setattr(
        runner, "validate_live_shared_runtime_semantics", validate_shared
    )
    monkeypatch.setattr(
        runner.frozen_collect,
        "validate_staged_collection",
        lambda *_args, **_kwargs: {"files": [], "worker_receipts": []},
    )

    def publish(_staging, _raw, manifest, pending, wrapper):
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(frozen_plan.canonical_bytes(wrapper))
        pending.unlink(missing_ok=True)

    monkeypatch.setattr(runner.frozen_collect, "_publish", publish)
    result = runner.run_collection(
        expected_readiness_sha256=readiness.report["readiness_sha256"],
        confirmation=runner.CONFIRMATION_TOKEN,
        data_root=data_root,
        state_root=state_root,
        freeze_root=freeze_root,
        readiness_path=readiness_path,
        external_freeze_receipt_path=receipt,
        readiness_loader=lambda **_kwargs: ({}, readiness),
        freeze_validator=lambda *_args, **_kwargs: _fake_freeze(),
        command_runner=lambda *_args, **_kwargs: None,
    )
    assert result["manifest"]["schema"] == frozen_collect.CORPUS_MANIFEST_SCHEMA
    assert len(shared_calls) == 1
    completion = (
        state_root
        / readiness.report["readiness_sha256"]
        / runner.COMPLETION_MARKER
    )
    assert completion.is_file()


def test_operations_do_not_mutate_frozen_source_or_prelock(
    response_blind_readiness,
) -> None:
    source_before = _tree_hashes(runner.FROZEN_SOURCE_ROOT)
    prelock_before = _tree_hashes(runner.PRELOCK_ROOT)
    runner.validate_frozen_source_hashes()
    runner.terminal_v4_failure_binding()
    runner.load_bound_readiness(
        prelock_root=runner.PRELOCK_ROOT,
        live_data_root=response_blind_readiness["data_root"],
        readiness_path=response_blind_readiness["readiness_path"],
        expected_prelock_sha256=runner.ORIGINAL_PRELOCK_SHA256,
        expected_readiness_sha256=response_blind_readiness[
            "readiness"
        ].report["readiness_sha256"],
    )
    assert _tree_hashes(runner.FROZEN_SOURCE_ROOT) == source_before
    assert _tree_hashes(runner.PRELOCK_ROOT) == prelock_before


def test_no_operational_module_reads_legacy_raw_values() -> None:
    operational = {
        "runner.py",
        "external_freeze.py",
        "evaluation_adapter.py",
        "v4_closeout.py",
    }
    for path in (runner.HERE / name for name in sorted(operational)):
        source = path.read_text(encoding="ascii")
        if path.name == "v4_closeout.py":
            assert "trajectory_csv_parsing_performed" in source
            assert "sha256_file(path)" in source
            assert "csv." not in source
        else:
            assert "data_v7/locked_transport_raw" not in source
