from __future__ import annotations

from pathlib import Path

import pytest

from . import evaluation_adapter, runner


def test_adapter_byte_verifies_frozen_numerical_sources() -> None:
    result = evaluation_adapter.verify_frozen_numerical_path()
    assert (
        result["run_evaluation.py"]
        == runner.FROZEN_RUN_EVALUATION_SHA256
    )
    assert result["corpus.py"] == runner.FROZEN_CORPUS_SHA256
    assert result["patched_calls"] == list(evaluation_adapter.PATCHED_CALLS)


def test_metadata_hooks_patch_only_two_calls_and_restore_on_exception(
    tmp_path: Path,
) -> None:
    frozen = evaluation_adapter.frozen_run
    original_readiness = frozen.corpus.load_bound_readiness
    original_freeze = frozen.external_freeze.validate_external_freeze_receipt
    original_transport = frozen.corpus.load_transport_corpus_index
    original_run = frozen.run_evaluation
    original_verify = frozen.verify_only

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        with evaluation_adapter.v5_metadata_hooks(
            readiness_path=tmp_path / "readiness.json"
        ):
            assert frozen.corpus.load_bound_readiness is not original_readiness
            assert (
                frozen.external_freeze.validate_external_freeze_receipt
                is not original_freeze
            )
            assert frozen.corpus.load_transport_corpus_index is original_transport
            assert frozen.run_evaluation is original_run
            assert frozen.verify_only is original_verify
            raise RuntimeError("synthetic evaluator failure")

    assert frozen.corpus.load_bound_readiness is original_readiness
    assert (
        frozen.external_freeze.validate_external_freeze_receipt
        is original_freeze
    )
    assert frozen.corpus.load_transport_corpus_index is original_transport
    assert frozen.run_evaluation is original_run
    assert frozen.verify_only is original_verify


def test_invocation_restores_hooks_and_preserves_frozen_hashes_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = evaluation_adapter.frozen_run
    original_readiness = frozen.corpus.load_bound_readiness
    original_freeze = frozen.external_freeze.validate_external_freeze_receipt
    source_before = evaluation_adapter.frozen_source_snapshot()
    shared_calls: list[Path] = []
    original_shared_snapshot = evaluation_adapter.shared_runtime_snapshot

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic numerical failure")

    def track_shared(prelock_root):
        shared_calls.append(prelock_root)
        return original_shared_snapshot(prelock_root)

    monkeypatch.setattr(evaluation_adapter, "_invoke_once", fail)
    monkeypatch.setattr(
        evaluation_adapter, "shared_runtime_snapshot", track_shared
    )
    with pytest.raises(RuntimeError, match="synthetic numerical failure"):
        evaluation_adapter.invoke_frozen_evaluation(
            "run",
            expected_prelock_sha256=runner.ORIGINAL_PRELOCK_SHA256,
            expected_readiness_sha256="7" * 64,
            prelock_root=runner.PRELOCK_ROOT,
            data_root=tmp_path / "data",
            state_root=tmp_path / "state_v3",
            readiness_path=tmp_path / "freeze_v5/readiness.json",
            external_freeze_receipt_path=tmp_path / "freeze_v5/receipt.json",
            output_dir=tmp_path / "output",
            live_external_freeze=False,
        )
    assert frozen.corpus.load_bound_readiness is original_readiness
    assert (
        frozen.external_freeze.validate_external_freeze_receipt
        is original_freeze
    )
    assert evaluation_adapter.frozen_source_snapshot() == source_before
    assert shared_calls == [runner.PRELOCK_ROOT, runner.PRELOCK_ROOT]


def test_run_verify_cli_accepts_all_path_overrides(tmp_path: Path) -> None:
    args = evaluation_adapter._parse_args(
        [
            "run-verify",
            "--expected-readiness-sha256",
            "7" * 64,
            "--prelock-root",
            str(tmp_path / "prelock"),
            "--data-root",
            str(tmp_path / "data"),
            "--state-root",
            str(tmp_path / "state_v3"),
            "--readiness",
            str(tmp_path / "freeze_v5/readiness.json"),
            "--external-freeze-receipt",
            str(tmp_path / "freeze_v5/receipt.json"),
            "--output",
            str(tmp_path / "output"),
            "--no-live-external-freeze",
        ]
    )
    assert args.command == "run-verify"
    assert args.prelock_root == tmp_path / "prelock"
    assert args.data_root == tmp_path / "data"
    assert args.state_root == tmp_path / "state_v3"
    assert args.readiness == tmp_path / "freeze_v5/readiness.json"
    assert args.external_freeze_receipt == tmp_path / "freeze_v5/receipt.json"
    assert args.output == tmp_path / "output"
    assert args.no_live_external_freeze is True


def test_readiness_binds_adapter_and_frozen_source_hashes(
    response_blind_readiness,
) -> None:
    report = response_blind_readiness["readiness"].report
    assert report["operational_code_sha256"]["evaluation_adapter.py"] == (
        runner.frozen_plan.sha256_file(runner.HERE / "evaluation_adapter.py")
    )
    assert report["frozen_evaluation_source_sha256"] == {
        "run_evaluation.py": runner.FROZEN_RUN_EVALUATION_SHA256,
        "corpus.py": runner.FROZEN_CORPUS_SHA256,
    }
