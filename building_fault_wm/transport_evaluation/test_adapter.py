from __future__ import annotations

from pathlib import Path

import pytest

from . import adapter


def test_exact_terminal_v4_audit_is_delegated_to_frozen_validator() -> None:
    result = adapter.verify_v4_terminal_audit_dispatch()
    assert result["terminal_v4_failure"]["binding"]["schema"] == (
        adapter.v5_runner.TERMINAL_BINDING_SCHEMA
    )
    assert result["terminal_v4_proxy_call_count"] == 1


def test_published_v5_metadata_preflight_stops_before_trajectory_loader() -> None:
    result = adapter.verify_v5_metadata_boundary()
    assert result["readiness_sha256"] == adapter.EXPECTED_READINESS_SHA256
    assert result["terminal_v4_proxy_call_count"] == 2
    assert result["trajectory_loader_called"] is False


def test_proxy_rejects_mixed_v4_v5_identity(
) -> None:
    original = (
        adapter.frozen_run.external_freeze.validate_external_freeze_receipt
    )
    proxy = adapter.TerminalV4FreezeProxy(original)
    with pytest.raises(ValueError, match="non-terminal-v4"):
        proxy.validate_external_freeze_receipt(
            adapter.v5_runner.V4_FREEZE_RECEIPT_PATH,
            prelock_root=adapter.v5_runner.PRELOCK_ROOT,
            readiness_path=adapter.v5_runner.V4_READINESS_PATH,
            expected_prelock_sha256=adapter.EXPECTED_PRELOCK_SHA256,
            expected_readiness_sha256=adapter.EXPECTED_READINESS_SHA256,
            live=False,
        )
    assert proxy.call_count == 0


def test_hook_restores_metadata_calls_and_preserves_numerical_callables() -> None:
    original_alias = adapter.v5_runner.frozen_external_freeze
    numerical = adapter._numerical_callables()
    with pytest.raises(RuntimeError, match="synthetic"):
        with adapter.terminal_v4_validation_proxy():
            assert adapter.v5_runner.frozen_external_freeze is not original_alias
            assert adapter._numerical_callables() == numerical
            raise RuntimeError("synthetic")
    assert adapter.v5_runner.frozen_external_freeze is original_alias
    assert adapter._numerical_callables() == numerical


def test_upstream_hashes_bind_v5_adapter_readiness_and_frozen_evaluator() -> None:
    assert adapter.upstream_hashes() == adapter.EXPECTED_UPSTREAM_SHA256
    paths = adapter.upstream_paths()
    assert paths["v5_readiness"] == adapter.v5_runner.READINESS_PATH
    assert paths["v5_evaluation_adapter"] == Path(
        adapter.v5_adapter.__file__
    ).resolve()
    assert paths["frozen_run_evaluation"] == Path(
        adapter.frozen_run.__file__
    ).resolve()


def test_cli_defaults_to_fresh_v6_output_and_exact_identity() -> None:
    args = adapter._parse_args(["run-verify"])
    assert args.output == adapter.DEFAULT_OUTPUT
    assert args.expected_prelock_sha256 == adapter.EXPECTED_PRELOCK_SHA256
    assert args.expected_readiness_sha256 == adapter.EXPECTED_READINESS_SHA256
    assert "evaluation_v6" in str(args.output)


def test_invocation_rejects_alternate_numerical_or_output_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="output_dir"):
        adapter.invoke_frozen_evaluation(
            "run",
            output_dir=tmp_path / "alternate",
        )
