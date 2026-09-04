from __future__ import annotations

import json
from pathlib import Path

import pytest

from building_fault_wm.ridge_arx.io import (
    write_json_once,
)

from . import attempt


def _public_receipt() -> dict:
    return {
        "revision": "d" * 40,
        "revision_committed_at_utc": "2026-07-23T10:00:00Z",
        "provider_verified_at_utc": "2026-07-23T10:01:00Z",
    }


def test_attempt_is_write_once_and_provider_time_precedes_it(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_public_receipt()), encoding="ascii")
    state = tmp_path / "state"
    output = tmp_path / "output"
    path = attempt.begin_attempt(
        state,
        recovery_prelock_sha256="a" * 64,
        recovery_public_freeze_receipt_path=receipt_path,
        recovery_public_freeze=_public_receipt(),
        v5_readiness_sha256="b" * 64,
        output_dir=output,
    )
    payload = attempt.validate_attempt(
        path,
        recovery_prelock_sha256="a" * 64,
        recovery_public_freeze_receipt_path=receipt_path,
        v5_readiness_sha256="b" * 64,
        output_dir=output,
    )
    assert payload["locked_trajectory_or_outcome_values_accessed"] is False
    with pytest.raises(FileExistsError, match="already has"):
        attempt.begin_attempt(
            state,
            recovery_prelock_sha256="a" * 64,
            recovery_public_freeze_receipt_path=receipt_path,
            recovery_public_freeze=_public_receipt(),
            v5_readiness_sha256="b" * 64,
            output_dir=output,
        )


def test_attempt_rejects_public_revision_after_provider_time(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="ascii")
    receipt = {
        **_public_receipt(),
        "revision_committed_at_utc": "2026-07-23T10:02:00Z",
    }
    with pytest.raises(ValueError, match="does not predate"):
        attempt.begin_attempt(
            tmp_path / "state",
            recovery_prelock_sha256="a" * 64,
            recovery_public_freeze_receipt_path=receipt_path,
            recovery_public_freeze=receipt,
            v5_readiness_sha256="b" * 64,
            output_dir=tmp_path / "output",
        )


def test_exclusive_lock_rejects_concurrent_invocation(tmp_path: Path) -> None:
    with attempt.exclusive_attempt_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another"):
            with attempt.exclusive_attempt_lock(tmp_path):
                pass


def test_completion_binds_frozen_completion_file(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_public_receipt()), encoding="ascii")
    state = tmp_path / "state"
    output = tmp_path / "output"
    attempt_path = attempt.begin_attempt(
        state,
        recovery_prelock_sha256="a" * 64,
        recovery_public_freeze_receipt_path=receipt_path,
        recovery_public_freeze=_public_receipt(),
        v5_readiness_sha256="b" * 64,
        output_dir=output,
    )
    write_json_once(output / "evaluation_complete.json", {"complete": True})
    attempt.record_completion(
        state,
        attempt_path=attempt_path,
        recovery_prelock_sha256="a" * 64,
        output_dir=output,
    )
    result = attempt.validate_completion(
        state,
        recovery_prelock_sha256="a" * 64,
        recovery_public_freeze_receipt_path=receipt_path,
        v5_readiness_sha256="b" * 64,
        output_dir=output,
    )
    assert result["completion"]["complete"] is True
