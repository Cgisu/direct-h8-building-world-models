"""One-shot persistent attempt state for the v6 evaluation recovery."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    write_json_once,
)


ATTEMPT_NAME = "evaluation_recovery_attempt.json"
FAILURE_NAME = "evaluation_recovery_failure.json"
COMPLETION_NAME = "evaluation_recovery_completion.json"
LOCK_NAME = "evaluation_recovery.lock"
ATTEMPT_SCHEMA = "direct-h8-evaluation-v6-recovery-attempt-v1"
FAILURE_SCHEMA = "direct-h8-evaluation-v6-recovery-failure-v1"
COMPLETION_SCHEMA = "direct-h8-evaluation-v6-recovery-completion-v1"


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} is not UTC")
    return parsed


@contextmanager
def exclusive_attempt_lock(state_root: Path) -> Iterator[None]:
    state_root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        state_root / LOCK_NAME,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another v6 recovery invocation is active") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _marker_paths(state_root: Path) -> dict[str, Path]:
    return {
        "attempt": state_root / ATTEMPT_NAME,
        "failure": state_root / FAILURE_NAME,
        "completion": state_root / COMPLETION_NAME,
    }


def begin_attempt(
    state_root: Path,
    *,
    recovery_prelock_sha256: str,
    recovery_public_freeze_receipt_path: Path,
    recovery_public_freeze: Mapping[str, object],
    v5_readiness_sha256: str,
    output_dir: Path,
) -> Path:
    markers = _marker_paths(state_root)
    if any(os.path.lexists(path) for path in markers.values()):
        raise FileExistsError(
            "v6 recovery digest already has an attempt, failure, or completion marker"
        )
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if os.path.lexists(output_dir) or os.path.lexists(staging):
        raise FileExistsError("v6 output or staging exists before the attempt")
    committed = _parse_utc(
        recovery_public_freeze.get("revision_committed_at_utc"),
        "recovery public revision time",
    )
    verified = _parse_utc(
        recovery_public_freeze.get("provider_verified_at_utc"),
        "trusted GitHub verification time",
    )
    if committed > verified:
        raise ValueError("recovery public revision does not predate the attempt")
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "stage": "before_frozen_evaluation",
        "recovery_prelock_sha256": recovery_prelock_sha256,
        "recovery_public_freeze_receipt_file_sha256": sha256_file(
            recovery_public_freeze_receipt_path
        ),
        "recovery_public_revision": recovery_public_freeze.get("revision"),
        "recovery_public_revision_committed_at_utc": (
            recovery_public_freeze.get("revision_committed_at_utc")
        ),
        "trusted_provider_verified_at_utc": (
            recovery_public_freeze.get("provider_verified_at_utc")
        ),
        "v5_readiness_sha256": v5_readiness_sha256,
        "output_dir": str(output_dir.resolve()),
        "output_exists": False,
        "staging_exists": False,
        "locked_trajectory_or_outcome_values_accessed": False,
        "retry_permitted_under_same_recovery_digest": False,
    }
    signed = {**payload, "attempt_sha256": canonical_sha256(payload)}
    return write_json_once(markers["attempt"], signed)


def validate_attempt(
    path: Path,
    *,
    recovery_prelock_sha256: str,
    recovery_public_freeze_receipt_path: Path,
    v5_readiness_sha256: str,
    output_dir: Path,
) -> dict:
    payload = strict_json(path)
    digest = payload.get("attempt_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "attempt_sha256"
    }
    expected_fields = {
        "schema",
        "stage",
        "recovery_prelock_sha256",
        "recovery_public_freeze_receipt_file_sha256",
        "recovery_public_revision",
        "recovery_public_revision_committed_at_utc",
        "trusted_provider_verified_at_utc",
        "v5_readiness_sha256",
        "output_dir",
        "output_exists",
        "staging_exists",
        "locked_trajectory_or_outcome_values_accessed",
        "retry_permitted_under_same_recovery_digest",
        "attempt_sha256",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema") != ATTEMPT_SCHEMA
        or payload.get("stage") != "before_frozen_evaluation"
        or payload.get("recovery_prelock_sha256") != recovery_prelock_sha256
        or payload.get("recovery_public_freeze_receipt_file_sha256")
        != sha256_file(recovery_public_freeze_receipt_path)
        or payload.get("v5_readiness_sha256") != v5_readiness_sha256
        or payload.get("output_dir") != str(output_dir.resolve())
        or payload.get("output_exists") is not False
        or payload.get("staging_exists") is not False
        or payload.get("locked_trajectory_or_outcome_values_accessed") is not False
        or payload.get("retry_permitted_under_same_recovery_digest") is not False
        or digest != canonical_sha256(unsigned)
    ):
        raise ValueError("v6 recovery attempt marker changed")
    committed = _parse_utc(
        payload.get("recovery_public_revision_committed_at_utc"),
        "attempt public revision time",
    )
    verified = _parse_utc(
        payload.get("trusted_provider_verified_at_utc"),
        "attempt trusted provider time",
    )
    if committed > verified:
        raise ValueError("v6 attempt predates its public recovery revision")
    return payload


def record_failure(
    state_root: Path,
    *,
    attempt_path: Path,
    error: BaseException,
    output_dir: Path,
) -> Path:
    markers = _marker_paths(state_root)
    if os.path.lexists(markers["failure"]) or os.path.lexists(
        markers["completion"]
    ):
        raise FileExistsError("v6 recovery already has a terminal marker")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    payload = {
        "schema": FAILURE_SCHEMA,
        "terminal": True,
        "attempt_file_sha256": sha256_file(attempt_path),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "output_exists": os.path.lexists(output_dir),
        "staging_exists": os.path.lexists(staging),
        "locked_trajectory_or_outcome_values_may_have_been_accessed": True,
        "retry_permitted_under_same_recovery_digest": False,
    }
    return write_json_once(
        markers["failure"],
        {**payload, "failure_sha256": canonical_sha256(payload)},
    )


def record_completion(
    state_root: Path,
    *,
    attempt_path: Path,
    recovery_prelock_sha256: str,
    output_dir: Path,
) -> Path:
    markers = _marker_paths(state_root)
    if os.path.lexists(markers["failure"]) or os.path.lexists(
        markers["completion"]
    ):
        raise FileExistsError("v6 recovery already has a terminal marker")
    frozen_completion = output_dir / "evaluation_complete.json"
    payload = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "attempt_file_sha256": sha256_file(attempt_path),
        "recovery_prelock_sha256": recovery_prelock_sha256,
        "output_dir": str(output_dir.resolve()),
        "frozen_evaluation_completion_file_sha256": sha256_file(
            frozen_completion
        ),
        "retry_permitted_under_same_recovery_digest": False,
    }
    return write_json_once(
        markers["completion"],
        {**payload, "completion_sha256": canonical_sha256(payload)},
    )


def validate_completion(
    state_root: Path,
    *,
    recovery_prelock_sha256: str,
    recovery_public_freeze_receipt_path: Path,
    v5_readiness_sha256: str,
    output_dir: Path,
) -> dict:
    markers = _marker_paths(state_root)
    if os.path.lexists(markers["failure"]):
        raise ValueError("v6 recovery is terminally failed")
    attempt_payload = validate_attempt(
        markers["attempt"],
        recovery_prelock_sha256=recovery_prelock_sha256,
        recovery_public_freeze_receipt_path=(
            recovery_public_freeze_receipt_path
        ),
        v5_readiness_sha256=v5_readiness_sha256,
        output_dir=output_dir,
    )
    payload = strict_json(markers["completion"])
    digest = payload.get("completion_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "completion_sha256"
    }
    expected_fields = {
        "schema",
        "complete",
        "attempt_file_sha256",
        "recovery_prelock_sha256",
        "output_dir",
        "frozen_evaluation_completion_file_sha256",
        "retry_permitted_under_same_recovery_digest",
        "completion_sha256",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema") != COMPLETION_SCHEMA
        or payload.get("complete") is not True
        or payload.get("attempt_file_sha256")
        != sha256_file(markers["attempt"])
        or payload.get("recovery_prelock_sha256") != recovery_prelock_sha256
        or payload.get("output_dir") != str(output_dir.resolve())
        or payload.get("frozen_evaluation_completion_file_sha256")
        != sha256_file(output_dir / "evaluation_complete.json")
        or payload.get("retry_permitted_under_same_recovery_digest") is not False
        or digest != canonical_sha256(unsigned)
    ):
        raise ValueError("v6 recovery completion marker changed")
    return {"attempt": attempt_payload, "completion": payload}
