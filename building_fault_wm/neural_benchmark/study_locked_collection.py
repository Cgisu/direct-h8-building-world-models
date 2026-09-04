from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from . import collect as collector
from .locked_state import (
    COLLECTION_ATTEMPT_MARKER,
    COLLECTION_COMPLETION_MARKER,
    COLLECTION_FAILURE_MARKER,
    EXTERNAL_FREEZE_RECEIPT,
    canonical_sha256,
    is_sha256,
    state_dir_for_digest,
    write_bytes_once,
    write_json_once,
)
from .protocol import CASES, TRAJECTORY_STEPS, sha256_file, strict_json_loads
from .provenance import load_strict_json, validate_prelock_bundle
from .study_config import StudyConfig


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CANONICAL_OUTPUT = HERE / "data_v4"
CANONICAL_LOCKED_RAW = CANONICAL_OUTPUT / collector.LOCKED_RAW_SUBDIR
CANONICAL_MANIFEST_ROOT = CANONICAL_OUTPUT / collector.MANIFEST_SUBDIR
CANONICAL_LOCKED_MANIFEST = (
    CANONICAL_MANIFEST_ROOT / "locked_test_all_corpus_manifest.json"
)
CANONICAL_PRELOCK_ROOT = PROJECT_ROOT / "artifacts" / "prelock_v4"
CANONICAL_PRELOCK_REGISTRY = CANONICAL_PRELOCK_ROOT / "prelock_registry.json"
CANONICAL_PRELOCK_BUNDLE = CANONICAL_PRELOCK_ROOT / "bundle"
CANONICAL_PRELOCK_DIGEST = (
    CANONICAL_PRELOCK_ROOT / "prelock_registry.canonical.sha256"
)

EXTERNAL_FREEZE_SCHEMA = "boptest-reliability-external-prelock-freeze-v1"
ATTEMPT_SCHEMA = "boptest-reliability-locked-collection-attempt-v1"
FAILURE_SCHEMA = "boptest-reliability-locked-collection-failure-v1"
COMPLETION_SCHEMA = "boptest-reliability-locked-collection-completion-v1"
CONFIRMATION_TOKEN = "I_UNDERSTAND_LOCKED_COLLECTION_IS_ONE_SHOT"
UTC_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

EXTERNAL_FREEZE_FIELDS = frozenset(
    {
        "schema",
        "prelock_registry_sha256",
        "timestamp_utc",
        "provider",
        "public_https_url",
    }
)
ATTEMPT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "started_at_utc",
        "pid",
        "prelock_registry_sha256",
        "prelock_registry_path",
        "prelock_registry_file_sha256",
        "prelock_artifact_root",
        "prelock_digest_record_path",
        "output_root",
        "locked_raw_root",
        "locked_manifest_path",
        "external_freeze_receipt_path",
        "external_freeze_receipt_sha256",
        "external_freeze_timestamp_utc",
        "external_freeze_provider",
        "external_freeze_public_https_url",
        "collector_command",
    }
)
COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "completed_at_utc",
        "prelock_registry_sha256",
        "prelock_registry_file_sha256",
        "attempt_marker_sha256",
        "external_freeze_receipt_sha256",
        "collector_exit_code",
        "locked_manifest_path",
        "locked_manifest_file_sha256",
        "locked_manifest_payload_sha256",
        "locked_raw_root",
        "collection_kind",
        "selected_cases",
        "counts",
        "locked_values_accessed_after_attempt",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or UTC_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be UTC at whole-second precision")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{label} is not a valid UTC timestamp") from error
    return parsed.replace(tzinfo=timezone.utc)


def _require_plain_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")


def _require_plain_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")


def _validate_public_https_url(value: object) -> str:
    if not isinstance(value, str) or value != value.strip() or len(value) > 2048:
        raise ValueError("external freeze URL must be a nonempty HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("external freeze URL must be a public HTTPS URL")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("external freeze URL must not name a local service")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        if "." not in hostname:
            raise ValueError("external freeze URL must have a public hostname")
    else:
        if not address.is_global:
            raise ValueError("external freeze URL must not use a private address")
    return value


def validate_external_freeze_receipt(
    receipt_path: Path, expected_prelock_sha256: str
) -> tuple[dict, bytes]:
    _require_plain_file(receipt_path, "external freeze receipt")
    receipt_bytes = receipt_path.read_bytes()
    try:
        payload = strict_json_loads(receipt_bytes.decode("ascii"))
    except UnicodeDecodeError as error:
        raise ValueError("external freeze receipt must be ASCII JSON") from error
    if not isinstance(payload, dict) or set(payload) != EXTERNAL_FREEZE_FIELDS:
        raise ValueError("external freeze receipt fields differ from the frozen schema")
    if payload["schema"] != EXTERNAL_FREEZE_SCHEMA:
        raise ValueError("external freeze receipt schema is invalid")
    if payload["prelock_registry_sha256"] != expected_prelock_sha256:
        raise ValueError("external freeze receipt binds a different pre-lock digest")
    _parse_utc_timestamp(payload["timestamp_utc"], "external freeze timestamp")
    provider = payload["provider"]
    if (
        not isinstance(provider, str)
        or provider != provider.strip()
        or not provider
        or len(provider) > 200
        or not provider.isascii()
        or any(ord(character) < 32 for character in provider)
    ):
        raise ValueError("external freeze provider is invalid")
    _validate_public_https_url(payload["public_https_url"])
    return payload, receipt_bytes


def _validate_fixed_layout() -> None:
    if collector.DEFAULT_OUTPUT != CANONICAL_OUTPUT:
        raise RuntimeError("collector default output differs from the formal locked layout")
    if CANONICAL_MANIFEST_ROOT != CANONICAL_OUTPUT / collector.MANIFEST_SUBDIR:
        raise RuntimeError("locked manifest root differs from the formal locked layout")
    if CANONICAL_LOCKED_RAW != CANONICAL_OUTPUT / collector.LOCKED_RAW_SUBDIR:
        raise RuntimeError("locked raw root differs from the formal locked layout")
    if CANONICAL_LOCKED_MANIFEST.parent != CANONICAL_MANIFEST_ROOT:
        raise RuntimeError("locked manifest escaped its canonical root")
    for path, label in (
        (CANONICAL_OUTPUT, "canonical output root"),
        (CANONICAL_MANIFEST_ROOT, "canonical manifest root"),
        (CANONICAL_PRELOCK_ROOT, "canonical pre-lock root"),
    ):
        if path.resolve() != path.absolute():
            raise ValueError(f"{label} traverses a symbolic link")
        if path.exists() and not path.is_dir():
            raise ValueError(f"{label} is not a directory")


def _read_digest_record(expected_prelock_sha256: str) -> None:
    _require_plain_file(CANONICAL_PRELOCK_DIGEST, "canonical pre-lock digest record")
    recorded = CANONICAL_PRELOCK_DIGEST.read_text(encoding="ascii")
    if recorded != f"{expected_prelock_sha256}\n":
        raise ValueError("canonical pre-lock digest record differs from the supplied digest")


def _collector_command(expected_prelock_sha256: str) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        "building_fault_wm.neural_benchmark.collect",
        "collect-locked",
        "--mode",
        "full",
        "--testcase-root",
        str(collector.DEFAULT_TESTCASE_ROOT),
        "--output-dir",
        str(CANONICAL_OUTPUT),
    ]
    for case in sorted(CASES):
        command.extend(("--case", case))
    command.extend(
        (
            "--confirm-locked-test",
            collector.LOCKED_CONFIRMATION,
            "--prelock-registry",
            str(CANONICAL_PRELOCK_REGISTRY),
            "--prelock-artifact-root",
            str(CANONICAL_PRELOCK_BUNDLE),
            "--expected-prelock-sha256",
            expected_prelock_sha256,
        )
    )
    return command


def _prepare_state_directory(expected_prelock_sha256: str) -> Path:
    state_dir = state_dir_for_digest(expected_prelock_sha256)
    state_root = state_dir.parent
    state_root.mkdir(parents=True, exist_ok=True)
    if state_root.is_symlink() or not state_root.is_dir():
        raise ValueError("locked state root is not a plain directory")
    state_dir.mkdir(exist_ok=True)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise ValueError("digest-scoped locked state is not a plain directory")
    allowed = {
        EXTERNAL_FREEZE_RECEIPT,
        COLLECTION_ATTEMPT_MARKER,
        COLLECTION_FAILURE_MARKER,
        COLLECTION_COMPLETION_MARKER,
    }
    unexpected = sorted(path.name for path in state_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(f"digest-scoped locked state contains unexpected entries: {unexpected}")
    return state_dir


def _manifest_evidence(expected_prelock_sha256: str) -> dict:
    _require_plain_file(CANONICAL_LOCKED_MANIFEST, "canonical locked manifest")
    _require_plain_directory(CANONICAL_LOCKED_RAW, "canonical locked raw corpus")
    wrapper = load_strict_json(CANONICAL_LOCKED_MANIFEST)
    if not isinstance(wrapper, dict) or set(wrapper) != {"manifest_sha256", "manifest"}:
        raise ValueError("locked manifest wrapper fields are invalid")
    manifest = wrapper["manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("locked corpus manifest payload is invalid")
    payload_sha256 = canonical_sha256(manifest)
    if wrapper["manifest_sha256"] != payload_sha256:
        raise ValueError("locked corpus manifest self-hash is invalid")
    required = {
        "collection_kind": "locked_test",
        "plan_mode": "full",
        "allowed_roles": ["locked_test"],
        "selected_cases": sorted(CASES),
        "prelock_registry_sha256": expected_prelock_sha256,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"locked corpus manifest {field} differs from the frozen contract")
    counts = manifest.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("cases") != len(CASES)
        or counts.get("trajectories") != 12 * len(CASES)
        or counts.get("rows") != 12 * len(CASES) * TRAJECTORY_STEPS
        or counts.get("roles") != {"locked_test": 12 * len(CASES)}
    ):
        raise ValueError("locked corpus manifest counts are invalid")
    return {
        "locked_manifest_file_sha256": sha256_file(CANONICAL_LOCKED_MANIFEST),
        "locked_manifest_payload_sha256": payload_sha256,
        "collection_kind": manifest["collection_kind"],
        "selected_cases": manifest["selected_cases"],
        "counts": counts,
    }


def _write_failure(
    state_dir: Path,
    *,
    expected_prelock_sha256: str,
    attempt_path: Path,
    receipt_sha256: str,
    error: BaseException,
) -> None:
    return_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
    payload = {
        "schema": FAILURE_SCHEMA,
        "stage": "locked_collection",
        "failed_at_utc": _utc_now(),
        "prelock_registry_sha256": expected_prelock_sha256,
        "attempt_marker_sha256": sha256_file(attempt_path),
        "external_freeze_receipt_sha256": receipt_sha256,
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:2000],
        "collector_return_code": return_code,
        "completion_written": False,
    }
    write_json_once(state_dir / COLLECTION_FAILURE_MARKER, payload)


def run_locked_collection(
    expected_prelock_sha256: str,
    external_freeze_receipt: Path,
    confirmation_token: str,
) -> Path:
    if confirmation_token != CONFIRMATION_TOKEN:
        raise ValueError(
            "formal locked collection requires --confirm-locked-collection "
            + CONFIRMATION_TOKEN
        )
    if not is_sha256(expected_prelock_sha256):
        raise ValueError("expected pre-lock digest must be a lowercase SHA-256")
    _validate_fixed_layout()
    _require_plain_file(CANONICAL_PRELOCK_REGISTRY, "canonical pre-lock registry")
    _require_plain_directory(CANONICAL_PRELOCK_BUNDLE, "canonical pre-lock bundle")
    _read_digest_record(expected_prelock_sha256)
    receipt, receipt_bytes = validate_external_freeze_receipt(
        external_freeze_receipt, expected_prelock_sha256
    )
    validate_prelock_bundle(
        CANONICAL_PRELOCK_REGISTRY,
        CANONICAL_PRELOCK_BUNDLE,
        StudyConfig(),
        expected_prelock_sha256,
    )
    started_at_utc = _utc_now()
    if _parse_utc_timestamp(
        receipt["timestamp_utc"], "external freeze timestamp"
    ) > _parse_utc_timestamp(started_at_utc, "locked collection attempt timestamp"):
        raise ValueError("external freeze timestamp is later than the collection attempt")

    state_dir = _prepare_state_directory(expected_prelock_sha256)
    attempt_path = state_dir / COLLECTION_ATTEMPT_MARKER
    terminal_paths = (
        state_dir / COLLECTION_FAILURE_MARKER,
        state_dir / COLLECTION_COMPLETION_MARKER,
    )
    if os.path.lexists(attempt_path) or any(os.path.lexists(path) for path in terminal_paths):
        raise FileExistsError(
            "this externally frozen pre-lock digest already has a locked collection attempt"
        )
    copied_receipt = state_dir / EXTERNAL_FREEZE_RECEIPT
    write_bytes_once(copied_receipt, receipt_bytes, identical_ok=True)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if sha256_file(copied_receipt) != receipt_sha256:
        raise IOError("copied external freeze receipt differs from its source")

    command = _collector_command(expected_prelock_sha256)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "stage": "locked_collection",
        "started_at_utc": started_at_utc,
        "pid": os.getpid(),
        "prelock_registry_sha256": expected_prelock_sha256,
        "prelock_registry_path": str(CANONICAL_PRELOCK_REGISTRY),
        "prelock_registry_file_sha256": sha256_file(CANONICAL_PRELOCK_REGISTRY),
        "prelock_artifact_root": str(CANONICAL_PRELOCK_BUNDLE),
        "prelock_digest_record_path": str(CANONICAL_PRELOCK_DIGEST),
        "output_root": str(CANONICAL_OUTPUT),
        "locked_raw_root": str(CANONICAL_LOCKED_RAW),
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "external_freeze_receipt_path": str(copied_receipt),
        "external_freeze_receipt_sha256": receipt_sha256,
        "external_freeze_timestamp_utc": receipt["timestamp_utc"],
        "external_freeze_provider": receipt["provider"],
        "external_freeze_public_https_url": receipt["public_https_url"],
        "collector_command": command,
    }
    if set(attempt) != ATTEMPT_FIELDS:
        raise AssertionError("locked collection attempt fields differ from schema")
    write_json_once(attempt_path, attempt)

    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        evidence = _manifest_evidence(expected_prelock_sha256)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "stage": "locked_collection",
            "completed_at_utc": _utc_now(),
            "prelock_registry_sha256": expected_prelock_sha256,
            "prelock_registry_file_sha256": sha256_file(CANONICAL_PRELOCK_REGISTRY),
            "attempt_marker_sha256": sha256_file(attempt_path),
            "external_freeze_receipt_sha256": receipt_sha256,
            "collector_exit_code": 0,
            "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
            "locked_raw_root": str(CANONICAL_LOCKED_RAW),
            "locked_values_accessed_after_attempt": True,
            **evidence,
        }
        if set(completion) != COMPLETION_FIELDS:
            raise AssertionError("locked collection completion fields differ from schema")
        write_json_once(state_dir / COLLECTION_COMPLETION_MARKER, completion)
    except BaseException as error:
        try:
            _write_failure(
                state_dir,
                expected_prelock_sha256=expected_prelock_sha256,
                attempt_path=attempt_path,
                receipt_sha256=receipt_sha256,
                error=error,
            )
        except BaseException as evidence_error:
            raise RuntimeError(
                "locked collection failed and immutable failure evidence could not be written"
            ) from evidence_error
        raise
    return CANONICAL_LOCKED_MANIFEST


def validate_locked_collection_completion(expected_prelock_sha256: str) -> dict:
    """Validate the one-shot collection receipt before confirmation reads locked values."""
    if not is_sha256(expected_prelock_sha256):
        raise ValueError("expected pre-lock digest must be a lowercase SHA-256")
    _validate_fixed_layout()
    _require_plain_file(CANONICAL_PRELOCK_REGISTRY, "canonical pre-lock registry")
    _require_plain_directory(CANONICAL_PRELOCK_BUNDLE, "canonical pre-lock bundle")
    _read_digest_record(expected_prelock_sha256)
    state_dir = state_dir_for_digest(expected_prelock_sha256)
    _require_plain_directory(state_dir, "digest-scoped locked state")
    attempt_path = state_dir / COLLECTION_ATTEMPT_MARKER
    completion_path = state_dir / COLLECTION_COMPLETION_MARKER
    receipt_path = state_dir / EXTERNAL_FREEZE_RECEIPT
    for path, label in (
        (attempt_path, "locked collection attempt marker"),
        (completion_path, "locked collection completion marker"),
        (receipt_path, "copied external freeze receipt"),
    ):
        _require_plain_file(path, label)
    if os.path.lexists(state_dir / COLLECTION_FAILURE_MARKER):
        raise ValueError("locked collection has immutable failure evidence")
    attempt = load_strict_json(attempt_path)
    completion = load_strict_json(completion_path)
    if not isinstance(attempt, dict) or set(attempt) != ATTEMPT_FIELDS:
        raise ValueError("locked collection attempt marker fields are invalid")
    if not isinstance(completion, dict) or set(completion) != COMPLETION_FIELDS:
        raise ValueError("locked collection completion marker fields are invalid")
    receipt, receipt_bytes = validate_external_freeze_receipt(
        receipt_path, expected_prelock_sha256
    )
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    evidence = _manifest_evidence(expected_prelock_sha256)
    freeze_time = _parse_utc_timestamp(
        receipt["timestamp_utc"], "external freeze timestamp"
    )
    attempt_time = _parse_utc_timestamp(
        attempt.get("started_at_utc"), "locked collection attempt timestamp"
    )
    completion_time = _parse_utc_timestamp(
        completion.get("completed_at_utc"), "locked collection completion timestamp"
    )
    if not freeze_time <= attempt_time <= completion_time:
        raise ValueError("locked collection chronology is invalid")
    expected_attempt = {
        "schema": ATTEMPT_SCHEMA,
        "stage": "locked_collection",
        "prelock_registry_sha256": expected_prelock_sha256,
        "prelock_registry_path": str(CANONICAL_PRELOCK_REGISTRY),
        "prelock_registry_file_sha256": sha256_file(CANONICAL_PRELOCK_REGISTRY),
        "prelock_artifact_root": str(CANONICAL_PRELOCK_BUNDLE),
        "prelock_digest_record_path": str(CANONICAL_PRELOCK_DIGEST),
        "output_root": str(CANONICAL_OUTPUT),
        "locked_raw_root": str(CANONICAL_LOCKED_RAW),
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "external_freeze_receipt_path": str(receipt_path),
        "external_freeze_receipt_sha256": receipt_sha256,
        "external_freeze_timestamp_utc": receipt["timestamp_utc"],
        "external_freeze_provider": receipt["provider"],
        "external_freeze_public_https_url": receipt["public_https_url"],
        "collector_command": _collector_command(expected_prelock_sha256),
    }
    for field, expected in expected_attempt.items():
        if attempt.get(field) != expected:
            raise ValueError(f"locked collection attempt {field} is invalid")
    if (
        isinstance(attempt.get("pid"), bool)
        or not isinstance(attempt.get("pid"), int)
        or attempt["pid"] <= 0
    ):
        raise ValueError("locked collection attempt pid is invalid")
    required_completion = {
        "schema": COMPLETION_SCHEMA,
        "stage": "locked_collection",
        "prelock_registry_sha256": expected_prelock_sha256,
        "prelock_registry_file_sha256": sha256_file(CANONICAL_PRELOCK_REGISTRY),
        "attempt_marker_sha256": sha256_file(attempt_path),
        "external_freeze_receipt_sha256": receipt_sha256,
        "collector_exit_code": 0,
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "locked_raw_root": str(CANONICAL_LOCKED_RAW),
        "locked_values_accessed_after_attempt": True,
        **evidence,
    }
    for field, expected in required_completion.items():
        if completion.get(field) != expected:
            raise ValueError(f"locked collection completion {field} is invalid")
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the digest-scoped, one-shot formal locked BOPTEST collection"
    )
    parser.add_argument("--expected-prelock-sha256", required=True)
    parser.add_argument("--external-freeze-receipt", type=Path, required=True)
    parser.add_argument("--confirm-locked-collection", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_locked_collection(
        args.expected_prelock_sha256,
        args.external_freeze_receipt,
        args.confirm_locked_collection,
    )
    print(
        json.dumps(
            {
                "locked_manifest": str(manifest),
                "prelock_registry_sha256": args.expected_prelock_sha256,
                "state_root": str(state_dir_for_digest(args.expected_prelock_sha256)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
