"""Write-once metadata closeout for the failed v5 evaluation invocation."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    write_json_once,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SERVICE_UNIT = "direct-h8-evaluation-v5.service"
FAILED_LOG = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_evaluation_v5.log"
)
FAILED_OUTPUT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_evaluation_v5"
)
FAILED_STAGING = (
    PROJECT_ROOT
    / "artifacts/.direct_h8_deterministic_transport_v3_evaluation_v5.staging"
)
DEFAULT_CLOSEOUT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_evaluation_v5_failed_attempt_closeout_v1"
    / "terminal_closeout.json"
)
SCHEMA = "direct-h8-evaluation-v5-terminal-metadata-closeout-v1"
EXPECTED_LOG_SHA256 = (
    "c36c15e7b1a03760625a59c657c06867130f5faa97faf5383ca3313159c33eed"
)
EXPECTED_READINESS_SHA256 = (
    "d245795503482417ac1d717782f33c56d05b0fa96d72f5156d5e954d4cdba74b"
)
ERROR_NEEDLES = (
    "evaluation_adapter.py",
    "runner.py",
    "terminal_v4_failure_binding",
    'ValueError: frozen evaluator requested a different readiness file',
)
SERVICE_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ExecStart",
)


def _service_metadata(unit: str = SERVICE_UNIT) -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            f"--property={','.join(SERVICE_FIELDS)}",
            "--no-pager",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in SERVICE_FIELDS:
            values[key] = value
    if set(values) != set(SERVICE_FIELDS):
        raise ValueError("failed v5 service metadata is incomplete")
    return values


def _validate_service_metadata(metadata: Mapping[str, str]) -> None:
    command = metadata.get("ExecStart", "")
    if (
        set(metadata) != set(SERVICE_FIELDS)
        or metadata.get("Id") != SERVICE_UNIT
        or metadata.get("LoadState") != "loaded"
        or metadata.get("ActiveState") != "failed"
        or metadata.get("SubState") != "failed"
        or metadata.get("Result") != "exit-code"
        or metadata.get("ExecMainCode") != "1"
        or metadata.get("ExecMainStatus") != "1"
        or not metadata.get("ExecMainStartTimestamp")
        or not metadata.get("ExecMainExitTimestamp")
        or (
            "building_fault_wm.transport_collection."
            "evaluation_adapter run-verify"
        )
        not in command
        or EXPECTED_READINESS_SHA256 not in command
        or "direct_h8_deterministic_transport_v3_evaluation_v5" not in command
    ):
        raise ValueError("failed v5 service metadata differs from the terminal run")


def _validate_failure_log(path: Path = FAILED_LOG) -> tuple[str, int]:
    digest = sha256_file(path)
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("failed v5 log is not UTF-8 metadata") from error
    if (
        digest != EXPECTED_LOG_SHA256
        or any(needle not in text for needle in ERROR_NEEDLES)
        or "load_transport_corpus_index" in text
        or "run_single" in text
    ):
        raise ValueError("failed v5 log is not the expected pre-evaluation traceback")
    return digest, len(content)


def build_closeout_payload(
    *,
    service_metadata: Mapping[str, str],
    log_path: Path = FAILED_LOG,
    output_path: Path = FAILED_OUTPUT,
    staging_path: Path = FAILED_STAGING,
) -> dict:
    _validate_service_metadata(service_metadata)
    log_sha256, log_bytes = _validate_failure_log(log_path)
    if os.path.lexists(output_path) or os.path.lexists(staging_path):
        raise ValueError("failed v5 invocation unexpectedly left evaluation output")
    payload = {
        "schema": SCHEMA,
        "terminal": True,
        "failure_stage": "metadata_validation_before_locked_corpus_loading",
        "service_unit": SERVICE_UNIT,
        "service_metadata": dict(service_metadata),
        "failure_log": {
            "path": log_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "bytes": log_bytes,
            "sha256": log_sha256,
        },
        "failed_output": {
            "path": output_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "exists": False,
        },
        "failed_staging": {
            "path": staging_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "exists": False,
        },
        "numerical_evaluation_started": False,
        "locked_trajectory_or_outcome_values_read_by_closeout": False,
        "retry_under_v5_adapter_permitted": False,
        "isolated_recovery_required": True,
    }
    return {**payload, "closeout_sha256": canonical_sha256(payload)}


def capture_terminal_closeout(
    output_path: Path = DEFAULT_CLOSEOUT,
) -> Path:
    payload = build_closeout_payload(service_metadata=_service_metadata())
    write_json_once(output_path, payload)
    validate_terminal_closeout(output_path)
    return output_path


def validate_terminal_closeout(
    path: Path = DEFAULT_CLOSEOUT,
    *,
    validate_live_filesystem: bool = True,
) -> dict:
    payload = strict_json(path)
    digest = payload.get("closeout_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "closeout_sha256"
    }
    expected_fields = {
        "schema",
        "terminal",
        "failure_stage",
        "service_unit",
        "service_metadata",
        "failure_log",
        "failed_output",
        "failed_staging",
        "numerical_evaluation_started",
        "locked_trajectory_or_outcome_values_read_by_closeout",
        "retry_under_v5_adapter_permitted",
        "isolated_recovery_required",
        "closeout_sha256",
    }
    service = payload.get("service_metadata")
    if not isinstance(service, dict):
        raise ValueError("failed v5 closeout service metadata is invalid")
    _validate_service_metadata(service)
    if (
        set(payload) != expected_fields
        or payload.get("schema") != SCHEMA
        or payload.get("terminal") is not True
        or payload.get("failure_stage")
        != "metadata_validation_before_locked_corpus_loading"
        or payload.get("service_unit") != SERVICE_UNIT
        or payload.get("numerical_evaluation_started") is not False
        or payload.get("locked_trajectory_or_outcome_values_read_by_closeout")
        is not False
        or payload.get("retry_under_v5_adapter_permitted") is not False
        or payload.get("isolated_recovery_required") is not True
        or digest != canonical_sha256(unsigned)
    ):
        raise ValueError("failed v5 closeout contract changed")
    if validate_live_filesystem:
        log_digest, log_bytes = _validate_failure_log()
        if payload.get("failure_log") != {
            "path": FAILED_LOG.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": log_bytes,
            "sha256": log_digest,
        }:
            raise ValueError("failed v5 closeout log binding changed")
        if (
            payload.get("failed_output")
            != {
                "path": FAILED_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
                "exists": False,
            }
            or payload.get("failed_staging")
            != {
                "path": FAILED_STAGING.relative_to(PROJECT_ROOT).as_posix(),
                "exists": False,
            }
            or os.path.lexists(FAILED_OUTPUT)
            or os.path.lexists(FAILED_STAGING)
        ):
            raise ValueError("failed v5 output-absence binding changed")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--no-live-filesystem", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "capture":
        result: object = str(capture_terminal_closeout(args.output.resolve()))
    else:
        result = validate_terminal_closeout(
            args.output.resolve(),
            validate_live_filesystem=not args.no_live_filesystem,
        )
    print(result)


if __name__ == "__main__":
    main()
