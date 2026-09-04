"""Prepare and verify the metadata-only v6 recovery prelock."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)

from . import adapter, closeout


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SCHEMA = "direct-h8-transport-evaluation-v6-recovery-prelock-v3"
REGISTRY_NAME = "recovery_prelock.json"
DIGEST_NAME = "recovery_prelock.canonical.sha256"
BUNDLE_NAME = "bundle"
CONTRACT_NAME = "recovery_contract.json"
CLOSEOUT_NAME = "failed_v5_terminal_closeout.json"
FAILURE_LOG_NAME = "failed_v5_evaluation.log"
REJECTED_V1_NAME = "rejected_v1_prelock.json"
REJECTION_CHAIN_NAME = "rejected_prelock_chain.json"
REJECTED_V1_PRELOCK = (
    PROJECT_ROOT / "artifacts/direct_h8_transport_evaluation_v6_prelock_v1"
)
REJECTED_V1_RECORD = (
    PROJECT_ROOT
    / "artifacts/direct_h8_transport_evaluation_v6_prelock_v1_rejection"
    / "terminal_rejection.json"
)
REJECTED_V2_PRELOCK = (
    PROJECT_ROOT / "artifacts/direct_h8_transport_evaluation_v6_prelock_v2"
)
REJECTION_CHAIN = (
    PROJECT_ROOT
    / "artifacts/direct_h8_transport_evaluation_v6_prelock_rejections_v1"
    / "rejection_chain.json"
)
REJECTED_V1_DIGEST = (
    "7c28c88852585d6ae99148075c4450c2acd816f762e8e0f78ad161527d9f6ed6"
)
REJECTED_V1_REGISTRY_SHA256 = (
    "8067382c37b6ad2f7bae7204e097dbe5d3cad8069cb6f24004dd199b841e03e4"
)
REJECTED_V1_DIGEST_FILE_SHA256 = (
    "617e94fe1eb1bddb8ed74119a8875fed01ce7de8e75de1f4debfc21cbc6d356c"
)
REJECTED_V2_DIGEST = (
    "12ea7e26dcce4c3048a48128f4189ec70898daef03233209d4563cee52e11a9d"
)
REJECTED_V2_REGISTRY_SHA256 = (
    "1cbae44c7256b7063087027c13b65a6dd2441634af7ac62c20beda4580af8260"
)
REJECTED_V2_DIGEST_FILE_SHA256 = (
    "1111e601cfecd529f23829c8e42e612f29c0c37e20830ccf0770919df0fb4db7"
)
REJECTED_V1_RECORD_SHA256 = (
    "04a5d6ea0f474b338ac936f583ca086a2e8ea2d3490b701ecf5ac4be01d28bf7"
)
DEFAULT_OUTPUT = adapter.DEFAULT_RECOVERY_PRELOCK
REQUIRED_SOURCE_FILES = {
    "__init__.py",
    "adapter.py",
    "attempt.py",
    "closeout.py",
    "freeze.py",
    "public_freeze.py",
    "PROTOCOL.md",
    "test_adapter.py",
    "test_attempt.py",
    "test_freeze.py",
}


def _source_paths() -> dict[str, Path]:
    paths = {
        path.name: path
        for path in HERE.iterdir()
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    if set(paths) != REQUIRED_SOURCE_FILES:
        raise ValueError("v6 recovery source inventory changed")
    return paths


def validate_rejected_v1_record(
    path: Path = REJECTED_V1_RECORD,
    *,
    validate_live_version_specific_absence: bool = True,
) -> dict:
    payload = strict_json(path)
    digest = payload.get("rejection_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "rejection_sha256"
    }
    expected_fields = {
        "schema",
        "rejected_prelock_canonical_sha256",
        "rejected_prelock_registry_file_sha256",
        "rejected_prelock_digest_file_sha256",
        "rejected_before_publication",
        "public_freeze_exists",
        "attempt_or_terminal_marker_exists",
        "evaluation_output_or_staging_exists",
        "scientific_values_parsed",
        "legacy_v4_csv_bytes_hashed",
        "reason",
        "publication_or_retry_under_v1_permitted",
        "replacement_prelock_path",
        "replacement_public_freeze_receipt_path",
        "replacement_identity_binding",
        "rejection_sha256",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema")
        != (
            "direct-h8-transport-evaluation-v6-prelock-v1-"
            "prepublication-rejection-v1"
        )
        or payload.get("rejected_prelock_canonical_sha256")
        != REJECTED_V1_DIGEST
        or payload.get("rejected_prelock_registry_file_sha256")
        != REJECTED_V1_REGISTRY_SHA256
        or payload.get("rejected_prelock_digest_file_sha256")
        != REJECTED_V1_DIGEST_FILE_SHA256
        or payload.get("rejected_before_publication") is not True
        or payload.get("public_freeze_exists") is not False
        or payload.get("attempt_or_terminal_marker_exists") is not False
        or payload.get("evaluation_output_or_staging_exists") is not False
        or payload.get("scientific_values_parsed") is not False
        or payload.get("legacy_v4_csv_bytes_hashed") is not True
        or payload.get("publication_or_retry_under_v1_permitted") is not False
        or payload.get("replacement_prelock_path")
        != "artifacts/direct_h8_transport_evaluation_v6_prelock_v2"
        or payload.get("replacement_public_freeze_receipt_path")
        != (
            "artifacts/direct_h8_transport_evaluation_v6_external_freeze_v2/"
            "external_freeze_receipt.json"
        )
        or digest != canonical_sha256(unsigned)
    ):
        raise ValueError("rejected v1 prelock record changed")
    if (
        sha256_file(REJECTED_V1_PRELOCK / REGISTRY_NAME)
        != REJECTED_V1_REGISTRY_SHA256
        or sha256_file(REJECTED_V1_PRELOCK / DIGEST_NAME)
        != REJECTED_V1_DIGEST_FILE_SHA256
        or (
            REJECTED_V1_PRELOCK / DIGEST_NAME
        ).read_text(encoding="ascii")
        != f"{REJECTED_V1_DIGEST}\n"
    ):
        raise ValueError("rejected v1 prelock identity changed")
    if validate_live_version_specific_absence:
        v1_receipt = (
            PROJECT_ROOT
            / "artifacts/direct_h8_transport_evaluation_v6_external_freeze_v1"
            / "external_freeze_receipt.json"
        )
        v1_state = (
            adapter.DEFAULT_STATE_BASE / REJECTED_V1_DIGEST
        )
        if any(
            os.path.lexists(candidate)
            for candidate in (v1_receipt, v1_state)
        ):
            raise ValueError("rejected v1 unexpectedly has a public or state artifact")
    return payload


def validate_rejection_chain(
    path: Path = REJECTION_CHAIN,
    *,
    validate_live_version_specific_absence: bool = True,
) -> dict:
    payload = strict_json(path)
    digest = payload.get("chain_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "chain_sha256"
    }
    rows = payload.get("rejected_prelocks")
    if (
        set(payload)
        != {
            "schema",
            "prior_v1_rejection_record_file_sha256",
            "rejected_prelocks",
            "replacement_prelock_path",
            "replacement_public_freeze_receipt_path",
            "replacement_identity_binding",
            "chain_sha256",
        }
        or payload.get("schema")
        != "direct-h8-transport-evaluation-v6-prepublication-rejection-chain-v1"
        or payload.get("prior_v1_rejection_record_file_sha256")
        != REJECTED_V1_RECORD_SHA256
        or not isinstance(rows, list)
        or len(rows) != 2
        or digest != canonical_sha256(unsigned)
        or payload.get("replacement_prelock_path")
        != "artifacts/direct_h8_transport_evaluation_v6_prelock_v3"
        or payload.get("replacement_public_freeze_receipt_path")
        != (
            "artifacts/direct_h8_transport_evaluation_v6_external_freeze_v3/"
            "external_freeze_receipt.json"
        )
    ):
        raise ValueError("rejected prelock chain changed")
    expected = {
        "v1": {
            "prelock": REJECTED_V1_PRELOCK,
            "digest": REJECTED_V1_DIGEST,
            "registry_sha256": REJECTED_V1_REGISTRY_SHA256,
            "digest_file_sha256": REJECTED_V1_DIGEST_FILE_SHA256,
            "receipt": (
                PROJECT_ROOT
                / "artifacts/direct_h8_transport_evaluation_v6_external_freeze_v1"
                / "external_freeze_receipt.json"
            ),
        },
        "v2": {
            "prelock": REJECTED_V2_PRELOCK,
            "digest": REJECTED_V2_DIGEST,
            "registry_sha256": REJECTED_V2_REGISTRY_SHA256,
            "digest_file_sha256": REJECTED_V2_DIGEST_FILE_SHA256,
            "receipt": (
                PROJECT_ROOT
                / "artifacts/direct_h8_transport_evaluation_v6_external_freeze_v2"
                / "external_freeze_receipt.json"
            ),
        },
    }
    row_fields = {
        "version",
        "prelock_path",
        "canonical_sha256",
        "registry_file_sha256",
        "digest_file_sha256",
        "public_freeze_receipt_path",
        "rejected_before_publication",
        "public_freeze_exists",
        "attempt_or_terminal_marker_exists",
        "evaluation_output_or_staging_existed_at_rejection",
        "scientific_values_parsed",
        "legacy_v4_csv_bytes_hashed",
        "reason",
        "publication_or_retry_permitted",
    }
    by_version = {
        row.get("version"): row for row in rows if isinstance(row, dict)
    }
    if set(by_version) != set(expected):
        raise ValueError("rejected prelock versions changed")
    for version, identity in expected.items():
        row = by_version[version]
        prelock = identity["prelock"]
        if (
            set(row) != row_fields
            or row.get("prelock_path")
            != prelock.relative_to(PROJECT_ROOT).as_posix()
            or row.get("canonical_sha256") != identity["digest"]
            or row.get("registry_file_sha256")
            != identity["registry_sha256"]
            or row.get("digest_file_sha256")
            != identity["digest_file_sha256"]
            or row.get("public_freeze_receipt_path")
            != identity["receipt"].relative_to(PROJECT_ROOT).as_posix()
            or row.get("rejected_before_publication") is not True
            or row.get("public_freeze_exists") is not False
            or row.get("attempt_or_terminal_marker_exists") is not False
            or row.get("evaluation_output_or_staging_existed_at_rejection")
            is not False
            or row.get("scientific_values_parsed") is not False
            or row.get("legacy_v4_csv_bytes_hashed") is not True
            or row.get("publication_or_retry_permitted") is not False
            or sha256_file(prelock / REGISTRY_NAME)
            != identity["registry_sha256"]
            or sha256_file(prelock / DIGEST_NAME)
            != identity["digest_file_sha256"]
            or (prelock / DIGEST_NAME).read_text(encoding="ascii")
            != f"{identity['digest']}\n"
        ):
            raise ValueError(f"rejected {version} prelock identity changed")
        state = adapter.DEFAULT_STATE_BASE / str(identity["digest"])
        if validate_live_version_specific_absence and (
            os.path.lexists(identity["receipt"]) or os.path.lexists(state)
        ):
            raise ValueError(
                f"rejected {version} unexpectedly has a public or state artifact"
            )
    if sha256_file(REJECTED_V1_RECORD) != REJECTED_V1_RECORD_SHA256:
        raise ValueError("prior v1 rejection record changed")
    return payload


def recovery_contract(closeout_path: Path) -> dict:
    closeout_payload = closeout.validate_terminal_closeout(closeout_path)
    rejected_v1 = validate_rejected_v1_record()
    rejection_chain = validate_rejection_chain()
    hashes = adapter.upstream_hashes()
    if hashes != adapter.EXPECTED_UPSTREAM_SHA256:
        raise ValueError("v6 recovery upstream input bytes changed")
    upstream = adapter.verify_upstream_inputs(live_external_freeze=False)
    terminal = adapter.verify_v4_terminal_audit_dispatch()
    boundary = adapter.verify_v5_metadata_boundary()
    return {
        "schema": "direct-h8-transport-evaluation-v6-recovery-contract-v3",
        "trajectory_or_result_values_parsed_while_preparing": False,
        "legacy_v4_trajectory_bytes_hashed_while_preparing": True,
        "v5_locked_trajectory_files_opened_while_preparing": False,
        "evaluation_result_files_opened_while_preparing": False,
        "repair_scope": "metadata_dispatch_only",
        "exact_v5_identity": {
            "prelock_registry_sha256": adapter.EXPECTED_PRELOCK_SHA256,
            "readiness_sha256": adapter.EXPECTED_READINESS_SHA256,
            "prelock_root": str(adapter.v5_runner.PRELOCK_ROOT.resolve()),
            "data_root": str(adapter.v5_runner.DATA_ROOT.resolve()),
            "state_root": str(adapter.v5_runner.STATE_ROOT.resolve()),
            "readiness_path": str(adapter.v5_runner.READINESS_PATH.resolve()),
            "external_freeze_receipt_path": str(
                adapter.v5_runner.EXTERNAL_FREEZE_RECEIPT.resolve()
            ),
        },
        "upstream_file_sha256": hashes,
        "failed_v5_closeout_payload_sha256": closeout_payload[
            "closeout_sha256"
        ],
        "failed_v5_closeout_file_sha256": sha256_file(closeout_path),
        "failed_v5_log_sha256": closeout_payload["failure_log"]["sha256"],
        "rejected_v1_prelock": {
            "record_file_sha256": sha256_file(REJECTED_V1_RECORD),
            "record_payload_sha256": rejected_v1["rejection_sha256"],
            "prelock_canonical_sha256": REJECTED_V1_DIGEST,
            "registry_file_sha256": REJECTED_V1_REGISTRY_SHA256,
            "publication_or_retry_permitted": False,
        },
        "rejected_prelock_chain": {
            "record_file_sha256": sha256_file(REJECTION_CHAIN),
            "record_payload_sha256": rejection_chain["chain_sha256"],
            "rejected_canonical_sha256": {
                row["version"]: row["canonical_sha256"]
                for row in rejection_chain["rejected_prelocks"]
            },
            "publication_or_retry_permitted": False,
        },
        "metadata_preflight": {
            "terminal_v4_binding_sha256": terminal[
                "terminal_v4_failure"
            ]["binding_sha256"],
            "terminal_v4_proxy_call_count": terminal[
                "terminal_v4_proxy_call_count"
            ],
            "v5_boundary": boundary,
            "v5_external_freeze_schema": upstream[
                "v5_external_freeze_schema"
            ],
            "identity_guard_schema": upstream["identity_guard_schema"],
            "trajectory_loader_called": False,
            "legacy_v4_raw_bytes_hashed_by_frozen_terminal_audit": True,
            "trajectory_values_parsed": False,
            "evaluation_result_csv_opened": False,
        },
        "fresh_output_namespace": str(adapter.DEFAULT_OUTPUT.resolve()),
        "one_shot_state_base": str(adapter.DEFAULT_STATE_BASE.resolve()),
        "hook_policy": {
            "temporary_alias": "runner.frozen_external_freeze",
            "exact_v5_tuple_routes_to": "published v5 adapter",
            "exact_terminal_v4_tuple_routes_to": "original frozen validator",
            "mixed_or_third_identity": "rejected",
            "terminal_v4_audit_must_delegate": True,
        },
        "unchanged": [
            "trajectory corpus",
            "checkpoints",
            "model architecture",
            "training",
            "inference",
            "metrics",
            "bootstrap",
            "gates",
            "output schemas",
        ],
        "public_freeze_required_before_evaluation": True,
        "one_attempt_per_recovery_digest": True,
        "trusted_provider_time_required_before_attempt": True,
    }


def prepare_local_prelock(
    output_root: Path = DEFAULT_OUTPUT,
    closeout_path: Path = closeout.DEFAULT_CLOSEOUT,
) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite v6 prelock: {output_root}")
    sources = _source_paths()
    contract = recovery_contract(closeout_path)
    bundle = output_root / BUNDLE_NAME
    bundle.mkdir(parents=True, exist_ok=False)
    for name, source in sorted(sources.items()):
        write_once(bundle / "source" / name, source.read_bytes())
    for name, source in sorted(adapter.upstream_paths().items()):
        write_once(bundle / "upstream" / name, source.read_bytes())
    write_once(bundle / CLOSEOUT_NAME, closeout_path.read_bytes())
    write_once(bundle / FAILURE_LOG_NAME, closeout.FAILED_LOG.read_bytes())
    write_once(bundle / REJECTED_V1_NAME, REJECTED_V1_RECORD.read_bytes())
    write_once(bundle / REJECTION_CHAIN_NAME, REJECTION_CHAIN.read_bytes())
    contract_path = write_json_once(bundle / CONTRACT_NAME, contract)
    inventory = tree_inventory(bundle)
    registry = {
        "schema": SCHEMA,
        "trajectory_or_result_values_parsed_while_preparing": False,
        "legacy_v4_trajectory_bytes_hashed_while_preparing": True,
        "v5_locked_trajectory_files_opened_while_preparing": False,
        "evaluation_result_files_opened_while_preparing": False,
        "contract_file_sha256": sha256_file(contract_path),
        "contract_payload_sha256": canonical_sha256(contract),
        "source_file_sha256_by_name": {
            name: sha256_file(path) for name, path in sorted(sources.items())
        },
        "upstream_file_sha256_by_name": adapter.upstream_hashes(),
        "bundle_inventory": inventory,
        "bundle_inventory_sha256": canonical_sha256(inventory),
    }
    write_json_once(output_root / REGISTRY_NAME, registry)
    write_once(
        output_root / DIGEST_NAME,
        f"{canonical_sha256(registry)}\n".encode("ascii"),
    )
    verify_local_prelock(output_root, closeout_path=closeout_path)
    return output_root


def verify_local_prelock(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    closeout_path: Path = closeout.DEFAULT_CLOSEOUT,
) -> dict:
    registry = strict_json(output_root / REGISTRY_NAME)
    if (
        set(registry)
        != {
            "schema",
            "trajectory_or_result_values_parsed_while_preparing",
            "legacy_v4_trajectory_bytes_hashed_while_preparing",
            "v5_locked_trajectory_files_opened_while_preparing",
            "evaluation_result_files_opened_while_preparing",
            "contract_file_sha256",
            "contract_payload_sha256",
            "source_file_sha256_by_name",
            "upstream_file_sha256_by_name",
            "bundle_inventory",
            "bundle_inventory_sha256",
        }
        or registry.get("schema") != SCHEMA
        or registry.get("trajectory_or_result_values_parsed_while_preparing")
        is not False
        or registry.get("legacy_v4_trajectory_bytes_hashed_while_preparing")
        is not True
        or registry.get("v5_locked_trajectory_files_opened_while_preparing")
        is not False
        or registry.get("evaluation_result_files_opened_while_preparing")
        is not False
        or (output_root / DIGEST_NAME).read_text(encoding="ascii")
        != f"{canonical_sha256(registry)}\n"
    ):
        raise ValueError("v6 recovery prelock identity changed")
    bundle = output_root / BUNDLE_NAME
    inventory = tree_inventory(bundle)
    if (
        registry.get("bundle_inventory") != inventory
        or registry.get("bundle_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise ValueError("v6 recovery prelock inventory changed")
    contract_path = bundle / CONTRACT_NAME
    contract = strict_json(contract_path)
    if (
        registry.get("contract_file_sha256") != sha256_file(contract_path)
        or registry.get("contract_payload_sha256")
        != canonical_sha256(contract)
        or contract != recovery_contract(closeout_path)
    ):
        raise ValueError("v6 recovery contract changed")
    source_hashes = {
        name: sha256_file(path)
        for name, path in sorted(_source_paths().items())
    }
    if registry.get("source_file_sha256_by_name") != source_hashes:
        raise ValueError("live v6 recovery source changed")
    for name, digest in source_hashes.items():
        if sha256_file(bundle / "source" / name) != digest:
            raise ValueError(f"v6 recovery source snapshot changed: {name}")
    upstream_hashes = adapter.upstream_hashes()
    if (
        upstream_hashes != adapter.EXPECTED_UPSTREAM_SHA256
        or registry.get("upstream_file_sha256_by_name") != upstream_hashes
    ):
        raise ValueError("live v6 upstream source or metadata changed")
    for name, digest in upstream_hashes.items():
        if sha256_file(bundle / "upstream" / name) != digest:
            raise ValueError(f"v6 upstream snapshot changed: {name}")
    if (
        sha256_file(bundle / CLOSEOUT_NAME) != sha256_file(closeout_path)
        or sha256_file(bundle / FAILURE_LOG_NAME)
        != closeout.EXPECTED_LOG_SHA256
        or sha256_file(bundle / REJECTED_V1_NAME)
        != sha256_file(REJECTED_V1_RECORD)
        or sha256_file(bundle / REJECTION_CHAIN_NAME)
        != sha256_file(REJECTION_CHAIN)
    ):
        raise ValueError("v6 failed-attempt evidence snapshot changed")
    return registry


def public_freeze_input_paths(
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Path]:
    verify_local_prelock(output_root)
    bundle = output_root / BUNDLE_NAME
    result = {
        REGISTRY_NAME: output_root / REGISTRY_NAME,
        DIGEST_NAME: output_root / DIGEST_NAME,
        CONTRACT_NAME: bundle / CONTRACT_NAME,
        CLOSEOUT_NAME: bundle / CLOSEOUT_NAME,
        FAILURE_LOG_NAME: bundle / FAILURE_LOG_NAME,
        REJECTED_V1_NAME: bundle / REJECTED_V1_NAME,
        REJECTION_CHAIN_NAME: bundle / REJECTION_CHAIN_NAME,
    }
    result.update(
        {
            f"source__{name}": bundle / "source" / name
            for name in sorted(REQUIRED_SOURCE_FILES)
        }
    )
    result.update(
        {
            f"upstream__{name}": bundle / "upstream" / name
            for name in sorted(adapter.EXPECTED_UPSTREAM_SHA256)
        }
    )
    if any(path.suffix.lower() == ".csv" for path in result.values()):
        raise AssertionError("v6 public freeze contains trajectory or result CSV")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--failed-v5-closeout", type=Path, default=closeout.DEFAULT_CLOSEOUT
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "prepare":
        result: object = str(
            prepare_local_prelock(
                args.output.resolve(), args.failed_v5_closeout.resolve()
            )
        )
    else:
        result = verify_local_prelock(
            args.output.resolve(),
            closeout_path=args.failed_v5_closeout.resolve(),
        )
    print(result)


if __name__ == "__main__":
    main()
