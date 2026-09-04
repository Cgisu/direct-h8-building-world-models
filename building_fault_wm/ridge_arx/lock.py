"""Metadata-only pre-lock for the schedule-matched ARX addendum.

The lock is prepared only after the ARX training grid and the ownership-corrected
v5 collection are complete. It validates collection evidence without calling the
numeric trajectory validator or opening a trajectory CSV.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping

from building_fault_wm.deterministic_transport import (
    collect as v3_collect,
    corpus as v3_corpus,
    plan as v3_plan,
)
from building_fault_wm.transport_collection import (
    external_freeze as transport_external_freeze,
    runner as transport_runner,
)
from building_fault_wm.neural_benchmark import protocol as boptest

from .config import CASES, FROZEN_CONFIG, PARENT_PACKAGE_DIGEST
from .io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)
from .train import (
    DEVELOPMENT_MANIFEST,
    FAULT_CONTRACT,
    SCALER_ROOT,
    SCHEDULE_ROOT,
    source_manifest,
    verify_parent_package,
    verify_training_grid,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_TRAINING_ROOT = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_training_v2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_prelock_v2"
)

PRELOCK_SCHEMA = "schedule-matched-recursive-ridge-arx-prelock-v1"
TRANSPORT_BINDING_SCHEMA = "schedule-matched-arx-v5-collection-binding-v1"
TRAINING_BINDING_SCHEMA = "schedule-matched-arx-training-binding-v1"
PARENT_BINDING_SCHEMA = "schedule-matched-arx-parent-assets-binding-v1"
SOURCE_SNAPSHOT_SCHEMA = "schedule-matched-arx-source-snapshot-v1"
REGISTRY_NAME = "addendum_prelock.json"
DIGEST_NAME = "addendum_prelock.canonical.sha256"
BUNDLE_NAME = "bundle"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _source_paths() -> dict[str, Path]:
    result = {
        path.name: path
        for path in sorted(HERE.iterdir())
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    required = {
        "__init__.py",
        "config.py",
        "io.py",
        "train.py",
        "lock.py",
        "external_freeze.py",
        "evaluate.py",
        "cli.py",
        "__main__.py",
        "PROTOCOL.md",
        "test_train.py",
        "test_lock.py",
        "test_evaluate.py",
        "test_cli.py",
    }
    if set(result) != required:
        raise ValueError(
            "addendum source set differs from its fixed pre-lock inventory"
        )
    return result


def complete_source_manifest() -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in sorted(_source_paths().items())
    }


def parent_asset_binding() -> dict:
    parent = verify_parent_package()
    schedules = {
        f"{case}/seed{seed}.json": sha256_file(
            SCHEDULE_ROOT / case / f"seed{seed}.json"
        )
        for case in CASES
        for seed in FROZEN_CONFIG.model_seeds
    }
    scalers = {
        f"{case}.json": sha256_file(SCALER_ROOT / f"{case}.json")
        for case in CASES
    }
    if len(schedules) != 15 or len(scalers) != 3:
        raise AssertionError("parent schedule/scaler grid is incomplete")
    return {
        "schema": PARENT_BINDING_SCHEMA,
        "parent_package": parent,
        "parent_package_digest": PARENT_PACKAGE_DIGEST,
        "development_manifest_file_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "fault_contract_file_sha256": sha256_file(FAULT_CONTRACT),
        "schedule_file_sha256_by_path": schedules,
        "fit_scaler_file_sha256_by_path": scalers,
    }


def _manifest_metadata(
    manifest_path: Path,
    completion: Mapping[str, object],
) -> dict:
    wrapper = strict_json(manifest_path)
    if set(wrapper) != {"manifest_sha256", "manifest"}:
        raise ValueError("v3 corpus manifest wrapper fields changed")
    manifest = wrapper.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != v3_corpus.MANIFEST_FIELDS:
        raise ValueError("v3 corpus manifest metadata fields changed")
    payload_sha256 = v3_plan.canonical_sha256(manifest)
    if (
        wrapper.get("manifest_sha256") != payload_sha256
        or completion.get("manifest_payload_sha256") != payload_sha256
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("v3 completion does not bind the corpus manifest")
    expected_counts = {
        "cases": len(v3_corpus.EXPECTED_CASES),
        "windows": v3_corpus.EXPECTED_WINDOWS,
        "branches": v3_corpus.EXPECTED_BRANCHES,
        "rows_per_branch": boptest.TRAJECTORY_STEPS,
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("v3 corpus manifest count grid changed")
    files = manifest.get("files")
    receipts = manifest.get("worker_receipts")
    if (
        not isinstance(files, list)
        or len(files) != v3_corpus.EXPECTED_BRANCHES
        or not isinstance(receipts, list)
    ):
        raise ValueError("v3 corpus manifest metadata grid is incomplete")
    return {
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": payload_sha256,
        "manifest_schema": manifest["schema"],
        "study_kind": manifest["study_kind"],
        "collection_kind": manifest["collection_kind"],
        "output_role": manifest["output_role"],
        "counts": manifest["counts"],
        "plan_sha256_by_case": manifest["plan_sha256_by_case"],
        "certificate_sha256": manifest["certificate_sha256"],
        "readiness_sha256": manifest["readiness_sha256"],
        "collection_code_sha256": manifest["collection_code_sha256"],
        "source_sha256_by_case": manifest["source_sha256_by_case"],
        "file_metadata_sha256": canonical_sha256(files),
        "worker_receipt_metadata_sha256": canonical_sha256(receipts),
    }


def bind_completed_transport_metadata(
    *,
    transport_prelock_root: Path,
    transport_live_data_root: Path,
    transport_readiness_path: Path,
    transport_external_freeze_receipt_path: Path,
    transport_state_root: Path,
    transport_manifest_path: Path,
    live_transport_external_freeze: bool = False,
) -> dict:
    """Bind completed v5 metadata without parsing a locked response value."""

    transport_runner.validate_namespace_separation(
        data_root=transport_live_data_root,
        state_root=transport_state_root,
        freeze_root=transport_readiness_path.parent,
    )
    readiness_document = strict_json(transport_readiness_path)
    expected_prelock = readiness_document.get("prelock_registry_sha256")
    expected_readiness = readiness_document.get("readiness_sha256")
    if (
        readiness_document.get("schema") != transport_runner.READINESS_SCHEMA
        or expected_prelock != transport_runner.ORIGINAL_PRELOCK_SHA256
        or not _valid_sha256(expected_readiness)
        or expected_readiness == transport_runner.TERMINAL_V4_READINESS_SHA256
        or readiness_document.get("namespaces")
        != {"data": "data", "state": "state_v3", "freeze": "freeze_v5"}
        or readiness_document.get("full_recollection_required") is not True
        or readiness_document.get("data_v7_raw_reuse_permitted") is not False
    ):
        raise ValueError("transport readiness is not the canonical v5 replacement")
    _, readiness = transport_runner.load_bound_readiness(
        prelock_root=transport_prelock_root,
        live_data_root=transport_live_data_root,
        readiness_path=transport_readiness_path,
        expected_prelock_sha256=str(expected_prelock),
        expected_readiness_sha256=str(expected_readiness),
    )
    freeze = transport_external_freeze.validate_external_freeze_receipt(
        transport_external_freeze_receipt_path,
        str(expected_prelock),
        str(expected_readiness),
        prelock_root=transport_prelock_root,
        readiness_path=transport_readiness_path,
        live=live_transport_external_freeze,
    )
    attempt, completion, attempt_sha256, completion_sha256 = (
        v3_corpus.validate_collection_completion(
            state_root=transport_state_root,
            readiness=readiness,
            manifest_path=transport_manifest_path,
            expected_prelock_sha256=str(expected_prelock),
            external_freeze=freeze,
            external_freeze_receipt_path=(
                transport_external_freeze_receipt_path
            ),
        )
    )
    metadata = _manifest_metadata(transport_manifest_path, completion)
    if attempt.get("locked_response_values_accessed") is not False:
        raise ValueError("v3 attempt did not precede locked-value access")
    if completion.get("locked_response_values_accessed_after_attempt") is not True:
        raise ValueError("v3 completion does not acknowledge value access")
    paths = {
        "prelock_root": str(transport_prelock_root.resolve()),
        "live_data_root": str(transport_live_data_root.resolve()),
        "readiness_path": str(transport_readiness_path.resolve()),
        "external_freeze_receipt_path": str(
            transport_external_freeze_receipt_path.resolve()
        ),
        "state_root": str(transport_state_root.resolve()),
        "manifest_path": str(transport_manifest_path.resolve()),
    }
    file_hashes = {
        "readiness": sha256_file(transport_readiness_path),
        "external_freeze_receipt": sha256_file(
            transport_external_freeze_receipt_path
        ),
        "attempt": attempt_sha256,
        "completion": completion_sha256,
        "manifest": sha256_file(transport_manifest_path),
    }
    return {
        "schema": TRANSPORT_BINDING_SCHEMA,
        "validation_scope": (
            "metadata_only_no_trajectory_csv_or_response_value_opened"
        ),
        "metadata_adapter": {
            "schema": transport_runner.READINESS_SCHEMA,
            "replacement_kind": readiness_document["replacement_kind"],
            "operational_code_sha256": readiness_document[
                "operational_code_sha256"
            ],
            "frozen_evaluation_source_sha256": readiness_document[
                "frozen_evaluation_source_sha256"
            ],
            "data_namespace": "data",
            "terminal_data_v7_rejected": True,
        },
        "paths": paths,
        "file_sha256_by_role": file_hashes,
        "transport_prelock_registry_sha256": expected_prelock,
        "transport_readiness_sha256": expected_readiness,
        "transport_certificate_sha256": readiness.expected_certificate_sha256,
        "transport_collection_code_sha256": readiness.collection_code_sha256,
        "transport_external_freeze": {
            "provider": freeze["provider"],
            "gist_id": freeze["gist_id"],
            "revision": freeze["revision"],
            "revision_committed_at_utc": freeze[
                "revision_committed_at_utc"
            ],
        },
        "attempt_started_at_utc": attempt["started_at_utc"],
        "completion_completed_at_utc": completion["completed_at_utc"],
        "manifest_metadata": metadata,
    }


def _training_binding(training_root: Path) -> dict:
    verified = verify_training_grid(training_root)
    return {
        "schema": TRAINING_BINDING_SCHEMA,
        "training_root": str(training_root.resolve()),
        "verification": verified,
        "training_source_manifest": source_manifest(),
    }


def prepare_prelock(
    *,
    output_root: Path = DEFAULT_OUTPUT,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    transport_prelock_root: Path,
    transport_live_data_root: Path,
    transport_readiness_path: Path,
    transport_external_freeze_receipt_path: Path,
    transport_state_root: Path,
    transport_manifest_path: Path,
    live_transport_external_freeze: bool = False,
) -> Path:
    """Create the addendum pre-lock after canonical v5 collection completion."""

    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite addendum pre-lock: {output_root}")
    source_hashes = complete_source_manifest()
    training = _training_binding(training_root)
    parent = parent_asset_binding()
    transport = bind_completed_transport_metadata(
        transport_prelock_root=transport_prelock_root,
        transport_live_data_root=transport_live_data_root,
        transport_readiness_path=transport_readiness_path,
        transport_external_freeze_receipt_path=(
            transport_external_freeze_receipt_path
        ),
        transport_state_root=transport_state_root,
        transport_manifest_path=transport_manifest_path,
        live_transport_external_freeze=live_transport_external_freeze,
    )

    bundle = output_root / BUNDLE_NAME
    bundle.mkdir(parents=True, exist_ok=False)
    write_json_once(
        bundle / "source_snapshot.json",
        {"schema": SOURCE_SNAPSHOT_SCHEMA, "file_sha256_by_name": source_hashes},
    )
    for name, source in sorted(_source_paths().items()):
        write_once(bundle / "source" / name, source.read_bytes())
    write_json_once(bundle / "training_binding.json", training)
    write_json_once(bundle / "parent_assets_binding.json", parent)
    write_json_once(bundle / "transport_collection_binding.json", transport)
    inventory = tree_inventory(bundle)
    registry = {
        "schema": PRELOCK_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "locked_response_values_accessed_while_preparing": False,
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
        "source_file_sha256_by_name": source_hashes,
        "training_binding_file_sha256": sha256_file(
            bundle / "training_binding.json"
        ),
        "parent_assets_binding_file_sha256": sha256_file(
            bundle / "parent_assets_binding.json"
        ),
        "transport_collection_binding_file_sha256": sha256_file(
            bundle / "transport_collection_binding.json"
        ),
        "bundle_inventory": inventory,
        "bundle_inventory_sha256": canonical_sha256(inventory),
    }
    write_json_once(output_root / REGISTRY_NAME, registry)
    digest = canonical_sha256(registry)
    write_once(output_root / DIGEST_NAME, f"{digest}\n".encode("ascii"))
    verify_prelock(output_root, verify_live_assets=True)
    return output_root


def _verify_inventory_files(root: Path, inventory: object, label: str) -> None:
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"{label} inventory is empty")
    for row in inventory:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError(f"{label} inventory row is invalid")
        relative = row["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"{label} inventory path is invalid")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"{label} inventory path escapes its root") from error
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"{label} inventory file changed: {relative}")


def verify_prelock(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    verify_live_assets: bool = True,
) -> dict:
    registry = strict_json(output_root / REGISTRY_NAME)
    if (
        registry.get("schema") != PRELOCK_SCHEMA
        or registry.get("secondary_only") is not True
        or registry.get("cannot_modify_v2_or_v3_gate") is not True
        or registry.get("locked_response_values_accessed_while_preparing")
        is not False
        or registry.get("config")
        != json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
    ):
        raise ValueError("addendum pre-lock identity changed")
    record = (output_root / DIGEST_NAME).read_text(encoding="ascii")
    expected = canonical_sha256(registry)
    if record != f"{expected}\n":
        raise ValueError("addendum pre-lock digest record changed")
    bundle = output_root / BUNDLE_NAME
    inventory = registry.get("bundle_inventory")
    if registry.get("bundle_inventory_sha256") != canonical_sha256(inventory):
        raise ValueError("addendum bundle inventory digest changed")
    _verify_inventory_files(bundle, inventory, "addendum bundle")
    if tree_inventory(bundle) != inventory:
        raise ValueError("addendum bundle contains unregistered files")

    snapshot = strict_json(bundle / "source_snapshot.json")
    source_hashes = snapshot.get("file_sha256_by_name")
    if (
        snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or source_hashes != registry.get("source_file_sha256_by_name")
    ):
        raise ValueError("addendum source snapshot changed")
    for name, digest in source_hashes.items():
        if sha256_file(bundle / "source" / name) != digest:
            raise ValueError(f"addendum source snapshot changed: {name}")

    if verify_live_assets:
        if complete_source_manifest() != source_hashes:
            raise ValueError("live addendum source differs from the pre-lock")
        training = strict_json(bundle / "training_binding.json")
        if training.get("schema") != TRAINING_BINDING_SCHEMA:
            raise ValueError("addendum training binding schema changed")
        live_training = _training_binding(Path(training["training_root"]))
        if live_training != training:
            raise ValueError("live ARX training grid differs from the pre-lock")
        if strict_json(bundle / "parent_assets_binding.json") != (
            parent_asset_binding()
        ):
            raise ValueError("live parent assets differ from the pre-lock")
        transport = strict_json(bundle / "transport_collection_binding.json")
        for role, digest in transport["file_sha256_by_role"].items():
            if role in {"attempt", "completion"}:
                readiness = str(transport["transport_readiness_sha256"])
                filename = (
                    v3_collect.ATTEMPT_MARKER
                    if role == "attempt"
                    else v3_collect.COMPLETION_MARKER
                )
                path = (
                    Path(transport["paths"]["state_root"])
                    / readiness
                    / filename
                )
            elif role == "readiness":
                path = Path(transport["paths"]["readiness_path"])
            elif role == "external_freeze_receipt":
                path = Path(
                    transport["paths"]["external_freeze_receipt_path"]
                )
            else:
                path = Path(transport["paths"]["manifest_path"])
            if sha256_file(path) != digest:
                raise ValueError(
                    f"live transport metadata changed after pre-lock: {role}"
                )
    return registry


def external_freeze_file_paths(
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Path]:
    verify_prelock(output_root, verify_live_assets=True)
    bundle = output_root / BUNDLE_NAME
    result = {
        "addendum_prelock.json": output_root / REGISTRY_NAME,
        "addendum_prelock.canonical.sha256": output_root / DIGEST_NAME,
        "source_snapshot.json": bundle / "source_snapshot.json",
        "training_binding.json": bundle / "training_binding.json",
        "parent_assets_binding.json": bundle / "parent_assets_binding.json",
        "transport_collection_binding.json": (
            bundle / "transport_collection_binding.json"
        ),
    }
    result.update(
        {
            f"source__{name}": path
            for name, path in sorted(
                {
                    item.name: item
                    for item in (bundle / "source").iterdir()
                    if item.is_file()
                }.items()
            )
        }
    )
    if any(path.is_symlink() or not path.is_file() for path in result.values()):
        raise ValueError("addendum external-freeze file set is incomplete")
    return result
