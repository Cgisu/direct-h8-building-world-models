"""Build and verify the value-blind v3 provenance bundle.

The builder deliberately has no code path that opens a v3 trajectory. It only
requires the prospective plans and asserts that the locked response destinations
do not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.runtime_provenance import (
    numerical_runtime_fingerprint,
    validate_numerical_runtime_fingerprint,
)

from . import collect, plan, train_grid, worker_collect
from .config import FROZEN_CONFIG


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_PARENT_ROOT = PROJECT_ROOT / "artifacts/direct_h8_publication_v2"
DEFAULT_TRAINING_ROOT = train_grid.CANONICAL_OUTPUT
DEFAULT_SHARED_SOURCE_ROOT = PROJECT_ROOT / "building_fault_wm"
DEFAULT_DATA_ROOT = collect.CANONICAL_DATA_ROOT
DEFAULT_PLAN_ROOT = DEFAULT_DATA_ROOT / "plans/full"
DEFAULT_CERTIFICATE = DEFAULT_DATA_ROOT / "disjointness_certificate.json"
DEFAULT_TESTCASE_ROOT = (Path.home() / "external/project1-boptest/testcases")
DEFAULT_OUTPUT = (
    collect.CANONICAL_PRELOCK_ROOT
)
DEFAULT_PARENT_DIGEST = train_grid.PARENT_DIGEST

SCHEMA = "direct-h8-deterministic-transport-prelock-v2"
COMPLETION_SCHEMA = "direct-h8-deterministic-transport-prelock-completion-v2"
REGISTRY_NAME = "prelock_registry.json"
DIGEST_NAME = "prelock_registry.canonical.sha256"
COMPLETION_NAME = "prelock_preparation_complete.json"
BUNDLE_NAME = "bundle"

REQUIRED_SOURCE_FILES = frozenset(
    {
        "PROTOCOL.md",
        "__init__.py",
        "build_plan_artifacts.py",
        "collect.py",
        "config.py",
        "corpus.py",
        "evaluate.py",
        "external_freeze.py",
        "gate.py",
        "independent_verify.py",
        "model.py",
        "plan.py",
        "prelock.py",
        "report.py",
        "run_evaluation.py",
        "test_collection.py",
        "test_build_plan_artifacts.py",
        "test_evaluate.py",
        "test_external_freeze.py",
        "test_gate.py",
        "test_independent_verify.py",
        "test_model.py",
        "test_plan.py",
        "test_prelock.py",
        "test_report.py",
        "test_run_evaluation.py",
        "test_train.py",
        "train.py",
        "train_grid.py",
        "worker_collect.py",
    }
)


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _parent_manifest_sha256(payload: object) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict:
    _plain_file(path, "JSON artifact")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="ascii"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {token}")
        ),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")
    return path


def _plain_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")
    return path


def _safe_relative_file(root: Path, relative: str, label: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} path must be nonempty and relative")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its root: {relative}") from error
    return _plain_file(path, label)


def _tree_files(root: Path, label: str) -> tuple[Path, ...]:
    root = _plain_directory(root, label)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"{label} contains a non-regular entry: {path}")
    return tuple(files)


def _inventory(root: Path, label: str) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _tree_files(root, label)
    ]


def _source_manifest(source_root: Path) -> dict[str, str]:
    source_root = _plain_directory(source_root, "v3 source root")
    paths = tuple(
        path
        for path in _tree_files(source_root, "v3 source root")
        if path.suffix in {".py", ".md"}
        and "__pycache__" not in path.parts
    )
    relative = {path.relative_to(source_root).as_posix(): path for path in paths}
    missing = REQUIRED_SOURCE_FILES - set(relative)
    if missing:
        raise ValueError(f"v3 source tree is incomplete: {sorted(missing)}")
    return {name: sha256_file(relative[name]) for name in sorted(relative)}


def _atomic_bytes(path: Path, content: bytes) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite pre-lock artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("ascii"),
    )


def _copy_file(source: Path, destination: Path) -> Path:
    source = _plain_file(source, "pre-lock copy source")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite pre-lock copy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as output_stream:
        temporary = Path(output_stream.name)
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    try:
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    if (
        source.stat().st_size != destination.stat().st_size
        or sha256_file(source) != sha256_file(destination)
    ):
        raise IOError(f"pre-lock copy differs from source: {source}")
    return destination


def _copy_inventory(
    source_root: Path,
    destination_root: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("copy inventory contains an invalid path")
        source = _safe_relative_file(source_root, relative, "copy source")
        if (
            source.stat().st_size != row.get("bytes")
            or sha256_file(source) != row.get("sha256")
        ):
            raise ValueError(f"copy source changed: {relative}")
        _copy_file(source, destination_root / relative)


def _expected_parent_paths() -> tuple[str, ...]:
    prefix = "experiment/prelock_bundle/frozen"
    paths = [
        "evidence/locked_fault_manifest.json",
        "experiment/prelock_bundle/development/source/run_config.json",
        f"{prefix}/frozen_fault_contract.json",
    ]
    for case in sorted(boptest.CASES):
        paths.append(f"{prefix}/fit_scalers/{case}.json")
        for seed in FROZEN_CONFIG.paired_model_seeds:
            paths.append(f"{prefix}/schedules/{case}/seed{seed}.json")
            for arm in ("legacy", "ungated_h8"):
                paths.append(
                    f"{prefix}/checkpoints/{case}/seed{seed}/{arm}_u0400.pt"
                )
    return tuple(sorted(paths))


def validate_parent_package(
    parent_root: Path,
    expected_digest: str = DEFAULT_PARENT_DIGEST,
) -> dict[str, object]:
    """Verify the complete immutable parent and its reused scientific subset."""

    parent_root = _plain_directory(parent_root, "parent publication package")
    manifest_path = _plain_file(
        parent_root / "package_manifest.json", "parent package manifest"
    )
    digest_path = _plain_file(
        parent_root / "package_manifest.canonical.sha256",
        "parent package digest record",
    )
    manifest = _strict_json(manifest_path)
    if manifest.get("schema") != "direct-h8-publication-package-manifest-v1":
        raise ValueError("parent package manifest schema changed")
    if _parent_manifest_sha256(manifest) != expected_digest:
        raise ValueError("parent package canonical digest changed")
    digest_record = digest_path.read_text(encoding="ascii")
    if digest_record != f"{expected_digest}\n":
        raise ValueError("parent package digest record changed")

    rows = manifest.get("artifact_inventory_excludes_manifest_and_digest")
    if not isinstance(rows, list) or not rows:
        raise ValueError("parent package inventory is incomplete")
    by_path: dict[str, dict] = {}
    total_bytes = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("parent package inventory row is invalid")
        relative = row["path"]
        if not isinstance(relative, str) or relative in by_path:
            raise ValueError("parent package inventory path is invalid or duplicated")
        path = _safe_relative_file(parent_root, relative, "parent artifact")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"parent artifact changed: {relative}")
        by_path[relative] = row
        total_bytes += int(row["bytes"])

    actual = {
        path.relative_to(parent_root).as_posix()
        for path in _tree_files(parent_root, "parent publication package")
        if path.name
        not in {"package_manifest.json", "package_manifest.canonical.sha256"}
    }
    if actual != set(by_path):
        raise ValueError("parent package on-disk inventory differs from its manifest")
    required = set(_expected_parent_paths())
    if not required <= set(by_path):
        raise ValueError(
            "parent reused artifact set is incomplete: "
            f"{sorted(required - set(by_path))}"
        )

    run_config = _strict_json(
        parent_root
        / "experiment/prelock_bundle/development/source/run_config.json"
    )
    parent_runtime = run_config.get("numerical_runtime")
    validate_numerical_runtime_fingerprint(
        parent_runtime, include_sklearn=True
    )
    return {
        "canonical_digest": expected_digest,
        "manifest_file_sha256": sha256_file(manifest_path),
        "digest_record_file_sha256": sha256_file(digest_path),
        "inventory_file_count": len(rows),
        "inventory_bytes": total_bytes,
        "inventory_sha256": canonical_sha256(rows),
        "reused_paths": list(_expected_parent_paths()),
        "parent_runtime_sha256": parent_runtime["sha256"],
    }


def _validate_training(
    training_root: Path,
    *,
    parent_summary: Mapping[str, object],
    source_manifest: Mapping[str, str],
    shared_runtime_source_manifest: Mapping[str, object],
    parent_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    training_root = _plain_directory(training_root, "v3 training root")
    grid_path = _plain_file(
        training_root / train_grid.GRID_RECEIPT, "complete training-grid receipt"
    )
    grid = _strict_json(grid_path)
    expected_parent = {
        key: parent_summary[key]
        for key in ("canonical_digest", "inventory_file_count", "inventory_bytes")
    }
    if (
        grid.get("schema") != train_grid.GRID_SCHEMA
        or grid.get("complete_grid") is not True
        or grid.get("parent_package") != expected_parent
        or grid.get("shared_runtime_source_manifest")
        != shared_runtime_source_manifest
    ):
        raise ValueError("v3 complete training-grid receipt is invalid")
    expected_config = json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
    if grid.get("config") != expected_config:
        raise ValueError("v3 training-grid configuration changed")
    recorded_source = grid.get("source_code_sha256")
    executable_training_sources = {
        "__init__.py",
        "config.py",
        "model.py",
        "train.py",
        "train_grid.py",
    }
    if (
        not isinstance(recorded_source, dict)
        or set(recorded_source)
        != executable_training_sources | {"PROTOCOL.md"}
        or any(
            source_manifest.get(name) != recorded_source[name]
            for name in executable_training_sources
        )
        or not boptest.valid_sha256(recorded_source.get("PROTOCOL.md"))
    ):
        raise ValueError("v3 training source binding changed")
    protocol_changed_after_training = (
        source_manifest["PROTOCOL.md"] != recorded_source["PROTOCOL.md"]
    )
    source_lock_path = _plain_file(
        training_root / train_grid.SOURCE_LOCK_NAME,
        "v3 training source lock",
    )
    source_lock = _strict_json(source_lock_path)
    expected_source_lock = {
        "schema": train_grid.SOURCE_LOCK_SCHEMA,
        "parent_package": expected_parent,
        "v3_training_source_manifest": recorded_source,
        "shared_runtime_source_manifest": shared_runtime_source_manifest,
        "config": expected_config,
    }
    if (
        source_lock != expected_source_lock
        or grid.get("training_source_lock_file_sha256")
        != sha256_file(source_lock_path)
    ):
        raise ValueError("v3 training source-lock binding changed")

    runs = grid.get("runs")
    expected_units = {
        (case, seed)
        for case in boptest.CASES
        for seed in FROZEN_CONFIG.paired_model_seeds
    }
    if not isinstance(runs, list) or {
        (row.get("case"), row.get("model_seed"))
        for row in runs
        if isinstance(row, dict)
    } != expected_units:
        raise ValueError("v3 training-grid unit identities are incomplete")
    rows_by_unit = {
        (str(row["case"]), int(row["model_seed"])): row for row in runs
    }
    if len(rows_by_unit) != len(expected_units):
        raise ValueError("v3 training-grid unit identities are duplicated")

    checkpoint_names = {
        f"update_{update:04d}.pt" for update in FROZEN_CONFIG.checkpoint_updates
    }
    for case, seed in sorted(expected_units):
        run_root = _plain_directory(
            training_root / case / f"seed{seed}", "v3 training unit"
        )
        receipt_path = _plain_file(
            run_root / "training_receipt.json", "v3 training receipt"
        )
        receipt = _strict_json(receipt_path)
        if (
            receipt.get("schema") != "boptest-deterministic-transport-training-v1"
            or receipt.get("model_seed") != seed
            or receipt.get("updates") != FROZEN_CONFIG.updates
            or receipt.get("selected_update") != FROZEN_CONFIG.updates
            or receipt.get("selection_rule")
            != "fixed_final_update_no_validation_selection"
            or receipt.get("config") != expected_config
        ):
            raise ValueError(f"v3 training receipt changed for {case}/seed{seed}")
        hashes = receipt.get("checkpoint_file_sha256")
        if not isinstance(hashes, dict) or set(hashes) != checkpoint_names:
            raise ValueError(f"v3 checkpoint grid is incomplete for {case}/seed{seed}")
        for name, digest in hashes.items():
            checkpoint = _plain_file(
                run_root / "checkpoints" / name, "v3 deterministic checkpoint"
            )
            if sha256_file(checkpoint) != digest:
                raise ValueError(f"v3 deterministic checkpoint changed: {checkpoint}")
        row = rows_by_unit[(case, seed)]
        schedule = (
            parent_root
            / "experiment/prelock_bundle/frozen/schedules"
            / case
            / f"seed{seed}.json"
        )
        if (
            row.get("schedule_file_sha256") != sha256_file(schedule)
            or row.get("selected_checkpoint_file_sha256")
            != hashes[f"update_{FROZEN_CONFIG.updates:04d}.pt"]
            or row.get("training_receipt_sha256") != sha256_file(receipt_path)
        ):
            raise ValueError(f"v3 grid receipt does not bind {case}/seed{seed}")

    inventory = _inventory(training_root, "v3 training root")
    if any(
        not str(row["path"]).endswith((".json", ".pt")) for row in inventory
    ):
        raise ValueError("v3 training tree contains an unsupported file")
    return inventory, {
        "grid_receipt_sha256": sha256_file(grid_path),
        "unit_count": len(expected_units),
        "checkpoint_count": len(expected_units) * len(checkpoint_names),
        "artifact_count": len(inventory),
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "executable_training_source_unchanged": True,
        "training_protocol_sha256_at_run": recorded_source["PROTOCOL.md"],
        "final_prelock_protocol_sha256": source_manifest["PROTOCOL.md"],
        "protocol_changed_after_training_before_outcome_lock": (
            protocol_changed_after_training
        ),
        "shared_runtime_source_manifest_sha256": (
            shared_runtime_source_manifest["sha256"]
        ),
        "shared_runtime_source_unchanged": True,
        "training_source_lock_file_sha256": sha256_file(source_lock_path),
    }


def _validate_plans(
    plan_root: Path, certificate_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    plan_root = _plain_directory(plan_root, "v3 plan root")
    expected_names = {f"{case}.json" for case in boptest.CASES}
    files = tuple(plan_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("v3 plan root contains a non-plain entry")
    if {path.name for path in files} != expected_names:
        raise ValueError("v3 plan root differs from the three-case contract")
    plans = {
        case: _strict_json(plan_root / f"{case}.json")
        for case in sorted(boptest.CASES)
    }
    for case, payload in plans.items():
        plan.validate_case_plan(payload)
        if payload.get("case") != case:
            raise ValueError(f"v3 plan case identity changed: {case}")

    certificate = _strict_json(certificate_path)
    certificate_digest = certificate.get("certificate_sha256")
    if not isinstance(certificate_digest, str):
        raise ValueError("v3 disjointness certificate has no digest")
    worker_collect.validate_certificate_grid(
        certificate, plans, certificate_digest
    )
    rows = [
        {
            "path": f"{case}.json",
            "bytes": (plan_root / f"{case}.json").stat().st_size,
            "sha256": sha256_file(plan_root / f"{case}.json"),
        }
        for case in sorted(boptest.CASES)
    ]
    return rows, {
        "plan_sha256_by_case": {
            case: plans[case]["plan_sha256"] for case in sorted(plans)
        },
        "plan_file_sha256_by_case": {
            case: sha256_file(plan_root / f"{case}.json")
            for case in sorted(plans)
        },
        "certificate_sha256": certificate_digest,
        "certificate_file_sha256": sha256_file(certificate_path),
        "window_count": sum(len(payload["entries"]) for payload in plans.values()),
        "branch_count": sum(
            len(payload["entries"]) * len(plan.POLICIES)
            for payload in plans.values()
        ),
        "response_values_used_for_selection": False,
    }


def _environment_pins(
    *,
    testcase_root: Path,
    validate_environment: bool,
) -> dict[str, object]:
    public_source = {
        "repository_url": boptest.BOPTEST_REPOSITORY_URL,
        "commit": boptest.BOPTEST_COMMIT,
        "license_path": boptest.BOPTEST_LICENSE_PATH,
        "license_sha256": boptest.BOPTEST_LICENSE_SHA256,
    }
    image = boptest.WORKER_IMAGE_ID
    if validate_environment:
        actual_source = collect.validate_public_checkout(testcase_root)
        if actual_source != {
            "repository_url": boptest.BOPTEST_REPOSITORY_URL,
            "commit": boptest.BOPTEST_COMMIT,
            "license_sha256": boptest.BOPTEST_LICENSE_SHA256,
        }:
            raise ValueError("live BOPTEST checkout differs from the frozen pin")
        image = collect.validate_worker_image()
    return {
        "public_source": public_source,
        "worker_image_id": image,
        "worker_boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "step_seconds": boptest.STEP_SECONDS,
        "warmup_seconds": boptest.WARMUP_SECONDS,
        "trajectory_steps": boptest.TRAJECTORY_STEPS,
        "live_environment_validated": validate_environment,
    }


def _validate_environment_pins_payload(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("environment pins are not an object")
    expected = {
        "public_source": {
            "repository_url": boptest.BOPTEST_REPOSITORY_URL,
            "commit": boptest.BOPTEST_COMMIT,
            "license_path": boptest.BOPTEST_LICENSE_PATH,
            "license_sha256": boptest.BOPTEST_LICENSE_SHA256,
        },
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "worker_boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "step_seconds": boptest.STEP_SECONDS,
        "warmup_seconds": boptest.WARMUP_SECONDS,
        "trajectory_steps": boptest.TRAJECTORY_STEPS,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"environment pin changed at {key}")
    if not isinstance(value.get("live_environment_validated"), bool):
        raise ValueError("environment live-validation flag is invalid")


def _default_forbidden_response_paths(data_root: Path) -> tuple[Path, ...]:
    raw = data_root / "locked_transport_raw"
    manifest = data_root / "manifests/locked_transport_corpus_manifest.json"
    return (
        raw,
        raw.parent / f".{raw.name}.staging",
        manifest,
        manifest.with_suffix(manifest.suffix + ".pending"),
        collect.STATE_ROOT,
        collect.CANONICAL_FREEZE_ROOT,
    )


def _portable_path_label(path: Path) -> str:
    absolute = path.resolve()
    try:
        relative = absolute.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"$EXTERNAL/{absolute.name}"
    return f"$PROJECT_ROOT/{relative.as_posix()}"


def _assert_no_response_artifacts(paths: Iterable[Path]) -> None:
    occupied = [path for path in paths if os.path.lexists(path)]
    if occupied:
        raise FileExistsError(
            "v3 response artifact exists before external freeze: "
            + ", ".join(str(path) for path in occupied)
        )


def _staging_path(output_dir: Path) -> Path:
    if not output_dir.name:
        raise ValueError("v3 pre-lock output must name a directory")
    return output_dir.parent / f".{output_dir.name}.prelock-staging"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_prelock(
    *,
    parent_root: Path = DEFAULT_PARENT_ROOT,
    expected_parent_digest: str = DEFAULT_PARENT_DIGEST,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    shared_source_root: Path = DEFAULT_SHARED_SOURCE_ROOT,
    source_root: Path = HERE,
    plan_root: Path = DEFAULT_PLAN_ROOT,
    certificate_path: Path = DEFAULT_CERTIFICATE,
    testcase_root: Path = DEFAULT_TESTCASE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    forbidden_response_paths: Sequence[Path] | None = None,
    validate_environment: bool = True,
    runtime_payload: Mapping[str, object] | None = None,
) -> Path:
    """Atomically publish a complete pre-lock without reading v3 responses."""

    if os.path.lexists(output_dir):
        raise FileExistsError(f"v3 pre-lock output already exists: {output_dir}")
    staging = _staging_path(output_dir)
    if os.path.lexists(staging):
        raise FileExistsError(f"stale v3 pre-lock staging exists: {staging}")
    response_paths = (
        tuple(forbidden_response_paths)
        if forbidden_response_paths is not None
        else _default_forbidden_response_paths(DEFAULT_DATA_ROOT)
    )
    _assert_no_response_artifacts(response_paths)

    source_before = _source_manifest(source_root)
    shared_source_before = train_grid.shared_runtime_source_manifest(
        shared_source_root
    )
    parent_before = validate_parent_package(parent_root, expected_parent_digest)
    training_before, training_summary = _validate_training(
        training_root,
        parent_summary=parent_before,
        source_manifest=source_before,
        shared_runtime_source_manifest=shared_source_before,
        parent_root=parent_root,
    )
    plan_rows_before, plan_summary = _validate_plans(
        plan_root, certificate_path
    )
    pins = _environment_pins(
        testcase_root=testcase_root,
        validate_environment=validate_environment,
    )
    runtime = (
        dict(runtime_payload)
        if runtime_payload is not None
        else numerical_runtime_fingerprint("cpu", include_sklearn=True)
    )
    validate_numerical_runtime_fingerprint(runtime, include_sklearn=True)

    staging.mkdir(parents=True)
    try:
        bundle = staging / BUNDLE_NAME
        bundle.mkdir()

        source_rows = [
            {
                "path": relative,
                "bytes": (source_root / relative).stat().st_size,
                "sha256": digest,
            }
            for relative, digest in source_before.items()
        ]
        _copy_inventory(source_root, bundle / "v3_source", source_rows)
        shared_source_rows = [
            {
                "path": relative,
                "bytes": (shared_source_root / relative).stat().st_size,
                "sha256": digest,
            }
            for relative, digest in shared_source_before["files"].items()
        ]
        _copy_inventory(
            shared_source_root,
            bundle / "shared_runtime_source",
            shared_source_rows,
        )
        _copy_inventory(plan_root, bundle / "plans/full", plan_rows_before)
        _copy_file(
            certificate_path,
            bundle / "plans/disjointness_certificate.json",
        )
        _copy_inventory(training_root, bundle / "training", training_before)

        parent_manifest = _strict_json(parent_root / "package_manifest.json")
        parent_rows = parent_manifest[
            "artifact_inventory_excludes_manifest_and_digest"
        ]
        parent_by_path = {row["path"]: row for row in parent_rows}
        reused_rows = [
            parent_by_path[relative] for relative in _expected_parent_paths()
        ]
        _copy_inventory(
            parent_root, bundle / "parent_selected", reused_rows
        )
        _copy_file(
            parent_root / "package_manifest.json",
            bundle / "parent_identity/package_manifest.json",
        )
        _copy_file(
            parent_root / "package_manifest.canonical.sha256",
            bundle / "parent_identity/package_manifest.canonical.sha256",
        )
        _write_json(bundle / "environment/pins.json", pins)
        _write_json(bundle / "environment/v3_numerical_runtime.json", runtime)

        bundle_inventory = _inventory(bundle, "v3 pre-lock bundle")
        registry = {
            "schema": SCHEMA,
            "study_kind": worker_collect.STUDY_KIND,
            "claim_boundary": "open_loop_boptest_world_model_prediction_only",
            "parent_package": parent_before,
            "source_code_sha256_by_path": source_before,
            "source_code_manifest_sha256": canonical_sha256(source_before),
            "shared_runtime_source_manifest": shared_source_before,
            "plans": plan_summary,
            "training": training_summary,
            "parent_reused_artifact_sha256_by_path": {
                row["path"]: row["sha256"] for row in reused_rows
            },
            "environment": pins,
            "v3_numerical_runtime": runtime,
            "bundle_inventory": bundle_inventory,
            "bundle_inventory_sha256": canonical_sha256(bundle_inventory),
            "locked_response_artifact_paths_checked": [
                _portable_path_label(path) for path in response_paths
            ],
            "locked_response_values_read": False,
            "locked_response_files_opened": False,
            "external_freeze_required": True,
        }
        digest = canonical_sha256(registry)
        registry_path = _write_json(staging / REGISTRY_NAME, registry)
        _atomic_bytes(staging / DIGEST_NAME, f"{digest}\n".encode("ascii"))
        completion = {
            "schema": COMPLETION_SCHEMA,
            "study_kind": worker_collect.STUDY_KIND,
            "canonical_registry_sha256": digest,
            "registry_file_sha256": sha256_file(registry_path),
            "bundle_inventory_sha256": registry["bundle_inventory_sha256"],
            "bundle_artifact_count": len(bundle_inventory),
            "parent_package_sha256": expected_parent_digest,
            "v3_plan_certificate_sha256": plan_summary["certificate_sha256"],
            "complete_training_grid_bound": True,
            "live_environment_validated": validate_environment,
            "locked_response_values_read": False,
            "locked_response_files_opened": False,
            "external_freeze_required": True,
        }
        _write_json(staging / COMPLETION_NAME, completion)

        if _source_manifest(source_root) != source_before:
            raise RuntimeError("v3 source changed during pre-lock preparation")
        if (
            train_grid.shared_runtime_source_manifest(shared_source_root)
            != shared_source_before
        ):
            raise RuntimeError(
                "shared runtime source changed during pre-lock preparation"
            )
        if (
            validate_parent_package(parent_root, expected_parent_digest)
            != parent_before
        ):
            raise RuntimeError("parent package changed during pre-lock preparation")
        training_after, training_summary_after = _validate_training(
            training_root,
            parent_summary=parent_before,
            source_manifest=source_before,
            shared_runtime_source_manifest=shared_source_before,
            parent_root=parent_root,
        )
        if (
            training_after != training_before
            or training_summary_after != training_summary
        ):
            raise RuntimeError("v3 training artifacts changed during pre-lock")
        plan_rows_after, plan_summary_after = _validate_plans(
            plan_root, certificate_path
        )
        if (
            plan_rows_after != plan_rows_before
            or plan_summary_after != plan_summary
        ):
            raise RuntimeError("v3 plans changed during pre-lock preparation")
        _assert_no_response_artifacts(response_paths)

        validate_prelock_bundle(
            staging,
            parent_root=parent_root,
            expected_parent_digest=expected_parent_digest,
            source_root=source_root,
            shared_source_root=shared_source_root,
        )
        _fsync_directory(staging)
        if os.path.lexists(output_dir):
            raise FileExistsError("v3 pre-lock destination changed during preparation")
        os.rename(staging, output_dir)
        _fsync_directory(output_dir.parent)
        return output_dir / REGISTRY_NAME
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def validate_prelock_bundle(
    root: Path,
    *,
    parent_root: Path | None = DEFAULT_PARENT_ROOT,
    expected_parent_digest: str = DEFAULT_PARENT_DIGEST,
    source_root: Path | None = None,
    shared_source_root: Path | None = DEFAULT_SHARED_SOURCE_ROOT,
) -> dict:
    """Validate all bundle bytes and, when supplied, their live parent/source."""

    root = _plain_directory(root, "v3 pre-lock root")
    expected_top = {
        BUNDLE_NAME,
        REGISTRY_NAME,
        DIGEST_NAME,
        COMPLETION_NAME,
    }
    if {path.name for path in root.iterdir()} != expected_top:
        raise ValueError("v3 pre-lock top-level inventory changed")
    registry_path = _plain_file(root / REGISTRY_NAME, "v3 pre-lock registry")
    digest_path = _plain_file(root / DIGEST_NAME, "v3 pre-lock digest")
    completion_path = _plain_file(
        root / COMPLETION_NAME, "v3 pre-lock completion"
    )
    registry = _strict_json(registry_path)
    digest = digest_path.read_text(encoding="ascii")
    expected_digest = canonical_sha256(registry)
    if digest != f"{expected_digest}\n":
        raise ValueError("v3 pre-lock registry digest changed")
    if (
        registry.get("schema") != SCHEMA
        or registry.get("locked_response_values_read") is not False
        or registry.get("locked_response_files_opened") is not False
        or registry.get("external_freeze_required") is not True
    ):
        raise ValueError("v3 pre-lock registry contract changed")

    bundle = _plain_directory(root / BUNDLE_NAME, "v3 pre-lock bundle")
    actual_inventory = _inventory(bundle, "v3 pre-lock bundle")
    if (
        registry.get("bundle_inventory") != actual_inventory
        or registry.get("bundle_inventory_sha256")
        != canonical_sha256(actual_inventory)
    ):
        raise ValueError("v3 pre-lock bundle inventory changed")

    completion = _strict_json(completion_path)
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("canonical_registry_sha256") != expected_digest
        or completion.get("registry_file_sha256") != sha256_file(registry_path)
        or completion.get("bundle_inventory_sha256")
        != registry["bundle_inventory_sha256"]
        or completion.get("locked_response_values_read") is not False
        or completion.get("locked_response_files_opened") is not False
    ):
        raise ValueError("v3 pre-lock completion contract changed")

    copied_parent = bundle / "parent_identity"
    copied_manifest = _strict_json(copied_parent / "package_manifest.json")
    if _parent_manifest_sha256(copied_manifest) != expected_parent_digest:
        raise ValueError("copied parent package manifest digest changed")
    if (
        (copied_parent / "package_manifest.canonical.sha256").read_text(
            encoding="ascii"
        )
        != f"{expected_parent_digest}\n"
    ):
        raise ValueError("copied parent digest record changed")
    parent_rows = copied_manifest[
        "artifact_inventory_excludes_manifest_and_digest"
    ]
    parent_by_path = {row["path"]: row for row in parent_rows}
    reused_hashes = registry.get("parent_reused_artifact_sha256_by_path")
    if not isinstance(reused_hashes, dict) or set(reused_hashes) != set(
        _expected_parent_paths()
    ):
        raise ValueError("parent reused-artifact registry is incomplete")
    for relative in _expected_parent_paths():
        row = parent_by_path.get(relative)
        if not isinstance(row, dict):
            raise ValueError(f"copied parent inventory lacks {relative}")
        if reused_hashes.get(relative) != row["sha256"]:
            raise ValueError(f"parent reused-artifact binding changed: {relative}")
        path = _safe_relative_file(
            bundle / "parent_selected", relative, "copied parent artifact"
        )
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"copied parent artifact changed: {relative}")
    parent_run_config = _strict_json(
        bundle
        / "parent_selected"
        / "experiment/prelock_bundle/development/source/run_config.json"
    )
    parent_runtime = parent_run_config.get("numerical_runtime")
    validate_numerical_runtime_fingerprint(
        parent_runtime, include_sklearn=True
    )
    if (
        not isinstance(registry.get("parent_package"), dict)
        or registry["parent_package"].get("parent_runtime_sha256")
        != parent_runtime["sha256"]
    ):
        raise ValueError("copied parent numerical runtime binding changed")

    copied_source = _source_manifest(bundle / "v3_source")
    if copied_source != registry.get("source_code_sha256_by_path"):
        raise ValueError("copied v3 source binding changed")
    if source_root is not None and _source_manifest(source_root) != copied_source:
        raise ValueError("live v3 source differs from the pre-lock bundle")
    copied_shared_source = train_grid.shared_runtime_source_manifest(
        bundle / "shared_runtime_source"
    )
    if copied_shared_source != registry.get("shared_runtime_source_manifest"):
        raise ValueError("copied shared runtime source binding changed")
    if (
        shared_source_root is not None
        and train_grid.shared_runtime_source_manifest(shared_source_root)
        != copied_shared_source
    ):
        raise ValueError("live shared runtime source differs from the pre-lock bundle")

    plans_root = bundle / "plans/full"
    _, plan_summary = _validate_plans(
        plans_root, bundle / "plans/disjointness_certificate.json"
    )
    if plan_summary != registry.get("plans"):
        raise ValueError("copied v3 plan binding changed")

    _, training_summary = _validate_training(
        bundle / "training",
        parent_summary=registry["parent_package"],
        source_manifest=copied_source,
        shared_runtime_source_manifest=copied_shared_source,
        parent_root=bundle / "parent_selected",
    )
    if training_summary != registry.get("training"):
        raise ValueError("copied v3 training binding changed")

    pins = _strict_json(bundle / "environment/pins.json")
    _validate_environment_pins_payload(pins)
    if pins != registry.get("environment"):
        raise ValueError("copied environment pins changed")
    runtime = _strict_json(bundle / "environment/v3_numerical_runtime.json")
    validate_numerical_runtime_fingerprint(runtime, include_sklearn=True)
    if runtime != registry.get("v3_numerical_runtime"):
        raise ValueError("copied v3 numerical runtime changed")

    if parent_root is not None:
        live_parent = validate_parent_package(
            parent_root, expected_parent_digest
        )
        if live_parent != registry.get("parent_package"):
            raise ValueError("live parent package differs from the v3 pre-lock")
    return registry


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument(
        "--shared-source-root", type=Path, default=DEFAULT_SHARED_SOURCE_ROOT
    )
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--testcase-root", type=Path, default=DEFAULT_TESTCASE_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.verify_only:
        registry = validate_prelock_bundle(
            args.output.resolve(),
            parent_root=args.parent_root.resolve(),
            source_root=HERE,
            shared_source_root=args.shared_source_root.resolve(),
        )
        print(canonical_sha256(registry))
        return
    path = prepare_prelock(
        parent_root=args.parent_root.resolve(),
        training_root=args.training_root.resolve(),
        shared_source_root=args.shared_source_root.resolve(),
        plan_root=args.plan_root.resolve(),
        certificate_path=args.certificate.resolve(),
        testcase_root=args.testcase_root.resolve(),
        output_dir=args.output.resolve(),
    )
    print(path)


if __name__ == "__main__":
    main()
