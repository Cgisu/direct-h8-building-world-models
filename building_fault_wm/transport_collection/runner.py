"""Prepare and run the ownership-corrected full transport recollection."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.deterministic_transport import (
    collect as frozen_collect,
)
from building_fault_wm.deterministic_transport import (
    external_freeze as frozen_external_freeze,
)
from building_fault_wm.deterministic_transport import (
    plan as frozen_plan,
)
from building_fault_wm.deterministic_transport import (
    prelock as frozen_prelock,
)
from building_fault_wm.deterministic_transport import (
    worker_collect as frozen_worker,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FROZEN_SOURCE_ROOT = frozen_collect.HERE
PRELOCK_ROOT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_prelock_v4"
)
DATA_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data"
)
LEGACY_DATA_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data_v7"
)
STATE_ROOT = HERE / ".direct_h8_transport_collection_state_v3"
FREEZE_ROOT = PROJECT_ROOT / "artifacts/direct_h8_transport_collection_freeze_v5"
READINESS_PATH = FREEZE_ROOT / "collection_readiness.json"
EXTERNAL_FREEZE_RECEIPT = FREEZE_ROOT / "external_freeze_receipt.json"
TESTCASE_ROOT = (Path.home() / "external/project1-boptest/testcases")

PLAN_ROOT_RELATIVE = Path("plans/full")
CERTIFICATE_RELATIVE = Path("disjointness_certificate.json")
RAW_RELATIVE = Path("locked_transport_raw")
MANIFEST_RELATIVE = Path("manifests/locked_transport_corpus_manifest.json")

ORIGINAL_PRELOCK_SHA256 = (
    "50dbd5d24537b61e109ff6634361ddb9ca9bceac2528b57394125a6667d80094"
)
TERMINAL_V4_READINESS_SHA256 = (
    "2bb8caf76d635189d1b0c738eca6a332a42639b63be7e5a8b3a9fee540e2df38"
)
EXPECTED_UID = 1005
EXPECTED_GID = 1006
CONFIRMATION_TOKEN = (
    "I_UNDERSTAND_DIRECT_H8_V5_FULL_RECOLLECTION_OPENS_LOCKED_OUTCOMES"
)

READINESS_SCHEMA = "direct-h8-transport-collection-readiness-v5"
TERMINAL_BINDING_SCHEMA = "direct-h8-transport-v4-terminal-failure-binding-v1"
ATTEMPT_MARKER = frozen_collect.ATTEMPT_MARKER
FAILURE_MARKER = frozen_collect.FAILURE_MARKER
COMPLETION_MARKER = frozen_collect.COMPLETION_MARKER

FROZEN_COLLECTION_CODE_SHA256 = {
    "collect.py": "0ac4006e9a5e08b9f76bd9c14bff292706582b3283045850e67f5fead19f4a27",
    "plan.py": "16f8124b55a503ac787087d19c5d97d4716463b9a93c6e6858bf49ec08fe8505",
    "worker_collect.py": (
        "5043b19d51edb1f66663fc993b34b89f8ce55b9cb3ad9081dab64e4c248bdcfd"
    ),
}
FROZEN_RUN_EVALUATION_SHA256 = (
    "431d0e325aa71ada4edc0e6b8e758b029045b8eac3c06452302b4ab0ca98450e"
)
FROZEN_CORPUS_SHA256 = (
    "a9247fc9a09e437a68c308e045b5a3545fccc38bce3513195ebed47b00d6b2ed"
)

V4_STATE_DIR = (
    FROZEN_SOURCE_ROOT
    / ".direct_h8_v3_locked_state_v2"
    / TERMINAL_V4_READINESS_SHA256
)
V4_ATTEMPT_PATH = V4_STATE_DIR / frozen_collect.ATTEMPT_MARKER
V4_FAILURE_PATH = V4_STATE_DIR / frozen_collect.FAILURE_MARKER
V4_READINESS_PATH = (
    PROJECT_ROOT
    / "artifacts/direct_h8_deterministic_transport_v3_external_freeze_v4"
    / "collection_readiness.json"
)
V4_FREEZE_RECEIPT_PATH = (
    PROJECT_ROOT
    / "artifacts/direct_h8_deterministic_transport_v3_external_freeze_v4"
    / "external_freeze_receipt.json"
)
EXPECTED_V4_FILE_SHA256 = {
    "attempt": "a768661a64acc57cf755c5df3877a0b452435a7de8e62945d6ad733984ecac2c",
    "failure": "0176f7e3624d0a0734c50bdc47b9f7a8518f04fe2f899c0dbc42d0cf37240350",
    "readiness_file": (
        "c15d7ad1322af5028147a6a432775340164d87de04961074dfcd301f75b5c7a7"
    ),
    "external_freeze_receipt": (
        "687e918213c8e480f78c628e17a818a3e55d52b9eb11100f89b0c3e0156f2e67"
    ),
}


def _require_plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")
    return path


def _require_plain_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")
    return path


def _load_json(path: Path, label: str) -> dict:
    value = boptest.strict_json_loads(
        _require_plain_file(path, label).read_text(encoding="ascii")
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_namespace_separation(
    *,
    data_root: Path,
    state_root: Path,
    freeze_root: Path,
    legacy_data_root: Path = LEGACY_DATA_ROOT,
) -> None:
    roots = {
        "data": data_root.resolve(),
        "state_v3": state_root.resolve(),
        "freeze_v5": freeze_root.resolve(),
    }
    for name, root in roots.items():
        if _is_within(root, legacy_data_root) or _is_within(
            legacy_data_root, root
        ):
            raise ValueError(
                f"{name} must be separate from the terminal data_v7 tree"
            )
    pairs = tuple(roots.items())
    for index, (left_name, left) in enumerate(pairs):
        for right_name, right in pairs[index + 1 :]:
            if _is_within(left, right) or _is_within(right, left):
                raise ValueError(
                    f"{left_name} and {right_name} namespaces overlap"
                )


def operational_code_hashes() -> dict[str, str]:
    return {
        name: frozen_plan.sha256_file(HERE / name)
        for name in (
            "PROTOCOL_ADDENDUM.md",
            "evaluation_adapter.py",
            "external_freeze.py",
            "runner.py",
            "v4_closeout.py",
        )
    }


def validate_frozen_source_hashes() -> dict[str, object]:
    collection = frozen_collect.collection_code_hashes()
    if collection != FROZEN_COLLECTION_CODE_SHA256:
        raise ValueError("frozen collection source differs from its prelock")
    run_evaluation_sha256 = frozen_plan.sha256_file(
        FROZEN_SOURCE_ROOT / "run_evaluation.py"
    )
    corpus_sha256 = frozen_plan.sha256_file(FROZEN_SOURCE_ROOT / "corpus.py")
    if run_evaluation_sha256 != FROZEN_RUN_EVALUATION_SHA256:
        raise ValueError("frozen run_evaluation.py changed")
    if corpus_sha256 != FROZEN_CORPUS_SHA256:
        raise ValueError("frozen corpus.py changed")
    return {
        "collection_code_sha256": collection,
        "run_evaluation.py": run_evaluation_sha256,
        "corpus.py": corpus_sha256,
    }


def _validate_scientific_prelock(prelock_root: Path) -> dict:
    registry = frozen_prelock.validate_prelock_bundle(
        prelock_root,
        parent_root=None,
        source_root=FROZEN_SOURCE_ROOT,
        shared_source_root=None,
    )
    digest = frozen_prelock.canonical_sha256(registry)
    if digest != ORIGINAL_PRELOCK_SHA256:
        raise ValueError("scientific prelock differs from the original digest")
    return registry


def validate_live_shared_runtime_semantics(
    *,
    prelock_root: Path,
    registry: Mapping[str, object],
) -> dict:
    """Reject functional drift while recording harmless source formatting drift."""

    manifest = registry.get("shared_runtime_source_manifest")
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("scientific prelock lacks its shared runtime manifest")
    expected_hashes = manifest["files"]
    frozen_root = prelock_root / "bundle/shared_runtime_source"
    live_root = PROJECT_ROOT / "building_fault_wm"
    live_hashes: dict[str, str] = {}
    ast_hashes: dict[str, str] = {}
    byte_differences: list[str] = []
    for relative, expected_sha256 in sorted(expected_hashes.items()):
        frozen = _require_plain_file(
            frozen_root / relative, "prelock shared runtime source"
        )
        live = _require_plain_file(
            live_root / relative, "live shared runtime source"
        )
        if frozen_plan.sha256_file(frozen) != expected_sha256:
            raise ValueError(f"prelock shared source hash changed: {relative}")
        live_sha256 = frozen_plan.sha256_file(live)
        live_hashes[relative] = live_sha256
        frozen_ast = ast.dump(
            ast.parse(frozen.read_text(encoding="utf-8")),
            include_attributes=False,
        ).encode("utf-8")
        live_ast = ast.dump(
            ast.parse(live.read_text(encoding="utf-8")),
            include_attributes=False,
        ).encode("utf-8")
        frozen_ast_sha256 = hashlib.sha256(frozen_ast).hexdigest()
        live_ast_sha256 = hashlib.sha256(live_ast).hexdigest()
        if live_ast_sha256 != frozen_ast_sha256:
            raise ValueError(
                f"live shared runtime has functional source drift: {relative}"
            )
        ast_hashes[relative] = frozen_ast_sha256
        if live_sha256 != expected_sha256:
            byte_differences.append(relative)
    payload = {
        "schema": "direct-h8-v5-live-shared-runtime-semantics-v1",
        "frozen_manifest_sha256": manifest["sha256"],
        "live_file_sha256": live_hashes,
        "ast_sha256": ast_hashes,
        "byte_different_ast_identical": byte_differences,
        "all_python_asts_identical": True,
    }
    return {
        **payload,
        "sha256": frozen_plan.canonical_sha256(payload),
    }


def terminal_v4_failure_binding() -> dict:
    from . import v4_closeout

    expected_paths = {
        "attempt": V4_ATTEMPT_PATH,
        "failure": V4_FAILURE_PATH,
        "readiness_file": V4_READINESS_PATH,
        "external_freeze_receipt": V4_FREEZE_RECEIPT_PATH,
    }
    actual_hashes = {
        name: frozen_plan.sha256_file(_require_plain_file(path, f"v4 {name}"))
        for name, path in expected_paths.items()
    }
    if actual_hashes != EXPECTED_V4_FILE_SHA256:
        raise ValueError("terminal v4 failure evidence changed")

    attempt = _load_json(V4_ATTEMPT_PATH, "terminal v4 attempt")
    failure = _load_json(V4_FAILURE_PATH, "terminal v4 failure")
    readiness = _load_json(V4_READINESS_PATH, "terminal v4 readiness")
    completion_path = V4_STATE_DIR / frozen_collect.COMPLETION_MARKER
    if (
        attempt.get("schema") != frozen_collect.ATTEMPT_SCHEMA
        or attempt.get("stage") != "locked_transport_collection"
        or attempt.get("readiness_sha256") != TERMINAL_V4_READINESS_SHA256
        or attempt.get("prelock_registry_sha256") != ORIGINAL_PRELOCK_SHA256
        or failure.get("schema") != frozen_collect.FAILURE_SCHEMA
        or failure.get("stage") != "locked_transport_collection_failed"
        or failure.get("readiness_sha256") != TERMINAL_V4_READINESS_SHA256
        or failure.get("attempt_marker_sha256") != actual_hashes["attempt"]
        or failure.get("error_type") != "PermissionError"
        or failure.get("raw_root_exists") is not True
        or failure.get("manifest_exists") is not False
        or failure.get("simulator_process_started") is not True
        or failure.get("locked_response_values_may_have_been_accessed")
        is not True
        or failure.get("retry_permitted_under_same_readiness_digest") is not False
        or readiness.get("readiness_sha256") != TERMINAL_V4_READINESS_SHA256
        or readiness.get("prelock_registry_sha256") != ORIGINAL_PRELOCK_SHA256
    ):
        raise ValueError("terminal v4 failure contract changed")
    if os.path.lexists(completion_path):
        raise ValueError("terminal v4 failure unexpectedly has a completion marker")
    unsigned_readiness = {
        key: value
        for key, value in readiness.items()
        if key != "readiness_sha256"
    }
    if (
        frozen_plan.canonical_sha256(unsigned_readiness)
        != TERMINAL_V4_READINESS_SHA256
    ):
        raise ValueError("terminal v4 readiness no longer self-verifies")
    frozen_external_freeze.validate_external_freeze_receipt(
        V4_FREEZE_RECEIPT_PATH,
        ORIGINAL_PRELOCK_SHA256,
        TERMINAL_V4_READINESS_SHA256,
        prelock_root=PRELOCK_ROOT,
        readiness_path=V4_READINESS_PATH,
        live=False,
    )
    closeout = v4_closeout.validate_closeout(
        validate_live_filesystem_hashes=True
    )
    closeout_payload = closeout["closeout"]
    raw_tree = closeout_payload["incomplete_raw_tree"]
    collection_log = closeout_payload["collection_log"]

    payload = {
        "schema": TERMINAL_BINDING_SCHEMA,
        "prelock_registry_sha256": ORIGINAL_PRELOCK_SHA256,
        "readiness_sha256": TERMINAL_V4_READINESS_SHA256,
        "attempt_file_sha256": actual_hashes["attempt"],
        "failure_file_sha256": actual_hashes["failure"],
        "readiness_file_sha256": actual_hashes["readiness_file"],
        "external_freeze_receipt_file_sha256": actual_hashes[
            "external_freeze_receipt"
        ],
        "terminal_closeout_file_sha256": frozen_plan.sha256_file(
            v4_closeout.OUTPUT
        ),
        "terminal_closeout_payload_sha256": closeout["closeout_sha256"],
        "collection_log_sha256": collection_log["sha256"],
        "incomplete_raw_inventory_canonical_sha256": raw_tree[
            "inventory_canonical_sha256"
        ],
        "attempt_marker_sha256_bound_by_failure": failure[
            "attempt_marker_sha256"
        ],
        "simulator_process_started": failure.get("simulator_process_started"),
        "locked_response_values_may_have_been_accessed": failure.get(
            "locked_response_values_may_have_been_accessed"
        ),
        "retry_permitted_under_same_readiness_digest": False,
        "data_v7_raw_reuse_permitted": False,
    }
    return {
        "binding_sha256": frozen_plan.canonical_sha256(payload),
        "binding": payload,
    }


def _write_copy_exclusive(source: Path, destination: Path) -> None:
    _require_plain_file(source, "frozen plan asset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite staged plan asset: {destination}")
    with source.open("rb") as source_stream, destination.open("xb") as output:
        shutil.copyfileobj(source_stream, output)
        output.flush()
        os.fsync(output.fileno())
    if frozen_plan.sha256_file(source) != frozen_plan.sha256_file(destination):
        raise ValueError(f"staged plan asset hash differs: {destination}")


def stage_frozen_plan_assets(
    *,
    data_root: Path = DATA_ROOT,
    state_root: Path = STATE_ROOT,
    freeze_root: Path = FREEZE_ROOT,
    prelock_root: Path = PRELOCK_ROOT,
) -> Path:
    validate_namespace_separation(
        data_root=data_root,
        state_root=state_root,
        freeze_root=freeze_root,
    )
    _validate_scientific_prelock(prelock_root)
    if os.path.lexists(data_root):
        raise FileExistsError(f"refusing to overwrite data: {data_root}")
    staging = data_root.parent / f".{data_root.name}.plan-staging"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale data plan staging exists: {staging}")
    frozen_plans = prelock_root / "bundle/plans/full"
    frozen_certificate = (
        prelock_root / "bundle/plans/disjointness_certificate.json"
    )
    staging.mkdir(parents=True)
    try:
        for case in sorted(boptest.CASES):
            _write_copy_exclusive(
                frozen_plans / f"{case}.json",
                staging / PLAN_ROOT_RELATIVE / f"{case}.json",
            )
        _write_copy_exclusive(
            frozen_certificate, staging / CERTIFICATE_RELATIVE
        )
        staging.rename(data_root)
        return data_root
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def validate_staged_plan_assets(
    *,
    data_root: Path,
    prelock_root: Path,
    require_plan_only: bool = False,
) -> tuple[dict[str, dict], tuple[Path, ...], dict, dict]:
    registry = _validate_scientific_prelock(prelock_root)
    live_plan_root = _require_plain_directory(
        data_root / PLAN_ROOT_RELATIVE, "data plan root"
    )
    live_certificate = _require_plain_file(
        data_root / CERTIFICATE_RELATIVE, "data certificate"
    )
    expected_relative = {
        "plans",
        "plans/full",
        CERTIFICATE_RELATIVE.as_posix(),
        *{
            (PLAN_ROOT_RELATIVE / f"{case}.json").as_posix()
            for case in sorted(boptest.CASES)
        },
    }
    actual_relative = {
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
    }
    if any(path.is_symlink() for path in data_root.rglob("*")):
        raise ValueError("data plan stage contains a symbolic link")
    if require_plan_only and actual_relative != expected_relative:
        raise ValueError(
            "data must contain only frozen plans and certificate before attempt"
        )
    frozen_plan_root = prelock_root / "bundle/plans/full"
    frozen_certificate = (
        prelock_root / "bundle/plans/disjointness_certificate.json"
    )
    for case in sorted(boptest.CASES):
        live = live_plan_root / f"{case}.json"
        frozen = frozen_plan_root / f"{case}.json"
        if frozen_plan.sha256_file(live) != frozen_plan.sha256_file(frozen):
            raise ValueError(f"data plan differs from prelock: {case}")
    if frozen_plan.sha256_file(live_certificate) != frozen_plan.sha256_file(
        frozen_certificate
    ):
        raise ValueError("data certificate differs from prelock")

    plans, paths = frozen_collect._load_plan_grid(live_plan_root)
    certificate = frozen_worker._load_json_strict(live_certificate)
    certificate_sha256 = certificate.get("certificate_sha256")
    if not isinstance(certificate_sha256, str):
        raise ValueError("data certificate has no canonical digest")
    frozen_worker.validate_certificate_grid(
        certificate, plans, certificate_sha256
    )
    if (
        registry["plans"]["plan_sha256_by_case"]
        != {case: plans[case]["plan_sha256"] for case in sorted(plans)}
        or registry["plans"]["certificate_sha256"] != certificate_sha256
    ):
        raise ValueError("data plan grid differs from the scientific prelock")
    return plans, paths, certificate, registry


def validate_host_identity() -> dict[str, object]:
    uid = os.getuid()
    gid = os.getgid()
    if uid != EXPECTED_UID or gid != EXPECTED_GID:
        raise PermissionError(
            f"host identity is {uid}:{gid}, expected {EXPECTED_UID}:{EXPECTED_GID}"
        )
    return {
        "expected_uid": EXPECTED_UID,
        "expected_gid": EXPECTED_GID,
        "observed_uid": uid,
        "observed_gid": gid,
        "validated": True,
    }


def ownership_probe_command(probe_root: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{probe_root}:/out",
        boptest.WORKER_IMAGE_ID,
        "/bin/bash",
        "-lc",
        (
            "set -eu && mkdir /out/nested && : > /out/nested/probe && "
            f"chown -R {EXPECTED_UID}:{EXPECTED_GID} /out"
        ),
    ]


def ownership_probe_contract_sha256() -> str:
    contract = {
        "schema": "direct-h8-v5-ownership-probe-command-v1",
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "mount": "<isolated-host-probe>:/out",
        "shell": (
            "set -eu && mkdir /out/nested && : > /out/nested/probe && "
            f"chown -R {EXPECTED_UID}:{EXPECTED_GID} /out"
        ),
    }
    return frozen_plan.canonical_sha256(contract)


def ownership_cleanup_command(probe_root: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{probe_root}:/out",
        boptest.WORKER_IMAGE_ID,
        "/bin/bash",
        "-lc",
        f"chown -R {EXPECTED_UID}:{EXPECTED_GID} /out",
    ]


def validate_isolated_ownership_probe(
    *,
    probe_parent: Path,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    identity = validate_host_identity()
    _require_plain_directory(probe_parent, "ownership-probe parent")
    probe = Path(
        tempfile.mkdtemp(
            prefix=".direct_h8_v5_ownership_probe.", dir=probe_parent
        )
    )
    command = ownership_probe_command(probe)
    try:
        command_runner(command, check=True, capture_output=True, text=True)
        paths = (probe, probe / "nested", probe / "nested/probe")
        for path in paths:
            if path.is_symlink() or not path.exists():
                raise ValueError("ownership probe output is incomplete")
            stat_result = path.stat()
            if (
                stat_result.st_uid != EXPECTED_UID
                or stat_result.st_gid != EXPECTED_GID
            ):
                raise PermissionError("ownership probe did not transfer host ownership")
        host_write = probe / "host_write_probe"
        with host_write.open("xb") as stream:
            stream.write(b"host-owned\n")
            stream.flush()
            os.fsync(stream.fileno())
        host_write.unlink()
        return {
            "schema": "direct-h8-v5-ownership-probe-v1",
            "worker_image_id": boptest.WORKER_IMAGE_ID,
            "output_uid": EXPECTED_UID,
            "output_gid": EXPECTED_GID,
            "recursive_paths_checked": 3,
            "host_write_delete_validated": True,
            "command_contract_sha256": ownership_probe_contract_sha256(),
            "validated": True,
            "host_identity": identity,
        }
    finally:
        if probe.exists() and not probe.is_symlink():
            try:
                shutil.rmtree(probe)
            except PermissionError:
                subprocess.run(
                    ownership_cleanup_command(probe),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                shutil.rmtree(probe)


def runtime_readiness_report(
    *,
    testcase_root: Path,
    probe_parent: Path,
    command_runner: Callable[..., object] = subprocess.run,
) -> tuple[dict[str, object], dict[str, str]]:
    public_source = frozen_collect.validate_public_checkout(testcase_root)
    frozen_collect.validate_worker_image()
    frozen_collect.validate_worker_entrypoint()
    ownership = validate_isolated_ownership_probe(
        probe_parent=probe_parent, command_runner=command_runner
    )
    return (
        {
            "worker_image_validated": True,
            "worker_entrypoint_validated": True,
            "host_identity": validate_host_identity(),
            "ownership_probe": ownership,
        },
        public_source,
    )


def _validate_runtime_report(report: Mapping[str, object]) -> None:
    identity = report.get("host_identity")
    probe = report.get("ownership_probe")
    if (
        report.get("worker_image_validated") is not True
        or report.get("worker_entrypoint_validated") is not True
        or not isinstance(identity, dict)
        or identity.get("expected_uid") != EXPECTED_UID
        or identity.get("expected_gid") != EXPECTED_GID
        or identity.get("observed_uid") != EXPECTED_UID
        or identity.get("observed_gid") != EXPECTED_GID
        or identity.get("validated") is not True
        or not isinstance(probe, dict)
        or probe.get("output_uid") != EXPECTED_UID
        or probe.get("output_gid") != EXPECTED_GID
        or probe.get("host_write_delete_validated") is not True
        or probe.get("recursive_paths_checked") != 3
        or probe.get("worker_image_id") != boptest.WORKER_IMAGE_ID
        or probe.get("command_contract_sha256")
        != ownership_probe_contract_sha256()
        or probe.get("validated") is not True
    ):
        raise ValueError("v5 runtime and ownership readiness is incomplete")


def _readiness_report(
    *,
    plans: Mapping[str, Mapping[str, object]],
    certificate: Mapping[str, object],
    certificate_path: Path,
    registry: Mapping[str, object],
    runtime_report: Mapping[str, object],
    public_source: Mapping[str, str],
    terminal_binding: Mapping[str, object],
    prelock_root: Path,
) -> dict:
    _validate_runtime_report(runtime_report)
    frozen_sources = validate_frozen_source_hashes()
    shared_runtime = validate_live_shared_runtime_semantics(
        prelock_root=prelock_root, registry=registry
    )
    code_hashes = operational_code_hashes()
    certificate_sha256 = certificate.get("certificate_sha256")
    payload = {
        "schema": READINESS_SCHEMA,
        "study_kind": frozen_worker.STUDY_KIND,
        "replacement_kind": "ownership_corrected_full_recollection",
        "namespaces": {
            "data": "data",
            "state": "state_v3",
            "freeze": "freeze_v5",
        },
        "plan_sha256_by_case": {
            case: plans[case]["plan_sha256"] for case in sorted(boptest.CASES)
        },
        "certificate_sha256": certificate_sha256,
        "certificate_file_sha256": frozen_plan.sha256_file(certificate_path),
        "prelock_registry_sha256": ORIGINAL_PRELOCK_SHA256,
        "prelock_registry_file_sha256": frozen_plan.sha256_file(
            prelock_root / frozen_prelock.REGISTRY_NAME
        ),
        "prelock_bundle_inventory_sha256": registry[
            "bundle_inventory_sha256"
        ],
        "collection_code_sha256": frozen_sources["collection_code_sha256"],
        "operational_code_sha256": code_hashes,
        "protocol_sha256": code_hashes["PROTOCOL_ADDENDUM.md"],
        "frozen_scientific_protocol_sha256": frozen_plan.sha256_file(
            FROZEN_SOURCE_ROOT / "PROTOCOL.md"
        ),
        "frozen_evaluation_source_sha256": {
            "run_evaluation.py": frozen_sources["run_evaluation.py"],
            "corpus.py": frozen_sources["corpus.py"],
        },
        "live_shared_runtime_validation": shared_runtime,
        "terminal_v4_failure": dict(terminal_binding),
        "source_sha256_by_case": {
            case: plans[case]["source_sha256"] for case in sorted(boptest.CASES)
        },
        "public_source": dict(public_source),
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "worker_boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "step_seconds": boptest.STEP_SECONDS,
        "warmup_seconds": boptest.WARMUP_SECONDS,
        "trajectory_steps": boptest.TRAJECTORY_STEPS,
        "window_count": sum(len(plans[case]["entries"]) for case in boptest.CASES),
        "branch_count": sum(
            len(plans[case]["entries"]) * len(frozen_plan.POLICIES)
            for case in boptest.CASES
        ),
        "runtime_validated": True,
        "worker_image_validated": True,
        "worker_entrypoint_validated": True,
        "host_identity": runtime_report["host_identity"],
        "ownership_probe": runtime_report["ownership_probe"],
        "full_recollection_required": True,
        "data_v7_raw_reuse_permitted": False,
        "locked_response_values_accessed": False,
        "state_created": False,
    }
    return {
        **payload,
        "readiness_sha256": frozen_plan.canonical_sha256(payload),
    }


def prepare_readiness(
    *,
    data_root: Path = DATA_ROOT,
    state_root: Path = STATE_ROOT,
    freeze_root: Path = FREEZE_ROOT,
    prelock_root: Path = PRELOCK_ROOT,
    testcase_root: Path = TESTCASE_ROOT,
    command_runner: Callable[..., object] = subprocess.run,
) -> frozen_collect.Readiness:
    validate_namespace_separation(
        data_root=data_root,
        state_root=state_root,
        freeze_root=freeze_root,
    )
    if os.path.lexists(state_root):
        raise FileExistsError("state_v3 exists before readiness freeze")
    plans, plan_paths, certificate, registry = validate_staged_plan_assets(
        data_root=data_root,
        prelock_root=prelock_root,
        require_plan_only=True,
    )
    for case, case_plan in plans.items():
        frozen_worker._validate_public_source(
            case_plan, boptest.CASES[case], testcase_root
        )
    runtime_report, public_source = runtime_readiness_report(
        testcase_root=testcase_root,
        probe_parent=data_root.parent,
        command_runner=command_runner,
    )
    terminal_binding = terminal_v4_failure_binding()
    report = _readiness_report(
        plans=plans,
        certificate=certificate,
        certificate_path=data_root / CERTIFICATE_RELATIVE,
        registry=registry,
        runtime_report=runtime_report,
        public_source=public_source,
        terminal_binding=terminal_binding,
        prelock_root=prelock_root,
    )
    return frozen_collect.Readiness(
        plans=plans,
        plan_paths=plan_paths,
        certificate=certificate,
        expected_certificate_sha256=str(certificate["certificate_sha256"]),
        collection_code_sha256=dict(FROZEN_COLLECTION_CODE_SHA256),
        report=report,
    )


def write_readiness_report(path: Path, report: Mapping[str, object]) -> Path:
    payload = dict(report)
    digest = payload.get("readiness_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "readiness_sha256"
    }
    _validate_runtime_report(payload)
    if (
        payload.get("schema") != READINESS_SCHEMA
        or payload.get("prelock_registry_sha256") != ORIGINAL_PRELOCK_SHA256
        or payload.get("locked_response_values_accessed") is not False
        or payload.get("state_created") is not False
        or payload.get("full_recollection_required") is not True
        or payload.get("data_v7_raw_reuse_permitted") is not False
        or not boptest.valid_sha256(digest)
        or digest == TERMINAL_V4_READINESS_SHA256
        or frozen_plan.canonical_sha256(unsigned) != digest
    ):
        raise ValueError("v5 readiness report does not self-verify")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v5 readiness: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(frozen_plan.canonical_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path


def load_bound_readiness(
    *,
    prelock_root: Path,
    live_data_root: Path,
    readiness_path: Path,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
) -> tuple[dict, frozen_collect.Readiness]:
    if expected_prelock_sha256 != ORIGINAL_PRELOCK_SHA256:
        raise ValueError("expected prelock is not the original scientific lock")
    if not boptest.valid_sha256(expected_readiness_sha256):
        raise ValueError("expected v5 readiness digest is not a SHA-256")
    if expected_readiness_sha256 == TERMINAL_V4_READINESS_SHA256:
        raise ValueError("terminal v4 readiness digest cannot be used by v5")
    plans, plan_paths, certificate, registry = validate_staged_plan_assets(
        data_root=live_data_root, prelock_root=prelock_root
    )
    report = _load_json(readiness_path, "v5 collection readiness")
    unsigned = {
        key: value for key, value in report.items() if key != "readiness_sha256"
    }
    current_terminal = terminal_v4_failure_binding()
    current_code = operational_code_hashes()
    frozen_sources = validate_frozen_source_hashes()
    shared_runtime = validate_live_shared_runtime_semantics(
        prelock_root=prelock_root, registry=registry
    )
    expected_plan_hashes = {
        case: plans[case]["plan_sha256"] for case in sorted(plans)
    }
    if (
        report.get("schema") != READINESS_SCHEMA
        or report.get("readiness_sha256") != expected_readiness_sha256
        or frozen_plan.canonical_sha256(unsigned) != expected_readiness_sha256
        or report.get("prelock_registry_sha256") != ORIGINAL_PRELOCK_SHA256
        or frozen_prelock.canonical_sha256(registry) != ORIGINAL_PRELOCK_SHA256
        or report.get("plan_sha256_by_case") != expected_plan_hashes
        or report.get("certificate_sha256")
        != certificate.get("certificate_sha256")
        or report.get("certificate_file_sha256")
        != frozen_plan.sha256_file(live_data_root / CERTIFICATE_RELATIVE)
        or report.get("collection_code_sha256")
        != frozen_sources["collection_code_sha256"]
        or report.get("operational_code_sha256") != current_code
        or report.get("protocol_sha256")
        != current_code["PROTOCOL_ADDENDUM.md"]
        or report.get("frozen_evaluation_source_sha256")
        != {
            "run_evaluation.py": frozen_sources["run_evaluation.py"],
            "corpus.py": frozen_sources["corpus.py"],
        }
        or report.get("live_shared_runtime_validation") != shared_runtime
        or report.get("terminal_v4_failure") != current_terminal
        or report.get("namespaces")
        != {"data": "data", "state": "state_v3", "freeze": "freeze_v5"}
        or report.get("full_recollection_required") is not True
        or report.get("data_v7_raw_reuse_permitted") is not False
        or report.get("locked_response_values_accessed") is not False
        or report.get("state_created") is not False
    ):
        raise ValueError("v5 readiness differs from the live bound contract")
    _validate_runtime_report(report)
    validate_host_identity()
    return registry, frozen_collect.Readiness(
        plans=plans,
        plan_paths=plan_paths,
        certificate=certificate,
        expected_certificate_sha256=str(certificate["certificate_sha256"]),
        collection_code_sha256=dict(FROZEN_COLLECTION_CODE_SHA256),
        report=report,
    )


def docker_worker_command(
    plan_path: Path,
    staging_raw: Path,
    readiness: frozen_collect.Readiness,
    *,
    plan_root: Path,
    certificate_path: Path,
    testcase_root: Path,
) -> list[str]:
    command = frozen_collect._docker_command(
        plan_path,
        staging_raw,
        readiness,
        plan_root=plan_root,
        certificate_path=certificate_path,
        testcase_root=testcase_root,
    )
    command[-1] = (
        command[-1] + f" && chown -R {EXPECTED_UID}:{EXPECTED_GID} /out"
    )
    return command


def _state_dir(state_root: Path, readiness_sha256: str) -> Path:
    if not boptest.valid_sha256(readiness_sha256):
        raise ValueError("v5 readiness digest is not a SHA-256")
    return state_root / readiness_sha256


def _write_state_once(state_root: Path, path: Path, payload: object) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    if state_root.is_symlink() or not state_root.is_dir():
        raise ValueError("state_v3 root is not a plain directory")
    if path.parent.parent.resolve() != state_root.resolve():
        raise ValueError("v5 state evidence leaves the digest-scoped root")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("v5 digest state is not a plain directory")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v5 state evidence: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"stale v5 state temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(frozen_plan.canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_no_prior_attempt(
    state_root: Path, readiness_sha256: str
) -> Path:
    state_dir = _state_dir(state_root, readiness_sha256)
    if state_root.exists() and (
        state_root.is_symlink() or not state_root.is_dir()
    ):
        raise ValueError("state_v3 root is not a plain directory")
    if state_dir.exists():
        if state_dir.is_symlink() or not state_dir.is_dir():
            raise ValueError("v5 digest state is not a plain directory")
        entries = sorted(path.name for path in state_dir.iterdir())
        if entries:
            raise FileExistsError(
                "this v5 readiness digest is terminal after its first attempt: "
                + ", ".join(entries)
            )
    return state_dir


def _operational_binding(readiness: frozen_collect.Readiness) -> dict:
    return {
        "operational_code_sha256": readiness.report[
            "operational_code_sha256"
        ],
        "terminal_v4_failure_binding_sha256": readiness.report[
            "terminal_v4_failure"
        ]["binding_sha256"],
        "data_namespace": "data",
        "state_namespace": "state_v3",
        "freeze_namespace": "freeze_v5",
        "full_recollection": True,
        "data_v7_raw_reused": False,
        "expected_output_owner": {"uid": EXPECTED_UID, "gid": EXPECTED_GID},
    }


def _attempt_payload(
    readiness: frozen_collect.Readiness,
    external_freeze: Mapping[str, object],
    *,
    external_freeze_receipt_path: Path,
    commands: Sequence[Sequence[str]],
    plan_root: Path,
    certificate_path: Path,
    raw_root: Path,
    manifest_path: Path,
    staging: Path,
) -> dict:
    payload = frozen_collect._attempt_payload(
        readiness,
        external_freeze,
        external_freeze_receipt_path=external_freeze_receipt_path,
        commands=commands,
        plan_root=plan_root,
        certificate_path=certificate_path,
        raw_root=raw_root,
        manifest_path=manifest_path,
        staging=staging,
    )
    return {**payload, **_operational_binding(readiness)}


def _failure_payload(
    *,
    readiness: frozen_collect.Readiness,
    attempt_path: Path,
    error: BaseException,
    simulator_process_started: bool,
    staging: Path,
    raw_root: Path,
    manifest_path: Path,
) -> dict:
    payload = frozen_collect._failure_payload(
        readiness=readiness,
        attempt_path=attempt_path,
        error=error,
        simulator_process_started=simulator_process_started,
        staging=staging,
        raw_root=raw_root,
        manifest_path=manifest_path,
    )
    return {**payload, **_operational_binding(readiness)}


def _completion_payload(
    *,
    readiness: frozen_collect.Readiness,
    attempt_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict:
    payload = frozen_collect._completion_payload(
        readiness=readiness,
        attempt_path=attempt_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    return {**payload, **_operational_binding(readiness)}


def _validate_output_ownership(root: Path) -> None:
    _require_plain_directory(root, "v5 staging output")
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_symlink():
            raise ValueError("v5 staging output contains a symbolic link")
        stat_result = path.stat()
        if stat_result.st_uid != EXPECTED_UID or stat_result.st_gid != EXPECTED_GID:
            raise PermissionError(f"v5 staging output has wrong owner: {path}")


def run_collection(
    *,
    expected_readiness_sha256: str,
    confirmation: str,
    data_root: Path = DATA_ROOT,
    state_root: Path = STATE_ROOT,
    freeze_root: Path = FREEZE_ROOT,
    prelock_root: Path = PRELOCK_ROOT,
    readiness_path: Path = READINESS_PATH,
    external_freeze_receipt_path: Path = EXTERNAL_FREEZE_RECEIPT,
    testcase_root: Path = TESTCASE_ROOT,
    command_runner: Callable[..., object] = subprocess.run,
    readiness_loader: Callable[..., tuple[dict, frozen_collect.Readiness]]
    | None = None,
    freeze_validator: Callable[..., dict] | None = None,
) -> dict:
    if expected_readiness_sha256 == TERMINAL_V4_READINESS_SHA256:
        raise ValueError("terminal v4 readiness digest cannot be retried")
    if confirmation != CONFIRMATION_TOKEN:
        raise ValueError("formal v5 collection requires the exact confirmation token")
    validate_namespace_separation(
        data_root=data_root,
        state_root=state_root,
        freeze_root=freeze_root,
    )
    validate_host_identity()
    loader = load_bound_readiness if readiness_loader is None else readiness_loader
    _, readiness = loader(
        prelock_root=prelock_root,
        live_data_root=data_root,
        readiness_path=readiness_path,
        expected_prelock_sha256=ORIGINAL_PRELOCK_SHA256,
        expected_readiness_sha256=expected_readiness_sha256,
    )
    if freeze_validator is None:
        from .external_freeze import validate_external_freeze_receipt

        freeze_validator = validate_external_freeze_receipt
    external_freeze = freeze_validator(
        external_freeze_receipt_path,
        ORIGINAL_PRELOCK_SHA256,
        expected_readiness_sha256,
        prelock_root=prelock_root,
        readiness_path=readiness_path,
        live=True,
    )

    state_dir = _assert_no_prior_attempt(state_root, expected_readiness_sha256)
    plan_root = data_root / PLAN_ROOT_RELATIVE
    certificate_path = data_root / CERTIFICATE_RELATIVE
    raw_root = data_root / RAW_RELATIVE
    manifest_path = data_root / MANIFEST_RELATIVE
    staging = raw_root.parent / f".{raw_root.name}.staging"
    pending = manifest_path.with_suffix(manifest_path.suffix + ".pending")
    commands = [
        docker_worker_command(
            plan_path,
            staging,
            readiness,
            plan_root=plan_root,
            certificate_path=certificate_path,
            testcase_root=testcase_root,
        )
        for plan_path in readiness.plan_paths
    ]
    attempt_path = state_dir / ATTEMPT_MARKER
    failure_path = state_dir / FAILURE_MARKER
    completion_path = state_dir / COMPLETION_MARKER
    _write_state_once(
        state_root,
        attempt_path,
        _attempt_payload(
            readiness,
            external_freeze,
            external_freeze_receipt_path=external_freeze_receipt_path,
            commands=commands,
            plan_root=plan_root,
            certificate_path=certificate_path,
            raw_root=raw_root,
            manifest_path=manifest_path,
            staging=staging,
        ),
    )
    simulator_process_started = False
    try:
        checked = frozen_collect._preflight_destination(raw_root, manifest_path)
        if checked != (raw_root, staging, manifest_path, pending):
            raise ValueError("frozen destination preflight returned different paths")
        staging.mkdir(parents=True)
        for command in commands:
            simulator_process_started = True
            command_runner(command, check=True)
        _validate_output_ownership(staging)
        if operational_code_hashes() != readiness.report[
            "operational_code_sha256"
        ]:
            raise ValueError("v5 operational code changed during collection")
        if frozen_collect.collection_code_hashes() != (
            readiness.collection_code_sha256
        ):
            raise ValueError("frozen worker code changed during collection")
        post_worker_registry = _validate_scientific_prelock(prelock_root)
        post_worker_shared = validate_live_shared_runtime_semantics(
            prelock_root=prelock_root,
            registry=post_worker_registry,
        )
        if post_worker_shared != readiness.report[
            "live_shared_runtime_validation"
        ]:
            raise ValueError("shared numerical runtime changed during collection")
        inventory = frozen_collect.validate_staged_collection(staging, readiness)
        manifest = frozen_collect.build_manifest(readiness, inventory)
        frozen_collect._publish(
            staging, raw_root, manifest_path, pending, manifest
        )
        _write_state_once(
            state_root,
            completion_path,
            _completion_payload(
                readiness=readiness,
                attempt_path=attempt_path,
                manifest_path=manifest_path,
                manifest=manifest,
            ),
        )
        return manifest
    except BaseException as error:
        try:
            _write_state_once(
                state_root,
                failure_path,
                _failure_payload(
                    readiness=readiness,
                    attempt_path=attempt_path,
                    error=error,
                    simulator_process_started=simulator_process_started,
                    staging=staging,
                    raw_root=raw_root,
                    manifest_path=manifest_path,
                ),
            )
        except BaseException as marker_error:
            raise marker_error from error
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "readiness", "collect"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--freeze-root", type=Path, default=FREEZE_ROOT)
    parser.add_argument("--prelock-root", type=Path, default=PRELOCK_ROOT)
    parser.add_argument("--readiness", type=Path, default=READINESS_PATH)
    parser.add_argument(
        "--external-freeze-receipt",
        type=Path,
        default=EXTERNAL_FREEZE_RECEIPT,
    )
    parser.add_argument("--testcase-root", type=Path, default=TESTCASE_ROOT)
    parser.add_argument("--expected-readiness-sha256")
    parser.add_argument("--confirm")
    parser.add_argument("--write-readiness", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    common = {
        "data_root": args.data_root.resolve(),
        "state_root": args.state_root.resolve(),
        "freeze_root": args.freeze_root.resolve(),
        "prelock_root": args.prelock_root.resolve(),
    }
    if args.command == "stage":
        if (
            args.expected_readiness_sha256 is not None
            or args.confirm is not None
            or args.write_readiness
        ):
            raise ValueError("stage received collection-only arguments")
        print(stage_frozen_plan_assets(**common))
        return
    if args.command == "readiness":
        if args.expected_readiness_sha256 is not None or args.confirm is not None:
            raise ValueError("readiness received collection-only arguments")
        readiness = prepare_readiness(
            **common,
            testcase_root=args.testcase_root.resolve(),
        )
        if args.write_readiness:
            write_readiness_report(args.readiness.resolve(), readiness.report)
        print(json.dumps(readiness.report, indent=2, sort_keys=True))
        return
    if not args.expected_readiness_sha256:
        raise ValueError("collect requires --expected-readiness-sha256")
    result = run_collection(
        **common,
        expected_readiness_sha256=args.expected_readiness_sha256,
        confirmation=args.confirm,
        readiness_path=args.readiness.resolve(),
        external_freeze_receipt_path=args.external_freeze_receipt.resolve(),
        testcase_root=args.testcase_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
