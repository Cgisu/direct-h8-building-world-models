"""Validate, stage, and atomically publish the paired v3 BOPTEST corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from building_fault_wm.neural_benchmark import protocol as boptest

from . import plan as v3_plan
from . import worker_collect


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CANONICAL_DATA_ROOT = (
    PROJECT_ROOT / "building_fault_wm/neural_benchmark/data_v7"
)
CANONICAL_PLAN_ROOT = CANONICAL_DATA_ROOT / "plans/full"
CANONICAL_CERTIFICATE = (
    CANONICAL_DATA_ROOT / "disjointness_certificate.json"
)
CANONICAL_RAW = CANONICAL_DATA_ROOT / "locked_transport_raw"
CANONICAL_MANIFEST = (
    CANONICAL_DATA_ROOT / "manifests/locked_transport_corpus_manifest.json"
)
CANONICAL_TESTCASE_ROOT = (Path.home() / "external/project1-boptest/testcases")
STATE_ROOT = HERE / ".direct_h8_v3_locked_state_v2"
CANONICAL_PRELOCK_ROOT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_prelock_v4"
)
CANONICAL_FREEZE_ROOT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_deterministic_transport_v3_external_freeze_v4"
)
CANONICAL_READINESS = CANONICAL_FREEZE_ROOT / "collection_readiness.json"
CANONICAL_EXTERNAL_FREEZE_RECEIPT = (
    CANONICAL_FREEZE_ROOT / "external_freeze_receipt.json"
)

CONFIRMATION_TOKEN = (
    "I_UNDERSTAND_DIRECT_H8_V3_PAIRED_COLLECTION_OPENS_LOCKED_OUTCOMES"
)
READINESS_SCHEMA = "direct-h8-transport-collection-readiness-v3"
CORPUS_MANIFEST_SCHEMA = "direct-h8-transport-corpus-manifest-v1"
ATTEMPT_SCHEMA = "direct-h8-transport-collection-attempt-v1"
FAILURE_SCHEMA = "direct-h8-transport-collection-failure-v1"
COMPLETION_SCHEMA = "direct-h8-transport-collection-completion-v1"
ATTEMPT_MARKER = "v3_paired_collection_attempt.json"
FAILURE_MARKER = "v3_paired_collection_failure.json"
COMPLETION_MARKER = "v3_paired_collection_completion.json"


@dataclass(frozen=True)
class Readiness:
    plans: dict[str, dict]
    plan_paths: tuple[Path, ...]
    certificate: dict
    expected_certificate_sha256: str
    collection_code_sha256: dict[str, str]
    report: dict


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def collection_code_hashes() -> dict[str, str]:
    return {
        name: v3_plan.sha256_file(HERE / name)
        for name in ("plan.py", "worker_collect.py", "collect.py")
    }


def _require_plain_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")


def _require_plain_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")


def _load_plan_grid(root: Path) -> tuple[dict[str, dict], tuple[Path, ...]]:
    _require_plain_directory(root, "v3 plan root")
    paths = tuple(sorted(root.iterdir()))
    expected = {f"{case}.json" for case in boptest.CASES}
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("v3 plan root contains a non-file or symbolic link")
    if {path.name for path in paths} != expected:
        raise ValueError("v3 plan root differs from the three-case contract")
    plans = {
        case: worker_collect._load_json_strict(root / f"{case}.json")
        for case in sorted(boptest.CASES)
    }
    for case, plan in plans.items():
        v3_plan.validate_case_plan(plan)
        if plan.get("case") != case:
            raise ValueError(f"v3 plan file identity differs for {case}")
    return plans, paths


def validate_public_checkout(testcase_root: Path) -> dict[str, str]:
    repository = testcase_root.parent
    _require_plain_directory(repository, "public BOPTEST checkout")
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != boptest.BOPTEST_COMMIT:
        raise ValueError("BOPTEST checkout differs from the frozen commit")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("BOPTEST public-source checkout is dirty")
    remote = subprocess.run(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote != boptest.BOPTEST_REPOSITORY_URL:
        raise ValueError("BOPTEST origin differs from the frozen public source")
    license_sha256 = v3_plan.sha256_file(
        repository / boptest.BOPTEST_LICENSE_PATH
    )
    if license_sha256 != boptest.BOPTEST_LICENSE_SHA256:
        raise ValueError("BOPTEST license differs from the frozen public source")
    return {
        "repository_url": remote,
        "commit": head,
        "license_sha256": license_sha256,
    }


def validate_worker_image() -> str:
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            boptest.WORKER_IMAGE_ID,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if result != boptest.WORKER_IMAGE_ID:
        raise ValueError("local BOPTEST image differs from the pinned image ID")
    return result


def validate_worker_entrypoint() -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{PROJECT_ROOT}:/workspace:ro",
        "-e",
        "PYTHONPATH=/workspace",
        boptest.WORKER_IMAGE_ID,
        "/bin/bash",
        "-lc",
        (
            ". /miniconda/bin/activate && conda activate pyfmi3 && "
            "python /workspace/building_fault_wm/"
            "direct_h8_deterministic_transport_v3/worker_collect.py --help"
        ),
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _readiness_payload(
    plans: Mapping[str, Mapping[str, object]],
    expected_certificate_sha256: str,
    *,
    runtime_validated: bool,
    public_source: Mapping[str, str],
    prelock_binding: Mapping[str, str],
) -> dict:
    return {
        "schema": READINESS_SCHEMA,
        "study_kind": worker_collect.STUDY_KIND,
        "plan_sha256_by_case": {
            case: plans[case]["plan_sha256"] for case in sorted(boptest.CASES)
        },
        "certificate_sha256": expected_certificate_sha256,
        "prelock_registry_sha256": prelock_binding[
            "prelock_registry_sha256"
        ],
        "prelock_registry_file_sha256": prelock_binding[
            "prelock_registry_file_sha256"
        ],
        "prelock_bundle_inventory_sha256": prelock_binding[
            "prelock_bundle_inventory_sha256"
        ],
        "collection_code_sha256": collection_code_hashes(),
        "protocol_sha256": v3_plan.sha256_file(HERE / "PROTOCOL.md"),
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
            len(plans[case]["entries"]) * len(v3_plan.POLICIES)
            for case in boptest.CASES
        ),
        "runtime_validated": runtime_validated,
        "worker_entrypoint_validated": runtime_validated,
        "locked_response_values_accessed": False,
        "state_created": False,
    }


def _validated_prelock_binding(prelock_root: Path) -> dict[str, str]:
    # Lazy import avoids the prelock -> collect dependency during module import.
    from . import prelock

    registry = prelock.validate_prelock_bundle(
        prelock_root,
        parent_root=prelock.DEFAULT_PARENT_ROOT,
        source_root=HERE,
    )
    digest = prelock.canonical_sha256(registry)
    digest_record = prelock_root / prelock.DIGEST_NAME
    if digest_record.read_text(encoding="ascii") != f"{digest}\n":
        raise ValueError("v3 prelock digest record changed")
    inventory_digest = registry.get("bundle_inventory_sha256")
    if not boptest.valid_sha256(inventory_digest):
        raise ValueError("v3 prelock bundle inventory digest is invalid")
    return {
        "prelock_registry_sha256": digest,
        "prelock_registry_file_sha256": v3_plan.sha256_file(
            prelock_root / prelock.REGISTRY_NAME
        ),
        "prelock_bundle_inventory_sha256": str(inventory_digest),
    }


def prepare_readiness(
    expected_certificate_sha256: str,
    *,
    plan_root: Path = CANONICAL_PLAN_ROOT,
    certificate_path: Path = CANONICAL_CERTIFICATE,
    testcase_root: Path = CANONICAL_TESTCASE_ROOT,
    prelock_root: Path = CANONICAL_PRELOCK_ROOT,
    validate_runtime: bool = False,
    prelock_binding: Mapping[str, str] | None = None,
) -> Readiness:
    if not boptest.valid_sha256(expected_certificate_sha256):
        raise ValueError("expected v3 certificate digest is not a SHA-256")
    plans, plan_paths = _load_plan_grid(plan_root)
    _require_plain_file(certificate_path, "v3 disjointness certificate")
    certificate = worker_collect._load_json_strict(certificate_path)
    worker_collect.validate_certificate_grid(
        certificate, plans, expected_certificate_sha256
    )
    public_source = validate_public_checkout(testcase_root)
    for case, plan in plans.items():
        adapter = boptest.CASES[case]
        worker_collect._validate_public_source(plan, adapter, testcase_root)
    if validate_runtime:
        validate_worker_image()
        validate_worker_entrypoint()
    binding = (
        _validated_prelock_binding(prelock_root)
        if prelock_binding is None
        else dict(prelock_binding)
    )
    expected_binding_fields = {
        "prelock_registry_sha256",
        "prelock_registry_file_sha256",
        "prelock_bundle_inventory_sha256",
    }
    if set(binding) != expected_binding_fields or any(
        not boptest.valid_sha256(value) for value in binding.values()
    ):
        raise ValueError("v3 readiness prelock binding is invalid")
    state_absent = not os.path.lexists(STATE_ROOT)
    if not state_absent:
        raise FileExistsError("v3 collection state exists before readiness freeze")

    # Bind the actual caller-supplied certificate path in parameterized uses.
    payload = {
        **_readiness_payload(
            plans,
            expected_certificate_sha256,
            runtime_validated=validate_runtime,
            public_source=public_source,
            prelock_binding=binding,
        ),
        "certificate_file_sha256": v3_plan.sha256_file(certificate_path),
    }
    report = {
        **payload,
        "readiness_sha256": v3_plan.canonical_sha256(payload),
    }
    return Readiness(
        plans=plans,
        plan_paths=plan_paths,
        certificate=certificate,
        expected_certificate_sha256=expected_certificate_sha256,
        collection_code_sha256=collection_code_hashes(),
        report=report,
    )


def readiness_report(
    expected_certificate_sha256: str,
    *,
    plan_root: Path = CANONICAL_PLAN_ROOT,
    certificate_path: Path = CANONICAL_CERTIFICATE,
    testcase_root: Path = CANONICAL_TESTCASE_ROOT,
    prelock_root: Path = CANONICAL_PRELOCK_ROOT,
    validate_runtime: bool = False,
    prelock_binding: Mapping[str, str] | None = None,
) -> dict:
    return prepare_readiness(
        expected_certificate_sha256,
        plan_root=plan_root,
        certificate_path=certificate_path,
        testcase_root=testcase_root,
        prelock_root=prelock_root,
        validate_runtime=validate_runtime,
        prelock_binding=prelock_binding,
    ).report


def write_readiness_report(path: Path, report: Mapping[str, object]) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v3 readiness: {path}")
    payload = dict(report)
    digest = payload.get("readiness_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "readiness_sha256"
    }
    if (
        not boptest.valid_sha256(digest)
        or v3_plan.canonical_sha256(unsigned) != digest
    ):
        raise ValueError("v3 readiness report does not self-verify")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(v3_plan.canonical_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path


def _docker_command(
    plan_path: Path,
    staging_raw: Path,
    readiness: Readiness,
    *,
    plan_root: Path = CANONICAL_PLAN_ROOT,
    certificate_path: Path = CANONICAL_CERTIFICATE,
    testcase_root: Path = CANONICAL_TESTCASE_ROOT,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--tmpfs",
        "/work:size=1g",
        "--tmpfs",
        "/tmp:exec,size=2g",
        "-w",
        "/work",
        "-v",
        f"{testcase_root.parent}:/public-boptest:ro",
        "-v",
        f"{PROJECT_ROOT}:/workspace:ro",
        "-v",
        f"{plan_root}:/v3-plans:ro",
        "-v",
        f"{certificate_path}:/certificate/disjointness_certificate.json:ro",
        "-v",
        f"{staging_raw}:/out",
        "-e",
        (
            "PYTHONPATH=/workspace:/boptest:/boptest/lib:/boptest/lib/kpis:"
            "/boptest/lib/forecast:/boptest/lib/data"
        ),
        boptest.WORKER_IMAGE_ID,
        "/bin/bash",
        "-lc",
        (
            ". /miniconda/bin/activate && conda activate pyfmi3 && "
            "cp /version.txt /work/version.txt && "
            "python /workspace/building_fault_wm/"
            "direct_h8_deterministic_transport_v3/worker_collect.py "
            f"--plan /v3-plans/{plan_path.name} --plan-root /v3-plans "
            "--disjointness-certificate "
            "/certificate/disjointness_certificate.json "
            f"--expected-certificate-sha256 "
            f"{readiness.expected_certificate_sha256} "
            "--output /out --testcase-root /public-boptest/testcases "
            f"--worker-image-id {boptest.WORKER_IMAGE_ID} "
            "--boptest-version-file /work/version.txt"
        ),
    ]


def _preflight_destination(
    raw_root: Path = CANONICAL_RAW,
    manifest_path: Path = CANONICAL_MANIFEST,
) -> tuple[Path, Path, Path, Path]:
    staging = raw_root.parent / f".{raw_root.name}.staging"
    pending = manifest_path.with_suffix(manifest_path.suffix + ".pending")
    occupied = [
        path
        for path in (raw_root, staging, manifest_path, pending)
        if os.path.lexists(path)
    ]
    if occupied:
        raise FileExistsError(
            "refusing v3 collection because a destination exists: "
            + ", ".join(str(path) for path in occupied)
        )
    return raw_root, staging, manifest_path, pending


def _validate_csv(
    path: Path,
    entry: Mapping[str, object],
    policy: str,
    adapter: boptest.CaseAdapter,
) -> dict:
    _require_plain_file(path, "v3 trajectory")
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != worker_collect.FIELDS:
            raise ValueError(f"v3 trajectory field schema mismatch: {path}")
        rows = list(reader)
    if len(rows) != boptest.TRAJECTORY_STEPS:
        raise ValueError(f"v3 trajectory row count is invalid: {path}")

    metadata = entry["policies"][policy]  # type: ignore[index]
    actions = np.asarray(metadata["action_levels"], dtype=float)
    numeric = worker_collect.FIELDS[5:]
    continuity = (
        ("next_zone_temperature_k", "zone_temperature_k"),
        ("next_hvac_electric_power_w", "hvac_electric_power_w"),
        ("next_auxiliary_1", "auxiliary_1"),
        ("next_auxiliary_2", "auxiliary_2"),
        ("next_outdoor_temperature_k", "outdoor_temperature_k"),
        (
            "next_global_horizontal_solar_w_m2",
            "global_horizontal_solar_w_m2",
        ),
        ("next_comfort_lower_k", "comfort_lower_k"),
        ("next_comfort_upper_k", "comfort_upper_k"),
        ("next_electricity_price", "electricity_price"),
    )
    for index, row in enumerate(rows):
        if row["case"] != adapter.case or row["role"] != worker_collect.OUTPUT_ROLE:
            raise ValueError(f"v3 trajectory identity mismatch: {path}")
        if int(row["day"]) != int(entry["day"]):
            raise ValueError(f"v3 trajectory day mismatch: {path}")
        if int(row["trajectory_seed"]) != int(metadata["trajectory_seed"]):
            raise ValueError(f"v3 trajectory identity seed mismatch: {path}")
        if int(row["step"]) != index:
            raise ValueError(f"v3 trajectory step mismatch: {path}")
        expected_time = (
            int(entry["day"]) * v3_plan.DAY_SECONDS
            + index * boptest.STEP_SECONDS
        )
        if float(row["time_s"]) != float(expected_time):
            raise ValueError(f"v3 trajectory timestamp mismatch: {path}")
        if float(row["normalized_action"]) != float(actions[index]):
            raise ValueError(f"v3 trajectory action mismatch: {path}")
        expected_setpoint = (
            adapter.base_setpoint_k
            + adapter.action_amplitude_k * float(actions[index])
        )
        if float(row["setpoint_k"]) != expected_setpoint:
            raise ValueError(f"v3 trajectory setpoint mismatch: {path}")
        if any(not math.isfinite(float(row[field])) for field in numeric):
            raise ValueError(f"v3 trajectory contains a non-finite value: {path}")
        if index:
            previous = rows[index - 1]
            for previous_name, current_name in continuity:
                if float(previous[previous_name]) != float(row[current_name]):
                    raise ValueError(f"v3 trajectory continuity mismatch: {path}")
    return {
        "path": str(path),
        "sha256": v3_plan.sha256_file(path),
        "rows": len(rows),
        "fields": len(worker_collect.FIELDS),
        "policy": policy,
        "day": int(entry["day"]),
        "scenario_seed": int(entry["scenario_seed"]),
        "trajectory_seed": int(metadata["trajectory_seed"]),
        "action_sha256": metadata["action_sha256"],
    }


def _load_worker_receipt(path: Path) -> dict:
    _require_plain_file(path, "v3 worker receipt")
    wrapper = v3_plan.load_json(path)
    if set(wrapper) != {"receipt_sha256", "receipt"}:
        raise ValueError("v3 worker receipt wrapper fields are invalid")
    receipt = wrapper["receipt"]
    if (
        not isinstance(receipt, dict)
        or wrapper["receipt_sha256"] != v3_plan.canonical_sha256(receipt)
    ):
        raise ValueError("v3 worker receipt self-hash is invalid")
    return receipt


def _validate_worker_receipt(
    path: Path,
    plan: Mapping[str, object],
    readiness: Readiness,
    expected_files: Sequence[Mapping[str, object]],
) -> dict:
    receipt = _load_worker_receipt(path)
    expected = {
        "schema": worker_collect.WORKER_RECEIPT_SCHEMA,
        "study_kind": worker_collect.STUDY_KIND,
        "collection_kind": "paired_locked_transport",
        "output_role": worker_collect.OUTPUT_ROLE,
        "row_schema": worker_collect.ROW_SCHEMA,
        "fields": list(worker_collect.FIELDS),
        "case": plan["case"],
        "plan_sha256": plan["plan_sha256"],
        "disjointness_certificate_sha256": readiness.expected_certificate_sha256,
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "boptest_commit": boptest.BOPTEST_COMMIT,
        "worker_code_sha256": readiness.collection_code_sha256,
        "source_sha256": plan["source_sha256"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"v3 worker receipt {field} differs")
    received_files = receipt.get("files")
    if not isinstance(received_files, list) or len(received_files) != len(
        expected_files
    ):
        raise ValueError("v3 worker receipt file grid is invalid")
    received_by_path = {
        item.get("path"): item
        for item in received_files
        if isinstance(item, dict)
    }
    expected_by_path = {item["path"]: item for item in expected_files}
    if set(received_by_path) != set(expected_by_path):
        raise ValueError("v3 worker receipt file paths differ")
    for relative, expected_file in expected_by_path.items():
        item = received_by_path[relative]
        if item.get("sha256") != expected_file["sha256"]:
            raise ValueError("v3 worker receipt file hash differs")
        if item.get("rows") != boptest.TRAJECTORY_STEPS:
            raise ValueError("v3 worker receipt row count differs")

    pairs = receipt.get("pairs")
    entries = plan.get("entries")
    if not isinstance(pairs, list) or not isinstance(entries, list):
        raise ValueError("v3 worker receipt pair grid is invalid")
    if len(pairs) != len(entries):
        raise ValueError("v3 worker receipt pair count differs")
    by_window = {
        item.get("window_id"): item for item in pairs if isinstance(item, dict)
    }
    if set(by_window) != {entry["window_id"] for entry in entries}:
        raise ValueError("v3 worker receipt window identities differ")
    for entry in entries:
        pair = by_window[entry["window_id"]]
        if pair.get("scenario_seed") != entry["scenario_seed"]:
            raise ValueError("v3 worker receipt scenario seed differs")
        if pair.get("day") != entry["day"]:
            raise ValueError("v3 worker receipt day differs")
        branches = pair.get("branches")
        if not isinstance(branches, dict) or set(branches) != set(
            v3_plan.POLICIES
        ):
            raise ValueError("v3 worker receipt branch grid differs")
        state_hash = pair.get("initialized_state_sha256")
        forecast_hash = pair.get("full_forecast_sha256")
        if not boptest.valid_sha256(state_hash) or not boptest.valid_sha256(
            forecast_hash
        ):
            raise ValueError("v3 worker receipt pair hashes are invalid")
        for policy in v3_plan.POLICIES:
            branch = branches[policy]
            metadata = entry["policies"][policy]
            relative = (
                f"{plan['case']}/"
                f"{worker_collect.expected_filename(entry, policy)}"
            )
            expected_file = expected_by_path[relative]
            if branch.get("trajectory_seed") != metadata["trajectory_seed"]:
                raise ValueError("v3 worker receipt trajectory seed differs")
            if branch.get("action_seed") != metadata["action_seed"]:
                raise ValueError("v3 worker receipt action seed differs")
            if branch.get("action_sha256") != metadata["action_sha256"]:
                raise ValueError("v3 worker receipt action hash differs")
            if branch.get("path") != relative:
                raise ValueError("v3 worker receipt branch path differs")
            if branch.get("sha256") != expected_file["sha256"]:
                raise ValueError("v3 worker receipt branch file hash differs")
            if branch.get("rows") != boptest.TRAJECTORY_STEPS:
                raise ValueError("v3 worker receipt branch row count differs")
            if branch.get("initialized_state_sha256") != state_hash:
                raise ValueError("v3 worker receipt paired state hashes differ")
            if branch.get("full_forecast_sha256") != forecast_hash:
                raise ValueError("v3 worker receipt paired forecast hashes differ")
    return receipt


def validate_staged_collection(
    staging_raw: Path,
    readiness: Readiness,
) -> dict:
    files: list[dict] = []
    receipts: list[dict] = []
    expected_paths: set[Path] = set()
    for case in sorted(boptest.CASES):
        plan = readiness.plans[case]
        adapter = boptest.CASES[case]
        case_files: list[dict] = []
        for entry in plan["entries"]:
            for policy in v3_plan.POLICIES:
                path = staging_raw / case / worker_collect.expected_filename(
                    entry, policy
                )
                expected_paths.add(path)
                validated = _validate_csv(path, entry, policy, adapter)
                relative = str(path.relative_to(staging_raw))
                validated["path"] = relative
                files.append(validated)
                case_files.append(validated)
        receipt_path = worker_collect.receipt_path(staging_raw, plan)
        expected_paths.add(receipt_path)
        receipt = _validate_worker_receipt(
            receipt_path, plan, readiness, case_files
        )
        receipts.append(
            {
                "case": case,
                "path": str(receipt_path.relative_to(staging_raw)),
                "sha256": v3_plan.sha256_file(receipt_path),
                "receipt_sha256": v3_plan.canonical_sha256(receipt),
            }
        )

    actual_paths = {
        path for path in staging_raw.rglob("*") if path.is_file() or path.is_symlink()
    }
    if any(path.is_symlink() for path in actual_paths):
        raise ValueError("v3 staging tree contains a symbolic link")
    if actual_paths != expected_paths:
        raise ValueError("v3 staging file inventory differs from the frozen plan")
    return {
        "files": files,
        "worker_receipts": receipts,
    }


def build_manifest(
    readiness: Readiness,
    inventory: Mapping[str, object],
) -> dict:
    payload = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "study_kind": worker_collect.STUDY_KIND,
        "collection_kind": "paired_locked_transport",
        "output_role": worker_collect.OUTPUT_ROLE,
        "row_schema": worker_collect.ROW_SCHEMA,
        "fields": list(worker_collect.FIELDS),
        "plan_sha256_by_case": readiness.report["plan_sha256_by_case"],
        "certificate_sha256": readiness.expected_certificate_sha256,
        "certificate_file_sha256": readiness.report[
            "certificate_file_sha256"
        ],
        "readiness_sha256": readiness.report["readiness_sha256"],
        "collection_code_sha256": readiness.collection_code_sha256,
        "source_sha256_by_case": readiness.report["source_sha256_by_case"],
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "worker_boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "boptest_commit": boptest.BOPTEST_COMMIT,
        "counts": {
            "cases": len(boptest.CASES),
            "windows": sum(
                len(readiness.plans[case]["entries"]) for case in boptest.CASES
            ),
            "branches": len(inventory["files"]),  # type: ignore[arg-type]
            "rows_per_branch": boptest.TRAJECTORY_STEPS,
        },
        "files": inventory["files"],
        "worker_receipts": inventory["worker_receipts"],
    }
    return {
        "manifest_sha256": v3_plan.canonical_sha256(payload),
        "manifest": payload,
    }


def _write_pending(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v3 pending manifest: {path}")
    with path.open("xb") as stream:
        stream.write(v3_plan.canonical_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _state_dir(readiness_sha256: str) -> Path:
    if not boptest.valid_sha256(readiness_sha256):
        raise ValueError("v3 readiness digest is not a SHA-256")
    return STATE_ROOT / readiness_sha256


def _write_state_once(path: Path, payload: object) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir():
        raise ValueError("v3 state root is not a plain directory")
    if path.parent.parent != STATE_ROOT:
        raise ValueError("v3 state evidence path leaves the digest-scoped root")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("v3 digest-scoped state path is not a plain directory")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v3 state evidence: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"stale v3 state temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(v3_plan.canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_no_prior_attempt(readiness_sha256: str) -> Path:
    state_dir = _state_dir(readiness_sha256)
    if STATE_ROOT.exists() and (
        STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir()
    ):
        raise ValueError("v3 state root is not a plain directory")
    if os.path.lexists(state_dir / ATTEMPT_MARKER):
        raise FileExistsError(
            "this frozen v3 readiness digest already has a collection attempt"
        )
    if state_dir.exists():
        if state_dir.is_symlink() or not state_dir.is_dir():
            raise ValueError("v3 digest-scoped state is not a plain directory")
        unexpected = sorted(path.name for path in state_dir.iterdir())
        if unexpected:
            raise ValueError(
                f"v3 digest-scoped state has unexpected entries: {unexpected}"
            )
    return state_dir


def _attempt_payload(
    readiness: Readiness,
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
    committed = datetime.fromisoformat(
        str(external_freeze["revision_committed_at_utc"]).replace(
            "Z", "+00:00"
        )
    )
    provider_verified = datetime.fromisoformat(
        str(external_freeze["provider_verified_at_utc"]).replace(
            "Z", "+00:00"
        )
    )
    if (
        committed.tzinfo is None
        or provider_verified.tzinfo is None
        or committed > provider_verified
    ):
        raise ValueError(
            "external freeze revision is not earlier than provider verification"
        )
    return {
        "schema": ATTEMPT_SCHEMA,
        "stage": "locked_transport_collection",
        "started_at_utc": external_freeze["provider_verified_at_utc"],
        "started_at_source": "github-api-http-date-before-attempt-marker",
        "host_marker_written_at_utc": _utc_now(),
        "pid": os.getpid(),
        "host_python_version": platform.python_version(),
        "certificate_sha256": readiness.expected_certificate_sha256,
        "readiness_sha256": readiness.report["readiness_sha256"],
        "prelock_registry_sha256": readiness.report[
            "prelock_registry_sha256"
        ],
        "external_freeze_receipt_path": str(external_freeze_receipt_path),
        "external_freeze_receipt_sha256": v3_plan.sha256_file(
            external_freeze_receipt_path
        ),
        "external_freeze_gist_id": external_freeze["gist_id"],
        "external_freeze_revision": external_freeze["revision"],
        "external_freeze_revision_committed_at_utc": external_freeze[
            "revision_committed_at_utc"
        ],
        "external_freeze_provider_verified_at_utc": external_freeze[
            "provider_verified_at_utc"
        ],
        "plan_sha256_by_case": readiness.report["plan_sha256_by_case"],
        "collection_code_sha256": readiness.collection_code_sha256,
        "protocol_sha256": readiness.report["protocol_sha256"],
        "worker_image_id": boptest.WORKER_IMAGE_ID,
        "worker_boptest_version": boptest.WORKER_BOPTEST_VERSION,
        "plan_root": str(plan_root),
        "certificate_path": str(certificate_path),
        "raw_root": str(raw_root),
        "manifest_path": str(manifest_path),
        "staging_root": str(staging),
        "collector_commands": [list(command) for command in commands],
        "locked_response_values_accessed": False,
    }


def _failure_payload(
    *,
    readiness: Readiness,
    attempt_path: Path,
    error: BaseException,
    simulator_process_started: bool,
    staging: Path,
    raw_root: Path,
    manifest_path: Path,
) -> dict:
    message = str(error).encode("ascii", "backslashreplace").decode("ascii")
    return {
        "schema": FAILURE_SCHEMA,
        "stage": "locked_transport_collection_failed",
        "failed_at_utc": _utc_now(),
        "certificate_sha256": readiness.expected_certificate_sha256,
        "readiness_sha256": readiness.report["readiness_sha256"],
        "attempt_marker_sha256": v3_plan.sha256_file(attempt_path),
        "error_type": type(error).__name__,
        "error_message": message,
        "simulator_process_started": simulator_process_started,
        "locked_response_values_may_have_been_accessed": simulator_process_started,
        "staging_root": str(staging),
        "staging_exists": staging.exists(),
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "retry_permitted_under_same_readiness_digest": False,
    }


def _completion_payload(
    *,
    readiness: Readiness,
    attempt_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict:
    return {
        "schema": COMPLETION_SCHEMA,
        "stage": "locked_transport_collection_complete",
        "completed_at_utc": _utc_now(),
        "certificate_sha256": readiness.expected_certificate_sha256,
        "readiness_sha256": readiness.report["readiness_sha256"],
        "prelock_registry_sha256": readiness.report[
            "prelock_registry_sha256"
        ],
        "attempt_marker_sha256": v3_plan.sha256_file(attempt_path),
        "collection_code_sha256": readiness.collection_code_sha256,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": v3_plan.sha256_file(manifest_path),
        "manifest_payload_sha256": manifest["manifest_sha256"],
        "locked_response_values_accessed_after_attempt": True,
    }


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError("refusing to seal a v3 tree containing a symbolic link")
        mode = path.stat().st_mode
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )


def _publish(
    staging: Path,
    raw_root: Path,
    manifest_path: Path,
    pending: Path,
    wrapper: Mapping[str, object],
) -> None:
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("v3 staging root is not a plain directory")
    if any(os.path.lexists(path) for path in (raw_root, manifest_path, pending)):
        raise FileExistsError("v3 publication destination became occupied")
    _write_pending(pending, wrapper)
    try:
        staging.rename(raw_root)
        _seal_tree(raw_root)
        os.link(pending, manifest_path)
        manifest_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        pending.unlink(missing_ok=True)


def run_collection(
    expected_certificate_sha256: str,
    expected_readiness_sha256: str,
    confirmation: str,
    *,
    plan_root: Path = CANONICAL_PLAN_ROOT,
    certificate_path: Path = CANONICAL_CERTIFICATE,
    raw_root: Path = CANONICAL_RAW,
    manifest_path: Path = CANONICAL_MANIFEST,
    testcase_root: Path = CANONICAL_TESTCASE_ROOT,
    prelock_root: Path = CANONICAL_PRELOCK_ROOT,
    readiness_path: Path = CANONICAL_READINESS,
    external_freeze_receipt_path: Path = CANONICAL_EXTERNAL_FREEZE_RECEIPT,
) -> dict:
    if confirmation != CONFIRMATION_TOKEN:
        raise ValueError(
            "formal v3 collection requires the exact confirmation token"
        )
    if not boptest.valid_sha256(expected_readiness_sha256):
        raise ValueError("expected v3 readiness digest is not a SHA-256")
    readiness = prepare_readiness(
        expected_certificate_sha256,
        plan_root=plan_root,
        certificate_path=certificate_path,
        testcase_root=testcase_root,
        prelock_root=prelock_root,
        validate_runtime=True,
    )
    if readiness.report["readiness_sha256"] != expected_readiness_sha256:
        raise ValueError("live v3 readiness differs from the externally frozen digest")
    if readiness.collection_code_sha256 != collection_code_hashes():
        raise ValueError("v3 collection code changed during readiness")
    from .external_freeze import validate_external_freeze_receipt

    external_freeze = validate_external_freeze_receipt(
        external_freeze_receipt_path,
        str(readiness.report["prelock_registry_sha256"]),
        expected_readiness_sha256,
        prelock_root=prelock_root,
        readiness_path=readiness_path,
        live=True,
    )

    state_dir = _assert_no_prior_attempt(expected_readiness_sha256)
    raw_root, staging, manifest_path, pending = _preflight_destination(
        raw_root, manifest_path
    )
    commands = [
        _docker_command(
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
        staging.mkdir(parents=True)
        for command in commands:
            simulator_process_started = True
            subprocess.run(command, check=True)
        if readiness.collection_code_sha256 != collection_code_hashes():
            raise ValueError("v3 collection code changed during execution")
        inventory = validate_staged_collection(staging, readiness)
        manifest = build_manifest(readiness, inventory)
        _publish(staging, raw_root, manifest_path, pending, manifest)
        _write_state_once(
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
        # A failed staging tree and immutable marker remain noncanonical evidence.
        try:
            _write_state_once(
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("readiness", "collect"))
    parser.add_argument("--expected-certificate-sha256", required=True)
    parser.add_argument("--expected-readiness-sha256")
    parser.add_argument("--confirm")
    parser.add_argument("--validate-runtime", action="store_true")
    parser.add_argument("--plan-root", type=Path, default=CANONICAL_PLAN_ROOT)
    parser.add_argument(
        "--certificate", type=Path, default=CANONICAL_CERTIFICATE
    )
    parser.add_argument("--raw-root", type=Path, default=CANONICAL_RAW)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--prelock-root", type=Path, default=CANONICAL_PRELOCK_ROOT)
    parser.add_argument("--readiness-file", type=Path, default=CANONICAL_READINESS)
    parser.add_argument(
        "--external-freeze-receipt",
        type=Path,
        default=CANONICAL_EXTERNAL_FREEZE_RECEIPT,
    )
    parser.add_argument("--write-readiness", action="store_true")
    parser.add_argument(
        "--testcase-root", type=Path, default=CANONICAL_TESTCASE_ROOT
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "readiness":
        if args.confirm is not None or args.expected_readiness_sha256 is not None:
            raise ValueError("readiness cannot accept collection-only arguments")
        report = readiness_report(
            args.expected_certificate_sha256,
            plan_root=args.plan_root,
            certificate_path=args.certificate,
            testcase_root=args.testcase_root,
            prelock_root=args.prelock_root,
            validate_runtime=args.validate_runtime,
        )
        if args.write_readiness:
            write_readiness_report(args.readiness_file.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if not args.expected_readiness_sha256:
        raise ValueError("collect requires --expected-readiness-sha256")
    result = run_collection(
        args.expected_certificate_sha256,
        args.expected_readiness_sha256,
        args.confirm,
        plan_root=args.plan_root,
        certificate_path=args.certificate,
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        testcase_root=args.testcase_root,
        prelock_root=args.prelock_root,
        readiness_path=args.readiness_file,
        external_freeze_receipt_path=args.external_freeze_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
