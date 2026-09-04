from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .protocol import (
    BOPTEST_COMMIT,
    BOPTEST_LICENSE_PATH,
    BOPTEST_LICENSE_SHA256,
    BOPTEST_REPOSITORY_URL,
    CASES,
    CORPUS_MANIFEST_SCHEMA,
    FULL_PLAN_SCHEMA,
    SMOKE_PLAN_SCHEMA,
    STEP_SECONDS,
    TRAJECTORY_STEPS,
    WORKER_BOPTEST_VERSION,
    WORKER_IMAGE_ID,
    WORKER_RECEIPT_SCHEMA,
    attach_plan_sha256,
    balanced_action_levels,
    build_case_plan,
    canonical_json,
    collector_code_hashes,
    sha256_file,
    smoke_view,
    strict_json_loads,
    validate_prelock_registry_payload,
    validate_plan,
)
from .worker_collect import (
    FIELDS,
    ROW_SCHEMA,
    collection_kind,
    expected_filename,
    receipt_path,
)


HERE = Path(__file__).resolve().parent
DEFAULT_TESTCASE_ROOT = (Path.home() / "external/project1-boptest/testcases")
DEFAULT_OUTPUT = HERE / "data_v4"
FULL_PLAN_SUBDIR = Path("plans/full")
SMOKE_PLAN_SUBDIR = Path("plans/smoke")
SMOKE_RAW_SUBDIR = Path("smoke_raw")
DEVELOPMENT_RAW_SUBDIR = Path("development_raw")
LOCKED_RAW_SUBDIR = Path("locked_test_raw")
MANIFEST_SUBDIR = Path("manifests")
SMOKE_ROLES = ("fit",)
DEVELOPMENT_ROLES = ("fit", "validation")
LOCKED_ROLES = ("locked_test",)
LOCKED_CONFIRMATION = "I_UNDERSTAND_LOCKED_TEST_IS_ONE_SHOT"


def _atomic_link_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary JSON exists: {temporary}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_immutable(path: Path, payload: object) -> None:
    """Write once; an identical retry is a no-op and any change is refused."""
    if path.exists():
        existing = strict_json_loads(path.read_text(encoding="ascii"))
        if canonical_json(existing) != canonical_json(payload):
            raise FileExistsError(f"refusing hash-changing overwrite of {path}")
        return
    _atomic_link_json(path, payload)


def validate_public_checkout(testcase_root: Path) -> None:
    repository = testcase_root.parent
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != BOPTEST_COMMIT:
        raise ValueError(f"BOPTEST checkout is {head}, expected {BOPTEST_COMMIT}")
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
    if remote != BOPTEST_REPOSITORY_URL:
        raise ValueError(f"BOPTEST origin is {remote}, expected {BOPTEST_REPOSITORY_URL}")
    if sha256_file(repository / BOPTEST_LICENSE_PATH) != BOPTEST_LICENSE_SHA256:
        raise ValueError("BOPTEST license hash differs from the frozen public source")


def _plan_path(output_dir: Path, mode: str, case: str) -> Path:
    subdir = FULL_PLAN_SUBDIR if mode == "full" else SMOKE_PLAN_SUBDIR
    return output_dir / subdir / f"{case}.json"


def _read_plan(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"frozen plan is missing: {path}")
    payload = strict_json_loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError(f"frozen plan is not a JSON object: {path}")
    return payload


def materialize_plans(
    testcase_root: Path,
    output_dir: Path,
    mode: str,
    cases: Sequence[str] | None = None,
) -> list[Path]:
    selected = tuple(cases or CASES)
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    paths: list[Path] = []
    for case in selected:
        adapter = CASES[case]
        full_path = _plan_path(output_dir, "full", case)
        if mode == "full":
            plan = attach_plan_sha256(build_case_plan(adapter, testcase_root))
            write_json_immutable(full_path, plan)
            paths.append(full_path)
            continue
        if mode != "smoke":
            raise ValueError("plan mode must be smoke or full")
        full_plan = _read_plan(full_path)
        validate_plan(full_plan, testcase_root, DEVELOPMENT_ROLES)
        plan = smoke_view(full_plan)
        smoke_path = _plan_path(output_dir, "smoke", case)
        write_json_immutable(smoke_path, plan)
        paths.append(smoke_path)
    return paths


def load_frozen_plans(
    testcase_root: Path,
    output_dir: Path,
    mode: str,
    cases: Sequence[str],
    allowed_roles: tuple[str, ...],
) -> list[Path]:
    paths = [_plan_path(output_dir, mode, case) for case in cases]
    for path in paths:
        validate_plan(_read_plan(path), testcase_root, allowed_roles)
    return paths


def validate_locked_plan_binding(
    plan_paths: Sequence[Path],
    prelock_plan_sha256_by_case: dict[str, str],
) -> dict[str, str]:
    """Require live locked-collection plans to equal the sealed development plans."""
    expected_cases = set(CASES)
    if set(prelock_plan_sha256_by_case) != expected_cases:
        raise ValueError("sealed pre-lock plan identities are incomplete")
    live: dict[str, str] = {}
    for path in plan_paths:
        plan = _read_plan(path)
        case = plan["case_adapter"]["case"]
        if case in live:
            raise ValueError(f"duplicate live full plan for case: {case}")
        live[case] = plan["plan_sha256"]
    if set(live) != expected_cases:
        raise ValueError("live locked-collection plan identities are incomplete")
    if live != prelock_plan_sha256_by_case:
        raise ValueError(
            "live full plans differ from sealed pre-lock development plans"
        )
    return live


def _docker_command(
    plan_path: Path,
    testcase_root: Path,
    raw_dir: Path,
    allowed_roles: tuple[str, ...],
    prelock_registry_path: Path | None = None,
    expected_prelock_sha256: str | None = None,
) -> list[str]:
    role_arguments = " ".join(f"--allowed-role {role}" for role in allowed_roles)
    if (prelock_registry_path is None) != (expected_prelock_sha256 is None):
        raise ValueError("pre-lock registry path and digest must be supplied together")
    prelock_mount: list[str] = []
    prelock_argument = ""
    if prelock_registry_path is not None:
        prelock_mount = [
            "-v",
            f"{prelock_registry_path.resolve()}:/prelock/prelock_registry.json:ro",
        ]
        prelock_argument = (
            " --prelock-registry /prelock/prelock_registry.json"
            f" --expected-prelock-sha256 {expected_prelock_sha256}"
        )
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
        f"{HERE.parents[1]}:/workspace:ro",
        "-v",
        f"{plan_path.parent}:/plans:ro",
        "-v",
        f"{raw_dir}:/out",
        *prelock_mount,
        "-e",
        (
            "PYTHONPATH=/workspace:/boptest:/boptest/lib:/boptest/lib/kpis:"
            "/boptest/lib/forecast:/boptest/lib/data"
        ),
        WORKER_IMAGE_ID,
        "/bin/bash",
        "-lc",
        (
            ". /miniconda/bin/activate && conda activate pyfmi3 && "
            "cp /version.txt /work/version.txt && "
            "python -m building_fault_wm.neural_benchmark.worker_collect "
            f"--plan /plans/{plan_path.name} --output /out "
            "--testcase-root /public-boptest/testcases "
            f"--worker-image-id {WORKER_IMAGE_ID} "
            f"--boptest-version-file /version.txt {role_arguments}"
            f"{prelock_argument}"
        ),
    ]


def validate_prelock_registry(path: Path, expected_sha256: str) -> dict:
    registry = strict_json_loads(path.read_text(encoding="ascii"))
    return validate_prelock_registry_payload(registry, expected_sha256)


def validate_worker_image() -> None:
    image_id = subprocess.run(
        ["docker", "image", "inspect", WORKER_IMAGE_ID, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if image_id != WORKER_IMAGE_ID:
        raise ValueError(f"Docker resolved {WORKER_IMAGE_ID} to unexpected {image_id}")
    version = subprocess.run(
        ["docker", "run", "--rm", WORKER_IMAGE_ID, "/bin/cat", "/version.txt"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != WORKER_BOPTEST_VERSION:
        raise ValueError(
            f"worker BOPTEST version is {version}, expected {WORKER_BOPTEST_VERSION}"
        )


def _expected_entries(plans: Iterable[dict], roles: set[str]) -> list[tuple[dict, dict]]:
    entries: list[tuple[dict, dict]] = []
    for plan in plans:
        for entry in plan["entries"]:
            if entry["role"] in roles:
                entries.append((plan, entry))
    return entries


def _validate_csv(path: Path, entry: dict, adapter) -> dict:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"trajectory field schema mismatch: {path}")
        rows = list(reader)
    if len(rows) != TRAJECTORY_STEPS:
        raise ValueError(f"trajectory has {len(rows)} rows, expected {TRAJECTORY_STEPS}: {path}")
    expected_actions = balanced_action_levels(int(entry["trajectory_seed"]))
    numeric_fields = FIELDS[5:]
    continuity = (
        ("next_zone_temperature_k", "zone_temperature_k"),
        ("next_hvac_electric_power_w", "hvac_electric_power_w"),
        ("next_auxiliary_1", "auxiliary_1"),
        ("next_auxiliary_2", "auxiliary_2"),
        ("next_outdoor_temperature_k", "outdoor_temperature_k"),
        ("next_global_horizontal_solar_w_m2", "global_horizontal_solar_w_m2"),
        ("next_comfort_lower_k", "comfort_lower_k"),
        ("next_comfort_upper_k", "comfort_upper_k"),
        ("next_electricity_price", "electricity_price"),
    )
    for index, row in enumerate(rows):
        if row["case"] != entry["case"] or row["role"] != entry["role"]:
            raise ValueError(f"trajectory case/role identity mismatch: {path}")
        if int(row["day"]) != int(entry["day"]):
            raise ValueError(f"trajectory day identity mismatch: {path}")
        if int(row["trajectory_seed"]) != int(entry["trajectory_seed"]):
            raise ValueError(f"trajectory seed identity mismatch: {path}")
        if int(row["step"]) != index:
            raise ValueError(f"trajectory step sequence mismatch: {path}")
        expected_time = int(entry["day"]) * 86_400 + index * STEP_SECONDS
        if float(row["time_s"]) != float(expected_time):
            raise ValueError(f"trajectory timestamp mismatch: {path}")
        if float(row["normalized_action"]) != float(expected_actions[index]):
            raise ValueError(f"trajectory action schedule mismatch: {path}")
        expected_setpoint = (
            adapter.base_setpoint_k
            + adapter.action_amplitude_k * float(expected_actions[index])
        )
        if float(row["setpoint_k"]) != expected_setpoint:
            raise ValueError(f"trajectory setpoint mismatch: {path}")
        if any(not math.isfinite(float(row[field])) for field in numeric_fields):
            raise ValueError(f"trajectory contains a non-finite numeric value: {path}")
        if index:
            previous = rows[index - 1]
            for previous_name, current_name in continuity:
                if float(previous[previous_name]) != float(row[current_name]):
                    raise ValueError(f"trajectory current/next continuity mismatch: {path}")
    return {
        "sha256": sha256_file(path),
        "rows": len(rows),
        "fields": len(FIELDS),
    }


def _read_and_validate_receipt(
    path: Path,
    plan: dict,
    allowed_roles: tuple[str, ...],
    expected_files: list[dict],
    prelock_registry_sha256: str | None,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"worker receipt is missing: {path}")
    wrapper = strict_json_loads(path.read_text(encoding="ascii"))
    payload = wrapper.get("receipt")
    actual_hash = hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()
    if wrapper.get("receipt_sha256") != actual_hash:
        raise ValueError(f"worker receipt hash mismatch: {path}")
    required = {
        "schema": WORKER_RECEIPT_SCHEMA,
        "collection_kind": collection_kind(plan, allowed_roles),
        "allowed_roles": sorted(set(allowed_roles)),
        "case": plan["case_adapter"]["case"],
        "plan_sha256": plan["plan_sha256"],
        "worker_image_id": WORKER_IMAGE_ID,
        "boptest_version": WORKER_BOPTEST_VERSION,
        "collector_code_sha256": collector_code_hashes(),
        "source_sha256": plan["source_sha256"],
        "prelock_registry_sha256": prelock_registry_sha256,
        "files": expected_files,
    }
    if payload != required:
        raise ValueError(f"worker receipt payload differs from collected files: {path}")
    return {"sha256": sha256_file(path), "receipt_sha256": actual_hash}


def _manifest_name(kind: str, cases: Sequence[str]) -> str:
    selected = tuple(sorted(cases))
    suffix = "all" if selected == tuple(sorted(CASES)) else "-".join(selected)
    return f"{kind}_{suffix}_corpus_manifest.json"


def build_corpus_manifest(
    output_dir: Path,
    raw_dir: Path,
    plan_paths: Sequence[Path],
    allowed_roles: tuple[str, ...],
    testcase_root: Path = DEFAULT_TESTCASE_ROOT,
    prelock_registry: dict | None = None,
    expected_prelock_sha256: str | None = None,
) -> tuple[Path, dict]:
    plan_records_input = sorted(
        ((path, _read_plan(path)) for path in plan_paths),
        key=lambda item: item[1]["case_adapter"]["case"],
    )
    if not plan_records_input:
        raise ValueError("cannot create a corpus manifest without plans")
    plan_paths = tuple(path for path, _ in plan_records_input)
    plans = [plan for _, plan in plan_records_input]
    modes = {plan["mode"] for plan in plans}
    if len(modes) != 1:
        raise ValueError("corpus plans mix full and smoke modes")
    for plan in plans:
        validate_plan(plan, testcase_root, allowed_roles)
    kind = collection_kind(plans[0], allowed_roles)
    if any(collection_kind(plan, allowed_roles) != kind for plan in plans):
        raise ValueError("corpus plans imply different collection kinds")
    if kind == "locked_test":
        validate_prelock_registry_payload(
            prelock_registry,
            expected_prelock_sha256,
        )
        prelock_registry_sha256 = expected_prelock_sha256
    else:
        if prelock_registry is not None or expected_prelock_sha256 is not None:
            raise ValueError("non-locked corpus cannot carry pre-lock inputs")
        prelock_registry_sha256 = None
    cases = tuple(sorted(plan["case_adapter"]["case"] for plan in plans))
    if len(set(cases)) != len(cases):
        raise ValueError("corpus contains duplicate case plans")
    roles = set(allowed_roles)
    planned_entries = _expected_entries(plans, roles)
    expected_paths = {
        raw_dir / entry["case"] / expected_filename(entry)
        for _, entry in planned_entries
    }
    actual_paths = {
        path
        for case in cases
        for path in (raw_dir / case).rglob("*.csv")
    }
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        stale = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f"corpus file bijection failed; missing={missing}, stale={stale}")

    expected_receipt_paths = {
        receipt_path(raw_dir, plan, allowed_roles) for plan in plans
    }
    receipts_dir = raw_dir / "_receipts"
    actual_receipt_paths = {
        path
        for path in receipts_dir.glob("*.json")
        if any(path.name.startswith(f"{case}_") for case in cases)
    }
    if actual_receipt_paths != expected_receipt_paths:
        missing = sorted(
            str(path) for path in expected_receipt_paths - actual_receipt_paths
        )
        stale = sorted(
            str(path) for path in actual_receipt_paths - expected_receipt_paths
        )
        raise ValueError(
            f"corpus receipt bijection failed; missing={missing}, stale={stale}"
        )

    files: list[dict] = []
    receipts: dict[str, dict] = {}
    for plan in plans:
        case = plan["case_adapter"]["case"]
        adapter = CASES[case]
        case_entries = [entry for entry in plan["entries"] if entry["role"] in roles]
        receipt_files: list[dict] = []
        for entry in case_entries:
            path = raw_dir / case / expected_filename(entry)
            validation = _validate_csv(path, entry, adapter)
            relative = str(path.relative_to(raw_dir))
            item = {
                "path": relative,
                "case": case,
                "role": entry["role"],
                "day": int(entry["day"]),
                "trajectory_seed": int(entry["trajectory_seed"]),
                "plan_sha256": plan["plan_sha256"],
                **validation,
            }
            files.append(item)
            receipt_files.append(
                {"path": relative, "sha256": validation["sha256"], "rows": validation["rows"]}
            )
        path = receipt_path(raw_dir, plan, allowed_roles)
        receipts[case] = {
            "path": str(path.relative_to(raw_dir)),
            **_read_and_validate_receipt(
                path,
                plan,
                allowed_roles,
                receipt_files,
                prelock_registry_sha256,
            ),
        }
    files.sort(key=lambda item: item["path"])
    role_counts = Counter(item["role"] for item in files)
    plan_records = {
        path.stem: {
            "path": str(path.relative_to(output_dir)),
            "file_sha256": sha256_file(path),
            "plan_sha256": plan["plan_sha256"],
            "schema": plan["schema"],
            "source_sha256": plan["source_sha256"],
            "selected_entries": sum(entry["role"] in roles for entry in plan["entries"]),
        }
        for path, plan in zip(plan_paths, plans)
    }
    payload = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "collection_kind": kind,
        "plan_mode": next(iter(modes)),
        "allowed_roles": sorted(roles),
        "selected_cases": list(cases),
        "public_source": plans[0]["public_source"],
        "boptest_commit": BOPTEST_COMMIT,
        "worker_runtime": {
            "image_id": WORKER_IMAGE_ID,
            "boptest_version": WORKER_BOPTEST_VERSION,
        },
        "collector_code_sha256": collector_code_hashes(),
        "prelock_registry_sha256": prelock_registry_sha256,
        "row_schema": {
            "name": ROW_SCHEMA,
            "fields": list(FIELDS),
            "step_seconds": STEP_SECONDS,
            "trajectory_steps": TRAJECTORY_STEPS,
            "transition_contract": "row r is (observation_r, context_r, action_r, observation_r+1, context_r+1)",
        },
        "plans": plan_records,
        "counts": {
            "cases": len(cases),
            "trajectories": len(files),
            "rows": sum(item["rows"] for item in files),
            "roles": {role: role_counts[role] for role in sorted(role_counts)},
        },
        "receipts": receipts,
        "files": files,
    }
    wrapper = {
        "manifest_sha256": hashlib.sha256(
            canonical_json(payload).encode("ascii")
        ).hexdigest(),
        "manifest": payload,
    }
    path = output_dir / MANIFEST_SUBDIR / _manifest_name(kind, cases)
    return path, wrapper


def write_corpus_manifest(
    output_dir: Path,
    raw_dir: Path,
    plan_paths: Sequence[Path],
    allowed_roles: tuple[str, ...],
    testcase_root: Path = DEFAULT_TESTCASE_ROOT,
    prelock_registry: dict | None = None,
    expected_prelock_sha256: str | None = None,
) -> Path:
    path, wrapper = build_corpus_manifest(
        output_dir,
        raw_dir,
        plan_paths,
        allowed_roles,
        testcase_root,
        prelock_registry,
        expected_prelock_sha256,
    )
    write_json_immutable(path, wrapper)
    return path


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def preflight_locked_destination(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Refuse any prior or partial all-case locked publication before simulation."""
    final_raw = output_dir / LOCKED_RAW_SUBDIR
    staging_raw = output_dir / f".{LOCKED_RAW_SUBDIR.name}.staging"
    manifest = output_dir / MANIFEST_SUBDIR / _manifest_name("locked_test", tuple(CASES))
    pending_manifest = manifest.with_suffix(manifest.suffix + ".pending")
    occupied = [
        path
        for path in (final_raw, staging_raw, manifest, pending_manifest)
        if _path_lexists(path)
    ]
    if occupied:
        raise FileExistsError(
            "refusing locked collection because a final or partial destination exists: "
            + ", ".join(str(path) for path in occupied)
        )
    return final_raw, staging_raw, manifest, pending_manifest


def publish_locked_collection(
    staging_raw: Path,
    final_raw: Path,
    manifest_path: Path,
    pending_manifest: Path,
    wrapper: dict,
) -> None:
    """Publish a complete staged raw corpus, then its already-built manifest."""
    if not staging_raw.is_dir():
        raise FileNotFoundError("locked staging directory is missing")
    if any(
        _path_lexists(path)
        for path in (final_raw, manifest_path, pending_manifest)
    ):
        raise FileExistsError("locked publication destination changed after preflight")
    write_json_immutable(pending_manifest, wrapper)
    os.rename(staging_raw, final_raw)
    try:
        os.link(pending_manifest, manifest_path)
    except Exception:
        raise RuntimeError(
            "locked raw data are complete but manifest publication failed; "
            f"preserve and audit {pending_manifest} before recovery"
        )
    else:
        pending_manifest.unlink()


def _raw_subdir(stage: str, mode: str) -> tuple[Path, tuple[str, ...]]:
    if stage == "collect-locked":
        return LOCKED_RAW_SUBDIR, LOCKED_ROLES
    if mode == "smoke":
        return SMOKE_RAW_SUBDIR, SMOKE_ROLES
    return DEVELOPMENT_RAW_SUBDIR, DEVELOPMENT_ROLES


def validate_collection_request(
    stage: str,
    mode: str,
    locked_confirmation: str | None,
    prelock_registry: Path | None = None,
    expected_prelock_sha256: str | None = None,
    selected_cases: Sequence[str] | None = None,
    prelock_artifact_root: Path | None = None,
) -> None:
    if stage == "collect-locked":
        if mode != "full":
            raise ValueError("collect-locked requires --mode full")
        if locked_confirmation != LOCKED_CONFIRMATION:
            raise ValueError(
                "collect-locked requires --confirm-locked-test "
                + LOCKED_CONFIRMATION
            )
        if (
            prelock_registry is None
            or expected_prelock_sha256 is None
            or prelock_artifact_root is None
        ):
            raise ValueError(
                "collect-locked requires --prelock-registry, "
                "--prelock-artifact-root, and --expected-prelock-sha256"
            )
        if tuple(sorted(selected_cases or ())) != tuple(sorted(CASES)):
            raise ValueError("collect-locked requires every frozen case in one invocation")
        return
    if any(
        value is not None
        for value in (
            locked_confirmation,
            prelock_registry,
            expected_prelock_sha256,
            prelock_artifact_root,
        )
    ):
        raise ValueError(
            "locked confirmation and pre-lock arguments are accepted only with "
            "collect-locked"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the frozen public BOPTEST corpus")
    parser.add_argument("stage", choices=("plan", "collect", "collect-locked"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--testcase-root", type=Path, default=DEFAULT_TESTCASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-locked-test")
    parser.add_argument("--prelock-registry", type=Path)
    parser.add_argument("--prelock-artifact-root", type=Path)
    parser.add_argument("--expected-prelock-sha256")
    args = parser.parse_args()
    selected = tuple(sorted(set(args.case or CASES)))
    validate_collection_request(
        args.stage,
        args.mode,
        args.confirm_locked_test,
        args.prelock_registry,
        args.expected_prelock_sha256,
        selected,
        args.prelock_artifact_root,
    )
    prelock_registry = None
    sealed_prelock_plans = None
    if args.stage == "collect-locked":
        from .provenance import (
            prelock_plan_sha256_by_case,
            validate_prelock_bundle,
        )
        from .study_config import StudyConfig

        prelock_registry = validate_prelock_bundle(
            args.prelock_registry,
            args.prelock_artifact_root,
            StudyConfig(),
            args.expected_prelock_sha256,
        )
        sealed_prelock_plans = prelock_plan_sha256_by_case(
            prelock_registry,
            args.prelock_artifact_root,
        )
    validate_public_checkout(args.testcase_root)

    if args.stage == "plan":
        paths = materialize_plans(
            args.testcase_root,
            args.output_dir,
            args.mode,
            selected,
        )
        for path in paths:
            print(path)
        return

    plan_mode = "full" if args.stage == "collect-locked" else args.mode
    raw_subdir, allowed_roles = _raw_subdir(args.stage, args.mode)
    plans = load_frozen_plans(
        args.testcase_root,
        args.output_dir,
        plan_mode,
        selected,
        allowed_roles,
    )
    if args.stage == "collect-locked":
        if sealed_prelock_plans is None:
            raise AssertionError("sealed pre-lock plan identities were not loaded")
        validate_locked_plan_binding(plans, sealed_prelock_plans)
    locked_publication: tuple[Path, Path, Path, Path] | None = None
    if args.stage == "collect-locked" and not args.dry_run:
        locked_publication = preflight_locked_destination(args.output_dir)
        final_raw, raw_dir, _, _ = locked_publication
        if final_raw != args.output_dir / raw_subdir:
            raise AssertionError("locked publication path differs from the frozen layout")
    else:
        raw_dir = args.output_dir / raw_subdir
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for direct BOPTEST collection")
    if not args.dry_run:
        validate_worker_image()
        raw_dir.mkdir(parents=True, exist_ok=True)
    for path in plans:
        command = _docker_command(
            path,
            args.testcase_root,
            raw_dir,
            allowed_roles,
            args.prelock_registry,
            args.expected_prelock_sha256,
        )
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)
    if not args.dry_run:
        if locked_publication is None:
            print(
                write_corpus_manifest(
                    args.output_dir,
                    raw_dir,
                    plans,
                    allowed_roles,
                    args.testcase_root,
                    prelock_registry,
                    args.expected_prelock_sha256,
                )
            )
        else:
            final_raw, staging_raw, manifest_path, pending_manifest = locked_publication
            built_path, wrapper = build_corpus_manifest(
                args.output_dir,
                raw_dir,
                plans,
                allowed_roles,
                args.testcase_root,
                prelock_registry,
                args.expected_prelock_sha256,
            )
            if built_path != manifest_path:
                raise AssertionError("locked manifest path changed after preflight")
            publish_locked_collection(
                staging_raw,
                final_raw,
                manifest_path,
                pending_manifest,
                wrapper,
            )
            print(manifest_path)


if __name__ == "__main__":
    main()
