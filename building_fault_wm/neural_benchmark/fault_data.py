"""Deterministic clean-trajectory and persistent-fault data layer.

The corpus index is metadata first: every source is schema/count/hash checked,
but trajectory values are read only when a role is explicitly loaded.  This
keeps the locked-test values out of FIT preprocessing and manifest generation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import pandas as pd

from .protocol import (
    BOPTEST_API_VERSION,
    BOPTEST_COMMIT,
    BOPTEST_LICENSE_NAME,
    BOPTEST_LICENSE_PATH,
    BOPTEST_LICENSE_SHA256,
    BOPTEST_REPOSITORY_URL,
    CASES,
    COLLECTOR_CODE_FILES,
    CORPUS_MANIFEST_SCHEMA,
    FULL_PLAN_SCHEMA,
    PLAN_SEED,
    SMOKE_PLAN_SCHEMA,
    STEP_SECONDS,
    TRAJECTORY_STEPS,
    WARMUP_SECONDS,
    WORKER_BOPTEST_VERSION,
    WORKER_IMAGE_ID,
    WORKER_RECEIPT_SCHEMA,
    canonical_json,
    plan_sha256,
    sha256_file,
    strict_json_loads,
    validate_prelock_registry_payload,
)
from .worker_collect import FIELDS, ROW_SCHEMA, expected_filename


Role = Literal["fit", "validation", "locked_test"]
Family = Literal["healthy", "bias", "drift", "stuck", "dropout"]

ROLES: tuple[Role, ...] = ("fit", "validation", "locked_test")
FAMILIES: tuple[Family, ...] = (
    "healthy",
    "bias",
    "drift",
    "stuck",
    "dropout",
)
HEALTH_CLASS = {family: index for index, family in enumerate(FAMILIES)}

OBSERVATION_COLUMNS = (
    "zone_temperature_k",
    "hvac_electric_power_w",
    "auxiliary_1",
    "auxiliary_2",
)
NEXT_OBSERVATION_COLUMNS = tuple(f"next_{name}" for name in OBSERVATION_COLUMNS)
FAULT_CHANNELS = OBSERVATION_COLUMNS[:2]
SOURCE_ACTION_COLUMNS = ("normalized_action", "setpoint_k")
ACTION_COLUMNS = ("normalized_action",)
CONTEXT_COLUMNS = (
    "outdoor_temperature_k",
    "global_horizontal_solar_w_m2",
    "comfort_lower_k",
    "comfort_upper_k",
    "electricity_price",
)
NEXT_CONTEXT_COLUMNS = tuple(f"next_{name}" for name in CONTEXT_COLUMNS)

CORPUS_SCHEMA = CORPUS_MANIFEST_SCHEMA
FAULT_MANIFEST_SCHEMA = "boptest-multicase-fault-manifest-v2"
CORPUS_LAYOUT = {
    "smoke": ("smoke", ("fit",), "smoke_raw", SMOKE_PLAN_SCHEMA),
    "development": (
        "full",
        ("fit", "validation"),
        "development_raw",
        FULL_PLAN_SCHEMA,
    ),
    "locked_test": (
        "full",
        ("locked_test",),
        "locked_test_raw",
        FULL_PLAN_SCHEMA,
    ),
}
PUBLIC_SOURCE = {
    "repository_url": BOPTEST_REPOSITORY_URL,
    "commit": BOPTEST_COMMIT,
    "license_name": BOPTEST_LICENSE_NAME,
    "license_path": BOPTEST_LICENSE_PATH,
    "license_sha256": BOPTEST_LICENSE_SHA256,
}
WORKER_RUNTIME = {
    "image_id": WORKER_IMAGE_ID,
    "boptest_version": WORKER_BOPTEST_VERSION,
}


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"corpus path escapes its root: {relative}") from error
    return candidate


def _csv_metadata(path: Path) -> tuple[tuple[str, ...], int]:
    """Read only a CSV header and record count, not trajectory values."""
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty trajectory CSV: {path}") from error
        rows = sum(1 for _ in reader)
    return header, rows


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    case: str
    role: Role
    day: int
    trajectory_seed: int

    @property
    def text(self) -> str:
        return f"{self.case}:{self.role}:day{self.day:03d}:seed{self.trajectory_seed}"


@dataclass(frozen=True)
class CorpusRecord:
    key: TrajectoryKey
    relative_path: str
    source_sha256: str
    rows: int
    step_seconds: int
    base_setpoint_k: float
    action_amplitude_k: float


@dataclass(frozen=True)
class CorpusIndex:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    collection_kind: str
    prelock_registry_sha256: str | None
    allowed_roles: tuple[Role, ...]
    records: tuple[CorpusRecord, ...]
    plan_sha256_by_case: tuple[tuple[str, str], ...]

    def records_for(
        self, role: Role, cases: Sequence[str] | None = None
    ) -> tuple[CorpusRecord, ...]:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        selected_cases = None if cases is None else set(cases)
        selected = tuple(
            record
            for record in self.records
            if record.key.role == role
            and (selected_cases is None or record.key.case in selected_cases)
        )
        if selected_cases is not None:
            actual_cases = {record.key.case for record in selected}
            if actual_cases != selected_cases:
                missing = sorted(selected_cases - actual_cases)
                raise ValueError(f"role {role} is missing requested cases: {missing}")
        if not selected:
            raise ValueError(f"corpus has no trajectories for role {role}")
        return selected

    @property
    def source_hashes(self) -> dict[str, str]:
        return {record.key.text: record.source_sha256 for record in self.records}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_code_hashes(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(COLLECTOR_CODE_FILES):
        raise ValueError(f"{label} collector code hashes are incomplete")
    if any(not _valid_sha256(digest) for digest in value.values()):
        raise ValueError(f"{label} collector code hash is invalid")
    return value


def _validate_plan_shape(plan: dict, case: str, expected_mode: str) -> None:
    expected_schema = FULL_PLAN_SCHEMA if expected_mode == "full" else SMOKE_PLAN_SCHEMA
    if plan.get("schema") != expected_schema:
        raise ValueError(f"plan schema mismatch for {case}")
    if plan.get("mode") != expected_mode:
        raise ValueError(f"plan mode mismatch for {case}")
    if plan.get("boptest_commit") != BOPTEST_COMMIT:
        raise ValueError(f"BOPTEST commit mismatch for {case}")
    if plan.get("boptest_api_version") != BOPTEST_API_VERSION:
        raise ValueError(f"BOPTEST API version mismatch for {case}")
    if plan.get("public_source") != PUBLIC_SOURCE:
        raise ValueError(f"public-source provenance mismatch for {case}")
    if plan.get("worker_runtime") != WORKER_RUNTIME:
        raise ValueError(f"worker runtime mismatch for {case}")
    _validate_code_hashes(plan.get("collector_code_sha256"), f"plan {case}")
    if plan.get("step_seconds") != STEP_SECONDS:
        raise ValueError(f"step cadence mismatch for {case}")
    if plan.get("warmup_seconds") != WARMUP_SECONDS:
        raise ValueError(f"warmup mismatch for {case}")
    if plan.get("trajectory_steps") != TRAJECTORY_STEPS:
        raise ValueError(f"trajectory length mismatch for {case}")
    if plan.get("plan_seed") != PLAN_SEED:
        raise ValueError(f"plan seed mismatch for {case}")
    adapter = plan.get("case_adapter", {})
    if case not in CASES or canonical_json(adapter) != canonical_json(asdict(CASES[case])):
        raise ValueError(f"plan case adapter mismatch for {case}")
    try:
        action_metadata = np.asarray(
            [adapter["base_setpoint_k"], adapter["action_amplitude_k"]],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"plan action metadata are invalid for {case}") from error
    if not np.isfinite(action_metadata).all() or action_metadata[1] <= 0:
        raise ValueError(f"plan action metadata are not finite for {case}")
    stored_plan_hash = plan.get("plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if stored_plan_hash != plan_sha256(payload):
        raise ValueError(f"canonical plan SHA-256 mismatch for {case}")

    source_hashes = plan.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != {"wrapped_fmu", "weather_csv"}
        or any(not _valid_sha256(value) for value in source_hashes.values())
        or source_hashes["wrapped_fmu"] != CASES[case].fmu_sha256
    ):
        raise ValueError(f"plan source hashes are invalid for {case}")

    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"plan has no entries for {case}")
    if expected_mode == "full":
        counts = {role: sum(entry.get("role") == role for entry in entries) for role in ROLES}
        if len(entries) != 40 or counts != {
            "fit": 20,
            "validation": 8,
            "locked_test": 12,
        }:
            raise ValueError(f"full plan role counts differ for {case}")
    else:
        if len(entries) != 1 or entries[0].get("role") != "fit":
            raise ValueError(f"smoke plan must contain exactly one FIT entry for {case}")
        if not _valid_sha256(plan.get("parent_full_plan_sha256")):
            raise ValueError(f"smoke plan parent hash is invalid for {case}")


def _entry_key(entry: dict, case: str) -> TrajectoryKey:
    if entry.get("case") != case:
        raise ValueError(f"trajectory entry case mismatch for {case}")
    role = entry.get("role")
    if role not in ROLES:
        raise ValueError(f"trajectory entry has invalid role for {case}")
    try:
        raw_day = entry["day"]
        raw_seed = entry["trajectory_seed"]
    except KeyError as error:
        raise ValueError(f"trajectory entry metadata are invalid for {case}") from error
    if (
        isinstance(raw_day, bool)
        or not isinstance(raw_day, int)
        or isinstance(raw_seed, bool)
        or not isinstance(raw_seed, int)
    ):
        raise ValueError(f"trajectory entry metadata are invalid for {case}")
    day = raw_day
    seed = raw_seed
    if day < 0 or seed < 0:
        raise ValueError(f"trajectory entry metadata are negative for {case}")
    return TrajectoryKey(case, role, day, seed)


def load_corpus_index(
    manifest_path: Path,
    *,
    prelock_registry: dict | None = None,
    expected_prelock_sha256: str | None = None,
) -> CorpusIndex:
    """Load one explicit, sealed corpus without reading trajectory values."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"corpus manifest is missing: {manifest_path}")
    if manifest_path.parent.name != "manifests":
        raise ValueError("corpus manifests must be supplied from the manifests directory")
    output_root = manifest_path.parent.parent
    wrapper = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    if not isinstance(wrapper, dict) or set(wrapper) != {
        "manifest_sha256",
        "manifest",
    }:
        raise ValueError("clean corpus wrapper schema mismatch")
    manifest = wrapper.get("manifest")
    recorded_manifest_hash = wrapper.get("manifest_sha256")
    if not isinstance(manifest, dict) or not _valid_sha256(recorded_manifest_hash):
        raise ValueError("clean corpus wrapper is incomplete")
    if _canonical_sha256(manifest) != recorded_manifest_hash:
        raise ValueError("clean corpus wrapper self-hash mismatch")
    expected_manifest_fields = {
        "schema",
        "collection_kind",
        "plan_mode",
        "allowed_roles",
        "selected_cases",
        "public_source",
        "boptest_commit",
        "worker_runtime",
        "collector_code_sha256",
        "prelock_registry_sha256",
        "row_schema",
        "plans",
        "counts",
        "receipts",
        "files",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("clean corpus manifest fields differ from the frozen schema")
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError("clean corpus schema mismatch")
    if manifest.get("boptest_commit") != BOPTEST_COMMIT:
        raise ValueError("clean corpus BOPTEST commit mismatch")

    kind = manifest.get("collection_kind")
    if kind not in CORPUS_LAYOUT:
        raise ValueError("clean corpus collection kind is invalid")
    expected_mode, expected_roles, raw_subdir, expected_plan_schema = CORPUS_LAYOUT[kind]
    prelock_registry_sha256 = manifest.get("prelock_registry_sha256")
    if kind == "locked_test":
        validate_prelock_registry_payload(
            prelock_registry,
            expected_prelock_sha256,
        )
        if prelock_registry_sha256 != expected_prelock_sha256:
            raise ValueError("locked corpus differs from the externally committed pre-lock digest")
    else:
        if prelock_registry_sha256 is not None:
            raise ValueError("non-locked corpus carries a pre-lock registry SHA-256")
        if prelock_registry is not None or expected_prelock_sha256 is not None:
            raise ValueError("non-locked corpus cannot be opened with pre-lock inputs")
    if manifest.get("plan_mode") != expected_mode:
        raise ValueError("clean corpus plan mode does not match its collection kind")
    if manifest.get("allowed_roles") != sorted(expected_roles):
        raise ValueError("clean corpus roles do not match its collection kind")
    if manifest.get("public_source") != PUBLIC_SOURCE:
        raise ValueError("clean corpus public-source provenance mismatch")
    if manifest.get("worker_runtime") != WORKER_RUNTIME:
        raise ValueError("clean corpus worker runtime mismatch")
    manifest_code_hashes = _validate_code_hashes(
        manifest.get("collector_code_sha256"), "manifest"
    )
    expected_row_schema = {
        "name": ROW_SCHEMA,
        "fields": list(FIELDS),
        "step_seconds": STEP_SECONDS,
        "trajectory_steps": TRAJECTORY_STEPS,
        "transition_contract": "row r is (observation_r, context_r, action_r, observation_r+1, context_r+1)",
    }
    if manifest.get("row_schema") != expected_row_schema:
        raise ValueError("clean corpus row contract mismatch")

    cases = manifest.get("selected_cases")
    if (
        not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, str) for case in cases)
        or cases != sorted(set(cases))
        or any(case not in CASES for case in cases)
    ):
        raise ValueError("clean corpus selected cases are invalid")
    suffix = "all" if tuple(cases) == tuple(sorted(CASES)) else "-".join(cases)
    expected_manifest_name = f"{kind}_{suffix}_corpus_manifest.json"
    if manifest_path.name != expected_manifest_name:
        raise ValueError("clean corpus manifest filename does not match its selected cases")
    raw_root = output_root / raw_subdir

    plan_metadata = manifest.get("plans")
    file_metadata = manifest.get("files")
    if not isinstance(plan_metadata, dict) or set(plan_metadata) != set(cases):
        raise ValueError("corpus manifest plans do not match selected cases")
    if not isinstance(file_metadata, list) or not file_metadata:
        raise ValueError("corpus manifest has no source files")

    expected_by_path: dict[str, tuple[TrajectoryKey, dict]] = {}
    plan_hashes: list[tuple[str, str]] = []
    loaded_plans: dict[str, dict] = {}
    for case, metadata in sorted(plan_metadata.items()):
        if not isinstance(metadata, dict) or set(metadata) != {
            "path",
            "file_sha256",
            "plan_sha256",
            "schema",
            "source_sha256",
            "selected_entries",
        }:
            raise ValueError(f"invalid plan metadata for {case}")
        expected_plan_relative = f"plans/{expected_mode}/{case}.json"
        if metadata.get("path") != expected_plan_relative:
            raise ValueError(f"plan path does not match hardened layout for {case}")
        plan_path = _safe_path(output_root, expected_plan_relative)
        if not plan_path.is_file():
            raise FileNotFoundError(f"plan file is missing for {case}")
        actual_plan_file_sha = sha256_file(plan_path)
        if actual_plan_file_sha != metadata.get("file_sha256"):
            raise ValueError(f"plan file SHA-256 mismatch for {case}")
        plan = strict_json_loads(plan_path.read_text(encoding="ascii"))
        if not isinstance(plan, dict):
            raise ValueError(f"plan payload is invalid for {case}")
        _validate_plan_shape(plan, case, expected_mode)
        canonical_plan_hash = plan["plan_sha256"]
        if metadata.get("plan_sha256") != canonical_plan_hash:
            raise ValueError(f"manifest canonical plan hash mismatch for {case}")
        if metadata.get("schema") != expected_plan_schema:
            raise ValueError(f"manifest plan schema mismatch for {case}")
        if metadata.get("source_sha256") != plan["source_sha256"]:
            raise ValueError(f"manifest plan source hashes mismatch for {case}")
        if plan.get("collector_code_sha256") != manifest_code_hashes:
            raise ValueError(f"plan/manifest collector code hashes differ for {case}")
        if plan.get("public_source") != manifest["public_source"]:
            raise ValueError(f"plan/manifest public-source provenance differs for {case}")
        if plan.get("worker_runtime") != manifest["worker_runtime"]:
            raise ValueError(f"plan/manifest worker runtime differs for {case}")
        loaded_plans[case] = plan
        plan_hashes.append((case, canonical_plan_hash))
        adapter = plan["case_adapter"]
        seen_keys: set[TrajectoryKey] = set()
        seen_days: set[int] = set()
        seen_seeds: set[int] = set()
        for entry in plan["entries"]:
            key = _entry_key(entry, case)
            if key in seen_keys:
                raise ValueError(f"duplicate trajectory entry: {key.text}")
            if key.day in seen_days:
                raise ValueError(
                    f"case/day is assigned to multiple trajectories or roles: "
                    f"{case}:day{key.day:03d}"
                )
            if key.trajectory_seed in seen_seeds:
                raise ValueError(f"duplicate trajectory seed in plan for {case}")
            seen_keys.add(key)
            seen_days.add(key.day)
            seen_seeds.add(key.trajectory_seed)
            if key.role not in expected_roles:
                continue
            relative = f"{case}/{expected_filename(entry)}"
            if relative in expected_by_path:
                raise ValueError(f"duplicate expected corpus path: {relative}")
            expected_by_path[relative] = (key, adapter)
        selected_entries = sum(
            entry.get("role") in expected_roles for entry in plan["entries"]
        )
        if metadata.get("selected_entries") != selected_entries:
            raise ValueError(f"manifest selected-entry count mismatch for {case}")

    listed_paths: set[str] = set()
    listed_keys: set[TrajectoryKey] = set()
    records: list[CorpusRecord] = []
    for metadata in file_metadata:
        if not isinstance(metadata, dict) or set(metadata) != {
            "path",
            "case",
            "role",
            "day",
            "trajectory_seed",
            "plan_sha256",
            "sha256",
            "rows",
            "fields",
        }:
            raise ValueError("invalid source-file metadata")
        relative = str(metadata.get("path", ""))
        if relative in listed_paths:
            raise ValueError(f"duplicate source path in corpus manifest: {relative}")
        listed_paths.add(relative)
        if relative not in expected_by_path:
            raise ValueError(f"source file is not assigned by a case plan: {relative}")
        key, adapter = expected_by_path[relative]
        expected_identity = {
            "case": key.case,
            "role": key.role,
            "day": key.day,
            "trajectory_seed": key.trajectory_seed,
        }
        if (
            any(metadata.get(name) != value for name, value in expected_identity.items())
            or isinstance(metadata.get("day"), bool)
            or not isinstance(metadata.get("day"), int)
            or isinstance(metadata.get("trajectory_seed"), bool)
            or not isinstance(metadata.get("trajectory_seed"), int)
        ):
            raise ValueError(f"source-file identity metadata mismatch: {relative}")
        expected_plan_hash = dict(plan_hashes)[key.case]
        if metadata.get("plan_sha256") != expected_plan_hash:
            raise ValueError(f"source-file plan hash mismatch: {relative}")
        if key in listed_keys:
            raise ValueError(f"duplicate trajectory source: {key.text}")
        listed_keys.add(key)
        path = _safe_path(raw_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"trajectory source is missing: {relative}")
        expected_hash = metadata.get("sha256")
        if not _valid_sha256(expected_hash):
            raise ValueError(f"invalid source SHA-256 metadata: {relative}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"trajectory source SHA-256 mismatch: {relative}")
        header, row_count = _csv_metadata(path)
        if header != tuple(FIELDS) or len(set(header)) != len(header):
            raise ValueError(f"trajectory CSV schema mismatch: {relative}")
        declared_rows = metadata.get("rows")
        if isinstance(declared_rows, bool) or not isinstance(declared_rows, int):
            raise ValueError(f"invalid row-count metadata: {relative}")
        if row_count != declared_rows or row_count != TRAJECTORY_STEPS:
            raise ValueError(f"trajectory row count mismatch: {relative}")
        if metadata.get("fields") != len(FIELDS):
            raise ValueError(f"trajectory field-count metadata mismatch: {relative}")
        records.append(
            CorpusRecord(
                key=key,
                relative_path=relative,
                source_sha256=expected_hash,
                rows=row_count,
                step_seconds=STEP_SECONDS,
                base_setpoint_k=float(adapter["base_setpoint_k"]),
                action_amplitude_k=float(adapter["action_amplitude_k"]),
            )
        )

    if listed_paths != set(expected_by_path):
        missing = sorted(set(expected_by_path) - listed_paths)
        extra = sorted(listed_paths - set(expected_by_path))
        raise ValueError(f"corpus is incomplete; missing={missing}, extra={extra}")
    disk_paths = {
        str(path.relative_to(raw_root))
        for case in cases
        for path in (raw_root / case).rglob("*.csv")
    }
    if disk_paths != listed_paths:
        raise ValueError("raw corpus contains unmanifested or missing CSV files")

    role_counts = Counter(record.key.role for record in records)
    expected_counts = {
        "cases": len(cases),
        "trajectories": len(records),
        "rows": sum(record.rows for record in records),
        "roles": {role: role_counts[role] for role in sorted(role_counts)},
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("clean corpus manifest counts mismatch")

    receipt_metadata = manifest.get("receipts")
    if not isinstance(receipt_metadata, dict) or set(receipt_metadata) != set(cases):
        raise ValueError("clean corpus receipts do not match selected cases")
    expected_receipt_paths: set[Path] = set()
    for case in cases:
        metadata = receipt_metadata[case]
        plan_hash = dict(plan_hashes)[case]
        expected_relative = f"_receipts/{case}_{kind}_{plan_hash[:16]}.json"
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"path", "sha256", "receipt_sha256"}
            or metadata.get("path") != expected_relative
        ):
            raise ValueError(f"worker receipt path mismatch for {case}")
        receipt = _safe_path(raw_root, expected_relative)
        expected_receipt_paths.add(receipt)
        if not receipt.is_file():
            raise FileNotFoundError(f"worker receipt is missing for {case}")
        if sha256_file(receipt) != metadata.get("sha256"):
            raise ValueError(f"worker receipt file hash mismatch for {case}")
        receipt_wrapper = strict_json_loads(receipt.read_text(encoding="ascii"))
        if not isinstance(receipt_wrapper, dict) or set(receipt_wrapper) != {
            "receipt_sha256",
            "receipt",
        }:
            raise ValueError(f"worker receipt wrapper mismatch for {case}")
        receipt_payload = receipt_wrapper.get("receipt")
        receipt_hash = receipt_wrapper.get("receipt_sha256")
        if (
            not isinstance(receipt_payload, dict)
            or not _valid_sha256(receipt_hash)
            or _canonical_sha256(receipt_payload) != receipt_hash
            or metadata.get("receipt_sha256") != receipt_hash
        ):
            raise ValueError(f"worker receipt self-hash mismatch for {case}")
        expected_receipt_payload = {
            "schema": WORKER_RECEIPT_SCHEMA,
            "collection_kind": kind,
            "allowed_roles": sorted(expected_roles),
            "case": case,
            "plan_sha256": plan_hash,
            "worker_image_id": WORKER_IMAGE_ID,
            "boptest_version": WORKER_BOPTEST_VERSION,
            "collector_code_sha256": manifest_code_hashes,
            "source_sha256": loaded_plans[case]["source_sha256"],
            "prelock_registry_sha256": prelock_registry_sha256,
            "files": [
                {
                    "path": record.relative_path,
                    "sha256": record.source_sha256,
                    "rows": record.rows,
                }
                for record in records
                if record.key.case == case
            ],
        }
        if receipt_payload != expected_receipt_payload:
            raise ValueError(f"worker receipt payload mismatch for {case}")
    actual_receipt_paths = {
        path
        for path in (raw_root / "_receipts").glob("*.json")
        if any(path.name.startswith(f"{case}_") for case in cases)
    }
    if actual_receipt_paths != expected_receipt_paths:
        raise ValueError("raw corpus contains unmanifested or missing worker receipts")

    records.sort(key=lambda record: record.key)
    return CorpusIndex(
        root=raw_root.resolve(),
        manifest_path=manifest_path,
        manifest_sha256=recorded_manifest_hash,
        collection_kind=kind,
        prelock_registry_sha256=prelock_registry_sha256,
        allowed_roles=tuple(expected_roles),
        records=tuple(records),
        plan_sha256_by_case=tuple(plan_hashes),
    )


def _exact_integer_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain finite integers")
    return values.astype(np.int64)


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite clean values in columns {tuple(columns)}")
    return values


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class CleanTrajectory:
    key: TrajectoryKey
    source: Path
    source_sha256: str
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    contexts: np.ndarray
    next_contexts: np.ndarray


def load_clean_trajectory(
    index: CorpusIndex,
    record: CorpusRecord,
    *,
    allow_locked_test: bool = False,
) -> CleanTrajectory:
    """Load and validate one whole trajectory against its indexed identity."""
    if record not in index.records:
        raise ValueError("trajectory record is not part of this corpus index")
    if record.key.role == "locked_test" and not allow_locked_test:
        raise PermissionError("locked-test values require explicit authorization")
    path = _safe_path(index.root, record.relative_path)
    if sha256_file(path) != record.source_sha256:
        raise ValueError(f"trajectory changed after indexing: {record.key.text}")
    header, rows = _csv_metadata(path)
    if header != tuple(FIELDS) or rows != record.rows:
        raise ValueError(f"trajectory schema/count changed after indexing: {record.key.text}")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != tuple(FIELDS) or len(frame) != record.rows:
        raise ValueError(f"trajectory frame is incomplete: {record.key.text}")

    expected_text = {
        "case": record.key.case,
        "role": record.key.role,
    }
    for column, expected in expected_text.items():
        if frame[column].isna().any() or set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"trajectory {column} identity mismatch: {record.key.text}")
    for column, expected in (
        ("day", record.key.day),
        ("trajectory_seed", record.key.trajectory_seed),
    ):
        values = _exact_integer_column(frame, column)
        if not np.equal(values, expected).all():
            raise ValueError(f"trajectory {column} identity mismatch: {record.key.text}")

    steps = _exact_integer_column(frame, "step")
    if not np.array_equal(steps, np.arange(record.rows)):
        raise ValueError(f"trajectory steps are duplicated or incomplete: {record.key.text}")
    times = _exact_integer_column(frame, "time_s")
    expected_times = record.key.day * 86_400 + steps * record.step_seconds
    if not np.array_equal(times, expected_times):
        raise ValueError(f"trajectory time grid mismatch: {record.key.text}")

    observations = _numeric_matrix(frame, OBSERVATION_COLUMNS)
    next_observations = _numeric_matrix(frame, NEXT_OBSERVATION_COLUMNS)
    source_actions = _numeric_matrix(frame, SOURCE_ACTION_COLUMNS)
    contexts = _numeric_matrix(frame, CONTEXT_COLUMNS)
    next_contexts = _numeric_matrix(frame, NEXT_CONTEXT_COLUMNS)
    if not np.isin(source_actions[:, 0], (-1.0, 0.0, 1.0)).all():
        raise ValueError(f"normalized actions leave the frozen alphabet: {record.key.text}")
    expected_setpoint = (
        record.base_setpoint_k
        + record.action_amplitude_k * source_actions[:, 0]
    )
    if not np.allclose(
        source_actions[:, 1], expected_setpoint, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"setpoint/action identity mismatch: {record.key.text}")
    actions = source_actions[:, :1]
    if not np.array_equal(next_observations[:-1], observations[1:]):
        raise ValueError(f"next-observation continuity mismatch: {record.key.text}")
    if not np.array_equal(next_contexts[:-1], contexts[1:]):
        raise ValueError(f"next-context continuity mismatch: {record.key.text}")
    return CleanTrajectory(
        key=record.key,
        source=path,
        source_sha256=record.source_sha256,
        observations=_readonly(observations),
        next_observations=_readonly(next_observations),
        actions=_readonly(actions),
        contexts=_readonly(contexts),
        next_contexts=_readonly(next_contexts),
    )


def load_role_trajectories(
    index: CorpusIndex,
    role: Role,
    *,
    cases: Sequence[str] | None = None,
    allow_locked_test: bool = False,
) -> list[CleanTrajectory]:
    if role == "locked_test" and not allow_locked_test:
        raise PermissionError("locked-test values require explicit authorization")
    records = index.records_for(role, cases)
    trajectories = [
        load_clean_trajectory(
            index,
            record,
            allow_locked_test=allow_locked_test,
        )
        for record in records
    ]
    if len({trajectory.key for trajectory in trajectories}) != len(trajectories):
        raise ValueError(f"duplicate whole trajectories loaded for role {role}")
    return trajectories


@dataclass(frozen=True)
class ScaleStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "ScaleStats":
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("scaler input must be a nonempty finite matrix")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(tuple(float(value) for value in mean), tuple(float(value) for value in scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if values.shape[-1] != len(mean):
            raise ValueError("scaler dimension mismatch")
        return (values - mean) / scale


@dataclass(frozen=True)
class FaultScalers:
    observation: ScaleStats
    action: ScaleStats
    context: ScaleStats
    fit_source_sha256: tuple[tuple[str, str], ...]


def fit_scalers(trajectories: Sequence[CleanTrajectory]) -> FaultScalers:
    """Fit all continuous transforms on aligned, clean FIT rows only."""
    if not trajectories:
        raise ValueError("FIT scaler requires at least one trajectory")
    keys = [trajectory.key for trajectory in trajectories]
    if len(keys) != len(set(keys)):
        raise ValueError("FIT scaler received duplicate trajectories")
    if any(key.role != "fit" for key in keys):
        raise ValueError("scalers may be fitted on FIT trajectories only")
    return FaultScalers(
        observation=ScaleStats.fit(
            np.concatenate([trajectory.observations[1:] for trajectory in trajectories])
        ),
        action=ScaleStats.fit(
            np.concatenate([trajectory.actions[:-1] for trajectory in trajectories])
        ),
        context=ScaleStats.fit(
            np.concatenate([trajectory.contexts[1:] for trajectory in trajectories])
        ),
        fit_source_sha256=tuple(
            sorted((trajectory.key.text, trajectory.source_sha256) for trajectory in trajectories)
        ),
    )


@dataclass(frozen=True)
class FaultSpec:
    fit_onsets: tuple[int, ...] = (32, 96)
    validation_onsets: tuple[int, ...] = (48, 112)
    locked_test_onsets: tuple[int, ...] = (64, 128)
    duration: int = 48
    anchor_offsets: tuple[int, ...] = (8, 16, 24, 32)
    evaluation_horizon: int = 8
    zone_bias_severities: tuple[float, ...] = (1.0, 2.0)
    power_bias_severities: tuple[float, ...] = (250.0, 500.0)
    zone_drift_severities: tuple[float, ...] = (0.025, 0.05)
    power_drift_severities: tuple[float, ...] = (7.5, 15.0)

    def onsets_for(self, role: Role) -> tuple[int, ...]:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        return {
            "fit": self.fit_onsets,
            "validation": self.validation_onsets,
            "locked_test": self.locked_test_onsets,
        }[role]

    def __post_init__(self) -> None:
        if self.duration <= 0 or self.evaluation_horizon <= 0:
            raise ValueError("fault onset, duration, and horizon must be positive")
        if tuple(sorted(set(self.anchor_offsets))) != self.anchor_offsets:
            raise ValueError("anchor offsets must be sorted and unique")
        for role in ROLES:
            onsets = self.onsets_for(role)
            if not onsets or tuple(sorted(set(onsets))) != onsets:
                raise ValueError(f"{role} fault onsets must be nonempty, sorted, and unique")
            for onset in onsets:
                stop = onset + self.duration
                if onset < 1 or stop > TRAJECTORY_STEPS:
                    raise ValueError(f"{role} fault interval leaves a whole trajectory")
                if any(
                    offset < 1 or onset + offset + self.evaluation_horizon >= stop
                    for offset in self.anchor_offsets
                ):
                    raise ValueError(
                        f"a scored H8 target leaves the active {role} fault interval"
                    )
        severity_groups = (
            self.zone_bias_severities,
            self.power_bias_severities,
            self.zone_drift_severities,
            self.power_drift_severities,
        )
        if any(not group or any(value <= 0 for value in group) for group in severity_groups):
            raise ValueError("bias and drift severities must be positive")

    @classmethod
    def from_dict(cls, payload: dict) -> "FaultSpec":
        tuple_keys = {
            "fit_onsets",
            "validation_onsets",
            "locked_test_onsets",
            "anchor_offsets",
            "zone_bias_severities",
            "power_bias_severities",
            "zone_drift_severities",
            "power_drift_severities",
        }
        return cls(
            **{
                key: tuple(value) if key in tuple_keys else value
                for key, value in payload.items()
            }
        )


@dataclass(frozen=True)
class FaultCell:
    cell_id: str
    trajectory: TrajectoryKey
    source_sha256: str
    fault_channel: str
    family: Family
    sign: int
    severity: float
    severity_unit: str
    onset: int
    stop: int
    anchors: tuple[int, ...]


@dataclass(frozen=True)
class FaultManifest:
    schema: str
    corpus_manifest_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    spec: FaultSpec
    cells: tuple[FaultCell, ...]

    def payload(self) -> dict:
        return {
            "schema": self.schema,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "source_sha256": dict(self.source_sha256),
            "spec": asdict(self.spec),
            "cells": [
                {
                    **asdict(cell),
                    "trajectory": asdict(cell.trajectory),
                }
                for cell in self.cells
            ],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.payload())

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload(), indent=2) + "\n", encoding="ascii")

    @classmethod
    def read(cls, path: Path) -> "FaultManifest":
        payload = strict_json_loads(path.read_text(encoding="ascii"))
        cells = tuple(
            FaultCell(
                **{
                    **cell,
                    "trajectory": TrajectoryKey(**cell["trajectory"]),
                    "anchors": tuple(cell["anchors"]),
                }
            )
            for cell in payload["cells"]
        )
        return cls(
            schema=payload["schema"],
            corpus_manifest_sha256=payload["corpus_manifest_sha256"],
            source_sha256=tuple(sorted(payload["source_sha256"].items())),
            spec=FaultSpec.from_dict(payload["spec"]),
            cells=cells,
        )


def _severity_grid(spec: FaultSpec, channel: str, family: Family) -> tuple[tuple[int, float, str], ...]:
    if family == "bias":
        values = spec.zone_bias_severities if channel == FAULT_CHANNELS[0] else spec.power_bias_severities
        unit = "K" if channel == FAULT_CHANNELS[0] else "W"
        return tuple((sign, severity, unit) for severity in values for sign in (-1, 1))
    if family == "drift":
        values = spec.zone_drift_severities if channel == FAULT_CHANNELS[0] else spec.power_drift_severities
        unit = "K_per_step" if channel == FAULT_CHANNELS[0] else "W_per_step"
        return tuple((sign, severity, unit) for severity in values for sign in (-1, 1))
    if family == "dropout":
        return ((0, 1.0, "unavailable_fraction"),)
    return ((0, 0.0, "none"),)


def fault_cell_signatures(
    spec: FaultSpec, role: Role
) -> set[tuple[str, str, int, float, str, int, int, int]]:
    """Return the exact metadata grid expected for one trajectory and role."""
    return {
        (
            channel,
            family,
            sign,
            float(severity),
            unit,
            onset,
            onset + offset,
            spec.evaluation_horizon,
        )
        for channel in FAULT_CHANNELS
        for family in FAMILIES
        for onset in spec.onsets_for(role)
        for sign, severity, unit in _severity_grid(spec, channel, family)
        for offset in spec.anchor_offsets
    }


def build_fault_manifest(
    index: CorpusIndex, spec: FaultSpec | None = None
) -> FaultManifest:
    """Build a fixed grid from source metadata without opening trajectory values."""
    spec = FaultSpec() if spec is None else spec
    cells: list[FaultCell] = []
    for record in index.records:
        for channel in FAULT_CHANNELS:
            for family in FAMILIES:
                for onset in spec.onsets_for(record.key.role):
                    for sign, severity, unit in _severity_grid(spec, channel, family):
                        stop = onset + spec.duration
                        anchors = tuple(onset + offset for offset in spec.anchor_offsets)
                        severity_text = format(severity, ".12g").replace("-", "m").replace(".", "p")
                        cell_id = (
                            f"{record.key.text}:{channel}:{family}:o{onset}:"
                            f"s{sign:+d}:v{severity_text}"
                        )
                        cells.append(
                            FaultCell(
                                cell_id=cell_id,
                                trajectory=record.key,
                                source_sha256=record.source_sha256,
                                fault_channel=channel,
                                family=family,
                                sign=sign,
                                severity=float(severity),
                                severity_unit=unit,
                                onset=onset,
                                stop=stop,
                                anchors=anchors,
                            )
                        )
    cells.sort(key=lambda cell: cell.cell_id)
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise AssertionError("fault cell identifiers are not unique")
    return FaultManifest(
        schema=FAULT_MANIFEST_SCHEMA,
        corpus_manifest_sha256=index.manifest_sha256,
        source_sha256=tuple(sorted(index.source_hashes.items())),
        spec=spec,
        cells=tuple(cells),
    )


def validate_fault_manifest(manifest: FaultManifest, index: CorpusIndex) -> None:
    if manifest.schema != FAULT_MANIFEST_SCHEMA:
        raise ValueError("fault manifest schema mismatch")
    if manifest.corpus_manifest_sha256 != index.manifest_sha256:
        raise ValueError("fault manifest corpus provenance mismatch")
    if dict(manifest.source_sha256) != index.source_hashes:
        raise ValueError("fault manifest source hashes mismatch")
    expected = build_fault_manifest(index, manifest.spec)
    if manifest.payload() != expected.payload():
        raise ValueError("fault manifest differs from the deterministic fixed grid")


@dataclass(frozen=True)
class FaultVariant:
    cell: FaultCell
    source: Path
    clean_observations: np.ndarray
    corrupted_observations: np.ndarray
    actions: np.ndarray
    contexts: np.ndarray
    availability: np.ndarray
    age: np.ndarray
    health_labels: np.ndarray


def apply_fault(trajectory: CleanTrajectory, cell: FaultCell) -> FaultVariant:
    """Apply one causal persistent fault while retaining clean counterfactuals."""
    if trajectory.key != cell.trajectory:
        raise ValueError("fault cell and clean trajectory identities differ")
    if trajectory.source_sha256 != cell.source_sha256:
        raise ValueError("fault cell and clean trajectory source hashes differ")
    if cell.fault_channel not in FAULT_CHANNELS or cell.family not in FAMILIES:
        raise ValueError("fault cell has an unknown channel or family")
    if not (1 <= cell.onset < cell.stop <= len(trajectory.observations)):
        raise ValueError("fault interval leaves the clean trajectory")
    if any(anchor + 8 >= cell.stop for anchor in cell.anchors):
        raise ValueError("a scored H8 target is not inside the active fault")

    clean = trajectory.observations.copy()
    corrupted = clean.copy()
    availability = np.ones_like(clean, dtype=bool)
    age = np.zeros_like(clean, dtype=float)
    labels = np.zeros((len(clean), len(FAULT_CHANNELS)), dtype=np.int64)
    channel = FAULT_CHANNELS.index(cell.fault_channel)
    active = slice(cell.onset, cell.stop)
    labels[active, channel] = HEALTH_CLASS[cell.family]
    if cell.family == "bias":
        corrupted[active, channel] += cell.sign * cell.severity
    elif cell.family == "drift":
        corrupted[active, channel] += cell.sign * cell.severity * np.arange(
            1, cell.stop - cell.onset + 1
        )
    elif cell.family == "stuck":
        corrupted[active, channel] = clean[cell.onset - 1, channel]
    elif cell.family == "dropout":
        corrupted[active, channel] = np.nan
        availability[active, channel] = False
        age[active, channel] = np.arange(1, cell.stop - cell.onset + 1)
    return FaultVariant(
        cell=cell,
        source=trajectory.source,
        clean_observations=_readonly(clean),
        corrupted_observations=_readonly(corrupted),
        actions=trajectory.actions,
        contexts=trajectory.contexts,
        availability=_readonly(availability),
        age=_readonly(age),
        health_labels=_readonly(labels),
    )


def iter_role_variants(
    index: CorpusIndex,
    manifest: FaultManifest,
    role: Role,
    *,
    cases: Sequence[str] | None = None,
    allow_locked_test: bool = False,
) -> Iterator[FaultVariant]:
    validate_fault_manifest(manifest, index)
    trajectories = load_role_trajectories(
        index,
        role,
        cases=cases,
        allow_locked_test=allow_locked_test,
    )
    by_key = {trajectory.key: trajectory for trajectory in trajectories}
    expected_cells = [
        cell
        for cell in manifest.cells
        if cell.trajectory in by_key
    ]
    if {cell.trajectory for cell in expected_cells} != set(by_key):
        raise ValueError(f"fault manifest is incomplete for role {role}")
    for cell in expected_cells:
        yield apply_fault(by_key[cell.trajectory], cell)


@dataclass(frozen=True)
class SequenceReference:
    variant_index: int
    aligned_start: int


@dataclass(frozen=True)
class RSSMSequenceBatch:
    previous_actions: np.ndarray
    corrupted_observations: np.ndarray
    availability: np.ndarray
    age: np.ndarray
    contexts: np.ndarray
    clean_targets: np.ndarray
    health_labels: np.ndarray
    source_rows: np.ndarray


def materialize_rssm_batch(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    references: Sequence[SequenceReference],
    *,
    length: int,
) -> RSSMSequenceBatch:
    """Materialize aligned ``a_(t-1), c_t, corrupted y_t, clean y_t`` arrays."""
    if not references or length <= 0:
        raise ValueError("RSSM batch requires references and a positive length")
    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for reference in references:
        if not 0 <= reference.variant_index < len(variants):
            raise IndexError("sequence reference variant index is invalid")
        variant = variants[reference.variant_index]
        start = reference.aligned_start
        stop = start + length
        if start < 0 or stop >= len(variant.clean_observations):
            raise ValueError("sequence reference crosses a whole trajectory")
        current = slice(start + 1, stop + 1)
        previous = slice(start, stop)
        actions.append(scalers.action.transform(variant.actions[previous]))
        observations.append(
            scalers.observation.transform(variant.corrupted_observations[current])
        )
        availability.append(variant.availability[current])
        ages.append(variant.age[current])
        contexts.append(scalers.context.transform(variant.contexts[current]))
        targets.append(scalers.observation.transform(variant.clean_observations[current]))
        labels.append(variant.health_labels[current])
        rows.append(np.arange(start + 1, stop + 1, dtype=np.int64))
    return RSSMSequenceBatch(
        previous_actions=np.stack(actions, axis=1),
        corrupted_observations=np.stack(observations, axis=1),
        availability=np.stack(availability, axis=1),
        age=np.stack(ages, axis=1),
        contexts=np.stack(contexts, axis=1),
        clean_targets=np.stack(targets, axis=1),
        health_labels=np.stack(labels, axis=1),
        source_rows=np.stack(rows, axis=1),
    )
