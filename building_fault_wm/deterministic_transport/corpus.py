"""Fail-closed adapter for the paired v3 locked-transport corpus.

This module deliberately separates metadata validation from value loading.  A
caller must first validate the pre-lock, externally frozen readiness digest,
and immutable collection state.  Only then does ``load_transport_corpus_index``
call the collector's full numeric validator and expose a ``CorpusIndex`` that
the inherited fault-data pipeline can read with ``allow_locked_test=True``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.fault_data import (
    CorpusIndex,
    CorpusRecord,
    TrajectoryKey,
)

from . import collect, plan, worker_collect
from .evaluate import PolicyTrajectoryMetadata, metadata_from_case_plan


EXPECTED_CASES = tuple(sorted(boptest.CASES))
EXPECTED_WINDOWS = 12 * len(EXPECTED_CASES)
EXPECTED_BRANCHES = EXPECTED_WINDOWS * len(plan.POLICIES)
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "study_kind",
        "collection_kind",
        "output_role",
        "row_schema",
        "fields",
        "plan_sha256_by_case",
        "certificate_sha256",
        "certificate_file_sha256",
        "readiness_sha256",
        "collection_code_sha256",
        "source_sha256_by_case",
        "worker_image_id",
        "worker_boptest_version",
        "boptest_commit",
        "counts",
        "files",
        "worker_receipts",
    }
)


@dataclass(frozen=True)
class ValidatedCollection:
    """Metadata and state proven before any locked response value is returned."""

    index: CorpusIndex
    plans: dict[str, dict]
    trajectory_metadata: dict[TrajectoryKey, PolicyTrajectoryMetadata]
    manifest: dict
    manifest_file_sha256: str
    readiness: collect.Readiness
    attempt: dict
    completion: dict
    attempt_file_sha256: str
    completion_file_sha256: str


def _require_plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")
    return path


def _require_plain_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")
    return path


def _safe_file(root: Path, relative: object, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
    ):
        raise ValueError(f"{label} path is invalid")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its root") from error
    return _require_plain_file(candidate, label)


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} is not a UTC timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is not a valid UTC timestamp") from error


def load_bound_readiness(
    *,
    prelock_root: Path,
    live_data_root: Path,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
) -> tuple[dict, collect.Readiness]:
    """Validate the pre-lock and reconstruct the exact collected readiness.

    The live plans and certificate must be byte-identical to their pre-lock
    copies.  This does not open the locked manifest or trajectory CSVs.
    """

    from . import prelock

    if not boptest.valid_sha256(expected_prelock_sha256):
        raise ValueError("expected pre-lock digest is not a SHA-256")
    if not boptest.valid_sha256(expected_readiness_sha256):
        raise ValueError("expected readiness digest is not a SHA-256")
    registry = prelock.validate_prelock_bundle(
        prelock_root,
        parent_root=None,
        source_root=prelock.HERE,
    )
    if prelock.canonical_sha256(registry) != expected_prelock_sha256:
        raise ValueError("validated pre-lock differs from the external digest")

    bundle = prelock_root / prelock.BUNDLE_NAME
    frozen_plan_root = bundle / "plans/full"
    frozen_certificate = bundle / "plans/disjointness_certificate.json"
    live_plan_root = live_data_root / "plans/full"
    live_certificate = live_data_root / "disjointness_certificate.json"
    _require_plain_directory(live_plan_root, "live v3 plan root")
    _require_plain_file(live_certificate, "live v3 certificate")
    for case in EXPECTED_CASES:
        frozen = _require_plain_file(
            frozen_plan_root / f"{case}.json", "pre-lock v3 plan"
        )
        live = _require_plain_file(
            live_plan_root / f"{case}.json", "live v3 plan"
        )
        if prelock.sha256_file(live) != prelock.sha256_file(frozen):
            raise ValueError(f"live v3 plan differs from pre-lock: {case}")
    if prelock.sha256_file(live_certificate) != prelock.sha256_file(
        frozen_certificate
    ):
        raise ValueError("live v3 certificate differs from pre-lock")

    plans, plan_paths = collect._load_plan_grid(frozen_plan_root)
    certificate = worker_collect._load_json_strict(frozen_certificate)
    certificate_sha256 = certificate.get("certificate_sha256")
    if not isinstance(certificate_sha256, str):
        raise ValueError("pre-lock certificate has no canonical digest")
    worker_collect.validate_certificate_grid(
        certificate, plans, certificate_sha256
    )
    public_source = {
        "repository_url": boptest.BOPTEST_REPOSITORY_URL,
        "commit": boptest.BOPTEST_COMMIT,
        "license_sha256": boptest.BOPTEST_LICENSE_SHA256,
    }
    prelock_binding = {
        "prelock_registry_sha256": expected_prelock_sha256,
        "prelock_registry_file_sha256": prelock.sha256_file(
            prelock_root / prelock.REGISTRY_NAME
        ),
        "prelock_bundle_inventory_sha256": str(
            registry["bundle_inventory_sha256"]
        ),
    }
    payload = {
        **collect._readiness_payload(
            plans,
            certificate_sha256,
            runtime_validated=True,
            public_source=public_source,
            prelock_binding=prelock_binding,
        ),
        "certificate_file_sha256": prelock.sha256_file(frozen_certificate),
    }
    report = {
        **payload,
        "readiness_sha256": plan.canonical_sha256(payload),
    }
    if report["readiness_sha256"] != expected_readiness_sha256:
        raise ValueError(
            "reconstructed readiness differs from the externally frozen digest"
        )
    if report["plan_sha256_by_case"] != registry["plans"][
        "plan_sha256_by_case"
    ]:
        raise ValueError("readiness plan grid differs from the pre-lock registry")
    if report["certificate_sha256"] != registry["plans"][
        "certificate_sha256"
    ]:
        raise ValueError("readiness certificate differs from the pre-lock registry")
    return registry, collect.Readiness(
        plans=plans,
        plan_paths=plan_paths,
        certificate=certificate,
        expected_certificate_sha256=certificate_sha256,
        collection_code_sha256=collect.collection_code_hashes(),
        report=report,
    )


def validate_collection_completion(
    *,
    state_root: Path,
    readiness: collect.Readiness,
    manifest_path: Path,
    expected_prelock_sha256: str,
    external_freeze: Mapping[str, object],
    external_freeze_receipt_path: Path,
) -> tuple[dict, dict, str, str]:
    """Validate immutable attempt/completion evidence before value access."""

    readiness_sha256 = str(readiness.report["readiness_sha256"])
    state_dir = state_root / readiness_sha256
    _require_plain_directory(state_dir, "v3 collection state")
    attempt_path = _require_plain_file(
        state_dir / collect.ATTEMPT_MARKER, "v3 collection attempt"
    )
    completion_path = _require_plain_file(
        state_dir / collect.COMPLETION_MARKER, "v3 collection completion"
    )
    if os.path.lexists(state_dir / collect.FAILURE_MARKER):
        raise ValueError("v3 collection state contains a failure marker")
    attempt = plan.load_json(attempt_path)
    completion = plan.load_json(completion_path)
    if attempt.get("schema") != collect.ATTEMPT_SCHEMA:
        raise ValueError("v3 collection attempt schema changed")
    if completion.get("schema") != collect.COMPLETION_SCHEMA:
        raise ValueError("v3 collection completion schema changed")
    if (
        attempt.get("stage") != "locked_transport_collection"
        or completion.get("stage") != "locked_transport_collection_complete"
    ):
        raise ValueError("v3 collection state stage changed")
    expected_common = {
        "certificate_sha256": readiness.expected_certificate_sha256,
        "readiness_sha256": readiness_sha256,
        "prelock_registry_sha256": expected_prelock_sha256,
    }
    for field, expected in expected_common.items():
        if attempt.get(field) != expected or completion.get(field) != expected:
            raise ValueError(f"v3 collection state {field} differs")
    if attempt.get("collection_code_sha256") != readiness.collection_code_sha256:
        raise ValueError("v3 attempt collection-code binding changed")
    if completion.get("collection_code_sha256") != readiness.collection_code_sha256:
        raise ValueError("v3 completion collection-code binding changed")
    attempt_sha256 = plan.sha256_file(attempt_path)
    if completion.get("attempt_marker_sha256") != attempt_sha256:
        raise ValueError("v3 completion binds a different collection attempt")
    if Path(str(completion.get("manifest_path"))).resolve() != manifest_path.resolve():
        raise ValueError("v3 completion binds a different corpus manifest path")
    if completion.get("manifest_file_sha256") != plan.sha256_file(manifest_path):
        raise ValueError("v3 completion corpus-manifest file hash changed")
    if completion.get("locked_response_values_accessed_after_attempt") is not True:
        raise ValueError("v3 completion does not acknowledge locked value access")
    if (
        attempt.get("plan_sha256_by_case")
        != readiness.report["plan_sha256_by_case"]
        or attempt.get("protocol_sha256") != readiness.report["protocol_sha256"]
        or attempt.get("worker_image_id") != boptest.WORKER_IMAGE_ID
        or attempt.get("worker_boptest_version")
        != boptest.WORKER_BOPTEST_VERSION
        or attempt.get("locked_response_values_accessed") is not False
    ):
        raise ValueError("v3 collection attempt frozen contract changed")
    if attempt.get("external_freeze_receipt_sha256") != plan.sha256_file(
        external_freeze_receipt_path
    ):
        raise ValueError("v3 attempt binds a different external-freeze receipt")
    for attempt_field, freeze_field in (
        ("external_freeze_gist_id", "gist_id"),
        ("external_freeze_revision", "revision"),
        (
            "external_freeze_revision_committed_at_utc",
            "revision_committed_at_utc",
        ),
    ):
        if attempt.get(attempt_field) != external_freeze.get(freeze_field):
            raise ValueError(f"v3 attempt {attempt_field} differs from the freeze")
    freeze_time = _parse_timestamp(
        external_freeze.get("revision_committed_at_utc"),
        "external freeze commit time",
    )
    attempt_time = _parse_timestamp(
        attempt.get("started_at_utc"), "collection attempt time"
    )
    completion_time = _parse_timestamp(
        completion.get("completed_at_utc"), "collection completion time"
    )
    if freeze_time > attempt_time:
        raise ValueError("external freeze was committed after collection began")
    if completion_time < attempt_time:
        raise ValueError("collection completion predates its attempt")
    return (
        attempt,
        completion,
        attempt_sha256,
        plan.sha256_file(completion_path),
    )


def _csv_header_count(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty v3 trajectory: {path}") from error
        return header, sum(1 for _ in reader)


def _metadata_from_plans(
    plans: Mapping[str, Mapping[str, object]],
) -> dict[TrajectoryKey, PolicyTrajectoryMetadata]:
    result: dict[TrajectoryKey, PolicyTrajectoryMetadata] = {}
    for case in EXPECTED_CASES:
        case_metadata = metadata_from_case_plan(plans[case])
        if set(result).intersection(case_metadata):
            raise ValueError("v3 plan grid repeats a trajectory identity")
        result.update(case_metadata)
    if len(result) != EXPECTED_BRANCHES:
        raise ValueError("v3 trajectory metadata grid is incomplete")
    return result


def load_transport_corpus_index(
    *,
    manifest_path: Path,
    raw_root: Path,
    readiness: collect.Readiness,
    expected_prelock_sha256: str,
    state_root: Path,
    external_freeze: Mapping[str, object],
    external_freeze_receipt_path: Path,
) -> ValidatedCollection:
    """Validate and adapt the custom paired corpus after all lock conditions."""

    _require_plain_file(manifest_path, "v3 corpus manifest")
    _require_plain_directory(raw_root, "v3 locked raw root")
    attempt, completion, attempt_sha, completion_sha = (
        validate_collection_completion(
            state_root=state_root,
            readiness=readiness,
            manifest_path=manifest_path,
            expected_prelock_sha256=expected_prelock_sha256,
            external_freeze=external_freeze,
            external_freeze_receipt_path=external_freeze_receipt_path,
        )
    )

    wrapper = plan.load_json(manifest_path)
    if set(wrapper) != {"manifest_sha256", "manifest"}:
        raise ValueError("v3 corpus manifest wrapper fields changed")
    manifest = wrapper.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("v3 corpus manifest fields changed")
    manifest_sha256 = plan.canonical_sha256(manifest)
    if wrapper.get("manifest_sha256") != manifest_sha256:
        raise ValueError("v3 corpus manifest self-hash changed")
    if completion.get("manifest_payload_sha256") != manifest_sha256:
        raise ValueError("v3 completion binds a different manifest payload")

    # This is the first operation allowed to parse response values.  It checks
    # every number, action, continuity relation, paired-state hash, receipt,
    # and complete raw-tree inventory against the frozen plans.
    inventory = collect.validate_staged_collection(raw_root, readiness)
    expected_wrapper = collect.build_manifest(readiness, inventory)
    if wrapper != expected_wrapper:
        raise ValueError("published v3 corpus differs from collector reconstruction")

    if (
        manifest["schema"] != collect.CORPUS_MANIFEST_SCHEMA
        or manifest["study_kind"] != worker_collect.STUDY_KIND
        or manifest["collection_kind"] != "paired_locked_transport"
        or manifest["output_role"] != "locked_test"
        or manifest["fields"] != list(worker_collect.FIELDS)
        or manifest["readiness_sha256"]
        != readiness.report["readiness_sha256"]
    ):
        raise ValueError("v3 corpus identity contract changed")
    expected_counts = {
        "cases": len(EXPECTED_CASES),
        "windows": EXPECTED_WINDOWS,
        "branches": EXPECTED_BRANCHES,
        "rows_per_branch": boptest.TRAJECTORY_STEPS,
    }
    if manifest["counts"] != expected_counts:
        raise ValueError("v3 corpus counts differ from the frozen grid")

    metadata = _metadata_from_plans(readiness.plans)
    expected_by_path: dict[str, tuple[TrajectoryKey, object, str]] = {}
    for case in EXPECTED_CASES:
        adapter = boptest.CASES[case]
        for entry in readiness.plans[case]["entries"]:
            for policy in plan.POLICIES:
                branch = entry["policies"][policy]
                key = TrajectoryKey(
                    case=case,
                    role="locked_test",
                    day=int(entry["day"]),
                    trajectory_seed=int(branch["trajectory_seed"]),
                )
                relative = (
                    f"{case}/{worker_collect.expected_filename(entry, policy)}"
                )
                expected_by_path[relative] = (
                    key,
                    adapter,
                    str(readiness.plans[case]["plan_sha256"]),
                )
    if len(expected_by_path) != EXPECTED_BRANCHES:
        raise ValueError("v3 expected source path grid is incomplete")

    records: list[CorpusRecord] = []
    listed: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise ValueError("v3 source-file metadata is invalid")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in listed:
            raise ValueError("v3 source path is invalid or duplicated")
        listed.add(relative)
        if relative not in expected_by_path:
            raise ValueError(f"v3 source lies outside the frozen plans: {relative}")
        key, adapter, _ = expected_by_path[relative]
        expected_identity = {
            "policy": metadata[key].policy,
            "day": key.day,
            "scenario_seed": metadata[key].scenario_seed,
            "trajectory_seed": key.trajectory_seed,
            "rows": boptest.TRAJECTORY_STEPS,
            "fields": len(worker_collect.FIELDS),
        }
        if any(item.get(field) != value for field, value in expected_identity.items()):
            raise ValueError(f"v3 source identity metadata differs: {relative}")
        source = _safe_file(raw_root, relative, "v3 trajectory")
        source_sha256 = item.get("sha256")
        if (
            not boptest.valid_sha256(source_sha256)
            or plan.sha256_file(source) != source_sha256
        ):
            raise ValueError(f"v3 source hash differs: {relative}")
        header, row_count = _csv_header_count(source)
        if (
            header != tuple(worker_collect.FIELDS)
            or row_count != boptest.TRAJECTORY_STEPS
        ):
            raise ValueError(f"v3 source schema/count differs: {relative}")
        records.append(
            CorpusRecord(
                key=key,
                relative_path=relative,
                source_sha256=str(source_sha256),
                rows=row_count,
                step_seconds=boptest.STEP_SECONDS,
                base_setpoint_k=float(adapter.base_setpoint_k),
                action_amplitude_k=float(adapter.action_amplitude_k),
            )
        )
    if listed != set(expected_by_path):
        raise ValueError("v3 corpus file grid is incomplete")
    records.sort(key=lambda record: record.key)
    index = CorpusIndex(
        root=raw_root.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
        collection_kind="locked_test",
        prelock_registry_sha256=expected_prelock_sha256,
        allowed_roles=("locked_test",),
        records=tuple(records),
        plan_sha256_by_case=tuple(
            sorted(
                (case, str(readiness.plans[case]["plan_sha256"]))
                for case in EXPECTED_CASES
            )
        ),
    )
    if set(metadata) != {record.key for record in records}:
        raise ValueError("v3 plan metadata and corpus identities differ")
    return ValidatedCollection(
        index=index,
        plans=readiness.plans,
        trajectory_metadata=metadata,
        manifest=manifest,
        manifest_file_sha256=plan.sha256_file(manifest_path),
        readiness=readiness,
        attempt=attempt,
        completion=completion,
        attempt_file_sha256=attempt_sha,
        completion_file_sha256=completion_sha,
    )
