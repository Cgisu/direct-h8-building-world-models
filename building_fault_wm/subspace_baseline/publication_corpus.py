"""Read-only adapter for the verified v6 publication transport corpus."""

from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from building_fault_wm.deterministic_transport.evaluate import (
    PolicyTrajectoryMetadata,
    metadata_from_case_plan,
)
from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.fault_data import (
    CorpusIndex,
    CorpusRecord,
    TrajectoryKey,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "external_evidence/neural_evaluation_package"
EXPECTED_PACKAGE_DIGEST = (
    "2251383ce5eb5fe7d1b62e05feb902d8e9b66cec1971ddb6df17cd528667b4c7"
)
LEGACY_ROOT = PACKAGE_ROOT / "legacy_v5_package"
CORPUS_ROOT = LEGACY_ROOT / "corpus"
MANIFEST_PATH = CORPUS_ROOT / "locked_transport_corpus_manifest.json"
RAW_ROOT = CORPUS_ROOT / "locked_transport_raw"
PLAN_ROOT = LEGACY_ROOT / "prelock/bundle/plans/full"
READINESS_PATH = LEGACY_ROOT / "external_freeze/collection_readiness.json"
COLLECTION_COMPLETION_PATH = (
    LEGACY_ROOT / "collection_state/v3_paired_collection_completion.json"
)
NEURAL_EVALUATION_ROOT = LEGACY_ROOT / "evaluation"
EXPECTED_CASES = tuple(sorted(boptest.CASES))
EXPECTED_BRANCHES = 72
EXPECTED_ROWS = 192


@dataclass(frozen=True)
class PublicationCollection:
    """Verified corpus index and policy metadata for numerical evaluation."""

    index: CorpusIndex
    trajectory_metadata: dict[TrajectoryKey, PolicyTrajectoryMetadata]
    manifest: dict
    manifest_file_sha256: str
    package_binding: dict


def _verify_external_package() -> dict:
    """Load the verifier carried by the separately deposited evidence package."""

    verifier_path = PACKAGE_ROOT / "publication_tools/verify_package.py"
    if not verifier_path.is_file():
        raise FileNotFoundError(
            "place the separately deposited neural evaluation package at "
            f"{PACKAGE_ROOT} before running the full comparator evaluation"
        )
    spec = importlib.util.spec_from_file_location("direct_h8_package_verifier", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evidence verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_package(
        PACKAGE_ROOT, expected_digest=EXPECTED_PACKAGE_DIGEST, require_read_only=True
    )


def package_binding() -> dict[str, object]:
    """Verify the immutable package without interpreting trajectory numbers."""

    verified = _verify_external_package()
    wrapper = strict_json(MANIFEST_PATH)
    readiness = strict_json(READINESS_PATH)
    completion = strict_json(COLLECTION_COMPLETION_PATH)
    manifest = wrapper.get("manifest")
    if (
        set(wrapper) != {"manifest", "manifest_sha256"}
        or not isinstance(manifest, dict)
        or wrapper.get("manifest_sha256") != canonical_sha256(manifest)
        or readiness.get("readiness_sha256") != manifest.get("readiness_sha256")
        or completion.get("manifest_payload_sha256")
        != wrapper.get("manifest_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(MANIFEST_PATH)
    ):
        raise ValueError("publication corpus identity changed")
    return {
        "publication_package_digest": verified["package_digest"],
        "legacy_package_digest": verified["legacy_v5_package_digest"],
        "publication_verification_payload_sha256": canonical_sha256(verified),
        "corpus_manifest_file_sha256": sha256_file(MANIFEST_PATH),
        "corpus_manifest_payload_sha256": wrapper["manifest_sha256"],
        "collection_readiness_file_sha256": sha256_file(READINESS_PATH),
        "collection_completion_file_sha256": sha256_file(
            COLLECTION_COMPLETION_PATH
        ),
        "neural_completion_file_sha256": sha256_file(
            NEURAL_EVALUATION_ROOT / "evaluation_complete.json"
        ),
        "neural_gate_core_file_sha256": sha256_file(
            NEURAL_EVALUATION_ROOT / "gate_core.csv"
        ),
    }


def _policy_metadata(
    plans: Mapping[str, Mapping[str, object]],
) -> dict[TrajectoryKey, PolicyTrajectoryMetadata]:
    metadata: dict[TrajectoryKey, PolicyTrajectoryMetadata] = {}
    for case in EXPECTED_CASES:
        case_metadata = metadata_from_case_plan(plans[case])
        if set(metadata).intersection(case_metadata):
            raise ValueError("publication plans repeat a trajectory identity")
        metadata.update(case_metadata)
    if len(metadata) != EXPECTED_BRANCHES:
        raise ValueError("publication policy metadata grid is incomplete")
    return metadata


def _safe_raw_file(relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("publication trajectory path is invalid")
    root = RAW_ROOT.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("publication trajectory path escapes corpus") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("publication trajectory is not a plain file")
    return path


def _header_and_rows(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError("publication trajectory is empty") from error
        return header, sum(1 for _ in reader)


def load_publication_collection() -> PublicationCollection:
    """Verify metadata, then expose the sealed corpus through ``CorpusIndex``."""

    binding = package_binding()
    wrapper = strict_json(MANIFEST_PATH)
    manifest = wrapper["manifest"]
    plans = {case: strict_json(PLAN_ROOT / f"{case}.json") for case in EXPECTED_CASES}
    if manifest.get("plan_sha256_by_case") != {
        case: plans[case].get("plan_sha256") for case in EXPECTED_CASES
    }:
        raise ValueError("publication plan grid differs from corpus manifest")
    metadata = _policy_metadata(plans)
    files = manifest.get("files")
    fields = manifest.get("fields")
    counts = manifest.get("counts")
    if (
        not isinstance(files, list)
        or len(files) != EXPECTED_BRANCHES
        or not isinstance(fields, list)
        or counts
        != {
            "cases": len(EXPECTED_CASES),
            "windows": 36,
            "branches": EXPECTED_BRANCHES,
            "rows_per_branch": EXPECTED_ROWS,
        }
        or manifest.get("collection_kind") != "paired_locked_transport"
        or manifest.get("output_role") != "locked_test"
    ):
        raise ValueError("publication corpus contract changed")
    records = []
    listed_keys = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("publication corpus inventory row is invalid")
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ValueError("publication corpus path is invalid")
        case = relative.split("/", 1)[0]
        key = TrajectoryKey(
            case=case,
            role="locked_test",
            day=int(item["day"]),
            trajectory_seed=int(item["trajectory_seed"]),
        )
        if key in listed_keys or key not in metadata:
            raise ValueError("publication trajectory identity is invalid")
        listed_keys.add(key)
        policy = metadata[key]
        if (
            item.get("policy") != policy.policy
            or item.get("scenario_seed") != policy.scenario_seed
            or item.get("rows") != EXPECTED_ROWS
            or item.get("fields") != len(fields)
        ):
            raise ValueError("publication trajectory metadata changed")
        path = _safe_raw_file(relative)
        if sha256_file(path) != item.get("sha256"):
            raise ValueError("publication trajectory hash changed")
        header, rows = _header_and_rows(path)
        if header != tuple(fields) or rows != EXPECTED_ROWS:
            raise ValueError("publication trajectory shape changed")
        adapter = boptest.CASES[case]
        records.append(
            CorpusRecord(
                key=key,
                relative_path=relative,
                source_sha256=str(item["sha256"]),
                rows=rows,
                step_seconds=boptest.STEP_SECONDS,
                base_setpoint_k=float(adapter.base_setpoint_k),
                action_amplitude_k=float(adapter.action_amplitude_k),
            )
        )
    if listed_keys != set(metadata):
        raise ValueError("publication trajectory grid is incomplete")
    records.sort(key=lambda record: record.key)
    readiness = strict_json(READINESS_PATH)
    index = CorpusIndex(
        root=RAW_ROOT.resolve(),
        manifest_path=MANIFEST_PATH.resolve(),
        manifest_sha256=str(wrapper["manifest_sha256"]),
        collection_kind="locked_test",
        prelock_registry_sha256=str(readiness["prelock_registry_sha256"]),
        allowed_roles=("locked_test",),
        records=tuple(records),
        plan_sha256_by_case=tuple(
            sorted(
                (case, str(plans[case]["plan_sha256"]))
                for case in EXPECTED_CASES
            )
        ),
    )
    return PublicationCollection(
        index=index,
        trajectory_metadata=metadata,
        manifest=manifest,
        manifest_file_sha256=sha256_file(MANIFEST_PATH),
        package_binding=binding,
    )
