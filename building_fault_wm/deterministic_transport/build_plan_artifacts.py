"""Build the value-blind v3 plan grid and prior-identity certificate once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping

from building_fault_wm.neural_benchmark import protocol as boptest

from . import plan
from . import worker_collect


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
MULTICASE_ROOT = PROJECT_ROOT / "building_fault_wm/neural_benchmark"
PARENT_PACKAGE_ROOT = PROJECT_ROOT / "artifacts/direct_h8_publication_v2"
PARENT_EXPERIMENT_ROOT = PARENT_PACKAGE_ROOT / "experiment"
V1_PLAN_ROOT = PARENT_EXPERIMENT_ROOT / "prelock_bundle/corpus/plans/full"
V2_PLAN_ROOT = PARENT_EXPERIMENT_ROOT / "prelock_bundle/recovery/v2/plans/full"
OUTPUT_ROOT = MULTICASE_ROOT / "data_v6"
PARENT_PACKAGE_DIGEST = (
    "b758859c6cb99d34930452c36e3fd59b5abd0e7f56b19710fa2b1998b23760b8"
)
PARENT_MANIFEST = "package_manifest.json"
PARENT_DIGEST_RECORD = "package_manifest.canonical.sha256"
LOCAL_SOURCE_PREFIX = "multicase:"
PARENT_SOURCE_LABEL = "immutable_parent_v2"
CSV_IDENTITY_FIELDS = ("case", "day", "trajectory_seed")
_IDENTITY_PATH = re.compile(
    rf"(?:^|/)(?P<case>{'|'.join(re.escape(case) for case in sorted(boptest.CASES))})/"
    r"day(?P<day>[0-9]+)_[^/]*_seed(?P<seed>[0-9]+)\.csv$"
)


def _strict_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON evidence is not a plain file: {path}")

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
        raise ValueError(f"JSON evidence is not an object: {path}")
    return payload


def _parent_payload_sha256(payload: object) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def verify_parent_package(
    parent_root: Path,
    *,
    expected_digest: str = PARENT_PACKAGE_DIGEST,
) -> dict:
    """Verify every byte listed by the immutable parent publication package."""

    if not plan.valid_sha256(expected_digest):
        raise ValueError("expected parent package digest is invalid")
    if parent_root.is_symlink() or not parent_root.is_dir():
        raise ValueError("immutable parent package root is invalid")
    manifest_path = parent_root / PARENT_MANIFEST
    digest_path = parent_root / PARENT_DIGEST_RECORD
    manifest = _strict_json(manifest_path)
    if _parent_payload_sha256(manifest) != expected_digest:
        raise ValueError("immutable parent package canonical digest changed")
    if (
        digest_path.is_symlink()
        or not digest_path.is_file()
        or digest_path.read_text(encoding="ascii").strip() != expected_digest
    ):
        raise ValueError("immutable parent package digest record changed")
    inventory = manifest.get("artifact_inventory_excludes_manifest_and_digest")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("immutable parent package inventory is incomplete")
    seen = set()
    byte_count = 0
    root_resolved = parent_root.resolve()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("immutable parent package inventory row is invalid")
        relative = item["path"]
        size = item["bytes"]
        digest = item["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not plan.valid_sha256(digest)
        ):
            raise ValueError("immutable parent package inventory row is malformed")
        seen.add(relative)
        candidate = parent_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"immutable parent artifact is not a plain file: {relative}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError("immutable parent inventory path escapes its root") from error
        if candidate.stat().st_size != size:
            raise ValueError(f"immutable parent artifact size changed: {relative}")
        if plan.sha256_file(candidate) != digest:
            raise ValueError(f"immutable parent artifact digest changed: {relative}")
        byte_count += size
    return {
        "canonical_digest": expected_digest,
        "inventory_file_count": len(inventory),
        "inventory_bytes": byte_count,
    }


def _identity_from_path(value: str) -> dict[str, object] | None:
    match = _IDENTITY_PATH.search(value.replace("\\", "/"))
    if match is None:
        return None
    return plan.normalize_trajectory_identity(
        {
            "case": match.group("case"),
            "day": int(match.group("day")),
            "trajectory_seed": int(match.group("seed")),
        }
    )


def _parse_csv_identities(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if (
            not isinstance(header, list)
            or len(header) != len(set(header))
            or any(field not in header for field in CSV_IDENTITY_FIELDS)
        ):
            raise ValueError(f"prior trajectory CSV identity header is invalid: {path}")
        identities = set()
        row_count = 0
        for row in reader:
            row_count += 1
            try:
                identity = plan.normalize_trajectory_identity(
                    {
                        "case": row["case"],
                        "day": int(row["day"]),
                        "trajectory_seed": int(row["trajectory_seed"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"prior trajectory CSV identity row is invalid: {path}"
                ) from error
            identities.add(
                (
                    str(identity["case"]),
                    int(identity["day"]),
                    int(identity["trajectory_seed"]),
                )
            )
    if row_count != boptest.TRAJECTORY_STEPS or len(identities) != 1:
        raise ValueError(
            f"prior trajectory CSV is not one complete 192-row identity: {path}"
        )
    identity = next(iter(identities))
    observed = {
        "case": identity[0],
        "day": identity[1],
        "trajectory_seed": identity[2],
    }
    named = _identity_from_path(path.as_posix())
    if named is not None and named != observed:
        raise ValueError(f"prior trajectory CSV filename identity differs: {path}")
    return [observed]


def _extract_json_identities(payload: object) -> list[dict[str, object]]:
    found: dict[tuple[str, int, int], dict[str, object]] = {}

    def add(identity: Mapping[str, object]) -> None:
        normalized = plan.normalize_trajectory_identity(identity)
        key = (
            str(normalized["case"]),
            int(normalized["day"]),
            int(normalized["trajectory_seed"]),
        )
        found[key] = normalized

    def walk(value: object) -> None:
        if isinstance(value, dict):
            explicit = None
            if all(field in value for field in CSV_IDENTITY_FIELDS):
                explicit = plan.normalize_trajectory_identity(value)
                add(explicit)
            referenced = None
            path_value = value.get("path")
            if isinstance(path_value, str):
                referenced = _identity_from_path(path_value)
                if referenced is not None:
                    add(referenced)
            if explicit is not None and referenced is not None and explicit != referenced:
                raise ValueError("JSON evidence path and explicit identity disagree")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return [found[key] for key in sorted(found)]


def _classify_evidence(path: Path, *, parent: bool) -> str | None:
    parts = path.parts
    name = path.name.lower()
    if path.suffix.lower() == ".csv" and any("raw" in part.lower() for part in parts):
        return "raw_csv"
    if path.suffix.lower() != ".json":
        return None
    if "_receipts" in parts or "receipt" in name:
        return "receipt_json"
    if "manifests" in parts or "manifest" in name:
        return "manifest_json"
    if parent and (
        "collection_state" in parts
        or ("control_evidence" in parts and "collection" in name)
    ):
        return "collection_record_json"
    return None


def _scan_evidence_source(
    root: Path,
    *,
    label: str,
    parent: bool,
) -> list[dict[str, object]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prior evidence source is not a plain directory: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        kind = _classify_evidence(path.relative_to(root), parent=parent)
        if kind is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"prior evidence is not a plain file: {path}")
        if kind == "raw_csv":
            identities = _parse_csv_identities(path)
        else:
            identities = _extract_json_identities(_strict_json(path))
        records.append(
            {
                "source": label,
                "path": path.relative_to(root).as_posix(),
                "kind": kind,
                "sha256": plan.sha256_file(path),
                "bytes": path.stat().st_size,
                "identities": identities,
            }
        )
    return records


def scan_prior_evidence(
    multicase_root: Path,
    parent_experiment_root: Path,
    *,
    parent_package_digest: str,
) -> dict:
    """Inventory identity-bearing evidence without reading any v3 destination."""

    scope = []
    records = []
    for suffix in ("", "_v1", "_v2", "_v3", "_v4", "_v5"):
        name = f"data{suffix}"
        root = multicase_root / name
        label = f"{LOCAL_SOURCE_PREFIX}{name}"
        present = root.exists()
        scope.append(
            {
                "label": label,
                "kind": "local_multicase_namespace",
                "present": present,
                "root_binding_sha256": None,
            }
        )
        if present:
            records.extend(_scan_evidence_source(root, label=label, parent=False))

    scope.append(
        {
            "label": PARENT_SOURCE_LABEL,
            "kind": "immutable_parent_package",
            "present": True,
            "root_binding_sha256": parent_package_digest,
        }
    )
    records.extend(
        _scan_evidence_source(
            parent_experiment_root,
            label=PARENT_SOURCE_LABEL,
            parent=True,
        )
    )
    v2_identities = [
        identity
        for record in records
        if record["source"] == f"{LOCAL_SOURCE_PREFIX}data_v5"
        and record["kind"] == "raw_csv"
        and str(record["path"]).startswith("locked_test_raw/")
        for identity in record["identities"]  # type: ignore[union-attr]
    ]
    return plan.build_prior_evidence_contract(
        scope,
        records,
        v2_locked_csv_identities=v2_identities,
    )


def _load_plan_grid(root: Path, label: str) -> dict[str, dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} plan root is not a plain directory")
    expected = {f"{case}.json" for case in boptest.CASES}
    paths = list(root.iterdir())
    if (
        {path.name for path in paths} != expected
        or any(path.is_symlink() or not path.is_file() for path in paths)
    ):
        raise ValueError(f"{label} plan root differs from the three-case contract")
    result = {}
    for case in sorted(boptest.CASES):
        payload = _strict_json(root / f"{case}.json")
        unsigned = {key: value for key, value in payload.items() if key != "plan_sha256"}
        if payload.get("plan_sha256") != _parent_payload_sha256(unsigned):
            raise ValueError(f"{label} parent plan self-hash changed for {case}")
        adapter = payload.get("case_adapter")
        if (
            not isinstance(adapter, dict)
            or adapter.get("case") != case
            or not isinstance(payload.get("entries"), list)
        ):
            raise ValueError(f"{label} parent plan identity changed for {case}")
        result[case] = payload
    return result


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(plan.canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def build_plan_artifacts(
    *,
    multicase_root: Path = MULTICASE_ROOT,
    parent_package_root: Path = PARENT_PACKAGE_ROOT,
    v1_plan_root: Path | None = None,
    v2_plan_root: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    expected_parent_digest: str = PARENT_PACKAGE_DIGEST,
    verify_parent: bool = True,
) -> dict:
    """Build all four canonical files, refusing any existing v3 destination."""

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"v3 plan output already exists: {output_root}")
    parent_experiment_root = parent_package_root / "experiment"
    if verify_parent:
        parent_verification = verify_parent_package(
            parent_package_root, expected_digest=expected_parent_digest
        )
    else:
        if not plan.valid_sha256(expected_parent_digest):
            raise ValueError("test parent package digest is invalid")
        parent_verification = {
            "canonical_digest": expected_parent_digest,
            "inventory_file_count": None,
            "inventory_bytes": None,
        }
    v1_root = (
        v1_plan_root
        if v1_plan_root is not None
        else parent_experiment_root / "prelock_bundle/corpus/plans/full"
    )
    v2_root = (
        v2_plan_root
        if v2_plan_root is not None
        else parent_experiment_root / "prelock_bundle/recovery/v2/plans/full"
    )
    v1_plans = _load_plan_grid(v1_root, "v1")
    v2_plans = _load_plan_grid(v2_root, "v2")

    # Selection is completed before prior files are scanned; no response value can rank it.
    v3_plans = {
        case: plan.build_case_plan(v2_plans[case]) for case in sorted(boptest.CASES)
    }
    prior_evidence = scan_prior_evidence(
        multicase_root,
        parent_experiment_root,
        parent_package_digest=expected_parent_digest,
    )
    certificate = plan.build_disjointness_certificate(
        v1_plans,
        v2_plans,
        v3_plans,
        prior_evidence=prior_evidence,
    )
    worker_collect.validate_certificate_grid(
        certificate,
        v3_plans,
        str(certificate["certificate_sha256"]),
    )

    output_root.mkdir(parents=False, exist_ok=False)
    plan_root = output_root / "plans/full"
    plan_root.mkdir(parents=True, exist_ok=False)
    for case in sorted(boptest.CASES):
        _write_exclusive(plan_root / f"{case}.json", v3_plans[case])
    certificate_path = output_root / "disjointness_certificate.json"
    _write_exclusive(certificate_path, certificate)
    return {
        "output_root": str(output_root),
        "plan_sha256_by_case": {
            case: v3_plans[case]["plan_sha256"] for case in sorted(boptest.CASES)
        },
        "certificate_sha256": certificate["certificate_sha256"],
        "certificate_file_sha256": plan.sha256_file(certificate_path),
        "prior_evidence_counts": prior_evidence["counts"],
        "prior_evidence_inventory_sha256": prior_evidence["inventory_sha256"],
        "parent_verification": parent_verification,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            build_plan_artifacts(output_root=args.output_root),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
