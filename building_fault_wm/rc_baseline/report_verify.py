"""Standard-library verifier for the sealed RC comparison report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "report_manifest.json"
DIGEST_NAME = "report_manifest.canonical.sha256"
MANIFEST_SCHEMA = "reviewer-rc-report-package-v1"
SUMMARY_SCHEMA = "reviewer-rc-report-summary-v1"
RESULT_SCHEMA = "reviewer-rc-comparison-result-v1"
EXPECTED_RESULT_PAYLOAD_SHA256 = (
    "d0e8c8237c078d03e9cdcc7afbcc125d3292f7d5e1a7e639a85c149aed7dac29"
)


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a plain file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict:
    payload = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("report inventory path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("report inventory path is unsafe")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _read_digest(path: Path) -> str:
    value = path.read_text(encoding="ascii")
    if not re.fullmatch(r"[0-9a-f]{64}\n", value):
        raise ValueError("report digest record is malformed")
    return value.strip()


def verify_report(
    root: Path, expected_digest: str | None = None, require_read_only: bool = True
) -> dict:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("report root is invalid")
    if require_read_only:
        for path in (root, *sorted(root.rglob("*"))):
            if path.is_symlink():
                raise ValueError("report contains a symbolic link")
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise ValueError(f"report path is writable: {path}")
    manifest = strict_json(root / MANIFEST_NAME)
    digest = _read_digest(root / DIGEST_NAME)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("descriptive_only") is not True
        or digest != canonical_sha256(manifest)
        or (expected_digest is not None and digest != expected_digest)
    ):
        raise ValueError("RC report manifest identity changed")
    inventory = manifest.get("artifact_inventory_excludes_manifest_and_digest")
    if (
        not isinstance(inventory, list)
        or canonical_sha256(inventory) != manifest.get("artifact_inventory_sha256")
    ):
        raise ValueError("RC report inventory digest changed")
    expected_paths = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("RC report inventory row is invalid")
        path = _safe_path(root, item["path"])
        if item["path"] in expected_paths:
            raise ValueError("RC report inventory path is duplicated")
        expected_paths.add(item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"RC report artifact changed: {item['path']}")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } - {MANIFEST_NAME, DIGEST_NAME}
    if observed != expected_paths:
        raise ValueError("RC report file set changed")
    summary = strict_json(root / "report_summary.json")
    result = strict_json(root / "comparison_result.json")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("complete") is not True
        or summary.get("descriptive_only") is not True
        or summary.get("comparison_result") != result
        or summary.get("candidate_count") != 180
        or canonical_sha256(summary) != manifest.get("summary_payload_sha256")
        or result.get("schema") != RESULT_SCHEMA
        or canonical_sha256(result) != EXPECTED_RESULT_PAYLOAD_SHA256
    ):
        raise ValueError("RC report result summary changed")
    with (root / "selected_hyperparameters.csv").open(
        newline="", encoding="ascii"
    ) as stream:
        selected = list(csv.DictReader(stream))
    if (
        len(selected) != 3
        or {row["selected_topology"] for row in selected} != {"2r2c"}
        or {row["case"] for row in selected}
        != {
            "bestest_hydronic_heat_pump",
            "multizone_office_simple_air",
            "twozone_apartment_hydronic",
        }
    ):
        raise ValueError("RC report selected-model table changed")
    audit = strict_json(root / "audit_receipt.json")
    reproduction = strict_json(root / "reproduction_receipt.json")
    completion = strict_json(root / "evaluation_complete.json")
    if sha256_file(
        root / "selected_hyperparameters_evaluation_snapshot.csv"
    ) != completion.get("file_sha256_by_name", {}).get(
        "selected_hyperparameters.csv"
    ):
        raise ValueError("RC report evaluation-snapshot binding changed")
    if (
        audit.get("complete") is not True
        or audit.get("recomputed_result_payload_sha256")
        != EXPECTED_RESULT_PAYLOAD_SHA256
        or reproduction.get("byte_identical_numerical_selection") is not True
        or completion.get("complete") is not True
        or completion.get("single_completed_heldout_pass") is not True
        or completion.get("row_count") != 207360
    ):
        raise ValueError("RC report audit binding changed")
    return {
        "schema": MANIFEST_SCHEMA,
        "report_digest": digest,
        "comparison_result_payload_sha256": EXPECTED_RESULT_PAYLOAD_SHA256,
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-digest")
    parser.add_argument("--allow-writable", action="store_true")
    args = parser.parse_args()
    result = verify_report(
        args.report,
        expected_digest=args.expected_digest,
        require_read_only=not args.allow_writable,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
