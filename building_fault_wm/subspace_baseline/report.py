"""Build the immutable reviewer subspace comparison report package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .report_verify import canonical_json_bytes, canonical_sha256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = PROJECT_ROOT / "artifacts/reviewer_subspace_evaluation_v3"
AUDIT_ROOT = PROJECT_ROOT / "artifacts/reviewer_subspace_audit_v5"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_subspace_report_v1"
EXPECTED_FILES = {
    "comparison_result.json": "6147a08803b87aefc4e85036293347832437875a549abc510e785da452f7830e",
    "selected_hyperparameters.csv": "f1655487c79cbd7e1db2b84c824c513b1b612b553d5f89c169e3044d3bb62a52",
    "evaluation_complete.json": "7a47e852aac61b6535c620497dcf91723d39aafb14050707e47c1605874ffcb0",
    "evaluation_provenance.json": "a6734a37e2567b01f5ad55d9fb9bd3d0c8c63f765b5183a0051dd53416e72880",
    "audit_receipt.json": "db9ce52bbd5045f334d7aebe96ebe9473a9504947cfb0d68e3edd3f5ddd14756",
}
RESULT_PAYLOAD_SHA256 = (
    "2f47710e7337e5b883e3c1f3b4841c1921f3fbddbf052d217dedf47a34cc921d"
)


def _write_once(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _strict(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("source JSON root changed")
    return payload


def _inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def build_report(output_root: Path = OUTPUT_ROOT) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite report: {output_root}")
    sources = {
        "comparison_result.json": EVALUATION_ROOT / "comparison_result.json",
        "selected_hyperparameters.csv": EVALUATION_ROOT
        / "selected_hyperparameters.csv",
        "evaluation_complete.json": EVALUATION_ROOT / "evaluation_complete.json",
        "evaluation_provenance.json": EVALUATION_ROOT
        / "evaluation_provenance.json",
        "audit_receipt.json": AUDIT_ROOT / "audit_receipt.json",
    }
    for name, path in sources.items():
        if sha256_file(path) != EXPECTED_FILES[name]:
            raise ValueError(f"source artifact changed: {name}")
    result = _strict(sources["comparison_result.json"])
    if canonical_sha256(result) != RESULT_PAYLOAD_SHA256:
        raise ValueError("comparison result payload changed")
    audit = _strict(sources["audit_receipt.json"])
    if (
        audit.get("complete") is not True
        or audit.get("recomputed_result_payload_sha256") != RESULT_PAYLOAD_SHA256
    ):
        raise ValueError("independent audit binding changed")
    output_root.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        _write_once(output_root / name, source.read_bytes())
    protocol = (
        "# Reviewer-motivated subspace comparison\n\n"
        "This package reports a descriptive, post-outcome comparison requested "
        "during supervisory review. Case-specific linear time-invariant state-"
        "space models were identified from fitting trajectories by a multi-"
        "experiment subspace procedure. Block rows, state order, and Kalman-"
        "innovation clipping were selected on development-validation H8 error. "
        "The immutable v6 publication corpus was then evaluated once under the "
        "existing paired two-hour/four-hour dwell, silent-fault, and equal-weight "
        "analysis. This comparator is data-driven, not a resistance-capacitance "
        "or physics-informed model, and assigns no confirmatory category.\n"
    )
    _write_once(output_root / "REPORT_PROTOCOL.md", protocol.encode("ascii"))
    verifier_source = Path(__file__).with_name("report_verify.py")
    _write_once(output_root / "verify_report.py", verifier_source.read_bytes())
    summary = {
        "schema": "reviewer-subspace-report-summary-v1",
        "complete": True,
        "descriptive_only": True,
        "confirmatory_category_assigned": False,
        "comparison_result": result,
        "selected_model_count": 3,
        "candidate_count": 162,
        "stable_candidate_count": 153,
        "claim_scope": (
            "three fixed BOPTEST simulators, synthetic silent sensor faults, "
            "paired two-hour and four-hour action dwell, open-loop H8 prediction"
        ),
    }
    _write_once(output_root / "report_summary.json", canonical_json_bytes(summary))
    bindings = {
        "audit_receipt_file_sha256": sha256_file(output_root / "audit_receipt.json"),
        "evaluation_completion_file_sha256": sha256_file(
            output_root / "evaluation_complete.json"
        ),
        "evaluation_provenance_file_sha256": sha256_file(
            output_root / "evaluation_provenance.json"
        ),
        "comparison_result_file_sha256": sha256_file(
            output_root / "comparison_result.json"
        ),
        "selected_hyperparameters_file_sha256": sha256_file(
            output_root / "selected_hyperparameters.csv"
        ),
        "comparison_result_payload_sha256": RESULT_PAYLOAD_SHA256,
    }
    inventory = _inventory(output_root)
    manifest = {
        "schema": "reviewer-subspace-report-package-v1",
        "complete": True,
        "descriptive_only": True,
        "confirmatory_category_assigned": False,
        "bindings": bindings,
        "summary_payload_sha256": canonical_sha256(summary),
        "artifact_inventory_excludes_manifest_and_digest": inventory,
        "artifact_inventory_sha256": canonical_sha256(inventory),
    }
    manifest_path = _write_once(
        output_root / "report_manifest.json", canonical_json_bytes(manifest)
    )
    _write_once(
        output_root / "report_manifest.canonical.sha256",
        (canonical_sha256(manifest) + "\n").encode("ascii"),
    )
    for directory in sorted(
        (path for path in output_root.rglob("*") if path.is_dir()), reverse=True
    ):
        directory.chmod(0o555)
    output_root.chmod(0o555)
    return manifest_path


if __name__ == "__main__":
    print(build_report())
