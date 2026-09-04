"""Build the immutable reviewer RC comparison report package."""

from __future__ import annotations

import os
from pathlib import Path

from .report_verify import canonical_json_bytes, canonical_sha256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_training_v1"
REPRODUCTION_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_selection_reproduction_v1"
READINESS_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_readiness_v1"
EVALUATION_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_evaluation_v1"
AUDIT_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_audit_v1"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_rc_report_v2"
RESULT_PAYLOAD_SHA256 = (
    "d0e8c8237c078d03e9cdcc7afbcc125d3292f7d5e1a7e639a85c149aed7dac29"
)
EXPECTED_FILES = {
    "comparison_result.json": "e03cf9dd0b9dcb25999c7a884420c42e25a8718031e08be00c49c88619a78e09",
    "selected_hyperparameters.csv": "34fcf076db74d44de2924976bae3609eea604f4fc38b94bf8bda3db116723135",
    "selected_hyperparameters_evaluation_snapshot.csv": "d64a3b9795624c2c3a607b9559f1bdae5a99d5f0997823d47c35e28469e6ca92",
    "evaluation_complete.json": "6b04461a8bdfed9063572c646d6e5698c268788d0b009c94ecd5a3867518d3c6",
    "evaluation_provenance.json": "9b68451b1b05092be60c857f3b93c31d4b6dd6935ac7fb4af236307d4ed00b43",
    "audit_receipt.json": "758abbc26b9a695885436f1b4238247c7b4e8b71fdf2338e2f0b9a2df7863148",
    "reproduction_receipt.json": "21b487c747b84d8522ec7cf6f848f22efa24dc2d1bc00a4fb3ceca28275c0fa4",
    "training_complete.json": "3dff5067d955e11ea5de0ea7083187471e91947bd3dc683cb6024110f3835632",
    "training_source_lock.json": "e37960eb8c0280e9b93864de450f1f9a432f5e61686d76438bcf69dc94ca0308",
    "readiness.json": "cde102f6399b42d899a2ff9cd1b798350c5509154eb2264183c7fb26ce824ec6",
    "models/bestest_hydronic_heat_pump.json": "00d1cef5b8a7e77c9678f2ccc48dea50b3101a5bb6bfa42738ed933e2340a444",
    "models/multizone_office_simple_air.json": "fdd16a7391a7f36ede2c45b4a1cdcb1fcdbdaa7a69a80d9b6706468d6a400fa1",
    "models/twozone_apartment_hydronic.json": "a629c6e293f4a239b8a50cd9c89ad1a50a4a06667aa1d936da98dffe87eebf49",
}


def _write_once(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _strict(path: Path) -> dict:
    import json

    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("source JSON root changed")
    return payload


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def build_report(output_root: Path = OUTPUT_ROOT) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite RC report: {output_root}")
    sources = {
        "comparison_result.json": EVALUATION_ROOT / "comparison_result.json",
        "selected_hyperparameters.csv": TRAINING_ROOT
        / "selected_hyperparameters.csv",
        "selected_hyperparameters_evaluation_snapshot.csv": EVALUATION_ROOT
        / "selected_hyperparameters.csv",
        "evaluation_complete.json": EVALUATION_ROOT / "evaluation_complete.json",
        "evaluation_provenance.json": EVALUATION_ROOT
        / "evaluation_provenance.json",
        "audit_receipt.json": AUDIT_ROOT / "audit_receipt.json",
        "reproduction_receipt.json": REPRODUCTION_ROOT / "reproduction_receipt.json",
        "training_complete.json": TRAINING_ROOT / "training_complete.json",
        "training_source_lock.json": TRAINING_ROOT / "training_source_lock.json",
        "readiness.json": READINESS_ROOT / "readiness.json",
        "models/bestest_hydronic_heat_pump.json": TRAINING_ROOT
        / "bestest_hydronic_heat_pump/model.json",
        "models/multizone_office_simple_air.json": TRAINING_ROOT
        / "multizone_office_simple_air/model.json",
        "models/twozone_apartment_hydronic.json": TRAINING_ROOT
        / "twozone_apartment_hydronic/model.json",
    }
    for name, path in sources.items():
        if sha256_file(path) != EXPECTED_FILES[name]:
            raise ValueError(f"RC report source changed: {name}")
    result = _strict(sources["comparison_result.json"])
    if canonical_sha256(result) != RESULT_PAYLOAD_SHA256:
        raise ValueError("RC comparison result payload changed")
    audit = _strict(sources["audit_receipt.json"])
    reproduction = _strict(sources["reproduction_receipt.json"])
    if (
        audit.get("complete") is not True
        or audit.get("recomputed_result_payload_sha256") != RESULT_PAYLOAD_SHA256
        or reproduction.get("byte_identical_numerical_selection") is not True
    ):
        raise ValueError("RC audit or reproduction binding changed")
    output_root.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        _write_once(output_root / name, source.read_bytes())
    protocol = (
        "# Reviewer-motivated resistance-capacitance comparison\n\n"
        "This package reports a descriptive, post-outcome physical-comparator "
        "analysis requested during supervisory review. Case-specific 1R1C and "
        "2R2C thermal networks were fitted from clean fitting trajectories. "
        "Topology, equipment-map regularization, and observer clipping were "
        "selected only on development-validation eight-step error. A small "
        "empirical equipment map predicts power, flow, and supply conditions; "
        "the zone-temperature transition is constrained by the RC heat balance. "
        "The immutable publication transport corpus was then evaluated once "
        "under the paired two-hour/four-hour dwell and silent-fault contract. "
        "The result assigns no confirmatory category and makes no control, "
        "energy, comfort, safety, or occupied-building claim.\n"
    )
    _write_once(output_root / "REPORT_PROTOCOL.md", protocol.encode("ascii"))
    verifier_source = Path(__file__).with_name("report_verify.py")
    _write_once(output_root / "verify_report.py", verifier_source.read_bytes())
    summary = {
        "schema": "reviewer-rc-report-summary-v1",
        "complete": True,
        "descriptive_only": True,
        "confirmatory_category_assigned": False,
        "comparison_result": result,
        "selected_model_count": 3,
        "candidate_count": 180,
        "selected_topology_by_case": {
            "bestest_hydronic_heat_pump": "2r2c",
            "multizone_office_simple_air": "2r2c",
            "twozone_apartment_hydronic": "2r2c",
        },
        "claim_scope": (
            "three fixed BOPTEST simulators, synthetic silent sensor faults, "
            "paired two-hour and four-hour action dwell, open-loop H8 prediction"
        ),
    }
    _write_once(output_root / "report_summary.json", canonical_json_bytes(summary))
    inventory = _inventory(output_root)
    manifest = {
        "schema": "reviewer-rc-report-package-v1",
        "complete": True,
        "descriptive_only": True,
        "confirmatory_category_assigned": False,
        "comparison_result_payload_sha256": RESULT_PAYLOAD_SHA256,
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
