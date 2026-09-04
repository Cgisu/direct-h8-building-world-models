"""Independent rerun comparison for RC development selection."""

from __future__ import annotations

import os
from pathlib import Path

from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    write_json_once,
)

from .config import CASES
from .study import DEFAULT_TRAINING_ROOT, verify_training


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/reviewer_rc_selection_reproduction_v1"
)


def compare_selections(
    reference_root: Path,
    production_root: Path = DEFAULT_TRAINING_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite RC reproduction audit: {output_root}")
    reference = verify_training(reference_root)
    production = verify_training(production_root)
    relative_files = [
        "training_source_lock.json",
        "selection_scores.csv",
        "selected_hyperparameters.csv",
        *(f"{case}/model.json" for case in CASES),
    ]
    identical = {}
    for relative in relative_files:
        reference_sha = sha256_file(reference_root / relative)
        production_sha = sha256_file(production_root / relative)
        if reference_sha != production_sha:
            raise ValueError(f"RC selection rerun differs: {relative}")
        identical[relative] = reference_sha
    receipt = {
        "schema": "reviewer-rc-selection-reproduction-v1",
        "complete": True,
        "heldout_values_accessed": False,
        "byte_identical_numerical_selection": True,
        "reference_training_completion_payload_sha256": canonical_sha256(reference),
        "production_training_completion_payload_sha256": canonical_sha256(production),
        "identical_file_sha256_by_relative_path": identical,
        "excluded_from_byte_identity": [
            "case selection receipts, because they contain wall-clock durations",
            "training completion records, because they bind those duration-bearing receipts",
        ],
    }
    return write_json_once(output_root / "reproduction_receipt.json", receipt)
