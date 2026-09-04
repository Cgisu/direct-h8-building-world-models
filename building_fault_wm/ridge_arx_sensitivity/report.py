"""Build a compact sealed report for the strengthened ARX sensitivity analysis."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)

from .audit import DEFAULT_AUDIT_ROOT
from .config import CASES, MODEL_SEEDS, POLICIES
from .study import (
    DEFAULT_EVALUATION_ROOT,
    DEFAULT_READINESS_ROOT,
    DEFAULT_TRAINING_ROOT,
    READINESS_NAME,
    TRAINING_COMPLETE_NAME,
    source_manifest,
    verify_evaluation,
    verify_readiness,
    verify_training_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT
    / "artifacts/post_outcome_strong_arx_sensitivity_report_v3"
)
REPORT_SCHEMA = "post-outcome-strong-arx-sealed-sensitivity-report-v3"
MANIFEST_NAME = "report_manifest.json"
DIGEST_NAME = "report_manifest.canonical.sha256"


def _copy_once(source: Path, destination: Path) -> Path:
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite report file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _frame(path: Path, frame: pd.DataFrame) -> Path:
    return write_once(
        path,
        frame.to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        ).encode("ascii"),
    )


def _verify_audit(root: Path) -> dict:
    complete = strict_json(root / "audit_complete.json")
    hashes = complete.get("file_sha256_by_name")
    if (
        complete.get("schema")
        != "post-outcome-strong-arx-audit-completion-v1"
        or complete.get("complete") is not True
        or not isinstance(hashes, dict)
    ):
        raise ValueError("standalone audit is incomplete")
    for name, digest in hashes.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"standalone audit changed: {name}")
    receipt = strict_json(root / "audit_receipt.json")
    if (
        receipt.get("scope") != "post_outcome_robustness"
        or receipt.get("exact_within_tolerance") is not True
    ):
        raise ValueError("standalone audit did not pass")
    return receipt


def build_report(
    *,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    readiness_root: Path = DEFAULT_READINESS_ROOT,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_REPORT_ROOT,
) -> Path:
    training = verify_training_grid(training_root)
    readiness = verify_readiness(
        training_root=training_root, readiness_root=readiness_root
    )
    completion = verify_evaluation(evaluation_root)
    audit = _verify_audit(audit_root)
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to overwrite report: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    selection_frames = []
    for case in CASES:
        for seed in MODEL_SEEDS:
            frame = pd.read_csv(
                training_root
                / case
                / f"seed{seed}"
                / "selection_scores.csv",
                float_precision="round_trip",
            )
            frame.insert(0, "model_seed", seed)
            frame.insert(0, "case", case)
            selection_frames.append(frame)
    selection_grid = pd.concat(selection_frames, ignore_index=True)
    if len(selection_grid) != 750:
        raise ValueError("report selection grid is incomplete")
    _frame(output_root / "selection_grid.csv", selection_grid)
    _copy_once(
        evaluation_root / "selected_hyperparameters.csv",
        output_root / "selected_hyperparameters.csv",
    )
    _copy_once(
        evaluation_root / "strong_arx_core.csv",
        output_root / "strong_arx_heldout_core.csv",
    )
    _copy_once(
        evaluation_root / "descriptive_by_horizon.csv",
        output_root / "descriptive_by_horizon.csv",
    )
    _copy_once(
        evaluation_root / "sensitivity_result.json",
        output_root / "sensitivity_result.json",
    )
    _copy_once(
        audit_root / "recomputed_headlines.json",
        output_root / "standalone_recomputed_headlines.json",
    )
    _copy_once(
        audit_root / "audit_receipt.json",
        output_root / "standalone_audit_receipt.json",
    )

    result = strict_json(evaluation_root / "sensitivity_result.json")
    aggregate_rows = []
    stratum_rows = []
    for policy in POLICIES:
        value = result["policy_results"][policy]
        aggregate_rows.append(
            {
                "policy": policy,
                "deterministic_wm_mae": value[
                    "mean_standardized_mae_by_arm"
                ]["deterministic_wm"],
                "strengthened_arx_mae": value[
                    "mean_standardized_mae_by_arm"
                ][
                    "post_outcome_strengthened_recursive_ridge_arx"
                ],
                "effect": value["point"],
                "ci95_lower": value["ci95_lower"],
                "ci95_upper": value["ci95_upper"],
                "ci90_lower": value["ci90_lower"],
                "ci90_upper": value["ci90_upper"],
            }
        )
        for kind in ("by_case", "by_family", "by_seed"):
            for identity, effect in value[kind].items():
                stratum_rows.append(
                    {
                        "policy": policy,
                        "stratum_kind": kind.removeprefix("by_"),
                        "stratum": identity,
                        "effect": effect,
                    }
                )
    _frame(
        output_root / "aggregate_2h_4h_comparison.csv",
        pd.DataFrame(aggregate_rows),
    )
    _frame(
        output_root / "case_family_seed_strata.csv",
        pd.DataFrame(stratum_rows),
    )
    provenance = strict_json(
        evaluation_root / "evaluation_provenance.json"
    )
    write_json_once(
        output_root / "runtime.json",
        {
            "scope": "post_outcome_robustness",
            "evaluation_runtime": provenance["runtime"],
            "selection_wall_seconds_total": sum(
                float(
                    strict_json(
                        training_root
                        / case
                        / f"seed{seed}"
                        / "selection_receipt.json"
                    )["wall_seconds"]
                )
                for case in CASES
                for seed in MODEL_SEEDS
            ),
        },
    )
    write_json_once(
        output_root / "exact_bindings.json",
        {
            "scope": "post_outcome_robustness",
            "source_manifest": source_manifest(),
            "readiness_file_sha256": sha256_file(
                readiness_root / READINESS_NAME
            ),
            "readiness_payload_sha256": canonical_sha256(readiness),
            "selection_grid_file_sha256": sha256_file(
                training_root / TRAINING_COMPLETE_NAME
            ),
            "selection_grid_payload_sha256": canonical_sha256(training),
            "evaluation_completion_file_sha256": sha256_file(
                evaluation_root / "evaluation_complete.json"
            ),
            "evaluation_completion_payload_sha256": canonical_sha256(
                completion
            ),
            "audit_receipt_payload_sha256": canonical_sha256(audit),
            "input_identity": readiness["input_identity"],
        },
    )
    boundary = {
        "scope": "post_outcome_robustness",
        "original_alpha_max_selected_count": 14,
        "original_fit_count": 15,
        "strengthened_history_max_selected_count": training[
            "selected_history_at_grid_max_count"
        ],
        "strengthened_alpha_max_selected_count": training[
            "selected_alpha_at_grid_max_count"
        ],
        "history_grid_max": 40,
        "alpha_grid_max": 100_000.0,
    }
    write_json_once(output_root / "boundary_flags.json", boundary)
    markdown = (
        "# Strengthened Ridge-ARX Sensitivity Report\n\n"
        "Scope: `post_outcome_robustness`. This analysis was motivated by the "
        "original alpha-grid boundary and is descriptive, not confirmatory.\n\n"
        "The directory contains the complete 750-candidate development selection "
        "grid, selected hyperparameters and boundary flags, exact held-out "
        "per-row Ridge-ARX core, paired 2 h/4 h bootstrap summaries, "
        "case/family/seed effects, runtime, exact code/input bindings, and a "
        "standalone numerical recomputation.\n"
    )
    write_once(output_root / "RESULTS.md", markdown.encode("ascii"))

    inventory = tree_inventory(output_root)
    manifest = {
        "schema": REPORT_SCHEMA,
        "scope": "post_outcome_robustness",
        "post_outcome": True,
        "confirmatory_category_assigned": False,
        "complete": True,
        "inventory_excludes_manifest_and_digest": inventory,
        "inventory_sha256": canonical_sha256(inventory),
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
    }
    manifest_path = write_json_once(output_root / MANIFEST_NAME, manifest)
    write_once(
        output_root / DIGEST_NAME,
        f"{canonical_sha256(manifest)}\n".encode("ascii"),
    )
    for path in output_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    output_root.chmod(0o555)
    return manifest_path


def verify_report(output_root: Path = DEFAULT_REPORT_ROOT) -> dict:
    manifest = strict_json(output_root / MANIFEST_NAME)
    if (
        manifest.get("schema") != REPORT_SCHEMA
        or manifest.get("scope") != "post_outcome_robustness"
        or manifest.get("post_outcome") is not True
        or manifest.get("confirmatory_category_assigned") is not False
        or manifest.get("complete") is not True
        or (output_root / DIGEST_NAME).read_text(encoding="ascii")
        != f"{canonical_sha256(manifest)}\n"
    ):
        raise ValueError("sealed sensitivity report identity changed")
    inventory = tree_inventory(
        output_root, exclude={MANIFEST_NAME, DIGEST_NAME}
    )
    if (
        manifest.get("inventory_excludes_manifest_and_digest") != inventory
        or manifest.get("inventory_sha256") != canonical_sha256(inventory)
        or manifest.get("file_count") != len(inventory)
        or manifest.get("total_bytes")
        != sum(int(row["bytes"]) for row in inventory)
    ):
        raise ValueError("sealed sensitivity report inventory changed")
    return manifest
