"""Prepare the metadata-only public-freeze input set for comparison analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

from building_fault_wm.deterministic_transport import (
    gate as v3_gate,
    run_evaluation as v3_run,
)
from building_fault_wm.ridge_arx import (
    evaluate as arx_evaluate,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)

from . import analysis


HERE = Path(__file__).resolve().parent
REGISTRY_NAME = "comparison_analysis_prelock.json"
DIGEST_NAME = "comparison_analysis_prelock.canonical.sha256"
BUNDLE_NAME = "bundle"
SCHEMA = "schedule-matched-arx-neural-comparison-prelock-v1"
REQUIRED_SOURCE_FILES = {
    "__init__.py",
    "analysis.py",
    "freeze.py",
    "public_freeze.py",
    "PROTOCOL.md",
    "test_analysis.py",
    "test_freeze.py",
}


def _source_paths() -> dict[str, Path]:
    paths = {
        path.name: path
        for path in HERE.iterdir()
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    if set(paths) != REQUIRED_SOURCE_FILES:
        raise ValueError("comparison analysis source inventory changed")
    return paths


def prospective_contract() -> dict:
    """Bind schemas and code only; no evaluation result value is accessed."""

    sources = {
        name: sha256_file(path) for name, path in sorted(_source_paths().items())
    }
    upstream = {
        "v3": {
            "output_schema": v3_run.OUTPUT_SCHEMA,
            "completion_schema": v3_run.COMPLETION_SCHEMA,
            "core_name": v3_run.CORE_NAME,
            "required_columns": list(v3_gate.REQUIRED_COLUMNS),
            "arms": list(v3_gate.ARMS),
            "run_evaluation_source_sha256": sha256_file(
                Path(v3_run.__file__).resolve()
            ),
            "gate_source_sha256": sha256_file(Path(v3_gate.__file__).resolve()),
        },
        "arx": {
            "output_schema": arx_evaluate.OUTPUT_SCHEMA,
            "completion_schema": arx_evaluate.COMPLETION_SCHEMA,
            "core_name": "arx_core.csv",
            "required_columns": list(arx_evaluate.CORE_COLUMNS),
            "arm": arx_evaluate.ARM,
            "evaluate_source_sha256": sha256_file(
                Path(arx_evaluate.__file__).resolve()
            ),
        },
    }
    return {
        "schema": "schedule-matched-arx-neural-comparison-contract-v1",
        "outcome_values_accessed": False,
        "source_file_sha256_by_name": sources,
        "upstream_evaluator_contracts": upstream,
        "pair_columns": list(analysis.PAIR_COLUMNS),
        "all_arms": list(analysis.ALL_ARMS),
        "primary_neural_arm": analysis.PRIMARY_NEURAL_ARM,
        "arx_arm": analysis.ARX_ARM,
        "primary_policy": analysis.PRIMARY_POLICY,
        "control_policy": analysis.CONTROL_POLICY,
        "primary_horizon": analysis.HORIZON,
        "horizons": list(analysis.HORIZONS),
        "silent_families": list(v3_gate.SILENT_FAMILIES),
        "bootstrap": {
            "draws": analysis.BOOTSTRAP_DRAWS,
            "seed": analysis.BOOTSTRAP_SEED,
            "generator": "numpy.random.PCG64",
            "hierarchy": "case/model_seed/window_within_case",
            "paired_across_arms_and_policies": True,
        },
        "margins": {
            "dominance_point_threshold": analysis.DOMINANCE_MARGIN,
            "equivalence_tost_margin": analysis.EQUIVALENCE_MARGIN,
            "case_family_equivalence_limit": (
                analysis.STRATUM_EQUIVALENCE_LIMIT
            ),
        },
        "claim_scope": {
            "inferential": (
                "H8 deterministic_wm versus ARX on silent faults under new_4h; "
                "old_2h is the persistence control"
            ),
            "descriptive_only": (
                "H1/H2/H4 and legacy/ungated_h8 RSSM comparisons"
            ),
        },
    }


def prepare_local_freeze_bundle(output_root: Path) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite comparison freeze bundle: {output_root}"
        )
    source_paths = _source_paths()
    contract = prospective_contract()
    bundle = output_root / BUNDLE_NAME
    bundle.mkdir(parents=True, exist_ok=False)
    for name, source in sorted(source_paths.items()):
        write_once(bundle / "source" / name, source.read_bytes())
    contract_path = write_json_once(bundle / "analysis_contract.json", contract)
    inventory = tree_inventory(bundle)
    registry = {
        "schema": SCHEMA,
        "outcome_values_accessed_while_preparing": False,
        "analysis_contract_file_sha256": sha256_file(contract_path),
        "analysis_contract_payload_sha256": canonical_sha256(contract),
        "bundle_inventory": inventory,
        "bundle_inventory_sha256": canonical_sha256(inventory),
    }
    write_json_once(output_root / REGISTRY_NAME, registry)
    write_once(
        output_root / DIGEST_NAME,
        f"{canonical_sha256(registry)}\n".encode("ascii"),
    )
    verify_local_freeze_bundle(output_root)
    return output_root


def verify_local_freeze_bundle(output_root: Path) -> dict:
    registry = strict_json(output_root / REGISTRY_NAME)
    if (
        registry.get("schema") != SCHEMA
        or registry.get("outcome_values_accessed_while_preparing") is not False
        or (output_root / DIGEST_NAME).read_text(encoding="ascii")
        != f"{canonical_sha256(registry)}\n"
    ):
        raise ValueError("comparison analysis prelock identity changed")
    bundle = output_root / BUNDLE_NAME
    inventory = tree_inventory(bundle)
    if (
        registry.get("bundle_inventory") != inventory
        or registry.get("bundle_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise ValueError("comparison analysis bundle inventory changed")
    contract = strict_json(bundle / "analysis_contract.json")
    if (
        registry.get("analysis_contract_file_sha256")
        != sha256_file(bundle / "analysis_contract.json")
        or registry.get("analysis_contract_payload_sha256")
        != canonical_sha256(contract)
        or contract != prospective_contract()
    ):
        raise ValueError("comparison analysis prospective contract changed")
    for name, source in sorted(_source_paths().items()):
        snapshot = bundle / "source" / name
        if sha256_file(snapshot) != sha256_file(source):
            raise ValueError(f"comparison analysis source snapshot changed: {name}")
    return registry


def public_freeze_input_paths(output_root: Path) -> dict[str, Path]:
    """Return the exact metadata/source files for a later public timestamp."""

    verify_local_freeze_bundle(output_root)
    bundle = output_root / BUNDLE_NAME
    paths = {
        REGISTRY_NAME: output_root / REGISTRY_NAME,
        DIGEST_NAME: output_root / DIGEST_NAME,
        "analysis_contract.json": bundle / "analysis_contract.json",
    }
    paths.update(
        {
            f"source__{name}": bundle / "source" / name
            for name in sorted(REQUIRED_SOURCE_FILES)
        }
    )
    if any(path.suffix == ".csv" for path in paths.values()):
        raise AssertionError("comparison public-freeze set contains result data")
    return paths
