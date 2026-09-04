"""Metadata-only local prelock for the comparison identity guard."""

from __future__ import annotations

import os
from pathlib import Path

from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
    write_once,
)

from .guard import BindingPaths, build_binding_contract, validate_binding_contract


HERE = Path(__file__).resolve().parent
SCHEMA = "schedule-matched-arx-neural-identity-guard-prelock-v1"
REGISTRY_NAME = "identity_guard_prelock.json"
DIGEST_NAME = "identity_guard_prelock.canonical.sha256"
BUNDLE_NAME = "bundle"
BINDING_NAME = "binding_contract.json"
REQUIRED_SOURCE_FILES = {
    "__init__.py",
    "guard.py",
    "freeze.py",
    "public_freeze.py",
    "PROTOCOL.md",
    "test_guard.py",
    "test_freeze.py",
}


def _source_paths() -> dict[str, Path]:
    result = {
        path.name: path
        for path in HERE.iterdir()
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    if set(result) != REQUIRED_SOURCE_FILES:
        raise ValueError("identity-guard source inventory changed")
    return result


def prepare_local_prelock(
    output_root: Path,
    paths: BindingPaths,
    *,
    live_external_freezes: bool,
) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite identity-guard prelock: {output_root}"
        )
    contract = build_binding_contract(
        paths, live_external_freezes=live_external_freezes
    )
    validate_binding_contract(contract)
    source_paths = _source_paths()
    bundle = output_root / BUNDLE_NAME
    bundle.mkdir(parents=True, exist_ok=False)
    for name, source in sorted(source_paths.items()):
        write_once(bundle / "source" / name, source.read_bytes())
    binding_path = write_json_once(bundle / BINDING_NAME, contract)
    inventory = tree_inventory(bundle)
    registry = {
        "schema": SCHEMA,
        "outcome_values_accessed_while_preparing": False,
        "binding_contract_file_sha256": sha256_file(binding_path),
        "binding_contract_payload_sha256": canonical_sha256(contract),
        "source_file_sha256_by_name": {
            name: sha256_file(path)
            for name, path in sorted(source_paths.items())
        },
        "bundle_inventory": inventory,
        "bundle_inventory_sha256": canonical_sha256(inventory),
    }
    write_json_once(output_root / REGISTRY_NAME, registry)
    write_once(
        output_root / DIGEST_NAME,
        f"{canonical_sha256(registry)}\n".encode("ascii"),
    )
    verify_local_prelock(output_root)
    return output_root


def verify_local_prelock(output_root: Path) -> dict:
    registry = strict_json(output_root / REGISTRY_NAME)
    if (
        set(registry)
        != {
            "schema",
            "outcome_values_accessed_while_preparing",
            "binding_contract_file_sha256",
            "binding_contract_payload_sha256",
            "source_file_sha256_by_name",
            "bundle_inventory",
            "bundle_inventory_sha256",
        }
        or registry.get("schema") != SCHEMA
        or registry.get("outcome_values_accessed_while_preparing") is not False
        or (output_root / DIGEST_NAME).read_text(encoding="ascii")
        != f"{canonical_sha256(registry)}\n"
    ):
        raise ValueError("identity-guard prelock identity changed")
    bundle = output_root / BUNDLE_NAME
    inventory = tree_inventory(bundle)
    if (
        registry.get("bundle_inventory") != inventory
        or registry.get("bundle_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise ValueError("identity-guard prelock inventory changed")
    binding_path = bundle / BINDING_NAME
    contract = strict_json(binding_path)
    validate_binding_contract(contract)
    if (
        registry.get("binding_contract_file_sha256")
        != sha256_file(binding_path)
        or registry.get("binding_contract_payload_sha256")
        != canonical_sha256(contract)
    ):
        raise ValueError("identity-guard binding contract changed")
    sources = _source_paths()
    expected_hashes = {
        name: sha256_file(path) for name, path in sorted(sources.items())
    }
    if registry.get("source_file_sha256_by_name") != expected_hashes:
        raise ValueError("live identity-guard source changed")
    for name, digest in expected_hashes.items():
        if sha256_file(bundle / "source" / name) != digest:
            raise ValueError(f"identity-guard source snapshot changed: {name}")
    return registry


def load_verified_binding_contract(output_root: Path) -> dict:
    verify_local_prelock(output_root)
    return strict_json(output_root / BUNDLE_NAME / BINDING_NAME)


def public_freeze_input_paths(output_root: Path) -> dict[str, Path]:
    verify_local_prelock(output_root)
    bundle = output_root / BUNDLE_NAME
    result = {
        REGISTRY_NAME: output_root / REGISTRY_NAME,
        DIGEST_NAME: output_root / DIGEST_NAME,
        BINDING_NAME: bundle / BINDING_NAME,
    }
    result.update(
        {
            f"source__{name}": bundle / "source" / name
            for name in sorted(REQUIRED_SOURCE_FILES)
        }
    )
    if any(path.suffix == ".csv" for path in result.values()):
        raise AssertionError("identity-guard public freeze contains result data")
    return result
