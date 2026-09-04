"""Train the fixed 15-model deterministic comparator grid from parent bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from building_fault_wm.neural_benchmark.fault_data import (
    build_fault_manifest,
    load_corpus_index,
)
from building_fault_wm.neural_benchmark.protocol import CASES
from building_fault_wm.neural_benchmark.study_train import (
    canonical_payload_sha256,
    prepare_case_training_data,
)

from .config import FROZEN_CONFIG, SHARED_RUNTIME_SOURCE_RELATIVE_PATHS
from .plan import sha256_file
from .train import load_parent_schedule, train_fixed_400


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = PROJECT_ROOT / "artifacts" / "direct_h8_publication_v2"
PARENT_DIGEST = (
    "b758859c6cb99d34930452c36e3fd59b5abd0e7f56b19710fa2b1998b23760b8"
)
PARENT_MANIFEST = PARENT_ROOT / "package_manifest.json"
PARENT_DIGEST_FILE = PARENT_ROOT / "package_manifest.canonical.sha256"
PARENT_PRELOCK = PARENT_ROOT / "experiment" / "prelock_bundle"
DEVELOPMENT_MANIFEST = (
    PARENT_PRELOCK / "corpus" / "manifests" / "development_all_corpus_manifest.json"
)
PARENT_SCHEDULE_ROOT = PARENT_PRELOCK / "frozen" / "schedules"
PARENT_SCALER_ROOT = PARENT_PRELOCK / "frozen" / "fit_scalers"
CANONICAL_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "direct_h8_deterministic_transport_v3_training_bound_v2"
)
GRID_RECEIPT = "training_grid_complete.json"
GRID_SCHEMA = "direct-h8-deterministic-transport-training-grid-v2"
SHARED_SOURCE_SCHEMA = "direct-h8-deterministic-transport-shared-source-v1"
SOURCE_LOCK_NAME = "training_source_lock.json"
SOURCE_LOCK_SCHEMA = "direct-h8-deterministic-transport-training-source-lock-v1"


def _canonical_sha256(payload: object) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def _strict_json(path: Path) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="ascii"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {value}")
        ),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def verify_parent_package() -> dict:
    """Hash every immutable parent file listed by its publication manifest."""

    if (
        PARENT_ROOT.is_symlink()
        or not PARENT_ROOT.is_dir()
        or not PARENT_MANIFEST.is_file()
        or not PARENT_DIGEST_FILE.is_file()
    ):
        raise FileNotFoundError("immutable parent publication package is incomplete")
    manifest = _strict_json(PARENT_MANIFEST)
    if _canonical_sha256(manifest) != PARENT_DIGEST:
        raise ValueError("parent package canonical digest changed")
    if PARENT_DIGEST_FILE.read_text(encoding="ascii").strip() != PARENT_DIGEST:
        raise ValueError("parent package digest record changed")
    inventory = manifest.get("artifact_inventory_excludes_manifest_and_digest")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("parent package inventory is incomplete")
    seen = set()
    total_bytes = 0
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("parent package inventory row is invalid")
        relative = str(item["path"])
        if relative in seen:
            raise ValueError("parent package inventory has duplicate paths")
        seen.add(relative)
        path = (PARENT_ROOT / relative).resolve()
        try:
            path.relative_to(PARENT_ROOT.resolve())
        except ValueError as error:
            raise ValueError("parent inventory path escapes its package") from error
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"parent artifact is not a plain file: {relative}")
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"parent artifact size changed: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"parent artifact SHA-256 changed: {relative}")
        total_bytes += int(item["bytes"])
    return {
        "canonical_digest": PARENT_DIGEST,
        "inventory_file_count": len(inventory),
        "inventory_bytes": total_bytes,
    }


def _load_frozen_scaler_payload(case: str) -> dict:
    path = PARENT_SCALER_ROOT / f"{case}.json"
    payload = _strict_json(path)
    if set(payload) != {
        "observation",
        "action",
        "context",
        "fit_source_sha256",
    }:
        raise ValueError(f"frozen scaler fields changed for {case}")
    return payload


def _validate_scalers(case: str, scalers) -> None:
    actual = asdict(scalers)
    expected = _load_frozen_scaler_payload(case)
    expected["fit_source_sha256"] = [
        tuple(item) for item in expected["fit_source_sha256"]
    ]
    if canonical_payload_sha256(actual) != canonical_payload_sha256(expected):
        raise ValueError(f"recomputed FIT scalers differ from parent bytes for {case}")


def _validate_completed_run(path: Path, model_seed: int) -> dict:
    receipt_path = path / "training_receipt.json"
    if path.is_symlink() or not path.is_dir() or not receipt_path.is_file():
        raise ValueError(f"incomplete existing v3 training run: {path}")
    receipt = _strict_json(receipt_path)
    json_config = json.loads(
        json.dumps(FROZEN_CONFIG.to_dict(), allow_nan=False)
    )
    required = {
        "schema": "boptest-deterministic-transport-training-v1",
        "model_seed": model_seed,
        "updates": FROZEN_CONFIG.updates,
        "checkpoint_updates": list(FROZEN_CONFIG.checkpoint_updates),
        "selected_update": FROZEN_CONFIG.updates,
        "selection_rule": "fixed_final_update_no_validation_selection",
        "config": json_config,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"existing v3 training receipt differs at {key}: {path}")
    hashes = receipt.get("checkpoint_file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        f"update_{update:04d}.pt" for update in FROZEN_CONFIG.checkpoint_updates
    }:
        raise ValueError(f"existing v3 checkpoint grid is incomplete: {path}")
    for name, expected in hashes.items():
        checkpoint = path / "checkpoints" / name
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
            raise ValueError(f"existing v3 checkpoint changed: {checkpoint}")
    return receipt


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "PROTOCOL.md",
        "__init__.py",
        "config.py",
        "model.py",
        "train.py",
        "train_grid.py",
    )
    return {name: sha256_file(root / name) for name in names}


def shared_runtime_source_manifest(
    root: Path | None = None,
) -> dict[str, object]:
    source_root = (
        PROJECT_ROOT / "building_fault_wm" if root is None else root.resolve()
    )
    files = {}
    for relative in SHARED_RUNTIME_SOURCE_RELATIVE_PATHS:
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root.resolve())
        except ValueError as error:
            raise ValueError("shared runtime source path escapes its root") from error
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"shared runtime source is not a plain file: {relative}")
        files[relative] = sha256_file(path)
    payload = {"schema": SHARED_SOURCE_SCHEMA, "files": files}
    return {**payload, "sha256": _canonical_sha256(payload)}


def validate_shared_runtime_source_manifest(
    value: object, root: Path | None = None
) -> None:
    if value != shared_runtime_source_manifest(root):
        raise ValueError("shared runtime source manifest changed")


def _bind_training_source(output_root: Path, payload: dict) -> Path:
    path = output_root / SOURCE_LOCK_NAME
    content = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode(
        "ascii"
    )
    if not os.path.lexists(path) and any(output_root.iterdir()):
        raise ValueError(
            "nonempty training root lacks a source lock; refusing retroactive binding"
        )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError:
        if _strict_json(path) != payload:
            raise ValueError(
                "existing training source lock differs; refusing mixed-source resume"
            )
        return path
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def train_grid(
    output_root: Path,
    *,
    cases: Iterable[str] = tuple(sorted(CASES)),
    seeds: Iterable[int] = FROZEN_CONFIG.paired_model_seeds,
    device: str = "cpu",
) -> Path:
    selected_cases = tuple(cases)
    selected_seeds = tuple(seeds)
    if not selected_cases or any(case not in CASES for case in selected_cases):
        raise ValueError("training cases must be a nonempty public-case subset")
    if not selected_seeds or any(
        seed not in FROZEN_CONFIG.paired_model_seeds for seed in selected_seeds
    ):
        raise ValueError("training seeds must be a nonempty paired-seed subset")
    if device != "cpu":
        raise ValueError("the frozen workstation training device is cpu")
    v3_source_before = _source_manifest()
    parent_before = verify_parent_package()
    shared_source_before = shared_runtime_source_manifest()
    source_lock = {
        "schema": SOURCE_LOCK_SCHEMA,
        "parent_package": parent_before,
        "v3_training_source_manifest": v3_source_before,
        "shared_runtime_source_manifest": shared_source_before,
        "config": json.loads(
            json.dumps(FROZEN_CONFIG.to_dict(), allow_nan=False)
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("v3 training output is not a plain directory")
    _bind_training_source(output_root, source_lock)
    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    if index.collection_kind != "development" or set(index.allowed_roles) != {
        "fit",
        "validation",
    }:
        raise ValueError("parent development corpus identity changed")
    manifest = build_fault_manifest(index)
    runs = []
    for case in selected_cases:
        variants, scalers = prepare_case_training_data(index, manifest, case)
        _validate_scalers(case, scalers)
        for model_seed in selected_seeds:
            run_dir = output_root / case / f"seed{model_seed}"
            schedule_path = (
                PARENT_SCHEDULE_ROOT
                / case
                / f"seed{model_seed}.json"
            )
            schedule = load_parent_schedule(schedule_path, variants)
            if run_dir.exists():
                receipt = _validate_completed_run(run_dir, model_seed)
                wall_seconds = None
                status = "verified_existing"
            else:
                start = time.monotonic()
                train_fixed_400(
                    variants,
                    scalers,
                    schedule,
                    model_seed=model_seed,
                    output_dir=run_dir,
                    device=device,
                )
                wall_seconds = time.monotonic() - start
                receipt = _validate_completed_run(run_dir, model_seed)
                status = "trained"
            runs.append(
                {
                    "case": case,
                    "model_seed": model_seed,
                    "status": status,
                    "wall_seconds": wall_seconds,
                    "schedule_file_sha256": sha256_file(schedule_path),
                    "schedule_payload_sha256": receipt["schedule_sha256"],
                    "final_model_state_sha256": receipt[
                        "final_model_state_sha256"
                    ],
                    "selected_checkpoint_file_sha256": receipt[
                        "checkpoint_file_sha256"
                    ]["update_0400.pt"],
                    "training_receipt_sha256": sha256_file(
                        run_dir / "training_receipt.json"
                    ),
                }
            )

    parent_after = verify_parent_package()
    if parent_after != parent_before:
        raise AssertionError("immutable parent package changed during v3 training")
    shared_source_after = shared_runtime_source_manifest()
    if shared_source_after != shared_source_before:
        raise AssertionError("shared runtime source changed during v3 training")
    v3_source_after = _source_manifest()
    if v3_source_after != v3_source_before:
        raise AssertionError("v3 executable source changed during v3 training")
    complete_grid = (
        set(selected_cases) == set(CASES)
        and set(selected_seeds) == set(FROZEN_CONFIG.paired_model_seeds)
    )
    payload = {
        "schema": GRID_SCHEMA,
        "complete_grid": complete_grid,
        "parent_package": parent_after,
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "fault_manifest_sha256": manifest.sha256,
        "source_code_sha256": v3_source_after,
        "shared_runtime_source_manifest": shared_source_after,
        "training_source_lock_file_sha256": sha256_file(
            output_root / SOURCE_LOCK_NAME
        ),
        "config": json.loads(
            json.dumps(FROZEN_CONFIG.to_dict(), allow_nan=False)
        ),
        "runs": sorted(runs, key=lambda row: (row["case"], row["model_seed"])),
    }
    receipt_name = GRID_RECEIPT if complete_grid else (
        f"training_subset_{_canonical_sha256({'cases': selected_cases, 'seeds': selected_seeds})[:16]}.json"
    )
    receipt_path = output_root / receipt_name
    if os.path.lexists(receipt_path):
        existing = _strict_json(receipt_path)
        comparable_existing = {
            **existing,
            "runs": [
                {**row, "status": "verified_existing", "wall_seconds": None}
                for row in existing["runs"]
            ],
        }
        comparable_payload = {
            **payload,
            "runs": [
                {**row, "status": "verified_existing", "wall_seconds": None}
                for row in payload["runs"]
            ],
        }
        if comparable_existing != comparable_payload:
            raise FileExistsError("existing v3 training-grid receipt differs")
        return receipt_path
    receipt_path.write_bytes(
        (
            json.dumps(payload, indent=2, allow_nan=False)
            + "\n"
        ).encode("ascii")
    )
    return receipt_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    parser.add_argument("--case", action="append", choices=tuple(sorted(CASES)))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = tuple(args.case) if args.case else tuple(sorted(CASES))
    seeds = tuple(args.seed) if args.seed else FROZEN_CONFIG.paired_model_seeds
    path = train_grid(
        args.output.resolve(),
        cases=cases,
        seeds=seeds,
        device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
