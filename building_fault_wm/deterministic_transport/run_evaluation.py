"""Run and independently verify the frozen v3 post-lock evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.fault_data import (
    FaultManifest,
    FaultScalers,
    FaultSpec,
    ScaleStats,
    build_fault_manifest,
    iter_role_variants,
    validate_fault_manifest,
)
from building_fault_wm.neural_benchmark.runtime_provenance import (
    numerical_runtime_fingerprint,
    validate_current_numerical_runtime_fingerprint,
)
from building_fault_wm.neural_benchmark.study_config import StudyConfig

from . import (
    collect,
    corpus,
    evaluate,
    external_freeze,
    gate,
    independent_verify,
    plan,
    prelock,
    train_grid,
)
from .config import FROZEN_CONFIG


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_PRELOCK_ROOT = prelock.DEFAULT_OUTPUT
DEFAULT_DATA_ROOT = collect.CANONICAL_DATA_ROOT
DEFAULT_STATE_ROOT = collect.STATE_ROOT
DEFAULT_READINESS = external_freeze.DEFAULT_READINESS
DEFAULT_EXTERNAL_FREEZE_RECEIPT = external_freeze.DEFAULT_RECEIPT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_evaluation"
)

OUTPUT_SCHEMA = "direct-h8-deterministic-transport-evaluation-v1"
PROVENANCE_SCHEMA = "direct-h8-deterministic-transport-evaluation-provenance-v1"
COMPLETION_SCHEMA = "direct-h8-deterministic-transport-evaluation-completion-v1"
FAULT_MANIFEST_NAME = "locked_transport_fault_manifest.json"
CORE_NAME = "gate_core.csv"
DETAIL_NAME = "detailed_diagnostics.csv"
DIAGNOSTIC_SUMMARY_NAME = "diagnostic_summary.csv"
GATE_RESULT_NAME = "gate_result.json"
PROVENANCE_NAME = "evaluation_provenance.json"
COMPLETION_NAME = "evaluation_complete.json"
PARENT_FAULT_MANIFEST_RELATIVE = "evidence/locked_fault_manifest.json"
PARENT_FROZEN_PREFIX = "experiment/prelock_bundle/frozen"


@dataclass(frozen=True)
class FrozenAssets:
    fault_spec: FaultSpec
    fault_source_file_sha256: str
    fault_contract_file_sha256: str
    scalers: dict[str, FaultScalers]
    scaler_file_sha256_by_case: dict[str, str]
    rssm_checkpoint: dict[tuple[str, int, str], tuple[Path, str]]
    deterministic_checkpoint: dict[tuple[str, int], tuple[Path, str]]
    deterministic_receipt_sha256: dict[tuple[str, int], str]
    deterministic_training_wall_seconds: dict[tuple[str, int], float | None]
    training_grid_file_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")
    return path


def _plain_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")
    return path


def _atomic_bytes(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("ascii"),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    content = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("ascii")
    _atomic_bytes(path, content)


def _scale_stats(payload: object, dimension: int, label: str) -> ScaleStats:
    if not isinstance(payload, dict) or set(payload) != {"mean", "scale"}:
        raise ValueError(f"{label} scaler fields are invalid")
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0.0).any()
    ):
        raise ValueError(f"{label} scaler values are invalid")
    return ScaleStats(
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
    )


def _load_scaler(path: Path, case: str, expected_sha256: str) -> FaultScalers:
    if plan.sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen FIT scaler hash differs for {case}")
    payload = plan.load_json(path)
    if set(payload) != {
        "observation",
        "action",
        "context",
        "fit_source_sha256",
    }:
        raise ValueError(f"frozen FIT scaler fields differ for {case}")
    sources = payload["fit_source_sha256"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].startswith(f"{case}:fit:")
            or not boptest.valid_sha256(item[1])
            for item in sources
        )
    ):
        raise ValueError(f"frozen FIT scaler source grid differs for {case}")
    return FaultScalers(
        observation=_scale_stats(payload["observation"], 4, "observation"),
        action=_scale_stats(payload["action"], 1, "action"),
        context=_scale_stats(payload["context"], 5, "context"),
        fit_source_sha256=tuple((str(a), str(b)) for a, b in sources),
    )


def _load_fault_spec(
    parent_selected: Path,
    reused_hashes: Mapping[str, str],
) -> tuple[FaultSpec, str, str]:
    manifest_path = _plain_file(
        parent_selected / PARENT_FAULT_MANIFEST_RELATIVE,
        "parent locked fault manifest",
    )
    manifest_sha256 = plan.sha256_file(manifest_path)
    if reused_hashes.get(PARENT_FAULT_MANIFEST_RELATIVE) != manifest_sha256:
        raise ValueError("parent locked fault manifest is not byte-bound")
    parent_manifest = FaultManifest.read(manifest_path)
    if (
        parent_manifest.schema
        != "boptest-multicase-fault-manifest-v2"
        or not parent_manifest.cells
    ):
        raise ValueError("parent locked fault manifest contract changed")

    contract_relative = f"{PARENT_FROZEN_PREFIX}/frozen_fault_contract.json"
    contract_path = _plain_file(
        parent_selected / contract_relative, "parent frozen fault contract"
    )
    contract_sha256 = plan.sha256_file(contract_path)
    if reused_hashes.get(contract_relative) != contract_sha256:
        raise ValueError("parent frozen fault contract is not byte-bound")
    contract = plan.load_json(contract_path)
    contract_spec_payload = contract.get("spec")
    if (
        contract.get("schema") != "boptest-multicase-frozen-fault-contract-v1"
        or not isinstance(contract_spec_payload, dict)
        or FaultSpec.from_dict(contract_spec_payload) != parent_manifest.spec
    ):
        raise ValueError(
            "parent locked fault manifest and frozen fault contract disagree"
        )
    if parent_manifest.spec.evaluation_horizon != 8:
        raise ValueError("parent locked fault specification is not H8")
    return parent_manifest.spec, manifest_sha256, contract_sha256


def load_frozen_assets(
    prelock_root: Path,
    registry: Mapping[str, object],
) -> FrozenAssets:
    """Load only artifacts already byte-bound by the validated pre-lock."""

    bundle = _plain_directory(prelock_root / prelock.BUNDLE_NAME, "pre-lock bundle")
    parent_selected = _plain_directory(
        bundle / "parent_selected", "pre-lock selected parent artifacts"
    )
    reused = registry.get("parent_reused_artifact_sha256_by_path")
    if not isinstance(reused, dict):
        raise ValueError("pre-lock parent reused-artifact map is missing")
    fault_spec, fault_sha, contract_sha = _load_fault_spec(
        parent_selected, reused
    )

    scalers: dict[str, FaultScalers] = {}
    scaler_hashes: dict[str, str] = {}
    rssm: dict[tuple[str, int, str], tuple[Path, str]] = {}
    for case in gate.CASES:
        scaler_relative = f"{PARENT_FROZEN_PREFIX}/fit_scalers/{case}.json"
        scaler_sha = reused.get(scaler_relative)
        if not isinstance(scaler_sha, str):
            raise ValueError(f"pre-lock scaler binding is missing for {case}")
        scaler_path = parent_selected / scaler_relative
        scalers[case] = _load_scaler(scaler_path, case, scaler_sha)
        scaler_hashes[case] = scaler_sha
        for seed in gate.CONFIRMATION_SEEDS:
            for arm in evaluate.RSSM_ARMS:
                relative = (
                    f"{PARENT_FROZEN_PREFIX}/checkpoints/{case}/"
                    f"seed{seed}/{arm}_u0400.pt"
                )
                digest = reused.get(relative)
                if not isinstance(digest, str):
                    raise ValueError(
                        f"pre-lock RSSM checkpoint binding is missing: {case}/{seed}/{arm}"
                    )
                path = _plain_file(
                    parent_selected / relative, "pre-lock RSSM checkpoint"
                )
                if plan.sha256_file(path) != digest:
                    raise ValueError(f"pre-lock RSSM checkpoint changed: {relative}")
                rssm[(case, seed, arm)] = (path, digest)

    training_root = _plain_directory(
        bundle / "training", "pre-lock deterministic training"
    )
    grid_path = _plain_file(
        training_root / "training_grid_complete.json",
        "pre-lock training-grid receipt",
    )
    grid = plan.load_json(grid_path)
    rows = grid.get("runs")
    if (
        grid.get("schema") != train_grid.GRID_SCHEMA
        or grid.get("complete_grid") is not True
        or not isinstance(rows, list)
    ):
        raise ValueError("pre-lock deterministic training grid changed")
    by_unit = {
        (row.get("case"), row.get("model_seed")): row
        for row in rows
        if isinstance(row, dict)
    }
    expected_units = {
        (case, seed)
        for case in gate.CASES
        for seed in gate.CONFIRMATION_SEEDS
    }
    if set(by_unit) != expected_units:
        raise ValueError("deterministic training-grid unit map is incomplete")

    deterministic: dict[tuple[str, int], tuple[Path, str]] = {}
    receipt_hashes: dict[tuple[str, int], str] = {}
    training_wall_seconds: dict[tuple[str, int], float | None] = {}
    for case, seed in sorted(expected_units):
        unit = training_root / case / f"seed{seed}"
        receipt_path = _plain_file(
            unit / "training_receipt.json", "deterministic training receipt"
        )
        row = by_unit[(case, seed)]
        receipt_sha = plan.sha256_file(receipt_path)
        if row.get("training_receipt_sha256") != receipt_sha:
            raise ValueError("training grid binds a different unit receipt")
        receipt = plan.load_json(receipt_path)
        hashes = receipt.get("checkpoint_file_sha256")
        selected_name = "update_0400.pt"
        if (
            receipt.get("schema")
            != "boptest-deterministic-transport-training-v1"
            or receipt.get("model_seed") != seed
            or receipt.get("selected_update") != 400
            or receipt.get("selection_rule")
            != "fixed_final_update_no_validation_selection"
            or receipt.get("config")
            != json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
            or not isinstance(hashes, dict)
            or row.get("selected_checkpoint_file_sha256")
            != hashes.get(selected_name)
        ):
            raise ValueError(
                f"deterministic selected-checkpoint contract changed: {case}/{seed}"
            )
        checkpoint = _plain_file(
            unit / "checkpoints" / selected_name,
            "deterministic selected checkpoint",
        )
        digest = str(hashes[selected_name])
        if plan.sha256_file(checkpoint) != digest:
            raise ValueError(
                f"deterministic selected checkpoint changed: {case}/{seed}"
            )
        deterministic[(case, seed)] = (checkpoint, digest)
        receipt_hashes[(case, seed)] = receipt_sha
        raw_wall_seconds = row.get("wall_seconds")
        if raw_wall_seconds is not None and (
            isinstance(raw_wall_seconds, bool)
            or not isinstance(raw_wall_seconds, (int, float))
            or not np.isfinite(float(raw_wall_seconds))
            or float(raw_wall_seconds) <= 0.0
        ):
            raise ValueError(
                f"deterministic training wall time is invalid: {case}/{seed}"
            )
        training_wall_seconds[(case, seed)] = (
            None
            if raw_wall_seconds is None
            else float(raw_wall_seconds)
        )
    return FrozenAssets(
        fault_spec=fault_spec,
        fault_source_file_sha256=fault_sha,
        fault_contract_file_sha256=contract_sha,
        scalers=scalers,
        scaler_file_sha256_by_case=scaler_hashes,
        rssm_checkpoint=rssm,
        deterministic_checkpoint=deterministic,
        deterministic_receipt_sha256=receipt_hashes,
        deterministic_training_wall_seconds=training_wall_seconds,
        training_grid_file_sha256=plan.sha256_file(grid_path),
    )


def _configure_runtime(registry: Mapping[str, object]) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    frozen = registry.get("v3_numerical_runtime")
    validate_current_numerical_runtime_fingerprint(
        frozen, include_sklearn=True
    )
    current = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    if current != frozen:
        raise ValueError("evaluation runtime differs from the pre-lock runtime")
    return current


def _load_models(
    assets: FrozenAssets,
    *,
    case: str,
    seed: int,
) -> dict[str, torch.nn.Module]:
    config = StudyConfig()
    models: dict[str, torch.nn.Module] = {}
    for arm in evaluate.RSSM_ARMS:
        path, digest = assets.rssm_checkpoint[(case, seed, arm)]
        models[arm] = evaluate.load_frozen_v2_rssm(
            path,
            case=case,
            model_seed=seed,
            arm=arm,
            expected_file_sha256=digest,
            config=config,
            device="cpu",
        )
    path, digest = assets.deterministic_checkpoint[(case, seed)]
    models["deterministic_wm"] = evaluate.load_deterministic_checkpoint(
        path,
        model_seed=seed,
        expected_file_sha256=digest,
        device="cpu",
    )
    return models


def _diagnostic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group = [
        "case",
        "policy",
        "model_seed",
        "arm",
        "fault_channel",
        "family",
        "horizon",
        "boundary_crossing",
        "raw_unit",
    ]
    result = (
        frame.groupby(group, as_index=False, dropna=False)
        .agg(
            rows=("standardized_abs_error", "size"),
            standardized_mae=("standardized_abs_error", "mean"),
            raw_mae=("raw_abs_error", "mean"),
            alternate_action_change_mean=(
                "action_prediction_change_standardized",
                "mean",
            ),
            action_transition_count_mean=("action_transition_count", "mean"),
        )
        .sort_values(group, kind="stable")
        .reset_index(drop=True)
    )
    numeric = result.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if result.empty or not np.isfinite(numeric).all():
        raise ValueError("diagnostic aggregation is empty or non-finite")
    return result


def _artifact_inventory(root: Path, exclude: set[str] | None = None) -> list[dict]:
    excluded = set() if exclude is None else exclude
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("evaluation output contains a symbolic link")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                rows.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": plan.sha256_file(path),
                    }
                )
        elif not path.is_dir():
            raise ValueError("evaluation output contains a non-regular entry")
    return rows


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )


def _input_hash_payload(
    *,
    prelock_sha256: str,
    readiness_sha256: str,
    freeze_receipt_path: Path,
    validated: corpus.ValidatedCollection,
    assets: FrozenAssets,
) -> dict:
    return {
        "prelock_registry_sha256": prelock_sha256,
        "readiness_sha256": readiness_sha256,
        "external_freeze_receipt_file_sha256": plan.sha256_file(
            freeze_receipt_path
        ),
        "corpus_manifest_file_sha256": validated.manifest_file_sha256,
        "corpus_manifest_payload_sha256": validated.index.manifest_sha256,
        "collection_attempt_file_sha256": validated.attempt_file_sha256,
        "collection_completion_file_sha256": validated.completion_file_sha256,
        "parent_locked_fault_manifest_file_sha256": (
            assets.fault_source_file_sha256
        ),
        "parent_frozen_fault_contract_file_sha256": (
            assets.fault_contract_file_sha256
        ),
        "fit_scaler_file_sha256_by_case": (
            assets.scaler_file_sha256_by_case
        ),
        "training_grid_file_sha256": assets.training_grid_file_sha256,
        "rssm_checkpoint_file_sha256": {
            f"{case}/seed{seed}/{arm}": digest
            for (case, seed, arm), (_, digest) in sorted(
                assets.rssm_checkpoint.items()
            )
        },
        "deterministic_checkpoint_file_sha256": {
            f"{case}/seed{seed}": digest
            for (case, seed), (_, digest) in sorted(
                assets.deterministic_checkpoint.items()
            )
        },
        "deterministic_training_receipt_file_sha256": {
            f"{case}/seed{seed}": digest
            for (case, seed), digest in sorted(
                assets.deterministic_receipt_sha256.items()
            )
        },
    }


def run_evaluation(
    *,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
    external_freeze_receipt_path: Path = DEFAULT_EXTERNAL_FREEZE_RECEIPT,
    output_dir: Path = DEFAULT_OUTPUT,
    live_external_freeze: bool = True,
) -> Path:
    """Evaluate the complete frozen grid and publish it as one atomic tree."""

    if os.path.lexists(output_dir):
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale evaluation staging exists: {staging}")

    registry, readiness = corpus.load_bound_readiness(
        prelock_root=prelock_root,
        live_data_root=data_root,
        expected_prelock_sha256=expected_prelock_sha256,
        expected_readiness_sha256=expected_readiness_sha256,
    )
    freeze = external_freeze.validate_external_freeze_receipt(
        external_freeze_receipt_path,
        expected_prelock_sha256,
        expected_readiness_sha256,
        prelock_root=prelock_root,
        readiness_path=readiness_path,
        live=live_external_freeze,
    )
    validated = corpus.load_transport_corpus_index(
        manifest_path=(
            data_root / "manifests/locked_transport_corpus_manifest.json"
        ),
        raw_root=data_root / "locked_transport_raw",
        readiness=readiness,
        expected_prelock_sha256=expected_prelock_sha256,
        state_root=state_root,
        external_freeze=freeze,
        external_freeze_receipt_path=external_freeze_receipt_path,
    )
    assets = load_frozen_assets(prelock_root, registry)
    runtime = _configure_runtime(registry)
    started_utc = _utc_now()
    started = time.perf_counter()
    starting_peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    staging.mkdir(parents=True)
    try:
        fault_manifest = build_fault_manifest(
            validated.index, assets.fault_spec
        )
        validate_fault_manifest(fault_manifest, validated.index)
        _write_json(staging / FAULT_MANIFEST_NAME, fault_manifest.payload())
        variants = tuple(
            iter_role_variants(
                validated.index,
                fault_manifest,
                "locked_test",
                allow_locked_test=True,
            )
        )
        if len(variants) != 72 * 44:
            raise ValueError("v3 fault-variant grid is incomplete")

        core_frames: list[pd.DataFrame] = []
        detail_frames: list[pd.DataFrame] = []
        unit_timing: list[dict[str, object]] = []
        for case in gate.CASES:
            case_variants = tuple(
                variant
                for variant in variants
                if variant.cell.trajectory.case == case
            )
            case_keys = {
                variant.cell.trajectory for variant in case_variants
            }
            case_metadata = {
                key: validated.trajectory_metadata[key] for key in case_keys
            }
            for seed in gate.CONFIRMATION_SEEDS:
                unit_started = time.perf_counter()
                load_started = time.perf_counter()
                models = _load_models(assets, case=case, seed=seed)
                load_seconds = time.perf_counter() - load_started
                prediction_seconds = {arm: 0.0 for arm in gate.ARMS}
                frames = evaluate.evaluate_transport_models(
                    models,
                    case_variants,
                    assets.scalers[case],
                    case_metadata,
                    model_seed=seed,
                    horizons=evaluate.EVALUATION_HORIZONS,
                    rssm_config=StudyConfig(),
                    device="cpu",
                    prediction_seconds_by_arm=prediction_seconds,
                )
                core_frames.append(frames.core)
                detail_frames.append(frames.detailed)
                unit_timing.append(
                    {
                        "case": case,
                        "model_seed": seed,
                        "model_load_seconds": float(load_seconds),
                        "prediction_seconds_by_arm": prediction_seconds,
                        "unit_wall_seconds": float(
                            time.perf_counter() - unit_started
                        ),
                        "core_rows": len(frames.core),
                    }
                )
                del models, frames
                gc.collect()

        core_frame = pd.concat(core_frames, ignore_index=True)
        detail_frame = pd.concat(detail_frames, ignore_index=True)
        expected_rows = 72 * 44 * 4 * 4 * 3 * 5
        if len(core_frame) != expected_rows or len(detail_frame) != expected_rows:
            raise ValueError("v3 evaluation row grid is incomplete")
        evaluate.assert_paired_rows(core_frame)
        core_path = staging / CORE_NAME
        detail_path = staging / DETAIL_NAME
        _write_csv(core_path, core_frame)
        _write_csv(detail_path, detail_frame)
        diagnostics = _diagnostic_summary(detail_frame)
        _write_csv(staging / DIAGNOSTIC_SUMMARY_NAME, diagnostics)

        # The persisted CSV, rather than the in-memory frame, is the gate source.
        persisted_core = pd.read_csv(core_path)
        persisted_core = persisted_core.loc[:, gate.REQUIRED_COLUMNS]
        gate_result = gate.analyze_gate(
            persisted_core,
            bootstrap_draws=gate.BOOTSTRAP_DRAWS,
            bootstrap_seed=gate.BOOTSTRAP_SEED,
        )
        _write_json(staging / GATE_RESULT_NAME, gate_result)

        input_hashes = _input_hash_payload(
            prelock_sha256=expected_prelock_sha256,
            readiness_sha256=expected_readiness_sha256,
            freeze_receipt_path=external_freeze_receipt_path,
            validated=validated,
            assets=assets,
        )
        elapsed = time.perf_counter() - started
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "study_kind": "direct_h8_deterministic_transport_v3",
            "evaluation_started_at_utc": started_utc,
            "evaluation_completed_at_utc": _utc_now(),
            "input_hashes": input_hashes,
            "external_freeze": {
                key: freeze[key]
                for key in (
                    "provider",
                    "gist_id",
                    "revision",
                    "owner_login",
                    "revision_committed_at_utc",
                    "revision_api_url",
                    "revision_html_url",
                )
            },
            "runtime": runtime,
            "runtime_policy": {
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
            "evaluation_contract": {
                "cases": list(gate.CASES),
                "model_seeds": list(gate.CONFIRMATION_SEEDS),
                "arms": list(gate.ARMS),
                "policies": list(gate.POLICIES),
                "horizons": list(evaluate.EVALUATION_HORIZONS),
                "history": evaluate.EVALUATION_HISTORY,
                "role": "locked_test",
                "fault_spec": asdict(assets.fault_spec),
                "bootstrap_draws": gate.BOOTSTRAP_DRAWS,
                "bootstrap_seed": gate.BOOTSTRAP_SEED,
            },
            "counts": {
                "clean_trajectories": len(validated.index.records),
                "fault_variants": len(variants),
                "gate_core_rows": len(core_frame),
                "detailed_rows": len(detail_frame),
                "diagnostic_summary_rows": len(diagnostics),
                "evaluation_units": len(unit_timing),
            },
            "model_resources": {
                "rssm_total_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in evaluate.ReliabilityGatedRSSM(
                            StudyConfig().model_config()
                        ).parameters()
                    )
                ),
                "rssm_active_observation_dynamics_parameters": 19_784,
                "deterministic_total_parameters": 19_789,
                "selected_checkpoint_bytes": {
                    **{
                        f"{case}/seed{seed}/{arm}": path.stat().st_size
                        for (case, seed, arm), (path, _) in sorted(
                            assets.rssm_checkpoint.items()
                        )
                    },
                    **{
                        f"{case}/seed{seed}/deterministic_wm": path.stat().st_size
                        for (case, seed), (path, _) in sorted(
                            assets.deterministic_checkpoint.items()
                        )
                    },
                },
                "training": {
                    "updates_per_model": FROZEN_CONFIG.updates,
                    "deterministic": {
                        "timing_source_file_sha256": (
                            assets.training_grid_file_sha256
                        ),
                        "wall_seconds_by_case_seed": {
                            f"{case}/seed{seed}": value
                            for (case, seed), value in sorted(
                                assets.deterministic_training_wall_seconds.items()
                            )
                        },
                        "available_count": sum(
                            value is not None
                            for value in assets.deterministic_training_wall_seconds.values()
                        ),
                        "unavailable_count": sum(
                            value is None
                            for value in assets.deterministic_training_wall_seconds.values()
                        ),
                        "unavailable_reason": (
                            "verified_existing grid entries do not record wall time"
                            if any(
                                value is None
                                for value in assets.deterministic_training_wall_seconds.values()
                            )
                            else None
                        ),
                    },
                    "rssm": {
                        "available": False,
                        "reason": (
                            "byte-bound parent checkpoint provenance contains "
                            "no training wall time"
                        ),
                    },
                },
            },
            "timing": {
                "total_wall_seconds": float(elapsed),
                "unit_timing": unit_timing,
                "peak_rss_kb_before_evaluation": int(starting_peak_rss_kb),
                "peak_rss_kb_after_evaluation": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            },
        }
        _write_json(staging / PROVENANCE_NAME, provenance)
        inventory = _artifact_inventory(staging)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "study_kind": "direct_h8_deterministic_transport_v3",
            "prelock_registry_sha256": expected_prelock_sha256,
            "readiness_sha256": expected_readiness_sha256,
            "corpus_manifest_payload_sha256": validated.index.manifest_sha256,
            "fault_manifest_sha256": fault_manifest.sha256,
            "gate_result_sha256": plan.canonical_sha256(gate_result),
            "provenance_file_sha256": plan.sha256_file(
                staging / PROVENANCE_NAME
            ),
            "artifact_inventory_excludes_completion": inventory,
            "artifact_inventory_sha256": plan.canonical_sha256(inventory),
            "complete": True,
        }
        _write_json(staging / COMPLETION_NAME, completion)
        _seal_tree(staging)
        if os.path.lexists(output_dir):
            raise FileExistsError("evaluation destination became occupied")
        staging.rename(output_dir)
        return output_dir
    except BaseException:
        # Keep an incomplete staging tree as explicit evidence.  It is never a
        # publishable result and cannot be mistaken for the atomic output.
        raise


def verify_evaluation_output(
    output_dir: Path,
    *,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    validated: corpus.ValidatedCollection,
    assets: FrozenAssets,
    external_freeze_receipt_path: Path,
) -> dict:
    """Reconstruct the gate from persisted CSV and verify every output hash."""

    _plain_directory(output_dir, "v3 evaluation output")
    completion = plan.load_json(
        _plain_file(output_dir / COMPLETION_NAME, "evaluation completion")
    )
    expected_fields = {
        "schema",
        "study_kind",
        "prelock_registry_sha256",
        "readiness_sha256",
        "corpus_manifest_payload_sha256",
        "fault_manifest_sha256",
        "gate_result_sha256",
        "provenance_file_sha256",
        "artifact_inventory_excludes_completion",
        "artifact_inventory_sha256",
        "complete",
    }
    if set(completion) != expected_fields:
        raise ValueError("evaluation completion fields changed")
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["complete"] is not True
        or completion["prelock_registry_sha256"] != expected_prelock_sha256
        or completion["readiness_sha256"] != expected_readiness_sha256
        or completion["corpus_manifest_payload_sha256"]
        != validated.index.manifest_sha256
    ):
        raise ValueError("evaluation completion identity changed")
    actual_inventory = _artifact_inventory(
        output_dir, exclude={COMPLETION_NAME}
    )
    if (
        completion["artifact_inventory_excludes_completion"]
        != actual_inventory
        or completion["artifact_inventory_sha256"]
        != plan.canonical_sha256(actual_inventory)
    ):
        raise ValueError("evaluation artifact inventory changed")

    fault_manifest = FaultManifest.read(output_dir / FAULT_MANIFEST_NAME)
    validate_fault_manifest(fault_manifest, validated.index)
    if (
        fault_manifest.spec != assets.fault_spec
        or fault_manifest.sha256 != completion["fault_manifest_sha256"]
    ):
        raise ValueError("persisted v3 fault manifest changed")

    core = pd.read_csv(output_dir / CORE_NAME).loc[:, gate.REQUIRED_COLUMNS]
    detail = pd.read_csv(output_dir / DETAIL_NAME)
    if tuple(detail.columns) != evaluate.DETAIL_COLUMNS:
        raise ValueError("persisted detailed diagnostic columns changed")
    pd.testing.assert_frame_equal(
        core.reset_index(drop=True),
        detail.loc[:, gate.REQUIRED_COLUMNS].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    reconstructed = gate.analyze_gate(
        core,
        bootstrap_draws=gate.BOOTSTRAP_DRAWS,
        bootstrap_seed=gate.BOOTSTRAP_SEED,
    )
    recorded = plan.load_json(output_dir / GATE_RESULT_NAME)
    if (
        recorded != reconstructed
        or completion["gate_result_sha256"]
        != plan.canonical_sha256(reconstructed)
    ):
        raise ValueError("persisted gate result does not reconstruct from CSV")
    independent_receipt = independent_verify.verify_files(
        output_dir / CORE_NAME,
        output_dir / GATE_RESULT_NAME,
    )

    provenance_path = output_dir / PROVENANCE_NAME
    provenance = plan.load_json(provenance_path)
    if (
        provenance.get("schema") != PROVENANCE_SCHEMA
        or completion["provenance_file_sha256"]
        != plan.sha256_file(provenance_path)
    ):
        raise ValueError("evaluation provenance changed")
    expected_inputs = _input_hash_payload(
        prelock_sha256=expected_prelock_sha256,
        readiness_sha256=expected_readiness_sha256,
        freeze_receipt_path=external_freeze_receipt_path,
        validated=validated,
        assets=assets,
    )
    if provenance.get("input_hashes") != expected_inputs:
        raise ValueError("evaluation provenance input binding changed")
    return {
        "schema": OUTPUT_SCHEMA,
        "verified": True,
        "independent_gate_verified": independent_receipt["verified"],
        "primary_architecture_category": recorded[
            "primary_architecture_category"
        ],
        "primary_supervision_category": recorded[
            "primary_supervision_category"
        ],
        "gate_core_rows": len(core),
        "artifact_inventory_sha256": completion[
            "artifact_inventory_sha256"
        ],
    }


def verify_only(
    *,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
    external_freeze_receipt_path: Path = DEFAULT_EXTERNAL_FREEZE_RECEIPT,
    output_dir: Path = DEFAULT_OUTPUT,
    live_external_freeze: bool = True,
) -> dict:
    registry, readiness = corpus.load_bound_readiness(
        prelock_root=prelock_root,
        live_data_root=data_root,
        expected_prelock_sha256=expected_prelock_sha256,
        expected_readiness_sha256=expected_readiness_sha256,
    )
    freeze = external_freeze.validate_external_freeze_receipt(
        external_freeze_receipt_path,
        expected_prelock_sha256,
        expected_readiness_sha256,
        prelock_root=prelock_root,
        readiness_path=readiness_path,
        live=live_external_freeze,
    )
    validated = corpus.load_transport_corpus_index(
        manifest_path=(
            data_root / "manifests/locked_transport_corpus_manifest.json"
        ),
        raw_root=data_root / "locked_transport_raw",
        readiness=readiness,
        expected_prelock_sha256=expected_prelock_sha256,
        state_root=state_root,
        external_freeze=freeze,
        external_freeze_receipt_path=external_freeze_receipt_path,
    )
    assets = load_frozen_assets(prelock_root, registry)
    return verify_evaluation_output(
        output_dir,
        expected_prelock_sha256=expected_prelock_sha256,
        expected_readiness_sha256=expected_readiness_sha256,
        validated=validated,
        assets=assets,
        external_freeze_receipt_path=external_freeze_receipt_path,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--expected-prelock-sha256", required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--prelock-root", type=Path, default=DEFAULT_PRELOCK_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument(
        "--external-freeze-receipt",
        type=Path,
        default=DEFAULT_EXTERNAL_FREEZE_RECEIPT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    kwargs = {
        "expected_prelock_sha256": args.expected_prelock_sha256,
        "expected_readiness_sha256": args.expected_readiness_sha256,
        "prelock_root": args.prelock_root.resolve(),
        "data_root": args.data_root.resolve(),
        "state_root": args.state_root.resolve(),
        "readiness_path": args.readiness.resolve(),
        "external_freeze_receipt_path": (
            args.external_freeze_receipt.resolve()
        ),
        "output_dir": args.output.resolve(),
    }
    if args.command == "run":
        result = run_evaluation(**kwargs)
        print(result)
    else:
        print(json.dumps(verify_only(**kwargs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
