from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from . import study_evaluate as study_evaluate_module
from . import study_gate as study_gate_module
from .baselines import (
    evaluate_arx_h8,
    evaluate_direct_h8_gru,
    evaluate_direct_h8_ridge,
)
from .fault_data import (
    FaultScalers,
    FaultSpec,
    ScaleStats,
    build_fault_manifest,
    fault_cell_signatures,
    iter_role_variants,
)
from .locked_state import (
    COLLECTION_COMPLETION_MARKER,
    CONFIRMATION_ATTEMPT_MARKER,
    CONFIRMATION_COMPLETION_MARKER,
    CONFIRMATION_FAILURE_MARKER,
    STATE_ROOT,
    state_dir_for_digest,
)
from .protocol import CASES, TRAJECTORY_STEPS, sha256_file
from .provenance import (
    FROZEN_FAULT_CONTRACT_SCHEMA,
    canonical_sha256,
    load_selected_baseline_models,
    load_strict_json,
    prelock_plan_sha256_by_case,
    validate_locked_corpus_binding,
    validate_prelock_bundle,
)
from .runtime_provenance import numerical_runtime_fingerprint
from .study_config import ARMS, StudyConfig
from .study_development import scientific_code_manifest
from .study_evaluate import evaluate_model_h8, load_model_checkpoint
from .study_gate import (
    canonical_frame_sha256,
    evaluate_study_gate,
    write_gate_analysis,
)
from .study_locked_collection import (
    CANONICAL_LOCKED_MANIFEST,
    CANONICAL_PRELOCK_ROOT,
    CANONICAL_PRELOCK_BUNDLE,
    CANONICAL_PRELOCK_REGISTRY,
    validate_locked_collection_completion,
)


HERE = Path(__file__).resolve().parent
CANONICAL_LOCKED_SOURCE_ROOT = HERE / "data_v4"
CONFIRMATION_TOKEN = "I_UNDERSTAND_LOCKED_EVALUATION_IS_ONE_SHOT"
RUN_CONFIG_SCHEMA = "boptest-reliability-confirmation-run-v1"
RUN_RECEIPT_SCHEMA = "boptest-reliability-confirmation-receipt-v1"
ATTEMPT_SCHEMA = "boptest-reliability-confirmation-attempt-v1"
FAILURE_SCHEMA = "boptest-reliability-confirmation-failure-v1"
COMPLETION_MARKER_SCHEMA = "boptest-reliability-confirmation-publication-v1"

ATTEMPT_MARKER = CONFIRMATION_ATTEMPT_MARKER
FAILURE_MARKER = CONFIRMATION_FAILURE_MARKER
COMPLETION_MARKER = CONFIRMATION_COMPLETION_MARKER
ATTEMPT_EVIDENCE_PATH = "confirmation_attempt.json"
COLLECTION_EVIDENCE_PATH = "locked_collection_completion.json"
ATTEMPT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "role",
        "one_shot",
        "started_at_utc",
        "locked_manifest_path",
        "output_path",
        "prelock_registry_sha256",
        "selected_update",
        "locked_corpus_manifest_sha256",
        "locked_manifest_file_sha256",
        "locked_collection_completion_sha256",
        "external_freeze_receipt_sha256",
        "runner_sha256",
        "scientific_code_manifest_sha256",
        "evaluation_policy",
        "evaluation_runtime",
        "attempt_identity_sha256",
    }
)

COMPETENCE_BASELINES = (
    "ridge_arx",
    "direct_h8_ridge",
    "deterministic_gru",
)
RESULT_SORT_COLUMNS = (
    "case",
    "model_seed",
    "arm",
    "update",
    "trajectory_day",
    "trajectory_seed",
    "fault_channel",
    "family",
    "sign",
    "severity",
    "onset",
    "anchor",
)
RUNNER_RELATIVE_PATH = "multicase_fault_benchmark/study_confirmation.py"
EVALUATION_POLICY = MappingProxyType(
    {
        "device": "cpu",
        "rssm_sampling": False,
        "native_threadpool_threads": 1,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "torch_deterministic_algorithms": True,
        "future_observations_available_to_imagination": False,
        "locked_variant_materializations": 1,
    }
)
EVALUATION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "role",
        "selected_update",
        "prelock_registry_sha256",
        "study_config_sha256",
        "confirmation_runner_sha256",
        "scientific_code_sha256_by_path",
        "scientific_code_manifest_sha256",
        "evaluator_sha256",
        "gate_sha256",
        "locked_corpus_manifest_sha256",
        "locked_fault_manifest_sha256",
        "attempt_marker_sha256",
        "attempt_marker_artifact",
        "locked_collection_completion_artifact",
        "rssm_frame_sha256",
        "baseline_frame_sha256",
        "rssm_result_artifact",
        "baseline_result_artifact",
        "evaluation_policy",
        "evaluation_runtime",
        "evaluation_source_manifest_artifact",
        "artifact_inventory",
    }
)
RUN_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "role",
        "one_shot",
        "claim_requires_completion_marker",
        "completed_at_utc",
        "decision",
        "gate_pass",
        "paper_claim_allowed",
        "selected_update",
        "prelock_registry_sha256",
        "locked_corpus_manifest_sha256",
        "locked_fault_manifest_sha256",
        "attempt_marker_sha256",
        "evaluation_receipt_sha256",
        "rssm_result",
        "baseline_result",
        "gate_result_sha256",
        "matched_result_sha256",
        "evaluation_policy",
        "evaluation_runtime",
        "scientific_code_sha256_by_path",
        "scientific_code_manifest_sha256",
        "wall_time",
        "artifact_inventory_excludes_this_receipt",
    }
)
COMPLETION_MARKER_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "completed_at_utc",
        "prelock_registry_sha256",
        "attempt_marker_sha256",
        "output_path",
        "run_complete_sha256",
        "decision",
        "gate_pass",
        "paper_claim_allowed",
    }
)
FAILURE_MARKER_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "failed_at_utc",
        "prelock_registry_sha256",
        "attempt_marker_sha256",
        "output_path",
        "error_type",
        "error_message",
        "output_published",
        "retry_allowed",
    }
)


@dataclass(frozen=True)
class FrozenAssets:
    selected_update: int
    scalers_by_case: Mapping[str, FaultScalers]
    rssm_models: Mapping[tuple[str, int, str], object]
    baseline_models: Mapping[str, Mapping[str, tuple[object, object]]]
    fault_contract: Mapping[str, object]
    plan_sha256_by_case: Mapping[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_canonical_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical_bytes(payload))


def _write_json_once(path: Path, payload: object) -> None:
    """Create an immutable marker without an overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite one-shot marker: {path}")
    content = _canonical_bytes(payload)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_file(root: Path, relative: object) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or Path(relative).as_posix() != relative
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
    ):
        raise ValueError("artifact path is not a canonical relative path")
    resolved_root = root.resolve()
    candidate = resolved_root
    for part in Path(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("artifact path traverses a symbolic link")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("artifact path escapes its root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact file is missing: {relative}")
    return resolved


def _reference_path(
    root: Path,
    reference: object,
    *,
    expected_kind: str,
    expected_identity: str,
) -> Path:
    if not isinstance(reference, dict) or set(reference) != {
        "path",
        "sha256",
        "kind",
        "identity",
    }:
        raise ValueError("pre-lock artifact reference fields are invalid")
    if (
        reference["kind"] != expected_kind
        or reference["identity"] != expected_identity
        or not _is_sha256(reference["sha256"])
    ):
        raise ValueError("pre-lock artifact reference identity is invalid")
    path = _safe_relative_file(root, reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("pre-lock artifact differs from its frozen SHA-256")
    return path


def _scale_stats(payload: object, *, dimension: int, label: str) -> ScaleStats:
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
    return ScaleStats(tuple(float(value) for value in mean), tuple(float(value) for value in scale))


def _load_frozen_scaler(path: Path, *, case: str, config: StudyConfig) -> FaultScalers:
    payload = load_strict_json(path)
    if not isinstance(payload, dict) or set(payload) != {
        "observation",
        "action",
        "context",
        "fit_source_sha256",
    }:
        raise ValueError(f"FIT scaler artifact fields are invalid for {case}")
    sources = payload["fit_source_sha256"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].startswith(f"{case}:fit:")
            or not _is_sha256(item[1])
            for item in sources
        )
    ):
        raise ValueError(f"FIT scaler source identities are invalid for {case}")
    return FaultScalers(
        observation=_scale_stats(
            payload["observation"], dimension=config.observation_dim, label="observation"
        ),
        action=_scale_stats(payload["action"], dimension=config.action_dim, label="action"),
        context=_scale_stats(
            payload["context"], dimension=config.context_dim, label="context"
        ),
        fit_source_sha256=tuple((str(key), str(digest)) for key, digest in sources),
    )


def confirmation_scientific_code_manifest() -> dict[str, str]:
    manifest = dict(scientific_code_manifest())
    manifest[RUNNER_RELATIVE_PATH] = sha256_file(Path(__file__).resolve())
    return dict(sorted(manifest.items()))


def _configure_evaluation_policy() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise RuntimeError(
                "confirmation requires a fresh process with one Torch interop thread"
            ) from error
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    unconstrained = [
        (pool.get("internal_api"), pool.get("num_threads"))
        for pool in threadpool_info()
        if pool.get("num_threads") != 1
    ]
    if unconstrained:
        raise RuntimeError(
            f"confirmation native thread pools are not fixed to one: {unconstrained}"
        )


def _expected_checkpoint_keys(config: StudyConfig, update: int) -> set[str]:
    return {
        f"{case}:seed{seed}:{arm}:u{update:04d}"
        for case in CASES
        for seed in config.confirmatory_seeds
        for arm in ARMS
    }


def _expected_baseline_keys(config: StudyConfig, baseline: str) -> set[str]:
    if baseline == "deterministic_gru":
        return {
            f"{case}:seed{seed}"
            for case in CASES
            for seed in config.confirmatory_seeds
        }
    return {f"{case}:seed0" for case in CASES}


def _load_frozen_assets(
    registry: dict,
    artifact_root: Path,
    config: StudyConfig,
) -> FrozenAssets:
    selected_update = registry.get("selected_update")
    if (
        isinstance(selected_update, bool)
        or not isinstance(selected_update, int)
        or selected_update not in config.validation_checkpoints
    ):
        raise ValueError("pre-lock selected update is invalid")

    scaler_references = registry.get("fit_scaler_artifact_by_case")
    if not isinstance(scaler_references, dict) or set(scaler_references) != set(CASES):
        raise ValueError("pre-lock FIT scaler identities are incomplete")
    scalers: dict[str, FaultScalers] = {}
    for case in CASES:
        path = _reference_path(
            artifact_root,
            scaler_references[case],
            expected_kind="fit_scaler",
            expected_identity=case,
        )
        scalers[case] = _load_frozen_scaler(path, case=case, config=config)

    checkpoint_references = registry.get("checkpoint_artifact_by_identity")
    expected_checkpoints = _expected_checkpoint_keys(config, selected_update)
    if not isinstance(checkpoint_references, dict) or set(checkpoint_references) != expected_checkpoints:
        raise ValueError("pre-lock RSSM checkpoint identities are incomplete")
    rssm_models: dict[tuple[str, int, str], object] = {}
    for case in CASES:
        for seed in config.confirmatory_seeds:
            for arm in ARMS:
                identity = f"{case}:seed{seed}:{arm}:u{selected_update:04d}"
                reference = checkpoint_references[identity]
                path = _reference_path(
                    artifact_root,
                    reference,
                    expected_kind="rssm_checkpoint",
                    expected_identity=identity,
                )
                payload = torch.load(path, map_location="cpu", weights_only=False)
                provenance = payload.get("provenance") if isinstance(payload, dict) else None
                if not isinstance(provenance, dict):
                    raise ValueError(f"checkpoint has no validated provenance: {identity}")
                rssm_models[(case, seed, arm)] = load_model_checkpoint(
                    path,
                    config,
                    case=case,
                    model_seed=seed,
                    arm=arm,
                    update=selected_update,
                    expected_checkpoint_sha256=reference["sha256"],
                    expected_provenance=provenance,
                    device="cpu",
                )

    baseline_references = registry.get("baseline_selection_artifact_by_arm")
    if not isinstance(baseline_references, dict) or set(baseline_references) != set(
        COMPETENCE_BASELINES
    ):
        raise ValueError("pre-lock baseline bundle identities are incomplete")
    baseline_models: dict[str, Mapping[str, tuple[object, object]]] = {}
    for baseline in COMPETENCE_BASELINES:
        path = _reference_path(
            artifact_root,
            baseline_references[baseline],
            expected_kind="baseline_validation_selection",
            expected_identity=baseline,
        )
        restored = load_selected_baseline_models(
            path,
            baseline=baseline,
            config=config,
        )
        if set(restored) != _expected_baseline_keys(config, baseline):
            raise ValueError(f"frozen {baseline} model identities are incomplete")
        baseline_models[baseline] = restored

    fault_reference = registry.get("fault_manifest_artifact")
    fault_path = _reference_path(
        artifact_root,
        fault_reference,
        expected_kind="frozen_fault_manifest",
        expected_identity="all_roles",
    )
    fault_contract = load_strict_json(fault_path)
    if not isinstance(fault_contract, dict):
        raise ValueError("frozen fault contract is not a JSON object")

    return FrozenAssets(
        selected_update=selected_update,
        scalers_by_case=MappingProxyType(scalers),
        rssm_models=MappingProxyType(rssm_models),
        baseline_models=MappingProxyType(baseline_models),
        fault_contract=MappingProxyType(fault_contract),
        plan_sha256_by_case=MappingProxyType(
            prelock_plan_sha256_by_case(registry, artifact_root)
        ),
    )


def _locked_source_root(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    if manifest_path != CANONICAL_LOCKED_MANIFEST.resolve():
        raise ValueError("confirmation requires the canonical locked-corpus location")
    if (
        manifest_path.name != "locked_test_all_corpus_manifest.json"
        or manifest_path.parent.name != "manifests"
    ):
        raise ValueError("confirmation requires the canonical all-case locked manifest")
    return manifest_path.parent.parent.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _staging_path(output_dir: Path) -> Path:
    if not output_dir.name:
        raise ValueError("confirmation output must name a directory")
    return output_dir.parent / f".{output_dir.name}.confirmation-staging"


def _preflight_destinations(
    source_root: Path,
    prelock_root: Path,
    output_dir: Path,
    expected_prelock_sha256: str,
) -> tuple[Path, Path, Path, Path]:
    output_dir = output_dir.resolve()
    staging = _staging_path(output_dir)
    forbidden_roots = (source_root, CANONICAL_PRELOCK_ROOT, STATE_ROOT)
    if any(
        _is_within(path, root)
        for path in (output_dir, staging)
        for root in forbidden_roots
    ):
        raise ValueError(
            "confirmation output must be outside source, pre-lock, and locked-state roots"
        )
    if any(os.path.lexists(path) for path in (output_dir, staging)):
        raise FileExistsError("confirmation output or staging destination already exists")
    state_dir = state_dir_for_digest(expected_prelock_sha256)
    started = state_dir / CONFIRMATION_ATTEMPT_MARKER
    failed = state_dir / CONFIRMATION_FAILURE_MARKER
    completed = state_dir / CONFIRMATION_COMPLETION_MARKER
    if any(os.path.lexists(path) for path in (started, failed, completed)):
        raise FileExistsError("the locked corpus already has a confirmation-attempt marker")
    return staging, started, failed, completed


def _locked_dependency_paths(manifest_path: Path, source_root: Path) -> list[str]:
    wrapper = load_strict_json(manifest_path)
    if not isinstance(wrapper, dict) or set(wrapper) != {"manifest_sha256", "manifest"}:
        raise ValueError("locked corpus wrapper fields are invalid")
    manifest = wrapper.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("collection_kind") != "locked_test":
        raise ValueError("evaluation source is not a locked corpus")
    paths = {str(manifest_path.resolve().relative_to(source_root.resolve()))}
    try:
        paths.update(metadata["path"] for metadata in manifest["plans"].values())
        paths.update(
            f"locked_test_raw/{metadata['path']}" for metadata in manifest["files"]
        )
        paths.update(
            f"locked_test_raw/{metadata['path']}"
            for metadata in manifest["receipts"].values()
        )
    except (KeyError, TypeError) as error:
        raise ValueError("locked corpus dependency metadata are incomplete") from error
    if len(paths) != 1 + len(manifest["plans"]) + len(manifest["files"]) + len(
        manifest["receipts"]
    ):
        raise ValueError("locked corpus dependency paths are duplicated")
    for relative in paths:
        _safe_relative_file(source_root, relative)
    return sorted(paths)


def _snapshot_locked_source(
    manifest_path: Path,
    source_root: Path,
    destination: Path,
) -> tuple[Path, list[str]]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("evaluation-source snapshot already exists")
    dependencies = _locked_dependency_paths(manifest_path, source_root)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in dependencies:
        source = _safe_relative_file(source_root, relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_sha256_before = sha256_file(source)
        shutil.copy2(source, target)
        target_sha256 = sha256_file(target)
        source_sha256_after = sha256_file(source)
        if (
            source_sha256_before != source_sha256_after
            or target_sha256 != source_sha256_before
            or os.path.samestat(source.stat(), target.stat())
        ):
            raise RuntimeError(f"evaluation-source snapshot changed bytes: {relative}")
    _fsync_directory(destination)
    snapshot_manifest = destination / str(manifest_path.resolve().relative_to(source_root))
    return snapshot_manifest, dependencies


def _frozen_fault_spec(contract: Mapping[str, object], config: StudyConfig) -> FaultSpec:
    required = {
        "schema",
        "development_corpus_manifest_sha256",
        "spec",
        "signatures_by_role",
        "signature_sha256_by_role",
    }
    if set(contract) != required or contract.get("schema") != FROZEN_FAULT_CONTRACT_SCHEMA:
        raise ValueError("frozen fault contract fields are invalid")
    spec_payload = contract.get("spec")
    signatures = contract.get("signatures_by_role")
    signature_hashes = contract.get("signature_sha256_by_role")
    if not isinstance(spec_payload, dict) or not isinstance(signatures, dict) or not isinstance(
        signature_hashes, dict
    ):
        raise ValueError("frozen fault contract payload is invalid")
    spec = FaultSpec.from_dict(spec_payload)
    if spec.evaluation_horizon != config.direct_horizon:
        raise ValueError("frozen fault contract and H8 study configuration differ")
    expected_rows = [list(row) for row in sorted(fault_cell_signatures(spec, "locked_test"))]
    if (
        signatures.get("locked_test") != expected_rows
        or signature_hashes.get("locked_test") != canonical_sha256(expected_rows)
    ):
        raise ValueError("locked fault signatures differ from the frozen contract")
    return spec


def _sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("confirmation result table is empty")
    missing = [column for column in RESULT_SORT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"confirmation result table is missing identities: {missing}")
    return frame.sort_values(list(RESULT_SORT_COLUMNS), kind="stable").reset_index(drop=True)


def _write_and_reload_result(path: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ordered = _sort_results(frame)
    content = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("ascii")
    _atomic_bytes(path, content)
    persisted = pd.read_csv(path)
    persisted = _sort_results(persisted)
    return persisted, {
        "path": path.name,
        "rows": len(persisted),
        "sha256": sha256_file(path),
        "canonical_frame_sha256": canonical_frame_sha256(persisted),
    }


def _evaluate_all(
    variants_by_case: Mapping[str, Sequence],
    assets: FrozenAssets,
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rssm_frames: list[pd.DataFrame] = []
    for case in CASES:
        variants = variants_by_case[case]
        for seed in config.confirmatory_seeds:
            for arm in ARMS:
                rssm_frames.append(
                    evaluate_model_h8(
                        assets.rssm_models[(case, seed, arm)],
                        variants,
                        assets.scalers_by_case[case],
                        config,
                        arm=arm,
                        case=case,
                        model_seed=seed,
                        update=assets.selected_update,
                        role="locked_test",
                        device="cpu",
                    )
                )

    baseline_frames: list[pd.DataFrame] = []
    for baseline in COMPETENCE_BASELINES:
        models = assets.baseline_models[baseline]
        for identity in sorted(models):
            case, seed_text = identity.split(":")
            seed = int(seed_text.removeprefix("seed"))
            receipt, model = models[identity]
            variants = variants_by_case[case]
            scalers = assets.scalers_by_case[case]
            if baseline == "ridge_arx":
                frame = evaluate_arx_h8(
                    model, variants, scalers, receipt, role="locked_test"
                )
            elif baseline == "direct_h8_ridge":
                frame = evaluate_direct_h8_ridge(
                    model, variants, scalers, receipt, role="locked_test"
                )
            else:
                if seed not in config.confirmatory_seeds:
                    raise ValueError("GRU baseline seed is outside the frozen set")
                frame = evaluate_direct_h8_gru(
                    model,
                    variants,
                    scalers,
                    receipt,
                    role="locked_test",
                    device="cpu",
                )
            baseline_frames.append(frame)
    return (
        pd.concat(rssm_frames, ignore_index=True),
        pd.concat(baseline_frames, ignore_index=True),
    )


def _artifact_inventory(root: Path, *, exclude: Sequence[Path] = ()) -> list[dict]:
    excluded = {path.resolve() for path in exclude}
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    ]


def _peak_memory() -> dict[str, int]:
    return {
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    }


def _publish_staging(staging: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite confirmation output")
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("confirmation staging directory is missing or invalid")
    if staging.parent.resolve() != output_dir.parent.resolve():
        raise ValueError("confirmation staging and output are not rename-compatible")
    os.rename(staging, output_dir)
    _fsync_directory(output_dir.parent)


@threadpool_limits.wrap(limits=1)
def run_confirmation(
    locked_manifest_path: Path,
    prelock_registry_path: Path,
    prelock_artifact_root: Path,
    expected_prelock_sha256: str,
    output_dir: Path,
    *,
    confirmation: str,
) -> Path:
    """Run one immutable confirmation attempt with no tuning surface."""
    if confirmation != CONFIRMATION_TOKEN:
        raise ValueError(f"confirmation requires --confirm {CONFIRMATION_TOKEN}")
    started = time.monotonic()
    stage_times: dict[str, float] = {}
    stage_started = time.monotonic()
    _configure_evaluation_policy()
    config = StudyConfig()
    evaluation_runtime = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    code_manifest = confirmation_scientific_code_manifest()
    code_manifest_sha256 = canonical_sha256(code_manifest)

    locked_manifest_path = locked_manifest_path.resolve()
    source_root = _locked_source_root(locked_manifest_path)
    prelock_registry_path = prelock_registry_path.resolve()
    prelock_artifact_root = prelock_artifact_root.resolve()
    if prelock_registry_path != CANONICAL_PRELOCK_REGISTRY.resolve():
        raise ValueError("confirmation requires the canonical pre-lock registry")
    if prelock_artifact_root != CANONICAL_PRELOCK_BUNDLE.resolve():
        raise ValueError("confirmation requires the canonical pre-lock bundle")
    output_dir = output_dir.resolve()
    staging, attempt_marker, failure_marker, completion_marker = _preflight_destinations(
        source_root,
        prelock_artifact_root,
        output_dir,
        expected_prelock_sha256,
    )

    registry = validate_prelock_bundle(
        prelock_registry_path,
        prelock_artifact_root,
        config,
        expected_prelock_sha256,
    )
    assets = _load_frozen_assets(registry, prelock_artifact_root, config)
    frozen_fault_spec = _frozen_fault_spec(assets.fault_contract, config)
    collection_completion = validate_locked_collection_completion(
        expected_prelock_sha256
    )
    collection_completion_path = (
        state_dir_for_digest(expected_prelock_sha256)
        / COLLECTION_COMPLETION_MARKER
    )
    collection_completion_sha256 = sha256_file(collection_completion_path)
    stage_times["pre_marker_validation_and_asset_instantiation"] = (
        time.monotonic() - stage_started
    )

    staging.mkdir(parents=True, exist_ok=False)
    run_config = {
        "schema": RUN_CONFIG_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "one_shot": True,
        "locked_manifest_path": str(locked_manifest_path),
        "output_path": str(output_dir),
        "prelock_registry_sha256": expected_prelock_sha256,
        "selected_update": assets.selected_update,
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "study_config": config.to_dict(),
        "locked_corpus_manifest_sha256": collection_completion[
            "locked_manifest_payload_sha256"
        ],
        "evaluation_policy": dict(EVALUATION_POLICY),
        "evaluation_runtime": evaluation_runtime,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "scientific_code_sha256_by_path": code_manifest,
        "scientific_code_manifest_sha256": code_manifest_sha256,
    }
    _write_canonical_json(staging / "run_config.json", run_config)

    attempt_payload = {
        "schema": ATTEMPT_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "one_shot": True,
        "started_at_utc": _utc_now(),
        "locked_manifest_path": str(locked_manifest_path),
        "output_path": str(output_dir),
        "prelock_registry_sha256": expected_prelock_sha256,
        "selected_update": assets.selected_update,
        "locked_corpus_manifest_sha256": collection_completion[
            "locked_manifest_payload_sha256"
        ],
        "locked_manifest_file_sha256": collection_completion[
            "locked_manifest_file_sha256"
        ],
        "locked_collection_completion_sha256": collection_completion_sha256,
        "external_freeze_receipt_sha256": collection_completion[
            "external_freeze_receipt_sha256"
        ],
        "runner_sha256": run_config["runner_sha256"],
        "scientific_code_manifest_sha256": code_manifest_sha256,
        "evaluation_policy": dict(EVALUATION_POLICY),
        "evaluation_runtime": evaluation_runtime,
    }
    attempt_payload["attempt_identity_sha256"] = canonical_sha256(attempt_payload)
    if set(attempt_payload) != ATTEMPT_FIELDS:
        raise AssertionError("confirmation-attempt fields differ from the frozen schema")
    marker_created = False
    try:
        _write_json_once(attempt_marker, attempt_payload)
        marker_created = True
        attempt_marker_sha256 = sha256_file(attempt_marker)

        stage_started = time.monotonic()
        original_index, issues = validate_locked_corpus_binding(
            locked_manifest_path,
            prelock_registry=registry,
            expected_prelock_sha256=expected_prelock_sha256,
            expected_plan_sha256_by_case=assets.plan_sha256_by_case,
        )
        if issues or original_index is None:
            raise ValueError("; ".join(issues) or "locked corpus binding failed")
        if (
            original_index.manifest_sha256
            != collection_completion["locked_manifest_payload_sha256"]
        ):
            raise ValueError("locked corpus identity differs from collection completion")
        prevalidated_source_inventory = _locked_dependency_paths(
            locked_manifest_path, source_root
        )
        snapshot_root = staging / "evaluation_source"
        snapshot_manifest, source_inventory = _snapshot_locked_source(
            locked_manifest_path, source_root, snapshot_root
        )
        if source_inventory != prevalidated_source_inventory:
            raise ValueError("locked corpus dependencies changed after the attempt marker")
        index, issues = validate_locked_corpus_binding(
            snapshot_manifest,
            prelock_registry=registry,
            expected_prelock_sha256=expected_prelock_sha256,
            artifact_root=snapshot_root,
            artifact_inventory=source_inventory,
            expected_plan_sha256_by_case=assets.plan_sha256_by_case,
        )
        if issues or index is None:
            raise ValueError("; ".join(issues) or "snapshot locked corpus binding failed")
        if original_index.manifest_sha256 != index.manifest_sha256:
            raise ValueError("locked source snapshot changed the corpus identity")
        attempt_evidence = snapshot_root / ATTEMPT_EVIDENCE_PATH
        shutil.copy2(attempt_marker, attempt_evidence)
        if sha256_file(attempt_evidence) != attempt_marker_sha256:
            raise RuntimeError("confirmation-attempt evidence changed during copying")
        collection_evidence = snapshot_root / COLLECTION_EVIDENCE_PATH
        shutil.copy2(collection_completion_path, collection_evidence)
        if sha256_file(collection_evidence) != collection_completion_sha256:
            raise RuntimeError("locked-collection evidence changed during copying")
        evaluation_inventory = sorted(
            [
                *source_inventory,
                ATTEMPT_EVIDENCE_PATH,
                COLLECTION_EVIDENCE_PATH,
            ]
        )
        stage_times["snapshot_and_revalidate_locked_source"] = (
            time.monotonic() - stage_started
        )

        stage_started = time.monotonic()
        fault_manifest = build_fault_manifest(index, frozen_fault_spec)
        _write_canonical_json(staging / "locked_fault_manifest.json", fault_manifest.payload())
        variants = list(
            iter_role_variants(
                index,
                fault_manifest,
                "locked_test",
                allow_locked_test=True,
            )
        )
        variants_by_case = {
            case: tuple(
                variant
                for variant in variants
                if variant.cell.trajectory.case == case
            )
            for case in CASES
        }
        if any(not variants_by_case[case] for case in CASES):
            raise ValueError("locked fault variants do not cover every frozen case")
        stage_times["single_locked_variant_materialization"] = (
            time.monotonic() - stage_started
        )

        stage_started = time.monotonic()
        rssm_frame, baseline_frame = _evaluate_all(
            variants_by_case, assets, config
        )
        stage_times["complete_h8_evaluation"] = time.monotonic() - stage_started

        stage_started = time.monotonic()
        rssm_frame, rssm_artifact = _write_and_reload_result(
            staging / "rssm_locked_h8.csv", rssm_frame
        )
        baseline_frame, baseline_artifact = _write_and_reload_result(
            staging / "baseline_locked_h8.csv", baseline_frame
        )
        final_runtime = numerical_runtime_fingerprint("cpu", include_sklearn=True)
        final_code_manifest = confirmation_scientific_code_manifest()
        if final_runtime != evaluation_runtime:
            raise RuntimeError("evaluation runtime changed during confirmation")
        if final_code_manifest != code_manifest:
            raise RuntimeError("scientific code changed during confirmation")

        source_manifest_relative = str(snapshot_manifest.relative_to(snapshot_root))
        evaluation_receipt = {
            "schema": study_gate_module.EVALUATION_PROVENANCE_SCHEMA,
            "stage": "confirmation",
            "role": "locked_test",
            "selected_update": assets.selected_update,
            "prelock_registry_sha256": expected_prelock_sha256,
            "study_config_sha256": canonical_sha256(config.to_dict()),
            "confirmation_runner_sha256": sha256_file(Path(__file__).resolve()),
            "scientific_code_sha256_by_path": final_code_manifest,
            "scientific_code_manifest_sha256": canonical_sha256(final_code_manifest),
            "evaluator_sha256": sha256_file(
                Path(study_evaluate_module.__file__).resolve()
            ),
            "gate_sha256": sha256_file(Path(study_gate_module.__file__).resolve()),
            "locked_corpus_manifest_sha256": index.manifest_sha256,
            "locked_fault_manifest_sha256": fault_manifest.sha256,
            "attempt_marker_sha256": attempt_marker_sha256,
            "attempt_marker_artifact": {
                "path": ATTEMPT_EVIDENCE_PATH,
                "sha256": attempt_marker_sha256,
                "kind": "confirmation_attempt_marker",
                "identity": expected_prelock_sha256,
            },
            "locked_collection_completion_artifact": {
                "path": COLLECTION_EVIDENCE_PATH,
                "sha256": collection_completion_sha256,
                "kind": "locked_collection_completion_marker",
                "identity": expected_prelock_sha256,
            },
            "rssm_frame_sha256": canonical_frame_sha256(rssm_frame),
            "baseline_frame_sha256": canonical_frame_sha256(baseline_frame),
            "rssm_result_artifact": rssm_artifact,
            "baseline_result_artifact": baseline_artifact,
            "evaluation_policy": dict(EVALUATION_POLICY),
            "evaluation_runtime": final_runtime,
            "evaluation_source_manifest_artifact": {
                "path": source_manifest_relative,
                "sha256": sha256_file(snapshot_manifest),
                "kind": "locked_evaluation_source_manifest",
                "identity": "locked_test",
            },
            "artifact_inventory": evaluation_inventory,
        }
        if set(evaluation_receipt) != EVALUATION_RECEIPT_FIELDS:
            raise AssertionError("evaluation receipt fields differ from the frozen schema")
        evaluation_receipt_path = staging / "evaluation_receipt.json"
        _write_canonical_json(evaluation_receipt_path, evaluation_receipt)
        persisted_evaluation_receipt = load_strict_json(evaluation_receipt_path)
        if persisted_evaluation_receipt != evaluation_receipt:
            raise RuntimeError("evaluation receipt changed during persistence")
        stage_times["persist_and_bind_results"] = time.monotonic() - stage_started

        stage_started = time.monotonic()
        paired, gate_result = evaluate_study_gate(
            rssm_frame,
            config,
            stage="confirmation",
            selected_update=assets.selected_update,
            baseline_frame=baseline_frame,
            prelock_registry=registry,
            prelock_artifact_root=prelock_artifact_root,
            expected_prelock_sha256=expected_prelock_sha256,
            evaluation_receipt=evaluation_receipt,
            evaluation_artifact_root=snapshot_root,
        )
        write_gate_analysis(staging / "gate", paired, gate_result)
        stage_times["single_confirmatory_gate"] = time.monotonic() - stage_started

        total_wall_seconds = time.monotonic() - started
        wall_time = {
            "schema": "boptest-reliability-confirmation-wall-time-v1",
            "device": "cpu",
            "stage_wall_seconds": stage_times,
            "total_wall_seconds": total_wall_seconds,
            "memory": _peak_memory(),
            "simulator_steps_consumed_by_runner": 0,
            "sealed_corpus_transition_rows": len(index.records) * TRAJECTORY_STEPS,
            "optimizer_updates": 0,
            "rssm_result_rows": len(rssm_frame),
            "baseline_result_rows": len(baseline_frame),
        }
        _write_canonical_json(staging / "wall_time.json", wall_time)

        run_complete_path = staging / "run_complete.json"
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "stage": "confirmation",
            "role": "locked_test",
            "one_shot": True,
            "claim_requires_completion_marker": True,
            "completed_at_utc": _utc_now(),
            "decision": gate_result["decision"],
            "gate_pass": bool(gate_result["gate_pass"]),
            "paper_claim_allowed": bool(gate_result["paper_claim_allowed"]),
            "selected_update": assets.selected_update,
            "prelock_registry_sha256": expected_prelock_sha256,
            "locked_corpus_manifest_sha256": index.manifest_sha256,
            "locked_fault_manifest_sha256": fault_manifest.sha256,
            "attempt_marker_sha256": attempt_marker_sha256,
            "evaluation_receipt_sha256": sha256_file(evaluation_receipt_path),
            "rssm_result": rssm_artifact,
            "baseline_result": baseline_artifact,
            "gate_result_sha256": sha256_file(staging / "gate" / "study_gate.json"),
            "matched_result_sha256": sha256_file(
                staging / "gate" / "matched_h8_arm_metrics.csv"
            ),
            "evaluation_policy": dict(EVALUATION_POLICY),
            "evaluation_runtime": final_runtime,
            "scientific_code_sha256_by_path": final_code_manifest,
            "scientific_code_manifest_sha256": canonical_sha256(final_code_manifest),
            "wall_time": wall_time,
            "artifact_inventory_excludes_this_receipt": _artifact_inventory(
                staging, exclude=(run_complete_path,)
            ),
        }
        if set(receipt) != RUN_RECEIPT_FIELDS:
            raise AssertionError("confirmation receipt fields differ from the frozen schema")
        _write_canonical_json(run_complete_path, receipt)
        _publish_staging(staging, output_dir)

        final_receipt = output_dir / "run_complete.json"
        completion_payload = {
            "schema": COMPLETION_MARKER_SCHEMA,
            "stage": "confirmation",
            "completed_at_utc": _utc_now(),
            "prelock_registry_sha256": expected_prelock_sha256,
            "attempt_marker_sha256": attempt_marker_sha256,
            "output_path": str(output_dir),
            "run_complete_sha256": sha256_file(final_receipt),
            "decision": gate_result["decision"],
            "gate_pass": bool(gate_result["gate_pass"]),
            "paper_claim_allowed": bool(gate_result["paper_claim_allowed"]),
        }
        if set(completion_payload) != COMPLETION_MARKER_FIELDS:
            raise AssertionError("confirmation completion fields differ from schema")
        _write_json_once(completion_marker, completion_payload)
        return final_receipt
    except BaseException as error:
        if marker_created and not completion_marker.exists():
            failure_payload = {
                "schema": FAILURE_SCHEMA,
                "stage": "confirmation",
                "failed_at_utc": _utc_now(),
                "prelock_registry_sha256": expected_prelock_sha256,
                "attempt_marker_sha256": sha256_file(attempt_marker),
                "output_path": str(output_dir),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "output_published": output_dir.is_dir(),
                "retry_allowed": False,
            }
            if set(failure_payload) != FAILURE_MARKER_FIELDS:
                raise AssertionError("confirmation failure fields differ from schema")
            try:
                _write_json_once(failure_marker, failure_payload)
            except BaseException:
                pass
        elif not marker_created:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_confirmation_completion(
    output_dir: Path, expected_prelock_sha256: str
) -> dict:
    """Return a claim-bearing receipt only after both sides of the commit agree."""
    output_dir = output_dir.resolve()
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("confirmation output is not a plain directory")
    state_dir = state_dir_for_digest(expected_prelock_sha256)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise ValueError("digest-scoped confirmation state is not a plain directory")
    attempt_path = state_dir / CONFIRMATION_ATTEMPT_MARKER
    completion_path = state_dir / CONFIRMATION_COMPLETION_MARKER
    failure_path = state_dir / CONFIRMATION_FAILURE_MARKER
    if os.path.lexists(failure_path):
        raise ValueError("confirmation has immutable failure evidence")
    for path, label in (
        (attempt_path, "confirmation attempt marker"),
        (completion_path, "confirmation completion marker"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is missing or is not a plain file")

    attempt = load_strict_json(attempt_path)
    completion = load_strict_json(completion_path)
    receipt_path = output_dir / "run_complete.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("published confirmation receipt is missing or invalid")
    receipt = load_strict_json(receipt_path)
    if not isinstance(attempt, dict) or set(attempt) != ATTEMPT_FIELDS:
        raise ValueError("confirmation attempt marker fields are invalid")
    if not isinstance(completion, dict) or set(completion) != COMPLETION_MARKER_FIELDS:
        raise ValueError("confirmation completion marker fields are invalid")
    if not isinstance(receipt, dict) or set(receipt) != RUN_RECEIPT_FIELDS:
        raise ValueError("published confirmation receipt fields are invalid")

    attempt_sha256 = sha256_file(attempt_path)
    receipt_sha256 = sha256_file(receipt_path)
    expected_completion = {
        "schema": COMPLETION_MARKER_SCHEMA,
        "stage": "confirmation",
        "prelock_registry_sha256": expected_prelock_sha256,
        "attempt_marker_sha256": attempt_sha256,
        "output_path": str(output_dir),
        "run_complete_sha256": receipt_sha256,
        "decision": receipt.get("decision"),
        "gate_pass": receipt.get("gate_pass"),
        "paper_claim_allowed": receipt.get("paper_claim_allowed"),
    }
    for field, expected in expected_completion.items():
        if completion.get(field) != expected:
            raise ValueError(f"confirmation completion {field} is invalid")
    expected_receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "one_shot": True,
        "claim_requires_completion_marker": True,
        "prelock_registry_sha256": expected_prelock_sha256,
        "attempt_marker_sha256": attempt_sha256,
    }
    for field, expected in expected_receipt.items():
        if receipt.get(field) != expected:
            raise ValueError(f"published confirmation receipt {field} is invalid")
    if receipt.get("paper_claim_allowed") is not receipt.get("gate_pass"):
        raise ValueError("published confirmation claim and gate flags differ")
    if receipt.get("paper_claim_allowed") != (receipt.get("decision") == "PASS"):
        raise ValueError("published confirmation decision and claim flag differ")

    evaluation_path = output_dir / "evaluation_receipt.json"
    gate_path = output_dir / "gate" / "study_gate.json"
    matched_path = output_dir / "gate" / "matched_h8_arm_metrics.csv"
    for path, expected_sha256, label in (
        (
            evaluation_path,
            receipt.get("evaluation_receipt_sha256"),
            "evaluation receipt",
        ),
        (gate_path, receipt.get("gate_result_sha256"), "gate result"),
        (matched_path, receipt.get("matched_result_sha256"), "matched result"),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"published {label} does not match the final receipt")
    gate = load_strict_json(gate_path)
    if not isinstance(gate, dict) or any(
        gate.get(field) != receipt.get(field)
        for field in ("decision", "gate_pass", "paper_claim_allowed")
    ):
        raise ValueError("published gate decision differs from the final receipt")

    inventory = receipt.get("artifact_inventory_excludes_this_receipt")
    if not isinstance(inventory, list):
        raise ValueError("published confirmation inventory is invalid")
    declared: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("published confirmation inventory entry is invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or relative in declared
        ):
            raise ValueError("published confirmation inventory path is invalid")
        try:
            path = _safe_relative_file(output_dir, relative)
        except (OSError, ValueError) as error:
            raise ValueError("published confirmation inventory path is invalid") from error
        if (
            path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise ValueError("published confirmation inventory artifact differs")
        declared.add(relative)
    actual = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if actual != declared | {"run_complete.json"}:
        raise ValueError("published confirmation inventory is incomplete")
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen one-shot multi-case locked confirmation"
    )
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--prelock-registry", type=Path, required=True)
    parser.add_argument("--prelock-artifact-root", type=Path, required=True)
    parser.add_argument("--expected-prelock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    receipt_path = run_confirmation(
        args.locked_manifest,
        args.prelock_registry,
        args.prelock_artifact_root,
        args.expected_prelock_sha256,
        args.output,
        confirmation=args.confirm,
    )
    receipt = validate_confirmation_completion(
        receipt_path.parent, args.expected_prelock_sha256
    )
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "gate_pass": receipt["gate_pass"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
