from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import (
    evaluate_arx_h8,
    evaluate_direct_h8_gru,
    evaluate_direct_h8_ridge,
    fit_arx_ridge,
    fit_direct_h8_gru,
    fit_direct_h8_ridge,
)
from .fault_data import (
    CorpusIndex,
    FaultScalers,
    FaultVariant,
    build_fault_manifest,
    iter_role_variants,
    load_corpus_index,
)
from .protocol import CASES, TRAJECTORY_STEPS, sha256_file
from .provenance import (
    canonical_sha256,
    load_selected_baseline_models,
    load_strict_json,
    write_baseline_selection_bundle,
    write_frozen_fault_contract,
    write_validation_selection,
)
from .runtime_provenance import (
    numerical_runtime_fingerprint,
    validate_current_numerical_runtime_fingerprint,
)
from .study_config import ARMS, StudyConfig
from .study_evaluate import (
    evaluate_model_h8,
    load_model_checkpoint,
    validation_selection_scores,
)
from .study_gate import (
    canonical_frame_sha256,
    evaluate_study_gate,
    write_gate_analysis,
)
from .study_train import (
    make_training_schedule,
    prepare_case_training_data,
    schedule_payload,
    train_case_seed,
    training_provenance,
)


RUNNER_SCHEMA = "boptest-reliability-development-run-v1"
RUN_RECEIPT_SCHEMA = "boptest-reliability-development-receipt-v1"
STAGING_SCHEMA = "boptest-reliability-development-staging-v1"
STAGING_MARKER = "staging_identity.json"
RUN_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "interpretation",
        "scientific_screen_enabled",
        "device",
        "corpus_manifest_sha256",
        "corpus_manifest_file_sha256",
        "runner_sha256",
        "scientific_code_sha256_by_path",
        "scientific_code_manifest_sha256",
        "numerical_runtime",
        "study_config_sha256",
        "study_config",
        "resume_requested",
        "staging_identity_sha256",
    }
)
RUN_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "interpretation",
        "scientific_screen_reported",
        "decision",
        "selected_update",
        "corpus_manifest_sha256",
        "corpus_manifest_file_sha256",
        "fault_manifest_sha256",
        "study_config_sha256",
        "runner_sha256",
        "scientific_code_sha256_by_path",
        "scientific_code_manifest_sha256",
        "numerical_runtime",
        "staging_identity_sha256",
        "rssm_result",
        "baseline_result",
        "gate_artifacts",
        "validation_selection_sha256",
        "baseline_bundle_sha256",
        "wall_time",
        "artifact_inventory_excludes_this_receipt",
    }
)
SCIENTIFIC_CODE_FILES = (
    "multicase_fault_benchmark/__init__.py",
    "multicase_fault_benchmark/STUDY_PROTOCOL.md",
    "multicase_fault_benchmark/protocol.py",
    "multicase_fault_benchmark/collect.py",
    "multicase_fault_benchmark/worker_collect.py",
    "multicase_fault_benchmark/study_config.py",
    "multicase_fault_benchmark/study_train.py",
    "multicase_fault_benchmark/reliability_model.py",
    "multicase_fault_benchmark/reliability_loss.py",
    "health_rssm/__init__.py",
    "health_rssm/model.py",
    "health_rssm/planning.py",
    "health_rssm/training.py",
    "multicase_fault_benchmark/fault_data.py",
    "multicase_fault_benchmark/baselines.py",
    "multicase_fault_benchmark/study_evaluate.py",
    "multicase_fault_benchmark/study_gate.py",
    "multicase_fault_benchmark/provenance.py",
    "multicase_fault_benchmark/runtime_provenance.py",
    "multicase_fault_benchmark/study_development.py",
    "multicase_fault_benchmark/study_prelock.py",
    "multicase_fault_benchmark/locked_state.py",
    "multicase_fault_benchmark/study_locked_collection.py",
    "multicase_fault_benchmark/study_confirmation.py",
)
RECOMPUTABLE_TOP_LEVEL = (
    "run_config.json",
    "frozen",
    "training_receipts.json",
    "rssm_validation_h8.csv",
    "baseline_selection_receipts.json",
    "baseline_selection_scores.csv",
    "baseline_gru_training_log.csv",
    "baseline_validation_h8.csv",
    "gate",
    "wall_time.json",
    "run_complete.json",
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


def integration_study_config() -> StudyConfig:
    """Return the only CLI-exposed reduced config, for integration tests only."""
    return replace(
        StudyConfig(),
        updates=1,
        checkpoint_every=1,
        validation_checkpoints=(1,),
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_canonical_json(path: Path, payload: object) -> None:
    content = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _atomic_bytes(path, content)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_value(payload: object) -> object:
    return json.loads(json.dumps(payload, allow_nan=False))


def _staging_path(output_dir: Path) -> Path:
    if not output_dir.name:
        raise ValueError("development output must name a directory")
    return output_dir.parent / f".{output_dir.name}.development-staging"


def scientific_code_manifest() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent.parent
    paths = {relative: package_root / relative for relative in SCIENTIFIC_CODE_FILES}
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"scientific code manifest is missing files: {missing}")
    return {relative: sha256_file(path) for relative, path in paths.items()}


def _staging_identity(
    *,
    output_dir: Path,
    manifest_path: Path,
    index: CorpusIndex,
    config: StudyConfig,
    device: torch.device,
    integration_only: bool,
) -> dict:
    manifest_path = manifest_path.resolve()
    code_manifest = scientific_code_manifest()
    runtime_fingerprint = numerical_runtime_fingerprint(
        device, include_sklearn=True
    )
    payload = {
        "schema": STAGING_SCHEMA,
        "stage": "development",
        "final_output_path": str(output_dir.resolve()),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "corpus_manifest_sha256": index.manifest_sha256,
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "scientific_code_sha256_by_path": code_manifest,
        "scientific_code_manifest_sha256": canonical_sha256(code_manifest),
        "numerical_runtime": runtime_fingerprint,
        "device": str(device),
        "integration_only": integration_only,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _validate_staging_topology(staging_dir: Path) -> None:
    allowed = {STAGING_MARKER, "training", *RECOMPUTABLE_TOP_LEVEL}
    for path in staging_dir.iterdir():
        if path.name not in allowed:
            raise ValueError(f"staging directory contains an unknown artifact: {path.name}")
        if path.is_symlink():
            raise ValueError(f"staging artifact is a symbolic link: {path.name}")
    marker = staging_dir / STAGING_MARKER
    if not marker.is_file():
        raise ValueError("staging identity marker is missing or not a file")
    training = staging_dir / "training"
    if training.exists() and not training.is_dir():
        raise ValueError("staging training artifact is not a directory")
    for name in RECOMPUTABLE_TOP_LEVEL:
        path = staging_dir / name
        if not path.exists():
            continue
        expected_directory = name in {"frozen", "gate"}
        if expected_directory and not path.is_dir():
            raise ValueError(f"recomputable staging artifact has the wrong type: {name}")
        if not expected_directory and not path.is_file():
            raise ValueError(f"recomputable staging artifact has the wrong type: {name}")


def _clear_recomputable_stages(staging_dir: Path) -> None:
    for name in RECOMPUTABLE_TOP_LEVEL:
        path = staging_dir / name
        if not path.exists():
            continue
        if path.is_symlink():
            raise ValueError(f"refusing to clear a symbolic-link artifact: {name}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _prepare_staging(
    *,
    output_dir: Path,
    identity: dict,
    resume: bool,
) -> Path:
    staging_dir = _staging_path(output_dir)
    if staging_dir.is_symlink():
        raise ValueError("development staging path is a symbolic link")
    if resume:
        if not staging_dir.is_dir():
            raise FileNotFoundError(
                f"resume requested but staging directory is missing: {staging_dir}"
            )
        _validate_staging_topology(staging_dir)
        recorded = load_strict_json(staging_dir / STAGING_MARKER)
        if not isinstance(recorded, dict):
            raise ValueError("staging identity marker is not a JSON object")
        validate_current_numerical_runtime_fingerprint(
            recorded.get("numerical_runtime"), include_sklearn=True
        )
        if recorded != identity:
            raise ValueError(
                "staging identity differs from the manifest/config/runner/device request"
            )
        _clear_recomputable_stages(staging_dir)
    else:
        if staging_dir.exists():
            raise FileExistsError(
                f"staging directory already exists; inspect it and rerun with --resume: "
                f"{staging_dir}"
            )
        staging_dir.mkdir(parents=True, exist_ok=False)
        _write_canonical_json(staging_dir / STAGING_MARKER, identity)
    return staging_dir


def _sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("development result table is empty")
    missing = [column for column in RESULT_SORT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"development result table is missing identities: {missing}")
    return frame.sort_values(list(RESULT_SORT_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _write_result_csv(path: Path, frame: pd.DataFrame) -> dict:
    frame = _sort_results(frame)
    _atomic_bytes(path, frame.to_csv(index=False, lineterminator="\n").encode("ascii"))
    return {
        "path": path.name,
        "rows": len(frame),
        "sha256": sha256_file(path),
        "canonical_frame_sha256": canonical_frame_sha256(frame),
    }


def _development_gate_artifacts(output_dir: Path) -> dict[str, dict]:
    paired_path = output_dir / "gate" / "matched_h8_arm_metrics.csv"
    result_path = output_dir / "gate" / "study_gate.json"
    paired = pd.read_csv(paired_path)
    result = load_strict_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("persisted development gate result is not a JSON object")
    return {
        "matched_metrics": {
            "path": str(paired_path.relative_to(output_dir)),
            "rows": len(paired),
            "sha256": sha256_file(paired_path),
            "canonical_frame_sha256": canonical_frame_sha256(paired),
        },
        "gate_result": {
            "path": str(result_path.relative_to(output_dir)),
            "sha256": sha256_file(result_path),
            "canonical_json_sha256": canonical_sha256(result),
        },
    }


def _validate_development_index(index: CorpusIndex) -> None:
    if index.collection_kind != "development":
        raise ValueError("development runner requires a development corpus")
    if set(index.allowed_roles) != {"fit", "validation"}:
        raise ValueError("development corpus roles must be exactly FIT and validation")
    if index.prelock_registry_sha256 is not None:
        raise ValueError("development corpus must not carry a locked-test binding")
    if {record.key.case for record in index.records} != set(CASES):
        raise ValueError("development corpus must contain every frozen BOPTEST case")
    counts = {
        (case, role): sum(
            record.key.case == case and record.key.role == role
            for record in index.records
        )
        for case in CASES
        for role in ("fit", "validation")
    }
    expected = {
        (case, role): count
        for case in CASES
        for role, count in (("fit", 20), ("validation", 8))
    }
    if counts != expected:
        raise ValueError("development corpus does not contain the frozen 20/8 split")


def _resolved_config(
    *, integration_only: bool, integration_config: StudyConfig | None
) -> StudyConfig:
    if integration_only:
        return (
            integration_study_config()
            if integration_config is None
            else integration_config
        )
    if integration_config is not None:
        raise ValueError("a custom config is allowed only with integration_only=True")
    config = StudyConfig()
    if config != StudyConfig():
        raise AssertionError("production development config is not frozen")
    return config


def _read_completion(
    path: Path, *, case: str, seed: int, config: StudyConfig
) -> dict:
    payload = load_strict_json(path)
    expected = {
        "schema": "boptest-reliability-rssm-training-complete-v2",
        "case": case,
        "model_seed": seed,
        "arms": list(ARMS),
        "updates": config.updates,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"training completion identity differs for {case}:seed{seed}")
    expected_checkpoints = {
        f"{arm}_u{update:04d}.pt"
        for arm in ARMS
        for update in config.validation_checkpoints
    }
    hashes = payload.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_checkpoints:
        raise ValueError(
            f"training completion checkpoint grid is incomplete for {case}:seed{seed}"
        )
    if not isinstance(payload.get("provenance"), dict):
        raise ValueError(f"training completion provenance is missing for {case}:seed{seed}")
    return payload


def _validate_training_tree(training_dir: Path, config: StudyConfig) -> None:
    if not training_dir.exists():
        return
    if training_dir.is_symlink() or not training_dir.is_dir():
        raise ValueError("staging training root is not a plain directory")
    expected_seeds = {f"seed{seed}" for seed in config.development_seeds}
    for case_path in training_dir.iterdir():
        if case_path.is_symlink() or not case_path.is_dir() or case_path.name not in CASES:
            raise ValueError(
                f"staging training root contains an unknown case artifact: {case_path.name}"
            )
        children = tuple(case_path.iterdir())
        if not children:
            raise ValueError(f"staging training case directory is empty: {case_path.name}")
        for seed_path in children:
            if (
                seed_path.is_symlink()
                or not seed_path.is_dir()
                or seed_path.name not in expected_seeds
            ):
                raise ValueError(
                    f"staging training case contains an unknown seed artifact: {seed_path}"
                )


def _validate_training_unit(
    run_dir: Path,
    *,
    index: CorpusIndex,
    fault_manifest,
    variants: list[FaultVariant],
    scalers: FaultScalers,
    config: StudyConfig,
    case: str,
    seed: int,
    device: torch.device,
) -> tuple[Path, dict]:
    """Validate one reusable unit from sources through every checkpoint tensor."""
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"training unit is not a plain directory: {run_dir}")
    expected_checkpoint_names = {
        f"{arm}_u{update:04d}.pt"
        for arm in ARMS
        for update in config.validation_checkpoints
    }
    expected_root_files = {
        "config.json",
        "fault_manifest.json",
        "fit_scalers.json",
        "training_schedule.json",
        "initial_state_hashes.json",
        "training_log.csv",
        "checkpoint_hashes.json",
        "training_complete.json",
    }
    root_children = tuple(run_dir.iterdir())
    if any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in root_children
    ):
        raise ValueError(f"training unit contains a non-plain artifact: {run_dir}")
    actual_root_files = {
        path.name for path in root_children if path.is_file()
    }
    actual_root_directories = {
        path.name for path in root_children if path.is_dir()
    }
    if actual_root_files != expected_root_files or actual_root_directories != {"checkpoints"}:
        raise ValueError(f"training unit inventory is incomplete or ambiguous: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_children = tuple(checkpoint_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in checkpoint_children):
        raise ValueError(f"training checkpoint directory is ambiguous: {checkpoint_dir}")
    if {path.name for path in checkpoint_children} != expected_checkpoint_names:
        raise ValueError(f"training checkpoint file grid is incomplete: {run_dir}")

    schedule = make_training_schedule(variants, config, case=case, model_seed=seed)
    expected_schedule = schedule_payload(schedule, variants)
    expected_provenance = training_provenance(
        index,
        fault_manifest,
        scalers,
        config,
        expected_schedule,
        device=device,
    )
    exact_json = {
        "config.json": config.to_dict(),
        "fault_manifest.json": fault_manifest.payload(),
        "fit_scalers.json": asdict(scalers),
        "training_schedule.json": expected_schedule,
    }
    for filename, expected in exact_json.items():
        if load_strict_json(run_dir / filename) != _json_value(expected):
            raise ValueError(f"reused training {filename} differs for {case}:seed{seed}")

    completion_path = run_dir / "training_complete.json"
    completion = _read_completion(
        completion_path, case=case, seed=seed, config=config
    )
    required_completion_fields = {
        "schema",
        "case",
        "model_seed",
        "arms",
        "updates",
        "wall_seconds",
        "device",
        "provenance",
        "initial_state_sha256",
        "training_log_sha256",
        "checkpoint_sha256",
        "checkpoint_core_state_sha256",
    }
    if set(completion) != required_completion_fields:
        raise ValueError(f"training completion fields are ambiguous for {case}:seed{seed}")
    recorded_provenance = completion.get("provenance")
    if not isinstance(recorded_provenance, dict):
        raise ValueError(f"training provenance is invalid for {case}:seed{seed}")
    validate_current_numerical_runtime_fingerprint(
        recorded_provenance.get("runtime"), include_sklearn=False
    )
    if completion["device"] != str(device) or completion["provenance"] != expected_provenance:
        raise ValueError(f"training completion binding differs for {case}:seed{seed}")
    wall_seconds = completion["wall_seconds"]
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0
    ):
        raise ValueError(f"training completion wall time is invalid for {case}:seed{seed}")

    initial_hashes = load_strict_json(run_dir / "initial_state_hashes.json")
    if (
        not isinstance(initial_hashes, dict)
        or set(initial_hashes) != set(ARMS)
        or any(not _is_sha256(value) for value in initial_hashes.values())
        or len(set(initial_hashes.values())) != 1
        or completion["initial_state_sha256"] != next(iter(initial_hashes.values()))
    ):
        raise ValueError(f"training initial-state hashes differ for {case}:seed{seed}")

    training_log_path = run_dir / "training_log.csv"
    if sha256_file(training_log_path) != completion["training_log_sha256"]:
        raise ValueError(f"training log hash differs for {case}:seed{seed}")
    log = pd.read_csv(training_log_path)
    expected_log_columns = (
        "update",
        "arm",
        "total",
        "observation_nll",
        "health_ce",
        "kl",
        "latent_overshooting_kl",
        "direct_h8",
        "gradient_norm",
    )
    if tuple(log.columns) != expected_log_columns or len(log) != config.updates * len(ARMS):
        raise ValueError(f"training log grid differs for {case}:seed{seed}")
    numeric = log.drop(columns="arm").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"training log contains non-finite values for {case}:seed{seed}")
    for update, rows in log.groupby("update", sort=False):
        if (
            isinstance(update, bool)
            or int(update) != update
            or int(update) not in range(1, config.updates + 1)
            or len(rows) != len(ARMS)
            or set(rows["arm"]) != set(ARMS)
        ):
            raise ValueError(f"training log identities differ for {case}:seed{seed}")
    if set(log["update"].astype(int)) != set(range(1, config.updates + 1)):
        raise ValueError(f"training log update coverage differs for {case}:seed{seed}")

    checkpoint_hashes = load_strict_json(run_dir / "checkpoint_hashes.json")
    if checkpoint_hashes != completion["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash receipt differs for {case}:seed{seed}")
    core_hashes = completion["checkpoint_core_state_sha256"]
    if (
        not isinstance(core_hashes, dict)
        or set(core_hashes) != expected_checkpoint_names
        or any(not _is_sha256(value) for value in core_hashes.values())
    ):
        raise ValueError(f"checkpoint core-hash grid differs for {case}:seed{seed}")
    for update in config.validation_checkpoints:
        for arm in ARMS:
            name = f"{arm}_u{update:04d}.pt"
            checkpoint_path = checkpoint_dir / name
            expected_hash = completion["checkpoint_sha256"].get(name)
            if not _is_sha256(expected_hash) or sha256_file(checkpoint_path) != expected_hash:
                raise ValueError(f"checkpoint file hash differs for {case}:seed{seed}:{name}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("core_state_sha256") != core_hashes[name]
            ):
                raise ValueError(f"checkpoint core receipt differs for {case}:seed{seed}:{name}")
            load_model_checkpoint(
                checkpoint_path,
                config,
                case=case,
                model_seed=seed,
                arm=arm,
                update=update,
                expected_checkpoint_sha256=expected_hash,
                expected_provenance=expected_provenance,
                device="cpu",
            )
    return completion_path, completion


def validate_training_unit(*args, **kwargs) -> tuple[Path, dict]:
    """Public strict validator for development and post-screen training reuse."""
    return _validate_training_unit(*args, **kwargs)


def _evaluate_checkpoint(
    *,
    completion_path: Path,
    completion: dict,
    variants: list[FaultVariant],
    scalers: FaultScalers,
    config: StudyConfig,
    case: str,
    seed: int,
    arm: str,
    update: int,
    device: torch.device,
) -> pd.DataFrame:
    checkpoint = completion_path.parent / "checkpoints" / f"{arm}_u{update:04d}.pt"
    model = load_model_checkpoint(
        checkpoint,
        config,
        case=case,
        model_seed=seed,
        arm=arm,
        update=update,
        expected_checkpoint_sha256=completion["checkpoint_sha256"][checkpoint.name],
        expected_provenance=completion["provenance"],
        device=device,
    )
    return evaluate_model_h8(
        model,
        variants,
        scalers,
        config,
        arm=arm,
        case=case,
        model_seed=seed,
        update=update,
        role="validation",
        device=device,
    )


def _artifact_inventory(root: Path, *, exclude: tuple[Path, ...] = ()) -> list[dict]:
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


def _peak_memory(device: torch.device) -> dict:
    # Linux reports ru_maxrss in KiB. It is a process peak, not an instantaneous sample.
    result = {
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def _publish_staging(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite development output: {output_dir}")
    if staging_dir.is_symlink() or not staging_dir.is_dir():
        raise ValueError("development staging directory disappeared before publication")
    if staging_dir.parent.resolve() != output_dir.parent.resolve():
        raise ValueError("development staging and final output are not rename-compatible")
    os.rename(staging_dir, output_dir)
    directory_fd = os.open(output_dir.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mask_integration_gate(result: dict) -> dict:
    """Remove the screening verdict when the frozen training budget was not used."""
    masked = json.loads(json.dumps(result, allow_nan=False))
    masked["decision"] = "INTEGRATION_ONLY"
    masked["paper_claim_allowed"] = False
    masked["gate_pass"] = False
    masked["confirmatory_conditions_evaluable"] = False
    masked["integration_only"] = {
        "scientific_verdict_reported": False,
        "reason": "reduced non-protocol config used only to exercise the pipeline",
    }
    screen = masked.get("development_screen", {})
    screen["paper_claim_allowed"] = False
    screen["evaluable"] = False
    screen["screen_pass"] = None
    screen["checks"] = {key: None for key in screen.get("checks", {})}
    masked["development_screen"] = screen
    masked["checks"] = {key: None for key in masked.get("checks", {})}
    return masked


def run_development(
    manifest_path: Path,
    output_dir: Path,
    *,
    device: torch.device | str = "cpu",
    integration_only: bool = False,
    integration_config: StudyConfig | None = None,
    resume: bool = False,
) -> Path:
    """Run the complete sealed FIT/validation study without opening locked data."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite development output: {output_dir}")
    config = _resolved_config(
        integration_only=integration_only,
        integration_config=integration_config,
    )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    index = load_corpus_index(manifest_path)
    _validate_development_index(index)
    final_output_dir = output_dir
    identity = _staging_identity(
        output_dir=final_output_dir,
        manifest_path=manifest_path,
        index=index,
        config=config,
        device=device,
        integration_only=integration_only,
    )
    output_dir = _prepare_staging(
        output_dir=final_output_dir,
        identity=identity,
        resume=resume,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.monotonic()
    stage_times: dict[str, float] = {}
    runner_path = Path(__file__).resolve()
    run_config = {
        "schema": RUNNER_SCHEMA,
        "stage": "development",
        "interpretation": (
            "integration_only_not_a_scientific_result"
            if integration_only
            else "frozen_development_screen"
        ),
        "scientific_screen_enabled": not integration_only,
        "device": str(device),
        "corpus_manifest_sha256": index.manifest_sha256,
        "corpus_manifest_file_sha256": identity["manifest_file_sha256"],
        "runner_sha256": sha256_file(runner_path),
        "scientific_code_sha256_by_path": identity[
            "scientific_code_sha256_by_path"
        ],
        "scientific_code_manifest_sha256": identity[
            "scientific_code_manifest_sha256"
        ],
        "numerical_runtime": identity["numerical_runtime"],
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "study_config": config.to_dict(),
        "resume_requested": resume,
        "staging_identity_sha256": identity["identity_sha256"],
    }
    if set(run_config) != RUN_CONFIG_FIELDS:
        raise AssertionError("development run_config fields differ from runner contract")
    _write_canonical_json(output_dir / "run_config.json", run_config)

    stage_started = time.monotonic()
    fault_manifest = build_fault_manifest(index)
    frozen_dir = output_dir / "frozen"
    write_frozen_fault_contract(
        frozen_dir / "frozen_fault_contract.json", index, config
    )
    variants_by_case: dict[str, dict[str, list[FaultVariant]]] = {}
    scalers_by_case: dict[str, FaultScalers] = {}
    for case in CASES:
        fit_variants, scalers = prepare_case_training_data(index, fault_manifest, case)
        validation_variants = list(
            iter_role_variants(
                index, fault_manifest, "validation", cases=(case,)
            )
        )
        variants_by_case[case] = {
            "fit": fit_variants,
            "validation": validation_variants,
        }
        scalers_by_case[case] = scalers
        _write_canonical_json(
            frozen_dir / "fit_scalers" / f"{case}.json", asdict(scalers)
        )
    stage_times["load_and_freeze_inputs"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    completions: dict[tuple[str, int], tuple[Path, dict]] = {}
    training_receipts: dict[str, dict] = {}
    resumed_training_units: list[str] = []
    new_training_units: list[str] = []
    _validate_training_tree(output_dir / "training", config)
    for case in CASES:
        for seed in config.development_seeds:
            identity_text = f"{case}:seed{seed}"
            run_dir = output_dir / "training" / case / f"seed{seed}"
            if run_dir.exists():
                completion_path, completion = _validate_training_unit(
                    run_dir,
                    index=index,
                    fault_manifest=fault_manifest,
                    variants=variants_by_case[case]["fit"],
                    scalers=scalers_by_case[case],
                    config=config,
                    case=case,
                    seed=seed,
                    device=device,
                )
                resumed_training_units.append(identity_text)
            else:
                completion_path = train_case_seed(
                    index,
                    case=case,
                    model_seed=seed,
                    output_dir=output_dir / "training",
                    config=config,
                    arms=ARMS,
                    device=device,
                )
                completion_path, completion = _validate_training_unit(
                    completion_path.parent,
                    index=index,
                    fault_manifest=fault_manifest,
                    variants=variants_by_case[case]["fit"],
                    scalers=scalers_by_case[case],
                    config=config,
                    case=case,
                    seed=seed,
                    device=device,
                )
                new_training_units.append(identity_text)
            if completion_path != run_dir / "training_complete.json":
                raise AssertionError("training completion path differs from its unit")
            if completion["device"] != str(device):
                raise AssertionError("validated training unit changed its device binding")
            completions[(case, seed)] = (completion_path, completion)
            training_receipts[identity_text] = {
                "path": str(completion_path.relative_to(output_dir)),
                "sha256": sha256_file(completion_path),
                "wall_seconds": float(completion["wall_seconds"]),
                "training_log_sha256": completion["training_log_sha256"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
                "provenance": completion["provenance"],
            }
    _write_canonical_json(output_dir / "training_receipts.json", training_receipts)
    stage_times["rssm_training"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    selection_frames = []
    for case in CASES:
        for seed in config.development_seeds:
            completion_path, completion = completions[(case, seed)]
            for update in config.validation_checkpoints:
                for arm in ("ungated_h8", "gated_h8"):
                    selection_frames.append(
                        _evaluate_checkpoint(
                            completion_path=completion_path,
                            completion=completion,
                            variants=variants_by_case[case]["validation"],
                            scalers=scalers_by_case[case],
                            config=config,
                            case=case,
                            seed=seed,
                            arm=arm,
                            update=update,
                            device=device,
                        )
                    )
    selection_frame = _sort_results(pd.concat(selection_frames, ignore_index=True))
    scores = validation_selection_scores(selection_frame, config)
    selected_rows = scores.loc[scores["selected"]]
    if len(selected_rows) != 1:
        raise AssertionError("validation selection did not produce one common update")
    selected_update = int(selected_rows.iloc[0]["update"])
    selection_payload = write_validation_selection(
        frozen_dir / "validation_selection.json",
        selection_frame,
        config,
        validation_rows_path=frozen_dir / "validation_selection_rows.csv",
    )
    if selection_payload["selected_update"] != selected_update:
        raise AssertionError("persisted validation selection changed the selected update")
    stage_times["checkpoint_selection"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    selected_primary = selection_frame.loc[
        selection_frame["update"] == selected_update
    ].copy()
    rssm_frames = [selected_primary]
    for case in CASES:
        for seed in config.development_seeds:
            completion_path, completion = completions[(case, seed)]
            for arm in ("legacy", "aux_h8", "huber_h8"):
                rssm_frames.append(
                    _evaluate_checkpoint(
                        completion_path=completion_path,
                        completion=completion,
                        variants=variants_by_case[case]["validation"],
                        scalers=scalers_by_case[case],
                        config=config,
                        case=case,
                        seed=seed,
                        arm=arm,
                        update=selected_update,
                        device=device,
                    )
                )
    rssm_frame = _sort_results(pd.concat(rssm_frames, ignore_index=True))
    rssm_result_path = output_dir / "rssm_validation_h8.csv"
    rssm_artifact = _write_result_csv(rssm_result_path, rssm_frame)
    stage_times["selected_rssm_evaluation"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    baseline_entries: dict[str, dict] = {
        "ridge_arx": {},
        "direct_h8_ridge": {},
        "deterministic_gru": {},
    }
    baseline_score_frames = []
    gru_training_logs = []
    baseline_receipts: dict[str, dict] = {}
    for case in CASES:
        fit_variants = variants_by_case[case]["fit"]
        validation_variants = variants_by_case[case]["validation"]
        scalers = scalers_by_case[case]

        arx_model, arx_scores, arx_receipt = fit_arx_ridge(
            fit_variants, validation_variants, scalers
        )
        direct_model, direct_scores, direct_receipt = fit_direct_h8_ridge(
            fit_variants, validation_variants, scalers
        )
        for baseline, model, table, receipt in (
            ("ridge_arx", arx_model, arx_scores, arx_receipt),
            ("direct_h8_ridge", direct_model, direct_scores, direct_receipt),
        ):
            baseline_identity = f"{case}:seed0"
            baseline_entries[baseline][baseline_identity] = (receipt, model, table)
            baseline_receipts[f"{baseline}:{baseline_identity}"] = receipt.payload
            baseline_score_frames.append(
                table.assign(baseline=baseline, case=case, model_seed=0)
            )

        for seed in config.development_seeds:
            fit_result = fit_direct_h8_gru(
                fit_variants,
                validation_variants,
                scalers,
                config,
                model_seed=seed,
                device=device,
            )
            baseline_identity = f"{case}:seed{seed}"
            baseline_entries["deterministic_gru"][baseline_identity] = (
                fit_result.receipt,
                fit_result.model,
                fit_result.score_table,
            )
            baseline_receipts[f"deterministic_gru:{baseline_identity}"] = (
                fit_result.receipt.payload
            )
            baseline_score_frames.append(
                fit_result.score_table.assign(
                    baseline="deterministic_gru",
                    case=case,
                    model_seed=seed,
                )
            )
            gru_training_logs.append(
                fit_result.training_log.assign(case=case, model_seed=seed)
            )

    bundle_paths: dict[str, Path] = {}
    for baseline, entries in baseline_entries.items():
        path = frozen_dir / "baselines" / f"{baseline}.pt"
        write_baseline_selection_bundle(
            path, baseline=baseline, entries=entries
        )
        bundle_paths[baseline] = path
    _write_canonical_json(
        output_dir / "baseline_selection_receipts.json", baseline_receipts
    )
    score_frame = pd.concat(baseline_score_frames, ignore_index=True).sort_values(
        ["baseline", "case", "model_seed", "validation_score"], kind="stable"
    )
    _atomic_bytes(
        output_dir / "baseline_selection_scores.csv",
        score_frame.to_csv(index=False, lineterminator="\n").encode("ascii"),
    )
    training_log = pd.concat(gru_training_logs, ignore_index=True).sort_values(
        ["case", "model_seed", "update"], kind="stable"
    )
    _atomic_bytes(
        output_dir / "baseline_gru_training_log.csv",
        training_log.to_csv(index=False, lineterminator="\n").encode("ascii"),
    )

    restored = {
        baseline: load_selected_baseline_models(
            path, baseline=baseline, config=config
        )
        for baseline, path in bundle_paths.items()
    }
    if any(
        set(restored[name]) != set(entries)
        for name, entries in baseline_entries.items()
    ):
        raise AssertionError(
            "a persisted baseline bundle did not reload its full identity set"
        )

    baseline_frames = []
    for case in CASES:
        variants = variants_by_case[case]["validation"]
        scalers = scalers_by_case[case]
        arx_receipt, arx_model = restored["ridge_arx"][f"{case}:seed0"]
        baseline_frames.append(
            evaluate_arx_h8(
                arx_model, variants, scalers, arx_receipt, role="validation"
            )
        )
        direct_receipt, direct_model = restored["direct_h8_ridge"][
            f"{case}:seed0"
        ]
        baseline_frames.append(
            evaluate_direct_h8_ridge(
                direct_model,
                variants,
                scalers,
                direct_receipt,
                role="validation",
            )
        )
        for seed in config.development_seeds:
            gru_receipt, gru_model = restored["deterministic_gru"][
                f"{case}:seed{seed}"
            ]
            baseline_frames.append(
                evaluate_direct_h8_gru(
                    gru_model,
                    variants,
                    scalers,
                    gru_receipt,
                    role="validation",
                    device=device,
                )
            )
    baseline_frame = _sort_results(pd.concat(baseline_frames, ignore_index=True))
    baseline_result_path = output_dir / "baseline_validation_h8.csv"
    baseline_artifact = _write_result_csv(baseline_result_path, baseline_frame)
    stage_times["baseline_fit_select_evaluate"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    paired, gate_result = evaluate_study_gate(
        rssm_frame,
        config,
        stage="development",
        selected_update=selected_update,
        baseline_frame=baseline_frame,
    )
    if integration_only:
        gate_result = _mask_integration_gate(gate_result)
    write_gate_analysis(output_dir / "gate", paired, gate_result)
    gate_artifacts = _development_gate_artifacts(output_dir)
    stage_times["development_gate"] = time.monotonic() - stage_started

    total_wall_seconds = time.monotonic() - started
    wall_time_payload = {
        "schema": "boptest-reliability-development-wall-time-v1",
        "device": str(device),
        "stage_wall_seconds": stage_times,
        "total_wall_seconds": total_wall_seconds,
        "memory": _peak_memory(device),
        "simulator_steps_consumed_by_runner": 0,
        "sealed_corpus_transition_rows": len(index.records) * TRAJECTORY_STEPS,
        "optimizer_updates": (
            len(CASES)
            * len(config.development_seeds)
            * config.updates
            * (len(ARMS) + 1)
        ),
        "training_units": {
            "resumed": resumed_training_units,
            "trained_this_attempt": new_training_units,
            "completed_training_wall_seconds": sum(
                float(completion["wall_seconds"])
                for _, completion in completions.values()
            ),
        },
    }
    _write_canonical_json(output_dir / "wall_time.json", wall_time_payload)

    receipt_path = output_dir / "run_complete.json"
    final_code_manifest = scientific_code_manifest()
    if (
        final_code_manifest != identity["scientific_code_sha256_by_path"]
        or canonical_sha256(final_code_manifest)
        != identity["scientific_code_manifest_sha256"]
    ):
        raise RuntimeError("scientific code changed while the development run was active")
    final_runtime = numerical_runtime_fingerprint(device, include_sklearn=True)
    if final_runtime != identity["numerical_runtime"]:
        raise RuntimeError(
            "numerical runtime changed while the development run was active"
        )
    decision = "INTEGRATION_ONLY" if integration_only else gate_result["decision"]
    if not integration_only and decision not in {"SCREEN_GO", "SCREEN_STOP", "INCOMPLETE"}:
        raise AssertionError("development gate returned an invalid screen decision")
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "stage": "development",
        "interpretation": run_config["interpretation"],
        "scientific_screen_reported": not integration_only,
        "decision": decision,
        "selected_update": selected_update,
        "corpus_manifest_sha256": index.manifest_sha256,
        "corpus_manifest_file_sha256": identity["manifest_file_sha256"],
        "fault_manifest_sha256": fault_manifest.sha256,
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "runner_sha256": sha256_file(runner_path),
        "scientific_code_sha256_by_path": final_code_manifest,
        "scientific_code_manifest_sha256": canonical_sha256(
            final_code_manifest
        ),
        "numerical_runtime": final_runtime,
        "staging_identity_sha256": identity["identity_sha256"],
        "rssm_result": rssm_artifact,
        "baseline_result": baseline_artifact,
        "gate_artifacts": gate_artifacts,
        "validation_selection_sha256": sha256_file(
            frozen_dir / "validation_selection.json"
        ),
        "baseline_bundle_sha256": {
            name: sha256_file(path) for name, path in sorted(bundle_paths.items())
        },
        "wall_time": wall_time_payload,
        "artifact_inventory_excludes_this_receipt": _artifact_inventory(
            output_dir, exclude=(receipt_path,)
        ),
    }
    if set(receipt) != RUN_RECEIPT_FIELDS:
        raise AssertionError("development receipt fields differ from runner contract")
    _write_canonical_json(receipt_path, receipt)
    _publish_staging(output_dir, final_output_dir)
    return final_output_dir / receipt_path.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sealed multi-case development screen"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="use one update to test wiring; suppress every scientific screen verdict",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only a validated staging directory for this exact run identity",
    )
    args = parser.parse_args()
    receipt_path = run_development(
        args.manifest,
        args.output,
        device=args.device,
        integration_only=args.integration_only,
        resume=args.resume,
    )
    receipt = load_strict_json(receipt_path)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "selected_update": receipt["selected_update"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
