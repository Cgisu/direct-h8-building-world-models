from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .baselines import fit_direct_h8_gru
from .fault_data import (
    CorpusIndex,
    FaultScalers,
    FaultVariant,
    build_fault_manifest,
    iter_role_variants,
    load_corpus_index,
)
from .protocol import CASES, sha256_file
from .provenance import (
    _corpus_dependency_paths,
    _validate_baseline_bundle,
    _validate_frame_artifact_metadata,
    _validate_json_artifact_metadata,
    build_prelock_registry,
    canonical_sha256,
    extend_gru_selection_bundle,
    load_selected_baseline_entries,
    load_strict_json,
    validate_prelock_bundle,
    write_baseline_selection_bundle,
    write_prelock_registry,
)
from .runtime_provenance import (
    numerical_runtime_fingerprint,
    validate_current_numerical_runtime_fingerprint,
)
from .study_config import ARMS, StudyConfig
from .study_development import (
    RUN_CONFIG_FIELDS,
    RUNNER_SCHEMA,
    RUN_RECEIPT_FIELDS,
    RUN_RECEIPT_SCHEMA,
    _validate_development_index,
    scientific_code_manifest,
    validate_training_unit,
)
from .study_gate import canonical_frame_sha256, evaluate_study_gate
from .study_train import prepare_case_training_data, train_case_seed


PREPARATION_SCHEMA = "boptest-reliability-prelock-preparation-v1"
PREPARATION_STAGING_SCHEMA = "boptest-reliability-prelock-staging-v1"
GRU_UNIT_SCHEMA = "boptest-reliability-prelock-gru-unit-v1"
STAGING_MARKER = "prelock_staging_identity.json"
REGISTRY_NAME = "prelock_registry.json"
DIGEST_NAME = "prelock_registry.canonical.sha256"
COMPLETION_NAME = "prelock_preparation_complete.json"
BUNDLE_NAME = "bundle"
WORK_NAME = "work"


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


def _write_json(path: Path, payload: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("ascii"),
    )


def _copy_file(source: Path, destination: Path) -> Path:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"prelock source is not a plain file: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite prelock artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as output_stream:
        temporary = Path(output_stream.name)
        shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    os.replace(temporary, destination)
    if sha256_file(source) != sha256_file(destination):
        raise IOError(f"copied prelock artifact differs from source: {source}")
    return destination


def _staging_path(output_dir: Path) -> Path:
    if not output_dir.name:
        raise ValueError("prelock output must name a directory")
    return output_dir.parent / f".{output_dir.name}.prelock-staging"


def _extra_seeds(config: StudyConfig) -> tuple[int, ...]:
    values = tuple(
        seed for seed in config.confirmatory_seeds if seed not in config.development_seeds
    )
    if len(values) != 2:
        raise ValueError("prelock preparation requires exactly confirmatory seeds 4 and 5")
    return values


def _validate_development_source(
    manifest_path: Path,
    development_dir: Path,
    config: StudyConfig,
) -> tuple[CorpusIndex, dict, dict, int]:
    if development_dir.is_symlink() or not development_dir.is_dir():
        raise ValueError("development run is not a plain directory")
    run_config = load_strict_json(development_dir / "run_config.json")
    run_complete = load_strict_json(development_dir / "run_complete.json")
    if not isinstance(run_config, dict) or set(run_config) != RUN_CONFIG_FIELDS:
        raise ValueError("development run_config fields differ from runner contract")
    if not isinstance(run_complete, dict) or set(run_complete) != RUN_RECEIPT_FIELDS:
        raise ValueError("development run_complete fields differ from runner contract")
    if (
        run_config["schema"] != RUNNER_SCHEMA
        or run_complete["schema"] != RUN_RECEIPT_SCHEMA
        or run_config["stage"] != "development"
        or run_complete["stage"] != "development"
        or run_config["interpretation"] != "frozen_development_screen"
        or run_complete["interpretation"] != "frozen_development_screen"
        or run_config["scientific_screen_enabled"] is not True
        or run_complete["scientific_screen_reported"] is not True
    ):
        raise ValueError("prelock source is not a production development screen")
    if run_complete["decision"] != "SCREEN_GO":
        raise ValueError("prelock preparation requires SCREEN_GO")

    index = load_corpus_index(manifest_path)
    _validate_development_index(index)
    manifest_file_sha256 = sha256_file(manifest_path)
    if (
        run_config["corpus_manifest_sha256"] != index.manifest_sha256
        or run_complete["corpus_manifest_sha256"] != index.manifest_sha256
        or run_config["corpus_manifest_file_sha256"] != manifest_file_sha256
        or run_complete["corpus_manifest_file_sha256"] != manifest_file_sha256
    ):
        raise ValueError("development receipts differ from the supplied corpus manifest")
    if run_config["staging_identity_sha256"] != run_complete[
        "staging_identity_sha256"
    ]:
        raise ValueError("development staging identity changed during the run")
    expected_config = json.loads(json.dumps(config.to_dict(), allow_nan=False))
    expected_config_sha256 = canonical_sha256(config.to_dict())
    if (
        run_config["study_config"] != expected_config
        or run_config["study_config_sha256"] != expected_config_sha256
        or run_complete["study_config_sha256"] != expected_config_sha256
    ):
        raise ValueError("development receipts differ from the frozen study config")
    current_code = scientific_code_manifest()
    if (
        run_config["scientific_code_sha256_by_path"] != current_code
        or run_complete["scientific_code_sha256_by_path"] != current_code
        or run_config["scientific_code_manifest_sha256"]
        != canonical_sha256(current_code)
        or run_complete["scientific_code_manifest_sha256"]
        != canonical_sha256(current_code)
    ):
        raise ValueError("development receipts differ from current scientific code")

    rssm = _validate_frame_artifact_metadata(
        run_complete["rssm_result"],
        {
            "path": run_complete["rssm_result"]["path"],
            "sha256": run_complete["rssm_result"]["sha256"],
        },
        development_dir,
        label="RSSM validation result",
    )
    baseline = _validate_frame_artifact_metadata(
        run_complete["baseline_result"],
        {
            "path": run_complete["baseline_result"]["path"],
            "sha256": run_complete["baseline_result"]["sha256"],
        },
        development_dir,
        label="baseline validation result",
    )
    gate_artifacts = run_complete["gate_artifacts"]
    if not isinstance(gate_artifacts, dict) or set(gate_artifacts) != {
        "matched_metrics",
        "gate_result",
    }:
        raise ValueError("development gate artifact metadata are invalid")
    persisted_paired = _validate_frame_artifact_metadata(
        gate_artifacts["matched_metrics"],
        {
            "path": gate_artifacts["matched_metrics"]["path"],
            "sha256": gate_artifacts["matched_metrics"]["sha256"],
        },
        development_dir,
        label="gate matched metrics",
    )
    persisted_result = _validate_json_artifact_metadata(
        gate_artifacts["gate_result"],
        {
            "path": gate_artifacts["gate_result"]["path"],
            "sha256": gate_artifacts["gate_result"]["sha256"],
        },
        development_dir,
        label="gate result",
    )
    selected_update = run_complete["selected_update"]
    if selected_update not in config.validation_checkpoints:
        raise ValueError("development selected update is outside the frozen grid")
    paired, result = evaluate_study_gate(
        rssm,
        config,
        stage="development",
        selected_update=selected_update,
        baseline_frame=baseline,
    )
    if canonical_frame_sha256(paired) != canonical_frame_sha256(persisted_paired):
        raise ValueError("development matched metrics do not recompute")
    if json.loads(json.dumps(result, allow_nan=False)) != persisted_result:
        raise ValueError("development gate result does not recompute")
    if result.get("decision") != "SCREEN_GO":
        raise ValueError("prelock preparation requires a recomputed SCREEN_GO")
    return index, run_config, run_complete, int(selected_update)


def _staging_identity(
    *,
    manifest_path: Path,
    development_dir: Path,
    output_dir: Path,
    index: CorpusIndex,
    selected_update: int,
    config: StudyConfig,
    device: torch.device,
) -> dict:
    code = scientific_code_manifest()
    payload = {
        "schema": PREPARATION_STAGING_SCHEMA,
        "stage": "prelock",
        "final_output_path": str(output_dir.resolve()),
        "development_manifest_path": str(manifest_path.resolve()),
        "development_manifest_file_sha256": sha256_file(manifest_path),
        "development_corpus_manifest_sha256": index.manifest_sha256,
        "development_run_path": str(development_dir.resolve()),
        "development_run_config_sha256": sha256_file(
            development_dir / "run_config.json"
        ),
        "development_run_complete_sha256": sha256_file(
            development_dir / "run_complete.json"
        ),
        "selected_update": selected_update,
        "extra_confirmatory_seeds": list(_extra_seeds(config)),
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "scientific_code_sha256_by_path": code,
        "scientific_code_manifest_sha256": canonical_sha256(code),
        "numerical_runtime": numerical_runtime_fingerprint(
            device, include_sklearn=True
        ),
        "device": str(device),
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _prepare_staging(output_dir: Path, identity: dict, *, resume: bool) -> Path:
    staging = _staging_path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite prelock output: {output_dir}")
    if resume:
        if staging.is_symlink() or not staging.is_dir():
            raise FileNotFoundError(f"prelock staging directory is missing: {staging}")
        allowed = {
            STAGING_MARKER,
            WORK_NAME,
            BUNDLE_NAME,
            REGISTRY_NAME,
            DIGEST_NAME,
            COMPLETION_NAME,
        }
        for path in staging.iterdir():
            if path.name not in allowed or path.is_symlink():
                raise ValueError(f"prelock staging contains an unknown artifact: {path}")
        recorded = load_strict_json(staging / STAGING_MARKER)
        if recorded != identity:
            raise ValueError("prelock staging identity differs from this request")
        validate_current_numerical_runtime_fingerprint(
            recorded.get("numerical_runtime"), include_sklearn=True
        )
        for name in (BUNDLE_NAME, REGISTRY_NAME, DIGEST_NAME, COMPLETION_NAME):
            path = staging / name
            if not path.exists():
                continue
            if path.is_symlink():
                raise ValueError(f"prelock recomputable artifact is a symlink: {name}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    else:
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(
                f"prelock staging exists; inspect it and rerun with --resume: {staging}"
            )
        staging.mkdir(parents=True, exist_ok=False)
        _write_json(staging / STAGING_MARKER, identity)
    (staging / WORK_NAME).mkdir(exist_ok=True)
    return staging


def _validate_work_topology(work: Path, config: StudyConfig) -> None:
    expected_seeds = {f"seed{seed}" for seed in _extra_seeds(config)}
    for kind_path in work.iterdir():
        if (
            kind_path.is_symlink()
            or not kind_path.is_dir()
            or kind_path.name not in {"rssm", "gru"}
        ):
            raise ValueError(f"prelock work contains an unknown artifact: {kind_path}")
        for case_path in kind_path.iterdir():
            if (
                case_path.is_symlink()
                or not case_path.is_dir()
                or case_path.name not in CASES
            ):
                raise ValueError(f"prelock work contains an unknown case: {case_path}")
            for seed_path in case_path.iterdir():
                if (
                    seed_path.is_symlink()
                    or not seed_path.is_dir()
                    or seed_path.name not in expected_seeds
                ):
                    raise ValueError(f"prelock work contains an unknown seed: {seed_path}")


def _gru_unit(
    unit_dir: Path,
    *,
    case: str,
    seed: int,
    fit_variants: list[FaultVariant],
    validation_variants: list[FaultVariant],
    scalers: FaultScalers,
    variants_by_case_role: dict[tuple[str, str], list[FaultVariant]],
    scalers_by_case: dict[str, FaultScalers],
    config: StudyConfig,
    device: torch.device,
) -> tuple[object, object, pd.DataFrame, dict]:
    identity = f"{case}:seed{seed}"
    bundle_path = unit_dir / "selected_bundle.pt"
    log_path = unit_dir / "training_log.csv"
    complete_path = unit_dir / "complete.json"
    if unit_dir.exists():
        if unit_dir.is_symlink() or not unit_dir.is_dir():
            raise ValueError(f"GRU work unit is not a plain directory: {unit_dir}")
        expected = {bundle_path.name, log_path.name, complete_path.name}
        if {path.name for path in unit_dir.iterdir()} != expected:
            raise ValueError(
                f"partial GRU work unit; remove only this directory before resume: {unit_dir}"
            )
        complete = load_strict_json(complete_path)
        if (
            not isinstance(complete, dict)
            or set(complete)
            != {
                "schema",
                "case",
                "model_seed",
                "bundle_sha256",
                "training_log_sha256",
                "selection_receipt_sha256",
                "study_config_sha256",
                "runtime",
                "device",
                "wall_seconds",
            }
            or complete["schema"] != GRU_UNIT_SCHEMA
            or complete["case"] != case
            or complete["model_seed"] != seed
            or complete["bundle_sha256"] != sha256_file(bundle_path)
            or complete["training_log_sha256"] != sha256_file(log_path)
            or complete["study_config_sha256"] != canonical_sha256(config.to_dict())
            or complete["device"] != str(device)
        ):
            raise ValueError(f"GRU work unit completion is invalid: {unit_dir}")
        validate_current_numerical_runtime_fingerprint(
            complete["runtime"], include_sklearn=True
        )
    else:
        unit_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        result = fit_direct_h8_gru(
            fit_variants,
            validation_variants,
            scalers,
            config,
            model_seed=seed,
            device=device,
        )
        write_baseline_selection_bundle(
            bundle_path,
            baseline="deterministic_gru",
            entries={identity: (result.receipt, result.model, result.score_table)},
        )
        _atomic_bytes(
            log_path,
            result.training_log.to_csv(index=False, lineterminator="\n").encode(
                "ascii"
            ),
        )
        _write_json(
            complete_path,
            {
                "schema": GRU_UNIT_SCHEMA,
                "case": case,
                "model_seed": seed,
                "bundle_sha256": sha256_file(bundle_path),
                "training_log_sha256": sha256_file(log_path),
                "selection_receipt_sha256": result.receipt.sha256,
                "study_config_sha256": canonical_sha256(config.to_dict()),
                "runtime": result.receipt.runtime_fingerprint,
                "device": str(device),
                "wall_seconds": time.monotonic() - started,
            },
        )
        complete = load_strict_json(complete_path)

    _validate_baseline_bundle(
        bundle_path,
        baseline="deterministic_gru",
        config=config,
        variants_by_case_role=variants_by_case_role,
        scalers=scalers_by_case,
        gru_seeds=(seed,),
        cases=(case,),
    )
    entries = load_selected_baseline_entries(
        bundle_path, baseline="deterministic_gru", config=config
    )
    if set(entries) != {identity}:
        raise ValueError(f"GRU work unit identity is invalid: {unit_dir}")
    receipt, model, score_table = entries[identity]
    validate_current_numerical_runtime_fingerprint(
        receipt.runtime_fingerprint, include_sklearn=True
    )
    if receipt.sha256 != complete["selection_receipt_sha256"]:
        raise ValueError(f"GRU work unit receipt differs: {unit_dir}")
    log = pd.read_csv(log_path)
    if len(log) != config.updates or set(log.columns) != {
        "update",
        "fit_smooth_l1",
        "gradient_norm",
    }:
        raise ValueError(f"GRU work unit training log is invalid: {unit_dir}")
    return receipt, model, score_table, complete


def _copy_corpus(manifest_path: Path, bundle: Path) -> Path:
    source_root = manifest_path.parent.parent.resolve()
    destination_root = bundle / "corpus"
    dependencies = _corpus_dependency_paths(source_root, manifest_path.resolve())
    for relative in sorted(dependencies):
        _copy_file(source_root / relative, destination_root / relative)
    return destination_root / manifest_path.resolve().relative_to(source_root)


def _assemble_bundle(
    bundle: Path,
    *,
    manifest_path: Path,
    development_dir: Path,
    work_dir: Path,
    selected_update: int,
    config: StudyConfig,
    gru_extension_entries: dict,
) -> tuple[dict, Path]:
    bundle.mkdir(parents=True, exist_ok=False)
    copied_manifest = _copy_corpus(manifest_path, bundle)
    development = bundle / "development"
    copied_run_config = _copy_file(
        development_dir / "run_config.json", development / "run_config.json"
    )
    copied_run_complete = _copy_file(
        development_dir / "run_complete.json", development / "run_complete.json"
    )
    copied_rssm = _copy_file(
        development_dir / "rssm_validation_h8.csv",
        development / "rssm_validation_h8.csv",
    )
    copied_baseline = _copy_file(
        development_dir / "baseline_validation_h8.csv",
        development / "baseline_validation_h8.csv",
    )
    copied_gate_paired = _copy_file(
        development_dir / "gate" / "matched_h8_arm_metrics.csv",
        development / "gate" / "matched_h8_arm_metrics.csv",
    )
    copied_gate_result = _copy_file(
        development_dir / "gate" / "study_gate.json",
        development / "gate" / "study_gate.json",
    )

    frozen = bundle / "frozen"
    copied_fault = _copy_file(
        development_dir / "frozen" / "frozen_fault_contract.json",
        frozen / "frozen_fault_contract.json",
    )
    copied_selection = _copy_file(
        development_dir / "frozen" / "validation_selection.json",
        frozen / "validation_selection.json",
    )
    _copy_file(
        development_dir / "frozen" / "validation_selection_rows.csv",
        frozen / "validation_selection_rows.csv",
    )
    scalers: dict[str, Path] = {}
    checkpoints: dict[str, Path] = {}
    schedules: dict[str, Path] = {}
    for case in CASES:
        scalers[case] = _copy_file(
            development_dir / "frozen" / "fit_scalers" / f"{case}.json",
            frozen / "fit_scalers" / f"{case}.json",
        )
        for seed in config.confirmatory_seeds:
            source_root = (
                development_dir / "training" / case / f"seed{seed}"
                if seed in config.development_seeds
                else work_dir / "rssm" / case / f"seed{seed}"
            )
            identity = f"{case}:seed{seed}"
            schedules[identity] = _copy_file(
                source_root / "training_schedule.json",
                frozen / "schedules" / case / f"seed{seed}.json",
            )
            for arm in ARMS:
                checkpoint_identity = (
                    f"{case}:seed{seed}:{arm}:u{selected_update:04d}"
                )
                checkpoints[checkpoint_identity] = _copy_file(
                    source_root
                    / "checkpoints"
                    / f"{arm}_u{selected_update:04d}.pt",
                    frozen
                    / "checkpoints"
                    / case
                    / f"seed{seed}"
                    / f"{arm}_u{selected_update:04d}.pt",
                )

    baseline_root = frozen / "baselines"
    ridge = _copy_file(
        development_dir / "frozen" / "baselines" / "ridge_arx.pt",
        baseline_root / "ridge_arx.pt",
    )
    direct = _copy_file(
        development_dir / "frozen" / "baselines" / "direct_h8_ridge.pt",
        baseline_root / "direct_h8_ridge.pt",
    )
    development_gru = _copy_file(
        development_dir / "frozen" / "baselines" / "deterministic_gru.pt",
        baseline_root / "development_deterministic_gru.pt",
    )
    confirmatory_gru = baseline_root / "deterministic_gru.pt"
    extend_gru_selection_bundle(
        confirmatory_gru,
        development_bundle=development_gru,
        extension_entries=gru_extension_entries,
        config=config,
    )
    registry = build_prelock_registry(
        artifact_root=bundle,
        development_run_config_path=copied_run_config,
        development_run_complete_path=copied_run_complete,
        development_rssm_result_path=copied_rssm,
        development_baseline_result_path=copied_baseline,
        development_gate_paired_path=copied_gate_paired,
        development_gate_result_path=copied_gate_result,
        development_gru_baseline_bundle=development_gru,
        development_corpus_manifest=copied_manifest,
        frozen_fault_contract_path=copied_fault,
        validation_selection_path=copied_selection,
        fit_scalers_by_case=scalers,
        checkpoints_by_identity=checkpoints,
        schedules_by_case_seed=schedules,
        baseline_bundles_by_arm={
            "ridge_arx": ridge,
            "direct_h8_ridge": direct,
            "deterministic_gru": confirmatory_gru,
        },
        config=config,
        selected_update=selected_update,
    )
    return registry, copied_manifest


def run_prelock_preparation(
    manifest_path: Path,
    development_dir: Path,
    output_dir: Path,
    *,
    device: torch.device | str = "cpu",
    resume: bool = False,
) -> Path:
    """Prepare the frozen five-seed bundle without reading locked-test values."""
    config = StudyConfig()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index, _, run_complete, selected_update = _validate_development_source(
        manifest_path, development_dir, config
    )
    identity = _staging_identity(
        manifest_path=manifest_path,
        development_dir=development_dir,
        output_dir=output_dir,
        index=index,
        selected_update=selected_update,
        config=config,
        device=device,
    )
    staging = _prepare_staging(output_dir, identity, resume=resume)
    work = staging / WORK_NAME
    _validate_work_topology(work, config)
    started = time.monotonic()
    fault_manifest = build_fault_manifest(index)
    variants_by_case_role: dict[tuple[str, str], list[FaultVariant]] = {}
    scalers_by_case: dict[str, FaultScalers] = {}
    fit_by_case: dict[str, list[FaultVariant]] = {}
    validation_by_case: dict[str, list[FaultVariant]] = {}
    for case in CASES:
        fit_variants, scalers = prepare_case_training_data(index, fault_manifest, case)
        validation_variants = list(
            iter_role_variants(index, fault_manifest, "validation", cases=(case,))
        )
        fit_by_case[case] = fit_variants
        validation_by_case[case] = validation_variants
        scalers_by_case[case] = scalers
        variants_by_case_role[(case, "fit")] = fit_variants
        variants_by_case_role[(case, "validation")] = validation_variants

    rssm_completions: dict[str, dict] = {}
    gru_completions: dict[str, dict] = {}
    gru_extension_entries: dict = {}
    for case in CASES:
        for seed in _extra_seeds(config):
            rssm_dir = work / "rssm" / case / f"seed{seed}"
            if not rssm_dir.exists():
                train_case_seed(
                    index,
                    case=case,
                    model_seed=seed,
                    output_dir=work / "rssm",
                    config=config,
                    arms=ARMS,
                    device=device,
                )
            _, rssm_completion = validate_training_unit(
                rssm_dir,
                index=index,
                fault_manifest=fault_manifest,
                variants=fit_by_case[case],
                scalers=scalers_by_case[case],
                config=config,
                case=case,
                seed=seed,
                device=device,
            )
            identity_text = f"{case}:seed{seed}"
            rssm_completions[identity_text] = rssm_completion
            receipt, model, score_table, gru_completion = _gru_unit(
                work / "gru" / case / f"seed{seed}",
                case=case,
                seed=seed,
                fit_variants=fit_by_case[case],
                validation_variants=validation_by_case[case],
                scalers=scalers_by_case[case],
                variants_by_case_role=variants_by_case_role,
                scalers_by_case=scalers_by_case,
                config=config,
                device=device,
            )
            gru_extension_entries[identity_text] = (receipt, model, score_table)
            gru_completions[identity_text] = gru_completion

    registry, copied_manifest = _assemble_bundle(
        staging / BUNDLE_NAME,
        manifest_path=manifest_path,
        development_dir=development_dir,
        work_dir=work,
        selected_update=selected_update,
        config=config,
        gru_extension_entries=gru_extension_entries,
    )
    registry_path = staging / REGISTRY_NAME
    digest = write_prelock_registry(registry_path, registry)
    _atomic_bytes(staging / DIGEST_NAME, f"{digest}\n".encode("ascii"))
    validate_prelock_bundle(
        registry_path, staging / BUNDLE_NAME, config, digest
    )
    current_code = scientific_code_manifest()
    if current_code != identity["scientific_code_sha256_by_path"]:
        raise RuntimeError("scientific code changed during prelock preparation")
    completion = {
        "schema": PREPARATION_SCHEMA,
        "stage": "prelock",
        "selected_update": selected_update,
        "extra_confirmatory_seeds": list(_extra_seeds(config)),
        "development_run_complete_sha256": sha256_file(
            development_dir / "run_complete.json"
        ),
        "development_corpus_manifest_sha256": index.manifest_sha256,
        "copied_development_manifest_sha256": sha256_file(copied_manifest),
        "registry_file_sha256": sha256_file(registry_path),
        "canonical_registry_sha256": digest,
        "artifact_inventory": registry["artifact_inventory"],
        "rssm_completion_sha256_by_case_seed": {
            identity_text: canonical_sha256(payload)
            for identity_text, payload in sorted(rssm_completions.items())
        },
        "gru_completion_sha256_by_case_seed": {
            identity_text: canonical_sha256(payload)
            for identity_text, payload in sorted(gru_completions.items())
        },
        "device": str(device),
        "numerical_runtime": identity["numerical_runtime"],
        "wall_seconds": time.monotonic() - started,
        "locked_values_read": False,
        "external_timestamp_required": True,
    }
    _write_json(staging / COMPLETION_NAME, completion)
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("prelock publication destination changed during preparation")
    os.rename(staging, output_dir)
    parent_fd = os.open(output_dir.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return output_dir / REGISTRY_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen post-SCREEN_GO prelock bundle"
    )
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    registry_path = run_prelock_preparation(
        args.development_manifest,
        args.development_run,
        args.output,
        device=args.device,
        resume=args.resume,
    )
    registry = load_strict_json(registry_path)
    print(
        json.dumps(
            {
                "registry": str(registry_path),
                "artifact_root": str(registry_path.parent / BUNDLE_NAME),
                "canonical_registry_sha256": canonical_sha256(registry),
                "external_timestamp_required": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
