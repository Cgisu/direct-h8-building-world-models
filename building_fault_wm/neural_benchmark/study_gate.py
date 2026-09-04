from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch

from .fault_data import FAULT_CHANNELS, FaultSpec, fault_cell_signatures
from .protocol import CASES, PRELOCK_REGISTRY_SCHEMA, TRAJECTORY_STEPS
from .provenance import (
    prelock_plan_sha256_by_case,
    validate_locked_corpus_binding,
    validate_prelock_registry_object,
)
from .runtime_provenance import validate_numerical_runtime_fingerprint
from .study_config import ARMS, StudyConfig
from .study_train import core_tensor_state_sha256


Stage = Literal["development", "confirmation"]

SILENT_FAMILIES = ("bias", "drift", "stuck")
COMPETENCE_BASELINE_ARMS = (
    "ridge_arx",
    "direct_h8_ridge",
    "deterministic_gru",
)
GATE_SCHEMA = "boptest-multicase-reliability-gate-v1"
PRELOCK_PROVENANCE_SCHEMA = PRELOCK_REGISTRY_SCHEMA
EVALUATION_PROVENANCE_SCHEMA = "boptest-multicase-evaluation-receipt-v1"
DEVELOPMENT_IMPROVEMENT_THRESHOLD = 0.075
DEVELOPMENT_BASELINE_RATIO_LIMIT = 1.15
CONFIRMATION_BASELINE_POINT_RATIO_LIMIT = 1.05
CONFIRMATION_BASELINE_CI_RATIO_LIMIT = 1.10
CONFIRMATION_BASELINE_CASE_RATIO_LIMIT = 1.10

_PAIR_KEYS = (
    "case",
    "role",
    "trajectory_day",
    "trajectory_seed",
    "model_seed",
    "update",
    "cell_id",
    "fault_channel",
    "family",
    "sign",
    "severity",
    "severity_unit",
    "onset",
    "anchor",
    "horizon",
)
_REQUIRED_RSSM_COLUMNS = {
    *_PAIR_KEYS,
    "arm",
    "standardized_abs_error",
    "target_raw",
    "prediction_raw",
    "alternate_action_prediction_raw",
    "alternate_action_standardized_abs_error",
    "action_prediction_change_standardized",
    "persistence_prediction_raw",
    "persistence_standardized_abs_error",
}
_CLUSTERS = ("case", "model_seed", "trajectory_day", "trajectory_seed")
_EQUAL_DIMENSIONS = ("family", "fault_channel", "sign", "severity")
_SCORE_COLUMNS = (*ARMS, "persistence")
_RAW_SCORE_COLUMNS = (
    *(f"{arm}_raw_abs_error" for arm in ARMS),
    "persistence_raw_abs_error",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON identity: {key}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a result table independently of its input row and column ordering."""
    if frame.empty or any(not isinstance(column, str) for column in frame.columns):
        raise ValueError("a canonical result frame must be nonempty with string columns")
    columns = sorted(frame.columns)
    canonical = frame.loc[:, columns].sort_values(
        columns, kind="stable", na_position="last"
    )
    content = canonical.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="null",
    ).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def _config_sha256(config: StudyConfig) -> str:
    return hashlib.sha256(_canonical_json(config.to_dict()).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_checkpoint_keys(
    config: StudyConfig, selected_update: int
) -> set[str]:
    return {
        f"{case}:seed{seed}:{arm}:u{selected_update:04d}"
        for case in CASES
        for seed in config.confirmatory_seeds
        for arm in ARMS
    }


def _required_schedule_keys(config: StudyConfig) -> set[str]:
    return {
        f"{case}:seed{seed}"
        for case in CASES
        for seed in config.confirmatory_seeds
    }


def _prelock_artifact_references(
    receipt: dict,
    config: StudyConfig,
    selected_update: int,
    issues: list[str],
) -> list[tuple[str, dict]]:
    references: list[tuple[str, dict]] = []
    single_fields = {
        "development_run_config_artifact": (
            "development_run_config",
            "development",
        ),
        "development_run_complete_artifact": (
            "development_run_receipt",
            "development",
        ),
        "development_gru_baseline_artifact": (
            "development_baseline_validation_selection",
            "deterministic_gru",
        ),
        "corpus_manifest_artifact": ("development_corpus_manifest", "development"),
        "fault_manifest_artifact": ("frozen_fault_manifest", "all_roles"),
        "validation_selection_artifact": (
            "validation_checkpoint_selection",
            f"update{selected_update:04d}",
        ),
    }
    for field, (kind, identity) in single_fields.items():
        value = receipt.get(field)
        if not isinstance(value, dict):
            issues.append(f"{field} is missing or is not an artifact reference")
        else:
            if value.get("kind") != kind or value.get("identity") != identity:
                issues.append(f"{field} kind/identity differs from the frozen contract")
            references.append((field, value))
    mapped_fields = {
        "fit_scaler_artifact_by_case": (
            set(CASES),
            "fit_scaler",
        ),
        "checkpoint_artifact_by_identity": (
            _required_checkpoint_keys(config, selected_update),
            "rssm_checkpoint",
        ),
        "training_schedule_artifact_by_case_seed": (
            _required_schedule_keys(config),
            "training_schedule",
        ),
        "baseline_selection_artifact_by_arm": (
            set(COMPETENCE_BASELINE_ARMS),
            "baseline_validation_selection",
        ),
    }
    for field, (expected_keys, kind) in mapped_fields.items():
        value = receipt.get(field)
        if not isinstance(value, dict) or set(value) != expected_keys:
            issues.append(f"{field} keys differ from the frozen identities")
            continue
        for identity, reference in value.items():
            if not isinstance(reference, dict):
                issues.append(f"{field}[{identity}] is not an artifact reference")
            else:
                if (
                    reference.get("kind") != kind
                    or reference.get("identity") != identity
                ):
                    issues.append(
                        f"{field}[{identity}] kind/identity differs from its key"
                    )
                references.append((f"{field}[{identity}]", reference))
    return references


def _evaluation_artifact_references(
    receipt: dict, issues: list[str]
) -> list[tuple[str, dict]]:
    references: list[tuple[str, dict]] = []
    expected = {
        "evaluation_source_manifest_artifact": (
            "locked_evaluation_source_manifest",
            "locked_test",
        ),
        "attempt_marker_artifact": (
            "confirmation_attempt_marker",
            receipt.get("prelock_registry_sha256"),
        ),
        "locked_collection_completion_artifact": (
            "locked_collection_completion_marker",
            receipt.get("prelock_registry_sha256"),
        ),
    }
    for field, (kind, identity) in expected.items():
        value = receipt.get(field)
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "kind",
            "identity",
        }:
            issues.append(f"{field} is missing or is not an artifact reference")
            continue
        if value.get("kind") != kind or value.get("identity") != identity:
            issues.append(f"{field} kind/identity differs from the locked-test contract")
        references.append((field, value))
    return references


def _validate_artifacts(
    receipt: dict,
    references: Sequence[tuple[str, dict]],
    artifact_root: Path | None,
    issues: list[str],
) -> None:
    if artifact_root is None:
        issues.append("confirmation artifact root is missing")
        return
    root = artifact_root.resolve()
    if not root.is_dir():
        issues.append("confirmation artifact root is not a directory")
        return
    declared_paths: list[str] = []
    declared_hashes: list[str] = []
    for label, reference in references:
        relative = reference.get("path")
        declared_hash = reference.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            issues.append(f"{label} has an invalid relative path")
            continue
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            issues.append(f"{label} escapes the confirmation artifact root")
            continue
        declared_paths.append(relative)
        if not _is_sha256(declared_hash):
            issues.append(f"{label} has an invalid SHA-256")
            continue
        declared_hashes.append(declared_hash)
        if not candidate.is_file():
            issues.append(f"{label} artifact is missing")
        elif _sha256_file(candidate) != declared_hash:
            issues.append(f"{label} artifact SHA-256 does not match")

    if len(declared_paths) != len(set(declared_paths)):
        issues.append("confirmation artifact paths are reused across identities")
    placeholder_hashes = {"0" * 64, "a" * 64, "f" * 64}
    if any(value in placeholder_hashes for value in declared_hashes):
        issues.append("confirmation artifacts contain a placeholder-only SHA-256")
    if len(declared_hashes) != len(set(declared_hashes)):
        issues.append("confirmation artifact hashes are reused across identities")

    inventory = receipt.get("artifact_inventory")
    if not isinstance(inventory, list) or any(
        not isinstance(item, str) for item in inventory
    ):
        issues.append("artifact_inventory is missing or invalid")
        return
    if len(inventory) != len(set(inventory)):
        issues.append("artifact_inventory contains duplicate paths")
    if not set(declared_paths).issubset(set(inventory)):
        issues.append("artifact_inventory omits a referenced artifact")
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if actual_files != set(inventory):
        issues.append("artifact root contains missing or unlisted files")


def _load_collection_completion_evidence(
    receipt: dict,
    artifact_root: Path | None,
    *,
    prelock_sha256: str,
    issues: list[str],
) -> dict | None:
    from .study_confirmation import COLLECTION_EVIDENCE_PATH
    from .study_locked_collection import (
        CANONICAL_LOCKED_MANIFEST,
        COMPLETION_FIELDS,
        COMPLETION_SCHEMA,
    )

    reference = receipt.get("locked_collection_completion_artifact")
    if not isinstance(reference, dict) or artifact_root is None:
        return None
    if reference.get("path") != COLLECTION_EVIDENCE_PATH:
        issues.append("locked-collection evidence path differs from the frozen contract")
        return None
    root = artifact_root.resolve()
    path = (root / COLLECTION_EVIDENCE_PATH).resolve()
    try:
        path.relative_to(root)
        completion = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError) as error:
        issues.append(f"locked-collection evidence is unreadable: {error}")
        return None
    if not isinstance(completion, dict) or set(completion) != COMPLETION_FIELDS:
        issues.append("locked-collection evidence fields differ from the frozen schema")
        return None
    source_reference = receipt.get("evaluation_source_manifest_artifact")
    source_manifest_sha256 = (
        source_reference.get("sha256")
        if isinstance(source_reference, dict)
        else None
    )
    expected_trajectories = 12 * len(CASES)
    expected = {
        "schema": COMPLETION_SCHEMA,
        "stage": "locked_collection",
        "prelock_registry_sha256": prelock_sha256,
        "collector_exit_code": 0,
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "locked_manifest_file_sha256": source_manifest_sha256,
        "locked_manifest_payload_sha256": receipt.get(
            "locked_corpus_manifest_sha256"
        ),
        "collection_kind": "locked_test",
        "selected_cases": sorted(CASES),
        "counts": {
            "cases": len(CASES),
            "trajectories": expected_trajectories,
            "rows": expected_trajectories * TRAJECTORY_STEPS,
            "roles": {"locked_test": expected_trajectories},
        },
        "locked_values_accessed_after_attempt": True,
    }
    for field, value in expected.items():
        if completion.get(field) != value:
            issues.append(f"locked-collection evidence {field} is invalid")
    return completion


def _validate_attempt_marker_evidence(
    receipt: dict,
    artifact_root: Path | None,
    *,
    selected_update: int,
    prelock_sha256: str,
    scientific_code_manifest_sha256: str,
    confirmation_runner_sha256: str,
    collection_completion: dict | None,
    issues: list[str],
) -> None:
    from .study_confirmation import (
        ATTEMPT_EVIDENCE_PATH,
        ATTEMPT_FIELDS,
        ATTEMPT_SCHEMA,
        EVALUATION_POLICY,
    )
    from .study_locked_collection import CANONICAL_LOCKED_MANIFEST

    reference = receipt.get("attempt_marker_artifact")
    if not isinstance(reference, dict):
        return
    if (
        reference.get("path") != ATTEMPT_EVIDENCE_PATH
        or reference.get("sha256") != receipt.get("attempt_marker_sha256")
    ):
        issues.append("attempt-marker artifact does not match its receipt identity")
        return
    if artifact_root is None:
        return
    root = artifact_root.resolve()
    marker_path = (root / ATTEMPT_EVIDENCE_PATH).resolve()
    try:
        marker_path.relative_to(root)
        marker = json.loads(
            marker_path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError) as error:
        issues.append(f"attempt-marker evidence is unreadable: {error}")
        return
    if not isinstance(marker, dict) or set(marker) != ATTEMPT_FIELDS:
        issues.append("attempt-marker evidence fields differ from the frozen schema")
        return
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "one_shot": True,
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "prelock_registry_sha256": prelock_sha256,
        "selected_update": selected_update,
        "locked_corpus_manifest_sha256": receipt.get(
            "locked_corpus_manifest_sha256"
        ),
        "runner_sha256": confirmation_runner_sha256,
        "scientific_code_manifest_sha256": scientific_code_manifest_sha256,
        "evaluation_policy": dict(EVALUATION_POLICY),
        "evaluation_runtime": receipt.get("evaluation_runtime"),
    }
    if collection_completion is not None:
        collection_reference = receipt.get(
            "locked_collection_completion_artifact"
        )
        expected.update(
            {
                "locked_manifest_file_sha256": collection_completion.get(
                    "locked_manifest_file_sha256"
                ),
                "locked_collection_completion_sha256": receipt.get(
                    "locked_collection_completion_artifact", {}
                ).get("sha256")
                if isinstance(collection_reference, dict)
                else None,
                "external_freeze_receipt_sha256": collection_completion.get(
                    "external_freeze_receipt_sha256"
                ),
            }
        )
    for field, value in expected.items():
        if marker.get(field) != value:
            issues.append(f"attempt-marker {field} differs from the locked run")
    for field in ("output_path",):
        value = marker.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            issues.append(f"attempt-marker {field} is not an absolute path")
    started_at = marker.get("started_at_utc")
    try:
        parsed = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        if not isinstance(started_at, str) or not started_at.endswith("Z"):
            raise ValueError("timestamp is not canonical UTC")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp has no timezone")
    except ValueError as error:
        issues.append(f"attempt-marker started_at_utc is invalid: {error}")
    identity_payload = dict(marker)
    identity = identity_payload.pop("attempt_identity_sha256", None)
    expected_identity = hashlib.sha256(
        _canonical_json(identity_payload).encode("ascii")
    ).hexdigest()
    if identity != expected_identity:
        issues.append("attempt-marker identity SHA-256 is invalid")


def validate_confirmation_provenance(
    prelock_registry: dict | None,
    evaluation_receipt: dict | None,
    rssm_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame | None,
    config: StudyConfig,
    *,
    selected_update: int,
    prelock_artifact_root: Path | None,
    evaluation_artifact_root: Path | None,
    expected_prelock_sha256: str | None,
) -> list[str]:
    """Validate the pre-lock freeze separately from post-run result binding."""
    issues: list[str] = []
    if not isinstance(prelock_registry, dict):
        issues.append("pre-lock provenance registry is missing or invalid")
        return issues
    actual_prelock_sha256 = hashlib.sha256(
        _canonical_json(prelock_registry).encode("ascii")
    ).hexdigest()
    if not _is_sha256(expected_prelock_sha256):
        issues.append("externally frozen pre-lock registry SHA-256 is missing")
    elif expected_prelock_sha256 != actual_prelock_sha256:
        issues.append("pre-lock registry differs from its externally frozen SHA-256")
    here = Path(__file__).resolve().parent
    if prelock_artifact_root is None:
        issues.append("confirmation pre-lock artifact root is missing")
    else:
        try:
            validate_prelock_registry_object(
                prelock_registry,
                prelock_artifact_root,
                config,
                actual_prelock_sha256,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            issues.append(f"canonical pre-lock validation failed: {error}")
    if prelock_registry.get("selected_update") != selected_update:
        issues.append("pre-lock selected update differs from confirmation")

    if not isinstance(evaluation_receipt, dict):
        issues.append("post-run evaluation receipt is missing or invalid")
        return issues
    from .study_confirmation import (
        EVALUATION_POLICY,
        EVALUATION_RECEIPT_FIELDS,
        RUNNER_RELATIVE_PATH,
        confirmation_scientific_code_manifest,
    )

    if set(evaluation_receipt) != EVALUATION_RECEIPT_FIELDS:
        issues.append("evaluation receipt fields differ from the frozen schema")
    scientific_code = confirmation_scientific_code_manifest()
    evaluation_exact = {
        "schema": EVALUATION_PROVENANCE_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "selected_update": selected_update,
        "prelock_registry_sha256": actual_prelock_sha256,
        "study_config_sha256": _config_sha256(config),
        "confirmation_runner_sha256": scientific_code[RUNNER_RELATIVE_PATH],
        "scientific_code_sha256_by_path": scientific_code,
        "scientific_code_manifest_sha256": hashlib.sha256(
            _canonical_json(scientific_code).encode("ascii")
        ).hexdigest(),
        "evaluator_sha256": _sha256_file(here / "study_evaluate.py"),
        "gate_sha256": _sha256_file(Path(__file__).resolve()),
        "rssm_frame_sha256": canonical_frame_sha256(rssm_frame),
        "evaluation_policy": dict(EVALUATION_POLICY),
    }
    if baseline_frame is not None:
        evaluation_exact["baseline_frame_sha256"] = canonical_frame_sha256(
            baseline_frame
        )
    for field, expected in evaluation_exact.items():
        if evaluation_receipt.get(field) != expected:
            issues.append(f"evaluation {field} does not match the locked input")
    try:
        validate_numerical_runtime_fingerprint(
            evaluation_receipt.get("evaluation_runtime"), include_sklearn=True
        )
    except (TypeError, ValueError) as error:
        issues.append(f"evaluation runtime fingerprint is invalid: {error}")
    if baseline_frame is None:
        issues.append("baseline_frame_sha256 cannot be verified without baseline rows")
    for field in (
        "locked_corpus_manifest_sha256",
        "locked_fault_manifest_sha256",
        "attempt_marker_sha256",
    ):
        if not _is_sha256(evaluation_receipt.get(field)):
            issues.append(f"evaluation {field} is invalid")
    for field, frame, filename in (
        ("rssm_result_artifact", rssm_frame, "rssm_locked_h8.csv"),
        ("baseline_result_artifact", baseline_frame, "baseline_locked_h8.csv"),
    ):
        metadata = evaluation_receipt.get(field)
        if frame is None:
            continue
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"path", "rows", "sha256", "canonical_frame_sha256"}
            or metadata.get("path") != filename
            or metadata.get("rows") != len(frame)
            or not _is_sha256(metadata.get("sha256"))
            or metadata.get("canonical_frame_sha256")
            != canonical_frame_sha256(frame)
        ):
            issues.append(f"evaluation {field} metadata differ from persisted rows")
    evaluation_references = _evaluation_artifact_references(
        evaluation_receipt, issues
    )
    _validate_artifacts(
        evaluation_receipt,
        evaluation_references,
        evaluation_artifact_root,
        issues,
    )
    collection_completion = _load_collection_completion_evidence(
        evaluation_receipt,
        evaluation_artifact_root,
        prelock_sha256=actual_prelock_sha256,
        issues=issues,
    )
    _validate_attempt_marker_evidence(
        evaluation_receipt,
        evaluation_artifact_root,
        selected_update=selected_update,
        prelock_sha256=actual_prelock_sha256,
        scientific_code_manifest_sha256=evaluation_exact[
            "scientific_code_manifest_sha256"
        ],
        confirmation_runner_sha256=evaluation_exact["confirmation_runner_sha256"],
        collection_completion=collection_completion,
        issues=issues,
    )
    source_reference = evaluation_receipt.get(
        "evaluation_source_manifest_artifact"
    )
    if evaluation_artifact_root is not None and isinstance(source_reference, dict):
        reference = source_reference
        relative = reference.get("path")
        if isinstance(relative, str):
            root = evaluation_artifact_root.resolve()
            manifest_path = (root / relative).resolve()
            try:
                manifest_path.relative_to(root)
            except ValueError:
                pass
            else:
                locked_digest = (
                    expected_prelock_sha256
                    if _is_sha256(expected_prelock_sha256)
                    else actual_prelock_sha256
                )
                expected_plans = None
                if prelock_artifact_root is not None:
                    try:
                        expected_plans = prelock_plan_sha256_by_case(
                            prelock_registry, prelock_artifact_root
                        )
                    except (KeyError, TypeError, ValueError, OSError) as error:
                        issues.append(
                            f"pre-lock plan binding failed: {error}"
                        )
                inventory = evaluation_receipt.get("artifact_inventory")
                marker_reference = evaluation_receipt.get("attempt_marker_artifact")
                excluded_evidence = {
                    marker_reference.get("path")
                    if isinstance(marker_reference, dict)
                    else None,
                    evaluation_receipt.get(
                        "locked_collection_completion_artifact", {}
                    ).get("path")
                    if isinstance(
                        evaluation_receipt.get(
                            "locked_collection_completion_artifact"
                        ),
                        dict,
                    )
                    else None,
                }
                locked_inventory = (
                    [item for item in inventory if item not in excluded_evidence]
                    if isinstance(inventory, list)
                    else []
                )
                locked_index, locked_issues = validate_locked_corpus_binding(
                    manifest_path,
                    prelock_registry=prelock_registry,
                    expected_prelock_sha256=locked_digest,
                    artifact_root=root,
                    artifact_inventory=locked_inventory,
                    expected_plan_sha256_by_case=expected_plans,
                )
                issues.extend(locked_issues)
                if (
                    locked_index is not None
                    and evaluation_receipt.get("locked_corpus_manifest_sha256")
                    != locked_index.manifest_sha256
                ):
                    issues.append(
                        "evaluation locked corpus identity differs from its source manifest"
                    )
    return issues


def _fit_observation_scales(
    prelock_registry: dict,
    artifact_root: Path,
    issues: list[str],
) -> dict[str, np.ndarray]:
    references = prelock_registry.get("fit_scaler_artifact_by_case")
    if not isinstance(references, dict) or set(references) != set(CASES):
        issues.append("FIT scaler registry is unavailable for metric verification")
        return {}
    root = artifact_root.resolve()
    scales: dict[str, np.ndarray] = {}
    for case, reference in references.items():
        try:
            path = (root / str(reference["path"])).resolve()
            path.relative_to(root)
            payload = json.loads(
                path.read_text(encoding="ascii"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            observation = payload["observation"]
            values = np.asarray(observation["scale"], dtype=float)
            means = np.asarray(observation["mean"], dtype=float)
            fit_sources = payload["fit_source_sha256"]
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            issues.append(f"FIT scaler artifact is not schema-valid for {case}")
            continue
        if (
            values.shape != (4,)
            or means.shape != (4,)
            or not np.isfinite(values).all()
            or not np.isfinite(means).all()
            or (values <= 0.0).any()
        ):
            issues.append(f"FIT observation scaler is invalid for {case}")
            continue
        if (
            not isinstance(fit_sources, list)
            or not fit_sources
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not str(item[0]).startswith(f"{case}:fit:")
                or not _is_sha256(item[1])
                for item in fit_sources
            )
        ):
            issues.append(f"FIT scaler sources are invalid for {case}")
            continue
        scales[case] = values
    return scales


def _aux_ungated_core_equivalence(
    prelock_registry: dict,
    artifact_root: Path,
    config: StudyConfig,
    selected_update: int,
    issues: list[str],
) -> bool | None:
    references = prelock_registry.get("checkpoint_artifact_by_identity")
    if not isinstance(references, dict):
        issues.append("checkpoint registry is unavailable for core-equivalence check")
        return None
    root = artifact_root.resolve()
    all_equal = True
    for case in CASES:
        for seed in config.confirmatory_seeds:
            hashes: dict[str, str] = {}
            for arm in ("ungated_h8", "aux_h8"):
                identity = f"{case}:seed{seed}:{arm}:u{selected_update:04d}"
                reference = references.get(identity)
                try:
                    path = (root / str(reference["path"])).resolve()
                    path.relative_to(root)
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except (KeyError, TypeError, ValueError, OSError, RuntimeError):
                    issues.append(f"checkpoint is unreadable for core check: {identity}")
                    return None
                expected = {
                    "schema": "boptest-reliability-rssm-checkpoint-v2",
                    "case": case,
                    "model_seed": seed,
                    "arm": arm,
                    "update": selected_update,
                    "config": config.to_dict(),
                }
                if any(payload.get(key) != value for key, value in expected.items()):
                    issues.append(f"checkpoint identity differs for core check: {identity}")
                    return None
                state_dict = payload.get("model_state_dict")
                try:
                    computed_core_hash = core_tensor_state_sha256(state_dict)
                except (AttributeError, TypeError, ValueError):
                    issues.append(
                        f"checkpoint model state is invalid for core check: {identity}"
                    )
                    return None
                if payload.get("core_state_sha256") != computed_core_hash:
                    issues.append(
                        f"checkpoint core-state SHA-256 differs from tensors: {identity}"
                    )
                    return None
                hashes[arm] = computed_core_hash
            all_equal = all_equal and hashes["ungated_h8"] == hashes["aux_h8"]
    return all_equal


def _validate_standardized_metrics(
    rssm_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame | None,
    scales: dict[str, np.ndarray],
    issues: list[str],
) -> None:
    if set(scales) != set(CASES):
        return
    channel_index = {channel: index for index, channel in enumerate(FAULT_CHANNELS)}

    def expected_error(frame: pd.DataFrame, prediction: str) -> np.ndarray:
        scale = np.asarray(
            [scales[case][channel_index[channel]] for case, channel in zip(
                frame["case"], frame["fault_channel"], strict=True
            )]
        )
        return np.abs(
            frame[prediction].to_numpy(dtype=float)
            - frame["target_raw"].to_numpy(dtype=float)
        ) / scale

    rssm_checks = {
        "standardized_abs_error": "prediction_raw",
        "alternate_action_standardized_abs_error": (
            "alternate_action_prediction_raw"
        ),
        "persistence_standardized_abs_error": "persistence_prediction_raw",
    }
    for metric, prediction in rssm_checks.items():
        if not np.allclose(
            rssm_frame[metric].to_numpy(dtype=float),
            expected_error(rssm_frame, prediction),
            rtol=1e-7,
            atol=1e-8,
        ):
            issues.append(f"RSSM {metric} differs from FIT-scaler/raw recomputation")
    action_scale = np.asarray(
        [
            scales[case][channel_index[channel]]
            for case, channel in zip(
                rssm_frame["case"], rssm_frame["fault_channel"], strict=True
            )
        ]
    )
    expected_action_change = np.abs(
        rssm_frame["prediction_raw"].to_numpy(dtype=float)
        - rssm_frame["alternate_action_prediction_raw"].to_numpy(dtype=float)
    ) / action_scale
    if not np.allclose(
        rssm_frame["action_prediction_change_standardized"].to_numpy(dtype=float),
        expected_action_change,
        rtol=1e-7,
        atol=1e-8,
    ):
        issues.append(
            "RSSM action prediction change differs from FIT-scaler/raw recomputation"
        )
    if baseline_frame is not None and not np.allclose(
        baseline_frame["standardized_abs_error"].to_numpy(dtype=float),
        expected_error(baseline_frame, "prediction_raw"),
        rtol=1e-7,
        atol=1e-8,
    ):
        issues.append("baseline standardized error differs from FIT-scaler/raw recomputation")


def _expected_role(stage: Stage) -> str:
    if stage == "development":
        return "validation"
    if stage == "confirmation":
        return "locked_test"
    raise ValueError(f"unknown study stage: {stage}")


def _expected_seeds(config: StudyConfig, stage: Stage) -> tuple[int, ...]:
    return (
        config.development_seeds
        if stage == "development"
        else config.confirmatory_seeds
    )


def _expected_trajectory_count(stage: Stage) -> int:
    return 8 if stage == "development" else 12


def _expected_fault_grid(role: str) -> set[tuple]:
    return set(fault_cell_signatures(FaultSpec(), role))


def _validate_cases_and_trajectories(frame: pd.DataFrame, stage: Stage) -> None:
    expected_cases = set(CASES)
    if set(frame["case"]) != expected_cases:
        raise ValueError("evaluation does not contain exactly the three frozen cases")
    expected_count = _expected_trajectory_count(stage)
    trajectories = frame[["case", "trajectory_day", "trajectory_seed"]].drop_duplicates()
    counts = trajectories.groupby("case").size()
    if set(counts.index) != expected_cases or not (counts == expected_count).all():
        raise ValueError(
            f"each case must contain exactly {expected_count} whole trajectories"
        )


def _validate_complete_grid(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> None:
    grid_columns = (
        "fault_channel",
        "family",
        "sign",
        "severity",
        "severity_unit",
        "onset",
        "anchor",
        "horizon",
    )
    roles = set(frame["role"])
    if len(roles) != 1:
        raise ValueError("a frozen-grid table must contain exactly one role")
    expected = _expected_fault_grid(next(iter(roles)))
    actual = {
        tuple(row)
        for row in frame.loc[:, grid_columns].drop_duplicates().itertuples(
            index=False, name=None
        )
    }
    if actual != expected:
        raise ValueError("evaluation rows differ from the frozen fault/anchor grid")
    identity = [*group_columns, *grid_columns]
    if frame.duplicated(identity).any():
        raise ValueError("evaluation contains a duplicate frozen-grid row")
    counts = frame.groupby(list(group_columns), dropna=False).size()
    if counts.empty or not (counts == len(expected)).all():
        raise ValueError("at least one model/trajectory group has an incomplete grid")


def _validate_rssm_frame(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    stage: Stage,
    selected_update: int,
) -> pd.DataFrame:
    missing = sorted(_REQUIRED_RSSM_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"RSSM evaluation is missing columns: {missing}")
    if selected_update not in config.validation_checkpoints:
        raise ValueError("selected update is outside the frozen checkpoint grid")
    updates = set(frame["update"])
    if not updates.issubset(set(config.validation_checkpoints)):
        raise ValueError("RSSM evaluation contains an update outside the frozen grid")
    role = _expected_role(stage)
    if set(frame["role"]) != {role}:
        raise ValueError(f"{stage} gate accepts {role} rows only")
    selected = frame.loc[frame["update"] == selected_update].copy()
    if selected.empty:
        raise ValueError("RSSM evaluation does not contain the common selected update")
    if stage == "confirmation" and updates != {selected_update}:
        raise ValueError("confirmation may evaluate the common selected update only")
    if set(selected["arm"]) != set(ARMS):
        raise ValueError("RSSM evaluation does not contain exactly the five frozen arms")
    if set(selected["model_seed"]) != set(_expected_seeds(config, stage)):
        raise ValueError("RSSM evaluation does not contain the frozen stage seeds")
    if set(selected["horizon"]) != {config.direct_horizon}:
        raise ValueError("RSSM evaluation is not the frozen H8 endpoint")
    numeric = selected[
        [
            "standardized_abs_error",
            "persistence_standardized_abs_error",
            "target_raw",
            "prediction_raw",
            "alternate_action_prediction_raw",
            "alternate_action_standardized_abs_error",
            "action_prediction_change_standardized",
            "persistence_prediction_raw",
            "severity",
        ]
    ].to_numpy(dtype=float)
    nonnegative_columns = (0, 1, 5, 6)
    if not np.isfinite(numeric).all() or any(
        (numeric[:, index] < 0).any() for index in nonnegative_columns
    ):
        raise ValueError("RSSM evaluation contains invalid error values")

    _validate_cases_and_trajectories(selected, stage)
    _validate_complete_grid(
        selected,
        group_columns=(
            "case",
            "trajectory_day",
            "trajectory_seed",
            "model_seed",
            "arm",
        ),
    )

    trajectories = selected[
        ["case", "trajectory_day", "trajectory_seed"]
    ].drop_duplicates()
    for case, case_trajectories in trajectories.groupby("case", sort=False):
        expected = {
            (case, int(day), int(seed), int(model_seed), arm)
            for day, seed in case_trajectories[
                ["trajectory_day", "trajectory_seed"]
            ].itertuples(index=False, name=None)
            for model_seed in _expected_seeds(config, stage)
            for arm in ARMS
        }
        actual = {
            tuple(row)
            for row in selected.loc[
                selected["case"] == case,
                [
                    "case",
                    "trajectory_day",
                    "trajectory_seed",
                    "model_seed",
                    "arm",
                ],
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        if actual != expected:
            raise ValueError("RSSM arm/seed/trajectory crossing is incomplete")
    return selected


def matched_arm_metrics(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    stage: Stage,
    selected_update: int,
) -> pd.DataFrame:
    """Validate and pair every RSSM arm at the same frozen H8 anchor."""
    selected = _validate_rssm_frame(
        frame, config, stage=stage, selected_update=selected_update
    )
    persistence_counts = selected.groupby(list(_PAIR_KEYS), dropna=False)[
        "persistence_standardized_abs_error"
    ].nunique(dropna=False)
    if not (persistence_counts == 1).all():
        raise ValueError("paired arms disagree on the persistence comparator")
    persistence = selected.groupby(list(_PAIR_KEYS), as_index=False, dropna=False)[
        "persistence_standardized_abs_error"
    ].first()
    persistence = persistence.rename(
        columns={"persistence_standardized_abs_error": "persistence"}
    )
    target_counts = selected.groupby(list(_PAIR_KEYS), dropna=False)[
        "target_raw"
    ].nunique(dropna=False)
    if not (target_counts == 1).all():
        raise ValueError("paired arms disagree on the clean raw target")
    model_independent_keys = [
        key for key in _PAIR_KEYS if key not in ("model_seed", "cell_id")
    ]
    cell_ids = selected.groupby(model_independent_keys, dropna=False)[
        "cell_id"
    ].nunique(dropna=False)
    if not (cell_ids == 1).all():
        raise ValueError("cell identity differs across model seeds or arms")
    for column in (
        "target_raw",
        "persistence_prediction_raw",
        "persistence_standardized_abs_error",
    ):
        counts = selected.groupby(model_independent_keys, dropna=False)[
            column
        ].nunique(dropna=False)
        if not (counts == 1).all():
            raise ValueError(f"model-independent {column} differs across seeds")
    selected["raw_abs_error"] = (
        selected["prediction_raw"] - selected["target_raw"]
    ).abs()
    selected["persistence_raw_abs_error"] = (
        selected["persistence_prediction_raw"] - selected["target_raw"]
    ).abs()
    paired = selected.pivot(
        index=list(_PAIR_KEYS), columns="arm", values="standardized_abs_error"
    ).reset_index()
    paired.columns.name = None
    if paired[list(ARMS)].isna().any().any():
        raise ValueError("at least one frozen H8 anchor is not paired across arms")
    paired = paired.merge(
        persistence, on=list(_PAIR_KEYS), how="inner", validate="one_to_one"
    )
    targets = selected.groupby(list(_PAIR_KEYS), as_index=False, dropna=False)[
        "target_raw"
    ].first()
    paired = paired.merge(
        targets, on=list(_PAIR_KEYS), how="inner", validate="one_to_one"
    )
    persistence_raw_counts = selected.groupby(list(_PAIR_KEYS), dropna=False)[
        "persistence_raw_abs_error"
    ].nunique(dropna=False)
    if not (persistence_raw_counts == 1).all():
        raise ValueError("paired arms disagree on raw persistence error")
    persistence_raw = selected.groupby(
        list(_PAIR_KEYS), as_index=False, dropna=False
    )["persistence_raw_abs_error"].first()
    paired = paired.merge(
        persistence_raw, on=list(_PAIR_KEYS), how="inner", validate="one_to_one"
    )
    extra_metrics = {
        "raw_abs_error": "raw_abs_error",
        "prediction_raw": "prediction_raw",
        "alternate_action_prediction_raw": "alternate_prediction_raw",
        "alternate_action_standardized_abs_error": "alternate_error",
        "action_prediction_change_standardized": "action_prediction_change",
    }
    for source, suffix in extra_metrics.items():
        wide = selected.pivot(
            index=list(_PAIR_KEYS), columns="arm", values=source
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(
            columns={arm: f"{arm}_{suffix}" for arm in ARMS}
        )
        paired = paired.merge(
            wide, on=list(_PAIR_KEYS), how="inner", validate="one_to_one"
        )
    paired["ungated_minus_gated"] = paired["ungated_h8"] - paired["gated_h8"]
    return paired.sort_values(list(_PAIR_KEYS), kind="stable").reset_index(drop=True)


def _equal_weight_values(
    paired: pd.DataFrame,
    *,
    families: Sequence[str],
    value_columns: Sequence[str],
    retain: Sequence[str] = (),
) -> pd.DataFrame:
    selected = paired.loc[paired["family"].isin(tuple(families))].copy()
    if selected.empty:
        raise ValueError("equal-weight aggregate selection is empty")
    retain = tuple(retain)
    value_columns = tuple(value_columns)
    if not value_columns or not set(value_columns).issubset(selected.columns):
        raise ValueError("equal-weight value columns are empty or missing")
    dimensions = tuple(dimension for dimension in _EQUAL_DIMENSIONS if dimension not in retain)
    group = [*_CLUSTERS, *retain, *dimensions]
    scores = selected.groupby(group, as_index=False, dropna=False)[list(value_columns)].mean()
    for dimension in reversed(dimensions):
        remaining = [column for column in group if column != dimension]
        scores = scores.groupby(remaining, as_index=False, dropna=False)[
            list(value_columns)
        ].mean()
        group = remaining
    return scores.sort_values(group, kind="stable").reset_index(drop=True)


def _equal_weight_scores(
    paired: pd.DataFrame,
    *,
    families: Sequence[str],
    retain: Sequence[str] = (),
) -> pd.DataFrame:
    return _equal_weight_values(
        paired,
        families=families,
        value_columns=_SCORE_COLUMNS,
        retain=retain,
    )


def _validate_bootstrap_grid(scores: pd.DataFrame, expected_seeds: Sequence[int]) -> None:
    if set(scores["model_seed"]) != set(expected_seeds):
        raise ValueError("bootstrap scores do not contain the frozen model seeds")
    for case, case_rows in scores.groupby("case", sort=False):
        trajectories = case_rows[["trajectory_day", "trajectory_seed"]].drop_duplicates()
        expected = {
            (int(seed), int(day), int(trajectory_seed))
            for seed in expected_seeds
            for day, trajectory_seed in trajectories.itertuples(index=False, name=None)
        }
        actual = {
            tuple(row)
            for row in case_rows[
                ["model_seed", "trajectory_day", "trajectory_seed"]
            ].itertuples(index=False, name=None)
        }
        if actual != expected:
            raise ValueError(f"bootstrap seed/trajectory grid is incomplete for {case}")


def _hierarchical_bootstrap_scores(
    scores: pd.DataFrame,
    config: StudyConfig,
    *,
    expected_seeds: Sequence[int],
    value_columns: Sequence[str] = _SCORE_COLUMNS,
) -> dict[str, np.ndarray]:
    """Resample cases and seeds as crossed clusters, then trajectories within case."""
    value_columns = tuple(value_columns)
    if not value_columns or not set(value_columns).issubset(scores.columns):
        raise ValueError("bootstrap value columns are empty or missing")
    _validate_bootstrap_grid(scores, expected_seeds)
    cases = tuple(sorted(scores["case"].unique()))
    seeds = tuple(sorted(expected_seeds))
    matrices: dict[str, dict[str, np.ndarray]] = {}
    trajectory_counts: dict[str, int] = {}
    for case in cases:
        case_rows = scores.loc[scores["case"] == case]
        trajectories = sorted(
            set(
                case_rows[["trajectory_day", "trajectory_seed"]].itertuples(
                    index=False, name=None
                )
            )
        )
        trajectory_counts[case] = len(trajectories)
        trajectory_index = {key: index for index, key in enumerate(trajectories)}
        matrices[case] = {}
        for column in value_columns:
            matrix = np.empty((len(seeds), len(trajectories)), dtype=float)
            for row in case_rows.itertuples(index=False):
                seed_index = seeds.index(int(row.model_seed))
                trajectory = (int(row.trajectory_day), int(row.trajectory_seed))
                matrix[seed_index, trajectory_index[trajectory]] = float(
                    getattr(row, column)
                )
            if not np.isfinite(matrix).all():
                raise ValueError("bootstrap matrix contains a non-finite value")
            matrices[case][column] = matrix

    rng = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    draws = {
        column: np.empty(config.bootstrap_draws, dtype=float)
        for column in value_columns
    }
    for draw in range(config.bootstrap_draws):
        sampled_cases = rng.integers(0, len(cases), size=len(cases))
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        case_values = {column: [] for column in value_columns}
        for case_index in sampled_cases:
            case = cases[int(case_index)]
            sampled_trajectories = rng.integers(
                0,
                trajectory_counts[case],
                size=trajectory_counts[case],
            )
            for column in value_columns:
                matrix = matrices[case][column]
                case_values[column].append(
                    float(matrix[np.ix_(sampled_seeds, sampled_trajectories)].mean())
                )
        for column in value_columns:
            draws[column][draw] = float(np.mean(case_values[column]))
    return draws


def _severity_level_scores(paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired.loc[paired["family"].isin(("bias", "drift"))].copy()
    lookup = selected[
        ["family", "fault_channel", "severity"]
    ].drop_duplicates()
    counts = lookup.groupby(["family", "fault_channel"])["severity"].nunique()
    if not (counts == 2).all():
        raise ValueError("bias/drift must each contain exactly two frozen severities")
    lookup["severity_level"] = lookup.groupby(
        ["family", "fault_channel"]
    )["severity"].rank(method="dense").astype(int)
    lookup["severity_level"] = lookup["severity_level"].map({1: "low", 2: "high"})
    selected = selected.merge(
        lookup,
        on=["family", "fault_channel", "severity"],
        how="left",
        validate="many_to_one",
    )
    return _equal_weight_scores(
        selected,
        families=("bias", "drift"),
        retain=("severity_level",),
    )


def _raw_mae_by_channel(
    paired: pd.DataFrame, families: Sequence[str]
) -> dict[str, dict]:
    scores = _equal_weight_values(
        paired,
        families=families,
        value_columns=_RAW_SCORE_COLUMNS,
        retain=("fault_channel",),
    )
    units = {
        "zone_temperature_k": "K",
        "hvac_electric_power_w": "W",
    }
    result: dict[str, dict] = {}
    for channel, group in scores.groupby("fault_channel"):
        means = group[list(_RAW_SCORE_COLUMNS)].mean()
        result[channel] = {
            "unit": units[channel],
            "mae": {column: float(means[column]) for column in _RAW_SCORE_COLUMNS},
        }
    return result


def _dropout_summary(paired: pd.DataFrame) -> dict:
    standardized = _equal_weight_scores(paired, families=("dropout",))
    point = standardized[list(_SCORE_COLUMNS)].mean()
    return {
        "standardized_mae": {
            column: float(point[column]) for column in _SCORE_COLUMNS
        },
        "raw_mae_by_channel": _raw_mae_by_channel(paired, ("dropout",)),
    }


def _action_use_summary(paired: pd.DataFrame) -> tuple[dict, bool]:
    alternate_columns = tuple(f"{arm}_alternate_error" for arm in ARMS)
    change_columns = tuple(f"{arm}_action_prediction_change" for arm in ARMS)
    scores = _equal_weight_values(
        paired,
        families=("healthy",),
        value_columns=(*ARMS, *alternate_columns, *change_columns),
    )
    report: dict[str, dict] = {}
    primary_passes: list[bool] = []
    for arm in ARMS:
        alternate = f"{arm}_alternate_error"
        change = f"{arm}_action_prediction_change"
        by_case = scores.groupby("case")[[arm, alternate, change]].mean()
        case_report = {
            case: {
                "realized_action_mae": float(row[arm]),
                "alternate_action_mae": float(row[alternate]),
                "prediction_change": float(row[change]),
                "realized_better": bool(row[arm] < row[alternate]),
                "nonzero_prediction_change": bool(row[change] > 0.0),
            }
            for case, row in by_case.iterrows()
        }
        report[arm] = {
            "aggregate_realized_action_mae": float(scores[arm].mean()),
            "aggregate_alternate_action_mae": float(scores[alternate].mean()),
            "aggregate_prediction_change": float(scores[change].mean()),
            "aggregate_realized_better": bool(
                scores[arm].mean() < scores[alternate].mean()
            ),
            "by_case": case_report,
        }
        if arm == "gated_h8":
            primary_passes.append(
                bool(scores[arm].mean() < scores[alternate].mean())
                and all(
                    item["nonzero_prediction_change"]
                    for item in case_report.values()
                )
            )
    return report, all(primary_passes)


def _baseline_scores(
    baseline_frame: pd.DataFrame | None,
    config: StudyConfig,
    *,
    stage: Stage,
    reference_cells: pd.DataFrame,
) -> tuple[
    dict[str, float],
    tuple[str, ...],
    dict[str, pd.DataFrame],
    dict[str, dict],
]:
    if baseline_frame is None:
        return {}, COMPETENCE_BASELINE_ARMS, {}, {}
    required = {
        "case",
        "role",
        "trajectory_day",
        "trajectory_seed",
        "arm",
        "cell_id",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "anchor",
        "horizon",
        "target_raw",
        "prediction_raw",
        "standardized_abs_error",
    }
    missing_columns = sorted(required - set(baseline_frame.columns))
    if missing_columns:
        raise ValueError(f"baseline evaluation is missing columns: {missing_columns}")
    missing_arms = tuple(
        arm for arm in COMPETENCE_BASELINE_ARMS if arm not in set(baseline_frame["arm"])
    )
    available_arms = tuple(
        arm for arm in COMPETENCE_BASELINE_ARMS if arm in set(baseline_frame["arm"])
    )
    if not available_arms:
        return {}, missing_arms, {}, {}
    selected = baseline_frame.loc[baseline_frame["arm"].isin(available_arms)].copy()
    role = _expected_role(stage)
    if set(selected["role"]) != {role}:
        raise ValueError(f"{stage} baseline gate accepts {role} rows only")
    if set(selected["horizon"]) != {8}:
        raise ValueError("baseline evaluation is not the frozen H8 endpoint")
    numeric = selected[
        ["standardized_abs_error", "target_raw", "prediction_raw", "severity"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 0] < 0).any():
        raise ValueError("baseline evaluation contains invalid values")
    _validate_cases_and_trajectories(selected, stage)
    cell_identity = [
        "case",
        "role",
        "trajectory_day",
        "trajectory_seed",
        "cell_id",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "anchor",
        "horizon",
    ]
    expected_cells = {
        tuple(row)
        for row in reference_cells[cell_identity]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    actual_cells = {
        tuple(row)
        for row in selected[cell_identity]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if actual_cells != expected_cells:
        raise ValueError("baseline and RSSM evaluations use different frozen cells")
    reference_targets = reference_cells[
        [*cell_identity, "target_raw"]
    ].drop_duplicates()
    if reference_targets.duplicated(cell_identity).any():
        raise ValueError("RSSM cells disagree on their model-independent raw target")
    baseline_target_counts = selected.groupby(cell_identity, dropna=False)[
        "target_raw"
    ].nunique(dropna=False)
    if not (baseline_target_counts == 1).all():
        raise ValueError("baseline raw targets differ across models or seeds")
    baseline_targets = selected.groupby(
        cell_identity, as_index=False, dropna=False
    )["target_raw"].first()
    target_comparison = reference_targets.merge(
        baseline_targets,
        on=cell_identity,
        how="inner",
        validate="one_to_one",
        suffixes=("_rssm", "_baseline"),
    )
    if len(target_comparison) != len(reference_targets) or not np.allclose(
        target_comparison["target_raw_rssm"].to_numpy(),
        target_comparison["target_raw_baseline"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("baseline and RSSM evaluations use different raw targets")
    if "model_seed" not in selected:
        selected["model_seed"] = 0
    else:
        seeds = selected["model_seed"].fillna(0).to_numpy(dtype=float)
        if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
            raise ValueError("baseline model seeds must be finite integers")
        selected["model_seed"] = seeds.astype(int)
    _validate_complete_grid(
        selected,
        group_columns=(
            "case",
            "trajectory_day",
            "trajectory_seed",
            "model_seed",
            "arm",
        ),
    )
    trajectories = selected[
        ["case", "trajectory_day", "trajectory_seed"]
    ].drop_duplicates()
    expected_gru_seeds = set(_expected_seeds(config, stage))
    for arm in available_arms:
        arm_rows = selected.loc[selected["arm"] == arm]
        arm_seeds = tuple(sorted(arm_rows["model_seed"].unique()))
        expected_arm_seeds = (
            expected_gru_seeds if arm == "deterministic_gru" else {0}
        )
        if set(arm_seeds) != expected_arm_seeds:
            raise ValueError(f"{arm} does not contain its frozen baseline seed set")
        for case, case_trajectories in trajectories.groupby("case", sort=False):
            expected = {
                (case, int(day), int(seed), int(model_seed), arm)
                for day, seed in case_trajectories[
                    ["trajectory_day", "trajectory_seed"]
                ].itertuples(index=False, name=None)
                for model_seed in arm_seeds
            }
            actual = {
                tuple(row)
                for row in arm_rows.loc[
                    arm_rows["case"] == case,
                    [
                        "case",
                        "trajectory_day",
                        "trajectory_seed",
                        "model_seed",
                        "arm",
                    ],
                ]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            }
            if actual != expected:
                raise ValueError("baseline arm/seed/trajectory crossing is incomplete")
    values: dict[str, float] = {}
    cluster_scores: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict] = {}
    selected["raw_abs_error"] = (
        selected["prediction_raw"] - selected["target_raw"]
    ).abs()
    for arm in available_arms:
        all_arm_rows = selected.loc[selected["arm"] == arm].copy()
        arm_rows = all_arm_rows.loc[
            all_arm_rows["family"].isin(SILENT_FAMILIES)
        ].copy()
        group = [
            "case",
            "model_seed",
            "trajectory_day",
            "trajectory_seed",
            *_EQUAL_DIMENSIONS,
        ]
        scores = arm_rows.groupby(group, as_index=False, dropna=False)[
            "standardized_abs_error"
        ].mean()
        for dimension in reversed(_EQUAL_DIMENSIONS):
            remaining = [column for column in group if column != dimension]
            scores = scores.groupby(remaining, as_index=False, dropna=False)[
                "standardized_abs_error"
            ].mean()
            group = remaining
        values[arm] = float(scores["standardized_abs_error"].mean())
        cluster_scores[arm] = scores.rename(
            columns={"standardized_abs_error": "baseline_error"}
        )
        arm_diagnostics: dict[str, dict] = {}
        for scope, families in {
            "silent_faults": SILENT_FAMILIES,
            "healthy": ("healthy",),
            "dropout": ("dropout",),
        }.items():
            raw_scores = _equal_weight_values(
                all_arm_rows,
                families=families,
                value_columns=("raw_abs_error",),
                retain=("fault_channel",),
            )
            arm_diagnostics[scope] = {
                channel: float(group["raw_abs_error"].mean())
                for channel, group in raw_scores.groupby("fault_channel")
            }
        dropout_standardized = _equal_weight_values(
            all_arm_rows,
            families=("dropout",),
            value_columns=("standardized_abs_error",),
        )
        arm_diagnostics["dropout_standardized_mae"] = float(
            dropout_standardized["standardized_abs_error"].mean()
        )
        diagnostics[arm] = arm_diagnostics
    return values, missing_arms, cluster_scores, diagnostics


def _competence_statistics(
    silent_scores: pd.DataFrame,
    *,
    rssm_arm: str,
    baseline_arm: str,
    baseline_clusters: pd.DataFrame,
    expected_seeds: Sequence[int],
    config: StudyConfig,
) -> dict:
    rssm = silent_scores[
        [*_CLUSTERS, rssm_arm]
    ].rename(columns={rssm_arm: "rssm_error"})
    baseline = baseline_clusters.copy()
    baseline_seeds = set(baseline["model_seed"])
    if baseline_seeds == {0}:
        baseline = baseline.drop(columns="model_seed").merge(
            pd.DataFrame({"model_seed": tuple(expected_seeds)}), how="cross"
        )
    elif baseline_seeds != set(expected_seeds):
        raise ValueError("competence baseline cannot be paired to the frozen RSSM seeds")
    paired = rssm.merge(
        baseline,
        on=list(_CLUSTERS),
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(rssm) or len(paired) != len(baseline):
        raise ValueError("competence comparison is not paired on every seed/trajectory")
    baseline_point = float(paired["baseline_error"].mean())
    if baseline_point <= 0.0:
        raise ValueError("competence baseline MAE denominator must be positive")
    aggregate_ratio = float(paired["rssm_error"].mean() / baseline_point)
    by_case = paired.groupby("case")[["rssm_error", "baseline_error"]].mean()
    if (by_case["baseline_error"] <= 0.0).any():
        raise ValueError("per-case competence denominator must be positive")
    case_ratios = {
        case: float(row["rssm_error"] / row["baseline_error"])
        for case, row in by_case.iterrows()
    }
    draws = _hierarchical_bootstrap_scores(
        paired,
        config,
        expected_seeds=expected_seeds,
        value_columns=("rssm_error", "baseline_error"),
    )
    if (draws["baseline_error"] <= 0.0).any():
        raise ValueError("bootstrap competence denominator must be positive")
    ratio_draws = draws["rssm_error"] / draws["baseline_error"]
    ratio_ci = np.quantile(ratio_draws, (0.025, 0.975))
    return {
        "rssm_arm": rssm_arm,
        "baseline_arm": baseline_arm,
        "aggregate_ratio": aggregate_ratio,
        "ratio_ci95": [float(ratio_ci[0]), float(ratio_ci[1])],
        "case_ratios": case_ratios,
    }


def evaluate_study_gate(
    rssm_frame: pd.DataFrame,
    config: StudyConfig,
    *,
    stage: Stage,
    selected_update: int,
    baseline_frame: pd.DataFrame | None = None,
    prelock_registry: dict | None = None,
    prelock_artifact_root: Path | None = None,
    expected_prelock_sha256: str | None = None,
    evaluation_receipt: dict | None = None,
    evaluation_artifact_root: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate the frozen reliability gate without permitting a development paper-pass."""
    paired = matched_arm_metrics(
        rssm_frame,
        config,
        stage=stage,
        selected_update=selected_update,
    )
    aux_output_equivalence = bool(
        np.array_equal(
            paired["aux_h8_prediction_raw"].to_numpy(),
            paired["ungated_h8_prediction_raw"].to_numpy(),
        )
        and np.array_equal(
            paired["aux_h8_alternate_prediction_raw"].to_numpy(),
            paired["ungated_h8_alternate_prediction_raw"].to_numpy(),
        )
        and np.array_equal(
            paired["aux_h8"].to_numpy(),
            paired["ungated_h8"].to_numpy(),
        )
    )
    expected_seeds = _expected_seeds(config, stage)
    silent_scores = _equal_weight_scores(paired, families=SILENT_FAMILIES)
    point = silent_scores[list(_SCORE_COLUMNS)].mean()
    if point["ungated_h8"] <= 0.0:
        raise ValueError("primary ungated MAE denominator must be positive")
    raw_improvement = float(point["ungated_h8"] - point["gated_h8"])
    relative_improvement = raw_improvement / float(point["ungated_h8"])
    silent_draws = _hierarchical_bootstrap_scores(
        silent_scores, config, expected_seeds=expected_seeds
    )
    delta_draws = silent_draws["ungated_h8"] - silent_draws["gated_h8"]
    raw_ci = np.quantile(delta_draws, (0.025, 0.975))

    by_case_scores = silent_scores.groupby("case")[list(_SCORE_COLUMNS)].mean()
    by_case = {
        case: float(row["ungated_h8"] - row["gated_h8"])
        for case, row in by_case_scores.iterrows()
    }
    family_scores = _equal_weight_scores(
        paired, families=SILENT_FAMILIES, retain=("family",)
    )
    by_family = {
        family: float(group["ungated_h8"].mean() - group["gated_h8"].mean())
        for family, group in family_scores.groupby("family")
    }
    severity_scores = _severity_level_scores(paired)
    by_severity = {
        level: float(group["ungated_h8"].mean() - group["gated_h8"].mean())
        for level, group in severity_scores.groupby("severity_level")
    }
    by_seed_scores = silent_scores.groupby("model_seed")[list(_SCORE_COLUMNS)].mean()
    by_seed = {
        str(int(seed)): float(row["ungated_h8"] - row["gated_h8"])
        for seed, row in by_seed_scores.iterrows()
    }
    positive_seed_count = sum(value > 0.0 for value in by_seed.values())

    healthy_scores = _equal_weight_scores(paired, families=("healthy",))
    healthy_point = healthy_scores[list(_SCORE_COLUMNS)].mean()
    if healthy_point["ungated_h8"] <= 0.0:
        raise ValueError("healthy ungated MAE denominator must be positive")
    healthy_degradation = float(
        (healthy_point["gated_h8"] - healthy_point["ungated_h8"])
        / healthy_point["ungated_h8"]
    )
    healthy_draws = _hierarchical_bootstrap_scores(
        healthy_scores, config, expected_seeds=expected_seeds
    )
    if (healthy_draws["ungated_h8"] <= 0.0).any():
        raise ValueError("healthy bootstrap produced a nonpositive denominator")
    healthy_relative_draws = (
        healthy_draws["gated_h8"] - healthy_draws["ungated_h8"]
    ) / healthy_draws["ungated_h8"]
    healthy_ci = np.quantile(healthy_relative_draws, (0.025, 0.975))

    persistence_by_case = {
        case: {
            "ungated_h8": float(row["ungated_h8"]),
            "gated_h8": float(row["gated_h8"]),
            "persistence": float(row["persistence"]),
            "both_beat_persistence": bool(
                row["ungated_h8"] < row["persistence"]
                and row["gated_h8"] < row["persistence"]
            ),
        }
        for case, row in by_case_scores.iterrows()
    }
    aux_improvement = float(point["aux_h8"] - point["gated_h8"])
    if point["huber_h8"] <= 0.0:
        raise ValueError("Huber MAE denominator must be positive")
    gated_vs_huber_relative_excess = float(
        (point["gated_h8"] - point["huber_h8"]) / point["huber_h8"]
    )
    raw_unit_mae = {
        "silent_faults": _raw_mae_by_channel(paired, SILENT_FAMILIES),
        "healthy": _raw_mae_by_channel(paired, ("healthy",)),
    }
    dropout = _dropout_summary(paired)
    action_use, primary_action_use_pass = _action_use_summary(paired)

    (
        comparator_scores,
        missing_comparators,
        comparator_clusters,
        baseline_diagnostics,
    ) = _baseline_scores(
        baseline_frame,
        config,
        stage=stage,
        reference_cells=paired,
    )
    provenance_issues = (
        validate_confirmation_provenance(
            prelock_registry,
            evaluation_receipt,
            rssm_frame,
            baseline_frame,
            config,
            selected_update=selected_update,
            prelock_artifact_root=prelock_artifact_root,
            evaluation_artifact_root=evaluation_artifact_root,
            expected_prelock_sha256=expected_prelock_sha256,
        )
        if stage == "confirmation"
        else []
    )
    aux_core_equivalence: bool | None = None
    if (
        stage == "confirmation"
        and isinstance(prelock_registry, dict)
        and prelock_artifact_root is not None
    ):
        fit_scales = _fit_observation_scales(
            prelock_registry, prelock_artifact_root, provenance_issues
        )
        _validate_standardized_metrics(
            rssm_frame, baseline_frame, fit_scales, provenance_issues
        )
        aux_core_equivalence = _aux_ungated_core_equivalence(
            prelock_registry,
            prelock_artifact_root,
            config,
            selected_update,
            provenance_issues,
        )
    aux_equivalence = (
        None
        if stage == "development" or aux_core_equivalence is None
        else bool(aux_core_equivalence and aux_output_equivalence)
    )
    rssm_scores = {arm: float(point[arm]) for arm in ARMS}
    strongest_rssm_arm = min(rssm_scores, key=rssm_scores.get)
    strongest_rssm_score = rssm_scores[strongest_rssm_arm]
    if not comparator_scores:
        competence = None
        strongest_comparator_arm = None
        strongest_comparator_score = None
    else:
        strongest_comparator_arm = min(comparator_scores, key=comparator_scores.get)
        strongest_comparator_score = comparator_scores[strongest_comparator_arm]
        competence = _competence_statistics(
            silent_scores,
            rssm_arm="gated_h8",
            baseline_arm=strongest_comparator_arm,
            baseline_clusters=comparator_clusters[strongest_comparator_arm],
            expected_seeds=expected_seeds,
            config=config,
        )
    competence_pass = (
        None
        if competence is None or missing_comparators
        else bool(
            competence["aggregate_ratio"]
            <= CONFIRMATION_BASELINE_POINT_RATIO_LIMIT
            and competence["ratio_ci95"][1]
            <= CONFIRMATION_BASELINE_CI_RATIO_LIMIT
            and all(
                ratio <= CONFIRMATION_BASELINE_CASE_RATIO_LIMIT
                for ratio in competence["case_ratios"].values()
            )
        )
    )

    checks: dict[str, bool | None] = {
        "silent_relative_improvement_at_least_10pct": bool(
            relative_improvement >= config.relative_improvement_threshold
        ),
        "silent_raw_improvement_ci_lower_above_zero": bool(raw_ci[0] > 0.0),
        "every_case_positive": bool(all(value > 0.0 for value in by_case.values())),
        "bias_drift_stuck_each_positive": bool(
            set(by_family) == set(SILENT_FAMILIES)
            and all(value > 0.0 for value in by_family.values())
        ),
        "both_frozen_severity_levels_positive": bool(
            set(by_severity) == {"low", "high"}
            and all(value > 0.0 for value in by_severity.values())
        ),
        "at_least_four_of_five_paired_seeds_positive": (
            bool(positive_seed_count >= 4) if stage == "confirmation" else None
        ),
        "healthy_point_degradation_at_most_5pct": bool(
            healthy_degradation <= config.healthy_degradation_limit
        ),
        "healthy_relative_ci_upper_at_most_5pct": bool(
            healthy_ci[1] <= config.healthy_degradation_limit
        ),
        "ungated_and_gated_each_beat_persistence_in_every_case": bool(
            all(item["both_beat_persistence"] for item in persistence_by_case.values())
        ),
        "gated_improves_over_auxiliary_only": (
            None
            if aux_equivalence is None
            else bool(aux_equivalence and aux_improvement > 0.0)
        ),
        "gated_within_2pct_of_huber": bool(
            gated_vs_huber_relative_excess <= config.huber_noninferiority_limit
        ),
        "rssm_meets_frozen_baseline_competence_bounds": competence_pass,
        "gated_h8_passes_healthy_action_use_diagnostic": (
            primary_action_use_pass
        ),
        "confirmation_provenance_integrity_bound": (
            None
            if stage == "development" or provenance_issues
            else True
        ),
    }
    development_screen_checks: dict[str, bool | None] = {
        "silent_relative_improvement_at_least_7p5pct": bool(
            relative_improvement >= DEVELOPMENT_IMPROVEMENT_THRESHOLD
        ),
        "every_case_positive": checks["every_case_positive"],
        "bias_drift_stuck_each_positive": checks[
            "bias_drift_stuck_each_positive"
        ],
        "at_least_two_of_three_paired_seeds_positive": bool(
            positive_seed_count >= 2
        ),
        "healthy_point_degradation_at_most_5pct": checks[
            "healthy_point_degradation_at_most_5pct"
        ],
        "ungated_and_gated_each_beat_persistence_in_every_case": checks[
            "ungated_and_gated_each_beat_persistence_in_every_case"
        ],
        "gated_within_1p15x_strongest_completed_baseline": (
            None
            if competence is None or missing_comparators
            else bool(
                float(point["gated_h8"]) / strongest_comparator_score
                <= DEVELOPMENT_BASELINE_RATIO_LIMIT
            )
        ),
    }
    screen_evaluable = all(
        value is not None for value in development_screen_checks.values()
    )
    screen_pass = bool(
        screen_evaluable
        and all(bool(value) for value in development_screen_checks.values())
    )
    evaluable = all(value is not None for value in checks.values())
    gate_pass = bool(
        stage == "confirmation"
        and evaluable
        and all(bool(value) for value in checks.values())
    )
    if stage == "development":
        decision = (
            "SCREEN_GO"
            if screen_pass
            else "SCREEN_STOP" if screen_evaluable else "INCOMPLETE"
        )
    elif not evaluable:
        decision = "INCOMPLETE"
    else:
        decision = "PASS" if gate_pass else "STOP"

    result = {
        "schema": GATE_SCHEMA,
        "stage": stage,
        "role": _expected_role(stage),
        "selected_update": selected_update,
        "decision": decision,
        "paper_claim_allowed": gate_pass,
        "gate_pass": gate_pass,
        "confirmatory_conditions_evaluable": evaluable,
        "development_screen": {
            "paper_claim_allowed": False,
            "evaluable": screen_evaluable,
            "screen_pass": screen_pass,
            "checks": development_screen_checks,
        },
        "bootstrap": {
            "method": "paired hierarchical percentile bootstrap",
            "clusters": [
                "case",
                "model_seed",
                "whole_trajectory_nested_within_case",
            ],
            "case_seed_structure": "crossed",
            "draws": config.bootstrap_draws,
            "seed": config.bootstrap_seed,
        },
        "primary": {
            "ungated_h8_mae": float(point["ungated_h8"]),
            "gated_h8_mae": float(point["gated_h8"]),
            "raw_improvement": raw_improvement,
            "relative_improvement": relative_improvement,
            "raw_improvement_ci95": [float(raw_ci[0]), float(raw_ci[1])],
            "case_raw_improvement": by_case,
            "family_raw_improvement": by_family,
            "severity_level_raw_improvement": by_severity,
            "seed_raw_improvement": by_seed,
            "positive_seed_count": positive_seed_count,
        },
        "healthy": {
            "ungated_h8_mae": float(healthy_point["ungated_h8"]),
            "gated_h8_mae": float(healthy_point["gated_h8"]),
            "relative_degradation": healthy_degradation,
            "relative_degradation_ci95": [
                float(healthy_ci[0]),
                float(healthy_ci[1]),
            ],
        },
        "raw_unit_mae_by_channel": raw_unit_mae,
        "dropout_separate_diagnostic": dropout,
        "action_use_diagnostic": {
            "scope": "equal-weight healthy cells; not used for checkpoint selection",
            "gated_h8_pass": primary_action_use_pass,
            "arms": action_use,
        },
        "persistence_by_case": persistence_by_case,
        "auxiliary_intervention": {
            "aux_h8_mae": float(point["aux_h8"]),
            "gated_h8_mae": float(point["gated_h8"]),
            "raw_improvement": aux_improvement,
            "aux_and_ungated_core_bit_identical": aux_core_equivalence,
            "aux_and_ungated_outputs_bit_identical": (
                aux_output_equivalence if stage == "confirmation" else None
            ),
            "auxiliary_intervention_isolated": aux_equivalence,
        },
        "huber_noninferiority": {
            "huber_h8_mae": float(point["huber_h8"]),
            "gated_h8_mae": float(point["gated_h8"]),
            "relative_excess": gated_vs_huber_relative_excess,
            "limit": config.huber_noninferiority_limit,
        },
        "competence": {
            "interpretation": (
                "confirmation requires aggregate ratio <=1.05, paired hierarchical "
                "95% upper ratio <=1.10, and every case ratio <=1.10"
            ),
            "rssm_scores": rssm_scores,
            "strongest_rssm_arm": strongest_rssm_arm,
            "strongest_rssm_score": strongest_rssm_score,
            "comparator_scores": comparator_scores,
            "missing_comparators": list(missing_comparators),
            "strongest_comparator_arm": strongest_comparator_arm,
            "strongest_comparator_score": strongest_comparator_score,
            "paired_comparison": competence,
            "raw_and_dropout_diagnostics": baseline_diagnostics,
            "confirmation_limits": {
                "aggregate_ratio": CONFIRMATION_BASELINE_POINT_RATIO_LIMIT,
                "ci95_upper_ratio": CONFIRMATION_BASELINE_CI_RATIO_LIMIT,
                "every_case_ratio": CONFIRMATION_BASELINE_CASE_RATIO_LIMIT,
            },
        },
        "provenance": {
            "required_for_paper_pass": True,
            "trust_boundary": (
                "the gate recomputes byte hashes, identities, inventories, and code "
                "bindings; upstream artifact producers remain responsible for "
                "schema-validating each referenced payload before the pre-lock digest "
                "is recorded"
            ),
            "timing_limit": (
                "hash validation cannot prove when the registry was frozen; its digest "
                "must be recorded externally before locked-test access"
            ),
            "prelock_registry_provided": prelock_registry is not None,
            "externally_frozen_prelock_sha256": expected_prelock_sha256,
            "actual_prelock_sha256": (
                hashlib.sha256(
                    _canonical_json(prelock_registry).encode("ascii")
                ).hexdigest()
                if isinstance(prelock_registry, dict)
                else None
            ),
            "evaluation_receipt_provided": evaluation_receipt is not None,
            "evaluation_receipt_sha256": (
                hashlib.sha256(
                    _canonical_json(evaluation_receipt).encode("ascii")
                ).hexdigest()
                if isinstance(evaluation_receipt, dict)
                else None
            ),
            "integrity_bound": stage == "confirmation" and not provenance_issues,
            "issues": provenance_issues,
        },
        "checks": checks,
    }
    return paired, result


def write_gate_analysis(output_dir: Path, paired: pd.DataFrame, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "matched_h8_arm_metrics.csv"
    result_path = output_dir / "study_gate.json"
    paired_tmp = paired_path.with_suffix(".csv.tmp")
    result_tmp = result_path.with_suffix(".json.tmp")
    paired.to_csv(paired_tmp, index=False)
    result_tmp.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    paired_tmp.replace(paired_path)
    result_tmp.replace(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen multi-case H8 gate")
    parser.add_argument("--rssm", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, nargs="*", default=())
    parser.add_argument(
        "--stage",
        choices=("development",),
        required=True,
        help="standalone gate is development-only; confirmation runs inside study_confirmation",
    )
    parser.add_argument("--selected-update", type=int, required=True)
    parser.add_argument("--prelock-registry", type=Path)
    parser.add_argument("--prelock-artifact-root", type=Path)
    parser.add_argument("--expected-prelock-sha256")
    parser.add_argument("--evaluation-receipt", type=Path)
    parser.add_argument("--evaluation-artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rssm = pd.concat((pd.read_csv(path) for path in args.rssm), ignore_index=True)
    baseline = (
        pd.concat((pd.read_csv(path) for path in args.baseline), ignore_index=True)
        if args.baseline
        else None
    )
    prelock_registry = (
        json.loads(
            args.prelock_registry.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if args.prelock_registry is not None
        else None
    )
    evaluation_receipt = (
        json.loads(
            args.evaluation_receipt.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if args.evaluation_receipt is not None
        else None
    )
    paired, result = evaluate_study_gate(
        rssm,
        StudyConfig(),
        stage=args.stage,
        selected_update=args.selected_update,
        baseline_frame=baseline,
        prelock_registry=prelock_registry,
        prelock_artifact_root=args.prelock_artifact_root,
        expected_prelock_sha256=args.expected_prelock_sha256,
        evaluation_receipt=evaluation_receipt,
        evaluation_artifact_root=args.evaluation_artifact_root,
    )
    write_gate_analysis(args.output_dir, paired, result)
    print(json.dumps({"decision": result["decision"], "gate_pass": result["gate_pass"]}))


if __name__ == "__main__":
    main()
