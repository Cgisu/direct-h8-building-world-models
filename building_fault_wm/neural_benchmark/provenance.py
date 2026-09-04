"""Schema-aware provenance freeze for the confirmatory fault benchmark.

The pre-lock registry is intentionally a small index over a self-contained artifact
root.  Corpus dependencies (plans, receipts, and clean CSVs) may appear in the root
without individual registry references because ``load_corpus_index`` verifies them
transitively from the referenced corpus manifest.  Every other file must have one
explicit semantic identity in the registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from .baselines import (
    ARXFeatureSpec,
    DirectH8FeatureSpec,
    DirectH8GRU,
    BaselineSelectionReceipt,
    RIDGE_ALPHAS,
    _make_gru_schedule,
    _canonical_sha256 as _baseline_canonical_sha256,
    _frame_sha256 as _baseline_frame_sha256,
    _ridge_state_sha256,
    _scalers_sha256,
    _tensor_state_sha256 as _baseline_tensor_state_sha256,
    _variant_identity_sha256,
    arx_one_step_dataset,
    baseline_producer_code_manifest,
    direct_h8_dataset,
)
from .fault_data import (
    CORPUS_LAYOUT,
    ROLES,
    CorpusIndex,
    FaultScalers,
    FaultSpec,
    build_fault_manifest,
    fault_cell_signatures,
    fit_scalers,
    iter_role_variants,
    load_corpus_index,
    load_role_trajectories,
)
from .protocol import CASES, PRELOCK_REGISTRY_SCHEMA, sha256_file
from .reliability_model import ReliabilityGatedRSSM
from .runtime_provenance import (
    fingerprint_device,
    validate_numerical_runtime_fingerprint,
)
from .study_config import ARMS, StudyConfig
from .study_train import (
    canonical_payload_sha256,
    core_tensor_state_sha256,
    make_training_schedule,
    schedule_payload,
    tensor_state_sha256,
    training_provenance,
)


FROZEN_FAULT_CONTRACT_SCHEMA = "boptest-multicase-frozen-fault-contract-v1"
VALIDATION_SELECTION_SCHEMA = (
    "boptest-multicase-validation-checkpoint-selection-v1"
)
BASELINE_BUNDLE_SCHEMA = "boptest-multicase-baseline-selection-bundle-v1"
RIDGE_STATE_SCHEMA = "boptest-multicase-selected-ridge-state-v1"
GRU_STATE_SCHEMA = "boptest-multicase-selected-gru-state-v2"
BASELINE_RECEIPT_SCHEMA = "boptest-baseline-selection-receipt-v1"
RSSM_CHECKPOINT_SCHEMA = "boptest-reliability-rssm-checkpoint-v2"
TRAINING_SCHEDULE_SCHEMA = "boptest-reliability-rssm-training-schedule-v1"

COMPETENCE_BASELINES = (
    "ridge_arx",
    "direct_h8_ridge",
    "deterministic_gru",
)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON identity: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=reject_constant,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_path(root: Path, relative: object) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or Path(relative).as_posix() != relative
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
    ):
        raise ValueError("artifact path is not a nonempty relative path")
    lexical = root.resolve()
    for part in Path(relative).parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise ValueError("artifact path traverses a symbolic link")
    candidate = lexical.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact path escapes its root") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact file is missing: {relative}")
    return candidate


def _relative_file(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"artifact is outside the pre-lock root: {path}") from error
    if not resolved.is_file() or relative == Path("."):
        raise FileNotFoundError(f"artifact is not a file: {path}")
    return str(relative)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("ascii")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False).encode("ascii")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _signature_rows(spec: FaultSpec, role: str) -> list[list[object]]:
    return [list(row) for row in sorted(fault_cell_signatures(spec, role))]


def frozen_fault_contract(index: CorpusIndex, config: StudyConfig) -> dict:
    if index.collection_kind != "development":
        raise ValueError("fault contract must be frozen from the development corpus")
    if set(index.allowed_roles) != {"fit", "validation"}:
        raise ValueError("development corpus has an unexpected role boundary")
    if {record.key.case for record in index.records} != set(CASES):
        raise ValueError("development corpus does not contain every frozen case")
    spec = FaultSpec()
    if spec.evaluation_horizon != config.direct_horizon:
        raise ValueError("fault contract and study horizon differ")
    signatures = {role: _signature_rows(spec, role) for role in ROLES}
    return {
        "schema": FROZEN_FAULT_CONTRACT_SCHEMA,
        "development_corpus_manifest_sha256": index.manifest_sha256,
        "spec": asdict(spec),
        "signatures_by_role": signatures,
        "signature_sha256_by_role": {
            role: canonical_sha256(rows) for role, rows in signatures.items()
        },
    }


def write_frozen_fault_contract(
    path: Path, index: CorpusIndex, config: StudyConfig
) -> dict:
    if path.exists():
        raise FileExistsError("refusing to overwrite the frozen fault contract")
    payload = frozen_fault_contract(index, config)
    _atomic_write_json(path, payload)
    return payload


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    if frame.empty or any(not isinstance(column, str) for column in frame.columns):
        raise ValueError("canonical frame must be nonempty with string columns")
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


def validation_selection_payload(
    validation_rows: pd.DataFrame,
    config: StudyConfig,
    *,
    validation_rows_artifact: Mapping[str, str],
) -> dict:
    # Local import keeps the registry module independent of the gate module.
    from .study_evaluate import validation_selection_scores

    scores = validation_selection_scores(validation_rows, config)
    selected = scores.loc[scores["selected"]]
    if len(selected) != 1:
        raise ValueError("validation selection did not identify exactly one checkpoint")
    score_rows = [
        {
            "update": int(row.update),
            "common_score": float(row.common_score),
            "selected": bool(row.selected),
        }
        for row in scores.sort_values("update", kind="stable").itertuples(index=False)
    ]
    return {
        "schema": VALIDATION_SELECTION_SCHEMA,
        "role": "validation",
        "arms": ["ungated_h8", "gated_h8"],
        "development_seeds": list(config.development_seeds),
        "candidate_updates": list(config.validation_checkpoints),
        "selected_update": int(selected.iloc[0]["update"]),
        "selection_metric": (
            "mean_equal_case_family_fault_channel_standardized_H8_MAE_"
            "across_ungated_h8_and_gated_h8"
        ),
        "score_rows": score_rows,
        "validation_rows_sha256": _canonical_frame_sha256(validation_rows),
        "validation_rows_artifact": dict(validation_rows_artifact),
    }


def write_validation_selection(
    path: Path,
    validation_rows: pd.DataFrame,
    config: StudyConfig,
    *,
    validation_rows_path: Path | None = None,
) -> dict:
    validation_rows_path = (
        path.with_name(f"{path.stem}_rows.csv")
        if validation_rows_path is None
        else validation_rows_path
    )
    if validation_rows_path.resolve().parent != path.resolve().parent:
        raise ValueError("validation rows must be a sibling of their selection record")
    if validation_rows_path.exists() or path.exists():
        raise FileExistsError("refusing to overwrite a validation freeze artifact")
    _atomic_write_csv(validation_rows_path, validation_rows)
    frozen_rows = pd.read_csv(validation_rows_path)
    payload = validation_selection_payload(
        frozen_rows,
        config,
        validation_rows_artifact={
            "path": validation_rows_path.name,
            "sha256": sha256_file(validation_rows_path),
        },
    )
    _atomic_write_json(path, payload)
    return payload


def _ridge_model_payload(model: Ridge) -> dict:
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError("selected Ridge state is non-finite")
    return {
        "schema": RIDGE_STATE_SCHEMA,
        "alpha": float(model.alpha),
        "fit_intercept": bool(model.fit_intercept),
        "n_features_in": int(model.n_features_in_),
        "coef": torch.as_tensor(coef.copy(), dtype=torch.float64),
        "intercept": torch.as_tensor(intercept.copy(), dtype=torch.float64),
        "state_sha256": _ridge_state_sha256(model),
    }


def _gru_model_payload(model: DirectH8GRU) -> dict:
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    return {
        "schema": GRU_STATE_SCHEMA,
        "model_config": {
            "observation_dim": model.observation_dim,
            "action_dim": model.action_dim,
            "context_dim": model.context_dim,
            "hidden_dim": model.hidden_dim,
            "history": model.spec.history,
            "horizon": model.spec.horizon,
        },
        "state_dict": state,
        "state_sha256": _baseline_tensor_state_sha256(state),
    }


def write_baseline_selection_bundle(
    path: Path,
    *,
    baseline: str,
    entries: Mapping[
        str, tuple[BaselineSelectionReceipt, Ridge | DirectH8GRU, pd.DataFrame]
    ],
) -> dict:
    """Write selected baseline models and their validation receipts as one artifact."""
    if path.exists():
        raise FileExistsError("refusing to overwrite a baseline selection bundle")
    if baseline not in COMPETENCE_BASELINES or not entries:
        raise ValueError("baseline bundle has an invalid or empty identity set")
    payload_entries: dict[str, dict] = {}
    for identity, (receipt, model, score_table) in sorted(entries.items()):
        if receipt.baseline != baseline:
            raise ValueError("baseline receipt differs from its bundle")
        if identity != f"{receipt.case}:seed{receipt.model_seed}":
            raise ValueError("baseline entry identity differs from its receipt")
        if (
            receipt.producer_code_sha256
            != baseline_producer_code_manifest()["sha256"]
        ):
            raise ValueError("baseline receipt producer code differs from its bundle")
        validate_numerical_runtime_fingerprint(
            receipt.runtime_fingerprint, include_sklearn=True
        )
        if baseline == "deterministic_gru":
            if not isinstance(model, DirectH8GRU):
                raise TypeError("GRU bundle requires DirectH8GRU models")
            model_payload = _gru_model_payload(model)
        else:
            if not isinstance(model, Ridge):
                raise TypeError("Ridge bundle requires sklearn Ridge models")
            model_payload = _ridge_model_payload(model)
        if receipt.score_table_sha256 != _baseline_frame_sha256(score_table):
            raise ValueError("baseline score table differs from its selection receipt")
        payload_entries[identity] = {
            "receipt": receipt.payload,
            "score_table": {
                "columns": list(score_table.columns),
                "records": score_table.to_dict(orient="records"),
            },
            "model": model_payload,
        }
    payload = {
        "schema": BASELINE_BUNDLE_SCHEMA,
        "baseline": baseline,
        "producer_code": baseline_producer_code_manifest(),
        "entries": payload_entries,
    }
    _atomic_torch_save(path, payload)
    return payload


def load_selected_baseline_entries(
    path: Path,
    *,
    baseline: str,
    config: StudyConfig,
) -> dict[
    str, tuple[BaselineSelectionReceipt, Ridge | DirectH8GRU, pd.DataFrame]
]:
    """Restore selected models and the validation tables that selected them."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "baseline", "producer_code", "entries"}
        or payload["schema"] != BASELINE_BUNDLE_SCHEMA
        or payload["baseline"] != baseline
        or payload["producer_code"] != baseline_producer_code_manifest()
        or not isinstance(payload["entries"], dict)
    ):
        raise ValueError("baseline selection bundle identity differs")
    restored: dict[
        str, tuple[BaselineSelectionReceipt, Ridge | DirectH8GRU, pd.DataFrame]
    ] = {}
    for identity, entry in sorted(payload["entries"].items()):
        if not isinstance(entry, dict) or set(entry) != {
            "receipt",
            "score_table",
            "model",
        }:
            raise ValueError(f"baseline bundle entry is invalid: {identity}")
        receipt_payload = entry["receipt"]
        if not isinstance(receipt_payload, dict):
            raise ValueError(f"baseline receipt is invalid: {identity}")
        receipt = BaselineSelectionReceipt(
            **{
                **receipt_payload,
                "candidate_grid": tuple(receipt_payload["candidate_grid"]),
            }
        )
        if identity != f"{receipt.case}:seed{receipt.model_seed}":
            raise ValueError(f"baseline entry identity differs from receipt: {identity}")
        if (
            receipt.producer_code_sha256
            != baseline_producer_code_manifest()["sha256"]
        ):
            raise ValueError(f"baseline receipt producer code differs: {identity}")
        validate_numerical_runtime_fingerprint(
            receipt.runtime_fingerprint, include_sklearn=True
        )
        table = _load_score_table(entry["score_table"])
        if receipt.score_table_sha256 != _baseline_frame_sha256(table):
            raise ValueError(f"baseline score table differs from receipt: {identity}")
        if baseline == "deterministic_gru":
            model = _restore_gru(entry["model"], receipt_payload, config)
        elif baseline in {"ridge_arx", "direct_h8_ridge"}:
            model = _restore_ridge(entry["model"], receipt_payload)
        else:
            raise ValueError(f"unknown baseline bundle: {baseline}")
        restored[identity] = (receipt, model, table)
    return restored


def load_selected_baseline_models(
    path: Path,
    *,
    baseline: str,
    config: StudyConfig,
) -> dict[str, tuple[BaselineSelectionReceipt, Ridge | DirectH8GRU]]:
    """Restore exactly the selected models carried by a frozen baseline bundle."""
    return {
        identity: (receipt, model)
        for identity, (receipt, model, _) in load_selected_baseline_entries(
            path, baseline=baseline, config=config
        ).items()
    }


def extend_gru_selection_bundle(
    path: Path,
    *,
    development_bundle: Path,
    extension_entries: Mapping[
        str, tuple[BaselineSelectionReceipt, DirectH8GRU, pd.DataFrame]
    ],
    config: StudyConfig,
) -> dict:
    """Write an exact three-to-five-seed extension of a screened GRU bundle."""
    if path.exists():
        raise FileExistsError("refusing to overwrite a GRU selection bundle")
    development_payload = torch.load(
        development_bundle, map_location="cpu", weights_only=True
    )
    development_entries = load_selected_baseline_entries(
        development_bundle, baseline="deterministic_gru", config=config
    )
    expected_development = {
        f"{case}:seed{seed}"
        for case in CASES
        for seed in config.development_seeds
    }
    expected_extension = {
        f"{case}:seed{seed}"
        for case in CASES
        for seed in config.confirmatory_seeds
        if seed not in config.development_seeds
    }
    if set(development_entries) != expected_development:
        raise ValueError("screened GRU bundle does not contain the development seed grid")
    if set(extension_entries) != expected_extension:
        raise ValueError("GRU extension does not contain exactly confirmatory seeds 4 and 5")

    extension_path = path.with_name(f".{path.name}.extension")
    if extension_path.exists():
        raise FileExistsError("stale GRU extension serialization exists")
    try:
        extension_payload = write_baseline_selection_bundle(
            extension_path,
            baseline="deterministic_gru",
            entries=extension_entries,
        )
        payload = {
            "schema": BASELINE_BUNDLE_SCHEMA,
            "baseline": "deterministic_gru",
            "producer_code": baseline_producer_code_manifest(),
            "entries": {
                **development_payload["entries"],
                **extension_payload["entries"],
            },
        }
        _validate_gru_bundle_extension(development_payload, payload)
        _atomic_torch_save(path, payload)
    finally:
        extension_path.unlink(missing_ok=True)
    restored = load_selected_baseline_entries(
        path, baseline="deterministic_gru", config=config
    )
    expected_all = {
        f"{case}:seed{seed}" for case in CASES for seed in config.confirmatory_seeds
    }
    if set(restored) != expected_all:
        raise AssertionError("persisted confirmatory GRU bundle is incomplete")
    return payload


def _artifact_reference(
    root: Path, path: Path, *, kind: str, identity: str
) -> dict:
    relative = _relative_file(root, path)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "kind": kind,
        "identity": identity,
    }


def _required_checkpoint_keys(config: StudyConfig, selected_update: int) -> set[str]:
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


def _validate_frame_artifact_metadata(
    metadata: object,
    reference: object,
    artifact_root: Path,
    *,
    label: str,
) -> pd.DataFrame:
    from .study_gate import canonical_frame_sha256

    required = {"path", "rows", "sha256", "canonical_frame_sha256"}
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required
        or not isinstance(metadata["path"], str)
        or not metadata["path"]
        or isinstance(metadata["rows"], bool)
        or not isinstance(metadata["rows"], int)
        or metadata["rows"] <= 0
        or not _is_sha256(metadata["sha256"])
        or not _is_sha256(metadata["canonical_frame_sha256"])
        or not isinstance(reference, dict)
    ):
        raise ValueError(f"development {label} metadata are invalid")
    path = _safe_path(artifact_root, reference.get("path"))
    if reference.get("sha256") != metadata["sha256"] or sha256_file(path) != metadata[
        "sha256"
    ]:
        raise ValueError(f"development {label} file differs from its receipt")
    frame = pd.read_csv(path)
    if len(frame) != metadata["rows"] or canonical_frame_sha256(frame) != metadata[
        "canonical_frame_sha256"
    ]:
        raise ValueError(f"development {label} rows differ from their receipt")
    return frame


def _validate_json_artifact_metadata(
    metadata: object,
    reference: object,
    artifact_root: Path,
    *,
    label: str,
) -> dict:
    required = {"path", "sha256", "canonical_json_sha256"}
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required
        or not isinstance(metadata["path"], str)
        or not metadata["path"]
        or not _is_sha256(metadata["sha256"])
        or not _is_sha256(metadata["canonical_json_sha256"])
        or not isinstance(reference, dict)
    ):
        raise ValueError(f"development {label} metadata are invalid")
    path = _safe_path(artifact_root, reference.get("path"))
    if reference.get("sha256") != metadata["sha256"] or sha256_file(path) != metadata[
        "sha256"
    ]:
        raise ValueError(f"development {label} file differs from its receipt")
    payload = load_strict_json(path)
    if not isinstance(payload, dict) or canonical_sha256(payload) != metadata[
        "canonical_json_sha256"
    ]:
        raise ValueError(f"development {label} JSON differs from its receipt")
    return payload


def _validate_development_screen_evidence(
    registry: dict,
    artifact_root: Path,
    config: StudyConfig,
    *,
    selected_update: int,
) -> None:
    """Recompute SCREEN_GO from the exact persisted development evidence."""
    from .study_gate import canonical_frame_sha256, evaluate_study_gate

    run_complete = load_strict_json(
        _safe_path(
            artifact_root, registry["development_run_complete_artifact"]["path"]
        )
    )
    if not isinstance(run_complete, dict):
        raise ValueError("development completion receipt is invalid")
    rssm = _validate_frame_artifact_metadata(
        run_complete.get("rssm_result"),
        registry.get("development_rssm_result_artifact"),
        artifact_root,
        label="RSSM validation result",
    )
    baseline = _validate_frame_artifact_metadata(
        run_complete.get("baseline_result"),
        registry.get("development_baseline_result_artifact"),
        artifact_root,
        label="baseline validation result",
    )
    gate_artifacts = run_complete.get("gate_artifacts")
    if not isinstance(gate_artifacts, dict) or set(gate_artifacts) != {
        "matched_metrics",
        "gate_result",
    }:
        raise ValueError("development gate artifact metadata are invalid")
    persisted_paired = _validate_frame_artifact_metadata(
        gate_artifacts["matched_metrics"],
        registry.get("development_gate_paired_artifact"),
        artifact_root,
        label="gate matched metrics",
    )
    persisted_result = _validate_json_artifact_metadata(
        gate_artifacts["gate_result"],
        registry.get("development_gate_result_artifact"),
        artifact_root,
        label="gate result",
    )
    recomputed_paired, recomputed_result = evaluate_study_gate(
        rssm,
        config,
        stage="development",
        selected_update=selected_update,
        baseline_frame=baseline,
    )
    if canonical_frame_sha256(recomputed_paired) != canonical_frame_sha256(
        persisted_paired
    ):
        raise ValueError("persisted development matched metrics do not recompute")
    normalized_result = json.loads(json.dumps(recomputed_result, allow_nan=False))
    if normalized_result != persisted_result:
        raise ValueError("persisted development gate result does not recompute")
    if (
        recomputed_result.get("decision") != "SCREEN_GO"
        or run_complete.get("decision") != "SCREEN_GO"
    ):
        raise ValueError("pre-lock requires a recomputed SCREEN_GO development result")


def _validate_development_run_binding(
    registry: dict,
    artifact_root: Path,
    config: StudyConfig,
    *,
    selected_update: int,
    corpus_manifest_sha256: str | None = None,
    fault_manifest_sha256: str | None = None,
) -> None:
    """Bind the pre-lock artifacts to the runner receipt that produced them."""
    # Local import avoids the intentional study_development -> provenance dependency.
    from .study_development import (
        RUN_CONFIG_FIELDS,
        RUNNER_SCHEMA,
        RUN_RECEIPT_FIELDS,
        RUN_RECEIPT_SCHEMA,
        scientific_code_manifest,
    )

    run_config = load_strict_json(
        _safe_path(
            artifact_root, registry["development_run_config_artifact"]["path"]
        )
    )
    run_complete = load_strict_json(
        _safe_path(
            artifact_root,
            registry["development_run_complete_artifact"]["path"],
        )
    )
    if not isinstance(run_config, dict) or set(run_config) != RUN_CONFIG_FIELDS:
        raise ValueError("development run_config fields differ from runner schema")
    if not isinstance(run_complete, dict) or set(run_complete) != RUN_RECEIPT_FIELDS:
        raise ValueError("development run_complete fields differ from runner schema")
    if (
        run_config["schema"] != RUNNER_SCHEMA
        or run_complete["schema"] != RUN_RECEIPT_SCHEMA
        or run_config["stage"] != "development"
        or run_complete["stage"] != "development"
        or run_config["interpretation"] != "frozen_development_screen"
        or run_complete["interpretation"] != "frozen_development_screen"
        or run_config["scientific_screen_enabled"] is not True
        or run_complete["scientific_screen_reported"] is not True
        or run_complete["decision"] != "SCREEN_GO"
    ):
        raise ValueError("pre-lock requires a completed SCREEN_GO development run")
    expected_config = config.to_dict()
    expected_config_json = json.loads(json.dumps(expected_config, allow_nan=False))
    expected_config_sha256 = canonical_sha256(expected_config)
    if (
        run_config["study_config"] != expected_config_json
        or run_config["study_config_sha256"] != expected_config_sha256
        or run_complete["study_config_sha256"] != expected_config_sha256
        or run_complete["selected_update"] != selected_update
    ):
        raise ValueError("development receipt differs from the frozen study config")

    current_code = scientific_code_manifest()
    current_code_sha256 = canonical_sha256(current_code)
    for document in (run_config, run_complete):
        if (
            document["scientific_code_sha256_by_path"] != current_code
            or document["scientific_code_manifest_sha256"]
            != current_code_sha256
        ):
            raise ValueError(
                "development scientific-code manifest differs from current code"
            )
    if run_config["scientific_code_sha256_by_path"] != run_complete[
        "scientific_code_sha256_by_path"
    ]:
        raise ValueError("development start and completion code manifests differ")
    runner_relative = "multicase_fault_benchmark/study_development.py"
    if (
        run_config["runner_sha256"] != current_code[runner_relative]
        or run_complete["runner_sha256"] != current_code[runner_relative]
    ):
        raise ValueError("development runner hash differs from its code manifest")
    registry_code_fields = {
        "protocol_sha256": "multicase_fault_benchmark/STUDY_PROTOCOL.md",
        "evaluator_sha256": "multicase_fault_benchmark/study_evaluate.py",
        "gate_sha256": "multicase_fault_benchmark/study_gate.py",
        "trainer_sha256": "multicase_fault_benchmark/study_train.py",
        "reliability_model_sha256": (
            "multicase_fault_benchmark/reliability_model.py"
        ),
        "rssm_backbone_sha256": "health_rssm/model.py",
        "reliability_loss_sha256": (
            "multicase_fault_benchmark/reliability_loss.py"
        ),
        "fault_data_sha256": "multicase_fault_benchmark/fault_data.py",
        "baselines_sha256": "multicase_fault_benchmark/baselines.py",
        "provenance_sha256": "multicase_fault_benchmark/provenance.py",
        "confirmation_runner_sha256": (
            "multicase_fault_benchmark/study_confirmation.py"
        ),
    }
    for registry_field, relative in registry_code_fields.items():
        if registry.get(registry_field) != current_code[relative]:
            raise ValueError(
                f"pre-lock {registry_field} is not bound to the development manifest"
            )

    for document in (run_config, run_complete):
        validate_numerical_runtime_fingerprint(
            document["numerical_runtime"], include_sklearn=True
        )
    if run_config["numerical_runtime"] != run_complete["numerical_runtime"]:
        raise ValueError("development start and completion runtimes differ")
    try:
        declared_device = torch.device(run_config["device"])
    except (TypeError, RuntimeError) as error:
        raise ValueError("development run_config device is invalid") from error
    recorded_device = fingerprint_device(run_config["numerical_runtime"])
    if declared_device.type != recorded_device.type or (
        declared_device.index is not None
        and declared_device.index != recorded_device.index
    ):
        raise ValueError("development device differs from its runtime fingerprint")

    for document, fields in (
        (
            run_config,
            (
                "corpus_manifest_sha256",
                "corpus_manifest_file_sha256",
                "staging_identity_sha256",
            ),
        ),
        (
            run_complete,
            (
                "corpus_manifest_sha256",
                "corpus_manifest_file_sha256",
                "fault_manifest_sha256",
                "staging_identity_sha256",
                "validation_selection_sha256",
            ),
        ),
    ):
        for field in fields:
            if not _is_sha256(document[field]):
                raise ValueError(f"development receipt has invalid {field}")
    if (
        run_config["corpus_manifest_sha256"]
        != run_complete["corpus_manifest_sha256"]
    ):
        raise ValueError("development corpus identity changed during the run")
    if (
        run_config["corpus_manifest_file_sha256"]
        != run_complete["corpus_manifest_file_sha256"]
        or run_complete["corpus_manifest_file_sha256"]
        != registry["corpus_manifest_artifact"]["sha256"]
    ):
        raise ValueError("development corpus manifest file changed during the run")
    if (
        run_config["staging_identity_sha256"]
        != run_complete["staging_identity_sha256"]
    ):
        raise ValueError("development staging identity changed during the run")
    if (
        corpus_manifest_sha256 is not None
        and run_complete["corpus_manifest_sha256"] != corpus_manifest_sha256
    ):
        raise ValueError("development receipt differs from the registered corpus")
    if (
        fault_manifest_sha256 is not None
        and run_complete["fault_manifest_sha256"] != fault_manifest_sha256
    ):
        raise ValueError("development receipt differs from the fault manifest")
    if (
        run_complete["validation_selection_sha256"]
        != registry["validation_selection_artifact"]["sha256"]
    ):
        raise ValueError("development receipt differs from validation selection")
    expected_baselines = {
        arm: reference["sha256"]
        for arm, reference in sorted(
            registry["baseline_selection_artifact_by_arm"].items()
        )
        if arm != "deterministic_gru"
    }
    expected_baselines["deterministic_gru"] = registry[
        "development_gru_baseline_artifact"
    ]["sha256"]
    if run_complete["baseline_bundle_sha256"] != expected_baselines:
        raise ValueError("development receipt differs from baseline bundles")

    inventory = run_complete["artifact_inventory_excludes_this_receipt"]
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("development receipt artifact inventory is invalid")
    inventory_by_path: dict[str, dict] = {}
    for entry in inventory:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "bytes"}
            or not isinstance(entry["path"], str)
            or not entry["path"]
            or not _is_sha256(entry["sha256"])
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] <= 0
            or entry["path"] in inventory_by_path
        ):
            raise ValueError("development receipt artifact inventory is invalid")
        inventory_by_path[entry["path"]] = entry
    required_original_hashes = {
        "run_config.json": registry["development_run_config_artifact"]["sha256"],
        "frozen/frozen_fault_contract.json": registry["fault_manifest_artifact"][
            "sha256"
        ],
        "frozen/validation_selection.json": registry[
            "validation_selection_artifact"
        ]["sha256"],
        **{
            f"frozen/baselines/{arm}.pt": expected_baselines[arm]
            for arm in sorted(expected_baselines)
        },
    }
    evidence_bindings = (
        (
            run_complete.get("rssm_result"),
            registry["development_rssm_result_artifact"],
            "RSSM validation result",
        ),
        (
            run_complete.get("baseline_result"),
            registry["development_baseline_result_artifact"],
            "baseline validation result",
        ),
    )
    for metadata, reference, label in evidence_bindings:
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("path"), str)
            or metadata.get("sha256") != reference["sha256"]
        ):
            raise ValueError(f"development {label} receipt binding is invalid")
        required_original_hashes[metadata["path"]] = reference["sha256"]
    gate_artifacts = run_complete.get("gate_artifacts")
    if not isinstance(gate_artifacts, dict) or set(gate_artifacts) != {
        "matched_metrics",
        "gate_result",
    }:
        raise ValueError("development gate artifact receipt binding is invalid")
    for key, reference_field in (
        ("matched_metrics", "development_gate_paired_artifact"),
        ("gate_result", "development_gate_result_artifact"),
    ):
        metadata = gate_artifacts[key]
        reference = registry[reference_field]
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("path"), str)
            or metadata.get("sha256") != reference["sha256"]
        ):
            raise ValueError("development gate artifact receipt binding is invalid")
        required_original_hashes[metadata["path"]] = reference["sha256"]
    for case in CASES:
        for seed in config.development_seeds:
            schedule_identity = f"{case}:seed{seed}"
            required_original_hashes[
                f"training/{case}/seed{seed}/training_schedule.json"
            ] = registry["training_schedule_artifact_by_case_seed"][
                schedule_identity
            ]["sha256"]
            for arm in ARMS:
                checkpoint_identity = (
                    f"{case}:seed{seed}:{arm}:u{selected_update:04d}"
                )
                required_original_hashes[
                    "training/"
                    f"{case}/seed{seed}/checkpoints/"
                    f"{arm}_u{selected_update:04d}.pt"
                ] = registry["checkpoint_artifact_by_identity"][
                    checkpoint_identity
                ]["sha256"]
    for relative, expected_sha256 in required_original_hashes.items():
        if (
            relative not in inventory_by_path
            or inventory_by_path[relative]["sha256"] != expected_sha256
        ):
            raise ValueError(
                f"development receipt inventory does not bind {relative}"
            )


def build_prelock_registry(
    *,
    artifact_root: Path,
    development_run_config_path: Path,
    development_run_complete_path: Path,
    development_rssm_result_path: Path,
    development_baseline_result_path: Path,
    development_gate_paired_path: Path,
    development_gate_result_path: Path,
    development_gru_baseline_bundle: Path,
    development_corpus_manifest: Path,
    frozen_fault_contract_path: Path,
    validation_selection_path: Path,
    fit_scalers_by_case: Mapping[str, Path],
    checkpoints_by_identity: Mapping[str, Path],
    schedules_by_case_seed: Mapping[str, Path],
    baseline_bundles_by_arm: Mapping[str, Path],
    config: StudyConfig,
    selected_update: int,
) -> dict:
    """Build and validate a canonical registry over an already staged root."""
    if selected_update not in config.validation_checkpoints:
        raise ValueError("selected update is outside the frozen checkpoint grid")
    if set(fit_scalers_by_case) != set(CASES):
        raise ValueError("FIT scaler paths do not cover every case")
    if set(checkpoints_by_identity) != _required_checkpoint_keys(config, selected_update):
        raise ValueError("checkpoint paths do not cover the confirmatory identity grid")
    if set(schedules_by_case_seed) != _required_schedule_keys(config):
        raise ValueError("schedule paths do not cover the confirmatory identity grid")
    if set(baseline_bundles_by_arm) != set(COMPETENCE_BASELINES):
        raise ValueError("baseline bundles do not cover the competence baselines")

    here = Path(__file__).resolve().parent
    registry = {
        "schema": PRELOCK_REGISTRY_SCHEMA,
        "stage": "prelock",
        "selected_update": selected_update,
        "protocol_sha256": sha256_file(here / "STUDY_PROTOCOL.md"),
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "evaluator_sha256": sha256_file(here / "study_evaluate.py"),
        "gate_sha256": sha256_file(here / "study_gate.py"),
        "trainer_sha256": sha256_file(here / "study_train.py"),
        "reliability_model_sha256": sha256_file(here / "reliability_model.py"),
        "rssm_backbone_sha256": sha256_file(here.parent / "health_rssm" / "model.py"),
        "reliability_loss_sha256": sha256_file(here / "reliability_loss.py"),
        "fault_data_sha256": sha256_file(here / "fault_data.py"),
        "baselines_sha256": sha256_file(here / "baselines.py"),
        "provenance_sha256": sha256_file(Path(__file__).resolve()),
        "confirmation_runner_sha256": sha256_file(here / "study_confirmation.py"),
        "development_run_config_artifact": _artifact_reference(
            artifact_root,
            development_run_config_path,
            kind="development_run_config",
            identity="development",
        ),
        "development_run_complete_artifact": _artifact_reference(
            artifact_root,
            development_run_complete_path,
            kind="development_run_receipt",
            identity="development",
        ),
        "development_rssm_result_artifact": _artifact_reference(
            artifact_root,
            development_rssm_result_path,
            kind="development_rssm_validation_result",
            identity="all_arms",
        ),
        "development_baseline_result_artifact": _artifact_reference(
            artifact_root,
            development_baseline_result_path,
            kind="development_baseline_validation_result",
            identity="competence_baselines",
        ),
        "development_gate_paired_artifact": _artifact_reference(
            artifact_root,
            development_gate_paired_path,
            kind="development_gate_matched_metrics",
            identity="development",
        ),
        "development_gate_result_artifact": _artifact_reference(
            artifact_root,
            development_gate_result_path,
            kind="development_gate_result",
            identity="SCREEN_GO",
        ),
        "development_gru_baseline_artifact": _artifact_reference(
            artifact_root,
            development_gru_baseline_bundle,
            kind="development_baseline_validation_selection",
            identity="deterministic_gru",
        ),
        "corpus_manifest_artifact": _artifact_reference(
            artifact_root,
            development_corpus_manifest,
            kind="development_corpus_manifest",
            identity="development",
        ),
        "fault_manifest_artifact": _artifact_reference(
            artifact_root,
            frozen_fault_contract_path,
            kind="frozen_fault_manifest",
            identity="all_roles",
        ),
        "validation_selection_artifact": _artifact_reference(
            artifact_root,
            validation_selection_path,
            kind="validation_checkpoint_selection",
            identity=f"update{selected_update:04d}",
        ),
        "fit_scaler_artifact_by_case": {
            case: _artifact_reference(
                artifact_root, path, kind="fit_scaler", identity=case
            )
            for case, path in sorted(fit_scalers_by_case.items())
        },
        "checkpoint_artifact_by_identity": {
            identity: _artifact_reference(
                artifact_root, path, kind="rssm_checkpoint", identity=identity
            )
            for identity, path in sorted(checkpoints_by_identity.items())
        },
        "training_schedule_artifact_by_case_seed": {
            identity: _artifact_reference(
                artifact_root, path, kind="training_schedule", identity=identity
            )
            for identity, path in sorted(schedules_by_case_seed.items())
        },
        "baseline_selection_artifact_by_arm": {
            arm: _artifact_reference(
                artifact_root,
                path,
                kind="baseline_validation_selection",
                identity=arm,
            )
            for arm, path in sorted(baseline_bundles_by_arm.items())
        },
    }
    registry["artifact_inventory"] = sorted(
        str(path.relative_to(artifact_root.resolve()))
        for path in artifact_root.resolve().rglob("*")
        if path.is_file()
    )
    issues = validate_prelock_registry_semantics(
        registry,
        artifact_root,
        config,
        selected_update=selected_update,
    )
    if issues:
        raise ValueError("invalid pre-lock artifact bundle: " + "; ".join(issues))
    return registry


def write_prelock_registry(path: Path, registry: dict) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite the pre-lock registry")
    _atomic_write_json(path, registry)
    return canonical_sha256(registry)


def validate_prelock_bundle(
    registry_path: Path,
    artifact_root: Path,
    config: StudyConfig,
    expected_sha256: str,
) -> dict:
    """Validate the external freeze before authorizing one-shot collection."""
    registry = load_strict_json(registry_path)
    return validate_prelock_registry_object(
        registry, artifact_root, config, expected_sha256
    )


def validate_prelock_registry_object(
    registry: object,
    artifact_root: Path,
    config: StudyConfig,
    expected_sha256: str,
) -> dict:
    """Validate an already loaded registry using the canonical pre-lock contract."""
    if not isinstance(registry, dict):
        raise ValueError("pre-lock registry is not a JSON object")
    actual_sha256 = canonical_sha256(registry)
    if not _is_sha256(expected_sha256) or actual_sha256 != expected_sha256:
        raise ValueError("pre-lock registry differs from its external canonical digest")
    here = Path(__file__).resolve().parent
    exact = {
        "schema": PRELOCK_REGISTRY_SCHEMA,
        "stage": "prelock",
        "protocol_sha256": sha256_file(here / "STUDY_PROTOCOL.md"),
        "study_config_sha256": canonical_sha256(config.to_dict()),
        "evaluator_sha256": sha256_file(here / "study_evaluate.py"),
        "gate_sha256": sha256_file(here / "study_gate.py"),
        "trainer_sha256": sha256_file(here / "study_train.py"),
        "reliability_model_sha256": sha256_file(here / "reliability_model.py"),
        "rssm_backbone_sha256": sha256_file(
            here.parent / "health_rssm" / "model.py"
        ),
        "reliability_loss_sha256": sha256_file(here / "reliability_loss.py"),
        "fault_data_sha256": sha256_file(here / "fault_data.py"),
        "baselines_sha256": sha256_file(here / "baselines.py"),
        "provenance_sha256": sha256_file(Path(__file__).resolve()),
        "confirmation_runner_sha256": sha256_file(here / "study_confirmation.py"),
    }
    expected_registry_fields = {
        *exact,
        "selected_update",
        "development_run_config_artifact",
        "development_run_complete_artifact",
        "development_rssm_result_artifact",
        "development_baseline_result_artifact",
        "development_gate_paired_artifact",
        "development_gate_result_artifact",
        "development_gru_baseline_artifact",
        "corpus_manifest_artifact",
        "fault_manifest_artifact",
        "validation_selection_artifact",
        "fit_scaler_artifact_by_case",
        "checkpoint_artifact_by_identity",
        "training_schedule_artifact_by_case_seed",
        "baseline_selection_artifact_by_arm",
        "artifact_inventory",
    }
    if set(registry) != expected_registry_fields:
        raise ValueError("pre-lock registry fields differ from the frozen schema")
    for field, expected in exact.items():
        if registry.get(field) != expected:
            raise ValueError(f"pre-lock {field} differs from the frozen implementation")
    selected_update = registry.get("selected_update")
    if (
        isinstance(selected_update, bool)
        or not isinstance(selected_update, int)
        or selected_update not in config.validation_checkpoints
    ):
        raise ValueError("pre-lock selected update is invalid")
    expected_mapping_keys = {
        "fit_scaler_artifact_by_case": set(CASES),
        "checkpoint_artifact_by_identity": _required_checkpoint_keys(
            config, selected_update
        ),
        "training_schedule_artifact_by_case_seed": _required_schedule_keys(config),
        "baseline_selection_artifact_by_arm": set(COMPETENCE_BASELINES),
    }
    references: list[dict] = []
    single_contracts = {
        "development_run_config_artifact": (
            "development_run_config",
            "development",
        ),
        "development_run_complete_artifact": (
            "development_run_receipt",
            "development",
        ),
        "development_rssm_result_artifact": (
            "development_rssm_validation_result",
            "all_arms",
        ),
        "development_baseline_result_artifact": (
            "development_baseline_validation_result",
            "competence_baselines",
        ),
        "development_gate_paired_artifact": (
            "development_gate_matched_metrics",
            "development",
        ),
        "development_gate_result_artifact": (
            "development_gate_result",
            "SCREEN_GO",
        ),
        "development_gru_baseline_artifact": (
            "development_baseline_validation_selection",
            "deterministic_gru",
        ),
        "corpus_manifest_artifact": (
            "development_corpus_manifest",
            "development",
        ),
        "fault_manifest_artifact": ("frozen_fault_manifest", "all_roles"),
        "validation_selection_artifact": (
            "validation_checkpoint_selection",
            f"update{selected_update:04d}",
        ),
    }
    for field, (kind, identity) in single_contracts.items():
        reference = registry.get(field)
        if not isinstance(reference, dict):
            raise ValueError(f"pre-lock {field} is not an artifact reference")
        if reference.get("kind") != kind or reference.get("identity") != identity:
            raise ValueError(f"pre-lock {field} identity differs from schema")
        references.append(reference)
    mapping_kinds = {
        "fit_scaler_artifact_by_case": "fit_scaler",
        "checkpoint_artifact_by_identity": "rssm_checkpoint",
        "training_schedule_artifact_by_case_seed": "training_schedule",
        "baseline_selection_artifact_by_arm": "baseline_validation_selection",
    }
    for field, expected_keys in expected_mapping_keys.items():
        mapping = registry.get(field)
        if not isinstance(mapping, dict) or set(mapping) != expected_keys:
            raise ValueError(f"pre-lock {field} identities are incomplete")
        for identity, reference in mapping.items():
            if (
                not isinstance(reference, dict)
                or reference.get("kind") != mapping_kinds[field]
                or reference.get("identity") != identity
            ):
                raise ValueError(f"pre-lock {field} identity differs from its key")
            references.append(reference)
    expected_reference_fields = {"path", "sha256", "kind", "identity"}
    declared_paths: list[str] = []
    declared_hashes: list[str] = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != expected_reference_fields:
            raise ValueError("pre-lock artifact reference fields differ from schema")
        path = _safe_path(artifact_root, reference["path"])
        if not _is_sha256(reference["sha256"]):
            raise ValueError("pre-lock artifact reference has an invalid digest")
        if sha256_file(path) != reference["sha256"]:
            raise ValueError("pre-lock artifact changed after registry construction")
        declared_paths.append(reference["path"])
        declared_hashes.append(reference["sha256"])
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("pre-lock artifact paths are reused across identities")
    if len(declared_hashes) != len(set(declared_hashes)):
        raise ValueError("pre-lock artifact hashes are reused across identities")
    inventory = registry.get("artifact_inventory")
    if (
        not isinstance(inventory, list)
        or any(not isinstance(path, str) for path in inventory)
        or len(inventory) != len(set(inventory))
        or not set(declared_paths).issubset(inventory)
    ):
        raise ValueError("pre-lock artifact inventory is invalid")
    actual_files = {
        str(path.relative_to(artifact_root.resolve()))
        for path in artifact_root.resolve().rglob("*")
        if path.is_file()
    }
    if actual_files != set(inventory):
        raise ValueError("pre-lock artifact root differs from its frozen inventory")
    _validate_development_run_binding(
        registry,
        artifact_root,
        config,
        selected_update=selected_update,
    )
    issues = validate_prelock_registry_semantics(
        registry,
        artifact_root,
        config,
        selected_update=selected_update,
    )
    if issues:
        raise ValueError("; ".join(issues))
    return registry


def _strict_scan_corpus_json(manifest_path: Path) -> None:
    output_root = manifest_path.parent.parent
    for path in sorted(output_root.rglob("*.json")):
        load_strict_json(path)


def _load_development_index(manifest_path: Path) -> CorpusIndex:
    _strict_scan_corpus_json(manifest_path)
    index = load_corpus_index(manifest_path)
    if index.collection_kind != "development":
        raise ValueError("pre-lock corpus is not the development corpus")
    if set(index.allowed_roles) != {"fit", "validation"}:
        raise ValueError("development corpus roles differ from FIT/validation")
    if {record.key.case for record in index.records} != set(CASES):
        raise ValueError("development corpus does not contain every frozen case")
    return index


def prelock_plan_sha256_by_case(
    registry: dict, artifact_root: Path
) -> dict[str, str]:
    """Return the full-plan identities already sealed in the development corpus."""
    try:
        relative = registry["corpus_manifest_artifact"]["path"]
    except (KeyError, TypeError) as error:
        raise ValueError("pre-lock corpus manifest reference is missing") from error
    index = _load_development_index(_safe_path(artifact_root, relative))
    values = dict(index.plan_sha256_by_case)
    if set(values) != set(CASES) or any(not _is_sha256(value) for value in values.values()):
        raise ValueError("pre-lock development plan identities are incomplete")
    return values


def _validate_fault_contract(
    path: Path, index: CorpusIndex, config: StudyConfig
) -> None:
    payload = load_strict_json(path)
    if not isinstance(payload, dict) or payload != frozen_fault_contract(index, config):
        raise ValueError("frozen fault contract differs from deterministic signatures")


def _load_scalers(
    registry: dict, root: Path, index: CorpusIndex
) -> dict[str, FaultScalers]:
    references = registry["fit_scaler_artifact_by_case"]
    scalers: dict[str, FaultScalers] = {}
    for case in CASES:
        path = _safe_path(root, references[case]["path"])
        payload = load_strict_json(path)
        expected = fit_scalers(load_role_trajectories(index, "fit", cases=(case,)))
        if payload != asdict(expected):
            raise ValueError(f"FIT scaler differs from clean FIT rows for {case}")
        scalers[case] = expected
    return scalers


def _validate_selection(
    path: Path, config: StudyConfig, selected_update: int
) -> Path:
    payload = load_strict_json(path)
    required = {
        "schema",
        "role",
        "arms",
        "development_seeds",
        "candidate_updates",
        "selected_update",
        "selection_metric",
        "score_rows",
        "validation_rows_sha256",
        "validation_rows_artifact",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("validation selection fields differ from the frozen schema")
    if (
        payload["schema"] != VALIDATION_SELECTION_SCHEMA
        or payload["role"] != "validation"
        or payload["arms"] != ["ungated_h8", "gated_h8"]
        or payload["development_seeds"] != list(config.development_seeds)
        or payload["candidate_updates"] != list(config.validation_checkpoints)
        or payload["selected_update"] != selected_update
        or payload["selection_metric"]
        != (
            "mean_equal_case_family_fault_channel_standardized_H8_MAE_"
            "across_ungated_h8_and_gated_h8"
        )
        or not _is_sha256(payload["validation_rows_sha256"])
    ):
        raise ValueError("validation selection identity differs from the frozen contract")
    source_reference = payload["validation_rows_artifact"]
    if (
        not isinstance(source_reference, dict)
        or set(source_reference) != {"path", "sha256"}
        or not _is_sha256(source_reference["sha256"])
    ):
        raise ValueError("validation row artifact reference is invalid")
    source_path = _safe_path(path.parent, source_reference["path"])
    if sha256_file(source_path) != source_reference["sha256"]:
        raise ValueError("validation row artifact changed after selection")
    source_rows = pd.read_csv(source_path)
    if _canonical_frame_sha256(source_rows) != payload["validation_rows_sha256"]:
        raise ValueError("validation rows differ from their canonical digest")
    from .study_evaluate import validation_selection_scores

    recomputed_scores = validation_selection_scores(source_rows, config)
    recomputed_rows = [
        {
            "update": int(row.update),
            "common_score": float(row.common_score),
            "selected": bool(row.selected),
        }
        for row in recomputed_scores.sort_values("update", kind="stable").itertuples(
            index=False
        )
    ]
    if payload["score_rows"] != recomputed_rows:
        raise ValueError("validation selection scores differ from source rows")
    rows = payload["score_rows"]
    if not isinstance(rows, list) or len(rows) != len(config.validation_checkpoints):
        raise ValueError("validation selection score grid is incomplete")
    frame = pd.DataFrame(rows)
    if set(frame.columns) != {"update", "common_score", "selected"}:
        raise ValueError("validation selection score columns are invalid")
    if frame["update"].tolist() != list(config.validation_checkpoints):
        raise ValueError("validation selection updates are reordered or incomplete")
    values = frame["common_score"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("validation selection scores are invalid")
    expected_index = int(np.lexsort((frame["update"].to_numpy(), values))[0])
    selected_mask = frame["selected"].map(lambda value: value is True).to_numpy()
    if selected_mask.sum() != 1 or not selected_mask[expected_index]:
        raise ValueError("validation checkpoint is not the deterministic best score")
    if int(frame.iloc[expected_index]["update"]) != selected_update:
        raise ValueError("registry selected update differs from validation selection")
    return source_path


def _expected_case_inputs(
    index: CorpusIndex, config: StudyConfig
) -> tuple[object, dict[str, list], dict[str, FaultScalers]]:
    manifest = build_fault_manifest(index)
    variants: dict[str, list] = {}
    scalers: dict[str, FaultScalers] = {}
    for case in CASES:
        scalers[case] = fit_scalers(
            load_role_trajectories(index, "fit", cases=(case,))
        )
        variants[case] = list(
            iter_role_variants(index, manifest, "fit", cases=(case,))
        )
    return manifest, variants, scalers


def _validate_schedules_and_checkpoints(
    registry: dict,
    root: Path,
    index: CorpusIndex,
    config: StudyConfig,
    selected_update: int,
    manifest: object,
    variants: Mapping[str, list],
    scalers: Mapping[str, FaultScalers],
) -> None:
    schedule_documents: dict[str, dict] = {}
    for identity, reference in sorted(
        registry["training_schedule_artifact_by_case_seed"].items()
    ):
        case, seed_text = identity.split(":")
        seed = int(seed_text.removeprefix("seed"))
        expected_schedule = make_training_schedule(
            variants[case], config, case=case, model_seed=seed
        )
        expected_document = schedule_payload(expected_schedule, variants[case])
        path = _safe_path(root, reference["path"])
        actual = load_strict_json(path)
        if actual != expected_document:
            raise ValueError(f"training schedule is not deterministic for {identity}")
        schedule_documents[identity] = expected_document

    expected_model = ReliabilityGatedRSSM(config.model_config())
    expected_state_keys = set(expected_model.state_dict())
    for identity, reference in sorted(
        registry["checkpoint_artifact_by_identity"].items()
    ):
        case, seed_text, arm, update_text = identity.split(":")
        seed = int(seed_text.removeprefix("seed"))
        update = int(update_text.removeprefix("u"))
        path = _safe_path(root, reference["path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected_identity = {
            "schema": RSSM_CHECKPOINT_SCHEMA,
            "case": case,
            "model_seed": seed,
            "arm": arm,
            "update": selected_update,
            "config": config.to_dict(),
        }
        if not isinstance(checkpoint, dict) or any(
            checkpoint.get(key) != value for key, value in expected_identity.items()
        ):
            raise ValueError(f"RSSM checkpoint identity differs for {identity}")
        if update != selected_update:
            raise ValueError(f"RSSM checkpoint update differs for {identity}")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict) or set(state) != expected_state_keys:
            raise ValueError(f"RSSM checkpoint tensors are incomplete for {identity}")
        if checkpoint.get("model_state_sha256") != tensor_state_sha256(state):
            raise ValueError(f"RSSM model-state hash differs for {identity}")
        if checkpoint.get("core_state_sha256") != core_tensor_state_sha256(state):
            raise ValueError(f"RSSM core-state hash differs for {identity}")
        expected_model.load_state_dict(state, strict=True)
        schedule = schedule_documents[f"{case}:seed{seed}"]
        provenance = checkpoint.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"RSSM training provenance is missing for {identity}")
        validate_numerical_runtime_fingerprint(
            provenance.get("runtime"), include_sklearn=False
        )
        expected_provenance = training_provenance(
            index,
            manifest,
            scalers[case],
            config,
            schedule,
            recorded_runtime=provenance["runtime"],
        )
        if provenance != expected_provenance:
            raise ValueError(f"RSSM training provenance differs for {identity}")


def _load_score_table(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict) or set(payload) != {"columns", "records"}:
        raise ValueError("baseline score table wrapper is invalid")
    columns = payload["columns"]
    records = payload["records"]
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
        or any(not isinstance(item, str) for item in columns)
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("baseline score table is empty or duplicated")
    if any(not isinstance(record, dict) or set(record) != set(columns) for record in records):
        raise ValueError("baseline score records differ from their columns")
    return pd.DataFrame(records, columns=columns)


def _validate_baseline_receipt_common(
    receipt: dict,
    *,
    baseline: str,
    case: str,
    seed: int,
    fit_variants: Sequence,
    validation_variants: Sequence,
    scalers: FaultScalers,
) -> None:
    required = set(BaselineSelectionReceipt.__dataclass_fields__)
    if set(receipt) != required:
        raise ValueError("baseline selection receipt fields differ from schema")
    if (
        receipt["schema"] != BASELINE_RECEIPT_SCHEMA
        or receipt["baseline"] != baseline
        or receipt["case"] != case
        or receipt["model_seed"] != seed
        or receipt["fit_role"] != "fit"
        or receipt["validation_role"] != "validation"
        or receipt["fit_variant_identity_sha256"]
        != _variant_identity_sha256(fit_variants)
        or receipt["validation_variant_identity_sha256"]
        != _variant_identity_sha256(validation_variants)
        or receipt["fit_scalers_sha256"] != _scalers_sha256(scalers)
        or receipt["producer_code_sha256"]
        != baseline_producer_code_manifest()["sha256"]
    ):
        raise ValueError("baseline selection receipt has mismatched data identity")
    validate_numerical_runtime_fingerprint(
        receipt["runtime_fingerprint"], include_sklearn=True
    )
    for field in (
        "feature_contract_sha256",
        "training_config_sha256",
        "score_table_sha256",
        "schedule_sha256",
        "selected_model_state_sha256",
    ):
        if not _is_sha256(receipt[field]):
            raise ValueError(f"baseline selection receipt has invalid {field}")
    if receipt["selection_metric"] != (
        "equal_mean_case_family_fault_channel_standardized_H8_MAE"
    ):
        raise ValueError("baseline selection metric differs from protocol")


def _restore_ridge(model_payload: object, receipt: dict) -> Ridge:
    required = {
        "schema",
        "alpha",
        "fit_intercept",
        "n_features_in",
        "coef",
        "intercept",
        "state_sha256",
    }
    if not isinstance(model_payload, dict) or set(model_payload) != required:
        raise ValueError("selected Ridge payload fields are invalid")
    if model_payload["schema"] != RIDGE_STATE_SCHEMA:
        raise ValueError("selected Ridge state schema differs")
    if (
        isinstance(model_payload["alpha"], bool)
        or not isinstance(model_payload["alpha"], (int, float))
        or not np.isfinite(float(model_payload["alpha"]))
        or type(model_payload["fit_intercept"]) is not bool
        or isinstance(model_payload["n_features_in"], bool)
        or not isinstance(model_payload["n_features_in"], int)
        or model_payload["n_features_in"] <= 0
    ):
        raise ValueError("selected Ridge scalar metadata are invalid")
    coef_tensor = model_payload["coef"]
    intercept_tensor = model_payload["intercept"]
    if not isinstance(coef_tensor, torch.Tensor) or not isinstance(
        intercept_tensor, torch.Tensor
    ):
        raise ValueError("selected Ridge arrays are not tensors")
    coef = coef_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    intercept = intercept_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    if (
        coef.ndim != 2
        or coef.shape[0] != 4
        or coef.shape[1] != int(model_payload["n_features_in"])
        or intercept.shape != (4,)
        or not np.isfinite(coef).all()
        or not np.isfinite(intercept).all()
    ):
        raise ValueError("selected Ridge arrays have invalid shape or values")
    model = Ridge(
        alpha=float(model_payload["alpha"]),
        fit_intercept=bool(model_payload["fit_intercept"]),
    )
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()
    model.n_features_in_ = int(model_payload["n_features_in"])
    state_hash = _ridge_state_sha256(model)
    if (
        model_payload["state_sha256"] != state_hash
        or receipt["selected_model_state_sha256"] != state_hash
        or float(receipt["selected_candidate"]) != float(model.alpha)
    ):
        raise ValueError("selected Ridge state differs from its receipt")
    return model


def _restore_gru(model_payload: object, receipt: dict, config: StudyConfig) -> DirectH8GRU:
    if not isinstance(model_payload, dict) or set(model_payload) != {
        "schema",
        "model_config",
        "state_dict",
        "state_sha256",
    }:
        raise ValueError("selected GRU payload fields are invalid")
    spec = DirectH8FeatureSpec(horizon=config.direct_horizon)
    expected_config = {
        "observation_dim": config.observation_dim,
        "action_dim": config.action_dim,
        "context_dim": config.context_dim,
        "hidden_dim": config.hidden_dim,
        "history": spec.history,
        "horizon": spec.horizon,
    }
    if (
        model_payload["schema"] != GRU_STATE_SCHEMA
        or model_payload["model_config"] != expected_config
    ):
        raise ValueError("selected GRU configuration differs from protocol")
    state = model_payload["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("selected GRU state dictionary is missing")
    state_hash = _baseline_tensor_state_sha256(state)
    if (
        model_payload["state_sha256"] != state_hash
        or receipt["selected_model_state_sha256"] != state_hash
    ):
        raise ValueError("selected GRU state differs from its receipt")
    model = DirectH8GRU(
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        spec=spec,
    )
    model.load_state_dict(state, strict=True)
    return model


def _baseline_expected_entries(
    *,
    baseline: str,
    config: StudyConfig,
    gru_seeds: Sequence[int] | None = None,
    cases: Sequence[str] | None = None,
) -> set[str]:
    selected_cases = tuple(CASES if cases is None else cases)
    if (
        not selected_cases
        or len(set(selected_cases)) != len(selected_cases)
        or not set(selected_cases).issubset(CASES)
    ):
        raise ValueError("baseline case scope must be a nonempty unique subset")
    if baseline == "deterministic_gru":
        seeds = (
            tuple(config.confirmatory_seeds)
            if gru_seeds is None
            else tuple(gru_seeds)
        )
    else:
        seeds = (0,)
    return {f"{case}:seed{seed}" for case in selected_cases for seed in seeds}


def _validate_baseline_bundle(
    path: Path,
    *,
    baseline: str,
    config: StudyConfig,
    variants_by_case_role: Mapping[tuple[str, str], Sequence],
    scalers: Mapping[str, FaultScalers],
    gru_seeds: Sequence[int] | None = None,
    cases: Sequence[str] | None = None,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "baseline",
        "producer_code",
        "entries",
    }:
        raise ValueError("baseline selection bundle fields differ from schema")
    if (
        payload["schema"] != BASELINE_BUNDLE_SCHEMA
        or payload["baseline"] != baseline
        or payload["producer_code"] != baseline_producer_code_manifest()
    ):
        raise ValueError("baseline selection bundle identity differs")
    expected_entries = _baseline_expected_entries(
        baseline=baseline,
        config=config,
        gru_seeds=gru_seeds,
        cases=cases,
    )
    entries = payload["entries"]
    if not isinstance(entries, dict) or set(entries) != expected_entries:
        raise ValueError("baseline bundle entries do not cover the frozen identities")
    for identity, entry in sorted(entries.items()):
        case, seed_text = identity.split(":")
        seed = int(seed_text.removeprefix("seed"))
        if not isinstance(entry, dict) or set(entry) != {"receipt", "score_table", "model"}:
            raise ValueError(f"baseline bundle entry is invalid: {identity}")
        receipt = entry["receipt"]
        if not isinstance(receipt, dict):
            raise ValueError(f"baseline receipt is missing: {identity}")
        fit_variants = variants_by_case_role[(case, "fit")]
        validation_variants = variants_by_case_role[(case, "validation")]
        _validate_baseline_receipt_common(
            receipt,
            baseline=baseline,
            case=case,
            seed=seed,
            fit_variants=fit_variants,
            validation_variants=validation_variants,
            scalers=scalers[case],
        )
        table = _load_score_table(entry["score_table"])
        if receipt["score_table_sha256"] != _baseline_frame_sha256(table):
            raise ValueError(f"baseline score table differs from receipt: {identity}")
        selected = table.loc[table["selected"].map(lambda value: value is True)]
        candidate_column = "update" if baseline == "deterministic_gru" else "alpha"
        if len(selected) != 1 or candidate_column not in table or "validation_score" not in table:
            raise ValueError(f"baseline score selection is invalid: {identity}")
        ordered = table.sort_values(
            ["validation_score", candidate_column], kind="stable"
        ).reset_index(drop=True)
        if not bool(ordered.loc[0, "selected"]):
            raise ValueError(f"baseline selected candidate is not best: {identity}")
        if (
            float(selected.iloc[0][candidate_column])
            != float(receipt["selected_candidate"])
            or float(selected.iloc[0]["validation_score"])
            != float(receipt["selected_validation_score"])
        ):
            raise ValueError(f"baseline selected score differs from receipt: {identity}")
        if baseline == "ridge_arx":
            contract = ARXFeatureSpec().contract
            model = _restore_ridge(entry["model"], receipt)
            fit_x, _, references = arx_one_step_dataset(
                fit_variants, scalers[case]
            )
            schedule = [
                {
                    "cell_id": fit_variants[variant_index].cell.cell_id,
                    "source": source,
                }
                for variant_index, source in references
            ]
            expected_schedule_sha256 = _baseline_canonical_sha256(schedule)
            expected_candidates = list(RIDGE_ALPHAS)
            expected_updates = 1
            expected_batch_size = len(fit_x)
        elif baseline == "direct_h8_ridge":
            contract = DirectH8FeatureSpec().contract
            model = _restore_ridge(entry["model"], receipt)
            fit_x, _, references = direct_h8_dataset(
                fit_variants, scalers[case]
            )
            schedule = [
                {
                    "cell_id": fit_variants[variant_index].cell.cell_id,
                    "anchor": anchor,
                }
                for variant_index, anchor in references
            ]
            expected_schedule_sha256 = _baseline_canonical_sha256(schedule)
            expected_candidates = list(RIDGE_ALPHAS)
            expected_updates = 1
            expected_batch_size = len(fit_x)
        else:
            model = _restore_gru(entry["model"], receipt, config)
            contract = model.feature_contract
            _, _, references = direct_h8_dataset(fit_variants, scalers[case])
            _, expected_schedule_sha256 = _make_gru_schedule(
                fit_variants,
                references,
                config,
                case=case,
                model_seed=seed,
            )
            expected_candidates = list(config.validation_checkpoints)
            expected_updates = config.updates
            expected_batch_size = config.gru_batch_size
        if receipt["feature_contract_sha256"] != _baseline_canonical_sha256(contract):
            raise ValueError(f"baseline feature contract differs: {identity}")
        if (
            receipt["candidate_grid"] != expected_candidates
            or sorted(float(value) for value in table[candidate_column])
            != sorted(float(value) for value in expected_candidates)
            or len(table) != len(expected_candidates)
            or receipt["training_updates"] != expected_updates
            or receipt["batch_size"] != expected_batch_size
            or receipt["schedule_sha256"] != expected_schedule_sha256
        ):
            raise ValueError(f"baseline training/selection grid differs: {identity}")
        if (
            baseline != "deterministic_gru"
            and model.n_features_in_ != fit_x.shape[1]
        ):
            raise ValueError(f"selected Ridge feature dimension differs: {identity}")
    return payload


def _validate_gru_bundle_extension(
    development_payload: dict,
    confirmatory_payload: dict,
) -> None:
    """Require every screened GRU entry to survive semantically unchanged."""

    def equal(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left, right)
            )
        if isinstance(left, dict) or isinstance(right, dict):
            return (
                isinstance(left, dict)
                and isinstance(right, dict)
                and set(left) == set(right)
                and all(equal(left[key], right[key]) for key in left)
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return (
                isinstance(left, (list, tuple))
                and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(equal(a, b) for a, b in zip(left, right, strict=True))
            )
        return bool(left == right)

    development_entries = development_payload.get("entries")
    confirmatory_entries = confirmatory_payload.get("entries")
    if not isinstance(development_entries, dict) or not isinstance(
        confirmatory_entries, dict
    ):
        raise ValueError("GRU bundle extension entries are invalid")
    if not set(development_entries) < set(confirmatory_entries):
        raise ValueError("confirmatory GRU bundle is not a strict seed extension")
    for identity, development_entry in development_entries.items():
        if not equal(development_entry, confirmatory_entries[identity]):
            raise ValueError(
                f"confirmatory GRU bundle changed screened entry: {identity}"
            )


def _corpus_dependency_paths(root: Path, manifest_path: Path) -> set[str]:
    wrapper = load_strict_json(manifest_path)
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("manifest"), dict):
        raise ValueError("corpus dependency manifest wrapper is invalid")
    manifest = wrapper["manifest"]
    kind = manifest.get("collection_kind")
    if kind not in CORPUS_LAYOUT:
        raise ValueError("corpus dependency collection kind is invalid")
    raw_subdir = CORPUS_LAYOUT[kind][2]
    output_root = manifest_path.parent.parent.resolve()
    dependencies = {manifest_path.resolve()}
    try:
        dependencies.update(
            _safe_path(output_root, metadata["path"])
            for metadata in manifest["plans"].values()
        )
        dependencies.update(
            _safe_path(output_root, f"{raw_subdir}/{metadata['path']}")
            for metadata in manifest["files"]
        )
        dependencies.update(
            _safe_path(output_root, f"{raw_subdir}/{metadata['path']}")
            for metadata in manifest["receipts"].values()
        )
    except (KeyError, TypeError) as error:
        raise ValueError("corpus dependency metadata are incomplete") from error
    return {str(path.relative_to(root.resolve())) for path in dependencies}


def _declared_reference_paths(registry: dict) -> set[str]:
    paths = {
        registry["development_run_config_artifact"]["path"],
        registry["development_run_complete_artifact"]["path"],
        registry["development_rssm_result_artifact"]["path"],
        registry["development_baseline_result_artifact"]["path"],
        registry["development_gate_paired_artifact"]["path"],
        registry["development_gate_result_artifact"]["path"],
        registry["development_gru_baseline_artifact"]["path"],
        registry["corpus_manifest_artifact"]["path"],
        registry["fault_manifest_artifact"]["path"],
        registry["validation_selection_artifact"]["path"],
    }
    for field in (
        "fit_scaler_artifact_by_case",
        "checkpoint_artifact_by_identity",
        "training_schedule_artifact_by_case_seed",
        "baseline_selection_artifact_by_arm",
    ):
        paths.update(reference["path"] for reference in registry[field].values())
    return paths


def validate_prelock_registry_semantics(
    registry: dict,
    artifact_root: Path,
    config: StudyConfig,
    *,
    selected_update: int,
) -> list[str]:
    """Parse and cross-check every pre-lock artifact without opening locked values."""
    issues: list[str] = []
    try:
        corpus_path = _safe_path(
            artifact_root, registry["corpus_manifest_artifact"]["path"]
        )
        index = _load_development_index(corpus_path)
        _validate_fault_contract(
            _safe_path(artifact_root, registry["fault_manifest_artifact"]["path"]),
            index,
            config,
        )
        validation_rows_path = _validate_selection(
            _safe_path(
                artifact_root, registry["validation_selection_artifact"]["path"]
            ),
            config,
            selected_update,
        )
        scalers = _load_scalers(registry, artifact_root, index)
        manifest, fit_variants, recomputed_scalers = _expected_case_inputs(index, config)
        _validate_development_run_binding(
            registry,
            artifact_root,
            config,
            selected_update=selected_update,
            corpus_manifest_sha256=index.manifest_sha256,
            fault_manifest_sha256=manifest.sha256,
        )
        _validate_development_screen_evidence(
            registry,
            artifact_root,
            config,
            selected_update=selected_update,
        )
        if {case: asdict(value) for case, value in scalers.items()} != {
            case: asdict(value) for case, value in recomputed_scalers.items()
        }:
            raise ValueError("registered FIT scalers differ from training scalers")
        _validate_schedules_and_checkpoints(
            registry,
            artifact_root,
            index,
            config,
            selected_update,
            manifest,
            fit_variants,
            scalers,
        )
        variants_by_case_role: dict[tuple[str, str], Sequence] = {}
        for case in CASES:
            variants_by_case_role[(case, "fit")] = fit_variants[case]
            variants_by_case_role[(case, "validation")] = list(
                iter_role_variants(index, manifest, "validation", cases=(case,))
            )
        development_gru_payload = _validate_baseline_bundle(
            _safe_path(
                artifact_root,
                registry["development_gru_baseline_artifact"]["path"],
            ),
            baseline="deterministic_gru",
            config=config,
            variants_by_case_role=variants_by_case_role,
            scalers=scalers,
            gru_seeds=config.development_seeds,
        )
        confirmatory_gru_payload: dict | None = None
        for baseline, reference in sorted(
            registry["baseline_selection_artifact_by_arm"].items()
        ):
            payload = _validate_baseline_bundle(
                _safe_path(artifact_root, reference["path"]),
                baseline=baseline,
                config=config,
                variants_by_case_role=variants_by_case_role,
                scalers=scalers,
            )
            if baseline == "deterministic_gru":
                confirmatory_gru_payload = payload
        if confirmatory_gru_payload is None:
            raise ValueError("confirmatory GRU bundle is missing")
        _validate_gru_bundle_extension(
            development_gru_payload, confirmatory_gru_payload
        )
        inventory = registry.get("artifact_inventory")
        if not isinstance(inventory, list) or any(
            not isinstance(item, str) for item in inventory
        ):
            raise ValueError("pre-lock artifact inventory is invalid")
        allowed = _declared_reference_paths(registry) | _corpus_dependency_paths(
            artifact_root, corpus_path
        )
        allowed.add(str(validation_rows_path.relative_to(artifact_root.resolve())))
        if set(inventory) != allowed or len(inventory) != len(set(inventory)):
            raise ValueError(
                "pre-lock inventory contains nonsemantic or missing dependencies"
            )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        issues.append(f"semantic pre-lock validation failed: {error}")
    return issues


def validate_locked_corpus_binding(
    manifest_path: Path,
    *,
    prelock_registry: dict | Path,
    expected_prelock_sha256: str,
    artifact_root: Path | None = None,
    artifact_inventory: Sequence[str] | None = None,
    expected_plan_sha256_by_case: Mapping[str, str] | None = None,
) -> tuple[CorpusIndex | None, list[str]]:
    """Verify locked metadata is chronologically bound without reading CSV values."""
    issues: list[str] = []
    try:
        if not _is_sha256(expected_prelock_sha256):
            raise ValueError("expected pre-lock digest is invalid")
        if isinstance(prelock_registry, Path):
            loaded_registry = load_strict_json(prelock_registry)
        else:
            loaded_registry = prelock_registry
        if not isinstance(loaded_registry, dict):
            raise ValueError("pre-lock registry object is invalid")
        _strict_scan_corpus_json(manifest_path)
        index = load_corpus_index(
            manifest_path,
            prelock_registry=loaded_registry,
            expected_prelock_sha256=expected_prelock_sha256,
        )
        if index.collection_kind != "locked_test":
            raise ValueError("evaluation source is not a locked-test corpus")
        if set(index.allowed_roles) != {"locked_test"}:
            raise ValueError("locked corpus exposes a non-locked role")
        if {record.key.case for record in index.records} != set(CASES):
            raise ValueError("locked corpus does not contain every frozen case")
        if index.prelock_registry_sha256 != expected_prelock_sha256:
            raise ValueError("locked corpus embeds a different pre-lock digest")
        if expected_plan_sha256_by_case is not None and (
            dict(index.plan_sha256_by_case) != dict(expected_plan_sha256_by_case)
        ):
            raise ValueError("locked corpus plans differ from the pre-lock plans")
        if artifact_root is not None or artifact_inventory is not None:
            if artifact_root is None or artifact_inventory is None:
                raise ValueError("locked corpus inventory inputs are incomplete")
            if (
                any(not isinstance(item, str) for item in artifact_inventory)
                or len(artifact_inventory) != len(set(artifact_inventory))
            ):
                raise ValueError("locked corpus inventory is invalid")
            dependencies = _corpus_dependency_paths(artifact_root, manifest_path)
            if set(artifact_inventory) != dependencies:
                raise ValueError("locked evaluation root contains non-corpus files")
        return index, issues
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        issues.append(f"locked corpus binding failed: {error}")
        return None, issues
