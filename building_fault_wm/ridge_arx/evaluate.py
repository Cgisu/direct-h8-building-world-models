"""Secondary recursive Ridge-ARX evaluation on the completed v3 corpus."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from building_fault_wm.deterministic_transport import (
    corpus as v3_corpus,
    evaluate as v3_evaluate,
)
from building_fault_wm.transport_collection import (
    external_freeze as transport_external_freeze,
    runner as transport_runner,
)
from building_fault_wm.neural_benchmark.fault_data import (
    FAULT_CHANNELS,
    FaultScalers,
    FaultSpec,
    FaultVariant,
    ScaleStats,
    TrajectoryKey,
    build_fault_manifest,
    iter_role_variants,
)

from .config import CASES, FROZEN_CONFIG, POLICIES, SILENT_FAMILIES
from .external_freeze import validate_external_freeze_receipt
from .io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    write_csv_once,
    write_json_once,
)
from .lock import (
    DEFAULT_OUTPUT as DEFAULT_PRELOCK_ROOT,
    bind_completed_transport_metadata,
    verify_prelock,
)
from .train import (
    DEFAULT_OUTPUT as DEFAULT_TRAINING_ROOT,
    FAULT_CONTRACT,
    SCALER_ROOT,
    load_model_run,
    recursive_prediction,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_evaluation_v1"
)
ARM = "schedule_matched_recursive_ridge_arx"
RAW_UNITS = {
    "zone_temperature_k": "K",
    "hvac_electric_power_w": "W",
}
CORE_COLUMNS = (
    "case",
    "policy",
    "window_id",
    "trajectory_day",
    "scenario_seed",
    "trajectory_seed",
    "model_seed",
    "arm",
    "cell_id",
    "fault_channel",
    "family",
    "sign",
    "severity",
    "onset",
    "anchor",
    "horizon",
    "standardized_abs_error",
)
DETAIL_COLUMNS = (
    *CORE_COLUMNS,
    "fault_channel_index",
    "severity_unit",
    "history_start",
    "history_stop",
    "target_index",
    "target_raw",
    "prediction_raw",
    "prediction_standardized",
    "raw_abs_error",
    "raw_unit",
    "boundary_crossing",
    "action_transition_count",
)
OUTPUT_SCHEMA = "schedule-matched-recursive-ridge-arx-evaluation-v1"
COMPLETION_SCHEMA = (
    "schedule-matched-recursive-ridge-arx-evaluation-completion-v1"
)


def _scale_stats(payload: object, dimension: int, label: str) -> ScaleStats:
    if not isinstance(payload, dict) or set(payload) != {"mean", "scale"}:
        raise ValueError(f"{label} scaler fields changed")
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0.0).any()
    ):
        raise ValueError(f"{label} scaler values changed")
    return ScaleStats(tuple(mean), tuple(scale))


def load_frozen_scaler(case: str) -> FaultScalers:
    if case not in CASES:
        raise ValueError("ARX scaler case is outside the frozen grid")
    payload = strict_json(SCALER_ROOT / f"{case}.json")
    if set(payload) != {
        "observation",
        "action",
        "context",
        "fit_source_sha256",
    }:
        raise ValueError("ARX frozen scaler fields changed")
    sources = payload["fit_source_sha256"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in sources
        )
    ):
        raise ValueError("ARX frozen scaler sources changed")
    return FaultScalers(
        observation=_scale_stats(payload["observation"], 4, "observation"),
        action=_scale_stats(payload["action"], 1, "action"),
        context=_scale_stats(payload["context"], 5, "context"),
        fit_source_sha256=tuple((str(a), str(b)) for a, b in sources),
    )


def load_frozen_fault_spec() -> FaultSpec:
    payload = strict_json(FAULT_CONTRACT)
    if (
        payload.get("schema")
        != "boptest-multicase-frozen-fault-contract-v1"
        or not isinstance(payload.get("spec"), dict)
    ):
        raise ValueError("ARX frozen fault contract changed")
    spec = FaultSpec.from_dict(payload["spec"])
    if spec.evaluation_horizon != 8:
        raise ValueError("ARX fault contract is not H8")
    return spec


def _validate_secondary_rows(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != CORE_COLUMNS:
        raise ValueError("ARX secondary core columns changed")
    identity = [column for column in CORE_COLUMNS if column != "standardized_abs_error"]
    if frame.empty or frame.duplicated(identity).any():
        raise ValueError("ARX secondary rows are empty or duplicated")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("ARX secondary rows contain non-finite values")
    if (
        set(frame["case"]) != set(CASES)
        or set(frame["policy"]) != set(POLICIES)
        or set(frame["model_seed"]) != set(FROZEN_CONFIG.model_seeds)
        or set(frame["arm"]) != {ARM}
        or set(frame["family"]) != set(SILENT_FAMILIES)
        or set(frame["horizon"]) != set(FROZEN_CONFIG.horizons)
    ):
        raise ValueError("ARX secondary evaluation grid is incomplete")
    for case in CASES:
        if frame.loc[frame["case"] == case, "window_id"].nunique() != 12:
            raise ValueError(f"ARX window grid is incomplete for {case}")
    pair_columns = [
        "case",
        "window_id",
        "scenario_seed",
        "model_seed",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "anchor",
        "horizon",
    ]
    counts = frame.groupby(pair_columns, dropna=False)["policy"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    if any(value != tuple(sorted(POLICIES)) for value in counts):
        raise ValueError("ARX secondary evaluation lacks paired policies")


def evaluate_variants(
    models: Mapping[int, object],
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    trajectory_metadata: Mapping[
        TrajectoryKey, v3_evaluate.PolicyTrajectoryMetadata
    ],
    *,
    horizons: Sequence[int] = FROZEN_CONFIG.horizons,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate supplied models without consuming a post-anchor observation."""

    normalized_horizons = tuple(horizons)
    if (
        not normalized_horizons
        or len(set(normalized_horizons)) != len(normalized_horizons)
        or any(value not in FROZEN_CONFIG.horizons for value in normalized_horizons)
    ):
        raise ValueError("ARX evaluation horizons changed")
    if set(models) != set(FROZEN_CONFIG.model_seeds):
        raise ValueError("ARX evaluation model grid is incomplete")
    silent = tuple(
        variant
        for variant in variants
        if variant.cell.family in SILENT_FAMILIES
    )
    v3_evaluate._validate_policy_metadata(silent, trajectory_metadata)
    rows: list[dict[str, object]] = []
    mean = np.asarray(scalers.observation.mean)
    scale = np.asarray(scalers.observation.scale)
    for model_seed, model in sorted(models.items()):
        for variant in silent:
            cell = variant.cell
            key = cell.trajectory
            metadata = trajectory_metadata[key]
            channel = FAULT_CHANNELS.index(cell.fault_channel)
            for anchor in cell.anchors:
                for horizon in normalized_horizons:
                    if (
                        anchor < FROZEN_CONFIG.history
                        or anchor + horizon >= cell.stop
                        or anchor + horizon >= len(variant.clean_observations)
                    ):
                        raise ValueError("ARX evaluation endpoint leaves its contract")
                    prediction = recursive_prediction(
                        model,
                        variant,
                        scalers,
                        anchor,
                        horizon,
                    )
                    target_raw = variant.clean_observations[anchor + horizon]
                    target_standardized = scalers.observation.transform(
                        target_raw[None]
                    )[0]
                    prediction_raw = prediction * scale + mean
                    boundary, transition_count = (
                        v3_evaluate.action_block_crosses_transition(
                            variant.actions, anchor, horizon
                        )
                    )
                    rows.append(
                        {
                            "case": key.case,
                            "policy": metadata.policy,
                            "window_id": metadata.window_id,
                            "trajectory_day": key.day,
                            "scenario_seed": metadata.scenario_seed,
                            "trajectory_seed": key.trajectory_seed,
                            "model_seed": model_seed,
                            "arm": ARM,
                            "cell_id": cell.cell_id,
                            "fault_channel": cell.fault_channel,
                            "family": cell.family,
                            "sign": cell.sign,
                            "severity": cell.severity,
                            "onset": cell.onset,
                            "anchor": anchor,
                            "horizon": horizon,
                            "standardized_abs_error": float(
                                abs(
                                    prediction[channel]
                                    - target_standardized[channel]
                                )
                            ),
                            "fault_channel_index": channel,
                            "severity_unit": cell.severity_unit,
                            "history_start": anchor - FROZEN_CONFIG.history + 1,
                            "history_stop": anchor,
                            "target_index": anchor + horizon,
                            "target_raw": float(target_raw[channel]),
                            "prediction_raw": float(prediction_raw[channel]),
                            "prediction_standardized": float(prediction[channel]),
                            "raw_abs_error": float(
                                abs(
                                    prediction_raw[channel]
                                    - target_raw[channel]
                                )
                            ),
                            "raw_unit": RAW_UNITS[cell.fault_channel],
                            "boundary_crossing": bool(boundary),
                            "action_transition_count": transition_count,
                        }
                    )
    detailed = pd.DataFrame(rows, columns=DETAIL_COLUMNS).sort_values(
        [
            "case",
            "window_id",
            "policy",
            "model_seed",
            "fault_channel",
            "family",
            "sign",
            "severity",
            "onset",
            "anchor",
            "horizon",
        ],
        kind="stable",
    ).reset_index(drop=True)
    core = detailed.loc[:, CORE_COLUMNS].copy()
    if set(models) == set(FROZEN_CONFIG.model_seeds) and set(
        normalized_horizons
    ) == set(FROZEN_CONFIG.horizons):
        _validate_secondary_rows(core)
    return core, detailed


def summarize_secondary(core: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight fault cells, then seeds/windows, for descriptive reporting."""

    if tuple(core.columns) != CORE_COLUMNS or core.empty:
        raise ValueError("ARX secondary summary input changed")
    cell_keys = [
        "case",
        "policy",
        "window_id",
        "model_seed",
        "fault_channel",
        "family",
        "sign",
        "severity",
        "onset",
        "horizon",
    ]
    cells = core.groupby(cell_keys, as_index=False, dropna=False)[
        "standardized_abs_error"
    ].mean()
    stratum_keys = [
        "case",
        "policy",
        "fault_channel",
        "family",
        "horizon",
    ]
    summary = cells.groupby(stratum_keys, as_index=False, dropna=False)[
        "standardized_abs_error"
    ].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(
        columns={
            "mean": "equal_cell_mean_standardized_mae",
            "std": "equal_cell_std_standardized_mae",
            "count": "equal_cell_count",
        }
    )
    return summary.sort_values(stratum_keys, kind="stable").reset_index(drop=True)


def _write_outputs(
    output_root: Path,
    core: pd.DataFrame,
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    provenance: dict,
) -> Path:
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite ARX addendum evaluation: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    core_path = write_csv_once(output_root / "arx_core.csv", core)
    detail_path = write_csv_once(
        output_root / "arx_detailed_diagnostics.csv", detailed
    )
    summary_path = write_csv_once(
        output_root / "arx_descriptive_summary.csv", summary
    )
    provenance_path = write_json_once(
        output_root / "evaluation_provenance.json", provenance
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "row_count": len(core),
        "file_sha256_by_name": {
            "arx_core.csv": sha256_file(core_path),
            "arx_detailed_diagnostics.csv": sha256_file(detail_path),
            "arx_descriptive_summary.csv": sha256_file(summary_path),
            "evaluation_provenance.json": sha256_file(provenance_path),
        },
    }
    return write_json_once(output_root / "evaluation_complete.json", completion)


def run_evaluation(
    *,
    prelock_root: Path,
    addendum_external_freeze_receipt_path: Path,
    training_root: Path,
    transport_prelock_root: Path,
    transport_live_data_root: Path,
    transport_readiness_path: Path,
    transport_external_freeze_receipt_path: Path,
    transport_state_root: Path,
    transport_manifest_path: Path,
    transport_raw_root: Path,
    output_root: Path = DEFAULT_OUTPUT,
    live_external_freezes: bool = True,
) -> Path:
    """Validate both freezes, then and only then open v5 trajectory values."""

    started = time.perf_counter()
    registry = verify_prelock(prelock_root, verify_live_assets=True)
    addendum_freeze = validate_external_freeze_receipt(
        addendum_external_freeze_receipt_path,
        prelock_root,
        live=live_external_freezes,
    )
    current_binding = bind_completed_transport_metadata(
        transport_prelock_root=transport_prelock_root,
        transport_live_data_root=transport_live_data_root,
        transport_readiness_path=transport_readiness_path,
        transport_external_freeze_receipt_path=(
            transport_external_freeze_receipt_path
        ),
        transport_state_root=transport_state_root,
        transport_manifest_path=transport_manifest_path,
        live_transport_external_freeze=live_external_freezes,
    )
    frozen_binding = strict_json(
        prelock_root / "bundle/transport_collection_binding.json"
    )
    if current_binding != frozen_binding:
        raise ValueError(
            "transport collection metadata differs from the ARX pre-lock"
        )

    readiness_document = strict_json(transport_readiness_path)
    expected_prelock = str(readiness_document["prelock_registry_sha256"])
    expected_readiness = str(readiness_document["readiness_sha256"])
    _, readiness = transport_runner.load_bound_readiness(
        prelock_root=transport_prelock_root,
        live_data_root=transport_live_data_root,
        readiness_path=transport_readiness_path,
        expected_prelock_sha256=expected_prelock,
        expected_readiness_sha256=expected_readiness,
    )
    transport_freeze = (
        transport_external_freeze.validate_external_freeze_receipt(
            transport_external_freeze_receipt_path,
            expected_prelock,
            expected_readiness,
            prelock_root=transport_prelock_root,
            readiness_path=transport_readiness_path,
            live=live_external_freezes,
        )
    )
    # First permitted locked-response access occurs inside this call.
    collection = v3_corpus.load_transport_corpus_index(
        manifest_path=transport_manifest_path,
        raw_root=transport_raw_root,
        readiness=readiness,
        expected_prelock_sha256=expected_prelock,
        state_root=transport_state_root,
        external_freeze=transport_freeze,
        external_freeze_receipt_path=(
            transport_external_freeze_receipt_path
        ),
    )
    spec = load_frozen_fault_spec()
    fault_manifest = build_fault_manifest(collection.index, spec)

    core_frames = []
    detailed_frames = []
    model_hashes: dict[str, str] = {}
    for case in CASES:
        variants = tuple(
            iter_role_variants(
                collection.index,
                fault_manifest,
                "locked_test",
                cases=(case,),
                allow_locked_test=True,
            )
        )
        models = {}
        for seed in FROZEN_CONFIG.model_seeds:
            run_dir = training_root / case / f"seed{seed}"
            model, _ = load_model_run(run_dir, case=case, model_seed=seed)
            models[seed] = model
            model_hashes[f"{case}/seed{seed}"] = sha256_file(
                run_dir / "model.json"
            )
        metadata = {
            key: value
            for key, value in collection.trajectory_metadata.items()
            if key.case == case
        }
        core, detailed = evaluate_variants(
            models, variants, load_frozen_scaler(case), metadata
        )
        core_frames.append(core)
        detailed_frames.append(detailed)
    core = pd.concat(core_frames, ignore_index=True)
    detailed = pd.concat(detailed_frames, ignore_index=True)
    _validate_secondary_rows(core)
    summary = summarize_secondary(core)
    provenance = {
        "schema": OUTPUT_SCHEMA,
        "secondary_only": True,
        "cannot_modify_v2_or_v3_gate": True,
        "interpretation": (
            "descriptive engineering-comparator addendum; no primary gate "
            "category or decision may be recomputed"
        ),
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
        "prelock_registry_sha256": canonical_sha256(registry),
        "addendum_external_freeze_receipt_sha256": sha256_file(
            addendum_external_freeze_receipt_path
        ),
        "addendum_external_freeze_revision": addendum_freeze["revision"],
        "transport_collection_binding_sha256": canonical_sha256(
            current_binding
        ),
        "transport_manifest_file_sha256": collection.manifest_file_sha256,
        "fault_manifest_sha256": fault_manifest.sha256,
        "fault_spec": asdict(spec),
        "model_file_sha256_by_case_seed": model_hashes,
        "rows": len(core),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "wall_seconds": time.perf_counter() - started,
            "device": "cpu",
        },
    }
    return _write_outputs(output_root, core, detailed, summary, provenance)
