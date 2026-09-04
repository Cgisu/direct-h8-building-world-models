"""Train and verify the fixed schedule-matched recursive Ridge-ARX grid."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge

from building_fault_wm.neural_benchmark.baselines import (
    ARXFeatureSpec,
    arx_one_step_features,
)
from building_fault_wm.neural_benchmark.fault_data import (
    FAULT_CHANNELS,
    FaultScalers,
    FaultVariant,
    SequenceReference,
    build_fault_manifest,
    iter_role_variants,
    load_corpus_index,
)
from building_fault_wm.neural_benchmark.study_train import (
    ScheduledBatch,
    canonical_payload_sha256,
    prepare_case_training_data,
)

from .config import CASES, FROZEN_CONFIG, PARENT_PACKAGE_DIGEST, ARXAddendumConfig
from .io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_csv_once,
    write_json_once,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PARENT_ROOT = PROJECT_ROOT / "artifacts/direct_h8_publication_v2"
PARENT_MANIFEST = PARENT_ROOT / "package_manifest.json"
PARENT_DIGEST_FILE = PARENT_ROOT / "package_manifest.canonical.sha256"
PARENT_PRELOCK = PARENT_ROOT / "experiment/prelock_bundle"
DEVELOPMENT_MANIFEST = (
    PARENT_PRELOCK / "corpus/manifests/development_all_corpus_manifest.json"
)
SCHEDULE_ROOT = PARENT_PRELOCK / "frozen/schedules"
SCALER_ROOT = PARENT_PRELOCK / "frozen/fit_scalers"
FAULT_CONTRACT = PARENT_PRELOCK / "frozen/frozen_fault_contract.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_training_v2"
)
TERMINAL_V1_ROOT = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_training_v1"
)
TERMINAL_V1_LOG = (
    PROJECT_ROOT / "artifacts/schedule_matched_arx_transport_training_v1.log"
)
TERMINAL_V1_CLOSEOUT = (
    PROJECT_ROOT
    / "artifacts/schedule_matched_arx_transport_training_v1_failed_attempt_closeout_v1.json"
)

SCHEDULE_SCHEMA = "boptest-reliability-rssm-training-schedule-v1"
MODEL_SCHEMA = "schedule-matched-recursive-ridge-arx-model-v1"
RUN_SCHEMA = "schedule-matched-recursive-ridge-arx-training-v1"
GRID_SCHEMA = "schedule-matched-recursive-ridge-arx-grid-v1"
SOURCE_LOCK_SCHEMA = "schedule-matched-recursive-ridge-arx-source-lock-v1"
TERMINAL_CLOSEOUT_SCHEMA = (
    "schedule-matched-recursive-ridge-arx-terminal-training-closeout-v1"
)
GRID_RECEIPT = "training_grid_complete.json"
SOURCE_LOCK = "training_source_lock.json"


def source_manifest() -> dict[str, str]:
    names = {
        path.name
        for path in HERE.iterdir()
        if path.is_file()
        and path.suffix in {".py", ".md"}
        and not path.name.startswith("test_")
    }
    return {name: sha256_file(HERE / name) for name in sorted(names)}


def verify_parent_package() -> dict:
    manifest = strict_json(PARENT_MANIFEST)
    parent_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    if hashlib.sha256(parent_bytes).hexdigest() != PARENT_PACKAGE_DIGEST:
        raise ValueError("immutable parent manifest canonical digest changed")
    record = PARENT_DIGEST_FILE.read_text(encoding="ascii")
    if record != f"{PARENT_PACKAGE_DIGEST}\n":
        raise ValueError("immutable parent digest record changed")
    inventory = manifest.get("artifact_inventory_excludes_manifest_and_digest")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("immutable parent inventory is incomplete")
    total = 0
    seen: set[str] = set()
    for row in inventory:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("immutable parent inventory row is invalid")
        relative = str(row["path"])
        if relative in seen:
            raise ValueError("immutable parent inventory repeats a path")
        seen.add(relative)
        path = (PARENT_ROOT / relative).resolve()
        try:
            path.relative_to(PARENT_ROOT.resolve())
        except ValueError as error:
            raise ValueError("immutable parent path escapes its root") from error
        if (
            path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"immutable parent artifact changed: {relative}")
        total += int(row["bytes"])
    return {
        "canonical_digest": PARENT_PACKAGE_DIGEST,
        "inventory_file_count": len(inventory),
        "inventory_bytes": total,
    }


def _schedule_body(payload: Mapping[str, object]) -> dict[str, object]:
    if set(payload) != {"schema", "updates", "sha256"}:
        raise ValueError("parent schedule fields changed")
    body = {"schema": payload["schema"], "updates": payload["updates"]}
    if payload["schema"] != SCHEDULE_SCHEMA:
        raise ValueError("parent schedule schema changed")
    raw = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    if payload["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("parent schedule payload hash changed")
    return body


def load_parent_schedule(
    path: Path,
    variants: Sequence[FaultVariant],
    config: ARXAddendumConfig = FROZEN_CONFIG,
) -> tuple[ScheduledBatch, ...]:
    payload = strict_json(path)
    body = _schedule_body(payload)
    by_cell = {variant.cell.cell_id: index for index, variant in enumerate(variants)}
    if len(by_cell) != len(variants):
        raise ValueError("FIT variants repeat a cell identity")
    updates = body["updates"]
    if not isinstance(updates, list) or len(updates) != config.updates:
        raise ValueError("parent schedule update count changed")
    result = []
    for expected_update, raw_update in enumerate(updates, start=1):
        if not isinstance(raw_update, dict) or set(raw_update) != {
            "update",
            "latent_seed",
            "references",
        }:
            raise ValueError("parent schedule update fields changed")
        references = raw_update["references"]
        if (
            raw_update["update"] != expected_update
            or not isinstance(references, list)
            or len(references) != config.batch_size
        ):
            raise ValueError("parent schedule update grid changed")
        mapped = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {
                "cell_id",
                "aligned_start",
            }:
                raise ValueError("parent schedule reference fields changed")
            cell_id = reference["cell_id"]
            start = reference["aligned_start"]
            if cell_id not in by_cell or not isinstance(start, int):
                raise ValueError("parent schedule reference cannot be mapped")
            variant = variants[by_cell[cell_id]]
            if (
                variant.cell.trajectory.role != "fit"
                or start != variant.cell.onset - 17
                or start < 0
                or start + config.sequence_length
                >= len(variant.clean_observations)
            ):
                raise ValueError("parent schedule reference leaves its frozen sequence")
            mapped.append(SequenceReference(by_cell[cell_id], start))
        result.append(
            ScheduledBatch(
                update=expected_update,
                latent_seed=int(raw_update["latent_seed"]),
                references=tuple(mapped),
            )
        )
    return tuple(result)


def _feature_from_standardized(
    standardized_observations: np.ndarray,
    availability: np.ndarray,
    age: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    source: int,
    scalers: FaultScalers,
    config: ARXAddendumConfig = FROZEN_CONFIG,
) -> np.ndarray:
    start = source - config.history + 1
    if (
        start < 0
        or source - config.history < 0
        or source + 1 >= len(standardized_observations)
    ):
        raise ValueError("ARX source leaves its causal history or next target")
    observation = standardized_observations[start : source + 1]
    mask = availability[start : source + 1].astype(float)
    observation = np.where(mask.astype(bool), observation, 0.0)
    log_age = np.log1p(age[start : source + 1])
    previous_action = scalers.action.transform(
        actions[source - config.history : source]
    )
    current_action = scalers.action.transform(actions[source : source + 1])
    known_context = scalers.context.transform(contexts[source : source + 2])
    features = np.concatenate(
        [
            observation.reshape(-1),
            mask.reshape(-1),
            log_age.reshape(-1),
            previous_action.reshape(-1),
            current_action.reshape(-1),
            known_context.reshape(-1),
        ]
    )
    if features.shape != (config.feature_dim,) or not np.isfinite(features).all():
        raise ValueError("ARX feature vector differs from the frozen contract")
    return features


def recursive_prediction(
    model: Ridge,
    variant: FaultVariant,
    scalers: FaultScalers,
    anchor: int,
    horizon: int,
    *,
    candidate_actions: np.ndarray | None = None,
    config: ARXAddendumConfig = FROZEN_CONFIG,
) -> np.ndarray:
    """Roll a fitted ARX without consuming any observation after ``anchor``."""

    if horizon not in config.horizons:
        raise ValueError("ARX rollout horizon is outside the frozen grid")
    if anchor < config.history or anchor + horizon >= len(
        variant.clean_observations
    ):
        raise ValueError("ARX rollout leaves its history or trajectory")
    actions = np.array(variant.actions, copy=True)
    if candidate_actions is not None:
        candidate = np.asarray(candidate_actions, dtype=float)
        expected = (horizon, config.action_dim)
        if candidate.shape != expected or not np.isfinite(candidate).all():
            raise ValueError("candidate action block differs from the frozen shape")
        actions[anchor : anchor + horizon] = candidate
    standardized = np.zeros_like(variant.corrupted_observations, dtype=float)
    standardized[: anchor + 1] = scalers.observation.transform(
        variant.corrupted_observations[: anchor + 1]
    )
    availability = np.zeros_like(variant.availability, dtype=bool)
    availability[: anchor + 1] = variant.availability[: anchor + 1]
    age = np.zeros_like(variant.age, dtype=float)
    age[: anchor + 1] = variant.age[: anchor + 1]
    prediction = None
    for source in range(anchor, anchor + horizon):
        features = _feature_from_standardized(
            standardized,
            availability,
            age,
            actions,
            variant.contexts,
            source,
            scalers,
            config,
        )
        prediction = np.asarray(model.predict(features[None])[0], dtype=float)
        if prediction.shape != (config.observation_dim,) or not np.isfinite(
            prediction
        ).all():
            raise ValueError("ARX rollout produced an invalid prediction")
        standardized[source + 1] = prediction
        availability[source + 1] = True
        age[source + 1] = 0.0
    if prediction is None:
        raise AssertionError("ARX rollout performed no recursive step")
    return prediction


def scheduled_one_step_dataset(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    schedule: Sequence[ScheduledBatch],
    *,
    require_complete: bool = True,
    config: ARXAddendumConfig = FROZEN_CONFIG,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, str], ...], str]:
    """Materialize exact schedule multiplicities and sequence-contained sources."""

    normalized = tuple(schedule)
    if require_complete and (
        len(normalized) != config.updates
        or tuple(item.update for item in normalized)
        != tuple(range(1, config.updates + 1))
    ):
        raise ValueError("ARX training requires the complete parent schedule")
    if not normalized or any(
        len(item.references) != config.batch_size for item in normalized
    ):
        raise ValueError("ARX schedule has an incomplete batch")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[tuple[str, str]] = []
    semantic = hashlib.sha256()
    for scheduled in normalized:
        for batch_index, reference in enumerate(scheduled.references):
            if not 0 <= reference.variant_index < len(variants):
                raise ValueError("ARX schedule variant index is invalid")
            variant = variants[reference.variant_index]
            first = reference.aligned_start + config.history
            stop = reference.aligned_start + config.sequence_length
            if stop - first != config.scheduled_sources_per_sequence:
                raise AssertionError("ARX scheduled source count changed")
            for source in range(first, stop):
                features.append(
                    arx_one_step_features(
                        variant,
                        source,
                        scalers,
                        ARXFeatureSpec(
                            history=config.history,
                            horizon=max(config.horizons),
                        ),
                    )
                )
                targets.append(
                    scalers.observation.transform(
                        variant.clean_observations[source + 1 : source + 2]
                    )[0]
                )
                groups.append(
                    (variant.cell.fault_channel, variant.cell.family)
                )
                semantic.update(
                    (
                        f"{scheduled.update}:{batch_index}:"
                        f"{variant.cell.cell_id}:{source}\n"
                    ).encode("ascii")
                )
    x = np.stack(features)
    y = np.stack(targets)
    if x.shape[1] != config.feature_dim or y.shape[1] != config.observation_dim:
        raise ValueError("ARX scheduled dataset dimensions changed")
    if require_complete and len(x) != config.scheduled_rows_per_model:
        raise ValueError("ARX scheduled dataset row count changed")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ARX scheduled dataset contains non-finite values")
    return x, y, tuple(groups), semantic.hexdigest()


def validation_score(
    model: Ridge,
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    config: ARXAddendumConfig = FROZEN_CONFIG,
) -> float:
    rows = []
    for variant in variants:
        for anchor in variant.cell.anchors:
            prediction = recursive_prediction(
                model,
                variant,
                scalers,
                anchor,
                max(config.horizons),
                config=config,
            )
            target = scalers.observation.transform(
                variant.clean_observations[
                    anchor + max(config.horizons) :
                    anchor + max(config.horizons) + 1
                ]
            )[0]
            channel = FAULT_CHANNELS.index(variant.cell.fault_channel)
            rows.append(
                {
                    "family": variant.cell.family,
                    "fault_channel": variant.cell.fault_channel,
                    "error": abs(prediction[channel] - target[channel]),
                }
            )
    if not rows:
        raise ValueError("ARX validation grid is empty")
    cells = pd.DataFrame(rows).groupby(
        ["family", "fault_channel"], as_index=False, dropna=False
    )["error"].mean()
    return float(cells["error"].mean())


def _ridge_state_payload(model: Ridge) -> dict:
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    state = {
        "alpha": float(model.alpha),
        "fit_intercept": bool(model.fit_intercept),
        "solver": str(model.solver),
        "n_features_in": int(model.n_features_in_),
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
    }
    return {**state, "state_sha256": canonical_sha256(state)}


def restore_model(payload: Mapping[str, object]) -> Ridge:
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("ARX model schema changed")
    config = payload.get("config")
    if config != json.loads(json.dumps(FROZEN_CONFIG.to_dict())):
        raise ValueError("ARX model frozen config changed")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("ARX model state is missing")
    recorded = state.get("state_sha256")
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    if recorded != canonical_sha256(body):
        raise ValueError("ARX model state hash changed")
    coef = np.asarray(body["coef"], dtype=np.float64)
    intercept = np.asarray(body["intercept"], dtype=np.float64)
    if coef.shape != (
        FROZEN_CONFIG.observation_dim,
        FROZEN_CONFIG.feature_dim,
    ) or intercept.shape != (FROZEN_CONFIG.observation_dim,):
        raise ValueError("ARX model coefficient shape changed")
    model = Ridge(
        alpha=float(body["alpha"]),
        fit_intercept=bool(body["fit_intercept"]),
        solver=str(body["solver"]),
    )
    model.coef_ = coef
    model.intercept_ = intercept
    model.n_features_in_ = int(body["n_features_in"])
    if model.n_features_in_ != FROZEN_CONFIG.feature_dim:
        raise ValueError("ARX model input width changed")
    return model


def fit_case_seed(
    fit_variants: Sequence[FaultVariant],
    validation_variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    schedule: Sequence[ScheduledBatch],
    *,
    case: str,
    model_seed: int,
    schedule_file_sha256: str,
    schedule_payload_sha256: str,
    output_dir: Path,
    config: ARXAddendumConfig = FROZEN_CONFIG,
) -> Path:
    if case not in CASES or model_seed not in config.model_seeds:
        raise ValueError("ARX case/model-seed identity is outside the frozen grid")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite ARX run: {output_dir}")
    x, y, groups, exposure_sha256 = scheduled_one_step_dataset(
        fit_variants, scalers, schedule, config=config
    )
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    weights *= len(weights) / weights.sum()
    candidates: dict[float, Ridge] = {}
    rows = []
    started = time.perf_counter()
    for alpha in config.alphas:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="cholesky",
        )
        model.fit(x, y, sample_weight=weights)
        score = validation_score(model, validation_variants, scalers, config)
        candidates[alpha] = model
        rows.append({"alpha": alpha, "validation_h8_mae": score})
    table = pd.DataFrame(rows).sort_values(
        ["validation_h8_mae", "alpha"], kind="stable"
    ).reset_index(drop=True)
    table["selected"] = False
    table.loc[0, "selected"] = True
    selected_alpha = float(table.loc[0, "alpha"])
    selected = candidates[selected_alpha]
    state = _ridge_state_payload(selected)
    model_payload = {
        "schema": MODEL_SCHEMA,
        "case": case,
        "model_seed": model_seed,
        "config": json.loads(json.dumps(config.to_dict())),
        "state": state,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = write_json_once(output_dir / "model.json", model_payload)
    score_path = write_csv_once(output_dir / "validation_scores.csv", table)
    receipt = {
        "schema": RUN_SCHEMA,
        "case": case,
        "model_seed": model_seed,
        "config": json.loads(json.dumps(config.to_dict())),
        "selection_rule": (
            "minimum_parent_validation_equal_family_channel_h8_mae_"
            "ties_smaller_alpha"
        ),
        "selected_alpha": selected_alpha,
        "selected_validation_h8_mae": float(
            table.loc[0, "validation_h8_mae"]
        ),
        "training_rows": len(x),
        "feature_dim": x.shape[1],
        "target_dim": y.shape[1],
        "active_coefficients": config.active_coefficients,
        "stratum_row_count": {
            f"{channel}/{family}": count
            for (channel, family), count in sorted(counts.items())
        },
        "scheduled_exposure_sha256": exposure_sha256,
        "schedule_file_sha256": schedule_file_sha256,
        "schedule_payload_sha256": schedule_payload_sha256,
        "fit_scalers_sha256": canonical_payload_sha256(asdict(scalers)),
        "fit_variant_identity_sha256": canonical_sha256(
            [variant.cell.cell_id for variant in fit_variants]
        ),
        "validation_variant_identity_sha256": canonical_sha256(
            [variant.cell.cell_id for variant in validation_variants]
        ),
        "model_state_sha256": state["state_sha256"],
        "model_file_sha256": sha256_file(model_path),
        "score_table_file_sha256": sha256_file(score_path),
        "score_table_payload_sha256": canonical_sha256(
            table.to_dict(orient="records")
        ),
        "wall_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "device": "cpu",
        },
    }
    write_json_once(output_dir / "training_receipt.json", receipt)
    return output_dir


def load_model_run(
    run_dir: Path,
    *,
    case: str,
    model_seed: int,
) -> tuple[Ridge, dict]:
    receipt = strict_json(run_dir / "training_receipt.json")
    if (
        receipt.get("schema") != RUN_SCHEMA
        or receipt.get("case") != case
        or receipt.get("model_seed") != model_seed
        or receipt.get("config")
        != json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
    ):
        raise ValueError("ARX training receipt identity changed")
    model_path = run_dir / "model.json"
    score_path = run_dir / "validation_scores.csv"
    if (
        receipt.get("model_file_sha256") != sha256_file(model_path)
        or receipt.get("score_table_file_sha256") != sha256_file(score_path)
    ):
        raise ValueError("ARX run artifact hash changed")
    payload = strict_json(model_path)
    if (
        payload.get("case") != case
        or payload.get("model_seed") != model_seed
    ):
        raise ValueError("ARX model identity changed")
    model = restore_model(payload)
    table = pd.read_csv(score_path, float_precision="round_trip")
    if (
        tuple(table.columns)
        != ("alpha", "validation_h8_mae", "selected")
        or len(table) != len(FROZEN_CONFIG.alphas)
        or set(float(value) for value in table["alpha"])
        != set(FROZEN_CONFIG.alphas)
        or not np.isfinite(
            table[["alpha", "validation_h8_mae"]].to_numpy(dtype=float)
        ).all()
        or int(table["selected"].astype(bool).sum()) != 1
    ):
        raise ValueError("ARX validation score table changed")
    selected = table.loc[table["selected"].astype(bool)].iloc[0]
    if (
        receipt.get("model_state_sha256") != payload["state"]["state_sha256"]
        or float(receipt.get("selected_alpha")) != float(selected["alpha"])
        or float(receipt.get("selected_validation_h8_mae"))
        != float(selected["validation_h8_mae"])
        or float(model.alpha) != float(selected["alpha"])
        or receipt.get("training_rows")
        != FROZEN_CONFIG.scheduled_rows_per_model
        or receipt.get("feature_dim") != FROZEN_CONFIG.feature_dim
        or receipt.get("target_dim") != FROZEN_CONFIG.observation_dim
        or receipt.get("active_coefficients")
        != FROZEN_CONFIG.active_coefficients
        or receipt.get("score_table_payload_sha256")
        != canonical_sha256(table.to_dict(orient="records"))
    ):
        raise ValueError("ARX receipt binds a different model state")
    return model, receipt


def _frozen_scaler_payload(case: str) -> dict:
    payload = strict_json(SCALER_ROOT / f"{case}.json")
    payload["fit_source_sha256"] = [
        tuple(item) for item in payload["fit_source_sha256"]
    ]
    return payload


def train_grid(
    output_root: Path = DEFAULT_OUTPUT,
    *,
    cases: Iterable[str] = CASES,
    seeds: Iterable[int] = FROZEN_CONFIG.model_seeds,
) -> Path:
    selected_cases = tuple(cases)
    selected_seeds = tuple(seeds)
    if not selected_cases or set(selected_cases) - set(CASES):
        raise ValueError("ARX cases are outside the frozen grid")
    if not selected_seeds or set(selected_seeds) - set(FROZEN_CONFIG.model_seeds):
        raise ValueError("ARX seeds are outside the frozen grid")
    source_before = source_manifest()
    parent_before = verify_parent_package()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("ARX training root is not a plain directory")
    lock_payload = {
        "schema": SOURCE_LOCK_SCHEMA,
        "parent_package": parent_before,
        "source_manifest": source_before,
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
    }
    source_lock_path = output_root / SOURCE_LOCK
    if os.path.lexists(source_lock_path):
        if strict_json(source_lock_path) != lock_payload:
            raise ValueError("ARX training source lock changed")
    else:
        if any(output_root.iterdir()):
            raise ValueError("nonempty ARX training root has no source lock")
        write_json_once(source_lock_path, lock_payload)

    index = load_corpus_index(DEVELOPMENT_MANIFEST)
    if index.collection_kind != "development":
        raise ValueError("parent development corpus kind changed")
    fault_manifest = build_fault_manifest(index)
    validation_by_case = {
        case: tuple(
            iter_role_variants(
                index,
                fault_manifest,
                "validation",
                cases=(case,),
            )
        )
        for case in selected_cases
    }
    runs = []
    for case in selected_cases:
        fit_variants, scalers = prepare_case_training_data(
            index, fault_manifest, case
        )
        if canonical_payload_sha256(asdict(scalers)) != canonical_payload_sha256(
            _frozen_scaler_payload(case)
        ):
            raise ValueError(f"ARX FIT scalers differ from parent bytes: {case}")
        for seed in selected_seeds:
            schedule_path = SCHEDULE_ROOT / case / f"seed{seed}.json"
            schedule_document = strict_json(schedule_path)
            schedule = load_parent_schedule(
                schedule_path, fit_variants, FROZEN_CONFIG
            )
            run_dir = output_root / case / f"seed{seed}"
            if run_dir.exists():
                _, receipt = load_model_run(
                    run_dir, case=case, model_seed=seed
                )
                status = "verified_existing"
            else:
                fit_case_seed(
                    fit_variants,
                    validation_by_case[case],
                    scalers,
                    schedule,
                    case=case,
                    model_seed=seed,
                    schedule_file_sha256=sha256_file(schedule_path),
                    schedule_payload_sha256=str(schedule_document["sha256"]),
                    output_dir=run_dir,
                )
                _, receipt = load_model_run(
                    run_dir, case=case, model_seed=seed
                )
                status = "trained"
            runs.append(
                {
                    "case": case,
                    "model_seed": seed,
                    "status": status,
                    "selected_alpha": receipt["selected_alpha"],
                    "selected_validation_h8_mae": receipt[
                        "selected_validation_h8_mae"
                    ],
                    "schedule_file_sha256": receipt[
                        "schedule_file_sha256"
                    ],
                    "schedule_payload_sha256": receipt[
                        "schedule_payload_sha256"
                    ],
                    "model_file_sha256": receipt["model_file_sha256"],
                    "training_receipt_sha256": sha256_file(
                        run_dir / "training_receipt.json"
                    ),
                }
            )
    if source_manifest() != source_before:
        raise AssertionError("ARX source changed during training")
    if verify_parent_package() != parent_before:
        raise AssertionError("immutable parent changed during ARX training")
    complete = (
        set(selected_cases) == set(CASES)
        and set(selected_seeds) == set(FROZEN_CONFIG.model_seeds)
    )
    payload = {
        "schema": GRID_SCHEMA,
        "complete_grid": complete,
        "parent_package": parent_before,
        "development_manifest_file_sha256": sha256_file(
            DEVELOPMENT_MANIFEST
        ),
        "fault_contract_file_sha256": sha256_file(FAULT_CONTRACT),
        "fault_manifest_sha256": fault_manifest.sha256,
        "fit_scaler_file_sha256_by_case": {
            case: sha256_file(SCALER_ROOT / f"{case}.json")
            for case in CASES
        },
        "source_manifest": source_before,
        "source_lock_file_sha256": sha256_file(source_lock_path),
        "config": json.loads(json.dumps(FROZEN_CONFIG.to_dict())),
        "runs": sorted(runs, key=lambda row: (row["case"], row["model_seed"])),
    }
    name = GRID_RECEIPT if complete else (
        f"training_subset_{canonical_sha256([selected_cases, selected_seeds])[:16]}.json"
    )
    path = output_root / name
    if os.path.lexists(path):
        existing = strict_json(path)
        normalized_existing = {
            **existing,
            "runs": [
                {**row, "status": "verified_existing"}
                for row in existing["runs"]
            ],
        }
        normalized_payload = {
            **payload,
            "runs": [
                {**row, "status": "verified_existing"}
                for row in payload["runs"]
            ],
        }
        if normalized_existing != normalized_payload:
            raise ValueError("existing ARX training-grid receipt changed")
        return path
    write_json_once(path, payload)
    return path


def verify_training_grid(output_root: Path = DEFAULT_OUTPUT) -> dict:
    receipt = strict_json(output_root / GRID_RECEIPT)
    expected_scalers = {
        case: sha256_file(SCALER_ROOT / f"{case}.json") for case in CASES
    }
    if (
        receipt.get("schema") != GRID_SCHEMA
        or receipt.get("complete_grid") is not True
        or receipt.get("parent_package") != verify_parent_package()
        or receipt.get("source_manifest") != source_manifest()
        or receipt.get("config")
        != json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
        or receipt.get("development_manifest_file_sha256")
        != sha256_file(DEVELOPMENT_MANIFEST)
        or receipt.get("fault_contract_file_sha256")
        != sha256_file(FAULT_CONTRACT)
        or receipt.get("fit_scaler_file_sha256_by_case") != expected_scalers
    ):
        raise ValueError("ARX complete training-grid identity changed")
    source_lock_path = output_root / SOURCE_LOCK
    source_lock = strict_json(source_lock_path)
    if (
        source_lock.get("schema") != SOURCE_LOCK_SCHEMA
        or source_lock.get("parent_package") != verify_parent_package()
        or source_lock.get("source_manifest") != source_manifest()
        or source_lock.get("config")
        != json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
        or receipt.get("source_lock_file_sha256")
        != sha256_file(source_lock_path)
    ):
        raise ValueError("ARX training source lock changed")
    expected = {
        (case, seed) for case in CASES for seed in FROZEN_CONFIG.model_seeds
    }
    runs = receipt.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != len(expected)
        or any(not isinstance(row, dict) for row in runs)
        or {
            (row.get("case"), row.get("model_seed"))
            for row in runs
            if isinstance(row, dict)
        }
        != expected
    ):
        raise ValueError("ARX training-grid run identities changed")
    by_identity = {
        (str(row["case"]), int(row["model_seed"])): row for row in runs
    }
    for case, seed in sorted(expected):
        run_dir = output_root / case / f"seed{seed}"
        _, run_receipt = load_model_run(
            run_dir,
            case=case,
            model_seed=seed,
        )
        grid_row = by_identity[(case, seed)]
        schedule_path = SCHEDULE_ROOT / case / f"seed{seed}.json"
        schedule_document = strict_json(schedule_path)
        _schedule_body(schedule_document)
        expected_schedule_file = sha256_file(schedule_path)
        expected_schedule_payload = schedule_document["sha256"]
        if (
            run_receipt.get("schedule_file_sha256")
            != expected_schedule_file
            or run_receipt.get("schedule_payload_sha256")
            != expected_schedule_payload
            or run_receipt.get("fit_scalers_sha256")
            != canonical_payload_sha256(
                _frozen_scaler_payload(case)
            )
            or grid_row.get("schedule_file_sha256")
            != expected_schedule_file
            or grid_row.get("schedule_payload_sha256")
            != expected_schedule_payload
            or grid_row.get("model_file_sha256")
            != run_receipt.get("model_file_sha256")
            or grid_row.get("training_receipt_sha256")
            != sha256_file(run_dir / "training_receipt.json")
        ):
            raise ValueError(
                f"ARX training run differs from parent assets: {case}/seed{seed}"
            )
    inventory = tree_inventory(output_root)
    return {
        "schema": GRID_SCHEMA,
        "receipt_file_sha256": sha256_file(output_root / GRID_RECEIPT),
        "inventory": inventory,
        "inventory_sha256": canonical_sha256(inventory),
        "runs": len(expected),
    }


def _terminal_v1_expected_paths() -> set[str]:
    case = CASES[0]
    seeds = FROZEN_CONFIG.model_seeds[:2]
    return {
        SOURCE_LOCK,
        "training_subset_51748a3aa06e4516.json",
        *{
            f"{case}/seed{seed}/{name}"
            for seed in seeds
            for name in (
                "model.json",
                "training_receipt.json",
                "validation_scores.csv",
            )
        },
    }


def build_terminal_v1_closeout(
    *,
    training_root: Path = TERMINAL_V1_ROOT,
    traceback_log: Path = TERMINAL_V1_LOG,
    output_path: Path = TERMINAL_V1_CLOSEOUT,
) -> Path:
    """Seal the parser-failed v1 partial grid without treating it as reusable."""

    inventory = tree_inventory(training_root)
    if {str(row["path"]) for row in inventory} != _terminal_v1_expected_paths():
        raise ValueError("terminal ARX v1 inventory differs from the failed attempt")
    source_lock_path = training_root / SOURCE_LOCK
    source_lock = strict_json(source_lock_path)
    if (
        source_lock.get("schema") != SOURCE_LOCK_SCHEMA
        or source_lock.get("config")
        != json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
    ):
        raise ValueError("terminal ARX v1 source lock is invalid")
    if traceback_log.is_symlink() or not traceback_log.is_file():
        raise ValueError("terminal ARX v1 traceback log is missing")
    log_text = traceback_log.read_text(encoding="ascii")
    expected_trace = (
        'train.py", line 621, in load_model_run',
        "ValueError: ARX receipt binds a different model state",
    )
    if not all(token in log_text for token in expected_trace):
        raise ValueError("terminal ARX v1 traceback differs from the parser failure")
    body = {
        "schema": TERMINAL_CLOSEOUT_SCHEMA,
        "terminal": True,
        "complete_grid": False,
        "continuation_under_v1_source_lock_permitted": False,
        "locked_transport_values_accessed": False,
        "failure_stage": "development_only_load_after_write_verification",
        "failure_class": "pandas_default_csv_float_parser_round_trip_mismatch",
        "failure_message": "ARX receipt binds a different model state",
        "replacement_training_namespace": (
            "artifacts/schedule_matched_arx_transport_training_v2"
        ),
        "training_root": str(training_root.resolve()),
        "source_lock_file_sha256": sha256_file(source_lock_path),
        "source_lock_payload_sha256": canonical_sha256(source_lock),
        "completed_unit_directories": [
            f"{CASES[0]}/seed{seed}" for seed in FROZEN_CONFIG.model_seeds[:2]
        ],
        "subset_receipt": {
            "path": "training_subset_51748a3aa06e4516.json",
            "sha256": sha256_file(
                training_root / "training_subset_51748a3aa06e4516.json"
            ),
        },
        "traceback_log": {
            "path": str(traceback_log.resolve()),
            "bytes": traceback_log.stat().st_size,
            "sha256": sha256_file(traceback_log),
        },
        "training_inventory": inventory,
        "training_inventory_sha256": canonical_sha256(inventory),
    }
    payload = {**body, "closeout_payload_sha256": canonical_sha256(body)}
    write_json_once(output_path, payload)
    verify_terminal_v1_closeout(
        training_root=training_root,
        traceback_log=traceback_log,
        closeout_path=output_path,
    )
    return output_path


def verify_terminal_v1_closeout(
    *,
    training_root: Path = TERMINAL_V1_ROOT,
    traceback_log: Path = TERMINAL_V1_LOG,
    closeout_path: Path = TERMINAL_V1_CLOSEOUT,
) -> dict:
    """Verify the exact terminal partial tree, source lock, and traceback."""

    payload = strict_json(closeout_path)
    body = {
        key: value
        for key, value in payload.items()
        if key != "closeout_payload_sha256"
    }
    if (
        payload.get("closeout_payload_sha256") != canonical_sha256(body)
        or payload.get("schema") != TERMINAL_CLOSEOUT_SCHEMA
        or payload.get("terminal") is not True
        or payload.get("complete_grid") is not False
        or payload.get("continuation_under_v1_source_lock_permitted") is not False
        or payload.get("locked_transport_values_accessed") is not False
        or payload.get("training_root") != str(training_root.resolve())
        or payload.get("completed_unit_directories")
        != [
            f"{CASES[0]}/seed{seed}"
            for seed in FROZEN_CONFIG.model_seeds[:2]
        ]
        or payload.get("replacement_training_namespace")
        != "artifacts/schedule_matched_arx_transport_training_v2"
    ):
        raise ValueError("terminal ARX v1 closeout identity changed")
    inventory = tree_inventory(training_root)
    if (
        {str(row["path"]) for row in inventory} != _terminal_v1_expected_paths()
        or payload.get("training_inventory") != inventory
        or payload.get("training_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise ValueError("terminal ARX v1 training tree changed")
    source_lock_path = training_root / SOURCE_LOCK
    source_lock = strict_json(source_lock_path)
    if (
        payload.get("source_lock_file_sha256") != sha256_file(source_lock_path)
        or payload.get("source_lock_payload_sha256")
        != canonical_sha256(source_lock)
    ):
        raise ValueError("terminal ARX v1 source lock changed")
    subset = payload.get("subset_receipt")
    if (
        not isinstance(subset, dict)
        or subset.get("path") != "training_subset_51748a3aa06e4516.json"
        or subset.get("sha256")
        != sha256_file(training_root / str(subset.get("path")))
    ):
        raise ValueError("terminal ARX v1 subset receipt changed")
    log = payload.get("traceback_log")
    if (
        not isinstance(log, dict)
        or log.get("path") != str(traceback_log.resolve())
        or log.get("bytes") != traceback_log.stat().st_size
        or log.get("sha256") != sha256_file(traceback_log)
    ):
        raise ValueError("terminal ARX v1 traceback log changed")
    return payload
