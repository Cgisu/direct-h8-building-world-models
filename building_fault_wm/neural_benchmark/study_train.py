from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from building_fault_wm.recurrent_models.training import (
    RSSMSequenceInputs,
    RSSMSequenceTargets,
)

from .fault_data import (
    FAMILIES,
    FAULT_CHANNELS,
    CorpusIndex,
    FaultManifest,
    FaultScalers,
    FaultVariant,
    SequenceReference,
    build_fault_manifest,
    fit_scalers,
    iter_role_variants,
    load_role_trajectories,
    materialize_rssm_batch,
    validate_fault_manifest,
)
from .reliability_loss import reliability_sequence_training_loss
from .reliability_model import ReliabilityGatedRSSM
from .runtime_provenance import (
    numerical_runtime_fingerprint,
    validate_numerical_runtime_fingerprint,
)
from .study_config import ARMS, ArmName, StudyConfig


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    latent_seed: int
    references: tuple[SequenceReference, ...]


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def tensor_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


GATE_STATE_PREFIXES = ("reliability_feature_net.", "health_head.")


def core_tensor_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    core = {
        name: value
        for name, value in state_dict.items()
        if not name.startswith(GATE_STATE_PREFIXES)
    }
    if not core or len(core) == len(state_dict):
        raise ValueError("model state does not expose the expected gate/core partition")
    return tensor_state_sha256(core)


def canonical_payload_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(content).hexdigest()


RSSM_PRODUCER_CODE_SCHEMA = "boptest-reliability-rssm-producer-code-v1"


def rssm_producer_code_manifest() -> dict:
    """Hash every implementation file that can change trained RSSM tensors."""
    here = Path(__file__).resolve().parent
    files = {
        "multicase_fault_benchmark/study_train.py": here / "study_train.py",
        "multicase_fault_benchmark/reliability_model.py": (
            here / "reliability_model.py"
        ),
        "multicase_fault_benchmark/reliability_loss.py": (
            here / "reliability_loss.py"
        ),
        "multicase_fault_benchmark/fault_data.py": here / "fault_data.py",
        "multicase_fault_benchmark/study_config.py": here / "study_config.py",
        "multicase_fault_benchmark/runtime_provenance.py": (
            here / "runtime_provenance.py"
        ),
        "health_rssm/model.py": here.parent / "health_rssm" / "model.py",
        "health_rssm/training.py": here.parent / "health_rssm" / "training.py",
    }
    payload = {
        "schema": RSSM_PRODUCER_CODE_SCHEMA,
        "files": {
            identity: hashlib.sha256(path.read_bytes()).hexdigest()
            for identity, path in sorted(files.items())
        },
    }
    return {**payload, "sha256": canonical_payload_sha256(payload)}


def validate_rssm_producer_code_manifest(value: object) -> None:
    if value != rssm_producer_code_manifest():
        raise ValueError(
            "RSSM producer-code manifest differs from the current implementation"
        )


def training_provenance(
    index: CorpusIndex,
    manifest: FaultManifest,
    scalers: FaultScalers,
    config: StudyConfig,
    schedule: dict,
    *,
    device: torch.device | str | None = None,
    recorded_runtime: Mapping | None = None,
) -> dict:
    """Return the immutable inputs that must accompany every checkpoint."""
    if (device is None) == (recorded_runtime is None):
        raise ValueError("provide exactly one of device or recorded_runtime")
    if recorded_runtime is None:
        runtime = numerical_runtime_fingerprint(device, include_sklearn=False)
    else:
        validate_numerical_runtime_fingerprint(
            recorded_runtime, include_sklearn=False
        )
        runtime = dict(recorded_runtime)
    scaler_payload = asdict(scalers)
    fit_sources = dict(scalers.fit_source_sha256)
    return {
        "corpus_manifest_sha256": index.manifest_sha256,
        "fault_manifest_sha256": manifest.sha256,
        "fit_scalers_sha256": canonical_payload_sha256(scaler_payload),
        "fit_source_sha256": canonical_payload_sha256(fit_sources),
        "fit_source_sha256_by_trajectory": fit_sources,
        "training_schedule_sha256": schedule["sha256"],
        "config_sha256": canonical_payload_sha256(config.to_dict()),
        "producer_code": rssm_producer_code_manifest(),
        "runtime": runtime,
    }


def create_matched_models(
    config: StudyConfig,
    model_seed: int,
    *,
    arms: Sequence[ArmName] = ARMS,
    device: torch.device | str = "cpu",
) -> dict[ArmName, ReliabilityGatedRSSM]:
    if model_seed not in config.confirmatory_seeds:
        raise ValueError("model seed is outside the frozen paired seed set")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("study arms must be nonempty and unique")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown study arms: {sorted(unknown)}")
    torch.manual_seed(model_seed)
    reference = ReliabilityGatedRSSM(config.model_config()).to(device)
    models = {arm: copy.deepcopy(reference).to(device) for arm in arms}
    reference_hash = tensor_state_sha256(reference.state_dict())
    for arm, model in models.items():
        if tensor_state_sha256(model.state_dict()) != reference_hash:
            raise AssertionError(f"arm {arm} did not receive matched initialization")
    model_list = list(models.values())
    for left_index, left in enumerate(model_list):
        for right in model_list[left_index + 1 :]:
            for left_parameter, right_parameter in zip(
                left.parameters(), right.parameters(), strict=True
            ):
                if left_parameter.data_ptr() == right_parameter.data_ptr():
                    raise AssertionError("paired study arms share parameter storage")
    return models


def _variant_groups(
    variants: Sequence[FaultVariant],
) -> dict[tuple[str, str], tuple[int, ...]]:
    groups: dict[tuple[str, str], list[int]] = {
        (channel, family): []
        for channel in FAULT_CHANNELS
        for family in FAMILIES
    }
    for index, variant in enumerate(variants):
        key = (variant.cell.fault_channel, variant.cell.family)
        if key not in groups:
            raise ValueError(f"variant has an unknown training stratum: {key}")
        groups[key].append(index)
    missing = [key for key, indices in groups.items() if not indices]
    if missing:
        raise ValueError(f"training variants are missing strata: {missing}")
    return {key: tuple(indices) for key, indices in groups.items()}


def make_training_schedule(
    variants: Sequence[FaultVariant],
    config: StudyConfig,
    *,
    case: str,
    model_seed: int,
) -> tuple[ScheduledBatch, ...]:
    """Create one deterministic, group-balanced schedule shared by all arms."""
    groups = _variant_groups(variants)
    group_keys = tuple(sorted(groups))
    rng = np.random.Generator(
        np.random.PCG64(stable_seed(config.schedule_seed, case, model_seed))
    )
    queue: list[tuple[str, str]] = []
    schedule: list[ScheduledBatch] = []
    selected_group_counts = {key: 0 for key in group_keys}
    for update in range(1, config.updates + 1):
        references: list[SequenceReference] = []
        for _ in range(config.batch_size):
            if not queue:
                queue = [group_keys[index] for index in rng.permutation(len(group_keys))]
            key = queue.pop()
            selected_group_counts[key] += 1
            candidates = groups[key]
            variant_index = candidates[int(rng.integers(len(candidates)))]
            onset = variants[variant_index].cell.onset
            aligned_start = onset - 17
            if aligned_start < 0:
                raise ValueError("training fault onset does not leave a healthy prefix")
            references.append(SequenceReference(variant_index, aligned_start))
        schedule.append(
            ScheduledBatch(
                update=update,
                latent_seed=stable_seed(
                    config.schedule_seed, case, model_seed, "latent", update
                ),
                references=tuple(references),
            )
        )
    counts = np.asarray(tuple(selected_group_counts.values()))
    if counts.max() - counts.min() > 1:
        raise AssertionError("training schedule is not balanced across channel/family strata")
    return tuple(schedule)


def schedule_payload(
    schedule: Sequence[ScheduledBatch], variants: Sequence[FaultVariant]
) -> dict:
    payload = {
        "schema": "boptest-reliability-rssm-training-schedule-v1",
        "updates": [
            {
                "update": item.update,
                "latent_seed": item.latent_seed,
                "references": [
                    {
                        "cell_id": variants[reference.variant_index].cell.cell_id,
                        "aligned_start": reference.aligned_start,
                    }
                    for reference in item.references
                ],
            }
            for item in schedule
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def materialize_torch_batch(
    variants: Sequence[FaultVariant],
    scalers: FaultScalers,
    scheduled: ScheduledBatch,
    config: StudyConfig,
    *,
    device: torch.device | str = "cpu",
) -> tuple[RSSMSequenceInputs, RSSMSequenceTargets]:
    batch = materialize_rssm_batch(
        variants,
        scalers,
        scheduled.references,
        length=config.sequence_length,
    )
    corrupted = torch.as_tensor(
        batch.corrupted_observations, dtype=torch.float32, device=device
    )
    clean = torch.as_tensor(batch.clean_targets, dtype=torch.float32, device=device)
    availability = torch.as_tensor(batch.availability, dtype=torch.bool, device=device)
    age = torch.as_tensor(batch.age, dtype=torch.float32, device=device)
    previous_actions = torch.as_tensor(
        batch.previous_actions, dtype=torch.float32, device=device
    )
    contexts = torch.as_tensor(batch.contexts, dtype=torch.float32, device=device)
    binary_health = torch.as_tensor(
        batch.health_labels != 0, dtype=torch.long, device=device
    )
    time_steps, batch_size = clean.shape[:2]
    valid_steps = torch.ones(
        (time_steps, batch_size), dtype=torch.bool, device=device
    )
    inputs = RSSMSequenceInputs(
        previous_actions=previous_actions,
        corrupted_observations=corrupted,
        availability=availability,
        age=age,
        contexts=contexts,
    )
    targets = RSSMSequenceTargets(
        clean_observations=clean,
        clean_observation_mask=torch.ones_like(clean, dtype=torch.bool),
        costs=torch.zeros((time_steps, batch_size, 1), device=device),
        cost_mask=torch.zeros(
            (time_steps, batch_size, 1), dtype=torch.bool, device=device
        ),
        constraints=torch.zeros((time_steps, batch_size, 1), device=device),
        constraint_mask=torch.zeros(
            (time_steps, batch_size, 1), dtype=torch.bool, device=device
        ),
        continuations=torch.zeros((time_steps, batch_size, 1), device=device),
        continuation_mask=torch.zeros(
            (time_steps, batch_size, 1), dtype=torch.bool, device=device
        ),
        health_labels=binary_health,
        health_mask=torch.ones_like(binary_health, dtype=torch.bool),
        valid_steps=valid_steps,
    )
    return inputs, targets


def prepare_case_training_data(
    index: CorpusIndex,
    manifest: FaultManifest,
    case: str,
) -> tuple[list[FaultVariant], FaultScalers]:
    validate_fault_manifest(manifest, index)
    clean_fit = load_role_trajectories(index, "fit", cases=(case,))
    scalers = fit_scalers(clean_fit)
    variants = list(iter_role_variants(index, manifest, "fit", cases=(case,)))
    if not variants or {variant.cell.trajectory.case for variant in variants} != {case}:
        raise ValueError(f"FIT variants are incomplete for case {case}")
    return variants, scalers


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2) + "\n").encode("ascii")
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


def _atomic_write_csv(path: Path, frame) -> None:
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


def train_case_seed(
    index: CorpusIndex,
    *,
    case: str,
    model_seed: int,
    output_dir: Path,
    config: StudyConfig | None = None,
    arms: Sequence[ArmName] = ARMS,
    device: torch.device | str = "cpu",
) -> Path:
    """Train one paired case/seed group and persist all provenance artifacts."""
    config = StudyConfig() if config is None else config
    if index.collection_kind not in {"smoke", "development"}:
        raise ValueError("training requires a smoke or development corpus")
    if "fit" not in index.allowed_roles:
        raise ValueError("training corpus does not contain the FIT role")
    manifest = build_fault_manifest(index)
    variants, scalers = prepare_case_training_data(index, manifest, case)
    schedule = make_training_schedule(
        variants, config, case=case, model_seed=model_seed
    )
    schedule_document = schedule_payload(schedule, variants)
    provenance = training_provenance(
        index,
        manifest,
        scalers,
        config,
        schedule_document,
        device=device,
    )
    models = create_matched_models(
        config, model_seed, arms=arms, device=device
    )
    optimizers = {
        arm: torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for arm, model in models.items()
    }
    run_dir = output_dir / case / f"seed{model_seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite training run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    initial_hashes = {
        arm: tensor_state_sha256(model.state_dict())
        for arm, model in models.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise AssertionError("paired study arms have different initial tensors")
    _atomic_write_json(run_dir / "config.json", config.to_dict())
    _atomic_write_json(run_dir / "fault_manifest.json", manifest.payload())
    _atomic_write_json(run_dir / "fit_scalers.json", asdict(scalers))
    _atomic_write_json(run_dir / "training_schedule.json", schedule_document)
    _atomic_write_json(run_dir / "initial_state_hashes.json", initial_hashes)

    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    log_rows: list[dict] = []
    checkpoint_core_hashes: dict[str, str] = {}
    for scheduled in schedule:
        inputs, targets = materialize_torch_batch(
            variants, scalers, scheduled, config, device=device
        )
        for arm in arms:
            model = models[arm]
            optimizer = optimizers[arm]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(scheduled.latent_seed)
            arm_config = config.arm_config(arm)
            output = reliability_sequence_training_loss(
                model,
                inputs,
                targets,
                config.loss_config(arm),
                gate_mode=arm_config.gate_mode,
                sample=True,
            )
            if not torch.isfinite(output.loss.total):
                raise FloatingPointError(
                    f"non-finite loss for {case}, seed {model_seed}, {arm}, "
                    f"update {scheduled.update}"
                )
            output.loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            log_rows.append(
                {
                    "update": scheduled.update,
                    "arm": arm,
                    "total": float(output.loss.total.detach()),
                    "observation_nll": float(
                        output.loss.ordinary.observation_nll.detach()
                    ),
                    "health_ce": float(output.loss.ordinary.health_ce.detach()),
                    "kl": float(output.loss.ordinary.kl.detach()),
                    "latent_overshooting_kl": float(
                        output.loss.latent_overshooting_kl.detach()
                    ),
                    "direct_h8": float(
                        output.loss.direct_horizon_smooth_l1.detach()
                    ),
                    "gradient_norm": float(gradient_norm.detach()),
                }
            )
        if scheduled.update % config.checkpoint_every == 0:
            for arm in arms:
                path = (
                    run_dir
                    / "checkpoints"
                    / f"{arm}_u{scheduled.update:04d}.pt"
                )
                _atomic_torch_save(
                    path,
                    {
                        "schema": "boptest-reliability-rssm-checkpoint-v2",
                        "case": case,
                        "model_seed": model_seed,
                        "arm": arm,
                        "update": scheduled.update,
                        "config": config.to_dict(),
                        "provenance": provenance,
                        "model_state_sha256": tensor_state_sha256(
                            models[arm].state_dict()
                        ),
                        "core_state_sha256": core_tensor_state_sha256(
                            models[arm].state_dict()
                        ),
                        "model_state_dict": models[arm].state_dict(),
                        "optimizer_state_dict": optimizers[arm].state_dict(),
                    },
                )
                checkpoint_core_hashes[path.name] = core_tensor_state_sha256(
                    models[arm].state_dict()
                )

    import pandas as pd

    training_log = run_dir / "training_log.csv"
    _atomic_write_csv(training_log, pd.DataFrame(log_rows))
    checkpoint_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((run_dir / "checkpoints").glob("*.pt"))
    }
    _atomic_write_json(run_dir / "checkpoint_hashes.json", checkpoint_hashes)
    completion = run_dir / "training_complete.json"
    _atomic_write_json(
        completion,
        {
            "schema": "boptest-reliability-rssm-training-complete-v2",
            "case": case,
            "model_seed": model_seed,
            "arms": list(arms),
            "updates": config.updates,
            "wall_seconds": time.monotonic() - started,
            "device": str(device),
            "provenance": provenance,
            "initial_state_sha256": next(iter(initial_hashes.values())),
            "training_log_sha256": hashlib.sha256(
                training_log.read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": checkpoint_hashes,
            "checkpoint_core_state_sha256": checkpoint_core_hashes,
        },
    )
    return completion
