from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from .fault_data import FaultScalers, ScaleStats
from .protocol import CASES, sha256_file
from .study_config import ARMS, StudyConfig
from .study_train import (
    core_tensor_state_sha256,
    create_matched_models,
    tensor_state_sha256,
)
from .test_study_train import make_variants, unit_scalers
from . import study_development as runner


def _fake_index(*, collection_kind: str = "development") -> SimpleNamespace:
    roles = ("fit", "validation") if collection_kind == "development" else ("locked_test",)
    records = []
    for case in CASES:
        for role, count in (("fit", 20), ("validation", 8)):
            if role not in roles:
                continue
            records.extend(
                SimpleNamespace(key=SimpleNamespace(case=case, role=role))
                for _ in range(count)
            )
    return SimpleNamespace(
        collection_kind=collection_kind,
        allowed_roles=roles,
        prelock_registry_sha256="f" * 64 if collection_kind == "locked_test" else None,
        records=tuple(records),
        manifest_sha256="a" * 64,
    )


def _scalers(case: str) -> FaultScalers:
    return FaultScalers(
        observation=ScaleStats((0.0,) * 4, (1.0,) * 4),
        action=ScaleStats((0.0,), (1.0,)),
        context=ScaleStats((0.0,) * 5, (1.0,) * 5),
        fit_source_sha256=((f"{case}:fit:day004:seed1", "b" * 64),),
    )


def _result_row(
    *, case: str, seed: int, arm: str, update: int, sequence: int = 0
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": case,
                "role": "validation",
                "trajectory_day": 1,
                "trajectory_seed": 10,
                "model_seed": seed,
                "arm": arm,
                "update": update,
                "cell_id": f"{case}:{arm}:{seed}:{sequence}",
                "fault_channel": "zone_temperature_k",
                "family": "healthy",
                "sign": 0,
                "severity": 0.0,
                "severity_unit": "none",
                "onset": 48,
                "anchor": 56,
                "horizon": 8,
                "target_raw": 1.0,
                "prediction_raw": 1.1,
                "standardized_abs_error": 0.1,
            }
        ]
    )


def test_development_index_rejects_locked_or_incomplete_corpus():
    with pytest.raises(ValueError, match="development corpus"):
        runner._validate_development_index(_fake_index(collection_kind="locked_test"))

    index = _fake_index()
    index.records = index.records[:-1]
    with pytest.raises(ValueError, match="20/8 split"):
        runner._validate_development_index(index)


def test_scientific_code_manifest_is_complete_and_content_addressed():
    manifest = runner.scientific_code_manifest()
    expected = {
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
    }
    assert set(manifest) == expected
    assert all(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
        for value in manifest.values()
    )


def test_production_refuses_custom_config_and_existing_output(tmp_path: Path):
    custom = runner.integration_study_config()
    with pytest.raises(ValueError, match="only with integration_only"):
        runner.run_development(
            tmp_path / "unused.json",
            tmp_path / "new",
            integration_config=custom,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_development(tmp_path / "unused.json", existing)


def test_staging_resume_is_exact_and_clears_only_recomputable_artifacts(
    tmp_path: Path,
):
    output = tmp_path / "result"
    runtime = runner.numerical_runtime_fingerprint("cpu", include_sklearn=True)
    identity = {
        "schema": "test",
        "binding": "same",
        "numerical_runtime": runtime,
    }
    staging = runner._prepare_staging(
        output_dir=output, identity=identity, resume=False
    )
    training_marker = staging / "training" / "keep.txt"
    training_marker.parent.mkdir()
    training_marker.write_text("expensive\n", encoding="ascii")
    (staging / "rssm_validation_h8.csv").write_text("stale\n", encoding="ascii")

    with pytest.raises(FileExistsError, match="--resume"):
        runner._prepare_staging(
            output_dir=output, identity=identity, resume=False
        )
    with pytest.raises(ValueError, match="identity differs"):
        runner._prepare_staging(
            output_dir=output,
            identity={
                "schema": "test",
                "binding": "different",
                "numerical_runtime": runtime,
            },
            resume=True,
        )

    resumed = runner._prepare_staging(
        output_dir=output, identity=identity, resume=True
    )
    assert resumed == staging
    assert training_marker.read_text(encoding="ascii") == "expensive\n"
    assert not (staging / "rssm_validation_h8.csv").exists()


def test_partial_training_unit_fails_closed(tmp_path: Path):
    run_dir = tmp_path / "training" / next(iter(CASES)) / "seed202608011"
    run_dir.mkdir(parents=True)
    (run_dir / "training_log.csv").write_text("partial\n", encoding="ascii")
    with pytest.raises(ValueError, match="inventory is incomplete or ambiguous"):
        runner._validate_training_unit(
            run_dir,
            index=_fake_index(),
            fault_manifest=SimpleNamespace(),
            variants=[],
            scalers=_scalers(next(iter(CASES))),
            config=runner.integration_study_config(),
            case=next(iter(CASES)),
            seed=runner.integration_study_config().development_seeds[0],
            device=runner.torch.device("cpu"),
        )


def test_complete_training_unit_is_semantically_reusable_and_tamper_evident(
    tmp_path: Path,
):
    config = runner.integration_study_config()
    seed = config.development_seeds[0]
    case = "case"
    variants = make_variants()
    scalers = unit_scalers()
    index = SimpleNamespace(manifest_sha256="a" * 64)

    class FakeManifest:
        sha256 = "b" * 64

        @staticmethod
        def payload():
            return {"schema": "test-fault-manifest"}

    manifest = FakeManifest()
    schedule = runner.make_training_schedule(
        variants, config, case=case, model_seed=seed
    )
    schedule_document = runner.schedule_payload(schedule, variants)
    provenance = runner.training_provenance(
        index,
        manifest,
        scalers,
        config,
        schedule_document,
        device="cpu",
    )
    models = create_matched_models(config, seed)
    run_dir = tmp_path / "training" / case / f"seed{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    for filename, payload in {
        "config.json": config.to_dict(),
        "fault_manifest.json": manifest.payload(),
        "fit_scalers.json": asdict(scalers),
        "training_schedule.json": schedule_document,
    }.items():
        (run_dir / filename).write_text(
            json.dumps(payload) + "\n", encoding="ascii"
        )
    initial_hashes = {
        arm: tensor_state_sha256(model.state_dict()) for arm, model in models.items()
    }
    (run_dir / "initial_state_hashes.json").write_text(
        json.dumps(initial_hashes) + "\n", encoding="ascii"
    )
    log = pd.DataFrame(
        [
            {
                "update": 1,
                "arm": arm,
                "total": 1.0,
                "observation_nll": 1.0,
                "health_ce": 0.0,
                "kl": 0.0,
                "latent_overshooting_kl": 0.0,
                "direct_h8": 0.0,
                "gradient_norm": 1.0,
            }
            for arm in ARMS
        ]
    )
    training_log = run_dir / "training_log.csv"
    log.to_csv(training_log, index=False)

    checkpoint_hashes = {}
    core_hashes = {}
    for arm, model in models.items():
        name = f"{arm}_u0001.pt"
        path = checkpoint_dir / name
        state = model.state_dict()
        state_hash = tensor_state_sha256(state)
        core_hash = core_tensor_state_sha256(state)
        torch.save(
            {
                "schema": "boptest-reliability-rssm-checkpoint-v2",
                "case": case,
                "model_seed": seed,
                "arm": arm,
                "update": 1,
                "config": config.to_dict(),
                "provenance": provenance,
                "model_state_sha256": state_hash,
                "core_state_sha256": core_hash,
                "model_state_dict": state,
                "optimizer_state_dict": {},
            },
            path,
        )
        checkpoint_hashes[name] = sha256_file(path)
        core_hashes[name] = core_hash
    (run_dir / "checkpoint_hashes.json").write_text(
        json.dumps(checkpoint_hashes) + "\n", encoding="ascii"
    )
    completion = {
        "schema": "boptest-reliability-rssm-training-complete-v2",
        "case": case,
        "model_seed": seed,
        "arms": list(ARMS),
        "updates": 1,
        "wall_seconds": 1.0,
        "device": "cpu",
        "provenance": provenance,
        "initial_state_sha256": next(iter(initial_hashes.values())),
        "training_log_sha256": sha256_file(training_log),
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_core_state_sha256": core_hashes,
    }
    completion_path = run_dir / "training_complete.json"
    completion_path.write_text(json.dumps(completion) + "\n", encoding="ascii")

    validated_path, validated = runner._validate_training_unit(
        run_dir,
        index=index,
        fault_manifest=manifest,
        variants=variants,
        scalers=scalers,
        config=config,
        case=case,
        seed=seed,
        device=torch.device("cpu"),
    )
    assert validated_path == completion_path
    assert validated == completion

    training_log.write_text(training_log.read_text(encoding="ascii") + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="training log hash differs"):
        runner._validate_training_unit(
            run_dir,
            index=index,
            fault_manifest=manifest,
            variants=variants,
            scalers=scalers,
            config=config,
            case=case,
            seed=seed,
            device=torch.device("cpu"),
        )


def test_integration_runner_exercises_full_flow_without_scientific_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = runner.integration_study_config()
    index = _fake_index()
    calls: dict[str, list] = {
        "roles": [],
        "train": [],
        "checkpoint_eval": [],
        "bundle_write": [],
        "bundle_load": [],
        "gate": [],
    }
    monkeypatch.setattr(runner, "load_corpus_index", lambda path: index)
    monkeypatch.setattr(
        runner,
        "build_fault_manifest",
        lambda index: SimpleNamespace(sha256="c" * 64),
    )

    def fake_frozen_contract(path, index, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema":"test"}\n', encoding="ascii")

    monkeypatch.setattr(runner, "write_frozen_fault_contract", fake_frozen_contract)

    def fake_prepare(index, manifest, case):
        calls["roles"].append((case, "fit"))
        return [(case, "fit")], _scalers(case)

    def fake_iter(index, manifest, role, *, cases):
        calls["roles"].append((cases[0], role))
        return iter([(cases[0], role)])

    monkeypatch.setattr(runner, "prepare_case_training_data", fake_prepare)
    monkeypatch.setattr(runner, "iter_role_variants", fake_iter)

    def write_fake_training_unit(output_dir, case, model_seed, config, device):
        run_dir = output_dir / case / f"seed{model_seed}"
        run_dir.mkdir(parents=True)
        checkpoint_hashes = {
            f"{arm}_u{update:04d}.pt": "d" * 64
            for arm in ARMS
            for update in config.validation_checkpoints
        }
        payload = {
            "schema": "boptest-reliability-rssm-training-complete-v2",
            "case": case,
            "model_seed": model_seed,
            "arms": list(ARMS),
            "updates": config.updates,
            "wall_seconds": 0.01,
            "device": str(device),
            "training_log_sha256": "e" * 64,
            "checkpoint_sha256": checkpoint_hashes,
            "provenance": {"test": True},
        }
        path = run_dir / "training_complete.json"
        path.write_text(json.dumps(payload) + "\n", encoding="ascii")
        return path

    def fake_train(
        index,
        *,
        case,
        model_seed,
        output_dir,
        config,
        arms,
        device,
    ):
        calls["train"].append((case, model_seed, tuple(arms), str(device)))
        return write_fake_training_unit(
            output_dir, case, model_seed, config, device
        )

    monkeypatch.setattr(runner, "train_case_seed", fake_train)

    def fake_validate_training_unit(
        run_dir,
        *,
        index,
        fault_manifest,
        variants,
        scalers,
        config,
        case,
        seed,
        device,
    ):
        path = run_dir / "training_complete.json"
        payload = json.loads(path.read_text(encoding="ascii"))
        assert payload["case"] == case
        assert payload["model_seed"] == seed
        assert payload["device"] == str(device)
        return path, payload

    monkeypatch.setattr(runner, "_validate_training_unit", fake_validate_training_unit)

    def fake_checkpoint_eval(**kwargs):
        calls["checkpoint_eval"].append(
            (kwargs["case"], kwargs["seed"], kwargs["arm"], kwargs["update"])
        )
        return _result_row(
            case=kwargs["case"],
            seed=kwargs["seed"],
            arm=kwargs["arm"],
            update=kwargs["update"],
        )

    monkeypatch.setattr(runner, "_evaluate_checkpoint", fake_checkpoint_eval)

    def fake_selection_scores(frame, config):
        assert set(frame["arm"]) == {"ungated_h8", "gated_h8"}
        assert set(frame["update"]) == set(config.validation_checkpoints)
        return pd.DataFrame(
            {
                "update": list(config.validation_checkpoints),
                "common_score": [1.0] * len(config.validation_checkpoints),
                "selected": [True] + [False] * (len(config.validation_checkpoints) - 1),
            }
        )

    def fake_write_selection(path, frame, config, *, validation_rows_path):
        validation_rows_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(validation_rows_path, index=False)
        payload = {"selected_update": config.validation_checkpoints[0]}
        path.write_text(json.dumps(payload) + "\n", encoding="ascii")
        return payload

    monkeypatch.setattr(runner, "validation_selection_scores", fake_selection_scores)
    monkeypatch.setattr(runner, "write_validation_selection", fake_write_selection)

    @dataclass
    class FakeReceipt:
        baseline: str
        case: str
        model_seed: int

        @property
        def payload(self):
            return {
                "baseline": self.baseline,
                "case": self.case,
                "model_seed": self.model_seed,
            }

    def ridge_fit(name):
        def fit(fit_variants, validation_variants, scalers):
            case = fit_variants[0][0]
            table = pd.DataFrame(
                {"alpha": [0.1], "validation_score": [1.0], "selected": [True]}
            )
            return object(), table, FakeReceipt(name, case, 0)

        return fit

    def fake_gru_fit(
        fit_variants,
        validation_variants,
        scalers,
        config,
        *,
        model_seed,
        device,
    ):
        case = fit_variants[0][0]
        table = pd.DataFrame(
            {"update": [1], "validation_score": [1.0], "selected": [True]}
        )
        log = pd.DataFrame(
            {"update": [1], "fit_smooth_l1": [1.0], "gradient_norm": [1.0]}
        )
        return SimpleNamespace(
            model=object(),
            score_table=table,
            training_log=log,
            receipt=FakeReceipt("deterministic_gru", case, model_seed),
        )

    monkeypatch.setattr(runner, "fit_arx_ridge", ridge_fit("ridge_arx"))
    monkeypatch.setattr(
        runner, "fit_direct_h8_ridge", ridge_fit("direct_h8_ridge")
    )
    monkeypatch.setattr(runner, "fit_direct_h8_gru", fake_gru_fit)

    stored_entries = {}

    def fake_write_bundle(path, *, baseline, entries):
        calls["bundle_write"].append((baseline, tuple(sorted(entries))))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(baseline.encode("ascii"))
        stored_entries[baseline] = dict(entries)

    def fake_load_bundle(path, *, baseline, config):
        calls["bundle_load"].append(baseline)
        return {
            identity: (receipt, model)
            for identity, (receipt, model, table) in stored_entries[baseline].items()
        }

    monkeypatch.setattr(runner, "write_baseline_selection_bundle", fake_write_bundle)
    monkeypatch.setattr(runner, "load_selected_baseline_models", fake_load_bundle)

    def baseline_frame(arm, receipt):
        return _result_row(
            case=receipt.case,
            seed=receipt.model_seed,
            arm=arm,
            update=0,
        )

    monkeypatch.setattr(
        runner,
        "evaluate_arx_h8",
        lambda model, variants, scalers, receipt, *, role: baseline_frame(
            "ridge_arx", receipt
        ),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_direct_h8_ridge",
        lambda model, variants, scalers, receipt, *, role: baseline_frame(
            "direct_h8_ridge", receipt
        ),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_direct_h8_gru",
        lambda model, variants, scalers, receipt, *, role, device: baseline_frame(
            "deterministic_gru", receipt
        ),
    )

    def fake_gate(rssm, config, *, stage, selected_update, baseline_frame):
        calls["gate"].append(
            (stage, selected_update, set(rssm["arm"]), set(baseline_frame["arm"]))
        )
        return pd.DataFrame({"paired": [1]}), {
            "decision": "SCREEN_GO",
            "paper_claim_allowed": False,
            "gate_pass": False,
            "confirmatory_conditions_evaluable": True,
            "development_screen": {
                "paper_claim_allowed": False,
                "evaluable": True,
                "screen_pass": True,
                "checks": {"test": True},
            },
            "checks": {"confirmatory_test": True},
        }

    monkeypatch.setattr(runner, "evaluate_study_gate", fake_gate)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"sealed":"test"}\n', encoding="ascii")
    output = tmp_path / "development"
    identity = runner._staging_identity(
        output_dir=output,
        manifest_path=manifest_path,
        index=index,
        config=config,
        device=runner.torch.device("cpu"),
        integration_only=True,
    )
    staging = runner._prepare_staging(
        output_dir=output, identity=identity, resume=False
    )
    first_case = next(iter(CASES))
    first_seed = config.development_seeds[0]
    write_fake_training_unit(
        staging / "training", first_case, first_seed, config, "cpu"
    )
    (staging / "rssm_validation_h8.csv").write_text("stale\n", encoding="ascii")
    receipt_path = runner.run_development(
        manifest_path,
        output,
        integration_only=True,
        integration_config=config,
        resume=True,
    )
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    gate = json.loads((output / "gate" / "study_gate.json").read_text(encoding="ascii"))

    assert receipt["decision"] == "INTEGRATION_ONLY"
    assert receipt["scientific_screen_reported"] is False
    assert gate["decision"] == "INTEGRATION_ONLY"
    assert gate["development_screen"]["screen_pass"] is None
    assert gate["development_screen"]["checks"] == {"test": None}
    assert gate["checks"] == {"confirmatory_test": None}
    assert len(calls["train"]) == len(CASES) * len(config.development_seeds) - 1
    assert len(calls["checkpoint_eval"]) == (
        len(CASES) * len(config.development_seeds) * (2 + 3)
    )
    assert all(role != "locked_test" for _, role in calls["roles"])
    assert {name for name, _ in calls["bundle_write"]} == {
        "ridge_arx",
        "direct_h8_ridge",
        "deterministic_gru",
    }
    assert set(calls["bundle_load"]) == {
        "ridge_arx",
        "direct_h8_ridge",
        "deterministic_gru",
    }
    assert calls["gate"] == [
        (
            "development",
            1,
            set(ARMS),
            {"ridge_arx", "direct_h8_ridge", "deterministic_gru"},
        )
    ]
    assert (output / "rssm_validation_h8.csv").is_file()
    assert (output / "baseline_validation_h8.csv").is_file()
    assert (output / "wall_time.json").is_file()
    assert (output / runner.STAGING_MARKER).is_file()
    assert not staging.exists()
    run_config = json.loads((output / "run_config.json").read_text(encoding="ascii"))
    assert set(run_config["scientific_code_sha256_by_path"]) == set(
        runner.SCIENTIFIC_CODE_FILES
    )
    assert (
        receipt["scientific_code_manifest_sha256"]
        == run_config["scientific_code_manifest_sha256"]
    )
    assert (
        receipt["scientific_code_sha256_by_path"]
        == run_config["scientific_code_sha256_by_path"]
    )
    assert receipt["numerical_runtime"] == run_config["numerical_runtime"]
    assert receipt["numerical_runtime"]["sklearn_version"] is not None
    assert receipt["wall_time"]["training_units"]["resumed"] == [
        f"{first_case}:seed{first_seed}"
    ]
    assert receipt["artifact_inventory_excludes_this_receipt"]
