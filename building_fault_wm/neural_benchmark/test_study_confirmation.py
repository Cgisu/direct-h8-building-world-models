from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from . import study_confirmation as confirmation_module
from .locked_state import COLLECTION_COMPLETION_MARKER
from .protocol import CASES, sha256_file
from .provenance import load_strict_json
from .study_config import ARMS, StudyConfig
from .study_confirmation import (
    ATTEMPT_MARKER,
    COMPLETION_MARKER,
    CONFIRMATION_TOKEN,
    EVALUATION_RECEIPT_FIELDS,
    FAILURE_MARKER,
    FrozenAssets,
    _configure_evaluation_policy,
    _parse_args,
    _reference_path,
    _snapshot_locked_source,
    run_confirmation,
    validate_confirmation_completion,
)


def _result_row(*, case: str, arm: str, model_seed: int, update: int) -> dict:
    return {
        "case": case,
        "model_seed": model_seed,
        "arm": arm,
        "update": update,
        "trajectory_day": 4,
        "trajectory_seed": 17,
        "fault_channel": "zone_temperature_k",
        "family": "healthy",
        "sign": 0,
        "severity": 0.0,
        "onset": 64,
        "anchor": 72,
        "value": float(model_seed % 1000),
    }


def _frozen_assets(config: StudyConfig) -> FrozenAssets:
    update = config.validation_checkpoints[0]
    rssm_models = {
        (case, seed, arm): object()
        for case in CASES
        for seed in config.confirmatory_seeds
        for arm in ARMS
    }
    baseline_models: dict[str, dict[str, tuple[object, object]]] = {
        "ridge_arx": {},
        "direct_h8_ridge": {},
        "deterministic_gru": {},
    }
    for case in CASES:
        for baseline in ("ridge_arx", "direct_h8_ridge"):
            receipt = SimpleNamespace(case=case, model_seed=0)
            baseline_models[baseline][f"{case}:seed0"] = (receipt, object())
        for seed in config.confirmatory_seeds:
            receipt = SimpleNamespace(case=case, model_seed=seed)
            baseline_models["deterministic_gru"][f"{case}:seed{seed}"] = (
                receipt,
                object(),
            )
    return FrozenAssets(
        selected_update=update,
        scalers_by_case={case: object() for case in CASES},
        rssm_models=rssm_models,
        baseline_models=baseline_models,
        fault_contract={"synthetic": True},
        plan_sha256_by_case={case: "1" * 64 for case in CASES},
    )


def _install_transaction_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_materialization: bool = False,
) -> tuple[dict[str, int], dict[str, Path | str]]:
    source_root = tmp_path / "locked_source"
    manifest = source_root / "manifests" / "locked_test_all_corpus_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        confirmation_module, "CANONICAL_LOCKED_MANIFEST", manifest.resolve()
    )
    canonical_prelock_root = tmp_path / "formal_prelock"
    prelock_root = canonical_prelock_root / "bundle"
    prelock_root.mkdir(parents=True)
    registry_path = canonical_prelock_root / "prelock_registry.json"
    registry_path.write_text("{}\n", encoding="ascii")
    digest = "d" * 64
    state_dir = tmp_path / "locked_state" / digest
    state_dir.mkdir(parents=True)
    (state_dir / COLLECTION_COMPLETION_MARKER).write_text(
        '{"synthetic":true}\n', encoding="ascii"
    )
    monkeypatch.setattr(
        confirmation_module, "CANONICAL_PRELOCK_REGISTRY", registry_path.resolve()
    )
    monkeypatch.setattr(
        confirmation_module, "CANONICAL_PRELOCK_BUNDLE", prelock_root.resolve()
    )
    monkeypatch.setattr(
        confirmation_module,
        "CANONICAL_PRELOCK_ROOT",
        canonical_prelock_root.resolve(),
    )
    monkeypatch.setattr(
        confirmation_module, "STATE_ROOT", state_dir.parent.resolve()
    )
    monkeypatch.setattr(
        confirmation_module, "state_dir_for_digest", lambda value: state_dir
    )
    config = StudyConfig()
    assets = _frozen_assets(config)
    index = SimpleNamespace(manifest_sha256="a" * 64, records=tuple(range(36)))
    fault_manifest = SimpleNamespace(
        sha256="b" * 64,
        payload=lambda: {"schema": "synthetic-locked-fault-manifest"},
    )
    variants = [
        SimpleNamespace(cell=SimpleNamespace(trajectory=SimpleNamespace(case=case)))
        for case in CASES
    ]
    counters = {
        "binding_validations": 0,
        "materializations": 0,
        "rssm": 0,
        "baseline": 0,
        "gate": 0,
    }

    monkeypatch.setattr(confirmation_module, "_configure_evaluation_policy", lambda: None)
    monkeypatch.setattr(
        confirmation_module,
        "numerical_runtime_fingerprint",
        lambda *args, **kwargs: {"runtime": "fixed-cpu"},
    )
    monkeypatch.setattr(
        confirmation_module,
        "confirmation_scientific_code_manifest",
        lambda: {"multicase_fault_benchmark/study_confirmation.py": "2" * 64},
    )
    monkeypatch.setattr(
        confirmation_module,
        "validate_prelock_bundle",
        lambda *args, **kwargs: {"selected_update": assets.selected_update},
    )
    monkeypatch.setattr(
        confirmation_module,
        "validate_locked_collection_completion",
        lambda value: {
            "locked_manifest_payload_sha256": index.manifest_sha256,
            "locked_manifest_file_sha256": sha256_file(manifest),
            "external_freeze_receipt_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        confirmation_module,
        "_load_frozen_assets",
        lambda *args, **kwargs: assets,
    )
    monkeypatch.setattr(confirmation_module, "_frozen_fault_spec", lambda *args: object())
    def validate_binding(*args, **kwargs):
        counters["binding_validations"] += 1
        return index, []

    monkeypatch.setattr(
        confirmation_module, "validate_locked_corpus_binding", validate_binding
    )
    monkeypatch.setattr(
        confirmation_module,
        "_locked_dependency_paths",
        lambda *args, **kwargs: [
            "manifests/locked_test_all_corpus_manifest.json"
        ],
    )

    def snapshot(source_manifest: Path, source: Path, destination: Path):
        target = destination / "manifests" / source_manifest.name
        target.parent.mkdir(parents=True)
        target.write_bytes(source_manifest.read_bytes())
        return target, [f"manifests/{source_manifest.name}"]

    monkeypatch.setattr(confirmation_module, "_snapshot_locked_source", snapshot)
    monkeypatch.setattr(
        confirmation_module,
        "build_fault_manifest",
        lambda *args, **kwargs: fault_manifest,
    )

    def materialize(*args, **kwargs):
        counters["materializations"] += 1
        if fail_materialization:
            raise RuntimeError("synthetic post-marker failure")
        return iter(variants)

    monkeypatch.setattr(confirmation_module, "iter_role_variants", materialize)

    def evaluate_rssm(
        model,
        case_variants,
        scalers,
        study_config,
        *,
        arm,
        case,
        model_seed,
        update,
        role,
        device,
    ):
        assert role == "locked_test"
        assert device == "cpu"
        counters["rssm"] += 1
        return pd.DataFrame(
            [_result_row(case=case, arm=arm, model_seed=model_seed, update=update)]
        )

    monkeypatch.setattr(confirmation_module, "evaluate_model_h8", evaluate_rssm)

    def baseline_frame(receipt, arm: str) -> pd.DataFrame:
        counters["baseline"] += 1
        return pd.DataFrame(
            [
                _result_row(
                    case=receipt.case,
                    arm=arm,
                    model_seed=receipt.model_seed,
                    update=0,
                )
            ]
        )

    monkeypatch.setattr(
        confirmation_module,
        "evaluate_arx_h8",
        lambda model, values, scalers, receipt, **kwargs: baseline_frame(
            receipt, "ridge_arx"
        ),
    )
    monkeypatch.setattr(
        confirmation_module,
        "evaluate_direct_h8_ridge",
        lambda model, values, scalers, receipt, **kwargs: baseline_frame(
            receipt, "direct_h8_ridge"
        ),
    )
    monkeypatch.setattr(
        confirmation_module,
        "evaluate_direct_h8_gru",
        lambda model, values, scalers, receipt, **kwargs: baseline_frame(
            receipt, "deterministic_gru"
        ),
    )

    def gate(rssm, gate_config, **kwargs):
        counters["gate"] += 1
        assert kwargs["stage"] == "confirmation"
        assert kwargs["selected_update"] == assets.selected_update
        assert len(rssm) == len(CASES) * len(config.confirmatory_seeds) * len(ARMS)
        assert len(kwargs["baseline_frame"]) == len(CASES) * (
            2 + len(config.confirmatory_seeds)
        )
        return pd.DataFrame([{"paired": 1.0}]), {
            "decision": "STOP",
            "gate_pass": False,
            "paper_claim_allowed": False,
        }

    monkeypatch.setattr(confirmation_module, "evaluate_study_gate", gate)
    return counters, {
        "source_root": source_root,
        "state_dir": state_dir,
        "manifest": manifest,
        "prelock_root": prelock_root,
        "registry": registry_path,
        "digest": digest,
    }


def test_confirmation_transaction_is_complete_one_shot_and_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    counters, paths = _install_transaction_fakes(monkeypatch, tmp_path)
    output = tmp_path / "confirmation_output"
    receipt_path = run_confirmation(
        paths["manifest"],
        paths["registry"],
        paths["prelock_root"],
        paths["digest"],
        output,
        confirmation=CONFIRMATION_TOKEN,
    )

    assert receipt_path == output / "run_complete.json"
    assert output.is_dir()
    assert not (tmp_path / ".confirmation_output.confirmation-staging").exists()
    assert counters == {
        "binding_validations": 2,
        "materializations": 1,
        "rssm": 75,
        "baseline": 21,
        "gate": 1,
    }
    state_dir = paths["state_dir"]
    assert (state_dir / ATTEMPT_MARKER).is_file()
    assert (state_dir / COMPLETION_MARKER).is_file()
    assert not (state_dir / FAILURE_MARKER).exists()

    receipt = load_strict_json(receipt_path)
    assert (
        validate_confirmation_completion(output, paths["digest"])["decision"]
        == "STOP"
    )
    assert receipt["decision"] == "STOP"
    assert receipt["one_shot"] is True
    assert receipt["claim_requires_completion_marker"] is True
    assert receipt["evaluation_runtime"] == {"runtime": "fixed-cpu"}
    evaluation = load_strict_json(output / "evaluation_receipt.json")
    assert evaluation["evaluation_runtime"] == {"runtime": "fixed-cpu"}
    assert evaluation["rssm_result_artifact"]["rows"] == 75
    assert evaluation["baseline_result_artifact"]["rows"] == 21
    assert evaluation["artifact_inventory"] == [
        "confirmation_attempt.json",
        "locked_collection_completion.json",
        "manifests/locked_test_all_corpus_manifest.json",
    ]

    with pytest.raises(FileExistsError, match="already has a confirmation-attempt"):
        run_confirmation(
            paths["manifest"],
            paths["registry"],
            paths["prelock_root"],
            paths["digest"],
            tmp_path / "alternate_output",
            confirmation=CONFIRMATION_TOKEN,
        )


def test_published_output_without_completion_marker_is_not_claim_bearing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, paths = _install_transaction_fakes(monkeypatch, tmp_path)
    output = tmp_path / "uncommitted_confirmation"
    run_confirmation(
        paths["manifest"],
        paths["registry"],
        paths["prelock_root"],
        paths["digest"],
        output,
        confirmation=CONFIRMATION_TOKEN,
    )
    (paths["state_dir"] / COMPLETION_MARKER).unlink()
    with pytest.raises(ValueError, match="completion marker"):
        validate_confirmation_completion(output, paths["digest"])


def test_post_marker_failure_is_preserved_and_cannot_resume_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    counters, paths = _install_transaction_fakes(
        monkeypatch, tmp_path, fail_materialization=True
    )
    output = tmp_path / "failed_confirmation"
    with pytest.raises(RuntimeError, match="synthetic post-marker failure"):
        run_confirmation(
            paths["manifest"],
            paths["registry"],
            paths["prelock_root"],
            paths["digest"],
            output,
            confirmation=CONFIRMATION_TOKEN,
        )
    state_dir = paths["state_dir"]
    assert counters["materializations"] == 1
    assert counters["rssm"] == counters["baseline"] == counters["gate"] == 0
    assert (state_dir / ATTEMPT_MARKER).is_file()
    assert (state_dir / FAILURE_MARKER).is_file()
    assert not (state_dir / COMPLETION_MARKER).exists()
    assert not output.exists()
    assert (tmp_path / ".failed_confirmation.confirmation-staging").is_dir()

    with pytest.raises(FileExistsError, match="already has a confirmation-attempt"):
        run_confirmation(
            paths["manifest"],
            paths["registry"],
            paths["prelock_root"],
            paths["digest"],
            tmp_path / "retry_elsewhere",
            confirmation=CONFIRMATION_TOKEN,
        )


def test_prelock_failure_does_not_consume_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, paths = _install_transaction_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        confirmation_module,
        "validate_prelock_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad pre-lock")),
    )
    output = tmp_path / "prelock_failure"
    with pytest.raises(ValueError, match="bad pre-lock"):
        run_confirmation(
            paths["manifest"],
            paths["registry"],
            paths["prelock_root"],
            paths["digest"],
            output,
            confirmation=CONFIRMATION_TOKEN,
        )
    state_dir = paths["state_dir"]
    assert not (state_dir / ATTEMPT_MARKER).exists()
    assert not (state_dir / FAILURE_MARKER).exists()
    assert not output.exists()
    assert not (tmp_path / ".prelock_failure.confirmation-staging").exists()


def test_confirmation_output_cannot_enter_reserved_state_or_prelock_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, paths = _install_transaction_fakes(monkeypatch, tmp_path)
    forbidden = (
        paths["state_dir"] / "poisoned-output",
        paths["prelock_root"].parent / "poisoned-output",
    )
    for output in forbidden:
        with pytest.raises(ValueError, match="locked-state roots"):
            run_confirmation(
                paths["manifest"],
                paths["registry"],
                paths["prelock_root"],
                paths["digest"],
                output,
                confirmation=CONFIRMATION_TOKEN,
            )
    assert not (paths["state_dir"] / ATTEMPT_MARKER).exists()


def test_locked_binding_failure_consumes_attempt_and_prevents_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, paths = _install_transaction_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        confirmation_module,
        "validate_locked_corpus_binding",
        lambda *args, **kwargs: (None, ["synthetic invalid locked binding"]),
    )
    output = tmp_path / "locked_binding_failure"
    with pytest.raises(ValueError, match="synthetic invalid locked binding"):
        run_confirmation(
            paths["manifest"],
            paths["registry"],
            paths["prelock_root"],
            paths["digest"],
            output,
            confirmation=CONFIRMATION_TOKEN,
        )
    state_dir = paths["state_dir"]
    assert (state_dir / ATTEMPT_MARKER).is_file()
    assert (state_dir / FAILURE_MARKER).is_file()
    assert not output.exists()
    assert (tmp_path / ".locked_binding_failure.confirmation-staging").is_dir()


def test_cli_has_no_device_seed_update_arm_or_resume_override():
    required = [
        "--locked-manifest",
        "locked.json",
        "--prelock-registry",
        "prelock.json",
        "--prelock-artifact-root",
        "bundle",
        "--expected-prelock-sha256",
        "d" * 64,
        "--output",
        "result",
        "--confirm",
        CONFIRMATION_TOKEN,
    ]
    parsed = _parse_args(required)
    assert not any(
        hasattr(parsed, name)
        for name in ("device", "seed", "selected_update", "arm", "resume")
    )
    with pytest.raises(SystemExit):
        _parse_args([*required, "--device", "cuda"])


def test_reference_loader_rejects_tampering_and_symlinks(tmp_path: Path):
    root = tmp_path / "root"
    path = root / "models" / "model.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen-model")
    reference = {
        "path": "models/model.pt",
        "sha256": sha256_file(path),
        "kind": "rssm_checkpoint",
        "identity": "case:seed1:gated_h8:u0100",
    }
    assert (
        _reference_path(
            root,
            reference,
            expected_kind="rssm_checkpoint",
            expected_identity="case:seed1:gated_h8:u0100",
        )
        == path
    )
    path.write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="frozen SHA-256"):
        _reference_path(
            root,
            reference,
            expected_kind="rssm_checkpoint",
            expected_identity="case:seed1:gated_h8:u0100",
        )

    target = tmp_path / "target.pt"
    target.write_bytes(b"target")
    symlink = root / "linked.pt"
    symlink.symlink_to(target)
    symlink_reference = {
        "path": "linked.pt",
        "sha256": sha256_file(target),
        "kind": "rssm_checkpoint",
        "identity": "linked",
    }
    with pytest.raises(ValueError, match="symbolic link"):
        _reference_path(
            root,
            symlink_reference,
            expected_kind="rssm_checkpoint",
            expected_identity="linked",
        )


def test_locked_snapshot_owns_independent_bytes(tmp_path: Path):
    source_root = tmp_path / "locked_source"
    manifest_path = (
        source_root / "manifests" / "locked_test_all_corpus_manifest.json"
    )
    plan_path = source_root / "plans" / "full" / "case.json"
    csv_path = source_root / "locked_test_raw" / "case" / "day.csv"
    receipt_path = source_root / "locked_test_raw" / "_receipts" / "case.json"
    for path, content in (
        (plan_path, "plan\n"),
        (csv_path, "value\n1\n"),
        (receipt_path, "receipt\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii")
    wrapper = {
        "manifest_sha256": "a" * 64,
        "manifest": {
            "collection_kind": "locked_test",
            "plans": {"case": {"path": "plans/full/case.json"}},
            "files": [{"path": "case/day.csv"}],
            "receipts": {"case": {"path": "_receipts/case.json"}},
        },
    }
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(wrapper) + "\n", encoding="ascii")

    snapshot_root = tmp_path / "snapshot"
    snapshot_manifest, inventory = _snapshot_locked_source(
        manifest_path, source_root, snapshot_root
    )
    copied_csv = snapshot_root / "locked_test_raw" / "case" / "day.csv"
    assert snapshot_manifest.is_file()
    assert len(inventory) == 4
    assert csv_path.stat().st_ino != copied_csv.stat().st_ino
    csv_path.write_text("value\n999\n", encoding="ascii")
    assert copied_csv.read_text(encoding="ascii") == "value\n1\n"


def test_wrong_confirmation_token_is_rejected_before_source_access(tmp_path: Path):
    with pytest.raises(ValueError, match="requires --confirm"):
        run_confirmation(
            tmp_path / "missing_locked_manifest.json",
            tmp_path / "missing_registry.json",
            tmp_path / "missing_bundle",
            "d" * 64,
            tmp_path / "output",
            confirmation="WRONG",
        )


def test_evaluation_policy_rejects_unconstrained_native_pools(monkeypatch):
    monkeypatch.setattr(
        confirmation_module,
        "threadpool_info",
        lambda: [{"internal_api": "openblas", "num_threads": 24}],
    )
    with pytest.raises(RuntimeError, match="native thread pools"):
        _configure_evaluation_policy()


def test_evaluation_receipt_contract_is_exact():
    assert "confirmation_runner_sha256" in EVALUATION_RECEIPT_FIELDS
    assert "evaluation_runtime" in EVALUATION_RECEIPT_FIELDS
    assert "locked_fault_manifest_sha256" in EVALUATION_RECEIPT_FIELDS
    assert "attempt_marker_sha256" in EVALUATION_RECEIPT_FIELDS
