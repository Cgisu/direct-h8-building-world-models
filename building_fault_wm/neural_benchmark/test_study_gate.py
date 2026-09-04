from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from .protocol import CASES, TRAJECTORY_STEPS
from .study_confirmation import (
    ATTEMPT_EVIDENCE_PATH,
    ATTEMPT_SCHEMA,
    COLLECTION_EVIDENCE_PATH,
    EVALUATION_POLICY,
    EVALUATION_RECEIPT_FIELDS,
    RUNNER_RELATIVE_PATH,
    confirmation_scientific_code_manifest,
)
from .study_locked_collection import (
    CANONICAL_LOCKED_MANIFEST,
    CANONICAL_LOCKED_RAW,
    COMPLETION_SCHEMA as COLLECTION_COMPLETION_SCHEMA,
)
from .study_config import ARMS, StudyConfig
from .study_train import core_tensor_state_sha256, tensor_state_sha256
from .runtime_provenance import numerical_runtime_fingerprint
from .study_gate import (
    COMPETENCE_BASELINE_ARMS,
    EVALUATION_PROVENANCE_SCHEMA,
    PRELOCK_PROVENANCE_SCHEMA,
    _config_sha256,
    _canonical_json,
    _equal_weight_scores,
    _expected_fault_grid,
    _required_checkpoint_keys,
    _required_schedule_keys,
    _sha256_file,
    canonical_frame_sha256,
    evaluate_study_gate,
    matched_arm_metrics,
    validate_confirmation_provenance,
)
from . import study_gate as study_gate_module


def _metric_value(
    arm: str,
    family: str,
    *,
    case_index: int,
    trajectory_index: int,
    seed_index: int,
) -> float:
    offset = 0.01 * (case_index + trajectory_index + seed_index)
    if family == "healthy":
        ungated = 2.0 + offset
        values = {
            "legacy": 2.2 + offset,
            "ungated_h8": ungated,
            "aux_h8": ungated,
            "gated_h8": 1.02 * ungated,
            "huber_h8": 2.01 + offset,
        }
    else:
        ungated = 10.0 + offset
        gated = 0.8 * ungated
        values = {
            "legacy": 10.5 + offset,
            "ungated_h8": ungated,
            "aux_h8": ungated,
            "gated_h8": gated,
            "huber_h8": 0.99 * gated,
        }
    return float(values[arm])


def _frames(stage: str) -> tuple[pd.DataFrame, pd.DataFrame, StudyConfig]:
    config = StudyConfig()
    role = "validation" if stage == "development" else "locked_test"
    seeds = config.development_seeds if stage == "development" else config.confirmatory_seeds
    trajectory_count = 8 if stage == "development" else 12
    rssm_rows: list[dict] = []
    baseline_rows: list[dict] = []
    grid = sorted(_expected_fault_grid(role))
    for case_index, case in enumerate(sorted(CASES)):
        for trajectory_index in range(trajectory_count):
            day = 10 + trajectory_index
            trajectory_seed = 10_000 * (case_index + 1) + trajectory_index
            for (
                channel,
                family,
                sign,
                severity,
                severity_unit,
                onset,
                anchor,
                horizon,
            ) in grid:
                cell_id = (
                    f"{case}:{role}:{day}:{trajectory_seed}:{channel}:{family}:"
                    f"{sign}:{severity}:{onset}"
                )
                common = {
                    "case": case,
                    "role": role,
                    "trajectory_day": day,
                    "trajectory_seed": trajectory_seed,
                    "cell_id": cell_id,
                    "fault_channel": channel,
                    "family": family,
                    "sign": sign,
                    "severity": severity,
                    "severity_unit": severity_unit,
                    "onset": onset,
                    "anchor": anchor,
                    "horizon": horizon,
                }
                for seed_index, model_seed in enumerate(seeds):
                    for arm in ARMS:
                        error = _metric_value(
                            arm,
                            family,
                            case_index=case_index,
                            trajectory_index=trajectory_index,
                            seed_index=seed_index,
                        )
                        rssm_rows.append(
                            {
                                **common,
                                "model_seed": model_seed,
                                "arm": arm,
                                "update": 100,
                                "target_raw": 0.0,
                                "prediction_raw": error,
                                "standardized_abs_error": error,
                                "alternate_action_prediction_raw": error + 1.0,
                                "alternate_action_standardized_abs_error": error
                                + 1.0,
                                "action_prediction_change_standardized": 1.0,
                                "persistence_prediction_raw": 12.0,
                                "persistence_standardized_abs_error": 12.0,
                            }
                        )
                for baseline_index, arm in enumerate(COMPETENCE_BASELINE_ARMS):
                    baseline_seeds = seeds if arm == "deterministic_gru" else (0,)
                    for baseline_seed in baseline_seeds:
                        baseline_rows.append(
                            {
                                **common,
                                "model_seed": baseline_seed,
                                "arm": arm,
                                "target_raw": 0.0,
                                "prediction_raw": 9.0 - 0.1 * baseline_index,
                                "standardized_abs_error": 9.0
                                - 0.1 * baseline_index,
                            }
                        )
    return pd.DataFrame(rssm_rows), pd.DataFrame(baseline_rows), config


def _confirmation_provenance(
    rssm: pd.DataFrame,
    baselines: pd.DataFrame,
    config: StudyConfig,
    artifact_root: Path,
    *,
    selected_update: int = 100,
) -> tuple[dict, Path, str, dict, Path]:
    here = Path(study_gate_module.__file__).resolve().parent
    prelock_root = artifact_root / "prelock"
    evaluation_root = artifact_root / "evaluation"
    prelock_root.mkdir(parents=True)
    evaluation_root.mkdir(parents=True)
    prelock_inventory: list[str] = []
    counter = 0

    def prelock_artifact(
        label: str,
        kind: str,
        identity: str,
        content: str | None = None,
        torch_payload: dict | None = None,
    ) -> dict:
        nonlocal counter
        counter += 1
        relative = f"artifact_{counter:03d}.txt"
        path = prelock_root / relative
        if torch_payload is not None:
            torch.save(torch_payload, path)
        else:
            path.write_text(
                f"{counter}:{label}\n" if content is None else content,
                encoding="ascii",
            )
        prelock_inventory.append(relative)
        return {
            "path": relative,
            "sha256": _sha256_file(path),
            "kind": kind,
            "identity": identity,
        }

    def checkpoint_artifact(identity: str) -> dict:
        case, seed_text, arm, update_text = identity.split(":")
        seed = int(seed_text.removeprefix("seed"))
        update = int(update_text.removeprefix("u"))
        core_identity = (
            f"{case}:seed{seed}:shared_core"
            if arm in ("ungated_h8", "aux_h8")
            else f"{case}:seed{seed}:{arm}:core"
        )
        state_dict = {
            "backbone.synthetic": torch.tensor(
                list(hashlib.sha256(core_identity.encode("ascii")).digest()),
                dtype=torch.uint8,
            ),
            "health_head.synthetic": torch.tensor(
                list(hashlib.sha256(identity.encode("ascii")).digest()),
                dtype=torch.uint8,
            ),
        }
        return prelock_artifact(
            f"checkpoint:{identity}",
            "rssm_checkpoint",
            identity,
            torch_payload={
                "schema": "boptest-reliability-rssm-checkpoint-v2",
                "case": case,
                "model_seed": seed,
                "arm": arm,
                "update": update,
                "config": config.to_dict(),
                "model_state_sha256": tensor_state_sha256(state_dict),
                "core_state_sha256": core_tensor_state_sha256(state_dict),
                "model_state_dict": state_dict,
            },
        )

    prelock = {
        "schema": PRELOCK_PROVENANCE_SCHEMA,
        "stage": "prelock",
        "selected_update": selected_update,
        "protocol_sha256": _sha256_file(here / "STUDY_PROTOCOL.md"),
        "study_config_sha256": _config_sha256(config),
        "evaluator_sha256": _sha256_file(here / "study_evaluate.py"),
        "gate_sha256": _sha256_file(Path(study_gate_module.__file__).resolve()),
        "trainer_sha256": _sha256_file(here / "study_train.py"),
        "reliability_model_sha256": _sha256_file(here / "reliability_model.py"),
        "rssm_backbone_sha256": _sha256_file(here.parent / "health_rssm" / "model.py"),
        "reliability_loss_sha256": _sha256_file(here / "reliability_loss.py"),
        "fault_data_sha256": _sha256_file(here / "fault_data.py"),
        "baselines_sha256": _sha256_file(here / "baselines.py"),
        "provenance_sha256": _sha256_file(here / "provenance.py"),
        "confirmation_runner_sha256": _sha256_file(here / "study_confirmation.py"),
        "development_run_config_artifact": prelock_artifact(
            "development run config", "development_run_config", "development"
        ),
        "development_run_complete_artifact": prelock_artifact(
            "development run receipt", "development_run_receipt", "development"
        ),
        "development_gru_baseline_artifact": prelock_artifact(
            "development GRU baseline",
            "development_baseline_validation_selection",
            "deterministic_gru",
        ),
        "corpus_manifest_artifact": prelock_artifact(
            "corpus manifest", "development_corpus_manifest", "development"
        ),
        "fault_manifest_artifact": prelock_artifact(
            "fault manifest", "frozen_fault_manifest", "all_roles"
        ),
        "validation_selection_artifact": prelock_artifact(
            "validation selection",
            "validation_checkpoint_selection",
            f"update{selected_update:04d}",
        ),
        "fit_scaler_artifact_by_case": {
            case: prelock_artifact(
                f"scaler:{case}",
                "fit_scaler",
                case,
                json.dumps(
                    {
                        "observation": {"mean": [0.0] * 4, "scale": [1.0] * 4},
                        "action": {"mean": [0.0], "scale": [1.0]},
                        "context": {"mean": [0.0] * 5, "scale": [1.0] * 5},
                        "fit_source_sha256": [
                            [f"{case}:fit:synthetic", hashlib.sha256(case.encode()).hexdigest()]
                        ],
                    }
                )
                + "\n",
            )
            for case in CASES
        },
        "checkpoint_artifact_by_identity": {
            key: checkpoint_artifact(key)
            for key in _required_checkpoint_keys(config, selected_update)
        },
        "training_schedule_artifact_by_case_seed": {
            key: prelock_artifact(f"schedule:{key}", "training_schedule", key)
            for key in _required_schedule_keys(config)
        },
        "baseline_selection_artifact_by_arm": {
            arm: prelock_artifact(
                f"baseline:{arm}", "baseline_validation_selection", arm
            )
            for arm in COMPETENCE_BASELINE_ARMS
        },
    }
    prelock["artifact_inventory"] = prelock_inventory
    prelock_digest = hashlib.sha256(
        _canonical_json(prelock).encode("ascii")
    ).hexdigest()
    evaluation_source = evaluation_root / "locked_source_manifest.txt"
    evaluation_source.write_text("locked evaluation source manifest\n", encoding="ascii")
    code_manifest = confirmation_scientific_code_manifest()
    rssm_frame_sha256 = canonical_frame_sha256(rssm)
    baseline_frame_sha256 = canonical_frame_sha256(baselines)
    evaluation_runtime = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    collection_path = evaluation_root / COLLECTION_EVIDENCE_PATH
    collection_payload = {
        "schema": COLLECTION_COMPLETION_SCHEMA,
        "stage": "locked_collection",
        "completed_at_utc": "2026-07-21T23:59:00Z",
        "prelock_registry_sha256": prelock_digest,
        "prelock_registry_file_sha256": "6" * 64,
        "attempt_marker_sha256": "7" * 64,
        "external_freeze_receipt_sha256": "8" * 64,
        "collector_exit_code": 0,
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "locked_manifest_file_sha256": _sha256_file(evaluation_source),
        "locked_manifest_payload_sha256": "1" * 64,
        "locked_raw_root": str(CANONICAL_LOCKED_RAW),
        "collection_kind": "locked_test",
        "selected_cases": sorted(CASES),
        "counts": {
            "cases": len(CASES),
            "trajectories": 12 * len(CASES),
            "rows": 12 * len(CASES) * TRAJECTORY_STEPS,
            "roles": {"locked_test": 12 * len(CASES)},
        },
        "locked_values_accessed_after_attempt": True,
    }
    collection_path.write_text(
        json.dumps(collection_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    collection_sha256 = _sha256_file(collection_path)
    attempt_path = evaluation_root / ATTEMPT_EVIDENCE_PATH
    attempt_payload = {
        "schema": ATTEMPT_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "one_shot": True,
        "started_at_utc": "2026-07-22T00:00:00Z",
        "locked_manifest_path": str(CANONICAL_LOCKED_MANIFEST),
        "output_path": str((artifact_root / "confirmation_output").resolve()),
        "prelock_registry_sha256": prelock_digest,
        "selected_update": selected_update,
        "locked_corpus_manifest_sha256": "1" * 64,
        "locked_manifest_file_sha256": _sha256_file(evaluation_source),
        "locked_collection_completion_sha256": collection_sha256,
        "external_freeze_receipt_sha256": "8" * 64,
        "runner_sha256": code_manifest[RUNNER_RELATIVE_PATH],
        "scientific_code_manifest_sha256": hashlib.sha256(
            _canonical_json(code_manifest).encode("ascii")
        ).hexdigest(),
        "evaluation_policy": dict(EVALUATION_POLICY),
        "evaluation_runtime": evaluation_runtime,
    }
    attempt_payload["attempt_identity_sha256"] = hashlib.sha256(
        _canonical_json(attempt_payload).encode("ascii")
    ).hexdigest()
    attempt_path.write_text(
        json.dumps(attempt_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    attempt_sha256 = _sha256_file(attempt_path)
    evaluation = {
        "schema": EVALUATION_PROVENANCE_SCHEMA,
        "stage": "confirmation",
        "role": "locked_test",
        "selected_update": selected_update,
        "prelock_registry_sha256": prelock_digest,
        "study_config_sha256": _config_sha256(config),
        "confirmation_runner_sha256": code_manifest[RUNNER_RELATIVE_PATH],
        "scientific_code_sha256_by_path": code_manifest,
        "scientific_code_manifest_sha256": hashlib.sha256(
            _canonical_json(code_manifest).encode("ascii")
        ).hexdigest(),
        "evaluator_sha256": _sha256_file(here / "study_evaluate.py"),
        "gate_sha256": _sha256_file(Path(study_gate_module.__file__).resolve()),
        "locked_corpus_manifest_sha256": "1" * 64,
        "locked_fault_manifest_sha256": "2" * 64,
        "attempt_marker_sha256": attempt_sha256,
        "attempt_marker_artifact": {
            "path": ATTEMPT_EVIDENCE_PATH,
            "sha256": attempt_sha256,
            "kind": "confirmation_attempt_marker",
            "identity": prelock_digest,
        },
        "locked_collection_completion_artifact": {
            "path": COLLECTION_EVIDENCE_PATH,
            "sha256": collection_sha256,
            "kind": "locked_collection_completion_marker",
            "identity": prelock_digest,
        },
        "rssm_frame_sha256": rssm_frame_sha256,
        "baseline_frame_sha256": baseline_frame_sha256,
        "rssm_result_artifact": {
            "path": "rssm_locked_h8.csv",
            "rows": len(rssm),
            "sha256": "4" * 64,
            "canonical_frame_sha256": rssm_frame_sha256,
        },
        "baseline_result_artifact": {
            "path": "baseline_locked_h8.csv",
            "rows": len(baselines),
            "sha256": "5" * 64,
            "canonical_frame_sha256": baseline_frame_sha256,
        },
        "evaluation_policy": dict(EVALUATION_POLICY),
        "evaluation_runtime": evaluation_runtime,
        "evaluation_source_manifest_artifact": {
            "path": evaluation_source.name,
            "sha256": _sha256_file(evaluation_source),
            "kind": "locked_evaluation_source_manifest",
            "identity": "locked_test",
        },
        "artifact_inventory": [
            evaluation_source.name,
            ATTEMPT_EVIDENCE_PATH,
            COLLECTION_EVIDENCE_PATH,
        ],
    }
    assert set(evaluation) == EVALUATION_RECEIPT_FIELDS
    return prelock, prelock_root, prelock_digest, evaluation, evaluation_root


def test_equal_weighting_is_nested_across_family_channel_sign_and_severity():
    rows = []
    family_channel_values = {
        ("bias", "zone_temperature_k"): 1.0,
        ("bias", "hvac_electric_power_w"): 3.0,
        ("drift", "zone_temperature_k"): 5.0,
        ("drift", "hvac_electric_power_w"): 5.0,
        ("stuck", "zone_temperature_k"): 9.0,
        ("stuck", "hvac_electric_power_w"): 9.0,
    }
    for (family, channel), value in family_channel_values.items():
        settings = ((0, 0.0),) if family == "stuck" else (
            (-1, 1.0),
            (-1, 2.0),
            (1, 1.0),
            (1, 2.0),
        )
        for sign, severity in settings:
            rows.append(
                {
                    "case": "case",
                    "model_seed": 1,
                    "trajectory_day": 1,
                    "trajectory_seed": 2,
                    "family": family,
                    "fault_channel": channel,
                    "sign": sign,
                    "severity": severity,
                    **{arm: value for arm in ARMS},
                    "persistence": value,
                }
            )
    score = _equal_weight_scores(
        pd.DataFrame(rows), families=("bias", "drift", "stuck")
    )
    # Bias averages to 2, drift to 5, and stuck to 9; families are then 1/3 each.
    assert len(score) == 1
    assert score.loc[0, "gated_h8"] == pytest.approx((2.0 + 5.0 + 9.0) / 3.0)


@pytest.fixture(scope="module")
def confirmation_frames():
    return _frames("confirmation")


def test_confirmation_provenance_uses_canonical_prelock_and_exact_receipt(
    confirmation_frames, tmp_path, monkeypatch
):
    rssm, baselines, config = confirmation_frames
    prelock, prelock_root, digest, evaluation, evaluation_root = (
        _confirmation_provenance(
            rssm, baselines, config, tmp_path / "provenance_artifacts"
        )
    )
    calls = {"prelock": 0}

    def validate_prelock(registry, root, frozen_config, expected):
        calls["prelock"] += 1
        assert registry is prelock
        assert root == prelock_root
        assert frozen_config == config
        assert expected == digest
        return registry

    monkeypatch.setattr(
        study_gate_module, "validate_prelock_registry_object", validate_prelock
    )
    monkeypatch.setattr(
        study_gate_module,
        "prelock_plan_sha256_by_case",
        lambda *args, **kwargs: {case: "6" * 64 for case in CASES},
    )
    monkeypatch.setattr(
        study_gate_module,
        "validate_locked_corpus_binding",
        lambda *args, **kwargs: (
            SimpleNamespace(manifest_sha256=evaluation["locked_corpus_manifest_sha256"]),
            [],
        ),
    )
    issues = validate_confirmation_provenance(
        prelock,
        evaluation,
        rssm,
        baselines,
        config,
        selected_update=100,
        prelock_artifact_root=prelock_root,
        evaluation_artifact_root=evaluation_root,
        expected_prelock_sha256=digest,
    )
    assert issues == []
    assert calls == {"prelock": 1}

    missing = dict(evaluation)
    missing.pop("attempt_marker_sha256")
    issues = validate_confirmation_provenance(
        prelock,
        missing,
        rssm,
        baselines,
        config,
        selected_update=100,
        prelock_artifact_root=prelock_root,
        evaluation_artifact_root=evaluation_root,
        expected_prelock_sha256=digest,
    )
    assert "evaluation receipt fields differ from the frozen schema" in issues


def test_standalone_gate_cli_rejects_confirmation_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study_gate",
            "--rssm",
            str(tmp_path / "unread.csv"),
            "--stage",
            "confirmation",
            "--selected-update",
            "100",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit):
        study_gate_module.main()


def test_forged_attempt_marker_is_rejected_semantically(
    confirmation_frames, tmp_path, monkeypatch
):
    rssm, baselines, config = confirmation_frames
    prelock, prelock_root, digest, evaluation, evaluation_root = (
        _confirmation_provenance(
            rssm, baselines, config, tmp_path / "forged_attempt"
        )
    )
    marker_path = evaluation_root / ATTEMPT_EVIDENCE_PATH
    marker = json.loads(marker_path.read_text(encoding="ascii"))
    marker["selected_update"] = 999
    marker.pop("attempt_identity_sha256")
    marker["attempt_identity_sha256"] = hashlib.sha256(
        _canonical_json(marker).encode("ascii")
    ).hexdigest()
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    forged_sha256 = _sha256_file(marker_path)
    evaluation["attempt_marker_sha256"] = forged_sha256
    evaluation["attempt_marker_artifact"]["sha256"] = forged_sha256

    monkeypatch.setattr(
        study_gate_module,
        "validate_prelock_registry_object",
        lambda *args, **kwargs: prelock,
    )
    monkeypatch.setattr(
        study_gate_module,
        "prelock_plan_sha256_by_case",
        lambda *args, **kwargs: {case: "6" * 64 for case in CASES},
    )
    monkeypatch.setattr(
        study_gate_module,
        "validate_locked_corpus_binding",
        lambda *args, **kwargs: (
            SimpleNamespace(
                manifest_sha256=evaluation["locked_corpus_manifest_sha256"]
            ),
            [],
        ),
    )
    issues = validate_confirmation_provenance(
        prelock,
        evaluation,
        rssm,
        baselines,
        config,
        selected_update=100,
        prelock_artifact_root=prelock_root,
        evaluation_artifact_root=evaluation_root,
        expected_prelock_sha256=digest,
    )
    assert "attempt-marker selected_update differs from the locked run" in issues


def test_complete_confirmation_gate_passes_with_constant_positive_effect(
    confirmation_frames,
    tmp_path,
    monkeypatch,
):
    rssm, baselines, config = confirmation_frames
    prelock, prelock_root, prelock_sha256, evaluation, evaluation_root = (
        _confirmation_provenance(
        rssm, baselines, config, tmp_path / "artifacts"
        )
    )
    monkeypatch.setattr(
        study_gate_module,
        "validate_confirmation_provenance",
        lambda *args, **kwargs: [],
    )
    paired, result = evaluate_study_gate(
        rssm,
        config,
        stage="confirmation",
        selected_update=100,
        baseline_frame=baselines,
        prelock_registry=prelock,
        prelock_artifact_root=prelock_root,
        expected_prelock_sha256=prelock_sha256,
        evaluation_receipt=evaluation,
        evaluation_artifact_root=evaluation_root,
    )
    assert len(paired) * len(ARMS) == len(rssm)
    assert result["decision"] == "PASS", result["provenance"]
    assert result["gate_pass"] is True
    assert result["paper_claim_allowed"] is True
    assert all(value is True for value in result["checks"].values())
    assert result["primary"]["relative_improvement"] == pytest.approx(0.2)
    assert result["primary"]["raw_improvement_ci95"][0] > 0.0
    assert result["primary"]["positive_seed_count"] == 5
    assert result["healthy"]["relative_degradation"] == pytest.approx(0.02)
    assert result["competence"]["missing_comparators"] == []
    assert result["competence"]["paired_comparison"]["rssm_arm"] == "gated_h8"
    assert result["competence"]["paired_comparison"]["ratio_ci95"][1] <= 1.10
    assert set(result["raw_unit_mae_by_channel"]["silent_faults"]) == {
        "zone_temperature_k",
        "hvac_electric_power_w",
    }
    assert "dropout" not in result["primary"]["family_raw_improvement"]
    assert result["dropout_separate_diagnostic"]["standardized_mae"]
    assert result["action_use_diagnostic"]["gated_h8_pass"] is True
    assert result["action_use_diagnostic"]["scope"].startswith(
        "equal-weight healthy"
    )
    assert result["auxiliary_intervention"] == {
        "aux_h8_mae": pytest.approx(result["primary"]["ungated_h8_mae"]),
        "gated_h8_mae": pytest.approx(result["primary"]["gated_h8_mae"]),
        "raw_improvement": pytest.approx(result["primary"]["raw_improvement"]),
        "aux_and_ungated_core_bit_identical": True,
        "aux_and_ungated_outputs_bit_identical": True,
        "auxiliary_intervention_isolated": True,
    }


def test_confirmation_is_incomplete_when_a_competence_baseline_is_missing(
    confirmation_frames,
    tmp_path,
    monkeypatch,
):
    rssm, baselines, config = confirmation_frames
    incomplete = baselines.loc[baselines["arm"] != "deterministic_gru"]
    prelock, prelock_root, prelock_sha256, evaluation, evaluation_root = (
        _confirmation_provenance(
        rssm, incomplete, config, tmp_path / "artifacts"
        )
    )
    monkeypatch.setattr(
        study_gate_module,
        "validate_confirmation_provenance",
        lambda *args, **kwargs: [],
    )
    _, result = evaluate_study_gate(
        rssm,
        config,
        stage="confirmation",
        selected_update=100,
        baseline_frame=incomplete,
        prelock_registry=prelock,
        prelock_artifact_root=prelock_root,
        expected_prelock_sha256=prelock_sha256,
        evaluation_receipt=evaluation,
        evaluation_artifact_root=evaluation_root,
    )
    assert result["decision"] == "INCOMPLETE"
    assert result["gate_pass"] is False
    assert result["checks"][
        "rssm_meets_frozen_baseline_competence_bounds"
    ] is None
    assert result["competence"]["missing_comparators"] == ["deterministic_gru"]


def test_development_is_screen_only_and_never_returns_a_paper_pass():
    rssm, baselines, config = _frames("development")
    _, result = evaluate_study_gate(
        rssm,
        config,
        stage="development",
        selected_update=100,
        baseline_frame=baselines,
    )
    assert result["decision"] == "SCREEN_GO"
    assert result["gate_pass"] is False
    assert result["paper_claim_allowed"] is False
    assert result["checks"][
        "at_least_four_of_five_paired_seeds_positive"
    ] is None
    missing_gru = baselines.loc[baselines["arm"] != "deterministic_gru"]
    _, incomplete = evaluate_study_gate(
        rssm,
        config,
        stage="development",
        selected_update=100,
        baseline_frame=missing_gru,
    )
    assert incomplete["decision"] == "INCOMPLETE"
    assert incomplete["development_screen"]["checks"][
        "gated_within_1p15x_strongest_completed_baseline"
    ] is None
    contaminated = rssm.iloc[[0]].copy()
    contaminated["role"] = "locked_test"
    contaminated["update"] = 200
    with pytest.raises(ValueError, match="validation rows only"):
        evaluate_study_gate(
            pd.concat([rssm, contaminated], ignore_index=True),
            config,
            stage="development",
            selected_update=100,
            baseline_frame=baselines,
        )


def test_incomplete_arm_pair_is_rejected_before_aggregation(confirmation_frames):
    rssm, _, config = confirmation_frames
    with pytest.raises(ValueError, match="incomplete grid"):
        matched_arm_metrics(
            rssm.iloc[:-1],
            config,
            stage="confirmation",
            selected_update=100,
        )


def test_conjunctive_gate_stops_on_case_and_healthy_failures(
    confirmation_frames,
    tmp_path,
    monkeypatch,
):
    rssm, baselines, config = confirmation_frames
    failed = rssm.copy()
    first_case = sorted(CASES)[0]
    silent_gated = (
        (failed["case"] == first_case)
        & failed["family"].isin(("bias", "drift", "stuck"))
        & (failed["arm"] == "gated_h8")
    )
    failed.loc[silent_gated, "prediction_raw"] = 12.0
    healthy_gated = (failed["family"] == "healthy") & (
        failed["arm"] == "gated_h8"
    )
    failed.loc[healthy_gated, "prediction_raw"] *= 1.10
    auxiliary = failed["arm"] == "aux_h8"
    failed.loc[auxiliary, "prediction_raw"] += 0.25
    failed["standardized_abs_error"] = (
        failed["prediction_raw"] - failed["target_raw"]
    ).abs()
    failed["action_prediction_change_standardized"] = (
        failed["prediction_raw"] - failed["alternate_action_prediction_raw"]
    ).abs()
    prelock, prelock_root, prelock_sha256, evaluation, evaluation_root = (
        _confirmation_provenance(
        failed, baselines, config, tmp_path / "artifacts"
        )
    )
    monkeypatch.setattr(
        study_gate_module,
        "validate_confirmation_provenance",
        lambda *args, **kwargs: [],
    )
    _, result = evaluate_study_gate(
        failed,
        config,
        stage="confirmation",
        selected_update=100,
        baseline_frame=baselines,
        prelock_registry=prelock,
        prelock_artifact_root=prelock_root,
        expected_prelock_sha256=prelock_sha256,
        evaluation_receipt=evaluation,
        evaluation_artifact_root=evaluation_root,
    )
    assert result["decision"] == "STOP", result["provenance"]
    assert result["gate_pass"] is False
    assert result["checks"]["every_case_positive"] is False
    assert result["checks"]["healthy_point_degradation_at_most_5pct"] is False
    assert result["checks"]["gated_improves_over_auxiliary_only"] is False
    assert result["auxiliary_intervention"][
        "aux_and_ungated_core_bit_identical"
    ] is True
    assert result["auxiliary_intervention"][
        "aux_and_ungated_outputs_bit_identical"
    ] is False


def test_confirmation_without_cross_bound_receipt_is_incomplete(confirmation_frames):
    rssm, baselines, config = confirmation_frames
    _, result = evaluate_study_gate(
        rssm,
        config,
        stage="confirmation",
        selected_update=100,
        baseline_frame=baselines,
    )
    assert result["decision"] == "INCOMPLETE"
    assert result["gate_pass"] is False
    assert result["checks"]["confirmation_provenance_integrity_bound"] is None
    assert result["provenance"]["issues"] == [
        "pre-lock provenance registry is missing or invalid"
    ]


def test_baselines_must_use_exact_rssm_trajectory_and_cell_identities(
    confirmation_frames,
    tmp_path,
):
    rssm, baselines, config = confirmation_frames
    shifted = baselines.copy()
    shifted["trajectory_day"] += 1_000
    shifted["trajectory_seed"] += 1_000
    shifted["cell_id"] = "shifted:" + shifted["cell_id"]
    prelock, prelock_root, prelock_sha256, evaluation, evaluation_root = (
        _confirmation_provenance(
        rssm, shifted, config, tmp_path / "artifacts"
        )
    )
    with pytest.raises(ValueError, match="different frozen cells"):
        evaluate_study_gate(
            rssm,
            config,
            stage="confirmation",
            selected_update=100,
            baseline_frame=shifted,
            prelock_registry=prelock,
            prelock_artifact_root=prelock_root,
            expected_prelock_sha256=prelock_sha256,
            evaluation_receipt=evaluation,
            evaluation_artifact_root=evaluation_root,
        )


def test_seed_specific_cell_ids_cannot_hide_model_independent_differences(
    confirmation_frames,
):
    rssm, _, config = confirmation_frames
    shifted = rssm.copy()
    second_seed = config.confirmatory_seeds[1]
    seed_rows = shifted["model_seed"] == second_seed
    shifted.loc[seed_rows, "cell_id"] = (
        shifted.loc[seed_rows, "cell_id"] + f":seed{second_seed}"
    )
    shifted.loc[seed_rows, "target_raw"] = 1.0
    with pytest.raises(ValueError, match="cell identity differs across model seeds"):
        matched_arm_metrics(
            shifted,
            config,
            stage="confirmation",
            selected_update=100,
        )
