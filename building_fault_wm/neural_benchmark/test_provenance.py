from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Ridge

from .baselines import (
    ARXFeatureSpec,
    BaselineSelectionReceipt,
    DirectH8FeatureSpec,
    DirectH8GRU,
    baseline_producer_code_manifest,
    _canonical_sha256 as baseline_canonical_sha256,
    _frame_sha256,
    _ridge_state_sha256,
)
from .fault_data import CorpusIndex, CorpusRecord, TrajectoryKey
from .protocol import CASES
from .runtime_provenance import numerical_runtime_fingerprint
from . import provenance as provenance_module
from .provenance import (
    BASELINE_RECEIPT_SCHEMA,
    _baseline_expected_entries,
    _gru_model_payload,
    _restore_gru,
    _restore_ridge,
    _safe_path,
    _validate_development_screen_evidence,
    _validate_gru_bundle_extension,
    _validate_selection,
    build_prelock_registry,
    frozen_fault_contract,
    load_strict_json,
    load_selected_baseline_models,
    validate_locked_corpus_binding,
    validate_prelock_bundle,
    write_baseline_selection_bundle,
    write_prelock_registry,
    write_validation_selection,
)
from .study_config import ARMS, StudyConfig
from . import study_evaluate as study_evaluate_module
from . import study_gate as study_gate_module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_baseline_entry_scope_supports_single_gru_work_units():
    config = StudyConfig()
    case = next(iter(CASES))
    seed = config.confirmatory_seeds[-1]
    assert _baseline_expected_entries(
        baseline="deterministic_gru",
        config=config,
        gru_seeds=(seed,),
        cases=(case,),
    ) == {f"{case}:seed{seed}"}
    assert _baseline_expected_entries(
        baseline="deterministic_gru",
        config=config,
        gru_seeds=(seed,),
    ) == {f"{selected}:seed{seed}" for selected in CASES}
    with pytest.raises(ValueError, match="baseline case scope"):
        _baseline_expected_entries(
            baseline="deterministic_gru",
            config=config,
            gru_seeds=(seed,),
            cases=(case, case),
        )
    with pytest.raises(ValueError, match="baseline case scope"):
        _baseline_expected_entries(
            baseline="deterministic_gru",
            config=config,
            gru_seeds=(seed,),
            cases=("unknown-case",),
        )


def test_strict_json_and_safe_paths_reject_ambiguous_inputs(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"identity":1,"identity":2}\n', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate JSON identity"):
        load_strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"score":NaN}\n', encoding="ascii")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_strict_json(nonfinite)

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("bound\n", encoding="ascii")
    assert _safe_path(tmp_path, "artifact.txt") == artifact
    for unsafe in ("../artifact.txt", "./artifact.txt", "x/../artifact.txt", "/tmp/x"):
        with pytest.raises(ValueError, match="artifact path"):
            _safe_path(tmp_path, unsafe)


def test_frozen_fault_contract_contains_all_role_signatures():
    records = tuple(
        CorpusRecord(
            key=TrajectoryKey(case, "fit", index, index),
            relative_path=f"{case}/{index}.csv",
            source_sha256=_digest(case),
            rows=192,
            step_seconds=900,
            base_setpoint_k=294.15,
            action_amplitude_k=0.75,
        )
        for index, case in enumerate(CASES)
    )
    index = CorpusIndex(
        root=Path("/tmp/not-read"),
        manifest_path=Path("/tmp/not-read/manifest.json"),
        manifest_sha256=_digest("manifest"),
        collection_kind="development",
        prelock_registry_sha256=None,
        allowed_roles=("fit", "validation"),
        records=records,
        plan_sha256_by_case=tuple((case, _digest(case + "plan")) for case in CASES),
    )
    payload = frozen_fault_contract(index, StudyConfig())
    assert set(payload["signatures_by_role"]) == {
        "fit",
        "validation",
        "locked_test",
    }
    assert all(payload["signatures_by_role"].values())
    assert payload["development_corpus_manifest_sha256"] == index.manifest_sha256


def test_confirmatory_gru_bundle_is_an_exact_development_seed_extension():
    screened_entry = {
        "receipt": {"selected_model_state_sha256": _digest("model")},
        "score_table": {"records": [{"score": 1.0}]},
        "model": {"state_dict": {"weight": torch.tensor([1.0, 2.0])}},
    }
    development = {"entries": {"case:seed1": screened_entry}}
    confirmatory = {
        "entries": {
            "case:seed1": {
                "receipt": dict(screened_entry["receipt"]),
                "score_table": json.loads(json.dumps(screened_entry["score_table"])),
                "model": {
                    "state_dict": {"weight": torch.tensor([1.0, 2.0])}
                },
            },
            "case:seed2": {"new": True},
        }
    }
    _validate_gru_bundle_extension(development, confirmatory)
    confirmatory["entries"]["case:seed1"]["model"]["state_dict"]["weight"][0] = 9.0
    with pytest.raises(ValueError, match="changed screened entry"):
        _validate_gru_bundle_extension(development, confirmatory)


def test_development_screen_evidence_is_recomputed(monkeypatch, tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    completion = root / "run_complete.json"
    completion.write_text(
        json.dumps(
            {
                "decision": "SCREEN_GO",
                "rssm_result": {},
                "baseline_result": {},
                "gate_artifacts": {
                    "matched_metrics": {},
                    "gate_result": {},
                },
            }
        )
        + "\n",
        encoding="ascii",
    )
    registry = {
        "development_run_complete_artifact": {"path": "run_complete.json"},
        "development_rssm_result_artifact": {},
        "development_baseline_result_artifact": {},
        "development_gate_paired_artifact": {},
        "development_gate_result_artifact": {},
    }
    rssm = pd.DataFrame({"kind": ["rssm"]})
    baseline = pd.DataFrame({"kind": ["baseline"]})
    paired = pd.DataFrame({"metric": [1.0]})

    def fake_frame(metadata, reference, artifact_root, *, label):
        if label.startswith("RSSM"):
            return rssm
        if label.startswith("baseline"):
            return baseline
        return paired

    persisted_result = {"decision": "SCREEN_GO", "checks": {"all": True}}
    monkeypatch.setattr(provenance_module, "_validate_frame_artifact_metadata", fake_frame)
    monkeypatch.setattr(
        provenance_module,
        "_validate_json_artifact_metadata",
        lambda *args, **kwargs: persisted_result,
    )
    monkeypatch.setattr(
        study_gate_module,
        "evaluate_study_gate",
        lambda *args, **kwargs: (paired.copy(), dict(persisted_result)),
    )
    _validate_development_screen_evidence(
        registry, root, StudyConfig(), selected_update=100
    )

    monkeypatch.setattr(
        study_gate_module,
        "evaluate_study_gate",
        lambda *args, **kwargs: (
            paired.copy(),
            {"decision": "SCREEN_STOP", "checks": {"all": False}},
        ),
    )
    with pytest.raises(ValueError, match="gate result does not recompute"):
        _validate_development_screen_evidence(
            registry, root, StudyConfig(), selected_update=100
        )


def test_validation_selection_rejects_nonbest_frozen_update(
    tmp_path: Path, monkeypatch
):
    config = StudyConfig()
    rows = [
        {"update": update, "common_score": float(update), "selected": update == 100}
        for update in config.validation_checkpoints
    ]
    source_rows = pd.DataFrame({"frozen_row": [np.nextafter(1.0, 2.0)]})
    monkeypatch.setattr(
        study_evaluate_module,
        "validation_selection_scores",
        lambda frame, frozen_config: pd.DataFrame(rows),
    )
    path = tmp_path / "selection.json"
    write_validation_selection(path, source_rows, config)
    _validate_selection(path, config, 100)

    payload = load_strict_json(path)
    payload["score_rows"][0]["selected"] = False
    payload["score_rows"][1]["selected"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="selection scores differ"):
        _validate_selection(path, config, 100)


def test_baseline_bundle_carries_loadable_ridge_and_detects_tensor_tamper(
    tmp_path: Path,
):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(24, 6))
    targets = rng.normal(size=(24, 4))
    model = Ridge(alpha=0.1).fit(features, targets)
    score_table = pd.DataFrame(
        {
            "alpha": [0.1, 1.0],
            "validation_score": [1.0, 2.0],
            "selected": [True, False],
        }
    )
    receipt = BaselineSelectionReceipt(
        schema=BASELINE_RECEIPT_SCHEMA,
        baseline="ridge_arx",
        case=next(iter(CASES)),
        model_seed=0,
        fit_role="fit",
        validation_role="validation",
        fit_variant_identity_sha256=_digest("fit"),
        validation_variant_identity_sha256=_digest("validation"),
        fit_scalers_sha256=_digest("scalers"),
        feature_contract_sha256=baseline_canonical_sha256(ARXFeatureSpec().contract),
        training_config_sha256=_digest("training"),
        selection_metric="equal_mean_case_family_fault_channel_standardized_H8_MAE",
        candidate_grid=(0.1, 1.0),
        selected_candidate=0.1,
        selected_validation_score=1.0,
        score_table_sha256=_frame_sha256(score_table),
        training_updates=1,
        batch_size=24,
        schedule_sha256=_digest("schedule"),
        selected_model_state_sha256=_ridge_state_sha256(model),
        producer_code_sha256=baseline_producer_code_manifest()["sha256"],
        runtime_fingerprint=numerical_runtime_fingerprint(
            "cpu", include_sklearn=True
        ),
    )
    path = tmp_path / "ridge_bundle.pt"
    write_baseline_selection_bundle(
        path,
        baseline="ridge_arx",
        entries={f"{receipt.case}:seed0": (receipt, model, score_table)},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    entry = payload["entries"][f"{receipt.case}:seed0"]
    restored = _restore_ridge(entry["model"], entry["receipt"])
    assert np.array_equal(restored.predict(features), model.predict(features))
    loaded = load_selected_baseline_models(
        path, baseline="ridge_arx", config=StudyConfig()
    )
    assert set(loaded) == {f"{receipt.case}:seed0"}

    entry["model"]["coef"][0, 0] += 1.0
    with pytest.raises(ValueError, match="differs from its receipt"):
        _restore_ridge(entry["model"], entry["receipt"])

    producer_tampered = torch.load(path, map_location="cpu", weights_only=True)
    producer_tampered["producer_code"]["files"][
        "multicase_fault_benchmark/baselines.py"
    ] = "0" * 64
    tampered_path = tmp_path / "ridge_bundle_tampered.pt"
    torch.save(producer_tampered, tampered_path)
    with pytest.raises(ValueError, match="bundle identity differs"):
        load_selected_baseline_models(
            tampered_path, baseline="ridge_arx", config=StudyConfig()
        )

    runtime_tampered = torch.load(path, map_location="cpu", weights_only=True)
    runtime_tampered["entries"][f"{receipt.case}:seed0"]["receipt"][
        "runtime_fingerprint"
    ]["numpy_version"] = "0.0.0"
    runtime_path = tmp_path / "ridge_bundle_runtime_tampered.pt"
    torch.save(runtime_tampered, runtime_path)
    with pytest.raises(ValueError, match="numerical runtime fingerprint"):
        load_selected_baseline_models(
            runtime_path, baseline="ridge_arx", config=StudyConfig()
        )


def test_gru_restore_uses_the_frozen_direct_h8_feature_spec():
    config = StudyConfig()
    model = DirectH8GRU(
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        spec=DirectH8FeatureSpec(horizon=config.direct_horizon),
    )
    payload = _gru_model_payload(model)
    receipt = {"selected_model_state_sha256": payload["state_sha256"]}

    restored = _restore_gru(payload, receipt, config)

    assert restored.spec == DirectH8FeatureSpec(history=40, horizon=8)
    assert restored.expected_feature_dim == model.expected_feature_dim
    assert all(
        torch.equal(restored.state_dict()[name], value)
        for name, value in model.state_dict().items()
    )

    stale_payload = {**payload, "model_config": {**payload["model_config"], "history": 8}}
    with pytest.raises(ValueError, match="configuration differs from protocol"):
        _restore_gru(stale_payload, receipt, config)


def test_locked_binding_requires_exact_external_prelock_digest(monkeypatch, tmp_path):
    expected = _digest("prelock")
    manifest = tmp_path / "manifests" / "locked_test_all_corpus_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="ascii")
    fake_index = SimpleNamespace(
        collection_kind="locked_test",
        allowed_roles=("locked_test",),
        records=tuple(
            SimpleNamespace(key=SimpleNamespace(case=case)) for case in CASES
        ),
        prelock_registry_sha256=expected,
        plan_sha256_by_case=tuple((case, _digest(case)) for case in CASES),
    )
    monkeypatch.setattr(provenance_module, "_strict_scan_corpus_json", lambda path: None)
    monkeypatch.setattr(
        provenance_module,
        "load_corpus_index",
        lambda path, **kwargs: fake_index,
    )
    _, issues = validate_locked_corpus_binding(
        manifest,
        prelock_registry={"schema": "test"},
        expected_prelock_sha256=expected,
    )
    assert issues == []

    _, issues = validate_locked_corpus_binding(
        manifest,
        prelock_registry={"schema": "test"},
        expected_prelock_sha256=expected,
        expected_plan_sha256_by_case={case: _digest("wrong") for case in CASES},
    )
    assert issues == [
        "locked corpus binding failed: locked corpus plans differ from the pre-lock plans"
    ]

    _, issues = validate_locked_corpus_binding(
        manifest,
        prelock_registry={"schema": "test"},
        expected_prelock_sha256=_digest("different"),
    )
    assert issues == [
        "locked corpus binding failed: locked corpus embeds a different pre-lock digest"
    ]


def test_prelock_bundle_rejects_noncanonical_reference_path(monkeypatch, tmp_path):
    config = StudyConfig()
    root = tmp_path / "bundle"
    root.mkdir()

    def artifact(name: str) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="ascii")
        return path

    corpus = artifact("corpus/manifests/development_all_corpus_manifest.json")
    rssm_result = artifact("development/rssm_validation_h8.csv")
    baseline_result = artifact("development/baseline_validation_h8.csv")
    gate_paired = artifact("development/gate/matched_h8_arm_metrics.csv")
    gate_result = artifact("development/gate/study_gate.json")
    fault = artifact("fault/frozen.json")
    selection = artifact("selection/checkpoint.json")
    scalers = {case: artifact(f"scalers/{case}.json") for case in CASES}
    checkpoints = {
        f"{case}:seed{seed}:{arm}:u0100": artifact(
            f"checkpoints/{case}_{seed}_{arm}.pt"
        )
        for case in CASES
        for seed in config.confirmatory_seeds
        for arm in ARMS
    }
    schedules = {
        f"{case}:seed{seed}": artifact(f"schedules/{case}_{seed}.json")
        for case in CASES
        for seed in config.confirmatory_seeds
    }
    baselines = {
        baseline: artifact(f"baselines/{baseline}.pt")
        for baseline in ("ridge_arx", "direct_h8_ridge", "deterministic_gru")
    }
    development_gru = artifact("baselines/development_deterministic_gru.pt")
    from .study_development import (
        RUNNER_SCHEMA,
        RUN_RECEIPT_SCHEMA,
        scientific_code_manifest,
    )

    code_manifest = scientific_code_manifest()
    code_manifest_sha256 = provenance_module.canonical_sha256(code_manifest)
    runtime = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    runner_sha256 = code_manifest[
        "multicase_fault_benchmark/study_development.py"
    ]
    corpus_identity = _digest("development-corpus")
    run_config = root / "run_config.json"
    run_config.write_text(
        json.dumps(
            {
                "schema": RUNNER_SCHEMA,
                "stage": "development",
                "interpretation": "frozen_development_screen",
                "scientific_screen_enabled": True,
                "device": "cpu",
                "corpus_manifest_sha256": corpus_identity,
                "corpus_manifest_file_sha256": hashlib.sha256(
                    corpus.read_bytes()
                ).hexdigest(),
                "runner_sha256": runner_sha256,
                "scientific_code_sha256_by_path": code_manifest,
                "scientific_code_manifest_sha256": code_manifest_sha256,
                "numerical_runtime": runtime,
                "study_config_sha256": provenance_module.canonical_sha256(
                    config.to_dict()
                ),
                "study_config": config.to_dict(),
                "resume_requested": False,
                "staging_identity_sha256": _digest("staging"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    required_inventory = {
        "run_config.json": hashlib.sha256(run_config.read_bytes()).hexdigest(),
        "frozen/frozen_fault_contract.json": hashlib.sha256(
            fault.read_bytes()
        ).hexdigest(),
        "frozen/validation_selection.json": hashlib.sha256(
            selection.read_bytes()
        ).hexdigest(),
        "rssm_validation_h8.csv": hashlib.sha256(rssm_result.read_bytes()).hexdigest(),
        "baseline_validation_h8.csv": hashlib.sha256(
            baseline_result.read_bytes()
        ).hexdigest(),
        "gate/matched_h8_arm_metrics.csv": hashlib.sha256(
            gate_paired.read_bytes()
        ).hexdigest(),
        "gate/study_gate.json": hashlib.sha256(gate_result.read_bytes()).hexdigest(),
        **{
            f"frozen/baselines/{arm}.pt": hashlib.sha256(
                (development_gru if arm == "deterministic_gru" else path).read_bytes()
            ).hexdigest()
            for arm, path in baselines.items()
        },
        **{
            f"training/{case}/seed{seed}/training_schedule.json": hashlib.sha256(
                schedules[f"{case}:seed{seed}"].read_bytes()
            ).hexdigest()
            for case in CASES
            for seed in config.development_seeds
        },
        **{
            (
                f"training/{case}/seed{seed}/checkpoints/"
                f"{arm}_u0100.pt"
            ): hashlib.sha256(
                checkpoints[f"{case}:seed{seed}:{arm}:u0100"].read_bytes()
            ).hexdigest()
            for case in CASES
            for seed in config.development_seeds
            for arm in ARMS
        },
    }
    run_complete_payload = {
        "schema": RUN_RECEIPT_SCHEMA,
        "stage": "development",
        "interpretation": "frozen_development_screen",
        "scientific_screen_reported": True,
        "decision": "SCREEN_GO",
        "selected_update": 100,
        "corpus_manifest_sha256": corpus_identity,
        "corpus_manifest_file_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "fault_manifest_sha256": _digest("fault-manifest"),
        "study_config_sha256": provenance_module.canonical_sha256(
            config.to_dict()
        ),
        "runner_sha256": runner_sha256,
        "scientific_code_sha256_by_path": code_manifest,
        "scientific_code_manifest_sha256": code_manifest_sha256,
        "numerical_runtime": runtime,
        "staging_identity_sha256": _digest("staging"),
        "rssm_result": {
            "path": "rssm_validation_h8.csv",
            "rows": 1,
            "sha256": hashlib.sha256(rssm_result.read_bytes()).hexdigest(),
            "canonical_frame_sha256": _digest("rssm-frame"),
        },
        "baseline_result": {
            "path": "baseline_validation_h8.csv",
            "rows": 1,
            "sha256": hashlib.sha256(baseline_result.read_bytes()).hexdigest(),
            "canonical_frame_sha256": _digest("baseline-frame"),
        },
        "gate_artifacts": {
            "matched_metrics": {
                "path": "gate/matched_h8_arm_metrics.csv",
                "rows": 1,
                "sha256": hashlib.sha256(gate_paired.read_bytes()).hexdigest(),
                "canonical_frame_sha256": _digest("paired-frame"),
            },
            "gate_result": {
                "path": "gate/study_gate.json",
                "sha256": hashlib.sha256(gate_result.read_bytes()).hexdigest(),
                "canonical_json_sha256": _digest("gate-result"),
            },
        },
        "validation_selection_sha256": required_inventory[
            "frozen/validation_selection.json"
        ],
        "baseline_bundle_sha256": {
            arm: required_inventory[f"frozen/baselines/{arm}.pt"]
            for arm in baselines
        },
        "wall_time": {},
        "artifact_inventory_excludes_this_receipt": [
            {"path": path, "sha256": digest, "bytes": 1}
            for path, digest in sorted(required_inventory.items())
        ],
    }
    run_complete = root / "run_complete.json"
    run_complete.write_text(
        json.dumps(run_complete_payload, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        provenance_module,
        "validate_prelock_registry_semantics",
        lambda *args, **kwargs: [],
    )
    registry = build_prelock_registry(
        artifact_root=root,
        development_run_config_path=run_config,
        development_run_complete_path=run_complete,
        development_rssm_result_path=rssm_result,
        development_baseline_result_path=baseline_result,
        development_gate_paired_path=gate_paired,
        development_gate_result_path=gate_result,
        development_gru_baseline_bundle=development_gru,
        development_corpus_manifest=corpus,
        frozen_fault_contract_path=fault,
        validation_selection_path=selection,
        fit_scalers_by_case=scalers,
        checkpoints_by_identity=checkpoints,
        schedules_by_case_seed=schedules,
        baseline_bundles_by_arm=baselines,
        config=config,
        selected_update=100,
    )
    registry_path = tmp_path / "prelock.json"
    digest = write_prelock_registry(registry_path, registry)
    assert validate_prelock_bundle(registry_path, root, config, digest) == registry

    tampered = json.loads(json.dumps(registry))
    original = tampered["fault_manifest_artifact"]["path"]
    tampered["fault_manifest_artifact"]["path"] = f"fault/../{original}"
    tampered_path = tmp_path / "tampered.json"
    tampered_digest = write_prelock_registry(tampered_path, tampered)
    with pytest.raises(ValueError, match="artifact path"):
        validate_prelock_bundle(tampered_path, root, config, tampered_digest)

    for index, code_path in enumerate(
        (
            "multicase_fault_benchmark/study_evaluate.py",
            "multicase_fault_benchmark/study_gate.py",
            "multicase_fault_benchmark/fault_data.py",
        )
    ):
        receipt_tampered = json.loads(json.dumps(run_complete_payload))
        receipt_tampered["scientific_code_sha256_by_path"][code_path] = "0" * 64
        receipt_tampered["scientific_code_manifest_sha256"] = (
            provenance_module.canonical_sha256(
                receipt_tampered["scientific_code_sha256_by_path"]
            )
        )
        run_complete.write_text(
            json.dumps(receipt_tampered, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="ascii",
        )
        receipt_registry = json.loads(json.dumps(registry))
        receipt_registry["development_run_complete_artifact"]["sha256"] = (
            hashlib.sha256(run_complete.read_bytes()).hexdigest()
        )
        receipt_registry_path = tmp_path / f"receipt_tampered_{index}.json"
        receipt_registry_digest = write_prelock_registry(
            receipt_registry_path, receipt_registry
        )
        with pytest.raises(ValueError, match="scientific-code manifest"):
            validate_prelock_bundle(
                receipt_registry_path, root, config, receipt_registry_digest
            )

    run_complete.write_text(
        json.dumps(run_complete_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    first_case = next(iter(CASES))
    checkpoint_identity = (
        f"{first_case}:seed{config.development_seeds[0]}:{ARMS[0]}:u0100"
    )
    schedule_identity = f"{first_case}:seed{config.development_seeds[0]}"
    substitutions = (
        (
            "checkpoint_artifact_by_identity",
            checkpoint_identity,
            checkpoints[checkpoint_identity],
        ),
        (
            "training_schedule_artifact_by_case_seed",
            schedule_identity,
            schedules[schedule_identity],
        ),
    )
    for index, (field, identity, substituted_artifact) in enumerate(substitutions):
        original_bytes = substituted_artifact.read_bytes()
        substituted_artifact.write_text(
            f"substituted after SCREEN_GO:{index}\n", encoding="ascii"
        )
        substituted_registry = json.loads(json.dumps(registry))
        substituted_registry[field][identity]["sha256"] = hashlib.sha256(
            substituted_artifact.read_bytes()
        ).hexdigest()
        substituted_path = tmp_path / f"substituted_{index}.json"
        substituted_digest = write_prelock_registry(
            substituted_path, substituted_registry
        )
        with pytest.raises(ValueError, match="does not bind training/"):
            validate_prelock_bundle(
                substituted_path, root, config, substituted_digest
            )
        substituted_artifact.write_bytes(original_bytes)
