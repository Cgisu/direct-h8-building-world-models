from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from . import study_prelock as prelock
from .runtime_provenance import numerical_runtime_fingerprint
from .study_config import StudyConfig
from .study_development import (
    RUN_CONFIG_FIELDS,
    RUNNER_SCHEMA,
    RUN_RECEIPT_FIELDS,
    RUN_RECEIPT_SCHEMA,
)


def _sha() -> str:
    return "a" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="ascii")


def test_screen_stop_is_rejected_before_prelock_staging(tmp_path: Path):
    development = tmp_path / "development"
    run_config = {field: None for field in RUN_CONFIG_FIELDS}
    run_config.update(
        {
            "schema": RUNNER_SCHEMA,
            "stage": "development",
            "interpretation": "frozen_development_screen",
            "scientific_screen_enabled": True,
        }
    )
    run_complete = {field: None for field in RUN_RECEIPT_FIELDS}
    run_complete.update(
        {
            "schema": RUN_RECEIPT_SCHEMA,
            "stage": "development",
            "interpretation": "frozen_development_screen",
            "scientific_screen_reported": True,
            "decision": "SCREEN_STOP",
        }
    )
    _write_json(development / "run_config.json", run_config)
    _write_json(development / "run_complete.json", run_complete)
    output = tmp_path / "prelock"
    with pytest.raises(ValueError, match="requires SCREEN_GO"):
        prelock.run_prelock_preparation(
            tmp_path / "not-opened-manifest.json", development, output
        )
    assert not output.exists()
    assert not prelock._staging_path(output).exists()


def test_prelock_resume_preserves_work_and_clears_only_recomputable(tmp_path: Path):
    output = tmp_path / "prelock"
    runtime = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    identity = {
        "schema": prelock.PREPARATION_STAGING_SCHEMA,
        "numerical_runtime": runtime,
        "identity_sha256": _sha(),
    }
    staging = prelock._prepare_staging(output, identity, resume=False)
    work = staging / prelock.WORK_NAME / "completed.txt"
    work.parent.mkdir(parents=True, exist_ok=True)
    work.write_text("complete\n", encoding="ascii")
    (staging / prelock.BUNDLE_NAME).mkdir()
    (staging / prelock.REGISTRY_NAME).write_text("{}\n", encoding="ascii")
    resumed = prelock._prepare_staging(output, identity, resume=True)
    assert resumed == staging
    assert work.read_text(encoding="ascii") == "complete\n"
    assert not (staging / prelock.BUNDLE_NAME).exists()
    assert not (staging / prelock.REGISTRY_NAME).exists()

    changed = {**identity, "identity_sha256": "b" * 64}
    with pytest.raises(ValueError, match="identity differs"):
        prelock._prepare_staging(output, changed, resume=True)


def test_prelock_runner_uses_only_extra_seeds_and_publishes_atomically(
    monkeypatch, tmp_path: Path
):
    config = StudyConfig()
    manifest = tmp_path / "development_manifest.json"
    manifest.write_text("{}\n", encoding="ascii")
    development = tmp_path / "development"
    development.mkdir()
    (development / "run_config.json").write_text("{}\n", encoding="ascii")
    (development / "run_complete.json").write_text("{}\n", encoding="ascii")
    output = tmp_path / "prelock"
    index = SimpleNamespace(manifest_sha256="c" * 64)
    selected_update = config.validation_checkpoints[0]
    monkeypatch.setattr(
        prelock,
        "_validate_development_source",
        lambda *args: (index, {}, {"decision": "SCREEN_GO"}, selected_update),
    )
    monkeypatch.setattr(prelock, "build_fault_manifest", lambda index: object())
    fake_scaler = SimpleNamespace()
    monkeypatch.setattr(
        prelock,
        "prepare_case_training_data",
        lambda index, manifest, case: ([SimpleNamespace(case=case)], fake_scaler),
    )
    monkeypatch.setattr(
        prelock,
        "iter_role_variants",
        lambda *args, cases: iter([SimpleNamespace(case=cases[0])]),
    )
    trained: list[tuple[str, int]] = []

    def fake_train(index, *, case, model_seed, output_dir, **kwargs):
        trained.append((case, model_seed))
        unit = output_dir / case / f"seed{model_seed}"
        unit.mkdir(parents=True)
        return unit / "training_complete.json"

    monkeypatch.setattr(prelock, "train_case_seed", fake_train)
    monkeypatch.setattr(
        prelock,
        "validate_training_unit",
        lambda run_dir, **kwargs: (
            run_dir / "training_complete.json",
            {"case": kwargs["case"], "seed": kwargs["seed"]},
        ),
    )
    monkeypatch.setattr(
        prelock,
        "_gru_unit",
        lambda unit_dir, **kwargs: (
            SimpleNamespace(),
            SimpleNamespace(),
            pd.DataFrame({"selected": [True]}),
            {"case": kwargs["case"], "seed": kwargs["seed"]},
        ),
    )

    def fake_assemble(bundle, **kwargs):
        bundle.mkdir()
        copied_manifest = bundle / "manifest.json"
        copied_manifest.write_text("{}\n", encoding="ascii")
        return {
            "schema": "test",
            "stage": "prelock",
            "artifact_inventory": ["manifest.json"],
        }, copied_manifest

    monkeypatch.setattr(prelock, "_assemble_bundle", fake_assemble)
    monkeypatch.setattr(prelock, "validate_prelock_bundle", lambda *args: {})
    registry_path = prelock.run_prelock_preparation(
        manifest, development, output, device="cpu"
    )
    expected = {
        (case, seed)
        for case in prelock.CASES
        for seed in config.confirmatory_seeds[3:]
    }
    assert set(trained) == expected
    assert len(trained) == 6
    assert registry_path == output / prelock.REGISTRY_NAME
    assert (output / prelock.DIGEST_NAME).is_file()
    assert (output / prelock.COMPLETION_NAME).is_file()
    assert not prelock._staging_path(output).exists()
    completion = json.loads(
        (output / prelock.COMPLETION_NAME).read_text(encoding="ascii")
    )
    assert completion["locked_values_read"] is False
    assert completion["external_timestamp_required"] is True


def test_gru_partial_unit_fails_closed(tmp_path: Path):
    unit = tmp_path / "gru" / "case" / "seed1"
    unit.mkdir(parents=True)
    (unit / "training_log.csv").write_text("partial\n", encoding="ascii")
    with pytest.raises(ValueError, match="remove only this directory"):
        prelock._gru_unit(
            unit,
            case="case",
            seed=1,
            fit_variants=[],
            validation_variants=[],
            scalers=SimpleNamespace(),
            variants_by_case_role={},
            scalers_by_case={},
            config=StudyConfig(),
            device=prelock.torch.device("cpu"),
        )
