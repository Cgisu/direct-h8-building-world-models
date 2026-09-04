from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark.runtime_provenance import (
    numerical_runtime_fingerprint,
)

from . import plan, prelock, train_grid
from .config import FROZEN_CONFIG


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _make_parent(root: Path) -> tuple[str, dict]:
    runtime = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    run_config = {"numerical_runtime": runtime}
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "evidence/locked_fault_manifest.json").write_bytes(
        b"locked-fault-manifest\n"
    )
    _write_json(
        root / "experiment/prelock_bundle/development/source/run_config.json",
        run_config,
    )
    prefix = root / "experiment/prelock_bundle/frozen"
    (prefix / "frozen_fault_contract.json").parent.mkdir(parents=True, exist_ok=True)
    (prefix / "frozen_fault_contract.json").write_bytes(b"fault-contract\n")
    for case in sorted(boptest.CASES):
        (prefix / "fit_scalers" / f"{case}.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (prefix / "fit_scalers" / f"{case}.json").write_bytes(
            f"scaler:{case}\n".encode("ascii")
        )
        for seed in FROZEN_CONFIG.paired_model_seeds:
            schedule = prefix / "schedules" / case / f"seed{seed}.json"
            schedule.parent.mkdir(parents=True, exist_ok=True)
            schedule.write_bytes(f"schedule:{case}:{seed}\n".encode("ascii"))
            checkpoint_root = prefix / "checkpoints" / case / f"seed{seed}"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            for arm in ("legacy", "ungated_h8"):
                (checkpoint_root / f"{arm}_u0400.pt").write_bytes(
                    f"{arm}:{case}:{seed}\n".encode("ascii")
                )
    rows = prelock._inventory(root, "fixture parent")
    manifest = {
        "schema": "direct-h8-publication-package-manifest-v1",
        "artifact_inventory_excludes_manifest_and_digest": rows,
    }
    digest = prelock._parent_manifest_sha256(manifest)
    _write_json(root / "package_manifest.json", manifest)
    (root / "package_manifest.canonical.sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    summary = prelock.validate_parent_package(root, digest)
    return digest, summary


def _make_training(
    root: Path,
    parent_root: Path,
    parent_summary: dict,
    source_root: Path,
    shared_source_root: Path,
) -> None:
    config = json.loads(json.dumps(FROZEN_CONFIG.to_dict()))
    source = prelock._source_manifest(source_root)
    shared_source = train_grid.shared_runtime_source_manifest(
        shared_source_root
    )
    runs = []
    for case in sorted(boptest.CASES):
        for seed in FROZEN_CONFIG.paired_model_seeds:
            run_root = root / case / f"seed{seed}"
            checkpoint_hashes = {}
            for update in FROZEN_CONFIG.checkpoint_updates:
                name = f"update_{update:04d}.pt"
                path = run_root / "checkpoints" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"v3:{case}:{seed}:{update}\n".encode("ascii"))
                checkpoint_hashes[name] = prelock.sha256_file(path)
            receipt = {
                "schema": "boptest-deterministic-transport-training-v1",
                "model_seed": seed,
                "updates": FROZEN_CONFIG.updates,
                "checkpoint_updates": list(FROZEN_CONFIG.checkpoint_updates),
                "selected_update": FROZEN_CONFIG.updates,
                "selection_rule": "fixed_final_update_no_validation_selection",
                "schedule_sha256": "0" * 64,
                "final_model_state_sha256": "1" * 64,
                "checkpoint_file_sha256": checkpoint_hashes,
                "config": config,
                "training_log": [],
            }
            receipt_path = run_root / "training_receipt.json"
            _write_json(receipt_path, receipt)
            schedule = (
                parent_root
                / "experiment/prelock_bundle/frozen/schedules"
                / case
                / f"seed{seed}.json"
            )
            runs.append(
                {
                    "case": case,
                    "model_seed": seed,
                    "status": "trained",
                    "wall_seconds": 1.0,
                    "schedule_file_sha256": prelock.sha256_file(schedule),
                    "schedule_payload_sha256": "2" * 64,
                    "final_model_state_sha256": "1" * 64,
                    "selected_checkpoint_file_sha256": checkpoint_hashes[
                        "update_0400.pt"
                    ],
                    "training_receipt_sha256": prelock.sha256_file(receipt_path),
                }
            )
    grid = {
        "schema": train_grid.GRID_SCHEMA,
        "complete_grid": True,
        "parent_package": {
            key: parent_summary[key]
            for key in ("canonical_digest", "inventory_file_count", "inventory_bytes")
        },
        "development_manifest_sha256": "3" * 64,
        "fault_manifest_sha256": "4" * 64,
        "source_code_sha256": {
            name: source[name]
            for name in (
                "PROTOCOL.md",
                "__init__.py",
                "config.py",
                "model.py",
                "train.py",
                "train_grid.py",
            )
        },
        "shared_runtime_source_manifest": shared_source,
        "config": config,
        "runs": runs,
    }
    source_lock = {
        "schema": train_grid.SOURCE_LOCK_SCHEMA,
        "parent_package": grid["parent_package"],
        "v3_training_source_manifest": grid["source_code_sha256"],
        "shared_runtime_source_manifest": shared_source,
        "config": config,
    }
    _write_json(root / train_grid.SOURCE_LOCK_NAME, source_lock)
    grid["training_source_lock_file_sha256"] = prelock.sha256_file(
        root / train_grid.SOURCE_LOCK_NAME
    )
    _write_json(root / "training_grid_complete.json", grid)
    _write_json(root / "training_subset_fixture.json", {"complete_grid": False})


def _parent_plan(case: str) -> dict:
    entries = []
    for index in range(28):
        entries.append(
            {
                "case": case,
                "day": 100 + 4 * index,
                "role": "fit" if index % 2 == 0 else "validation",
                "trajectory_seed": 1000 + index,
                "temperature_stratum": index % 5,
                "mean_outdoor_temperature_k": 270.0 + index,
            }
        )
    entries.append(
        {
            "case": case,
            "day": 10,
            "role": "locked_test",
            "trajectory_seed": 9,
            "temperature_stratum": 0,
            "mean_outdoor_temperature_k": 275.0,
        }
    )
    return {
        "case_adapter": asdict(boptest.CASES[case]),
        "entries": entries,
        "plan_sha256": prelock.canonical_sha256({"case": case, "parent": "v2"}),
        "source_sha256": {
            "weather_csv": prelock.canonical_sha256({"case": case, "weather": 1}),
            "wrapped_fmu": boptest.CASES[case].fmu_sha256,
        },
    }


def _make_plans(root: Path) -> tuple[Path, Path]:
    plan_root = root / "plans/full"
    v1_plans = {}
    v2_plans = {}
    v3_plans = {}
    v2_identities = []
    for case in sorted(boptest.CASES):
        parent = _parent_plan(case)
        v2_plans[case] = parent
        v3_plans[case] = plan.build_case_plan(parent)
        v1_plans[case] = {
            "plan_sha256": prelock.canonical_sha256({"case": case, "parent": "v1"}),
            "entries": [
                {
                    "case": case,
                    "day": 1,
                    "role": "fit",
                    "trajectory_seed": 1,
                }
            ],
        }
        v2_identities.append(
            {"case": case, "day": 10, "trajectory_seed": 9}
        )
        _write_json(plan_root / f"{case}.json", v3_plans[case])
    prior_evidence = plan.build_prior_evidence_contract(
        [
            {
                "label": "fixture",
                "kind": "local_multicase_namespace",
                "present": True,
                "root_binding_sha256": None,
            }
        ],
        [
            {
                "source": "fixture",
                "path": "v2_manifest.json",
                "kind": "manifest_json",
                "sha256": "a" * 64,
                "bytes": 1,
                "identities": v2_identities,
            }
        ],
        v2_locked_csv_identities=v2_identities,
    )
    certificate = plan.build_disjointness_certificate(
        v1_plans,
        v2_plans,
        v3_plans,
        prior_evidence=prior_evidence,
    )
    certificate_path = root / "disjointness_certificate.json"
    _write_json(certificate_path, certificate)
    return plan_root, certificate_path


def _fixture(tmp_path: Path) -> dict:
    source_root = tmp_path / "source"
    source_root.mkdir()
    for path in prelock.HERE.iterdir():
        if path.is_file() and path.suffix in {".py", ".md"}:
            shutil.copyfile(path, source_root / path.name)
    shared_source_root = tmp_path / "shared-source"
    for relative in train_grid.SHARED_RUNTIME_SOURCE_RELATIVE_PATHS:
        source = prelock.DEFAULT_SHARED_SOURCE_ROOT / relative
        destination = shared_source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    parent = tmp_path / "parent"
    digest, parent_summary = _make_parent(parent)
    training = tmp_path / "training"
    _make_training(
        training,
        parent,
        parent_summary,
        source_root,
        shared_source_root,
    )
    plan_root, certificate = _make_plans(tmp_path / "data")
    return {
        "parent_root": parent,
        "expected_parent_digest": digest,
        "training_root": training,
        "shared_source_root": shared_source_root,
        "source_root": source_root,
        "plan_root": plan_root,
        "certificate_path": certificate,
        "testcase_root": tmp_path / "unused-testcases",
        "forbidden_response_paths": (tmp_path / "responses",),
        "validate_environment": False,
        "runtime_payload": numerical_runtime_fingerprint(
            "cpu", include_sklearn=True
        ),
    }


def test_prepare_and_validate_are_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    kwargs = _fixture(tmp_path)
    first = tmp_path / "prelock-a"
    second = tmp_path / "prelock-b"
    prelock.prepare_prelock(output_dir=first, **kwargs)
    prelock.prepare_prelock(output_dir=second, **kwargs)
    assert (first / prelock.REGISTRY_NAME).read_bytes() == (
        second / prelock.REGISTRY_NAME
    ).read_bytes()
    registry = prelock.validate_prelock_bundle(
        first,
        parent_root=kwargs["parent_root"],
        expected_parent_digest=kwargs["expected_parent_digest"],
        source_root=kwargs["source_root"],
        shared_source_root=kwargs["shared_source_root"],
    )
    paths = {row["path"] for row in registry["bundle_inventory"]}
    assert "v3_source/evaluate.py" in paths
    assert "v3_source/gate.py" in paths
    assert "v3_source/test_prelock.py" in paths
    assert "shared_runtime_source/health_rssm/model.py" in paths
    assert (
        "shared_runtime_source/multicase_fault_benchmark/study_train.py"
        in paths
    )
    assert "training/training_grid_complete.json" in paths
    assert (
        "parent_selected/experiment/prelock_bundle/frozen/"
        "frozen_fault_contract.json"
    ) in paths
    assert registry["training"]["unit_count"] == 15
    assert registry["training"]["checkpoint_count"] == 60
    assert registry["plans"]["branch_count"] == 72
    assert registry["locked_response_values_read"] is False
    checked = registry["locked_response_artifact_paths_checked"]
    assert checked == ["$EXTERNAL/responses"]
    assert not any(value.startswith("/") for value in checked)


def test_prepare_fails_closed_when_checkpoint_is_missing(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    missing = (
        kwargs["training_root"]
        / "bestest_hydronic_heat_pump"
        / "seed202608011/checkpoints/update_0400.pt"
    )
    missing.unlink()
    with pytest.raises(ValueError, match="not a plain file"):
        prelock.prepare_prelock(output_dir=tmp_path / "prelock", **kwargs)


def test_prepare_refuses_existing_response_artifact(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    response = kwargs["forbidden_response_paths"][0]
    response.write_bytes(b"must not be opened")
    with pytest.raises(FileExistsError, match="response artifact exists"):
        prelock.prepare_prelock(output_dir=tmp_path / "prelock", **kwargs)


def test_verifier_detects_bundle_tampering(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    output = tmp_path / "prelock"
    prelock.prepare_prelock(output_dir=output, **kwargs)
    target = output / "bundle/environment/pins.json"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bundle inventory changed"):
        prelock.validate_prelock_bundle(
            output,
            parent_root=kwargs["parent_root"],
            expected_parent_digest=kwargs["expected_parent_digest"],
        )


def test_verifier_detects_live_shared_source_change(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    output = tmp_path / "prelock"
    prelock.prepare_prelock(output_dir=output, **kwargs)
    target = kwargs["shared_source_root"] / "health_rssm/training.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="live shared runtime source"):
        prelock.validate_prelock_bundle(
            output,
            parent_root=kwargs["parent_root"],
            expected_parent_digest=kwargs["expected_parent_digest"],
            source_root=kwargs["source_root"],
            shared_source_root=kwargs["shared_source_root"],
        )
