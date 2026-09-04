from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from building_fault_wm.neural_benchmark import fault_data
from building_fault_wm.neural_benchmark.fault_data import (
    ACTION_COLUMNS,
    CONTEXT_COLUMNS,
    FAULT_CHANNELS,
    FAMILIES,
    HEALTH_CLASS,
    NEXT_CONTEXT_COLUMNS,
    NEXT_OBSERVATION_COLUMNS,
    OBSERVATION_COLUMNS,
    SOURCE_ACTION_COLUMNS,
    FaultManifest,
    SequenceReference,
    apply_fault,
    build_fault_manifest,
    fault_cell_signatures,
    fit_scalers,
    load_clean_trajectory,
    load_corpus_index,
    load_role_trajectories,
    materialize_rssm_batch,
    validate_fault_manifest,
)
from building_fault_wm.neural_benchmark.protocol import (
    BOPTEST_API_VERSION,
    BOPTEST_COMMIT,
    BOPTEST_LICENSE_NAME,
    BOPTEST_LICENSE_PATH,
    BOPTEST_LICENSE_SHA256,
    BOPTEST_REPOSITORY_URL,
    CASES as PROTOCOL_CASES,
    CORPUS_MANIFEST_SCHEMA,
    FULL_PLAN_SCHEMA,
    PLAN_SEED,
    PRELOCK_REGISTRY_SCHEMA,
    SMOKE_PLAN_SCHEMA,
    STEP_SECONDS,
    TRAJECTORY_STEPS,
    WARMUP_SECONDS,
    WORKER_BOPTEST_VERSION,
    WORKER_IMAGE_ID,
    WORKER_RECEIPT_SCHEMA,
    attach_plan_sha256,
    canonical_json,
    collector_code_hashes,
    plan_sha256,
    sha256_file,
    smoke_view,
)
from building_fault_wm.neural_benchmark.worker_collect import FIELDS, ROW_SCHEMA


CASES = ("bestest_hydronic_heat_pump", "twozone_apartment_hydronic")
TEST_PRELOCK_REGISTRY = {
    "schema": PRELOCK_REGISTRY_SCHEMA,
    "stage": "prelock",
    "test_fixture": True,
}
TEST_PRELOCK_DIGEST = hashlib.sha256(
    canonical_json(TEST_PRELOCK_REGISTRY).encode("ascii")
).hexdigest()


def _observation(case_index: int, day: int, step: int) -> tuple[float, ...]:
    return (
        288.0 + case_index + 0.02 * day + 0.01 * step,
        800.0 + 100.0 * case_index + 3.0 * day + 2.5 * step,
        0.1 * case_index + 0.005 * step,
        1500.0 + 50.0 * case_index + 4.0 * step,
    )


def _context(case_index: int, day: int, step: int) -> tuple[float, ...]:
    return (
        275.0 + case_index + 0.03 * day + 0.02 * step,
        float((step % 24) * 20),
        293.15,
        297.15,
        0.2 + 0.01 * ((step // 8) % 3),
    )


def _write_trajectory(
    path: Path,
    *,
    case: str,
    case_index: int,
    role: str,
    day: int,
    seed: int,
    base_setpoint: float,
    amplitude: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(TRAJECTORY_STEPS):
        level = float((-1, 0, 1)[(step // 8) % 3])
        values = (
            case,
            role,
            day,
            seed,
            step,
            day * 86_400 + step * STEP_SECONDS,
            level,
            base_setpoint + amplitude * level,
            *_observation(case_index, day, step),
            *_observation(case_index, day, step + 1),
            *_context(case_index, day, step),
            *_context(case_index, day, step + 1),
        )
        rows.append(dict(zip(FIELDS, values)))
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def _write_manifest_wrapper(path: Path, payload: dict) -> None:
    _write_json(
        path,
        {
            "manifest_sha256": _canonical_sha256(payload),
            "manifest": payload,
        },
    )


def _read_manifest_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))["manifest"]


def _make_corpus(root: Path, kind: str = "development") -> Path:
    layout = {
        "smoke": ("smoke", ("fit",), "smoke_raw", SMOKE_PLAN_SCHEMA),
        "development": (
            "full",
            ("fit", "validation"),
            "development_raw",
            FULL_PLAN_SCHEMA,
        ),
        "locked_test": (
            "full",
            ("locked_test",),
            "locked_test_raw",
            FULL_PLAN_SCHEMA,
        ),
    }
    mode, selected_roles, raw_subdir, plan_schema = layout[kind]
    selected_cases = sorted(CASES)
    public_source = {
        "repository_url": BOPTEST_REPOSITORY_URL,
        "commit": BOPTEST_COMMIT,
        "license_name": BOPTEST_LICENSE_NAME,
        "license_path": BOPTEST_LICENSE_PATH,
        "license_sha256": BOPTEST_LICENSE_SHA256,
    }
    runtime = {
        "image_id": WORKER_IMAGE_ID,
        "boptest_version": WORKER_BOPTEST_VERSION,
    }
    code_hashes = collector_code_hashes()
    prelock_registry_sha256 = TEST_PRELOCK_DIGEST if kind == "locked_test" else None
    plan_paths: dict[str, Path] = {}
    plans: dict[str, dict] = {}
    files: list[dict] = []
    receipts: dict[str, dict] = {}
    for case_index, case in enumerate(CASES):
        adapter = PROTOCOL_CASES[case]
        base_setpoint = adapter.base_setpoint_k
        amplitude = adapter.action_amplitude_k
        entries = []
        roles = ("fit",) * 20 + ("validation",) * 8 + ("locked_test",) * 12
        for entry_index, role in enumerate(roles):
            day = 4 + 9 * entry_index
            seed = 1000 + 100 * case_index + entry_index
            entries.append(
                {
                    "case": case,
                    "day": day,
                    "role": role,
                    "temperature_stratum": entry_index // 8,
                    "mean_outdoor_temperature_k": 280.0 + 0.1 * entry_index,
                    "trajectory_seed": seed,
                }
            )
        full_plan = attach_plan_sha256(
            {
                "schema": FULL_PLAN_SCHEMA,
                "mode": "full",
                "boptest_commit": BOPTEST_COMMIT,
                "boptest_api_version": BOPTEST_API_VERSION,
                "public_source": public_source,
                "worker_runtime": runtime,
                "collector_code_sha256": code_hashes,
                "step_seconds": STEP_SECONDS,
                "warmup_seconds": WARMUP_SECONDS,
                "trajectory_steps": TRAJECTORY_STEPS,
                "plan_seed": PLAN_SEED,
                "case_adapter": asdict(adapter),
                "source_sha256": {
                    "wrapped_fmu": adapter.fmu_sha256,
                    "weather_csv": "b" * 64,
                },
                "entries": entries,
            }
        )
        plan = smoke_view(full_plan) if mode == "smoke" else full_plan
        plan_path = root / "plans" / mode / f"{case}.json"
        _write_json(plan_path, plan)
        plan_paths[case] = plan_path
        plans[case] = plan
        receipt_files = []
        for entry in plan["entries"]:
            if entry["role"] not in selected_roles:
                continue
            relative = (
                f"{case}/day{entry['day']:03d}_{entry['role']}_"
                f"seed{entry['trajectory_seed']}.csv"
            )
            path = root / raw_subdir / relative
            _write_trajectory(
                path,
                case=case,
                case_index=case_index,
                role=entry["role"],
                day=entry["day"],
                seed=entry["trajectory_seed"],
                base_setpoint=base_setpoint,
                amplitude=amplitude,
            )
            digest = sha256_file(path)
            files.append(
                {
                    "path": relative,
                    "case": case,
                    "role": entry["role"],
                    "day": entry["day"],
                    "trajectory_seed": entry["trajectory_seed"],
                    "plan_sha256": plan["plan_sha256"],
                    "sha256": digest,
                    "rows": TRAJECTORY_STEPS,
                    "fields": len(FIELDS),
                }
            )
            receipt_files.append(
                {"path": relative, "sha256": digest, "rows": TRAJECTORY_STEPS}
            )
        receipt_payload = {
            "schema": WORKER_RECEIPT_SCHEMA,
            "collection_kind": kind,
            "allowed_roles": sorted(selected_roles),
            "case": case,
            "plan_sha256": plan["plan_sha256"],
            "worker_image_id": WORKER_IMAGE_ID,
            "boptest_version": WORKER_BOPTEST_VERSION,
            "collector_code_sha256": code_hashes,
            "source_sha256": plan["source_sha256"],
            "prelock_registry_sha256": prelock_registry_sha256,
            "files": receipt_files,
        }
        receipt_wrapper = {
            "receipt_sha256": _canonical_sha256(receipt_payload),
            "receipt": receipt_payload,
        }
        receipt_relative = (
            f"_receipts/{case}_{kind}_{plan['plan_sha256'][:16]}.json"
        )
        receipt_path = root / raw_subdir / receipt_relative
        _write_json(receipt_path, receipt_wrapper)
        receipts[case] = {
            "path": receipt_relative,
            "sha256": sha256_file(receipt_path),
            "receipt_sha256": receipt_wrapper["receipt_sha256"],
        }
    files.sort(key=lambda item: item["path"])
    role_counts = Counter(item["role"] for item in files)
    payload = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "collection_kind": kind,
        "plan_mode": mode,
        "allowed_roles": sorted(selected_roles),
        "selected_cases": selected_cases,
        "public_source": public_source,
        "boptest_commit": BOPTEST_COMMIT,
        "worker_runtime": runtime,
        "collector_code_sha256": code_hashes,
        "prelock_registry_sha256": prelock_registry_sha256,
        "row_schema": {
            "name": ROW_SCHEMA,
            "fields": list(FIELDS),
            "step_seconds": STEP_SECONDS,
            "trajectory_steps": TRAJECTORY_STEPS,
            "transition_contract": "row r is (observation_r, context_r, action_r, observation_r+1, context_r+1)",
        },
        "plans": {
            case: {
                "path": str(plan_paths[case].relative_to(root)),
                "file_sha256": sha256_file(plan_paths[case]),
                "plan_sha256": plans[case]["plan_sha256"],
                "schema": plan_schema,
                "source_sha256": plans[case]["source_sha256"],
                "selected_entries": sum(
                    entry["role"] in selected_roles for entry in plans[case]["entries"]
                ),
            }
            for case in selected_cases
        },
        "counts": {
            "cases": len(selected_cases),
            "trajectories": len(files),
            "rows": len(files) * TRAJECTORY_STEPS,
            "roles": {role: role_counts[role] for role in sorted(role_counts)},
        },
        "receipts": receipts,
        "files": files,
    }
    suffix = "-".join(selected_cases)
    manifest_path = root / "manifests" / f"{kind}_{suffix}_corpus_manifest.json"
    _write_manifest_wrapper(manifest_path, payload)
    return manifest_path


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    return _make_corpus(tmp_path / "corpus")


@pytest.fixture
def locked_manifest(tmp_path: Path) -> Path:
    return _make_corpus(tmp_path / "locked_corpus", "locked_test")


@pytest.fixture
def smoke_manifest(tmp_path: Path) -> Path:
    return _make_corpus(tmp_path / "smoke_corpus", "smoke")


def _cell(manifest, trajectory, channel, family, *, sign=0, severity=None, onset=32):
    matches = [
        cell
        for cell in manifest.cells
        if cell.trajectory == trajectory.key
        and cell.fault_channel == channel
        and cell.family == family
        and cell.sign == sign
        and cell.onset == onset
        and (severity is None or cell.severity == severity)
    ]
    assert len(matches) == 1
    return matches[0]


def test_metadata_index_role_integrity_hashes_and_locked_value_boundary(corpus_root):
    index = load_corpus_index(corpus_root)
    assert index.manifest_path == corpus_root.resolve()
    assert index.root.name == "development_raw"
    assert index.collection_kind == "development"
    assert index.prelock_registry_sha256 is None
    assert index.allowed_roles == ("fit", "validation")
    assert len(index.records) == 56
    assert len(index.source_hashes) == 56
    assert Counter(record.key.role for record in index.records) == {
        "fit": 40,
        "validation": 16,
    }
    assert {record.key.case for record in index.records} == set(CASES)
    for record in index.records:
        assert sha256_file(index.root / record.relative_path) == record.source_sha256

    fit = load_role_trajectories(index, "fit")
    validation = load_role_trajectories(index, "validation")
    assert {item.key.role for item in fit} == {"fit"}
    assert {item.key.role for item in validation} == {"validation"}
    with pytest.raises(PermissionError, match="explicit authorization"):
        load_role_trajectories(index, "locked_test")
    with pytest.raises(ValueError, match="missing requested cases"):
        index.records_for("fit", [*CASES, "absent_case"])


def test_fit_scalers_use_clean_aligned_fit_rows_only(corpus_root):
    index = load_corpus_index(corpus_root)
    fit = load_role_trajectories(index, "fit")
    validation = load_role_trajectories(index, "validation")
    scalers = fit_scalers(fit)
    expected_observation = np.concatenate([item.observations[1:] for item in fit])
    expected_action = np.concatenate([item.actions[:-1] for item in fit])
    expected_context = np.concatenate([item.contexts[1:] for item in fit])
    np.testing.assert_allclose(scalers.observation.mean, expected_observation.mean(0))
    np.testing.assert_allclose(scalers.action.mean, expected_action.mean(0))
    np.testing.assert_allclose(scalers.context.mean, expected_context.mean(0))
    assert all(":fit:" in key for key, _ in scalers.fit_source_sha256)
    with pytest.raises(ValueError, match="FIT trajectories only"):
        fit_scalers([fit[0], validation[0]])
    with pytest.raises(ValueError, match="duplicate"):
        fit_scalers([fit[0], fit[0]])


def test_manifest_is_deterministic_fixed_complete_and_h8_active(corpus_root, tmp_path):
    index = load_corpus_index(corpus_root)
    first = build_fault_manifest(index)
    second = build_fault_manifest(index)
    assert first.payload() == second.payload()
    assert first.sha256 == second.sha256
    assert len(first.cells) == len(index.records) * len(FAULT_CHANNELS) * 22
    assert len({cell.cell_id for cell in first.cells}) == len(first.cells)
    validate_fault_manifest(first, index)
    for cell in first.cells:
        assert cell.stop - cell.onset == 48
        assert cell.onset in first.spec.onsets_for(cell.trajectory.role)
        assert cell.anchors == tuple(cell.onset + offset for offset in (8, 16, 24, 32))
        assert all(anchor + 8 < cell.stop for anchor in cell.anchors)
        assert cell.source_sha256 == index.source_hashes[cell.trajectory.text]
    counts = Counter(
        (cell.trajectory, cell.fault_channel, cell.family) for cell in first.cells
    )
    for record in index.records:
        for channel in FAULT_CHANNELS:
            assert {
                family: counts[(record.key, channel, family)] for family in FAMILIES
            } == {"healthy": 2, "bias": 8, "drift": 8, "stuck": 2, "dropout": 2}
    path = tmp_path / "fault_manifest.json"
    first.write(path)
    loaded = FaultManifest.read(path)
    assert loaded.payload() == first.payload()
    assert loaded.sha256 == first.sha256


def test_fault_formulas_labels_availability_ages_and_clean_targets(corpus_root):
    index = load_corpus_index(corpus_root)
    trajectory = load_role_trajectories(index, "fit")[0]
    manifest = build_fault_manifest(index)

    bias_cell = _cell(
        manifest,
        trajectory,
        "zone_temperature_k",
        "bias",
        sign=-1,
        severity=2.0,
    )
    bias = apply_fault(trajectory, bias_cell)
    np.testing.assert_allclose(
        bias.corrupted_observations[32:80, 0] - trajectory.observations[32:80, 0],
        -2.0,
    )
    assert bias.availability.all() and not bias.age.any()
    assert np.all(bias.health_labels[32:80, 0] == HEALTH_CLASS["bias"])
    assert not bias.health_labels[:, 1].any()

    drift_cell = _cell(
        manifest,
        trajectory,
        "hvac_electric_power_w",
        "drift",
        sign=1,
        severity=15.0,
    )
    drift = apply_fault(trajectory, drift_cell)
    np.testing.assert_allclose(
        drift.corrupted_observations[32:80, 1] - trajectory.observations[32:80, 1],
        15.0 * np.arange(1, 49),
    )
    assert np.all(drift.health_labels[32:80, 1] == HEALTH_CLASS["drift"])

    stuck_cell = _cell(
        manifest, trajectory, "zone_temperature_k", "stuck", onset=96
    )
    stuck = apply_fault(trajectory, stuck_cell)
    np.testing.assert_array_equal(
        stuck.corrupted_observations[96:144, 0],
        np.full(48, trajectory.observations[95, 0]),
    )
    assert np.all(stuck.health_labels[96:144, 0] == HEALTH_CLASS["stuck"])

    dropout_cell = _cell(
        manifest, trajectory, "hvac_electric_power_w", "dropout", onset=96
    )
    dropout = apply_fault(trajectory, dropout_cell)
    assert np.isnan(dropout.corrupted_observations[96:144, 1]).all()
    assert not dropout.availability[96:144, 1].any()
    np.testing.assert_array_equal(dropout.age[96:144, 1], np.arange(1, 49))
    assert dropout.availability[:96].all() and dropout.availability[144:].all()
    assert not dropout.age[:96].any() and not dropout.age[144:].any()
    assert np.all(dropout.health_labels[96:144, 1] == HEALTH_CLASS["dropout"])

    healthy_cell = _cell(
        manifest, trajectory, "zone_temperature_k", "healthy", onset=32
    )
    healthy = apply_fault(trajectory, healthy_cell)
    np.testing.assert_array_equal(healthy.corrupted_observations, trajectory.observations)
    assert not healthy.health_labels.any()
    for variant in (bias, drift, stuck, dropout, healthy):
        np.testing.assert_array_equal(variant.clean_observations, trajectory.observations)
        np.testing.assert_array_equal(variant.actions, trajectory.actions)
        np.testing.assert_array_equal(variant.contexts, trajectory.contexts)


def test_manifest_and_fit_preprocessing_never_load_locked_values(corpus_root, monkeypatch):
    index = load_corpus_index(corpus_root)
    original = fault_data.load_clean_trajectory
    loaded_roles = []

    def recording_loader(index_arg, record, **kwargs):
        loaded_roles.append(record.key.role)
        return original(index_arg, record, **kwargs)

    monkeypatch.setattr(fault_data, "load_clean_trajectory", recording_loader)
    manifest = build_fault_manifest(index)
    assert loaded_roles == []
    fit = load_role_trajectories(index, "fit")
    fit_scalers(fit)
    assert loaded_roles == ["fit"] * 40
    assert all(cell.trajectory.role in {"fit", "validation"} for cell in manifest.cells)


def test_locked_index_is_metadata_only_and_values_remain_guarded(
    locked_manifest, monkeypatch
):
    def forbidden_value_load(*args, **kwargs):
        raise AssertionError("locked trajectory values were parsed during metadata indexing")

    monkeypatch.setattr(pd, "read_csv", forbidden_value_load)
    with pytest.raises(ValueError, match="pre-lock registry"):
        load_corpus_index(locked_manifest)
    index = load_corpus_index(
        locked_manifest,
        prelock_registry=TEST_PRELOCK_REGISTRY,
        expected_prelock_sha256=TEST_PRELOCK_DIGEST,
    )
    assert index.root.name == "locked_test_raw"
    assert index.collection_kind == "locked_test"
    assert index.prelock_registry_sha256 == TEST_PRELOCK_DIGEST
    assert index.allowed_roles == ("locked_test",)
    assert len(index.records) == 24
    assert {record.key.role for record in index.records} == {"locked_test"}
    manifest = build_fault_manifest(index)
    assert manifest.cells
    with pytest.raises(PermissionError, match="explicit authorization"):
        load_role_trajectories(index, "locked_test")


def test_loader_enforces_prelock_digest_scope(tmp_path):
    development = _make_corpus(tmp_path / "development", "development")
    payload = _read_manifest_payload(development)
    payload["prelock_registry_sha256"] = "8" * 64
    _write_manifest_wrapper(development, payload)
    with pytest.raises(ValueError, match="non-locked corpus carries"):
        load_corpus_index(development)

    locked = _make_corpus(tmp_path / "locked", "locked_test")
    payload = _read_manifest_payload(locked)
    payload["prelock_registry_sha256"] = None
    _write_manifest_wrapper(locked, payload)
    with pytest.raises(ValueError, match="externally committed"):
        load_corpus_index(
            locked,
            prelock_registry=TEST_PRELOCK_REGISTRY,
            expected_prelock_sha256=TEST_PRELOCK_DIGEST,
        )


def test_role_specific_onsets_are_frozen_and_transfer_across_splits(
    corpus_root, locked_manifest
):
    development = load_corpus_index(corpus_root)
    locked = load_corpus_index(
        locked_manifest,
        prelock_registry=TEST_PRELOCK_REGISTRY,
        expected_prelock_sha256=TEST_PRELOCK_DIGEST,
    )
    development_faults = build_fault_manifest(development)
    locked_faults = build_fault_manifest(locked)
    spec = development_faults.spec
    assert spec.onsets_for("fit") == (32, 96)
    assert spec.onsets_for("validation") == (48, 112)
    assert spec.onsets_for("locked_test") == (64, 128)
    for role in ("fit", "validation", "locked_test"):
        signatures = fault_cell_signatures(spec, role)
        assert len(signatures) == 176
        assert {item[5] for item in signatures} == set(spec.onsets_for(role))
    assert {
        cell.onset for cell in development_faults.cells if cell.trajectory.role == "fit"
    } == {32, 96}
    assert {
        cell.onset
        for cell in development_faults.cells
        if cell.trajectory.role == "validation"
    } == {48, 112}
    assert {cell.onset for cell in locked_faults.cells} == {64, 128}


def test_smoke_manifest_maps_only_to_smoke_raw(smoke_manifest):
    index = load_corpus_index(smoke_manifest)
    assert index.root.name == "smoke_raw"
    assert index.collection_kind == "smoke"
    assert index.allowed_roles == ("fit",)
    assert len(index.records) == len(CASES)


def test_v2_wrapper_seal_counts_identity_and_disk_bijection_are_enforced(tmp_path):
    seal_path = _make_corpus(tmp_path / "seal")
    wrapper = json.loads(seal_path.read_text(encoding="ascii"))
    wrapper["manifest"]["counts"]["rows"] += 1
    _write_json(seal_path, wrapper)
    with pytest.raises(ValueError, match="self-hash"):
        load_corpus_index(seal_path)

    count_path = _make_corpus(tmp_path / "counts")
    payload = _read_manifest_payload(count_path)
    payload["counts"]["trajectories"] += 1
    _write_manifest_wrapper(count_path, payload)
    with pytest.raises(ValueError, match="counts mismatch"):
        load_corpus_index(count_path)

    identity_path = _make_corpus(tmp_path / "identity")
    payload = _read_manifest_payload(identity_path)
    payload["files"][0]["role"] = "validation"
    _write_manifest_wrapper(identity_path, payload)
    with pytest.raises(ValueError, match="identity metadata mismatch"):
        load_corpus_index(identity_path)

    disk_path = _make_corpus(tmp_path / "disk")
    payload = _read_manifest_payload(disk_path)
    stale = (
        disk_path.parents[1]
        / "development_raw"
        / payload["selected_cases"][0]
        / "unmanifested.csv"
    )
    stale.write_text("stale\n", encoding="ascii")
    with pytest.raises(ValueError, match="unmanifested or missing CSV"):
        load_corpus_index(disk_path)


def test_manifest_plan_and_receipt_reject_duplicate_json_keys(tmp_path):
    manifest_path = _make_corpus(tmp_path / "manifest_duplicate")
    original = manifest_path.read_text(encoding="ascii").lstrip()
    manifest_path.write_text(
        '{"manifest_sha256":"0",' + original[1:],
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_corpus_index(manifest_path)

    plan_manifest = _make_corpus(tmp_path / "plan_duplicate")
    payload = _read_manifest_payload(plan_manifest)
    case = CASES[0]
    plan_path = plan_manifest.parents[1] / payload["plans"][case]["path"]
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    plan_original = plan_path.read_text(encoding="ascii").lstrip()
    plan_path.write_text(
        json.dumps({"schema": plan["schema"]})[:-1] + "," + plan_original[1:],
        encoding="ascii",
    )
    payload["plans"][case]["file_sha256"] = sha256_file(plan_path)
    _write_manifest_wrapper(plan_manifest, payload)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_corpus_index(plan_manifest)

    receipt_manifest = _make_corpus(tmp_path / "receipt_duplicate")
    payload = _read_manifest_payload(receipt_manifest)
    case = CASES[0]
    receipt_path = (
        receipt_manifest.parents[1]
        / "development_raw"
        / payload["receipts"][case]["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt_original = receipt_path.read_text(encoding="ascii").lstrip()
    receipt_path.write_text(
        json.dumps({"receipt_sha256": receipt["receipt_sha256"]})[:-1]
        + ","
        + receipt_original[1:],
        encoding="ascii",
    )
    payload["receipts"][case]["sha256"] = sha256_file(receipt_path)
    _write_manifest_wrapper(receipt_manifest, payload)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_corpus_index(receipt_manifest)


def test_plan_file_and_canonical_hashes_are_independently_enforced(tmp_path):
    file_hash_path = _make_corpus(tmp_path / "plan_file")
    payload = _read_manifest_payload(file_hash_path)
    case = CASES[0]
    plan_path = file_hash_path.parents[1] / payload["plans"][case]["path"]
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="plan file SHA-256 mismatch"):
        load_corpus_index(file_hash_path)

    canonical_path = _make_corpus(tmp_path / "plan_canonical")
    payload = _read_manifest_payload(canonical_path)
    plan_path = canonical_path.parents[1] / payload["plans"][case]["path"]
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    plan["step_seconds"] = STEP_SECONDS + 1
    _write_json(plan_path, plan)
    payload["plans"][case]["file_sha256"] = sha256_file(plan_path)
    _write_manifest_wrapper(canonical_path, payload)
    with pytest.raises(ValueError, match="step cadence|canonical plan"):
        load_corpus_index(canonical_path)


def test_loader_requires_an_explicit_manifest_file(corpus_root, tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest is missing"):
        load_corpus_index(corpus_root.parents[1])
    copied = tmp_path / corpus_root.name
    copied.write_bytes(corpus_root.read_bytes())
    with pytest.raises(ValueError, match="manifests directory"):
        load_corpus_index(copied)

    legacy = _make_corpus(tmp_path / "legacy")
    payload = _read_manifest_payload(legacy)
    payload["schema"] = "boptest-multicase-clean-corpus-v1"
    _write_manifest_wrapper(legacy, payload)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_corpus_index(legacy)


def test_time_major_rssm_alignment_and_batch_boundaries(corpus_root):
    index = load_corpus_index(corpus_root)
    fit = load_role_trajectories(index, "fit")
    scalers = fit_scalers(fit)
    manifest = build_fault_manifest(index)
    first_cell = _cell(
        manifest, fit[0], "zone_temperature_k", "bias", sign=1, severity=1.0
    )
    second_cell = _cell(
        manifest, fit[1], "hvac_electric_power_w", "healthy"
    )
    variants = [apply_fault(fit[0], first_cell), apply_fault(fit[1], second_cell)]
    batch = materialize_rssm_batch(
        variants,
        scalers,
        [SequenceReference(0, 47), SequenceReference(1, 7)],
        length=4,
    )
    assert batch.previous_actions.shape == (4, 2, len(ACTION_COLUMNS))
    assert batch.corrupted_observations.shape == (4, 2, len(OBSERVATION_COLUMNS))
    assert batch.availability.shape == batch.corrupted_observations.shape
    assert batch.age.shape == batch.corrupted_observations.shape
    assert batch.contexts.shape == (4, 2, len(CONTEXT_COLUMNS))
    assert batch.clean_targets.shape == batch.corrupted_observations.shape
    assert batch.health_labels.shape == (4, 2, len(FAULT_CHANNELS))
    np.testing.assert_array_equal(batch.source_rows[:, 0], np.arange(48, 52))
    np.testing.assert_array_equal(batch.source_rows[:, 1], np.arange(8, 12))
    np.testing.assert_allclose(
        batch.previous_actions[:, 0],
        scalers.action.transform(variants[0].actions[47:51]),
    )
    np.testing.assert_allclose(
        batch.corrupted_observations[:, 0],
        scalers.observation.transform(variants[0].corrupted_observations[48:52]),
    )
    np.testing.assert_allclose(
        batch.contexts[:, 0],
        scalers.context.transform(variants[0].contexts[48:52]),
    )
    np.testing.assert_allclose(
        batch.clean_targets[:, 0],
        scalers.observation.transform(variants[0].clean_observations[48:52]),
    )
    with pytest.raises(ValueError, match="crosses a whole trajectory"):
        materialize_rssm_batch(
            variants,
            scalers,
            [SequenceReference(0, TRAJECTORY_STEPS - 2)],
            length=2,
        )


def test_source_tampering_duplicate_manifest_entries_and_incomplete_rows_rejected(
    corpus_root,
):
    index = load_corpus_index(corpus_root)
    fit_record = index.records_for("fit")[0]
    fit_path = index.root / fit_record.relative_path
    fit_path.write_bytes(fit_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after indexing"):
        load_clean_trajectory(index, fit_record)

    duplicate_root = _make_corpus(corpus_root.parents[2] / "duplicate")
    payload = _read_manifest_payload(duplicate_root)
    payload["files"].append(dict(payload["files"][0]))
    _write_manifest_wrapper(duplicate_root, payload)
    with pytest.raises(ValueError, match="duplicate source path"):
        load_corpus_index(duplicate_root)

    incomplete_root = _make_corpus(corpus_root.parents[2] / "incomplete")
    payload = _read_manifest_payload(incomplete_root)
    source = incomplete_root.parents[1] / "development_raw" / payload["files"][0]["path"]
    lines = source.read_text(encoding="ascii").splitlines()
    source.write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")
    payload["files"][0]["sha256"] = sha256_file(source)
    payload["files"][0]["rows"] = TRAJECTORY_STEPS - 1
    _write_manifest_wrapper(incomplete_root, payload)
    with pytest.raises(ValueError, match="row count mismatch"):
        load_corpus_index(incomplete_root)


def test_duplicate_steps_are_rejected_after_valid_hash_indexing(corpus_root):
    payload = _read_manifest_payload(corpus_root)
    metadata = next(
        item for item in payload["files"] if "_fit_" in item["path"]
    )
    path = corpus_root.parents[1] / "development_raw" / metadata["path"]
    frame = pd.read_csv(path)
    frame.loc[1, "step"] = 0
    frame.to_csv(path, index=False)
    metadata["sha256"] = sha256_file(path)
    receipt_metadata = payload["receipts"][metadata["case"]]
    receipt_path = corpus_root.parents[1] / "development_raw" / receipt_metadata["path"]
    receipt_wrapper = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt_file = next(
        item
        for item in receipt_wrapper["receipt"]["files"]
        if item["path"] == metadata["path"]
    )
    receipt_file["sha256"] = metadata["sha256"]
    receipt_wrapper["receipt_sha256"] = _canonical_sha256(receipt_wrapper["receipt"])
    _write_json(receipt_path, receipt_wrapper)
    receipt_metadata["sha256"] = sha256_file(receipt_path)
    receipt_metadata["receipt_sha256"] = receipt_wrapper["receipt_sha256"]
    _write_manifest_wrapper(corpus_root, payload)
    index = load_corpus_index(corpus_root)
    record = next(item for item in index.records if item.relative_path == metadata["path"])
    with pytest.raises(ValueError, match="duplicated or incomplete"):
        load_clean_trajectory(index, record)


def test_same_case_day_cannot_cross_roles(corpus_root):
    manifest = _read_manifest_payload(corpus_root)
    case = CASES[0]
    plan_metadata = manifest["plans"][case]
    plan_path = corpus_root.parents[1] / plan_metadata["path"]
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    plan["entries"][1]["day"] = plan["entries"][0]["day"]
    plan["plan_sha256"] = plan_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    _write_json(plan_path, plan)
    plan_metadata["file_sha256"] = sha256_file(plan_path)
    plan_metadata["plan_sha256"] = plan["plan_sha256"]
    _write_manifest_wrapper(corpus_root, manifest)
    with pytest.raises(ValueError, match="assigned to multiple.*roles"):
        load_corpus_index(corpus_root)


def test_declared_clean_and_next_schema_are_complete():
    assert tuple(FIELDS[8:12]) == OBSERVATION_COLUMNS
    assert tuple(FIELDS[12:16]) == NEXT_OBSERVATION_COLUMNS
    assert tuple(FIELDS[6:8]) == SOURCE_ACTION_COLUMNS
    assert ACTION_COLUMNS == ("normalized_action",)
    assert tuple(FIELDS[16:21]) == CONTEXT_COLUMNS
    assert tuple(FIELDS[21:26]) == NEXT_CONTEXT_COLUMNS
