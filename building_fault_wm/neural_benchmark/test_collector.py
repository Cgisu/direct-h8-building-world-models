from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from . import protocol as protocol_module
from .collect import (
    DEVELOPMENT_RAW_SUBDIR,
    DEVELOPMENT_ROLES,
    LOCKED_CONFIRMATION,
    LOCKED_RAW_SUBDIR,
    LOCKED_ROLES,
    SMOKE_RAW_SUBDIR,
    SMOKE_ROLES,
    _docker_command,
    _raw_subdir,
    build_corpus_manifest,
    materialize_plans,
    preflight_locked_destination,
    publish_locked_collection,
    validate_collection_request,
    validate_locked_plan_binding,
    validate_prelock_registry,
    write_corpus_manifest,
    write_json_immutable,
)
from .protocol import (
    BOPTEST_API_VERSION,
    BOPTEST_COMMIT,
    CASES,
    PRELOCK_REGISTRY_SCHEMA,
    FULL_PLAN_SCHEMA,
    START_DAYS,
    STEP_SECONDS,
    TRAJECTORY_STEPS,
    WORKER_BOPTEST_VERSION,
    WORKER_IMAGE_ID,
    action_payload,
    attach_plan_sha256,
    balanced_action_levels,
    build_case_plan,
    canonical_json,
    plan_sha256,
    smoke_view,
    validate_plan,
)
from .worker_collect import (
    FIELDS,
    canonical_observation,
    collect_plan,
    expected_filename,
    receipt_path,
)
from .fault_data import load_corpus_index


TESTCASE_ROOT = (Path.home() / "external/project1-boptest/testcases")


@pytest.fixture(scope="module")
def full_plans() -> dict[str, dict]:
    return {
        case: attach_plan_sha256(build_case_plan(adapter, TESTCASE_ROOT))
        for case, adapter in CASES.items()
    }


@pytest.fixture(scope="module")
def bestest_full(full_plans: dict[str, dict]) -> dict:
    return full_plans["bestest_hydronic_heat_pump"]


@pytest.fixture(scope="module")
def bestest_smoke(bestest_full: dict) -> dict:
    return smoke_view(bestest_full)


def _test_prelock_registry(value: int = 1) -> tuple[dict, str]:
    registry = {
        "schema": PRELOCK_REGISTRY_SCHEMA,
        "stage": "prelock",
        "value": value,
    }
    digest = hashlib.sha256(canonical_json(registry).encode("ascii")).hexdigest()
    return registry, digest


def test_balanced_action_schedule_is_deterministic_and_exciting():
    left = balanced_action_levels(1234)
    right = balanced_action_levels(1234)
    np.testing.assert_array_equal(left, right)
    assert len(left) == TRAJECTORY_STEPS
    blocks = left.reshape(-1, 8)[:, 0]
    assert Counter(blocks) == {-1.0: 8, 0.0: 8, 1.0: 8}
    assert np.all(blocks[1:] != blocks[:-1])


def test_balanced_actions_cover_every_frozen_trajectory_seed(full_plans):
    for plan in full_plans.values():
        for entry in plan["entries"]:
            levels = balanced_action_levels(entry["trajectory_seed"])
            blocks = levels.reshape(-1, 8)[:, 0]
            assert Counter(blocks) == {-1.0: 8, 0.0: 8, 1.0: 8}
            assert np.all(blocks[1:] != blocks[:-1])


def test_actions_are_broadcast_and_bounded():
    for adapter in CASES.values():
        low, low_payload = action_payload(adapter, -1.0)
        high, high_payload = action_payload(adapter, 1.0)
        assert high - low == 2.0 * adapter.action_amplitude_k
        assert len(low_payload) == 2 * len(adapter.action_pairs)
        assert len(high_payload) == 2 * len(adapter.action_pairs)
        for activate, value in adapter.action_pairs:
            assert low_payload[activate] == high_payload[activate] == 1.0
            assert low_payload[value] == low
            assert high_payload[value] == high


def test_case_plans_have_frozen_nonoverlapping_weather_stratified_roles(full_plans):
    for plan in full_plans.values():
        assert plan["schema"] == FULL_PLAN_SCHEMA
        assert plan["boptest_commit"] == BOPTEST_COMMIT
        assert plan["boptest_api_version"] == BOPTEST_API_VERSION
        assert plan_sha256(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        ) == plan["plan_sha256"]
        entries = plan["entries"]
        assert len(entries) == 40
        assert sorted(entry["day"] for entry in entries) == list(START_DAYS)
        assert min(np.diff(sorted(entry["day"] for entry in entries))) >= 9
        assert Counter(entry["role"] for entry in entries) == {
            "fit": 20,
            "validation": 8,
            "locked_test": 12,
        }
        expected_by_stratum = {
            0: {"fit": 4, "validation": 2, "locked_test": 2},
            1: {"fit": 4, "validation": 2, "locked_test": 2},
            2: {"fit": 4, "validation": 2, "locked_test": 2},
            3: {"fit": 4, "validation": 1, "locked_test": 3},
            4: {"fit": 4, "validation": 1, "locked_test": 3},
        }
        for stratum, expected in expected_by_stratum.items():
            roles = Counter(
                entry["role"]
                for entry in entries
                if entry["temperature_stratum"] == stratum
            )
            assert roles == expected


def test_canonical_observation_reductions_are_exact():
    for adapter in CASES.values():
        keys = sorted(
            {
                *adapter.zone_keys,
                *adapter.power_keys,
                *adapter.auxiliary_1_keys,
                *adapter.auxiliary_2_keys,
            }
        )
        state = {key: float(index + 1) for index, key in enumerate(keys)}

        def reduce(names, operation):
            values = np.asarray([state[name] for name in names])
            return float(values.mean() if operation == "mean" else values.sum())

        expected = (
            reduce(adapter.zone_keys, "mean"),
            reduce(adapter.power_keys, "sum"),
            reduce(adapter.auxiliary_1_keys, adapter.auxiliary_1_reduction),
            reduce(adapter.auxiliary_2_keys, adapter.auxiliary_2_reduction),
        )
        assert canonical_observation(adapter, state) == expected


def _patch_plan_builders(monkeypatch, frozen_plans: dict[str, dict]) -> None:
    payloads = {
        case: {key: value for key, value in plan.items() if key != "plan_sha256"}
        for case, plan in frozen_plans.items()
    }

    def fake_build(adapter, testcase_root, *, plan_seed=protocol_module.PLAN_SEED):
        return copy.deepcopy(payloads[adapter.case])

    monkeypatch.setattr(protocol_module, "build_case_plan", fake_build)
    monkeypatch.setattr(
        "building_fault_wm.neural_benchmark.collect.build_case_plan",
        fake_build,
    )


def _patch_plan_builder(monkeypatch, frozen_plan: dict) -> None:
    _patch_plan_builders(
        monkeypatch,
        {frozen_plan["case_adapter"]["case"]: frozen_plan},
    )


def test_full_and_smoke_plans_coexist_and_full_bytes_never_change(
    tmp_path, monkeypatch, bestest_full
):
    _patch_plan_builder(monkeypatch, bestest_full)
    case = bestest_full["case_adapter"]["case"]
    full_path = materialize_plans(TESTCASE_ROOT, tmp_path, "full", (case,))[0]
    full_bytes = full_path.read_bytes()
    smoke_path = materialize_plans(TESTCASE_ROOT, tmp_path, "smoke", (case,))[0]
    assert full_path != smoke_path
    assert full_path.read_bytes() == full_bytes
    assert len(json.loads(full_path.read_text())["entries"]) == 40
    smoke = json.loads(smoke_path.read_text())
    assert len(smoke["entries"]) == 1
    assert smoke["entries"][0]["role"] == "fit"
    materialize_plans(TESTCASE_ROOT, tmp_path, "full", (case,))
    materialize_plans(TESTCASE_ROOT, tmp_path, "smoke", (case,))
    assert full_path.read_bytes() == full_bytes
    assert not (tmp_path / "plans/full/twozone_apartment_hydronic.json").exists()


def test_immutable_json_refuses_hash_changing_overwrite(tmp_path):
    path = tmp_path / "frozen.json"
    write_json_immutable(path, {"value": 1})
    original = path.read_bytes()
    write_json_immutable(path, {"value": 1})
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError, match="hash-changing"):
        write_json_immutable(path, {"value": 2})
    assert path.read_bytes() == original


def test_plan_validation_rejects_resealed_semantic_tampering(
    monkeypatch, bestest_full
):
    _patch_plan_builder(monkeypatch, bestest_full)
    mutations = (
        lambda plan: plan.update(schema="wrong"),
        lambda plan: plan.update(boptest_commit="0" * 40),
        lambda plan: plan.update(boptest_api_version="wrong"),
        lambda plan: plan.update(step_seconds=300),
        lambda plan: plan.update(warmup_seconds=0),
        lambda plan: plan.update(trajectory_steps=10),
        lambda plan: plan.update(plan_seed=1),
        lambda plan: plan["case_adapter"].update(base_setpoint_k=1.0),
        lambda plan: plan["source_sha256"].update(weather_csv="0" * 64),
        lambda plan: plan["entries"][0].update(trajectory_seed=1),
        lambda plan: plan["entries"].append(copy.deepcopy(plan["entries"][0])),
    )
    for mutate in mutations:
        changed = copy.deepcopy(bestest_full)
        changed.pop("plan_sha256")
        mutate(changed)
        changed = attach_plan_sha256(changed)
        with pytest.raises(ValueError):
            validate_plan(changed, TESTCASE_ROOT, DEVELOPMENT_ROLES)


def test_plan_validation_checks_mounted_source_hashes(monkeypatch, bestest_full):
    original = protocol_module.sha256_file

    def changed_fmu(path):
        if Path(path).name == "wrapped.fmu":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(protocol_module, "sha256_file", changed_fmu)
    with pytest.raises(ValueError, match="FMU hash mismatch"):
        validate_plan(bestest_full, TESTCASE_ROOT, DEVELOPMENT_ROLES)


def test_role_routing_is_disjoint_and_locked_is_explicit(bestest_full, bestest_smoke):
    _, development = validate_plan(bestest_full, TESTCASE_ROOT, DEVELOPMENT_ROLES)
    _, locked = validate_plan(bestest_full, TESTCASE_ROOT, LOCKED_ROLES)
    _, smoke = validate_plan(bestest_smoke, TESTCASE_ROOT, SMOKE_ROLES)
    assert {entry["role"] for entry in development} == {"fit", "validation"}
    assert {entry["role"] for entry in locked} == {"locked_test"}
    assert {entry["role"] for entry in smoke} == {"fit"}
    assert _raw_subdir("collect", "smoke") == (SMOKE_RAW_SUBDIR, SMOKE_ROLES)
    assert _raw_subdir("collect", "full") == (
        DEVELOPMENT_RAW_SUBDIR,
        DEVELOPMENT_ROLES,
    )
    assert _raw_subdir("collect-locked", "full") == (LOCKED_RAW_SUBDIR, LOCKED_ROLES)
    with pytest.raises(ValueError, match="smoke collection"):
        validate_plan(bestest_smoke, TESTCASE_ROOT, LOCKED_ROLES)
    with pytest.raises(ValueError, match="development-only or locked-only"):
        validate_plan(bestest_full, TESTCASE_ROOT, ("fit", "locked_test"))


def test_locked_collection_requires_exact_one_shot_confirmation():
    registry = Path("prelock.json")
    artifact_root = Path("prelock_bundle")
    digest = "1" * 64
    validate_collection_request("collect", "smoke", None)
    validate_collection_request("collect", "full", None)
    validate_collection_request(
        "collect-locked",
        "full",
        LOCKED_CONFIRMATION,
        registry,
        digest,
        tuple(CASES),
        artifact_root,
    )
    with pytest.raises(ValueError, match="requires --mode full"):
        validate_collection_request(
            "collect-locked",
            "smoke",
            LOCKED_CONFIRMATION,
            registry,
            digest,
            tuple(CASES),
            artifact_root,
        )
    with pytest.raises(ValueError, match="requires --confirm-locked-test"):
        validate_collection_request(
            "collect-locked",
            "full",
            None,
            registry,
            digest,
            tuple(CASES),
            artifact_root,
        )
    with pytest.raises(ValueError, match="requires --prelock-registry"):
        validate_collection_request("collect-locked", "full", LOCKED_CONFIRMATION)
    with pytest.raises(ValueError, match="accepted only"):
        validate_collection_request("collect", "full", LOCKED_CONFIRMATION)
    with pytest.raises(ValueError, match="every frozen case"):
        validate_collection_request(
            "collect-locked",
            "full",
            LOCKED_CONFIRMATION,
            registry,
            digest,
            (next(iter(CASES)),),
            artifact_root,
        )


def test_prelock_registry_requires_the_externally_supplied_canonical_digest(tmp_path):
    registry = {"schema": PRELOCK_REGISTRY_SCHEMA, "stage": "prelock", "value": 1}
    path = tmp_path / "prelock.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="ascii")
    digest = hashlib.sha256(canonical_json(registry).encode("ascii")).hexdigest()
    assert validate_prelock_registry(path, digest) == registry
    with pytest.raises(ValueError, match="externally supplied digest"):
        validate_prelock_registry(path, "0" * 64)
    path.write_text(
        '{"schema":"boptest-multicase-prelock-registry-v1",'
        '"stage":"prelock","stage":"prelock"}\n',
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_prelock_registry(path, digest)
    path.write_text(
        '{"schema":"boptest-multicase-prelock-registry-v1",'
        '"stage":"prelock","value":NaN}\n',
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        validate_prelock_registry(path, digest)


def test_locked_collection_requires_live_plans_to_match_prelock_plans(
    tmp_path, full_plans
):
    plan_paths = []
    expected = {}
    for case, plan in sorted(full_plans.items()):
        path = tmp_path / "plans" / "full" / f"{case}.json"
        write_json_immutable(path, plan)
        plan_paths.append(path)
        expected[case] = plan["plan_sha256"]

    assert validate_locked_plan_binding(plan_paths, expected) == expected

    mismatched = dict(expected)
    mismatched["bestest_hydronic_heat_pump"] = "0" * 64
    with pytest.raises(ValueError, match="differ from sealed pre-lock"):
        validate_locked_plan_binding(plan_paths, mismatched)


def test_locked_preflight_and_publication_are_session_atomic(tmp_path):
    final_raw, staging_raw, manifest, pending = preflight_locked_destination(tmp_path)
    staging_raw.mkdir(parents=True)
    (staging_raw / "complete.txt").write_text("complete\n", encoding="ascii")
    wrapper = {"manifest_sha256": "1" * 64, "manifest": {"complete": True}}
    publish_locked_collection(staging_raw, final_raw, manifest, pending, wrapper)
    assert final_raw.is_dir()
    assert (final_raw / "complete.txt").read_text(encoding="ascii") == "complete\n"
    assert json.loads(manifest.read_text(encoding="ascii")) == wrapper
    assert not pending.exists()
    with pytest.raises(FileExistsError, match="partial destination"):
        preflight_locked_destination(tmp_path)


@pytest.mark.parametrize(
    "relative",
    (
        LOCKED_RAW_SUBDIR,
        Path(f".{LOCKED_RAW_SUBDIR.name}.staging"),
        Path("manifests/locked_test_all_corpus_manifest.json"),
        Path("manifests/locked_test_all_corpus_manifest.json.pending"),
    ),
)
def test_locked_preflight_rejects_every_partial_destination(tmp_path, relative):
    occupied = tmp_path / relative
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.touch()
    with pytest.raises(FileExistsError, match="partial destination"):
        preflight_locked_destination(tmp_path)


def test_docker_command_uses_pinned_image_and_explicit_roles(tmp_path):
    plan_path = tmp_path / "plans/full/case.json"
    raw_path = tmp_path / "development_raw"
    command = _docker_command(plan_path, TESTCASE_ROOT, raw_path, DEVELOPMENT_ROLES)
    assert WORKER_IMAGE_ID in command
    assert all("latest" not in part for part in command)
    shell = command[-1]
    assert "--allowed-role fit" in shell
    assert "--allowed-role validation" in shell
    assert "locked_test" not in shell
    assert str(raw_path) + ":/out" in command
    assert f"{TESTCASE_ROOT.parent}:/public-boptest:ro" in command
    assert "--testcase-root /public-boptest/testcases" in shell
    assert f"{TESTCASE_ROOT}:/cases:ro" not in command
    assert "cp /version.txt /work/version.txt" in shell
    assert "--prelock-registry" not in shell
    registry_path = tmp_path / "prelock.json"
    locked = _docker_command(
        plan_path,
        TESTCASE_ROOT,
        raw_path,
        LOCKED_ROLES,
        registry_path,
        "1" * 64,
    )[-1]
    assert "--allowed-role locked_test" in locked
    assert "--prelock-registry /prelock/prelock_registry.json" in locked
    assert f"--expected-prelock-sha256 {'1' * 64}" in locked


class FakeTestCase:
    def __init__(self, fmu_path, uncertainty_path):
        self.case = Path(fmu_path).parents[1].name
        self.adapter = CASES[self.case]
        self.time = 0
        self.step = STEP_SECONDS

    def set_step(self, step):
        self.step = int(step)
        return 200, "ok", None

    def set_scenario(self, scenario):
        return 200, "ok", scenario

    def _state(self, time_s):
        keys = sorted(
            {
                *self.adapter.zone_keys,
                *self.adapter.power_keys,
                *self.adapter.auxiliary_1_keys,
                *self.adapter.auxiliary_2_keys,
            }
        )
        return {
            "time": float(time_s),
            **{key: float(time_s + index + 1) for index, key in enumerate(keys)},
        }

    def initialize(self, start_time, warmup_seconds):
        self.time = int(start_time)
        return 200, "ok", self._state(self.time)

    def get_forecast(self, names, horizon, interval):
        count = int(horizon // interval) + 1
        times = [self.time + index * int(interval) for index in range(count)]
        forecast = {"time": [float(value) for value in times]}
        for offset, name in enumerate(names, start=1):
            forecast[name] = [float(value + offset) for value in times]
        return 200, "ok", forecast

    def advance(self, payload):
        self.time += self.step
        return 200, "ok", self._state(self.time)


class ShiftedForecastTestCase(FakeTestCase):
    def get_forecast(self, names, horizon, interval):
        status, message, forecast = super().get_forecast(names, horizon, interval)
        forecast["time"][1] += 1.0
        return status, message, forecast


class ShiftedNextStateTestCase(FakeTestCase):
    def advance(self, payload):
        status, message, state = super().advance(payload)
        state["time"] += 1.0
        return status, message, state


class ShortForecastTestCase(FakeTestCase):
    def get_forecast(self, names, horizon, interval):
        status, message, forecast = super().get_forecast(names, horizon, interval)
        return status, message, {key: value[:-1] for key, value in forecast.items()}


class LongForecastTestCase(FakeTestCase):
    def get_forecast(self, names, horizon, interval):
        status, message, forecast = super().get_forecast(names, horizon, interval)
        return status, message, {
            key: [*value, value[-1] + STEP_SECONDS]
            for key, value in forecast.items()
        }


def _write_smoke_plan(root: Path, plan: dict) -> Path:
    path = root / "plans/smoke" / f"{plan['case_adapter']['case']}.json"
    write_json_immutable(path, plan)
    return path


def test_fake_testcase_proves_causal_alignment_and_refuses_overwrite(
    tmp_path, monkeypatch, bestest_full, bestest_smoke
):
    _patch_plan_builder(monkeypatch, bestest_full)
    raw = tmp_path / SMOKE_RAW_SUBDIR
    paths = collect_plan(
        bestest_smoke,
        raw,
        allowed_roles=SMOKE_ROLES,
        testcase_root=TESTCASE_ROOT,
        test_case_factory=FakeTestCase,
    )
    assert len(paths) == 1
    with paths[0].open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == FIELDS
    assert len(rows) == TRAJECTORY_STEPS
    assert float(rows[0]["time_s"]) == float(int(rows[0]["day"]) * 86_400)
    for index in range(TRAJECTORY_STEPS - 1):
        assert rows[index]["next_zone_temperature_k"] == rows[index + 1]["zone_temperature_k"]
        assert rows[index]["next_outdoor_temperature_k"] == rows[index + 1]["outdoor_temperature_k"]
    receipt = receipt_path(raw, bestest_smoke, SMOKE_ROLES)
    assert receipt.exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        collect_plan(
            bestest_smoke,
            raw,
            allowed_roles=SMOKE_ROLES,
            testcase_root=TESTCASE_ROOT,
            test_case_factory=FakeTestCase,
        )


@pytest.mark.parametrize(
    "factory",
    (
        ShiftedForecastTestCase,
        ShiftedNextStateTestCase,
        ShortForecastTestCase,
        LongForecastTestCase,
    ),
)
def test_worker_rejects_shifted_simulator_or_forecast_clock(
    tmp_path, monkeypatch, bestest_full, bestest_smoke, factory
):
    _patch_plan_builder(monkeypatch, bestest_full)
    with pytest.raises(ValueError, match="time|forecast length"):
        collect_plan(
            bestest_smoke,
            tmp_path / factory.__name__,
            allowed_roles=SMOKE_ROLES,
            testcase_root=TESTCASE_ROOT,
            test_case_factory=factory,
        )


def test_worker_enforces_prelock_digest_scope(
    tmp_path, monkeypatch, bestest_full, bestest_smoke
):
    _patch_plan_builder(monkeypatch, bestest_full)
    with pytest.raises(ValueError, match="pre-lock registry"):
        collect_plan(
            bestest_full,
            tmp_path / "locked_missing_digest",
            allowed_roles=LOCKED_ROLES,
            testcase_root=TESTCASE_ROOT,
            test_case_factory=FakeTestCase,
        )
    with pytest.raises(ValueError, match="non-locked collection cannot"):
        collect_plan(
            bestest_smoke,
            tmp_path / "smoke_with_digest",
            allowed_roles=SMOKE_ROLES,
            testcase_root=TESTCASE_ROOT,
            prelock_registry={"schema": PRELOCK_REGISTRY_SCHEMA, "stage": "prelock"},
            expected_prelock_sha256="1" * 64,
            test_case_factory=FakeTestCase,
        )
    registry, digest = _test_prelock_registry()
    with pytest.raises(ValueError, match="externally supplied digest"):
        collect_plan(
            bestest_full,
            tmp_path / "locked_mismatched_registry",
            allowed_roles=LOCKED_ROLES,
            testcase_root=TESTCASE_ROOT,
            prelock_registry=registry,
            expected_prelock_sha256="0" * 64,
            test_case_factory=FakeTestCase,
        )
    assert not (tmp_path / "locked_mismatched_registry").exists()


def test_locked_worker_receipt_embeds_prelock_digest(
    tmp_path, monkeypatch, bestest_full
):
    _patch_plan_builder(monkeypatch, bestest_full)
    registry, digest = _test_prelock_registry(2)
    raw = tmp_path / LOCKED_RAW_SUBDIR
    paths = collect_plan(
        bestest_full,
        raw,
        allowed_roles=LOCKED_ROLES,
        testcase_root=TESTCASE_ROOT,
        prelock_registry=registry,
        expected_prelock_sha256=digest,
        test_case_factory=FakeTestCase,
    )
    assert len(paths) == 12
    receipt = json.loads(
        receipt_path(raw, bestest_full, LOCKED_ROLES).read_text(encoding="ascii")
    )
    assert receipt["receipt"]["prelock_registry_sha256"] == digest


def test_all_case_locked_collection_is_staged_sealed_and_bound(
    tmp_path, monkeypatch, full_plans
):
    _patch_plan_builders(monkeypatch, full_plans)
    registry, digest = _test_prelock_registry(3)
    final_raw, staging_raw, manifest_path, pending = preflight_locked_destination(
        tmp_path
    )
    plan_paths = []
    for case, plan in sorted(full_plans.items()):
        plan_path = tmp_path / "plans/full" / f"{case}.json"
        write_json_immutable(plan_path, plan)
        plan_paths.append(plan_path)
        collect_plan(
            plan,
            staging_raw,
            allowed_roles=LOCKED_ROLES,
            testcase_root=TESTCASE_ROOT,
            prelock_registry=registry,
            expected_prelock_sha256=digest,
            test_case_factory=FakeTestCase,
        )
    built_path, wrapper = build_corpus_manifest(
        tmp_path,
        staging_raw,
        plan_paths,
        LOCKED_ROLES,
        TESTCASE_ROOT,
        registry,
        digest,
    )
    assert built_path == manifest_path
    publish_locked_collection(
        staging_raw,
        final_raw,
        manifest_path,
        pending,
        wrapper,
    )
    index = load_corpus_index(
        manifest_path,
        prelock_registry=registry,
        expected_prelock_sha256=digest,
    )
    assert index.collection_kind == "locked_test"
    assert index.prelock_registry_sha256 == digest
    assert len(index.records) == 36
    assert not staging_raw.exists()


def test_manifest_enforces_prelock_digest_scope(bestest_full, bestest_smoke, tmp_path):
    full_path = tmp_path / "plans/full/bestest_hydronic_heat_pump.json"
    smoke_path = tmp_path / "plans/smoke/bestest_hydronic_heat_pump.json"
    write_json_immutable(full_path, bestest_full)
    write_json_immutable(smoke_path, bestest_smoke)
    with pytest.raises(ValueError, match="pre-lock registry"):
        write_corpus_manifest(
            tmp_path,
            tmp_path / LOCKED_RAW_SUBDIR,
            (full_path,),
            LOCKED_ROLES,
            TESTCASE_ROOT,
        )
    with pytest.raises(ValueError, match="non-locked corpus cannot"):
        write_corpus_manifest(
            tmp_path,
            tmp_path / SMOKE_RAW_SUBDIR,
            (smoke_path,),
            SMOKE_ROLES,
            TESTCASE_ROOT,
            prelock_registry={"schema": PRELOCK_REGISTRY_SCHEMA, "stage": "prelock"},
            expected_prelock_sha256="3" * 64,
        )


def _collect_fake_smoke(root: Path, plan: dict) -> tuple[Path, Path]:
    plan_path = _write_smoke_plan(root, plan)
    raw = root / SMOKE_RAW_SUBDIR
    collect_plan(
        plan,
        raw,
        allowed_roles=SMOKE_ROLES,
        testcase_root=TESTCASE_ROOT,
        test_case_factory=FakeTestCase,
    )
    return plan_path, raw


def test_manifest_is_complete_canonical_and_self_hashed(
    tmp_path, monkeypatch, bestest_full, bestest_smoke
):
    _patch_plan_builder(monkeypatch, bestest_full)
    plan_path, raw = _collect_fake_smoke(tmp_path, bestest_smoke)
    manifest_path = write_corpus_manifest(
        tmp_path,
        raw,
        (plan_path,),
        SMOKE_ROLES,
        TESTCASE_ROOT,
    )
    first_bytes = manifest_path.read_bytes()
    wrapper = json.loads(first_bytes)
    expected_hash = hashlib.sha256(
        canonical_json(wrapper["manifest"]).encode("ascii")
    ).hexdigest()
    assert wrapper["manifest_sha256"] == expected_hash
    manifest = wrapper["manifest"]
    assert manifest["allowed_roles"] == ["fit"]
    assert manifest["worker_runtime"] == {
        "image_id": WORKER_IMAGE_ID,
        "boptest_version": WORKER_BOPTEST_VERSION,
    }
    assert manifest["row_schema"]["fields"] == list(FIELDS)
    assert manifest["counts"] == {
        "cases": 1,
        "trajectories": 1,
        "rows": TRAJECTORY_STEPS,
        "roles": {"fit": 1},
    }
    write_corpus_manifest(
        tmp_path,
        raw,
        (plan_path,),
        SMOKE_ROLES,
        TESTCASE_ROOT,
    )
    assert manifest_path.read_bytes() == first_bytes


def test_manifest_rejects_missing_and_stale_files(
    tmp_path, monkeypatch, bestest_full, bestest_smoke
):
    _patch_plan_builder(monkeypatch, bestest_full)
    plan_path, raw = _collect_fake_smoke(tmp_path, bestest_smoke)
    entry = bestest_smoke["entries"][0]
    trajectory = raw / entry["case"] / expected_filename(entry)
    backup = trajectory.with_suffix(".bak")
    trajectory.rename(backup)
    with pytest.raises(ValueError, match="bijection"):
        write_corpus_manifest(
            tmp_path, raw, (plan_path,), SMOKE_ROLES, TESTCASE_ROOT
        )
    backup.rename(trajectory)
    stale = trajectory.parent / "stale.csv"
    stale.write_text("stale\n", encoding="ascii")
    with pytest.raises(ValueError, match="bijection"):
        write_corpus_manifest(
            tmp_path, raw, (plan_path,), SMOKE_ROLES, TESTCASE_ROOT
        )


def test_manifest_rejects_stale_worker_receipt(
    tmp_path, monkeypatch, bestest_full, bestest_smoke
):
    _patch_plan_builder(monkeypatch, bestest_full)
    plan_path, raw = _collect_fake_smoke(tmp_path, bestest_smoke)
    stale = raw / "_receipts" / (
        bestest_smoke["case_adapter"]["case"] + "_development_stale.json"
    )
    stale.write_text("{}\n", encoding="ascii")
    with pytest.raises(ValueError, match="receipt bijection"):
        write_corpus_manifest(
            tmp_path, raw, (plan_path,), SMOKE_ROLES, TESTCASE_ROOT
        )


def test_manifest_rejects_trajectory_changed_after_worker_receipt(
    tmp_path, monkeypatch, bestest_full, bestest_smoke
):
    _patch_plan_builder(monkeypatch, bestest_full)
    plan_path, raw = _collect_fake_smoke(tmp_path, bestest_smoke)
    entry = bestest_smoke["entries"][0]
    trajectory = raw / entry["case"] / expected_filename(entry)
    with trajectory.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    rows[-1]["next_hvac_electric_power_w"] = str(
        float(rows[-1]["next_hvac_electric_power_w"]) + 1.0
    )
    with trajectory.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="receipt payload"):
        write_corpus_manifest(
            tmp_path, raw, (plan_path,), SMOKE_ROLES, TESTCASE_ROOT
        )


def test_manifest_bytes_are_invariant_to_plan_argument_order(
    tmp_path, monkeypatch, full_plans
):
    selected_full = {
        case: full_plans[case]
        for case in ("bestest_hydronic_heat_pump", "twozone_apartment_hydronic")
    }
    _patch_plan_builders(monkeypatch, selected_full)
    plan_paths = []
    raw = tmp_path / SMOKE_RAW_SUBDIR
    for full in selected_full.values():
        plan = smoke_view(full)
        plan_paths.append(_write_smoke_plan(tmp_path, plan))
        collect_plan(
            plan,
            raw,
            allowed_roles=SMOKE_ROLES,
            testcase_root=TESTCASE_ROOT,
            test_case_factory=FakeTestCase,
        )
    manifest = write_corpus_manifest(
        tmp_path,
        raw,
        tuple(reversed(plan_paths)),
        SMOKE_ROLES,
        TESTCASE_ROOT,
    )
    first = manifest.read_bytes()
    write_corpus_manifest(
        tmp_path,
        raw,
        tuple(plan_paths),
        SMOKE_ROLES,
        TESTCASE_ROOT,
    )
    assert manifest.read_bytes() == first
