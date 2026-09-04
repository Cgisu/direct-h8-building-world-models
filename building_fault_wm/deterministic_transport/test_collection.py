from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from building_fault_wm.neural_benchmark import protocol as boptest
from building_fault_wm.neural_benchmark import fault_data

from . import collect
from . import external_freeze
from . import plan as v3_plan
from . import worker_collect


PRELOCK_BINDING = {
    "prelock_registry_sha256": "7" * 64,
    "prelock_registry_file_sha256": "8" * 64,
    "prelock_bundle_inventory_sha256": "9" * 64,
}


def _external_freeze_payload() -> dict:
    return {
        "gist_id": "abc123",
        "revision": "a" * 40,
        "revision_committed_at_utc": "2026-07-23T00:00:00Z",
        "provider_verified_at_utc": "2026-07-23T00:00:01Z",
    }


def _entry(case: str, index: int) -> dict:
    day = 10 + 4 * index
    policies = {}
    for policy in v3_plan.POLICIES:
        action_seed = boptest.stable_seed(
            v3_plan.PLAN_SEED, case, day, policy, "actions"
        )
        levels = v3_plan.policy_levels(policy, action_seed)
        policies[policy] = {
            "trajectory_seed": boptest.stable_seed(
                v3_plan.PLAN_SEED, case, day, policy, "identity"
            ),
            "action_seed": action_seed,
            "action_levels": [int(value) for value in levels],
            "action_sha256": hashlib.sha256(
                levels.astype(np.int8).tobytes()
            ).hexdigest(),
            "transition_counts": v3_plan.transition_counts(levels),
            "dwell_steps": 8 if policy == "old_2h" else 16,
        }
    start, stop = v3_plan.exposure_interval(day)
    return {
        "window_id": f"{case}:day{day:03d}",
        "case": case,
        "day": day,
        "role": "locked_transport",
        "source_plan_role": "fit" if index % 2 == 0 else "validation",
        "source_plan_trajectory_seed": 30_000 + index,
        "temperature_stratum": index % 5,
        "mean_outdoor_temperature_k": 275.0 + index,
        "scenario_seed": boptest.stable_seed(
            v3_plan.PLAN_SEED, case, day, "scenario"
        ),
        "exposure_start_s": start,
        "exposure_stop_s": stop,
        "policies": policies,
    }


def _case_plan(case: str) -> dict:
    payload = {
        "schema": v3_plan.PLAN_SCHEMA,
        "plan_seed": v3_plan.PLAN_SEED,
        "case": case,
        "parent_v2_plan_sha256": "a" * 64,
        "source_sha256": {
            "wrapped_fmu": "b" * 64,
            "weather_csv": "c" * 64,
        },
        "case_adapter": asdict(boptest.CASES[case]),
        "step_seconds": boptest.STEP_SECONDS,
        "warmup_seconds": boptest.WARMUP_SECONDS,
        "trajectory_steps": boptest.TRAJECTORY_STEPS,
        "policies": list(v3_plan.POLICIES),
        "selection_rule": {
            "base_per_temperature_stratum": v3_plan.BASE_PER_STRATUM,
            "extra_windows": v3_plan.EXTRA_WINDOWS,
            "extra_windows_distinct_strata": True,
            "response_values_used": False,
        },
        "entries": [_entry(case, index) for index in range(12)],
    }
    return {**payload, "plan_sha256": v3_plan.canonical_sha256(payload)}


@pytest.fixture
def plan_grid() -> dict[str, dict]:
    return {case: _case_plan(case) for case in sorted(boptest.CASES)}


@pytest.fixture
def certificate(plan_grid: dict[str, dict]) -> dict:
    prior_evidence = v3_plan.build_prior_evidence_contract(
        [
            {
                "label": "fixture",
                "kind": "local_multicase_namespace",
                "present": True,
                "root_binding_sha256": None,
            }
        ],
        [],
        v2_locked_csv_identities=[],
    )
    identity_proof, identity_by_case = (
        v3_plan.build_identity_disjointness_proof(plan_grid, prior_evidence)
    )
    payload = {
        "schema": v3_plan.CERTIFICATE_SCHEMA,
        "v1_plan_sha256_by_case": {
            case: "d" * 64 for case in sorted(boptest.CASES)
        },
        "v2_plan_sha256_by_case": {
            case: "e" * 64 for case in sorted(boptest.CASES)
        },
        "v3_plan_sha256_by_case": {
            case: plan_grid[case]["plan_sha256"]
            for case in sorted(boptest.CASES)
        },
        "prior_evidence": prior_evidence,
        "identity_proof": identity_proof,
        "cases": {
            case: {
                "selected_days": sorted(
                    entry["day"] for entry in plan_grid[case]["entries"]
                ),
                "v2_locked_days": [],
                "selected_window_count": 12,
                "selected_intervals_sha256": v3_plan.canonical_sha256(
                    [
                        v3_plan.exposure_interval(entry["day"])
                        for entry in plan_grid[case]["entries"]
                    ]
                ),
                "all_selected_vs_v1_disjoint": True,
                "all_selected_vs_v2_locked_disjoint": True,
                "no_selected_day_previously_collected_in_v2": True,
                **identity_by_case[case],
            }
            for case in sorted(boptest.CASES)
        },
    }
    return {
        **payload,
        "certificate_sha256": v3_plan.canonical_sha256(payload),
    }


class FakeTestCase:
    def __init__(
        self,
        adapter: boptest.CaseAdapter,
        branch: int,
        events: list[tuple],
        *,
        mismatch_state: bool = False,
        mismatch_forecast: bool = False,
    ) -> None:
        self.adapter = adapter
        self.branch = branch
        self.events = events
        self.mismatch_state = mismatch_state
        self.mismatch_forecast = mismatch_forecast
        self.time = 0
        self.actions: list[dict[str, float]] = []

    def _state(self, time_s: int) -> dict[str, float]:
        state: dict[str, float] = {"time": float(time_s)}
        for offset, key in enumerate(
            (
                *self.adapter.zone_keys,
                *self.adapter.power_keys,
                *self.adapter.auxiliary_1_keys,
                *self.adapter.auxiliary_2_keys,
            )
        ):
            state[key] = float(offset + 1) + time_s / 1_000_000.0
        if self.mismatch_state and self.branch == 1:
            state["mismatch"] = 1.0
        return state

    def set_step(self, step: int):
        self.events.append(("step", self.branch, step))
        return 200, "ok", {}

    def set_scenario(self, scenario: dict):
        self.events.append(("scenario", self.branch, scenario["seed"]))
        return 200, "ok", {}

    def initialize(self, start_time: int, warmup: int):
        self.time = start_time
        self.events.append(("initialize", self.branch, start_time, warmup))
        return 200, "ok", self._state(start_time)

    def get_forecast(self, keys: list[str], horizon: int, interval: int):
        self.events.append(("forecast", self.branch, horizon, interval))
        count = boptest.TRAJECTORY_STEPS + 1
        forecast: dict[str, list[float]] = {
            "time": [
                float(self.time + index * interval) for index in range(count)
            ]
        }
        for offset, key in enumerate(keys):
            forecast[key] = [
                float(offset + 1) + index / 100.0 for index in range(count)
            ]
        if self.mismatch_forecast and self.branch == 1:
            forecast[keys[0]][0] += 1.0
        return 200, "ok", forecast

    def advance(self, action: dict[str, float]):
        self.events.append(("advance", self.branch))
        self.actions.append(dict(action))
        self.time += boptest.STEP_SECONDS
        return 200, "ok", self._state(self.time)


class FakeFactory:
    def __init__(
        self,
        *,
        mismatch_state: bool = False,
        mismatch_forecast: bool = False,
    ) -> None:
        self.events: list[tuple] = []
        self.instances: list[FakeTestCase] = []
        self.mismatch_state = mismatch_state
        self.mismatch_forecast = mismatch_forecast

    def __call__(self, fmu_path: str, uncertainty_path: str) -> FakeTestCase:
        case = Path(fmu_path).parents[1].name
        instance = FakeTestCase(
            boptest.CASES[case],
            len(self.instances),
            self.events,
            mismatch_state=self.mismatch_state,
            mismatch_forecast=self.mismatch_forecast,
        )
        self.instances.append(instance)
        return instance


def test_pair_uses_frozen_actions_and_hashes_before_advancing(
    tmp_path: Path, plan_grid: dict[str, dict]
) -> None:
    case = "bestest_hydronic_heat_pump"
    entry = plan_grid[case]["entries"][0]
    factory = FakeFactory()
    paths, receipt = worker_collect.collect_pair(
        entry,
        boptest.CASES[case],
        tmp_path,
        testcase_root=Path("/fake/testcases"),
        test_case_factory=factory,
    )

    assert len(paths) == 2
    assert all("_locked_test_policy-" in path.name for path in paths)
    assert receipt["initialized_state_sha256"]
    assert receipt["full_forecast_sha256"]
    assert (
        receipt["branches"]["old_2h"]["initialized_state_sha256"]
        == receipt["branches"]["new_4h"]["initialized_state_sha256"]
    )
    assert (
        receipt["branches"]["old_2h"]["full_forecast_sha256"]
        == receipt["branches"]["new_4h"]["full_forecast_sha256"]
    )

    first_advance = next(
        index for index, event in enumerate(factory.events) if event[0] == "advance"
    )
    assert {event[1] for event in factory.events[:first_advance] if event[0] == "forecast"} == {
        0,
        1,
    }
    assert [event[2] for event in factory.events if event[0] == "scenario"] == [
        entry["scenario_seed"],
        entry["scenario_seed"],
    ]

    for branch_index, policy in enumerate(v3_plan.POLICIES):
        expected = np.asarray(
            entry["policies"][policy]["action_levels"], dtype=float
        )
        instance = factory.instances[branch_index]
        observed = [
            (
                action[instance.adapter.action_pairs[0][1]]
                - instance.adapter.base_setpoint_k
            )
            / instance.adapter.action_amplitude_k
            for action in instance.actions
        ]
        np.testing.assert_array_equal(observed, expected)
        for activate, _ in instance.adapter.action_pairs:
            assert all(action[activate] == 1.0 for action in instance.actions)


@pytest.mark.parametrize("kind", ("state", "forecast"))
def test_pair_rejects_initialization_mismatch_before_any_advance(
    tmp_path: Path, plan_grid: dict[str, dict], kind: str
) -> None:
    case = "bestest_hydronic_heat_pump"
    factory = FakeFactory(
        mismatch_state=kind == "state",
        mismatch_forecast=kind == "forecast",
    )
    with pytest.raises(ValueError, match=f"paired v3 .*{kind}.* hashes differ"):
        worker_collect.collect_pair(
            plan_grid[case]["entries"][0],
            boptest.CASES[case],
            tmp_path,
            testcase_root=Path("/fake/testcases"),
            test_case_factory=factory,
        )
    assert not any(event[0] == "advance" for event in factory.events)
    assert not list(tmp_path.rglob("*.csv"))


def test_pair_refuses_overwrite_without_constructing_simulator(
    tmp_path: Path, plan_grid: dict[str, dict]
) -> None:
    case = "bestest_hydronic_heat_pump"
    entry = plan_grid[case]["entries"][0]
    worker_collect.collect_pair(
        entry,
        boptest.CASES[case],
        tmp_path,
        testcase_root=Path("/fake/testcases"),
        test_case_factory=FakeFactory(),
    )
    forbidden = FakeFactory()
    with pytest.raises(FileExistsError, match="overwrite"):
        worker_collect.collect_pair(
            entry,
            boptest.CASES[case],
            tmp_path,
            testcase_root=Path("/fake/testcases"),
            test_case_factory=forbidden,
        )
    assert forbidden.instances == []


def test_csv_has_exact_loader_schema_actions_and_continuity(
    tmp_path: Path, plan_grid: dict[str, dict]
) -> None:
    case = "bestest_hydronic_heat_pump"
    entry = plan_grid[case]["entries"][0]
    paths, _ = worker_collect.collect_pair(
        entry,
        boptest.CASES[case],
        tmp_path,
        testcase_root=Path("/fake/testcases"),
        test_case_factory=FakeFactory(),
    )
    for policy, path in zip(v3_plan.POLICIES, paths, strict=True):
        result = collect._validate_csv(
            path, entry, policy, boptest.CASES[case]
        )
        assert result["rows"] == 192
        assert result["fields"] == 26
        with path.open(newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        assert tuple(rows[0]) == worker_collect.FIELDS
        assert {row["role"] for row in rows} == {"locked_test"}
        assert all(
            rows[index - 1]["next_zone_temperature_k"]
            == rows[index]["zone_temperature_k"]
            for index in range(1, len(rows))
        )


def test_existing_clean_trajectory_loader_accepts_v3_branch(
    tmp_path: Path, plan_grid: dict[str, dict]
) -> None:
    case = "bestest_hydronic_heat_pump"
    entry = plan_grid[case]["entries"][0]
    paths, _ = worker_collect.collect_pair(
        entry,
        boptest.CASES[case],
        tmp_path,
        testcase_root=Path("/fake/testcases"),
        test_case_factory=FakeFactory(),
    )
    policy = "new_4h"
    path = paths[1]
    metadata = entry["policies"][policy]
    record = fault_data.CorpusRecord(
        key=fault_data.TrajectoryKey(
            case=case,
            role="locked_test",
            day=entry["day"],
            trajectory_seed=metadata["trajectory_seed"],
        ),
        relative_path=str(path.relative_to(tmp_path)),
        source_sha256=v3_plan.sha256_file(path),
        rows=boptest.TRAJECTORY_STEPS,
        step_seconds=boptest.STEP_SECONDS,
        base_setpoint_k=boptest.CASES[case].base_setpoint_k,
        action_amplitude_k=boptest.CASES[case].action_amplitude_k,
    )
    index = fault_data.CorpusIndex(
        root=tmp_path,
        manifest_path=tmp_path / "synthetic-index.json",
        manifest_sha256="f" * 64,
        collection_kind="paired_locked_transport",
        prelock_registry_sha256=None,
        allowed_roles=("locked_test",),
        records=(record,),
        plan_sha256_by_case=((case, plan_grid[case]["plan_sha256"]),),
    )
    loaded = fault_data.load_clean_trajectory(
        index, record, allow_locked_test=True
    )
    assert loaded.observations.shape == (192, 4)
    assert loaded.contexts.shape == (192, 5)
    np.testing.assert_array_equal(
        loaded.actions[:, 0],
        np.asarray(metadata["action_levels"], dtype=float),
    )


def test_csv_validator_rejects_broken_continuity(
    tmp_path: Path, plan_grid: dict[str, dict]
) -> None:
    case = "bestest_hydronic_heat_pump"
    entry = plan_grid[case]["entries"][0]
    paths, _ = worker_collect.collect_pair(
        entry,
        boptest.CASES[case],
        tmp_path,
        testcase_root=Path("/fake/testcases"),
        test_case_factory=FakeFactory(),
    )
    path = paths[0]
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    rows[1]["zone_temperature_k"] = "999"
    changed = tmp_path / "broken.csv"
    with changed.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=worker_collect.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="continuity mismatch"):
        collect._validate_csv(
            changed, entry, "old_2h", boptest.CASES[case]
        )


def test_collect_plan_writes_complete_provenance_receipt(
    monkeypatch,
    tmp_path: Path,
    plan_grid: dict[str, dict],
    certificate: dict,
) -> None:
    case = "bestest_hydronic_heat_pump"
    monkeypatch.setattr(
        worker_collect, "_validate_public_source", lambda *args, **kwargs: None
    )
    paths = worker_collect.collect_plan(
        plan_grid[case],
        tmp_path,
        plans=plan_grid,
        certificate=certificate,
        expected_certificate_sha256=certificate["certificate_sha256"],
        testcase_root=Path("/fake/testcases"),
        test_case_factory=FakeFactory(),
    )
    assert len(paths) == 24
    receipt = worker_collect.receipt_path(tmp_path, plan_grid[case])
    wrapper = v3_plan.load_json(receipt)
    payload = wrapper["receipt"]
    assert wrapper["receipt_sha256"] == v3_plan.canonical_sha256(payload)
    assert payload["plan_sha256"] == plan_grid[case]["plan_sha256"]
    assert payload["source_sha256"] == plan_grid[case]["source_sha256"]
    assert payload["worker_image_id"] == boptest.WORKER_IMAGE_ID
    assert payload["worker_code_sha256"] == worker_collect.worker_code_hashes()
    assert len(payload["pairs"]) == 12
    assert len(payload["files"]) == 24


def test_certificate_rejects_changed_plan_binding(
    plan_grid: dict[str, dict], certificate: dict
) -> None:
    changed = copy.deepcopy(plan_grid)
    changed["bestest_hydronic_heat_pump"]["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plan (canonical SHA-256|grid)"):
        worker_collect.validate_certificate_grid(
            certificate, changed, certificate["certificate_sha256"]
        )


def test_certificate_rejects_resealed_interval_inventory(
    plan_grid: dict[str, dict], certificate: dict
) -> None:
    changed = copy.deepcopy(certificate)
    changed.pop("certificate_sha256")
    changed["cases"]["bestest_hydronic_heat_pump"][
        "selected_intervals_sha256"
    ] = "0" * 64
    changed["certificate_sha256"] = v3_plan.canonical_sha256(changed)
    with pytest.raises(ValueError, match="interval hash differs"):
        worker_collect.validate_certificate_grid(
            changed, plan_grid, changed["certificate_sha256"]
        )


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"one","schema":"two"}\n', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        worker_collect._load_json_strict(path)


def _write_grid(root: Path, plans: dict[str, dict]) -> None:
    root.mkdir(parents=True)
    for case, plan in plans.items():
        (root / f"{case}.json").write_bytes(v3_plan.canonical_bytes(plan))


def test_readiness_is_value_blind_and_creates_no_collection_state(
    monkeypatch,
    tmp_path: Path,
    plan_grid: dict[str, dict],
    certificate: dict,
) -> None:
    plan_root = tmp_path / "plans"
    certificate_path = tmp_path / "certificate.json"
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.json"
    _write_grid(plan_root, plan_grid)
    certificate_path.write_bytes(v3_plan.canonical_bytes(certificate))
    monkeypatch.setattr(
        collect,
        "validate_public_checkout",
        lambda root: {
            "repository_url": boptest.BOPTEST_REPOSITORY_URL,
            "commit": boptest.BOPTEST_COMMIT,
            "license_sha256": boptest.BOPTEST_LICENSE_SHA256,
        },
    )
    monkeypatch.setattr(
        worker_collect, "_validate_public_source", lambda *args, **kwargs: None
    )
    report = collect.readiness_report(
        certificate["certificate_sha256"],
        plan_root=plan_root,
        certificate_path=certificate_path,
        testcase_root=tmp_path / "public/testcases",
        prelock_binding=PRELOCK_BINDING,
    )
    assert report["readiness_sha256"]
    assert report["locked_response_values_accessed"] is False
    assert report["state_created"] is False
    assert report["branch_count"] == 72
    assert not raw.exists()
    assert not manifest.exists()


def test_docker_command_uses_v3_paths_and_pinned_image(
    plan_grid: dict[str, dict], certificate: dict, tmp_path: Path
) -> None:
    report = {
        "plan_sha256_by_case": {
            case: plan["plan_sha256"] for case, plan in plan_grid.items()
        },
        "certificate_sha256": certificate["certificate_sha256"],
    }
    readiness = collect.Readiness(
        plans=plan_grid,
        plan_paths=tuple(
            tmp_path / f"{case}.json" for case in sorted(boptest.CASES)
        ),
        certificate=certificate,
        expected_certificate_sha256=certificate["certificate_sha256"],
        collection_code_sha256=collect.collection_code_hashes(),
        report=report,
    )
    command = collect._docker_command(
        readiness.plan_paths[0],
        tmp_path / "staging",
        readiness,
        plan_root=tmp_path / "plans",
        certificate_path=tmp_path / "certificate.json",
        testcase_root=tmp_path / "public/testcases",
    )
    rendered = " ".join(command)
    assert boptest.WORKER_IMAGE_ID in command
    assert (
        "/workspace/building_fault_wm/"
        "direct_h8_deterministic_transport_v3/worker_collect.py"
        in rendered
    )
    assert "python -m building_fault_wm" not in rendered
    assert "--plan-root /v3-plans" in rendered
    assert "--expected-certificate-sha256" in rendered
    assert "direct_h8_supervision_study_v2" not in rendered


def test_publication_is_atomic_and_refuses_every_existing_destination(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "locked_transport_raw"
    manifest = tmp_path / "manifests/manifest.json"
    _, staging, _, pending = collect._preflight_destination(raw, manifest)
    staging.mkdir(parents=True)
    (staging / "complete.txt").write_text("complete\n", encoding="ascii")
    wrapper = {"manifest_sha256": "1" * 64, "manifest": {"complete": True}}
    collect._publish(staging, raw, manifest, pending, wrapper)
    assert raw.is_dir()
    assert manifest.is_file()
    assert not pending.exists()
    assert json.loads(manifest.read_text(encoding="ascii")) == wrapper
    assert manifest.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="destination exists"):
        collect._preflight_destination(raw, manifest)


def test_collection_rejects_wrong_confirmation_before_readiness(
    monkeypatch, tmp_path: Path
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("readiness must not run for a wrong token")

    monkeypatch.setattr(collect, "prepare_readiness", forbidden)
    with pytest.raises(ValueError, match="exact confirmation token"):
        collect.run_collection(
            "a" * 64,
            "b" * 64,
            "wrong",
            raw_root=tmp_path / "raw",
            manifest_path=tmp_path / "manifest.json",
        )


def test_collection_rejects_readiness_drift_before_preflight(
    monkeypatch,
    tmp_path: Path,
    plan_grid: dict[str, dict],
    certificate: dict,
) -> None:
    report = {"readiness_sha256": "1" * 64}
    readiness = collect.Readiness(
        plans=plan_grid,
        plan_paths=tuple(),
        certificate=certificate,
        expected_certificate_sha256=certificate["certificate_sha256"],
        collection_code_sha256=collect.collection_code_hashes(),
        report=report,
    )
    monkeypatch.setattr(collect, "prepare_readiness", lambda *a, **k: readiness)
    raw = tmp_path / "raw"
    with pytest.raises(ValueError, match="externally frozen digest"):
        collect.run_collection(
            certificate["certificate_sha256"],
            "2" * 64,
            collect.CONFIRMATION_TOKEN,
            raw_root=raw,
            manifest_path=tmp_path / "manifest.json",
        )
    assert not raw.exists()


def _run_readiness(
    tmp_path: Path,
    plan_grid: dict[str, dict],
    certificate: dict,
    *,
    readiness_sha256: str = "1" * 64,
) -> collect.Readiness:
    return collect.Readiness(
        plans=plan_grid,
        plan_paths=(tmp_path / "plans/bestest_hydronic_heat_pump.json",),
        certificate=certificate,
        expected_certificate_sha256=certificate["certificate_sha256"],
        collection_code_sha256=collect.collection_code_hashes(),
        report={
            "readiness_sha256": readiness_sha256,
            **PRELOCK_BINDING,
            "plan_sha256_by_case": {
                case: plan["plan_sha256"] for case, plan in plan_grid.items()
            },
            "protocol_sha256": "2" * 64,
            "certificate_file_sha256": "3" * 64,
            "source_sha256_by_case": {
                case: plan["source_sha256"] for case, plan in plan_grid.items()
            },
        },
    )


def test_attempt_precedes_simulator_and_failure_is_terminal(
    monkeypatch,
    tmp_path: Path,
    plan_grid: dict[str, dict],
    certificate: dict,
) -> None:
    digest = "1" * 64
    readiness = _run_readiness(
        tmp_path, plan_grid, certificate, readiness_sha256=digest
    )
    state_root = tmp_path / "state"
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.json"
    freeze_receipt = tmp_path / "external_freeze_receipt.json"
    freeze_receipt.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(collect, "STATE_ROOT", state_root)
    monkeypatch.setattr(collect, "prepare_readiness", lambda *a, **k: readiness)
    monkeypatch.setattr(
        external_freeze,
        "validate_external_freeze_receipt",
        lambda *args, **kwargs: _external_freeze_payload(),
    )

    calls = 0

    def fail_after_check(command, check):
        nonlocal calls
        calls += 1
        attempt = state_root / digest / collect.ATTEMPT_MARKER
        assert attempt.is_file()
        assert attempt.stat().st_mode & 0o777 == 0o444
        payload = worker_collect._load_json_strict(attempt)
        assert payload["locked_response_values_accessed"] is False
        assert payload["readiness_sha256"] == digest
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(subprocess, "run", fail_after_check)
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        collect.run_collection(
            certificate["certificate_sha256"],
            digest,
            collect.CONFIRMATION_TOKEN,
            raw_root=raw,
            manifest_path=manifest,
            plan_root=tmp_path / "plans",
            certificate_path=tmp_path / "certificate.json",
            testcase_root=tmp_path / "public/testcases",
            external_freeze_receipt_path=freeze_receipt,
        )
    assert calls == 1
    state_dir = state_root / digest
    attempt = worker_collect._load_json_strict(
        state_dir / collect.ATTEMPT_MARKER
    )
    failure = worker_collect._load_json_strict(
        state_dir / collect.FAILURE_MARKER
    )
    assert failure["attempt_marker_sha256"] == v3_plan.sha256_file(
        state_dir / collect.ATTEMPT_MARKER
    )
    assert failure["simulator_process_started"] is True
    assert failure["locked_response_values_may_have_been_accessed"] is True
    assert failure["retry_permitted_under_same_readiness_digest"] is False
    assert not (state_dir / collect.COMPLETION_MARKER).exists()
    assert attempt["collector_commands"]
    assert (raw.parent / f".{raw.name}.staging").is_dir()

    with pytest.raises(FileExistsError, match="already has a collection attempt"):
        collect.run_collection(
            certificate["certificate_sha256"],
            digest,
            collect.CONFIRMATION_TOKEN,
            raw_root=raw,
            manifest_path=manifest,
            plan_root=tmp_path / "plans",
            certificate_path=tmp_path / "certificate.json",
            testcase_root=tmp_path / "public/testcases",
            external_freeze_receipt_path=freeze_receipt,
        )
    assert calls == 1


def test_success_writes_completion_only_after_publication(
    monkeypatch,
    tmp_path: Path,
    plan_grid: dict[str, dict],
    certificate: dict,
) -> None:
    digest = "4" * 64
    readiness = _run_readiness(
        tmp_path, plan_grid, certificate, readiness_sha256=digest
    )
    state_root = tmp_path / "state"
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.json"
    freeze_receipt = tmp_path / "external_freeze_receipt.json"
    freeze_receipt.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(collect, "STATE_ROOT", state_root)
    monkeypatch.setattr(collect, "prepare_readiness", lambda *a, **k: readiness)
    monkeypatch.setattr(
        external_freeze,
        "validate_external_freeze_receipt",
        lambda *args, **kwargs: _external_freeze_payload(),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(
        collect,
        "validate_staged_collection",
        lambda staging, ready: {"files": [], "worker_receipts": []},
    )
    wrapper = {
        "manifest_sha256": "5" * 64,
        "manifest": {"schema": "synthetic-success"},
    }
    monkeypatch.setattr(collect, "build_manifest", lambda *args: wrapper)

    result = collect.run_collection(
        certificate["certificate_sha256"],
        digest,
        collect.CONFIRMATION_TOKEN,
        raw_root=raw,
        manifest_path=manifest,
        plan_root=tmp_path / "plans",
        certificate_path=tmp_path / "certificate.json",
        testcase_root=tmp_path / "public/testcases",
        external_freeze_receipt_path=freeze_receipt,
    )
    assert result == wrapper
    state_dir = state_root / digest
    assert (state_dir / collect.ATTEMPT_MARKER).is_file()
    assert (state_dir / collect.COMPLETION_MARKER).is_file()
    assert not (state_dir / collect.FAILURE_MARKER).exists()
    completion = worker_collect._load_json_strict(
        state_dir / collect.COMPLETION_MARKER
    )
    assert completion["manifest_file_sha256"] == v3_plan.sha256_file(manifest)
    assert completion["manifest_payload_sha256"] == wrapper["manifest_sha256"]
    assert completion["locked_response_values_accessed_after_attempt"] is True
    assert raw.is_dir()
    assert manifest.is_file()


def test_state_evidence_is_exclusive_and_read_only(
    monkeypatch, tmp_path: Path
) -> None:
    digest = "6" * 64
    monkeypatch.setattr(collect, "STATE_ROOT", tmp_path / "state")
    path = collect._state_dir(digest) / collect.ATTEMPT_MARKER
    collect._write_state_once(path, {"attempt": 1})
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="overwrite v3 state evidence"):
        collect._write_state_once(path, {"attempt": 2})
    assert worker_collect._load_json_strict(path) == {"attempt": 1}
