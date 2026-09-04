"""Collect paired, prospectively locked BOPTEST transport trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from building_fault_wm.neural_benchmark import protocol as boptest

if __package__:
    from . import plan as v3_plan
else:
    import plan as v3_plan


STUDY_KIND = "direct_h8_deterministic_transport_v3"
WORKER_RECEIPT_SCHEMA = "boptest-direct-h8-transport-worker-receipt-v1"
ROW_SCHEMA = "boptest-direct-h8-transport-transition-v1"
OUTPUT_ROLE = "locked_test"

# Keep this byte-for-byte compatible with the existing CleanTrajectory loader.
FIELDS = (
    "case",
    "role",
    "day",
    "trajectory_seed",
    "step",
    "time_s",
    "normalized_action",
    "setpoint_k",
    "zone_temperature_k",
    "hvac_electric_power_w",
    "auxiliary_1",
    "auxiliary_2",
    "next_zone_temperature_k",
    "next_hvac_electric_power_w",
    "next_auxiliary_1",
    "next_auxiliary_2",
    "outdoor_temperature_k",
    "global_horizontal_solar_w_m2",
    "comfort_lower_k",
    "comfort_upper_k",
    "electricity_price",
    "next_outdoor_temperature_k",
    "next_global_horizontal_solar_w_m2",
    "next_comfort_lower_k",
    "next_comfort_upper_k",
    "next_electricity_price",
)


def worker_code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {
        name: v3_plan.sha256_file(here / name)
        for name in ("plan.py", "worker_collect.py", "collect.py")
    }


def _require_plain_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")


def _load_json_strict(path: Path) -> dict:
    _require_plain_file(path, "v3 JSON artifact")
    value = boptest.strict_json_loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"v3 JSON artifact is not an object: {path}")
    return value


def _load_plan_grid(root: Path) -> dict[str, dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"v3 plan root is not a plain directory: {root}")
    paths = list(root.iterdir())
    expected = {f"{case}.json" for case in boptest.CASES}
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("v3 plan root contains a non-file or symbolic link")
    if {path.name for path in paths} != expected:
        raise ValueError("v3 plan root differs from the three-case contract")
    return {
        case: _load_json_strict(root / f"{case}.json")
        for case in sorted(boptest.CASES)
    }


def _reduce(
    state: Mapping[str, object], keys: Iterable[str], reduction: str
) -> float:
    values = np.asarray([float(state[key]) for key in keys], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("BOPTEST state contains a non-finite canonical observation")
    if reduction == "mean":
        return float(values.mean())
    if reduction == "sum":
        return float(values.sum())
    raise ValueError(f"unknown canonical reduction {reduction}")


def canonical_observation(
    adapter: boptest.CaseAdapter, state: Mapping[str, object]
) -> tuple[float, float, float, float]:
    return (
        _reduce(state, adapter.zone_keys, "mean"),
        _reduce(state, adapter.power_keys, "sum"),
        _reduce(state, adapter.auxiliary_1_keys, adapter.auxiliary_1_reduction),
        _reduce(state, adapter.auxiliary_2_keys, adapter.auxiliary_2_reduction),
    )


def _forecast_value(
    forecast: Mapping[str, object], keys: Iterable[str], index: int
) -> float:
    values = np.asarray(
        [float(forecast[key][index]) for key in keys],  # type: ignore[index]
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("BOPTEST forecast contains a non-finite value")
    return float(values.mean())


def _context(
    adapter: boptest.CaseAdapter,
    forecast: Mapping[str, object],
    index: int,
) -> tuple[float, float, float, float, float]:
    return (
        _forecast_value(forecast, ("TDryBul",), index),
        _forecast_value(forecast, ("HGloHor",), index),
        _forecast_value(forecast, adapter.lower_forecast_keys, index),
        _forecast_value(forecast, adapter.upper_forecast_keys, index),
        _forecast_value(forecast, (adapter.price_forecast_key,), index),
    )


def _canonical_simulator_value(value: object) -> object:
    """Normalize simulator responses without rounding any numeric value."""

    if isinstance(value, np.generic):
        return _canonical_simulator_value(value.item())
    if isinstance(value, np.ndarray):
        return [_canonical_simulator_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("simulator response contains a non-string key")
            result[key] = _canonical_simulator_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_simulator_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("simulator response contains a non-finite number")
        return value
    raise ValueError(f"unsupported simulator response value: {type(value).__name__}")


def canonical_simulator_sha256(value: object) -> str:
    normalized = _canonical_simulator_value(value)
    return hashlib.sha256(v3_plan.canonical_bytes(normalized)).hexdigest()


def expected_filename(entry: Mapping[str, object], policy: str) -> str:
    if policy not in v3_plan.POLICIES:
        raise ValueError(f"unknown v3 policy in filename: {policy}")
    metadata = entry["policies"][policy]  # type: ignore[index]
    return (
        f"day{int(entry['day']):03d}_{OUTPUT_ROLE}_"
        f"policy-{policy}_seed{int(metadata['trajectory_seed'])}.csv"
    )


def receipt_path(output_dir: Path, plan: Mapping[str, object]) -> Path:
    return (
        output_dir
        / "_receipts"
        / f"{plan['case']}_paired_{str(plan['plan_sha256'])[:16]}.json"
    )


def _certificate_payload(certificate: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }


def validate_certificate_grid(
    certificate: Mapping[str, object],
    plans: Mapping[str, Mapping[str, object]],
    expected_certificate_sha256: str,
) -> None:
    if set(plans) != set(boptest.CASES):
        raise ValueError("v3 plan grid does not cover all public cases")
    if not boptest.valid_sha256(expected_certificate_sha256):
        raise ValueError("expected v3 certificate digest is not a SHA-256")
    for case, plan in plans.items():
        v3_plan.validate_case_plan(plan)
        if plan.get("case") != case:
            raise ValueError(f"v3 plan grid identity differs for {case}")
    if certificate.get("schema") != v3_plan.CERTIFICATE_SCHEMA:
        raise ValueError("v3 disjointness certificate schema mismatch")
    actual = v3_plan.canonical_sha256(_certificate_payload(certificate))
    if certificate.get("certificate_sha256") != actual:
        raise ValueError("v3 disjointness certificate self-hash mismatch")
    if actual != expected_certificate_sha256:
        raise ValueError("v3 disjointness certificate differs from the frozen digest")

    expected_hashes = {
        case: plans[case].get("plan_sha256") for case in sorted(boptest.CASES)
    }
    if certificate.get("v3_plan_sha256_by_case") != expected_hashes:
        raise ValueError("v3 disjointness certificate binds a different plan grid")
    for field in ("v1_plan_sha256_by_case", "v2_plan_sha256_by_case"):
        hashes = certificate.get(field)
        if (
            not isinstance(hashes, dict)
            or set(hashes) != set(boptest.CASES)
            or any(not boptest.valid_sha256(value) for value in hashes.values())
        ):
            raise ValueError(f"v3 disjointness certificate {field} is invalid")
    prior_evidence = certificate.get("prior_evidence")
    if not isinstance(prior_evidence, dict):
        raise ValueError("v3 disjointness certificate omits prior evidence")
    v3_plan.validate_prior_evidence_contract(prior_evidence)
    expected_identity_proof, expected_identity_by_case = (
        v3_plan.build_identity_disjointness_proof(plans, prior_evidence)
    )
    if certificate.get("identity_proof") != expected_identity_proof:
        raise ValueError("v3 disjointness certificate identity proof differs")
    cases = certificate.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(boptest.CASES):
        raise ValueError("v3 disjointness certificate case grid is incomplete")
    for case in sorted(boptest.CASES):
        record = cases[case]
        if not isinstance(record, dict):
            raise ValueError(f"v3 certificate record is invalid for {case}")
        entries = plans[case].get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"v3 plan entries are invalid for {case}")
        selected_days = sorted(int(entry["day"]) for entry in entries)
        if record.get("selected_days") != selected_days:
            raise ValueError(f"v3 certificate selected days differ for {case}")
        if record.get("selected_window_count") != v3_plan.EXPECTED_WINDOWS_PER_CASE:
            raise ValueError(f"v3 certificate window count differs for {case}")
        expected_intervals = [
            v3_plan.exposure_interval(day) for day in selected_days
        ]
        if record.get("selected_intervals_sha256") != v3_plan.canonical_sha256(
            expected_intervals
        ):
            raise ValueError(f"v3 certificate interval hash differs for {case}")
        locked_days = record.get("v2_locked_days")
        if (
            not isinstance(locked_days, list)
            or any(
                isinstance(day, bool) or not isinstance(day, int) or day < 1
                for day in locked_days
            )
            or locked_days != sorted(set(locked_days))
            or set(locked_days) & set(selected_days)
        ):
            raise ValueError(f"v3 certificate prior-day inventory is invalid for {case}")
        for field in (
            "all_selected_vs_v1_disjoint",
            "all_selected_vs_v2_locked_disjoint",
            "no_selected_day_previously_collected_in_v2",
        ):
            if record.get(field) is not True:
                raise ValueError(f"v3 certificate does not prove {field} for {case}")
        for field, expected in expected_identity_by_case[case].items():
            if record.get(field) != expected:
                raise ValueError(
                    f"v3 certificate prior identity proof differs for {case}: {field}"
                )


def _validate_public_source(
    plan: Mapping[str, object],
    adapter: boptest.CaseAdapter,
    testcase_root: Path,
) -> None:
    source = plan.get("source_sha256")
    if not isinstance(source, dict) or set(source) != {"wrapped_fmu", "weather_csv"}:
        raise ValueError("v3 plan source hashes are incomplete")
    model_root = testcase_root / adapter.case / "models"
    observed = {
        "wrapped_fmu": v3_plan.sha256_file(model_root / "wrapped.fmu"),
        "weather_csv": v3_plan.sha256_file(
            model_root / "Resources" / "weather.csv"
        ),
    }
    if source != observed:
        raise ValueError("mounted BOPTEST source differs from the frozen v3 plan")


def validate_collection_contract(
    plan: Mapping[str, object],
    *,
    plans: Mapping[str, Mapping[str, object]],
    certificate: Mapping[str, object],
    expected_certificate_sha256: str,
    testcase_root: Path,
) -> tuple[boptest.CaseAdapter, tuple[dict, ...]]:
    v3_plan.validate_case_plan(plan)
    case = plan.get("case")
    if case not in boptest.CASES:
        raise ValueError("v3 worker plan names an unknown case")
    adapter = boptest.CASES[str(case)]
    if boptest.canonical_json(plan.get("case_adapter")) != boptest.canonical_json(
        asdict(adapter)
    ):
        raise ValueError("v3 plan adapter differs from the frozen public adapter")
    if plan.get("step_seconds") != boptest.STEP_SECONDS:
        raise ValueError("v3 plan cadence differs from 15 minutes")
    if plan.get("warmup_seconds") != boptest.WARMUP_SECONDS:
        raise ValueError("v3 plan warmup differs from one day")
    if plan.get("trajectory_steps") != boptest.TRAJECTORY_STEPS:
        raise ValueError("v3 plan trajectory length differs from 192 rows")

    validate_certificate_grid(certificate, plans, expected_certificate_sha256)
    sealed = plans.get(adapter.case)
    if sealed is None or v3_plan.canonical_bytes(plan) != v3_plan.canonical_bytes(
        sealed
    ):
        raise ValueError("v3 worker plan differs from the validated plan grid")
    _validate_public_source(plan, adapter, testcase_root)

    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise ValueError("v3 worker plan entries are invalid")
    if plan.get("policies") != list(v3_plan.POLICIES):
        raise ValueError("v3 plan policy order differs from the frozen contract")
    identities: set[int] = set()
    for entry in entries:
        day = int(entry["day"])
        if entry.get("window_id") != f"{adapter.case}:day{day:03d}":
            raise ValueError("v3 window identity differs from its case and day")
        if (
            entry.get("exposure_start_s"),
            entry.get("exposure_stop_s"),
        ) != v3_plan.exposure_interval(day):
            raise ValueError("v3 exposure interval differs from its frozen day")
        scenario_seed = entry.get("scenario_seed")
        if (
            isinstance(scenario_seed, bool)
            or not isinstance(scenario_seed, int)
            or scenario_seed < 0
        ):
            raise ValueError("v3 scenario seed is invalid")
        if scenario_seed != boptest.stable_seed(
            v3_plan.PLAN_SEED, adapter.case, day, "scenario"
        ):
            raise ValueError("v3 scenario seed differs from deterministic regeneration")
        metadata_grid = entry.get("policies")
        if not isinstance(metadata_grid, dict):
            raise ValueError("v3 policy metadata are invalid")
        for policy in v3_plan.POLICIES:
            metadata = metadata_grid[policy]
            action_seed = metadata.get("action_seed")
            if (
                isinstance(action_seed, bool)
                or not isinstance(action_seed, int)
                or action_seed < 0
            ):
                raise ValueError("v3 action seed is invalid")
            expected_action_seed = boptest.stable_seed(
                v3_plan.PLAN_SEED, adapter.case, day, policy, "actions"
            )
            if action_seed != expected_action_seed:
                raise ValueError("v3 action seed differs from deterministic regeneration")
            identity = metadata.get("trajectory_seed")
            if (
                isinstance(identity, bool)
                or not isinstance(identity, int)
                or identity < 0
                or identity in identities
            ):
                raise ValueError("v3 trajectory identity seeds are invalid or repeated")
            expected_identity = boptest.stable_seed(
                v3_plan.PLAN_SEED, adapter.case, day, policy, "identity"
            )
            if identity != expected_identity:
                raise ValueError(
                    "v3 trajectory identity differs from deterministic regeneration"
                )
            identities.add(identity)
            frozen = np.asarray(metadata.get("action_levels"), dtype=float)
            regenerated = v3_plan.policy_levels(
                policy, action_seed
            )
            if not np.array_equal(frozen, regenerated):
                raise ValueError("v3 frozen action array changed")
            expected_dwell = 8 if policy == "old_2h" else 16
            if metadata.get("dwell_steps") != expected_dwell:
                raise ValueError("v3 frozen action dwell metadata changed")
        left = int(metadata_grid[v3_plan.POLICIES[0]]["trajectory_seed"])
        right = int(metadata_grid[v3_plan.POLICIES[1]]["trajectory_seed"])
        if left == right:
            raise ValueError("paired policy branches must have distinct identities")
    return adapter, tuple(dict(entry) for entry in entries)


def _require_exact_time(actual: object, expected: int, name: str) -> float:
    value = float(actual)
    if not np.isfinite(value) or value != float(expected):
        raise ValueError(f"{name} is {value}, expected exactly {expected}")
    return value


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v3 trajectory: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"stale v3 trajectory temporary exists: {temporary}")
    try:
        with temporary.open("x", newline="", encoding="ascii") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v3 worker receipt: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"stale v3 receipt temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(v3_plan.canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_test_case(
    factory: Callable,
    adapter: boptest.CaseAdapter,
    testcase_root: Path,
):
    test_case = factory(
        str(testcase_root / adapter.case / "models" / "wrapped.fmu"),
        "/boptest/lib/forecast/forecast_uncertainty_params.json",
    )
    status, message, _ = test_case.set_step(boptest.STEP_SECONDS)
    if status != 200:
        raise RuntimeError(message)
    return test_case


def _initialize_branch(
    test_case,
    adapter: boptest.CaseAdapter,
    entry: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object], str, str]:
    scenario_seed = int(entry["scenario_seed"])
    status, message, _ = test_case.set_scenario(
        {
            "electricity_price": "dynamic",
            "time_period": None,
            "temperature_uncertainty": "none",
            "solar_uncertainty": "none",
            "seed": scenario_seed,
        }
    )
    if status != 200:
        raise RuntimeError(message)
    start_time = int(entry["day"]) * v3_plan.DAY_SECONDS
    status, message, state = test_case.initialize(
        start_time, boptest.WARMUP_SECONDS
    )
    if status != 200:
        raise RuntimeError(message)
    status, message, forecast = test_case.get_forecast(
        list(adapter.forecast_keys),
        boptest.TRAJECTORY_STEPS * boptest.STEP_SECONDS,
        boptest.STEP_SECONDS,
    )
    if status != 200:
        raise RuntimeError(message)
    if not isinstance(state, Mapping) or not isinstance(forecast, Mapping):
        raise ValueError("BOPTEST initialization returned a non-mapping response")
    forecast_fields = ("time", *adapter.forecast_keys)
    if set(forecast) != set(forecast_fields):
        raise ValueError("forecast fields differ from the complete requested forecast")
    if any(
        len(forecast[key]) != boptest.TRAJECTORY_STEPS + 1  # type: ignore[arg-type]
        for key in forecast_fields
    ):
        raise ValueError("forecast length differs from the frozen v3 trajectory")
    _require_exact_time(state["time"], start_time, "initialized state time")
    for index, value in enumerate(forecast["time"]):  # type: ignore[union-attr]
        _require_exact_time(
            value,
            start_time + index * boptest.STEP_SECONDS,
            "forecast time",
        )
    return (
        state,
        forecast,
        canonical_simulator_sha256(state),
        canonical_simulator_sha256(forecast),
    )


def _advance_branch(
    test_case,
    adapter: boptest.CaseAdapter,
    entry: Mapping[str, object],
    policy: str,
    state: Mapping[str, object],
    forecast: Mapping[str, object],
) -> list[dict]:
    metadata = entry["policies"][policy]  # type: ignore[index]
    levels = np.asarray(metadata["action_levels"], dtype=float)
    trajectory_seed = int(metadata["trajectory_seed"])
    start_time = int(entry["day"]) * v3_plan.DAY_SECONDS
    rows: list[dict] = []
    for step in range(boptest.TRAJECTORY_STEPS):
        expected_time = start_time + step * boptest.STEP_SECONDS
        state_time = _require_exact_time(state["time"], expected_time, "state time")
        _require_exact_time(
            forecast["time"][step], expected_time, "forecast time"  # type: ignore[index]
        )
        normalized_action = float(levels[step])
        setpoint, action = boptest.action_payload(adapter, normalized_action)
        status, message, next_state = test_case.advance(action)
        if status != 200:
            raise RuntimeError(message)
        if not isinstance(next_state, Mapping):
            raise ValueError("BOPTEST advance returned a non-mapping state")
        next_time = expected_time + boptest.STEP_SECONDS
        _require_exact_time(next_state["time"], next_time, "next-state time")
        _require_exact_time(
            forecast["time"][step + 1],  # type: ignore[index]
            next_time,
            "next forecast time",
        )
        values = (
            adapter.case,
            OUTPUT_ROLE,
            int(entry["day"]),
            trajectory_seed,
            step,
            state_time,
            normalized_action,
            setpoint,
            *canonical_observation(adapter, state),
            *canonical_observation(adapter, next_state),
            *_context(adapter, forecast, step),
            *_context(adapter, forecast, step + 1),
        )
        rows.append(dict(zip(FIELDS, values, strict=True)))
        state = next_state
    return rows


def collect_pair(
    entry: Mapping[str, object],
    adapter: boptest.CaseAdapter,
    output_dir: Path,
    *,
    testcase_root: Path,
    test_case_factory: Callable,
) -> tuple[list[Path], dict]:
    final_paths = {
        policy: output_dir / adapter.case / expected_filename(entry, policy)
        for policy in v3_plan.POLICIES
    }
    if any(os.path.lexists(path) for path in final_paths.values()):
        raise FileExistsError("refusing to overwrite an existing v3 trajectory pair")

    branches = {
        policy: _new_test_case(test_case_factory, adapter, testcase_root)
        for policy in v3_plan.POLICIES
    }
    initialized = {
        policy: _initialize_branch(branches[policy], adapter, entry)
        for policy in v3_plan.POLICIES
    }
    state_hashes = {policy: values[2] for policy, values in initialized.items()}
    forecast_hashes = {policy: values[3] for policy, values in initialized.items()}
    if len(set(state_hashes.values())) != 1:
        raise ValueError("paired v3 initialized-state hashes differ")
    if len(set(forecast_hashes.values())) != 1:
        raise ValueError("paired v3 full-forecast hashes differ")

    written: list[Path] = []
    branch_receipts: dict[str, dict] = {}
    for policy in v3_plan.POLICIES:
        state, forecast, state_sha256, forecast_sha256 = initialized[policy]
        rows = _advance_branch(
            branches[policy],
            adapter,
            entry,
            policy,
            state,
            forecast,
        )
        path = final_paths[policy]
        _write_csv_exclusive(path, rows)
        written.append(path)
        metadata = entry["policies"][policy]  # type: ignore[index]
        branch_receipts[policy] = {
            "trajectory_seed": int(metadata["trajectory_seed"]),
            "action_seed": int(metadata["action_seed"]),
            "action_sha256": metadata["action_sha256"],
            "initialized_state_sha256": state_sha256,
            "full_forecast_sha256": forecast_sha256,
            "path": str(path.relative_to(output_dir)),
            "sha256": v3_plan.sha256_file(path),
            "rows": boptest.TRAJECTORY_STEPS,
        }
    pair_receipt = {
        "window_id": entry["window_id"],
        "day": int(entry["day"]),
        "scenario_seed": int(entry["scenario_seed"]),
        "initialized_state_sha256": next(iter(state_hashes.values())),
        "full_forecast_sha256": next(iter(forecast_hashes.values())),
        "branches": branch_receipts,
    }
    return written, pair_receipt


def collect_plan(
    plan: Mapping[str, object],
    output_dir: Path,
    *,
    plans: Mapping[str, Mapping[str, object]],
    certificate: Mapping[str, object],
    expected_certificate_sha256: str,
    testcase_root: Path = Path("/public-boptest/testcases"),
    worker_image_id: str = boptest.WORKER_IMAGE_ID,
    boptest_version: str = boptest.WORKER_BOPTEST_VERSION,
    test_case_factory: Callable | None = None,
) -> list[Path]:
    if worker_image_id != boptest.WORKER_IMAGE_ID:
        raise ValueError("worker image ID differs from the pinned immutable image")
    if boptest_version != boptest.WORKER_BOPTEST_VERSION:
        raise ValueError("runtime BOPTEST version differs from the pinned version")
    adapter, entries = validate_collection_contract(
        plan,
        plans=plans,
        certificate=certificate,
        expected_certificate_sha256=expected_certificate_sha256,
        testcase_root=testcase_root,
    )
    final_paths = [
        output_dir / adapter.case / expected_filename(entry, policy)
        for entry in entries
        for policy in v3_plan.POLICIES
    ]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("v3 plan maps multiple branches to one trajectory path")
    if any(os.path.lexists(path) for path in final_paths):
        raise FileExistsError("refusing to overwrite an existing v3 trajectory")
    runtime_receipt = receipt_path(output_dir, plan)
    if os.path.lexists(runtime_receipt):
        raise FileExistsError(f"refusing to overwrite v3 worker receipt: {runtime_receipt}")

    if test_case_factory is None:
        from testcase import TestCase

        test_case_factory = TestCase

    written: list[Path] = []
    pair_receipts: list[dict] = []
    for entry in entries:
        pair_paths, pair_receipt = collect_pair(
            entry,
            adapter,
            output_dir,
            testcase_root=testcase_root,
            test_case_factory=test_case_factory,
        )
        written.extend(pair_paths)
        pair_receipts.append(pair_receipt)
        for path in pair_paths:
            print(path, flush=True)

    receipt_payload = {
        "schema": WORKER_RECEIPT_SCHEMA,
        "study_kind": STUDY_KIND,
        "collection_kind": "paired_locked_transport",
        "output_role": OUTPUT_ROLE,
        "row_schema": ROW_SCHEMA,
        "fields": list(FIELDS),
        "case": adapter.case,
        "plan_sha256": plan["plan_sha256"],
        "disjointness_certificate_sha256": expected_certificate_sha256,
        "worker_image_id": worker_image_id,
        "boptest_version": boptest_version,
        "boptest_commit": boptest.BOPTEST_COMMIT,
        "worker_code_sha256": worker_code_hashes(),
        "source_sha256": plan["source_sha256"],
        "pairs": pair_receipts,
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": v3_plan.sha256_file(path),
                "rows": boptest.TRAJECTORY_STEPS,
            }
            for path in written
        ],
    }
    wrapper = {
        "receipt_sha256": v3_plan.canonical_sha256(receipt_payload),
        "receipt": receipt_payload,
    }
    _write_json_exclusive(runtime_receipt, wrapper)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--disjointness-certificate", type=Path, required=True)
    parser.add_argument("--expected-certificate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--testcase-root",
        type=Path,
        default=Path("/public-boptest/testcases"),
    )
    parser.add_argument("--worker-image-id", required=True)
    parser.add_argument(
        "--boptest-version-file", type=Path, default=Path("/version.txt")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require_plain_file(args.plan, "v3 worker plan")
    _require_plain_file(
        args.disjointness_certificate, "v3 disjointness certificate"
    )
    plans = _load_plan_grid(args.plan_root)
    collect_plan(
        _load_json_strict(args.plan),
        args.output,
        plans=plans,
        certificate=_load_json_strict(args.disjointness_certificate),
        expected_certificate_sha256=args.expected_certificate_sha256,
        testcase_root=args.testcase_root,
        worker_image_id=args.worker_image_id,
        boptest_version=args.boptest_version_file.read_text(
            encoding="ascii"
        ).strip(),
    )


if __name__ == "__main__":
    main()
