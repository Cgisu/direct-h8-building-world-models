from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from building_fault_wm.neural_benchmark.protocol import (
    STEP_SECONDS,
    TRAJECTORY_STEPS,
    WARMUP_SECONDS,
    WORKER_BOPTEST_VERSION,
    WORKER_IMAGE_ID,
    WORKER_RECEIPT_SCHEMA,
    action_payload,
    balanced_action_levels,
    canonical_json,
    collector_code_hashes,
    sha256_file,
    strict_json_loads,
    validate_prelock_registry_payload,
    validate_plan,
)


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
ROW_SCHEMA = "boptest-canonical-transition-v2"


def _reduce(state: dict, keys: Iterable[str], reduction: str) -> float:
    values = np.asarray([float(state[key]) for key in keys], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("BOPTEST state contains a non-finite canonical observation")
    if reduction == "mean":
        return float(values.mean())
    if reduction == "sum":
        return float(values.sum())
    raise ValueError(f"unknown canonical reduction {reduction}")


def canonical_observation(adapter, state: dict) -> tuple[float, float, float, float]:
    return (
        _reduce(state, adapter.zone_keys, "mean"),
        _reduce(state, adapter.power_keys, "sum"),
        _reduce(state, adapter.auxiliary_1_keys, adapter.auxiliary_1_reduction),
        _reduce(state, adapter.auxiliary_2_keys, adapter.auxiliary_2_reduction),
    )


def _forecast_value(forecast: dict, keys: Iterable[str], index: int) -> float:
    values = np.asarray([float(forecast[key][index]) for key in keys], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("BOPTEST forecast contains a non-finite value")
    return float(values.mean())


def _context(adapter, forecast: dict, index: int) -> tuple[float, float, float, float, float]:
    return (
        _forecast_value(forecast, ("TDryBul",), index),
        _forecast_value(forecast, ("HGloHor",), index),
        _forecast_value(forecast, adapter.lower_forecast_keys, index),
        _forecast_value(forecast, adapter.upper_forecast_keys, index),
        _forecast_value(forecast, (adapter.price_forecast_key,), index),
    )


def expected_filename(entry: dict) -> str:
    return (
        f"day{int(entry['day']):03d}_{entry['role']}_"
        f"seed{int(entry['trajectory_seed'])}.csv"
    )


def collection_kind(plan: dict, allowed_roles: Iterable[str]) -> str:
    roles = set(allowed_roles)
    if plan["mode"] == "smoke" and roles == {"fit"}:
        return "smoke"
    if roles == {"fit", "validation"}:
        return "development"
    if roles == {"locked_test"}:
        return "locked_test"
    raise ValueError("unsupported plan mode and role subset")


def receipt_path(
    output_dir: Path,
    plan: dict,
    allowed_roles: Iterable[str],
) -> Path:
    kind = collection_kind(plan, allowed_roles)
    case = plan["case_adapter"]["case"]
    return output_dir / "_receipts" / f"{case}_{kind}_{plan['plan_sha256'][:16]}.json"


def _require_exact_time(actual: object, expected: int, name: str) -> float:
    value = float(actual)
    if not np.isfinite(value) or value != float(expected):
        raise ValueError(f"{name} is {value}, expected exactly {expected}")
    return value


def _write_csv_exclusive(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite collected trajectory: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary trajectory exists: {temporary}")
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


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite worker receipt: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary receipt exists: {temporary}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_plan(
    plan: dict,
    output_dir: Path,
    *,
    allowed_roles: tuple[str, ...],
    testcase_root: Path = Path("/cases"),
    worker_image_id: str = WORKER_IMAGE_ID,
    boptest_version: str = WORKER_BOPTEST_VERSION,
    prelock_registry: dict | None = None,
    expected_prelock_sha256: str | None = None,
    test_case_factory: Callable | None = None,
) -> list[Path]:
    if worker_image_id != WORKER_IMAGE_ID:
        raise ValueError("worker image ID differs from the pinned immutable image")
    if boptest_version != WORKER_BOPTEST_VERSION:
        raise ValueError("runtime BOPTEST version differs from the pinned version")
    adapter, entries = validate_plan(plan, testcase_root, allowed_roles)
    kind = collection_kind(plan, allowed_roles)
    if kind == "locked_test":
        validate_prelock_registry_payload(
            prelock_registry,
            expected_prelock_sha256,
        )
        prelock_registry_sha256 = expected_prelock_sha256
    else:
        if prelock_registry is not None or expected_prelock_sha256 is not None:
            raise ValueError("non-locked collection cannot carry pre-lock inputs")
        prelock_registry_sha256 = None
    final_paths = [output_dir / adapter.case / expected_filename(entry) for entry in entries]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("plan maps multiple entries to the same trajectory path")
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {len(existing)} trajectories")
    runtime_receipt = receipt_path(output_dir, plan, allowed_roles)
    if runtime_receipt.exists():
        raise FileExistsError(f"refusing to overwrite worker receipt: {runtime_receipt}")

    if test_case_factory is None:
        from testcase import TestCase

        test_case_factory = TestCase

    test_case = test_case_factory(
        str(testcase_root / adapter.case / "models" / "wrapped.fmu"),
        "/boptest/lib/forecast/forecast_uncertainty_params.json",
    )
    status, message, _ = test_case.set_step(STEP_SECONDS)
    if status != 200:
        raise RuntimeError(message)
    written: list[Path] = []
    for entry, final_path in zip(entries, final_paths):
        seed = int(entry["trajectory_seed"])
        status, message, _ = test_case.set_scenario(
            {
                "electricity_price": "dynamic",
                "time_period": None,
                "temperature_uncertainty": "none",
                "solar_uncertainty": "none",
                "seed": seed,
            }
        )
        if status != 200:
            raise RuntimeError(message)
        start_time = int(entry["day"]) * 86_400
        status, message, state = test_case.initialize(start_time, WARMUP_SECONDS)
        if status != 200:
            raise RuntimeError(message)
        status, message, forecast = test_case.get_forecast(
            list(adapter.forecast_keys),
            TRAJECTORY_STEPS * STEP_SECONDS,
            STEP_SECONDS,
        )
        if status != 200:
            raise RuntimeError(message)
        forecast_fields = ("time", *adapter.forecast_keys)
        if any(len(forecast[key]) != TRAJECTORY_STEPS + 1 for key in forecast_fields):
            raise ValueError("forecast length differs from the frozen trajectory")
        levels = balanced_action_levels(seed)
        rows: list[dict] = []
        for step in range(TRAJECTORY_STEPS):
            expected_time = start_time + step * STEP_SECONDS
            state_time = _require_exact_time(state["time"], expected_time, "state time")
            _require_exact_time(forecast["time"][step], expected_time, "forecast time")
            normalized_action = float(levels[step])
            setpoint, payload = action_payload(adapter, normalized_action)
            status, message, next_state = test_case.advance(payload)
            if status != 200:
                raise RuntimeError(message)
            next_time = expected_time + STEP_SECONDS
            _require_exact_time(next_state["time"], next_time, "next-state time")
            _require_exact_time(
                forecast["time"][step + 1], next_time, "next forecast time"
            )
            observation = canonical_observation(adapter, state)
            next_observation = canonical_observation(adapter, next_state)
            context = _context(adapter, forecast, step)
            next_context = _context(adapter, forecast, step + 1)
            rows.append(
                dict(
                    zip(
                        FIELDS,
                        (
                            adapter.case,
                            entry["role"],
                            int(entry["day"]),
                            seed,
                            step,
                            state_time,
                            normalized_action,
                            setpoint,
                            *observation,
                            *next_observation,
                            *context,
                            *next_context,
                        ),
                    )
                )
            )
            state = next_state
        _write_csv_exclusive(final_path, rows)
        written.append(final_path)
        print(final_path, flush=True)

    receipt_payload = {
        "schema": WORKER_RECEIPT_SCHEMA,
        "collection_kind": kind,
        "allowed_roles": sorted(set(allowed_roles)),
        "case": adapter.case,
        "plan_sha256": plan["plan_sha256"],
        "worker_image_id": worker_image_id,
        "boptest_version": boptest_version,
        "collector_code_sha256": collector_code_hashes(),
        "source_sha256": plan["source_sha256"],
        "prelock_registry_sha256": prelock_registry_sha256,
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "rows": TRAJECTORY_STEPS,
            }
            for path in written
        ],
    }
    wrapper = {
        "receipt_sha256": hashlib.sha256(
            canonical_json(receipt_payload).encode("ascii")
        ).hexdigest(),
        "receipt": receipt_payload,
    }
    _write_json_exclusive(runtime_receipt, wrapper)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one frozen BOPTEST case plan")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-role", action="append", required=True)
    parser.add_argument("--testcase-root", type=Path, default=Path("/cases"))
    parser.add_argument("--worker-image-id", required=True)
    parser.add_argument(
        "--boptest-version-file", type=Path, default=Path("/version.txt")
    )
    parser.add_argument("--prelock-registry", type=Path)
    parser.add_argument("--expected-prelock-sha256")
    args = parser.parse_args()
    plan = strict_json_loads(args.plan.read_text(encoding="ascii"))
    if not isinstance(plan, dict):
        raise ValueError("worker plan must be a JSON object")
    version = args.boptest_version_file.read_text(encoding="ascii").strip()
    prelock_registry = (
        strict_json_loads(args.prelock_registry.read_text(encoding="ascii"))
        if args.prelock_registry is not None
        else None
    )
    collect_plan(
        plan,
        args.output,
        allowed_roles=tuple(args.allowed_role),
        testcase_root=args.testcase_root,
        worker_image_id=args.worker_image_id,
        boptest_version=version,
        prelock_registry=prelock_registry,
        expected_prelock_sha256=args.expected_prelock_sha256,
    )


if __name__ == "__main__":
    main()
