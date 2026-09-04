from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


BOPTEST_COMMIT = "0f8a467cb1823f005b6512937e9333c65e1e483e"
BOPTEST_API_VERSION = "1.0.0-dev"
BOPTEST_REPOSITORY_URL = "https://github.com/ibpsa/project1-boptest.git"
BOPTEST_LICENSE_NAME = "Revised BSD-3-Clause with enhancements paragraph"
BOPTEST_LICENSE_PATH = "license.md"
BOPTEST_LICENSE_SHA256 = (
    "4f541c48eb7fea31e3bc0cb21256af3467e5d61e0db4392ed8d9b9355e7db95a"
)
WORKER_IMAGE_ID = (
    "sha256:28b5ebfe981237c3d3c381519c7a5feab526ff45603e8825fe43a6293c61910f"
)
WORKER_BOPTEST_VERSION = "1.0.0-dev"
WORKER_RECEIPT_SCHEMA = "boptest-multicase-worker-receipt-v3"
CORPUS_MANIFEST_SCHEMA = "boptest-multicase-clean-corpus-v3"
PRELOCK_REGISTRY_SCHEMA = "boptest-multicase-prelock-registry-v1"
FULL_PLAN_SCHEMA = "boptest-multicase-clean-full-plan-v2"
SMOKE_PLAN_SCHEMA = "boptest-multicase-clean-smoke-view-v2"
COLLECTOR_CODE_FILES = ("protocol.py", "collect.py", "worker_collect.py")
STEP_SECONDS = 900
WARMUP_SECONDS = 86_400
TRAJECTORY_STEPS = 192
START_DAYS = tuple(4 + 9 * index for index in range(40))
PLAN_SEED = 202607310

Role = Literal["fit", "validation", "locked_test"]


@dataclass(frozen=True)
class CaseAdapter:
    case: str
    climate: str
    fmu_sha256: str
    zone_keys: tuple[str, ...]
    power_keys: tuple[str, ...]
    auxiliary_1_keys: tuple[str, ...]
    auxiliary_2_keys: tuple[str, ...]
    auxiliary_1_reduction: Literal["mean", "sum"]
    auxiliary_2_reduction: Literal["mean", "sum"]
    auxiliary_1_name: str
    auxiliary_1_unit: str
    auxiliary_2_name: str
    auxiliary_2_unit: str
    outdoor_key: str
    solar_key: str
    lower_forecast_keys: tuple[str, ...]
    upper_forecast_keys: tuple[str, ...]
    price_forecast_key: str
    action_pairs: tuple[tuple[str, str], ...]
    base_setpoint_k: float
    action_amplitude_k: float

    @property
    def forecast_keys(self) -> tuple[str, ...]:
        return (
            "TDryBul",
            "HGloHor",
            *self.lower_forecast_keys,
            *self.upper_forecast_keys,
            self.price_forecast_key,
        )


CASES: dict[str, CaseAdapter] = {
    "bestest_hydronic_heat_pump": CaseAdapter(
        case="bestest_hydronic_heat_pump",
        climate="Brussels TMY",
        fmu_sha256="674b9500c1c89fdbec64f2c053af8ddcd06afebe63bd7c1b7fc708692a11863e",
        zone_keys=("reaTZon_y",),
        power_keys=("reaPHeaPum_y", "reaPFan_y", "reaPPumEmi_y"),
        auxiliary_1_keys=("reaTSup_y",),
        auxiliary_2_keys=("reaQFloHea_y",),
        auxiliary_1_reduction="mean",
        auxiliary_2_reduction="sum",
        auxiliary_1_name="radiant_supply_temperature",
        auxiliary_1_unit="K",
        auxiliary_2_name="delivered_heating_power",
        auxiliary_2_unit="W",
        outdoor_key="weaSta_reaWeaTDryBul_y",
        solar_key="weaSta_reaWeaHGloHor_y",
        lower_forecast_keys=("LowerSetp[1]",),
        upper_forecast_keys=("UpperSetp[1]",),
        price_forecast_key="PriceElectricPowerDynamic",
        action_pairs=(("oveTSet_activate", "oveTSet_u"),),
        base_setpoint_k=294.15,
        action_amplitude_k=0.75,
    ),
    "twozone_apartment_hydronic": CaseAdapter(
        case="twozone_apartment_hydronic",
        climate="Milan TMY",
        fmu_sha256="88916efe9fbd0543d78c323eb8221692cb4b2411ffa1b84c2dbc9e23a2d19d72",
        zone_keys=("dayZon_reaTRooAir_y", "nigZon_reaTRooAir_y"),
        power_keys=(
            "hydronicSystem_reaPeleHeaPum_y",
            "hydronicSystem_reaPPum_y",
        ),
        auxiliary_1_keys=("dayZon_reaMFloHea_y", "nigZon_reaMFloHea_y"),
        auxiliary_2_keys=("dayZon_reaPowFlooHea_y", "nigZon_reaPowFlooHea_y"),
        auxiliary_1_reduction="sum",
        auxiliary_2_reduction="sum",
        auxiliary_1_name="zone_heating_mass_flow",
        auxiliary_1_unit="kg/s",
        auxiliary_2_name="delivered_floor_heating_power",
        auxiliary_2_unit="W",
        outdoor_key="weatherStation_reaWeaTDryBul_y",
        solar_key="weatherStation_reaWeaHGloHor_y",
        lower_forecast_keys=("LowerSetp[Day]", "LowerSetp[Night]"),
        upper_forecast_keys=("UpperSetp[Day]", "UpperSetp[Night]"),
        price_forecast_key="PriceElectricPowerDynamic",
        action_pairs=(
            (
                "thermostatDayZon_oveTsetZon_activate",
                "thermostatDayZon_oveTsetZon_u",
            ),
            (
                "thermostatNigZon_oveTsetZon_activate",
                "thermostatNigZon_oveTsetZon_u",
            ),
        ),
        base_setpoint_k=294.15,
        action_amplitude_k=0.75,
    ),
    "multizone_office_simple_air": CaseAdapter(
        case="multizone_office_simple_air",
        climate="Chicago TMY3",
        fmu_sha256="27da2cf751bdfa5d3f55ffc29151bebeba04a2f3c151b3ac8a91f084e966afe0",
        zone_keys=tuple(
            f"hvac_reaZon{zone}_TZon_y"
            for zone in ("Cor", "Eas", "Nor", "Sou", "Wes")
        ),
        power_keys=(
            "chi_reaPChi_y",
            "chi_reaPPumDis_y",
            "heaPum_reaPHeaPum_y",
            "heaPum_reaPPumDis_y",
            "hvac_reaAhu_PFanSup_y",
            "hvac_reaAhu_PPumCoo_y",
            "hvac_reaAhu_PPumHea_y",
        ),
        auxiliary_1_keys=("hvac_reaAhu_V_flow_sup_y",),
        auxiliary_2_keys=("hvac_reaAhu_TSup_y",),
        auxiliary_1_reduction="sum",
        auxiliary_2_reduction="mean",
        auxiliary_1_name="ahu_supply_air_flow",
        auxiliary_1_unit="m3/s",
        auxiliary_2_name="ahu_supply_air_temperature",
        auxiliary_2_unit="K",
        outdoor_key="weaSta_reaWeaTDryBul_y",
        solar_key="weaSta_reaWeaHGloHor_y",
        lower_forecast_keys=tuple(
            f"LowerSetp[{zone}]" for zone in ("cor", "eas", "nor", "sou", "wes")
        ),
        upper_forecast_keys=tuple(
            f"UpperSetp[{zone}]" for zone in ("cor", "eas", "nor", "sou", "wes")
        ),
        price_forecast_key="PriceElectricPowerDynamic",
        action_pairs=tuple(
            (
                f"hvac_oveZonSup{zone}_TZonCooSet_activate",
                f"hvac_oveZonSup{zone}_TZonCooSet_u",
            )
            for zone in ("Cor", "Eas", "Nor", "Sou", "Wes")
        ),
        base_setpoint_k=297.15,
        action_amplitude_k=0.75,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collector_code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in COLLECTOR_CODE_FILES}


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def balanced_action_levels(seed: int) -> np.ndarray:
    """Return 24 balanced two-hour PRBS blocks with no adjacent repeat."""
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = np.asarray(
        tuple(itertools.permutations((-1, 0, 1))), dtype=int
    )
    values: list[int] = []
    for _ in range(8):
        candidates = permutations
        if values:
            candidates = candidates[candidates[:, 0] != values[-1]]
        block = candidates[int(rng.integers(len(candidates)))]
        values.extend(int(value) for value in block)
    return np.repeat(np.asarray(values, dtype=float), 8)


def action_payload(adapter: CaseAdapter, normalized_action: float) -> tuple[float, dict[str, float]]:
    if normalized_action not in (-1.0, 0.0, 1.0):
        raise ValueError("normalized action must be -1, 0, or 1")
    setpoint = adapter.base_setpoint_k + adapter.action_amplitude_k * normalized_action
    payload: dict[str, float] = {}
    for activate, value in adapter.action_pairs:
        payload[activate] = 1.0
        payload[value] = float(setpoint)
    return float(setpoint), payload


def _mean_weather_temperature(weather_csv: Path, start_day: int) -> float:
    lower = start_day * 86_400
    upper = (start_day + 2) * 86_400
    values: list[float] = []
    with weather_csv.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            time_s = float(row["time"])
            if lower <= time_s < upper:
                values.append(float(row["TDryBul"]))
    if len(values) != TRAJECTORY_STEPS:
        raise ValueError(
            f"weather window day {start_day} has {len(values)} rows, expected {TRAJECTORY_STEPS}"
        )
    return float(np.mean(values))


def build_case_plan(
    adapter: CaseAdapter,
    testcase_root: Path,
    *,
    plan_seed: int = PLAN_SEED,
) -> dict:
    model_root = testcase_root / adapter.case / "models"
    fmu_path = model_root / "wrapped.fmu"
    weather_path = model_root / "Resources" / "weather.csv"
    license_path = testcase_root.parent / BOPTEST_LICENSE_PATH
    if sha256_file(fmu_path) != adapter.fmu_sha256:
        raise ValueError(f"FMU hash mismatch for {adapter.case}")
    if sha256_file(license_path) != BOPTEST_LICENSE_SHA256:
        raise ValueError("BOPTEST public-source license hash mismatch")
    temperatures = {
        day: _mean_weather_temperature(weather_path, day) for day in START_DAYS
    }
    ordered = sorted(START_DAYS, key=lambda day: (temperatures[day], day))
    validation_counts = (2, 2, 2, 1, 1)
    entries: list[dict] = []
    for stratum in range(5):
        days = ordered[8 * stratum : 8 * (stratum + 1)]
        rng = np.random.Generator(
            np.random.PCG64(stable_seed(plan_seed, adapter.case, stratum))
        )
        shuffled = [int(value) for value in rng.permutation(days)]
        validation_count = validation_counts[stratum]
        roles: Sequence[Role] = (
            ("fit",) * 4
            + ("validation",) * validation_count
            + ("locked_test",) * (4 - validation_count)
        )
        for day, role in zip(shuffled, roles):
            entries.append(
                {
                    "case": adapter.case,
                    "day": day,
                    "role": role,
                    "temperature_stratum": stratum,
                    "mean_outdoor_temperature_k": temperatures[day],
                    "trajectory_seed": stable_seed(plan_seed, adapter.case, day),
                }
            )
    entries.sort(key=lambda item: item["day"])
    return {
        "schema": FULL_PLAN_SCHEMA,
        "mode": "full",
        "boptest_commit": BOPTEST_COMMIT,
        "boptest_api_version": BOPTEST_API_VERSION,
        "public_source": {
            "repository_url": BOPTEST_REPOSITORY_URL,
            "commit": BOPTEST_COMMIT,
            "license_name": BOPTEST_LICENSE_NAME,
            "license_path": BOPTEST_LICENSE_PATH,
            "license_sha256": BOPTEST_LICENSE_SHA256,
        },
        "worker_runtime": {
            "image_id": WORKER_IMAGE_ID,
            "boptest_version": WORKER_BOPTEST_VERSION,
        },
        "collector_code_sha256": collector_code_hashes(),
        "step_seconds": STEP_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "trajectory_steps": TRAJECTORY_STEPS,
        "plan_seed": plan_seed,
        "case_adapter": asdict(adapter),
        "source_sha256": {
            "wrapped_fmu": sha256_file(fmu_path),
            "weather_csv": sha256_file(weather_path),
        },
        "entries": entries,
    }


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def strict_json_loads(content: str) -> object:
    """Parse standards-compliant JSON while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        content,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_prelock_registry_payload(
    registry: object,
    expected_sha256: str,
) -> dict:
    """Bind a pre-lock registry object to a separately supplied canonical digest."""
    if not valid_sha256(expected_sha256):
        raise ValueError("expected pre-lock registry SHA-256 is invalid")
    if not isinstance(registry, dict):
        raise ValueError("pre-lock registry must be a JSON object")
    if registry.get("schema") != PRELOCK_REGISTRY_SCHEMA:
        raise ValueError("pre-lock registry schema mismatch")
    if registry.get("stage") != "prelock":
        raise ValueError("pre-lock registry stage mismatch")
    actual_sha256 = hashlib.sha256(
        canonical_json(registry).encode("ascii")
    ).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("pre-lock registry differs from the externally supplied digest")
    return registry


def plan_sha256(plan: dict) -> str:
    return hashlib.sha256(canonical_json(plan).encode("ascii")).hexdigest()


def attach_plan_sha256(payload: dict) -> dict:
    plan = dict(payload)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def smoke_view(full_plan: dict) -> dict:
    validate_plan_sha256(full_plan)
    if full_plan.get("schema") != FULL_PLAN_SCHEMA or full_plan.get("mode") != "full":
        raise ValueError("smoke views require a frozen full plan")
    fit_entries = [entry for entry in full_plan["entries"] if entry["role"] == "fit"]
    fit_entries.sort(
        key=lambda entry: (
            abs(entry["temperature_stratum"] - 2),
            entry["day"],
        )
    )
    payload = {
        **{key: value for key, value in full_plan.items() if key not in {"entries", "plan_sha256"}},
        "schema": SMOKE_PLAN_SCHEMA,
        "mode": "smoke",
        "parent_full_plan_sha256": full_plan["plan_sha256"],
        "entries": fit_entries[:1],
    }
    return attach_plan_sha256(payload)


def validate_plan_sha256(plan: dict) -> None:
    recorded = plan.get("plan_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("plan is missing a valid plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan_sha256(payload) != recorded:
        raise ValueError("plan SHA-256 does not match its canonical payload")


def validate_plan(
    plan: dict,
    testcase_root: Path,
    allowed_roles: Sequence[Role],
) -> tuple[CaseAdapter, tuple[dict, ...]]:
    """Validate a complete frozen plan and return its permitted entries."""
    validate_plan_sha256(plan)
    if plan.get("boptest_commit") != BOPTEST_COMMIT:
        raise ValueError("plan BOPTEST commit differs from the frozen commit")
    if plan.get("boptest_api_version") != BOPTEST_API_VERSION:
        raise ValueError("plan BOPTEST API version differs from the frozen version")
    if plan.get("worker_runtime") != {
        "image_id": WORKER_IMAGE_ID,
        "boptest_version": WORKER_BOPTEST_VERSION,
    }:
        raise ValueError("plan worker runtime is not the pinned immutable runtime")
    if plan.get("collector_code_sha256") != collector_code_hashes():
        raise ValueError("collector code hashes differ from the frozen plan")
    expected_public_source = {
        "repository_url": BOPTEST_REPOSITORY_URL,
        "commit": BOPTEST_COMMIT,
        "license_name": BOPTEST_LICENSE_NAME,
        "license_path": BOPTEST_LICENSE_PATH,
        "license_sha256": BOPTEST_LICENSE_SHA256,
    }
    if plan.get("public_source") != expected_public_source:
        raise ValueError("plan public-source provenance differs from the frozen source")
    if plan.get("step_seconds") != STEP_SECONDS:
        raise ValueError("plan step_seconds differs from the frozen cadence")
    if plan.get("warmup_seconds") != WARMUP_SECONDS:
        raise ValueError("plan warmup_seconds differs from the frozen warmup")
    if plan.get("trajectory_steps") != TRAJECTORY_STEPS:
        raise ValueError("plan trajectory_steps differs from the frozen length")
    if plan.get("plan_seed") != PLAN_SEED:
        raise ValueError("plan seed differs from the frozen split seed")

    adapter_payload = plan.get("case_adapter")
    if not isinstance(adapter_payload, dict) or adapter_payload.get("case") not in CASES:
        raise ValueError("plan names an unknown case adapter")
    adapter = CASES[adapter_payload["case"]]
    if canonical_json(adapter_payload) != canonical_json(asdict(adapter)):
        raise ValueError("plan adapter differs from the frozen adapter")

    rebuilt_full = attach_plan_sha256(build_case_plan(adapter, testcase_root))
    mode = plan.get("mode")
    if mode == "full":
        if (
            plan.get("schema") != FULL_PLAN_SCHEMA
            or canonical_json(plan) != canonical_json(rebuilt_full)
        ):
            raise ValueError("full plan differs from deterministic regeneration")
    elif mode == "smoke":
        if (
            plan.get("schema") != SMOKE_PLAN_SCHEMA
            or canonical_json(plan) != canonical_json(smoke_view(rebuilt_full))
        ):
            raise ValueError("smoke view differs from its frozen full plan")
    else:
        raise ValueError("plan mode must be full or smoke")

    roles = tuple(dict.fromkeys(allowed_roles))
    role_set = set(roles)
    if mode == "smoke":
        if role_set != {"fit"}:
            raise ValueError("smoke collection permits only the FIT role")
    elif role_set not in ({"fit", "validation"}, {"locked_test"}):
        raise ValueError("full collection must be development-only or locked-only")
    entries = tuple(entry for entry in plan["entries"] if entry["role"] in role_set)
    if not entries:
        raise ValueError("plan contains no entries for the requested role subset")
    if any(entry["case"] != adapter.case for entry in entries):
        raise ValueError("plan entry case differs from its adapter")
    if len({(entry["day"], entry["role"]) for entry in entries}) != len(entries):
        raise ValueError("plan contains duplicate day/role entries")
    return adapter, entries
