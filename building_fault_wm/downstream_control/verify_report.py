"""Independently verify the sealed downstream-control report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import stat
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "artifacts/direct_h8_downstream_control_v1"
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "artifacts/direct_h8_downstream_control_protocol_v1/protocol.json"
)
EXPECTED_PROTOCOL_DIGEST = (
    "07ec27d1da4f254f4883671f0f920e32d66d160aab05d3777a520cf7f17ae035"
)
POLICIES = (
    "constant_zero",
    "legacy_rssm",
    "direct_h8_rssm",
    "deterministic_wm",
)
CONDITIONS = (
    "clean",
    "zone_bias_negative",
    "zone_bias_positive",
    "zone_drift_negative",
    "zone_drift_positive",
    "zone_stuck",
)
CONTRASTS = (
    ("legacy_rssm", "constant_zero"),
    ("direct_h8_rssm", "constant_zero"),
    ("deterministic_wm", "constant_zero"),
    ("direct_h8_rssm", "legacy_rssm"),
    ("deterministic_wm", "direct_h8_rssm"),
)
ENDPOINTS = (
    "cost_tot",
    "tdis_tot",
    "control_cost_proxy",
    "control_discomfort_kh",
    "control_energy_kwh",
)
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1
PCG_MULTIPLIER = (2549297995355413924 << 64) | 4865540595714422341


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(content).hexdigest()


def strict_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"{label} is not finite")
    if abs(actual - expected) > 1e-12 + 1e-12 * abs(expected):
        raise ValueError(f"{label} differs: {actual!r} != {expected!r}")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header is missing: {path}")
        rows = [dict(row) for row in reader]
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV row shape differs: {path}")
    return list(reader.fieldnames), rows


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot reduce an empty value set")
    return float(statistics.median(values))


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot reduce an empty value set")
    return float(statistics.fmean(values))


def quantile_linear(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _hashmix(value: int, hash_constant: int) -> tuple[int, int]:
    value = (value ^ hash_constant) & MASK32
    hash_constant = (hash_constant * 0x931E8875) & MASK32
    value = (value * hash_constant) & MASK32
    return (value ^ (value >> 16)) & MASK32, hash_constant


def _mix(left: int, right: int) -> int:
    value = (0xCA01F9DD * left - 0x4973F715 * right) & MASK32
    return (value ^ (value >> 16)) & MASK32


def _seed_words(entropy: int) -> list[int]:
    words: list[int] = []
    if entropy == 0:
        return [0]
    while entropy:
        words.append(entropy & MASK32)
        entropy >>= 32
    return words


def _pcg64_seed(entropy: int) -> tuple[int, int]:
    """Reproduce the PCG64 seed state used to create the sealed intervals."""
    entropy_words = _seed_words(entropy)
    pool = [0, 0, 0, 0]
    hash_constant = 0x43B0D7E5
    for index in range(4):
        source = entropy_words[index] if index < len(entropy_words) else 0
        pool[index], hash_constant = _hashmix(source, hash_constant)
    for source_index in range(4):
        for destination_index in range(4):
            if source_index == destination_index:
                continue
            hashed, hash_constant = _hashmix(
                pool[source_index], hash_constant
            )
            pool[destination_index] = _mix(pool[destination_index], hashed)
    for source in entropy_words[4:]:
        for destination_index in range(4):
            hashed, hash_constant = _hashmix(source, hash_constant)
            pool[destination_index] = _mix(pool[destination_index], hashed)

    generated: list[int] = []
    hash_constant = 0x8B51F9DD
    for index in range(8):
        value = pool[index % 4] ^ hash_constant
        hash_constant = (hash_constant * 0x58F38DED) & MASK32
        value = (value * hash_constant) & MASK32
        generated.append((value ^ (value >> 16)) & MASK32)
    words64 = [
        generated[index] | (generated[index + 1] << 32)
        for index in range(0, 8, 2)
    ]
    initial_state = (words64[0] << 64) | words64[1]
    initial_sequence = (words64[2] << 64) | words64[3]
    increment = ((initial_sequence << 1) | 1) & MASK128

    def step(state: int) -> int:
        return (state * PCG_MULTIPLIER + increment) & MASK128

    state = step(0)
    state = (state + initial_state) & MASK128
    return step(state), increment


class Pcg64Integers:
    """Minimal PCG64 integer stream matching the sealed bootstrap generator."""

    def __init__(self, seed: int) -> None:
        self.state, self.increment = _pcg64_seed(seed)
        self.cached_uint32: int | None = None

    def next_uint64(self) -> int:
        self.state = (
            self.state * PCG_MULTIPLIER + self.increment
        ) & MASK128
        high = self.state >> 64
        value = (high ^ (self.state & MASK64)) & MASK64
        rotation = high >> 58
        return (
            (value >> rotation) | (value << ((-rotation) & 63))
        ) & MASK64

    def next_uint32(self) -> int:
        if self.cached_uint32 is not None:
            value = self.cached_uint32
            self.cached_uint32 = None
            return value
        value = self.next_uint64()
        self.cached_uint32 = value >> 32
        return value & MASK32

    def below(self, upper_exclusive: int) -> int:
        if not 1 <= upper_exclusive <= MASK32:
            raise ValueError("bounded PCG64 range is unsupported")
        range_inclusive = upper_exclusive - 1
        threshold = (MASK32 - range_inclusive) % upper_exclusive
        while True:
            product = self.next_uint32() * upper_exclusive
            if (product & MASK32) >= threshold:
                return product >> 32


def paired_bootstrap(
    values: list[float], seed: int, draws: int = 10_000
) -> tuple[float, float]:
    generator = Pcg64Integers(seed)
    medians = []
    for _ in range(draws):
        sample = [values[generator.below(len(values))] for _ in values]
        medians.append(median(sample))
    return quantile_linear(medians, 0.025), quantile_linear(medians, 0.975)


def verify_manifest(
    report: Path, expected_report_digest: str | None, require_read_only: bool
) -> str:
    manifest_path = report / "report_manifest.json"
    digest_path = report / "report_manifest.canonical.sha256"
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "direct-h8-downstream-control-report-manifest-v1":
        raise ValueError("report manifest schema differs")
    recorded = digest_path.read_text(encoding="ascii").strip()
    calculated = canonical_sha256(manifest)
    if recorded != calculated:
        raise ValueError("report manifest canonical digest is invalid")
    if expected_report_digest is not None and calculated != expected_report_digest:
        raise ValueError("report digest differs from the expected identity")
    without_payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if manifest.get("payload_sha256") != canonical_sha256(without_payload):
        raise ValueError("report manifest payload digest is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("report manifest file inventory is empty")
    declared = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("report manifest file entry differs")
        relative = str(item["path"])
        if relative in declared or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("report manifest contains an unsafe or duplicate path")
        declared[relative] = item
    excluded = {manifest_path.name, digest_path.name}
    actual = {
        path.relative_to(report).as_posix()
        for path in report.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if actual != set(declared):
        raise ValueError("report file tree differs from its manifest")
    for relative, item in declared.items():
        path = report / relative
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"report byte count differs: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"report file digest differs: {relative}")
    if require_read_only:
        for path in (report, *report.rglob("*")):
            if path.stat().st_mode & WRITE_BITS:
                raise ValueError(f"sealed report remains writable: {path}")
    return calculated


def verify_protocol(protocol_path: Path) -> dict:
    payload = strict_json(protocol_path)
    if canonical_sha256(payload) != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("frozen downstream protocol digest differs")
    if payload.get("schema") != "direct-h8-downstream-control-protocol-v1":
        raise ValueError("downstream protocol schema differs")
    if payload.get("status") != "reviewer_motivated_post_primary_analysis_descriptive":
        raise ValueError("downstream chronology label differs")
    if payload.get("policies") != list(POLICIES):
        raise ValueError("downstream policy set differs")
    if payload.get("conditions") != list(CONDITIONS):
        raise ValueError("downstream condition set differs")
    if payload.get("paired_contrasts") != [
        {"candidate": candidate, "reference": reference}
        for candidate, reference in CONTRASTS
    ]:
        raise ValueError("downstream contrast set differs")
    if len(payload.get("windows", [])) != 12:
        raise ValueError("downstream window set differs")
    return payload


def expected_zone_delta(
    frame: list[dict[str, str]], condition: str
) -> list[float]:
    anchor = float(frame[47]["true_zone_temperature_k"])
    expected = []
    for row in frame:
        step = int(row["step"])
        active = 48 <= step < 96
        if not active or condition == "clean":
            expected.append(0.0)
        elif condition == "zone_bias_negative":
            expected.append(-2.0)
        elif condition == "zone_bias_positive":
            expected.append(2.0)
        elif condition == "zone_drift_negative":
            expected.append(-0.05 * (step - 48 + 1))
        elif condition == "zone_drift_positive":
            expected.append(0.05 * (step - 48 + 1))
        elif condition == "zone_stuck":
            expected.append(anchor - float(row["true_zone_temperature_k"]))
        else:
            raise ValueError(f"unknown condition in report: {condition}")
    return expected


def verify_episode_files(
    report: Path, summaries: list[dict[str, str]], protocol: dict
) -> None:
    expected_days = [int(row["day"]) for row in protocol["windows"]]
    expected_rows = {
        (day, condition, policy)
        for day in expected_days
        for condition in CONDITIONS
        for policy in POLICIES
    }
    actual_rows = {
        (int(row["day"]), row["condition"], row["policy"])
        for row in summaries
    }
    if len(summaries) != 288 or actual_rows != expected_rows:
        raise ValueError("episode summary does not contain the fixed 288 branches")
    for row in summaries:
        if any(not math.isfinite(float(row[endpoint])) for endpoint in ENDPOINTS):
            raise ValueError("episode summary contains a missing endpoint")
    for day in expected_days:
        same_day = [row for row in summaries if int(row["day"]) == day]
        if len({row["initialized_state_sha256"] for row in same_day}) != 1:
            raise ValueError(f"day {day} branches do not share initialization")
        if len({row["forecast_sha256"] for row in same_day}) != 1:
            raise ValueError(f"day {day} branches do not share the forecast")
        reference = [row for row in same_day if row["policy"] == "constant_zero"]
        for endpoint in ENDPOINTS:
            if len({float(row[endpoint]) for row in reference}) != 1:
                raise ValueError(f"constant policy changed with sensor condition: {day}:{endpoint}")
    for summary in summaries:
        day = int(summary["day"])
        condition = str(summary["condition"])
        policy = str(summary["policy"])
        stem = f"day{day:03d}_{condition}_{policy}"
        _, frame = read_csv(report / f"{stem}_trajectory.csv")
        _, decisions = read_csv(report / f"{stem}_decisions.csv")
        if len(frame) != 192 or [int(row["step"]) for row in frame] != list(range(192)):
            raise ValueError(f"trajectory step grid differs: {stem}")
        if any(
            float(row["outcome_time_s"]) - float(row["time_s"]) != 900.0
            for row in frame
        ):
            raise ValueError(f"trajectory time increment differs: {stem}")
        control = [as_bool(row["control_stage"]) for row in frame]
        if sum(control) != 144 or any(control[:48]) or not all(control[48:]):
            raise ValueError(f"control-stage mask differs: {stem}")
        action = [float(row["normalized_action"]) for row in frame]
        if not set(action).issubset({-1.0, 0.0, 1.0}):
            raise ValueError(f"action alphabet differs: {stem}")
        if any(value != 0.0 for value in action[:48]):
            raise ValueError(f"common-history action differs: {stem}")
        for row, action_value in zip(frame, action, strict=True):
            assert_close(
                float(row["setpoint_k"]),
                294.15 + 0.75 * action_value,
                f"setpoint map:{stem}",
            )
        for row, expected in zip(
            frame, expected_zone_delta(frame, condition), strict=True
        ):
            observed = float(row["visible_zone_temperature_k"]) - float(
                row["true_zone_temperature_k"]
            )
            assert_close(observed, expected, f"zone fault waveform:{stem}")
        for visible, true in (
            ("visible_hvac_electric_power_w", "true_hvac_electric_power_w"),
            ("visible_auxiliary_1", "true_auxiliary_1"),
            ("visible_auxiliary_2", "true_auxiliary_2"),
        ):
            if any(float(row[visible]) != float(row[true]) for row in frame):
                raise ValueError(f"non-faulted observation differs: {stem}:{visible}")
        if policy == "constant_zero":
            if any(value != 0.0 for value in action) or decisions:
                raise ValueError(f"constant reference policy differs: {stem}")
            expected_decisions = 0
        else:
            if len(decisions) != 27 or sum(
                as_bool(row["selected"]) for row in decisions
            ) != 9:
                raise ValueError(f"model decision table differs: {stem}")
            expected_decisions = 9
            grouped: dict[int, list[dict[str, str]]] = {}
            for row in decisions:
                grouped.setdefault(int(row["decision_id"]), []).append(row)
            for decision_id, group in sorted(grouped.items()):
                expected_step = 48 + 16 * int(decision_id)
                if (
                    len(group) != 3
                    or {float(row["action_level"]) for row in group}
                    != {-1.0, 0.0, 1.0}
                    or sum(as_bool(row["selected"]) for row in group) != 1
                    or {int(row["decision_step"]) for row in group}
                    != {expected_step}
                ):
                    raise ValueError(f"candidate set differs: {stem}:{decision_id}")
                selected_action = float(
                    next(row["action_level"] for row in group if as_bool(row["selected"]))
                )
                if any(
                    value != selected_action
                    for value in action[expected_step : expected_step + 16]
                ):
                    raise ValueError(f"selected action or dwell differs: {stem}:{decision_id}")
        if int(summary["decisions"]) != expected_decisions:
            raise ValueError(f"episode decision count differs: {stem}")
        selected = [row for row, active in zip(frame, control, strict=True) if active]
        duration_h = 0.25
        cost = duration_h * sum(
            max(float(row["outcome_hvac_electric_power_w"]), 0.0)
            / 1000.0
            * float(row["outcome_electricity_price"])
            for row in selected
        )
        discomfort = duration_h * sum(
            float(row["outcome_discomfort_k"]) for row in selected
        )
        energy = duration_h * sum(
            max(float(row["outcome_hvac_electric_power_w"]), 0.0) / 1000.0
            for row in selected
        )
        assert_close(cost, float(summary["control_cost_proxy"]), f"cost proxy:{stem}")
        assert_close(
            discomfort,
            float(summary["control_discomfort_kh"]),
            f"discomfort proxy:{stem}",
        )
        assert_close(energy, float(summary["control_energy_kwh"]), f"energy:{stem}")
        for level, field in (
            (-1.0, "action_minus_fraction"),
            (0.0, "action_zero_fraction"),
            (1.0, "action_plus_fraction"),
        ):
            assert_close(
                sum(
                    value == level
                    for value, active in zip(action, control, strict=True)
                    if active
                )
                / 144.0,
                float(summary[field]),
                f"action fraction:{stem}:{level}",
            )
        control_actions = [
            value for value, active in zip(action, control, strict=True) if active
        ]
        changes = sum(
            previous != current
            for previous, current in zip(
                [0.0, *control_actions[:-1]], control_actions, strict=True
            )
        )
        if changes != int(summary["action_changes"]):
            raise ValueError(f"action-change count differs: {stem}")


def reconstruct_analysis(
    summaries: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aggregate_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        condition_rows = [
            row for row in summaries if row["condition"] == condition
        ]
        for policy in POLICIES:
            selected = sorted(
                (row for row in condition_rows if row["policy"] == policy),
                key=lambda row: int(row["day"]),
            )
            for endpoint in ENDPOINTS:
                values = [float(row[endpoint]) for row in selected]
                aggregate_rows.append(
                    {
                        "condition": condition,
                        "policy": policy,
                        "endpoint": endpoint,
                        "windows": len(values),
                        "median": median(values),
                        "mean": mean(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
        for candidate_name, reference_name in CONTRASTS:
            candidate = {
                int(row["day"]): row
                for row in condition_rows
                if row["policy"] == candidate_name
            }
            reference = {
                int(row["day"]): row
                for row in condition_rows
                if row["policy"] == reference_name
            }
            if candidate.keys() != reference.keys():
                raise ValueError("paired downstream window identity differs")
            for endpoint in ENDPOINTS:
                differences = [
                    float(candidate[day][endpoint])
                    - float(reference[day][endpoint])
                    for day in sorted(candidate)
                ]
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"{condition}:{candidate_name}:{reference_name}:{endpoint}".encode(
                            "ascii"
                        )
                    ).digest()[:8],
                    "little",
                )
                low, high = paired_bootstrap(differences, seed)
                paired_rows.append(
                    {
                        "condition": condition,
                        "candidate": candidate_name,
                        "reference": reference_name,
                        "endpoint": endpoint,
                        "windows": len(differences),
                        "median_paired_difference": median(differences),
                        "mean_paired_difference": mean(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                        "improved_windows": sum(value < 0.0 for value in differences),
                        "tied_windows": sum(value == 0.0 for value in differences),
                    }
                )
    return aggregate_rows, paired_rows


def compare_rows(
    actual_columns: list[str],
    actual: list[dict[str, str]],
    expected: list[dict[str, object]],
    label: str,
) -> None:
    expected_columns = list(expected[0]) if expected else []
    if actual_columns != expected_columns or len(actual) != len(expected):
        raise ValueError(f"{label} shape or columns differ")
    for row_index, (actual_row, expected_row) in enumerate(
        zip(actual, expected, strict=True)
    ):
        for column in expected_columns:
            expected_value = expected_row[column]
            if isinstance(expected_value, (int, float)):
                assert_close(
                    float(actual_row[column]),
                    float(expected_value),
                    f"{label}:{row_index}:{column}",
                )
            elif actual_row[column] != str(expected_value):
                raise ValueError(f"{label} identity column differs: {column}")


def verify_report(
    report: Path = DEFAULT_REPORT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    expected_report_digest: str | None = None,
    require_read_only: bool = True,
) -> dict[str, object]:
    report_digest = verify_manifest(report, expected_report_digest, require_read_only)
    protocol = verify_protocol(protocol_path)
    metadata = strict_json(report / "run_metadata.json")
    if (
        metadata.get("schema") != "direct-h8-downstream-control-result-v1"
        or metadata.get("pilot") is not False
        or metadata.get("protocol_file_sha256") != sha256_file(protocol_path)
        or metadata.get("protocol_canonical_sha256") != EXPECTED_PROTOCOL_DIGEST
        or metadata.get("episodes") != 288
    ):
        raise ValueError("downstream run metadata differs")
    _, summaries = read_csv(report / "episode_summary.csv")
    verify_episode_files(report, summaries, protocol)
    expected_aggregate, expected_paired = reconstruct_analysis(summaries)
    aggregate_columns, aggregate_rows = read_csv(report / "aggregate_summary.csv")
    compare_rows(
        aggregate_columns,
        aggregate_rows,
        expected_aggregate,
        "aggregate summary",
    )
    paired_columns, paired_rows = read_csv(report / "paired_effects.csv")
    compare_rows(
        paired_columns,
        paired_rows,
        expected_paired,
        "paired effects",
    )
    return {
        "schema": "direct-h8-downstream-control-verification-v1",
        "report_digest": report_digest,
        "protocol_digest": EXPECTED_PROTOCOL_DIGEST,
        "episodes": 288,
        "windows": 12,
        "conditions": len(CONDITIONS),
        "policies": len(POLICIES),
        "trajectory_rows": 288 * 192,
        "reconstructed_aggregate_rows": len(expected_aggregate),
        "reconstructed_paired_rows": len(expected_paired),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-report-digest")
    parser.add_argument("--allow-writable", action="store_true")
    args = parser.parse_args()
    receipt = verify_report(
        args.report,
        args.protocol,
        expected_report_digest=args.expected_report_digest,
        require_read_only=not args.allow_writable,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
