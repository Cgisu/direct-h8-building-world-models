"""Verify a sealed multi-case downstream report from persisted files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from . import protocol


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def verify_manifest(report: Path) -> dict:
    manifest_path = report / "report_manifest.json"
    digest_path = report / "report_manifest.canonical.sha256"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "direct-h8-downstream-multicase-manifest-v2":
        raise ValueError("report manifest schema differs")
    payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if manifest.get("payload_sha256") != protocol.canonical_sha256(payload):
        raise ValueError("report manifest payload digest differs")
    if digest_path.read_text(encoding="ascii").strip() != protocol.canonical_sha256(
        manifest
    ):
        raise ValueError("report manifest canonical digest differs")
    declared = {str(item["path"]): item for item in manifest["files"]}
    actual = {
        path.relative_to(report).as_posix()
        for path in report.rglob("*")
        if path.is_file()
        and path.name
        not in {"report_manifest.json", "report_manifest.canonical.sha256"}
    }
    if set(declared) != actual:
        raise ValueError("report file inventory differs")
    for relative, item in declared.items():
        path = report / relative
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"report byte count differs: {relative}")
        if protocol.sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"report file hash differs: {relative}")
    return manifest


def trajectory_receipt(report: Path, row: dict[str, str], pilot: bool) -> None:
    case = row["case"]
    day = int(row["day"])
    condition = row["condition"]
    policy_name = row["policy"]
    stem = f"{case}_day{day:03d}_{condition}_{policy_name}"
    trajectory = read_rows(report / f"{stem}_trajectory.csv")
    decisions = read_rows(report / f"{stem}_decisions.csv")
    expected_steps = 96 if pilot else protocol.EPISODE_STEPS
    if len(trajectory) != expected_steps:
        raise ValueError(f"trajectory length differs: {stem}")
    if any(
        item["case"] != case
        or item["condition"] != condition
        or item["policy"] != policy_name
        for item in trajectory
    ):
        raise ValueError(f"trajectory identity differs: {stem}")
    control = [item for item in trajectory if item["control_stage"] == "True"]
    if len(control) != expected_steps - protocol.HISTORY_STEPS:
        raise ValueError(f"control-stage length differs: {stem}")
    duration_h = protocol.STEP_SECONDS / 3600.0
    cost_budget = (
        protocol.COST_REFERENCE_POWER_DENSITY_KW_M2
        * duration_h
        * sum(max(float(item["outcome_electricity_price"]), 0.0) for item in control)
    )
    discomfort_budget = duration_h * sum(
        max(
            (
                float(item["outcome_comfort_upper_k"])
                - float(item["outcome_comfort_lower_k"])
            )
            / 2.0,
            0.0,
        )
        for item in control
    )
    control_cost = duration_h * sum(
        max(float(item["outcome_hvac_electric_power_w"]), 0.0)
        / 1000.0
        * float(item["outcome_electricity_price"])
        for item in control
    )
    control_discomfort = duration_h * sum(
        float(item["outcome_discomfort_k"]) for item in control
    )
    control_energy = duration_h * sum(
        max(float(item["outcome_hvac_electric_power_w"]), 0.0) / 1000.0
        for item in control
    )
    for field, actual in (
        ("cost_budget", cost_budget),
        ("discomfort_budget", discomfort_budget),
        ("control_cost_proxy", control_cost),
        ("control_discomfort_kh", control_discomfort),
        ("control_energy_kwh", control_energy),
    ):
        if not close(float(row[field]), actual):
            raise ValueError(f"{field} does not re-derive: {stem}")
    action_values = [float(item["normalized_action"]) for item in control]
    for level, field in (
        (-1.0, "action_minus_fraction"),
        (0.0, "action_zero_fraction"),
        (1.0, "action_plus_fraction"),
    ):
        actual = sum(value == level for value in action_values) / len(action_values)
        if not close(float(row[field]), actual):
            raise ValueError(f"{field} does not re-derive: {stem}")
    expected_decision_rows = (
        0
        if policy_name == "constant_zero"
        else ((expected_steps - protocol.HISTORY_STEPS) // protocol.ACTION_DWELL_STEPS)
        * len(protocol.ACTION_LEVELS)
    )
    if len(decisions) != expected_decision_rows:
        raise ValueError(f"decision row count differs: {stem}")


def verify_scores(rows: list[dict[str, str]]) -> None:
    reference: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row["policy"] == "constant_zero":
            reference[(row["case"], row["day"], row["condition"])] = row
    for row in rows:
        neutral = reference[(row["case"], row["day"], row["condition"])]
        if not close(float(row["cost_budget"]), float(neutral["cost_budget"])):
            raise ValueError("paired cost budget differs")
        if not close(
            float(row["discomfort_budget"]), float(neutral["discomfort_budget"])
        ):
            raise ValueError("paired discomfort budget differs")
        expected = protocol.scalar_score(
            float(row["cost_tot"]),
            float(row["tdis_tot"]),
            float(neutral["cost_tot"]),
            float(neutral["tdis_tot"]),
            float(neutral["cost_budget"]),
            float(neutral["discomfort_budget"]),
        )
        if not close(float(row["operational_score"]), expected):
            raise ValueError("operational score does not re-derive")


def verify_aggregate(report: Path, summaries: list[dict[str, str]]) -> None:
    aggregate = read_rows(report / "aggregate_summary.csv")
    paired = read_rows(report / "paired_effects.csv")
    panel = read_rows(report / "finite_panel_summary.csv")
    if len(aggregate) != 3 * 6 * 6 * 6:
        raise ValueError("aggregate summary grid differs")
    if len(paired) != 3 * 6 * 9 * 6:
        raise ValueError("paired-effect grid differs")
    if len(panel) != 2 * 6 * 6:
        raise ValueError("finite-panel summary grid differs")
    lookup: dict[tuple[str, str, str, str], list[float]] = {}
    for row in summaries:
        for endpoint in (
            "operational_score",
            "cost_tot",
            "tdis_tot",
            "control_cost_proxy",
            "control_discomfort_kh",
            "control_energy_kwh",
        ):
            lookup.setdefault(
                (row["case"], row["condition"], row["policy"], endpoint), []
            ).append(float(row[endpoint]))
    for row in aggregate:
        values = lookup[
            (row["case"], row["condition"], row["policy"], row["endpoint"])
        ]
        if len(values) != 12:
            raise ValueError("aggregate source window count differs")
        if not close(float(row["median"]), statistics.median(values)):
            raise ValueError("aggregate median does not re-derive")
        if not close(float(row["mean"]), statistics.fmean(values)):
            raise ValueError("aggregate mean does not re-derive")


def verify(report: Path) -> dict[str, object]:
    protocol_payload = protocol.validate_frozen_protocol()
    verify_manifest(report)
    metadata = read_json(report / "run_metadata.json")
    if metadata.get("schema") != "direct-h8-downstream-multicase-result-v2":
        raise ValueError("run metadata schema differs")
    if metadata.get("protocol_file_sha256") != protocol.sha256_file(
        protocol.PROTOCOL_PATH
    ):
        raise ValueError("run protocol file receipt differs")
    if metadata.get("protocol_canonical_sha256") != protocol.canonical_sha256(
        protocol_payload
    ):
        raise ValueError("run protocol canonical receipt differs")
    if metadata.get("parallel_workers") != protocol.PARALLEL_WORKERS:
        raise ValueError("run parallel-worker count differs")
    pilot = bool(metadata.get("pilot"))
    summaries = read_rows(report / "episode_summary.csv")
    expected = 36 if pilot else 3 * 12 * 6 * 6
    if len(summaries) != expected or int(metadata.get("episodes", -1)) != expected:
        raise ValueError("episode summary grid differs")
    keys = {
        (row["case"], row["day"], row["condition"], row["policy"])
        for row in summaries
    }
    if len(keys) != expected:
        raise ValueError("episode summary identities are duplicated")
    for row in summaries:
        trajectory_receipt(report, row, pilot)
    verify_scores(summaries)
    if not pilot:
        verify_aggregate(report, summaries)
    return {
        "schema": "direct-h8-downstream-multicase-verification-v2",
        "report": report.name,
        "pilot": pilot,
        "episodes": expected,
        "manifest_sha256": protocol.sha256_file(report / "report_manifest.json"),
        "protocol_sha256": protocol.sha256_file(protocol.PROTOCOL_PATH),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, nargs="?", default=protocol.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.report), indent=2))


if __name__ == "__main__":
    main()
