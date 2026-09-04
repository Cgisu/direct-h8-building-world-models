"""Value-blind window selection and paired action schedules for v3."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from building_fault_wm.neural_benchmark.protocol import (
    CASES,
    STEP_SECONDS,
    TRAJECTORY_STEPS,
    WARMUP_SECONDS,
    balanced_action_levels,
    stable_seed,
)


PLAN_SEED = 202607312
PLAN_SCHEMA = "direct-h8-deterministic-transport-plan-v1"
CERTIFICATE_SCHEMA = "direct-h8-deterministic-transport-certificate-v2"
PRIOR_EVIDENCE_SCHEMA = "direct-h8-prior-trajectory-evidence-v1"
POLICIES = ("old_2h", "new_4h")
BASE_PER_STRATUM = 2
EXTRA_WINDOWS = 2
EXPECTED_WINDOWS_PER_CASE = 12
DAY_SECONDS = 86_400
EVIDENCE_KINDS = frozenset(
    {
        "raw_csv",
        "manifest_json",
        "receipt_json",
        "collection_record_json",
    }
)
EVIDENCE_SCOPE_KINDS = frozenset(
    {"local_multicase_namespace", "immutable_parent_package"}
)


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def normalize_trajectory_identity(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("trajectory identity is not an object")
    case = value.get("case")
    day = value.get("day")
    trajectory_seed = value.get("trajectory_seed")
    if case not in CASES:
        raise ValueError("trajectory identity has an unknown public case")
    if isinstance(day, bool) or not isinstance(day, int) or day < 1:
        raise ValueError("trajectory identity day is invalid")
    if (
        isinstance(trajectory_seed, bool)
        or not isinstance(trajectory_seed, int)
        or trajectory_seed < 0
    ):
        raise ValueError("trajectory identity seed is invalid")
    return {
        "case": str(case),
        "day": day,
        "trajectory_seed": trajectory_seed,
    }


def _identity_key(value: Mapping[str, object]) -> tuple[str, int, int]:
    identity = normalize_trajectory_identity(value)
    return (
        str(identity["case"]),
        int(identity["day"]),
        int(identity["trajectory_seed"]),
    )


def _sorted_identities(
    values: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    unique = {_identity_key(value) for value in values}
    return [
        {"case": case, "day": day, "trajectory_seed": seed}
        for case, day, seed in sorted(unique)
    ]


def build_prior_evidence_contract(
    scope: Sequence[Mapping[str, object]],
    evidence_files: Sequence[Mapping[str, object]],
    *,
    v2_locked_csv_identities: Sequence[Mapping[str, object]],
) -> dict:
    """Build a portable, self-verifying inventory of all known prior identities."""

    normalized_scope = []
    for item in scope:
        if not isinstance(item, Mapping):
            raise ValueError("prior evidence scope row is not an object")
        label = item.get("label")
        kind = item.get("kind")
        present = item.get("present")
        binding = item.get("root_binding_sha256")
        if (
            not isinstance(label, str)
            or not label
            or not label.isascii()
            or "/" in label
        ):
            raise ValueError("prior evidence scope label is invalid")
        if kind not in EVIDENCE_SCOPE_KINDS:
            raise ValueError("prior evidence scope kind is invalid")
        if not isinstance(present, bool):
            raise ValueError("prior evidence scope presence flag is invalid")
        if binding is not None and not valid_sha256(binding):
            raise ValueError("prior evidence root binding is invalid")
        normalized_scope.append(
            {
                "label": label,
                "kind": kind,
                "present": present,
                "root_binding_sha256": binding,
            }
        )
    normalized_scope.sort(key=lambda item: str(item["label"]))
    labels = [str(item["label"]) for item in normalized_scope]
    if len(labels) != len(set(labels)):
        raise ValueError("prior evidence scope labels repeat")
    present_labels = {
        str(item["label"]) for item in normalized_scope if item["present"] is True
    }

    normalized_files = []
    logical_paths = set()
    for item in evidence_files:
        if not isinstance(item, Mapping):
            raise ValueError("prior evidence file row is not an object")
        source = item.get("source")
        path = item.get("path")
        kind = item.get("kind")
        digest = item.get("sha256")
        size = item.get("bytes")
        identities = item.get("identities")
        if source not in present_labels:
            raise ValueError("prior evidence file names an absent or unknown source")
        if (
            not isinstance(path, str)
            or not path
            or not path.isascii()
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ValueError("prior evidence relative path is invalid")
        logical_path = f"{source}/{path}"
        if logical_path in logical_paths:
            raise ValueError("prior evidence file path repeats")
        logical_paths.add(logical_path)
        if kind not in EVIDENCE_KINDS:
            raise ValueError("prior evidence file kind is invalid")
        if not valid_sha256(digest):
            raise ValueError("prior evidence file SHA-256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("prior evidence file size is invalid")
        if not isinstance(identities, Sequence) or isinstance(
            identities, (str, bytes)
        ):
            raise ValueError("prior evidence file identities are invalid")
        normalized_identities = _sorted_identities(identities)
        normalized_files.append(
            {
                "source": source,
                "path": path,
                "kind": kind,
                "sha256": digest,
                "bytes": size,
                "identities": normalized_identities,
                "identity_count": len(normalized_identities),
                "identities_sha256": canonical_sha256(normalized_identities),
            }
        )
    normalized_files.sort(key=lambda item: (str(item["source"]), str(item["path"])))

    known_identities = _sorted_identities(
        identity
        for item in normalized_files
        for identity in item["identities"]  # type: ignore[union-attr]
    )
    known_seeds = sorted(
        {int(identity["trajectory_seed"]) for identity in known_identities}
    )
    v2_identities = _sorted_identities(v2_locked_csv_identities)
    known_keys = {_identity_key(identity) for identity in known_identities}
    if any(_identity_key(identity) not in known_keys for identity in v2_identities):
        raise ValueError("v2 locked CSV identity is absent from prior evidence")
    counts = {
        "scope_records": len(normalized_scope),
        "present_scope_records": len(present_labels),
        "evidence_files": len(normalized_files),
        "evidence_bytes": sum(int(item["bytes"]) for item in normalized_files),
        "identity_occurrences": sum(
            int(item["identity_count"]) for item in normalized_files
        ),
        "known_identities": len(known_identities),
        "known_seeds": len(known_seeds),
        "v2_locked_csv_identities": len(v2_identities),
    }
    payload = {
        "schema": PRIOR_EVIDENCE_SCHEMA,
        "scope": normalized_scope,
        "scope_sha256": canonical_sha256(normalized_scope),
        "inventory": normalized_files,
        "inventory_sha256": canonical_sha256(normalized_files),
        "known_identities": known_identities,
        "known_identities_sha256": canonical_sha256(known_identities),
        "known_seeds": known_seeds,
        "known_seeds_sha256": canonical_sha256(known_seeds),
        "v2_locked_csv_identities": v2_identities,
        "v2_locked_csv_identities_sha256": canonical_sha256(v2_identities),
        "counts": counts,
    }
    return {**payload, "contract_sha256": canonical_sha256(payload)}


def validate_prior_evidence_contract(contract: Mapping[str, object]) -> None:
    if contract.get("schema") != PRIOR_EVIDENCE_SCHEMA:
        raise ValueError("prior trajectory evidence schema mismatch")
    scope = contract.get("scope")
    inventory = contract.get("inventory")
    v2_identities = contract.get("v2_locked_csv_identities")
    if (
        not isinstance(scope, Sequence)
        or isinstance(scope, (str, bytes))
        or not isinstance(inventory, Sequence)
        or isinstance(inventory, (str, bytes))
        or not isinstance(v2_identities, Sequence)
        or isinstance(v2_identities, (str, bytes))
    ):
        raise ValueError("prior trajectory evidence contract is incomplete")
    base_inventory = []
    for item in inventory:
        if not isinstance(item, Mapping):
            raise ValueError("prior trajectory evidence inventory row is invalid")
        base_inventory.append(
            {
                key: item.get(key)
                for key in (
                    "source",
                    "path",
                    "kind",
                    "sha256",
                    "bytes",
                    "identities",
                )
            }
        )
    rebuilt = build_prior_evidence_contract(
        scope,
        base_inventory,
        v2_locked_csv_identities=v2_identities,
    )
    if canonical_bytes(rebuilt) != canonical_bytes(dict(contract)):
        raise ValueError("prior trajectory evidence contract does not self-verify")


def selected_trajectory_identities(
    v3_plans: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    identities = []
    for case in sorted(CASES):
        plan = v3_plans.get(case)
        entries = plan.get("entries") if isinstance(plan, Mapping) else None
        if not isinstance(entries, list):
            raise ValueError(f"v3 plan entries are incomplete for {case}")
        for entry in entries:
            policies = entry.get("policies") if isinstance(entry, Mapping) else None
            if not isinstance(policies, Mapping) or set(policies) != set(POLICIES):
                raise ValueError(f"v3 policy identity grid is incomplete for {case}")
            for policy in POLICIES:
                metadata = policies[policy]
                if not isinstance(metadata, Mapping):
                    raise ValueError("v3 policy identity metadata are invalid")
                identities.append(
                    normalize_trajectory_identity(
                        {
                            "case": case,
                            "day": entry.get("day"),
                            "trajectory_seed": metadata.get("trajectory_seed"),
                        }
                    )
                )
    if len(identities) != len({_identity_key(item) for item in identities}):
        raise ValueError("v3 trajectory identities repeat")
    seeds = [int(item["trajectory_seed"]) for item in identities]
    if len(seeds) != len(set(seeds)):
        raise ValueError("v3 trajectory identity seeds repeat")
    return sorted(
        identities,
        key=lambda item: (
            str(item["case"]),
            int(item["day"]),
            int(item["trajectory_seed"]),
        ),
    )


def build_identity_disjointness_proof(
    v3_plans: Mapping[str, Mapping[str, object]],
    prior_evidence: Mapping[str, object],
) -> tuple[dict, dict[str, dict]]:
    validate_prior_evidence_contract(prior_evidence)
    selected = selected_trajectory_identities(v3_plans)
    prior = prior_evidence["known_identities"]
    prior_seeds = prior_evidence["known_seeds"]
    assert isinstance(prior, list)
    assert isinstance(prior_seeds, list)
    selected_keys = {_identity_key(item) for item in selected}
    prior_keys = {_identity_key(item) for item in prior}
    selected_seed_set = {int(item["trajectory_seed"]) for item in selected}
    prior_seed_set = {int(seed) for seed in prior_seeds}
    selected_days = {(str(item["case"]), int(item["day"])) for item in selected}
    prior_days = {(str(item["case"]), int(item["day"])) for item in prior}
    if selected_days & prior_days:
        raise ValueError("a selected v3 day occurs in prior trajectory evidence")
    if selected_keys & prior_keys:
        raise ValueError("a selected v3 trajectory identity occurs in prior evidence")
    if selected_seed_set & prior_seed_set:
        raise ValueError("a selected v3 trajectory seed occurs in prior evidence")

    proof = {
        "selected_identity_count": len(selected),
        "selected_identities_sha256": canonical_sha256(selected),
        "selected_seed_count": len(selected_seed_set),
        "selected_seeds_sha256": canonical_sha256(sorted(selected_seed_set)),
        "known_prior_identity_count": len(prior),
        "known_prior_identities_sha256": canonical_sha256(prior),
        "known_prior_seed_count": len(prior_seed_set),
        "known_prior_seeds_sha256": canonical_sha256(sorted(prior_seed_set)),
        "no_selected_day_in_prior_evidence": True,
        "no_selected_identity_in_prior_evidence": True,
        "no_selected_seed_in_prior_evidence": True,
    }
    by_case = {}
    for case in sorted(CASES):
        selected_case = [item for item in selected if item["case"] == case]
        prior_case = [item for item in prior if item["case"] == case]
        prior_case_days = sorted({int(item["day"]) for item in prior_case})
        by_case[case] = {
            "selected_identity_count": len(selected_case),
            "selected_identities_sha256": canonical_sha256(selected_case),
            "known_prior_identity_count": len(prior_case),
            "known_prior_identities_sha256": canonical_sha256(prior_case),
            "prior_observed_days": prior_case_days,
            "prior_observed_days_sha256": canonical_sha256(prior_case_days),
            "no_selected_day_in_prior_evidence": True,
            "no_selected_identity_in_prior_evidence": True,
            "no_selected_seed_in_prior_evidence": True,
        }
    return proof, by_case


def _rank(prefix: str, case: str, entry: Mapping[str, object]) -> str:
    payload = (
        f"{prefix}:{PLAN_SEED}:{case}:{int(entry['temperature_stratum'])}:"
        f"{int(entry['day'])}:{entry['role']}:{int(entry['trajectory_seed'])}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def exposure_interval(day: int) -> tuple[int, int]:
    if isinstance(day, bool) or not isinstance(day, int) or day < 1:
        raise ValueError("trajectory day must be a positive integer")
    return (
        day * DAY_SECONDS - WARMUP_SECONDS,
        day * DAY_SECONDS + TRAJECTORY_STEPS * STEP_SECONDS,
    )


def intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def four_hour_action_levels(seed: int) -> np.ndarray:
    """Return 12 balanced four-hour blocks with no adjacent repeat."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("action seed must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = np.asarray(
        tuple(itertools.permutations((-1, 0, 1))), dtype=np.int8
    )
    blocks: list[int] = []
    for _ in range(4):
        candidates = permutations
        if blocks:
            candidates = candidates[candidates[:, 0] != blocks[-1]]
        selected = candidates[int(rng.integers(len(candidates)))]
        blocks.extend(int(value) for value in selected)
    values = np.repeat(np.asarray(blocks, dtype=float), 16)
    _validate_policy_array(values, block_steps=16, blocks_per_level=4)
    return values


def _validate_policy_array(
    values: np.ndarray, *, block_steps: int, blocks_per_level: int
) -> None:
    if values.shape != (TRAJECTORY_STEPS,) or not np.isfinite(values).all():
        raise ValueError("action policy must cover one finite trajectory")
    if not np.isin(values, (-1.0, 0.0, 1.0)).all():
        raise ValueError("action policy leaves the frozen action alphabet")
    counts = Counter(float(value) for value in values)
    if counts != {-1.0: 64, 0.0: 64, 1.0: 64}:
        raise ValueError("action policy marginal counts are not balanced")
    blocks = values[::block_steps]
    expected_blocks = 3 * blocks_per_level
    if len(blocks) != expected_blocks:
        raise ValueError("action policy block count is invalid")
    if any(
        not np.equal(
            values[index * block_steps : (index + 1) * block_steps],
            blocks[index],
        ).all()
        for index in range(expected_blocks)
    ):
        raise ValueError("action policy changes inside a frozen dwell block")
    if np.equal(blocks[:-1], blocks[1:]).any():
        raise ValueError("adjacent action blocks repeat")
    if Counter(float(value) for value in blocks) != {
        -1.0: blocks_per_level,
        0.0: blocks_per_level,
        1.0: blocks_per_level,
    }:
        raise ValueError("action policy block counts are not balanced")


def policy_levels(policy: str, seed: int) -> np.ndarray:
    if policy == "old_2h":
        values = balanced_action_levels(seed)
        _validate_policy_array(values, block_steps=8, blocks_per_level=8)
        return values
    if policy == "new_4h":
        return four_hour_action_levels(seed)
    raise ValueError(f"unknown v3 action policy: {policy}")


def transition_counts(values: np.ndarray) -> dict[str, int]:
    counts = {
        f"{int(left):+d}->{int(right):+d}": 0
        for left in (-1, 0, 1)
        for right in (-1, 0, 1)
    }
    for left, right in zip(values[:-1], values[1:], strict=True):
        counts[f"{int(left):+d}->{int(right):+d}"] += 1
    return counts


def _candidate_pool(parent_plan: Mapping[str, object]) -> tuple[str, list[dict]]:
    adapter = parent_plan.get("case_adapter")
    entries = parent_plan.get("entries")
    if not isinstance(adapter, dict) or not isinstance(entries, list):
        raise ValueError("parent v2 plan is incomplete")
    case = adapter.get("case")
    if case not in CASES:
        raise ValueError("parent v2 plan has an unknown case")
    pool = [
        dict(entry)
        for entry in entries
        if isinstance(entry, dict) and entry.get("role") in {"fit", "validation"}
    ]
    if len(pool) != 28:
        raise ValueError("parent v2 plan does not expose 28 uncollected candidates")
    if len({int(entry["day"]) for entry in pool}) != len(pool):
        raise ValueError("parent v2 candidate days are not unique")
    strata = Counter(int(entry["temperature_stratum"]) for entry in pool)
    if set(strata) != set(range(5)) or any(count < 4 for count in strata.values()):
        raise ValueError("parent v2 candidate temperature strata are incomplete")
    return str(case), pool


def select_windows(parent_plan: Mapping[str, object]) -> list[dict]:
    case, pool = _candidate_pool(parent_plan)
    selected: list[dict] = []
    for stratum in range(5):
        candidates = [
            entry for entry in pool if int(entry["temperature_stratum"]) == stratum
        ]
        candidates.sort(key=lambda entry: (_rank("v3-window", case, entry), int(entry["day"])))
        selected.extend(candidates[:BASE_PER_STRATUM])

    selected_days = {int(entry["day"]) for entry in selected}
    remaining = [entry for entry in pool if int(entry["day"]) not in selected_days]
    remaining.sort(key=lambda entry: (_rank("v3-extra", case, entry), int(entry["day"])))
    first = remaining[0]
    second = next(
        entry
        for entry in remaining[1:]
        if int(entry["temperature_stratum"])
        != int(first["temperature_stratum"])
    )
    selected.extend((first, second))
    if len(selected) != EXPECTED_WINDOWS_PER_CASE:
        raise AssertionError("v3 window selection has the wrong size")

    result = []
    for entry in sorted(selected, key=lambda item: int(item["day"])):
        day = int(entry["day"])
        scenario_seed = stable_seed(PLAN_SEED, case, day, "scenario")
        policies = {}
        for policy in POLICIES:
            identity_seed = stable_seed(PLAN_SEED, case, day, policy, "identity")
            action_seed = stable_seed(PLAN_SEED, case, day, policy, "actions")
            values = policy_levels(policy, action_seed)
            policies[policy] = {
                "trajectory_seed": identity_seed,
                "action_seed": action_seed,
                "action_levels": [int(value) for value in values],
                "action_sha256": hashlib.sha256(
                    values.astype(np.int8).tobytes()
                ).hexdigest(),
                "transition_counts": transition_counts(values),
                "dwell_steps": 8 if policy == "old_2h" else 16,
            }
        start, stop = exposure_interval(day)
        result.append(
            {
                "window_id": f"{case}:day{day:03d}",
                "case": case,
                "day": day,
                "role": "locked_transport",
                "source_plan_role": str(entry["role"]),
                "source_plan_trajectory_seed": int(entry["trajectory_seed"]),
                "temperature_stratum": int(entry["temperature_stratum"]),
                "mean_outdoor_temperature_k": float(
                    entry["mean_outdoor_temperature_k"]
                ),
                "scenario_seed": scenario_seed,
                "exposure_start_s": start,
                "exposure_stop_s": stop,
                "policies": policies,
            }
        )
    return result


def build_case_plan(parent_plan: Mapping[str, object]) -> dict:
    case, _ = _candidate_pool(parent_plan)
    entries = select_windows(parent_plan)
    payload = {
        "schema": PLAN_SCHEMA,
        "plan_seed": PLAN_SEED,
        "case": case,
        "parent_v2_plan_sha256": parent_plan.get("plan_sha256"),
        "source_sha256": parent_plan.get("source_sha256"),
        "case_adapter": parent_plan.get("case_adapter"),
        "step_seconds": STEP_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "trajectory_steps": TRAJECTORY_STEPS,
        "policies": list(POLICIES),
        "selection_rule": {
            "base_per_temperature_stratum": BASE_PER_STRATUM,
            "extra_windows": EXTRA_WINDOWS,
            "extra_windows_distinct_strata": True,
            "response_values_used": False,
        },
        "entries": entries,
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}


def _entry_intervals(entries: Iterable[Mapping[str, object]]) -> list[tuple[int, int]]:
    return [exposure_interval(int(entry["day"])) for entry in entries]


def build_disjointness_certificate(
    v1_plans: Mapping[str, Mapping[str, object]],
    v2_plans: Mapping[str, Mapping[str, object]],
    v3_plans: Mapping[str, Mapping[str, object]],
    *,
    prior_evidence: Mapping[str, object],
) -> dict:
    if set(v1_plans) != set(CASES) or set(v2_plans) != set(CASES):
        raise ValueError("parent plan grids do not cover all public cases")
    if set(v3_plans) != set(CASES):
        raise ValueError("v3 plan grid does not cover all public cases")
    identity_proof, identity_by_case = build_identity_disjointness_proof(
        v3_plans, prior_evidence
    )
    v2_observed = prior_evidence.get("v2_locked_csv_identities")
    if not isinstance(v2_observed, list):
        raise ValueError("prior evidence omits v2 locked CSV identities")
    v2_observed_by_case = {
        case: {
            _identity_key(identity)
            for identity in v2_observed
            if identity.get("case") == case
        }
        for case in CASES
    }
    cases = {}
    for case in sorted(CASES):
        v1_entries = v1_plans[case].get("entries")
        v2_entries = v2_plans[case].get("entries")
        v3_entries = v3_plans[case].get("entries")
        if not all(isinstance(value, list) for value in (v1_entries, v2_entries, v3_entries)):
            raise ValueError(f"plan entries are incomplete for {case}")
        assert isinstance(v1_entries, list)
        assert isinstance(v2_entries, list)
        assert isinstance(v3_entries, list)
        selected_days = {int(entry["day"]) for entry in v3_entries}
        v2_locked_days = {
            int(entry["day"])
            for entry in v2_entries
            if entry.get("role") == "locked_test"
        }
        expected_v2_identities = {
            _identity_key(entry)
            for entry in v2_entries
            if entry.get("role") == "locked_test"
        }
        observed_v2_identities = v2_observed_by_case[case]
        if observed_v2_identities != expected_v2_identities:
            raise ValueError(
                f"v2 disk identity inventory differs from its locked plan for {case}"
            )
        disk_days = {identity[1] for identity in observed_v2_identities}
        if selected_days & disk_days:
            raise ValueError(f"a selected v3 day was already collected for {case}")
        selected_intervals = _entry_intervals(v3_entries)
        v1_intervals = _entry_intervals(v1_entries)
        v2_locked_intervals = _entry_intervals(
            entry for entry in v2_entries if entry.get("role") == "locked_test"
        )
        if any(
            intervals_overlap(selected, previous)
            for selected in selected_intervals
            for previous in (*v1_intervals, *v2_locked_intervals)
        ):
            raise ValueError(f"a selected v3 exposure overlaps prior data for {case}")
        cases[case] = {
            "selected_days": sorted(selected_days),
            "v2_locked_days": sorted(v2_locked_days),
            "selected_window_count": len(v3_entries),
            "selected_intervals_sha256": canonical_sha256(selected_intervals),
            "all_selected_vs_v1_disjoint": True,
            "all_selected_vs_v2_locked_disjoint": True,
            "no_selected_day_previously_collected_in_v2": True,
            **identity_by_case[case],
        }
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "v1_plan_sha256_by_case": {
            case: v1_plans[case].get("plan_sha256") for case in sorted(CASES)
        },
        "v2_plan_sha256_by_case": {
            case: v2_plans[case].get("plan_sha256") for case in sorted(CASES)
        },
        "v3_plan_sha256_by_case": {
            case: v3_plans[case].get("plan_sha256") for case in sorted(CASES)
        },
        "prior_evidence": prior_evidence,
        "identity_proof": identity_proof,
        "cases": cases,
    }
    return {**payload, "certificate_sha256": canonical_sha256(payload)}


def validate_case_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("v3 plan schema mismatch")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != canonical_sha256(payload):
        raise ValueError("v3 plan canonical SHA-256 mismatch")
    case = plan.get("case")
    entries = plan.get("entries")
    if case not in CASES or not isinstance(entries, list):
        raise ValueError("v3 plan case or entries are invalid")
    if len(entries) != EXPECTED_WINDOWS_PER_CASE:
        raise ValueError("v3 plan window count is invalid")
    if len({entry["window_id"] for entry in entries}) != len(entries):
        raise ValueError("v3 plan window IDs are not unique")
    strata = Counter(int(entry["temperature_stratum"]) for entry in entries)
    if set(strata) != set(range(5)) or min(strata.values()) < 2:
        raise ValueError("v3 plan temperature coverage is invalid")
    for entry in entries:
        if entry.get("case") != case or entry.get("role") != "locked_transport":
            raise ValueError("v3 entry identity is invalid")
        if entry.get("source_plan_role") not in {"fit", "validation"}:
            raise ValueError("v3 entry was not drawn from an unused source role")
        policies = entry.get("policies")
        if not isinstance(policies, dict) or set(policies) != set(POLICIES):
            raise ValueError("v3 entry policy grid is incomplete")
        for policy in POLICIES:
            metadata = policies[policy]
            if not isinstance(metadata, dict):
                raise ValueError("v3 policy metadata are invalid")
            levels = np.asarray(metadata.get("action_levels"), dtype=float)
            expected = policy_levels(policy, int(metadata["action_seed"]))
            if not np.array_equal(levels, expected):
                raise ValueError("v3 frozen action array changed")
            if metadata.get("action_sha256") != hashlib.sha256(
                expected.astype(np.int8).tobytes()
            ).hexdigest():
                raise ValueError("v3 frozen action SHA-256 changed")
            if metadata.get("transition_counts") != transition_counts(expected):
                raise ValueError("v3 frozen transition counts changed")
        numeric = np.asarray(
            [
                entry["mean_outdoor_temperature_k"],
                entry["exposure_start_s"],
                entry["exposure_stop_s"],
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all() or numeric[2] <= numeric[1]:
            raise ValueError("v3 entry numeric metadata are invalid")


def load_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="ascii"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
