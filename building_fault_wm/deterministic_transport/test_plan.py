from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from building_fault_wm.neural_benchmark.protocol import CASES

from .plan import (
    POLICIES,
    build_case_plan,
    build_disjointness_certificate,
    build_prior_evidence_contract,
    four_hour_action_levels,
    intervals_overlap,
    policy_levels,
    select_windows,
    validate_case_plan,
)


def _parent_plan(case: str, *, day_offset: int = 9) -> dict:
    entries = []
    for index in range(40):
        stratum = index // 8
        within = index % 8
        validation_count = 2 if stratum < 3 else 1
        role = (
            "fit"
            if within < 4
            else "validation"
            if within < 4 + validation_count
            else "locked_test"
        )
        entries.append(
            {
                "case": case,
                "day": day_offset + 9 * index,
                "role": role,
                "temperature_stratum": stratum,
                "mean_outdoor_temperature_k": 275.0 + index,
                "trajectory_seed": 10_000 + index,
            }
        )
    return {
        "plan_sha256": f"{1:064x}",
        "source_sha256": {"wrapped_fmu": f"{2:064x}", "weather_csv": f"{3:064x}"},
        "case_adapter": {"case": case},
        "entries": entries,
    }


def _prior_evidence(v2: dict[str, dict], extra: list[dict] | None = None) -> dict:
    locked = [
        {
            "case": case,
            "day": entry["day"],
            "trajectory_seed": entry["trajectory_seed"],
        }
        for case in sorted(CASES)
        for entry in v2[case]["entries"]
        if entry["role"] == "locked_test"
    ]
    identities = [*locked, *(extra or [])]
    return build_prior_evidence_contract(
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
                "path": "manifest.json",
                "kind": "manifest_json",
                "sha256": "a" * 64,
                "bytes": 1,
                "identities": identities,
            }
        ],
        v2_locked_csv_identities=locked,
    )


def test_four_hour_policy_is_balanced_and_persistent() -> None:
    first = four_hour_action_levels(123)
    second = four_hour_action_levels(123)
    assert np.array_equal(first, second)
    assert first.shape == (192,)
    assert Counter(first) == {-1.0: 64, 0.0: 64, 1.0: 64}
    blocks = first[::16]
    assert Counter(blocks) == {-1.0: 4, 0.0: 4, 1.0: 4}
    assert not np.equal(blocks[:-1], blocks[1:]).any()


def test_policies_match_marginals_but_not_dwell() -> None:
    old = policy_levels("old_2h", 456)
    new = policy_levels("new_4h", 456)
    assert Counter(old) == Counter(new)
    assert np.all([len(set(old[i : i + 8])) == 1 for i in range(0, 192, 8)])
    assert np.all([len(set(new[i : i + 16])) == 1 for i in range(0, 192, 16)])
    with pytest.raises(ValueError, match="unknown"):
        policy_levels("other", 456)


def test_selection_is_value_blind_deterministic_and_stratified() -> None:
    parent = _parent_plan(next(iter(CASES)))
    left = select_windows(parent)
    right = select_windows(parent)
    assert left == right
    assert len(left) == 12
    counts = Counter(entry["temperature_stratum"] for entry in left)
    assert set(counts) == set(range(5))
    assert min(counts.values()) >= 2
    assert {entry["source_plan_role"] for entry in left} <= {"fit", "validation"}
    assert all(set(entry["policies"]) == set(POLICIES) for entry in left)


def test_case_plan_self_hash_and_action_arrays_validate() -> None:
    plan = build_case_plan(_parent_plan(next(iter(CASES))))
    validate_case_plan(plan)
    plan["entries"][0]["policies"]["new_4h"]["action_levels"][0] = 9
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    from .plan import canonical_sha256

    plan["plan_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="action array"):
        validate_case_plan(plan)


def test_disjointness_certificate_rejects_prior_collected_day() -> None:
    v1 = {case: _parent_plan(case, day_offset=4) for case in CASES}
    v2 = {case: _parent_plan(case, day_offset=9) for case in CASES}
    v3 = {case: build_case_plan(v2[case]) for case in CASES}
    certificate = build_disjointness_certificate(
        v1, v2, v3, prior_evidence=_prior_evidence(v2)
    )
    assert certificate["certificate_sha256"]
    first_case = next(iter(CASES))
    selected = v3[first_case]["entries"][0]
    collision = {
        "case": first_case,
        "day": selected["day"],
        "trajectory_seed": 2_000_000_000,
    }
    with pytest.raises(ValueError, match="selected v3 day"):
        build_disjointness_certificate(
            v1,
            v2,
            v3,
            prior_evidence=_prior_evidence(v2, [collision]),
        )


def test_disjointness_certificate_rejects_prior_seed_collision() -> None:
    v1 = {case: _parent_plan(case, day_offset=4) for case in CASES}
    v2 = {case: _parent_plan(case, day_offset=9) for case in CASES}
    v3 = {case: build_case_plan(v2[case]) for case in CASES}
    first_case = next(iter(CASES))
    selected_seed = v3[first_case]["entries"][0]["policies"]["old_2h"][
        "trajectory_seed"
    ]
    collision = {
        "case": first_case,
        "day": 999,
        "trajectory_seed": selected_seed,
    }
    with pytest.raises(ValueError, match="trajectory seed"):
        build_disjointness_certificate(
            v1,
            v2,
            v3,
            prior_evidence=_prior_evidence(v2, [collision]),
        )


def test_interval_overlap_is_half_open() -> None:
    assert intervals_overlap((0, 10), (9, 20))
    assert not intervals_overlap((0, 10), (10, 20))
