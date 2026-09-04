from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from building_fault_wm.neural_benchmark import protocol as boptest

from . import build_plan_artifacts as builder
from . import plan


def _parent_plan(case: str, day_offset: int) -> dict:
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
    payload = {
        "schema": "fixture-parent-plan-v1",
        "case_adapter": {"case": case},
        "source_sha256": {
            "wrapped_fmu": "1" * 64,
            "weather_csv": "2" * 64,
        },
        "entries": entries,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    return {**payload, "plan_sha256": digest}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _write_identity_csv(path: Path, identity: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case", "day", "trajectory_seed"]
        )
        writer.writeheader()
        for _ in range(boptest.TRAJECTORY_STEPS):
            writer.writerow(identity)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, dict]]:
    multicase = tmp_path / "multicase"
    parent = tmp_path / "parent"
    experiment = parent / "experiment"
    v1_root = experiment / "v1_plans"
    v2_root = experiment / "v2_plans"
    v2_plans = {}
    for case in sorted(boptest.CASES):
        v1 = _parent_plan(case, 4)
        v2 = _parent_plan(case, 9)
        v2_plans[case] = v2
        _write_json(v1_root / f"{case}.json", v1)
        _write_json(v2_root / f"{case}.json", v2)
        for entry in v2["entries"]:
            if entry["role"] != "locked_test":
                continue
            identity = {
                "case": case,
                "day": entry["day"],
                "trajectory_seed": entry["trajectory_seed"],
            }
            name = (
                f"day{entry['day']:03d}_locked_test_"
                f"seed{entry['trajectory_seed']}.csv"
            )
            _write_identity_csv(
                multicase / "data_v5/locked_test_raw" / case / name,
                identity,
            )
    experiment.mkdir(parents=True, exist_ok=True)
    return multicase, parent, v1_root, v2_root, v2_plans


def test_builder_writes_only_canonical_plan_artifacts_exclusively(
    tmp_path: Path,
) -> None:
    multicase, parent, v1_root, v2_root, _ = _fixture(tmp_path)
    output = multicase / "data_v6"
    result = builder.build_plan_artifacts(
        multicase_root=multicase,
        parent_package_root=parent,
        v1_plan_root=v1_root,
        v2_plan_root=v2_root,
        output_root=output,
        expected_parent_digest="f" * 64,
        verify_parent=False,
    )
    expected = {
        *(f"plans/full/{case}.json" for case in boptest.CASES),
        "disjointness_certificate.json",
    }
    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == expected
    certificate = plan.load_json(output / "disjointness_certificate.json")
    assert certificate["schema"] == plan.CERTIFICATE_SCHEMA
    assert certificate["identity_proof"][
        "no_selected_identity_in_prior_evidence"
    ]
    assert result["prior_evidence_counts"]["v2_locked_csv_identities"] == 36
    with pytest.raises(FileExistsError):
        builder.build_plan_artifacts(
            multicase_root=multicase,
            parent_package_root=parent,
            v1_plan_root=v1_root,
            v2_plan_root=v2_root,
            output_root=output,
            expected_parent_digest="f" * 64,
            verify_parent=False,
        )


def test_builder_rejects_prior_day_collision_before_creating_output(
    tmp_path: Path,
) -> None:
    multicase, parent, v1_root, v2_root, v2_plans = _fixture(tmp_path)
    case = sorted(boptest.CASES)[0]
    selected = plan.build_case_plan(v2_plans[case])["entries"][0]
    collision = {
        "case": case,
        "day": selected["day"],
        "trajectory_seed": 2_000_000_000,
    }
    _write_identity_csv(
        multicase
        / "data_v3/smoke_raw"
        / case
        / (
            f"day{collision['day']:03d}_fit_"
            f"seed{collision['trajectory_seed']}.csv"
        ),
        collision,
    )
    output = multicase / "data_v6"
    with pytest.raises(ValueError, match="selected v3 day"):
        builder.build_plan_artifacts(
            multicase_root=multicase,
            parent_package_root=parent,
            v1_plan_root=v1_root,
            v2_plan_root=v2_root,
            output_root=output,
            expected_parent_digest="f" * 64,
            verify_parent=False,
        )
    assert not output.exists()


def test_json_identity_evidence_rejects_path_metadata_disagreement() -> None:
    payload = {
        "files": [
            {
                "path": (
                    "bestest_hydronic_heat_pump/"
                    "day009_fit_seed123.csv"
                ),
                "case": "bestest_hydronic_heat_pump",
                "day": 10,
                "trajectory_seed": 123,
            }
        ]
    }
    with pytest.raises(ValueError, match="path and explicit identity disagree"):
        builder._extract_json_identities(payload)


def test_prior_evidence_contract_detects_resealed_inventory_change() -> None:
    contract = plan.build_prior_evidence_contract(
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
                "path": "raw.csv",
                "kind": "raw_csv",
                "sha256": "a" * 64,
                "bytes": 10,
                "identities": [],
            }
        ],
        v2_locked_csv_identities=[],
    )
    contract["inventory"][0]["bytes"] = 11
    with pytest.raises(ValueError, match="does not self-verify"):
        plan.validate_prior_evidence_contract(contract)
