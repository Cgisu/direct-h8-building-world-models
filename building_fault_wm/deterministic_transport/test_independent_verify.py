from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from . import gate
from .independent_verify import (
    FLOAT_ABS_TOLERANCE,
    assert_semantically_equal,
    reconstruct_gate,
    validate_core,
    verify_files,
)
from .test_gate import _frame


def _vary_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    values = {
        "bias": {
            "legacy": 1.0,
            "ungated_h8": 0.5,
            "deterministic_wm": 1.0,
        },
        "drift": {
            "legacy": 1.0,
            "ungated_h8": 1.0,
            "deterministic_wm": 1.0,
        },
        "stuck": {
            "legacy": 1.0,
            "ungated_h8": 1.5,
            "deterministic_wm": 1.0,
        },
    }
    result["standardized_abs_error"] = [
        values[family][arm]
        for family, arm in zip(result["family"], result["arm"], strict=True)
    ]
    return result


def test_independent_reconstruction_matches_primary_gate() -> None:
    frame = _vary_by_family(
        _frame(ungated=0.8, legacy=1.0, deterministic=1.0)
    )
    observed = reconstruct_gate(frame)
    expected = gate.analyze_gate(frame)
    assert_semantically_equal(expected, observed)
    assert (
        observed["transport"]["A"]["persistent_across_dwell"]
        == expected["transport"]["A"]["persistent_across_dwell"]
    )
    assert (
        observed["transport"]["D"]["persistent_across_dwell"]
        == expected["transport"]["D"]["persistent_across_dwell"]
    )


def test_independent_weighting_is_unchanged_by_family_row_replication() -> None:
    frame = _vary_by_family(
        _frame(ungated=1.0, legacy=1.0, deterministic=1.0)
    )
    expected = reconstruct_gate(frame)
    bias = frame.loc[frame["family"] == "bias"].copy()
    copies = [frame]
    for index in range(7):
        replica = bias.copy()
        replica["cell_id"] += f":replica{index}"
        replica["onset"] += index + 1
        replica["anchor"] += index + 1
        copies.append(replica)
    observed = reconstruct_gate(pd.concat(copies, ignore_index=True))
    assert_semantically_equal(expected, observed)
    assert expected["results"]["new_4h"]["A"]["point"] == pytest.approx(0.0)


def test_independent_validator_rejects_arm_and_policy_unpairing() -> None:
    frame = _frame(ungated=0.8, legacy=1.0, deterministic=1.0)
    missing_arm = frame.drop(
        frame.loc[
            (frame["arm"] == "legacy")
            & (frame["policy"] == "old_2h")
        ].index[0]
    )
    with pytest.raises(ValueError, match="model-arm rows"):
        validate_core(missing_arm)

    missing_policy = frame.drop(
        frame.loc[
            (frame["policy"] == "new_4h")
            & (frame["arm"] == "legacy")
        ].index[0]
    )
    with pytest.raises(ValueError, match="model-arm rows|policy branches"):
        validate_core(missing_policy)


def test_file_verifier_detects_semantic_tampering(tmp_path: Path) -> None:
    frame = _frame(ungated=0.8, legacy=1.0, deterministic=1.0)
    core_path = tmp_path / "gate_core.csv"
    result_path = tmp_path / "gate_result.json"
    frame.to_csv(core_path, index=False)
    result = gate.analyze_gate(frame)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    receipt = verify_files(core_path, result_path)
    assert receipt["verified"] is True
    assert receipt["architecture_persistent_across_dwell"] is True

    result["results"]["new_4h"]["A"]["point"] += 10 * FLOAT_ABS_TOLERANCE
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="numeric mismatch"):
        verify_files(core_path, result_path)


def test_independent_module_does_not_import_primary_gate() -> None:
    source = Path(__file__).with_name("independent_verify.py").read_text(
        encoding="ascii"
    )
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "gate" not in imported
    assert not any(name.endswith(".gate") for name in imported)
