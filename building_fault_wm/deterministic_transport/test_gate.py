from __future__ import annotations

import pandas as pd

from .gate import (
    ARMS,
    CASES,
    CONFIRMATION_SEEDS,
    POLICIES,
    REQUIRED_COLUMNS,
    SILENT_FAMILIES,
    _pivot_scores,
    analyze_gate,
)


def _frame(*, ungated: float, legacy: float, deterministic: float) -> pd.DataFrame:
    errors = {
        "legacy": legacy,
        "ungated_h8": ungated,
        "deterministic_wm": deterministic,
    }
    rows = []
    for case in CASES:
        for window_index in range(12):
            for policy_index, policy in enumerate(POLICIES):
                for model_seed in CONFIRMATION_SEEDS:
                    for family in SILENT_FAMILIES:
                        for channel_index, channel in enumerate(
                            ("zone_temperature_k", "hvac_electric_power_w")
                        ):
                            for arm in ARMS:
                                rows.append(
                                    {
                                        "case": case,
                                        "policy": policy,
                                        "window_id": f"{case}:w{window_index:02d}",
                                        "trajectory_day": 9 + 9 * window_index,
                                        "scenario_seed": 1000 + window_index,
                                        "trajectory_seed": (
                                            2000
                                            + 2 * window_index
                                            + policy_index
                                        ),
                                        "model_seed": model_seed,
                                        "arm": arm,
                                        "cell_id": (
                                            f"{case}:w{window_index}:{policy}:"
                                            f"{family}:{channel}:seed{model_seed}"
                                        ),
                                        "fault_channel": channel,
                                        "family": family,
                                        "sign": 1,
                                        "severity": float(channel_index + 1),
                                        "onset": 64,
                                        "anchor": 72,
                                        "horizon": 8,
                                        "standardized_abs_error": errors[arm],
                                    }
                                )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_gate_detects_rssm_and_h8_advantages() -> None:
    result = analyze_gate(
        _frame(ungated=0.8, legacy=1.0, deterministic=1.0),
        bootstrap_draws=100,
    )
    assert result["primary_architecture_category"] == "RSSM_ADVANTAGE"
    assert result["primary_supervision_category"] == "H8_BENEFIT"
    assert result["transport"]["A"]["persistent_across_dwell"] is True
    assert result["transport"]["D"]["persistent_across_dwell"] is True


def test_gate_is_symmetric_for_deterministic_advantage() -> None:
    result = analyze_gate(
        _frame(ungated=1.25, legacy=1.5, deterministic=1.0),
        bootstrap_draws=100,
    )
    assert (
        result["primary_architecture_category"]
        == "DETERMINISTIC_WM_ADVANTAGE"
    )
    assert result["primary_supervision_category"] == "H8_BENEFIT"


def test_gate_can_conclude_practical_equivalence() -> None:
    result = analyze_gate(
        _frame(ungated=0.99, legacy=1.0, deterministic=1.0),
        bootstrap_draws=100,
    )
    assert result["primary_architecture_category"] == "PRACTICAL_EQUIVALENCE"
    assert result["primary_supervision_category"] == "PRACTICAL_EQUIVALENCE"


def test_family_weighting_is_equal_under_row_replication() -> None:
    frame = _frame(ungated=1.0, legacy=1.0, deterministic=1.0)
    family_error = {"bias": 9.0, "drift": 3.0, "stuck": 3.0}
    frame["standardized_abs_error"] = frame["family"].map(family_error)
    expected = _pivot_scores(frame)

    replicated = frame.loc[frame["family"] == "bias"].copy()
    copies = [frame]
    for index in range(9):
        copy = replicated.copy()
        copy["cell_id"] = copy["cell_id"] + f":replica{index}"
        copy["onset"] = copy["onset"] + index + 1
        copy["anchor"] = copy["anchor"] + index + 1
        copies.append(copy)
    observed = _pivot_scores(pd.concat(copies, ignore_index=True))

    assert expected.loc[:, list(ARMS)].eq(5.0).all().all()
    pd.testing.assert_frame_equal(expected, observed)
