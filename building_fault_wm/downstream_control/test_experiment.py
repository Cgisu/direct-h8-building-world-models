from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol
from .experiment import FaultInjector, analyze, predicted_metrics, select_candidate


def test_protocol_uses_all_response_unseen_windows_and_fixed_inputs() -> None:
    payload = protocol.protocol_payload()
    assert payload["schema"] == "direct-h8-downstream-control-protocol-v1"
    assert len(payload["windows"]) == 12
    assert len({row["day"] for row in payload["windows"]}) == 12
    assert payload["policies"] == list(protocol.POLICIES)
    assert payload["conditions"] == list(protocol.CONDITIONS)
    assert len(payload["inputs"]["rssm_checkpoints"]) == 10
    assert len(payload["inputs"]["deterministic_checkpoints"]) == 5
    assert len(payload["inputs"]["implementation"]) == len(
        protocol.IMPLEMENTATION_PATHS
    )
    assert payload["development_pilot"]["day"] not in {
        row["day"] for row in payload["windows"]
    }
    assert payload["paired_contrasts"] == [
        {"candidate": candidate, "reference": reference}
        for candidate, reference in protocol.CONTRASTS
    ]
    assert str(Path.home()) not in json.dumps(payload)


def test_fault_injector_matches_fixed_fault_contract() -> None:
    clean = np.asarray([294.0, 1000.0, 300.0, 2000.0])
    bias = FaultInjector("zone_bias_positive")
    bias.apply(protocol.FAULT_START - 1, clean)
    np.testing.assert_allclose(
        bias.apply(protocol.FAULT_START, clean),
        [296.0, 1000.0, 300.0, 2000.0],
    )
    np.testing.assert_allclose(bias.apply(protocol.FAULT_STOP, clean), clean)

    negative_bias = FaultInjector("zone_bias_negative")
    negative_bias.apply(protocol.FAULT_START - 1, clean)
    assert negative_bias.apply(protocol.FAULT_START, clean)[0] == 292.0

    drift = FaultInjector("zone_drift_positive")
    drift.apply(protocol.FAULT_START - 1, clean)
    assert drift.apply(protocol.FAULT_START, clean)[0] == 294.05
    assert drift.apply(protocol.FAULT_START + 47, clean)[0] == 296.4

    negative_drift = FaultInjector("zone_drift_negative")
    negative_drift.apply(protocol.FAULT_START - 1, clean)
    assert np.isclose(
        negative_drift.apply(protocol.FAULT_START + 47, clean)[0], 291.6
    )

    stuck = FaultInjector("zone_stuck")
    stuck.apply(protocol.FAULT_START - 1, clean)
    changed = clean.copy()
    changed[0] = 290.0
    assert stuck.apply(protocol.FAULT_START, changed)[0] == 294.0


def test_candidate_rule_is_cost_first_only_within_comfort_feasible_set() -> None:
    rows = [
        {
            "action_level": -1.0,
            "comfort_feasible": False,
            "predicted_discomfort_kh": 0.1,
            "predicted_cost_proxy": 0.1,
        },
        {
            "action_level": 0.0,
            "comfort_feasible": True,
            "predicted_discomfort_kh": 0.0,
            "predicted_cost_proxy": 0.4,
        },
        {
            "action_level": 1.0,
            "comfort_feasible": True,
            "predicted_discomfort_kh": 0.0,
            "predicted_cost_proxy": 0.6,
        },
    ]
    assert select_candidate(rows)["action_level"] == 0.0
    for row in rows:
        row["comfort_feasible"] = False
    rows[0]["predicted_discomfort_kh"] = 0.2
    rows[1]["predicted_discomfort_kh"] = 0.1
    rows[2]["predicted_discomfort_kh"] = 0.3
    assert select_candidate(rows)["action_level"] == 0.0


def test_predicted_metrics_use_physical_units() -> None:
    prediction = np.zeros((protocol.CONTROL_HORIZON, 4))
    prediction[:, 0] = 293.0
    prediction[:, 1] = 2000.0
    contexts = np.zeros((protocol.CONTROL_HORIZON, 5))
    contexts[:, 2] = 294.0
    contexts[:, 3] = 300.0
    contexts[:, 4] = 0.25
    result = predicted_metrics(prediction, contexts)
    assert result["predicted_discomfort_kh"] == 2.0
    assert result["predicted_energy_kwh"] == 4.0
    assert result["predicted_cost_proxy"] == 1.0
    assert result["comfort_feasible"] is False


def test_analysis_includes_the_two_model_comparisons() -> None:
    rows = []
    for day in (9, 72):
        for index, policy_name in enumerate(protocol.POLICIES):
            row = {
                "day": day,
                "condition": "clean",
                "policy": policy_name,
            }
            row.update(
                {
                    endpoint: float(day + index)
                    for endpoint in (
                        "cost_tot",
                        "tdis_tot",
                        "control_cost_proxy",
                        "control_discomfort_kh",
                        "control_energy_kwh",
                    )
                }
            )
            rows.append(row)
    aggregate, paired = analyze(pd.DataFrame(rows))
    assert len(aggregate) == len(protocol.POLICIES) * 5
    assert len(paired) == len(protocol.CONTRASTS) * 5
    contrasts = set(zip(paired.candidate, paired.reference))
    assert ("direct_h8_rssm", "legacy_rssm") in contrasts
    assert ("deterministic_wm", "direct_h8_rssm") in contrasts
