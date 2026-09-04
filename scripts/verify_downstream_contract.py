#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_fault_wm.downstream_control import test_experiment

TESTS = (
    "test_fault_injector_matches_fixed_fault_contract",
    "test_candidate_rule_is_cost_first_only_within_comfort_feasible_set",
    "test_predicted_metrics_use_physical_units",
    "test_analysis_includes_the_two_model_comparisons",
)

for name in TESTS:
    getattr(test_experiment, name)()

print("DOWNSTREAM CONTRACT TESTS: PASS")
print(f"  tests: {len(TESTS)}")
