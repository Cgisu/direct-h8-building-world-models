"""Public BOPTEST multi-case telemetry-fault benchmark utilities."""

from .protocol import CASES, CaseAdapter, build_case_plan

__all__ = ["CASES", "CaseAdapter", "build_case_plan"]
