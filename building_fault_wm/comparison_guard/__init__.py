"""Outcome-blind identity guard for the frozen neural-versus-ARX analysis."""

from .guard import BindingPaths, build_binding_contract, run_guarded_analysis

__all__ = ("BindingPaths", "build_binding_contract", "run_guarded_analysis")
