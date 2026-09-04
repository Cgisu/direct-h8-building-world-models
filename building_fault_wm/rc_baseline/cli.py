"""Command-line entry points for the RC comparator workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audit import DEFAULT_AUDIT_ROOT, run_audit
from .evaluation import (
    DEFAULT_EVALUATION_ROOT,
    DEFAULT_READINESS_ROOT,
    prepare_readiness,
    run_evaluation,
    verify_evaluation,
    verify_readiness,
)
from .study import DEFAULT_TRAINING_ROOT, run_development_selection, verify_training
from .reproducibility import DEFAULT_OUTPUT_ROOT as DEFAULT_REPRODUCTION_ROOT
from .reproducibility import compare_selections


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RC comparator workflow")
    parser.add_argument(
        "command",
        choices=("select", "reproduction", "readiness", "evaluate", "audit", "verify"),
    )
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--readiness-root", type=Path, default=DEFAULT_READINESS_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--reference-training-root", type=Path)
    parser.add_argument(
        "--reproduction-root", type=Path, default=DEFAULT_REPRODUCTION_ROOT
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "select":
        path = run_development_selection(args.training_root)
    elif args.command == "reproduction":
        if args.reference_training_root is None:
            raise ValueError("--reference-training-root is required")
        path = compare_selections(
            args.reference_training_root, args.training_root, args.reproduction_root
        )
    elif args.command == "readiness":
        path = prepare_readiness(
            args.training_root, args.reproduction_root, args.readiness_root
        )
    elif args.command == "evaluate":
        path = run_evaluation(
            args.training_root,
            args.reproduction_root,
            args.readiness_root,
            args.evaluation_root,
        )
    elif args.command == "audit":
        path = run_audit(args.evaluation_root, args.audit_root)
    else:
        verify_training(args.training_root)
        verify_readiness(
            args.training_root, args.reproduction_root, args.readiness_root
        )
        path = args.evaluation_root / "evaluation_complete.json"
        verify_evaluation(args.evaluation_root)
    print(path)
