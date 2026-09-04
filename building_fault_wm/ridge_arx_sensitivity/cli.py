"""Command line entry points for the strengthened ARX robustness analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import audit, report, study
from .config import CASES, MODEL_SEEDS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-training")
    subparsers.add_parser("recover-selection-v2")
    fit = subparsers.add_parser("fit")
    fit.add_argument("--case", required=True, choices=CASES)
    fit.add_argument("--seed", required=True, type=int, choices=MODEL_SEEDS)
    subparsers.add_parser("finalize-training")
    subparsers.add_parser("prepare-readiness")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("audit")
    subparsers.add_parser("report")
    subparsers.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-training":
        path = study.prepare_training_root()
    elif args.command == "recover-selection-v2":
        path = study.recover_v2_selection_runs()
    elif args.command == "fit":
        path = study.fit_case_seed(case=args.case, model_seed=args.seed)
    elif args.command == "finalize-training":
        path = study.finalize_training_grid()
    elif args.command == "prepare-readiness":
        path = study.prepare_readiness()
    elif args.command == "evaluate":
        path = study.run_evaluation()
    elif args.command == "audit":
        path = audit.run_audit()
    elif args.command == "report":
        path = report.build_report()
    elif args.command == "verify":
        study.verify_training_grid()
        study.verify_readiness()
        study.verify_evaluation()
        audit_receipt = audit.DEFAULT_AUDIT_ROOT / "audit_receipt.json"
        if not audit_receipt.is_file():
            raise ValueError("standalone audit is missing")
        path = report.DEFAULT_REPORT_ROOT / report.MANIFEST_NAME
        report.verify_report()
    else:
        raise AssertionError(args.command)
    print(Path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
