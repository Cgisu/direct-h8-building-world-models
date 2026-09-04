"""Command-line entry point for the reviewer subspace comparison."""

from __future__ import annotations

import argparse

from .audit import run_audit
from .study import prepare_readiness, run_development_selection, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("select", "readiness", "evaluate", "audit", "all")
    )
    args = parser.parse_args()
    if args.stage in {"select", "all"}:
        print(run_development_selection())
    if args.stage in {"readiness", "all"}:
        print(prepare_readiness())
    if args.stage in {"evaluate", "all"}:
        print(run_evaluation())
    if args.stage in {"audit", "all"}:
        print(run_audit())


if __name__ == "__main__":
    main()

