"""Command-line interface for the secondary schedule-matched ARX addendum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from building_fault_wm.transport_collection import (
    runner as transport_runner,
)

from . import evaluate, external_freeze, lock, train
from .config import CASES, FROZEN_CONFIG


def _transport_arguments(
    parser: argparse.ArgumentParser, *, include_raw: bool
) -> None:
    parser.add_argument(
        "--transport-prelock-root",
        type=Path,
        default=transport_runner.PRELOCK_ROOT,
    )
    parser.add_argument(
        "--transport-live-data-root",
        type=Path,
        default=transport_runner.DATA_ROOT,
    )
    parser.add_argument(
        "--transport-readiness",
        type=Path,
        default=transport_runner.READINESS_PATH,
    )
    parser.add_argument(
        "--transport-external-freeze-receipt",
        type=Path,
        default=transport_runner.EXTERNAL_FREEZE_RECEIPT,
    )
    parser.add_argument(
        "--transport-state-root",
        type=Path,
        default=transport_runner.STATE_ROOT,
    )
    parser.add_argument(
        "--transport-manifest",
        type=Path,
        default=transport_runner.DATA_ROOT / transport_runner.MANIFEST_RELATIVE,
    )
    if include_raw:
        parser.add_argument(
            "--transport-raw-root",
            type=Path,
            default=transport_runner.DATA_ROOT / transport_runner.RAW_RELATIVE,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser(
        "train", help="train or verify the schedule-matched ARX grid"
    )
    training.add_argument("--output", type=Path, default=train.DEFAULT_OUTPUT)
    training.add_argument("--case", action="append", choices=CASES)
    training.add_argument(
        "--seed",
        action="append",
        type=int,
        choices=FROZEN_CONFIG.model_seeds,
    )

    prepare = subparsers.add_parser(
        "prepare-lock",
        help="prepare the metadata-only addendum pre-lock after v5 completion",
    )
    prepare.add_argument("--output", type=Path, default=lock.DEFAULT_OUTPUT)
    prepare.add_argument(
        "--training-root", type=Path, default=lock.DEFAULT_TRAINING_ROOT
    )
    prepare.add_argument("--live-transport-freeze", action="store_true")
    _transport_arguments(prepare, include_raw=False)

    verify_lock = subparsers.add_parser(
        "verify-lock", help="verify the addendum pre-lock and live bindings"
    )
    verify_lock.add_argument("--prelock-root", type=Path, default=lock.DEFAULT_OUTPUT)

    freeze_files = subparsers.add_parser(
        "freeze-files",
        help="print the exact files that must be externally frozen",
    )
    freeze_files.add_argument("--prelock-root", type=Path, default=lock.DEFAULT_OUTPUT)

    create_freeze = subparsers.add_parser(
        "create-public-freeze",
        help="create, pin, write once, and live-verify the public freeze",
    )
    create_freeze.add_argument(
        "--prelock-root", type=Path, default=lock.DEFAULT_OUTPUT
    )
    create_freeze.add_argument("--receipt", type=Path, required=True)

    verify_freeze = subparsers.add_parser(
        "verify-freeze", help="verify a revision-pinned public freeze"
    )
    verify_freeze.add_argument("--prelock-root", type=Path, default=lock.DEFAULT_OUTPUT)
    verify_freeze.add_argument("--receipt", type=Path, required=True)
    verify_freeze.add_argument("--offline", action="store_true")

    run = subparsers.add_parser(
        "evaluate",
        help="evaluate only after the addendum and v5 freezes validate",
    )
    run.add_argument("--prelock-root", type=Path, default=lock.DEFAULT_OUTPUT)
    run.add_argument("--addendum-freeze-receipt", type=Path, required=True)
    run.add_argument(
        "--training-root", type=Path, default=train.DEFAULT_OUTPUT
    )
    run.add_argument("--output", type=Path, default=evaluate.DEFAULT_OUTPUT)
    run.add_argument("--offline-freeze-verification", action="store_true")
    _transport_arguments(run, include_raw=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        path = train.train_grid(
            args.output,
            cases=CASES if args.case is None else tuple(args.case),
            seeds=(
                FROZEN_CONFIG.model_seeds
                if args.seed is None
                else tuple(args.seed)
            ),
        )
        print(path)
        return 0
    if args.command == "prepare-lock":
        path = lock.prepare_prelock(
            output_root=args.output,
            training_root=args.training_root,
            transport_prelock_root=args.transport_prelock_root,
            transport_live_data_root=args.transport_live_data_root,
            transport_readiness_path=args.transport_readiness,
            transport_external_freeze_receipt_path=(
                args.transport_external_freeze_receipt
            ),
            transport_state_root=args.transport_state_root,
            transport_manifest_path=args.transport_manifest,
            live_transport_external_freeze=args.live_transport_freeze,
        )
        print(path)
        return 0
    if args.command == "verify-lock":
        registry = lock.verify_prelock(args.prelock_root)
        print(json.dumps(registry, sort_keys=True))
        return 0
    if args.command == "freeze-files":
        files = lock.external_freeze_file_paths(args.prelock_root)
        print(
            json.dumps(
                {name: str(path.resolve()) for name, path in sorted(files.items())},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "create-public-freeze":
        receipt = external_freeze.create_public_freeze(
            prelock_root=args.prelock_root,
            receipt_path=args.receipt,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "verify-freeze":
        receipt = external_freeze.validate_external_freeze_receipt(
            args.receipt,
            args.prelock_root,
            live=not args.offline,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "evaluate":
        completion = evaluate.run_evaluation(
            prelock_root=args.prelock_root,
            addendum_external_freeze_receipt_path=args.addendum_freeze_receipt,
            training_root=args.training_root,
            transport_prelock_root=args.transport_prelock_root,
            transport_live_data_root=args.transport_live_data_root,
            transport_readiness_path=args.transport_readiness,
            transport_external_freeze_receipt_path=(
                args.transport_external_freeze_receipt
            ),
            transport_state_root=args.transport_state_root,
            transport_manifest_path=args.transport_manifest,
            transport_raw_root=args.transport_raw_root,
            output_root=args.output,
            live_external_freezes=not args.offline_freeze_verification,
        )
        print(completion)
        return 0
    raise AssertionError(f"unhandled ARX addendum command: {args.command}")
