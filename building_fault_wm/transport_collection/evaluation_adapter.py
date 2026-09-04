"""Route v5 metadata into the byte-verified frozen numerical evaluator."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from building_fault_wm.deterministic_transport import (
    run_evaluation as frozen_run,
)

from . import external_freeze, runner


DEFAULT_OUTPUT = (
    runner.PROJECT_ROOT
    / "artifacts/direct_h8_deterministic_transport_v3_evaluation_v5"
)
PATCHED_CALLS = (
    "corpus.load_bound_readiness",
    "external_freeze.validate_external_freeze_receipt",
)


def shared_runtime_snapshot(
    prelock_root: Path = runner.PRELOCK_ROOT,
) -> dict:
    registry = runner._validate_scientific_prelock(prelock_root)
    return runner.validate_live_shared_runtime_semantics(
        prelock_root=prelock_root,
        registry=registry,
    )


def frozen_source_snapshot() -> dict[str, str]:
    """Hash frozen top-level source, excluding state and bytecode caches."""

    paths = tuple(
        sorted(
            path
            for path in runner.FROZEN_SOURCE_ROOT.iterdir()
            if path.is_file() and path.suffix in {".py", ".md"}
        )
    )
    return {
        path.name: runner.frozen_plan.sha256_file(path)
        for path in paths
    }


def verify_frozen_numerical_path(
    prelock_root: Path = runner.PRELOCK_ROOT,
) -> dict[str, object]:
    hashes = runner.validate_frozen_source_hashes()
    shared_runtime = shared_runtime_snapshot(prelock_root)
    module_path = Path(str(frozen_run.__file__)).resolve()
    expected_path = (runner.FROZEN_SOURCE_ROOT / "run_evaluation.py").resolve()
    if module_path != expected_path:
        raise ValueError("imported evaluator is not the frozen run_evaluation.py")
    if runner.frozen_plan.sha256_file(module_path) != (
        runner.FROZEN_RUN_EVALUATION_SHA256
    ):
        raise ValueError("imported frozen evaluator hash changed")
    return {
        "run_evaluation.py": hashes["run_evaluation.py"],
        "corpus.py": hashes["corpus.py"],
        "patched_calls": list(PATCHED_CALLS),
        "live_shared_runtime_validation": shared_runtime,
    }


@contextmanager
def v5_metadata_hooks(
    *,
    readiness_path: Path,
) -> Iterator[None]:
    """Temporarily replace only readiness and external-freeze metadata calls."""

    original_readiness_loader = frozen_run.corpus.load_bound_readiness
    original_freeze_validator = (
        frozen_run.external_freeze.validate_external_freeze_receipt
    )
    original_transport_loader = frozen_run.corpus.load_transport_corpus_index
    original_run_function = frozen_run.run_evaluation
    original_verify_function = frozen_run.verify_only

    def load_bound_readiness_v5(
        *,
        prelock_root: Path,
        live_data_root: Path,
        expected_prelock_sha256: str,
        expected_readiness_sha256: str,
    ):
        return runner.load_bound_readiness(
            prelock_root=prelock_root,
            live_data_root=live_data_root,
            readiness_path=readiness_path,
            expected_prelock_sha256=expected_prelock_sha256,
            expected_readiness_sha256=expected_readiness_sha256,
        )

    def validate_external_freeze_v5(
        receipt_path: Path,
        expected_prelock_sha256: str,
        expected_readiness_sha256: str,
        *,
        prelock_root: Path,
        readiness_path: Path,
        live: bool = True,
    ):
        if readiness_path.resolve() != readiness_path_bound.resolve():
            raise ValueError("frozen evaluator requested a different readiness file")
        return external_freeze.validate_external_freeze_receipt(
            receipt_path,
            expected_prelock_sha256,
            expected_readiness_sha256,
            prelock_root=prelock_root,
            readiness_path=readiness_path_bound,
            live=live,
        )

    readiness_path_bound = readiness_path.resolve()
    frozen_run.corpus.load_bound_readiness = load_bound_readiness_v5
    frozen_run.external_freeze.validate_external_freeze_receipt = (
        validate_external_freeze_v5
    )
    try:
        if (
            frozen_run.corpus.load_transport_corpus_index
            is not original_transport_loader
            or frozen_run.run_evaluation is not original_run_function
            or frozen_run.verify_only is not original_verify_function
        ):
            raise RuntimeError("a frozen numerical callable changed during hook setup")
        yield
    finally:
        frozen_run.corpus.load_bound_readiness = original_readiness_loader
        frozen_run.external_freeze.validate_external_freeze_receipt = (
            original_freeze_validator
        )
        if (
            frozen_run.corpus.load_transport_corpus_index
            is not original_transport_loader
            or frozen_run.run_evaluation is not original_run_function
            or frozen_run.verify_only is not original_verify_function
        ):
            raise RuntimeError("a frozen numerical callable changed during evaluation")


def _invoke_once(
    command: str,
    *,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    prelock_root: Path,
    data_root: Path,
    state_root: Path,
    readiness_path: Path,
    external_freeze_receipt_path: Path,
    output_dir: Path,
    live_external_freeze: bool,
):
    kwargs = {
        "expected_prelock_sha256": expected_prelock_sha256,
        "expected_readiness_sha256": expected_readiness_sha256,
        "prelock_root": prelock_root,
        "data_root": data_root,
        "state_root": state_root,
        "readiness_path": readiness_path,
        "external_freeze_receipt_path": external_freeze_receipt_path,
        "output_dir": output_dir,
        "live_external_freeze": live_external_freeze,
    }
    if command == "run":
        return frozen_run.run_evaluation(**kwargs)
    if command == "verify":
        return frozen_run.verify_only(**kwargs)
    raise ValueError(f"unsupported evaluation command: {command}")


def invoke_frozen_evaluation(
    command: str,
    *,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    prelock_root: Path = runner.PRELOCK_ROOT,
    data_root: Path = runner.DATA_ROOT,
    state_root: Path = runner.STATE_ROOT,
    readiness_path: Path = runner.READINESS_PATH,
    external_freeze_receipt_path: Path = runner.EXTERNAL_FREEZE_RECEIPT,
    output_dir: Path = DEFAULT_OUTPUT,
    live_external_freeze: bool = True,
):
    if command not in {"run", "verify"}:
        raise ValueError("evaluation command must be run or verify")
    runner.validate_namespace_separation(
        data_root=data_root,
        state_root=state_root,
        freeze_root=readiness_path.parent,
    )
    verified_path = verify_frozen_numerical_path(prelock_root)
    source_before = frozen_source_snapshot()
    shared_before = verified_path["live_shared_runtime_validation"]
    try:
        with v5_metadata_hooks(readiness_path=readiness_path):
            return _invoke_once(
                command,
                expected_prelock_sha256=expected_prelock_sha256,
                expected_readiness_sha256=expected_readiness_sha256,
                prelock_root=prelock_root,
                data_root=data_root,
                state_root=state_root,
                readiness_path=readiness_path,
                external_freeze_receipt_path=external_freeze_receipt_path,
                output_dir=output_dir,
                live_external_freeze=live_external_freeze,
            )
    finally:
        source_after = frozen_source_snapshot()
        shared_after = shared_runtime_snapshot(prelock_root)
        if source_after != source_before:
            raise RuntimeError("frozen evaluator source changed during invocation")
        if shared_after != shared_before:
            raise RuntimeError(
                "shared numerical runtime changed during evaluation"
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify", "run-verify"))
    parser.add_argument(
        "--expected-prelock-sha256",
        default=runner.ORIGINAL_PRELOCK_SHA256,
    )
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--prelock-root", type=Path, default=runner.PRELOCK_ROOT)
    parser.add_argument("--data-root", type=Path, default=runner.DATA_ROOT)
    parser.add_argument("--state-root", type=Path, default=runner.STATE_ROOT)
    parser.add_argument("--readiness", type=Path, default=runner.READINESS_PATH)
    parser.add_argument(
        "--external-freeze-receipt",
        type=Path,
        default=runner.EXTERNAL_FREEZE_RECEIPT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-live-external-freeze", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    common = {
        "expected_prelock_sha256": args.expected_prelock_sha256,
        "expected_readiness_sha256": args.expected_readiness_sha256,
        "prelock_root": args.prelock_root.resolve(),
        "data_root": args.data_root.resolve(),
        "state_root": args.state_root.resolve(),
        "readiness_path": args.readiness.resolve(),
        "external_freeze_receipt_path": (
            args.external_freeze_receipt.resolve()
        ),
        "output_dir": args.output.resolve(),
        "live_external_freeze": not args.no_live_external_freeze,
    }
    commands = ("run", "verify") if args.command == "run-verify" else (args.command,)
    results = []
    for command in commands:
        result = invoke_frozen_evaluation(command, **common)
        results.append(
            {
                "command": command,
                "result": str(result) if isinstance(result, Path) else result,
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
