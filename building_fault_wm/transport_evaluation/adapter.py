"""Run the frozen evaluator through an exact-identity metadata dispatcher."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from building_fault_wm.deterministic_transport import (
    run_evaluation as frozen_run,
)
from building_fault_wm.transport_collection import (
    evaluation_adapter as v5_adapter,
)
from building_fault_wm.transport_collection import (
    external_freeze as v5_external_freeze,
)
from building_fault_wm.transport_collection import runner as v5_runner
from building_fault_wm.ridge_arx.io import (
    sha256_file,
    strict_json,
)

from . import closeout
from . import attempt


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
EXPECTED_PRELOCK_SHA256 = v5_runner.ORIGINAL_PRELOCK_SHA256
EXPECTED_READINESS_SHA256 = (
    "d245795503482417ac1d717782f33c56d05b0fa96d72f5156d5e954d4cdba74b"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/direct_h8_deterministic_transport_v3_evaluation_v6"
)
DEFAULT_RECOVERY_PRELOCK = (
    PROJECT_ROOT / "artifacts/direct_h8_transport_evaluation_v6_prelock_v3"
)
DEFAULT_RECOVERY_FREEZE_RECEIPT = (
    PROJECT_ROOT
    / "artifacts/direct_h8_transport_evaluation_v6_external_freeze_v3"
    / "external_freeze_receipt.json"
)
DEFAULT_STATE_BASE = HERE / ".direct_h8_evaluation_recovery_state_v1"
V5_COLLECTION_STATE = (
    v5_runner.STATE_ROOT / EXPECTED_READINESS_SHA256
)
IDENTITY_GUARD_PRELOCK = (
    PROJECT_ROOT
    / "artifacts/schedule_matched_arx_neural_identity_guard_prelock_v1"
)
IDENTITY_GUARD_FREEZE_RECEIPT = (
    PROJECT_ROOT
    / "artifacts/schedule_matched_arx_neural_identity_guard_external_freeze_v1"
    / "external_freeze_receipt.json"
)
EXPECTED_UPSTREAM_SHA256 = {
    "v5_readiness": (
        "ab48044cff4907eb6988a341568c57cd37a564a201e1a8fe6accb4a5b71fbf2d"
    ),
    "v5_external_freeze_receipt": (
        "a7cbb158b628433f972bedd883b5f553ca59bf608fc77e07e9968e1cc450e5ce"
    ),
    "v5_evaluation_adapter": (
        "1c8985deea0dc870be947fe51087ac7a8cc1f5e9b4ef632dbf84ea4e70815673"
    ),
    "v5_runner": (
        "6c0a0fe16d73c8da9e80d5f5463fad88e95985402c79272be6610d7771052f0c"
    ),
    "v5_external_freeze": (
        "a08175562de9485abec38583c07836d4786b251c26a8ce0810e042f515cac0d4"
    ),
    "frozen_run_evaluation": (
        "431d0e325aa71ada4edc0e6b8e758b029045b8eac3c06452302b4ab0ca98450e"
    ),
    "frozen_corpus": (
        "a9247fc9a09e437a68c308e045b5a3545fccc38bce3513195ebed47b00d6b2ed"
    ),
    "frozen_external_freeze": (
        "2eaa9ae171ba26e14a2434062ed8aadb0395537758fd5b5d571967f45a74535f"
    ),
    "v5_collection_attempt": (
        "4afe1c7945977d439e1ca30bdd3b228242f72333126efb009431b31605bf5c19"
    ),
    "v5_collection_completion": (
        "c1d597491a6ac741a047443b4a27b0cc6e61a931cfba7882b0ac23882be567ab"
    ),
    "v5_collection_manifest": (
        "97ddbc78b4840f32ad4a7619ec459dc7d898202f29f8939bc1bc73765b782a29"
    ),
    "identity_guard_prelock_digest": (
        "893ea727fb7618a79317dc2dbe92ac06af5e3f45bce829ca7b0d388dafdb9458"
    ),
    "identity_guard_prelock_registry": (
        "1c8140eb4a32ee05f225ae125e57f90e9d875a826d6b4d883e918f83aa9215cf"
    ),
    "identity_guard_external_freeze_receipt": (
        "104b48f495c8bd589df23e750f7a338e0e4445fe8e5524aedc84f8d8ab622c33"
    ),
}
NUMERICAL_CALLABLE_NAMES = (
    "run_evaluation",
    "verify_only",
    "load_frozen_assets",
    "verify_evaluation_output",
)


def upstream_paths() -> dict[str, Path]:
    return {
        "v5_readiness": v5_runner.READINESS_PATH,
        "v5_external_freeze_receipt": v5_runner.EXTERNAL_FREEZE_RECEIPT,
        "v5_evaluation_adapter": Path(v5_adapter.__file__).resolve(),
        "v5_runner": Path(v5_runner.__file__).resolve(),
        "v5_external_freeze": Path(v5_external_freeze.__file__).resolve(),
        "frozen_run_evaluation": Path(frozen_run.__file__).resolve(),
        "frozen_corpus": Path(frozen_run.corpus.__file__).resolve(),
        "frozen_external_freeze": Path(
            frozen_run.external_freeze.__file__
        ).resolve(),
        "v5_collection_attempt": (
            V5_COLLECTION_STATE / v5_runner.ATTEMPT_MARKER
        ),
        "v5_collection_completion": (
            V5_COLLECTION_STATE / v5_runner.COMPLETION_MARKER
        ),
        "v5_collection_manifest": (
            v5_runner.DATA_ROOT / v5_runner.MANIFEST_RELATIVE
        ),
        "identity_guard_prelock_digest": (
            IDENTITY_GUARD_PRELOCK / "identity_guard_prelock.canonical.sha256"
        ),
        "identity_guard_prelock_registry": (
            IDENTITY_GUARD_PRELOCK / "identity_guard_prelock.json"
        ),
        "identity_guard_external_freeze_receipt": (
            IDENTITY_GUARD_FREEZE_RECEIPT
        ),
    }


def upstream_hashes() -> dict[str, str]:
    return {
        name: sha256_file(path) for name, path in sorted(upstream_paths().items())
    }


def verify_upstream_inputs(*, live_external_freeze: bool) -> dict:
    from building_fault_wm.comparison_guard import (
        freeze as identity_guard_freeze,
    )
    from building_fault_wm.comparison_guard import (
        public_freeze as identity_guard_public_freeze,
    )

    hashes = upstream_hashes()
    if hashes != EXPECTED_UPSTREAM_SHA256:
        raise ValueError("v5 adapter, metadata, or frozen evaluator bytes changed")
    readiness = strict_json(v5_runner.READINESS_PATH)
    receipt = strict_json(v5_runner.EXTERNAL_FREEZE_RECEIPT)
    if (
        readiness.get("readiness_sha256") != EXPECTED_READINESS_SHA256
        or readiness.get("prelock_registry_sha256")
        != EXPECTED_PRELOCK_SHA256
        or receipt.get("readiness_sha256") != EXPECTED_READINESS_SHA256
        or receipt.get("prelock_registry_sha256") != EXPECTED_PRELOCK_SHA256
    ):
        raise ValueError("v5 readiness or freeze receipt identity changed")
    verified_freeze = v5_external_freeze.validate_external_freeze_receipt(
        v5_runner.EXTERNAL_FREEZE_RECEIPT,
        EXPECTED_PRELOCK_SHA256,
        EXPECTED_READINESS_SHA256,
        prelock_root=v5_runner.PRELOCK_ROOT,
        readiness_path=v5_runner.READINESS_PATH,
        live=live_external_freeze,
    )
    numerical = v5_adapter.verify_frozen_numerical_path(v5_runner.PRELOCK_ROOT)
    if (
        numerical.get("run_evaluation.py")
        != EXPECTED_UPSTREAM_SHA256["frozen_run_evaluation"]
        or numerical.get("corpus.py")
        != EXPECTED_UPSTREAM_SHA256["frozen_corpus"]
    ):
        raise ValueError("v5 adapter resolved a different frozen numerical path")
    identity_guard = identity_guard_freeze.verify_local_prelock(
        IDENTITY_GUARD_PRELOCK
    )
    identity_guard_receipt = (
        identity_guard_public_freeze.validate_public_freeze_receipt(
            IDENTITY_GUARD_FREEZE_RECEIPT,
            IDENTITY_GUARD_PRELOCK,
            live=live_external_freeze,
        )
    )
    return {
        "upstream_file_sha256": hashes,
        "v5_external_freeze_schema": verified_freeze.get("schema"),
        "frozen_numerical_path": numerical,
        "identity_guard_schema": identity_guard.get("schema"),
        "identity_guard_external_freeze_schema": (
            identity_guard_receipt.get("schema")
        ),
    }


def _numerical_callables() -> dict[str, object]:
    return {name: getattr(frozen_run, name) for name in NUMERICAL_CALLABLE_NAMES}


class TerminalV4FreezeProxy:
    """Expose only the exact terminal-v4 call to the original validator."""

    def __init__(self, original_validator) -> None:
        self._original_validator = original_validator
        self.call_count = 0

    def validate_external_freeze_receipt(
        self,
        receipt_path: Path,
        expected_prelock_sha256: str,
        expected_readiness_sha256: str,
        *,
        prelock_root: Path,
        readiness_path: Path,
        live: bool = True,
    ):
        is_exact_terminal_v4 = (
            receipt_path.resolve() == v5_runner.V4_FREEZE_RECEIPT_PATH.resolve()
            and expected_prelock_sha256 == EXPECTED_PRELOCK_SHA256
            and expected_readiness_sha256
            == v5_runner.TERMINAL_V4_READINESS_SHA256
            and prelock_root.resolve() == v5_runner.PRELOCK_ROOT.resolve()
            and readiness_path.resolve() == v5_runner.V4_READINESS_PATH.resolve()
            and live is False
        )
        if not is_exact_terminal_v4:
            raise ValueError(
                "recovery proxy rejected a non-terminal-v4 freeze identity"
            )
        self.call_count += 1
        return self._original_validator(
            receipt_path,
            expected_prelock_sha256,
            expected_readiness_sha256,
            prelock_root=prelock_root,
            readiness_path=readiness_path,
            live=live,
        )


@contextmanager
def terminal_v4_validation_proxy() -> Iterator[TerminalV4FreezeProxy]:
    """Repair only the shared-module alias used by the v4 terminal audit."""

    original_alias = v5_runner.frozen_external_freeze
    if original_alias is not frozen_run.external_freeze:
        raise RuntimeError("frozen external-freeze alias changed before recovery")
    original_validator = original_alias.validate_external_freeze_receipt
    original_readiness_loader = frozen_run.corpus.load_bound_readiness
    original_transport_loader = frozen_run.corpus.load_transport_corpus_index
    numerical_before = _numerical_callables()
    proxy = TerminalV4FreezeProxy(original_validator)
    v5_runner.frozen_external_freeze = proxy
    try:
        if (
            frozen_run.corpus.load_bound_readiness is not original_readiness_loader
            or frozen_run.corpus.load_transport_corpus_index
            is not original_transport_loader
            or _numerical_callables() != numerical_before
        ):
            raise RuntimeError("frozen callable changed during proxy setup")
        yield proxy
    finally:
        v5_runner.frozen_external_freeze = original_alias
        if (
            frozen_run.corpus.load_bound_readiness is not original_readiness_loader
            or frozen_run.corpus.load_transport_corpus_index
            is not original_transport_loader
            or _numerical_callables() != numerical_before
        ):
            raise RuntimeError("frozen numerical callable changed during recovery")


def verify_v4_terminal_audit_dispatch(
) -> dict:
    """Regression check for the exact metadata path that failed in v5."""

    with terminal_v4_validation_proxy() as proxy:
        terminal = v5_runner.terminal_v4_failure_binding()
    if (
        terminal.get("binding", {}).get("schema")
        != v5_runner.TERMINAL_BINDING_SCHEMA
        or proxy.call_count != 1
    ):
        raise ValueError("recovery proxy did not validate terminal-v4 exactly once")
    return {
        "terminal_v4_failure": terminal,
        "terminal_v4_proxy_call_count": proxy.call_count,
    }


def verify_v5_metadata_boundary() -> dict:
    """Exercise published v5 metadata routing without loading trajectory CSVs."""

    transport_loader = frozen_run.corpus.load_transport_corpus_index
    with terminal_v4_validation_proxy() as proxy:
        with v5_adapter.v5_metadata_hooks(
            readiness_path=v5_runner.READINESS_PATH
        ):
            registry, readiness = frozen_run.corpus.load_bound_readiness(
                prelock_root=v5_runner.PRELOCK_ROOT,
                live_data_root=v5_runner.DATA_ROOT,
                expected_prelock_sha256=EXPECTED_PRELOCK_SHA256,
                expected_readiness_sha256=EXPECTED_READINESS_SHA256,
            )
            receipt = (
                frozen_run.external_freeze.validate_external_freeze_receipt(
                    v5_runner.EXTERNAL_FREEZE_RECEIPT,
                    EXPECTED_PRELOCK_SHA256,
                    EXPECTED_READINESS_SHA256,
                    prelock_root=v5_runner.PRELOCK_ROOT,
                    readiness_path=v5_runner.READINESS_PATH,
                    live=False,
                )
            )
            if frozen_run.corpus.load_transport_corpus_index is not transport_loader:
                raise RuntimeError("trajectory loader changed during metadata preflight")
    if (
        proxy.call_count != 2
        or readiness.report.get("readiness_sha256")
        != EXPECTED_READINESS_SHA256
    ):
        raise ValueError("published v5 metadata preflight did not reach its boundary")
    return {
        "prelock_registry_sha256": (
            v5_runner.frozen_prelock.canonical_sha256(registry)
        ),
        "readiness_sha256": readiness.report["readiness_sha256"],
        "v5_external_freeze_schema": receipt.get("schema"),
        "terminal_v4_proxy_call_count": proxy.call_count,
        "trajectory_loader_called": False,
    }


def invoke_frozen_evaluation(
    command: str,
    *,
    expected_prelock_sha256: str = EXPECTED_PRELOCK_SHA256,
    expected_readiness_sha256: str = EXPECTED_READINESS_SHA256,
    prelock_root: Path = v5_runner.PRELOCK_ROOT,
    data_root: Path = v5_runner.DATA_ROOT,
    state_root: Path = v5_runner.STATE_ROOT,
    readiness_path: Path = v5_runner.READINESS_PATH,
    external_freeze_receipt_path: Path = v5_runner.EXTERNAL_FREEZE_RECEIPT,
    recovery_prelock_root: Path = DEFAULT_RECOVERY_PRELOCK,
    recovery_public_freeze_receipt: Path = DEFAULT_RECOVERY_FREEZE_RECEIPT,
    closeout_path: Path = closeout.DEFAULT_CLOSEOUT,
    output_dir: Path = DEFAULT_OUTPUT,
    live_external_freezes: bool = True,
    attempt_path: Path | None = None,
):
    if command not in {"run", "verify"}:
        raise ValueError("evaluation command must be run or verify")
    if (
        expected_prelock_sha256 != EXPECTED_PRELOCK_SHA256
        or expected_readiness_sha256 != EXPECTED_READINESS_SHA256
    ):
        raise ValueError("recovery accepts only the exact frozen v5 identity")
    exact_paths = {
        "prelock_root": (prelock_root, v5_runner.PRELOCK_ROOT),
        "data_root": (data_root, v5_runner.DATA_ROOT),
        "state_root": (state_root, v5_runner.STATE_ROOT),
        "readiness_path": (readiness_path, v5_runner.READINESS_PATH),
        "external_freeze_receipt_path": (
            external_freeze_receipt_path,
            v5_runner.EXTERNAL_FREEZE_RECEIPT,
        ),
        "closeout_path": (closeout_path, closeout.DEFAULT_CLOSEOUT),
        "output_dir": (output_dir, DEFAULT_OUTPUT),
    }
    changed_paths = [
        name
        for name, (actual, expected) in exact_paths.items()
        if actual.resolve() != expected.resolve()
    ]
    if changed_paths:
        raise ValueError(
            "recovery path differs from its frozen identity: "
            + ", ".join(changed_paths)
        )

    from . import freeze, public_freeze

    closeout.validate_terminal_closeout(closeout_path)
    freeze.verify_local_prelock(
        recovery_prelock_root, closeout_path=closeout_path
    )
    public_freeze.validate_public_freeze_receipt(
        recovery_public_freeze_receipt,
        recovery_prelock_root,
        live=live_external_freezes,
    )
    recovery_digest = (
        recovery_prelock_root / freeze.DIGEST_NAME
    ).read_text(encoding="ascii").strip()
    if command == "run":
        if attempt_path is None:
            raise ValueError("frozen run requires a write-once recovery attempt")
        attempt.validate_attempt(
            attempt_path,
            recovery_prelock_sha256=recovery_digest,
            recovery_public_freeze_receipt_path=(
                recovery_public_freeze_receipt
            ),
            v5_readiness_sha256=EXPECTED_READINESS_SHA256,
            output_dir=output_dir,
        )
    verified_before = verify_upstream_inputs(
        live_external_freeze=live_external_freezes
    )
    source_before = v5_adapter.frozen_source_snapshot()
    shared_before = v5_adapter.shared_runtime_snapshot(prelock_root)
    verify_v4_terminal_audit_dispatch()
    verify_v5_metadata_boundary()
    kwargs = {
        "expected_prelock_sha256": expected_prelock_sha256,
        "expected_readiness_sha256": expected_readiness_sha256,
        "prelock_root": prelock_root,
        "data_root": data_root,
        "state_root": state_root,
        "readiness_path": readiness_path,
        "external_freeze_receipt_path": external_freeze_receipt_path,
        "output_dir": output_dir,
        "live_external_freeze": live_external_freezes,
    }
    try:
        with terminal_v4_validation_proxy() as proxy:
            result = v5_adapter.invoke_frozen_evaluation(command, **kwargs)
        if proxy.call_count != 2:
            raise RuntimeError(
                "published v5 adapter did not perform two terminal-v4 audits"
            )
        return result
    finally:
        if v5_adapter.frozen_source_snapshot() != source_before:
            raise RuntimeError("frozen evaluator source changed during recovery")
        if v5_adapter.shared_runtime_snapshot(prelock_root) != shared_before:
            raise RuntimeError("shared numerical runtime changed during recovery")
        if verify_upstream_inputs(
            live_external_freeze=False
        )["upstream_file_sha256"] != verified_before["upstream_file_sha256"]:
            raise RuntimeError("recovery upstream bytes changed during invocation")


def _recovery_digest(recovery_prelock_root: Path) -> str:
    from . import freeze

    record = (recovery_prelock_root / freeze.DIGEST_NAME).read_text(
        encoding="ascii"
    )
    if (
        len(record) != 65
        or not record.endswith("\n")
        or any(character not in "0123456789abcdef" for character in record[:-1])
    ):
        raise ValueError("v6 recovery prelock digest is malformed")
    return record[:-1]


def recovery_state_root(recovery_prelock_root: Path) -> Path:
    return DEFAULT_STATE_BASE / _recovery_digest(recovery_prelock_root)


def run_verify_persistent(**kwargs) -> list[dict]:
    """Run and verify exactly once under a public recovery digest."""

    from . import freeze, public_freeze

    recovery_prelock_root = Path(
        kwargs.get("recovery_prelock_root", DEFAULT_RECOVERY_PRELOCK)
    )
    recovery_receipt_path = Path(
        kwargs.get(
            "recovery_public_freeze_receipt",
            DEFAULT_RECOVERY_FREEZE_RECEIPT,
        )
    )
    output_dir = Path(kwargs.get("output_dir", DEFAULT_OUTPUT))
    freeze.verify_local_prelock(recovery_prelock_root)
    public_receipt = public_freeze.validate_public_freeze_receipt(
        recovery_receipt_path,
        recovery_prelock_root,
        live=True,
    )
    digest = _recovery_digest(recovery_prelock_root)
    state_root = recovery_state_root(recovery_prelock_root)
    with attempt.exclusive_attempt_lock(state_root):
        attempt_path = attempt.begin_attempt(
            state_root,
            recovery_prelock_sha256=digest,
            recovery_public_freeze_receipt_path=recovery_receipt_path,
            recovery_public_freeze=public_receipt,
            v5_readiness_sha256=EXPECTED_READINESS_SHA256,
            output_dir=output_dir,
        )
        try:
            run_result = invoke_frozen_evaluation(
                "run", attempt_path=attempt_path, **kwargs
            )
            verify_result = invoke_frozen_evaluation("verify", **kwargs)
            completion_path = attempt.record_completion(
                state_root,
                attempt_path=attempt_path,
                recovery_prelock_sha256=digest,
                output_dir=output_dir,
            )
        except BaseException as error:
            attempt.record_failure(
                state_root,
                attempt_path=attempt_path,
                error=error,
                output_dir=output_dir,
            )
            raise
    return [
        {"command": "run", "result": str(run_result)},
        {"command": "verify", "result": verify_result},
        {"recovery_completion": str(completion_path)},
    ]


def verify_completed_evaluation(**kwargs) -> dict:
    recovery_prelock_root = Path(
        kwargs.get("recovery_prelock_root", DEFAULT_RECOVERY_PRELOCK)
    )
    output_dir = Path(kwargs.get("output_dir", DEFAULT_OUTPUT))
    state = attempt.validate_completion(
        recovery_state_root(recovery_prelock_root),
        recovery_prelock_sha256=_recovery_digest(recovery_prelock_root),
        recovery_public_freeze_receipt_path=Path(
            kwargs.get(
                "recovery_public_freeze_receipt",
                DEFAULT_RECOVERY_FREEZE_RECEIPT,
            )
        ),
        v5_readiness_sha256=EXPECTED_READINESS_SHA256,
        output_dir=output_dir,
    )
    result = invoke_frozen_evaluation("verify", **kwargs)
    return {"recovery_state": state, "frozen_verification": result}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "run-verify"))
    parser.add_argument(
        "--expected-prelock-sha256", default=EXPECTED_PRELOCK_SHA256
    )
    parser.add_argument(
        "--expected-readiness-sha256", default=EXPECTED_READINESS_SHA256
    )
    parser.add_argument("--prelock-root", type=Path, default=v5_runner.PRELOCK_ROOT)
    parser.add_argument("--data-root", type=Path, default=v5_runner.DATA_ROOT)
    parser.add_argument("--state-root", type=Path, default=v5_runner.STATE_ROOT)
    parser.add_argument("--readiness", type=Path, default=v5_runner.READINESS_PATH)
    parser.add_argument(
        "--external-freeze-receipt",
        type=Path,
        default=v5_runner.EXTERNAL_FREEZE_RECEIPT,
    )
    parser.add_argument(
        "--recovery-prelock-root",
        type=Path,
        default=DEFAULT_RECOVERY_PRELOCK,
    )
    parser.add_argument(
        "--recovery-public-freeze-receipt",
        type=Path,
        default=DEFAULT_RECOVERY_FREEZE_RECEIPT,
    )
    parser.add_argument(
        "--failed-v5-closeout", type=Path, default=closeout.DEFAULT_CLOSEOUT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-live-external-freezes", action="store_true")
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
        "external_freeze_receipt_path": args.external_freeze_receipt.resolve(),
        "recovery_prelock_root": args.recovery_prelock_root.resolve(),
        "recovery_public_freeze_receipt": (
            args.recovery_public_freeze_receipt.resolve()
        ),
        "closeout_path": args.failed_v5_closeout.resolve(),
        "output_dir": args.output.resolve(),
        "live_external_freezes": not args.no_live_external_freezes,
    }
    if args.command == "run-verify":
        results: object = run_verify_persistent(**common)
    else:
        results = verify_completed_evaluation(**common)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
