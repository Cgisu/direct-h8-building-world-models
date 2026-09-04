"""Bind the intended v5 and ARX study instances before result CSV access."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from building_fault_wm.deterministic_transport import (
    evaluate as v3_evaluate,
    gate as v3_gate,
    run_evaluation as v3_run,
)
from building_fault_wm.transport_collection import (
    external_freeze as transport_external_freeze,
    runner as transport_runner,
)
from building_fault_wm.ridge_arx import (
    evaluate as arx_evaluate,
    external_freeze as arx_external_freeze,
    lock as arx_lock,
    train as arx_train,
)
from building_fault_wm.ridge_arx.comparison_analysis import (
    analysis as frozen_comparison,
    freeze as comparison_freeze,
    public_freeze as comparison_public_freeze,
)
from building_fault_wm.ridge_arx.io import (
    canonical_sha256,
    sha256_file,
    strict_json,
    tree_inventory,
    write_json_once,
)


SCHEMA = "schedule-matched-arx-neural-identity-binding-v1"
PROVENANCE_SCHEMA = "schedule-matched-arx-neural-identity-guard-provenance-v1"
COMPLETION_SCHEMA = "schedule-matched-arx-neural-identity-guard-completion-v1"
EXPECTED_TRANSPORT_PRELOCK_SHA256 = (
    "50dbd5d24537b61e109ff6634361ddb9ca9bceac2528b57394125a6667d80094"
)
EXPECTED_TRANSPORT_READINESS_SHA256 = (
    "d245795503482417ac1d717782f33c56d05b0fa96d72f5156d5e954d4cdba74b"
)
EXPECTED_COMPARISON_PRELOCK_SHA256 = (
    "812db4961dfd98424c045db8bc1812662874e109b95d7ca650e77d3588e85e11"
)
TRAINING_GRID_NAME = "training_grid_complete.json"
TRAINING_SOURCE_LOCK_NAME = "training_source_lock.json"


@dataclass(frozen=True)
class BindingPaths:
    transport_prelock_root: Path
    transport_live_data_root: Path
    transport_readiness_path: Path
    transport_external_freeze_receipt_path: Path
    transport_state_root: Path
    transport_manifest_path: Path
    arx_prelock_root: Path
    arx_external_freeze_receipt_path: Path
    arx_training_root: Path
    comparison_prelock_root: Path
    comparison_public_freeze_receipt_path: Path


def _digest_record(path: Path, label: str) -> str:
    value = path.read_text(encoding="ascii")
    if (
        len(value) != 65
        or value[-1] != "\n"
        or any(character not in "0123456789abcdef" for character in value[:-1])
    ):
        raise ValueError(f"{label} digest record is malformed")
    return value[:-1]


def _model_hashes(training_grid: Mapping[str, object]) -> dict[str, str]:
    rows = training_grid.get("runs")
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError("ARX training grid does not contain exactly 15 runs")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ARX training-grid run is not an object")
        case = row.get("case")
        seed = row.get("model_seed")
        digest = row.get("model_file_sha256")
        if (
            not isinstance(case, str)
            or not isinstance(seed, int)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("ARX training-grid model identity changed")
        key = f"{case}/seed{seed}"
        if key in result:
            raise ValueError("ARX training-grid model identity is duplicated")
        result[key] = digest
    if set(result) != {
        f"{case}/seed{seed}"
        for case in arx_evaluate.CASES
        for seed in arx_evaluate.FROZEN_CONFIG.model_seeds
    }:
        raise ValueError("ARX training-grid model set changed")
    return dict(sorted(result.items()))


def build_binding_contract(
    paths: BindingPaths, *, live_external_freezes: bool
) -> dict:
    """Build metadata identities without opening a trajectory or result CSV."""

    readiness = strict_json(paths.transport_readiness_path)
    if (
        readiness.get("schema") != transport_runner.READINESS_SCHEMA
        or readiness.get("prelock_registry_sha256")
        != EXPECTED_TRANSPORT_PRELOCK_SHA256
        or readiness.get("readiness_sha256")
        != EXPECTED_TRANSPORT_READINESS_SHA256
        or readiness.get("locked_response_values_accessed") is not False
        or readiness.get("data_v7_raw_reuse_permitted") is not False
        or readiness.get("namespaces")
        != {"data": "data", "state": "state_v3", "freeze": "freeze_v5"}
    ):
        raise ValueError("transport readiness is not the intended v5 replacement")
    transport_receipt = (
        transport_external_freeze.validate_external_freeze_receipt(
            paths.transport_external_freeze_receipt_path,
            EXPECTED_TRANSPORT_PRELOCK_SHA256,
            EXPECTED_TRANSPORT_READINESS_SHA256,
            prelock_root=paths.transport_prelock_root,
            readiness_path=paths.transport_readiness_path,
            live=live_external_freezes,
        )
    )

    arx_registry = arx_lock.verify_prelock(
        paths.arx_prelock_root, verify_live_assets=True
    )
    arx_digest = _digest_record(
        paths.arx_prelock_root / arx_lock.DIGEST_NAME, "ARX prelock"
    )
    if arx_digest != canonical_sha256(arx_registry):
        raise ValueError("ARX prelock registry digest changed")
    arx_receipt = arx_external_freeze.validate_external_freeze_receipt(
        paths.arx_external_freeze_receipt_path,
        paths.arx_prelock_root,
        live=live_external_freezes,
    )
    arx_transport_binding_path = (
        paths.arx_prelock_root / "bundle/transport_collection_binding.json"
    )
    arx_training_binding_path = (
        paths.arx_prelock_root / "bundle/training_binding.json"
    )
    arx_transport_binding = strict_json(arx_transport_binding_path)
    arx_training_binding = strict_json(arx_training_binding_path)
    if (
        arx_transport_binding.get("transport_prelock_registry_sha256")
        != EXPECTED_TRANSPORT_PRELOCK_SHA256
        or arx_transport_binding.get("transport_readiness_sha256")
        != EXPECTED_TRANSPORT_READINESS_SHA256
        or Path(str(arx_training_binding.get("training_root"))).resolve()
        != paths.arx_training_root.resolve()
    ):
        raise ValueError("ARX prelock binds a different transport or training grid")
    transport_roles = arx_transport_binding.get("file_sha256_by_role")
    manifest_metadata = arx_transport_binding.get("manifest_metadata")
    if not isinstance(transport_roles, dict) or not isinstance(
        manifest_metadata, dict
    ):
        raise ValueError("ARX transport binding metadata changed")
    if (
        transport_roles.get("readiness")
        != sha256_file(paths.transport_readiness_path)
        or transport_roles.get("external_freeze_receipt")
        != sha256_file(paths.transport_external_freeze_receipt_path)
        or transport_roles.get("manifest")
        != sha256_file(paths.transport_manifest_path)
    ):
        raise ValueError("ARX prelock transport file binding changed")

    arx_train.verify_training_grid(paths.arx_training_root)
    training_grid_path = paths.arx_training_root / TRAINING_GRID_NAME
    training_source_lock_path = paths.arx_training_root / TRAINING_SOURCE_LOCK_NAME
    training_grid = strict_json(training_grid_path)
    training_inventory = tree_inventory(paths.arx_training_root)

    comparison_registry = comparison_freeze.verify_local_freeze_bundle(
        paths.comparison_prelock_root
    )
    comparison_digest = _digest_record(
        paths.comparison_prelock_root / comparison_freeze.DIGEST_NAME,
        "comparison prelock",
    )
    if (
        comparison_digest != EXPECTED_COMPARISON_PRELOCK_SHA256
        or comparison_digest != canonical_sha256(comparison_registry)
    ):
        raise ValueError("comparison prelock is not immutable v2")
    comparison_receipt = (
        comparison_public_freeze.validate_public_freeze_receipt(
            paths.comparison_public_freeze_receipt_path,
            paths.comparison_prelock_root,
            live=live_external_freezes,
        )
    )

    expected_v3_input_hashes = {
        "prelock_registry_sha256": EXPECTED_TRANSPORT_PRELOCK_SHA256,
        "readiness_sha256": EXPECTED_TRANSPORT_READINESS_SHA256,
        "external_freeze_receipt_file_sha256": sha256_file(
            paths.transport_external_freeze_receipt_path
        ),
        "corpus_manifest_file_sha256": manifest_metadata.get(
            "manifest_file_sha256"
        ),
        "corpus_manifest_payload_sha256": manifest_metadata.get(
            "manifest_payload_sha256"
        ),
        "collection_attempt_file_sha256": transport_roles.get("attempt"),
        "collection_completion_file_sha256": transport_roles.get("completion"),
    }
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in expected_v3_input_hashes.values()
    ):
        raise ValueError("v5 transport binding lacks an expected input hash")
    expected_v3_evaluation_contract = {
        "cases": list(v3_gate.CASES),
        "model_seeds": list(v3_gate.CONFIRMATION_SEEDS),
        "arms": list(v3_gate.ARMS),
        "policies": list(v3_gate.POLICIES),
        "horizons": list(v3_evaluate.EVALUATION_HORIZONS),
        "history": v3_evaluate.EVALUATION_HISTORY,
        "role": "locked_test",
        "fault_spec": json.loads(
            json.dumps(asdict(arx_evaluate.load_frozen_fault_spec()))
        ),
        "bootstrap_draws": v3_gate.BOOTSTRAP_DRAWS,
        "bootstrap_seed": v3_gate.BOOTSTRAP_SEED,
    }

    return {
        "schema": SCHEMA,
        "outcome_values_accessed": False,
        "transport": {
            "prelock_registry_sha256": EXPECTED_TRANSPORT_PRELOCK_SHA256,
            "readiness_sha256": EXPECTED_TRANSPORT_READINESS_SHA256,
            "readiness_file_sha256": sha256_file(
                paths.transport_readiness_path
            ),
            "external_freeze_receipt_file_sha256": sha256_file(
                paths.transport_external_freeze_receipt_path
            ),
            "external_freeze_revision": transport_receipt["revision"],
            "manifest_file_sha256": sha256_file(paths.transport_manifest_path),
            "transport_binding_payload_sha256": canonical_sha256(
                arx_transport_binding
            ),
        },
        "neural_evaluation": {
            "completion_schema": v3_run.COMPLETION_SCHEMA,
            "provenance_schema": v3_run.PROVENANCE_SCHEMA,
            "study_kind": "direct_h8_deterministic_transport_v3",
            "run_evaluation_source_sha256": sha256_file(
                Path(v3_run.__file__).resolve()
            ),
            "gate_source_sha256": sha256_file(Path(v3_gate.__file__).resolve()),
            "expected_input_hashes": expected_v3_input_hashes,
            "external_freeze_revision": transport_receipt["revision"],
            "evaluation_contract": expected_v3_evaluation_contract,
        },
        "arx": {
            "prelock_registry_sha256": arx_digest,
            "prelock_registry_file_sha256": sha256_file(
                paths.arx_prelock_root / arx_lock.REGISTRY_NAME
            ),
            "external_freeze_receipt_file_sha256": sha256_file(
                paths.arx_external_freeze_receipt_path
            ),
            "external_freeze_revision": arx_receipt["revision"],
            "training_grid_file_sha256": sha256_file(training_grid_path),
            "training_source_lock_file_sha256": sha256_file(
                training_source_lock_path
            ),
            "training_tree_inventory_sha256": canonical_sha256(
                training_inventory
            ),
            "training_binding_file_sha256": sha256_file(
                arx_training_binding_path
            ),
            "training_binding_payload_sha256": canonical_sha256(
                arx_training_binding
            ),
            "transport_binding_file_sha256": sha256_file(
                arx_transport_binding_path
            ),
            "transport_binding_payload_sha256": canonical_sha256(
                arx_transport_binding
            ),
            "transport_manifest_file_sha256": manifest_metadata[
                "manifest_file_sha256"
            ],
            "evaluate_source_sha256": sha256_file(
                Path(arx_evaluate.__file__).resolve()
            ),
            "config": json.loads(
                json.dumps(arx_evaluate.FROZEN_CONFIG.to_dict())
            ),
            "model_file_sha256_by_case_seed": _model_hashes(training_grid),
        },
        "comparison": {
            "prelock_registry_sha256": comparison_digest,
            "public_freeze_receipt_file_sha256": sha256_file(
                paths.comparison_public_freeze_receipt_path
            ),
            "public_freeze_revision": comparison_receipt["revision"],
            "analysis_source_sha256": sha256_file(
                Path(frozen_comparison.__file__).resolve()
            ),
        },
    }


def validate_binding_contract(contract: Mapping[str, object]) -> None:
    if (
        set(contract) != {
            "schema",
            "outcome_values_accessed",
            "transport",
            "neural_evaluation",
            "arx",
            "comparison",
        }
        or contract.get("schema") != SCHEMA
        or contract.get("outcome_values_accessed") is not False
        or not all(
            isinstance(contract.get(key), dict)
            for key in ("transport", "neural_evaluation", "arx", "comparison")
        )
    ):
        raise ValueError("identity-guard binding contract fields changed")
    transport = contract["transport"]
    comparison = contract["comparison"]
    if (
        transport.get("prelock_registry_sha256")
        != EXPECTED_TRANSPORT_PRELOCK_SHA256
        or transport.get("readiness_sha256")
        != EXPECTED_TRANSPORT_READINESS_SHA256
        or comparison.get("prelock_registry_sha256")
        != EXPECTED_COMPARISON_PRELOCK_SHA256
    ):
        raise ValueError("identity-guard fixed study identities changed")


def verify_v3_evaluation_metadata(root: Path, contract: Mapping[str, object]) -> dict:
    """Verify the complete intended v5 neural output without parsing its core CSV."""

    _, completion = frozen_comparison._verify_v3_output(root)
    expected = contract["neural_evaluation"]
    if (
        completion.get("schema") != expected["completion_schema"]
        or completion.get("study_kind") != expected["study_kind"]
        or completion.get("prelock_registry_sha256")
        != EXPECTED_TRANSPORT_PRELOCK_SHA256
        or completion.get("readiness_sha256")
        != EXPECTED_TRANSPORT_READINESS_SHA256
        or completion.get("corpus_manifest_payload_sha256")
        != expected["expected_input_hashes"]["corpus_manifest_payload_sha256"]
    ):
        raise ValueError("neural evaluation is a different study instance")
    provenance = strict_json(root / v3_run.PROVENANCE_NAME)
    if (
        provenance.get("schema") != expected["provenance_schema"]
        or provenance.get("study_kind") != expected["study_kind"]
        or completion.get("provenance_file_sha256")
        != sha256_file(root / v3_run.PROVENANCE_NAME)
    ):
        raise ValueError("neural evaluation provenance identity changed")
    input_hashes = provenance.get("input_hashes")
    if not isinstance(input_hashes, dict) or any(
        input_hashes.get(key) != value
        for key, value in expected["expected_input_hashes"].items()
    ):
        raise ValueError("neural evaluation provenance binds different inputs")
    external = provenance.get("external_freeze")
    if (
        not isinstance(external, dict)
        or external.get("revision") != expected["external_freeze_revision"]
    ):
        raise ValueError("neural evaluation provenance binds a different freeze")
    if provenance.get("evaluation_contract") != expected["evaluation_contract"]:
        raise ValueError("neural evaluation contract changed")
    return completion


def verify_arx_evaluation_metadata(root: Path, contract: Mapping[str, object]) -> dict:
    """Verify the complete intended ARX output without parsing its core CSV."""

    _, completion = frozen_comparison._verify_arx_output(root)
    expected = contract["arx"]
    provenance = strict_json(root / "evaluation_provenance.json")
    if (
        provenance.get("schema") != arx_evaluate.OUTPUT_SCHEMA
        or provenance.get("secondary_only") is not True
        or provenance.get("cannot_modify_v2_or_v3_gate") is not True
        or provenance.get("prelock_registry_sha256")
        != expected["prelock_registry_sha256"]
        or provenance.get("addendum_external_freeze_receipt_sha256")
        != expected["external_freeze_receipt_file_sha256"]
        or provenance.get("addendum_external_freeze_revision")
        != expected["external_freeze_revision"]
        or provenance.get("transport_collection_binding_sha256")
        != expected["transport_binding_payload_sha256"]
        or provenance.get("transport_manifest_file_sha256")
        != expected["transport_manifest_file_sha256"]
        or provenance.get("config") != expected["config"]
        or provenance.get("model_file_sha256_by_case_seed")
        != expected["model_file_sha256_by_case_seed"]
        or provenance.get("rows") != completion.get("row_count")
    ):
        raise ValueError("ARX evaluation provenance binds a different study instance")
    return completion


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def run_guarded_analysis(
    *,
    paths: BindingPaths,
    identity_prelock_root: Path,
    identity_public_freeze_receipt_path: Path,
    v3_output_root: Path,
    arx_output_root: Path,
    output_root: Path,
    live_external_freezes: bool = True,
) -> Path:
    """Validate every identity before delegating to the immutable comparison."""

    if os.path.lexists(output_root):
        raise FileExistsError(
            f"refusing to overwrite guarded comparison output: {output_root}"
        )
    from .freeze import load_verified_binding_contract
    from .public_freeze import validate_public_freeze_receipt

    contract = load_verified_binding_contract(identity_prelock_root)
    validate_public_freeze_receipt(
        identity_public_freeze_receipt_path,
        identity_prelock_root,
        live=live_external_freezes,
    )
    current = build_binding_contract(
        paths, live_external_freezes=live_external_freezes
    )
    if current != contract:
        raise ValueError("live metadata differs from the identity-guard prelock")
    v3_completion = verify_v3_evaluation_metadata(v3_output_root, contract)
    arx_completion = verify_arx_evaluation_metadata(arx_output_root, contract)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging.",
            dir=output_root.parent,
        )
    )
    try:
        comparison_output = staging / "comparison"
        comparison_complete = frozen_comparison.run_bound_analysis(
            v3_output_root=v3_output_root,
            arx_output_root=arx_output_root,
            freeze_bundle_root=paths.comparison_prelock_root,
            public_freeze_receipt_path=(
                paths.comparison_public_freeze_receipt_path
            ),
            output_root=comparison_output,
            live_public_freeze=live_external_freezes,
        )
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "identity_wrapper_prelock_registry_sha256": _digest_record(
                identity_prelock_root / "identity_guard_prelock.canonical.sha256",
                "identity guard",
            ),
            "identity_wrapper_public_freeze_receipt_file_sha256": (
                sha256_file(identity_public_freeze_receipt_path)
            ),
            "neural_evaluation_completion_file_sha256": sha256_file(
                v3_output_root / v3_run.COMPLETION_NAME
            ),
            "arx_evaluation_completion_file_sha256": sha256_file(
                arx_output_root / "evaluation_complete.json"
            ),
            "v5_readiness_sha256": EXPECTED_TRANSPORT_READINESS_SHA256,
            "binding_contract_payload_sha256": canonical_sha256(contract),
            "v3_completion_payload_sha256": canonical_sha256(v3_completion),
            "arx_completion_payload_sha256": canonical_sha256(arx_completion),
            "frozen_comparison_completion_file_sha256": sha256_file(
                comparison_complete
            ),
            "all_identity_checks_preceded_result_csv_access": True,
        }
        write_json_once(staging / "identity_guard_provenance.json", provenance)
        inventory = tree_inventory(staging)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "complete": True,
            "secondary_only": True,
            "artifact_inventory_excludes_completion": inventory,
            "artifact_inventory_sha256": canonical_sha256(inventory),
            "provenance_payload_sha256": canonical_sha256(provenance),
        }
        write_json_once(staging / "identity_guard_complete.json", completion)
        _seal_tree(staging)
        if os.path.lexists(output_root):
            raise FileExistsError(
                "guarded comparison destination became occupied"
            )
        staging.rename(output_root)
        return output_root / "identity_guard_complete.json"
    except Exception:
        try:
            shutil.rmtree(staging)
        except OSError:
            try:
                staging.chmod(stat.S_IRWXU)
            except OSError:
                pass
            for path in staging.rglob("*"):
                try:
                    path.chmod(stat.S_IRWXU)
                except OSError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
        raise
