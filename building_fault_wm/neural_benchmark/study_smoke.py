from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .fault_data import build_fault_manifest, load_corpus_index
from .study_config import ARMS, StudyConfig
from .study_evaluate import evaluate_model_h8, load_model_checkpoint, result_sha256
from .study_train import prepare_case_training_data, train_case_seed


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_smoke(manifest_path: Path, output_dir: Path) -> Path:
    """Run one real update through every RSSM arm and the H8 evaluator."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output_dir}")
    index = load_corpus_index(manifest_path)
    if index.collection_kind != "smoke" or index.allowed_roles != ("fit",):
        raise ValueError("integration smoke requires one sealed smoke corpus")
    cases = tuple(sorted({record.key.case for record in index.records}))
    if len(cases) != 3 or len(index.records) != 3:
        raise ValueError("integration smoke requires one FIT trajectory in all three cases")

    config = replace(
        StudyConfig(),
        updates=1,
        checkpoint_every=1,
        validation_checkpoints=(1,),
    )
    model_seed = config.development_seeds[0]
    started = time.monotonic()
    manifest = build_fault_manifest(index)
    frames = []
    completion_hashes: dict[str, str] = {}
    core_invariants: dict[str, bool] = {}
    for case in cases:
        completion_path = train_case_seed(
            index,
            case=case,
            model_seed=model_seed,
            output_dir=output_dir / "training",
            config=config,
            arms=ARMS,
        )
        completion = json.loads(completion_path.read_text(encoding="ascii"))
        completion_hashes[case] = hashlib.sha256(
            completion_path.read_bytes()
        ).hexdigest()
        core_hashes = completion["checkpoint_core_state_sha256"]
        core_invariants[case] = (
            core_hashes["ungated_h8_u0001.pt"]
            == core_hashes["aux_h8_u0001.pt"]
        )
        if not core_invariants[case]:
            raise AssertionError("auxiliary negative-control RSSM core diverged")

        variants, scalers = prepare_case_training_data(index, manifest, case)
        run_dir = completion_path.parent
        for arm in ARMS:
            checkpoint = run_dir / "checkpoints" / f"{arm}_u0001.pt"
            model = load_model_checkpoint(
                checkpoint,
                config,
                case=case,
                model_seed=model_seed,
                arm=arm,
                update=1,
                expected_checkpoint_sha256=completion["checkpoint_sha256"][
                    checkpoint.name
                ],
                expected_provenance=completion["provenance"],
            )
            frames.append(
                evaluate_model_h8(
                    model,
                    variants,
                    scalers,
                    config,
                    arm=arm,
                    case=case,
                    model_seed=model_seed,
                    update=1,
                    role="fit",
                )
            )

    frame = pd.concat(frames, ignore_index=True)
    if set(frame["role"]) != {"fit"} or set(frame["case"]) != set(cases):
        raise AssertionError("smoke evaluation crossed a role or case boundary")
    expected_rows = 176 * len(ARMS) * len(cases)
    if len(frame) != expected_rows:
        raise AssertionError(
            f"smoke evaluation produced {len(frame)} rows, expected {expected_rows}"
        )
    invariants = frame.groupby(["case", "cell_id", "anchor"])[
        ["target_raw", "persistence_prediction_raw"]
    ].nunique()
    if (invariants != 1).any().any():
        raise AssertionError("paired smoke targets or persistence values differ by arm")

    result_path = output_dir / "fit_h8_results.csv"
    result_content = frame.to_csv(index=False).encode("ascii")
    _atomic_bytes(result_path, result_content)
    summary = {
        "schema": "boptest-reliability-rssm-integration-smoke-v1",
        "interpretation": "integration_only_not_a_scientific_result",
        "corpus_manifest_sha256": index.manifest_sha256,
        "fault_manifest_sha256": manifest.sha256,
        "cases": list(cases),
        "model_seed": model_seed,
        "arms": list(ARMS),
        "updates": 1,
        "rows": len(frame),
        "rows_by_case_arm": {
            f"{case}:{arm}": int(count)
            for (case, arm), count in frame.groupby(["case", "arm"]).size().items()
        },
        "aux_ungated_core_identical": core_invariants,
        "training_completion_sha256": completion_hashes,
        "result_sha256": result_sha256(frame),
        "wall_seconds": time.monotonic() - started,
    }
    summary_path = output_dir / "smoke_complete.json"
    _atomic_bytes(
        summary_path, (json.dumps(summary, indent=2) + "\n").encode("ascii")
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sealed three-case RSSM smoke")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run_smoke(args.manifest, args.output))


if __name__ == "__main__":
    main()
