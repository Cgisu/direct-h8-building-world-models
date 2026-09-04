#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/downstream"
SUMMARY_FILES = (
    "aggregate_summary.csv",
    "episode_summary.csv",
    "finite_panel_summary.csv",
    "paired_effects.csv",
    "run_metadata.json",
)

def canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def main() -> int:
    manifest = load_object(RESULTS / "report_manifest.json")
    payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if manifest.get("payload_sha256") != canonical_sha256(payload):
        raise ValueError("downstream report manifest payload digest differs")
    if (
        RESULTS.joinpath("report_manifest.canonical.sha256")
        .read_text(encoding="ascii")
        .strip()
        != canonical_sha256(manifest)
    ):
        raise ValueError("downstream report manifest canonical digest differs")
    declared = {str(row["path"]): row for row in manifest["files"]}
    for name in SUMMARY_FILES:
        path = RESULTS / name
        row = declared.get(name)
        if row is None:
            raise ValueError(f"summary absent from sealed manifest: {name}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"summary byte count differs: {name}")
        if file_sha256(path) != str(row["sha256"]):
            raise ValueError(f"summary digest differs: {name}")

    protocol = load_object(RESULTS / "protocol.json")
    if (
        RESULTS.joinpath("protocol.canonical.sha256")
        .read_text(encoding="ascii")
        .strip()
        != canonical_sha256(protocol)
    ):
        raise ValueError("downstream protocol canonical digest differs")
    metadata = load_object(RESULTS / "run_metadata.json")
    if metadata.get("protocol_file_sha256") != file_sha256(RESULTS / "protocol.json"):
        raise ValueError("downstream run protocol file receipt differs")
    if metadata.get("protocol_canonical_sha256") != canonical_sha256(protocol):
        raise ValueError("downstream run protocol canonical receipt differs")
    tuning = load_object(RESULTS / "tuning.json")
    receipt = protocol.get("tuning_receipt", {})
    if receipt.get("sha256") != file_sha256(RESULTS / "tuning.json"):
        raise ValueError("downstream tuning file receipt differs")
    tuning_payload = {
        key: value for key, value in tuning.items() if key != "payload_sha256"
    }
    if tuning.get("payload_sha256") != canonical_sha256(tuning_payload):
        raise ValueError("downstream tuning payload digest differs")
    if receipt.get("canonical_sha256") != tuning.get("payload_sha256"):
        raise ValueError("downstream tuning canonical receipt differs")
    if int(metadata.get("episodes", -1)) != 1296:
        raise ValueError("downstream episode count differs")
    print("DOWNSTREAM RESULT SUBSET: PASS")
    print(f"  sealed summaries: {len(SUMMARY_FILES)}")
    print(f"  episodes: {metadata['episodes']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
