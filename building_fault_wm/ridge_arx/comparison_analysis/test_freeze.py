from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from . import freeze
from . import public_freeze


def test_public_freeze_set_is_metadata_and_source_only(tmp_path: Path) -> None:
    root = freeze.prepare_local_freeze_bundle(tmp_path / "freeze")
    paths = freeze.public_freeze_input_paths(root)
    assert paths
    assert not any(path.suffix == ".csv" for path in paths.values())
    contract = json.loads(
        paths["analysis_contract.json"].read_text(encoding="ascii")
    )
    assert contract["outcome_values_accessed"] is False
    assert contract["bootstrap"]["draws"] == 10_000
    assert contract["upstream_evaluator_contracts"]["v3"][
        "completion_schema"
    ]
    assert contract["upstream_evaluator_contracts"]["arx"][
        "completion_schema"
    ]


def test_source_snapshot_hash_drift_is_rejected(tmp_path: Path) -> None:
    root = freeze.prepare_local_freeze_bundle(tmp_path / "freeze")
    source = root / "bundle/source/analysis.py"
    os.chmod(source, 0o644)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inventory|source|contract"):
        freeze.verify_local_freeze_bundle(root)


def test_public_receipt_validates_offline_against_metadata_only_set(
    tmp_path: Path,
) -> None:
    root = freeze.prepare_local_freeze_bundle(tmp_path / "freeze")
    gist = "c" * 32
    revision = "d" * 40
    owner = "owner"
    receipt = {
        "schema": public_freeze.SCHEMA,
        "provider": public_freeze.PROVIDER,
        "public": True,
        "prelock_registry_sha256": public_freeze._prelock_digest(root),
        "gist_id": gist,
        "revision": revision,
        "owner_login": owner,
        "provider_created_at_utc": "2026-08-01T00:00:00Z",
        "provider_updated_at_utc": "2026-08-01T00:01:00Z",
        "revision_committed_at_utc": "2026-08-01T00:00:30Z",
        "revision_api_url": f"https://api.github.com/gists/{gist}/{revision}",
        "revision_html_url": (
            f"https://gist.github.com/{owner}/{gist}/{revision}"
        ),
        "file_sha256_by_name": public_freeze.expected_file_hashes(root),
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="ascii")
    assert public_freeze.validate_public_freeze_receipt(
        path, root, live=False
    ) == receipt
