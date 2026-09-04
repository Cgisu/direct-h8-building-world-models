from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from . import freeze, public_freeze
from .test_guard import base_contract, _dummy_paths


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(
        freeze,
        "build_binding_contract",
        lambda *args, **kwargs: base_contract(),
    )
    return freeze.prepare_local_prelock(
        tmp_path / "prelock",
        _dummy_paths(tmp_path),
        live_external_freezes=False,
    )


def test_local_prelock_and_public_set_are_metadata_source_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    freeze.verify_local_prelock(root)
    paths = freeze.public_freeze_input_paths(root)
    assert paths
    assert not any(path.suffix == ".csv" for path in paths.values())
    assert set(paths) == {
        freeze.REGISTRY_NAME,
        freeze.DIGEST_NAME,
        freeze.BINDING_NAME,
        *(
            f"source__{name}"
            for name in freeze.REQUIRED_SOURCE_FILES
        ),
    }


def test_source_snapshot_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
    source = root / "bundle/source/guard.py"
    os.chmod(source, 0o644)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inventory|snapshot"):
        freeze.verify_local_prelock(root)


def test_offline_public_receipt_binds_exact_prelock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare(tmp_path, monkeypatch)
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
