from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import pandas as pd

from . import closeout, freeze, public_freeze


def test_terminal_closeout_is_metadata_only_and_live_bound() -> None:
    payload = closeout.validate_terminal_closeout()
    assert payload["numerical_evaluation_started"] is False
    assert payload["locked_trajectory_or_outcome_values_read_by_closeout"] is False
    assert payload["failed_output"]["exists"] is False
    assert payload["failed_staging"]["exists"] is False


def test_prelock_hashes_only_legacy_v4_csv_bytes_and_never_parses_csv_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open
    opened_csv: list[Path] = []
    transport_loader_calls = 0
    pandas_read_csv_calls = 0

    def tracked_open(self: Path, *args, **kwargs):
        if self.suffix.lower() == ".csv":
            opened_csv.append(self)
        return original_open(self, *args, **kwargs)

    def forbidden_transport_loader(*args, **kwargs):
        nonlocal transport_loader_calls
        transport_loader_calls += 1
        raise AssertionError("v5 locked trajectory loader was called")

    def forbidden_pandas_read_csv(*args, **kwargs):
        nonlocal pandas_read_csv_calls
        pandas_read_csv_calls += 1
        raise AssertionError("CSV values were parsed")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(
        freeze.adapter.frozen_run.corpus,
        "load_transport_corpus_index",
        forbidden_transport_loader,
    )
    monkeypatch.setattr(pd, "read_csv", forbidden_pandas_read_csv)
    root = freeze.prepare_local_prelock(tmp_path / "prelock")
    freeze.verify_local_prelock(root)
    paths = freeze.public_freeze_input_paths(root)
    assert transport_loader_calls == 0
    assert pandas_read_csv_calls == 0
    assert opened_csv
    assert all(
        "data_v7/locked_transport_raw" in path.as_posix()
        for path in opened_csv
    )
    assert not any("data" in path.as_posix() for path in opened_csv)
    assert not any("evaluation" in path.name for path in opened_csv)
    assert paths
    assert not any(path.suffix.lower() == ".csv" for path in paths.values())


def test_rejected_prelocks_are_bound_and_have_no_public_or_state_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = freeze.validate_rejected_v1_record()
    assert payload["rejected_before_publication"] is True
    assert payload["scientific_values_parsed"] is False
    assert payload["legacy_v4_csv_bytes_hashed"] is True
    assert payload["publication_or_retry_under_v1_permitted"] is False
    chain = freeze.validate_rejection_chain()
    assert [row["version"] for row in chain["rejected_prelocks"]] == ["v1", "v2"]
    assert all(
        row["evaluation_output_or_staging_existed_at_rejection"] is False
        for row in chain["rejected_prelocks"]
    )
    assert all(
        row["publication_or_retry_permitted"] is False
        for row in chain["rejected_prelocks"]
    )

    # A future legitimate v3 output must not invalidate historical rejection
    # evidence; v3 attempt state separately enforces output freshness.
    occupied_future_output = tmp_path / "future-v3-output"
    occupied_future_output.mkdir()
    monkeypatch.setattr(freeze.adapter, "DEFAULT_OUTPUT", occupied_future_output)
    freeze.validate_rejected_v1_record()
    freeze.validate_rejection_chain()


def test_snapshot_drift_is_rejected(tmp_path: Path) -> None:
    root = freeze.prepare_local_prelock(tmp_path / "prelock")
    snapshot = root / "bundle/source/adapter.py"
    os.chmod(snapshot, 0o644)
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inventory|snapshot"):
        freeze.verify_local_prelock(root)


def test_offline_public_receipt_binds_exact_prelock(tmp_path: Path) -> None:
    root = freeze.prepare_local_prelock(tmp_path / "prelock")
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
        "provider_created_at_utc": "2026-07-23T10:00:00Z",
        "provider_updated_at_utc": "2026-07-23T10:01:00Z",
        "revision_committed_at_utc": "2026-07-23T10:00:30Z",
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
