"""Revision-pinned public freeze for the outcome-blind comparison contract."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from building_fault_wm.ridge_arx import (
    external_freeze as base,
)
from building_fault_wm.ridge_arx.io import (
    sha256_file,
    strict_json,
    write_json_once,
)

from .freeze import DIGEST_NAME, public_freeze_input_paths


SCHEMA = "schedule-matched-arx-neural-comparison-github-gist-freeze-v1"
PROVIDER = "github-gist"
RECEIPT_FIELDS = base.RECEIPT_FIELDS


def _prelock_digest(freeze_root: Path) -> str:
    record = (freeze_root / DIGEST_NAME).read_text(encoding="ascii")
    if (
        len(record) != 65
        or not record.endswith("\n")
        or not re.fullmatch(r"[0-9a-f]{64}", record[:-1])
    ):
        raise ValueError("comparison prelock digest record is malformed")
    return record[:-1]


def expected_file_hashes(freeze_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in sorted(public_freeze_input_paths(freeze_root).items())
    }


def build_public_freeze_receipt(
    gist_id: str,
    revision: str,
    owner_login: str,
    *,
    freeze_root: Path,
) -> dict:
    gist, revision, owner = base._identity(
        {
            "gist_id": gist_id,
            "revision": revision,
            "owner_login": owner_login,
        }
    )
    api_url = f"{base.GITHUB_API}/gists/{gist}/{revision}"
    html_url = f"https://gist.github.com/{owner}/{gist}/{revision}"
    response = base._fetch(api_url)
    history = base._history_entry(response.payload, revision)
    local_paths = public_freeze_input_paths(freeze_root)
    receipt = {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "public": True,
        "prelock_registry_sha256": _prelock_digest(freeze_root),
        "gist_id": gist,
        "revision": revision,
        "owner_login": owner,
        "provider_created_at_utc": response.payload.get("created_at"),
        "provider_updated_at_utc": response.payload.get("updated_at"),
        "revision_committed_at_utc": history.get("committed_at"),
        "revision_api_url": api_url,
        "revision_html_url": html_url,
        "file_sha256_by_name": {
            name: sha256_file(path)
            for name, path in sorted(local_paths.items())
        },
    }
    if set(receipt) != RECEIPT_FIELDS:
        raise AssertionError("comparison public-freeze receipt fields changed")
    for field in (
        "provider_created_at_utc",
        "provider_updated_at_utc",
        "revision_committed_at_utc",
    ):
        base._parse_utc(receipt[field], field)
    base._validate_remote(response, receipt, local_paths)
    return receipt


def create_public_freeze(
    *,
    freeze_root: Path,
    receipt_path: Path,
) -> dict:
    """Create the public record; callers must invoke this explicitly."""

    if os.path.lexists(receipt_path):
        raise FileExistsError(
            f"refusing to overwrite comparison freeze receipt: {receipt_path}"
        )
    local_paths = public_freeze_input_paths(freeze_root)
    with tempfile.TemporaryDirectory(prefix="arx-comparison-freeze-") as raw:
        staging = Path(raw)
        staged = []
        for name, source in sorted(local_paths.items()):
            destination = staging / name
            destination.write_bytes(source.read_bytes())
            staged.append(destination)
        result = subprocess.run(
            [
                "gh",
                "gist",
                "create",
                "--public",
                "--desc",
                "Frozen neural-versus-ARX comparison analysis",
                *(str(path) for path in staged),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    output = result.stdout.strip().splitlines()
    if not output:
        raise ValueError("gh gist create returned no public URL")
    gist_id = output[-1].rstrip("/").split("/")[-1]
    if not base.GIST_ID_PATTERN.fullmatch(gist_id):
        raise ValueError("gh gist create returned an invalid Gist ID")
    current = base._fetch(f"{base.GITHUB_API}/gists/{gist_id}")
    owner_payload = current.payload.get("owner")
    history = current.payload.get("history")
    if (
        current.payload.get("public") is not True
        or not isinstance(owner_payload, dict)
        or not isinstance(owner_payload.get("login"), str)
        or not isinstance(history, list)
        or not history
        or not isinstance(history[0], dict)
        or not isinstance(history[0].get("version"), str)
    ):
        raise ValueError("new comparison Gist lacks a pinnable public revision")
    receipt = build_public_freeze_receipt(
        gist_id,
        str(history[0]["version"]),
        str(owner_payload["login"]),
        freeze_root=freeze_root,
    )
    write_json_once(receipt_path, receipt)
    validated = validate_public_freeze_receipt(
        receipt_path, freeze_root, live=True
    )
    if any(validated.get(key) != value for key, value in receipt.items()):
        raise RuntimeError("comparison public-freeze receipt failed validation")
    return validated


def validate_public_freeze_receipt(
    receipt_path: Path,
    freeze_root: Path,
    *,
    live: bool = True,
) -> dict:
    receipt = strict_json(receipt_path)
    if set(receipt) != RECEIPT_FIELDS:
        raise ValueError("comparison public-freeze receipt fields changed")
    gist, revision, owner = base._identity(receipt)
    api_url = f"{base.GITHUB_API}/gists/{gist}/{revision}"
    html_url = f"https://gist.github.com/{owner}/{gist}/{revision}"
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("provider") != PROVIDER
        or receipt.get("public") is not True
        or receipt.get("prelock_registry_sha256")
        != _prelock_digest(freeze_root)
        or receipt.get("revision_api_url") != api_url
        or receipt.get("revision_html_url") != html_url
        or receipt.get("file_sha256_by_name")
        != expected_file_hashes(freeze_root)
    ):
        raise ValueError("comparison public-freeze receipt identity changed")
    committed = base._parse_utc(
        receipt.get("revision_committed_at_utc"), "comparison revision time"
    )
    created = base._parse_utc(
        receipt.get("provider_created_at_utc"), "comparison Gist created time"
    )
    updated = base._parse_utc(
        receipt.get("provider_updated_at_utc"), "comparison Gist updated time"
    )
    if not created <= committed <= updated:
        raise ValueError("comparison public-freeze chronology changed")
    if not live:
        return receipt
    response = base._fetch(api_url)
    base._validate_remote(
        response, receipt, public_freeze_input_paths(freeze_root)
    )
    return {
        **receipt,
        "provider_verified_at_utc": response.provider_verified_at_utc,
    }
