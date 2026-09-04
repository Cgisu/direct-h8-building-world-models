"""Revision-pinned public receipt for the identity-guard prelock."""

from __future__ import annotations

from pathlib import Path

from building_fault_wm.ridge_arx import (
    external_freeze as base,
)
from building_fault_wm.ridge_arx.io import (
    sha256_file,
    strict_json,
)

from .freeze import DIGEST_NAME, public_freeze_input_paths


SCHEMA = "schedule-matched-arx-neural-identity-guard-github-gist-freeze-v1"
PROVIDER = "github-gist"
RECEIPT_FIELDS = base.RECEIPT_FIELDS


def _prelock_digest(prelock_root: Path) -> str:
    value = (prelock_root / DIGEST_NAME).read_text(encoding="ascii")
    if (
        len(value) != 65
        or value[-1] != "\n"
        or any(character not in "0123456789abcdef" for character in value[:-1])
    ):
        raise ValueError("identity-guard prelock digest is malformed")
    return value[:-1]


def expected_file_hashes(prelock_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in sorted(public_freeze_input_paths(prelock_root).items())
    }


def build_public_freeze_receipt(
    gist_id: str,
    revision: str,
    owner_login: str,
    *,
    prelock_root: Path,
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
    local_paths = public_freeze_input_paths(prelock_root)
    receipt = {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "public": True,
        "prelock_registry_sha256": _prelock_digest(prelock_root),
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
        raise AssertionError("identity-guard public receipt fields changed")
    for field in (
        "provider_created_at_utc",
        "provider_updated_at_utc",
        "revision_committed_at_utc",
    ):
        base._parse_utc(receipt[field], field)
    base._validate_remote(response, receipt, local_paths)
    return receipt


def validate_public_freeze_receipt(
    receipt_path: Path,
    prelock_root: Path,
    *,
    live: bool = True,
) -> dict:
    receipt = strict_json(receipt_path)
    if set(receipt) != RECEIPT_FIELDS:
        raise ValueError("identity-guard public receipt fields changed")
    gist, revision, owner = base._identity(receipt)
    api_url = f"{base.GITHUB_API}/gists/{gist}/{revision}"
    html_url = f"https://gist.github.com/{owner}/{gist}/{revision}"
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("provider") != PROVIDER
        or receipt.get("public") is not True
        or receipt.get("prelock_registry_sha256")
        != _prelock_digest(prelock_root)
        or receipt.get("revision_api_url") != api_url
        or receipt.get("revision_html_url") != html_url
        or receipt.get("file_sha256_by_name")
        != expected_file_hashes(prelock_root)
    ):
        raise ValueError("identity-guard public receipt identity changed")
    committed = base._parse_utc(
        receipt.get("revision_committed_at_utc"),
        "identity-guard revision time",
    )
    created = base._parse_utc(
        receipt.get("provider_created_at_utc"),
        "identity-guard Gist created time",
    )
    updated = base._parse_utc(
        receipt.get("provider_updated_at_utc"),
        "identity-guard Gist updated time",
    )
    if not created <= committed <= updated:
        raise ValueError("identity-guard public receipt chronology changed")
    if not live:
        return receipt
    response = base._fetch(api_url)
    base._validate_remote(
        response, receipt, public_freeze_input_paths(prelock_root)
    )
    return {
        **receipt,
        "provider_verified_at_utc": response.provider_verified_at_utc,
    }
