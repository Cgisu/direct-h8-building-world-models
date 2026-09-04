"""Validate a revision-pinned public GitHub Gist freeze for the addendum."""

from __future__ import annotations

import email.utils
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .io import sha256_file, strict_json, write_json_once
from .lock import DIGEST_NAME, external_freeze_file_paths


SCHEMA = "schedule-matched-recursive-ridge-arx-github-gist-freeze-v1"
PROVIDER = "github-gist"
GITHUB_API = "https://api.github.com"
GIST_ID_PATTERN = re.compile(r"^[0-9a-f]+$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
OWNER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "provider",
        "public",
        "prelock_registry_sha256",
        "gist_id",
        "revision",
        "owner_login",
        "provider_created_at_utc",
        "provider_updated_at_utc",
        "revision_committed_at_utc",
        "revision_api_url",
        "revision_html_url",
        "file_sha256_by_name",
    }
)


@dataclass(frozen=True)
class ProviderResponse:
    payload: dict
    provider_verified_at_utc: str


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} is not UTC")
    return parsed


def _identity(receipt: Mapping[str, object]) -> tuple[str, str, str]:
    gist = receipt.get("gist_id")
    revision = receipt.get("revision")
    owner = receipt.get("owner_login")
    if not isinstance(gist, str) or not GIST_ID_PATTERN.fullmatch(gist):
        raise ValueError("addendum Gist ID is invalid")
    if (
        not isinstance(revision, str)
        or not REVISION_PATTERN.fullmatch(revision)
    ):
        raise ValueError("addendum Gist revision is invalid")
    if not isinstance(owner, str) or not OWNER_PATTERN.fullmatch(owner):
        raise ValueError("addendum Gist owner is invalid")
    return gist, revision, owner


def _prelock_digest(prelock_root: Path) -> str:
    record = (prelock_root / DIGEST_NAME).read_text(encoding="ascii")
    if (
        len(record) != 65
        or not record.endswith("\n")
        or not re.fullmatch(r"[0-9a-f]{64}", record[:-1])
    ):
        raise ValueError("addendum pre-lock digest record is malformed")
    return record[:-1]


def expected_file_hashes(prelock_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in sorted(
            external_freeze_file_paths(prelock_root).items()
        )
    }


def _fetch(revision_api_url: str) -> ProviderResponse:
    request = urllib.request.Request(
        revision_api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "wm-buildings-arx-addendum-freeze-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"GitHub returned HTTP {response.status}")
        body = response.read()
        date_header = response.headers.get("Date")
    if not date_header:
        raise ValueError("GitHub response has no trusted Date header")
    parsed_date = email.utils.parsedate_to_datetime(date_header)
    if parsed_date.tzinfo is None:
        raise ValueError("GitHub Date header has no timezone")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GitHub Gist response is not an object")
    return ProviderResponse(
        payload=payload,
        provider_verified_at_utc=parsed_date.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )


def _history_entry(payload: Mapping[str, object], revision: str) -> dict:
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("GitHub Gist response has no revision history")
    matches = [
        item
        for item in history
        if isinstance(item, dict) and item.get("version") == revision
    ]
    if len(matches) != 1:
        raise ValueError("GitHub Gist response lacks the unique pinned revision")
    return matches[0]


def _validate_remote(
    response: ProviderResponse,
    receipt: Mapping[str, object],
    local_paths: Mapping[str, Path],
) -> None:
    gist, revision, owner = _identity(receipt)
    api_url = f"{GITHUB_API}/gists/{gist}/{revision}"
    payload = response.payload
    if payload.get("id") != gist or payload.get("public") is not True:
        raise ValueError("GitHub response is not the required public Gist")
    owner_payload = payload.get("owner")
    if not isinstance(owner_payload, dict) or owner_payload.get("login") != owner:
        raise ValueError("addendum Gist owner differs")
    if payload.get("html_url") != f"https://gist.github.com/{owner}/{gist}":
        raise ValueError("addendum Gist HTML URL is noncanonical")
    created = payload.get("created_at")
    updated = payload.get("updated_at")
    created_at = _parse_utc(created, "GitHub Gist created time")
    updated_at = _parse_utc(updated, "GitHub Gist updated time")
    verified_at = _parse_utc(
        response.provider_verified_at_utc, "GitHub HTTP Date"
    )
    if (
        not created_at <= updated_at <= verified_at
        or receipt.get("provider_created_at_utc") != created
        or receipt.get("provider_updated_at_utc") != updated
    ):
        raise ValueError("addendum Gist chronology differs")
    history = _history_entry(payload, revision)
    committed = _parse_utc(
        history.get("committed_at"), "remote revision time"
    )
    if (
        not created_at <= committed <= updated_at
        or receipt.get("revision_committed_at_utc")
        != history.get("committed_at")
        or history.get("url") != api_url
    ):
        raise ValueError("addendum Gist revision differs")
    history_user = history.get("user")
    if not isinstance(history_user, dict) or history_user.get("login") != owner:
        raise ValueError("addendum Gist revision author differs")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(local_paths):
        raise ValueError("addendum Gist file set differs")
    expected_hashes: dict[str, str] = {}
    for name, path in sorted(local_paths.items()):
        metadata = files.get(name)
        content = path.read_bytes()
        if (
            not isinstance(metadata, dict)
            or metadata.get("filename") != name
            or metadata.get("truncated") is not False
            or metadata.get("size") != len(content)
            or metadata.get("content", "").encode("utf-8") != content
        ):
            raise ValueError(f"addendum Gist file differs: {name}")
        expected_hashes[name] = sha256_file(path)
    if receipt.get("file_sha256_by_name") != expected_hashes:
        raise ValueError("addendum Gist receipt hashes differ")


def build_external_freeze_receipt(
    gist_id: str,
    revision: str,
    owner_login: str,
    *,
    prelock_root: Path,
) -> dict:
    identity = {
        "gist_id": gist_id,
        "revision": revision,
        "owner_login": owner_login,
    }
    gist, revision, owner = _identity(identity)
    api_url = f"{GITHUB_API}/gists/{gist}/{revision}"
    html_url = f"https://gist.github.com/{owner}/{gist}/{revision}"
    response = _fetch(api_url)
    history = _history_entry(response.payload, revision)
    local_paths = external_freeze_file_paths(prelock_root)
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
        raise AssertionError("addendum freeze receipt field set is incomplete")
    for field in (
        "provider_created_at_utc",
        "provider_updated_at_utc",
        "revision_committed_at_utc",
    ):
        _parse_utc(receipt[field], field)
    _validate_remote(response, receipt, local_paths)
    return receipt


def create_public_freeze(
    *,
    prelock_root: Path,
    receipt_path: Path,
) -> dict:
    """Create, pin, write once, and live-verify the public Gist freeze."""

    if os.path.lexists(receipt_path):
        raise FileExistsError(
            f"refusing to overwrite addendum freeze receipt: {receipt_path}"
        )
    local_paths = external_freeze_file_paths(prelock_root)
    with tempfile.TemporaryDirectory(prefix="arx-addendum-freeze-") as raw:
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
                "Schedule-matched recursive Ridge-ARX transport addendum freeze",
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
    if not GIST_ID_PATTERN.fullmatch(gist_id):
        raise ValueError("gh gist create returned an invalid Gist ID")
    current = _fetch(f"{GITHUB_API}/gists/{gist_id}")
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
        raise ValueError("new Gist response lacks a pinnable public revision")
    receipt = build_external_freeze_receipt(
        gist_id,
        str(history[0]["version"]),
        str(owner_payload["login"]),
        prelock_root=prelock_root,
    )
    write_json_once(receipt_path, receipt)
    validated = validate_external_freeze_receipt(
        receipt_path, prelock_root, live=True
    )
    if any(validated.get(key) != value for key, value in receipt.items()):
        raise RuntimeError("written addendum freeze receipt failed live validation")
    return validated


def validate_external_freeze_receipt(
    receipt_path: Path,
    prelock_root: Path,
    *,
    live: bool = True,
) -> dict:
    receipt = strict_json(receipt_path)
    if set(receipt) != RECEIPT_FIELDS:
        raise ValueError("addendum external-freeze receipt fields differ")
    gist, revision, owner = _identity(receipt)
    api_url = f"{GITHUB_API}/gists/{gist}/{revision}"
    html_url = f"https://gist.github.com/{owner}/{gist}/{revision}"
    expected_hashes = expected_file_hashes(prelock_root)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("provider") != PROVIDER
        or receipt.get("public") is not True
        or receipt.get("prelock_registry_sha256")
        != _prelock_digest(prelock_root)
        or receipt.get("revision_api_url") != api_url
        or receipt.get("revision_html_url") != html_url
        or receipt.get("file_sha256_by_name") != expected_hashes
    ):
        raise ValueError("addendum external-freeze receipt identity changed")
    committed = _parse_utc(
        receipt.get("revision_committed_at_utc"),
        "addendum Gist revision time",
    )
    created = _parse_utc(
        receipt.get("provider_created_at_utc"),
        "addendum Gist created time",
    )
    updated = _parse_utc(
        receipt.get("provider_updated_at_utc"),
        "addendum Gist updated time",
    )
    if not created <= committed <= updated:
        raise ValueError("addendum external-freeze chronology changed")
    if not live:
        return receipt

    response = _fetch(api_url)
    local_paths = external_freeze_file_paths(prelock_root)
    _validate_remote(response, receipt, local_paths)
    return {
        **receipt,
        "provider_verified_at_utc": response.provider_verified_at_utc,
    }
