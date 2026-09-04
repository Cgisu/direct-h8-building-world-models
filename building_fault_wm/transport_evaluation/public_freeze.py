"""Revision-pinned public freeze for the v6 recovery prelock."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from building_fault_wm.ridge_arx import (
    external_freeze as base,
)
from building_fault_wm.ridge_arx.io import (
    sha256_file,
    strict_json,
    write_json_once,
)

from . import adapter, freeze


SCHEMA = "direct-h8-transport-evaluation-v6-recovery-github-gist-freeze-v3"
PROVIDER = "github-gist"
RECEIPT_FIELDS = base.RECEIPT_FIELDS


def _prelock_digest(prelock_root: Path) -> str:
    freeze.verify_local_prelock(prelock_root)
    record = (prelock_root / freeze.DIGEST_NAME).read_text(encoding="ascii")
    if (
        len(record) != 65
        or not record.endswith("\n")
        or any(character not in "0123456789abcdef" for character in record[:-1])
    ):
        raise ValueError("v6 recovery prelock digest is malformed")
    return record[:-1]


def expected_file_hashes(prelock_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in sorted(
            freeze.public_freeze_input_paths(prelock_root).items()
        )
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
    local_paths = freeze.public_freeze_input_paths(prelock_root)
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
        raise AssertionError("v6 public-freeze receipt fields changed")
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
        raise ValueError("v6 public-freeze receipt fields changed")
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
        raise ValueError("v6 public-freeze receipt identity changed")
    committed = base._parse_utc(
        receipt.get("revision_committed_at_utc"), "v6 recovery revision time"
    )
    created = base._parse_utc(
        receipt.get("provider_created_at_utc"), "v6 recovery Gist created time"
    )
    updated = base._parse_utc(
        receipt.get("provider_updated_at_utc"), "v6 recovery Gist updated time"
    )
    if not created <= committed <= updated:
        raise ValueError("v6 public-freeze chronology changed")
    if not live:
        return receipt
    response = base._fetch(api_url)
    base._validate_remote(
        response, receipt, freeze.public_freeze_input_paths(prelock_root)
    )
    return {
        **receipt,
        "provider_verified_at_utc": response.provider_verified_at_utc,
    }


def create_public_freeze(
    *,
    prelock_root: Path,
    receipt_path: Path = adapter.DEFAULT_RECOVERY_FREEZE_RECEIPT,
) -> dict:
    if os.path.lexists(receipt_path):
        raise FileExistsError(
            f"refusing to overwrite v6 public-freeze receipt: {receipt_path}"
        )
    local_paths = freeze.public_freeze_input_paths(prelock_root)
    with tempfile.TemporaryDirectory(prefix="direct-h8-eval-v6-freeze-") as raw:
        staging = Path(raw)
        staged: list[Path] = []
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
                "Direct-H8 evaluation v6 metadata-dispatch recovery freeze v3",
                *(str(path) for path in staged),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise ValueError("gh gist create returned no public URL")
    gist_id = lines[-1].rstrip("/").split("/")[-1]
    if not base.GIST_ID_PATTERN.fullmatch(gist_id):
        raise ValueError("gh gist create returned an invalid Gist ID")
    current = base._fetch(f"{base.GITHUB_API}/gists/{gist_id}")
    owner = current.payload.get("owner")
    history = current.payload.get("history")
    if (
        current.payload.get("public") is not True
        or not isinstance(owner, dict)
        or not isinstance(owner.get("login"), str)
        or not isinstance(history, list)
        or not history
        or not isinstance(history[0], dict)
        or not isinstance(history[0].get("version"), str)
    ):
        raise ValueError("new v6 Gist response lacks a pinnable revision")
    receipt = build_public_freeze_receipt(
        gist_id,
        str(history[0]["version"]),
        str(owner["login"]),
        prelock_root=prelock_root,
    )
    write_json_once(receipt_path, receipt)
    return validate_public_freeze_receipt(
        receipt_path, prelock_root, live=True
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "validate"))
    parser.add_argument(
        "--prelock-root", type=Path, default=adapter.DEFAULT_RECOVERY_PRELOCK
    )
    parser.add_argument(
        "--receipt", type=Path, default=adapter.DEFAULT_RECOVERY_FREEZE_RECEIPT
    )
    parser.add_argument("--no-live", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "create":
        result = create_public_freeze(
            prelock_root=args.prelock_root.resolve(),
            receipt_path=args.receipt.resolve(),
        )
    else:
        result = validate_public_freeze_receipt(
            args.receipt.resolve(),
            args.prelock_root.resolve(),
            live=not args.no_live,
        )
    print(result)


if __name__ == "__main__":
    main()
