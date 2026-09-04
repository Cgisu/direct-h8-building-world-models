"""Create and verify the public, revision-pinned v5 operational freeze."""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from building_fault_wm.deterministic_transport import (
    plan as frozen_plan,
)

from . import runner


DEFAULT_PRELOCK_ROOT = runner.PRELOCK_ROOT
DEFAULT_READINESS = runner.READINESS_PATH
DEFAULT_RECEIPT = runner.EXTERNAL_FREEZE_RECEIPT

SCHEMA = "direct-h8-transport-collection-v5-github-gist-freeze-v1"
PROVIDER = "github-gist"
GITHUB_API = "https://api.github.com"
GIST_ID_PATTERN = re.compile(r"^[0-9a-f]+$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
OWNER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
FREEZE_FILENAMES = frozenset(
    {
        "PROTOCOL_ADDENDUM.md",
        "collection_readiness.json",
        "evaluation_adapter.py",
        "external_freeze.py",
        "prelock_registry.canonical.sha256",
        "runner.py",
        "v4_closeout.py",
        "v4_terminal_failure_closeout.json",
        "disjointness_certificate.json",
        "v3_paired_collection_attempt.json",
        "v3_paired_collection_failure.json",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "provider",
        "prelock_registry_sha256",
        "readiness_sha256",
        "readiness_file_sha256",
        "protocol_addendum_sha256",
        "runner_sha256",
        "external_freeze_sha256",
        "evaluation_adapter_sha256",
        "terminal_v4_failure_binding_sha256",
        "terminal_v4_closeout_file_sha256",
        "terminal_v4_closeout_payload_sha256",
        "terminal_v4_collection_log_sha256",
        "terminal_v4_raw_inventory_canonical_sha256",
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
class GitHubResponse:
    payload: dict
    provider_date_utc: str


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a plain file: {path}")
    return path


def _strict_json_bytes(content: bytes, label: str) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    value = json.loads(
        content.decode("ascii"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {label}: {token}")
        ),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} is not UTC")
    return parsed


def _validate_identity(
    gist_id: object, revision: object, owner_login: object
) -> tuple[str, str, str]:
    if not isinstance(gist_id, str) or not GIST_ID_PATTERN.fullmatch(gist_id):
        raise ValueError("GitHub Gist ID is invalid")
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        raise ValueError("GitHub Gist revision is invalid")
    if (
        not isinstance(owner_login, str)
        or not OWNER_PATTERN.fullmatch(owner_login)
    ):
        raise ValueError("GitHub owner login is invalid")
    return gist_id, revision, owner_login


def _urls(gist_id: str, revision: str, owner: str) -> tuple[str, str]:
    return (
        f"{GITHUB_API}/gists/{gist_id}/{revision}",
        f"https://gist.github.com/{owner}/{gist_id}/{revision}",
    )


def _fetch_json(url: str) -> GitHubResponse:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "wm-buildings-direct-h8-v5-freeze-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"GitHub returned HTTP {response.status}")
        content = response.read()
        date_header = response.headers.get("Date")
    if not date_header:
        raise ValueError("GitHub response has no trusted Date header")
    parsed_date = email.utils.parsedate_to_datetime(date_header)
    if parsed_date.tzinfo is None:
        raise ValueError("GitHub Date header has no timezone")
    return GitHubResponse(
        payload=_strict_json_bytes(content, "GitHub Gist response"),
        provider_date_utc=parsed_date.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )


def _prelock_digest(prelock_root: Path) -> str:
    record = _plain_file(
        prelock_root / "prelock_registry.canonical.sha256",
        "scientific prelock digest",
    ).read_text(encoding="ascii")
    if record != f"{runner.ORIGINAL_PRELOCK_SHA256}\n":
        raise ValueError("scientific prelock digest record changed")
    registry = _strict_json_bytes(
        _plain_file(
            prelock_root / "prelock_registry.json", "scientific prelock registry"
        ).read_bytes(),
        "scientific prelock registry",
    )
    if frozen_plan.canonical_sha256(registry) != runner.ORIGINAL_PRELOCK_SHA256:
        raise ValueError("scientific prelock registry changed")
    return runner.ORIGINAL_PRELOCK_SHA256


def _readiness_payload(
    path: Path,
    expected_prelock_sha256: str,
    *,
    prelock_root: Path,
) -> tuple[dict, str]:
    payload = _strict_json_bytes(
        _plain_file(path, "v5 collection readiness").read_bytes(),
        "v5 collection readiness",
    )
    digest = payload.get("readiness_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "readiness_sha256"
    }
    current_code = runner.operational_code_hashes()
    terminal = runner.terminal_v4_failure_binding()
    registry = runner._validate_scientific_prelock(prelock_root)
    shared_runtime = runner.validate_live_shared_runtime_semantics(
        prelock_root=prelock_root,
        registry=registry,
    )
    if (
        payload.get("schema") != runner.READINESS_SCHEMA
        or not _valid_sha256(digest)
        or frozen_plan.canonical_sha256(unsigned) != digest
        or payload.get("prelock_registry_sha256") != expected_prelock_sha256
        or payload.get("operational_code_sha256") != current_code
        or payload.get("protocol_sha256")
        != current_code["PROTOCOL_ADDENDUM.md"]
        or payload.get("terminal_v4_failure") != terminal
        or payload.get("live_shared_runtime_validation") != shared_runtime
        or payload.get("full_recollection_required") is not True
        or payload.get("data_v7_raw_reuse_permitted") is not False
        or payload.get("locked_response_values_accessed") is not False
        or payload.get("state_created") is not False
    ):
        raise ValueError("v5 collection readiness is not a valid outcome-blind lock")
    runner._validate_runtime_report(payload)
    return payload, str(digest)


def freeze_file_paths(
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
) -> dict[str, Path]:
    result = {
        "PROTOCOL_ADDENDUM.md": runner.HERE / "PROTOCOL_ADDENDUM.md",
        "collection_readiness.json": readiness_path,
        "evaluation_adapter.py": runner.HERE / "evaluation_adapter.py",
        "external_freeze.py": runner.HERE / "external_freeze.py",
        "prelock_registry.canonical.sha256": (
            prelock_root / "prelock_registry.canonical.sha256"
        ),
        "runner.py": runner.HERE / "runner.py",
        "v4_closeout.py": runner.HERE / "v4_closeout.py",
        "v4_terminal_failure_closeout.json": (
            runner.HERE / "v4_terminal_failure_closeout.json"
        ),
        "disjointness_certificate.json": (
            prelock_root / "bundle/plans/disjointness_certificate.json"
        ),
        "v3_paired_collection_attempt.json": runner.V4_ATTEMPT_PATH,
        "v3_paired_collection_failure.json": runner.V4_FAILURE_PATH,
    }
    if set(result) != FREEZE_FILENAMES:
        raise AssertionError("v5 external-freeze file set is incomplete")
    for name, path in result.items():
        _plain_file(path, f"v5 freeze source {name}")
    return result


def read_freeze_files(
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
) -> dict[str, bytes]:
    return {
        name: path.read_bytes()
        for name, path in sorted(
            freeze_file_paths(prelock_root, readiness_path).items()
        )
    }


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
        raise ValueError("GitHub Gist response does not contain the pinned revision")
    return matches[0]


def _validate_remote(
    response: GitHubResponse,
    receipt: Mapping[str, object],
    local_files: Mapping[str, bytes],
) -> None:
    gist_id, revision, owner = _validate_identity(
        receipt.get("gist_id"),
        receipt.get("revision"),
        receipt.get("owner_login"),
    )
    api_url, html_url = _urls(gist_id, revision, owner)
    payload = response.payload
    if payload.get("id") != gist_id or payload.get("public") is not True:
        raise ValueError("GitHub response is not the required public Gist")
    remote_owner = payload.get("owner")
    if not isinstance(remote_owner, dict) or remote_owner.get("login") != owner:
        raise ValueError("GitHub Gist owner differs from the freeze receipt")
    if payload.get("html_url") != f"https://gist.github.com/{owner}/{gist_id}":
        raise ValueError("GitHub Gist HTML URL is noncanonical")
    if receipt.get("revision_api_url") != api_url:
        raise ValueError("v5 freeze receipt API URL is noncanonical")
    if receipt.get("revision_html_url") != html_url:
        raise ValueError("v5 freeze receipt HTML URL is noncanonical")

    created = payload.get("created_at")
    updated = payload.get("updated_at")
    created_at = _parse_utc(created, "GitHub Gist created_at")
    updated_at = _parse_utc(updated, "GitHub Gist updated_at")
    provider_at = _parse_utc(response.provider_date_utc, "GitHub HTTP Date")
    if not created_at <= updated_at <= provider_at:
        raise ValueError("GitHub Gist chronology is invalid")
    if receipt.get("provider_created_at_utc") != created:
        raise ValueError("v5 freeze receipt created_at differs from GitHub")
    if receipt.get("provider_updated_at_utc") != updated:
        raise ValueError("v5 freeze receipt updated_at differs from GitHub")

    history = _history_entry(payload, revision)
    committed = history.get("committed_at")
    committed_at = _parse_utc(committed, "GitHub Gist revision committed_at")
    if not created_at <= committed_at <= updated_at:
        raise ValueError("GitHub Gist revision chronology is invalid")
    if receipt.get("revision_committed_at_utc") != committed:
        raise ValueError("v5 freeze receipt revision time differs from GitHub")
    if history.get("url") != api_url:
        raise ValueError("GitHub Gist history URL does not pin the revision")
    history_user = history.get("user")
    if not isinstance(history_user, dict) or history_user.get("login") != owner:
        raise ValueError("GitHub Gist revision author differs from its owner")

    remote_files = payload.get("files")
    if not isinstance(remote_files, dict) or set(remote_files) != FREEZE_FILENAMES:
        raise ValueError("GitHub Gist file set differs from the v5 freeze contract")
    if set(local_files) != FREEZE_FILENAMES:
        raise ValueError("local v5 freeze file set is incomplete")
    expected_hashes: dict[str, str] = {}
    for name, expected in sorted(local_files.items()):
        metadata = remote_files.get(name)
        if not isinstance(metadata, dict) or metadata.get("filename") != name:
            raise ValueError(f"GitHub Gist metadata is invalid for {name}")
        if metadata.get("truncated") is not False:
            raise ValueError(f"GitHub Gist file is truncated: {name}")
        content = metadata.get("content")
        if not isinstance(content, str) or content.encode("utf-8") != expected:
            raise ValueError(f"GitHub Gist file differs byte-for-byte: {name}")
        if metadata.get("size") != len(expected):
            raise ValueError(f"GitHub Gist file size differs: {name}")
        expected_hashes[name] = _sha256_bytes(expected)
    if receipt.get("file_sha256_by_name") != expected_hashes:
        raise ValueError("v5 freeze receipt file hashes differ")


def _validate_receipt_shape(
    receipt: object,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    *,
    readiness: Mapping[str, object],
    files: Mapping[str, bytes],
) -> dict:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError("v5 external-freeze receipt fields differ")
    code = readiness.get("operational_code_sha256")
    terminal = readiness.get("terminal_v4_failure")
    if not isinstance(code, dict) or not isinstance(terminal, dict):
        raise ValueError("v5 readiness code/failure binding is missing")
    expected_explicit = {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "prelock_registry_sha256": expected_prelock_sha256,
        "readiness_sha256": expected_readiness_sha256,
        "readiness_file_sha256": _sha256_bytes(
            files["collection_readiness.json"]
        ),
        "protocol_addendum_sha256": code["PROTOCOL_ADDENDUM.md"],
        "runner_sha256": code["runner.py"],
        "external_freeze_sha256": code["external_freeze.py"],
        "evaluation_adapter_sha256": code["evaluation_adapter.py"],
        "terminal_v4_failure_binding_sha256": terminal["binding_sha256"],
        "terminal_v4_closeout_file_sha256": terminal["binding"][
            "terminal_closeout_file_sha256"
        ],
        "terminal_v4_closeout_payload_sha256": terminal["binding"][
            "terminal_closeout_payload_sha256"
        ],
        "terminal_v4_collection_log_sha256": terminal["binding"][
            "collection_log_sha256"
        ],
        "terminal_v4_raw_inventory_canonical_sha256": terminal["binding"][
            "incomplete_raw_inventory_canonical_sha256"
        ],
    }
    if any(receipt.get(key) != value for key, value in expected_explicit.items()):
        raise ValueError("v5 external-freeze receipt identity differs")
    _validate_identity(
        receipt.get("gist_id"),
        receipt.get("revision"),
        receipt.get("owner_login"),
    )
    for field in (
        "provider_created_at_utc",
        "provider_updated_at_utc",
        "revision_committed_at_utc",
    ):
        _parse_utc(receipt.get(field), field)
    hashes = receipt.get("file_sha256_by_name")
    expected_hashes = {
        name: _sha256_bytes(content) for name, content in sorted(files.items())
    }
    if hashes != expected_hashes:
        raise ValueError("v5 external-freeze receipt hashes are invalid")
    return receipt


def build_external_freeze_receipt(
    gist_id: str,
    revision: str,
    owner_login: str,
    *,
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
) -> dict:
    gist_id, revision, owner_login = _validate_identity(
        gist_id, revision, owner_login
    )
    prelock_digest = _prelock_digest(prelock_root)
    readiness, readiness_digest = _readiness_payload(
        readiness_path,
        prelock_digest,
        prelock_root=prelock_root,
    )
    files = read_freeze_files(prelock_root, readiness_path)
    api_url, html_url = _urls(gist_id, revision, owner_login)
    response = _fetch_json(api_url)
    history = _history_entry(response.payload, revision)
    code = readiness["operational_code_sha256"]
    terminal = readiness["terminal_v4_failure"]
    receipt = {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "prelock_registry_sha256": prelock_digest,
        "readiness_sha256": readiness_digest,
        "readiness_file_sha256": _sha256_bytes(
            files["collection_readiness.json"]
        ),
        "protocol_addendum_sha256": code["PROTOCOL_ADDENDUM.md"],
        "runner_sha256": code["runner.py"],
        "external_freeze_sha256": code["external_freeze.py"],
        "evaluation_adapter_sha256": code["evaluation_adapter.py"],
        "terminal_v4_failure_binding_sha256": terminal["binding_sha256"],
        "terminal_v4_closeout_file_sha256": terminal["binding"][
            "terminal_closeout_file_sha256"
        ],
        "terminal_v4_closeout_payload_sha256": terminal["binding"][
            "terminal_closeout_payload_sha256"
        ],
        "terminal_v4_collection_log_sha256": terminal["binding"][
            "collection_log_sha256"
        ],
        "terminal_v4_raw_inventory_canonical_sha256": terminal["binding"][
            "incomplete_raw_inventory_canonical_sha256"
        ],
        "gist_id": gist_id,
        "revision": revision,
        "owner_login": owner_login,
        "provider_created_at_utc": response.payload.get("created_at"),
        "provider_updated_at_utc": response.payload.get("updated_at"),
        "revision_committed_at_utc": history.get("committed_at"),
        "revision_api_url": api_url,
        "revision_html_url": html_url,
        "file_sha256_by_name": {
            name: _sha256_bytes(content) for name, content in sorted(files.items())
        },
    }
    _validate_receipt_shape(
        receipt,
        prelock_digest,
        readiness_digest,
        readiness=readiness,
        files=files,
    )
    _validate_remote(response, receipt, files)
    return receipt


def _write_once(path: Path, payload: object) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite v5 freeze evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    ).encode("ascii")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def validate_external_freeze_receipt(
    receipt_path: Path,
    expected_prelock_sha256: str,
    expected_readiness_sha256: str,
    *,
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
    live: bool = True,
) -> dict:
    if _prelock_digest(prelock_root) != expected_prelock_sha256:
        raise ValueError("live scientific prelock differs from the freeze receipt")
    readiness, readiness_digest = _readiness_payload(
        readiness_path,
        expected_prelock_sha256,
        prelock_root=prelock_root,
    )
    if readiness_digest != expected_readiness_sha256:
        raise ValueError("live v5 readiness differs from the expected digest")
    files = read_freeze_files(prelock_root, readiness_path)
    receipt = _validate_receipt_shape(
        _strict_json_bytes(
            _plain_file(receipt_path, "v5 external-freeze receipt").read_bytes(),
            "v5 external-freeze receipt",
        ),
        expected_prelock_sha256,
        expected_readiness_sha256,
        readiness=readiness,
        files=files,
    )
    if live:
        response = _fetch_json(str(receipt["revision_api_url"]))
        _validate_remote(response, receipt, files)
        receipt = {
            **receipt,
            "provider_verified_at_utc": response.provider_date_utc,
        }
    return receipt


def create_public_gist(
    *,
    prelock_root: Path = DEFAULT_PRELOCK_ROOT,
    readiness_path: Path = DEFAULT_READINESS,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict:
    files = freeze_file_paths(prelock_root, readiness_path)
    command = [
        "gh",
        "gist",
        "create",
        "--public",
        "--desc",
        "Direct-H8 ownership-corrected outcome-blind collection freeze v5",
        *(str(files[name]) for name in sorted(files)),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    url = result.stdout.strip().splitlines()[-1]
    gist_id = url.rstrip("/").split("/")[-1]
    current = _fetch_json(f"{GITHUB_API}/gists/{gist_id}")
    owner = current.payload.get("owner")
    history = current.payload.get("history")
    if (
        not isinstance(owner, dict)
        or not isinstance(owner.get("login"), str)
        or not isinstance(history, list)
        or not history
        or not isinstance(history[0], dict)
    ):
        raise ValueError("new GitHub Gist response has incomplete identity")
    receipt = build_external_freeze_receipt(
        gist_id,
        str(history[0].get("version")),
        str(owner["login"]),
        prelock_root=prelock_root,
        readiness_path=readiness_path,
    )
    _write_once(receipt_path, receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--prelock-root", type=Path, default=DEFAULT_PRELOCK_ROOT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--expected-prelock-sha256")
    parser.add_argument("--expected-readiness-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "create":
        receipt = create_public_gist(
            prelock_root=args.prelock_root.resolve(),
            readiness_path=args.readiness.resolve(),
            receipt_path=args.receipt.resolve(),
        )
    else:
        if not args.expected_prelock_sha256 or not args.expected_readiness_sha256:
            raise ValueError("verify requires both expected digests")
        receipt = validate_external_freeze_receipt(
            args.receipt.resolve(),
            args.expected_prelock_sha256,
            args.expected_readiness_sha256,
            prelock_root=args.prelock_root.resolve(),
            readiness_path=args.readiness.resolve(),
            live=True,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
