from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATE_ROOT = HERE / ".locked_study_state_v1"

EXTERNAL_FREEZE_RECEIPT = "external_freeze_receipt.json"
COLLECTION_ATTEMPT_MARKER = "locked_collection_attempt.json"
COLLECTION_FAILURE_MARKER = "locked_collection_failure.json"
COLLECTION_COMPLETION_MARKER = "locked_collection_completion.json"
CONFIRMATION_ATTEMPT_MARKER = "locked_confirmation_attempt.json"
CONFIRMATION_FAILURE_MARKER = "locked_confirmation_failure.json"
CONFIRMATION_COMPLETION_MARKER = "locked_confirmation_completion.json"


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload).rstrip(b"\n")).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_once(path: Path, content: bytes, *, identical_ok: bool = False) -> None:
    """Atomically create evidence without permitting a hash-changing overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if identical_ok and path.is_file() and not path.is_symlink():
            if path.read_bytes() == content:
                return
        raise FileExistsError(f"refusing to overwrite one-shot evidence: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        os.chmod(path, 0o444)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_once(path: Path, payload: object) -> None:
    write_bytes_once(path, canonical_bytes(payload))


def state_dir_for_digest(digest: str) -> Path:
    if not is_sha256(digest):
        raise ValueError("pre-lock digest must be a lowercase SHA-256")
    return STATE_ROOT / digest
