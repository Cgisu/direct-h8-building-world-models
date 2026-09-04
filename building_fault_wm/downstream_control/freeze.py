"""Freeze the downstream control protocol before response generation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import protocol


def write_exclusive(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze() -> str:
    payload = protocol.protocol_payload()
    write_exclusive(
        protocol.PROTOCOL_PATH,
        (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("ascii"),
    )
    digest = protocol.canonical_sha256(payload)
    write_exclusive(protocol.PROTOCOL_DIGEST_PATH, (digest + "\n").encode("ascii"))
    protocol.validate_frozen_protocol()
    print(protocol.PROTOCOL_PATH)
    print(digest)
    return digest


if __name__ == "__main__":
    freeze()
