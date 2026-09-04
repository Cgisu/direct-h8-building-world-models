"""Freeze the multi-case downstream protocol after development-only tuning."""

from __future__ import annotations

import os

from . import protocol


def main() -> None:
    if os.path.lexists(protocol.PROTOCOL_ROOT):
        raise FileExistsError(
            f"refusing to overwrite frozen protocol: {protocol.PROTOCOL_ROOT}"
        )
    payload = protocol.protocol_payload()
    protocol.PROTOCOL_ROOT.mkdir(parents=True)
    protocol.PROTOCOL_PATH.write_bytes(protocol.canonical_bytes(payload) + b"\n")
    protocol.PROTOCOL_DIGEST_PATH.write_text(
        protocol.canonical_sha256(payload) + "\n", encoding="ascii"
    )
    protocol.validate_frozen_protocol()
    print(protocol.PROTOCOL_PATH)
    print(protocol.PROTOCOL_DIGEST_PATH)


if __name__ == "__main__":
    main()
