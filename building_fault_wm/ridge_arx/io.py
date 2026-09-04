"""Strict serialization and immutable-file helpers for the addendum."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


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
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"hash input is not a plain file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input is not a plain file: {path}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token in {path}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def write_once(path: Path, content: bytes, *, mode: int = 0o444) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite addendum artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json_once(path: Path, payload: object) -> Path:
    return write_once(
        path,
        (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("ascii"),
    )


def write_csv_once(path: Path, frame) -> Path:
    return write_once(path, frame.to_csv(index=False).encode("ascii"))


def tree_inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"inventory root is not a plain directory: {root}")
    excluded = set() if exclude is None else set(exclude)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"inventory contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows

