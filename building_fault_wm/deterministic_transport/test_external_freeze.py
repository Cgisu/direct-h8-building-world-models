from __future__ import annotations

from datetime import datetime, timezone

import pytest

from .external_freeze import (
    FREEZE_FILENAMES,
    PROVIDER,
    RECEIPT_FIELDS,
    SCHEMA,
    _parse_utc,
    _validate_identity,
    _validate_receipt_shape,
)


def test_external_freeze_identity_is_strict() -> None:
    assert _validate_identity("abc123", "1" * 40, "research-user") == (
        "abc123",
        "1" * 40,
        "research-user",
    )
    with pytest.raises(ValueError, match="revision"):
        _validate_identity("abc123", "short", "research-user")


def test_external_freeze_receipt_shape_binds_both_digests() -> None:
    prelock = "1" * 64
    readiness = "2" * 64
    receipt = {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "prelock_registry_sha256": prelock,
        "readiness_sha256": readiness,
        "gist_id": "abc123",
        "revision": "3" * 40,
        "owner_login": "research-user",
        "provider_created_at_utc": "2026-07-23T00:00:00Z",
        "provider_updated_at_utc": "2026-07-23T00:00:01Z",
        "revision_committed_at_utc": "2026-07-23T00:00:01Z",
        "revision_api_url": "https://api.github.com/gists/abc123/" + "3" * 40,
        "revision_html_url": (
            "https://gist.github.com/research-user/abc123/" + "3" * 40
        ),
        "file_sha256_by_name": {
            name: "4" * 64 for name in FREEZE_FILENAMES
        },
    }
    assert set(receipt) == RECEIPT_FIELDS
    assert _validate_receipt_shape(receipt, prelock, readiness) == receipt
    with pytest.raises(ValueError, match="identity"):
        _validate_receipt_shape(receipt, prelock, "5" * 64)


def test_utc_parser_rejects_naive_time() -> None:
    parsed = _parse_utc("2026-07-23T00:00:00Z", "test")
    assert parsed == datetime(2026, 7, 23, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="UTC"):
        _parse_utc("2026-07-23T00:00:00", "test")
