from __future__ import annotations

import pytest

from .cli import build_parser


def test_cli_exposes_all_explicit_phases() -> None:
    parser = build_parser()
    for command in (
        "train",
        "prepare-lock",
        "verify-lock",
        "freeze-files",
        "create-public-freeze",
        "verify-freeze",
        "evaluate",
    ):
        if command in {"create-public-freeze", "verify-freeze", "evaluate"}:
            with pytest.raises(SystemExit):
                parser.parse_args([command])
        else:
            assert parser.parse_args([command]).command == command
