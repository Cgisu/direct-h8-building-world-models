"""Repository-level isolation for tests of completed frozen studies."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_completed_v3_readiness_state(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the pre-collection readiness test without completed live state."""

    if request.node.name != (
        "test_readiness_is_value_blind_and_creates_no_collection_state"
    ):
        return
    monkeypatch.setattr(
        request.node.module.collect,
        "STATE_ROOT",
        tmp_path / "isolated-v3-state",
    )
