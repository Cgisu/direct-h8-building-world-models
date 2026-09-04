"""Skip the tests that the compact public release cannot run.

The release omits the sealed evidence packages and renames the study module
directories, so a subset of the provenance and closeout tests cannot pass here.
They are skipped with an explicit reason rather than deleted; see
``scripts/non_portable_tests.txt`` for the list and the reasoning, and the
README section "Running the full test suite".

Neither documented verification command depends on this file: both run only the
portable set in ``scripts/portable_tests.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent
LISTING = ROOT / "scripts/non_portable_tests.txt"
REASON = (
    "not runnable from the compact public release: needs the separately "
    "deposited evidence packages or the pre-release source layout; see "
    "scripts/non_portable_tests.txt"
)


def _listing() -> tuple[list[str], set[str]]:
    ignored: list[str] = []
    skipped: set[str] = set()
    if not LISTING.is_file():
        return ignored, skipped
    for line in LISTING.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("COLLECT_IGNORE "):
            ignored.append(line.split(" ", 1)[1].strip())
        else:
            skipped.add(line)
    return ignored, skipped


COLLECT_IGNORE, NON_PORTABLE = _listing()
collect_ignore = COLLECT_IGNORE


def pytest_collection_modifyitems(items) -> None:
    if not NON_PORTABLE:
        return
    marker = pytest.mark.skip(reason=REASON)
    for item in items:
        if item.nodeid in NON_PORTABLE:
            item.add_marker(marker)
