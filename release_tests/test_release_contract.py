"""Portable release contract, exercised through the standard test entry point.

Each stage of ``scripts/verify_all.sh`` is reported as its own test so that a
reviewer running ``pytest`` sees which part of the contract passed. The unit
test targets are read from ``scripts/portable_tests.txt``, the same list the
shell entry point uses, so the two cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_TESTS = ROOT / "scripts/portable_tests.txt"


def _environment() -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON": sys.executable,
    }
    suffix = os.getuid() if hasattr(os, "getuid") else os.getpid()
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"direct-h8-mpl-{suffix}"
    cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(cache)
    return environment


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.fail(
            f"command failed: {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}",
            pytrace=False,
        )


def unit_test_targets() -> list[str]:
    lines = PORTABLE_TESTS.read_text(encoding="ascii").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/check_dependencies.py",
        "scripts/verify_repository.py",
        "scripts/verify_downstream_subset.py",
        "scripts/verify_downstream_contract.py",
    ],
)
def test_verification_script(script: str) -> None:
    _run([sys.executable, script])


@pytest.mark.parametrize("target", unit_test_targets())
def test_portable_unit_target(target: str) -> None:
    _run([sys.executable, "-m", "unittest", target])


def test_shell_entry_point_matches_the_python_entry_point() -> None:
    _run(["bash", "scripts/verify_all.sh"])


def _plant(names: tuple[str, ...]) -> list[Path]:
    created = []
    for name in names:
        target = ROOT / name
        if target.exists():
            continue
        (target / "nested").mkdir(parents=True)
        (target / "nested" / "untracked.bin").write_bytes(bytes(16))
        created.append(target)
    return created


def test_repository_check_ignores_a_virtualenv_under_any_name() -> None:
    """A venv is recognised by its pyvenv.cfg, not by being called .venv."""

    target = ROOT / "venv-with-an-unusual-name"
    if target.exists():
        pytest.skip("directory already present")
    (target / "lib").mkdir(parents=True)
    (target / "pyvenv.cfg").write_text("home = /usr/bin" + chr(10), encoding="ascii")
    (target / "lib" / "installed.py").write_text("x = 1" + chr(10), encoding="ascii")
    try:
        _run([sys.executable, "scripts/verify_repository.py"])
    finally:
        shutil.rmtree(target)


def test_repository_check_ignores_untracked_working_directories() -> None:
    """A working copy carries caches and a virtualenv; verification must pass."""

    created = _plant(("__pycache__", ".venv", "direct_h8_transport.egg-info"))
    if not created:
        pytest.skip("the checkout already carries these directories")
    try:
        _run([sys.executable, "scripts/verify_repository.py"])
    finally:
        for target in created:
            shutil.rmtree(target)


def test_repository_check_fails_closed_on_an_unreadable_checkout() -> None:
    """A .git that cannot be enumerated must fail, never silently downgrade."""

    if (ROOT / ".git").exists():
        pytest.skip("this is a real checkout; the failure mode cannot be staged")
    created = _plant((".git",))
    try:
        result = subprocess.run(
            [sys.executable, "scripts/verify_repository.py"],
            cwd=ROOT,
            env=_environment(),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pytest.fail(
                "verification passed on a checkout whose file list is unreadable",
                pytrace=False,
            )
        if "ls-files" not in result.stderr and "git" not in result.stderr:
            pytest.fail(
                f"expected a git enumeration error, got: {result.stderr}",
                pytrace=False,
            )
    finally:
        for target in created:
            shutil.rmtree(target)
