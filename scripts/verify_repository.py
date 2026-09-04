#!/usr/bin/env python3
"""Verify the public repository against its sealed file manifest.

In a Git checkout the release file set is exactly what Git tracks, so a file
cannot escape the manifest by living under an ignored directory name. In a
plain archive extract there is no index to consult, so the tree is enumerated
directly and only the directories a working copy legitimately creates are
skipped; the archive form is additionally pinned by its published SHA-256.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "FILE_MANIFEST.sha256"
REQUIRED = (
    "building_fault_wm/neural_benchmark/reliability_model.py",
    "building_fault_wm/neural_benchmark/data/manifests/locked_transport_corpus_manifest.json",
    "building_fault_wm/deterministic_transport/model.py",
    "building_fault_wm/transport_collection/runner.py",
    "building_fault_wm/transport_evaluation/adapter.py",
    "building_fault_wm/ridge_arx/train.py",
    "building_fault_wm/ridge_arx_sensitivity/study.py",
    "building_fault_wm/subspace_baseline/study.py",
    "building_fault_wm/rc_baseline/model.py",
    "building_fault_wm/comparison_guard/guard.py",
    "building_fault_wm/downstream_control/experiment.py",
    "building_fault_wm/downstream_multicase/experiment.py",
    "results/neural/primary_estimands.csv",
    "results/ridge_arx/sensitivity_result.json",
    "results/subspace/comparison_result.json",
    "results/rc/comparison_result.json",
    "results/downstream/episode_summary.csv",
    "results/downstream/finite_panel_summary.csv",
    "results/downstream/paired_effects.csv",
    "results/downstream/protocol.json",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "scripts/portable_tests.txt",
)
FORBIDDEN_TOP_LEVEL = {"artifacts", "built_outputs", "generated", "evidence_summaries"}
# Directories a working checkout legitimately creates but the manifest never
# covers: version control, virtual environments, caches, and build output.
# This mirrors .gitignore, so a fresh clone verifies without pruning anything.
IGNORED_PARTS = {
    ".eggs",
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".matplotlib-cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "env",
    "venv",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
# Local development paths and non-release product identifiers must not appear
# anywhere in the published repository.
FORBIDDEN_BYTES = tuple(bytes.fromhex(value) for value in (
    "2f686f6d652f61696469",
    "636f646578", "63686174677074", "636f70696c6f74", "636c61756465",
))
# Author and institutional identity belongs only in the citation and licensing
# files; it must not leak into code, data, or result files.
IDENTITY_BYTES = tuple(bytes.fromhex(value) for value in (
    "40756165752e61632e6165",
    "756e69746564206172616220656d69726174657320756e6976657273697479",
    "63656e74657220666f7220616920616e64206469676974616c20696e6e6f766174696f6e",
    "697361696173206768656272656869776574",
    "6e617a6172207a616b69",
    "696768656272656869776574",
    "6e7a616b6940",
))
IDENTITY_ALLOWED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
}

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

class EnumerationError(RuntimeError):
    """A Git checkout is present but its tracked file list cannot be read."""

def tracked_files() -> list[str] | None:
    """Every file Git tracks, so nothing escapes by living under an ignored name.

    Fails closed. When a checkout is present the tracked list is the only
    trustworthy file set, so an unreadable index is an error rather than a
    silent downgrade to the weaker tree walk.
    """

    if not (ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise EnumerationError(
            "this is a Git checkout but git could not be run "
            f"({error}); install git, or verify a clean archive extract instead"
        ) from error
    if result.returncode:
        raise EnumerationError(
            "this is a Git checkout but 'git ls-files' failed: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return sorted(
        name
        for name in result.stdout.decode("utf-8").split(chr(0))
        if name and name != MANIFEST.name
    )

def virtualenv_roots() -> set[Path]:
    """Directories holding a pyvenv.cfg, whatever the user named them.

    Matching on the marker file rather than on a fixed list of names means a
    virtual environment called .venv-ci or venv311 is skipped just like .venv.
    """

    return {marker.parent for marker in ROOT.rglob("pyvenv.cfg")}


def extracted_files() -> list[str]:
    """Every file in a plain extract, minus directories a working copy creates."""

    environments = virtualenv_roots()
    names = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if set(relative.parts) & IGNORED_PARTS or path.suffix in IGNORED_SUFFIXES:
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if any(environment in path.parents for environment in environments):
            continue
        names.append(relative.as_posix())
    return sorted(names)

def release_files() -> tuple[list[str], str]:
    names = tracked_files()
    if names is not None:
        return names, "git-tracked"
    return extracted_files(), "extracted tree"

def main() -> int:
    errors = []
    try:
        names, mode = release_files()
    except EnumerationError as error:
        print("CODE REPOSITORY: FAIL", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 1
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    for name in FORBIDDEN_TOP_LEVEL:
        if (ROOT / name).exists():
            errors.append(f"excluded top-level directory is present: {name}")
    expected = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        value, relative = line.split("  ", 1)
        expected[relative] = value
    print(f"  enumeration mode: {mode}")
    actual = {}
    identity_files = 0
    for name in names:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"listed release file is missing: {name}")
            continue
        data = path.read_bytes()
        lowered = data.lower()
        if any(token in lowered for token in FORBIDDEN_BYTES):
            errors.append(f"excluded content is present: {name}")
        if any(token in lowered for token in IDENTITY_BYTES):
            if name in IDENTITY_ALLOWED:
                identity_files += 1
            else:
                errors.append(f"identity material outside the citation files: {name}")
        actual[name] = digest(path)
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        errors.append("repository file manifest does not match")
        for name in missing:
            errors.append(f"  manifested file is missing: {name}")
        for name in extra:
            errors.append(f"  unmanifested file is present: {name}")
        for name in changed:
            errors.append(f"  digest differs: {name}")
    if errors:
        print("CODE REPOSITORY: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("CODE REPOSITORY: PASS")
    print(f"  manifested files: {len(actual)}")
    print(
        f"  identity-bearing files: {identity_files} of "
        f"{len(IDENTITY_ALLOWED)} permitted"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
