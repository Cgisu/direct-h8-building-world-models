#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys

EXPECTED_MATPLOTLIB = "3.10.9"

MODULES = (
    "matplotlib",
    "nfoursid",
    "numpy",
    "pandas",
    "PIL",
    "requests",
    "scipy",
    "sklearn",
    "threadpoolctl",
    "torch",
)

missing = []
for name in MODULES:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        missing.append(name)

if missing:
    print("VERIFICATION DEPENDENCIES: MISSING", file=sys.stderr)
    print("  modules: " + ", ".join(missing), file=sys.stderr)
    print(
        f"  install with: {sys.executable} -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2)

matplotlib = importlib.import_module("matplotlib")
if matplotlib.__version__ != EXPECTED_MATPLOTLIB:
    print("VERIFICATION DEPENDENCIES: VERSION MISMATCH", file=sys.stderr)
    print(
        f"  matplotlib: {matplotlib.__version__}; "
        f"expected {EXPECTED_MATPLOTLIB}",
        file=sys.stderr,
    )
    raise SystemExit(2)

print(f"VERIFICATION INTERPRETER: {sys.executable}")
print("VERIFICATION DEPENDENCIES: PASS")
