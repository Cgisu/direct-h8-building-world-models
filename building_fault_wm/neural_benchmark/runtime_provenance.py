"""Canonical numerical-runtime fingerprints for persisted trained artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Mapping

import numpy as np
import torch


RUNTIME_FINGERPRINT_SCHEMA = "boptest-multicase-numerical-runtime-v1"


def _canonical_sha256(payload: object) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(content).hexdigest()


def _resolved_device(device: torch.device | str) -> torch.device:
    value = torch.device(device)
    if value.type not in {"cpu", "cuda"}:
        raise ValueError("numerical runtime supports only CPU or CUDA devices")
    if value.type == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA runtime fingerprint requested without usable CUDA")
    index = torch.cuda.current_device() if value.index is None else value.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError("CUDA runtime fingerprint has an invalid device index")
    return torch.device("cuda", index)


def numerical_runtime_fingerprint(
    device: torch.device | str,
    *,
    include_sklearn: bool,
) -> dict:
    resolved = _resolved_device(device)
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        device_payload = {
            "type": "cuda",
            "index": resolved.index,
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(resolved)),
        }
    else:
        device_payload = {
            "type": "cpu",
            "index": None,
            "name": None,
            "capability": None,
        }
    sklearn_version = None
    if include_sklearn:
        import sklearn

        sklearn_version = sklearn.__version__
    payload = {
        "schema": RUNTIME_FINGERPRINT_SCHEMA,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn_version,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device_payload,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def fingerprint_device(value: object) -> torch.device:
    if not isinstance(value, Mapping) or not isinstance(value.get("device"), Mapping):
        raise ValueError("numerical runtime fingerprint has no device identity")
    device = value["device"]
    device_type = device.get("type")
    index = device.get("index")
    if device_type == "cpu" and index is None:
        return torch.device("cpu")
    if (
        device_type == "cuda"
        and not isinstance(index, bool)
        and isinstance(index, int)
        and index >= 0
    ):
        return torch.device("cuda", index)
    raise ValueError("numerical runtime fingerprint device identity is invalid")


def validate_numerical_runtime_fingerprint(
    value: object,
    *,
    include_sklearn: bool,
) -> None:
    """Validate a recorded producer runtime without requiring this host to match it."""
    required = {
        "schema",
        "python_version",
        "python_implementation",
        "torch_version",
        "numpy_version",
        "sklearn_version",
        "torch_cuda_version",
        "cudnn_version",
        "device",
        "sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("numerical runtime fingerprint fields are invalid")
    if value["schema"] != RUNTIME_FINGERPRINT_SCHEMA:
        raise ValueError("numerical runtime fingerprint schema is invalid")
    for field in (
        "python_version",
        "python_implementation",
        "torch_version",
        "numpy_version",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"numerical runtime fingerprint {field} is invalid")
    sklearn_version = value["sklearn_version"]
    if include_sklearn:
        if not isinstance(sklearn_version, str) or not sklearn_version:
            raise ValueError("numerical runtime fingerprint sklearn_version is invalid")
    elif sklearn_version is not None:
        raise ValueError("numerical runtime fingerprint unexpectedly records sklearn")
    if value["torch_cuda_version"] is not None and (
        not isinstance(value["torch_cuda_version"], str)
        or not value["torch_cuda_version"]
    ):
        raise ValueError("numerical runtime fingerprint CUDA version is invalid")
    if value["cudnn_version"] is not None and (
        isinstance(value["cudnn_version"], bool)
        or not isinstance(value["cudnn_version"], int)
        or value["cudnn_version"] < 0
    ):
        raise ValueError("numerical runtime fingerprint cuDNN version is invalid")
    device = value["device"]
    if not isinstance(device, Mapping) or set(device) != {
        "type",
        "index",
        "name",
        "capability",
    }:
        raise ValueError("numerical runtime fingerprint device identity is invalid")
    if device["type"] == "cpu":
        if any(device[field] is not None for field in ("index", "name", "capability")):
            raise ValueError("numerical runtime fingerprint CPU identity is invalid")
    elif device["type"] == "cuda":
        capability = device["capability"]
        if (
            isinstance(device["index"], bool)
            or not isinstance(device["index"], int)
            or device["index"] < 0
            or not isinstance(device["name"], str)
            or not device["name"]
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in capability
            )
            or value["torch_cuda_version"] is None
        ):
            raise ValueError("numerical runtime fingerprint CUDA identity is invalid")
    else:
        raise ValueError("numerical runtime fingerprint device type is invalid")
    payload = {key: value[key] for key in required if key != "sha256"}
    if value["sha256"] != _canonical_sha256(payload):
        raise ValueError("numerical runtime fingerprint self-hash is invalid")


def validate_current_numerical_runtime_fingerprint(
    value: object,
    *,
    include_sklearn: bool,
) -> None:
    """Require the recorded runtime to equal this host, for resume/reuse only."""
    validate_numerical_runtime_fingerprint(value, include_sklearn=include_sklearn)
    expected = numerical_runtime_fingerprint(
        fingerprint_device(value), include_sklearn=include_sklearn
    )
    if value != expected:
        raise ValueError(
            "numerical runtime fingerprint differs from the current environment"
        )
