from __future__ import annotations

import copy
import platform

import numpy as np
import pytest
import torch

from .runtime_provenance import (
    RUNTIME_FINGERPRINT_SCHEMA,
    _canonical_sha256,
    fingerprint_device,
    numerical_runtime_fingerprint,
    validate_current_numerical_runtime_fingerprint,
    validate_numerical_runtime_fingerprint,
)


def test_cpu_runtime_fingerprint_is_complete_and_self_authenticating():
    fingerprint = numerical_runtime_fingerprint("cpu", include_sklearn=False)
    assert set(fingerprint) == {
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
    assert fingerprint["schema"] == RUNTIME_FINGERPRINT_SCHEMA
    assert fingerprint["python_version"] == platform.python_version()
    assert fingerprint["torch_version"] == str(torch.__version__)
    assert fingerprint["numpy_version"] == np.__version__
    assert fingerprint["sklearn_version"] is None
    assert fingerprint["device"] == {
        "type": "cpu",
        "index": None,
        "name": None,
        "capability": None,
    }
    assert fingerprint_device(fingerprint) == torch.device("cpu")
    validate_numerical_runtime_fingerprint(fingerprint, include_sklearn=False)


def test_runtime_fingerprint_rejects_environment_and_digest_tamper():
    fingerprint = numerical_runtime_fingerprint("cpu", include_sklearn=True)
    assert isinstance(fingerprint["sklearn_version"], str)
    changed = copy.deepcopy(fingerprint)
    changed["torch_version"] = "0.0.0"
    with pytest.raises(ValueError, match="self-hash"):
        validate_numerical_runtime_fingerprint(changed, include_sklearn=True)

    changed = copy.deepcopy(fingerprint)
    changed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash"):
        validate_numerical_runtime_fingerprint(changed, include_sklearn=True)

    validate_current_numerical_runtime_fingerprint(
        fingerprint, include_sklearn=True
    )
    current_mismatch = numerical_runtime_fingerprint(
        "cpu", include_sklearn=True
    )
    current_mismatch["python_version"] = "0.0.0"
    current_mismatch["sha256"] = _canonical_sha256(
        {key: value for key, value in current_mismatch.items() if key != "sha256"}
    )
    validate_numerical_runtime_fingerprint(
        current_mismatch, include_sklearn=True
    )
    with pytest.raises(ValueError, match="current environment"):
        validate_current_numerical_runtime_fingerprint(
            current_mismatch, include_sklearn=True
        )


def test_runtime_fingerprint_rejects_invalid_device_identity():
    fingerprint = numerical_runtime_fingerprint("cpu", include_sklearn=False)
    fingerprint["device"] = {
        "type": "cuda",
        "index": -1,
        "name": "invalid",
        "capability": [0, 0],
    }
    with pytest.raises(ValueError, match="CUDA identity"):
        validate_numerical_runtime_fingerprint(
            fingerprint, include_sklearn=False
        )
