from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from . import runner


def synthetic_runtime_report() -> dict:
    identity = {
        "expected_uid": runner.EXPECTED_UID,
        "expected_gid": runner.EXPECTED_GID,
        "observed_uid": runner.EXPECTED_UID,
        "observed_gid": runner.EXPECTED_GID,
        "validated": True,
    }
    return {
        "worker_image_validated": True,
        "worker_entrypoint_validated": True,
        "host_identity": identity,
        "ownership_probe": {
            "schema": "direct-h8-v5-ownership-probe-v1",
            "worker_image_id": runner.boptest.WORKER_IMAGE_ID,
            "output_uid": runner.EXPECTED_UID,
            "output_gid": runner.EXPECTED_GID,
            "recursive_paths_checked": 3,
            "host_write_delete_validated": True,
            "command_contract_sha256": (
                runner.ownership_probe_contract_sha256()
            ),
            "validated": True,
            "host_identity": identity,
        },
    }


@pytest.fixture(scope="session")
def response_blind_readiness(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("direct_h8_v5_readiness")
    data_root = root / "data"
    state_root = root / "state_v3"
    freeze_root = root / "freeze_v5"
    runner.stage_frozen_plan_assets(
        data_root=data_root,
        state_root=state_root,
        freeze_root=freeze_root,
        prelock_root=runner.PRELOCK_ROOT,
    )
    public_source = {
        "repository_url": runner.boptest.BOPTEST_REPOSITORY_URL,
        "commit": runner.boptest.BOPTEST_COMMIT,
        "license_sha256": runner.boptest.BOPTEST_LICENSE_SHA256,
    }
    with (
        patch.object(
            runner.frozen_worker, "_validate_public_source", return_value=None
        ),
        patch.object(
            runner,
            "runtime_readiness_report",
            return_value=(synthetic_runtime_report(), public_source),
        ),
    ):
        readiness = runner.prepare_readiness(
            data_root=data_root,
            state_root=state_root,
            freeze_root=freeze_root,
            prelock_root=runner.PRELOCK_ROOT,
            testcase_root=runner.TESTCASE_ROOT,
        )
    readiness_path = freeze_root / "collection_readiness.json"
    runner.write_readiness_report(readiness_path, readiness.report)
    return {
        "root": root,
        "data_root": data_root,
        "state_root": state_root,
        "freeze_root": freeze_root,
        "readiness_path": readiness_path,
        "readiness": readiness,
    }
