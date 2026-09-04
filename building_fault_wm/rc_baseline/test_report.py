"""Contract tests for the sealed reviewer RC report."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .report import build_report
from .report_verify import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
    strict_json,
    verify_report,
)


class RcReportContractTests(unittest.TestCase):
    @staticmethod
    def _make_directories_writable(root: Path) -> None:
        for path in (root, *sorted(root.rglob("*"))):
            if path.is_dir():
                path.chmod(0o755)

    def test_correctly_built_report_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "report"
            try:
                build_report(root)
                result = verify_report(root)
                self.assertTrue(result["verified"])
            finally:
                if root.exists():
                    self._make_directories_writable(root)

    def test_evaluation_snapshot_must_match_completion_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "report"
            try:
                build_report(root)
                snapshot = root / "selected_hyperparameters_evaluation_snapshot.csv"
                snapshot.chmod(0o644)
                with snapshot.open("ab") as stream:
                    stream.write(b"\n")

                manifest_path = root / "report_manifest.json"
                manifest_path.chmod(0o644)
                manifest = strict_json(manifest_path)
                inventory = manifest[
                    "artifact_inventory_excludes_manifest_and_digest"
                ]
                for item in inventory:
                    if item["path"] == snapshot.name:
                        item["bytes"] = snapshot.stat().st_size
                        item["sha256"] = sha256_file(snapshot)
                        break
                else:
                    self.fail("evaluation snapshot missing from report inventory")
                manifest["artifact_inventory_sha256"] = canonical_sha256(inventory)
                manifest_path.write_bytes(canonical_json_bytes(manifest))

                digest_path = root / "report_manifest.canonical.sha256"
                digest_path.chmod(0o644)
                digest_path.write_text(
                    canonical_sha256(manifest) + "\n", encoding="ascii"
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "^RC report evaluation-snapshot binding changed$",
                ):
                    verify_report(root, require_read_only=False)
            finally:
                if root.exists():
                    self._make_directories_writable(root)


if __name__ == "__main__":
    unittest.main()
