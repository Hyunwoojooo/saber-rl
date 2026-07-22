#!/usr/bin/env python3
"""Fail-closed tests for deterministic full-package fingerprints."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from build_artifact import fingerprint_tree as build_artifact_fingerprint_tree
from source_fingerprint import (
    FingerprintError,
    make_fingerprint,
    tree_sha256,
    validate_fingerprint_payload,
    verify_freeze_in_git,
    verify_current,
)
from verify_artifact import fingerprint_tree as verify_artifact_fingerprint_tree


class SourceFingerprintTests(unittest.TestCase):
    def test_artifact_consumers_share_the_execution_fingerprint_domain(self) -> None:
        files = {
            "config.json": "1" * 64,
            "code/src/weights.py": "a" * 64,
        }
        expected = tree_sha256(files)
        self.assertEqual(build_artifact_fingerprint_tree(files), expected)
        self.assertEqual(verify_artifact_fingerprint_tree(files), expected)

    def test_git_preflight_requires_every_frozen_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            root = (
                repo
                / "experiment_FPP/ddpg_saber_random_support_count_matched_ls512"
            )
            root.mkdir(parents=True)
            source = root / "run.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            fingerprint = make_fingerprint("freeze", root)
            freeze = root / "freeze_manifest.json"
            freeze.write_text(json.dumps({
                "source_fingerprint": fingerprint,
                "frozen_files_sha256": fingerprint["files_sha256"],
            }, sort_keys=True) + "\n", encoding="utf-8")
            for args in (
                ("init",),
                ("config", "user.email", "fingerprint-test@example.invalid"),
                ("config", "user.name", "Fingerprint Test"),
                ("add", "."),
                ("commit", "-m", "frozen source"),
            ):
                subprocess.run(
                    ["git", *args], cwd=repo, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                stdout=subprocess.PIPE, text=True,
            ).stdout.strip()
            verified_commit, hashes = verify_freeze_in_git(
                freeze, commit, root=root, repo=repo
            )
            self.assertEqual(verified_commit, commit)
            self.assertEqual(hashes, fingerprint["files_sha256"])

            ignored = root / "analysis/ignored.py"
            ignored.parent.mkdir()
            ignored.write_text("# untracked\n", encoding="utf-8")
            drift = make_fingerprint("freeze", root)
            freeze.write_text(json.dumps({
                "source_fingerprint": drift,
                "frozen_files_sha256": drift["files_sha256"],
            }, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(FingerprintError):
                verify_freeze_in_git(freeze, commit, root=root, repo=repo)

            ignored.unlink()
            ignored.parent.rmdir()
            source.write_text("VALUE = 2\n", encoding="utf-8")
            drift = make_fingerprint("freeze", root)
            freeze.write_text(json.dumps({
                "source_fingerprint": drift,
                "frozen_files_sha256": drift["files_sha256"],
            }, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(FingerprintError, "Git blob differs"):
                verify_freeze_in_git(freeze, commit, root=root, repo=repo)

    def test_exclusions_are_stable_but_source_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "code/src/module.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (root / "freeze_manifest.json").write_text("{}\n", encoding="utf-8")
            cache = root / "code/src/__pycache__/module.pyc"
            cache.parent.mkdir()
            cache.write_bytes(b"cache")
            payload = make_fingerprint("smoke", root)
            verify_current(payload, root=root, expected_stage="smoke")
            (root / "freeze_manifest.json").write_text("changed\n", encoding="utf-8")
            cache.write_bytes(b"changed cache")
            verify_current(payload, root=root, expected_stage="smoke")
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(FingerprintError, "changed=.*module.py"):
                verify_current(payload, root=root, expected_stage="smoke")

    def test_stage_tree_and_path_validation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run.py").write_text("# source\n", encoding="utf-8")
            payload = make_fingerprint("gate", root)
            with self.assertRaisesRegex(FingerprintError, "stage mismatch"):
                validate_fingerprint_payload(payload, expected_stage="smoke")
            forged = json.loads(json.dumps(payload))
            digest = forged["files_sha256"].pop("run.py")
            forged["files_sha256"]["../run.py"] = digest
            with self.assertRaisesRegex(FingerprintError, "unsafe"):
                validate_fingerprint_payload(forged, expected_stage="gate")
            forged = json.loads(json.dumps(payload))
            forged["tree_sha256"] = "0" * 64
            with self.assertRaisesRegex(FingerprintError, "tree"):
                validate_fingerprint_payload(forged, expected_stage="gate")


if __name__ == "__main__":
    unittest.main()
