#!/usr/bin/env python3
"""End-to-end tests for compact artifact construction and standalone verification."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_artifact import build_artifact
from test_analysis import make_fixture, write_json


SOURCE_COMMIT = "a" * 40
CANONICAL_COMMIT = "b" * 40


def make_fake_execution_source(root: Path) -> Path:
    experiment = root / "experiment_FPP/ddpg_saber_query_replay_ls512"
    write_json(
        experiment / "config.json",
        {
            "trainer_method": "b3_support_runtime_opt",
            "support_query_mode": "replay_action",
            "weight_scope": "critic_only",
            "weighting_mode": "batch_norm",
            "local_search_max": 512,
            "total_steps": 1_000_000,
            "seeds": [1, 2, 3, 4, 5],
        },
    )
    files = {
        "run.py": "#!/usr/bin/env python3\n",
        "code/src/train_fetch_pick_and_place_lite_v0.py": "# synthetic train fixture\n",
        "code/src/metacog_lite_v0/weights.py": "# batch.action_exec replay query fixture\n",
        "code/src/test_current_policy_legacy_parity.py": "print('passed')\n",
        "code/src/test_support_query_modes.py": "print('passed')\n",
        "code/src/test_ddpg_query_loss_isolation.py": "print('passed')\n",
        "validate_config_parity.py": "print('passed')\n",
    }
    for relative, content in files.items():
        path = experiment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return experiment


def make_analysis_source(actual: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "query_ablation.py",
        "analyze_results.py",
        "build_artifact.py",
        "verify_artifact.py",
    ):
        shutil.copy2(actual / name, destination / name)
    # Avoid recursive artifact builds while still exercising captured passing
    # test evidence and the standalone verifier's required-evidence gate.
    for name in ("test_analysis.py", "test_artifact.py"):
        (destination / name).write_text("print('synthetic prepackage test passed')\n", encoding="utf-8")
    return destination


class ArtifactTests(unittest.TestCase):
    def test_build_sanitize_copy_and_standalone_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            main_root, replay_root = make_fixture(repo / "experiment_FPP/runs")
            experiment = make_fake_execution_source(repo)
            analysis = make_analysis_source(Path(__file__).resolve().parent, root / "analysis")
            write_json(
                replay_root / "_execution/launch_metadata.json",
                {
                    "schema_version": 1,
                    "experiment": "ddpg_saber_query_replay_ls512",
                    "source_commit": SOURCE_COMMIT,
                    "origin_main": SOURCE_COMMIT,
                    "head_matches_origin_main": True,
                    "worktree_clean": True,
                    "steps": 1_000_000,
                    "seeds": [1, 2, 3, 4, 5],
                    "support_query_mode": "replay_action",
                },
            )
            artifact = root / "artifact"
            result = build_artifact(
                repo=repo,
                main_root=main_root,
                replay_root=replay_root,
                artifact_root=artifact,
                experiment_root=experiment,
                analysis_root=analysis,
                execution_source_commit=SOURCE_COMMIT,
                canonical_source_commit=CANONICAL_COMMIT,
                run_tests=True,
                require_execution_metadata=True,
            )
            self.assertEqual(result["analysis_recomputation"], "verified")
            packaged_summary = json.loads(
                (artifact / "data/main/seed_1/summary.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(packaged_summary["final_lite_metrics"]["legacy/optional_nan"])
            manifest = json.loads((artifact / "input_manifest.json").read_text(encoding="utf-8"))
            summary_record = next(
                record
                for record in manifest["methods"]["main"]["runs"][0]["files"]
                if record["role"] == "summary"
            )
            self.assertEqual(
                summary_record["transformations"]["nonfinite_json_numbers_replaced_with_null"],
                1,
            )
            self.assertEqual(
                summary_record["transformations"]["nonfinite_json_paths"],
                ["$/final_lite_metrics/legacy~1optional_nan"],
            )

            copied = root / "copied-artifact"
            shutil.copytree(artifact, copied)
            completed = subprocess.run(
                [
                    sys.executable,
                    "code/verify_artifact.py",
                    "--artifact-root",
                    ".",
                ],
                cwd=copied,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"analysis_recomputation": "verified"', completed.stdout)

    def test_tamper_and_symlink_fail(self) -> None:
        # Hash and symlink failures are covered independently with the public
        # verifier against the artifact built once for this test.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            main_root, replay_root = make_fixture(repo / "experiment_FPP/runs")
            experiment = make_fake_execution_source(repo)
            analysis = make_analysis_source(Path(__file__).resolve().parent, root / "analysis")
            write_json(
                replay_root / "_execution/launch_metadata.json",
                {
                    "experiment": "ddpg_saber_query_replay_ls512",
                    "source_commit": SOURCE_COMMIT,
                    "origin_main": SOURCE_COMMIT,
                    "head_matches_origin_main": True,
                    "worktree_clean": True,
                    "steps": 1_000_000,
                    "seeds": [1, 2, 3, 4, 5],
                },
            )
            artifact = root / "artifact"
            build_artifact(
                repo=repo,
                main_root=main_root,
                replay_root=replay_root,
                artifact_root=artifact,
                experiment_root=experiment,
                analysis_root=analysis,
                execution_source_commit=SOURCE_COMMIT,
                canonical_source_commit=CANONICAL_COMMIT,
            )
            verifier = artifact / "code/verify_artifact.py"
            target = artifact / "README.md"
            target.write_text(target.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, str(verifier), "--artifact-root", str(artifact)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("mismatch", failed.stderr.lower())

            # Restore by rebuilding, then add an unmanifested symlink.
            shutil.rmtree(artifact)
            build_artifact(
                repo=repo,
                main_root=main_root,
                replay_root=replay_root,
                artifact_root=artifact,
                experiment_root=experiment,
                analysis_root=analysis,
                execution_source_commit=SOURCE_COMMIT,
                canonical_source_commit=CANONICAL_COMMIT,
            )
            (artifact / "tests/forbidden-link").symlink_to(artifact / "README.md")
            failed = subprocess.run(
                [sys.executable, str(artifact / "code/verify_artifact.py"), "--artifact-root", str(artifact)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("symlink", failed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
