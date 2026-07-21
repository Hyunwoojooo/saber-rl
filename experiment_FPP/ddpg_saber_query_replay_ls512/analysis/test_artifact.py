#!/usr/bin/env python3
"""End-to-end tests for compact artifact construction and standalone verification."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_artifact import (
    ArtifactBuildError,
    build_artifact,
    resolve_execution_source_commit,
    write_manifests,
)
from test_analysis import make_fixture, write_json


EXPERIMENT_REL = Path("experiment_FPP/ddpg_saber_query_replay_ls512")
MAIN_SOURCE_REL = Path("experiment_FPP/b3_critic_only_runtime_opt_v4_ls512")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return completed.stdout.strip()


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
        "analysis/reference.txt": "tracked analysis fixture\n",
    }
    for relative, content in files.items():
        path = experiment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return experiment


def make_fake_main_source(root: Path) -> Path:
    experiment = root / MAIN_SOURCE_REL
    write_json(
        experiment / "config.json",
        {
            "trainer_method": "b3_support_runtime_opt",
            "weight_scope": "critic_only",
            "weighting_mode": "batch_norm",
            "local_search_max": 512,
            "total_steps": 1_000_000,
            "seeds": [1, 2, 3, 4, 5],
        },
    )
    files = {
        "run.py": "#!/usr/bin/env python3\n",
        "code/src/train_fetch_pick_and_place_lite_v0.py": "# synthetic canonical trainer\n",
        "code/src/metacog_lite_v0/weights.py": "# synthetic canonical weights\n",
    }
    for relative, content in files.items():
        path = experiment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return experiment


def make_committed_sources(repo: Path) -> tuple[Path, str, str, str, str]:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init")
    git(repo, "config", "user.name", "Artifact Test")
    git(repo, "config", "user.email", "artifact-test@example.invalid")
    make_fake_main_source(repo)
    git(repo, "add", MAIN_SOURCE_REL.as_posix())
    git(repo, "commit", "-m", "Add canonical fixture")
    canonical_commit = git(repo, "rev-parse", "HEAD")
    experiment = make_fake_execution_source(repo)
    git(repo, "add", EXPERIMENT_REL.as_posix())
    git(repo, "commit", "-m", "Add Replay fixture")
    source_commit = git(repo, "rev-parse", "HEAD")
    source_tree = git(repo, "rev-parse", f"{source_commit}:{EXPERIMENT_REL.as_posix()}/code/src")
    config_blob = git(repo, "rev-parse", f"{source_commit}:{EXPERIMENT_REL.as_posix()}/config.json")
    return experiment, source_commit, canonical_commit, source_tree, config_blob


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
    def test_source_cleanliness_excludes_only_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            experiment, source_commit, _canonical, _tree, _blob = make_committed_sources(repo)
            tracked_analysis = experiment / "analysis/reference.txt"
            tracked_analysis.write_text("dirty analysis fixture\n", encoding="utf-8")
            (experiment / "analysis/untracked.txt").write_text(
                "untracked analysis fixture\n", encoding="utf-8"
            )
            self.assertEqual(
                resolve_execution_source_commit(
                    repo, source_commit, require_clean=True
                ),
                source_commit,
            )

            trainer = experiment / "code/src/metacog_lite_v0/weights.py"
            trainer.write_text(
                trainer.read_text(encoding="utf-8") + "# dirty trainer\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ArtifactBuildError, "differs outside analysis"
            ):
                resolve_execution_source_commit(
                    repo, source_commit, require_clean=True
                )

            git(repo, "add", str(trainer.relative_to(repo)))
            git(repo, "commit", "-m", "Change trainer after execution")
            with self.assertRaisesRegex(
                ArtifactBuildError, "paths differing from execution commit"
            ):
                resolve_execution_source_commit(
                    repo, source_commit, require_clean=True
                )

    def test_build_sanitize_copy_and_standalone_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            main_root, replay_root = make_fixture(repo / "experiment_FPP/runs")
            experiment, source_commit, canonical_commit, source_tree, config_blob = (
                make_committed_sources(repo)
            )
            analysis = make_analysis_source(Path(__file__).resolve().parent, root / "analysis")
            write_json(
                replay_root / "_execution/launch_metadata.json",
                {
                    "schema_version": 1,
                    "experiment": "ddpg_saber_query_replay_ls512",
                    "source_commit": source_commit,
                    "origin_main": source_commit,
                    "head_matches_origin_main": True,
                    "worktree_clean": True,
                    "source_tree_git_oid": source_tree,
                    "config_git_blob_oid": config_blob,
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
                execution_source_commit=source_commit,
                canonical_source_commit=canonical_commit,
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
            self.assertIn(
                '"source_provenance": "commit_tree_blob_chains_verified"',
                completed.stdout,
            )
            self.assertIn(
                '"canonical_main_run_execution_commit": "not_captured"',
                completed.stdout,
            )
            self.assertTrue((artifact / "code/main_source/config.json").is_file())

    def test_rehashed_source_tamper_still_fails_git_blob_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            main_root, replay_root = make_fixture(repo / "experiment_FPP/runs")
            experiment, source_commit, canonical_commit, source_tree, config_blob = (
                make_committed_sources(repo)
            )
            analysis = make_analysis_source(Path(__file__).resolve().parent, root / "analysis")
            write_json(
                replay_root / "_execution/launch_metadata.json",
                {
                    "experiment": "ddpg_saber_query_replay_ls512",
                    "source_commit": source_commit,
                    "origin_main": source_commit,
                    "head_matches_origin_main": True,
                    "worktree_clean": True,
                    "source_tree_git_oid": source_tree,
                    "config_git_blob_oid": config_blob,
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
                execution_source_commit=source_commit,
                canonical_source_commit=canonical_commit,
            )

            path_tamper = root / "path-tamper"
            shutil.copytree(artifact, path_tamper)
            path_provenance = json.loads(
                (path_tamper / "code/source_provenance.json").read_text(encoding="utf-8")
            )
            path_provenance["replay_execution"]["source_path_at_commit"] = (
                "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512"
            )
            write_json(path_tamper / "code/source_provenance.json", path_provenance)
            write_manifests(path_tamper, source_commit, canonical_commit)
            failed_path = subprocess.run(
                [
                    sys.executable,
                    str(path_tamper / "code/verify_artifact.py"),
                    "--artifact-root",
                    str(path_tamper),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed_path.returncode, 2)
            self.assertIn("source_path_at_commit", failed_path.stderr)

            omission = root / "omission-tamper"
            shutil.copytree(artifact, omission)
            omitted_relative = "validate_config_parity.py"
            omitted_packaged = f"code/execution_source/{omitted_relative}"
            (omission / omitted_packaged).unlink()
            omission_manifest = json.loads(
                (omission / "input_manifest.json").read_text(encoding="utf-8")
            )
            omission_manifest["frozen_execution_source_files"] = [
                record
                for record in omission_manifest["frozen_execution_source_files"]
                if record["packaged_path"] != omitted_packaged
            ]
            write_json(omission / "input_manifest.json", omission_manifest)
            omission_provenance = json.loads(
                (omission / "code/source_provenance.json").read_text(encoding="utf-8")
            )
            omission_provenance["replay_execution"]["selected_files"] = [
                record
                for record in omission_provenance["replay_execution"]["selected_files"]
                if record["packaged_path"] != omitted_packaged
            ]
            write_json(omission / "code/source_provenance.json", omission_provenance)
            write_manifests(omission, source_commit, canonical_commit)
            failed_omission = subprocess.run(
                [
                    sys.executable,
                    str(omission / "code/verify_artifact.py"),
                    "--artifact-root",
                    str(omission),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed_omission.returncode, 2)
            self.assertIn("incomplete relative", failed_omission.stderr)

            target_relative = "code/execution_source/code/src/metacog_lite_v0/weights.py"
            target = artifact / target_relative
            target.write_text(target.read_text(encoding="utf-8") + "# forged\n", encoding="utf-8")
            altered_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            altered_size = target.stat().st_size
            input_manifest = json.loads(
                (artifact / "input_manifest.json").read_text(encoding="utf-8")
            )
            input_record = next(
                record
                for record in input_manifest["frozen_execution_source_files"]
                if record["packaged_path"] == target_relative
            )
            input_record["source_sha256"] = altered_sha
            input_record["source_size_bytes"] = altered_size
            write_json(artifact / "input_manifest.json", input_manifest)
            provenance = json.loads(
                (artifact / "code/source_provenance.json").read_text(encoding="utf-8")
            )
            provenance_record = next(
                record
                for record in provenance["replay_execution"]["selected_files"]
                if record["packaged_path"] == target_relative
            )
            provenance_record["sha256"] = altered_sha
            provenance_record["size_bytes"] = altered_size
            write_json(artifact / "code/source_provenance.json", provenance)
            write_manifests(artifact, source_commit, canonical_commit)
            failed = subprocess.run(
                [sys.executable, str(artifact / "code/verify_artifact.py"), "--artifact-root", str(artifact)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("git blob mismatch", failed.stderr.lower())

    def test_launch_tree_oid_mismatch_fails_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            main_root, replay_root = make_fixture(repo / "experiment_FPP/runs")
            experiment, source_commit, canonical_commit, _source_tree, config_blob = (
                make_committed_sources(repo)
            )
            analysis = make_analysis_source(Path(__file__).resolve().parent, root / "analysis")
            write_json(
                replay_root / "_execution/launch_metadata.json",
                {
                    "experiment": "ddpg_saber_query_replay_ls512",
                    "source_commit": source_commit,
                    "origin_main": source_commit,
                    "head_matches_origin_main": True,
                    "worktree_clean": True,
                    "source_tree_git_oid": "0" * 40,
                    "config_git_blob_oid": config_blob,
                    "steps": 1_000_000,
                    "seeds": [1, 2, 3, 4, 5],
                },
            )
            with self.assertRaisesRegex(ArtifactBuildError, "source_tree_git_oid"):
                build_artifact(
                    repo=repo,
                    main_root=main_root,
                    replay_root=replay_root,
                    artifact_root=root / "artifact",
                    experiment_root=experiment,
                    analysis_root=analysis,
                    execution_source_commit=source_commit,
                    canonical_source_commit=canonical_commit,
                )

    def test_tamper_and_symlink_fail(self) -> None:
        # Hash and symlink failures are covered independently with the public
        # verifier against the artifact built once for this test.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            main_root, replay_root = make_fixture(repo / "experiment_FPP/runs")
            experiment, source_commit, canonical_commit, source_tree, config_blob = (
                make_committed_sources(repo)
            )
            analysis = make_analysis_source(Path(__file__).resolve().parent, root / "analysis")
            write_json(
                replay_root / "_execution/launch_metadata.json",
                {
                    "experiment": "ddpg_saber_query_replay_ls512",
                    "source_commit": source_commit,
                    "origin_main": source_commit,
                    "head_matches_origin_main": True,
                    "worktree_clean": True,
                    "source_tree_git_oid": source_tree,
                    "config_git_blob_oid": config_blob,
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
                execution_source_commit=source_commit,
                canonical_source_commit=canonical_commit,
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
                execution_source_commit=source_commit,
                canonical_source_commit=canonical_commit,
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
