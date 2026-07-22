#!/usr/bin/env python3
"""End-to-end tests for batch-wide Random-Support artifact construction."""

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
    BuildError,
    build_artifact,
    verify_freeze_against_commit,
    write_manifests,
)
from test_analysis import make_fixture, write_json


SOURCE_COMMIT = "a" * 40
CANONICAL_COMMIT = "b" * 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def fingerprint_tree(files_sha256: dict[str, str]) -> str:
    canonical = json.dumps(
        files_sha256, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"random-support-count-matched-source-fingerprint-v1\0" + canonical
    ).hexdigest()


def fingerprint_payload(stage: str, files_sha256: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": "full_experiment_package",
        "stage": stage,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "exclusions": {
            "names": ["freeze_manifest.json"],
            "directory_names": ["__pycache__"],
            "suffixes": [".nbc", ".nbi", ".pyc"],
        },
        "file_count": len(files_sha256),
        "tree_sha256": fingerprint_tree(files_sha256),
        "files_sha256": files_sha256,
    }


def validation_payload(
    kind: str, repo: Path, files_sha256: dict[str, str]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": kind,
        "passed": True,
        "performance_used_for_gate_decision": False,
        "steps": 5_000 if kind == "smoke" else 50_000,
        "seeds": [1],
        "source_fingerprint": fingerprint_payload(kind, files_sha256),
        "runs": [
            {
                "seed": 1,
                "stage": kind,
                "steps": 5_000 if kind == "smoke" else 50_000,
                "run_dir": str(repo / "machine-specific/raw-run"),
                "assignment_diagnostics": {
                    "exact_k_mismatch_count": 0,
                    "raw_histogram_mismatch_count": 0,
                    "normalized_histogram_mismatch_count": 0,
                    "invariant_failure_count": 0,
                },
                "checkpoint": {
                    "assignment_rng_state_present": True,
                    "diagnostic_state_present": True,
                },
                "log": {"resume_exercised": True},
            }
        ],
    }


def make_fake_execution_source(repo: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    experiment = repo / "experiment_FPP/ddpg_saber_random_support_count_matched_ls512"
    write_json(
        experiment / "config.json",
        {
            "run_slug": "ddpg_saber_random_support_count_matched_ls512",
            "support_query_mode": "current_policy",
            "support_assignment_mode": "batch_wide_count_matched",
            "assignment_seed_offset": 30_000,
            "assignment_diagnostic_window_updates": 1_000,
            "seeds": [1, 2, 3, 4, 5],
            "total_steps": 1_000_000,
        },
    )
    files = {
        "run.py": "#!/usr/bin/env python3\n",
        "source_fingerprint.py": "# synthetic source fingerprint helper\n",
        "test_source_fingerprint.py": "print('passed')\n",
        "validate_config_parity.py": "# synthetic validator\n",
        "validate_results.py": "# synthetic completion gate\n",
        "code/src/train_fetch_pick_and_place_lite_v0.py": "# synthetic trainer CLI\n",
        "code/src/metacog_lite_v0/weights.py": "# batch-wide assignment semantics\n",
        "code/src/metacog_lite_v0/trainer.py": "# assignment checkpoint state\n",
        "code/src/metacog_lite_v0/numba_support_scan_fpp.py": "# canonical scanner\n",
        "code/src/test_batch_wide_count_matched.py": "print('passed')\n",
        "code/src/test_trainer_resume_contract.py": "print('passed')\n",
        "code/src/test_resume_output_rollback.py": "print('passed')\n",
        "code/src/weight_parity_fixture.py": "# parity fixture\n",
    }
    for relative, content in files.items():
        path = experiment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    frozen = {
        path.relative_to(experiment).as_posix(): sha256(path)
        for path in sorted(experiment.rglob("*"))
        if path.is_file()
    }
    smoke = validation_payload("smoke", repo, frozen)
    gate = validation_payload("gate", repo, frozen)
    write_json(
        experiment / "freeze_manifest.json",
        {
            "schema_version": 1,
            "repository_head_at_freeze": SOURCE_COMMIT,
            "production_config_frozen": True,
            "performance_used_for_gate_decision": False,
            "smoke_validation": smoke,
            "gate_validation": gate,
            "source_fingerprint": fingerprint_payload("freeze", frozen),
            "frozen_files_sha256": frozen,
        },
    )
    return experiment, smoke, gate


def prepare_inputs(root: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    repo = root / "repo"
    main_root, control_root = make_fixture(repo / "experiment_FPP/runs")
    summary_path = control_root / "synthetic_control_s1_1000000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_lite_metrics"] = {
        "b3/support_to_success_corr": float("nan")
    }
    summary_path.write_text(
        json.dumps(summary, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    experiment, _smoke, _gate = make_fake_execution_source(repo)
    execution = {
        "schema_version": 1,
        "experiment": "ddpg_saber_random_support_count_matched_ls512",
        "source_commit": SOURCE_COMMIT,
        "origin_main": SOURCE_COMMIT,
        "head_matches_origin_main": True,
        "worktree_clean": True,
        "support_query_mode": "current_policy",
        "support_assignment_mode": "batch_wide_count_matched",
        "assignment_seed_rule": "30000+seed",
        "weight_scope": "critic_only",
        "steps": 1_000_000,
        "seeds": [1, 2, 3, 4, 5],
        "source_tree_git_oid": "c" * 40,
        "config_git_blob_oid": git_blob_oid(experiment / "config.json"),
        "freeze_manifest_git_blob_oid": git_blob_oid(
            experiment / "freeze_manifest.json"
        ),
    }
    metadata_path = control_root / "_execution/launch_metadata.json"
    write_json(metadata_path, execution)
    write_json(
        control_root / "_execution/completion_validation.json",
        {
            "schema_version": 1,
            "kind": "production",
            "passed": True,
            "performance_used_for_gate_decision": False,
            "steps": 1_000_000,
            "seeds": [1, 2, 3, 4, 5],
            "source_fingerprint": fingerprint_payload(
                "production",
                json.loads(
                    (experiment / "freeze_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["frozen_files_sha256"],
            ),
        },
    )
    return repo, main_root, control_root, experiment, execution


def build_synthetic(root: Path) -> Path:
    repo, main_root, control_root, experiment, execution = prepare_inputs(root)
    artifact = root / "artifact"
    build_artifact(
        repo=repo,
        main_root=main_root,
        control_root=control_root,
        artifact_root=artifact,
        experiment_root=experiment,
        analysis_root=Path(__file__).resolve().parent,
        execution_source_commit=SOURCE_COMMIT,
        canonical_source_commit=CANONICAL_COMMIT,
        execution_metadata_path=control_root / "_execution/launch_metadata.json",
        execution_metadata=execution,
        run_tests=False,
        allow_synthetic_verification=True,
    )
    # Convert the synthetic-only bypass into ordinary passing test evidence so
    # the copied standalone CLI can be exercised without a test-only flag.
    write_json(
        artifact / "tests/test_evidence.json",
        {
            "schema_version": 1,
            "status": "passed",
            "commands": [{"test": "synthetic-fixture", "returncode": 0, "passed": True}],
        },
    )
    write_manifests(artifact, SOURCE_COMMIT, CANONICAL_COMMIT)
    return artifact


def run_standalone(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "code/verify_artifact.py", "--artifact-root", "."],
        cwd=artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ArtifactTests(unittest.TestCase):
    def test_every_frozen_file_must_be_tracked_and_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            experiment = (
                repo
                / "experiment_FPP/ddpg_saber_random_support_count_matched_ls512"
            )
            weights = experiment / "code/src/metacog_lite_v0/weights.py"
            ignored_analysis = experiment / "analysis/random_support_analysis.py"
            weights.parent.mkdir(parents=True)
            ignored_analysis.parent.mkdir(parents=True)
            weights.write_text("# frozen weights\n", encoding="utf-8")
            ignored_analysis.write_text("# frozen analysis\n", encoding="utf-8")
            write_json(
                experiment / "freeze_manifest.json",
                {
                    "frozen_files_sha256": {
                        "analysis/random_support_analysis.py": sha256(ignored_analysis),
                        "code/src/metacog_lite_v0/weights.py": sha256(weights),
                    }
                },
            )
            (repo / ".gitignore").write_text(
                "experiment_FPP/ddpg_saber_random_support_count_matched_ls512/analysis/\n",
                encoding="utf-8",
            )
            for args in (
                ("init",),
                ("config", "user.email", "artifact-test@example.invalid"),
                ("config", "user.name", "Artifact Test"),
                ("add", "."),
                ("commit", "-m", "initial frozen source"),
            ):
                subprocess.run(
                    ["git", *args], cwd=repo, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
            first_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                stdout=subprocess.PIPE, text=True,
            ).stdout.strip()
            with self.assertRaisesRegex(BuildError, "not tracked"):
                verify_freeze_against_commit(repo, experiment, first_commit)

            subprocess.run(
                ["git", "add", "-f", str(ignored_analysis.relative_to(repo))],
                cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "force-add ignored analysis"],
                cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            source_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                stdout=subprocess.PIPE, text=True,
            ).stdout.strip()
            verified = verify_freeze_against_commit(
                repo, experiment, source_commit
            )
            self.assertEqual(
                set(verified),
                {
                    "analysis/random_support_analysis.py",
                    "code/src/metacog_lite_v0/weights.py",
                },
            )

            ignored_analysis.write_text(
                "# post-commit drift\n", encoding="utf-8"
            )
            manifest = json.loads(
                (experiment / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            manifest["frozen_files_sha256"][
                "analysis/random_support_analysis.py"
            ] = sha256(ignored_analysis)
            write_json(experiment / "freeze_manifest.json", manifest)
            with self.assertRaisesRegex(BuildError, "commit bytes differ"):
                verify_freeze_against_commit(repo, experiment, source_commit)

    def test_build_sanitize_recompute_and_standalone_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = build_synthetic(Path(temporary))
            completed = run_standalone(artifact)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"status": "verified"', completed.stdout)
            manifest = json.loads(
                (artifact / "input_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["methods"]["control"]["runs"][0]["packaged_run_dir"],
                "data/control/seed_1",
            )
            smoke_text = (
                artifact / "technical_gate/smoke_validation.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(Path(temporary)), smoke_text)
            self.assertIn("$" + "{REPO_ROOT}", smoke_text)
            packaged_summary = json.loads(
                (
                    artifact / "data/control/seed_1/summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsNone(
                packaged_summary["final_lite_metrics"][
                    "b3/support_to_success_corr"
                ]
            )
            validation = json.loads(
                (artifact / "results/validation.json").read_text(encoding="utf-8")
            )
            differences = validation["config_differences_by_seed"]["1"]
            by_field = {item["field"]: item for item in differences}
            self.assertEqual(
                set(by_field),
                {
                    "assignment_diagnostic_window_updates",
                    "output_root",
                    "run_name",
                    "seeds.assignment_seed",
                    "support_assignment_mode",
                },
            )
            self.assertIsNone(by_field["seeds.assignment_seed"]["main"])
            self.assertEqual(by_field["seeds.assignment_seed"]["control"], 30_001)
            self.assertEqual(
                validation["control_bookkeeping_by_seed"]["1"],
                {
                    "assignment_seed": 30_001,
                    "assignment_diagnostic_window_updates": 1_000,
                },
            )

    def test_hash_tamper_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = build_synthetic(Path(temporary))
            tampered = Path(temporary) / "tampered"
            shutil.copytree(artifact, tampered)
            target = tampered / "README.md"
            target.write_text(target.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
            failed = run_standalone(tampered)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("mismatch", failed.stderr.lower())

            linked = Path(temporary) / "linked"
            shutil.copytree(artifact, linked)
            (linked / "tests/forbidden-link").symlink_to(linked / "README.md")
            failed = run_standalone(linked)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("symlink", failed.stderr.lower())

    def test_builder_is_atomic_on_provenance_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, main_root, control_root, experiment, execution = prepare_inputs(root)
            artifact = root / "artifact"
            artifact.mkdir()
            marker = artifact / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            bad = dict(execution)
            bad["source_commit"] = "c" * 40
            with self.assertRaisesRegex(Exception, "metadata"):
                build_artifact(
                    repo=repo,
                    main_root=main_root,
                    control_root=control_root,
                    artifact_root=artifact,
                    experiment_root=experiment,
                    analysis_root=Path(__file__).resolve().parent,
                    execution_source_commit=SOURCE_COMMIT,
                    canonical_source_commit=CANONICAL_COMMIT,
                    execution_metadata_path=control_root / "_execution/launch_metadata.json",
                    execution_metadata=bad,
                    replace=True,
                    run_tests=False,
                    allow_synthetic_verification=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
