#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_module("analyze_results", HERE / "analyze_results.py")
package = load_module("ddpg_shuffled_package_test", HERE / "package_artifact.py")
fixtures = load_module("ddpg_shuffled_fixture_test", HERE / "test_analyze_results.py")


def init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "artifact-test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Artifact Test"], cwd=root, check=True)
    marker = root / "marker.txt"
    marker.write_text("synthetic\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "synthetic"], cwd=root, check=True)


def test_compact_metrics_primary_rows_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = fixtures.write_run(root, "shuffled", 1)
        source = run / "metrics_selected_10k.csv"
        destination = root / "compact.csv"
        assert package.compact_metrics(source, destination) == 100
        rows = analysis.evaluation_rows(destination)
        assert list(rows) == list(analysis.EVAL_STEPS)
        assert sum(1 for _ in destination.open(encoding="utf-8")) == 101


def test_end_to_end_package_and_standalone_verifier() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        init_repo(repo)
        canonical_modules = (
            HERE.parent / "b3_critic_only_runtime_opt_v4_ls512" / "code" / "src" / "metacog_lite_v0"
        )
        packaged_canonical_modules = repo / package.MAIN_SOURCE_REL / "code" / "src" / "metacog_lite_v0"
        packaged_canonical_modules.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(canonical_modules, packaged_canonical_modules)
        main_root, shuffled_root = fixtures.build_synthetic_roots(repo)
        artifact = repo / "paper_artifact"
        commit = "1" * 40
        package.build_artifact(
            repo,
            main_root,
            shuffled_root,
            artifact,
            execution_source_commit=commit,
            source_root=HERE,
        )
        assert (artifact / "artifact_manifest.json").is_file()
        assert (artifact / "frozen_source.sha256").is_file()
        assert len(list((artifact / "data" / "main").iterdir())) == 5
        assert len(list((artifact / "data" / "shuffled").iterdir())) == 5
        process = subprocess.run(
            [sys.executable, str(artifact / "verify_artifact.py"), "--artifact-dir", str(artifact)],
            cwd=artifact,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode:
            raise AssertionError(f"verifier failed\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}")
        result = json.loads(process.stdout)
        assert result["status"] == "PASS"
        assert result["production_run_count"] == 10
        assert result["evaluation_points_per_run"] == 100
        assert result["bootstrap_assignments"] == 10_000
        assert result["sign_flip_assignments"] == 32


def main() -> int:
    test_compact_metrics_primary_rows_only()
    test_end_to_end_package_and_standalone_verifier()
    print("test_artifact_tooling.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
