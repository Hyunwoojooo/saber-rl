#!/usr/bin/env python3
"""Fail-closed tests for the replay-query Git HEAD-bound source freeze."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import freeze_source


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE)


def _must_fail(callable_, *args, **kwargs) -> None:
    try:
        callable_(*args, **kwargs)
    except (RuntimeError, FileNotFoundError):
        return
    raise AssertionError("expected fail-closed freeze rejection")


def test_head_binding_contract() -> None:
    with tempfile.TemporaryDirectory() as temp:
        repo = Path(temp) / "repo"
        root = repo / "experiment"
        launcher_rel = Path("launcher.sh")
        repo.mkdir()
        root.mkdir()
        (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / launcher_rel).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        _git(repo, "init", "-q")
        _git(repo, "add", "experiment/source.py", launcher_rel.as_posix())
        _git(
            repo,
            "-c",
            "user.name=Freeze Test",
            "-c",
            "user.email=freeze@example.invalid",
            "commit",
            "-q",
            "-m",
            "freeze fixture",
        )

        manifest = freeze_source.build_manifest(root, repo, launcher_rel)
        assert manifest["schema_version"] == 2
        assert manifest["head_binding_verified"] is True
        assert manifest["recorded_commit_was_current_head_at_write"] is True
        assert len(manifest["files"]) == 1
        assert len(manifest["entrypoints"]) == 1
        assert manifest["files"][0]["head_bound"] is True
        manifest_path = repo / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        freeze_source.verify(manifest_path, root, repo, launcher_rel)

        (root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        _must_fail(freeze_source.build_manifest, root, repo, launcher_rel)
        _must_fail(freeze_source.verify, manifest_path, root, repo, launcher_rel)

        (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
        _must_fail(freeze_source.build_manifest, root, repo, launcher_rel)
        (root / "untracked.py").unlink()

        (repo / launcher_rel).write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        _must_fail(freeze_source.build_manifest, root, repo, launcher_rel)
        _must_fail(freeze_source.verify, manifest_path, root, repo, launcher_rel)


def main() -> int:
    test_head_binding_contract()
    print("test_freeze_source_head_binding.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
