#!/usr/bin/env python3
"""Freeze source only when every production file is bound to a committed Git blob."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_MANIFEST = ROOT / "freeze_manifest.json"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "runs", "logs", "reports"}
EXCLUDED_NAMES = {
    "freeze_manifest.json",
    "config_parity.json",
    "source_parity.json",
    "checkpoint_resume_audit.json",
    "completion_validation.json",
}
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json"}
ROOT_LAUNCHER_REPO_PATH = Path(
    "metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Replay-Action-Query-2M.sh"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_source_relative(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.name in EXCLUDED_NAMES:
        return False
    return relative.suffix in SOURCE_SUFFIXES


def source_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and _is_source_relative(path.relative_to(root)):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _git(
    args: list[str],
    repo_root: Path,
    *,
    binary: bool = False,
    check: bool = True,
) -> str | bytes | subprocess.CompletedProcess[bytes]:
    if not check:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=not binary,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo_root}: {str(stderr).strip()}"
        ) from exc


def git_head(repo_root: Path = REPO_ROOT) -> str:
    return str(_git(["rev-parse", "HEAD"], repo_root)).strip()


def _repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"source path is outside repository: {path}") from exc


def _committed_source_paths(commit: str, root: Path, repo_root: Path) -> set[str]:
    root_repo_path = _repo_path(root, repo_root)
    listing = str(
        _git(
            ["ls-tree", "-r", "--name-only", commit, "--", root_repo_path],
            repo_root,
        )
    )
    paths: set[str] = set()
    for raw in listing.splitlines():
        repo_path = raw.strip()
        if not repo_path:
            continue
        relative = Path(repo_path).relative_to(Path(root_repo_path))
        if _is_source_relative(relative):
            paths.add(repo_path)
    return paths


def _head_bound_entry(
    path: Path,
    *,
    logical_path: str,
    commit: str,
    repo_root: Path,
    role: str,
) -> dict[str, Any]:
    repo_path = _repo_path(path, repo_root)
    working = path.read_bytes()
    committed = _git(["show", f"{commit}:{repo_path}"], repo_root, binary=True)
    if not isinstance(committed, bytes):
        raise AssertionError("binary git show unexpectedly returned text")
    if working != committed:
        raise RuntimeError(
            f"source is not byte-identical to HEAD {commit}: {repo_path}"
        )
    digest = sha256_bytes(working)
    blob_oid = str(
        _git(["rev-parse", f"{commit}:{repo_path}"], repo_root)
    ).strip()
    return {
        "path": logical_path,
        "repo_path": repo_path,
        "role": role,
        "size": len(working),
        "sha256": digest,
        "head_sha256": digest,
        "git_blob_oid": blob_oid,
        "head_bound": True,
    }


def build_manifest(
    root: Path = ROOT,
    repo_root: Path = REPO_ROOT,
    launcher_repo_path: Path = ROOT_LAUNCHER_REPO_PATH,
) -> dict[str, object]:
    commit = git_head(repo_root)
    paths = source_files(root)
    current_repo_paths = {_repo_path(path, repo_root) for path in paths}
    committed_repo_paths = _committed_source_paths(commit, root, repo_root)
    if current_repo_paths != committed_repo_paths:
        missing = sorted(committed_repo_paths - current_repo_paths)
        uncommitted = sorted(current_repo_paths - committed_repo_paths)
        raise RuntimeError(
            "source set is not identical to HEAD; "
            f"missing_from_worktree={missing}, untracked_or_uncommitted={uncommitted}"
        )

    entries = [
        _head_bound_entry(
            path,
            logical_path=path.relative_to(root).as_posix(),
            commit=commit,
            repo_root=repo_root,
            role="experiment_source",
        )
        for path in paths
    ]
    launcher = repo_root / launcher_repo_path
    if not launcher.is_file():
        raise FileNotFoundError(f"root launcher is missing: {launcher}")
    entrypoints = [
        _head_bound_entry(
            launcher,
            logical_path=launcher_repo_path.as_posix(),
            commit=commit,
            repo_root=repo_root,
            role="root_launcher",
        )
    ]
    return {
        "schema_version": 2,
        "experiment": "DDPG-HER-B3-Replay-Action-Query-2M",
        "source_commit": commit,
        "head_binding_verified": True,
        "recorded_commit_was_current_head_at_write": True,
        "files": entries,
        "entrypoints": entrypoints,
    }


def _verify_entry(
    path: Path,
    row: dict[str, Any],
    *,
    commit: str,
    repo_root: Path,
) -> None:
    repo_path = _repo_path(path, repo_root)
    if repo_path != row.get("repo_path") or row.get("head_bound") is not True:
        raise RuntimeError(f"invalid Git binding metadata: {repo_path}")
    working = path.read_bytes()
    committed = _git(["show", f"{commit}:{repo_path}"], repo_root, binary=True)
    if not isinstance(committed, bytes):
        raise AssertionError("binary git show unexpectedly returned text")
    digest = sha256_bytes(working)
    committed_digest = sha256_bytes(committed)
    blob_oid = str(
        _git(["rev-parse", f"{commit}:{repo_path}"], repo_root)
    ).strip()
    if (
        working != committed
        or len(working) != int(row.get("size", -1))
        or digest != row.get("sha256")
        or committed_digest != row.get("head_sha256")
        or committed_digest != row.get("sha256")
        or blob_oid != row.get("git_blob_oid")
    ):
        raise RuntimeError(f"source or Git blob binding changed: {repo_path}")


def verify(
    manifest_path: Path,
    root: Path = ROOT,
    repo_root: Path = REPO_ROOT,
    launcher_repo_path: Path = ROOT_LAUNCHER_REPO_PATH,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("head_binding_verified") is not True
        or manifest.get("recorded_commit_was_current_head_at_write") is not True
    ):
        raise RuntimeError("source freeze lacks required Git HEAD binding")
    commit = str(manifest.get("source_commit", ""))
    ancestor = _git(
        ["merge-base", "--is-ancestor", commit, "HEAD"],
        repo_root,
        check=False,
    )
    if not isinstance(ancestor, subprocess.CompletedProcess) or ancestor.returncode != 0:
        raise RuntimeError(f"recorded source commit is not an ancestor of HEAD: {commit}")

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source freeze has no files")
    expected = {str(row.get("path", "")): row for row in rows if isinstance(row, dict)}
    if len(expected) != len(rows):
        raise RuntimeError("duplicate or malformed paths in source freeze manifest")
    actual_paths = {
        path.relative_to(root).as_posix(): path for path in source_files(root)
    }
    if set(expected) != set(actual_paths):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise RuntimeError(f"frozen source set changed: missing={missing}, extra={extra}")
    committed_paths = _committed_source_paths(commit, root, repo_root)
    manifest_repo_paths = {str(row.get("repo_path", "")) for row in expected.values()}
    if committed_paths != manifest_repo_paths:
        raise RuntimeError("manifest source set differs from its recorded Git commit")
    for relative, row in expected.items():
        _verify_entry(actual_paths[relative], row, commit=commit, repo_root=repo_root)

    entrypoints = manifest.get("entrypoints")
    if not isinstance(entrypoints, list) or len(entrypoints) != 1:
        raise RuntimeError("freeze manifest must bind exactly one root launcher")
    launcher_row = entrypoints[0]
    if (
        not isinstance(launcher_row, dict)
        or launcher_row.get("path") != launcher_repo_path.as_posix()
        or launcher_row.get("role") != "root_launcher"
    ):
        raise RuntimeError("root launcher binding metadata is invalid")
    launcher = repo_root / launcher_repo_path
    if not launcher.is_file():
        raise FileNotFoundError(f"root launcher is missing: {launcher}")
    _verify_entry(launcher, launcher_row, commit=commit, repo_root=repo_root)
    print(
        f"source freeze verified: {len(expected)} files + 1 launcher, "
        f"commit={commit}, head_bound=true"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write:
        if args.manifest.exists():
            raise FileExistsError(f"refusing to overwrite source freeze: {args.manifest}")
        manifest = build_manifest()
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote HEAD-bound source freeze: {args.manifest} "
            f"({len(manifest['files'])} files + 1 launcher)"
        )
    else:
        verify(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
