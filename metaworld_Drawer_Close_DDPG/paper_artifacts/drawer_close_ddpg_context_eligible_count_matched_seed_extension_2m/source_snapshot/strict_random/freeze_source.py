#!/usr/bin/env python3
"""Freeze and verify the exact strict-control source used by queued runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def build_manifest() -> dict[str, object]:
    entries = []
    for path in source_files():
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M",
        "source_commit": git_head(),
        "files": entries,
    }


def verify(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in manifest["files"]}
    actual_paths = {path.relative_to(ROOT).as_posix(): path for path in source_files()}
    if set(expected) != set(actual_paths):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise RuntimeError(f"frozen source set changed: missing={missing}, extra={extra}")
    failures = []
    for relative, row in expected.items():
        path = actual_paths[relative]
        if path.stat().st_size != int(row["size"]) or sha256(path) != row["sha256"]:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"frozen source content changed: {failures}")
    print(f"source freeze verified: {len(expected)} files, commit={manifest['source_commit']}")


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
        print(f"wrote source freeze: {args.manifest} ({len(manifest['files'])} files)")
    else:
        verify(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
