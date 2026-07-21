#!/usr/bin/env python3
"""Deterministically fingerprint the complete strict-control source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
ALGORITHM = "sha256"
SCOPE = "full_experiment_package"
STAGES = ("smoke", "gate", "production", "freeze")
EXCLUDED_NAMES = frozenset({"freeze_manifest.json"})
EXCLUDED_DIR_NAMES = frozenset({"__pycache__"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".nbi", ".nbc"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class FingerprintError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(relative: Path) -> bool:
    return (
        relative.name in EXCLUDED_NAMES
        or any(part in EXCLUDED_DIR_NAMES for part in relative.parts)
        or relative.suffix in EXCLUDED_SUFFIXES
    )


def collect_source_hashes(root: Path = HERE) -> dict[str, str]:
    root = root.resolve()
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if excluded(relative):
            continue
        if path.is_symlink():
            raise FingerprintError(f"source symlink is forbidden: {relative.as_posix()}")
        if path.is_file():
            output[relative.as_posix()] = sha256(path)
    if not output:
        raise FingerprintError(f"source package is empty: {root}")
    return output


def tree_sha256(files_sha256: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(files_sha256), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"strict-source-fingerprint-v1\0" + canonical).hexdigest()


def make_fingerprint(stage: str, root: Path = HERE) -> dict[str, Any]:
    if stage not in STAGES:
        raise FingerprintError(f"invalid fingerprint stage: {stage!r}")
    files = collect_source_hashes(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "scope": SCOPE,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exclusions": {
            "names": sorted(EXCLUDED_NAMES),
            "directory_names": sorted(EXCLUDED_DIR_NAMES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "file_count": len(files),
        "tree_sha256": tree_sha256(files),
        "files_sha256": files,
    }


def validate_fingerprint_payload(
    payload: Mapping[str, Any], *, expected_stage: str | None = None
) -> dict[str, str]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "scope": SCOPE,
        "exclusions": {
            "names": sorted(EXCLUDED_NAMES),
            "directory_names": sorted(EXCLUDED_DIR_NAMES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FingerprintError(f"fingerprint {key} mismatch")
    stage = payload.get("stage")
    if stage not in STAGES or (expected_stage is not None and stage != expected_stage):
        raise FingerprintError(
            f"fingerprint stage mismatch: expected {expected_stage!r}, got {stage!r}"
        )
    if not isinstance(payload.get("created_utc"), str) or not payload["created_utc"]:
        raise FingerprintError("fingerprint creation time is missing")
    raw_files = payload.get("files_sha256")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise FingerprintError("fingerprint file mapping is missing")
    files: dict[str, str] = {}
    for raw_relative, digest in raw_files.items():
        if not isinstance(raw_relative, str) or not raw_relative:
            raise FingerprintError(f"invalid fingerprint path: {raw_relative!r}")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_relative
            or any(part in {"", ".", ".."} for part in relative.parts)
            or excluded(relative)
        ):
            raise FingerprintError(f"unsafe/excluded fingerprint path: {raw_relative!r}")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise FingerprintError(f"invalid fingerprint digest: {raw_relative}")
        files[raw_relative] = digest
    if payload.get("file_count") != len(files):
        raise FingerprintError("fingerprint file count mismatch")
    if payload.get("tree_sha256") != tree_sha256(files):
        raise FingerprintError("fingerprint tree digest mismatch")
    return files


def load_fingerprint(path: Path, *, expected_stage: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FingerprintError(f"invalid fingerprint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FingerprintError(f"fingerprint is not an object: {path}")
    return payload, validate_fingerprint_payload(payload, expected_stage=expected_stage)


def verify_current(
    payload: Mapping[str, Any], *, root: Path = HERE,
    expected_stage: str | None = None,
) -> dict[str, str]:
    recorded = validate_fingerprint_payload(payload, expected_stage=expected_stage)
    current = collect_source_hashes(root)
    if current != recorded:
        added = sorted(set(current) - set(recorded))
        removed = sorted(set(recorded) - set(current))
        changed = sorted(
            key for key in set(current) & set(recorded)
            if current[key] != recorded[key]
        )
        raise FingerprintError(
            f"source fingerprint drift: added={added}, removed={removed}, changed={changed}"
        )
    return recorded


def find_repo_root(start: Path = HERE) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "experiment_FPP").is_dir():
            return candidate
    raise FingerprintError(f"repository root not found from {start}")


def git_bytes(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FingerprintError(detail or f"git {' '.join(args)} failed")
    return completed.stdout


def verify_freeze_in_git(
    freeze_manifest: Path, commitish: str, *, root: Path = HERE,
    repo: Path | None = None,
) -> tuple[str, dict[str, str]]:
    """Verify every frozen path is current, tracked, and byte-identical in Git."""
    repo = (repo or find_repo_root(root)).resolve()
    root = root.resolve()
    try:
        package_relative = root.relative_to(repo)
    except ValueError as exc:
        raise FingerprintError("source package is outside the repository") from exc
    commit = git_bytes(repo, "rev-parse", f"{commitish}^{{commit}}").decode().strip()
    if not HEX40.fullmatch(commit):
        raise FingerprintError(f"invalid resolved commit: {commit!r}")
    try:
        freeze = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FingerprintError(f"invalid freeze manifest {freeze_manifest}: {exc}") from exc
    if not isinstance(freeze, Mapping):
        raise FingerprintError("freeze manifest is not an object")
    raw_hashes = freeze.get("frozen_files_sha256")
    if not isinstance(raw_hashes, Mapping) or not raw_hashes:
        raise FingerprintError("freeze manifest has no frozen file mapping")
    hashes: dict[str, str] = {}
    for relative, digest in raw_hashes.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise FingerprintError("invalid freeze manifest mapping")
        hashes[relative] = digest
    # Reuse the strict schema/path/digest validation by constructing the
    # deterministic portion of a fingerprint payload.
    validate_fingerprint_payload({
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "scope": SCOPE,
        "stage": "freeze",
        "created_utc": "git-preflight",
        "exclusions": {
            "names": sorted(EXCLUDED_NAMES),
            "directory_names": sorted(EXCLUDED_DIR_NAMES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "file_count": len(hashes),
        "tree_sha256": tree_sha256(hashes),
        "files_sha256": hashes,
    }, expected_stage="freeze")
    current = collect_source_hashes(root)
    if current != hashes:
        raise FingerprintError("current package differs from frozen file mapping")
    for relative, expected_hash in sorted(hashes.items()):
        object_spec = f"{commit}:{(package_relative / relative).as_posix()}"
        git_bytes(repo, "cat-file", "-e", object_spec)
        committed = git_bytes(repo, "show", object_spec)
        if hashlib.sha256(committed).hexdigest() != expected_hash:
            raise FingerprintError(
                f"Git blob differs from frozen SHA256: {relative}"
            )
    return commit, hashes


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--stage", choices=STAGES, required=True)
    write.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--fingerprint", type=Path, required=True)
    verify.add_argument("--expected-stage", choices=STAGES, required=True)
    verify_git = subparsers.add_parser("verify-git")
    verify_git.add_argument("--freeze-manifest", type=Path, required=True)
    verify_git.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        if args.command == "write":
            payload = make_fingerprint(args.stage)
            write_atomic(args.output.resolve(), payload)
            print(json.dumps({
                "path": str(args.output.resolve()),
                "stage": args.stage,
                "file_count": payload["file_count"],
                "tree_sha256": payload["tree_sha256"],
            }, sort_keys=True))
        elif args.command == "verify":
            payload, _ = load_fingerprint(
                args.fingerprint.resolve(), expected_stage=args.expected_stage
            )
            verify_current(payload, expected_stage=args.expected_stage)
            print(json.dumps({
                "path": str(args.fingerprint.resolve()),
                "stage": args.expected_stage,
                "file_count": payload["file_count"],
                "tree_sha256": payload["tree_sha256"],
                "verified": True,
            }, sort_keys=True))
        else:
            commit, hashes = verify_freeze_in_git(
                args.freeze_manifest.resolve(), args.commit
            )
            print(json.dumps({
                "commit": commit,
                "file_count": len(hashes),
                "freeze_manifest": str(args.freeze_manifest.resolve()),
                "tree_sha256": tree_sha256(hashes),
                "verified": True,
            }, sort_keys=True))
    except (FingerprintError, OSError) as exc:
        print(f"source fingerprint failed: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
