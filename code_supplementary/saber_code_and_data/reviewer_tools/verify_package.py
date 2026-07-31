#!/usr/bin/env python3
"""Verify top-level hashes, presets, internal artifacts, and anonymity."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_top_manifest() -> list[str]:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["MANIFEST.sha256 is missing"]
    errors: list[str] = []
    listed: set[Path] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split("  ", maxsplit=1)
        path = ROOT / relative
        listed.add(Path(relative))
        if not path.is_file():
            errors.append(f"{relative}: missing")
        elif sha256(path) != digest:
            errors.append(f"{relative}: sha256 mismatch")
    actual = {
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if (
            path.is_file()
            and path.name != "MANIFEST.sha256"
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
            and path.relative_to(ROOT).parts[0:1] != ("reviewer_runs",)
        )
    }
    for missing in sorted(actual - listed):
        errors.append(f"{missing}: absent from manifest")
    for stale in sorted(listed - actual):
        errors.append(f"{stale}: stale manifest entry")
    return errors


def package_path(value: str) -> tuple[Path | None, str | None]:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"unsafe package path: {value}"
    target = (ROOT / relative).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError:
        return None, f"package path escapes root: {value}"
    return target, None


def verify_source_snapshot_manifest() -> list[str]:
    path = ROOT / "provenance/source_snapshot_manifest.json"
    if not path.is_file():
        return ["provenance/source_snapshot_manifest.json is missing"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for snapshot in payload.get("source_snapshots", []):
        snapshot_id = str(snapshot.get("source_snapshot_id", ""))
        if not snapshot_id.startswith("snapshot_"):
            errors.append(f"invalid anonymous source_snapshot_id: {snapshot_id!r}")
        listed: set[Path] = set()
        for item in snapshot.get("files", []):
            relative = Path(str(item["path"]))
            listed.add(relative)
            target, error = package_path(relative.as_posix())
            if error:
                errors.append(error)
            elif target is None or not target.is_file():
                errors.append(f"{relative}: source snapshot file missing")
            elif sha256(target) != item["sha256"]:
                errors.append(f"{relative}: source snapshot sha256 mismatch")
        source_root, error = package_path(str(snapshot.get("package_path", "")))
        if error:
            errors.append(error)
        elif source_root is not None and source_root.is_dir():
            actual = {
                item.relative_to(ROOT)
                for item in source_root.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts
            }
            for missing in sorted(actual - listed):
                errors.append(f"{missing}: absent from source snapshot manifest")
            for stale in sorted(listed - actual):
                errors.append(f"{stale}: stale source snapshot entry")
    return errors


def verify_selection_manifests() -> list[str]:
    root = ROOT / "results/selected_artifacts"
    errors: list[str] = []
    for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = artifact_dir / "SELECTION_MANIFEST.json"
        if not manifest.is_file():
            errors.append(f"{artifact_dir.relative_to(ROOT)}: selection manifest missing")
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        listed: set[Path] = set()
        for item in payload.get("files", []):
            relative = Path(str(item["path"]))
            listed.add(relative)
            target = artifact_dir / relative
            if not target.is_file():
                errors.append(f"{target.relative_to(ROOT)}: selected file missing")
            elif target.stat().st_size != int(item["bytes"]):
                errors.append(f"{target.relative_to(ROOT)}: selected byte-size mismatch")
            elif sha256(target) != item["sha256"]:
                errors.append(f"{target.relative_to(ROOT)}: selected sha256 mismatch")
        actual = {
            item.relative_to(artifact_dir)
            for item in artifact_dir.rglob("*")
            if item.is_file() and item != manifest and "__pycache__" not in item.parts
        }
        for missing in sorted(actual - listed):
            errors.append(
                f"{artifact_dir.relative_to(ROOT) / missing}: absent from selection manifest"
            )
        for stale in sorted(listed - actual):
            errors.append(
                f"{artifact_dir.relative_to(ROOT) / stale}: stale selection entry"
            )
    return errors


def verify_evidence_paths() -> list[str]:
    tables = (
        (ROOT / "CLAIM_TO_ARTIFACT.csv", "evidence_path"),
        (
            ROOT / "results/derived/technical_supplement/artifact_index.csv",
            "package_relative_path",
        ),
    )
    errors: list[str] = []
    for csv_path, field in tables:
        if not csv_path.is_file():
            errors.append(f"{csv_path.relative_to(ROOT)} is missing")
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or field not in rows[0]:
            errors.append(f"{csv_path.relative_to(ROOT)}: missing {field} column")
            continue
        for row in rows:
            value = row[field]
            target, error = package_path(value)
            if error:
                errors.append(error)
            elif target is None or not target.exists():
                errors.append(f"{csv_path.relative_to(ROOT)}: missing path {value}")
    return errors


def run_tool(name: str, *args: str) -> list[str]:
    command = [sys.executable, str(ROOT / "reviewer_tools" / name), *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        return [f"{name} failed:\n{result.stdout}{result.stderr}"]
    print(result.stdout.strip())
    return []


def main() -> int:
    errors = verify_top_manifest()
    errors.extend(verify_source_snapshot_manifest())
    errors.extend(verify_selection_manifests())
    errors.extend(verify_evidence_paths())
    errors.extend(run_tool("audit_anonymity.py"))
    errors.extend(run_tool("launch_experiment.py", "--list"))
    errors.extend(run_tool("reproduce_main_results.py", "--check"))
    errors.extend(run_tool("reproduce_supplement_results.py", "--check"))

    package_manifest = ROOT / "provenance/package_manifest.json"
    if package_manifest.is_file():
        payload = json.loads(package_manifest.read_text(encoding="utf-8"))
        print(
            f"Package inventory: {payload.get('file_count')} files, "
            f"{payload.get('total_bytes')} bytes"
        )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Package verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
