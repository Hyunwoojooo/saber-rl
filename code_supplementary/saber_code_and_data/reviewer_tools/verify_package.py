#!/usr/bin/env python3
"""Verify top-level hashes, presets, internal artifacts, and anonymity."""

from __future__ import annotations

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


def run_tool(name: str, *args: str) -> list[str]:
    command = [sys.executable, str(ROOT / "reviewer_tools" / name), *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        return [f"{name} failed:\n{result.stdout}{result.stderr}"]
    print(result.stdout.strip())
    return []


def main() -> int:
    errors = verify_top_manifest()
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
