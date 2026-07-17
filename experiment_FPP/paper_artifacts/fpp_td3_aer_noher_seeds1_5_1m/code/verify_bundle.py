#!/usr/bin/env python3
"""Verify the self-contained FPP TD3 AER three-arm result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ARTIFACT_ID = "fpp_td3_aer_noher_seeds1_5_1m"
SOURCE_COMMIT = "8c53c789f6f5cb6a45beceb4462a209cf6100d7e"
METHODS = ("td3_er", "td3_annealing_er", "td3_aer")
ANALYSIS_FILES = {
    "analysis_results.json",
    "analysis_summary.md",
    "mean_curves.csv",
    "method_summary.csv",
    "pairwise_summary.csv",
    "per_seed_results.csv",
}
FORBIDDEN_NAMES = {
    "checkpoint.pt",
    "best_checkpoint.pt",
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
}


class VerificationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise VerificationError(f"unsafe checksum path: {name!r}")
    return path


def verify_inventory(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise VerificationError("missing SHA256SUMS")
    expected: dict[str, str] = {}
    for number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise VerificationError(f"invalid SHA256SUMS line {number}")
        name = match.group(2)
        safe_relative(name)
        if name in expected:
            raise VerificationError(f"duplicate SHA256SUMS path: {name}")
        expected[name] = match.group(1)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    if set(expected) != actual:
        raise VerificationError(
            f"checksum inventory mismatch: missing={sorted(actual-set(expected))}, "
            f"extra={sorted(set(expected)-actual)}"
        )
    for name, digest in expected.items():
        path = root / safe_relative(name)
        if path.is_symlink() or sha256(path) != digest:
            raise VerificationError(f"checksum mismatch or symlink: {name}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VerificationError(f"symlink not allowed: {path}")
        if path.is_file() and (path.name in FORBIDDEN_NAMES or path.suffix == ".log"):
            raise VerificationError(f"forbidden raw/large file: {path}")


def verify_manifest(root: Path) -> None:
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0",
        "artifact_id": ARTIFACT_ID,
        "source_commit_full": SOURCE_COMMIT,
        "methods": list(METHODS),
        "seeds": [1, 2, 3, 4, 5],
        "total_steps_per_run": 1_000_000,
        "all_15_runs_complete": True,
        "her_ratio": 0.0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise VerificationError(f"bundle manifest mismatch: {key}")


def verify_arms(root: Path) -> None:
    arms = root / "arms"
    actual = {path.name for path in arms.iterdir() if path.is_dir()}
    if actual != set(METHODS):
        raise VerificationError(f"arm set mismatch: {sorted(actual)}")
    for method in METHODS:
        arm = arms / method
        verifier = arm / "code" / "verify_artifact.py"
        completed = subprocess.run(
            [sys.executable, str(verifier), str(arm)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise VerificationError(
                f"nested {method} verification failed:\n{completed.stdout}"
            )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise VerificationError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def compare_analysis(stored: Path, reproduced: Path) -> None:
    exact = (
        "method_summary.csv",
        "pairwise_summary.csv",
        "mean_curves.csv",
        "analysis_summary.md",
    )
    for name in exact:
        if (stored / name).read_bytes() != (reproduced / name).read_bytes():
            raise VerificationError(f"reproduced analysis differs: {name}")

    stored_fields, stored_rows = read_csv(stored / "per_seed_results.csv")
    reproduced_fields, reproduced_rows = read_csv(reproduced / "per_seed_results.csv")
    if stored_fields != reproduced_fields or len(stored_rows) != len(reproduced_rows):
        raise VerificationError("per-seed analysis schema/count mismatch")
    value_fields = [name for name in stored_fields if not name.endswith("_run_dir")]
    for index, (left, right) in enumerate(zip(stored_rows, reproduced_rows), 1):
        if {key: left[key] for key in value_fields} != {key: right[key] for key in value_fields}:
            raise VerificationError(f"per-seed numerical mismatch at row {index}")

    left: dict[str, Any] = json.loads((stored / "analysis_results.json").read_text())
    right: dict[str, Any] = json.loads((reproduced / "analysis_results.json").read_text())
    left.pop("source_runs", None)
    right.pop("source_runs", None)
    if left != right:
        raise VerificationError("analysis_results.json numerical/config mismatch")


def verify_analysis(root: Path) -> None:
    stored = root / "analysis"
    if {path.name for path in stored.iterdir() if path.is_file()} != ANALYSIS_FILES:
        raise VerificationError("analysis file set mismatch")
    analyzer = root / "code" / "analyze_td3_aer_noher.py"
    with tempfile.TemporaryDirectory(prefix="verify_fpp_td3_aer_") as temp:
        reproduced = Path(temp)
        command = [
            sys.executable,
            str(analyzer),
            "--td3-er-root",
            str(root / "arms/td3_er/data/td3_er"),
            "--td3-annealing-er-root",
            str(root / "arms/td3_annealing_er/data/td3_annealing_er"),
            "--td3-aer-root",
            str(root / "arms/td3_aer/data/td3_aer"),
            "--seeds",
            "1-5",
            "--output-dir",
            str(reproduced),
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise VerificationError(f"analysis reproduction failed:\n{completed.stdout}")
        compare_analysis(stored, reproduced)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    try:
        root = args.artifact_root.resolve()
        verify_inventory(root)
        verify_manifest(root)
        verify_arms(root)
        verify_analysis(root)
    except (VerificationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[bundle-verify-fail] {exc}", file=sys.stderr)
        return 1
    print(f"[bundle-verify-pass] {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
