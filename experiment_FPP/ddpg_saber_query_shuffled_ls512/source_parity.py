#!/usr/bin/env python3
"""Check protected DDPG implementation files against the canonical Main source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MUTABLE_CORE_FILES = {"config.py", "trainer.py", "weights.py"}
ALLOWED_QUERY_ENTRY_POINTS = (
    "code/src/metacog_lite_v0/config.py",
    "code/src/metacog_lite_v0/trainer.py",
    "code/src/metacog_lite_v0/weights.py",
    "code/src/train_fetch_pick_and_place_lite_v0.py",
    "config.json",
    "run.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_paths(source_root: Path) -> list[Path]:
    module_root = source_root / "code" / "src" / "metacog_lite_v0"
    if not module_root.is_dir():
        raise FileNotFoundError(module_root)
    return sorted(
        path.relative_to(source_root)
        for path in module_root.glob("*.py")
        if path.name not in MUTABLE_CORE_FILES
    )


def build_report(canonical_root: Path, intervention_root: Path, output: Path | None = None) -> dict[str, object]:
    canonical_root = canonical_root.resolve()
    intervention_root = intervention_root.resolve()
    canonical_paths = protected_paths(canonical_root)
    intervention_paths = protected_paths(intervention_root)
    errors: list[str] = []
    if canonical_paths != intervention_paths:
        missing = sorted(path.as_posix() for path in set(canonical_paths) - set(intervention_paths))
        extra = sorted(path.as_posix() for path in set(intervention_paths) - set(canonical_paths))
        errors.append(f"protected path coverage differs: missing={missing}, extra={extra}")
    records: list[dict[str, object]] = []
    for relative in sorted(set(canonical_paths) & set(intervention_paths)):
        canonical = canonical_root / relative
        intervention = intervention_root / relative
        canonical_sha = sha256(canonical)
        intervention_sha = sha256(intervention)
        matches = canonical_sha == intervention_sha
        if not matches:
            errors.append(f"protected source differs: {relative.as_posix()}")
        records.append(
            {
                "path": relative.as_posix(),
                "canonical_sha256": canonical_sha,
                "intervention_sha256": intervention_sha,
                "byte_identical": matches,
            }
        )
    report = {
        "passed": not errors,
        "definition": "Protected DDPG agent/network/replay/scanner/task/environment files are byte-identical to canonical Main.",
        "mutable_core_files": sorted(MUTABLE_CORE_FILES),
        "allowed_query_entry_points": list(ALLOWED_QUERY_ENTRY_POINTS),
        "protected_file_count": len(records),
        "protected_files": records,
        "errors": errors,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"source parity failed: {errors}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.canonical_root, args.intervention_root, args.output)
    print(json.dumps({"passed": report["passed"], "protected_file_count": report["protected_file_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
