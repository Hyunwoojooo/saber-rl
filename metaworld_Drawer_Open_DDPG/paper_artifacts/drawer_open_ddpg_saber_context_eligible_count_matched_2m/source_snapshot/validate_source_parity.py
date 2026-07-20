#!/usr/bin/env python3
"""Prove that all non-control DrawerOpen learning components remain byte-identical."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DRAWER_ROOT = EXPERIMENT_DIR.parent
ORIGINAL_SRC = DRAWER_ROOT / "code" / "src"
CONTROL_SRC = EXPERIMENT_DIR / "code" / "src"

# These files define the environment, DDPG networks/optimizer/loss, replay/HER,
# reward recomputation inputs, task bins, and output conventions. They must be
# byte-identical to the matched Main source.
EXACT_FILES = (
    "metacog_lite_v0/agent.py",
    "metacog_lite_v0/networks.py",
    "metacog_lite_v0/replay.py",
    "metacog_lite_v0/envs.py",
    "metacog_lite_v0/task_drawer_open.py",
    "metacog_lite_v0/io_utils.py",
)

# Changes are isolated to strict assignment/config/logging/checkpoint support.
CONTROL_FILES = (
    "train_drawer_open_lite_v0.py",
    "metacog_lite_v0/config.py",
    "metacog_lite_v0/weights.py",
    "metacog_lite_v0/numba_support_scan_drawer_open.py",
    "metacog_lite_v0/trainer.py",
    "metacog_lite_v0/assignment_diagnostics.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report() -> dict[str, object]:
    exact = []
    failures = []
    for relative in EXACT_FILES:
        original = ORIGINAL_SRC / relative
        control = CONTROL_SRC / relative
        original_hash = sha256(original)
        control_hash = sha256(control)
        matched = original_hash == control_hash
        row = {
            "path": relative,
            "original_sha256": original_hash,
            "control_sha256": control_hash,
            "byte_identical": matched,
        }
        exact.append(row)
        if not matched:
            failures.append(relative)
    controlled = []
    for relative in CONTROL_FILES:
        path = CONTROL_SRC / relative
        if not path.is_file():
            failures.append(relative)
            continue
        controlled.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "exact_files": exact,
        "controlled_change_files": controlled,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "source_parity.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"source parity failed: {report['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
