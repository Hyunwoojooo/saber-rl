#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EVAL_STEPS = list(range(10_000, 2_000_001, 10_000))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(steps: list[int], values: list[float]) -> float:
    area = sum((b - a) * (x + y) / 2 for a, b, x, y in zip(steps, steps[1:], values, values[1:]))
    return area / 1_990_000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for row in manifest["files"]:
        path = root / row["path"]
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise AssertionError(f"manifest mismatch: {row['path']}")
    per_seed = {}
    with (root / "per_seed_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method_key"] == "context":
                per_seed[int(row["seed"])] = float(row["auc"])
    if set(per_seed) != {1, 2, 3, 4, 5}:
        raise AssertionError("missing context seeds")
    for seed in range(1, 6):
        seed_dir = root / f"seed_{seed}"
        config = json.loads((seed_dir / "config.json").read_text())
        summary = json.loads((seed_dir / "summary.json").read_text())
        if config["support_query_mode"] != "context_count_only" or summary["support_query_mode"] != "context_count_only":
            raise AssertionError(f"seed {seed}: mode mismatch")
        with (seed_dir / "metrics_selected_10k.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        steps = [int(float(row["step"])) for row in rows]
        values = [float(row["eval/env_success_mean"]) for row in rows]
        if steps != EVAL_STEPS or len(values) != 200 or not all(math.isfinite(v) for v in values):
            raise AssertionError(f"seed {seed}: invalid exact grid")
        if abs(auc(steps, values) - per_seed[seed]) > 1e-12:
            raise AssertionError(f"seed {seed}: AUC mismatch")
        with (seed_dir / "context_diagnostics.csv").open(newline="") as handle:
            diagnostics = list(csv.DictReader(handle))
        if not diagnostics:
            raise AssertionError(f"seed {seed}: missing diagnostics")
        for row in diagnostics:
            if float(row["query/invariant_violation_count"]) != 0.0:
                raise AssertionError(f"seed {seed}: invariant violation")
            if float(row["query/action_membership_lookup_count"]) != 0.0:
                raise AssertionError(f"seed {seed}: action lookup")
            if abs(float(row["query/context_count_eligible_rate"]) - float(row["query/support_positive_rate"])) > 1e-7:
                raise AssertionError(f"seed {seed}: mask mismatch")
    print(json.dumps({"passed": True, "artifact": str(root), "files": len(manifest["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
