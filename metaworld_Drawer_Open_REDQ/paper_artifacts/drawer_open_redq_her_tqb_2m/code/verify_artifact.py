#!/usr/bin/env python3
"""Verify hashes, exact evaluation grids, and aggregate DrawerOpen metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


EVAL_GRID = tuple(range(10_000, 2_000_001, 10_000))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _close(left: float, right: float, context: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context}: {left} != {right}")


def _verify_manifest(root: Path) -> None:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    payload = manifest["payload"]
    if int(manifest["payload_file_count"]) != len(payload):
        raise ValueError("Manifest file count mismatch")
    for item in payload:
        path = root / item["path"]
        data = path.read_bytes()
        if len(data) != int(item["size_bytes"]):
            raise ValueError(f"Size mismatch: {path}")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")


def _verify_grids(root: Path) -> None:
    files = sorted((root / "data").glob("*/metrics_eval_10k.csv"))
    if not files:
        raise ValueError("No exact-grid metric files")
    for path in files:
        rows = _read_csv(path)
        steps = tuple(int(float(row["step"])) for row in rows)
        if steps != EVAL_GRID:
            raise ValueError(f"Evaluation grid mismatch: {path}")
        if any(not row.get("eval/env_success_mean", "") for row in rows):
            raise ValueError(f"Missing evaluation value: {path}")


def _verify_aggregates(root: Path) -> None:
    per_seed = _read_csv(root / "per_seed_performance.csv")
    aggregates = _read_csv(root / "aggregate_performance.csv")
    for aggregate in aggregates:
        seeds = [int(value) for value in aggregate["seeds"].split(",")]
        rows = [row for row in per_seed if int(row["seed"]) in seeds]
        if len(rows) != len(seeds):
            raise ValueError(f"Missing per-seed rows for {aggregate['cohort']}")
        for source, mean_key, std_key in (
            ("final_success", "final_success_mean", "final_success_sample_std"),
            ("best_success", "best_success_mean", "best_success_sample_std"),
            ("success_auc", "success_auc_mean", "success_auc_sample_std"),
            ("best_final_drop", "best_final_drop_mean", "best_final_drop_sample_std"),
        ):
            values = [float(row[source]) for row in rows]
            _close(statistics.fmean(values), float(aggregate[mean_key]), mean_key)
            _close(_sample_std(values), float(aggregate[std_key]), std_key)
        hits = [int(row["first_hit_0.90"]) for row in rows if row["hit90_reached"] == "True"]
        if len(hits) != int(aggregate["hit90_count"]):
            raise ValueError(f"Hit90 count mismatch for {aggregate['cohort']}")
        if hits:
            _close(
                float(statistics.median(hits)),
                float(aggregate["hit90_median_step_among_hits"]),
                "Hit90 median",
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _verify_manifest(root)
    _verify_grids(root)
    _verify_aggregates(root)
    print(f"OK: {root.name}")


if __name__ == "__main__":
    main()
