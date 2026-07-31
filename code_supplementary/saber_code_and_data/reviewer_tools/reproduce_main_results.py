#!/usr/bin/env python3
"""Recompute primary aggregate and paired statistics from included seed rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/derived/primary_seed_results"
RESAMPLES = 10_000
BOOTSTRAP_SEED = 2027
TOLERANCE = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap(differences: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(differences)
    means = [
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(RESAMPLES)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)


def require_close(errors: list[str], label: str, left: float, right: float) -> None:
    if not close(left, right):
        errors.append(f"{label}: recomputed={left:.17g}, stored={right:.17g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    per_seed = read_csv(DATA / "per_seed_metrics.csv")
    aggregate = read_csv(DATA / "aggregate_metrics.csv")
    paired = read_csv(DATA / "paired_statistics.csv")
    errors: list[str] = []

    rows_by_group: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    rows_by_seed: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in per_seed:
        group = (row["task"], row["backbone"], row["method"])
        rows_by_group.setdefault(group, []).append(row)
        rows_by_seed[(*group, int(row["seed"]))] = row

    recomputed_aggregate: list[dict[str, object]] = []
    for stored in aggregate:
        group = (stored["task"], stored["backbone"], stored["method"])
        rows = rows_by_group[group]
        item: dict[str, object] = {
            "task": group[0],
            "backbone": group[1],
            "method": group[2],
            "n_seeds": len(rows),
        }
        if len(rows) != int(stored["n_seeds"]):
            errors.append(f"{group}: n_seeds mismatch")
        for metric, prefix in (
            ("auc", "auc"),
            ("final_quarter_auc", "final_quarter_auc"),
            ("final_success", "final_success"),
        ):
            values = [float(row[metric]) for row in rows]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values)
            item[f"{prefix}_mean"] = mean
            item[f"{prefix}_sample_sd"] = sd
            require_close(errors, f"{group} {metric} mean", mean, float(stored[f"{prefix}_mean"]))
            require_close(
                errors,
                f"{group} {metric} sample_sd",
                sd,
                float(stored[f"{prefix}_sample_sd"]),
            )
        hits = [int(row["hit90_step"]) for row in rows if row["hit90_step"]]
        item["hit90_reached"] = len(hits)
        item["hit90_total"] = len(rows)
        item["hit90_median_reached"] = statistics.median(hits) if hits else None
        if len(hits) != int(stored["hit90_reached"]):
            errors.append(f"{group}: hit90 count mismatch")
        recomputed_aggregate.append(item)

    recomputed_pairs: list[dict[str, object]] = []
    for stored in paired:
        task = stored["task"]
        backbone = stored["backbone"]
        left_method, right_method = stored["comparison"].split("-", maxsplit=1)
        metric = stored["metric"]
        seeds = sorted(
            seed
            for (row_task, row_backbone, method, seed) in rows_by_seed
            if row_task == task and row_backbone == backbone and method == left_method
        )
        differences = [
            float(rows_by_seed[(task, backbone, left_method, seed)][metric])
            - float(rows_by_seed[(task, backbone, right_method, seed)][metric])
            for seed in seeds
        ]
        mean = statistics.fmean(differences)
        low, high = bootstrap(differences)
        positives = sum(value > 0.0 for value in differences)
        require_close(errors, f"{task}/{backbone}/{stored['comparison']} mean", mean, float(stored["mean_difference"]))
        require_close(errors, f"{task}/{backbone}/{stored['comparison']} ci_low", low, float(stored["ci95_low"]))
        require_close(errors, f"{task}/{backbone}/{stored['comparison']} ci_high", high, float(stored["ci95_high"]))
        if positives != int(stored["positive_pairs"]):
            errors.append(f"{task}/{backbone}/{stored['comparison']}: positive pair mismatch")
        recomputed_pairs.append(
            {
                "task": task,
                "backbone": backbone,
                "comparison": stored["comparison"],
                "metric": metric,
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": positives,
                "n_pairs": len(differences),
            }
        )

    report = {
        "seed_rows": len(per_seed),
        "aggregate_rows": len(aggregate),
        "paired_rows": len(paired),
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "primary_reproduction_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
