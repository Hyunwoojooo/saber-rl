#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


GRID = tuple(range(10_000, 1_000_001, 10_000))
METHODS = ("td3_her", "td3_saber")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_012_027


def find_repo_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "version_v0_lite").is_dir() and (parent / "experiment_FPP").is_dir():
            return parent
    return Path.cwd()


def finite_float(value: str) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def load_curve(path: Path) -> list[float]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "eval/success_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns {required} in {path}")
        for row in reader:
            step_value = finite_float(row.get("step", ""))
            success = finite_float(row.get("eval/success_mean", ""))
            if step_value is None or success is None:
                continue
            step = int(step_value)
            if step not in GRID:
                raise ValueError(f"Unexpected evaluation step {step} in {path}")
            if step in values:
                raise ValueError(f"Duplicate evaluation step {step} in {path}")
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"Invalid success {success} at step {step} in {path}")
            values[step] = success
    missing = sorted(set(GRID) - set(values))
    if missing:
        raise ValueError(f"Missing evaluation steps in {path}: {missing}")
    return [values[step] for step in GRID]


def normalized_auc(curve: list[float]) -> float:
    if len(curve) != len(GRID):
        raise ValueError("AUC requires the frozen 100-point evaluation grid")
    area = sum(
        (GRID[index + 1] - GRID[index]) * (curve[index] + curve[index + 1]) / 2.0
        for index in range(len(GRID) - 1)
    )
    return area / float(GRID[-1] - GRID[0])


def first_hit(curve: list[float], threshold: float) -> int | None:
    return next((step for step, value in zip(GRID, curve) if value >= threshold), None)


def sample_std(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.stdev(data) if len(data) > 1 else 0.0


def quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(differences: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_sign_flip_p(differences: list[float]) -> float:
    n = len(differences)
    observed = abs(statistics.mean(differences))
    extreme = 0
    for mask in range(1 << n):
        statistic = abs(
            sum(value if mask & (1 << index) else -value for index, value in enumerate(differences)) / n
        )
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / float(1 << n)


def load_seed(artifact_root: Path, method: str, seed: int) -> dict[str, Any]:
    run_dir = artifact_root / "data" / method / str(seed)
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics_selected_10k.csv"
    config_path = run_dir / "config.json"
    for path in (summary_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    if summary.get("total_steps") != 1_000_000 or config.get("total_steps") != 1_000_000:
        raise ValueError(f"Incomplete run: {run_dir}")
    if config.get("seeds", {}).get("env_seed") != seed:
        raise ValueError(f"Seed mismatch: {run_dir}")
    curve = load_curve(metrics_path)
    auc = normalized_auc(curve)
    summary_auc = float(summary["eval_AUC_success"])
    if abs(auc - summary_auc) > 1e-9:
        raise ValueError(
            f"AUC mismatch for {run_dir}: curve={auc:.12f}, summary={summary_auc:.12f}"
        )
    return {
        "seed": seed,
        "curve": curve,
        "auc": auc,
        "final": curve[-1],
        "best": max(curve),
        "first_hit_0.90": first_hit(curve, 0.90),
    }


def aggregate(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    auc = [row["auc"] for row in rows]
    final = [row["final"] for row in rows]
    best = [row["best"] for row in rows]
    reached = [row["first_hit_0.90"] for row in rows if row["first_hit_0.90"] is not None]
    return {
        "method": method,
        "n": len(rows),
        "auc_mean": statistics.mean(auc),
        "auc_sample_std": sample_std(auc),
        "final_mean": statistics.mean(final),
        "final_sample_std": sample_std(final),
        "best_mean": statistics.mean(best),
        "best_sample_std": sample_std(best),
        "hit90_reached": len(reached),
        "hit90_median_step_reached_only": statistics.median(reached) if reached else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description="Analyze the frozen paired FPP TD3 n=10 experiment.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo_root / "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_by_method = {
        method: [load_seed(args.artifact_root, method, seed) for seed in range(1, 11)]
        for method in METHODS
    }
    paired_rows = []
    for seed in range(1, 11):
        her = rows_by_method["td3_her"][seed - 1]
        saber = rows_by_method["td3_saber"][seed - 1]
        paired_rows.append(
            {
                "seed": seed,
                "her_auc": her["auc"],
                "saber_auc": saber["auc"],
                "paired_auc_difference": saber["auc"] - her["auc"],
                "her_final": her["final"],
                "saber_final": saber["final"],
                "paired_final_difference": saber["final"] - her["final"],
                "her_best": her["best"],
                "saber_best": saber["best"],
                "her_first_hit_0.90": her["first_hit_0.90"] or "",
                "saber_first_hit_0.90": saber["first_hit_0.90"] or "",
            }
        )

    differences = [float(row["paired_auc_difference"]) for row in paired_rows]
    ci_low, ci_high = paired_bootstrap_ci(differences)
    p_value = exact_sign_flip_p(differences)
    aggregates = [aggregate(method, rows_by_method[method]) for method in METHODS]
    mean_difference = statistics.mean(differences)
    interpretation = (
        "SABER improves paired mean AUC over 10 seeds."
        if ci_low > 0.0
        else "SABER has higher mean AUC, but uncertainty includes zero."
        if mean_difference > 0.0
        else "The paired mean AUC difference is non-positive; reduce the TD3 claim."
    )

    output_dir = args.artifact_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_results.csv", paired_rows)
    write_csv(output_dir / "aggregate_results.csv", aggregates)
    result = {
        "schema_version": 1,
        "primary_endpoint": "trapezoidal success AUC over 10k-1M divided by 990k",
        "evaluation_grid": list(GRID),
        "n_pairs": 10,
        "paired_auc_difference_mean": mean_difference,
        "paired_auc_difference_sample_std": sample_std(differences),
        "paired_bootstrap_ci_95": [ci_low, ci_high],
        "paired_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_bootstrap_seed": BOOTSTRAP_SEED,
        "exact_sign_flip_p_two_sided": p_value,
        "exact_sign_flip_permutations": 1 << len(differences),
        "aggregate": aggregates,
        "interpretation": interpretation,
        "prohibited_processing": ["interpolation", "smoothing", "forward_fill"],
    }
    (output_dir / "analysis_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
