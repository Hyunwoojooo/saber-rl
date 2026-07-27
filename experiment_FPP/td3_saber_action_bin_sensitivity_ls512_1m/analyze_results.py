#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Mapping


SEEDS = (1, 2, 3, 4, 5)
EVAL_GRID = tuple(range(10_000, 1_000_001, 10_000))
FINAL_QUARTER_GRID = tuple(range(750_000, 1_000_001, 10_000))
SUCCESS_FIELDS = ("eval/env_success_mean", "eval/success_mean")
SUPPORT_FIELD = "lite/psi_A_policy_rate"


def finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def metrics_path(run_dir: Path) -> Path:
    for filename in ("metrics_selected_10k.csv", "metrics.csv"):
        candidate = run_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{run_dir}: missing metrics CSV")


def seed_and_bins(run_dir: Path) -> tuple[int, int]:
    path = run_dir / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or "env_seed" not in seeds:
        raise ValueError(f"{path}: missing seeds.env_seed")
    return int(seeds["env_seed"]), int(payload.get("action_bins_per_dim", 5))


def discover_runs(root: Path, expected_bins: int) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    summaries = sorted(root.rglob("summary.json"))
    found: dict[int, Path] = {}
    for summary in summaries:
        run_dir = summary.parent
        if not (run_dir / "config.json").is_file():
            continue
        metrics_path(run_dir)
        seed, bins = seed_and_bins(run_dir)
        if bins != expected_bins:
            raise ValueError(
                f"{run_dir}: action_bins_per_dim={bins}, expected {expected_bins}"
            )
        if seed in found:
            raise ValueError(
                f"{root}: duplicate seed {seed}: {found[seed]} and {run_dir}"
            )
        found[seed] = run_dir
    missing = [seed for seed in SEEDS if seed not in found]
    extra = sorted(seed for seed in found if seed not in SEEDS)
    if missing or extra:
        raise ValueError(
            f"{root}: expected exactly seeds {list(SEEDS)}, "
            f"missing={missing}, extra={extra}"
        )
    return {seed: found[seed] for seed in SEEDS}


def row_step(row: Mapping[str, str]) -> int | None:
    for field in ("step", "global_step", "env_step"):
        value = finite(row.get(field))
        if value is not None:
            return int(value)
    return None


def row_success(row: Mapping[str, str]) -> float | None:
    for field in SUCCESS_FIELDS:
        value = finite(row.get(field))
        if value is not None:
            return value
    return None


def insert_unique(
    values: dict[int, float],
    step: int,
    value: float,
    *,
    source: Path,
    label: str,
) -> None:
    previous = values.get(step)
    if previous is not None and not math.isclose(
        previous, value, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{source}: conflicting {label} values at step {step}: "
            f"{previous} versus {value}"
        )
    values[step] = value


def load_curve_and_support(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    success: dict[int, float] = {}
    support: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = row_step(row)
            if step is None:
                continue
            success_value = row_success(row)
            if success_value is not None:
                insert_unique(
                    success,
                    step,
                    success_value,
                    source=path,
                    label="success",
                )
            support_value = finite(row.get(SUPPORT_FIELD))
            if support_value is not None:
                insert_unique(
                    support,
                    step,
                    support_value,
                    source=path,
                    label="support-positive rate",
                )
    return success, support


def normalized_auc(curve: Mapping[int, float], grid: tuple[int, ...]) -> float:
    missing = [step for step in grid if step not in curve]
    if missing:
        raise ValueError(
            f"missing exact evaluation checkpoints {missing[:10]}; "
            "interpolation and forward filling are not allowed"
        )
    area = sum(
        0.5 * (curve[left] + curve[right]) * (right - left)
        for left, right in zip(grid, grid[1:])
    )
    return float(area / (grid[-1] - grid[0]))


def analyze_run(run_dir: Path, seed: int, bins: int) -> dict[str, object]:
    curve, support = load_curve_and_support(metrics_path(run_dir))
    missing_support = [step for step in EVAL_GRID if step not in support]
    if missing_support:
        raise ValueError(
            f"{run_dir}: missing {SUPPORT_FIELD} on exact 10k checkpoints "
            f"{missing_support[:10]}"
        )
    aligned_support = [support[step] for step in EVAL_GRID]
    return {
        "seed": seed,
        "action_bins_per_dim": bins,
        "run_dir": str(run_dir),
        "auc": normalized_auc(curve, EVAL_GRID),
        "final_quarter_auc": normalized_auc(curve, FINAL_QUARTER_GRID),
        "support_positive_rate": statistics.fmean(aligned_support),
        "support_snapshot_count": len(aligned_support),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in ("auc", "final_quarter_auc", "support_positive_rate"):
        values = [float(row[field]) for row in rows]
        result[f"{field}_mean"] = statistics.fmean(values)
        result[f"{field}_sd"] = statistics.stdev(values)
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_summary(
    candidate: list[dict[str, object]],
    reference: list[dict[str, object]],
    field: str,
) -> dict[str, object]:
    candidate_by_seed = {int(row["seed"]): float(row[field]) for row in candidate}
    reference_by_seed = {int(row["seed"]): float(row[field]) for row in reference}
    differences = [
        candidate_by_seed[seed] - reference_by_seed[seed] for seed in SEEDS
    ]
    rng = random.Random(27_027)
    means = [
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in SEEDS)
        for _ in range(10_000)
    ]
    return {
        "field": field,
        "mean_difference": statistics.fmean(differences),
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "positive_pairs": sum(value > 0.0 for value in differences),
        "differences_by_seed": {
            str(seed): differences[index] for index, seed in enumerate(SEEDS)
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_new_root = (
        repo_root
        / "experiment_FPP/runs/td3_saber_action_bin_sensitivity_ls512_1m"
    )
    default_k5_root = (
        repo_root
        / "experiment_FPP/paper_artifacts/"
        "fpp_td3_saber_query_ablation_ls512/data/saber"
    )
    parser = argparse.ArgumentParser(
        description="Analyze FPP TD3 SABER K=3/5/7 action-bin sensitivity."
    )
    parser.add_argument("--k3-root", type=Path, default=default_new_root / "k3")
    parser.add_argument("--k5-root", type=Path, default=default_k5_root)
    parser.add_argument("--k7-root", type=Path, default=default_new_root / "k7")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    specs = {
        "k3": (args.k3_root, 3),
        "k5_main": (args.k5_root, 5),
        "k7": (args.k7_root, 7),
    }
    per_seed: dict[str, list[dict[str, object]]] = {}
    aggregates: dict[str, dict[str, float]] = {}
    for label, (root, bins) in specs.items():
        rows = [
            analyze_run(run_dir, seed, bins)
            for seed, run_dir in discover_runs(root, bins).items()
        ]
        per_seed[label] = rows
        aggregates[label] = aggregate(rows)

    paired: dict[str, dict[str, object]] = {}
    for label in ("k3", "k7"):
        for field in ("auc", "final_quarter_auc"):
            paired[f"{label}-k5_main:{field}"] = paired_summary(
                per_seed[label],
                per_seed["k5_main"],
                field,
            )

    payload = {
        "protocol": {
            "seeds": list(SEEDS),
            "evaluation_grid": [EVAL_GRID[0], EVAL_GRID[-1], 10_000],
            "final_quarter_grid": [
                FINAL_QUARTER_GRID[0],
                FINAL_QUARTER_GRID[-1],
                10_000,
            ],
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 27_027,
            "smoothing": False,
            "interpolation": False,
            "support_snapshot_grid": [EVAL_GRID[0], EVAL_GRID[-1], 10_000],
        },
        "per_seed": per_seed,
        "aggregate": aggregates,
        "paired_against_k5_main": paired,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
