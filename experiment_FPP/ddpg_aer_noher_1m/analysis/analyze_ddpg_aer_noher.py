#!/usr/bin/env python3
"""Analyze the paired FetchPickAndPlace DDPG no-HER AER experiment.

The primary endpoint is the trapezoidal success AUC on the frozen evaluation
grid 10k, 20k, ..., 1M, divided by 990k.  Curves must contain every grid point;
this script never interpolates, smooths, or forward-fills observations.

Only Python's standard library is required.  Run directories are discovered
from config.json contents rather than directory-name conventions, so one common
root or three method-specific roots can be supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


GRID = tuple(range(10_000, 1_000_001, 10_000))
METHODS = ("ddpg_er", "ddpg_annealing_er", "ddpg_aer")
COMPARISONS = (
    ("ddpg_aer_vs_ddpg_annealing_er", "ddpg_aer", "ddpg_annealing_er"),
    ("ddpg_aer_vs_ddpg_er", "ddpg_aer", "ddpg_er"),
)
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_012_027
EXPECTED_TOTAL_STEPS = 1_000_000
EXPECTED_ENV_ID = "FetchPickAndPlace-v4"

# These fields define the common DDPG/environment/evaluation backbone.  AER
# selection and LR-schedule-specific fields are deliberately not included.
COMMON_CONFIG_KEYS = (
    "task_name",
    "reward_type",
    "env_id",
    "max_episode_steps",
    "total_steps",
    "start_steps",
    "warmup_steps",
    "updates_per_step",
    "batch_size",
    "buffer_capacity",
    "her_ratio",
    "gamma",
    "tau",
    "actor_lr",
    "critic_lr",
    "hidden_dim",
    "noise_std",
    "l2_action_penalty",
    "eval_freq",
    "eval_episodes",
    "final_eval_episodes",
    "checkpoint_every_steps",
    "success_terminates",
    "timeout_bootstrap",
)
PAIRED_SEED_KEYS = (
    "env_seed",
    "torch_seed",
    "numpy_seed",
    "action_noise_seed",
    "eval_seed",
)


@dataclass(frozen=True)
class RunResult:
    method: str
    seed: int
    run_dir: Path
    metrics_path: Path
    curve: tuple[float, ...]
    auc: float
    final: float
    best: float
    first_hits: dict[float, int | None]
    runtime_total_sec: float | None
    final_steps_per_sec: float | None
    config: dict[str, Any]


def find_repo_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "version_v0_lite").is_dir() and (parent / "experiment_FPP").is_dir():
            return parent
    return Path.cwd()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def config_seed(config: dict[str, Any]) -> int | None:
    seeds = config.get("seeds")
    value = seeds.get("env_seed") if isinstance(seeds, dict) else config.get("env_seed")
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def parse_seed_spec(spec: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in (item.strip() for item in spec.split(",")):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending seed range is not allowed: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds in {spec!r}")
    if len(seeds) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to at most 20 paired seeds")
    return tuple(seeds)


def find_metrics(run_dir: Path) -> Path:
    preferred = run_dir / "metrics_selected_10k.csv"
    if preferred.is_file():
        return preferred
    fallback = run_dir / "metrics.csv"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No metrics_selected_10k.csv or metrics.csv in {run_dir}")


def load_curve(path: Path) -> tuple[float, ...]:
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "eval/success_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns {sorted(required)} in {path}")
        for row_number, row in enumerate(reader, start=2):
            success = finite_float(row.get("eval/success_mean"))
            if success is None:
                continue
            step_number = finite_float(row.get("step"))
            if step_number is None or not step_number.is_integer():
                raise ValueError(f"Invalid evaluation step on row {row_number} in {path}")
            step = int(step_number)
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
    return tuple(values[step] for step in GRID)


def normalized_auc(curve: Sequence[float]) -> float:
    if len(curve) != len(GRID):
        raise ValueError(f"AUC requires exactly {len(GRID)} points on the frozen grid")
    area = sum(
        (GRID[index + 1] - GRID[index]) * (curve[index] + curve[index + 1]) / 2.0
        for index in range(len(GRID) - 1)
    )
    return area / float(GRID[-1] - GRID[0])


def first_hit(curve: Sequence[float], threshold: float) -> int | None:
    return next((step for step, value in zip(GRID, curve) if value >= threshold), None)


def sample_std(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.stdev(data) if len(data) > 1 else 0.0


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("Paired bootstrap requires at least one difference")
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_sign_flip_p(differences: Sequence[float]) -> float:
    if not differences:
        raise ValueError("Exact sign-flip test requires at least one difference")
    n = len(differences)
    observed = abs(statistics.mean(differences))
    extreme = 0
    for mask in range(1 << n):
        statistic = abs(
            sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(differences)
            )
            / n
        )
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / float(1 << n)


def discover_run(root: Path, method: str, seed: int, expected_total_steps: int) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(root)
    matches: list[Path] = []
    for config_path in sorted(root.rglob("config.json")):
        try:
            config = read_json(config_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if config.get("method") != method:
            continue
        if config_seed(config) != seed:
            continue
        if config.get("total_steps") != expected_total_steps:
            continue
        matches.append(config_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one complete config for method={method}, seed={seed}, "
            f"total_steps={expected_total_steps} under {root}; found {len(matches)}: {matches}"
        )
    return matches[0]


def validate_basic_config(
    config: dict[str, Any], method: str, seed: int, expected_total_steps: int, run_dir: Path
) -> None:
    expected = {
        "method": method,
        "env_id": EXPECTED_ENV_ID,
        "reward_type": "sparse",
        "total_steps": expected_total_steps,
        "eval_freq": 10_000,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{run_dir}: config.{key} expected {value!r}, got {config.get(key)!r}")
    her_ratio = finite_float(config.get("her_ratio"))
    if her_ratio is None or abs(her_ratio) > 1e-15:
        raise ValueError(f"{run_dir}: standalone experiment requires config.her_ratio=0, got {her_ratio}")
    if config_seed(config) != seed:
        raise ValueError(f"{run_dir}: env seed does not match requested seed {seed}")


def load_run(root: Path, method: str, seed: int, expected_total_steps: int) -> RunResult:
    run_dir = discover_run(root, method, seed, expected_total_steps)
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    for path in (summary_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    config = read_json(config_path)
    validate_basic_config(config, method, seed, expected_total_steps, run_dir)
    if summary.get("total_steps") != expected_total_steps:
        raise ValueError(f"{run_dir}: incomplete summary total_steps={summary.get('total_steps')!r}")
    if summary.get("method") != method:
        raise ValueError(f"{run_dir}: summary method mismatch: {summary.get('method')!r}")

    metrics_path = find_metrics(run_dir)
    curve = load_curve(metrics_path)
    auc = normalized_auc(curve)
    reported_auc = finite_float(summary.get("eval_AUC_success"))
    if reported_auc is not None and abs(auc - reported_auc) > 1e-9:
        raise ValueError(
            f"{run_dir}: AUC mismatch: exact-grid={auc:.12f}, summary={reported_auc:.12f}"
        )
    return RunResult(
        method=method,
        seed=seed,
        run_dir=run_dir.resolve(),
        metrics_path=metrics_path.resolve(),
        curve=curve,
        auc=auc,
        final=curve[-1],
        best=max(curve),
        first_hits={threshold: first_hit(curve, threshold) for threshold in THRESHOLDS},
        runtime_total_sec=finite_float(summary.get("runtime_total_sec")),
        final_steps_per_sec=finite_float(summary.get("final_steps_per_sec")),
        config=config,
    )


def validate_paired_configs(rows_by_method: dict[str, list[RunResult]], seeds: Sequence[int]) -> None:
    reference = rows_by_method[METHODS[0]][0].config
    for key in COMMON_CONFIG_KEYS:
        if key not in reference:
            raise ValueError(f"Reference config is missing common field {key!r}")
        expected = reference[key]
        for method in METHODS:
            for run in rows_by_method[method]:
                if key not in run.config:
                    raise ValueError(f"{run.run_dir}: missing common config field {key!r}")
                if run.config[key] != expected:
                    raise ValueError(
                        f"Common config mismatch for {key}: expected {expected!r}, "
                        f"got {run.config[key]!r} in {run.run_dir}"
                    )

    for seed in seeds:
        runs = {method: rows_by_method[method][seeds.index(seed)] for method in METHODS}
        reference_seeds = runs[METHODS[0]].config.get("seeds", {})
        if not isinstance(reference_seeds, dict):
            raise ValueError(f"{runs[METHODS[0]].run_dir}: config.seeds must be an object")
        for key in PAIRED_SEED_KEYS:
            if key not in reference_seeds:
                raise ValueError(f"{runs[METHODS[0]].run_dir}: missing config.seeds.{key}")
            expected = reference_seeds[key]
            for method, run in runs.items():
                method_seeds = run.config.get("seeds", {})
                actual = method_seeds.get(key) if isinstance(method_seeds, dict) else None
                if actual != expected:
                    raise ValueError(
                        f"Paired seed mismatch for seed={seed}, field={key}: "
                        f"{METHODS[0]}={expected!r}, {method}={actual!r}"
                    )


def aggregate_method(method: str, rows: Sequence[RunResult]) -> dict[str, Any]:
    aucs = [run.auc for run in rows]
    finals = [run.final for run in rows]
    bests = [run.best for run in rows]
    runtimes = [run.runtime_total_sec for run in rows if run.runtime_total_sec is not None]
    throughputs = [run.final_steps_per_sec for run in rows if run.final_steps_per_sec is not None]
    result: dict[str, Any] = {
        "method": method,
        "n": len(rows),
        "auc_mean": statistics.mean(aucs),
        "auc_sample_std": sample_std(aucs),
        "auc_median": statistics.median(aucs),
        "final_mean": statistics.mean(finals),
        "final_sample_std": sample_std(finals),
        "best_mean": statistics.mean(bests),
        "best_sample_std": sample_std(bests),
        "runtime_total_sec_mean": statistics.mean(runtimes) if runtimes else None,
        "final_steps_per_sec_mean": statistics.mean(throughputs) if throughputs else None,
    }
    for threshold in THRESHOLDS:
        hits = [run.first_hits[threshold] for run in rows if run.first_hits[threshold] is not None]
        label = f"hit{int(round(threshold * 100)):02d}"
        result[f"{label}_reached"] = len(hits)
        result[f"{label}_median_step_reached_only"] = statistics.median(hits) if hits else None
    return result


def compare(
    label: str,
    method: str,
    baseline: str,
    rows_by_method: dict[str, list[RunResult]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    method_rows = rows_by_method[method]
    baseline_rows = rows_by_method[baseline]
    differences = [left.auc - right.auc for left, right in zip(method_rows, baseline_rows)]
    final_differences = [left.final - right.final for left, right in zip(method_rows, baseline_rows)]
    ci_low, ci_high = paired_bootstrap_ci(differences, seed=bootstrap_seed)
    tolerance = 1e-15
    return {
        "comparison": label,
        "method": method,
        "baseline": baseline,
        "difference_definition": f"{method}_minus_{baseline}",
        "n_pairs": len(differences),
        "paired_auc_difference_mean": statistics.mean(differences),
        "paired_auc_difference_sample_std": sample_std(differences),
        "paired_auc_difference_median": statistics.median(differences),
        "paired_auc_bootstrap_ci_95_low": ci_low,
        "paired_auc_bootstrap_ci_95_high": ci_high,
        "paired_auc_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_auc_bootstrap_seed": bootstrap_seed,
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(differences),
        "exact_sign_flip_permutations": 1 << len(differences),
        "auc_wins": sum(value > tolerance for value in differences),
        "auc_ties": sum(abs(value) <= tolerance for value in differences),
        "auc_losses": sum(value < -tolerance for value in differences),
        "paired_final_difference_mean": statistics.mean(final_differences),
        "paired_final_difference_sample_std": sample_std(final_differences),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def per_seed_rows(
    rows_by_method: dict[str, list[RunResult]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        runs = {method: rows_by_method[method][index] for method in METHODS}
        row: dict[str, Any] = {"seed": seed}
        for method, run in runs.items():
            row[f"{method}_auc"] = run.auc
            row[f"{method}_final"] = run.final
            row[f"{method}_best"] = run.best
            for threshold in THRESHOLDS:
                label = int(round(threshold * 100))
                row[f"{method}_first_hit_{label:02d}"] = run.first_hits[threshold] or ""
            row[f"{method}_runtime_total_sec"] = run.runtime_total_sec
            row[f"{method}_final_steps_per_sec"] = run.final_steps_per_sec
            row[f"{method}_run_dir"] = str(run.run_dir)
        row["aer_minus_er_auc"] = runs["ddpg_aer"].auc - runs["ddpg_er"].auc
        row["aer_minus_annealing_er_auc"] = (
            runs["ddpg_aer"].auc - runs["ddpg_annealing_er"].auc
        )
        row["aer_minus_er_final"] = runs["ddpg_aer"].final - runs["ddpg_er"].final
        row["aer_minus_annealing_er_final"] = (
            runs["ddpg_aer"].final - runs["ddpg_annealing_er"].final
        )
        rows.append(row)
    return rows


def mean_curve_rows(rows_by_method: dict[str, list[RunResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for grid_index, step in enumerate(GRID):
        row: dict[str, Any] = {"step": step}
        for method in METHODS:
            values = [run.curve[grid_index] for run in rows_by_method[method]]
            row[f"{method}_mean"] = statistics.mean(values)
            row[f"{method}_sample_std"] = sample_std(values)
        rows.append(row)
    return rows


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# FPP DDPG no-HER AER paired analysis",
        "",
        "Primary endpoint: trapezoidal `eval/success_mean` AUC on the exact "
        "10k, 20k, ..., 1M grid, divided by 990k.",
        "",
        "No interpolation, smoothing, forward filling, or best-checkpoint selection was used.",
        "",
        "## Method summaries",
        "",
        "| Method | n | AUC mean | AUC SD | Final mean | Best mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for aggregate in result["method_aggregates"]:
        lines.append(
            f"| {aggregate['method']} | {aggregate['n']} | "
            f"{aggregate['auc_mean']:.6f} | {aggregate['auc_sample_std']:.6f} | "
            f"{aggregate['final_mean']:.6f} | {aggregate['best_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired AUC comparisons",
            "",
            "Positive differences favor `ddpg_aer`.",
            "",
            "| Comparison | Mean difference | 95% paired bootstrap CI | Exact sign-flip p | W/T/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in result["comparisons"]:
        lines.append(
            f"| {comparison['comparison']} | "
            f"{comparison['paired_auc_difference_mean']:.6f} | "
            f"[{comparison['paired_auc_bootstrap_ci_95_low']:.6f}, "
            f"{comparison['paired_auc_bootstrap_ci_95_high']:.6f}] | "
            f"{comparison['exact_sign_flip_p_two_sided']:.6g} | "
            f"{comparison['auc_wins']}/{comparison['auc_ties']}/{comparison['auc_losses']} |"
        )
    lines.extend(
        [
            "",
            "The exact sign-flip test is two-sided and enumerates every paired sign assignment. "
            "The bootstrap resamples paired seed differences, not unpaired method values.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    default_runs_root = repo_root / "experiment_FPP/runs/fpp_ddpg_aer_noher_1m"
    parser = argparse.ArgumentParser(
        description="Analyze paired ddpg_aer, ddpg_er, and ddpg_annealing_er FPP runs."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default_runs_root,
        help="Common root searched recursively for all methods (default: %(default)s).",
    )
    parser.add_argument("--ddpg-aer-root", type=Path, default=None)
    parser.add_argument("--ddpg-er-root", type=Path, default=None)
    parser.add_argument("--ddpg-annealing-er-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RUNS_ROOT/analysis).",
    )
    parser.add_argument(
        "--seeds",
        default="1-10",
        help="Comma-separated seeds and inclusive ranges (default: 1-10).",
    )
    parser.add_argument(
        "--expected-total-steps", type=int, default=EXPECTED_TOTAL_STEPS, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_spec(args.seeds)
    roots = {
        "ddpg_aer": args.ddpg_aer_root or args.runs_root,
        "ddpg_er": args.ddpg_er_root or args.runs_root,
        "ddpg_annealing_er": args.ddpg_annealing_er_root or args.runs_root,
    }
    rows_by_method = {
        method: [
            load_run(roots[method], method, seed, args.expected_total_steps) for seed in seeds
        ]
        for method in METHODS
    }
    validate_paired_configs(rows_by_method, seeds)

    aggregates = [aggregate_method(method, rows_by_method[method]) for method in METHODS]
    comparisons = [
        compare(label, method, baseline, rows_by_method, BOOTSTRAP_SEED + index)
        for index, (label, method, baseline) in enumerate(COMPARISONS)
    ]
    seed_rows = per_seed_rows(rows_by_method, seeds)
    curves = mean_curve_rows(rows_by_method)
    result = {
        "schema_version": 1,
        "experiment": "fpp_ddpg_aer_noher_1m",
        "methods": list(METHODS),
        "seeds": list(seeds),
        "n_pairs": len(seeds),
        "primary_endpoint": "trapezoidal success AUC on exact 10k..1M grid divided by 990k",
        "evaluation_metric": "eval/success_mean",
        "evaluation_grid": list(GRID),
        "auc_denominator_steps": GRID[-1] - GRID[0],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "method_aggregates": aggregates,
        "comparisons": comparisons,
        "source_runs": {
            method: [
                {
                    "seed": run.seed,
                    "run_dir": str(run.run_dir),
                    "metrics_path": str(run.metrics_path),
                }
                for run in rows_by_method[method]
            ]
            for method in METHODS
        },
        "validation": {
            "expected_total_steps": args.expected_total_steps,
            "expected_env_id": EXPECTED_ENV_ID,
            "required_her_ratio": 0.0,
            "paired_common_config_keys": list(COMMON_CONFIG_KEYS),
            "paired_seed_keys": list(PAIRED_SEED_KEYS),
            "prohibited_processing": ["interpolation", "smoothing", "forward_fill"],
        },
    }

    output_dir = args.output_dir or args.runs_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_results.csv", seed_rows)
    write_csv(output_dir / "method_summary.csv", aggregates)
    write_csv(output_dir / "pairwise_summary.csv", comparisons)
    write_csv(output_dir / "mean_curves.csv", curves)
    (output_dir / "analysis_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "analysis_summary.md").write_text(markdown_summary(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
