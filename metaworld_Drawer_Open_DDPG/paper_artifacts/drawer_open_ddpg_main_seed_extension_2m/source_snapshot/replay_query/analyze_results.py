#!/usr/bin/env python3
"""Exact-grid analysis for the DrawerOpen DDPG replay-action query ablation.

The analysis is deliberately standard-library only.  It never interpolates,
smooths, forward-fills, or treats episode rows as evaluation rows.  An
evaluation row is a row whose ``eval/env_success_mean`` cell is non-empty and
finite.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 2_000_001, 10_000))
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
BOOTSTRAP_RESAMPLE_COUNT = len(SEEDS) ** len(SEEDS)  # 5^5 = 3,125.
SIGN_FLIP_COUNT = 2 ** len(SEEDS)  # 32.
ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    root: Path


CONTRASTS = (
    ("main_minus_replay", "SABER Main - Replay-Action Query", "main", "replay"),
    ("replay_minus_her", "Replay-Action Query - HER", "replay", "her"),
    ("main_minus_her", "SABER Main - HER", "main", "her"),
)


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def metrics_path_for(run_dir: Path) -> Path:
    for name in ("metrics_selected_10k.csv", "metrics.csv"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{run_dir}: missing metrics_selected_10k.csv or metrics.csv")


def seed_from_run(run_dir: Path) -> int:
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = read_json(config_path)
        seeds = config.get("seeds")
        if isinstance(seeds, dict) and "env_seed" in seeds:
            return int(seeds["env_seed"])
    match = re.search(r"(?:^|_)s(\d+)(?:_|$)", run_dir.name)
    if match is None:
        # Compact artifacts intentionally use data/<arm>/seed_N directories.
        # Supporting that explicit layout keeps the packaged analysis script
        # directly reusable without inventing symlinks or reconstructed names.
        match = re.fullmatch(r"seed_(\d+)", run_dir.name)
    if match is None:
        raise ValueError(f"{run_dir}: cannot infer seed from config or run name")
    return int(match.group(1))


def discover_runs(root: Path, seeds: Sequence[int] = SEEDS) -> dict[int, Path]:
    """Discover exactly the requested seeds while ignoring documented extra cohorts.

    The historical B3 result root also contains diagnostic seeds 6--10.  Those
    are intentionally not part of the primary matched 1--5 cohort.
    """

    if not root.is_dir():
        raise FileNotFoundError(f"result root does not exist: {root}")
    candidates = [root] if (root / "summary.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir()
    )
    wanted = {int(seed) for seed in seeds}
    found: dict[int, Path] = {}
    for run_dir in candidates:
        if not (run_dir / "summary.json").is_file():
            continue
        try:
            seed = seed_from_run(run_dir)
        except ValueError:
            continue
        if seed not in wanted:
            continue
        metrics_path_for(run_dir)
        if seed in found:
            raise ValueError(f"{root}: duplicate seed {seed}: {found[seed]} and {run_dir}")
        found[seed] = run_dir
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{root}: missing primary seeds {missing}")
    return {int(seed): found[int(seed)] for seed in seeds}


def load_eval_curve(
    metrics_path: Path,
    required_steps: Sequence[int] = EVAL_STEPS,
) -> tuple[list[int], list[float]]:
    values_by_step: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or PRIMARY_SUCCESS_FIELD not in reader.fieldnames:
            raise ValueError(f"{metrics_path}: missing {PRIMARY_SUCCESS_FIELD!r} column")
        for row in reader:
            # This non-empty primary field is the sole definition of an eval row.
            raw_success = row.get(PRIMARY_SUCCESS_FIELD, "")
            if raw_success in (None, ""):
                continue
            success = finite_float(raw_success)
            step_value = finite_float(row.get("step"))
            if success is None or step_value is None:
                raise ValueError(f"{metrics_path}: non-finite evaluation row")
            step = int(step_value)
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"{metrics_path}: success outside [0, 1] at step {step}: {success}")
            if step in values_by_step:
                raise ValueError(f"{metrics_path}: duplicate evaluation row at step {step}")
            values_by_step[step] = success

    expected = [int(step) for step in required_steps]
    missing = [step for step in expected if step not in values_by_step]
    unexpected = sorted(set(values_by_step) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{metrics_path}: exact evaluation grid mismatch; missing={missing}, "
            f"unexpected={unexpected}; interpolation/forward-fill are forbidden"
        )
    return expected, [values_by_step[step] for step in expected]


def normalized_auc(steps: Sequence[int], values: Sequence[float]) -> float:
    if len(steps) != len(values) or len(steps) < 2:
        raise ValueError("AUC requires at least two aligned observed points")
    if any(int(right) <= int(left) for left, right in zip(steps, steps[1:])):
        raise ValueError("AUC steps must be strictly increasing")
    area = sum(
        (float(right) - float(left)) * (float(y_left) + float(y_right)) / 2.0
        for left, right, y_left, y_right in zip(steps, steps[1:], values, values[1:])
    )
    span = float(steps[-1]) - float(steps[0])
    return area / span


def first_hit90(steps: Sequence[int], values: Sequence[float]) -> int | None:
    for step, value in zip(steps, values):
        if float(value) >= 0.90:
            return int(step)
    return None


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(float(value) for value in values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(math.fsum((float(value) - center) ** 2 for value in values) / (len(values) - 1))


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exhaustive_paired_bootstrap(differences: Sequence[float]) -> tuple[float, float, int]:
    """Ordered exhaustive paired bootstrap over all n^n index sequences."""

    values = [float(value) for value in differences]
    n = len(values)
    if n != len(SEEDS):
        raise ValueError(f"exhaustive bootstrap requires {len(SEEDS)} paired seeds, got {n}")
    means = [mean([values[index] for index in indices]) for indices in itertools.product(range(n), repeat=n)]
    if len(means) != BOOTSTRAP_RESAMPLE_COUNT:
        raise AssertionError("ordered exhaustive bootstrap count changed")
    return percentile(means, 0.025), percentile(means, 0.975), len(means)


def exact_two_sided_sign_flip(differences: Sequence[float]) -> tuple[float, int]:
    values = [float(value) for value in differences]
    if len(values) != len(SEEDS):
        raise ValueError(f"sign-flip test requires {len(SEEDS)} paired seeds")
    observed = abs(mean(values))
    statistics = [
        abs(mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    extreme = sum(statistic >= observed - 1e-15 for statistic in statistics)
    return extreme / len(statistics), len(statistics)


def analyze_run(method: MethodSpec, seed: int, run_dir: Path) -> tuple[dict[str, object], list[float]]:
    metrics_path = metrics_path_for(run_dir)
    steps, values = load_eval_curve(metrics_path)
    auc = normalized_auc(steps, values)
    best = max(values)
    final = values[-1]
    hit90 = first_hit90(steps, values)
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    native_auc = finite_float(summary.get("eval_AUC_success"))
    native_final = finite_float(summary.get("final_eval_env_success"))
    native_best = finite_float(summary.get("best_eval_env_success"))
    return (
        {
            "method_key": method.key,
            "method": method.label,
            "seed": int(seed),
            "run_dir": run_dir.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "evaluation_points": len(steps),
            "auc": auc,
            "final_success": final,
            "best_success": best,
            "best_to_final_drop": best - final,
            "hit90": hit90,
            "summary_auc": native_auc,
            "summary_auc_difference": None if native_auc is None else native_auc - auc,
            "summary_final_difference": None if native_final is None else native_final - final,
            "summary_best_difference": None if native_best is None else native_best - best,
        },
        values,
    )


def aggregate_rows(per_seed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method_key in ("her", "main", "replay"):
        rows = sorted(
            (row for row in per_seed if row["method_key"] == method_key),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{method_key}: aggregate requires exact seeds {list(SEEDS)}")
        result: dict[str, object] = {
            "method_key": method_key,
            "method": rows[0]["method"],
            "seed_count": len(rows),
        }
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            values = [float(row[field]) for row in rows]
            result[f"{field}_mean"] = mean(values)
            result[f"{field}_sample_sd"] = sample_sd(values)
        hits = [int(row["hit90"]) for row in rows if row["hit90"] not in (None, "")]
        result["hit90_seed_count"] = len(hits)
        result["hit90_median_reached"] = None if not hits else float(median(hits))
        result["hit90_mean_reached"] = None if not hits else mean(hits)
        output.append(result)
    return output


def paired_comparison_rows(per_seed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (str(row["method_key"]), int(row["seed"])): row
        for row in per_seed
    }
    output: list[dict[str, object]] = []
    for key, label, left, right in CONTRASTS:
        differences = [
            float(indexed[(left, seed)]["auc"]) - float(indexed[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        ci_low, ci_high, bootstrap_count = exhaustive_paired_bootstrap(differences)
        p_value, sign_count = exact_two_sided_sign_flip(differences)
        leave_one_out = [mean([value for index, value in enumerate(differences) if index != omitted]) for omitted in range(len(SEEDS))]
        row: dict[str, object] = {
            "contrast_key": key,
            "contrast": label,
            "left_method": left,
            "right_method": right,
            "paired_seed_count": len(SEEDS),
            "mean_auc_difference": mean(differences),
            "sample_sd_auc_difference": sample_sd(differences),
            "bootstrap_ci95_lower": ci_low,
            "bootstrap_ci95_upper": ci_high,
            "bootstrap_resample_count": bootstrap_count,
            "exact_two_sided_sign_flip_pvalue": p_value,
            "sign_flip_assignment_count": sign_count,
            "positive_seed_count": sum(value > ZERO_TOLERANCE for value in differences),
            "zero_seed_count": sum(abs(value) <= ZERO_TOLERANCE for value in differences),
            "negative_seed_count": sum(value < -ZERO_TOLERANCE for value in differences),
            "leave_one_seed_out_mean_min": min(leave_one_out),
            "leave_one_seed_out_mean_max": max(leave_one_out),
        }
        for seed, difference in zip(SEEDS, differences):
            row[f"seed_{seed}_auc_difference"] = difference
        output.append(row)
    return output


def mean_curve_rows(curves: Mapping[tuple[str, int], Sequence[float]], labels: Mapping[str, str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for point_index, step in enumerate(EVAL_STEPS):
        row: dict[str, object] = {"step": step}
        for method_key in ("her", "main", "replay"):
            values = [float(curves[(method_key, seed)][point_index]) for seed in SEEDS]
            row[f"{method_key}_mean_success"] = mean(values)
            row[f"{method_key}_sample_sd_success"] = sample_sd(values)
            row[f"{method_key}_method"] = labels[method_key]
        output.append(row)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def analyze(
    methods: Sequence[MethodSpec],
    output_dir: Path | None = None,
) -> dict[str, object]:
    if [method.key for method in methods] != ["her", "main", "replay"]:
        raise ValueError("methods must be ordered as her, main, replay")
    per_seed: list[dict[str, object]] = []
    curves: dict[tuple[str, int], Sequence[float]] = {}
    for method in methods:
        for seed, run_dir in discover_runs(method.root).items():
            row, values = analyze_run(method, seed, run_dir)
            per_seed.append(row)
            curves[(method.key, seed)] = values
    aggregates = aggregate_rows(per_seed)
    comparisons = paired_comparison_rows(per_seed)
    labels = {method.key: method.label for method in methods}
    mean_curves = mean_curve_rows(curves, labels)
    result = {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "evaluation_steps": list(EVAL_STEPS),
        "evaluation_field": PRIMARY_SUCCESS_FIELD,
        "auc_convention": "observed-only trapezoid normalized by (2,000,000 - 10,000)",
        "smoothing": False,
        "interpolation": False,
        "forward_fill": False,
        "extrapolation": False,
        "bootstrap": "ordered exhaustive paired bootstrap, 5^5=3125 index sequences",
        "per_seed": per_seed,
        "aggregate": aggregates,
        "paired_comparisons": comparisons,
        "mean_curves": mean_curves,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "per_seed_results.csv", per_seed)
        write_csv(output_dir / "aggregate_results.csv", aggregates)
        write_csv(output_dir / "paired_comparisons.csv", comparisons)
        write_csv(output_dir / "exact_grid_mean_curves.csv", mean_curves)
        (output_dir / "analysis_results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    repo = default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--her-root",
        type=Path,
        default=repo / "metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_saber_context_eligible_count_matched_2m/data/her",
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=repo / "metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_saber_context_eligible_count_matched_2m/data/main",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=repo / "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Replay-Action-Query-2M",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = (
        MethodSpec("her", "DrawerOpen DDPG HER", args.her_root.resolve()),
        MethodSpec("main", "DrawerOpen DDPG HER + SABER Main", args.main_root.resolve()),
        MethodSpec("replay", "DrawerOpen DDPG Replay-Action Query", args.replay_root.resolve()),
    )
    result = analyze(methods, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": "passed",
                "output_dir": args.output_dir.resolve().as_posix(),
                "seed_count": len(SEEDS),
                "evaluation_points_per_seed": len(EVAL_STEPS),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLE_COUNT,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
