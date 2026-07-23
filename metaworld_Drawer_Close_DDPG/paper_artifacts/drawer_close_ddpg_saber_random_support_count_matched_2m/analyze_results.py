#!/usr/bin/env python3
"""Exact-grid analysis for the DrawerClose DDPG Random-Support control.

The analysis uses only ``eval/env_success_mean`` observed at the exact
10k--2M evaluation grid.  It performs no fallback, smoothing, interpolation,
forward-fill, or extrapolation.
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
from typing import Mapping, Sequence


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 2_000_001, 10_000))
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
METHOD_ORDER = ("her", "main", "random")
CONTRASTS = (
    (
        "main_minus_random",
        "SABER Main - Random-Support Count-Matched",
        "main",
        "random",
    ),
    (
        "random_minus_her",
        "Random-Support Count-Matched - HER",
        "random",
        "her",
    ),
    ("main_minus_her", "SABER Main - HER", "main", "her"),
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    root: Path


def finite_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        math.fsum((value - center) ** 2 for value in values)
        / (len(values) - 1)
    )


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exhaustive_paired_bootstrap(
    differences: Sequence[float],
) -> tuple[float, float, int]:
    if len(differences) != 5:
        raise ValueError("exhaustive bootstrap requires exactly five seeds")
    means = [
        mean([differences[index] for index in indices])
        for indices in itertools.product(range(5), repeat=5)
    ]
    return percentile(means, 0.025), percentile(means, 0.975), len(means)


def exact_two_sided_sign_flip(
    differences: Sequence[float],
) -> tuple[float, int]:
    if len(differences) != 5:
        raise ValueError("sign-flip test requires exactly five seeds")
    observed = abs(mean(differences))
    statistics = [
        abs(mean([sign * value for sign, value in zip(signs, differences)]))
        for signs in itertools.product((-1.0, 1.0), repeat=5)
    ]
    return (
        sum(value >= observed - 1e-15 for value in statistics)
        / len(statistics),
        len(statistics),
    )


def seed_from_dir(path: Path) -> int:
    if re.fullmatch(r"seed_[1-5]", path.name):
        return int(path.name.split("_")[-1])
    match = re.search(r"(?:^|_)s([1-5])(?:_|$)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot infer seed from {path}")


def discover_runs(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    found: dict[int, Path] = {}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            seed = seed_from_dir(child)
        except ValueError:
            continue
        if seed in found:
            raise ValueError(f"{root}: duplicate seed {seed}")
        if not (child / "metrics_selected_10k.csv").is_file():
            raise ValueError(f"{child}: missing compact metrics")
        found[seed] = child
    if set(found) != set(SEEDS):
        raise ValueError(f"{root}: seed coverage mismatch {sorted(found)}")
    return {seed: found[seed] for seed in SEEDS}


def load_curve(path: Path) -> list[float]:
    steps: list[int] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"step", PRIMARY_SUCCESS_FIELD}
        if not required.issubset(fields):
            raise ValueError(f"{path}: missing {sorted(required - fields)}")
        for line, row in enumerate(reader, start=2):
            raw_step = finite_float(row.get("step"))
            value = finite_float(row.get(PRIMARY_SUCCESS_FIELD))
            if raw_step is None or value is None:
                raise ValueError(f"{path}:{line}: non-finite selected row")
            step = int(raw_step)
            if raw_step != step or not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}:{line}: invalid step/success")
            steps.append(step)
            values.append(value)
    if steps != list(EVAL_STEPS):
        raise ValueError(
            f"{path}: exact-grid mismatch ({len(steps)} vs {len(EVAL_STEPS)})"
        )
    return values


def normalized_auc(values: Sequence[float]) -> float:
    area = math.fsum(
        (right - left) * (y_left + y_right) / 2.0
        for left, right, y_left, y_right in zip(
            EVAL_STEPS, EVAL_STEPS[1:], values, values[1:]
        )
    )
    return area / (EVAL_STEPS[-1] - EVAL_STEPS[0])


def first_hit90(values: Sequence[float]) -> int | None:
    return next(
        (
            step
            for step, value in zip(EVAL_STEPS, values)
            if value >= 0.9
        ),
        None,
    )


def portable(path: Path, artifact_root: Path) -> str:
    try:
        return path.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != fields:
                raise ValueError(f"{path}: inconsistent row schema")
            writer.writerow(
                {key: "" if value is None else value for key, value in row.items()}
            )


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def analyze(methods: Sequence[MethodSpec], output_dir: Path) -> dict[str, object]:
    if [method.key for method in methods] != list(METHOD_ORDER):
        raise ValueError("methods must be ordered her, main, random")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir.resolve()

    per_seed: list[dict[str, object]] = []
    curves: dict[tuple[str, int], list[float]] = {}
    labels = {method.key: method.label for method in methods}
    for method in methods:
        for seed, run_dir in discover_runs(method.root).items():
            metrics = run_dir / "metrics_selected_10k.csv"
            values = load_curve(metrics)
            curves[(method.key, seed)] = values
            best = max(values)
            final = values[-1]
            summary = read_json(run_dir / "summary.json")
            summary_auc = finite_float(summary.get("eval_AUC_success"))
            summary_final = finite_float(
                summary.get("final_eval_env_success")
            )
            summary_best = finite_float(summary.get("best_eval_env_success"))
            if (
                summary_auc is None
                or summary_final is None
                or summary_best is None
            ):
                raise ValueError(
                    f"{run_dir}/summary.json: missing trustworthy "
                    "AUC/final/best fields"
                )
            exact_auc = normalized_auc(values)
            per_seed.append(
                {
                    "method_key": method.key,
                    "method": method.label,
                    "seed": seed,
                    "run_dir": portable(run_dir, artifact_root),
                    "metrics_path": portable(metrics, artifact_root),
                    "evaluation_points": len(values),
                    "auc": exact_auc,
                    "final_success": final,
                    "best_success": best,
                    "best_to_final_drop": best - final,
                    "hit90": first_hit90(values),
                    "summary_auc": summary_auc,
                    "summary_auc_difference": summary_auc - exact_auc,
                    "summary_final_difference": summary_final - final,
                    "summary_best_difference": summary_best - best,
                }
            )

    aggregate: list[dict[str, object]] = []
    for method_key in METHOD_ORDER:
        rows = [row for row in per_seed if row["method_key"] == method_key]
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{method_key}: incomplete seed ordering")
        item: dict[str, object] = {
            "method_key": method_key,
            "method": rows[0]["method"],
            "seed_count": len(rows),
        }
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            values = [float(row[field]) for row in rows]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_sample_sd"] = sample_sd(values)
        hits = [int(row["hit90"]) for row in rows if row["hit90"] is not None]
        item["hit90_seed_count"] = len(hits)
        item["hit90_median_reached"] = (
            None if not hits else float(median(hits))
        )
        item["hit90_mean_reached"] = None if not hits else mean(hits)
        aggregate.append(item)

    index = {
        (str(row["method_key"]), int(row["seed"])): row for row in per_seed
    }
    paired: list[dict[str, object]] = []
    for key, label, left, right in CONTRASTS:
        differences = [
            float(index[(left, seed)]["auc"])
            - float(index[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        lower, upper, bootstrap_count = exhaustive_paired_bootstrap(differences)
        sign_p, sign_count = exact_two_sided_sign_flip(differences)
        leave_one_out = [
            mean(
                [
                    value
                    for index_, value in enumerate(differences)
                    if index_ != omitted
                ]
            )
            for omitted in range(5)
        ]
        row: dict[str, object] = {
            "contrast_key": key,
            "contrast": label,
            "left_method": left,
            "right_method": right,
            "paired_seed_count": 5,
            "mean_auc_difference": mean(differences),
            "sample_sd_auc_difference": sample_sd(differences),
            "bootstrap_ci95_lower": lower,
            "bootstrap_ci95_upper": upper,
            "bootstrap_resample_count": bootstrap_count,
            "exact_two_sided_sign_flip_pvalue": sign_p,
            "sign_flip_assignment_count": sign_count,
            "positive_seed_count": sum(value > 1e-12 for value in differences),
            "zero_seed_count": sum(abs(value) <= 1e-12 for value in differences),
            "negative_seed_count": sum(value < -1e-12 for value in differences),
            "leave_one_seed_out_mean_min": min(leave_one_out),
            "leave_one_seed_out_mean_max": max(leave_one_out),
        }
        for seed, difference in zip(SEEDS, differences):
            row[f"seed_{seed}_auc_difference"] = difference
        paired.append(row)

    mean_curves: list[dict[str, object]] = []
    for point_index, step in enumerate(EVAL_STEPS):
        row: dict[str, object] = {"step": step}
        for method_key in METHOD_ORDER:
            values = [
                curves[(method_key, seed)][point_index] for seed in SEEDS
            ]
            row[f"{method_key}_mean_success"] = mean(values)
            row[f"{method_key}_sample_sd_success"] = sample_sd(values)
            row[f"{method_key}_method"] = labels[method_key]
        mean_curves.append(row)

    write_csv(output_dir / "per_seed_results.csv", per_seed)
    write_csv(output_dir / "aggregate_results.csv", aggregate)
    write_csv(output_dir / "paired_comparisons.csv", paired)
    write_csv(output_dir / "exact_grid_mean_curves.csv", mean_curves)
    payload = {
        "schema_version": 1,
        "auc_convention": (
            "observed-only trapezoid normalized by "
            "(2,000,000 - 10,000)"
        ),
        "bootstrap": (
            "ordered exhaustive paired bootstrap, "
            "5^5=3125 index sequences"
        ),
        "evaluation_field": PRIMARY_SUCCESS_FIELD,
        "evaluation_steps": list(EVAL_STEPS),
        "extrapolation": False,
        "forward_fill": False,
        "interpolation": False,
        "mean_curves": mean_curves,
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "per_seed": per_seed,
        "seeds": list(SEEDS),
        "smoothing": False,
    }
    write_json(output_dir / "analysis_results.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--her-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analyze(
        (
            MethodSpec("her", "DrawerClose DDPG HER", args.her_root),
            MethodSpec(
                "main",
                "DrawerClose DDPG HER + SABER Main",
                args.main_root,
            ),
            MethodSpec(
                "random",
                "DrawerClose DDPG Random-Support Count-Matched",
                args.random_root,
            ),
        ),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
