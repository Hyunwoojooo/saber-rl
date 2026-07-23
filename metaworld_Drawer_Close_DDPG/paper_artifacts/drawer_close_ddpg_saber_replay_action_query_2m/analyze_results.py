#!/usr/bin/env python3
"""Deterministic exact-grid analysis for DrawerClose DDPG replay-query results."""

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
METHOD_ORDER = ("her", "main", "replay")
CONTRASTS = (
    ("main_minus_replay", "SABER Main - Replay-Action Query", "main", "replay"),
    ("replay_minus_her", "Replay-Action Query - HER", "replay", "her"),
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


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def metrics_path_for(run_dir: Path) -> Path:
    selected = run_dir / "metrics_selected_10k.csv"
    if selected.is_file():
        return selected
    raw = run_dir / "metrics.csv"
    if raw.is_file():
        return raw
    raise FileNotFoundError(f"{run_dir}: missing metrics_selected_10k.csv/metrics.csv")


def seed_from_run(run_dir: Path) -> int:
    match = re.search(r"(?:^|_)s(?:eed_)?([1-5])(?:_|$)", run_dir.name)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"seed_([1-5])", run_dir.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot determine seed from {run_dir}")


def discover_runs(root: Path, seeds: Sequence[int] = SEEDS) -> dict[int, Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    found: dict[int, Path] = {}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            seed = seed_from_run(child)
        except ValueError:
            continue
        if seed in seeds and metrics_path_for(child).is_file():
            if seed in found:
                raise ValueError(f"{root}: duplicate run for seed {seed}")
            found[seed] = child
    missing = sorted(set(seeds) - set(found))
    extra = sorted(set(found) - set(seeds))
    if missing or extra:
        raise ValueError(f"{root}: seed coverage mismatch missing={missing} extra={extra}")
    return {seed: found[seed] for seed in seeds}


def load_eval_curve(
    path: Path,
    expected_steps: Sequence[int] = EVAL_STEPS,
) -> tuple[list[int], list[float]]:
    expected = set(expected_steps)
    actual_steps: list[int] = []
    values: list[float] = []
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {"step", PRIMARY_SUCCESS_FIELD} - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row_index, row in enumerate(reader, start=2):
            raw_success = row.get(PRIMARY_SUCCESS_FIELD)
            if raw_success in (None, "") or str(raw_success).lower() == "nan":
                continue
            raw_step = finite_float(row.get("step"))
            success = finite_float(raw_success)
            if raw_step is None or success is None:
                raise ValueError(f"{path}:{row_index}: non-finite evaluation row")
            step = int(raw_step)
            if raw_step != step:
                raise ValueError(f"{path}:{row_index}: non-integral step")
            if step not in expected:
                raise ValueError(f"{path}:{row_index}: unexpected evaluation step {step}")
            if step in seen:
                raise ValueError(f"{path}:{row_index}: duplicate evaluation step {step}")
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"{path}:{row_index}: success outside [0,1]")
            seen.add(step)
            actual_steps.append(step)
            values.append(success)
    if actual_steps != list(expected_steps):
        raise ValueError(
            f"{path}: exact evaluation grid mismatch "
            f"(got {len(actual_steps)}, expected {len(expected_steps)})"
        )
    return actual_steps, values


def normalized_auc(steps: Sequence[int], values: Sequence[float]) -> float:
    if len(steps) != len(values) or len(steps) < 2:
        raise ValueError("AUC requires aligned nontrivial sequences")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("AUC steps must be strictly increasing")
    area = math.fsum(
        (right - left) * (y_left + y_right) / 2.0
        for left, right, y_left, y_right in zip(
            steps, steps[1:], values, values[1:]
        )
    )
    return area / (steps[-1] - steps[0])


def first_hit90(steps: Sequence[int], values: Sequence[float]) -> int | None:
    return next((step for step, value in zip(steps, values) if value >= 0.9), None)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
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
        raise ValueError("ordered exhaustive bootstrap requires exactly five seeds")
    means = [
        mean([differences[index] for index in indices])
        for indices in itertools.product(range(5), repeat=5)
    ]
    return percentile(means, 0.025), percentile(means, 0.975), len(means)


def exact_two_sided_sign_flip(differences: Sequence[float]) -> tuple[float, int]:
    if len(differences) != 5:
        raise ValueError("exact sign-flip requires exactly five seeds")
    observed = abs(mean(differences))
    statistics = [
        abs(mean([sign * value for sign, value in zip(signs, differences)]))
        for signs in itertools.product((-1.0, 1.0), repeat=5)
    ]
    return (
        sum(value >= observed - 1e-15 for value in statistics) / len(statistics),
        len(statistics),
    )


def analyze_run(
    method: MethodSpec,
    seed: int,
    run_dir: Path,
) -> tuple[dict[str, object], list[float]]:
    metrics_path = metrics_path_for(run_dir)
    steps, values = load_eval_curve(metrics_path)
    best = max(values)
    final = values[-1]
    return (
        {
            "method_key": method.key,
            "method": method.label,
            "seed": seed,
            "auc": normalized_auc(steps, values),
            "final_success": final,
            "best_success": best,
            "best_to_final_drop": best - final,
            "hit90": first_hit90(steps, values),
            "evaluation_points": len(steps),
            "run_dir": run_dir.as_posix(),
            "metrics_path": metrics_path.as_posix(),
        },
        values,
    )


def aggregate_rows(
    per_seed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for method_key in METHOD_ORDER:
        rows = [row for row in per_seed if row["method_key"] == method_key]
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{method_key}: incomplete seed coverage")
        output: dict[str, object] = {
            "method_key": method_key,
            "method": rows[0]["method"],
            "seed_count": len(rows),
        }
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            values = [float(row[field]) for row in rows]
            output[f"{field}_mean"] = mean(values)
            output[f"{field}_sample_sd"] = sample_sd(values)
        hits = [int(row["hit90"]) for row in rows if row["hit90"] is not None]
        output["hit90_seed_count"] = len(hits)
        output["hit90_median_reached"] = None if not hits else float(median(hits))
        result.append(output)
    return result


def paired_comparison_rows(
    per_seed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    index = {
        (str(row["method_key"]), int(row["seed"])): row
        for row in per_seed
    }
    result: list[dict[str, object]] = []
    for contrast_key, label, left, right in CONTRASTS:
        differences = [
            float(index[(left, seed)]["auc"]) - float(index[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        lower, upper, bootstrap_count = exhaustive_paired_bootstrap(differences)
        sign_flip_p, assignment_count = exact_two_sided_sign_flip(differences)
        leave_one_out = [
            mean([value for index_, value in enumerate(differences) if index_ != omitted])
            for omitted in range(5)
        ]
        row: dict[str, object] = {
            "contrast_key": contrast_key,
            "contrast": label,
            "left_method_key": left,
            "right_method_key": right,
            "mean_auc_difference": mean(differences),
            "sample_sd_auc_difference": sample_sd(differences),
            "bootstrap_ci95_lower": lower,
            "bootstrap_ci95_upper": upper,
            "bootstrap_resample_count": bootstrap_count,
            "exact_two_sided_sign_flip_pvalue": sign_flip_p,
            "sign_flip_assignment_count": assignment_count,
            "positive_seed_count": sum(value > 1e-12 for value in differences),
            "zero_seed_count": sum(abs(value) <= 1e-12 for value in differences),
            "negative_seed_count": sum(value < -1e-12 for value in differences),
            "leave_one_seed_out_mean_min": min(leave_one_out),
            "leave_one_seed_out_mean_max": max(leave_one_out),
        }
        for seed, difference in zip(SEEDS, differences):
            row[f"seed_{seed}_auc_difference"] = difference
        result.append(row)
    return result


def mean_curve_rows(
    curves: Mapping[tuple[str, int], Sequence[float]],
    labels: Mapping[str, str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, step in enumerate(EVAL_STEPS):
        row: dict[str, object] = {"step": step}
        for method_key in METHOD_ORDER:
            values = [float(curves[(method_key, seed)][index]) for seed in SEEDS]
            row[f"{method_key}_mean_success"] = mean(values)
            row[f"{method_key}_sample_sd_success"] = sample_sd(values)
            row[f"{method_key}_label"] = labels[method_key]
        result.append(row)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != fields:
                raise ValueError(f"{path}: inconsistent row schema")
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def analyze(methods: Sequence[MethodSpec], output_dir: Path) -> dict[str, object]:
    if [method.key for method in methods] != list(METHOD_ORDER):
        raise ValueError("methods must be ordered as her, main, replay")
    runs = {method.key: discover_runs(method.root) for method in methods}
    per_seed: list[dict[str, object]] = []
    curves: dict[tuple[str, int], list[float]] = {}
    for method in methods:
        for seed in SEEDS:
            row, values = analyze_run(method, seed, runs[method.key][seed])
            row["run_dir"] = f"data/{method.key}/seed_{seed}"
            row["metrics_path"] = (
                f"data/{method.key}/seed_{seed}/metrics_selected_10k.csv"
            )
            per_seed.append(row)
            curves[(method.key, seed)] = values
    aggregate = aggregate_rows(per_seed)
    paired = paired_comparison_rows(per_seed)
    labels = {method.key: method.label for method in methods}
    mean_curves = mean_curve_rows(curves, labels)
    payload: dict[str, object] = {
        "schema_version": 1,
        "task": "drawer_close",
        "seeds": list(SEEDS),
        "evaluation_steps": list(EVAL_STEPS),
        "evaluation_field": PRIMARY_SUCCESS_FIELD,
        "auc_convention": (
            "observed-only trapezoid normalized by (2,000,000 - 10,000)"
        ),
        "smoothing": False,
        "interpolation": False,
        "forward_fill": False,
        "extrapolation": False,
        "bootstrap": "ordered exhaustive paired bootstrap, 5^5=3125",
        "exact_sign_flip": "all 2^5=32 assignments",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "mean_curves": mean_curves,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_results.csv", per_seed)
    write_csv(output_dir / "aggregate_results.csv", aggregate)
    write_csv(output_dir / "paired_comparisons.csv", paired)
    write_csv(output_dir / "exact_grid_mean_curves.csv", mean_curves)
    write_json(output_dir / "analysis_results.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--her-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = (
        MethodSpec("her", "DrawerClose DDPG HER", args.her_root.resolve()),
        MethodSpec(
            "main",
            "DrawerClose DDPG HER + SABER Main",
            args.main_root.resolve(),
        ),
        MethodSpec(
            "replay",
            "DrawerClose DDPG Replay-Action Query",
            args.replay_root.resolve(),
        ),
    )
    result = analyze(methods, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": "analyzed",
                "seed_runs": len(result["per_seed"]),
                "evaluation_points_per_seed": len(EVAL_STEPS),
                "output": args.output_dir.resolve().as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
