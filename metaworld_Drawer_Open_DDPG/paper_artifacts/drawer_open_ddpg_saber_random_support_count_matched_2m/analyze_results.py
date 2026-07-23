#!/usr/bin/env python3
"""Analysis helpers for the DrawerOpen DDPG Random-Support paper artifact."""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 2_000_001, 10_000))
PRIMARY_FIELD = "eval/env_success_mean"
METHODS = ("her", "main", "random")
METHOD_LABELS = {
    "her": "DrawerOpen DDPG HER",
    "main": "DrawerOpen DDPG HER + SABER Main",
    "random": "DrawerOpen DDPG Random-Support (Count Matched)",
}
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


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(float(value) for value in values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        math.fsum((float(value) - center) ** 2 for value in values)
        / (len(values) - 1)
    )


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_exact_curve(path: Path) -> list[float]:
    """Load only finite, observed eval/env_success_mean rows on the exact grid."""

    found: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "step" not in fields or PRIMARY_FIELD not in fields:
            raise ValueError(f"{path}: missing step or {PRIMARY_FIELD}")
        for row in reader:
            raw = row.get(PRIMARY_FIELD)
            if raw in ("", None):
                continue
            step_value = finite_float(row.get("step"))
            success = finite_float(raw)
            if step_value is None or success is None:
                raise ValueError(f"{path}: non-finite evaluation row")
            step = int(step_value)
            if float(step) != step_value:
                raise ValueError(f"{path}: non-integral evaluation step {step_value}")
            if step not in EVAL_STEPS:
                raise ValueError(f"{path}: unexpected evaluation step {step}")
            if step in found:
                raise ValueError(f"{path}: duplicate evaluation step {step}")
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"{path}: success outside [0, 1] at {step}")
            found[step] = success
    if tuple(found) != EVAL_STEPS:
        missing = sorted(set(EVAL_STEPS) - set(found))
        raise ValueError(f"{path}: exact evaluation grid mismatch; missing={missing[:10]}")
    return [found[step] for step in EVAL_STEPS]


def normalized_auc(values: Sequence[float]) -> float:
    if len(values) != len(EVAL_STEPS):
        raise ValueError("AUC requires the exact 200-point evaluation grid")
    area = math.fsum(
        (right - left) * (y_left + y_right) / 2.0
        for left, right, y_left, y_right in zip(
            EVAL_STEPS, EVAL_STEPS[1:], values, values[1:]
        )
    )
    return area / (EVAL_STEPS[-1] - EVAL_STEPS[0])


def run_metrics(values: Sequence[float]) -> dict[str, object]:
    best = max(values)
    hit90 = next(
        (step for step, value in zip(EVAL_STEPS, values) if value >= 0.90), None
    )
    return {
        "evaluation_points": len(values),
        "auc": normalized_auc(values),
        "final_success": values[-1],
        "best_success": best,
        "best_to_final_drop": best - values[-1],
        "hit90": hit90,
    }


def exhaustive_paired_statistics(
    differences: Sequence[float],
) -> dict[str, object]:
    values = [float(value) for value in differences]
    if len(values) != len(SEEDS):
        raise ValueError("paired analysis requires exactly five seeds")
    bootstrap_means = [
        mean([values[index] for index in indices])
        for indices in itertools.product(range(5), repeat=5)
    ]
    sign_flip_statistics = [
        abs(mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1.0, 1.0), repeat=5)
    ]
    observed = abs(mean(values))
    leave_one_out = [
        mean([value for index, value in enumerate(values) if index != omitted])
        for omitted in range(5)
    ]
    result: dict[str, object] = {
        "mean_auc_difference": mean(values),
        "sample_sd_auc_difference": sample_sd(values),
        "bootstrap_ci95_lower": percentile(bootstrap_means, 0.025),
        "bootstrap_ci95_upper": percentile(bootstrap_means, 0.975),
        "bootstrap_resample_count": len(bootstrap_means),
        "exact_two_sided_sign_flip_pvalue": (
            sum(value >= observed - 1e-15 for value in sign_flip_statistics)
            / len(sign_flip_statistics)
        ),
        "sign_flip_assignment_count": len(sign_flip_statistics),
        "positive_seed_count": sum(value > 1e-12 for value in values),
        "zero_seed_count": sum(abs(value) <= 1e-12 for value in values),
        "negative_seed_count": sum(value < -1e-12 for value in values),
        "leave_one_seed_out_mean_min": min(leave_one_out),
        "leave_one_seed_out_mean_max": max(leave_one_out),
    }
    for seed, value in zip(SEEDS, values):
        result[f"seed_{seed}_auc_difference"] = value
    return result


def analyze_curves(
    curves: Mapping[tuple[str, int], Sequence[float]],
) -> dict[str, list[dict[str, object]]]:
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(curves) != expected:
        raise ValueError("analysis requires HER/Main/Random seeds 1-5")

    per_seed: list[dict[str, object]] = []
    metrics: dict[tuple[str, int], dict[str, object]] = {}
    for method in METHODS:
        for seed in SEEDS:
            result = run_metrics(curves[(method, seed)])
            metrics[(method, seed)] = result
            per_seed.append(
                {
                    "method_key": method,
                    "method": METHOD_LABELS[method],
                    "seed": seed,
                    **result,
                }
            )

    aggregate: list[dict[str, object]] = []
    for method in METHODS:
        row: dict[str, object] = {
            "method_key": method,
            "method": METHOD_LABELS[method],
            "seed_count": len(SEEDS),
        }
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            values = [float(metrics[(method, seed)][field]) for seed in SEEDS]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_sample_sd"] = sample_sd(values)
        hits = [
            int(metrics[(method, seed)]["hit90"])
            for seed in SEEDS
            if metrics[(method, seed)]["hit90"] is not None
        ]
        row["hit90_seed_count"] = len(hits)
        row["hit90_median_reached"] = None if not hits else median(hits)
        row["hit90_mean_reached"] = None if not hits else mean(hits)
        aggregate.append(row)

    paired: list[dict[str, object]] = []
    for key, label, left, right in CONTRASTS:
        differences = [
            float(metrics[(left, seed)]["auc"])
            - float(metrics[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        paired.append(
            {
                "contrast_key": key,
                "contrast": label,
                "left_method": left,
                "right_method": right,
                "paired_seed_count": len(SEEDS),
                **exhaustive_paired_statistics(differences),
            }
        )

    mean_curves: list[dict[str, object]] = []
    for index, step in enumerate(EVAL_STEPS):
        row: dict[str, object] = {"step": step}
        for method in METHODS:
            values = [float(curves[(method, seed)][index]) for seed in SEEDS]
            row[f"{method}_mean_success"] = mean(values)
            row[f"{method}_sample_sd_success"] = sample_sd(values)
            row[f"{method}_method"] = METHOD_LABELS[method]
        mean_curves.append(row)

    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "mean_curves": mean_curves,
    }
