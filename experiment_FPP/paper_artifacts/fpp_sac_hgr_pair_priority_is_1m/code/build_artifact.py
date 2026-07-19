#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


ARTIFACT_NAME = "fpp_sac_hgr_pair_priority_is_1m"
PAPER_LABEL = "HGR (matched SAC+HER)"
SOURCE_COMMIT_SHORT = "c2be4e8"
SOURCE_COMMIT_FULL = "c2be4e85a451af61dabba8e54517e9d6d4c1cbf2"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "3aa0c06576d9a5ecae26e538792d2be6f577bf72ff8af173b528dac408f89772"
)
FROZEN_SOURCE_MANIFEST_LINES = 28
TOTAL_STEPS = 1_000_000
HGR_EVAL_FREQ = 25_000
HGR_EVAL_GRID = tuple(range(HGR_EVAL_FREQ, TOTAL_STEPS + 1, HGR_EVAL_FREQ))
COMMON_EVAL_FREQ = 50_000
COMMON_EVAL_GRID = tuple(range(COMMON_EVAL_FREQ, TOTAL_STEPS + 1, COMMON_EVAL_FREQ))
SEEDS = (1, 2, 3, 4, 5)

HGR_REQUIRED_METRICS = (
    "hgr/relabel_fraction",
    "hgr/fallback_fraction",
    "hgr/beta_episode",
    "hgr/beta_goal",
    "hgr/is_weight_mean",
    "hgr/is_weight_min",
    "hgr/episode_priority_mean",
    "hgr/pair_priority_count",
    "hgr/priority_safety_replacement_rate",
    "runtime/hgr_sampling_sec_per_update",
)

BACKBONE_FIELDS = (
    "task_name",
    "env_id",
    "reward_type",
    "max_episode_steps",
    "total_steps",
    "start_steps",
    "warmup_steps",
    "batch_size",
    "buffer_capacity",
    "her_ratio",
    "updates_per_step",
    "gamma",
    "tau",
    "actor_lr",
    "critic_lr",
    "hidden_dim",
    "noise_std",
    "sac_alpha",
    "sac_alpha_lr",
    "sac_auto_alpha",
    "sac_target_entropy",
    "eval_freq",
    "eval_episodes",
    "success_terminates",
    "timeout_bootstrap",
)

AGGREGATE_FIELDS = (
    "method_key",
    "paper_label",
    "n_seeds",
    "total_steps",
    "eval_grid_start",
    "eval_grid_end",
    "eval_grid_frequency",
    "eval_grid_points",
    "final_success_mean",
    "final_success_sample_std",
    "best_success_mean",
    "best_success_sample_std",
    "success_auc_mean",
    "success_auc_sample_std",
    "best_final_drop_mean",
    "best_final_drop_sample_std",
    "hit90_count",
    "hit90_total",
    "hit90_median_step_among_hits",
    "runtime_hours_mean",
    "runtime_hours_sample_std",
    *tuple(
        value
        for metric in HGR_REQUIRED_METRICS
        for value in (f"final_{metric}_mean", f"final_{metric}_sample_std")
    ),
)

PER_SEED_FIELDS = (
    "method_key",
    "paper_label",
    "seed",
    "run_name",
    "source_run_path",
    "total_steps",
    "eval_grid_points",
    "final_success",
    "best_success",
    "first_best_step",
    "success_auc",
    "best_final_drop",
    "first_hit_0.90",
    "hit90_reached",
    "runtime_total_sec",
    "runtime_hours",
    "steps_per_sec",
    *tuple(f"final_{metric}" for metric in HGR_REQUIRED_METRICS),
)

PAIRED_FIELDS = (
    "comparison",
    "reference_label",
    "metric",
    "delta_definition",
    "direction_favoring_hgr",
    "n_pairs",
    "paired_seeds",
    "seed_deltas",
    "mean_delta",
    "sample_std_delta",
    "ci95_low",
    "ci95_high",
    "bootstrap_method",
    "bootstrap_distribution_size",
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    root: Path
    run_name: Callable[[int], str]
    metrics_name: str
    logical_root: str
    eval_grid: tuple[int, ...]
    audit_full_summary: bool


@dataclass
class RunResult:
    method_key: str
    paper_label: str
    seed: int
    run_name: str
    source_run_path: str
    run_dir: Path
    config: dict[str, object]
    summary: dict[str, object]
    metrics_fieldnames: list[str]
    selected_rows: list[dict[str, str]]
    eval_rows: list[dict[str, str]]
    eval_steps: list[int]
    eval_values: list[float]
    final_success: float
    best_success: float
    first_best_step: int
    success_auc: float
    best_final_drop: float
    first_hit_90: int | None
    runtime_total_sec: float
    steps_per_sec: float
    final_hgr_metrics: dict[str, float]


def _repo_root_from(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "experiment_FPP").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {path}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sanitize_json(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item, replacements) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        result = value
        for source, replacement in replacements.items():
            result = result.replace(source, replacement)
        return result
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: object, *, context: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Missing numeric value for {context}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {context}: {value!r}")
    return result


def _optional_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _summary_number(summary: dict[str, object], keys: Sequence[str], context: str) -> float:
    for key in keys:
        if key in summary and summary[key] not in (None, ""):
            return _finite_float(summary[key], context=f"{context}.{key}")
    raise ValueError(f"Missing summary value for {context}; tried {list(keys)}")


def _seed_from_config(config: dict[str, object]) -> int:
    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("config.seeds must be an object")
    return int(_finite_float(seeds.get("env_seed"), context="config.seeds.env_seed"))


def _load_metrics(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _extract_grid_rows(
    path: Path,
    eval_grid: Sequence[int],
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    fieldnames, rows = _load_metrics(path)
    if "step" not in fieldnames or "eval/env_success_mean" not in fieldnames:
        raise ValueError(f"Missing step/evaluation columns: {path}")

    selected_rows: list[dict[str, str]] = []
    eval_by_step: dict[int, dict[str, str]] = {}
    selected_counts = {step: 0 for step in eval_grid}
    grid_set = set(eval_grid)
    for row_index, row in enumerate(rows, start=2):
        step = int(_finite_float(row.get("step"), context=f"{path}:{row_index}:step"))
        if step in grid_set:
            selected_rows.append(row)
            selected_counts[step] += 1
        eval_value = _optional_float(row.get("eval/env_success_mean"))
        if math.isfinite(eval_value):
            if step in eval_by_step and step in grid_set:
                raise ValueError(f"Duplicate evaluation row at step {step}: {path}")
            if step in grid_set:
                eval_by_step[step] = row

    missing = [step for step in eval_grid if step not in eval_by_step]
    bad_selected_counts = {
        step: count for step, count in selected_counts.items() if count != 2
    }
    if missing or bad_selected_counts:
        raise ValueError(
            f"Exact grid validation failed for {path}: missing={missing}, "
            f"selected_row_counts={bad_selected_counts}"
        )
    if len(selected_rows) != 2 * len(eval_grid):
        raise ValueError(
            f"Expected {2 * len(eval_grid)} selected rows in {path}, got {len(selected_rows)}"
        )
    eval_rows = [eval_by_step[step] for step in eval_grid]
    return fieldnames, selected_rows, eval_rows


def _normalized_trapezoid(
    steps: Sequence[int], values: Sequence[float], expected_grid: Sequence[int]
) -> float:
    if tuple(steps) != tuple(expected_grid):
        raise ValueError(f"AUC requires the exact observed grid {tuple(expected_grid)}")
    if len(values) != len(expected_grid):
        raise ValueError(f"AUC requires {len(expected_grid)} values, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AUC values must all be finite")
    area = sum(
        0.5 * (values[index] + values[index + 1]) * (steps[index + 1] - steps[index])
        for index in range(len(steps) - 1)
    )
    return area / float(steps[-1] - steps[0])


def _first_hit(steps: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    for step, value in zip(steps, values):
        if value >= threshold:
            return int(step)
    return None


def _assert_close(actual: float, expected: float, context: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{context} mismatch: recomputed={actual}, summary={expected}")


def _load_run(spec: MethodSpec, seed: int) -> RunResult:
    run_name = spec.run_name(seed)
    run_dir = spec.root / run_name
    required = [
        run_dir / "summary.json",
        run_dir / "config.json",
        run_dir / spec.metrics_name,
    ]
    if spec.key == "hgr":
        required.append(run_dir / "env_snapshot.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files for {run_name}: {missing}")

    summary = _read_json(run_dir / "summary.json")
    config = _read_json(run_dir / "config.json")
    config_steps = int(_finite_float(config.get("total_steps"), context=f"{run_name}.config.total_steps"))
    summary_steps = int(_finite_float(summary.get("total_steps"), context=f"{run_name}.summary.total_steps"))
    if config_steps != TOTAL_STEPS or summary_steps != TOTAL_STEPS:
        raise ValueError(
            f"Run {run_name} is not complete at 1M: config={config_steps}, summary={summary_steps}"
        )
    if _seed_from_config(config) != seed:
        raise ValueError(f"Seed mismatch for {run_name}")
    if str(config.get("run_name", run_name)) != run_name:
        raise ValueError(f"Run name mismatch in config for {run_name}")

    fieldnames, selected_rows, eval_rows = _extract_grid_rows(
        run_dir / spec.metrics_name,
        spec.eval_grid,
    )
    eval_steps = [int(float(row["step"])) for row in eval_rows]
    eval_values = [
        _finite_float(row["eval/env_success_mean"], context=f"{run_name}.eval@{step}")
        for step, row in zip(eval_steps, eval_rows)
    ]
    if not all(0.0 <= value <= 1.0 for value in eval_values):
        raise ValueError(f"Evaluation success outside [0, 1] for {run_name}")

    final_success = eval_values[-1]
    best_success = max(eval_values)
    first_best_step = eval_steps[eval_values.index(best_success)]
    success_auc = _normalized_trapezoid(eval_steps, eval_values, spec.eval_grid)
    first_hit_90 = _first_hit(eval_steps, eval_values, 0.90)
    best_final_drop = best_success - final_success

    summary_final = _summary_number(
        summary,
        ("final_eval_env_success", "final_eval_success"),
        run_name,
    )
    summary_best = _summary_number(
        summary,
        ("best_eval_env_success", "best_eval_success"),
        run_name,
    )
    summary_auc = _summary_number(summary, ("eval_AUC_success",), run_name)
    _assert_close(final_success, summary_final, f"{run_name}.final_success")
    if spec.audit_full_summary:
        _assert_close(best_success, summary_best, f"{run_name}.best_success")
        _assert_close(success_auc, summary_auc, f"{run_name}.success_auc")

        summary_hit = summary.get("first_hit_0.90", "")
        expected_hit = None if summary_hit in (None, "") else int(float(summary_hit))
        if first_hit_90 != expected_hit:
            raise ValueError(
                f"{run_name}.first_hit_0.90 mismatch: "
                f"recomputed={first_hit_90}, summary={expected_hit}"
            )

    final_hgr_metrics: dict[str, float] = {}
    if spec.key == "hgr":
        missing_columns = [metric for metric in HGR_REQUIRED_METRICS if metric not in fieldnames]
        if missing_columns:
            raise ValueError(f"Missing required HGR columns in {run_name}: {missing_columns}")
        final_hgr_metrics = {
            metric: _finite_float(
                eval_rows[-1].get(metric),
                context=f"{run_name}.final.{metric}",
            )
            for metric in HGR_REQUIRED_METRICS
        }
        for row, step in zip(eval_rows, eval_steps):
            for metric in HGR_REQUIRED_METRICS:
                _finite_float(row.get(metric), context=f"{run_name}@{step}.{metric}")

    runtime_total_sec = _summary_number(summary, ("runtime_total_sec",), run_name)
    steps_per_sec = _summary_number(summary, ("final_steps_per_sec",), run_name)
    logical_path = f"{spec.logical_root}/{run_name}"
    return RunResult(
        method_key=spec.key,
        paper_label=spec.label,
        seed=seed,
        run_name=run_name,
        source_run_path=logical_path,
        run_dir=run_dir,
        config=config,
        summary=summary,
        metrics_fieldnames=fieldnames,
        selected_rows=selected_rows,
        eval_rows=eval_rows,
        eval_steps=eval_steps,
        eval_values=eval_values,
        final_success=final_success,
        best_success=best_success,
        first_best_step=first_best_step,
        success_auc=success_auc,
        best_final_drop=best_final_drop,
        first_hit_90=first_hit_90,
        runtime_total_sec=runtime_total_sec,
        steps_per_sec=steps_per_sec,
        final_hgr_metrics=final_hgr_metrics,
    )


def _validate_matched_backbone(results: Sequence[RunResult]) -> None:
    by_seed: dict[int, list[RunResult]] = {seed: [] for seed in SEEDS}
    for result in results:
        by_seed[result.seed].append(result)
    for seed, seed_results in by_seed.items():
        if {result.method_key for result in seed_results} != {"hgr", "her", "saber"}:
            raise ValueError(f"Missing matched methods for seed {seed}")
        reference = next(result for result in seed_results if result.method_key == "her")
        for result in seed_results:
            for field in BACKBONE_FIELDS:
                if result.config.get(field) != reference.config.get(field):
                    raise ValueError(
                        f"Backbone mismatch seed={seed} field={field}: "
                        f"HER={reference.config.get(field)!r}, {result.method_key}={result.config.get(field)!r}"
                    )
            if _seed_from_config(result.config) != seed:
                raise ValueError(f"Seed mismatch after backbone validation: {result.run_name}")


def _sample_std(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return statistics.mean(finite)


def _format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".15g")
    return str(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_number(row.get(field)) for field in fieldnames})


def _write_selected_metrics(path: Path, result: RunResult) -> None:
    rows = [{field: row.get(field, "") for field in result.metrics_fieldnames} for row in result.selected_rows]
    _write_csv(path, result.metrics_fieldnames, rows)


def _write_hgr_metrics(path: Path, result: RunResult) -> None:
    all_hgr = [field for field in result.metrics_fieldnames if field.startswith("hgr/")]
    extras = sorted(field for field in all_hgr if field not in HGR_REQUIRED_METRICS)
    fieldnames = [
        "step",
        "eval/env_success_mean",
        *HGR_REQUIRED_METRICS,
        *extras,
    ]
    rows = [{field: row.get(field, "") for field in fieldnames} for row in result.eval_rows]
    _write_csv(path, fieldnames, rows)


def _aggregate_rows(results: Sequence[RunResult]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method_key in ("hgr", "her", "saber"):
        group = sorted(
            (result for result in results if result.method_key == method_key),
            key=lambda result: result.seed,
        )
        hits = [result.first_hit_90 for result in group if result.first_hit_90 is not None]
        eval_grid = tuple(group[0].eval_steps)
        if any(tuple(result.eval_steps) != eval_grid for result in group):
            raise ValueError(f"Inconsistent evaluation grid within method {method_key}")
        row: dict[str, object] = {
            "method_key": method_key,
            "paper_label": group[0].paper_label,
            "n_seeds": len(group),
            "total_steps": TOTAL_STEPS,
            "eval_grid_start": eval_grid[0],
            "eval_grid_end": eval_grid[-1],
            "eval_grid_frequency": eval_grid[1] - eval_grid[0],
            "eval_grid_points": len(eval_grid),
            "final_success_mean": _mean(result.final_success for result in group),
            "final_success_sample_std": _sample_std(result.final_success for result in group),
            "best_success_mean": _mean(result.best_success for result in group),
            "best_success_sample_std": _sample_std(result.best_success for result in group),
            "success_auc_mean": _mean(result.success_auc for result in group),
            "success_auc_sample_std": _sample_std(result.success_auc for result in group),
            "best_final_drop_mean": _mean(result.best_final_drop for result in group),
            "best_final_drop_sample_std": _sample_std(result.best_final_drop for result in group),
            "hit90_count": len(hits),
            "hit90_total": len(group),
            "hit90_median_step_among_hits": statistics.median(hits) if hits else None,
            "runtime_hours_mean": _mean(result.runtime_total_sec / 3600.0 for result in group),
            "runtime_hours_sample_std": _sample_std(
                result.runtime_total_sec / 3600.0 for result in group
            ),
        }
        for metric in HGR_REQUIRED_METRICS:
            values = [result.final_hgr_metrics.get(metric, math.nan) for result in group]
            row[f"final_{metric}_mean"] = _mean(values)
            row[f"final_{metric}_sample_std"] = _sample_std(values)
        output.append(row)
    return output


def _per_seed_rows(results: Sequence[RunResult]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    order = {"hgr": 0, "her": 1, "saber": 2}
    for result in sorted(results, key=lambda item: (order[item.method_key], item.seed)):
        row: dict[str, object] = {
            "method_key": result.method_key,
            "paper_label": result.paper_label,
            "seed": result.seed,
            "run_name": result.run_name,
            "source_run_path": result.source_run_path,
            "total_steps": TOTAL_STEPS,
            "eval_grid_points": len(result.eval_steps),
            "final_success": result.final_success,
            "best_success": result.best_success,
            "first_best_step": result.first_best_step,
            "success_auc": result.success_auc,
            "best_final_drop": result.best_final_drop,
            "first_hit_0.90": result.first_hit_90,
            "hit90_reached": int(result.first_hit_90 is not None),
            "runtime_total_sec": result.runtime_total_sec,
            "runtime_hours": result.runtime_total_sec / 3600.0,
            "steps_per_sec": result.steps_per_sec,
        }
        for metric in HGR_REQUIRED_METRICS:
            row[f"final_{metric}"] = result.final_hgr_metrics.get(metric)
        output.append(row)
    return output


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _exact_paired_bootstrap_ci(deltas: Sequence[float]) -> tuple[float, float, int]:
    finite = [value for value in deltas if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan, 0
    n = len(finite)
    bootstrap_means = [
        statistics.mean(finite[index] for index in indices)
        for indices in itertools.product(range(n), repeat=n)
    ]
    return (
        _quantile(bootstrap_means, 0.025),
        _quantile(bootstrap_means, 0.975),
        len(bootstrap_means),
    )


def _metrics_on_grid(result: RunResult, grid: Sequence[int]) -> dict[str, float | None]:
    by_step = dict(zip(result.eval_steps, result.eval_values))
    missing = [step for step in grid if step not in by_step]
    if missing:
        raise ValueError(f"Missing common-grid points for {result.run_name}: {missing}")
    values = [by_step[step] for step in grid]
    final_success = values[-1]
    best_success = max(values)
    first_hit_90 = _first_hit(grid, values, 0.90)
    return {
        "final_success": final_success,
        "best_success": best_success,
        "success_auc": _normalized_trapezoid(grid, values, grid),
        "best_final_drop": best_success - final_success,
        "hit90_reached": float(first_hit_90 is not None),
        "first_hit_0.90_steps_both_hit": (
            float(first_hit_90) if first_hit_90 is not None else None
        ),
    }


def _paired_rows(results: Sequence[RunResult]) -> list[dict[str, object]]:
    by_method_seed = {(result.method_key, result.seed): result for result in results}
    common_metrics = {
        (method_key, seed): _metrics_on_grid(result, COMMON_EVAL_GRID)
        for (method_key, seed), result in by_method_seed.items()
    }
    metric_specs: tuple[tuple[str, str], ...] = (
        ("final_success", "higher"),
        ("best_success", "higher"),
        ("success_auc", "higher"),
        ("best_final_drop", "lower"),
        ("hit90_reached", "higher"),
        ("first_hit_0.90_steps_both_hit", "lower"),
    )
    output: list[dict[str, object]] = []
    for reference_key in ("her", "saber"):
        reference_label = by_method_seed[(reference_key, SEEDS[0])].paper_label
        comparison = f"HGR-minus-{reference_key.upper()}"
        for metric_name, favorable_direction in metric_specs:
            pairs: list[tuple[int, float]] = []
            for seed in SEEDS:
                hgr_value = common_metrics[("hgr", seed)][metric_name]
                reference_value = common_metrics[(reference_key, seed)][metric_name]
                if hgr_value is None or reference_value is None:
                    continue
                delta = float(hgr_value) - float(reference_value)
                if math.isfinite(delta):
                    pairs.append((seed, delta))
            deltas = [delta for _, delta in pairs]
            ci_low, ci_high, distribution_size = _exact_paired_bootstrap_ci(deltas)
            output.append(
                {
                    "comparison": comparison,
                    "reference_label": reference_label,
                    "metric": metric_name,
                    "delta_definition": (
                        f"{PAPER_LABEL} minus {reference_label} on common 50k grid"
                    ),
                    "direction_favoring_hgr": favorable_direction,
                    "n_pairs": len(pairs),
                    "paired_seeds": ";".join(str(seed) for seed, _ in pairs),
                    "seed_deltas": ";".join(
                        f"s{seed}:{format(delta, '.15g')}" for seed, delta in pairs
                    ),
                    "mean_delta": _mean(deltas),
                    "sample_std_delta": _sample_std(deltas),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_method": (
                        "paired nonparametric percentile; exhaustive enumeration of n^n resamples"
                    ),
                    "bootstrap_distribution_size": distribution_size,
                }
            )
    return output


def _markdown_number(value: object, digits: int = 3) -> str:
    if value in (None, ""):
        return "NA"
    numeric = _optional_float(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _find_row(rows: Sequence[dict[str, object]], key: str, value: str) -> dict[str, object]:
    return next(row for row in rows if str(row.get(key)) == value)


def _write_readme(
    path: Path,
    aggregate_rows: Sequence[dict[str, object]],
    paired_rows: Sequence[dict[str, object]],
) -> None:
    aggregate_lines = []
    for row in aggregate_rows:
        aggregate_lines.append(
            "| {label} | {final} +/- {final_std} | {best} +/- {best_std} | "
            "{auc} +/- {auc_std} | {drop} +/- {drop_std} | {hits}/{total} | {median} |".format(
                label=row["paper_label"],
                final=_markdown_number(row["final_success_mean"]),
                final_std=_markdown_number(row["final_success_sample_std"]),
                best=_markdown_number(row["best_success_mean"]),
                best_std=_markdown_number(row["best_success_sample_std"]),
                auc=_markdown_number(row["success_auc_mean"]),
                auc_std=_markdown_number(row["success_auc_sample_std"]),
                drop=_markdown_number(row["best_final_drop_mean"]),
                drop_std=_markdown_number(row["best_final_drop_sample_std"]),
                hits=row["hit90_count"],
                total=row["hit90_total"],
                median=(
                    f"{int(float(row['hit90_median_step_among_hits'])) // 1000}k"
                    if row["hit90_median_step_among_hits"] not in (None, "")
                    else "NA"
                ),
            )
        )

    key_pairs = [
        row
        for row in paired_rows
        if row["metric"] in {"final_success", "success_auc", "hit90_reached"}
    ]
    paired_lines = []
    for row in key_pairs:
        paired_lines.append(
            "| {comparison} | {metric} | {n} | {mean} | [{low}, {high}] |".format(
                comparison=row["comparison"],
                metric=row["metric"],
                n=row["n_pairs"],
                mean=_markdown_number(row["mean_delta"]),
                low=_markdown_number(row["ci95_low"]),
                high=_markdown_number(row["ci95_high"]),
            )
        )

    hgr = _find_row(aggregate_rows, "method_key", "hgr")
    diagnostics_lines = []
    for metric in HGR_REQUIRED_METRICS:
        diagnostics_lines.append(
            f"| `{metric}` | {_markdown_number(hgr[f'final_{metric}_mean'], 6)} | "
            f"{_markdown_number(hgr[f'final_{metric}_sample_std'], 6)} |"
        )

    boundary = (
        "> This is an algorithmically faithful HGR pair-priority and importance-correction\n"
        "> implementation transferred to a matched SAC+HER backbone. It is not an exact\n"
        "> reproduction of the original HGR paper’s environment and training configuration."
    )
    text = f"""# FetchPickAndPlace SAC+HER+HGR Pair-Priority + IS 1M Artifact

Paper label: **{PAPER_LABEL}**

{boundary}

## Scope

This artifact contains FetchPickAndPlace-v4 sparse-reward results for seeds 1-5
at 1,000,000 environment steps. HGR is compared with the matched SAC+HER
baseline and the main critic-only SABER SAC+HER run set. Backbone settings,
seed streams, configured evaluation frequency, and evaluation episode count are
validated before any output is written.

## Aggregate Results

All values after `+/-` are sample standard deviations across five seeds
(`statistics.stdev`, denominator `n-1`). HGR rows use the complete observed
25k grid (40 evaluations). The two reference artifacts preserve evaluations on
the common 50k grid (20 evaluations), so their aggregate rows use that grid.

| Method | Final success | Best success | Success AUC | Best-final drop | Hit90 | Hit90 median among hits |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(aggregate_lines)}

## Paired Comparisons

Each delta is HGR minus the same-seed reference. Confidence intervals are
two-sided 95% paired nonparametric percentile bootstrap intervals. With five
seeds, all `5^5 = 3,125` resamples are enumerated, so there is no Monte Carlo
seed or sampling error. Every paired metric is recomputed on the exact common
`{{50k, 100k, ..., 1M}}` grid (20 points); HGR's intervening 25k points are not
used in a paired delta. The full metric set is in `paired_vs_her_saber.csv`.

| Comparison | Metric | Pairs | Mean delta | 95% bootstrap CI |
|---|---|---:|---:|---:|
{chr(10).join(paired_lines)}

## Metric Definitions

- Final success is `eval/env_success_mean` at exactly 1,000,000 steps.
- HGR primary best success is the maximum over exactly
  `{{25k, 50k, ..., 1M}}`; HGR primary Success AUC is normalized trapezoidal
  area on those 40 observed points: `trapz(success, step) / (1,000,000 - 25,000)`.
- Reference aggregates and every paired comparison use exactly
  `{{50k, 100k, ..., 1M}}`; their AUC denominator is `1,000,000 - 50,000`.
  Missing points are rejected; no values are interpolated or carried forward.
- Best-final drop is `best_success - final_success`.
- Hit90 is the first observed grid step with success at least 0.90. The median
  uses only seeds that hit; the count always reports the denominator.
- HGR summary-native final, best, AUC, and Hit90 values are audited against the
  recomputed complete 25k grid. For the compact reference artifacts, final is
  audited against the summary while best, AUC, and Hit90 are recomputed on the
  preserved common 50k grid.

## Final HGR Diagnostics

The table uses each seed's 1M evaluation row. Full 25k-grid traces, including
all logged `hgr/*` fields, are stored in each `hgr_metrics_selected.csv`.

| Metric | Mean | Sample std |
|---|---:|---:|
{chr(10).join(diagnostics_lines)}

## Included Files

Each HGR seed directory contains only:

- `summary.json`
- `config.json`
- `env_snapshot.json`
- `metrics_selected_25k.csv` (80 source rows: train and evaluation rows at each grid step)
- `hgr_metrics_selected.csv` (40 evaluation rows)

Root analysis outputs are `aggregate_results.csv`, `per_seed_results.csv`, and
`paired_vs_her_saber.csv`. `completion_validation.json` records the strict 1M
completion gate, `input_manifest.json` hashes every input, and
`provenance/frozen_source.sha256` preserves the launch-time source manifest.
`code/build_artifact.py` rebuilds the artifact and `code/verify_artifact.py`
performs an independent integrity check.

The launch-time source manifest captures the exact frozen experiment files,
but an exact Python/package dependency lock was not captured at launch. The
saved `env_snapshot.json` records the runtime environment metadata that was
available; exact dependency-level reproduction therefore has this provenance
limitation.

Checkpoints, raw `metrics.csv`, raw JSONL, and logs are intentionally excluded.
`artifact_manifest.json` records SHA256 and byte size for every payload file.
The manifest excludes itself because a file cannot contain its own final hash;
this exclusion is explicit in the manifest schema.

## Source Runs

- HGR: `experiment_FPP/runs/sac_hgr_tdpriority_1m/`
- SAC+HER reference artifact: `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/`
- SAC+HER+SABER reference artifact: `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/`
- Exact launch-time HGR source commit: `{SOURCE_COMMIT_FULL}`
- The result commit may add a repository-root portability fix to `run.py`;
  the algorithm, trainer, and frozen configuration remain unchanged.

## Rebuild

From the repository root, with the HGR raw run root and compact SAC reference
artifacts available:

```bash
python3 experiment_FPP/paper_artifacts/{ARTIFACT_NAME}/code/build_artifact.py
python3 experiment_FPP/paper_artifacts/{ARTIFACT_NAME}/code/verify_artifact.py
```
"""
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, artifact_root: Path, results: Sequence[RunResult]) -> None:
    payload_files = sorted(
        candidate
        for candidate in artifact_root.rglob("*")
        if candidate.is_file() and candidate != path
    )
    files = [
        {
            "path": candidate.relative_to(artifact_root).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for candidate in payload_files
    ]
    source_runs = [
        {
            "method_key": result.method_key,
            "paper_label": result.paper_label,
            "seed": result.seed,
            "run_name": result.run_name,
            "source_run_path": result.source_run_path,
            "total_steps": TOTAL_STEPS,
            "eval_grid_points": len(result.eval_steps),
        }
        for result in sorted(results, key=lambda item: (item.method_key, item.seed))
    ]
    payload = {
        "schema_version": 1,
        "artifact": ARTIFACT_NAME,
        "paper_label": PAPER_LABEL,
        "source_commit": SOURCE_COMMIT_SHORT,
        "source_commit_full": SOURCE_COMMIT_FULL,
        "sample_std_ddof": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_scope": "all artifact payload files except artifact_manifest.json",
        "manifest_self_excluded": True,
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "source_runs": source_runs,
        "exclusions": ["checkpoint files", "raw metrics.csv", "raw JSONL", "logs"],
    }
    _write_json(path, payload)


def _write_input_manifest(path: Path, results: Sequence[RunResult]) -> None:
    entries: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: (item.method_key, item.seed)):
        names = ["summary.json", "config.json", result.run_dir.joinpath(
            "metrics.csv" if result.method_key == "hgr" else "metrics_selected_10k.csv"
        ).name]
        if result.method_key == "hgr":
            names.insert(2, "env_snapshot.json")
        for name in names:
            source = result.run_dir / name
            entries.append(
                {
                    "method_key": result.method_key,
                    "seed": result.seed,
                    "logical_path": f"{result.source_run_path}/{name}",
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact": ARTIFACT_NAME,
            "source_commit": SOURCE_COMMIT_FULL,
            "manifest_scope": "all JSON and metric inputs used to build or compare this artifact",
            "file_count": len(entries),
            "files": entries,
        },
    )


def _write_completion_validation(path: Path, hgr_results: Sequence[RunResult]) -> None:
    runs: list[dict[str, object]] = []
    for result in hgr_results:
        final_lite = result.summary.get("final_lite_metrics")
        if not isinstance(final_lite, dict):
            raise ValueError(f"Missing final_lite_metrics for {result.run_name}")
        total_updates = int(
            _finite_float(result.summary.get("total_updates"), context=f"{result.run_name}.updates")
        )
        episodes = int(
            _finite_float(result.summary.get("episodes"), context=f"{result.run_name}.episodes")
        )
        replay_size = int(
            _finite_float(
                result.summary.get("final_replay_size"), context=f"{result.run_name}.replay_size"
            )
        )
        hgr_tables = int(
            _finite_float(final_lite.get("hgr/pair_table_count"), context=f"{result.run_name}.tables")
        )
        hgr_pairs = int(
            _finite_float(final_lite.get("hgr/pair_priority_count"), context=f"{result.run_name}.pairs")
        )
        expected = {
            "total_steps": TOTAL_STEPS,
            "episodes": 20_000,
            "total_updates": 995_001,
            "final_replay_size": 300_000,
            "eval_points": len(HGR_EVAL_GRID),
            "hgr_tables": 6_000,
            "hgr_pairs": 7_650_000,
        }
        actual = {
            "total_steps": int(result.summary["total_steps"]),
            "episodes": episodes,
            "total_updates": total_updates,
            "final_replay_size": replay_size,
            "eval_points": len(result.eval_rows),
            "hgr_tables": hgr_tables,
            "hgr_pairs": hgr_pairs,
        }
        if actual != expected:
            raise ValueError(f"Completion gate failed for {result.run_name}: {actual} != {expected}")
        runs.append(
            {
                "seed": result.seed,
                "run_name": result.run_name,
                **actual,
                "exact_eval_grid": True,
                "final_success": result.final_success,
                "best_success": result.best_success,
                "success_auc": result.success_auc,
                "summary_sha256": _sha256(result.run_dir / "summary.json"),
                "ok": True,
            }
        )
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact": ARTIFACT_NAME,
            "source_commit": SOURCE_COMMIT_FULL,
            "strict_gate": "experiment_FPP/sac_hgr_tdpriority_1m/validate_results.py",
            "ok": True,
            "runs": runs,
        },
    )


def _copy_hgr_seed_payload(result: RunResult, artifact_root: Path) -> None:
    destination = artifact_root / result.run_name
    destination.mkdir(parents=True, exist_ok=True)
    source_repo_root = result.run_dir.parents[3]
    replacements = {str(source_repo_root): "${REPO_ROOT}"}
    for name in ("summary.json", "config.json", "env_snapshot.json"):
        payload = _sanitize_json(_read_json(result.run_dir / name), replacements)
        _write_json(destination / name, payload)
    _write_selected_metrics(destination / "metrics_selected_25k.csv", result)
    _write_hgr_metrics(destination / "hgr_metrics_selected.csv", result)


def _assert_output_seed_contents(artifact_root: Path, hgr_results: Sequence[RunResult]) -> None:
    expected = {
        "summary.json",
        "config.json",
        "env_snapshot.json",
        "metrics_selected_25k.csv",
        "hgr_metrics_selected.csv",
    }
    for result in hgr_results:
        destination = artifact_root / result.run_name
        actual = {path.name for path in destination.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError(
                f"Unexpected seed payload for {result.run_name}: expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )


def _parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_repo = _repo_root_from(script_path)
    parser = argparse.ArgumentParser(
        description="Build the FPP SAC-HGR pair-priority + IS 1M paper artifact."
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--hgr-runs-root",
        type=Path,
        default=None,
        help="Defaults to <repo>/experiment_FPP/runs/sac_hgr_tdpriority_1m",
    )
    parser.add_argument("--her-runs-root", type=Path, default=None)
    parser.add_argument("--saber-runs-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--frozen-source-manifest",
        type=Path,
        default=None,
        help="Launch-time SHA256 manifest; defaults to the copy already in the artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    hgr_root = (
        args.hgr_runs_root.resolve()
        if args.hgr_runs_root is not None
        else repo_root / "experiment_FPP/runs/sac_hgr_tdpriority_1m"
    )
    her_root = (
        args.her_runs_root.resolve()
        if args.her_runs_root is not None
        else repo_root
        / "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512"
    )
    saber_root = (
        args.saber_runs_root.resolve()
        if args.saber_runs_root is not None
        else repo_root
        / "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512"
    )
    artifact_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / f"experiment_FPP/paper_artifacts/{ARTIFACT_NAME}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    specs = (
        MethodSpec(
            key="hgr",
            label=PAPER_LABEL,
            root=hgr_root,
            run_name=lambda seed: f"fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000",
            metrics_name="metrics.csv",
            logical_root="experiment_FPP/runs/sac_hgr_tdpriority_1m",
            eval_grid=HGR_EVAL_GRID,
            audit_full_summary=True,
        ),
        MethodSpec(
            key="her",
            label="SAC+HER",
            root=her_root,
            run_name=lambda seed: f"fetch_pick_and_place_sac_her_baseline_ls512_s{seed}_1000000_ls512",
            metrics_name="metrics_selected_10k.csv",
            logical_root=(
                "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
                "sac_her_baseline_ls512"
            ),
            eval_grid=COMMON_EVAL_GRID,
            audit_full_summary=False,
        ),
        MethodSpec(
            key="saber",
            label="SABER (matched SAC+HER)",
            root=saber_root,
            run_name=lambda seed: (
                f"fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s{seed}_1000000_ls512"
            ),
            metrics_name="metrics_selected_10k.csv",
            logical_root=(
                "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
                "sac_saber_critic_only_v4_ls512"
            ),
            eval_grid=COMMON_EVAL_GRID,
            audit_full_summary=False,
        ),
    )
    results = [_load_run(spec, seed) for spec in specs for seed in SEEDS]
    _validate_matched_backbone(results)

    hgr_results = sorted(
        (result for result in results if result.method_key == "hgr"),
        key=lambda result: result.seed,
    )
    for result in hgr_results:
        _copy_hgr_seed_payload(result, artifact_root)
    _assert_output_seed_contents(artifact_root, hgr_results)

    aggregate_rows = _aggregate_rows(results)
    per_seed_rows = _per_seed_rows(results)
    paired_rows = _paired_rows(results)
    _write_csv(artifact_root / "aggregate_results.csv", AGGREGATE_FIELDS, aggregate_rows)
    _write_csv(artifact_root / "per_seed_results.csv", PER_SEED_FIELDS, per_seed_rows)
    _write_csv(artifact_root / "paired_vs_her_saber.csv", PAIRED_FIELDS, paired_rows)
    _write_readme(artifact_root / "README.md", aggregate_rows, paired_rows)
    _write_input_manifest(artifact_root / "input_manifest.json", results)
    _write_completion_validation(artifact_root / "completion_validation.json", hgr_results)

    frozen_source = (
        args.frozen_source_manifest.resolve()
        if args.frozen_source_manifest is not None
        else Path(__file__).resolve().parents[1] / "provenance/frozen_source.sha256"
    )
    if not frozen_source.is_file():
        raise FileNotFoundError(f"Missing frozen source manifest: {frozen_source}")
    frozen_sha256 = _sha256(frozen_source)
    if frozen_sha256 != FROZEN_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "Frozen source manifest SHA256 mismatch: "
            f"expected={FROZEN_SOURCE_MANIFEST_SHA256}, actual={frozen_sha256}"
        )
    frozen_lines = len(frozen_source.read_text(encoding="utf-8").splitlines())
    if frozen_lines != FROZEN_SOURCE_MANIFEST_LINES:
        raise ValueError(
            "Frozen source manifest line-count mismatch: "
            f"expected={FROZEN_SOURCE_MANIFEST_LINES}, actual={frozen_lines}"
        )
    frozen_destination = artifact_root / "provenance/frozen_source.sha256"
    frozen_destination.parent.mkdir(parents=True, exist_ok=True)
    if frozen_source.resolve() != frozen_destination.resolve():
        shutil.copyfile(frozen_source, frozen_destination)
    _write_manifest(artifact_root / "artifact_manifest.json", artifact_root, results)

    print(f"artifact={artifact_root}")
    print(f"hgr_runs_verified={len(hgr_results)}")
    print(f"comparison_runs_verified={len(results) - len(hgr_results)}")
    print(f"hgr_eval_grid_points_per_run={len(HGR_EVAL_GRID)}")
    print(f"common_comparison_grid_points_per_run={len(COMMON_EVAL_GRID)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
