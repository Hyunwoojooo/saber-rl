#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "analysis" / "result_pipeline" / "config.json"
DEFAULT_OUTPUT = ROOT / "analysis" / "result_pipeline" / "output"


METRICS = [
    "final_success",
    "best_success",
    "success_auc",
    "last_quarter_auc",
    "steps_per_sec",
    "runtime_total_sec",
]


PLOT_GROUP_LABELS = {
    "fpp_ddpg": "FetchPickAndPlace DDPG",
    "fpp_td3": "FetchPickAndPlace TD3",
    "fpp_sac": "FetchPickAndPlace SAC",
    "fpp_ddpg_per": "FetchPickAndPlace DDPG PER Controls",
    "fetchslide_ddpg": "FetchSlide DDPG",
    "fetchslide_td3": "FetchSlide TD3",
    "fetchslide_sac": "FetchSlide SAC",
    "drawer_ddpg": "Meta-World DrawerOpen DDPG",
    "drawer_td3": "Meta-World DrawerOpen TD3",
    "drawer_controls": "Meta-World DrawerOpen TD3 Controls",
}

METHOD_ORDER = {
    "HER": 0,
    "HER+SABER": 1,
    "Random-support": 2,
    "Shuffled-weight": 3,
    "Random-fixed-count": 4,
    "Random-fixed-count k94": 4,
    "HER+PER-style": 5,
    "HER+PER-full": 6,
    "Actor-only weighting": 7,
    "Actor+critic weighting": 8,
}

METHOD_COLORS = {
    "HER": "#4C566A",
    "HER+SABER": "#006D77",
    "Random-support": "#B56576",
    "Shuffled-weight": "#6D597A",
    "Random-fixed-count": "#D97706",
    "Random-fixed-count k94": "#D97706",
    "HER+PER-style": "#2563EB",
    "HER+PER-full": "#9333EA",
    "Actor-only weighting": "#2F855A",
    "Actor+critic weighting": "#C2410C",
}


@dataclass(frozen=True)
class MethodRow:
    id: str
    task: str
    backbone: str
    method: str
    plot_group: str
    expected_total_steps: int | None
    seeds: tuple[int, ...]
    summary_globs: tuple[str, ...]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None and data[key] != "":
            return data[key]
    return None


def extract_seed(path: Path, summary: dict[str, Any]) -> int | None:
    seeds = summary.get("seeds")
    if isinstance(seeds, dict):
        seed = seeds.get("env_seed") or seeds.get("numpy_seed") or seeds.get("torch_seed")
        if seed is not None:
            return int(seed)
    for pattern in [r"_s(\d+)_", r"_s(\d+)\b"]:
        match = re.search(pattern, str(path))
        if match:
            return int(match.group(1))
    return None


def parse_method_rows(config: dict[str, Any]) -> list[MethodRow]:
    rows: list[MethodRow] = []
    for raw in config["rows"]:
        rows.append(
            MethodRow(
                id=str(raw["id"]),
                task=str(raw["task"]),
                backbone=str(raw["backbone"]),
                method=str(raw["method"]),
                plot_group=str(raw.get("plot_group") or raw["id"]),
                expected_total_steps=(
                    int(raw["expected_total_steps"]) if raw.get("expected_total_steps") is not None else None
                ),
                seeds=tuple(int(s) for s in raw.get("seeds", [])),
                summary_globs=tuple(str(g) for g in raw["summary_globs"]),
            )
        )
    return rows


def expand_summaries(row: MethodRow) -> list[Path]:
    paths: list[Path] = []
    for pattern in row.summary_globs:
        paths.extend(ROOT.glob(pattern))
    return sorted(set(paths))


def select_curve_file(run_dir: Path) -> Path | None:
    for name in ["metrics_selected_10k.csv", "metrics.csv"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def per_seed_record(row: MethodRow, summary_path: Path) -> dict[str, Any] | None:
    summary = load_json(summary_path)
    seed = extract_seed(summary_path, summary)
    if seed is None:
        return None
    if row.seeds and seed not in row.seeds:
        return None
    total_steps = as_float(summary.get("total_steps"))
    if row.expected_total_steps is not None and total_steps is not None:
        if int(total_steps) != int(row.expected_total_steps):
            return None
    record = {
        "method_id": row.id,
        "task": row.task,
        "backbone": row.backbone,
        "method": row.method,
        "plot_group": row.plot_group,
        "seed": seed,
        "summary_path": str(summary_path.relative_to(ROOT)),
        "run_dir": str(summary_path.parent.relative_to(ROOT)),
        "total_steps": int(total_steps) if total_steps is not None else "",
        "final_success": as_float(
            first_present(
                summary,
                [
                    "final_eval_success",
                    "final_success",
                    "final_eval_goal_success",
                    "final_eval_env_success",
                    "final_eval_goal_success_final",
                    "final_eval_env_success_final",
                ],
            )
        ),
        "best_success": as_float(first_present(summary, ["best_eval_success", "best_success"])),
        "success_auc": as_float(first_present(summary, ["eval_AUC_success", "auc_eval_mean", "auc_trapz_0_to_total"])),
        "last_quarter_auc": as_float(summary.get("last_quarter_auc")),
        "hit90_step": as_float(summary.get("first_hit_0.90")),
        "runtime_total_sec": as_float(summary.get("runtime_total_sec")),
        "steps_per_sec": as_float(first_present(summary, ["final_steps_per_sec", "steps_per_sec"])),
    }
    return record


def collect_per_seed(rows: list[MethodRow]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        seen_seeds: set[int] = set()
        for summary_path in expand_summaries(row):
            record = per_seed_record(row, summary_path)
            if record is None:
                continue
            seed = int(record["seed"])
            if seed in seen_seeds:
                warnings.append(f"duplicate seed ignored: {row.id} seed={seed} path={summary_path}")
                continue
            seen_seeds.add(seed)
            records.append(record)
        expected = set(row.seeds)
        missing = sorted(expected - seen_seeds)
        if missing:
            warnings.append(f"missing seeds for {row.id}: {missing}")
    return sorted(records, key=lambda r: (r["task"], r["backbone"], r["method_id"], int(r["seed"]))), warnings


def numeric_values(records: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = as_float(record.get(metric))
        if value is not None:
            values.append(value)
    return values


def mean_or_blank(values: list[float]) -> float | str:
    return mean(values) if values else ""


def std_or_blank(values: list[float]) -> float | str:
    return stdev(values) if len(values) >= 2 else ""


def fmt(value: Any, ndigits: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{ndigits}f}".rstrip("0").rstrip(".")
    return str(value)


def stable_offset(text: str, modulo: int = 1000000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def aggregate_records(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in per_seed:
        grouped[str(record["method_id"])].append(record)
    out: list[dict[str, Any]] = []
    for method_id, records in sorted(grouped.items()):
        first = records[0]
        row: dict[str, Any] = {
            "method_id": method_id,
            "task": first["task"],
            "backbone": first["backbone"],
            "method": first["method"],
            "n": len(records),
            "seeds": " ".join(str(r["seed"]) for r in sorted(records, key=lambda r: int(r["seed"]))),
        }
        for metric in METRICS:
            values = numeric_values(records, metric)
            row[f"{metric}_mean"] = mean_or_blank(values)
            row[f"{metric}_std"] = std_or_blank(values)
        hit_values = numeric_values(records, "hit90_step")
        row["hit90_reached"] = len(hit_values)
        row["median_hit90_step"] = median(hit_values) if hit_values else ""
        out.append(row)
    return out


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = p * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_mean_ci(values: list[float], iterations: int, ci: float, rng: random.Random) -> tuple[float, float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0], values[0]
    samples: list[float] = []
    n = len(values)
    for _ in range(iterations):
        samples.append(mean(values[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    alpha = (1.0 - ci) / 2.0
    return mean(values), percentile(samples, alpha), percentile(samples, 1.0 - alpha)


def aggregate_bootstrap(per_seed: list[dict[str, Any]], iterations: int, ci: float, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in per_seed:
        grouped[str(record["method_id"])].append(record)
    out: list[dict[str, Any]] = []
    for method_id, records in sorted(grouped.items()):
        rng = random.Random(seed + stable_offset(method_id))
        first = records[0]
        for metric in METRICS:
            values = numeric_values(records, metric)
            ci_values = bootstrap_mean_ci(values, iterations, ci, rng)
            if ci_values is None:
                continue
            estimate, lo, hi = ci_values
            out.append(
                {
                    "method_id": method_id,
                    "task": first["task"],
                    "backbone": first["backbone"],
                    "method": first["method"],
                    "metric": metric,
                    "n": len(values),
                    "mean": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci": ci,
                    "bootstrap_iterations": iterations,
                }
            )
    return out


def paired_differences(config: dict[str, Any], per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_seed = {(str(r["method_id"]), int(r["seed"])): r for r in per_seed}
    out: list[dict[str, Any]] = []
    for pair in config.get("pairs", []):
        baseline = str(pair["baseline"])
        treatment = str(pair["treatment"])
        seeds = sorted({s for method, s in by_method_seed if method in {baseline, treatment}})
        for seed in seeds:
            b = by_method_seed.get((baseline, seed))
            t = by_method_seed.get((treatment, seed))
            if b is None or t is None:
                continue
            for metric in METRICS:
                bv = as_float(b.get(metric))
                tv = as_float(t.get(metric))
                if bv is None or tv is None:
                    continue
                out.append(
                    {
                        "pair_id": pair["id"],
                        "baseline_id": baseline,
                        "treatment_id": treatment,
                        "task": t["task"],
                        "backbone": t["backbone"],
                        "seed": seed,
                        "metric": metric,
                        "baseline": bv,
                        "treatment": tv,
                        "delta": tv - bv,
                    }
                )
    return out


def paired_bootstrap(paired: list[dict[str, Any]], iterations: int, ci: float, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in paired:
        key = (str(row["pair_id"]), str(row["metric"]))
        grouped[key].append(float(row["delta"]))
        meta[key] = row
    out: list[dict[str, Any]] = []
    for key, deltas in sorted(grouped.items()):
        pair_id, metric = key
        rng = random.Random(seed + stable_offset(pair_id + metric))
        ci_values = bootstrap_mean_ci(deltas, iterations, ci, rng)
        if ci_values is None:
            continue
        estimate, lo, hi = ci_values
        row_meta = meta[key]
        out.append(
            {
                "pair_id": pair_id,
                "baseline_id": row_meta["baseline_id"],
                "treatment_id": row_meta["treatment_id"],
                "task": row_meta["task"],
                "backbone": row_meta["backbone"],
                "metric": metric,
                "n": len(deltas),
                "mean_delta": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "ci": ci,
                "bootstrap_iterations": iterations,
            }
        )
    return out


def curve_success_value(row: dict[str, str]) -> float | None:
    for key in ["eval/success_mean", "eval/goal_success_mean", "eval/env_success_mean"]:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


LOG_EVAL_RE = re.compile(
    r"\bstep=(?P<step>\d+)\b.*?\beval_goal=(?P<goal>-?\d+(?:\.\d+)?)\b.*?\beval_env=(?P<env>-?\d+(?:\.\d+)?)\b"
)


def load_curve_points_from_log(record: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = ROOT / str(record["run_dir"])
    log_file = run_dir / "run.log"
    if not log_file.exists():
        return []
    by_step: dict[int, float] = {}
    for line in log_file.read_text(errors="replace").splitlines():
        match = LOG_EVAL_RE.search(line)
        if match is None:
            continue
        step = int(match.group("step"))
        success = as_float(match.group("env"))
        if success is None:
            success = as_float(match.group("goal"))
        if success is None:
            continue
        by_step[step] = success
    points: list[dict[str, Any]] = []
    for step, success in sorted(by_step.items()):
        points.append(
            {
                "method_id": record["method_id"],
                "task": record["task"],
                "backbone": record["backbone"],
                "method": record["method"],
                "plot_group": record["plot_group"],
                "seed": record["seed"],
                "step": step,
                "success": success,
                "curve_file": str(log_file.relative_to(ROOT)),
            }
        )
    return points


def load_curve_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = ROOT / str(record["run_dir"])
    curve_file = select_curve_file(run_dir)
    if curve_file is None:
        return load_curve_points_from_log(record)
    by_step: dict[int, float] = {}
    with curve_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = as_float(row.get("step"))
            success = curve_success_value(row)
            if step is None or success is None:
                continue
            by_step[int(step)] = success
    points: list[dict[str, Any]] = []
    for step, success in sorted(by_step.items()):
        points.append(
            {
                "method_id": record["method_id"],
                "task": record["task"],
                "backbone": record["backbone"],
                "method": record["method"],
                "plot_group": record["plot_group"],
                "seed": record["seed"],
                "step": step,
                "success": success,
                "curve_file": str(curve_file.relative_to(ROOT)),
            }
        )
    return points or load_curve_points_from_log(record)


def collect_curves(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for record in per_seed:
        curves.extend(load_curve_points(record))
    return curves


def aggregate_curves(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        grouped[(str(row["method_id"]), int(row["step"]))].append(row)
    out: list[dict[str, Any]] = []
    for (_method_id, step), rows in sorted(grouped.items()):
        values = [float(r["success"]) for r in rows]
        first = rows[0]
        out.append(
            {
                "method_id": first["method_id"],
                "task": first["task"],
                "backbone": first["backbone"],
                "method": first["method"],
                "plot_group": first["plot_group"],
                "step": step,
                "n": len(values),
                "success_mean": mean(values),
                "success_std": stdev(values) if len(values) >= 2 else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row.get(field), 4) for field in fields) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    aggregate: list[dict[str, Any]],
    paired_ci: list[dict[str, Any]],
    warnings: list[str],
    figures: list[Path],
) -> None:
    main_fields = [
        "method_id",
        "n",
        "final_success_mean",
        "final_success_std",
        "success_auc_mean",
        "success_auc_std",
        "median_hit90_step",
        "hit90_reached",
        "steps_per_sec_mean",
    ]
    delta_fields = ["pair_id", "metric", "n", "mean_delta", "ci_low", "ci_high"]
    final_deltas = [row for row in paired_ci if row["metric"] in {"final_success", "success_auc", "last_quarter_auc"}]
    lines = [
        "# SABER Result Statistics Pipeline Report",
        "",
        "Generated from configured `summary.json` and metric CSV artifacts.",
        "",
        "## Aggregate Metrics",
        "",
        markdown_table(aggregate, main_fields),
        "",
        "## Paired Bootstrap CI",
        "",
        markdown_table(final_deltas, delta_fields),
        "",
        "## Figures",
        "",
    ]
    if figures:
        lines.extend(f"- `{figure.relative_to(ROOT)}`" for figure in figures)
    else:
        lines.append("- No figures were generated. Install matplotlib or Pillow to enable PNG plots.")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def method_sort_key(method: str) -> tuple[int, str]:
    return METHOD_ORDER.get(method, 100), method


def format_step_tick(step: int) -> str:
    if step >= 1_000_000:
        value = step / 1_000_000
        return f"{value:g}M"
    if step >= 1000:
        return f"{step // 1000}k"
    return str(step)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def step_ticks(max_step: int) -> list[int]:
    if max_step <= 1_000_000:
        return [0, 250_000, 500_000, 750_000, 1_000_000]
    if max_step <= 2_000_000:
        return [0, 500_000, 1_000_000, 1_500_000, 2_000_000]
    return [int(round(max_step * frac)) for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]


def group_raw_curves(curves: list[dict[str, Any]]) -> dict[str, dict[str, dict[int, list[dict[str, Any]]]]]:
    grouped: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in curves:
        grouped[str(row["plot_group"])][str(row["method"])][int(row["seed"])].append(row)
    return grouped


def plot_curves(
    curve_aggregate: list[dict[str, Any]],
    output_dir: Path,
    curves: list[dict[str, Any]] | None = None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return plot_curves_with_pillow(curve_aggregate, output_dir, curves)
    by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in curve_aggregate:
        by_group[str(row["plot_group"])][str(row["method"])].append(row)
    raw_by_group = group_raw_curves(curves or [])
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    for group, methods in sorted(by_group.items()):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        all_steps = [int(row["step"]) for rows in methods.values() for row in rows]
        max_step = max(all_steps) if all_steps else 1
        for method, rows in sorted(methods.items(), key=lambda item: method_sort_key(item[0])):
            rows = sorted(rows, key=lambda r: int(r["step"]))
            color = METHOD_COLORS.get(method, None)
            raw_seed_rows = raw_by_group.get(group, {}).get(method, {})
            for seed_rows in raw_seed_rows.values():
                seed_rows = sorted(seed_rows, key=lambda r: int(r["step"]))
                seed_xs = [int(r["step"]) for r in seed_rows]
                seed_ys = [float(r["success"]) for r in seed_rows]
                ax.plot(seed_xs, seed_ys, color=color, alpha=0.13, linewidth=0.7, zorder=1)
            xs = [int(r["step"]) for r in rows]
            ys = [float(r["success_mean"]) for r in rows]
            stds = [float(r["success_std"]) for r in rows]
            ax.plot(xs, ys, label=method, color=color, linewidth=2.4, zorder=3)
            if any(s > 0 for s in stds):
                lo = [max(0.0, y - s) for y, s in zip(ys, stds)]
                hi = [min(1.0, y + s) for y, s in zip(ys, stds)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.set_xlabel("Environment steps", fontsize=12)
        ax.set_ylabel("Evaluation success", fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_step)
        ticks = [tick for tick in step_ticks(max_step) if tick <= max_step]
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_step_tick(tick) for tick in ticks], fontsize=11)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
        fig.tight_layout(pad=0.25)
        base = figure_dir / f"{group}_success_curve"
        pdf_path = base.with_suffix(".pdf")
        png_path = base.with_suffix(".png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.extend([pdf_path, png_path])
    return figures


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (0, 92, 175),
        (200, 70, 0),
        (0, 135, 85),
        (150, 80, 170),
        (215, 160, 0),
        (80, 80, 80),
    ]
    return palette[index % len(palette)]


def plot_curves_with_pillow(
    curve_aggregate: list[dict[str, Any]],
    output_dir: Path,
    curves: list[dict[str, Any]] | None = None,
) -> list[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return []

    by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in curve_aggregate:
        by_group[str(row["plot_group"])][str(row["method"])].append(row)
    raw_by_group = group_raw_curves(curves or [])

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    width, height = 1800, 1080
    left, right, top, bottom = 150, 60, 46, 132
    plot_w = width - left - right
    plot_h = height - top - bottom
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold_font_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 32)
        small_font = ImageFont.truetype(str(font_path), 28)
        bold_font = ImageFont.truetype(str(bold_font_path), 32)
    except Exception:
        font = small_font = bold_font = ImageFont.load_default()

    for group, methods in sorted(by_group.items()):
        all_steps = [int(row["step"]) for rows in methods.values() for row in rows]
        if not all_steps:
            continue
        x_min, x_max = 0, max(all_steps)
        if x_max <= 0:
            x_max = 1

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        for y_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = top + int((1.0 - y_value) * plot_h)
            draw.line([(left, y), (width - right, y)], fill=(225, 225, 225), width=2)
            draw.text((42, y - 18), f"{y_value:.2f}", fill=(70, 70, 70), font=small_font)

        for step in step_ticks(x_max):
            if step > x_max:
                continue
            x = left + int((step - x_min) / (x_max - x_min) * plot_w)
            draw.line([(x, top), (x, height - bottom)], fill=(238, 238, 238), width=1)
            label = format_step_tick(step)
            draw.text((x - 28, height - bottom + 22), label, fill=(70, 70, 70), font=small_font)

        draw.line([(left, top), (left, height - bottom)], fill=(40, 40, 40), width=3)
        draw.line([(left, height - bottom), (width - right, height - bottom)], fill=(40, 40, 40), width=3)
        draw.text((left + plot_w // 2 - 140, height - 46), "Environment steps", fill=(30, 30, 30), font=font)
        draw.text((left, 8), "Evaluation success", fill=(30, 30, 30), font=font)

        for method_index, (method, rows) in enumerate(sorted(methods.items(), key=lambda item: method_sort_key(item[0]))):
            color = hex_to_rgb(METHOD_COLORS.get(method, "#4C566A"))
            rows = sorted(rows, key=lambda r: int(r["step"]))

            def point(step: int, success: float) -> tuple[int, int]:
                x = left + int((step - x_min) / (x_max - x_min) * plot_w)
                y = top + int((1.0 - success) * plot_h)
                return x, y

            raw_seed_rows = raw_by_group.get(group, {}).get(method, {})
            raw_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            raw_draw = ImageDraw.Draw(raw_overlay)
            for seed_rows in raw_seed_rows.values():
                seed_rows = sorted(seed_rows, key=lambda r: int(r["step"]))
                seed_points = [point(int(row["step"]), float(row["success"])) for row in seed_rows]
                if len(seed_points) >= 2:
                    raw_draw.line(seed_points, fill=(*color, 40), width=3)
            image = Image.alpha_composite(image.convert("RGBA"), raw_overlay).convert("RGB")
            draw = ImageDraw.Draw(image)

            points = [point(int(row["step"]), float(row["success_mean"])) for row in rows]
            if len(points) >= 2:
                band_hi = [
                    point(int(row["step"]), min(1.0, float(row["success_mean"]) + float(row["success_std"])))
                    for row in rows
                ]
                band_lo = [
                    point(int(row["step"]), max(0.0, float(row["success_mean"]) - float(row["success_std"])))
                    for row in reversed(rows)
                ]
                overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.polygon(band_hi + band_lo, fill=(*color, 28))
                image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.line(points, fill=color, width=7)
            elif len(points) == 1:
                x, y = points[0]
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)

            legend_x = left + 26
            legend_y = top + 26 + method_index * 44
            draw.line([(legend_x, legend_y + 16), (legend_x + 54, legend_y + 16)], fill=color, width=7)
            draw.text((legend_x + 68, legend_y), method, fill=(35, 35, 35), font=bold_font)

        base = figure_dir / f"{group}_success_curve"
        png_path = base.with_suffix(".png")
        pdf_path = base.with_suffix(".pdf")
        image.save(png_path)
        image.save(pdf_path, "PDF", resolution=300.0)
        figures.extend([pdf_path, png_path])

    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SABER result tables, paired CIs, and learning curves.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_json(args.config)
    rows = parse_method_rows(config)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed, warnings = collect_per_seed(rows)
    aggregate = aggregate_records(per_seed)
    boot_cfg = config.get("bootstrap", {})
    iterations = int(boot_cfg.get("iterations", 10000))
    ci = float(boot_cfg.get("ci", 0.95))
    seed = int(boot_cfg.get("seed", 27027))
    aggregate_ci = aggregate_bootstrap(per_seed, iterations, ci, seed)
    paired = paired_differences(config, per_seed)
    paired_ci = paired_bootstrap(paired, iterations, ci, seed)
    curves = collect_curves(per_seed)
    curve_aggregate = aggregate_curves(curves)

    write_csv(output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(output_dir / "aggregate_bootstrap_ci.csv", aggregate_ci)
    write_csv(output_dir / "paired_seed_differences.csv", paired)
    write_csv(output_dir / "paired_bootstrap_ci.csv", paired_ci)
    write_csv(output_dir / "learning_curves_long.csv", curves)
    write_csv(output_dir / "learning_curves_aggregate.csv", curve_aggregate)
    figures = plot_curves(curve_aggregate, output_dir, curves)
    write_report(output_dir / "report.md", aggregate, paired_ci, warnings, figures)

    print(f"wrote {len(per_seed)} per-seed rows")
    print(f"wrote {len(aggregate)} aggregate rows")
    print(f"wrote {len(paired)} paired seed-difference rows")
    print(f"wrote {len(curves)} learning curve points")
    print(f"wrote {len(figures)} figures")
    if warnings:
        print(f"warnings: {len(warnings)}")
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
