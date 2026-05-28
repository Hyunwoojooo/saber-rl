#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = EXPERIMENT_ROOT / "experiment" / "FetchSlide"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "analysis" / "FetchSlide"
SABER_METHOD = "b3_critic_only"
COMPARATORS = [
    "b0",
    "b3_original",
    "b3_critic_only_random",
    "b3_critic_only_shuffled",
    "td_error",
    "per_style",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SABER FetchSlide runs.")
    parser.add_argument("--input_root", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = collect_summaries(input_root)
    write_csv(output_root / "by_seed.csv", rows, by_seed_fields())
    aggregate = aggregate_rows(rows)
    write_csv(output_root / "aggregate.csv", aggregate, aggregate_fields())
    paired = paired_rows(rows)
    paired_b0 = [row for row in paired if row.get("comparator") == "b0"]
    write_csv(output_root / "paired_vs_b0.csv", paired_b0, paired_fields())
    write_csv(output_root / "paired_vs_controls.csv", paired, paired_fields())
    runtime = runtime_rows(rows)
    write_csv(output_root / "runtime_pareto.csv", runtime, runtime_fields())
    write_report(output_root / "report.md", rows, aggregate, paired, runtime)
    print(output_root)
    return 0


def collect_summaries(root: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "task": "FetchSlide",
                "method": payload.get("method", summary_path.parents[1].name),
                "variant": payload.get("control_mode", "none"),
                "seed": int(payload.get("seed", -1)),
                "run_dir": payload.get("run_dir", str(summary_path.parent)),
                "final_success": as_float(payload.get("final_success")),
                "best_success": as_float(payload.get("best_success")),
                "best_step": as_int(payload.get("best_step")),
                "auc_eval_mean": as_float(payload.get("auc_eval_mean")),
                "auc_trapz_0_to_total": as_float(payload.get("auc_trapz_0_to_total")),
                "first_hit_0.25": payload.get("first_hit_0.25"),
                "first_hit_0.50": payload.get("first_hit_0.50"),
                "first_hit_0.75": payload.get("first_hit_0.75"),
                "first_hit_0.90": payload.get("first_hit_0.90"),
                "first_hit_1.00": payload.get("first_hit_1.00"),
                "drop_from_best_to_final": as_float(payload.get("drop_from_best_to_final")),
                "last_quarter_auc": as_float(payload.get("last_quarter_auc")),
                "final_window_success_mean": as_float(payload.get("final_window_success_mean")),
                "final_window_success_std": as_float(payload.get("final_window_success_std")),
                "runtime_total_sec": as_float(payload.get("runtime_total_sec")),
                "steps_per_sec": as_float(payload.get("steps_per_sec")),
                "local_search_max": as_int(payload.get("local_search_max")),
            }
        )
    return rows


def aggregate_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    out = []
    for method, items in sorted(groups.items()):
        record: Dict[str, Any] = {"task": "FetchSlide", "method": method, "n": len(items)}
        for field in [
            "final_success",
            "best_success",
            "auc_eval_mean",
            "auc_trapz_0_to_total",
            "drop_from_best_to_final",
            "last_quarter_auc",
            "runtime_total_sec",
            "steps_per_sec",
        ]:
            values = finite_values(item.get(field) for item in items)
            record[f"{field}_mean"] = mean(values) if values else ""
            record[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        for threshold in ("0.25", "0.50", "0.75", "0.90", "1.00"):
            hits = [as_int(item.get(f"first_hit_{threshold}")) for item in items]
            hits = [hit for hit in hits if hit is not None]
            record[f"hit_count_{threshold}"] = len(hits)
            record[f"mean_hit_step_{threshold}"] = mean(hits) if hits else ""
        out.append(record)
    return out


def paired_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    by_method_seed = {(row["method"], row["seed"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows if row["method"] == SABER_METHOD})
    out = []
    for comparator in COMPARATORS:
        for seed in seeds:
            saber = by_method_seed.get((SABER_METHOD, seed))
            other = by_method_seed.get((comparator, seed))
            if not saber or not other:
                continue
            out.append(
                {
                    "task": "FetchSlide",
                    "seed": seed,
                    "comparator": comparator,
                    "final_diff": diff(saber, other, "final_success"),
                    "AUC_diff": diff(saber, other, "auc_eval_mean"),
                    "drop_advantage": diff(other, saber, "drop_from_best_to_final"),
                    "first_hit_0.50_diff": hit_diff(saber, other, "first_hit_0.50"),
                }
            )
    return out


def runtime_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[row["seed"]][row["method"]] = row
    out = []
    for seed, methods in sorted(by_seed.items()):
        b0 = methods.get("b0")
        base = methods.get(SABER_METHOD)
        for method, row in sorted(methods.items()):
            out.append(
                {
                    "task": "FetchSlide",
                    "seed": seed,
                    "method": method,
                    "runtime_total_sec": row.get("runtime_total_sec"),
                    "steps_per_sec": row.get("steps_per_sec"),
                    "overhead_vs_B0": ratio(row, b0, "runtime_total_sec"),
                    "overhead_vs_SABER": ratio(row, base, "runtime_total_sec"),
                    "AUC_retention_vs_SABER": ratio(row, base, "auc_eval_mean"),
                    "final_retention_vs_SABER": ratio(row, base, "final_success"),
                }
            )
    return out


def write_report(
    path: Path,
    rows: list[Dict[str, Any]],
    aggregate: list[Dict[str, Any]],
    paired: list[Dict[str, Any]],
    runtime: list[Dict[str, Any]],
) -> None:
    lines = [
        "# FetchSlide SABER Report",
        "",
        f"Runs parsed: {len(rows)}",
        "",
        "## Main Performance",
        "",
        "| method | n | final mean | AUC mean | drop mean | steps/sec mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {method} | {n} | {final} | {auc} | {drop} | {sps} |".format(
                method=row["method"],
                n=row["n"],
                final=fmt(row.get("final_success_mean")),
                auc=fmt(row.get("auc_eval_mean_mean")),
                drop=fmt(row.get("drop_from_best_to_final_mean")),
                sps=fmt(row.get("steps_per_sec_mean")),
            )
        )
    lines.extend(["", "## Paired Vs Controls", ""])
    if paired:
        grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for row in paired:
            grouped[row["comparator"]].append(row)
        lines.extend(["| comparator | n | final diff | AUC diff | drop advantage |", "| --- | ---: | ---: | ---: | ---: |"])
        for comp, items in sorted(grouped.items()):
            lines.append(
                f"| {comp} | {len(items)} | {fmt(mean_field(items, 'final_diff'))} | "
                f"{fmt(mean_field(items, 'AUC_diff'))} | {fmt(mean_field(items, 'drop_advantage'))} |"
            )
    else:
        lines.append("No seed-paired SABER comparisons available yet.")
    lines.extend(["", "## Runtime", ""])
    if runtime:
        lines.extend(["| method | seed | overhead vs B0 | AUC retention vs SABER |", "| --- | ---: | ---: | ---: |"])
        for row in runtime[:50]:
            lines.append(
                f"| {row['method']} | {row['seed']} | {fmt(row.get('overhead_vs_B0'))} | "
                f"{fmt(row.get('AUC_retention_vs_SABER'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def by_seed_fields() -> list[str]:
    return [
        "task",
        "method",
        "variant",
        "seed",
        "run_dir",
        "final_success",
        "best_success",
        "best_step",
        "auc_eval_mean",
        "auc_trapz_0_to_total",
        "first_hit_0.25",
        "first_hit_0.50",
        "first_hit_0.75",
        "first_hit_0.90",
        "first_hit_1.00",
        "drop_from_best_to_final",
        "last_quarter_auc",
        "final_window_success_mean",
        "final_window_success_std",
        "runtime_total_sec",
        "steps_per_sec",
        "local_search_max",
    ]


def aggregate_fields() -> list[str]:
    base = ["task", "method", "n"]
    metrics = [
        "final_success",
        "best_success",
        "auc_eval_mean",
        "auc_trapz_0_to_total",
        "drop_from_best_to_final",
        "last_quarter_auc",
        "runtime_total_sec",
        "steps_per_sec",
    ]
    fields = base[:]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    for threshold in ("0.25", "0.50", "0.75", "0.90", "1.00"):
        fields.extend([f"hit_count_{threshold}", f"mean_hit_step_{threshold}"])
    return fields


def paired_fields() -> list[str]:
    return ["task", "seed", "comparator", "final_diff", "AUC_diff", "drop_advantage", "first_hit_0.50_diff"]


def runtime_fields() -> list[str]:
    return [
        "task",
        "seed",
        "method",
        "runtime_total_sec",
        "steps_per_sec",
        "overhead_vs_B0",
        "overhead_vs_SABER",
        "AUC_retention_vs_SABER",
        "final_retention_vs_SABER",
    ]


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def as_int(value: Any) -> int | None:
    try:
        if value in {"", None}:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def finite_values(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        f = as_float(value)
        if math.isfinite(f):
            out.append(f)
    return out


def diff(a: Dict[str, Any], b: Dict[str, Any], field: str) -> float | str:
    av = as_float(a.get(field))
    bv = as_float(b.get(field))
    if math.isfinite(av) and math.isfinite(bv):
        return av - bv
    return ""


def ratio(a: Dict[str, Any] | None, b: Dict[str, Any] | None, field: str) -> float | str:
    if not a or not b:
        return ""
    av = as_float(a.get(field))
    bv = as_float(b.get(field))
    if math.isfinite(av) and math.isfinite(bv) and abs(bv) > 1e-12:
        return av / bv
    return ""


def hit_diff(a: Dict[str, Any], b: Dict[str, Any], field: str) -> int | str:
    av = as_int(a.get(field))
    bv = as_int(b.get(field))
    if av is None or bv is None:
        return ""
    return av - bv


def mean_field(rows: list[Dict[str, Any]], field: str) -> float | str:
    values = finite_values(row.get(field) for row in rows)
    return mean(values) if values else ""


def fmt(value: Any) -> str:
    f = as_float(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
