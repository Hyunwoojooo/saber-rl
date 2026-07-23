#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BATCH_KEYS = [
    "support_positive_rate",
    "td_abs_before_q1_mean",
    "td_abs_before_q2_mean",
    "td_abs_before_mean",
    "td_abs_after_q1_mean",
    "td_abs_after_q2_mean",
    "td_abs_after_mean",
    "td_abs_reduction_mean",
    "psi_td_abs_pearson",
    "psi_td_abs_spearman",
    "top10_td_support_overlap",
    "top20_td_support_overlap",
    "top10_td_support_enrichment",
    "top20_td_support_enrichment",
    "support_positive_td_abs_before_mean",
    "support_negative_td_abs_before_mean",
    "support_positive_td_abs_after_mean",
    "support_negative_td_abs_after_mean",
    "support_positive_td_abs_reduction_mean",
    "support_negative_td_abs_reduction_mean",
    "support_positive_td_abs_reduction_over_omega_mean",
    "support_negative_td_abs_reduction_over_omega_mean",
    "support_positive_td_abs_after_over_before_mean",
    "support_negative_td_abs_after_over_before_mean",
    "target_variance_support_positive",
    "target_variance_support_negative",
    "w_raw_mean",
    "omega_critic_mean",
]

SAMPLE_KEYS = [
    "psi_q_rank",
    "td_abs_before",
    "td_abs_after",
    "td_abs_reduction",
    "td_abs_before_q1",
    "td_abs_before_q2",
    "td_abs_before_mean",
    "td_abs_after_q1",
    "td_abs_after_q2",
    "td_abs_after_mean",
    "td_abs_reduction_mean",
    "td_abs_reduction_over_omega",
    "td_abs_after_over_before",
    "w_raw",
    "omega_critic",
    "local_pool_size",
    "support_candidate_count",
    "phase_id",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            vals.append(x)
    return np.asarray(vals, dtype=np.float64)


def summarize_values(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = finite_values(rows, key)
        out[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
        out[f"{key}_std"] = float(np.std(values)) if values.size else float("nan")
        out[f"{key}_n"] = int(values.size)
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if int(np.sum(mask)) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * float(i + j - 1)
        i = j
    return ranks


def sample_level_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out = summarize_values(samples, SAMPLE_KEYS)
    psi = finite_values(samples, "psi_q_rank")
    td = finite_values(samples, "td_abs_before")
    n = min(psi.size, td.size)
    if n:
        out["sample_psi_td_abs_pearson"] = pearson(psi[:n], td[:n])
        out["sample_psi_td_abs_spearman"] = pearson(rankdata_average(psi[:n]), rankdata_average(td[:n]))
    else:
        out["sample_psi_td_abs_pearson"] = float("nan")
        out["sample_psi_td_abs_spearman"] = float("nan")
    support = [row for row in samples if bool(row.get("support_positive", False))]
    nonsupport = [row for row in samples if not bool(row.get("support_positive", False))]
    out["sample_count"] = len(samples)
    out["sample_support_positive_count"] = len(support)
    out["sample_support_positive_rate"] = float(len(support) / max(1, len(samples)))
    out["sample_support_positive_reduction_mean"] = float(np.mean(finite_values(support, "td_abs_reduction"))) if support else float("nan")
    out["sample_support_negative_reduction_mean"] = float(np.mean(finite_values(nonsupport, "td_abs_reduction"))) if nonsupport else float("nan")
    out["sample_support_positive_reduction_over_omega_mean"] = float(np.mean(finite_values(support, "td_abs_reduction_over_omega"))) if support else float("nan")
    out["sample_support_negative_reduction_over_omega_mean"] = float(np.mean(finite_values(nonsupport, "td_abs_reduction_over_omega"))) if nonsupport else float("nan")
    out["sample_support_positive_after_over_before_mean"] = float(np.mean(finite_values(support, "td_abs_after_over_before"))) if support else float("nan")
    out["sample_support_negative_after_over_before_mean"] = float(np.mean(finite_values(nonsupport, "td_abs_after_over_before"))) if nonsupport else float("nan")
    out["sample_target_variance_support_positive"] = float(np.var(finite_values(support, "target_value"))) if support else float("nan")
    out["sample_target_variance_support_negative"] = float(np.var(finite_values(nonsupport, "target_value"))) if nonsupport else float("nan")
    phases = finite_values(samples, "phase_id").astype(np.int64)
    out["sample_phase_unique_count"] = int(np.unique(phases).size) if phases.size else 0
    if support:
        support_phases = finite_values(support, "phase_id").astype(np.int64)
        out["sample_support_phase_unique_count"] = int(np.unique(support_phases).size) if support_phases.size else 0
    else:
        out["sample_support_phase_unique_count"] = 0
    return out


def summarize_run(run_dir: Path) -> dict[str, Any]:
    batch_rows = load_jsonl(run_dir / "diagnostic_metrics.jsonl")
    sample_rows = load_jsonl(run_dir / "diagnostic_samples.jsonl")
    batch_rows = [row for row in batch_rows if row.get("type") == "diagnostic_batch"]
    sample_rows = [row for row in sample_rows if row.get("type") == "diagnostic_sample"]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "diagnostic_batch_count": len(batch_rows),
        "diagnostic_sample_count": len(sample_rows),
    }
    summary.update(summarize_values(batch_rows, BATCH_KEYS))
    summary.update(sample_level_summary(sample_rows))
    return summary


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"run_count": len(runs)}
    keys = sorted({key for run in runs for key in run if key.endswith("_mean") or key.startswith("sample_")})
    for key in keys:
        vals = []
        for run in runs:
            value = run.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                vals.append(float(value))
        if vals:
            out[f"{key}_across_runs_mean"] = float(np.mean(vals))
            out[f"{key}_across_runs_std"] = float(np.std(vals))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze FPP TD3 SABER sample-level diagnostic logs.")
    parser.add_argument("--root", default="experiment_FPP/runs/fpp_td3_saber_diagnostic_ls512_1m")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    run_dirs = sorted(path.parent for path in root.glob("*/diagnostic_samples.jsonl"))
    if root.is_dir() and (root / "diagnostic_samples.jsonl").exists():
        run_dirs = [root]
    runs = [summarize_run(path) for path in run_dirs]
    payload = {"root": str(root), "runs": runs, "aggregate": aggregate_runs(runs)}
    output = Path(args.output) if args.output else root / "diagnostic_analysis_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
