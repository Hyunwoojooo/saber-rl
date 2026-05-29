from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

METHOD_B0 = "B0"
METHOD_B3 = "b3_critic_only_runtime_opt_v4"
SEEDS = [1, 2, 3, 4, 5]
TAIL_WINDOWS = [50_000, 100_000, 200_000, 300_000]
THRESHOLDS = [0.5, 0.8, 0.9, 1.0]
PHASES = ["approach", "grasp", "lift", "place"]


def _num(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def _std(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return stdev(vals) if len(vals) > 1 else 0.0


def _corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def _paired_effect(diff: list[float]) -> dict[str, float]:
    vals = [v for v in diff if math.isfinite(v)]
    if not vals:
        return {"mean": math.nan, "std": math.nan, "stderr": math.nan, "t": math.nan, "cohens_dz": math.nan}
    sd = stdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return {
        "mean": mean(vals),
        "std": sd,
        "stderr": se,
        "t": mean(vals) / se if se > 0.0 else math.nan,
        "cohens_dz": mean(vals) / sd if sd > 0.0 else math.nan,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_root(method: str) -> str:
    return METHOD_B0 if method == METHOD_B0 else method


def _load_selected_curve(exp_root: Path, method: str, run_dir: str) -> list[dict[str, float]]:
    path = exp_root / "runs" / _run_root(method) / run_dir / "metrics_selected_10k.csv"
    rows = []
    for row in _read_csv(path):
        rows.append({key: _num(value) for key, value in row.items()})
    return rows


def _load_summary(exp_root: Path) -> list[dict[str, str]]:
    return _read_csv(exp_root / "results" / "b0_vs_b3_runtime_opt_v4_summary.csv")


def _seed_rows(summary_rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return sorted([r for r in summary_rows if r["method"] == method], key=lambda r: int(r["seed"]))


def _tail_eval(curve: list[dict[str, float]], total_steps: int, window: int) -> float:
    vals = [r.get("eval/success_mean", math.nan) for r in curve if total_steps - window < int(r.get("step", -1)) <= total_steps]
    return _mean(vals)


def _first_hit(curve: list[dict[str, float]], threshold: float) -> float:
    for row in curve:
        value = row.get("eval/success_mean", math.nan)
        if math.isfinite(value) and value >= threshold:
            return row.get("step", math.nan)
    return math.nan


def build_committed_tables(exp_root: Path) -> dict[str, object]:
    summary = _load_summary(exp_root)
    curves: dict[tuple[str, int], list[dict[str, float]]] = {}
    for row in summary:
        curves[(row["method"], int(row["seed"]))] = _load_selected_curve(exp_root, row["method"], row["run_dir"])

    method_rows = []
    for method in [METHOD_B0, METHOD_B3]:
        rows = _seed_rows(summary, method)
        method_rows.append(
            {
                "method": method,
                "n": len(rows),
                "final_success_mean": _mean([_num(r["final_eval_success"]) for r in rows]),
                "final_success_std": _std([_num(r["final_eval_success"]) for r in rows]),
                "best_success_mean": _mean([_num(r["best_eval_success"]) for r in rows]),
                "auc_mean": _mean([_num(r["eval_AUC_success"]) for r in rows]),
                "auc_std": _std([_num(r["eval_AUC_success"]) for r in rows]),
                "steps_per_sec_mean": _mean([_num(r["final_steps_per_sec"]) for r in rows]),
                "runtime_hours_mean": _mean([_num(r["runtime_total_sec"]) / 3600.0 for r in rows]),
                "lookup_time_mean": _mean([_num(r.get("time_local_context_lookup", "")) for r in rows]),
                "support_rate_mean": _mean([_num(r.get("lite_psi_q_rank_rate", "")) for r in rows]),
                "mean_weight_mean": _mean([_num(r.get("lite_mean_w_P", "")) for r in rows]),
            }
        )

    tail_rows = []
    for method in [METHOD_B0, METHOD_B3]:
        rows = _seed_rows(summary, method)
        for window in TAIL_WINDOWS:
            vals = []
            for row in rows:
                seed = int(row["seed"])
                total = int(_num(row["total_steps"]))
                vals.append(_tail_eval(curves[(method, seed)], total, window))
            tail_rows.append({"method": method, "tail_window": window, "mean": _mean(vals), "std": _std(vals), "seed_values": vals})

    threshold_rows = []
    for method in [METHOD_B0, METHOD_B3]:
        rows = _seed_rows(summary, method)
        for threshold in THRESHOLDS:
            step_hits = []
            wall_hits = []
            for row in rows:
                seed = int(row["seed"])
                hit = _first_hit(curves[(method, seed)], threshold)
                step_hits.append(hit)
                steps_per_sec = _num(row["final_steps_per_sec"])
                wall_hits.append(hit / steps_per_sec / 3600.0 if math.isfinite(hit) and steps_per_sec > 0 else math.nan)
            threshold_rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "mean_step": _mean(step_hits),
                    "std_step": _std(step_hits),
                    "mean_wall_hours": _mean(wall_hits),
                    "std_wall_hours": _std(wall_hits),
                    "hits": sum(1 for v in step_hits if math.isfinite(v)),
                }
            )

    b0_by_seed = {int(r["seed"]): r for r in _seed_rows(summary, METHOD_B0)}
    b3_by_seed = {int(r["seed"]): r for r in _seed_rows(summary, METHOD_B3)}
    paired_rows = []
    paired_metrics: dict[str, list[float]] = {
        "final_success": [],
        "auc": [],
        "runtime_hours": [],
        "steps_per_sec_ratio": [],
        "tail100k_success": [],
        "tail200k_success": [],
    }
    for seed in SEEDS:
        b0 = b0_by_seed[seed]
        b3 = b3_by_seed[seed]
        b0_curve = curves[(METHOD_B0, seed)]
        b3_curve = curves[(METHOD_B3, seed)]
        b0_tail100 = _tail_eval(b0_curve, int(_num(b0["total_steps"])), 100_000)
        b3_tail100 = _tail_eval(b3_curve, int(_num(b3["total_steps"])), 100_000)
        b0_tail200 = _tail_eval(b0_curve, int(_num(b0["total_steps"])), 200_000)
        b3_tail200 = _tail_eval(b3_curve, int(_num(b3["total_steps"])), 200_000)
        paired_rows.append(
            {
                "seed": seed,
                "final_success_delta_b3_minus_b0": _num(b3["final_eval_success"]) - _num(b0["final_eval_success"]),
                "auc_delta_b3_minus_b0": _num(b3["eval_AUC_success"]) - _num(b0["eval_AUC_success"]),
                "runtime_hours_delta_b3_minus_b0": (_num(b3["runtime_total_sec"]) - _num(b0["runtime_total_sec"])) / 3600.0,
                "steps_per_sec_ratio_b3_over_b0": _num(b3["final_steps_per_sec"]) / _num(b0["final_steps_per_sec"]),
                "tail100k_success_delta_b3_minus_b0": b3_tail100 - b0_tail100,
                "tail200k_success_delta_b3_minus_b0": b3_tail200 - b0_tail200,
            }
        )
        paired_metrics["final_success"].append(paired_rows[-1]["final_success_delta_b3_minus_b0"])
        paired_metrics["auc"].append(paired_rows[-1]["auc_delta_b3_minus_b0"])
        paired_metrics["runtime_hours"].append(paired_rows[-1]["runtime_hours_delta_b3_minus_b0"])
        paired_metrics["steps_per_sec_ratio"].append(paired_rows[-1]["steps_per_sec_ratio_b3_over_b0"])
        paired_metrics["tail100k_success"].append(paired_rows[-1]["tail100k_success_delta_b3_minus_b0"])
        paired_metrics["tail200k_success"].append(paired_rows[-1]["tail200k_success_delta_b3_minus_b0"])

    paired_effect_rows = []
    for metric, diffs in paired_metrics.items():
        effect = _paired_effect(diffs)
        effect["metric"] = metric
        paired_effect_rows.append(effect)

    return {
        "method_summary": method_rows,
        "tail_success": tail_rows,
        "thresholds": threshold_rows,
        "paired_seed_deltas": paired_rows,
        "paired_effects": paired_effect_rows,
        "curves": curves,
        "summary_rows": summary,
    }


def _raw_run_dir(raw_root: Path, method: str, seed: int) -> Path:
    if method == METHOD_B0:
        return raw_root / "B0" / f"fetch_pick_and_place_b0_s{seed}_1000000_ls1024"
    return raw_root / METHOD_B3 / f"fetch_pick_and_place_b3_critic_only_runtime_opt_v4_s{seed}_1000000_ls1024"


def _raw_metrics_path(raw_root: Path, method: str, seed: int) -> Path:
    return _raw_run_dir(raw_root, method, seed) / "metrics.csv"


def build_raw_tables(raw_root: Path) -> dict[str, list[dict[str, object]]]:
    if not all(_raw_metrics_path(raw_root, METHOD_B3, seed).exists() for seed in SEEDS):
        return {}

    b3_windows = []
    runtime_cols = [
        "eval/success_mean",
        "runtime/steps_per_sec",
        "time/local_context_lookup",
        "time/numba_support_scan",
        "time/actor_critic_update",
        "runtime/full_context_scan_rate",
        "runtime/predicate_early_exit_rate",
        "runtime/support_cache_hit_rate",
        "runtime/local_scan_cache_hit_rate",
        "runtime/exact_negative_short_circuit_rate",
        "runtime/full_context_filter_count_mean",
        "runtime/batch_query_unique_rate",
        "runtime/batch_query_coalescing_saving_rate",
        "lite/psi_q_rank_rate",
        "lite/q_candidate_fallback_rate",
        "lite/ESS_w_normalized",
        "lite/mean_len_A_Q_replay_local",
        "lite/local_support_count_mean",
        "lite/local_support_count_p50",
        "lite/mean_num_scanned_candidates",
        "lite/mean_num_coarse_candidates",
        "loss/td_error_abs_mean",
        "loss/td_error_abs_p90",
    ]
    for window_name, lo, hi in [("early", 0, 300_000), ("mid", 300_000, 700_000), ("late", 700_000, 1_000_000)]:
        acc = {col: [] for col in runtime_cols}
        for seed in SEEDS:
            for row in _read_csv(_raw_metrics_path(raw_root, METHOD_B3, seed)):
                step = int(_num(row.get("step")))
                if lo < step <= hi:
                    for col in runtime_cols:
                        value = _num(row.get(col, ""))
                        if math.isfinite(value):
                            acc[col].append(value)
        record: dict[str, object] = {"window": window_name}
        for col, values in acc.items():
            record[f"{col}_mean"] = _mean(values)
            record[f"{col}_std"] = _std(values)
        b3_windows.append(record)

    phase_rows = []
    for window_name, lo, hi in [("early", 0, 300_000), ("mid", 300_000, 700_000), ("late", 700_000, 1_000_000)]:
        sums = {phase: {"count": 0.0, "psi": 0.0, "fallback": 0.0, "weight": 0.0} for phase in PHASES}
        for seed in SEEDS:
            for row in _read_csv(_raw_metrics_path(raw_root, METHOD_B3, seed)):
                step = int(_num(row.get("step")))
                if lo < step <= hi:
                    for phase in PHASES:
                        count = _num(row.get(f"lite/phase_{phase}_count_rate", ""))
                        psi = _num(row.get(f"lite/phase_{phase}_psi_q_rank_rate", ""))
                        fallback = _num(row.get(f"lite/phase_{phase}_fallback_rate", ""))
                        weight = _num(row.get(f"lite/phase_{phase}_mean_w_P", ""))
                        if math.isfinite(count):
                            sums[phase]["count"] += count
                            if count > 0.0:
                                sums[phase]["psi"] += count * psi
                                sums[phase]["fallback"] += count * fallback
                                sums[phase]["weight"] += count * weight
        total_count = sum(v["count"] for v in sums.values())
        for phase in PHASES:
            count = sums[phase]["count"]
            phase_rows.append(
                {
                    "window": window_name,
                    "phase": phase,
                    "sample_share": count / total_count if total_count > 0.0 else math.nan,
                    "support_rate_weighted": sums[phase]["psi"] / count if count > 0.0 else math.nan,
                    "fallback_rate_weighted": sums[phase]["fallback"] / count if count > 0.0 else math.nan,
                    "mean_weight_weighted": sums[phase]["weight"] / count if count > 0.0 else math.nan,
                }
            )

    return {"b3_raw_windows": b3_windows, "phase_support": phase_rows}


def write_outputs(out_dir: Path, committed: dict[str, object], raw: dict[str, list[dict[str, object]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_map = {
        "method_summary": committed["method_summary"],
        "tail_success": committed["tail_success"],
        "thresholds": committed["thresholds"],
        "paired_seed_deltas": committed["paired_seed_deltas"],
        "paired_effects": committed["paired_effects"],
    }
    for name, rows in table_map.items():
        _write_csv(out_dir / f"fpp_b0_b3_v4_{name}.csv", rows)  # type: ignore[arg-type]
    for name, rows in raw.items():
        _write_csv(out_dir / f"fpp_b0_b3_v4_{name}.csv", rows)

    json_payload = {name: rows for name, rows in table_map.items()}
    json_payload.update(raw)
    (out_dir / "fpp_b0_b3_v4_deep_analysis_tables.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate B0 vs B3 v4 FPP analysis tables.")
    exp_root_default = Path(__file__).resolve().parents[1]
    repo_root_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--experiment-root", type=Path, default=exp_root_default)
    parser.add_argument("--raw-runs-root", type=Path, default=repo_root_default / "FetchPickAndPlace" / "runs")
    parser.add_argument("--out", type=Path, default=exp_root_default / "results")
    args = parser.parse_args()

    committed = build_committed_tables(args.experiment_root)
    raw = build_raw_tables(args.raw_runs_root)
    write_outputs(args.out, committed, raw)
    print(args.out)
    if not raw:
        print(f"raw metrics not found under {args.raw_runs_root}; wrote committed-result tables only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
