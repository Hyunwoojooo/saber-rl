from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from experiment_registry import EXPERIMENTS, repo_root_from_shared

FIELDS = [
    "task", "method", "variant", "seed", "final_success", "best_success", "best_step",
    "auc_eval_mean", "auc_trapz_0_to_total", "first_hit_0.25", "first_hit_0.50",
    "first_hit_0.75", "first_hit_0.90", "first_hit_1.00", "drop_from_best_to_final",
    "runtime_total_sec", "steps_per_sec",
]


def _num(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _seed_from_config(config: dict) -> int:
    seeds = config.get("seeds", {})
    return int(seeds.get("env_seed", seeds.get("torch_seed", -1)))


def collect(root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(root.glob("*/fetch_pick_and_place_*_s*_*/summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        config = json.loads(config_path.read_text())
        run_name = config.get("run_name", run_dir.name)
        method = str(config.get("method", summary.get("method", "")))
        variant = run_name.replace("fetch_pick_and_place_", "")
        final_success = _num(summary.get("final_eval_env_success", summary.get("final_eval_success")))
        best_success = _num(summary.get("best_eval_env_success", summary.get("best_eval_success")))
        auc = _num(summary.get("eval_AUC_success"))
        rows.append(
            {
                "task": "FetchPickAndPlace",
                "method": method,
                "variant": variant,
                "seed": _seed_from_config(config),
                "final_success": final_success,
                "best_success": best_success,
                "best_step": summary.get("best_eval_step", ""),
                "auc_eval_mean": auc,
                "auc_trapz_0_to_total": auc,
                "first_hit_0.25": summary.get("first_hit_0.25", ""),
                "first_hit_0.50": summary.get("first_hit_0.50", ""),
                "first_hit_0.75": summary.get("first_hit_0.75", ""),
                "first_hit_0.90": summary.get("first_hit_0.90", ""),
                "first_hit_1.00": summary.get("first_hit_1.00", ""),
                "drop_from_best_to_final": best_success - final_success if math.isfinite(best_success) and math.isfinite(final_success) else math.nan,
                "runtime_total_sec": _num(summary.get("runtime_total_sec")),
                "steps_per_sec": _num(summary.get("final_steps_per_sec")),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"].rsplit("_s", 1)[0]].append(row)
    out = []
    for method, group in sorted(grouped.items()):
        record = {"method": method, "n": len(group)}
        for key in ["final_success", "best_success", "auc_eval_mean", "drop_from_best_to_final", "runtime_total_sec", "steps_per_sec"]:
            vals = [_num(row.get(key)) for row in group]
            vals = [v for v in vals if math.isfinite(v)]
            record[f"{key}_mean"] = mean(vals) if vals else math.nan
            record[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(record)
    return out


def paired(rows: list[dict]) -> list[dict]:
    by_seed = defaultdict(dict)
    for row in rows:
        method = row["variant"].rsplit("_s", 1)[0]
        by_seed[int(row["seed"])][method] = row
    saber_key = "b3_critic_only"
    out = []
    for seed, methods in sorted(by_seed.items()):
        saber = methods.get(saber_key)
        if not saber:
            continue
        for method, comp in sorted(methods.items()):
            if method == saber_key:
                continue
            out.append(
                {
                    "seed": seed,
                    "comparison": f"SABER - {method}",
                    "final_diff": _num(saber["final_success"]) - _num(comp["final_success"]),
                    "AUC_diff": _num(saber["auc_eval_mean"]) - _num(comp["auc_eval_mean"]),
                    "drop_advantage": _num(comp["drop_from_best_to_final"]) - _num(saber["drop_from_best_to_final"]),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SABER FetchPickAndPlace runs.")
    default_root = repo_root_from_shared() / "SABER" / "FetchPickAndPlace" / "runs"
    parser.add_argument("--runs-root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=repo_root_from_shared() / "SABER" / "FetchPickAndPlace" / "analysis")
    args = parser.parse_args()
    rows = collect(args.runs_root)
    write_csv(args.out / "by_seed.csv", rows, FIELDS)
    agg = aggregate(rows)
    agg_fields = sorted({key for row in agg for key in row}) if agg else ["method", "n"]
    write_csv(args.out / "aggregate.csv", agg, agg_fields)
    pair_rows = paired(rows)
    write_csv(args.out / "paired_vs_controls.csv", pair_rows, ["seed", "comparison", "final_diff", "AUC_diff", "drop_advantage"])
    write_csv(args.out / "runtime_pareto.csv", agg, agg_fields)
    report = args.out / "report.md"
    report.write_text(
        "# FetchPickAndPlace SABER Report\n\n"
        f"Runs root: `{args.runs_root}`\n\n"
        f"Runs parsed: {len(rows)}\n\n"
        "Generated files: `by_seed.csv`, `aggregate.csv`, `paired_vs_controls.csv`, `runtime_pareto.csv`.\n",
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
