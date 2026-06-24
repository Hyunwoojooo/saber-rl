from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable


EVAL_RE = re.compile(
    r"step=(?P<step>\d+).*?"
    r"eval_goal=(?P<goal>[-+0-9.eE]+).*?"
    r"eval_env=(?P<env>[-+0-9.eE]+).*?"
    r"eval_ret=(?P<ret>[-+0-9.eE]+)"
)

THRESHOLDS = (0.25, 0.50, 0.75, 0.90)
TAIL_WINDOWS = (5, 10, 20, 50)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def num(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean_or_nan(values: Iterable[float]) -> float:
    vals = finite(values)
    return mean(vals) if vals else math.nan


def sd_or_zero(values: Iterable[float]) -> float:
    vals = finite(values)
    return stdev(vals) if len(vals) > 1 else 0.0


def median_or_nan(values: Iterable[float]) -> float:
    vals = finite(values)
    return median(vals) if vals else math.nan


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_eval_history(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": float(match.group("step")),
                "goal_success": float(match.group("goal")),
                "env_success": float(match.group("env")),
                "return": float(match.group("ret")),
            }
        )
    return rows


def auc_success(history: list[dict[str, float]], key: str = "env_success") -> float:
    if not history:
        return math.nan
    if len(history) == 1:
        return float(history[0][key])
    total = 0.0
    for left, right in zip(history, history[1:]):
        total += 0.5 * (left[key] + right[key]) * (right["step"] - left["step"])
    span = max(1.0, history[-1]["step"] - history[0]["step"])
    return total / span


def first_hit(history: list[dict[str, float]], threshold: float, key: str = "env_success") -> float:
    for row in history:
        if row[key] >= threshold:
            return row["step"]
    return math.nan


def tail_mean(history: list[dict[str, float]], n_rows: int, key: str = "env_success") -> float:
    if not history:
        return math.nan
    return mean_or_nan(row[key] for row in history[-n_rows:])


def tail_metric(rows: list[dict[str, str]], key: str, n_rows: int = 50) -> float:
    vals = [num(row.get(key)) for row in rows[-n_rows:]]
    return mean_or_nan(vals)


def seed_from_run_dir(run_dir: Path, config: dict[str, object]) -> int:
    seeds = config.get("seeds")
    if isinstance(seeds, dict):
        seed = seeds.get("env_seed", seeds.get("torch_seed"))
        if seed is not None:
            return int(seed)
    match = re.search(r"_s(\d+)_", run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer seed from {run_dir}")
    return int(match.group(1))


def summarize_run(run_dir: Path) -> dict[str, object]:
    config = read_json(run_dir / "config.json")
    final_row = read_json(run_dir / "metrics_final_row.json")
    selected_rows = read_csv(run_dir / "metrics_selected_10k.csv")
    history = parse_eval_history(run_dir / "run.log")
    if not history:
        raise ValueError(f"No eval history found in {run_dir / 'run.log'}")

    seed = seed_from_run_dir(run_dir, config)
    requested_total_steps = int(num(config.get("total_steps")))
    final_train_step = int(num(final_row.get("step")))
    last_eval_step = int(history[-1]["step"])
    best_row = max(history, key=lambda row: row["env_success"])
    csv_eval_rows = sum(1 for row in selected_rows if row.get("eval/success_mean") not in ("", None))

    record: dict[str, object] = {
        "method": "TD-error",
        "seed": seed,
        "run_dir": str(run_dir),
        "requested_total_steps": requested_total_steps,
        "final_train_step": final_train_step,
        "last_eval_step": last_eval_step,
        "completion_rate": final_train_step / requested_total_steps if requested_total_steps else math.nan,
        "n_eval": len(history),
        "csv_eval_rows": csv_eval_rows,
        "eval_source": "run.log",
        "run_complete": final_train_step >= int(0.99 * requested_total_steps),
        "final_success": history[-1]["env_success"],
        "best_success": best_row["env_success"],
        "best_eval_step": int(best_row["step"]),
        "auc_success": auc_success(history),
        "runtime_total_sec": num(final_row.get("runtime/total_sec")),
        "runtime_hours": num(final_row.get("runtime/total_sec")) / 3600.0,
        "final_steps_per_sec": num(final_row.get("runtime/steps_per_sec")),
        "td_error_abs_mean_final": num(final_row.get("td_error/abs_mean")),
        "td_error_abs_p90_final": num(final_row.get("td_error/abs_p90")),
        "td_error_weight_mean_final": num(final_row.get("td_error/weight_mean")),
        "td_error_weight_ess_final": num(final_row.get("td_error/weight_ESS")),
        "td_error_abs_mean_tail50": tail_metric(selected_rows, "td_error/abs_mean"),
        "td_error_weight_mean_tail50": tail_metric(selected_rows, "td_error/weight_mean"),
        "td_error_weight_ess_tail50": tail_metric(selected_rows, "td_error/weight_ESS"),
    }

    for n_rows in TAIL_WINDOWS:
        record[f"tail{n_rows}_success"] = tail_mean(history, n_rows)
    for threshold in THRESHOLDS:
        record[f"first_hit_{threshold:.2f}"] = first_hit(history, threshold)

    return record


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    fields = [
        "requested_total_steps",
        "final_train_step",
        "last_eval_step",
        "completion_rate",
        "n_eval",
        "final_success",
        "best_success",
        "auc_success",
        "tail5_success",
        "tail10_success",
        "tail20_success",
        "tail50_success",
        "runtime_hours",
        "final_steps_per_sec",
        "td_error_abs_mean_final",
        "td_error_abs_p90_final",
        "td_error_weight_mean_final",
        "td_error_weight_ess_final",
        "td_error_abs_mean_tail50",
        "td_error_weight_mean_tail50",
        "td_error_weight_ess_tail50",
    ]
    out: dict[str, object] = {"method": "TD-error", "n": len(records)}
    for field in fields:
        vals = [num(row.get(field)) for row in records]
        out[f"{field}_mean"] = mean_or_nan(vals)
        out[f"{field}_sd"] = sd_or_zero(vals)

    for threshold in THRESHOLDS:
        key = f"first_hit_{threshold:.2f}"
        vals = [num(row.get(key)) for row in records]
        out[f"{key}_mean"] = mean_or_nan(vals)
        out[f"{key}_sd"] = sd_or_zero(vals)
        out[f"{key}_median"] = median_or_nan(vals)
        out[f"{key}_hits"] = len(finite(vals))

    out["all_runs_complete"] = all(bool(row["run_complete"]) for row in records)
    out["min_last_eval_step"] = min(int(row["last_eval_step"]) for row in records)
    out["max_last_eval_step"] = max(int(row["last_eval_step"]) for row in records)
    return out


def clean_json(value: object) -> object:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def csv_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def write_markdown(path: Path, records: list[dict[str, object]], summary: dict[str, object]) -> None:
    def fmt(value: object, digits: int = 3) -> str:
        v = num(value)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    def fmt_step(value: object) -> str:
        v = num(value)
        if not math.isfinite(v):
            return "n/a"
        return f"{int(round(v / 1000.0))}k"

    lines = [
        "# FPP TD-error Partial Log Aggregation",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Status",
        "",
        "- Source: `experiment_FPP/runs/TD-error/*_1000000_ls1024/run.log`.",
        "- `metrics_selected_10k.csv` has empty evaluation columns for these runs, so evaluation curves were reconstructed from `run.log` lines.",
        "- The run directories request 1M steps, but all five runs stop around 633k train steps and have last evaluation at 630k.",
        "- Treat this as a partial diagnostic artifact, not as a 1M main-paper TD-error baseline.",
        "- AUC is normalized over the observed evaluation span, matching the FPP trainer convention for an available history.",
        "",
        "## Aggregate",
        "",
        "| method | n | actual final step | last eval | final | best | AUC | tail10 | tail50 | hit90 median | hit90 hits | runtime h | steps/s | complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| TD-error | "
        f"{summary['n']} | "
        f"{fmt_step(summary['final_train_step_mean'])} | "
        f"{fmt_step(summary['last_eval_step_mean'])} | "
        f"{fmt(summary['final_success_mean'])} +/- {fmt(summary['final_success_sd'])} | "
        f"{fmt(summary['best_success_mean'])} +/- {fmt(summary['best_success_sd'])} | "
        f"{fmt(summary['auc_success_mean'])} +/- {fmt(summary['auc_success_sd'])} | "
        f"{fmt(summary['tail10_success_mean'])} +/- {fmt(summary['tail10_success_sd'])} | "
        f"{fmt(summary['tail50_success_mean'])} +/- {fmt(summary['tail50_success_sd'])} | "
        f"{fmt_step(summary['first_hit_0.90_median'])} | "
        f"{summary['first_hit_0.90_hits']}/5 | "
        f"{fmt(summary['runtime_hours_mean'], 2)} | "
        f"{fmt(summary['final_steps_per_sec_mean'], 2)} | "
        f"{summary['all_runs_complete']} |",
        "",
        "## Per Seed",
        "",
        "| seed | final train step | last eval | final | best | AUC | tail10 | hit90 | runtime h | steps/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(records, key=lambda item: int(item["seed"])):
        lines.append(
            "| "
            f"{row['seed']} | "
            f"{fmt_step(row['final_train_step'])} | "
            f"{fmt_step(row['last_eval_step'])} | "
            f"{fmt(row['final_success'])} | "
            f"{fmt(row['best_success'])} | "
            f"{fmt(row['auc_success'])} | "
            f"{fmt(row['tail10_success'])} | "
            f"{fmt_step(row['first_hit_0.90'])} | "
            f"{fmt(row['runtime_hours'], 2)} | "
            f"{fmt(row['final_steps_per_sec'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The partial TD-error runs are surprisingly strong by 630k, but they are not comparable to 1M FPP rows without either completing TD-error to 1M or truncating all methods to the same horizon.",
            "- Because the evaluation columns in CSV are empty, the trustworthy evaluation source for this artifact is `run.log`; the generated files record this explicitly.",
            "- For the AAAI draft, this row should stay out of the main 1M table until complete 1M TD-error runs or same-horizon comparison tables are prepared.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate FPP TD-error partial runs from run.log eval lines.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=repo_root() / "experiment_FPP" / "runs" / "TD-error",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root() / "experiment_FPP" / "results",
    )
    args = parser.parse_args()

    run_dirs = sorted(args.runs_root.glob("fetch_pick_and_place_td_error_s*_1000000_ls1024"))
    if not run_dirs:
        raise FileNotFoundError(f"No TD-error run dirs found under {args.runs_root}")

    records = [summarize_run(run_dir) for run_dir in run_dirs]
    records = sorted(records, key=lambda row: int(row["seed"]))
    summary = aggregate(records)
    tag = "fpp_td_error_partial_630k"

    per_seed_fields = [
        "method",
        "seed",
        "run_dir",
        "requested_total_steps",
        "final_train_step",
        "last_eval_step",
        "completion_rate",
        "n_eval",
        "csv_eval_rows",
        "eval_source",
        "run_complete",
        "final_success",
        "best_success",
        "best_eval_step",
        "auc_success",
        "tail5_success",
        "tail10_success",
        "tail20_success",
        "tail50_success",
        "first_hit_0.25",
        "first_hit_0.50",
        "first_hit_0.75",
        "first_hit_0.90",
        "runtime_total_sec",
        "runtime_hours",
        "final_steps_per_sec",
        "td_error_abs_mean_final",
        "td_error_abs_p90_final",
        "td_error_weight_mean_final",
        "td_error_weight_ess_final",
        "td_error_abs_mean_tail50",
        "td_error_weight_mean_tail50",
        "td_error_weight_ess_tail50",
    ]
    summary_fields = ["method", "n"] + sorted(key for key in summary if key not in {"method", "n"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{tag}_per_seed.csv", records, per_seed_fields)
    write_csv(args.out_dir / f"{tag}_method_summary.csv", [summary], summary_fields)
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": str(args.runs_root),
        "eval_source": "run.log",
        "status": "partial_not_1m",
        "method_summary": [summary],
        "per_seed": records,
    }
    (args.out_dir / f"{tag}_summary.json").write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / f"{tag}_analysis.md", records, summary)
    print(args.out_dir / f"{tag}_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
