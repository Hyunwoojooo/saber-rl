#!/usr/bin/env python3
"""Validate DrawerClose DDPG SABER diagnostics and build compact artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


PHASES = ("approach", "contact", "closing", "closed")
CORE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("step", "step"),
    ("episode", "episode"),
    ("train/her_relabel_ratio", "her_relabel_ratio"),
    ("lite/local_support_count_mean", "local_support_count_mean"),
    ("diagnostic/support_positive_rate", "support_positive_rate"),
    ("diagnostic/td_error_support_pos_mean", "td_support_pos_mean"),
    ("diagnostic/td_error_support_neg_mean", "td_support_neg_mean"),
    ("diagnostic/td_error_pre_mean", "td_pre_mean"),
    ("diagnostic/td_error_post_mean", "td_post_mean"),
    ("diagnostic/td_error_post_support_pos_mean", "td_post_support_pos_mean"),
    ("diagnostic/td_error_post_support_neg_mean", "td_post_support_neg_mean"),
    ("diagnostic/td_error_enrichment_ratio", "td_enrichment_ratio"),
    ("diagnostic/td_error_top10_threshold", "td_top10_threshold"),
    ("diagnostic/td_error_top10_support_rate", "td_top10_support_rate"),
    ("diagnostic/td_error_top10_overlap_rate", "td_top10_overlap_rate"),
    ("diagnostic/td_error_top10_enrichment_ratio", "td_top10_enrichment_ratio"),
    ("diagnostic/support_to_td_error_corr", "support_td_corr"),
    ("diagnostic/weight_to_td_error_corr", "weight_td_corr"),
    ("diagnostic/residual_reduction_mean", "residual_reduction_mean"),
    ("diagnostic/residual_reduction_support_pos_mean", "residual_support_pos_mean"),
    ("diagnostic/residual_reduction_support_neg_mean", "residual_support_neg_mean"),
    ("diagnostic/residual_correction_enrichment_ratio", "residual_enrichment_ratio"),
    ("diagnostic/residual_reduction_support_pos_rate", "residual_positive_rate_pos"),
    ("diagnostic/residual_reduction_support_neg_rate", "residual_positive_rate_neg"),
    ("diagnostic/weighted_residual_reduction_mean", "weighted_residual_reduction_mean"),
    ("diagnostic/support_to_residual_reduction_corr", "support_residual_corr"),
    ("diagnostic/weight_to_residual_reduction_corr", "weight_residual_corr"),
    ("diagnostic/same_minibatch_one_update", "same_minibatch_one_update"),
)


def parse_args() -> argparse.Namespace:
    experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=experiment_root / "results_2m_diag" / "DDPG-HER-B3-Critic-Only-2M-diagnostic",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_root / "diagnostic_artifact",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--expected-final-step", type=int, default=2_000_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def finite(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: Any) -> float:
    return float(value) if finite(value) else float("nan")


def finite_values(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if finite(value)]


def finite_mean(values: Iterable[Any]) -> float:
    items = finite_values(values)
    return mean(items) if items else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def locate_seed_run(root: Path, seed: int) -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in root.rglob("config.json"):
        try:
            config = read_json(config_path)
        except (OSError, json.JSONDecodeError):
            continue
        seeds = config.get("seeds", {})
        if int(seeds.get("env_seed", -1)) != seed:
            continue
        if config.get("task_name") != "drawer_close":
            continue
        if config.get("method") not in {"b3_support_only", "b3_support_runtime_opt"}:
            continue
        if config.get("weight_scope") != "critic_only":
            continue
        run_dir = config_path.parent
        if not (run_dir / "metrics.csv").is_file():
            continue
        summary_path = run_dir / "summary.json"
        completed = 0
        if summary_path.is_file():
            completed = int(read_json(summary_path).get("total_steps", 0))
        candidates.append((completed, run_dir))
    if not candidates:
        raise FileNotFoundError(f"no Main SABER diagnostic run found for seed {seed} under {root}")
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime_ns), reverse=True)
    return candidates[0][1]


def diagnostic_columns() -> tuple[tuple[str, str], ...]:
    phase_columns: list[tuple[str, str]] = []
    for phase in PHASES:
        phase_columns.extend(
            [
                (f"diagnostic/phase_{phase}_batch_rate", f"phase_{phase}_batch_rate"),
                (
                    f"diagnostic/phase_{phase}_support_positive_rate",
                    f"phase_{phase}_support_rate",
                ),
                (f"diagnostic/phase_{phase}_td_error_mean", f"phase_{phase}_td_mean"),
                (
                    f"diagnostic/phase_{phase}_residual_reduction_mean",
                    f"phase_{phase}_residual_mean",
                ),
            ]
        )
    return CORE_COLUMNS + tuple(phase_columns)


def compact_snapshots(rows: Sequence[Mapping[str, str]], seed: int) -> list[dict[str, Any]]:
    columns = diagnostic_columns()
    missing = [source for source, _ in columns if source not in rows[0]] if rows else []
    if missing:
        raise ValueError(f"missing diagnostic columns for seed {seed}: {missing}")
    by_step: dict[int, Mapping[str, str]] = {}
    for row in rows:
        if not finite(row.get("step")):
            continue
        if number(row.get("diagnostic/same_minibatch_one_update")) != 1.0:
            continue
        if not finite(row.get("diagnostic/support_positive_rate")):
            continue
        by_step[int(float(row["step"]))] = row
    compact: list[dict[str, Any]] = []
    for step, row in sorted(by_step.items()):
        out: dict[str, Any] = {"seed": seed}
        for source, target in columns:
            value = row.get(source, "")
            out[target] = int(float(value)) if target in {"step", "episode"} and finite(value) else value
        out["step"] = step
        compact.append(out)
    return compact


def weighted_phase_mean(rows: Sequence[Mapping[str, Any]], phase: str, metric: str) -> float:
    numerator = 0.0
    denominator = 0.0
    weight_key = f"phase_{phase}_batch_rate"
    value_key = f"phase_{phase}_{metric}"
    for row in rows:
        weight = number(row.get(weight_key))
        value = number(row.get(value_key))
        if math.isfinite(weight) and weight > 0.0 and math.isfinite(value):
            numerator += weight * value
            denominator += weight
    return numerator / denominator if denominator > 0.0 else float("nan")


def summarize_seed(seed: int, run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "seed": seed,
        "run_path": str(run_dir.resolve()),
        "snapshot_count": len(rows),
        "first_step": int(rows[0]["step"]),
        "final_step": int(rows[-1]["step"]),
        "support_positive_rate": finite_mean(row.get("support_positive_rate") for row in rows),
        "support_td_corr": finite_mean(row.get("support_td_corr") for row in rows),
        "td_top10_support_rate": finite_mean(row.get("td_top10_support_rate") for row in rows),
        "td_top10_enrichment_ratio": finite_mean(
            row.get("td_top10_enrichment_ratio") for row in rows
        ),
        "td_support_pos_mean": finite_mean(row.get("td_support_pos_mean") for row in rows),
        "td_support_neg_mean": finite_mean(row.get("td_support_neg_mean") for row in rows),
        "residual_support_pos_mean": finite_mean(
            row.get("residual_support_pos_mean") for row in rows
        ),
        "residual_support_neg_mean": finite_mean(
            row.get("residual_support_neg_mean") for row in rows
        ),
        "support_residual_corr": finite_mean(row.get("support_residual_corr") for row in rows),
    }
    stats["td_support_difference"] = stats["td_support_pos_mean"] - stats["td_support_neg_mean"]
    stats["residual_support_difference"] = (
        stats["residual_support_pos_mean"] - stats["residual_support_neg_mean"]
    )
    for phase in PHASES:
        stats[f"phase_{phase}_batch_rate"] = finite_mean(
            row.get(f"phase_{phase}_batch_rate") for row in rows
        )
        stats[f"phase_{phase}_support_rate"] = weighted_phase_mean(rows, phase, "support_rate")
        stats[f"phase_{phase}_td_mean"] = weighted_phase_mean(rows, phase, "td_mean")
        stats[f"phase_{phase}_residual_mean"] = weighted_phase_mean(rows, phase, "residual_mean")
    return stats


def bootstrap_interval(values: Sequence[float], seed: int = 2701) -> tuple[float, float]:
    finite_items = [value for value in values if math.isfinite(value)]
    if not finite_items:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(10_000):
        means.append(mean(rng.choice(finite_items) for _ in finite_items))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def aggregate_seed_stats(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "support_positive_rate",
        "support_td_corr",
        "td_top10_support_rate",
        "td_top10_enrichment_ratio",
        "td_support_pos_mean",
        "td_support_neg_mean",
        "td_support_difference",
        "residual_support_pos_mean",
        "residual_support_neg_mean",
        "residual_support_difference",
        "support_residual_corr",
    ]
    metrics.extend(f"phase_{phase}_{suffix}" for phase in PHASES for suffix in (
        "batch_rate", "support_rate", "td_mean", "residual_mean"
    ))
    output = []
    for index, metric in enumerate(metrics):
        values = finite_values(row.get(metric) for row in seed_rows)
        low, high = bootstrap_interval(values, seed=2701 + index)
        output.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": mean(values) if values else float("nan"),
                "sample_sd": stdev(values) if len(values) > 1 else 0.0,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return output


def window_rows(seed: int, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    windows = (
        ("early_0_500k", 0, 500_000),
        ("mid_500k_1m", 500_000, 1_000_000),
        ("late_1m_2m", 1_000_000, 2_000_000),
    )
    output = []
    for name, lower, upper in windows:
        selected = [row for row in rows if lower < int(row["step"]) <= upper]
        output.append(
            {
                "seed": seed,
                "window": name,
                "snapshot_count": len(selected),
                "support_positive_rate": finite_mean(
                    row.get("support_positive_rate") for row in selected
                ),
                "support_td_corr": finite_mean(row.get("support_td_corr") for row in selected),
                "td_top10_enrichment_ratio": finite_mean(
                    row.get("td_top10_enrichment_ratio") for row in selected
                ),
                "residual_support_difference": (
                    finite_mean(row.get("residual_support_pos_mean") for row in selected)
                    - finite_mean(row.get("residual_support_neg_mean") for row in selected)
                ),
            }
        )
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    seed_stats: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"input_root": str(input_root), "seeds": {}}

    for seed in args.seeds:
        run_dir = locate_seed_run(input_root, seed)
        required = ("metrics.csv", "metrics.jsonl", "config.json", "env_snapshot.json", "summary.json", "checkpoint.pt")
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"seed {seed} missing files: {missing} in {run_dir}")
        raw_rows = read_csv(run_dir / "metrics.csv")
        snapshots = compact_snapshots(raw_rows, seed)
        if not snapshots:
            raise ValueError(f"seed {seed} has no finite diagnostic snapshots")
        final_step = max(int(float(row["step"])) for row in raw_rows if finite(row.get("step")))
        if final_step != args.expected_final_step and not args.allow_incomplete:
            raise ValueError(
                f"seed {seed} final step {final_step} != expected {args.expected_final_step}"
            )

        seed_root = output_root / "data" / "seeds" / f"seed_{seed:02d}"
        snapshot_fields = ["seed"] + [target for _, target in diagnostic_columns()]
        write_gzip_csv(seed_root / "diagnostic_snapshots.csv.gz", snapshots, snapshot_fields)
        for name in ("config.json", "env_snapshot.json", "summary.json"):
            seed_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / name, seed_root / name)

        stats = summarize_seed(seed, run_dir, snapshots)
        seed_stats.append(stats)
        windows.extend(window_rows(seed, snapshots))
        phase_total_error = max(
            abs(sum(number(row.get(f"phase_{phase}_batch_rate")) for phase in PHASES) - 1.0)
            for row in snapshots
        )
        validation["seeds"][str(seed)] = {
            "run_path": str(run_dir.resolve()),
            "raw_metric_rows": len(raw_rows),
            "diagnostic_snapshot_rows": len(snapshots),
            "first_diagnostic_step": int(snapshots[0]["step"]),
            "final_diagnostic_step": int(snapshots[-1]["step"]),
            "final_training_step": final_step,
            "support_activation": all(finite(row.get("support_positive_rate")) for row in snapshots),
            "sample_level_td_and_top10": (
                all(finite(row.get("td_top10_support_rate")) for row in snapshots)
                and any(finite(row.get("support_td_corr")) for row in snapshots)
                and any(finite(row.get("td_support_pos_mean")) for row in snapshots)
                and any(finite(row.get("td_support_neg_mean")) for row in snapshots)
            ),
            "same_minibatch_pre_post_residual": (
                all(number(row.get("same_minibatch_one_update")) == 1.0 for row in snapshots)
                and all(finite(row.get("td_pre_mean")) for row in snapshots)
                and all(finite(row.get("td_post_mean")) for row in snapshots)
                and any(finite(row.get("residual_support_pos_mean")) for row in snapshots)
                and any(finite(row.get("residual_support_neg_mean")) for row in snapshots)
            ),
            "phase_conditioned_support": phase_total_error < 1e-5,
            "max_phase_fraction_sum_error": phase_total_error,
        }

    seed_fields = list(seed_stats[0].keys())
    write_csv(output_root / "results" / "diagnostic_by_seed.csv", seed_stats, seed_fields)
    aggregate = aggregate_seed_stats(seed_stats)
    write_csv(
        output_root / "results" / "diagnostic_aggregate.csv",
        aggregate,
        ("metric", "n_seeds", "mean", "sample_sd", "bootstrap_ci_low", "bootstrap_ci_high"),
    )
    write_csv(
        output_root / "results" / "diagnostic_by_phase_window.csv",
        windows,
        (
            "seed", "window", "snapshot_count", "support_positive_rate", "support_td_corr",
            "td_top10_enrichment_ratio", "residual_support_difference",
        ),
    )

    validation["all_four_diagnostics_complete"] = all(
        all(
            seed_info[key]
            for key in (
                "support_activation",
                "sample_level_td_and_top10",
                "same_minibatch_pre_post_residual",
                "phase_conditioned_support",
            )
        )
        for seed_info in validation["seeds"].values()
    )
    validation_path = output_root / "validation" / "validation_summary.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation["all_four_diagnostics_complete"]:
        raise RuntimeError("one or more diagnostic contracts failed; see validation_summary.json")

    manifest = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps({"files": manifest}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
