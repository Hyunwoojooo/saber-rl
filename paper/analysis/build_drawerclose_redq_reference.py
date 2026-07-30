#!/usr/bin/env python3
"""Build the compact DrawerClose REDQ scaling-reference artifact.

The builder fails closed on run completion, configuration parity, and the
exact 10k--2M evaluation grid. Raw outputs remain in canonical locations and
are bound by repository-relative SHA256 records.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from build_supplementary_results import (
    artifact_manifest,
    exact_curve,
    metrics,
    read_config,
    read_raw_series,
    source_entry,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = (
    ROOT
    / "metaworld_Drawer_Close_REDQ"
    / "results_2m"
    / "REDQ-HER-TQB-B0-2M"
)
OUTPUT = (
    ROOT
    / "metaworld_Drawer_Close_REDQ"
    / "paper_artifacts"
    / "drawer_close_redq_her_tqb_utd4_2m"
)
TABLE = ROOT / "paper" / "tables" / "supplementary_drawerclose_redq_table.tex"
PRIMARY_ROOT = ROOT / "paper" / "supplementary_artifacts" / "primary_seed_results"
PRIMARY_SUMMARY_TEMPLATES = {
    "HER": (
        "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B0-2M/"
        "drawer_close_DDPG-HER-B0-2M_s{seed}_2m_cuda/summary.json"
    ),
    "SABER": (
        "metaworld_Drawer_Close_DDPG/results_2m/"
        "DDPG-HER-B3-Critic-Only-2M/"
        "drawer_close_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/summary.json"
    ),
}
SOURCE_COMMIT = "0c72f95707e0befb19190a56c778332c1e2bbeb4"
SEEDS = tuple(range(1, 6))
BUDGET = 2_000_000
GRID_STEP = 10_000
GRID = tuple(range(GRID_STEP, BUDGET + 1, GRID_STEP))
SUCCESS_FIELD = "eval/env_success_mean"

EXPECTED_CONFIG: dict[str, object] = {
    "task_name": "drawer_close",
    "reward_type": "sparse",
    "metaworld_reward_function_version": "v2",
    "total_steps": BUDGET,
    "max_episode_steps": 200,
    "batch_size": 256,
    "buffer_capacity": 300_000,
    "warmup_steps": 5_000,
    "start_steps": 10_000,
    "updates_per_step": 4,
    "redq_ensemble_size": 10,
    "target_subset_size": 2,
    "her_ratio": 0.8,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "target_q_bounding": True,
    "target_q_min": -50.0,
    "target_q_max": 0.0,
    "sac_auto_alpha": True,
    "target_entropy": -4.0,
    "eval_freq": GRID_STEP,
    "eval_episodes": 20,
    "weight_scope": "none",
}


def run_dir(seed: int) -> Path:
    return RAW_ROOT / f"drawer_close_REDQ-HER-TQB-B0-2M_s{seed}_2m_cuda"


def values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_config(config: Mapping[str, object], path: Path, seed: int) -> None:
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if not values_match(config.get(key), expected)
    }
    seeds = config.get("seeds")
    observed_seed = seeds.get("env_seed") if isinstance(seeds, dict) else None
    if observed_seed != seed:
        mismatches["seeds.env_seed"] = {
            "expected": seed,
            "observed": observed_seed,
        }
    if mismatches:
        raise ValueError(f"{path}: configuration mismatch: {mismatches}")


def load_primary_rows() -> list[dict[str, object]]:
    path = PRIMARY_ROOT / "per_seed_metrics.csv"
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["task"] != "DrawerClose" or row["backbone"] != "DDPG":
                continue
            if row["method"] not in {"HER", "SABER"}:
                continue
            seed = int(row["seed"])
            summary_path = ROOT / PRIMARY_SUMMARY_TEMPLATES[row["method"]].format(
                seed=seed
            )
            summary = read_config(summary_path)
            speed = float(summary.get("final_steps_per_sec", float("nan")))
            runtime = float(summary.get("runtime_total_sec", float("nan")))
            if not math.isfinite(speed) or speed <= 0.0:
                raise ValueError(f"{summary_path}: invalid final_steps_per_sec")
            rows.append(
                {
                    "task": "DrawerClose",
                    "backbone": "DDPG",
                    "method": f"{row['method']} (DDPG)",
                    "seed": seed,
                    "utd": 1,
                    "critic_configuration": "1 critic",
                    "auc": float(row["auc"]),
                    "final_quarter_auc": float(row["final_quarter_auc"]),
                    "final_success": float(row["final_success"]),
                    "hit90_step": int(float(row["hit90_step"])),
                    "runtime_hours": runtime / 3600.0 if math.isfinite(runtime) else None,
                    "steps_per_second": speed,
                }
            )
    expected = {
        (method, seed)
        for method in ("HER (DDPG)", "SABER (DDPG)")
        for seed in SEEDS
    }
    observed = {(str(row["method"]), int(row["seed"])) for row in rows}
    if observed != expected:
        raise ValueError(
            f"incomplete primary reference rows: {sorted(expected - observed)}"
        )
    return rows


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def aggregate(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    order = ("HER (DDPG)", "SABER (DDPG)", "REDQ-HER TQB")
    output: list[dict[str, object]] = []
    for method in order:
        group = [row for row in rows if row["method"] == method]
        if len(group) != len(SEEDS):
            raise ValueError(
                f"{method}: expected {len(SEEDS)} seeds, found {len(group)}"
            )
        result: dict[str, object] = {
            "task": "DrawerClose",
            "backbone": group[0]["backbone"],
            "method": method,
            "n_seeds": len(group),
            "utd": group[0]["utd"],
            "critic_configuration": group[0]["critic_configuration"],
        }
        for field in ("auc", "final_quarter_auc", "final_success"):
            mean, sd = mean_sd([float(row[field]) for row in group])
            result[f"{field}_mean"] = mean
            result[f"{field}_sample_sd"] = sd
        hits = [
            int(row["hit90_step"])
            for row in group
            if row["hit90_step"] is not None
        ]
        result["hit90_reached"] = len(hits)
        result["hit90_total"] = len(group)
        result["hit90_median_reached"] = (
            statistics.median(hits) if hits else None
        )
        runtimes = [
            float(row["runtime_hours"])
            for row in group
            if row["runtime_hours"] is not None
        ]
        speeds = [
            float(row["steps_per_second"])
            for row in group
            if row["steps_per_second"] is not None
        ]
        result["runtime_hours_mean"] = (
            statistics.fmean(runtimes) if runtimes else None
        )
        result["runtime_hours_sample_sd"] = (
            statistics.stdev(runtimes) if len(runtimes) > 1 else None
        )
        result["steps_per_second_mean"] = (
            statistics.fmean(speeds) if speeds else None
        )
        result["steps_per_second_sample_sd"] = (
            statistics.stdev(speeds) if len(speeds) > 1 else None
        )
        output.append(result)
    return output


def format_hit90(row: Mapping[str, object]) -> str:
    reached = int(row["hit90_reached"])
    total = int(row["hit90_total"])
    median = row["hit90_median_reached"]
    if median is None:
        return f"{reached}/{total}"
    return f"{reached}/{total} ({float(median) / 1_000_000:.3f}M)"


def write_table(rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "% Generated by paper/analysis/build_drawerclose_redq_reference.py.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Cohort & Method/Variant & AUC & FQ-AUC & Final & Hit90 & Steps/s \\\\",
        "\\midrule",
    ]
    display = {
        "HER (DDPG)": ("DrawerClose--DDPG", r"\shortstack[l]{HER\\UTD 1; 1 critic}"),
        "SABER (DDPG)": ("", r"\shortstack[l]{SABER\\UTD 1; 1 critic}"),
        "REDQ-HER TQB": (
            "DrawerClose--REDQ",
            r"\shortstack[l]{REDQ-HER TQB\\UTD 4; 10 critics/2 target}",
        ),
    }
    for row in rows:
        cohort, method = display[str(row["method"])]
        lines.append(
            f"{cohort} & {method} & "
            f"${float(row['auc_mean']):.3f} \\pm "
            f"{float(row['auc_sample_sd']):.3f}$ & "
            f"${float(row['final_quarter_auc_mean']):.3f} \\pm "
            f"{float(row['final_quarter_auc_sample_sd']):.3f}$ & "
            f"${float(row['final_success_mean']):.3f} \\pm "
            f"{float(row['final_success_sample_sd']):.3f}$ & "
            f"{format_hit90(row)} & "
            f"{float(row['steps_per_second_mean']):.1f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{DrawerClose update-budget and architecture reference "
            "over seeds 1--5. Values are mean $\\pm$ sample standard "
            "deviation on the exact 10k--2M grid. FQ-AUC covers 1.5M--2M; "
            "Final is final-checkpoint success; Hit90 gives reaching seeds "
            "and their median first-hit step. REDQ-HER uses four updates per "
            "environment step and a 10-critic ensemble with two target "
            "critics sampled per backup, so it is not a resource-matched "
            "SABER comparison. Steps/s is the descriptive seed mean defined "
            "in Section~2.3.}",
            "\\label{tab:supp-drawerclose-redq}",
            "\\end{table*}",
            "",
        ]
    )
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_primary_rows()
    sources: dict[str, dict[str, object]] = {}

    for seed in SEEDS:
        directory = run_dir(seed)
        metrics_path = directory / "metrics.csv"
        config_path = directory / "config.json"
        summary_path = directory / "summary.json"
        config = read_config(config_path)
        validate_config(config, config_path, seed)

        observed = read_raw_series(metrics_path, SUCCESS_FIELD)
        if set(observed) != set(GRID):
            missing = sorted(set(GRID) - set(observed))
            extra = sorted(set(observed) - set(GRID))
            raise ValueError(
                f"{metrics_path}: non-exact evaluation grid; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        curve = exact_curve(metrics_path, GRID)
        result = metrics(curve, BUDGET, GRID_STEP)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("total_steps", -1)) != BUDGET:
            raise ValueError(f"{summary_path}: incomplete training budget")
        if int(summary.get("total_updates", -1)) != 7_980_004:
            raise ValueError(f"{summary_path}: unexpected total update count")
        if not math.isclose(
            float(summary.get("eval_AUC_success", float("nan"))),
            float(result["auc"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{summary_path}: stored AUC disagrees with recomputation"
            )

        rows.append(
            {
                "task": "DrawerClose",
                "backbone": "REDQ",
                "method": "REDQ-HER TQB",
                "seed": seed,
                "utd": 4,
                "critic_configuration": "10 critics (2 target)",
                **result,
                "runtime_hours": float(summary["runtime_total_sec"]) / 3600.0,
                "steps_per_second": float(summary["final_steps_per_sec"]),
            }
        )

        for name in (
            "best_checkpoint.json",
            "config.json",
            "env_snapshot.json",
            "metrics.csv",
            "run.log",
            "summary.json",
        ):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            sources[str(path.relative_to(ROOT))] = source_entry(path)

    for template in PRIMARY_SUMMARY_TEMPLATES.values():
        for seed in SEEDS:
            path = ROOT / template.format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            sources[str(path.relative_to(ROOT))] = source_entry(path)

    for path in (
        ROOT / "metaworld_Drawer_Close_REDQ" / "REDQ-HER-TQB-B0-2M.sh",
        ROOT / "metaworld_Drawer_Close_REDQ" / "README.md",
        PRIMARY_ROOT / "per_seed_metrics.csv",
        PRIMARY_ROOT / "artifact_manifest.json",
        Path(__file__).resolve(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[str(path.relative_to(ROOT))] = source_entry(path)

    aggregates = aggregate(rows)
    per_fields = (
        "task",
        "backbone",
        "method",
        "seed",
        "utd",
        "critic_configuration",
        "auc",
        "final_quarter_auc",
        "final_success",
        "hit90_step",
        "runtime_hours",
        "steps_per_second",
    )
    aggregate_fields = (
        "task",
        "backbone",
        "method",
        "n_seeds",
        "utd",
        "critic_configuration",
        "auc_mean",
        "auc_sample_sd",
        "final_quarter_auc_mean",
        "final_quarter_auc_sample_sd",
        "final_success_mean",
        "final_success_sample_sd",
        "hit90_reached",
        "hit90_total",
        "hit90_median_reached",
        "runtime_hours_mean",
        "runtime_hours_sample_sd",
        "steps_per_second_mean",
        "steps_per_second_sample_sd",
    )
    write_csv(OUTPUT / "per_seed_metrics.csv", rows, per_fields)
    write_csv(OUTPUT / "aggregate_metrics.csv", aggregates, aggregate_fields)
    write_json(
        OUTPUT / "validation.json",
        {
            "complete": True,
            "source_commit": SOURCE_COMMIT,
            "seeds": list(SEEDS),
            "training_steps_per_seed": BUDGET,
            "updates_per_seed": 7_980_004,
            "evaluation_grid_start": GRID[0],
            "evaluation_grid_end": GRID[-1],
            "evaluation_grid_stride": GRID_STEP,
            "evaluation_points_per_seed": len(GRID),
            "smoothing": False,
            "interpolation": False,
            "forward_filling": False,
            "resource_matched_to_ddpg_saber": False,
            "validation_checks": [
                "five complete seed runs",
                "exact 200-point evaluation grid",
                "configuration parity across REDQ seeds",
                "stored and recomputed AUC agreement",
                "2M environment steps and 7,980,004 updates per seed",
            ],
        },
    )
    write_json(
        OUTPUT / "source_manifest.json",
        {
            "scope": "DrawerClose REDQ-HER UTD-4 scaling reference",
            "source_commit": SOURCE_COMMIT,
            "raw_data_duplicated": False,
            "sources": [sources[key] for key in sorted(sources)],
        },
    )
    (OUTPUT / "README.md").write_text(
        "# DrawerClose REDQ-HER scaling reference\n\n"
        "This compact artifact validates and summarizes REDQ-HER TQB seeds "
        "1--5 on DrawerClose. Every seed completed 2M environment steps, "
        "7,980,004 updates, and the exact 200-point 10k--2M evaluation grid. "
        "AUC and final-quarter AUC are recomputed without smoothing, "
        "interpolation, extrapolation, or forward filling.\n\n"
        "REDQ-HER uses UTD 4, ten critics, a random two-critic target subset, "
        "automatic entropy adjustment, and target-Q bounding to [-50,0]. "
        "The DDPG-HER and SABER rows are imported from the verified primary "
        "seed artifact for context. This is a scaling and architecture "
        "reference, not a resource-matched post-HER allocation control.\n\n"
        "Raw run outputs are not copied here. source_manifest.json binds "
        "their canonical repository-relative paths and SHA256 digests; "
        "artifact_manifest.json binds this compact package.\n",
        encoding="utf-8",
    )
    write_table(aggregates)
    artifact_manifest(OUTPUT)
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
