#!/usr/bin/env python3
"""Build task-wide REDQ and FPP-only high-replay-ratio references."""

from __future__ import annotations

import csv
from typing import Mapping, Sequence

from build_supplementary_results import (
    ROOT,
    SEEDS,
    TABLE_ROOT,
    aggregate_metric_rows,
    artifact_manifest,
    bind_run_sources,
    exact_curve,
    final_steps_per_second,
    hit90_text,
    metrics,
    normalized_auc,
    read_config,
    source_entry,
    write_csv,
    write_json,
)


OUTPUT = ROOT / "paper" / "supplementary_artifacts" / "update_budget_references"
ROW_END = r"\\"
HIGH_RR_GRID = tuple(
    step for step in range(50_000, 1_000_001, 50_000)
    if step != 800_000
)
HIGH_RR_FQ_GRID = tuple(step for step in HIGH_RR_GRID if step >= 750_000)

HIGH_RR_SPECS = (
    {
        "base_backbone": "DDPG", "method": "HER", "utd": 1,
        "critics": "1 critic", "runtime": "summary",
        "path": (
            "experiment_FPP/runs/B0/fetch_pick_and_place_b0_s{seed}_"
            "1000000_ls1024/metrics_selected_10k.csv"
        ),
    },
    {
        "base_backbone": "DDPG", "method": "HER high-RR", "utd": 4,
        "critics": "1 critic", "runtime": "segments",
        "path": (
            "experiment_FPP/paper_artifacts/"
            "fpp_ddpg_td3_her_high_rr_x4_1m/data/ddpg/seed_{seed}/"
            "eval_curve_10k.csv"
        ),
    },
    {
        "base_backbone": "TD3", "method": "HER", "utd": 1,
        "critics": "2 critics", "runtime": "summary",
        "path": (
            "experiment_FPP/paper_artifacts/"
            "fpp_td3_her_vs_saber_n10_1m/data/td3_her/{seed}/"
            "metrics_selected_10k.csv"
        ),
    },
    {
        "base_backbone": "TD3", "method": "HER high-RR", "utd": 4,
        "critics": "2 critics", "runtime": "segments",
        "path": (
            "experiment_FPP/paper_artifacts/"
            "fpp_ddpg_td3_her_high_rr_x4_1m/data/td3/seed_{seed}/"
            "eval_curve_10k.csv"
        ),
    },
    {
        "base_backbone": "SAC", "method": "HER", "utd": 1,
        "critics": "2 critics", "runtime": "summary",
        "path": (
            "experiment_FPP/paper_artifacts/"
            "fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/"
            "fetch_pick_and_place_sac_her_baseline_ls512_s{seed}_"
            "1000000_ls512/metrics_selected_10k.csv"
        ),
    },
    {
        "base_backbone": "SAC", "method": "HER high-RR", "utd": 4,
        "critics": "2 critics", "runtime": "summary",
        "path": (
            "experiment_FPP/paper_artifacts/"
            "fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/"
            "fetch_pick_and_place_sac_her_high_rr_x4_s{seed}_"
            "1000000_ls512/metrics_selected_10k.csv"
        ),
    },
)

EXPECTED_HIGH_RR_AUC = {
    ("DDPG", "HER"): 0.555,
    ("DDPG", "HER high-RR"): 0.649,
    ("TD3", "HER"): 0.381,
    ("TD3", "HER high-RR"): 0.467,
    ("SAC", "HER"): 0.464,
    ("SAC", "HER high-RR"): 0.768,
}

EXPECTED_HIGH_RR_CONFIG_DIFFERENCE_KEYS = {
    "DDPG": {
        "device", "output_root", "run_name", "updates_per_step",
    },
    "TD3": {
        "diagnostic_freq", "diagnostic_full_batch_metrics",
        "diagnostic_log_samples", "diagnostic_samples_per_log",
        "local_scan_cache_enabled", "method", "numba_support_scan",
        "output_root", "resume", "resume_path", "run_name",
        "save_step_checkpoints", "support_cache_enabled",
        "support_diagnostic_audit_interval", "support_predicate_early_exit",
        "support_short_circuit", "updates_per_step",
    },
    "SAC": {
        "eval_freq", "method", "numba_support_scan", "output_root",
        "random_fixed_count_k", "resume", "resume_path", "run_name",
        "save_step_checkpoints", "support_predicate_early_exit",
        "support_runtime_impl", "support_short_circuit", "updates_per_step",
    },
}


def high_rr_config_difference_audit() -> dict[str, object]:
    indexed = {
        (str(spec["base_backbone"]), str(spec["method"])): spec
        for spec in HIGH_RR_SPECS
    }
    rows = []
    for backbone in ("DDPG", "TD3", "SAC"):
        for seed in SEEDS:
            baseline = ROOT / str(indexed[(backbone, "HER")]["path"]).format(
                seed=seed
            )
            high_rr = ROOT / str(
                indexed[(backbone, "HER high-RR")]["path"]
            ).format(seed=seed)
            baseline_config = read_config(baseline.parent / "config.json")
            high_rr_config = read_config(high_rr.parent / "config.json")
            differences = sorted(
                key for key in set(baseline_config) | set(high_rr_config)
                if baseline_config.get(key) != high_rr_config.get(key)
            )
            if set(differences) != EXPECTED_HIGH_RR_CONFIG_DIFFERENCE_KEYS[backbone]:
                raise ValueError(
                    f"{backbone} seed {seed}: unexpected UTD1/UTD4 config "
                    f"differences {differences}"
                )
            rows.append({"backbone": backbone, "seed": seed, "keys": differences})
    return {
        "status": "passed_with_documented_differences",
        "strict_one_factor_control": False,
        "shared_analysis_grid": list(HIGH_RR_GRID),
        "note": (
            "Besides updates_per_step, retained configs differ in procedural, "
            "runtime, naming, device, resume/checkpoint, diagnostic, and SAC "
            "evaluation-cadence fields; Table 16 is a scaling reference."
        ),
        "per_seed_difference_keys": rows,
    }

RUN_SPECS = (
    {
        "task": "FPP",
        "base_backbone": "SAC",
        "method": "HER",
        "utd": 1,
        "critics": "2 critics",
        "budget": 1_000_000,
        "grid_step": 50_000,
        "path": (
            "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
            "sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_"
            "ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv"
        ),
    },
    {
        "task": "FPP",
        "base_backbone": "SAC",
        "method": "SABER",
        "utd": 1,
        "critics": "2 critics",
        "budget": 1_000_000,
        "grid_step": 50_000,
        "path": (
            "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
            "sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_"
            "critic_only_v4_ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv"
        ),
    },
    {
        "task": "FPP",
        "base_backbone": "SAC",
        "method": "HER high-RR",
        "utd": 4,
        "critics": "2 critics",
        "budget": 1_000_000,
        "grid_step": 50_000,
        "path": (
            "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
            "sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_"
            "high_rr_x4_s{seed}_1000000_ls512/metrics_selected_10k.csv"
        ),
    },
    {
        "task": "FPP",
        "base_backbone": "SAC",
        "method": "REDQ-HER TQB",
        "utd": 4,
        "critics": "10 critics/2 target",
        "budget": 1_000_000,
        "grid_step": 50_000,
        "path": (
            "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
            "redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_"
            "redq_her_utd4_n10_m2_tqbound_ls512_s{seed}_1000000/"
            "metrics_selected_10k.csv"
        ),
    },
    {
        "task": "DrawerOpen",
        "base_backbone": "DDPG",
        "method": "HER",
        "utd": 1,
        "critics": "1 critic",
        "budget": 2_000_000,
        "grid_step": 10_000,
        "path": (
            "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B0-2M/"
            "drawer_open_DDPG-HER-B0-2M_s{seed}_2m_cuda/metrics.csv"
        ),
    },
    {
        "task": "DrawerOpen",
        "base_backbone": "DDPG",
        "method": "SABER",
        "utd": 1,
        "critics": "1 critic",
        "budget": 2_000_000,
        "grid_step": 10_000,
        "path": (
            "metaworld_Drawer_Open_DDPG/results_2m/"
            "DDPG-HER-B3-Critic-Only-2M/drawer_open_DDPG-HER-B3-"
            "Critic-Only-2M_s{seed}_2m_cuda/metrics.csv"
        ),
    },
    {
        "task": "DrawerOpen",
        "base_backbone": "DDPG",
        "method": "REDQ-HER TQB",
        "utd": 4,
        "critics": "10 critics/2 target",
        "budget": 2_000_000,
        "grid_step": 10_000,
        "path": (
            "metaworld_Drawer_Open_REDQ/results_2m/REDQ-HER-TQB-B0-2M/"
            "drawer_open_REDQ-HER-TQB-B0-2M_s{seed}_2m_cuda/metrics.csv"
        ),
    },
    {
        "task": "DrawerClose",
        "base_backbone": "DDPG",
        "method": "HER",
        "utd": 1,
        "critics": "1 critic",
        "budget": 2_000_000,
        "grid_step": 10_000,
        "path": (
            "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B0-2M/"
            "drawer_close_DDPG-HER-B0-2M_s{seed}_2m_cuda/metrics.csv"
        ),
    },
    {
        "task": "DrawerClose",
        "base_backbone": "DDPG",
        "method": "SABER",
        "utd": 1,
        "critics": "1 critic",
        "budget": 2_000_000,
        "grid_step": 10_000,
        "path": (
            "metaworld_Drawer_Close_DDPG/results_2m/"
            "DDPG-HER-B3-Critic-Only-2M/drawer_close_DDPG-HER-B3-"
            "Critic-Only-2M_s{seed}_2m_cuda/metrics.csv"
        ),
    },
    {
        "task": "DrawerClose",
        "base_backbone": "DDPG",
        "method": "REDQ-HER TQB",
        "utd": 4,
        "critics": "10 critics/2 target",
        "budget": 2_000_000,
        "grid_step": 10_000,
        "path": (
            "metaworld_Drawer_Close_REDQ/results_2m/REDQ-HER-TQB-B0-2M/"
            "drawer_close_REDQ-HER-TQB-B0-2M_s{seed}_2m_cuda/metrics.csv"
        ),
    },
)

EXPECTED_AUC = {
    ("FPP", "HER"): 0.471,
    ("FPP", "SABER"): 0.583,
    ("FPP", "HER high-RR"): 0.768,
    ("FPP", "REDQ-HER TQB"): 0.867,
    ("DrawerOpen", "HER"): 0.207,
    ("DrawerOpen", "SABER"): 0.754,
    ("DrawerOpen", "REDQ-HER TQB"): 0.672,
    ("DrawerClose", "HER"): 0.251,
    ("DrawerClose", "SABER"): 0.324,
    ("DrawerClose", "REDQ-HER TQB"): 0.188,
}


def aggregate_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    return {
        (str(row["task"]), str(row["method"])): row
        for row in rows
    }


def metrics_on_high_rr_grid(curve: Mapping[int, float]) -> dict[str, object]:
    return {
        "auc": normalized_auc(curve, HIGH_RR_GRID),
        "final_quarter_auc": normalized_auc(curve, HIGH_RR_FQ_GRID),
        "final_success": curve[1_000_000],
        "hit90_step": next(
            (step for step in HIGH_RR_GRID if curve[step] >= 0.9), None
        ),
    }


def segmented_steps_per_second(metrics_path) -> float:
    segments_path = metrics_path.parent / "runtime_segments.csv"
    with segments_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{segments_path}: empty runtime segment table")
    active_seconds = sum(float(row["active_seconds"]) for row in rows)
    if active_seconds <= 0.0:
        raise ValueError(f"{segments_path}: non-positive active runtime")
    return 1_000_000 / active_seconds


def build_high_rr_rows(
    sources: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for spec in HIGH_RR_SPECS:
        for seed in SEEDS:
            path = ROOT / str(spec["path"]).format(seed=seed)
            curve = exact_curve(path, HIGH_RR_GRID)
            speed = (
                segmented_steps_per_second(path)
                if spec["runtime"] == "segments"
                else final_steps_per_second(path)
            )
            rows.append(
                {
                    "task": "FPP",
                    "base_backbone": spec["base_backbone"],
                    "method": spec["method"],
                    "seed": seed,
                    "utd": spec["utd"],
                    "critic_configuration": spec["critics"],
                    "budget": 1_000_000,
                    "grid_contract": "50k--1M excluding 800k",
                    "grid_points": len(HIGH_RR_GRID),
                    **metrics_on_high_rr_grid(curve),
                    "steps_per_second": speed,
                }
            )
            bind_run_sources(sources, path)
            if spec["runtime"] == "segments":
                segment_path = path.parent / "runtime_segments.csv"
                sources[str(segment_path.relative_to(ROOT))] = source_entry(
                    segment_path
                )
    aggregate = aggregate_metric_rows(
        rows,
        (
            "task", "base_backbone", "method", "utd",
            "critic_configuration", "budget", "grid_contract",
            "grid_points",
        ),
    )
    indexed = {
        (str(row["base_backbone"]), str(row["method"])): row
        for row in aggregate
    }
    for key, expected in EXPECTED_HIGH_RR_AUC.items():
        observed = round(float(indexed[key]["auc_mean"]), 3)
        if observed != expected:
            raise ValueError(
                f"High-RR {key}: expected AUC {expected:.3f}, "
                f"got {observed:.3f}"
            )
    return rows, aggregate


def metric_cells(row: Mapping[str, object]) -> str:
    return (
        f"${float(row['auc_mean']):.3f} \\pm "
        f"{float(row['auc_sample_sd']):.3f}$ & "
        f"${float(row['final_quarter_auc_mean']):.3f} \\pm "
        f"{float(row['final_quarter_auc_sample_sd']):.3f}$ & "
        f"${float(row['final_success_mean']):.3f} \\pm "
        f"{float(row['final_success_sample_sd']):.3f}$ & "
        f"{hit90_text(row)} & {float(row['steps_per_second_mean']):.1f}"
    )


def write_high_rr_table(
    indexed: Mapping[tuple[str, str], Mapping[str, object]]
) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Cohort & Method/Variant & AUC & FQ-AUC & Final & Hit90 & Steps/s "
        + ROW_END,
        "\\midrule",
    ]
    critic_text = {"DDPG": "1 critic", "TD3": "2 critics", "SAC": "2 critics"}
    for backbone_index, backbone in enumerate(("DDPG", "TD3", "SAC")):
        for method_index, method in enumerate(("HER", "HER high-RR")):
            row = indexed[(backbone, method)]
            cohort = f"FPP--{backbone}" if method_index == 0 else ""
            if method == "HER high-RR":
                resume_mark = "$^{\\dagger}$" if backbone == "TD3" else ""
                display = (
                    f"\\shortstack[l]{{HER high-RR \\(4\\times\\)"
                    f"{resume_mark}\\\\{critic_text[backbone]}}}"
                )
            else:
                display = (
                    "\\shortstack[l]{HER\\\\"
                    f"UTD 1; {critic_text[backbone]}}}"
                )
            lines.append(
                f"{cohort} & {display} & {metric_cells(row)} " + ROW_END
            )
        if backbone_index != 2:
            lines.append("\\addlinespace[2pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{FPP HER update-ratio references over seeds 1--5 "
            "on one exact shared 19-point grid: 50k--1M at 50k intervals, "
            "excluding 800k because that evaluation was not retained for "
            "TD3 High-RR seeds 1, 3, and 4. No value is interpolated. "
            "High-RR \\(4\\times\\) uses four gradient updates per "
            "environment step instead of one within each backbone, but "
            "retained configs also differ in "
            "procedural/runtime fields and SAC evaluation cadence; this is "
            "not a strict one-factor or wall-clock-matched control. "
            "$^{\\dagger}$ TD3 High-RR resumed after "
            "reboot at step 800001 (seeds 1--4) or 630001 (seed 5); exact "
            "equivalence to uninterrupted runs is not claimed. High-RR "
            "DDPG/TD3 steps/s values sum retained active-runtime segments and "
            "exclude reboot downtime and unretained discarded work. "
            "Steps/s is descriptive.}",
            "\\label{tab:supp-high-rr}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_high_rr_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_redq_table(indexed: Mapping[tuple[str, str], Mapping[str, object]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Task & Method/Variant & AUC & FQ-AUC & Final & Hit90 & Steps/s "
        + ROW_END,
        "\\midrule",
    ]
    for task_index, task in enumerate(("FPP", "DrawerOpen", "DrawerClose")):
        backbone = "SAC" if task == "FPP" else "DDPG"
        critic_text = "2 critics" if task == "FPP" else "1 critic"
        for method_index, method in enumerate(("HER", "SABER", "REDQ-HER TQB")):
            row = indexed[(task, method)]
            task_cell = task if method_index == 0 else ""
            if method == "REDQ-HER TQB":
                display = r"\shortstack[l]{REDQ-HER TQB\\UTD 4; 10 critics/2 target}"
            elif method == "SABER":
                display = (
                    f"\\shortstack[l]{{SABER (Ours)\\\\"
                    f"{backbone}; UTD 1; {critic_text}}}"
                )
            else:
                display = (
                    f"\\shortstack[l]{{{method} ({backbone})\\\\"
                    f"UTD 1; {critic_text}}}"
                )
            lines.append(
                f"{task_cell} & {display} & {metric_cells(row)} " + ROW_END
            )
        if task_index != 2:
            lines.append("\\addlinespace[2pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{REDQ architecture-and-compute references over seeds "
            "1--5. FPP uses the exact shared 50k--1M grid; Drawer tasks use "
            "10k--2M. REDQ-HER uses UTD 4, ten critics, a random two-critic "
            "target subset, and target-Q bounding, whereas HER "
            "and SABER rows use UTD 1. Values are compared only within task. "
            "Steps/s is descriptive.}",
            "\\label{tab:supp-redq-references}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_redq_references_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    per_seed: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    for spec in RUN_SPECS:
        budget = int(spec["budget"])
        grid_step = int(spec["grid_step"])
        grid = tuple(range(grid_step, budget + 1, grid_step))
        for seed in SEEDS:
            path = ROOT / str(spec["path"]).format(seed=seed)
            curve = exact_curve(path, grid)
            per_seed.append(
                {
                    "task": spec["task"],
                    "base_backbone": spec["base_backbone"],
                    "method": spec["method"],
                    "seed": seed,
                    "utd": spec["utd"],
                    "critic_configuration": spec["critics"],
                    "budget": budget,
                    "grid_step": grid_step,
                    **metrics(curve, budget, grid_step),
                    "steps_per_second": final_steps_per_second(path),
                }
            )
            bind_run_sources(sources, path)

    aggregate = aggregate_metric_rows(
        per_seed,
        (
            "task",
            "base_backbone",
            "method",
            "utd",
            "critic_configuration",
            "budget",
            "grid_step",
        ),
    )
    indexed = aggregate_index(aggregate)
    for key, expected in EXPECTED_AUC.items():
        observed = round(float(indexed[key]["auc_mean"]), 3)
        if observed != expected:
            raise ValueError(f"{key}: expected AUC {expected:.3f}, got {observed:.3f}")

    high_rr_per_seed, high_rr_aggregate = build_high_rr_rows(sources)
    high_rr_indexed = {
        (str(row["base_backbone"]), str(row["method"])): row
        for row in high_rr_aggregate
    }

    per_fields = (
        "task", "base_backbone", "method", "seed", "utd",
        "critic_configuration", "budget", "grid_step", "auc",
        "final_quarter_auc", "final_success", "hit90_step",
        "steps_per_second",
    )
    aggregate_fields = (
        "task", "base_backbone", "method", "utd",
        "critic_configuration", "budget", "grid_step", "n_seeds",
        "auc_mean", "auc_sample_sd", "final_quarter_auc_mean",
        "final_quarter_auc_sample_sd", "final_success_mean",
        "final_success_sample_sd", "hit90_reached", "hit90_total",
        "hit90_median_reached", "steps_per_second_mean",
        "steps_per_second_sample_sd",
    )
    write_csv(OUTPUT / "per_seed_metrics.csv", per_seed, per_fields)
    write_csv(OUTPUT / "aggregate_metrics.csv", aggregate, aggregate_fields)
    high_rr_per_fields = (
        "task", "base_backbone", "method", "seed", "utd",
        "critic_configuration", "budget", "grid_contract", "grid_points",
        "auc", "final_quarter_auc", "final_success", "hit90_step",
        "steps_per_second",
    )
    high_rr_aggregate_fields = (
        "task", "base_backbone", "method", "utd", "critic_configuration",
        "budget", "grid_contract", "grid_points", "n_seeds", "auc_mean",
        "auc_sample_sd", "final_quarter_auc_mean",
        "final_quarter_auc_sample_sd", "final_success_mean",
        "final_success_sample_sd", "hit90_reached", "hit90_total",
        "hit90_median_reached", "steps_per_second_mean",
        "steps_per_second_sample_sd",
    )
    write_csv(
        OUTPUT / "high_rr_per_seed_metrics.csv",
        high_rr_per_seed,
        high_rr_per_fields,
    )
    write_csv(
        OUTPUT / "high_rr_aggregate_metrics.csv",
        high_rr_aggregate,
        high_rr_aggregate_fields,
    )
    config_difference_audit = high_rr_config_difference_audit()
    write_json(
        OUTPUT / "high_rr_config_difference_audit.json",
        config_difference_audit,
    )
    compact_manifest = (
        ROOT / "experiment_FPP" / "paper_artifacts" /
        "fpp_ddpg_td3_her_high_rr_x4_1m" / "artifact_manifest.json"
    )
    sources[str(compact_manifest.relative_to(ROOT))] = source_entry(
        compact_manifest
    )
    write_json(
        OUTPUT / "source_manifest.json",
        {
            "scope": "FPP high-RR and task-wide REDQ scaling references",
            "high_rr_table_grid": list(HIGH_RR_GRID),
            "high_rr_table_grid_note": (
                "one exact 19-point grid; 800k excluded for every row because "
                "TD3 High-RR seeds 1, 3, and 4 did not retain that evaluation"
            ),
            "high_rr_strict_one_factor_control": False,
            "high_rr_config_difference_audit": (
                "high_rr_config_difference_audit.json"
            ),
            "sources": [sources[key] for key in sorted(sources)],
        },
    )
    (OUTPUT / "README.md").write_text(
        "# Update-budget and REDQ references\n\n"
        "Exact-grid metrics for FPP DDPG, TD3, and SAC HER High-RR references "
        "and REDQ-HER TQB on FPP, DrawerOpen, and DrawerClose. Table 16 uses "
        "one exact 19-point FPP grid (50k--1M excluding 800k) for all six "
        "UTD1/UTD4 rows, without interpolation. The TD3 High-RR cohort is a "
        "resumed trajectory and is labeled accordingly. Its throughput sums "
        "retained active-runtime segments. REDQ changes UTD, critic "
        "architecture, target construction, and entropy optimization. "
        "The retained UTD1/UTD4 configs also differ in procedural/runtime "
        "fields and SAC evaluation cadence, so High-RR is not a strict "
        "one-factor control. Canonical runs and compact artifacts are bound "
        "by repository-relative "
        "paths and SHA256 without raw duplication.\n",
        encoding="utf-8",
    )
    artifact_manifest(OUTPUT)
    write_high_rr_table(high_rr_indexed)
    write_redq_table(indexed)
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
