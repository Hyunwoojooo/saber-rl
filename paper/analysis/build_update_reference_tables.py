#!/usr/bin/env python3
"""Build task-wide REDQ and FPP-only high-replay-ratio references."""

from __future__ import annotations

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
    write_csv,
    write_json,
)


OUTPUT = ROOT / "paper" / "supplementary_artifacts" / "update_budget_references"
ROW_END = r"\\"

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


def write_high_rr_table(indexed: Mapping[tuple[str, str], Mapping[str, object]]) -> None:
    her = indexed[("FPP", "HER")]
    high_rr = indexed[("FPP", "HER high-RR")]
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
        "FPP--SAC & \\shortstack[l]{HER\\\\UTD 1; 2 critics} & "
        + metric_cells(her)
        + " "
        + ROW_END,
        " & \\shortstack[l]{HER high-RR\\\\UTD 4; 2 critics} & "
        + metric_cells(high_rr)
        + " "
        + ROW_END,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{FPP SAC update-ratio reference over seeds 1--5 on the "
        "exact shared 50k--1M grid. High-RR changes only the number of "
        "updates per environment step from one to four within HER. It was "
        "evaluated only on FPP SAC and is not a task-wide or wall-clock-"
        "matched control. Steps/s is descriptive.}",
        "\\label{tab:supp-high-rr}",
        "\\end{table*}",
        "",
    ]
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
    write_json(
        OUTPUT / "source_manifest.json",
        {
            "scope": "FPP high-RR and task-wide REDQ scaling references",
            "sources": [sources[key] for key in sorted(sources)],
        },
    )
    (OUTPUT / "README.md").write_text(
        "# Update-budget and REDQ references\n\n"
        "Exact-grid metrics for the FPP SAC high-replay-ratio reference and "
        "REDQ-HER TQB on FPP, DrawerOpen, and DrawerClose. High-RR changes "
        "UTD from one to four only on FPP SAC. REDQ changes UTD, critic "
        "architecture, target construction, and entropy optimization. Raw "
        "runs are bound by repository-relative paths and SHA256 without "
        "duplication.\n",
        encoding="utf-8",
    )
    artifact_manifest(OUTPUT)
    write_high_rr_table(indexed)
    write_redq_table(indexed)
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
