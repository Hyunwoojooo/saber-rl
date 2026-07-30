#!/usr/bin/env python3
"""Build mechanism-control, phase-count, and paired-prioritization supplements."""

from __future__ import annotations

import csv
import itertools
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from build_supplementary_results import (
    ROOT,
    SEEDS,
    SUCCESS_FIELD,
    TABLE_ROOT,
    aggregate_metric_rows,
    artifact_manifest,
    bind_run_sources,
    exact_curve,
    final_steps_per_second,
    fmt,
    hit90_text,
    metrics,
    percentile,
    source_entry,
    write_csv,
    write_json,
)


OUTPUT_ROOT = ROOT / "paper" / "supplementary_artifacts"
BATCH_SIZE = 256

TASK_SPECS = {
    "FPP": (1_000_000, 10_000),
    "DrawerOpen": (2_000_000, 10_000),
    "DrawerClose": (2_000_000, 10_000),
}

VARIANT_LABELS = {
    "main": "SABER (Main)",
    "replay_action": "Replay-Action Query",
    "shuffled_policy": "Shuffled-Policy Query",
    "random_support": "Random-Support",
    "context_eligible": "Context-Eligible Count-Matched",
}

TABLE_VARIANT_LABELS = {
    **VARIANT_LABELS,
    "replay_action": "\\shortstack[l]{Replay-Action\\\\Query}",
    "shuffled_policy": "\\shortstack[l]{Shuffled-Policy\\\\Query}",
    "context_eligible": "\\shortstack[l]{Context-Eligible\\\\Count-Matched}",
}

MECHANISM_SPECS = {
    ("FPP", "main"): (
        "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512/"
        "fetch_pick_and_place_b3_critic_only_runtime_opt_v4_ls512_s{seed}_"
        "1000000_ls512/metrics_selected_10k.csv"
    ),
    ("FPP", "replay_action"): (
        "experiment_FPP/paper_artifacts/fpp_ddpg_saber_query_replay_ls512_1m/"
        "data/replay/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("FPP", "shuffled_policy"): (
        "experiment_FPP/paper_artifacts/fpp_ddpg_saber_query_shuffled_ls512_1m/"
        "data/shuffled/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("FPP", "random_support"): (
        "experiment_FPP/paper_artifacts/"
        "fpp_ddpg_saber_random_support_count_matched_ls512_1m/"
        "data/control/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("FPP", "context_eligible"): (
        "experiment_FPP/paper_artifacts/"
        "fpp_ddpg_saber_context_eligible_count_matched_ls512_1m/"
        "data/control/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("DrawerOpen", "main"): (
        "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M/"
        "drawer_open_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/metrics.csv"
    ),
    ("DrawerOpen", "replay_action"): (
        "metaworld_Drawer_Open_DDPG/paper_artifacts/"
        "drawer_open_ddpg_saber_replay_action_query_2m/"
        "data/replay/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("DrawerOpen", "shuffled_policy"): (
        "metaworld_Drawer_Open_DDPG/paper_artifacts/"
        "drawer_open_ddpg_saber_shuffled_policy_query_2m/"
        "data/shuffled/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("DrawerOpen", "random_support"): (
        "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Random-Support-2M/"
        "drawer_open_DDPG-HER-B3-Random-Support-2M_s{seed}_2m_cuda/metrics.csv"
    ),
    ("DrawerOpen", "context_eligible"): (
        "metaworld_Drawer_Open_DDPG/paper_artifacts/"
        "drawer_open_ddpg_saber_context_eligible_count_matched_2m/"
        "data/strict/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("DrawerClose", "main"): (
        "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M/"
        "drawer_close_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/metrics.csv"
    ),
    ("DrawerClose", "replay_action"): (
        "metaworld_Drawer_Close_DDPG/paper_artifacts/"
        "drawer_close_ddpg_saber_replay_action_query_2m/"
        "data/replay/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("DrawerClose", "shuffled_policy"): (
        "metaworld_Drawer_Close_DDPG/paper_artifacts/"
        "drawer_close_ddpg_saber_shuffled_policy_query_2m/"
        "data/shuffled/seed_{seed}/metrics_selected_10k.csv"
    ),
    ("DrawerClose", "random_support"): (
        "metaworld_Drawer_Close_DDPG/results_2m/"
        "DDPG-HER-B3-Random-Support-2M/"
        "drawer_close_DDPG-HER-B3-Random-Support-2M_s{seed}_2m_cuda/metrics.csv"
    ),
    ("DrawerClose", "context_eligible"): (
        "metaworld_Drawer_Close_DDPG/paper_artifacts/"
        "drawer_close_ddpg_saber_context_eligible_count_matched_2m/"
        "data/strict/seed_{seed}/metrics_selected_10k.csv"
    ),
}

EXPECTED_MAIN_MINUS_CONTROL = {
    ("FPP", "replay_action"): 0.011,
    ("FPP", "shuffled_policy"): 0.043,
    ("FPP", "random_support"): 0.056,
    ("FPP", "context_eligible"): 0.033,
    ("DrawerOpen", "replay_action"): 0.436,
    ("DrawerOpen", "shuffled_policy"): 0.105,
    ("DrawerOpen", "random_support"): 0.047,
    ("DrawerOpen", "context_eligible"): 0.149,
    ("DrawerClose", "replay_action"): 0.049,
    ("DrawerClose", "shuffled_policy"): 0.033,
    ("DrawerClose", "random_support"): 0.048,
    ("DrawerClose", "context_eligible"): 0.030,
}

EXPECTED_MECHANISM_CI = {
    ("FPP", "replay_action"): (-0.028, 0.059),
    ("FPP", "shuffled_policy"): (-0.002, 0.085),
    ("FPP", "random_support"): (0.014, 0.101),
    ("FPP", "context_eligible"): (-0.006, 0.076),
    ("DrawerOpen", "replay_action"): (0.121, 0.729),
    ("DrawerOpen", "shuffled_policy"): (-0.080, 0.335),
    ("DrawerOpen", "random_support"): (-0.042, 0.136),
    ("DrawerOpen", "context_eligible"): (-0.042, 0.442),
    ("DrawerClose", "replay_action"): (-0.006, 0.128),
    ("DrawerClose", "shuffled_policy"): (-0.063, 0.119),
    ("DrawerClose", "random_support"): (-0.004, 0.115),
    ("DrawerClose", "context_eligible"): (-0.033, 0.092),
}

OPEN_PHASE_SOURCE = (
    ROOT
    / "metaworld_Drawer_Open_DDPG/paper_artifacts/"
    "drawer_open_ddpg_saber_fixed_target_diagnostic_all_10_2m/"
    "results/diagnostic_by_seed.csv"
)
CLOSE_PHASE_SOURCE = (
    ROOT
    / "metaworld_Drawer_Close_DDPG/experiments/"
    "DDPG-HER-B3-Critic-Only-Diagnostic-2M/diagnostic_artifact_all_10/"
    "results/diagnostic_by_seed.csv"
)
OPEN_PHASE_DEFINITION = (
    ROOT
    / "metaworld_Drawer_Open_DDPG/experiments/"
    "DDPG-HER-B3-Critic-Only-Fixed-Target-Diagnostic-2M/code/src/"
    "metacog_lite_v0/task_drawer_open.py"
)
CLOSE_PHASE_DEFINITION = (
    ROOT
    / "metaworld_Drawer_Close_DDPG/experiments/"
    "DDPG-HER-B3-Critic-Only-Diagnostic-2M/code/src/"
    "metacog_lite_v0/task_drawer_close.py"
)

PHASE_FIELDS = {
    "DrawerOpen": (
        ("approach", "phase_approach_batch_fraction", "phase_approach_support_rate"),
        ("contact", "phase_contact_batch_fraction", "phase_contact_support_rate"),
        ("opening", "phase_opening_batch_fraction", "phase_opening_support_rate"),
        ("open", "phase_open_batch_fraction", "phase_open_support_rate"),
    ),
    "DrawerClose": (
        ("approach", "phase_approach_batch_rate", "phase_approach_support_rate"),
        ("contact", "phase_contact_batch_rate", "phase_contact_support_rate"),
        ("closing", "phase_closing_batch_rate", "phase_closing_support_rate"),
        ("closed", "phase_closed_batch_rate", "phase_closed_support_rate"),
    ),
}


def signed(value: float) -> str:
    return f"{value:+.3f}"


def delta_ci_text(row: Mapping[str, object]) -> str:
    return (
        f"{signed(float(row['mean_difference']))} "
        f"[{float(row['ci95_low']):.3f},{float(row['ci95_high']):.3f}]"
    )


def mechanism_interval(
    task: str, differences: Sequence[float]
) -> tuple[float, float, str, int, int | None]:
    """Reproduce the frozen Main Table 2 interval conventions."""
    if task == "FPP":
        rng = random.Random(27027)
        count = len(differences)
        means = [
            statistics.fmean(
                differences[rng.randrange(count)] for _ in range(count)
            )
            for _ in range(10_000)
        ]
        return (
            percentile(means, 0.025),
            percentile(means, 0.975),
            "paired_percentile_bootstrap",
            10_000,
            27027,
        )

    count = len(differences)
    means = [
        statistics.fmean(differences[index] for index in assignment)
        for assignment in itertools.product(range(count), repeat=count)
    ]
    return (
        percentile(means, 0.025),
        percentile(means, 0.975),
        "exact_paired_bootstrap_enumeration",
        count**count,
        None,
    )


def build_mechanism_controls() -> None:
    out = OUTPUT_ROOT / "mechanism_controls_ddpg"
    per_seed: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}

    for task, (budget, grid_step) in TASK_SPECS.items():
        grid = tuple(range(grid_step, budget + 1, grid_step))
        for variant in VARIANT_LABELS:
            template = MECHANISM_SPECS[(task, variant)]
            for seed in SEEDS:
                path = ROOT / template.format(seed=seed)
                curve = exact_curve(path, grid)
                row = {
                    "task": task,
                    "backbone": "DDPG",
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "seed": seed,
                    **metrics(curve, budget, grid_step),
                    "steps_per_second": final_steps_per_second(path),
                }
                per_seed.append(row)
                bind_run_sources(sources, path)

    aggregate = aggregate_metric_rows(
        per_seed, ("task", "backbone", "variant", "variant_label")
    )
    aggregate_by_key = {
        (str(row["task"]), str(row["variant"])): row for row in aggregate
    }

    pairs: list[dict[str, object]] = []
    for task in TASK_SPECS:
        main_rows = {
            int(row["seed"]): row
            for row in per_seed
            if row["task"] == task and row["variant"] == "main"
        }
        for variant in tuple(VARIANT_LABELS)[1:]:
            control_rows = {
                int(row["seed"]): row
                for row in per_seed
                if row["task"] == task and row["variant"] == variant
            }
            differences = [
                float(main_rows[seed]["auc"]) - float(control_rows[seed]["auc"])
                for seed in SEEDS
            ]
            low, high, interval_method, resamples, interval_seed = (
                mechanism_interval(task, differences)
            )
            pair = {
                "task": task,
                "backbone": "DDPG",
                "comparison": f"main_minus_{variant}",
                "control_variant": variant,
                "control_label": VARIANT_LABELS[variant],
                "metric": "auc",
                "mean_difference": statistics.fmean(differences),
                "ci95_low": low,
                "ci95_high": high,
                "interval_method": interval_method,
                "bootstrap_resamples": resamples,
                "bootstrap_seed": interval_seed,
                "positive_pairs": sum(value > 0 for value in differences),
                "n_pairs": len(differences),
            }
            expected = EXPECTED_MAIN_MINUS_CONTROL[(task, variant)]
            if round(float(pair["mean_difference"]), 3) != expected:
                raise ValueError(
                    f"{task}/{variant}: delta {pair['mean_difference']:.6f} "
                    f"does not reproduce Main Table 2 value {expected:.3f}"
                )
            expected_low, expected_high = EXPECTED_MECHANISM_CI[(task, variant)]
            if (
                round(float(pair["ci95_low"]), 3) != expected_low
                or round(float(pair["ci95_high"]), 3) != expected_high
            ):
                raise ValueError(
                    f"{task}/{variant}: interval "
                    f"[{pair['ci95_low']:.6f},{pair['ci95_high']:.6f}] does not "
                    f"reproduce Main Table 2 [{expected_low:.3f},{expected_high:.3f}]"
                )
            pairs.append(pair)

    per_seed_fields = (
        "task", "backbone", "variant", "variant_label", "seed", "auc",
        "final_quarter_auc", "final_success", "hit90_step", "steps_per_second",
    )
    aggregate_fields = (
        "task", "backbone", "variant", "variant_label", "n_seeds",
        "auc_mean", "auc_sample_sd", "final_quarter_auc_mean",
        "final_quarter_auc_sample_sd", "final_success_mean",
        "final_success_sample_sd", "hit90_reached", "hit90_total",
        "hit90_median_reached", "steps_per_second_mean",
        "steps_per_second_sample_sd",
    )
    pair_fields = (
        "task", "backbone", "comparison", "control_variant", "control_label",
        "metric", "mean_difference", "ci95_low", "ci95_high",
        "interval_method", "bootstrap_resamples", "bootstrap_seed",
        "positive_pairs", "n_pairs",
    )
    write_csv(out / "per_seed_metrics.csv", per_seed, per_seed_fields)
    write_csv(out / "aggregate_metrics.csv", aggregate, aggregate_fields)
    write_csv(out / "paired_statistics.csv", pairs, pair_fields)
    write_json(
        out / "source_manifest.json",
        {
            "scope": "DDPG mechanism controls on FPP, DrawerOpen, and DrawerClose",
            "sources": [sources[key] for key in sorted(sources)],
        },
    )
    (out / "README.md").write_text(
        "# DDPG mechanism controls\n\n"
        "Exact-grid metrics for SABER Main and the four mechanism controls on "
        "FPP, DrawerOpen, and DrawerClose over paired seeds 1--5. The paired "
        "statistics reproduce the Main Table 2 Main-minus-control AUC "
        "differences. No curve is smoothed or interpolated.\n",
        encoding="utf-8",
    )
    artifact_manifest(out)

    pair_by_key = {
        (str(row["task"]), str(row["control_variant"])): row for row in pairs
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        "Cohort & Method/Control & AUC & FQ-AUC & Final & Hit90 & Steps/s & $\\Delta$AUC \\\\",
        "\\midrule",
    ]
    for task_index, task in enumerate(TASK_SPECS):
        for variant_index, variant in enumerate(VARIANT_LABELS):
            row = aggregate_by_key[(task, variant)]
            task_cell = f"{task}--DDPG" if variant_index == 0 else ""
            delta = "--" if variant == "main" else delta_ci_text(pair_by_key[(task, variant)])
            lines.append(
                f"{task_cell} & {TABLE_VARIANT_LABELS[variant]} & "
                f"${fmt(float(row['auc_mean']))} \\pm {fmt(float(row['auc_sample_sd']))}$ & "
                f"${fmt(float(row['final_quarter_auc_mean']))} \\pm "
                f"{fmt(float(row['final_quarter_auc_sample_sd']))}$ & "
                f"${fmt(float(row['final_success_mean']))} \\pm "
                f"{fmt(float(row['final_success_sample_sd']))}$ & "
                f"{hit90_text(row)} & {float(row['steps_per_second_mean']):.1f} & "
                f"${delta}$ \\\\"
            )
        if task_index != len(TASK_SPECS) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{DDPG mechanism controls over paired seeds 1--5. Values are mean "
            "$\\pm$ sample SD on each task's exact grid. Hit90 gives reaching seeds and "
            "their reached-seed median step. The final column reports SABER Main-minus-control "
            "AUC differences with frozen paired-bootstrap 95\\% CIs; positive values "
            "favor Main. FPP uses 10,000 seeded resamples and Drawer controls enumerate "
            "all $5^5$ paired resamples, matching the validated source artifacts. Steps/s "
            "is descriptive rather than hardware-normalized.}",
            "\\label{tab:supp-mechanism-controls}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_mechanism_controls_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def integer_count(value: float, label: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 0.01:
        raise ValueError(f"{label}: expected integer count, got {value}")
    return rounded


def build_phase_seed_results() -> None:
    out = OUTPUT_ROOT / "phase_seed_results"
    rows: list[dict[str, object]] = []
    for task, source in (
        ("DrawerOpen", OPEN_PHASE_SOURCE),
        ("DrawerClose", CLOSE_PHASE_SOURCE),
    ):
        for source_row in read_csv(source):
            seed = int(source_row["seed"])
            snapshot_count = int(source_row["snapshot_count"])
            total_samples = snapshot_count * BATCH_SIZE
            seed_phase_total = 0
            for phase, fraction_field, support_field in PHASE_FIELDS[task]:
                fraction = float(source_row[fraction_field])
                support_rate = float(source_row[support_field])
                phase_count = integer_count(
                    fraction * total_samples, f"{task}/seed{seed}/{phase}"
                )
                support_count = integer_count(
                    support_rate * phase_count,
                    f"{task}/seed{seed}/{phase}/support",
                )
                seed_phase_total += phase_count
                rows.append(
                    {
                        "task": task,
                        "backbone": "DDPG",
                        "cohort": "primary" if seed <= 5 else "extension",
                        "seed": seed,
                        "phase": phase,
                        "snapshot_count": snapshot_count,
                        "batch_size": BATCH_SIZE,
                        "total_logged_samples": total_samples,
                        "phase_sample_count": phase_count,
                        "phase_fraction": fraction,
                        "support_positive_count": support_count,
                        "support_positive_rate": support_rate,
                    }
                )
            if seed_phase_total != total_samples:
                raise ValueError(
                    f"{task}/seed{seed}: phase counts {seed_phase_total} "
                    f"do not sum to {total_samples}"
                )

    fields = (
        "task", "backbone", "cohort", "seed", "phase", "snapshot_count",
        "batch_size", "total_logged_samples", "phase_sample_count",
        "phase_fraction", "support_positive_count", "support_positive_rate",
    )
    write_csv(out / "phase_by_seed.csv", rows, fields)

    cohort_rows: list[dict[str, object]] = []
    for task in PHASE_FIELDS:
        for cohort in ("primary", "extension"):
            for phase, _, _ in PHASE_FIELDS[task]:
                selected = [
                    row for row in rows
                    if row["task"] == task
                    and row["cohort"] == cohort
                    and row["phase"] == phase
                ]
                phase_count = sum(int(row["phase_sample_count"]) for row in selected)
                support_count = sum(
                    int(row["support_positive_count"]) for row in selected
                )
                total = sum(int(row["total_logged_samples"]) for row in selected)
                cohort_rows.append(
                    {
                        "task": task,
                        "backbone": "DDPG",
                        "cohort": cohort,
                        "phase": phase,
                        "n_seeds": len(selected),
                        "total_logged_samples": total,
                        "phase_sample_count": phase_count,
                        "phase_fraction": phase_count / total,
                        "support_positive_count": support_count,
                        "support_positive_rate": support_count / phase_count,
                    }
                )
    cohort_fields = (
        "task", "backbone", "cohort", "phase", "n_seeds",
        "total_logged_samples", "phase_sample_count", "phase_fraction",
        "support_positive_count", "support_positive_rate",
    )
    write_csv(out / "phase_cohort_counts.csv", cohort_rows, cohort_fields)
    write_json(
        out / "source_manifest.json",
        {
            "scope": "Drawer phase counts and seed-level support rates",
            "batch_size": BATCH_SIZE,
            "sources": [
                source_entry(OPEN_PHASE_SOURCE),
                source_entry(CLOSE_PHASE_SOURCE),
                source_entry(OPEN_PHASE_DEFINITION),
                source_entry(CLOSE_PHASE_DEFINITION),
            ],
        },
    )
    (out / "README.md").write_text(
        "# Drawer phase counts and seed-level results\n\n"
        "Exact phase sample counts and support-positive counts reconstructed "
        "from the validated per-seed diagnostic summaries. Each seed contains "
        "9,976 logged minibatches of 256 samples. Seeds 1--5 are primary and "
        "seeds 6--10 are separate extensions. The source manifest also binds "
        "the task files that define the 0.08, 0.12, and 0.05 phase thresholds.\n",
        encoding="utf-8",
    )
    artifact_manifest(out)

    indexed = {
        (str(row["task"]), int(row["seed"]), str(row["phase"])): row
        for row in rows
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Task & Seed & Approach & Contact & Opening/Closing & Open/Closed \\\\",
        "\\midrule",
    ]
    for task_index, task in enumerate(PHASE_FIELDS):
        phase_names = [item[0] for item in PHASE_FIELDS[task]]
        for seed in SEEDS:
            cells = []
            for phase in phase_names:
                row = indexed[(task, seed, phase)]
                cells.append(
                    f"{int(row['phase_sample_count']):,} "
                    f"({float(row['support_positive_rate']):.3f})"
                )
            task_cell = task if seed == 1 else ""
            lines.append(f"{task_cell} & {seed} & " + " & ".join(cells) + " \\\\")
        if task_index != len(PHASE_FIELDS) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Primary Drawer phase counts and seed-level support activation. "
            "Each cell gives the exact number of logged samples in the phase, followed "
            "by the support-positive rate in parentheses. Each seed contributes "
            "2,553,856 samples (9,976 minibatches of size 256). Opening/Open apply to "
            "DrawerOpen and Closing/Closed to DrawerClose. Exact seeds 6--10 extension "
            "rows and support-positive counts are provided in the phase artifact.}",
            "\\label{tab:supp-phase-seed-counts}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_phase_seed_counts_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_prioritization_pair_table() -> None:
    source = OUTPUT_ROOT / "primary_seed_results" / "paired_statistics.csv"
    rows = read_csv(source)
    indexed = {
        (row["task"], row["backbone"], row["comparison"], row["metric"]): row
        for row in rows
    }
    task_order = (
        ("FPP", "DDPG"), ("FPP", "TD3"), ("FPP", "SAC"),
        ("DrawerOpen", "DDPG"), ("DrawerOpen", "TD3"),
        ("DrawerClose", "DDPG"), ("DrawerClose", "TD3"),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Cohort & Comparator & $\\Delta$AUC (95\\% CI) & $\\Delta$FQ-AUC (95\\% CI) \\\\",
        "\\midrule",
    ]
    for cohort_index, (task, backbone) in enumerate(task_order):
        for comparator_index, comparator in enumerate(("HGR", "PER")):
            auc = indexed[(task, backbone, f"SABER-{comparator}", "auc")]
            fq = indexed[
                (task, backbone, f"SABER-{comparator}", "final_quarter_auc")
            ]
            cohort = f"{task}--{backbone}" if comparator_index == 0 else ""
            lines.append(
                f"{cohort} & {comparator} & ${delta_ci_text(auc)}$ & "
                f"${delta_ci_text(fq)}$ \\\\"
            )
        if cohort_index != len(task_order) - 1:
            lines.append("\\addlinespace[1pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Paired SABER-minus-prioritization differences over seeds 1--5. "
            "Positive values favor SABER. Intervals are 10,000-resample paired "
            "percentile-bootstrap 95\\% CIs computed from within-seed differences.}",
            "\\label{tab:supp-prioritization-paired}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_primary_paired_comparisons_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    build_mechanism_controls()
    build_phase_seed_results()
    build_prioritization_pair_table()
    print("Built mechanism controls, phase counts, and prioritization CIs.")


if __name__ == "__main__":
    main()
