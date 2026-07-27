#!/usr/bin/env python3
"""Audit final-quarter success AUC for every primary comparison run.

This script is intentionally read-only. It validates each exact evaluation grid,
recomputes the frozen full-budget AUC, and reports normalized success AUC over
the final 25% of the environment-step budget without smoothing or interpolation.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(1, 6))
METHODS = ("HER", "SABER", "HGR", "PER")

# task, backbone, budget, exact common evaluation spacing
COHORTS = (
    ("FPP", "DDPG", 1_000_000, 10_000),
    ("FPP", "TD3", 1_000_000, 10_000),
    ("FPP", "SAC", 1_000_000, 50_000),
    ("DrawerOpen", "DDPG", 2_000_000, 10_000),
    ("DrawerOpen", "TD3", 2_000_000, 10_000),
    ("DrawerClose", "DDPG", 2_000_000, 10_000),
    ("DrawerClose", "TD3", 2_000_000, 10_000),
)

# kind, metric, path, optional method key for combined CSVs
SPECS = {
    ("FPP", "DDPG", "HER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/B0/fetch_pick_and_place_b0_s{seed}_1000000_ls1024/metrics_selected_10k.csv", None),
    ("FPP", "DDPG", "SABER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512/fetch_pick_and_place_b3_critic_only_runtime_opt_v4_ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("FPP", "DDPG", "HGR"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_ddpg_hgr_pair_priority_is_1m/fetch_pick_and_place_ddpg_hgr_tdpriority_s{seed}_1000000/metrics_selected_10k.csv", None),
    ("FPP", "DDPG", "PER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/ddpg_her_per_full_ls512/fetch_pick_and_place_ddpg_her_per_full_ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("FPP", "TD3", "HER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/data/td3_her/{seed}/metrics_selected_10k.csv", None),
    ("FPP", "TD3", "SABER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/data/td3_saber/{seed}/metrics_selected_10k.csv", None),
    ("FPP", "TD3", "HGR"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_hgr_pair_priority_is_1m/fetch_pick_and_place_td3_hgr_tdpriority_s{seed}_1000000/metrics_selected_10k.csv", None),
    ("FPP", "TD3", "PER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_sac_her_per_full_1m/data/td3_her_per_full/seed{seed}/eval_metrics.csv", None),
    ("FPP", "SAC", "HER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("FPP", "SAC", "SABER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("FPP", "SAC", "HGR"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_sac_hgr_pair_priority_is_1m/fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000/metrics_selected_25k.csv", None),
    ("FPP", "SAC", "PER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_sac_her_per_full_1m/data/sac_her_per_full/seed{seed}/eval_metrics.csv", None),
    ("DrawerOpen", "DDPG", "HER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B0-2M/drawer_open_DDPG-HER-B0-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerOpen", "DDPG", "SABER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M/drawer_open_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerOpen", "DDPG", "HGR"): (
        "combined", "eval_env_success_mean",
        "metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/drawer_open_ddpg_hgr_pair_priority_is_2m/evaluation_curves_10k.csv", "hgr"),
    ("DrawerOpen", "DDPG", "PER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-PER-Full-2M/drawer_open_ddpg_her_per_full_s{seed}_2000000/metrics.csv", None),
    ("DrawerOpen", "TD3", "HER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B0-2M/drawer_open_TD3-HER-B0-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerOpen", "TD3", "SABER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Critic-Only-2M/drawer_open_TD3-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerOpen", "TD3", "HGR"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_TD3_HGR/runs/td3_hgr_pair_priority_is_2m/TD3-HER-HGR-2m/drawer_open_td3_hgr_pair_priority_is_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerOpen", "TD3", "PER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Open_TD3/results_2m/TD3-HER-PER-Full-2M/drawer_open_td3_her_per_full_s{seed}_2000000/metrics.csv", None),
    ("DrawerClose", "DDPG", "HER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B0-2M/drawer_close_DDPG-HER-B0-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerClose", "DDPG", "SABER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M/drawer_close_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerClose", "DDPG", "HGR"): (
        "combined", "eval_env_success_mean",
        "metaworld_Drawer_Close_DDPG_HGR/paper_artifacts/drawer_close_ddpg_hgr_pair_priority_is_2m/evaluation_curves_10k.csv", "hgr"),
    ("DrawerClose", "DDPG", "PER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-PER-Full-2M/drawer_close_ddpg_her_per_full_s{seed}_2000000/metrics.csv", None),
    ("DrawerClose", "TD3", "HER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B0-2M/drawer_close_TD3-HER-B0-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerClose", "TD3", "SABER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B3-Critic-Only-2M/drawer_close_TD3-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerClose", "TD3", "HGR"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_TD3_HGR/runs/td3_hgr_pair_priority_is_2m/TD3-HER-HGR-2m/drawer_close_td3_hgr_pair_priority_is_s{seed}_2m_cuda/metrics.csv", None),
    ("DrawerClose", "TD3", "PER"): (
        "raw", "eval/env_success_mean",
        "metaworld_Drawer_Close_TD3/results_2m/TD3-HER-PER-Full-2M/drawer_close_td3_her_per_full_s{seed}_2000000/metrics.csv", None),
}

# Frozen table values, used only as an independent provenance check.
EXPECTED_FULL_AUC = {
    ("FPP", "DDPG"): {"HER": .539, "SABER": .609, "HGR": .338, "PER": .460},
    ("FPP", "TD3"): {"HER": .372, "SABER": .442, "HGR": .367, "PER": .369},
    ("FPP", "SAC"): {"HER": .471, "SABER": .583, "HGR": .484, "PER": .690},
    ("DrawerOpen", "DDPG"): {"HER": .207, "SABER": .754, "HGR": .688, "PER": .825},
    ("DrawerOpen", "TD3"): {"HER": .127, "SABER": .362, "HGR": .371, "PER": .406},
    ("DrawerClose", "DDPG"): {"HER": .251, "SABER": .324, "HGR": .349, "PER": .308},
    ("DrawerClose", "TD3"): {"HER": .329, "SABER": .549, "HGR": .602, "PER": .524},
}


def finite(value: str | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_curve(spec: tuple[str, str, str, str | None], seed: int) -> dict[int, float]:
    kind, metric, path_template, method_key = spec
    path = ROOT / path_template.format(seed=seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    curve: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", metric}
        if kind == "combined":
            required |= {"seed", "method_key"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            if kind == "combined" and (
                int(row["seed"]) != seed or row["method_key"] != method_key
            ):
                continue
            value = finite(row[metric])
            if value is None:
                continue
            step = int(float(row["step"]))
            if step in curve:
                raise ValueError(f"{path}: duplicate evaluation at step {step}")
            curve[step] = value
    return curve


def normalized_auc(curve: dict[int, float], grid: tuple[int, ...]) -> float:
    area = sum(
        0.5 * (curve[left] + curve[right]) * (right - left)
        for left, right in zip(grid, grid[1:])
    )
    return area / (grid[-1] - grid[0])


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(differences: tuple[float, ...]) -> tuple[float, float]:
    rng = random.Random(2027)
    n = len(differences)
    means = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(10_000)
    ]
    return percentile(means, .025), percentile(means, .975)


def main() -> None:
    results: dict[tuple[str, str, str], tuple[float, ...]] = {}
    print("task,backbone,method,full_auc_mean,final_quarter_auc_mean,final_quarter_auc_sd")
    for task, backbone, budget, grid_step in COHORTS:
        full_grid = tuple(range(grid_step, budget + 1, grid_step))
        quarter_start = 3 * budget // 4
        quarter_grid = tuple(range(quarter_start, budget + 1, grid_step))
        if quarter_grid[0] != quarter_start or quarter_grid[-1] != budget:
            raise ValueError(f"{task}/{backbone}: final-quarter endpoints not on grid")
        for method in METHODS:
            seed_full = []
            seed_quarter = []
            for seed in SEEDS:
                observed = read_curve(SPECS[(task, backbone, method)], seed)
                missing = sorted(set(full_grid) - set(observed))
                if missing:
                    raise ValueError(
                        f"{task}/{backbone}/{method}/seed{seed}: missing {missing[:5]}"
                    )
                curve = {step: observed[step] for step in full_grid}
                if any(not 0 <= value <= 1 for value in curve.values()):
                    raise ValueError(f"{task}/{backbone}/{method}/seed{seed}: invalid success")
                seed_full.append(normalized_auc(curve, full_grid))
                seed_quarter.append(normalized_auc(curve, quarter_grid))
            full_mean = statistics.fmean(seed_full)
            expected = EXPECTED_FULL_AUC[(task, backbone)][method]
            if round(full_mean, 3) != expected:
                raise ValueError(
                    f"{task}/{backbone}/{method}: full AUC {full_mean:.9f} != table {expected:.3f}"
                )
            quarter_values = tuple(seed_quarter)
            results[(task, backbone, method)] = quarter_values
            print(
                f"{task},{backbone},{method},{full_mean:.9f},"
                f"{statistics.fmean(quarter_values):.9f},"
                f"{statistics.stdev(quarter_values):.9f}"
            )

    print("\npaired_final_quarter_comparisons")
    print("task,backbone,comparison,mean_delta,ci_low,ci_high,positive_pairs")
    for task, backbone, _, _ in COHORTS:
        saber = results[(task, backbone, "SABER")]
        for control in ("HER", "HGR", "PER"):
            other = results[(task, backbone, control)]
            differences = tuple(a - b for a, b in zip(saber, other))
            low, high = paired_bootstrap_ci(differences)
            print(
                f"{task},{backbone},SABER-{control},"
                f"{statistics.fmean(differences):.9f},{low:.9f},{high:.9f},"
                f"{sum(value > 0 for value in differences)}/{len(differences)}"
            )


if __name__ == "__main__":
    main()
