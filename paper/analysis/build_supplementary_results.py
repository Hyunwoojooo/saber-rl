#!/usr/bin/env python3
"""Build supplementary sensitivity, scope, and seed-level result artifacts.

The builder is fail-closed on evaluation grids and never smooths, interpolates,
extrapolates, or forward-fills. Raw experiment files remain in their canonical
locations; the generated artifacts contain statistics and cryptographic source
manifests only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from audit_primary_final_quarter_auc import (
    COHORTS,
    EXPECTED_FULL_AUC,
    METHODS,
    SEEDS,
    SPECS,
    read_curve,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "paper/supplementary_artifacts"
TABLE_ROOT = ROOT / "paper/tables"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2027
SUCCESS_FIELD = "eval/env_success_mean"
SUPPORT_FIELD = "lite/psi_A_policy_rate"


ACTION_BIN_SPECS = {
    "k3": {
        "label": "K=3",
        "bins": 3,
        "path": (
            "experiment_FPP/paper_artifacts/"
            "fpp_td3_saber_action_bin_sensitivity_ls512_1m/k3/data/"
            "fetch_pick_and_place_td3_saber_action_bins_3_critic_only_v4_"
            "ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv"
        ),
    },
    "k5_main": {
        "label": "K=5 (Main)",
        "bins": 5,
        "path": (
            "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/"
            "data/td3_saber/{seed}/metrics_selected_10k.csv"
        ),
    },
    "k7": {
        "label": "K=7",
        "bins": 7,
        "path": (
            "experiment_FPP/runs/td3_saber_action_bin_sensitivity_ls512_1m/"
            "k7/fetch_pick_and_place_td3_saber_action_bins_7_critic_only_"
            "v4_ls512_s{seed}_1000000_ls512/metrics_selected_10k.csv"
        ),
    },
}


ACTOR_SCOPE_SPECS = {
    ("FPP", "critic_only"): (
        1_000_000,
        "experiment_FPP/paper_artifacts/"
        "fpp_ddpg_saber_actor_only_v4_ls512_1m/data/critic_only/"
        "seed_{seed}/metrics_selected_10k.csv",
    ),
    ("FPP", "actor_only"): (
        1_000_000,
        "experiment_FPP/paper_artifacts/"
        "fpp_ddpg_saber_actor_only_v4_ls512_1m/data/actor_only/"
        "seed_{seed}/metrics_selected_10k.csv",
    ),
    ("FPP", "actor_critic"): (
        1_000_000,
        "experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512/"
        "fetch_pick_and_place_ddpg_saber_actor_critic_v4_ls512_s{seed}_"
        "1000000_ls512/metrics_selected_10k.csv",
    ),
    ("DrawerOpen", "critic_only"): (
        2_000_000,
        "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M/"
        "drawer_open_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/"
        "metrics.csv",
    ),
    ("DrawerOpen", "actor_only"): (
        2_000_000,
        "metaworld_Drawer_Open_DDPG/paper_artifacts/"
        "drawer_open_ddpg_saber_actor_only_2m/data/actor_only/"
        "seed_{seed}/metrics_selected_10k.csv",
    ),
    ("DrawerOpen", "actor_critic"): (
        2_000_000,
        "metaworld_Drawer_Open_DDPG/paper_artifacts/"
        "drawer_open_ddpg_saber_actor_critic_2m/data/actor_critic/"
        "seed_{seed}/metrics_selected_10k.csv",
    ),
    ("DrawerClose", "critic_only"): (
        2_000_000,
        "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M/"
        "drawer_close_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda/"
        "metrics.csv",
    ),
    ("DrawerClose", "actor_only"): (
        2_000_000,
        "metaworld_Drawer_Close_DDPG/paper_artifacts/"
        "drawer_close_ddpg_saber_actor_only_2m/data/actor_only/"
        "seed_{seed}/metrics_selected_10k.csv",
    ),
    ("DrawerClose", "actor_critic"): (
        2_000_000,
        "metaworld_Drawer_Close_DDPG/paper_artifacts/"
        "drawer_close_ddpg_saber_actor_critic_2m/data/actor_critic/"
        "seed_{seed}/metrics_selected_10k.csv",
    ),
}

SCOPE_LABELS = {
    "critic_only": "Critic-Only (Main)",
    "actor_only": "Actor-Only",
    "actor_critic": "Actor+Critic",
}


def finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def bind_run_sources(
    sources: dict[str, dict[str, object]], metrics_path: Path
) -> None:
    paths = [metrics_path]
    for name in ("config.json", "env_snapshot.json", "summary.json"):
        path = metrics_path.parent / name
        if path.is_file():
            paths.append(path)
    for path in paths:
        sources[str(path.relative_to(ROOT))] = source_entry(path)


def read_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def final_steps_per_second(metrics_path: Path) -> float:
    summary_path = metrics_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_config(summary_path)
    value = finite(summary.get("final_steps_per_sec"))
    if value is None or value <= 0.0:
        raise ValueError(f"{summary_path}: missing positive final_steps_per_sec")
    return value


def config_seed(config: Mapping[str, object], path: Path) -> int:
    seeds = config.get("seeds")
    if not isinstance(seeds, dict) or "env_seed" not in seeds:
        raise ValueError(f"{path}: missing seeds.env_seed")
    return int(seeds["env_seed"])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: "" if row.get(field) is None else row.get(field) for field in fields}
            )


def read_raw_series(path: Path, field: str) -> dict[int, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", field}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            value = finite(row.get(field))
            step_value = finite(row.get("step"))
            if value is None or step_value is None:
                continue
            step = int(step_value)
            previous = values.get(step)
            if previous is not None and not math.isclose(
                previous, value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{path}: conflicting {field} values at step {step}: "
                    f"{previous} versus {value}"
                )
            values[step] = value
    return values


def exact_curve(path: Path, grid: tuple[int, ...]) -> dict[int, float]:
    observed = read_raw_series(path, SUCCESS_FIELD)
    missing = sorted(set(grid) - set(observed))
    if missing:
        raise ValueError(f"{path}: missing exact checkpoints {missing[:10]}")
    curve = {step: observed[step] for step in grid}
    if any(not 0.0 <= value <= 1.0 for value in curve.values()):
        raise ValueError(f"{path}: success outside [0,1]")
    return curve


def normalized_auc(curve: Mapping[int, float], grid: tuple[int, ...]) -> float:
    area = math.fsum(
        0.5 * (curve[left] + curve[right]) * (right - left)
        for left, right in zip(grid, grid[1:])
    )
    return area / (grid[-1] - grid[0])


def first_hit90(curve: Mapping[int, float], grid: tuple[int, ...]) -> int | None:
    return next((step for step in grid if curve[step] >= 0.9), None)


def metrics(curve: Mapping[int, float], budget: int, grid_step: int) -> dict[str, object]:
    grid = tuple(range(grid_step, budget + 1, grid_step))
    quarter_grid = tuple(range(3 * budget // 4, budget + 1, grid_step))
    if quarter_grid[0] != 3 * budget // 4 or quarter_grid[-1] != budget:
        raise ValueError(f"invalid final-quarter grid for budget={budget}")
    return {
        "auc": normalized_auc(curve, grid),
        "final_quarter_auc": normalized_auc(curve, quarter_grid),
        "final_success": curve[budget],
        "hit90_step": first_hit90(curve, grid),
    }


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(differences: Sequence[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(differences)
    means = [
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def aggregate_metric_rows(
    per_seed: Sequence[Mapping[str, object]], group_fields: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in per_seed:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, rows in grouped.items():
        result = dict(zip(group_fields, key))
        result["n_seeds"] = len(rows)
        for field in ("auc", "final_quarter_auc", "final_success"):
            values = [float(row[field]) for row in rows]
            result[f"{field}_mean"], result[f"{field}_sample_sd"] = mean_sd(values)
        hits = [int(row["hit90_step"]) for row in rows if row["hit90_step"] is not None]
        result["hit90_reached"] = len(hits)
        result["hit90_total"] = len(rows)
        result["hit90_median_reached"] = statistics.median(hits) if hits else None
        if all(row.get("steps_per_second") is not None for row in rows):
            speeds = [float(row["steps_per_second"]) for row in rows]
            result["steps_per_second_mean"], result[
                "steps_per_second_sample_sd"
            ] = mean_sd(speeds)
        output.append(result)
    return output


def artifact_manifest(directory: Path) -> None:
    entries = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        entries.append(source_entry(path))
    write_json(
        directory / "artifact_manifest.json",
        {"root": str(directory.relative_to(ROOT)), "files": entries},
    )


def fmt(value: float) -> str:
    return f"{value:.3f}"


def hit90_text(row: Mapping[str, object]) -> str:
    reached = int(row["hit90_reached"])
    total = int(row["hit90_total"])
    median = row["hit90_median_reached"]
    if median is None:
        return f"{reached}/{total}"
    return f"{reached}/{total} ({float(median) / 1_000_000:.3f}M)"


def build_action_bin_sensitivity() -> None:
    out = OUTPUT_ROOT / "action_bin_sensitivity"
    grid = tuple(range(10_000, 1_000_001, 10_000))
    rows: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    for variant, spec in ACTION_BIN_SPECS.items():
        for seed in SEEDS:
            path = ROOT / str(spec["path"]).format(seed=seed)
            config_path = path.parent / "config.json"
            config = read_config(config_path)
            if int(config.get("action_bins_per_dim", -1)) != int(spec["bins"]):
                raise ValueError(f"{config_path}: action-bin setting does not match label")
            if config.get("weight_scope") != "critic_only":
                raise ValueError(f"{config_path}: expected critic_only scope")
            if config_seed(config, config_path) != seed:
                raise ValueError(f"{config_path}: seed mismatch")
            curve = exact_curve(path, grid)
            row = {
                "variant": variant,
                "label": spec["label"],
                "action_bins_per_dim": spec["bins"],
                "seed": seed,
                **metrics(curve, 1_000_000, 10_000),
            }
            support = read_raw_series(path, SUPPORT_FIELD)
            missing = sorted(set(grid) - set(support))
            if missing:
                raise ValueError(f"{path}: missing support snapshots {missing[:10]}")
            row["support_positive_rate"] = statistics.fmean(support[step] for step in grid)
            row["steps_per_second"] = final_steps_per_second(path)
            rows.append(row)
            bind_run_sources(sources, path)

    aggregates = aggregate_metric_rows(rows, ("variant", "label", "action_bins_per_dim"))
    by_variant = {
        str(row["variant"]): [item for item in rows if item["variant"] == row["variant"]]
        for row in aggregates
    }
    for aggregate in aggregates:
        values = [float(row["support_positive_rate"]) for row in by_variant[str(aggregate["variant"])]]
        aggregate["support_positive_rate_mean"], aggregate["support_positive_rate_sample_sd"] = mean_sd(values)

    paired_rows = []
    main_by_seed = {
        int(row["seed"]): row for row in rows if row["variant"] == "k5_main"
    }
    for variant in ("k3", "k7"):
        candidate = {int(row["seed"]): row for row in rows if row["variant"] == variant}
        for field in ("auc", "final_quarter_auc"):
            differences = [
                float(main_by_seed[seed][field]) - float(candidate[seed][field])
                for seed in SEEDS
            ]
            low, high = paired_bootstrap(differences)
            paired_rows.append(
                {
                    "comparison": f"k5_main_minus_{variant}",
                    "metric": field,
                    "mean_difference": statistics.fmean(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_pairs": sum(value > 0 for value in differences),
                    "n_pairs": len(differences),
                }
            )

    per_fields = (
        "variant", "label", "action_bins_per_dim", "seed", "auc",
        "final_quarter_auc", "final_success", "hit90_step", "support_positive_rate",
        "steps_per_second",
    )
    aggregate_fields = (
        "variant", "label", "action_bins_per_dim", "n_seeds", "auc_mean",
        "auc_sample_sd", "final_quarter_auc_mean", "final_quarter_auc_sample_sd",
        "final_success_mean", "final_success_sample_sd", "hit90_reached",
        "hit90_total", "hit90_median_reached", "support_positive_rate_mean",
        "support_positive_rate_sample_sd", "steps_per_second_mean",
        "steps_per_second_sample_sd",
    )
    paired_fields = (
        "comparison", "metric", "mean_difference", "ci95_low", "ci95_high",
        "positive_pairs", "n_pairs",
    )
    write_csv(out / "per_seed_metrics.csv", rows, per_fields)
    write_csv(out / "aggregate_metrics.csv", aggregates, aggregate_fields)
    write_csv(out / "paired_statistics.csv", paired_rows, paired_fields)
    write_json(
        out / "source_manifest.json",
        {
            "protocol": {
                "task": "FetchPickAndPlace-v4",
                "backbone": "TD3",
                "seeds": list(SEEDS),
                "evaluation_grid": "10k--1M by 10k",
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "smoothing": False,
                "interpolation": False,
            },
            "sources": list(sources.values()),
        },
    )
    (out / "README.md").write_text(
        "# FPP TD3 action-bin sensitivity\n\n"
        "Integrated seeds 1--5 analysis for K=3, Main K=5, and K=7. "
        "All metrics use the exact 10k--1M evaluation grid without smoothing, "
        "interpolation, extrapolation, or forward filling. Raw run files are "
        "not duplicated; `source_manifest.json` binds every source CSV by "
        "repository-relative path and SHA256.\n",
        encoding="utf-8",
    )
    artifact_manifest(out)

    agg_by_variant = {str(row["variant"]): row for row in aggregates}
    pair_by_variant = {
        str(row["comparison"]).split("k5_main_minus_", 1)[1]: row
        for row in paired_rows if row["metric"] == "auc"
    }
    tex = [
        "% Generated by paper/analysis/build_supplementary_results.py.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "Cohort & Method/Variant & AUC & FQ-AUC & Final & Hit90 & Steps/s & "
        "Support+ & $\\Delta$AUC \\\\",
        "\\midrule",
    ]
    for variant in ("k3", "k5_main", "k7"):
        row = agg_by_variant[variant]
        delta = "--"
        if variant != "k5_main":
            pair = pair_by_variant[variant]
            delta = (
                f"{float(pair['mean_difference']):+.3f}\\; "
                f"[{float(pair['ci95_low']):.3f},{float(pair['ci95_high']):.3f}]"
            )
        cohort = "FPP--TD3" if variant == "k3" else ""
        variant_label = f"$K={int(row['action_bins_per_dim'])}$"
        if variant == "k5_main":
            variant_label += " (Main)"
        tex.append(
            f"{cohort} & {variant_label} & "
            f"${fmt(float(row['auc_mean']))} \\pm {fmt(float(row['auc_sample_sd']))}$ & "
            f"${fmt(float(row['final_quarter_auc_mean']))} \\pm "
            f"{fmt(float(row['final_quarter_auc_sample_sd']))}$ & "
            f"${fmt(float(row['final_success_mean']))} \\pm "
            f"{fmt(float(row['final_success_sample_sd']))}$ & "
            f"{hit90_text(row)} & {float(row['steps_per_second_mean']):.1f} & "
            f"${fmt(float(row['support_positive_rate_mean']))} \\pm "
            f"{fmt(float(row['support_positive_rate_sample_sd']))}$ & "
            f"${delta}$ \\\\"
        )
    tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{FPP TD3 action-bin sensitivity over paired seeds 1--5. "
            "Values are mean $\\pm$ sample SD. Support+ is the support-positive "
            "rate. The final column reports Main-minus-variant paired AUC "
            "differences with 10,000-resample percentile-bootstrap 95\\% CIs; "
            "positive values favor Main $K=5$.}",
            "\\label{tab:supp-action-bin-sensitivity}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_action_bin_sensitivity_table.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )


def build_actor_scope() -> None:
    out = OUTPUT_ROOT / "actor_scope_ddpg"
    rows: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    missing: list[dict[str, object]] = []
    for (task, scope), (budget, template) in ACTOR_SCOPE_SPECS.items():
        expected_paths = [ROOT / template.format(seed=seed) for seed in SEEDS]
        present_paths = [path for path in expected_paths if path.is_file()]
        if not present_paths:
            missing.append(
                {
                    "task": task,
                    "backbone": "DDPG",
                    "scope": scope,
                    "status": "pending_completed_results",
                    "expected_metrics": [
                        str(path.relative_to(ROOT)) for path in expected_paths
                    ],
                }
            )
            continue
        if len(present_paths) != len(expected_paths):
            absent = [
                str(path.relative_to(ROOT))
                for path in expected_paths
                if not path.is_file()
            ]
            raise FileNotFoundError(
                f"{task}/{scope}: partial five-seed cohort; missing {absent}"
            )
        grid = tuple(range(10_000, budget + 1, 10_000))
        for seed in SEEDS:
            path = ROOT / template.format(seed=seed)
            config_path = path.parent / "config.json"
            if config_path.is_file():
                config = read_config(config_path)
                if config.get("weight_scope") != scope:
                    raise ValueError(
                        f"{config_path}: weight_scope={config.get('weight_scope')!r}, expected {scope!r}"
                    )
                if config_seed(config, config_path) != seed:
                    raise ValueError(f"{config_path}: seed mismatch")
                if int(config.get("total_steps", -1)) != budget:
                    raise ValueError(f"{config_path}: training-budget mismatch")
            elif scope != "critic_only":
                raise FileNotFoundError(config_path)
            curve = exact_curve(path, grid)
            rows.append(
                {
                    "task": task,
                    "backbone": "DDPG",
                    "scope": scope,
                    "scope_label": SCOPE_LABELS[scope],
                    "seed": seed,
                    **metrics(curve, budget, 10_000),
                    "steps_per_second": final_steps_per_second(path),
                }
            )
            bind_run_sources(sources, path)

    aggregates = aggregate_metric_rows(
        rows, ("task", "backbone", "scope", "scope_label")
    )
    expected_main = {
        "FPP": EXPECTED_FULL_AUC[("FPP", "DDPG")]["SABER"],
        "DrawerOpen": EXPECTED_FULL_AUC[("DrawerOpen", "DDPG")]["SABER"],
        "DrawerClose": EXPECTED_FULL_AUC[("DrawerClose", "DDPG")]["SABER"],
    }
    for row in aggregates:
        if row["scope"] == "critic_only" and round(float(row["auc_mean"]), 3) != expected_main[str(row["task"])]:
            raise ValueError(f"{row['task']}: actor-scope Main does not match Table 1")

    paired_rows = []
    for task in ("FPP", "DrawerOpen", "DrawerClose"):
        main = {
            int(row["seed"]): row
            for row in rows if row["task"] == task and row["scope"] == "critic_only"
        }
        for control in ("actor_only", "actor_critic"):
            candidate = {
                int(row["seed"]): row
                for row in rows if row["task"] == task and row["scope"] == control
            }
            if not candidate:
                continue
            for field in ("auc", "final_quarter_auc"):
                differences = [
                    float(main[seed][field]) - float(candidate[seed][field])
                    for seed in SEEDS
                ]
                low, high = paired_bootstrap(differences)
                paired_rows.append(
                    {
                        "task": task,
                        "comparison": f"critic_only_minus_{control}",
                        "metric": field,
                        "mean_difference": statistics.fmean(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                        "positive_pairs": sum(value > 0 for value in differences),
                        "n_pairs": len(differences),
                    }
                )

    per_fields = (
        "task", "backbone", "scope", "scope_label", "seed", "auc",
        "final_quarter_auc", "final_success", "hit90_step",
        "steps_per_second",
    )
    aggregate_fields = (
        "task", "backbone", "scope", "scope_label", "n_seeds", "auc_mean",
        "auc_sample_sd", "final_quarter_auc_mean", "final_quarter_auc_sample_sd",
        "final_success_mean", "final_success_sample_sd", "hit90_reached",
        "hit90_total", "hit90_median_reached", "steps_per_second_mean",
        "steps_per_second_sample_sd",
    )
    paired_fields = (
        "task", "comparison", "metric", "mean_difference", "ci95_low",
        "ci95_high", "positive_pairs", "n_pairs",
    )
    write_csv(out / "per_seed_metrics.csv", rows, per_fields)
    write_csv(out / "aggregate_metrics.csv", aggregates, aggregate_fields)
    write_csv(out / "paired_statistics.csv", paired_rows, paired_fields)
    write_json(out / "missing_sources.json", missing)
    write_json(
        out / "source_manifest.json",
        {
            "protocol": {
                "backbone": "DDPG",
                "seeds": list(SEEDS),
                "evaluation_grids": {"FPP": "10k--1M", "Drawer": "10k--2M"},
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "smoothing": False,
                "interpolation": False,
            },
            "sources": list(sources.values()),
            "missing": missing,
        },
    )
    (out / "README.md").write_text(
        "# DDPG actor-scope ablations\n\n"
        "Integrated exact-grid analysis for Critic-Only (Main), Actor-Only, "
        "and Actor+Critic on FPP, DrawerOpen, and DrawerClose. Conditions with "
        "no completed five-seed cohort are recorded in `missing_sources.json` "
        "rather than inferred; partially present cohorts fail closed. All "
        "reported cells use paired seeds 1--5.\n",
        encoding="utf-8",
    )
    artifact_manifest(out)

    agg = {(str(row["task"]), str(row["scope"])): row for row in aggregates}
    pair = {
        (str(row["task"]), str(row["comparison"])): row
        for row in paired_rows if row["metric"] == "auc"
    }
    tex = [
        "% Generated by paper/analysis/build_supplementary_results.py.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Cohort & Method/Variant & AUC & FQ-AUC & Final & Hit90 & Steps/s & "
        "$\\Delta$AUC (Main--variant) \\\\",
        "\\midrule",
    ]
    for task in ("FPP", "DrawerOpen", "DrawerClose"):
        for scope in ("critic_only", "actor_only", "actor_critic"):
            row = agg.get((task, scope))
            task_cell = f"{task}--DDPG" if scope == "critic_only" else ""
            if row is None:
                tex.append(
                    f"{task_cell} & {SCOPE_LABELS[scope]} & "
                    "\\multicolumn{6}{c}{\\textit{pending}} \\\\"
                )
                continue
            delta = "--"
            if scope != "critic_only":
                contrast = pair[(task, f"critic_only_minus_{scope}")]
                delta = (
                    f"{float(contrast['mean_difference']):+.3f} "
                    f"[{float(contrast['ci95_low']):.3f},{float(contrast['ci95_high']):.3f}]"
                )
            tex.append(
                f"{task_cell} & {SCOPE_LABELS[scope]} & "
                f"${fmt(float(row['auc_mean']))} \\pm {fmt(float(row['auc_sample_sd']))}$ & "
                f"${fmt(float(row['final_quarter_auc_mean']))} \\pm "
                f"{fmt(float(row['final_quarter_auc_sample_sd']))}$ & "
                f"${fmt(float(row['final_success_mean']))} \\pm "
                f"{fmt(float(row['final_success_sample_sd']))}$ & "
                f"{hit90_text(row)} & "
                f"{float(row['steps_per_second_mean']):.1f} & ${delta}$ \\\\"
            )
        if task != "DrawerClose":
            tex.append("\\addlinespace[2pt]")
    tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{DDPG actor-scope ablations over paired seeds 1--5. "
            "Values are mean $\\pm$ sample standard deviation. Hit90 reports "
            "the number of reaching seeds and the reached-seed median step. "
            "Steps/s is the descriptive seed mean defined in Section~2.3. "
            "The final column gives Main-minus-variant paired AUC differences "
            "with 95\\% bootstrap confidence intervals. Pending cells are "
            "left unreported until a complete five-seed cohort is available.}",
            "\\label{tab:supp-actor-scope}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / "supplementary_actor_scope_table.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )


def primary_rows_and_curves() -> tuple[list[dict[str, object]], dict, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str, str, int], dict[int, float]] = {}
    sources: dict[str, dict[str, object]] = {}
    for task, backbone, budget, grid_step in COHORTS:
        grid = tuple(range(grid_step, budget + 1, grid_step))
        for method in METHODS:
            spec = SPECS[(task, backbone, method)]
            for seed in SEEDS:
                path = ROOT / spec[2].format(seed=seed)
                observed = read_curve(spec, seed)
                missing = sorted(set(grid) - set(observed))
                if missing:
                    raise ValueError(
                        f"{task}/{backbone}/{method}/seed{seed}: missing {missing[:10]}"
                    )
                curve = {step: observed[step] for step in grid}
                curves[(task, backbone, method, seed)] = curve
                rows.append(
                    {
                        "task": task,
                        "backbone": backbone,
                        "method": method,
                        "seed": seed,
                        **metrics(curve, budget, grid_step),
                    }
                )
                sources[str(path.relative_to(ROOT))] = source_entry(path)
    return rows, curves, list(sources.values())


def write_primary_seed_tables(rows: Sequence[Mapping[str, object]]) -> None:
    by_key = {
        (str(row["task"]), str(row["backbone"]), str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    tex = ["% Generated by paper/analysis/build_supplementary_results.py."]
    for task, backbone, _, _ in COHORTS:
        label_task = task.lower().replace("open", "-open").replace("close", "-close")
        tex.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                "\\small",
                "\\setlength{\\tabcolsep}{7pt}",
                "\\begin{tabular}{llcccc}",
                "\\toprule",
                "Method & Seed & AUC & FQ-AUC & Final & Hit90 step \\\\",
                "\\midrule",
            ]
        )
        for method_index, method in enumerate(METHODS):
            for seed in SEEDS:
                row = by_key[(task, backbone, method, seed)]
                method_cell = method if seed == SEEDS[0] else ""
                hit = "--" if row["hit90_step"] is None else f"{int(row['hit90_step']) / 1_000_000:.3f}M"
                tex.append(
                    f"{method_cell} & {seed} & {float(row['auc']):.6f} & "
                    f"{float(row['final_quarter_auc']):.6f} & "
                    f"{float(row['final_success']):.3f} & {hit} \\\\"
                )
            if method_index != len(METHODS) - 1:
                tex.append("\\addlinespace[2pt]")
        tex.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                f"\\caption{{Seed-level primary results for {task} with {backbone}. "
                "Metrics use the exact shared checkpoint grid without smoothing "
                "or interpolation.}",
                f"\\label{{tab:supp-seeds-{label_task}-{backbone.lower()}}}",
                "\\end{table*}",
                "",
            ]
        )
    (TABLE_ROOT / "supplementary_primary_seed_results.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )


def build_primary_seed_results() -> dict:
    out = OUTPUT_ROOT / "primary_seed_results"
    rows, curves, sources = primary_rows_and_curves()
    aggregates = aggregate_metric_rows(rows, ("task", "backbone", "method"))
    for row in aggregates:
        expected = EXPECTED_FULL_AUC[(str(row["task"]), str(row["backbone"]))][str(row["method"])]
        if round(float(row["auc_mean"]), 3) != expected:
            raise ValueError(f"{row['task']}/{row['backbone']}/{row['method']}: Table 1 mismatch")

    paired_rows = []
    for task, backbone, _, _ in COHORTS:
        saber = {
            int(row["seed"]): row
            for row in rows
            if row["task"] == task and row["backbone"] == backbone and row["method"] == "SABER"
        }
        for control in ("HER", "HGR", "PER"):
            comparator = {
                int(row["seed"]): row
                for row in rows
                if row["task"] == task and row["backbone"] == backbone and row["method"] == control
            }
            for field in ("auc", "final_quarter_auc"):
                differences = [
                    float(saber[seed][field]) - float(comparator[seed][field])
                    for seed in SEEDS
                ]
                low, high = paired_bootstrap(differences)
                paired_rows.append(
                    {
                        "task": task,
                        "backbone": backbone,
                        "comparison": f"SABER-{control}",
                        "metric": field,
                        "mean_difference": statistics.fmean(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                        "positive_pairs": sum(value > 0 for value in differences),
                        "n_pairs": len(differences),
                    }
                )

    mean_curve_rows = []
    for task, backbone, budget, grid_step in COHORTS:
        grid = tuple(range(grid_step, budget + 1, grid_step))
        for method in METHODS:
            for step in grid:
                values = [curves[(task, backbone, method, seed)][step] for seed in SEEDS]
                mean, sd = mean_sd(values)
                mean_curve_rows.append(
                    {
                        "task": task,
                        "backbone": backbone,
                        "method": method,
                        "step": step,
                        "mean_success": mean,
                        "sample_sd": sd,
                        "n_seeds": len(SEEDS),
                    }
                )

    per_fields = (
        "task", "backbone", "method", "seed", "auc", "final_quarter_auc",
        "final_success", "hit90_step",
    )
    aggregate_fields = (
        "task", "backbone", "method", "n_seeds", "auc_mean", "auc_sample_sd",
        "final_quarter_auc_mean", "final_quarter_auc_sample_sd", "final_success_mean",
        "final_success_sample_sd", "hit90_reached", "hit90_total",
        "hit90_median_reached",
    )
    paired_fields = (
        "task", "backbone", "comparison", "metric", "mean_difference",
        "ci95_low", "ci95_high", "positive_pairs", "n_pairs",
    )
    curve_fields = (
        "task", "backbone", "method", "step", "mean_success", "sample_sd", "n_seeds",
    )
    write_csv(out / "per_seed_metrics.csv", rows, per_fields)
    write_csv(out / "aggregate_metrics.csv", aggregates, aggregate_fields)
    write_csv(out / "paired_statistics.csv", paired_rows, paired_fields)
    write_csv(out / "mean_learning_curves.csv", mean_curve_rows, curve_fields)
    write_json(
        out / "source_manifest.json",
        {
            "protocol": {
                "cohorts": [list(cohort) for cohort in COHORTS],
                "methods": list(METHODS),
                "seeds": list(SEEDS),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "smoothing": False,
                "interpolation": False,
            },
            "sources": sources,
        },
    )
    (out / "README.md").write_text(
        "# Primary seed-level results\n\n"
        "Seed-level and aggregate metrics for the seven primary task--backbone "
        "cohorts and HER, SABER, HGR, and PER. AUCs use each cohort's exact "
        "shared evaluation grid. No smoothing, interpolation, extrapolation, "
        "or forward filling is used. Raw curves remain at the paths bound in "
        "`source_manifest.json`.\n",
        encoding="utf-8",
    )
    write_primary_seed_tables(rows)
    artifact_manifest(out)
    return {"rows": rows, "curves": curves, "mean_curves": mean_curve_rows, "sources": sources}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    build_action_bin_sensitivity()
    build_actor_scope()
    build_primary_seed_results()
    (OUTPUT_ROOT / "README.md").write_text(
        "# Supplementary analysis artifacts\n\n"
        "Generated by `paper/analysis/build_supplementary_results.py`. Each "
        "subdirectory contains exact-grid results, a source manifest, and a "
        "self-excluding artifact manifest. Raw experiment data are referenced "
        "by path and hash rather than duplicated.\n",
        encoding="utf-8",
    )
    print("Built action-bin sensitivity, actor-scope, and primary seed artifacts.")


if __name__ == "__main__":
    main()
