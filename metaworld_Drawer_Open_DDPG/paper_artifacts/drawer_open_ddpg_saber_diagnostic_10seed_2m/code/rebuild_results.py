"""Rebuild all result tables from the compact artifact payload only.

The statistical unit is a seed.  Diagnostic snapshots are averaged within a
seed before any across-seed inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import spearmanr, t as student_t, ttest_1samp, wilcoxon


ARTIFACT_ID = "DRAWEROPEN-DDPG-SABER-DIAGNOSTIC-10SEED-2M"
SEEDS = tuple(range(1, 11))
PHASES = ("approach", "contact", "opening", "open")
SUCCESS_THRESHOLD = 0.90

RESULT_FILENAMES = (
    "performance_by_seed.csv",
    "performance_aggregate.csv",
    "performance_reproducibility_reference.csv",
    "successful_9seed_td_residual.csv",
    "successful_9seed_td_residual_summary.csv",
    "phase_conditioned_allocation.csv",
    "hit90_pre_post_by_seed.csv",
    "hit90_pre_post_summary.csv",
    "seed7_400k_windows.csv",
    "scanner_runtime.csv",
    "exploratory_correlation.csv",
)


def _is_finite(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any) -> float:
    if not _is_finite(value):
        return float("nan")
    return float(value)


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if _is_finite(value)]


def _mean(values: Iterable[Any]) -> float:
    finite = _finite(values)
    return sum(finite) / len(finite) if finite else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-15:
        return float("nan")
    return numerator / denominator


def _canonical(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == 0.0:
            return "0"
        return format(value, ".12g")
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(row.get(key)) for key in fieldnames})


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _auc(eval_rows: Sequence[Mapping[str, str]]) -> float:
    if not eval_rows:
        return float("nan")
    if len(eval_rows) == 1:
        return _float(eval_rows[0]["eval_success_mean"])
    steps = [int(row["step"]) for row in eval_rows]
    values = [_float(row["eval_success_mean"]) for row in eval_rows]
    area = 0.0
    for left, right, y_left, y_right in zip(steps[:-1], steps[1:], values[:-1], values[1:]):
        area += (right - left) * (y_left + y_right) * 0.5
    return area / max(1, steps[-1] - steps[0])


def _first_hit(eval_rows: Sequence[Mapping[str, str]], threshold: float) -> int | None:
    for row in eval_rows:
        if _float(row["eval_success_mean"]) >= threshold:
            return int(row["step"])
    return None


def _t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    finite = _finite(values)
    if len(finite) < 2:
        return float("nan"), float("nan")
    center = _mean(finite)
    variance = sum((value - center) ** 2 for value in finite) / (len(finite) - 1)
    sem = math.sqrt(variance / len(finite))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, len(finite) - 1))
    return center - critical * sem, center + critical * sem


def _wilcoxon_p(values: Sequence[float]) -> float:
    finite = _finite(values)
    if not finite or all(abs(value) < 1e-15 for value in finite):
        return 1.0
    return float(wilcoxon(finite, alternative="two-sided", method="auto").pvalue)


def _ttest_p(values: Sequence[float]) -> float:
    finite = _finite(values)
    if len(finite) < 2:
        return float("nan")
    return float(ttest_1samp(finite, popmean=0.0).pvalue)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def _seed_dir(artifact_root: Path, seed: int) -> Path:
    return artifact_root / "data" / "seeds" / f"seed_{seed:02d}"


def _load_seed_payload(artifact_root: Path, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    seed_dir = _seed_dir(artifact_root, seed)
    eval_rows = _read_csv(seed_dir / "eval_10k.csv")
    snapshots = _read_csv(seed_dir / "diagnostic_snapshots.csv.gz")
    summary = _load_json(seed_dir / "summary.json")
    return eval_rows, snapshots, summary


def _performance(eval_rows: Sequence[Mapping[str, str]], seed: int, cohort: str) -> dict[str, Any]:
    hit90 = _first_hit(eval_rows, SUCCESS_THRESHOLD)
    return {
        "artifact_id": ARTIFACT_ID,
        "cohort": cohort,
        "seed": seed,
        "evaluation_points": len(eval_rows),
        "final_success": _float(eval_rows[-1]["eval_success_mean"]),
        "best_success": max(_float(row["eval_success_mean"]) for row in eval_rows),
        "auc_success": _auc(eval_rows),
        "hit90_step": hit90,
        "reached_hit90": hit90 is not None,
        "successful": hit90 is not None,
    }


def _row_difference(row: Mapping[str, str], positive: str, negative: str) -> float:
    pos = _float(row.get(positive))
    neg = _float(row.get(negative))
    return pos - neg if math.isfinite(pos) and math.isfinite(neg) else float("nan")


def _weighted_phase_support(rows: Sequence[Mapping[str, str]], phase: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        fraction = _float(row.get(f"phase_{phase}_frac"))
        support = _float(row.get(f"phase_{phase}_support_rate"))
        if math.isfinite(fraction) and math.isfinite(support) and fraction > 0.0:
            numerator += fraction * support
            denominator += fraction
    return numerator / denominator if denominator > 0.0 else float("nan")


def _diagnostic_seed_stats(seed: int, rows: Sequence[Mapping[str, str]], successful: bool) -> dict[str, Any]:
    td_differences = [_row_difference(row, "td_support_pos_mean", "td_support_neg_mean") for row in rows]
    residual_differences = [
        _row_difference(row, "residual_support_pos_mean", "residual_support_neg_mean") for row in rows
    ]
    residual_rate_differences = [
        _row_difference(row, "residual_positive_rate_pos", "residual_positive_rate_neg") for row in rows
    ]
    tail = [row for row in rows if int(row["step"]) > 1_600_000]
    result: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "seed": seed,
        "successful": successful,
        "snapshot_count": len(rows),
        "support_mean": _mean(row.get("support_rate") for row in rows),
        "support_final_400k_mean": _mean(row.get("support_rate") for row in tail),
        "td_support_pos_mean": _mean(row.get("td_support_pos_mean") for row in rows),
        "td_support_neg_mean": _mean(row.get("td_support_neg_mean") for row in rows),
        "td_difference": _mean(td_differences),
        "residual_support_pos_mean": _mean(row.get("residual_support_pos_mean") for row in rows),
        "residual_support_neg_mean": _mean(row.get("residual_support_neg_mean") for row in rows),
        "residual_difference": _mean(residual_differences),
        "residual_positive_rate_pos": _mean(row.get("residual_positive_rate_pos") for row in rows),
        "residual_positive_rate_neg": _mean(row.get("residual_positive_rate_neg") for row in rows),
        "residual_positive_rate_difference": _mean(residual_rate_differences),
        "support_td_corr": _mean(row.get("support_td_corr") for row in rows),
        "support_residual_corr": _mean(row.get("support_residual_corr") for row in rows),
        "normalized_ess": _mean(row.get("ess_normalized") for row in rows),
        "scan_truncated_rate": _mean(row.get("scan_truncated_rate") for row in rows),
    }
    for phase in PHASES:
        result[f"phase_{phase}_fraction"] = _mean(row.get(f"phase_{phase}_frac") for row in rows)
        result[f"phase_{phase}_support_rate"] = _weighted_phase_support(rows, phase)
    return result


def _subset_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean(row.get(key) for row in rows)


def _stats_row(metric: str, values: Sequence[float]) -> dict[str, Any]:
    finite = _finite(values)
    ci_low, ci_high = _t_interval(finite)
    return {
        "artifact_id": ARTIFACT_ID,
        "metric": metric,
        "n_seeds": len(finite),
        "mean_difference": _mean(finite),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "positive_seed_count": sum(value > 0.0 for value in finite),
        "negative_seed_count": sum(value < 0.0 for value in finite),
        "zero_seed_count": sum(value == 0.0 for value in finite),
        "one_sample_t_p": _ttest_p(finite),
        "wilcoxon_two_sided_p": _wilcoxon_p(finite),
        "inference_unit": "seed",
    }


def _metric_in_snapshot(row: Mapping[str, str], metric: str) -> float:
    if metric == "td_difference":
        return _row_difference(row, "td_support_pos_mean", "td_support_neg_mean")
    if metric == "residual_difference":
        return _row_difference(row, "residual_support_pos_mean", "residual_support_neg_mean")
    if metric == "residual_positive_rate_difference":
        return _row_difference(row, "residual_positive_rate_pos", "residual_positive_rate_neg")
    return _float(row.get(metric))


def _window_seed_mean(rows: Sequence[Mapping[str, str]], metric: str) -> float:
    return _mean(_metric_in_snapshot(row, metric) for row in rows)


def _scanner_seed_row(seed: int, rows: Sequence[Mapping[str, str]], summary: Mapping[str, Any], successful: bool) -> dict[str, Any]:
    fields = (
        "scan_truncated_rate",
        "numba_enabled",
        "numba_warmed",
        "numba_fallback_rate",
        "audit_policy_support_mismatch",
        "audit_local_count_abs_diff",
        "audit_support_count_abs_diff",
        "predicate_early_exit_rate",
        "predicate_scanned_count_mean",
        "support_cache_hit_rate",
        "local_scan_cache_hit_rate",
        "local_context_time_sec",
        "actor_critic_update_time_sec",
        "numba_scan_time_sec",
    )
    result: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "row_scope": "seed",
        "seed": seed,
        "successful": successful,
        "runtime_total_sec": summary.get("runtime_total_sec"),
        "final_steps_per_sec": summary.get("final_steps_per_sec"),
    }
    for field in fields:
        result[field] = _mean(row.get(field) for row in rows)
    return result


def build_results(artifact_root: Path) -> dict[str, Any]:
    data_root = artifact_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"compact data directory is missing: {data_root}")

    payloads: dict[int, tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]] = {}
    performance_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        payloads[seed] = _load_seed_payload(artifact_root, seed)
        eval_rows, _snapshots, _summary = payloads[seed]
        cohort = "diagnostic_rerun_seeds_1_5" if seed <= 5 else "diagnostic_extension_seeds_6_10"
        performance_rows.append(_performance(eval_rows, seed, cohort))

    successful_seeds = tuple(row["seed"] for row in performance_rows if row["successful"])
    if successful_seeds != (1, 2, 3, 4, 5, 6, 8, 9, 10):
        raise ValueError(f"unexpected successful-seed definition from Hit90: {successful_seeds}")

    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename in RESULT_FILENAMES:
        path = results_dir / filename
        if path.exists():
            path.unlink()

    performance_fields = (
        "artifact_id", "cohort", "seed", "evaluation_points", "final_success", "best_success",
        "auc_success", "hit90_step", "reached_hit90", "successful",
    )
    _write_csv(results_dir / "performance_by_seed.csv", performance_fields, performance_rows)

    aggregate_performance: list[dict[str, Any]] = []
    subsets = (
        ("diagnostic_all_10", performance_rows),
        ("diagnostic_successful_9", [row for row in performance_rows if row["successful"]]),
        ("diagnostic_rerun_seeds_1_5", performance_rows[:5]),
        ("diagnostic_extension_seeds_6_10", performance_rows[5:]),
    )
    for name, subset in subsets:
        successes = sum(bool(row["successful"]) for row in subset)
        wilson_low, wilson_high = _wilson_interval(successes, len(subset))
        auc_ci_low, auc_ci_high = _t_interval([float(row["auc_success"]) for row in subset])
        hit_steps = [int(row["hit90_step"]) for row in subset if row["hit90_step"] is not None]
        aggregate_performance.append({
            "artifact_id": ARTIFACT_ID,
            "cohort": name,
            "n_seeds": len(subset),
            "hit90_seed_count": successes,
            "success_rate": successes / len(subset),
            "success_rate_wilson95_low": wilson_low,
            "success_rate_wilson95_high": wilson_high,
            "final_success_mean": _subset_mean(subset, "final_success"),
            "auc_success_mean": _subset_mean(subset, "auc_success"),
            "auc_success_ci95_low": auc_ci_low,
            "auc_success_ci95_high": auc_ci_high,
            "hit90_mean_successful_only": _mean(hit_steps),
            "hit90_mean_scope": "reached_Hit90_only",
        })
    aggregate_fields = (
        "artifact_id", "cohort", "n_seeds", "hit90_seed_count", "success_rate",
        "success_rate_wilson95_low", "success_rate_wilson95_high", "final_success_mean",
        "auc_success_mean", "auc_success_ci95_low", "auc_success_ci95_high",
        "hit90_mean_successful_only", "hit90_mean_scope",
    )
    _write_csv(results_dir / "performance_aggregate.csv", aggregate_fields, aggregate_performance)

    reference_rows: list[dict[str, Any]] = []
    reference_perf: list[dict[str, Any]] = []
    for seed in range(1, 6):
        ref_eval = _read_csv(data_root / "reproduction_reference" / f"seed_{seed:02d}_eval_10k.csv")
        ref = _performance(ref_eval, seed, "original_non_diagnostic_reference")
        diag = performance_rows[seed - 1]
        reference_perf.append(ref)
        reference_rows.append({
            "artifact_id": ARTIFACT_ID,
            "row_scope": "seed",
            "seed": seed,
            "original_final_success": ref["final_success"],
            "diagnostic_final_success": diag["final_success"],
            "original_auc_success": ref["auc_success"],
            "diagnostic_auc_success": diag["auc_success"],
            "auc_difference_diagnostic_minus_original": float(diag["auc_success"]) - float(ref["auc_success"]),
            "original_hit90_step": ref["hit90_step"],
            "diagnostic_hit90_step": diag["hit90_step"],
            "unmatched_extension_reference_only": True,
        })
    reference_rows.append({
        "artifact_id": ARTIFACT_ID,
        "row_scope": "seed_1_5_mean",
        "seed": None,
        "original_final_success": _subset_mean(reference_perf, "final_success"),
        "diagnostic_final_success": _subset_mean(performance_rows[:5], "final_success"),
        "original_auc_success": _subset_mean(reference_perf, "auc_success"),
        "diagnostic_auc_success": _subset_mean(performance_rows[:5], "auc_success"),
        "auc_difference_diagnostic_minus_original": _subset_mean(performance_rows[:5], "auc_success") - _subset_mean(reference_perf, "auc_success"),
        "original_hit90_step": _mean(row["hit90_step"] for row in reference_perf if row["hit90_step"] is not None),
        "diagnostic_hit90_step": _mean(row["hit90_step"] for row in performance_rows[:5] if row["hit90_step"] is not None),
        "unmatched_extension_reference_only": True,
    })
    reference_fields = (
        "artifact_id", "row_scope", "seed", "original_final_success", "diagnostic_final_success",
        "original_auc_success", "diagnostic_auc_success", "auc_difference_diagnostic_minus_original",
        "original_hit90_step", "diagnostic_hit90_step", "unmatched_extension_reference_only",
    )
    _write_csv(results_dir / "performance_reproducibility_reference.csv", reference_fields, reference_rows)

    diagnostic_stats = {
        seed: _diagnostic_seed_stats(seed, payloads[seed][1], seed in successful_seeds) for seed in SEEDS
    }
    successful_stats = [diagnostic_stats[seed] for seed in successful_seeds]
    td_fields = (
        "artifact_id", "seed", "successful", "snapshot_count", "support_mean",
        "support_final_400k_mean", "td_support_pos_mean", "td_support_neg_mean", "td_difference",
        "residual_support_pos_mean", "residual_support_neg_mean", "residual_difference",
        "residual_positive_rate_pos", "residual_positive_rate_neg", "residual_positive_rate_difference",
        "support_td_corr", "support_residual_corr", "normalized_ess", "scan_truncated_rate",
    )
    _write_csv(
        results_dir / "successful_9seed_td_residual.csv",
        td_fields,
        [diagnostic_stats[seed] for seed in successful_seeds],
    )

    td_pos_mean = _subset_mean(successful_stats, "td_support_pos_mean")
    td_neg_mean = _subset_mean(successful_stats, "td_support_neg_mean")
    residual_pos_mean = _subset_mean(successful_stats, "residual_support_pos_mean")
    residual_neg_mean = _subset_mean(successful_stats, "residual_support_neg_mean")
    rate_pos_mean = _subset_mean(successful_stats, "residual_positive_rate_pos")
    rate_neg_mean = _subset_mean(successful_stats, "residual_positive_rate_neg")
    summary_rows = []
    summary_specs = (
        ("pre_update_absolute_td_error", "td_support_pos_mean", "td_support_neg_mean", "td_difference", td_pos_mean, td_neg_mean),
        ("same_minibatch_residual_reduction", "residual_support_pos_mean", "residual_support_neg_mean", "residual_difference", residual_pos_mean, residual_neg_mean),
        ("same_minibatch_residual_positive_rate", "residual_positive_rate_pos", "residual_positive_rate_neg", "residual_positive_rate_difference", rate_pos_mean, rate_neg_mean),
    )
    for metric, _pos_key, _neg_key, diff_key, pos_mean, neg_mean in summary_specs:
        stat = _stats_row(metric, [float(row[diff_key]) for row in successful_stats])
        stat.update({
            "support_positive_mean": pos_mean,
            "support_negative_mean": neg_mean,
            "positive_to_negative_ratio": _ratio(pos_mean, neg_mean),
            "same_minibatch": metric != "pre_update_absolute_td_error",
            "successful_seed_definition": "reached_eval_success_at_least_0.90",
        })
        summary_rows.append(stat)
    td_summary_fields = (
        "artifact_id", "metric", "n_seeds", "support_positive_mean", "support_negative_mean",
        "mean_difference", "positive_to_negative_ratio", "ci95_low", "ci95_high",
        "positive_seed_count", "negative_seed_count", "zero_seed_count", "one_sample_t_p",
        "wilcoxon_two_sided_p", "same_minibatch", "inference_unit", "successful_seed_definition",
    )
    _write_csv(results_dir / "successful_9seed_td_residual_summary.csv", td_summary_fields, summary_rows)

    phase_rows: list[dict[str, Any]] = []
    for seed in successful_seeds:
        stats = diagnostic_stats[seed]
        for phase in PHASES:
            phase_rows.append({
                "artifact_id": ARTIFACT_ID,
                "row_scope": "seed",
                "seed": seed,
                "phase": phase,
                "batch_fraction": stats[f"phase_{phase}_fraction"],
                "support_positive_rate_within_phase": stats[f"phase_{phase}_support_rate"],
                "phase_goal_basis": "desired_goal_train_including_HER_relabels",
            })
    phase_aggregate: dict[str, dict[str, float]] = {}
    for phase in PHASES:
        matching = [row for row in phase_rows if row["phase"] == phase]
        aggregate = {
            "batch_fraction": _mean(row["batch_fraction"] for row in matching),
            "support_positive_rate_within_phase": _mean(row["support_positive_rate_within_phase"] for row in matching),
        }
        phase_aggregate[phase] = aggregate
        phase_rows.append({
            "artifact_id": ARTIFACT_ID,
            "row_scope": "successful_9seed_mean",
            "seed": None,
            "phase": phase,
            **aggregate,
            "phase_goal_basis": "desired_goal_train_including_HER_relabels",
        })
    phase_fields = (
        "artifact_id", "row_scope", "seed", "phase", "batch_fraction",
        "support_positive_rate_within_phase", "phase_goal_basis",
    )
    _write_csv(results_dir / "phase_conditioned_allocation.csv", phase_fields, phase_rows)

    hit_metrics = (
        "support_rate",
        "td_difference",
        "residual_difference",
        "residual_positive_rate_difference",
        "support_td_corr",
        "support_residual_corr",
        "hand_handle_dist_mean",
    )
    hit_rows: list[dict[str, Any]] = []
    for seed in successful_seeds:
        hit90 = int(performance_rows[seed - 1]["hit90_step"])
        snapshots = payloads[seed][1]
        pre = [row for row in snapshots if int(row["step"]) < hit90]
        post = [row for row in snapshots if int(row["step"]) >= hit90]
        for metric in hit_metrics:
            pre_mean = _window_seed_mean(pre, metric)
            post_mean = _window_seed_mean(post, metric)
            hit_rows.append({
                "artifact_id": ARTIFACT_ID,
                "seed": seed,
                "hit90_step": hit90,
                "metric": metric,
                "pre_hit90_mean": pre_mean,
                "post_hit90_mean": post_mean,
                "post_minus_pre": post_mean - pre_mean,
                "inference_unit": "seed",
            })
    hit_fields = (
        "artifact_id", "seed", "hit90_step", "metric", "pre_hit90_mean", "post_hit90_mean",
        "post_minus_pre", "inference_unit",
    )
    _write_csv(results_dir / "hit90_pre_post_by_seed.csv", hit_fields, hit_rows)

    hit_summary_rows: list[dict[str, Any]] = []
    for metric in hit_metrics:
        matching = [row for row in hit_rows if row["metric"] == metric]
        stat = _stats_row(metric, [float(row["post_minus_pre"]) for row in matching])
        stat.update({
            "pre_hit90_mean": _mean(row["pre_hit90_mean"] for row in matching),
            "post_hit90_mean": _mean(row["post_hit90_mean"] for row in matching),
            "temporal_interpretation": "descriptive_pre_post_not_causal",
        })
        hit_summary_rows.append(stat)
    hit_summary_fields = (
        "artifact_id", "metric", "n_seeds", "pre_hit90_mean", "post_hit90_mean",
        "mean_difference", "ci95_low", "ci95_high", "positive_seed_count", "negative_seed_count",
        "zero_seed_count", "one_sample_t_p", "wilcoxon_two_sided_p", "inference_unit",
        "temporal_interpretation",
    )
    _write_csv(results_dir / "hit90_pre_post_summary.csv", hit_summary_fields, hit_summary_rows)

    window_bounds = ((0, 400_000), (400_000, 800_000), (800_000, 1_200_000), (1_200_000, 1_600_000), (1_600_000, 2_000_000))
    window_metrics = (
        "support_rate", "hand_handle_dist_mean", "phase_contact_frac", "phase_opening_frac",
        "td_error_abs_mean", "residual_difference", "critic_loss",
    )
    seed7_window_rows: list[dict[str, Any]] = []
    for lower, upper in window_bounds:
        by_seed: dict[int, list[dict[str, str]]] = {
            seed: [row for row in payloads[seed][1] if lower < int(row["step"]) <= upper] for seed in SEEDS
        }
        out: dict[str, Any] = {
            "artifact_id": ARTIFACT_ID,
            "window_start_exclusive": lower,
            "window_end_inclusive": upper,
            "window_label": f"{max(5_000, lower + 1)}-{upper}",
        }
        for metric in window_metrics:
            successful_seed_means = [_window_seed_mean(by_seed[seed], metric) for seed in successful_seeds]
            out[f"successful_9seed_{metric}_mean"] = _mean(successful_seed_means)
            out[f"seed7_{metric}_mean"] = _window_seed_mean(by_seed[7], metric)
        out["successful_9seed_support_ge_0_95_snapshot_rate"] = _mean(
            _mean(float(_float(row["support_rate"]) >= 0.95) for row in by_seed[seed]) for seed in successful_seeds
        )
        out["seed7_support_ge_0_95_snapshot_rate"] = _mean(
            float(_float(row["support_rate"]) >= 0.95) for row in by_seed[7]
        )
        out["successful_9seed_approach_support_rate"] = _mean(
            _weighted_phase_support(by_seed[seed], "approach") for seed in successful_seeds
        )
        out["seed7_approach_support_rate"] = _weighted_phase_support(by_seed[7], "approach")
        out["successful_9seed_open_support_rate"] = _mean(
            _weighted_phase_support(by_seed[seed], "open") for seed in successful_seeds
        )
        out["seed7_open_support_rate"] = _weighted_phase_support(by_seed[7], "open")
        seed7_window_rows.append(out)
    seed7_fields = (
        "artifact_id", "window_start_exclusive", "window_end_inclusive", "window_label",
        "successful_9seed_support_rate_mean", "seed7_support_rate_mean",
        "successful_9seed_hand_handle_dist_mean_mean", "seed7_hand_handle_dist_mean_mean",
        "successful_9seed_phase_contact_frac_mean", "seed7_phase_contact_frac_mean",
        "successful_9seed_phase_opening_frac_mean", "seed7_phase_opening_frac_mean",
        "successful_9seed_td_error_abs_mean_mean", "seed7_td_error_abs_mean_mean",
        "successful_9seed_residual_difference_mean", "seed7_residual_difference_mean",
        "successful_9seed_critic_loss_mean", "seed7_critic_loss_mean",
        "successful_9seed_support_ge_0_95_snapshot_rate", "seed7_support_ge_0_95_snapshot_rate",
        "successful_9seed_approach_support_rate", "seed7_approach_support_rate",
        "successful_9seed_open_support_rate", "seed7_open_support_rate",
    )
    _write_csv(results_dir / "seed7_400k_windows.csv", seed7_fields, seed7_window_rows)

    scanner_rows = [
        _scanner_seed_row(seed, payloads[seed][1], payloads[seed][2], seed in successful_seeds) for seed in SEEDS
    ]
    scanner_numeric_fields = tuple(
        key for key in scanner_rows[0] if key not in {"artifact_id", "row_scope", "seed", "successful"}
    )
    for scope, selected in (
        ("diagnostic_all_10_mean", scanner_rows),
        ("successful_9seed_mean", [row for row in scanner_rows if row["successful"]]),
    ):
        aggregate = {"artifact_id": ARTIFACT_ID, "row_scope": scope, "seed": None, "successful": None}
        for field in scanner_numeric_fields:
            aggregate[field] = _mean(row[field] for row in selected)
        scanner_rows.append(aggregate)
    scanner_fields = (
        "artifact_id", "row_scope", "seed", "successful", "runtime_total_sec", "final_steps_per_sec",
        "scan_truncated_rate", "numba_enabled", "numba_warmed", "numba_fallback_rate",
        "audit_policy_support_mismatch", "audit_local_count_abs_diff", "audit_support_count_abs_diff",
        "predicate_early_exit_rate", "predicate_scanned_count_mean", "support_cache_hit_rate",
        "local_scan_cache_hit_rate", "local_context_time_sec", "actor_critic_update_time_sec",
        "numba_scan_time_sec",
    )
    _write_csv(results_dir / "scanner_runtime.csv", scanner_fields, scanner_rows)

    correlation_metrics = (
        ("support_mean", "support_mean"),
        ("support_final_400k_mean", "support_final_400k_mean"),
        ("td_difference", "td_difference"),
        ("residual_difference", "residual_difference"),
        ("residual_positive_rate_difference", "residual_positive_rate_difference"),
        ("support_td_corr", "support_td_corr"),
        ("support_residual_corr", "support_residual_corr"),
    )
    correlation_rows: list[dict[str, Any]] = []
    for label, key in correlation_metrics:
        x = [float(diagnostic_stats[seed][key]) for seed in successful_seeds]
        for outcome in ("auc_success", "hit90_step"):
            y = [float(performance_rows[seed - 1][outcome]) for seed in successful_seeds]
            result = spearmanr(x, y)
            correlation_rows.append({
                "artifact_id": ARTIFACT_ID,
                "diagnostic_metric": label,
                "performance_outcome": outcome,
                "spearman_rho": float(result.statistic),
                "two_sided_p": float(result.pvalue),
                "n_seeds": len(successful_seeds),
                "seed_ids": "1;2;3;4;5;6;8;9;10",
                "multiple_comparison_corrected": False,
                "analysis_class": "exploratory",
                "inference_unit": "seed",
            })
    correlation_fields = (
        "artifact_id", "diagnostic_metric", "performance_outcome", "spearman_rho", "two_sided_p",
        "n_seeds", "seed_ids", "multiple_comparison_corrected", "analysis_class", "inference_unit",
    )
    _write_csv(results_dir / "exploratory_correlation.csv", correlation_fields, correlation_rows)

    all_aggregate = next(row for row in aggregate_performance if row["cohort"] == "diagnostic_all_10")
    td_summary = {row["metric"]: row for row in summary_rows}
    seed7_tail = seed7_window_rows[-1]
    return {
        "result_files": [str(results_dir / filename) for filename in RESULT_FILENAMES],
        "performance_all_10": all_aggregate,
        "successful_seeds": successful_seeds,
        "diagnostic_seed_stats": diagnostic_stats,
        "td_residual_summary": td_summary,
        "phase_successful_9": phase_aggregate,
        "hit90_summary": {row["metric"]: row for row in hit_summary_rows},
        "seed7_final_400k": seed7_tail,
        "scanner_seed_rows": scanner_rows[:10],
        "correlations": correlation_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Artifact root containing data/ and results/.",
    )
    args = parser.parse_args()
    result = build_results(args.artifact_root.resolve())
    print(f"rebuilt {len(result['result_files'])} result CSV files from compact data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
