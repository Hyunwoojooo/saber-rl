#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

import numpy as np


EVAL_STEPS = tuple(range(10_000, 1_000_001, 10_000))
SEEDS = (1, 2, 3, 4, 5)
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
FALLBACK_SUCCESS_FIELD = "eval/success_mean"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    root: Path


COMPARISONS = (
    ("Main SABER - Replay-query", "saber", "replay"),
    ("Main SABER - Shuffled-policy", "saber", "shuffled"),
    ("Main SABER - Context-count-only", "saber", "context_only"),
)

QUERY_MODES = {
    "saber": "current_policy",
    "replay": "replay_action",
    "shuffled": "shuffled_policy",
    "context_only": "context_count_only",
}

MECHANISM_FIELDS = (
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/query_bin_present_given_context_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/raw_weight_std",
    "query/raw_weight_ESS",
    "query/normalized_weight_mean",
    "query/normalized_weight_std",
    "query/normalized_weight_ESS",
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "query/support_positive_rate_phase_approach",
    "query/support_positive_rate_phase_grasp",
    "query/support_positive_rate_phase_lift",
    "query/support_positive_rate_phase_place",
)

CONFIG_IGNORE_FIELDS = {
    "run_name",
    "output_root",
    "resume",
    "resume_path",
    "support_query_mode",
    "thread_hint",
}


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def row_step(row: Mapping[str, str]) -> int | None:
    for key in ("step", "global_step", "env_step"):
        value = finite_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def row_success(row: Mapping[str, str]) -> float | None:
    primary = finite_float(row.get(PRIMARY_SUCCESS_FIELD))
    if primary is not None:
        return primary
    return finite_float(row.get(FALLBACK_SUCCESS_FIELD))


def load_eval_curve(
    metrics_path: Path,
    required_steps: Sequence[int] = EVAL_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    values_by_step: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = row_step(row)
            success = row_success(row)
            if step is None or success is None:
                continue
            previous = values_by_step.get(step)
            if previous is not None and not math.isclose(
                previous, success, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{metrics_path}: duplicate evaluation rows disagree at step "
                    f"{step}: {previous} versus {success}"
                )
            values_by_step[step] = success

    missing = [int(step) for step in required_steps if int(step) not in values_by_step]
    if missing:
        raise ValueError(
            f"{metrics_path}: missing exact evaluation checkpoints {missing}; "
            "interpolation and forward filling are not allowed"
        )
    steps = np.asarray(required_steps, dtype=np.float64)
    values = np.asarray([values_by_step[int(step)] for step in required_steps], dtype=np.float64)
    return steps, values


MECHANISM_FALLBACK_FIELDS = {
    "query/support_positive_rate": "lite/psi_q_rank_rate",
    "query/raw_weight_mean": "lite/mean_w_P",
    "query/raw_weight_std": "lite/std_w_P",
    "query/raw_weight_ESS": "lite/ESS_w",
    "query/support_positive_rate_phase_approach": "b3/support_by_phase_approach",
    "query/support_positive_rate_phase_grasp": "b3/support_by_phase_grasp",
    "query/support_positive_rate_phase_lift": "b3/support_by_phase_lift",
    "query/support_positive_rate_phase_place": "b3/support_by_phase_place",
}


def load_mechanism_snapshots(
    metrics_path: Path,
    required_steps: Sequence[int] = EVAL_STEPS,
) -> dict[str, float | int | None]:
    required = {int(step) for step in required_steps}
    values: dict[str, dict[int, float]] = {field: {} for field in MECHANISM_FIELDS}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = row_step(row)
            if step is None or step not in required or row_success(row) is None:
                continue
            for field in MECHANISM_FIELDS:
                value = finite_float(row.get(field))
                if value is None:
                    value = finite_float(row.get(MECHANISM_FALLBACK_FIELDS.get(field, "")))
                if value is None:
                    continue
                previous = values[field].get(step)
                if previous is not None and not math.isclose(
                    previous, value, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"{metrics_path}: duplicate mechanism rows disagree for {field} "
                        f"at step {step}: {previous} versus {value}"
                    )
                values[field][step] = value

    output: dict[str, float | int | None] = {}
    for field, by_step in values.items():
        if not by_step:
            output[field] = None
            output[f"{field}__checkpoint_count"] = 0
            continue
        missing = sorted(required - set(by_step))
        if missing:
            raise ValueError(
                f"{metrics_path}: mechanism field {field} is present but missing exact "
                f"evaluation checkpoints {missing}"
            )
        output[field] = float(np.mean([by_step[step] for step in required_steps]))
        output[f"{field}__checkpoint_count"] = len(by_step)
    return output


def normalized_auc(steps: np.ndarray, values: np.ndarray) -> float:
    if steps.size < 2 or values.size != steps.size:
        raise ValueError("AUC requires at least two aligned evaluation points")
    span = float(steps[-1] - steps[0])
    if span <= 0.0:
        raise ValueError("AUC evaluation steps must span a positive interval")
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, steps) / span)


def first_hit90(steps: np.ndarray, values: np.ndarray) -> int | None:
    reached = np.flatnonzero(values >= 0.90)
    return int(steps[int(reached[0])]) if reached.size else None


def metrics_path_for(run_dir: Path) -> Path:
    for name in ("metrics_selected_10k.csv", "metrics.csv"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{run_dir}: missing metrics_selected_10k.csv and metrics.csv"
    )


def seed_for_run(run_dir: Path) -> int:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{run_dir}: missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = config.get("seeds")
    if not isinstance(seeds, dict) or "env_seed" not in seeds:
        raise ValueError(f"{config_path}: missing seeds.env_seed")
    return int(seeds["env_seed"])


def discover_runs(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    candidates = [root] if (root / "summary.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir()
    )
    found: dict[int, Path] = {}
    for run_dir in candidates:
        if not (run_dir / "summary.json").is_file():
            continue
        metrics_path_for(run_dir)
        seed = seed_for_run(run_dir)
        if seed in found:
            raise ValueError(
                f"{root}: duplicate seed {seed} runs: {found[seed]} and {run_dir}"
            )
        found[seed] = run_dir
    missing = [seed for seed in SEEDS if seed not in found]
    extra = sorted(seed for seed in found if seed not in SEEDS)
    if missing or extra:
        raise ValueError(
            f"{root}: expected exactly seeds {list(SEEDS)}, missing={missing}, extra={extra}"
        )
    return {seed: found[seed] for seed in SEEDS}


def display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def analyze_method(
    spec: MethodSpec,
    repo_root: Path,
    required_steps: Sequence[int] = EVAL_STEPS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, run_dir in discover_runs(spec.root).items():
        metrics_path = metrics_path_for(run_dir)
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        native_auc = finite_float(summary.get("eval_AUC_success"))
        if native_auc is None:
            raise ValueError(f"{summary_path}: missing finite eval_AUC_success")
        steps, values = load_eval_curve(metrics_path, required_steps)
        auc = normalized_auc(steps, values)
        mechanism = load_mechanism_snapshots(metrics_path, required_steps)
        query_mode = config.get("support_query_mode")
        if query_mode is None and spec.key == "saber":
            query_mode = "current_policy"
        rows.append(
            {
                "method_key": spec.key,
                "method": spec.label,
                "seed": seed,
                "run_dir": display_path(run_dir, repo_root),
                "metrics_path": display_path(metrics_path, repo_root),
                "summary_path": display_path(summary_path, repo_root),
                "support_query_mode": query_mode,
                "native_auc": native_auc,
                "recomputed_auc": auc,
                "native_minus_recomputed_auc": native_auc - auc,
                "final_success": float(values[-1]),
                "hit90": first_hit90(steps, values),
                "checkpoint_count": int(steps.size),
                "_config": config,
                **mechanism,
            }
        )
    return rows


def _canonical_variant_config(config: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in config.items()
        if key not in CONFIG_IGNORE_FIELDS
    }


def validate_condition_configs(per_seed: Sequence[Mapping[str, object]]) -> None:
    by_method = {
        key: [row for row in per_seed if row["method_key"] == key]
        for key in ("her", "saber", "replay", "shuffled", "context_only")
    }
    for key, rows in by_method.items():
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{key}: config validation requires exact seeds {list(SEEDS)}")

    for key, expected_mode in QUERY_MODES.items():
        modes = {row["support_query_mode"] for row in by_method[key]}
        if modes != {expected_mode}:
            raise ValueError(
                f"{key}: expected support_query_mode={expected_mode!r}, "
                f"got {sorted(repr(mode) for mode in modes)}"
            )

    for seed_index, seed in enumerate(SEEDS):
        reference = _canonical_variant_config(by_method["saber"][seed_index]["_config"])
        for key in ("replay", "shuffled", "context_only"):
            candidate = _canonical_variant_config(by_method[key][seed_index]["_config"])
            if candidate != reference:
                differing = sorted(
                    field
                    for field in set(reference) | set(candidate)
                    if reference.get(field) != candidate.get(field)
                )
                raise ValueError(
                    f"{key} seed {seed}: non-query config differs from main SABER: {differing}"
                )

    for key in QUERY_MODES:
        for row in by_method[key]:
            config = row["_config"]
            required = {
                "total_steps": 1_000_000,
                "batch_size": 256,
                "her_ratio": 0.8,
                "updates_per_step": 1,
                "eval_freq": 10_000,
                "eval_episodes": 20,
                "local_search_max": 512,
                "M_B": 16,
                "M_Q_min": 4,
                "N_local_min": 4,
                "action_bins_per_dim": 5,
                "lambda_support": 1.0,
                "weight_scope": "critic_only",
                "weighting_mode": "batch_norm",
                "support_scan_mode": "deterministic_tail_v1",
            }
            mismatched = {
                field: (config.get(field), expected)
                for field, expected in required.items()
                if config.get(field) != expected
            }
            if mismatched:
                raise ValueError(
                    f"{key} seed {row['seed']}: fixed production config mismatch {mismatched}"
                )


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def aggregate_rows(
    per_seed: Sequence[Mapping[str, object]],
    specs: Sequence[MethodSpec],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for spec in specs:
        rows = [row for row in per_seed if row["method_key"] == spec.key]
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{spec.label}: seed order/coverage is not exactly {list(SEEDS)}")
        auc = [float(row["recomputed_auc"]) for row in rows]
        native = [float(row["native_auc"]) for row in rows]
        final = [float(row["final_success"]) for row in rows]
        reached = [int(row["hit90"]) for row in rows if row["hit90"] is not None]
        output.append(
            {
                "method_key": spec.key,
                "method": spec.label,
                "n": len(rows),
                "auc_mean": float(np.mean(auc)),
                "auc_std": sample_std(auc),
                "native_auc_mean": float(np.mean(native)),
                "native_auc_std": sample_std(native),
                "final_success_mean": float(np.mean(final)),
                "final_success_std": sample_std(final),
                "hit90_reached_seed_count": len(reached),
                "hit90_median_reached": float(median(reached)) if reached else None,
            }
        )
    return output


def mechanism_per_seed_rows(
    per_seed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in per_seed:
        for field in MECHANISM_FIELDS:
            output.append(
                {
                    "method_key": row["method_key"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "metric": field,
                    "mean_over_exact_eval_snapshots": row.get(field),
                    "checkpoint_count": row.get(f"{field}__checkpoint_count", 0),
                }
            )
    return output


def mechanism_aggregate_rows(
    mechanism_per_seed: Sequence[Mapping[str, object]],
    specs: Sequence[MethodSpec],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for spec in specs:
        for field in MECHANISM_FIELDS:
            rows = [
                row
                for row in mechanism_per_seed
                if row["method_key"] == spec.key and row["metric"] == field
            ]
            finite = [
                float(row["mean_over_exact_eval_snapshots"])
                for row in rows
                if row["mean_over_exact_eval_snapshots"] is not None
            ]
            output.append(
                {
                    "method_key": spec.key,
                    "method": spec.label,
                    "metric": field,
                    "available_seed_count": len(finite),
                    "mean": float(np.mean(finite)) if finite else None,
                    "sample_std": sample_std(finite) if len(finite) >= 2 else None,
                }
            )
    return output


def paired_rows(
    per_seed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_method: dict[str, dict[int, Mapping[str, object]]] = {}
    for row in per_seed:
        by_method.setdefault(str(row["method_key"]), {})[int(row["seed"])] = row
    output: list[dict[str, object]] = []
    for comparison, left_key, right_key in COMPARISONS:
        if set(by_method.get(left_key, {})) != set(SEEDS):
            raise ValueError(f"{left_key}: paired seed coverage is not exactly {list(SEEDS)}")
        if set(by_method.get(right_key, {})) != set(SEEDS):
            raise ValueError(f"{right_key}: paired seed coverage is not exactly {list(SEEDS)}")
        for seed in SEEDS:
            left = by_method[left_key][seed]
            right = by_method[right_key][seed]
            left_hit = left["hit90"]
            right_hit = right["hit90"]
            output.append(
                {
                    "comparison": comparison,
                    "seed": seed,
                    "left_method": left["method"],
                    "right_method": right["method"],
                    "auc_difference": float(left["recomputed_auc"])
                    - float(right["recomputed_auc"]),
                    "final_success_difference": float(left["final_success"])
                    - float(right["final_success"]),
                    "left_hit90": left_hit,
                    "right_hit90": right_hit,
                    "hit90_difference": (
                        int(left_hit) - int(right_hit)
                        if left_hit is not None and right_hit is not None
                        else None
                    ),
                }
            )
    return output


def paired_percentile_bootstrap(
    differences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires finite paired differences")
    if int(resamples) <= 0:
        raise ValueError("paired bootstrap resamples must be positive")
    rng = np.random.default_rng(int(rng_seed))
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean_auc_difference": float(values.mean()),
        "bootstrap_mean": float(means.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("exact sign-flip test requires finite paired differences")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 1 << int(values.size)
    for mask in range(total):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(values.size)],
            dtype=np.float64,
        )
        if abs(float(np.mean(values * signs))) >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def bootstrap_rows(
    paired: Sequence[Mapping[str, object]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for comparison, _left, _right in COMPARISONS:
        rows = [row for row in paired if row["comparison"] == comparison]
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{comparison}: paired rows are not ordered seeds 1-5")
        result = paired_percentile_bootstrap(
            [float(row["auc_difference"]) for row in rows],
            resamples=resamples,
            rng_seed=rng_seed,
        )
        output.append(
            {
                "comparison": comparison,
                "n": len(rows),
                "resamples": int(resamples),
                "rng_seed": int(rng_seed),
                "exact_sign_flip_pvalue_two_sided": exact_sign_flip_pvalue(
                    [float(row["auc_difference"]) for row in rows]
                ),
                **result,
            }
        )
    return output


def leave_one_seed_out_rows(
    paired: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for comparison, _left, _right in COMPARISONS:
        rows = [row for row in paired if row["comparison"] == comparison]
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{comparison}: paired rows are not ordered seeds 1-5")
        for omitted_seed in SEEDS:
            retained = [
                float(row["auc_difference"])
                for row in rows
                if int(row["seed"]) != omitted_seed
            ]
            output.append(
                {
                    "comparison": comparison,
                    "omitted_seed": omitted_seed,
                    "retained_seed_count": len(retained),
                    "mean_auc_difference": float(np.mean(retained)),
                }
            )
    return output


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def render_report(
    per_seed: Sequence[Mapping[str, object]],
    aggregate: Sequence[Mapping[str, object]],
    bootstrap: Sequence[Mapping[str, object]],
    mechanism_aggregate: Sequence[Mapping[str, object]],
    command: str,
) -> str:
    ranking = sorted(aggregate, key=lambda row: float(row["auc_mean"]), reverse=True)
    if not bootstrap:
        raise ValueError("report requires paired bootstrap rows")
    report_resamples = int(bootstrap[0]["resamples"])
    report_rng_seed = int(bootstrap[0]["rng_seed"])
    lines = [
        "# TD3 SABER Support-Query Ablation",
        "",
        "All methods are recomputed on exact evaluations at 10k, 20k, ..., 1M.",
        "No interpolation, smoothing, extrapolation, or forward filling is used.",
        f"Success uses `{PRIMARY_SUCCESS_FIELD}` with `{FALLBACK_SUCCESS_FIELD}` only as fallback.",
        "Only the support-predicate query action changes across Main SABER and the three variants.",
        "Replay-query retains the sampled transition in the support pool, so self-support is a stated limitation.",
        "Context-count-only has no action membership gate and may therefore change the positive count.",
        "",
        "## Discovered Runs",
        "",
        "| Method | Seed | Run |",
        "|---|---:|---|",
    ]
    for row in per_seed:
        lines.append(f"| {row['method']} | {row['seed']} | `{row['run_dir']}` |")
    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "Mean is arithmetic mean and standard deviation is sample std (ddof=1).",
            "",
            "| Method | AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        median_hit = "" if row["hit90_median_reached"] is None else f"{float(row['hit90_median_reached']):.0f}"
        lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} +/- {float(row['auc_std']):.6f} | "
            f"{float(row['final_success_mean']):.6f} +/- {float(row['final_success_std']):.6f} | "
            f"{row['hit90_reached_seed_count']}/{row['n']} | {median_hit} |"
        )
    lines.extend(
        [
            "",
            f"AUC ranking: {' > '.join(str(row['method']) for row in ranking)}.",
            "",
            "## Paired AUC Differences",
            "",
            f"Intervals are 95% paired percentile bootstrap CIs with {report_resamples:,} resamples",
            f"and RNG seed {report_rng_seed}. Paired seed differences are resampled directly.",
            "Exact sign-flip p-values are two-sided over all 32 sign assignments for five paired seeds.",
            "",
            "| Comparison | Mean delta | 95% CI | Exact sign-flip p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap:
        lines.append(
            f"| {row['comparison']} | {float(row['mean_auc_difference']):.6f} | "
            f"[{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}] | "
            f"{float(row['exact_sign_flip_pvalue_two_sided']):.6f} |"
        )
    mechanism_lookup = {
        (str(row["method_key"]), str(row["metric"])): row
        for row in mechanism_aggregate
    }
    lines.extend(
        [
            "",
            "## Mechanism Snapshots",
            "",
            "Values are means over exact 10k evaluation snapshots, then arithmetic means across available seeds.",
            "Blank entries were not logged by the legacy run and are not imputed.",
            "",
            "| Method | Context eligible | Support positive | Query bin present given context |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        key = str(row["method_key"])
        values = []
        for metric in (
            "query/context_count_eligible_rate",
            "query/support_positive_rate",
            "query/query_bin_present_given_context_rate",
        ):
            value = mechanism_lookup[(key, metric)]["mean"]
            values.append("" if value is None else f"{float(value):.6f}")
        lines.append(f"| {row['method']} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "The intervals, sign-flip values, and leave-one-seed-out summaries are reported as generated; this report does not add a separate significance claim.",
            "",
            "## Reproduction",
            "",
            f"`{command}`",
            "",
        ]
    )
    return "\n".join(lines)


def resolve_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the TD3 SABER support-query ablation.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--her-root",
        type=Path,
        default=Path("experiment_FPP/runs/td3_her_baseline_ls512"),
    )
    parser.add_argument(
        "--saber-root",
        type=Path,
        default=Path("experiment_FPP/runs/td3_saber_critic_only_v4_ls512"),
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path("experiment_FPP/runs/td3_saber_query_replay_ls512"),
    )
    parser.add_argument(
        "--shuffled-root",
        type=Path,
        default=Path("experiment_FPP/runs/td3_saber_query_shuffled_ls512"),
    )
    parser.add_argument(
        "--context-only-root",
        type=Path,
        default=Path("experiment_FPP/runs/td3_saber_context_count_only_ls512"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_FPP/reports/td3_saber_query_ablation_ls512"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    specs = [
        MethodSpec("her", "TD3+HER", resolve_path(repo_root, args.her_root)),
        MethodSpec("saber", "TD3+HER+SABER (Main)", resolve_path(repo_root, args.saber_root)),
        MethodSpec("replay", "Replay-query", resolve_path(repo_root, args.replay_root)),
        MethodSpec("shuffled", "Shuffled-policy-query", resolve_path(repo_root, args.shuffled_root)),
        MethodSpec(
            "context_only",
            "Context-count-only",
            resolve_path(repo_root, args.context_only_root),
        ),
    ]
    per_seed = [
        row
        for spec in specs
        for row in analyze_method(spec, repo_root=repo_root)
    ]
    validate_condition_configs(per_seed)
    aggregate = aggregate_rows(per_seed, specs)
    mechanism_per_seed = mechanism_per_seed_rows(per_seed)
    mechanism_aggregate = mechanism_aggregate_rows(mechanism_per_seed, specs)
    paired = paired_rows(per_seed)
    bootstrap = bootstrap_rows(
        paired,
        resamples=args.bootstrap_resamples,
        rng_seed=args.bootstrap_seed,
    )
    leave_one_out = leave_one_seed_out_rows(paired)

    output_dir = resolve_path(repo_root, args.output_dir)
    write_csv(
        output_dir / "per_seed_results.csv",
        per_seed,
        (
            "method_key", "method", "seed", "run_dir", "metrics_path", "summary_path",
            "support_query_mode",
            "native_auc", "recomputed_auc", "native_minus_recomputed_auc",
            "final_success", "hit90", "checkpoint_count",
        ),
    )
    write_csv(
        output_dir / "aggregate_results.csv",
        aggregate,
        (
            "method_key", "method", "n", "auc_mean", "auc_std", "native_auc_mean",
            "native_auc_std", "final_success_mean", "final_success_std",
            "hit90_reached_seed_count", "hit90_median_reached",
        ),
    )
    write_csv(
        output_dir / "paired_differences.csv",
        paired,
        (
            "comparison", "seed", "left_method", "right_method", "auc_difference",
            "final_success_difference", "left_hit90", "right_hit90", "hit90_difference",
        ),
    )
    write_csv(
        output_dir / "paired_bootstrap_ci.csv",
        bootstrap,
        (
            "comparison", "n", "resamples", "rng_seed", "mean_auc_difference",
            "bootstrap_mean", "ci_lower", "ci_upper",
            "exact_sign_flip_pvalue_two_sided",
        ),
    )
    write_csv(
        output_dir / "leave_one_seed_out.csv",
        leave_one_out,
        ("comparison", "omitted_seed", "retained_seed_count", "mean_auc_difference"),
    )
    write_csv(
        output_dir / "mechanism_per_seed.csv",
        mechanism_per_seed,
        (
            "method_key", "method", "seed", "metric",
            "mean_over_exact_eval_snapshots", "checkpoint_count",
        ),
    )
    write_csv(
        output_dir / "mechanism_summary.csv",
        mechanism_aggregate,
        ("method_key", "method", "metric", "available_seed_count", "mean", "sample_std"),
    )
    # Keep generated reports portable and free of machine-specific interpreter paths.
    command = " ".join(shlex.quote(part) for part in ["python", *sys.argv])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(
        render_report(per_seed, aggregate, bootstrap, mechanism_aggregate, command),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
