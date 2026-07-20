#!/usr/bin/env python3
"""Exact-grid analysis for the FPP DDPG shuffled-policy-query ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

import numpy as np


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 1_000_001, 10_000))
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027
EXPECTED_TOTAL_STEPS = 1_000_000
EXPECTED_TOTAL_UPDATES = 995_001
EXPECTED_REPLAY_SIZE = 300_000

METHODS = (
    ("main", "DDPG+HER+SABER (Main)", "current_policy"),
    ("shuffled", "DDPG+HER+SABER (Shuffled-policy query)", "shuffled_policy"),
)

DIAGNOSTIC_FIELDS = (
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/query_bin_present_given_context_rate",
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/normalized_weight_mean",
)

DIAGNOSTIC_FALLBACKS = {
    "query/support_positive_rate": "lite/psi_q_rank_rate",
    "query/raw_weight_mean": "lite/mean_w_P",
}

SHUFFLED_REQUIRED_DIAGNOSTICS = DIAGNOSTIC_FIELDS
FINITE_TRAIN_FIELDS = (
    "loss/critic",
    "loss/actor",
    "loss/td_error_abs_mean",
    "value/q_mean",
    "value/target_mean",
    "train/replay_size",
)

ALLOWED_CONFIG_DIFFS = ("output_root", "run_name", "support_query_mode")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    root: Path
    expected_query_mode: str


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def row_step(row: Mapping[str, str]) -> int | None:
    for field in ("step", "global_step", "env_step"):
        value = finite_float(row.get(field))
        if value is not None:
            return int(value)
    return None


def row_success(row: Mapping[str, str]) -> float | None:
    """Return only the protocol's primary success field; never use a fallback."""

    return finite_float(row.get(PRIMARY_SUCCESS_FIELD))


def query_mode(config: Mapping[str, object], method_key: str) -> str | None:
    value = config.get("support_query_mode")
    if value is None and method_key == "main":
        return "current_policy"
    return str(value) if value is not None else None


def seed_for_config(config: Mapping[str, object], path: Path) -> int:
    try:
        return int(config["seeds"]["env_seed"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing integer seeds.env_seed") from exc


def metrics_path_for(run_dir: Path) -> Path:
    for name in ("metrics_selected_10k.csv", "metrics.csv"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{run_dir}: missing metrics_selected_10k.csv and metrics.csv")


def evaluation_rows(
    metrics_path: Path,
    required_steps: Sequence[int] = EVAL_STEPS,
) -> dict[int, dict[str, str]]:
    required = {int(step) for step in required_steps}
    selected: dict[int, dict[str, str]] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or PRIMARY_SUCCESS_FIELD not in reader.fieldnames:
            raise ValueError(f"{metrics_path}: missing primary field {PRIMARY_SUCCESS_FIELD}")
        for row in reader:
            step = row_step(row)
            success = row_success(row)
            if step not in required or success is None:
                continue
            previous = selected.get(int(step))
            if previous is not None:
                previous_success = row_success(previous)
                if previous_success is None or not math.isclose(
                    previous_success, success, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"{metrics_path}: conflicting primary evaluations at step {step}"
                    )
                continue
            selected[int(step)] = dict(row)
    missing = sorted(required - set(selected))
    extra = sorted(set(selected) - required)
    if missing or extra:
        raise ValueError(
            f"{metrics_path}: exact evaluation grid mismatch; missing={missing}, extra={extra}; "
            "smoothing/interpolation/forward filling are forbidden"
        )
    return {int(step): selected[int(step)] for step in required_steps}


def load_eval_curve(
    metrics_path: Path,
    required_steps: Sequence[int] = EVAL_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    rows = evaluation_rows(metrics_path, required_steps)
    steps = np.asarray(required_steps, dtype=np.float64)
    values = np.asarray([row_success(rows[int(step)]) for step in required_steps], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{metrics_path}: non-finite primary success value")
    return steps, values


def normalized_auc(steps: np.ndarray, values: np.ndarray) -> float:
    if steps.size < 2 or values.size != steps.size:
        raise ValueError("AUC requires at least two aligned evaluation points")
    span = float(steps[-1] - steps[0])
    if span <= 0.0:
        raise ValueError("AUC grid must span a positive interval")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, steps) / span)


def first_hit90(steps: np.ndarray, values: np.ndarray) -> int | None:
    reached = np.flatnonzero(values >= 0.90)
    return int(steps[int(reached[0])]) if reached.size else None


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample standard deviation needs at least two values")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def discover_runs(root: Path, method_key: str) -> dict[int, Path]:
    """Select exactly completed 1M seeds 1-5, ignoring other seeds and partial runs."""

    if not root.is_dir():
        raise FileNotFoundError(f"missing run root: {root}")
    candidates = [root] if (root / "config.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir()
    )
    found: dict[int, Path] = {}
    for run_dir in candidates:
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if not config_path.is_file() or not summary_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        try:
            seed = seed_for_config(config, config_path)
        except ValueError:
            continue
        if seed not in SEEDS:
            continue
        if int(config.get("total_steps", -1)) != EXPECTED_TOTAL_STEPS:
            continue
        if int(summary.get("total_steps", -1)) != EXPECTED_TOTAL_STEPS:
            continue
        metrics_path_for(run_dir)
        if seed in found:
            raise ValueError(f"{root}: duplicate completed 1M seed {seed}: {found[seed]} and {run_dir}")
        found[seed] = run_dir
    missing = sorted(set(SEEDS) - set(found))
    if missing:
        raise ValueError(f"{method_key}: missing completed 1M seeds {missing} under {root}")
    return {seed: found[seed] for seed in SEEDS}


def load_diagnostics(
    rows: Mapping[int, Mapping[str, str]],
) -> tuple[dict[str, float | int | None], dict[str, list[float]]]:
    snapshots: dict[str, list[float]] = {field: [] for field in DIAGNOSTIC_FIELDS}
    for row in rows.values():
        for field in DIAGNOSTIC_FIELDS:
            value = finite_float(row.get(field))
            if value is None:
                fallback = DIAGNOSTIC_FALLBACKS.get(field)
                value = finite_float(row.get(fallback)) if fallback else None
            if value is not None:
                snapshots[field].append(value)
    summary: dict[str, float | int | None] = {}
    for field, values in snapshots.items():
        summary[field] = float(np.mean(values)) if values else None
        summary[f"{field}__checkpoint_count"] = len(values)
    return summary, snapshots


def validate_completion(
    method_key: str,
    expected_mode: str,
    run_dir: Path,
    config: Mapping[str, object],
    summary: Mapping[str, object],
    eval_rows: Mapping[int, Mapping[str, str]],
    values: np.ndarray,
    diagnostics: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    errors: list[str] = []
    if int(config.get("total_steps", -1)) != EXPECTED_TOTAL_STEPS:
        errors.append("config.total_steps != 1000000")
    if int(summary.get("total_steps", -1)) != EXPECTED_TOTAL_STEPS:
        errors.append("summary.total_steps != 1000000")
    if int(summary.get("total_updates", -1)) != EXPECTED_TOTAL_UPDATES:
        errors.append("summary.total_updates != 995001")
    if int(summary.get("final_replay_size", -1)) != EXPECTED_REPLAY_SIZE:
        errors.append("summary.final_replay_size != 300000")
    if config.get("method") != "b3_support_runtime_opt":
        errors.append("method is not b3_support_runtime_opt")
    if config.get("weight_scope") != "critic_only":
        errors.append("weight_scope is not critic_only")
    actual_mode = query_mode(config, method_key)
    if actual_mode != expected_mode:
        errors.append(f"support_query_mode={actual_mode!r}, expected {expected_mode!r}")
    if tuple(eval_rows) != EVAL_STEPS or len(eval_rows) != 100:
        errors.append("evaluation grid is not the exact 100-point 10k..1M grid")
    if not np.all(np.isfinite(values)):
        errors.append("primary evaluation success contains NaN/Inf")
    if np.any(values < 0.0) or np.any(values > 1.0):
        errors.append("primary evaluation success is outside [0,1]")

    for step, row in eval_rows.items():
        for field in FINITE_TRAIN_FIELDS:
            if finite_float(row.get(field)) is None:
                errors.append(f"step {step}: missing/non-finite {field}")
    if method_key == "shuffled":
        for field in SHUFFLED_REQUIRED_DIAGNOSTICS:
            field_values = diagnostics.get(field, ())
            if len(field_values) != len(EVAL_STEPS):
                errors.append(f"{field}: expected 100 finite snapshots, got {len(field_values)}")
        normalized = diagnostics.get("query/normalized_weight_mean", ())
        if any(not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-5) for value in normalized):
            errors.append("query/normalized_weight_mean deviates from 1")

    native_auc = finite_float(summary.get("eval_AUC_success"))
    auc = normalized_auc(np.asarray(EVAL_STEPS, dtype=np.float64), values)
    if native_auc is None or not math.isclose(native_auc, auc, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"summary eval_AUC_success mismatch: native={native_auc}, recomputed={auc}")
    final_native = finite_float(summary.get("final_eval_env_success"))
    if final_native is None or not math.isclose(final_native, float(values[-1]), rel_tol=0.0, abs_tol=1e-12):
        errors.append("summary final_eval_env_success mismatch")

    return {
        "method_key": method_key,
        "seed": seed_for_config(config, run_dir / "config.json"),
        "run_dir": run_dir.name,
        "passed": not errors,
        "errors": errors,
        "evaluation_checkpoint_count": len(eval_rows),
        "total_steps": summary.get("total_steps"),
        "total_updates": summary.get("total_updates"),
        "final_replay_size": summary.get("final_replay_size"),
        "support_query_mode": actual_mode,
        "weight_scope": config.get("weight_scope"),
    }


def analyze_method(spec: MethodSpec) -> tuple[list[dict[str, object]], dict[int, np.ndarray], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    curves: dict[int, np.ndarray] = {}
    validations: list[dict[str, object]] = []
    for seed, run_dir in discover_runs(spec.root, spec.key).items():
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        metrics_path = metrics_path_for(run_dir)
        rows = evaluation_rows(metrics_path)
        steps = np.asarray(EVAL_STEPS, dtype=np.float64)
        values = np.asarray([row_success(rows[step]) for step in EVAL_STEPS], dtype=np.float64)
        diag_summary, diag_snapshots = load_diagnostics(rows)
        validation = validate_completion(
            spec.key, spec.expected_query_mode, run_dir, config, summary, rows, values, diag_snapshots
        )
        validations.append(validation)
        if not validation["passed"]:
            raise ValueError(f"{run_dir}: completion validation failed: {validation['errors']}")
        auc = normalized_auc(steps, values)
        native_auc = float(summary["eval_AUC_success"])
        results.append(
            {
                "method_key": spec.key,
                "method": spec.label,
                "seed": seed,
                "support_query_mode": query_mode(config, spec.key),
                "auc": auc,
                "native_auc": native_auc,
                "native_minus_recomputed_auc": native_auc - auc,
                "final_success": float(values[-1]),
                "best_success": float(np.max(values)),
                "hit90": first_hit90(steps, values),
                "evaluation_checkpoint_count": len(values),
                "run_dir": run_dir.name,
                "metrics_file": metrics_path.name,
                "_config": config,
                **diag_summary,
            }
        )
        curves[seed] = values
    return results, curves, validations


def normalized_config(config: Mapping[str, object], method_key: str) -> dict[str, object]:
    normalized = dict(config)
    normalized["support_query_mode"] = query_mode(config, method_key)
    return normalized


def config_diff(main: Mapping[str, object], shuffled: Mapping[str, object]) -> dict[str, dict[str, object]]:
    fields = sorted(set(main) | set(shuffled))
    return {
        field: {"main": main.get(field), "shuffled": shuffled.get(field)}
        for field in fields
        if main.get(field) != shuffled.get(field)
    }


def validate_config_parity(per_seed: Sequence[Mapping[str, object]]) -> dict[str, object]:
    indexed = {(str(row["method_key"]), int(row["seed"])): row for row in per_seed}
    rows: list[dict[str, object]] = []
    all_passed = True
    for seed in SEEDS:
        main = normalized_config(indexed[("main", seed)]["_config"], "main")  # type: ignore[arg-type]
        shuffled = normalized_config(indexed[("shuffled", seed)]["_config"], "shuffled")  # type: ignore[arg-type]
        diff = config_diff(main, shuffled)
        unexpected = sorted(set(diff) - set(ALLOWED_CONFIG_DIFFS))
        expected_changed = sorted(diff)
        passed = not unexpected and expected_changed == sorted(ALLOWED_CONFIG_DIFFS)
        all_passed = all_passed and passed
        rows.append(
            {
                "seed": seed,
                "passed": passed,
                "allowed_differences": list(ALLOWED_CONFIG_DIFFS),
                "actual_differences": diff,
                "unexpected_fields": unexpected,
            }
        )
    report = {"passed": all_passed, "allowed_differences": list(ALLOWED_CONFIG_DIFFS), "seeds": rows}
    if not all_passed:
        raise ValueError(f"Main/shuffled config parity failed: {report}")
    return report


def aggregate_rows(per_seed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, label, _mode in METHODS:
        rows = sorted((row for row in per_seed if row["method_key"] == key), key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"{key}: aggregate requires exact seeds 1-5")
        auc = [float(row["auc"]) for row in rows]
        final = [float(row["final_success"]) for row in rows]
        best = [float(row["best_success"]) for row in rows]
        reached = [int(row["hit90"]) for row in rows if row["hit90"] is not None]
        output.append(
            {
                "method_key": key,
                "method": label,
                "n": 5,
                "auc_mean": float(np.mean(auc)),
                "auc_sample_std": sample_std(auc),
                "final_success_mean": float(np.mean(final)),
                "final_success_sample_std": sample_std(final),
                "best_success_mean": float(np.mean(best)),
                "best_success_sample_std": sample_std(best),
                "hit90_reached_seed_count": len(reached),
                "hit90_median_reached": float(median(reached)) if reached else None,
            }
        )
    return output


def paired_percentile_bootstrap(
    differences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires exactly five finite differences")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, 5, size=(int(resamples), 5))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5))
    return {
        "bootstrap_mean": float(np.mean(means)),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
    }


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("sign-flip test requires exactly five finite differences")
    observed = abs(float(np.mean(values)))
    extreme = 0
    for mask in range(1 << 5):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(5)])
        if abs(float(np.mean(values * signs))) >= observed - 1e-15:
            extreme += 1
    return float(extreme / 32)


def paired_outputs(per_seed: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    indexed = {(str(row["method_key"]), int(row["seed"])): row for row in per_seed}
    rows: list[dict[str, object]] = []
    differences: list[float] = []
    for seed in SEEDS:
        main = indexed[("main", seed)]
        shuffled = indexed[("shuffled", seed)]
        difference = float(main["auc"]) - float(shuffled["auc"])
        differences.append(difference)
        rows.append(
            {
                "comparison": "Main SABER - Shuffled-policy query",
                "seed": seed,
                "main_auc": main["auc"],
                "shuffled_auc": shuffled["auc"],
                "auc_difference": difference,
                "final_success_difference": float(main["final_success"]) - float(shuffled["final_success"]),
                "main_hit90": main["hit90"],
                "shuffled_hit90": shuffled["hit90"],
            }
        )
    values = np.asarray(differences, dtype=np.float64)
    loo = [float(np.mean(np.delete(values, index))) for index in range(5)]
    inference: dict[str, object] = {
        "comparison": "Main SABER - Shuffled-policy query",
        "n": 5,
        "mean_auc_difference": float(np.mean(values)),
        "sample_sd_auc_difference": sample_std(differences),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_rng_seed": BOOTSTRAP_SEED,
        **paired_percentile_bootstrap(differences),
        "sign_flip_assignment_count": 32,
        "exact_two_sided_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
        "positive_seed_count": int(np.sum(values > 1e-15)),
        "zero_seed_count": int(np.sum(np.abs(values) <= 1e-15)),
        "negative_seed_count": int(np.sum(values < -1e-15)),
        "leave_one_seed_out_means": loo,
        "leave_one_seed_out_mean_min": min(loo),
        "leave_one_seed_out_mean_max": max(loo),
    }
    return rows, inference


def mean_curve_rows(curves: Mapping[str, Mapping[int, np.ndarray]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    main = np.stack([curves["main"][seed] for seed in SEEDS])
    shuffled = np.stack([curves["shuffled"][seed] for seed in SEEDS])
    for index, step in enumerate(EVAL_STEPS):
        output.append(
            {
                "step": step,
                "main_success_mean": float(np.mean(main[:, index])),
                "main_success_sample_std": sample_std(main[:, index].tolist()),
                "shuffled_success_mean": float(np.mean(shuffled[:, index])),
                "shuffled_success_sample_std": sample_std(shuffled[:, index].tolist()),
            }
        )
    return output


def diagnostic_rows(per_seed: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row in per_seed:
        for field in DIAGNOSTIC_FIELDS:
            seed_rows.append(
                {
                    "method_key": row["method_key"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "metric": field,
                    "mean_over_exact_eval_snapshots": row.get(field),
                    "finite_checkpoint_count": row.get(f"{field}__checkpoint_count", 0),
                }
            )
    for key, label, _mode in METHODS:
        for field in DIAGNOSTIC_FIELDS:
            values = [
                float(row["mean_over_exact_eval_snapshots"])
                for row in seed_rows
                if row["method_key"] == key and row["metric"] == field and row["mean_over_exact_eval_snapshots"] is not None
            ]
            summary_rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "metric": field,
                    "available_seed_count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "sample_std": sample_std(values) if len(values) >= 2 else None,
                }
            )
    return seed_rows, summary_rows


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def json_ready(row: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def run_analysis(main_root: Path, shuffled_root: Path, output_dir: Path) -> dict[str, object]:
    specs = (
        MethodSpec("main", METHODS[0][1], main_root, "current_policy"),
        MethodSpec("shuffled", METHODS[1][1], shuffled_root, "shuffled_policy"),
    )
    per_seed: list[dict[str, object]] = []
    curves: dict[str, dict[int, np.ndarray]] = {}
    validations: list[dict[str, object]] = []
    for spec in specs:
        rows, method_curves, method_validations = analyze_method(spec)
        per_seed.extend(rows)
        curves[spec.key] = method_curves
        validations.extend(method_validations)
    parity = validate_config_parity(per_seed)
    aggregate = aggregate_rows(per_seed)
    paired, inference = paired_outputs(per_seed)
    leave_one_seed_out = [
        {
            "omitted_seed": seed,
            "retained_seed_count": 4,
            "mean_auc_difference": inference["leave_one_seed_out_means"][index],
        }
        for index, seed in enumerate(SEEDS)
    ]
    mean_curves = mean_curve_rows(curves)
    diagnostics_per_seed, diagnostics_summary = diagnostic_rows(per_seed)
    validation = {
        "passed": all(bool(row["passed"]) for row in validations) and bool(parity["passed"]),
        "protocol": {
            "primary_success_field": PRIMARY_SUCCESS_FIELD,
            "exact_evaluation_steps": list(EVAL_STEPS),
            "evaluation_checkpoint_count_per_seed": 100,
            "auc_denominator": 990_000,
            "smoothing": False,
            "interpolation": False,
            "forward_fill": False,
            "extrapolation": False,
        },
        "run_count": len(validations),
        "runs": validations,
        "config_parity": parity,
    }
    payload = {
        "protocol": validation["protocol"],
        "methods": [json_ready(row) for row in aggregate],
        "per_seed": [json_ready(row) for row in per_seed],
        "paired_differences": paired,
        "paired_inference": inference,
        "leave_one_seed_out": leave_one_seed_out,
        "query_diagnostics": diagnostics_summary,
        "validation_passed": validation["passed"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_seed_results.csv",
        [json_ready(row) for row in per_seed],
        ("method_key", "method", "seed", "support_query_mode", "auc", "native_auc", "native_minus_recomputed_auc", "final_success", "best_success", "hit90", "evaluation_checkpoint_count", "run_dir", "metrics_file"),
    )
    write_csv(
        output_dir / "aggregate_results.csv",
        aggregate,
        ("method_key", "method", "n", "auc_mean", "auc_sample_std", "final_success_mean", "final_success_sample_std", "best_success_mean", "best_success_sample_std", "hit90_reached_seed_count", "hit90_median_reached"),
    )
    write_csv(
        output_dir / "paired_differences.csv",
        paired,
        ("comparison", "seed", "main_auc", "shuffled_auc", "auc_difference", "final_success_difference", "main_hit90", "shuffled_hit90"),
    )
    write_csv(output_dir / "paired_inference.csv", [inference], tuple(key for key in inference if key != "leave_one_seed_out_means"))
    write_csv(
        output_dir / "leave_one_seed_out.csv",
        leave_one_seed_out,
        ("omitted_seed", "retained_seed_count", "mean_auc_difference"),
    )
    write_csv(
        output_dir / "exact_grid_mean_curves.csv",
        mean_curves,
        ("step", "main_success_mean", "main_success_sample_std", "shuffled_success_mean", "shuffled_success_sample_std"),
    )
    write_csv(
        output_dir / "query_diagnostics_per_seed.csv",
        diagnostics_per_seed,
        ("method_key", "method", "seed", "metric", "mean_over_exact_eval_snapshots", "finite_checkpoint_count"),
    )
    write_csv(
        output_dir / "query_diagnostics_summary.csv",
        diagnostics_summary,
        ("method_key", "method", "metric", "available_seed_count", "mean", "sample_std"),
    )
    (output_dir / "analysis_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "config_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--shuffled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_analysis(args.main_root.resolve(), args.shuffled_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"validation_passed": payload["validation_passed"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
