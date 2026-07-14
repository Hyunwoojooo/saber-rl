#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
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
    expected_query_mode: str
    expected_assignment_mode: str
    count_matched: bool = False
    descriptive_only: bool = False


COMPARISONS = (
    (
        "Main SABER - Context-eligible count-matched",
        "main",
        "control",
    ),
)

QUERY_MODES = {
    "main": "current_policy",
    "control": "current_policy",
    "context_only": "context_count_only",
}

ASSIGNMENT_MODES = {
    "main": "support",
    "control": "context_eligible_count_matched",
    "context_only": "support",
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
    "support_assignment_mode",
    "assignment_diagnostic_window_updates",
    "thread_hint",
}

REQUIRED_PRODUCTION_CONFIG = {
    "task_name": "fetch_pick_and_place",
    "env_id": "FetchPickAndPlace-v4",
    "method": "td3_saber_runtime_opt",
    "reward_type": "sparse",
    "total_steps": 1_000_000,
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "batch_size": 256,
    "buffer_capacity": 300_000,
    "her_ratio": 0.8,
    "updates_per_step": 1,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "local_search_max": 512,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "weight_scope": "critic_only",
    "weighting_mode": "batch_norm",
    "policy_support_mode": "bin",
    "support_scan_mode": "deterministic_tail_v1",
    "td3_policy_delay": 2,
    "td3_target_policy_noise": 0.2,
    "td3_target_noise_clip": 0.5,
    "success_terminates": False,
    "timeout_bootstrap": True,
}

DIAGNOSTIC_WEIGHTED_FIELDS = (
    "context_eligible_count_mean",
    "main_positive_count_mean",
    "assigned_positive_count_mean",
    "context_eligible_rate_mean",
    "main_positive_rate_mean",
    "assigned_positive_rate_mean",
    "overlap_count_mean",
    "overlap_over_k_mean",
    "jaccard_mean",
    "reference_raw_weight_mean_mean",
    "assigned_raw_weight_mean_mean",
    "reference_raw_weight_std_mean",
    "assigned_raw_weight_std_mean",
    "reference_raw_weight_ESS_mean",
    "assigned_raw_weight_ESS_mean",
    "reference_normalized_weight_mean_mean",
    "assigned_normalized_weight_mean_mean",
    "reference_normalized_weight_std_mean",
    "assigned_normalized_weight_std_mean",
    "reference_normalized_weight_ESS_mean",
    "assigned_normalized_weight_ESS_mean",
)


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
        query_mode = str(config.get("support_query_mode", "current_policy"))
        assignment_mode = str(
            config.get("support_assignment_mode", "support")
        )
        rows.append(
            {
                "method_key": spec.key,
                "method": spec.label,
                "descriptive_only": spec.descriptive_only,
                "seed": seed,
                "run_dir": display_path(run_dir, repo_root),
                "metrics_path": display_path(metrics_path, repo_root),
                "summary_path": display_path(summary_path, repo_root),
                "support_query_mode": query_mode,
                "support_assignment_mode": assignment_mode,
                "native_auc": native_auc,
                "recomputed_auc": auc,
                "native_minus_recomputed_auc": native_auc - auc,
                "final_success": float(values[-1]),
                "best_success": float(np.max(values)),
                "hit90": first_hit90(steps, values),
                "checkpoint_count": int(steps.size),
                "_config": config,
                "_summary": summary,
                "_run_dir": run_dir,
                **mechanism,
            }
        )
    return rows


def _canonical_variant_config(config: Mapping[str, object]) -> dict[str, object]:
    canonical = {
        key: value
        for key, value in config.items()
        if key not in CONFIG_IGNORE_FIELDS
    }
    canonical["support_query_mode"] = str(
        config.get("support_query_mode", "current_policy")
    )
    seeds = dict(canonical.get("seeds", {}))
    seeds.pop("assignment_seed", None)
    canonical["seeds"] = seeds
    return canonical


def validate_condition_configs(per_seed: Sequence[Mapping[str, object]]) -> None:
    by_method = {
        key: [row for row in per_seed if row["method_key"] == key]
        for key in QUERY_MODES
    }
    for key, rows in by_method.items():
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(
                f"{key}: config validation requires exact seeds {list(SEEDS)}"
            )
        modes = {row["support_query_mode"] for row in rows}
        if modes != {QUERY_MODES[key]}:
            raise ValueError(
                f"{key}: expected support_query_mode={QUERY_MODES[key]!r}, "
                f"got {sorted(repr(mode) for mode in modes)}"
            )
        assignment_modes = {
            row["support_assignment_mode"] for row in rows
        }
        if assignment_modes != {ASSIGNMENT_MODES[key]}:
            raise ValueError(
                f"{key}: expected support_assignment_mode="
                f"{ASSIGNMENT_MODES[key]!r}, got "
                f"{sorted(repr(mode) for mode in assignment_modes)}"
            )

        for row in rows:
            config = row["_config"]
            mismatched = {
                field: (config.get(field), expected)
                for field, expected in REQUIRED_PRODUCTION_CONFIG.items()
                if config.get(field) != expected
            }
            if mismatched:
                raise ValueError(
                    f"{key} seed {row['seed']}: fixed production config "
                    f"mismatch {mismatched}"
                )
            seed = int(row["seed"])
            seeds = dict(config.get("seeds", {}))
            expected_seeds = {
                "env_seed": seed,
                "torch_seed": seed,
                "numpy_seed": seed,
                "action_noise_seed": seed,
                "weight_seed": 20_000 + seed,
                "eval_seed": 10_000 + seed,
            }
            seed_mismatches = {
                field: (seeds.get(field), expected)
                for field, expected in expected_seeds.items()
                if seeds.get(field) != expected
            }
            if seed_mismatches:
                raise ValueError(
                    f"{key} seed {seed}: seed mapping mismatch "
                    f"{seed_mismatches}"
                )
            if (
                key == "control"
                and seeds.get("assignment_seed") != 30_000 + seed
            ):
                raise ValueError(
                    f"control seed {seed}: assignment_seed must be "
                    f"{30_000 + seed}"
                )

    for seed_index, seed in enumerate(SEEDS):
        reference = _canonical_variant_config(
            by_method["main"][seed_index]["_config"]
        )
        candidate = _canonical_variant_config(
            by_method["control"][seed_index]["_config"]
        )
        if candidate != reference:
            differing = sorted(
                field
                for field in set(reference) | set(candidate)
                if reference.get(field) != candidate.get(field)
            )
            raise ValueError(
                f"control seed {seed}: non-assignment config differs from "
                f"Main SABER: {differing}"
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
        best = [float(row["best_success"]) for row in rows]
        native = [float(row["native_auc"]) for row in rows]
        final = [float(row["final_success"]) for row in rows]
        reached = [int(row["hit90"]) for row in rows if row["hit90"] is not None]
        output.append(
            {
                "descriptive_only": spec.descriptive_only,
                "method_key": spec.key,
                "method": spec.label,
                "n": len(rows),
                "auc_mean": float(np.mean(auc)),
                "auc_std": sample_std(auc),
                "native_auc_mean": float(np.mean(native)),
                "native_auc_std": sample_std(native),
                "final_success_mean": float(np.mean(final)),
                "final_success_std": sample_std(final),
                "best_success_mean": float(np.mean(best)),
                "best_success_std": sample_std(best),
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
                    "best_success_difference": float(left["best_success"])
                    - float(right["best_success"]),
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


def run_source_parity(
    repo_root: Path,
    recorded_evidence: Path | None = None,
) -> dict[str, object]:
    if recorded_evidence is not None:
        evidence = json.loads(recorded_evidence.read_text(encoding="utf-8"))
        if evidence.get("passed") is not True:
            raise ValueError(
                f"{recorded_evidence}: recorded source parity did not pass"
            )
        return {
            "mode": "recorded",
            "passed": True,
            "evidence_path": display_path(recorded_evidence, repo_root),
            "test": evidence.get("test"),
            "stdout": evidence.get("stdout", ""),
        }

    test_path = (
        repo_root
        / "experiment_FPP"
        / "td3_saber_context_eligible_count_matched_ls512"
        / "code"
        / "src"
        / "test_current_policy_legacy_parity.py"
    )
    if not test_path.is_file():
        raise FileNotFoundError(f"source parity test missing: {test_path}")
    env = os.environ.copy()
    env["PARITY_NUMBA"] = "0"
    result = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Main source parity test failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "mode": "executed",
        "passed": True,
        "test": display_path(test_path, repo_root),
        "stdout": result.stdout.strip(),
    }


def load_assignment_diagnostics(
    diagnostics_path: Path,
    expected_updates: int,
) -> dict[str, object]:
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{diagnostics_path}: no diagnostic windows")

    expected_start = 1
    total_updates = 0
    weighted_sums = {field: 0.0 for field in DIAGNOSTIC_WEIGHTED_FIELDS}
    mismatch_fields = (
        "exact_k_mismatch_count",
        "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch_count",
        "invariant_failure_count",
    )
    mismatch_totals = {field: 0 for field in mismatch_fields}
    normalized_sum_abs_diff_max = 0.0

    for index, row in enumerate(rows, start=1):
        window_index = int(float(row["window_index"]))
        start = int(float(row["update_start"]))
        end = int(float(row["update_end"]))
        updates = int(float(row["updates"]))
        complete = int(float(row["complete_window"]))
        if window_index != index:
            raise ValueError(
                f"{diagnostics_path}: window index {window_index} != {index}"
            )
        if start != expected_start or end != start + updates - 1:
            raise ValueError(
                f"{diagnostics_path}: non-contiguous window {index}"
            )
        if updates <= 0 or updates > 1_000:
            raise ValueError(
                f"{diagnostics_path}: invalid window size {updates}"
            )
        if index < len(rows) and updates != 1_000:
            raise ValueError(
                f"{diagnostics_path}: non-final window {index} is partial"
            )
        if complete != int(updates == 1_000):
            raise ValueError(
                f"{diagnostics_path}: complete flag mismatch in window {index}"
            )

        for field in mismatch_fields:
            value = int(float(row[field]))
            mismatch_totals[field] += value
            if value != 0:
                raise ValueError(
                    f"{diagnostics_path}: {field}={value} in window {index}"
                )

        main_count = float(row["main_positive_count_mean"])
        assigned_count = float(row["assigned_positive_count_mean"])
        eligible_count = float(row["context_eligible_count_mean"])
        if not math.isclose(
            main_count,
            assigned_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{diagnostics_path}: exact-k mismatch in window {index}"
            )
        if main_count > eligible_count + 1e-12:
            raise ValueError(
                f"{diagnostics_path}: k exceeds E in window {index}"
            )

        for prefix in ("raw_weight", "normalized_weight"):
            for statistic in ("mean", "std", "ESS"):
                reference = float(
                    row[f"reference_{prefix}_{statistic}_mean"]
                )
                assigned = float(
                    row[f"assigned_{prefix}_{statistic}_mean"]
                )
                if not math.isclose(
                    reference,
                    assigned,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        f"{diagnostics_path}: {prefix} {statistic} "
                        f"differs in window {index}"
                    )

        for field in DIAGNOSTIC_WEIGHTED_FIELDS:
            value = finite_float(row.get(field))
            if value is None:
                raise ValueError(
                    f"{diagnostics_path}: missing finite {field}"
                )
            weighted_sums[field] += value * updates
        normalized_sum_abs_diff_max = max(
            normalized_sum_abs_diff_max,
            float(row["normalized_weight_sum_abs_diff_max"]),
        )
        total_updates += updates
        expected_start = end + 1

    if total_updates != int(expected_updates):
        raise ValueError(
            f"{diagnostics_path}: diagnostics cover {total_updates} updates, "
            f"summary reports {expected_updates}"
        )
    return {
        "window_count": len(rows),
        "total_updates": total_updates,
        **{
            field: weighted_sums[field] / total_updates
            for field in DIAGNOSTIC_WEIGHTED_FIELDS
        },
        **mismatch_totals,
        "normalized_weight_sum_abs_diff_max": (
            normalized_sum_abs_diff_max
        ),
    }


def diagnostic_seed_rows(
    per_seed: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rows = [row for row in per_seed if row["method_key"] == "control"]
    for row in rows:
        run_dir = Path(row["_run_dir"])
        diagnostics_path = run_dir / "assignment_diagnostics_1000.csv"
        if not diagnostics_path.is_file():
            raise FileNotFoundError(
                f"{run_dir}: missing assignment_diagnostics_1000.csv"
            )
        summary = dict(row["_summary"])
        aggregate = load_assignment_diagnostics(
            diagnostics_path,
            int(summary["total_updates"]),
        )
        output.append(
            {
                "method_key": "control",
                "method": row["method"],
                "seed": row["seed"],
                "diagnostics_path": display_path(
                    diagnostics_path,
                    repo_root,
                ),
                **aggregate,
            }
        )
    return output


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    leave_one_out: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    command: str,
) -> str:
    if len(bootstrap) != 1:
        raise ValueError("report requires one primary paired comparison")
    inference = bootstrap[0]
    ci_lower = float(inference["ci_lower"])
    ci_upper = float(inference["ci_upper"])
    loso_values = [
        float(row["mean_auc_difference"]) for row in leave_one_out
    ]
    if ci_lower > 0.0:
        interpretation = (
            "The generated paired CI is above zero. Within FPP TD3, this "
            "supports additional value from membership-aligned assignment."
        )
    elif ci_lower <= 0.0 <= ci_upper:
        interpretation = (
            "The generated paired CI includes zero, so an independent "
            "membership-alignment contribution is not confirmed."
        )
    else:
        interpretation = (
            "The control is similar or higher on paired AUC; SABER should "
            "be interpreted here as context-gated, count-preserving critic "
            "reweighting rather than an isolated membership effect."
        )

    lines = [
        "# FPP TD3 Context-Eligible Count-Matched Control",
        "",
        "E is the pre-membership context-eligible mask, M is the Main "
        "current-policy membership mask, k=sum(M), and Z uniformly samples "
        "exactly k positions without replacement from E.",
        "Main weights M; the control weights Z. Natural M/Z overlap is allowed.",
        "The isolated variable is membership-aligned sample assignment. "
        "Query semantics remain current-policy and per-update counts and "
        "raw/normalized weight histograms are matched.",
        "",
        "Performance uses exact evaluations at 10k, 20k, ..., 1M. "
        "No interpolation, smoothing, extrapolation, or forward filling is used.",
        f"Success uses {PRIMARY_SUCCESS_FIELD} with {FALLBACK_SUCCESS_FIELD} "
        "only as fallback.",
        "AUC is trapezoidal area normalized over 10k to 1M. Mean is the "
        "arithmetic mean and std is sample std (ddof=1).",
        "Context-count-only is a count-unmatched descriptive reference.",
        "Random-support is excluded from the primary table because its "
        "generation-code provenance is not established.",
        "",
        "## Discovered Runs",
        "",
        "| Method | Seed | Run |",
        "|---|---:|---|",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['run_dir']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Native vs Exact-Grid AUC",
            "",
            "| Method | Seed | Native | Exact-grid | Native - exact | "
            "Final | Best | Hit90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed:
        hit = "" if row["hit90"] is None else str(int(row["hit90"]))
        lines.append(
            f"| {row['method']} | {row['seed']} | "
            f"{float(row['native_auc']):.6f} | "
            f"{float(row['recomputed_auc']):.6f} | "
            f"{float(row['native_minus_recomputed_auc']):.6f} | "
            f"{float(row['final_success']):.6f} | "
            f"{float(row['best_success']):.6f} | {hit} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Method | AUC mean +/- std | Final mean +/- std | "
            "Best mean +/- std | Hit90 reached | Median Hit90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        median_hit = (
            ""
            if row["hit90_median_reached"] is None
            else f"{float(row['hit90_median_reached']):.0f}"
        )
        lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} +/- "
            f"{float(row['auc_std']):.6f} | "
            f"{float(row['final_success_mean']):.6f} +/- "
            f"{float(row['final_success_std']):.6f} | "
            f"{float(row['best_success_mean']):.6f} +/- "
            f"{float(row['best_success_std']):.6f} | "
            f"{row['hit90_reached_seed_count']}/{row['n']} | "
            f"{median_hit} |"
        )

    differences = [
        float(row["auc_difference"])
        for row in paired_rows(per_seed)
    ]
    lines.extend(
        [
            "",
            "## Primary Paired AUC",
            "",
            f"Main - control mean: "
            f"{float(inference['mean_auc_difference']):.6f}.",
            f"95% paired percentile bootstrap CI: "
            f"[{ci_lower:.6f}, {ci_upper:.6f}] using "
            f"{int(inference['resamples']):,} resamples and RNG seed "
            f"{int(inference['rng_seed'])}.",
            f"Exact two-sided sign-flip value: "
            f"{float(inference['exact_sign_flip_pvalue_two_sided']):.6f}.",
            f"Positive deltas: "
            f"{int(np.sum(np.asarray(differences) > 0.0))}/{len(differences)}.",
            f"Leave-one-seed-out mean range: "
            f"[{min(loso_values):.6f}, {max(loso_values):.6f}].",
            "",
            interpretation,
            "This is an FPP TD3 result only; it does not establish a "
            "universal or task-agnostic mechanism.",
            "",
            "## Assignment Diagnostics",
            "",
            "| Seed | Updates | E rate | Main rate | Assigned rate | "
            "Overlap/k | Jaccard |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics:
        lines.append(
            f"| {row['seed']} | {row['total_updates']} | "
            f"{float(row['context_eligible_rate_mean']):.6f} | "
            f"{float(row['main_positive_rate_mean']):.6f} | "
            f"{float(row['assigned_positive_rate_mean']):.6f} | "
            f"{float(row['overlap_over_k_mean']):.6f} | "
            f"{float(row['jaccard_mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "All exact-k, histogram, normalized-sum, and window-contiguity "
            "checks passed before this report was written.",
            "",
            "## Reproduction",
            "",
            command,
            "",
        ]
    )
    return "\n".join(lines)

def resolve_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the FPP TD3 context-eligible count-matched control."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--main-root",
        type=Path,
        default=Path(
            "experiment_FPP/runs/td3_saber_critic_only_v4_ls512"
        ),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path(
            "experiment_FPP/runs/"
            "td3_saber_context_eligible_count_matched_ls512"
        ),
    )
    parser.add_argument(
        "--context-only-root",
        type=Path,
        default=Path(
            "experiment_FPP/runs/td3_saber_context_count_only_ls512"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiment_FPP/reports/"
            "td3_saber_context_eligible_count_matched_ls512"
        ),
    )
    parser.add_argument(
        "--source-parity-evidence",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    specs = [
        MethodSpec(
            "main",
            "TD3+HER+SABER (Main)",
            resolve_path(repo_root, args.main_root),
            "current_policy",
            "support",
        ),
        MethodSpec(
            "control",
            "Context-eligible count-matched",
            resolve_path(repo_root, args.control_root),
            "current_policy",
            "context_eligible_count_matched",
            count_matched=True,
        ),
        MethodSpec(
            "context_only",
            "Context-count-only (count-unmatched reference)",
            resolve_path(repo_root, args.context_only_root),
            "context_count_only",
            "support",
            descriptive_only=True,
        ),
    ]
    recorded_source_parity = (
        None
        if args.source_parity_evidence is None
        else resolve_path(repo_root, args.source_parity_evidence)
    )
    source_parity = run_source_parity(
        repo_root,
        recorded_source_parity,
    )

    per_seed = [
        row
        for spec in specs
        for row in analyze_method(spec, repo_root=repo_root)
    ]
    validate_condition_configs(per_seed)
    diagnostics = diagnostic_seed_rows(per_seed, repo_root)
    aggregate = aggregate_rows(per_seed, specs)
    paired = paired_rows(per_seed)
    bootstrap = bootstrap_rows(
        paired,
        resamples=args.bootstrap_resamples,
        rng_seed=args.bootstrap_seed,
    )
    leave_one_out = leave_one_seed_out_rows(paired)
    if len(bootstrap) != 1:
        raise ValueError("expected exactly one primary paired comparison")
    differences = [float(row["auc_difference"]) for row in paired]
    loso_means = [
        float(row["mean_auc_difference"]) for row in leave_one_out
    ]
    bootstrap[0].update(
        {
            "sample_std_auc_difference": sample_std(differences),
            "positive_delta_count": int(
                np.sum(np.asarray(differences) > 0.0)
            ),
            "zero_delta_count": int(
                np.sum(np.asarray(differences) == 0.0)
            ),
            "negative_delta_count": int(
                np.sum(np.asarray(differences) < 0.0)
            ),
            "leave_one_seed_out_mean_min": float(np.min(loso_means)),
            "leave_one_seed_out_mean_max": float(np.max(loso_means)),
        }
    )

    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_seed_results.csv",
        per_seed,
        (
            "method_key",
            "method",
            "descriptive_only",
            "seed",
            "run_dir",
            "metrics_path",
            "summary_path",
            "support_query_mode",
            "support_assignment_mode",
            "native_auc",
            "recomputed_auc",
            "native_minus_recomputed_auc",
            "final_success",
            "best_success",
            "hit90",
            "checkpoint_count",
        ),
    )
    write_csv(
        output_dir / "aggregate_results.csv",
        aggregate,
        (
            "method_key",
            "method",
            "descriptive_only",
            "n",
            "auc_mean",
            "auc_std",
            "native_auc_mean",
            "native_auc_std",
            "final_success_mean",
            "final_success_std",
            "best_success_mean",
            "best_success_std",
            "hit90_reached_seed_count",
            "hit90_median_reached",
        ),
    )
    write_csv(
        output_dir / "paired_differences.csv",
        paired,
        (
            "comparison",
            "seed",
            "left_method",
            "right_method",
            "auc_difference",
            "final_success_difference",
            "best_success_difference",
            "left_hit90",
            "right_hit90",
            "hit90_difference",
        ),
    )
    write_csv(
        output_dir / "paired_inference.csv",
        bootstrap,
        (
            "comparison",
            "n",
            "resamples",
            "rng_seed",
            "mean_auc_difference",
            "sample_std_auc_difference",
            "bootstrap_mean",
            "ci_lower",
            "ci_upper",
            "exact_sign_flip_pvalue_two_sided",
            "positive_delta_count",
            "zero_delta_count",
            "negative_delta_count",
            "leave_one_seed_out_mean_min",
            "leave_one_seed_out_mean_max",
        ),
    )
    write_csv(
        output_dir / "leave_one_seed_out.csv",
        leave_one_out,
        (
            "comparison",
            "omitted_seed",
            "retained_seed_count",
            "mean_auc_difference",
        ),
    )
    write_csv(
        output_dir / "assignment_diagnostic_summary.csv",
        diagnostics,
        (
            "method_key",
            "method",
            "seed",
            "diagnostics_path",
            "window_count",
            "total_updates",
            *DIAGNOSTIC_WEIGHTED_FIELDS,
            "exact_k_mismatch_count",
            "raw_histogram_mismatch_count",
            "normalized_histogram_mismatch_count",
            "invariant_failure_count",
            "normalized_weight_sum_abs_diff_max",
        ),
    )

    command = shlex.join(
        [
            "python",
            "experiment_FPP/"
            "td3_saber_context_eligible_count_matched_ls512/"
            "analyze_results.py",
            "--repo-root",
            ".",
            "--output-dir",
            display_path(output_dir, repo_root),
        ]
    )
    validation = {
        "schema_version": 1,
        "exact_evaluation_steps": list(EVAL_STEPS),
        "evaluation_checkpoint_count": len(EVAL_STEPS),
        "interpolation": False,
        "smoothing": False,
        "forward_fill": False,
        "success_field": PRIMARY_SUCCESS_FIELD,
        "success_fallback_field": FALLBACK_SUCCESS_FIELD,
        "bootstrap": {
            "resamples": int(args.bootstrap_resamples),
            "rng_seed": int(args.bootstrap_seed),
            "paired_differences_resampled": True,
        },
        "source_parity": source_parity,
        "config_parity": {
            "passed": True,
            "main_control_non_assignment_config_equal": True,
            "seed_mapping_equal": True,
            "evaluation_protocol_equal": True,
        },
        "assignment_diagnostics": {
            "passed": True,
            "all_invariant_counts_zero": True,
            "all_windows_contiguous": True,
            "all_updates_covered": True,
            "seed_window_counts": {
                str(row["seed"]): int(row["window_count"])
                for row in diagnostics
            },
        },
        "method_seed_coverage": {
            spec.key: list(SEEDS) for spec in specs
        },
        "primary_comparison": COMPARISONS[0][0],
        "context_only_is_count_unmatched_reference": True,
        "random_support_in_primary_table": False,
    }
    write_json(output_dir / "validation.json", validation)
    (output_dir / "report.md").write_text(
        render_report(
            per_seed,
            aggregate,
            bootstrap,
            leave_one_out,
            diagnostics,
            command,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
