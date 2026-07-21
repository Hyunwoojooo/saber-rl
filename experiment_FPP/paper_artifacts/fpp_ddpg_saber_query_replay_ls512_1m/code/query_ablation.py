#!/usr/bin/env python3
"""Fail-closed exact-grid analysis for the FPP DDPG SABER query ablation.

This module intentionally uses only the Python standard library.  The same
code is copied into the compact paper artifact and used by its standalone
verifier, so all published tables are recomputed from the packaged per-seed
inputs rather than trusted as opaque outputs.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_ID = "fpp_ddpg_saber_query_replay_ls512_1m"
SCHEMA_VERSION = 1
MAIN_KEY = "main"
REPLAY_KEY = "replay"
METHOD = "b3_support_runtime_opt"
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 1_000_000
TOTAL_UPDATES = 995_001
EVAL_FREQ = 10_000
GRID = tuple(range(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ))
AUC_DENOMINATOR = GRID[-1] - GRID[0]
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
FALLBACK_SUCCESS_FIELD = "eval/success_mean"
FLOAT_TOLERANCE = 1e-12

QUERY_MODES = {MAIN_KEY: "current_policy", REPLAY_KEY: "replay_action"}
PAIR_CONFIG_EXEMPT_KEYS = frozenset(
    {
        "support_query_mode",
        "run_name",
        "output_root",
    }
)

EXPECTED_CONFIG: Mapping[str, Any] = {
    "method": METHOD,
    "task_name": "fetch_pick_and_place",
    "env_id": "FetchPickAndPlace-v4",
    "resolved_env_id": "FetchPickAndPlace-v4",
    "reward_type": "sparse",
    "total_steps": TOTAL_STEPS,
    "max_episode_steps": 50,
    "hidden_dim": 256,
    "buffer_capacity": 300_000,
    "batch_size": 256,
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.98,
    "tau": 0.005,
    "noise_std": 0.2,
    "her_ratio": 0.8,
    "updates_per_step": 1,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "eval_freq": EVAL_FREQ,
    "eval_episodes": 20,
    "checkpoint_every_steps": 50_000,
    "save_step_checkpoints": True,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "N_gap_min": 128,
    "local_search_max": 512,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "policy_support_mode": "bin",
    "support_scan_mode": "deterministic_tail_v1",
    "weighting_mode": "batch_norm",
    "weight_scope": "critic_only",
    "support_cache_enabled": True,
    "support_cache_max_entries": 200_000,
    "local_scan_cache_enabled": True,
    "local_scan_cache_max_entries": 50_000,
    "support_short_circuit": True,
}

EXPECTED_ENV_CORE: Mapping[tuple[str, ...], Any] = {
    ("train_env", "env_id"): "FetchPickAndPlace-v4",
    ("train_env", "obs_dim"): 25,
    ("train_env", "goal_dim"): 3,
    ("train_env", "state_dim"): 28,
    ("train_env", "action_dim"): 4,
    ("train_env", "action_low"): [-1.0, -1.0, -1.0, -1.0],
    ("train_env", "action_high"): [1.0, 1.0, 1.0, 1.0],
    ("train_env", "max_episode_steps"): 50,
    ("train_env", "distance_threshold"): 0.05,
    ("train_env", "spec_kwargs", "reward_type"): "sparse",
    ("eval_env", "env_id"): "FetchPickAndPlace-v4",
    ("eval_env", "obs_dim"): 25,
    ("eval_env", "goal_dim"): 3,
    ("eval_env", "state_dim"): 28,
    ("eval_env", "action_dim"): 4,
    ("eval_env", "action_low"): [-1.0, -1.0, -1.0, -1.0],
    ("eval_env", "action_high"): [1.0, 1.0, 1.0, 1.0],
    ("eval_env", "max_episode_steps"): 50,
    ("eval_env", "distance_threshold"): 0.05,
    ("eval_env", "spec_kwargs", "reward_type"): "sparse",
    ("task_spec", "task"): "FetchPickAndPlace",
    ("task_spec", "epsilon_g"): 0.05,
}

DIAGNOSTIC_FIELDS = (
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/query_bin_present_given_context_rate",
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "query/raw_weight_mean",
    "query/raw_weight_std",
    "query/raw_weight_ESS",
    "query/normalized_weight_mean",
    "query/normalized_weight_std",
    "query/normalized_weight_ESS",
    "query/support_positive_rate_phase_approach",
    "query/support_positive_rate_phase_grasp",
    "query/support_positive_rate_phase_lift",
    "query/support_positive_rate_phase_place",
)
DIAGNOSTIC_FALLBACKS: Mapping[str, str] = {
    "query/support_positive_rate": "lite/psi_q_rank_rate",
    "query/raw_weight_mean": "lite/mean_w_P",
    "query/raw_weight_std": "lite/std_w_P",
    "query/raw_weight_ESS": "lite/ESS_w",
    "query/support_positive_rate_phase_approach": "b3/support_by_phase_approach",
    "query/support_positive_rate_phase_grasp": "b3/support_by_phase_grasp",
    "query/support_positive_rate_phase_lift": "b3/support_by_phase_lift",
    "query/support_positive_rate_phase_place": "b3/support_by_phase_place",
}

PER_SEED_FIELDS = (
    "method_key",
    "method",
    "support_query_mode",
    "seed",
    "run_name",
    "total_steps",
    "total_updates",
    "eval_grid_points",
    "success_auc",
    "native_success_auc",
    "native_minus_recomputed_auc",
    "final_success",
    "best_success",
    "first_hit_0.90",
    "hit90_reached",
    "success_primary_checkpoint_count",
    "success_fallback_checkpoint_count",
)
AGGREGATE_FIELDS = (
    "method_key",
    "method",
    "support_query_mode",
    "n_seeds",
    "success_auc_mean",
    "success_auc_sample_std",
    "final_success_mean",
    "final_success_sample_std",
    "best_success_mean",
    "best_success_sample_std",
    "hit90_reached_count",
    "hit90_total",
    "hit90_step_mean_reached_only",
    "hit90_step_sample_std_reached_only",
    "hit90_step_median_reached_only",
)
PAIRED_FIELDS = (
    "comparison",
    "delta_definition",
    "seed",
    "main_auc",
    "replay_auc",
    "main_minus_replay_auc",
    "main_final_success",
    "replay_final_success",
    "main_minus_replay_final_success",
    "main_best_success",
    "replay_best_success",
    "main_minus_replay_best_success",
    "main_first_hit_0.90",
    "replay_first_hit_0.90",
    "main_minus_replay_hit90_step",
    "hit90_pair_complete",
)
INFERENCE_FIELDS = (
    "comparison",
    "metric",
    "delta_definition",
    "direction_favoring_main",
    "n_pairs",
    "paired_seeds",
    "mean_delta",
    "sample_std_delta",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "bootstrap_resamples",
    "bootstrap_seed",
    "positive_count",
    "negative_count",
    "zero_count",
    "exact_sign_flip_p_two_sided",
    "exact_sign_flip_assignments",
)
LOSO_FIELDS = (
    "comparison",
    "omitted_seed",
    "retained_seeds",
    "retained_seed_count",
    "mean_main_minus_replay_auc",
    "positive_count",
    "negative_count",
    "zero_count",
)
MEAN_CURVE_FIELDS = (
    "method_key",
    "method",
    "support_query_mode",
    "step",
    "n_seeds",
    "success_mean",
    "success_sample_std",
)
DIAGNOSTIC_PER_SEED_FIELDS = (
    "method_key",
    "method",
    "support_query_mode",
    "seed",
    "metric",
    "mean_over_finite_exact_eval_snapshots",
    "finite_checkpoint_count",
    "missing_or_nonfinite_checkpoint_count",
    "source_column",
)
DIAGNOSTIC_AGGREGATE_FIELDS = (
    "method_key",
    "method",
    "support_query_mode",
    "metric",
    "available_seed_count",
    "mean_of_seed_means",
    "sample_std_of_seed_means",
    "total_finite_checkpoint_count",
    "total_missing_or_nonfinite_checkpoint_count",
)


class AnalysisError(ValueError):
    """Raised whenever a run or rendered result violates the frozen protocol."""


class MissingRunError(AnalysisError):
    """Raised when exact paired seeds are unavailable."""


@dataclass(frozen=True)
class DiagnosticSummary:
    mean: float | None
    finite_count: int
    missing_count: int
    source_column: str | None


@dataclass(frozen=True)
class RunResult:
    method_key: str
    method: str
    query_mode: str
    seed: int
    run_dir: Path
    run_name: str
    metrics_path: Path
    metrics_rows: tuple[dict[str, str], ...]
    curve: tuple[float, ...]
    success_sources: tuple[str, ...]
    auc: float
    native_auc: float
    final: float
    best: float
    first_hit_90: int | None
    config: dict[str, Any]
    summary: dict[str, Any]
    env_snapshot: dict[str, Any]
    diagnostics: dict[str, DiagnosticSummary]


@dataclass(frozen=True)
class AnalysisBundle:
    seeds: tuple[int, ...]
    runs_by_method: dict[str, tuple[RunResult, ...]]
    per_seed_rows: tuple[dict[str, Any], ...]
    aggregate_rows: tuple[dict[str, Any], ...]
    paired_rows: tuple[dict[str, Any], ...]
    inference_rows: tuple[dict[str, Any], ...]
    leave_one_seed_out_rows: tuple[dict[str, Any], ...]
    mean_curve_rows: tuple[dict[str, Any], ...]
    diagnostic_per_seed_rows: tuple[dict[str, Any], ...]
    diagnostic_aggregate_rows: tuple[dict[str, Any], ...]


def find_repo_root(path: Path) -> Path:
    start = path.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "experiment_FPP").is_dir() and (
            (candidate / ".git").exists() or (candidate / "version_v0_lite").is_dir()
        ):
            return candidate
    raise AnalysisError(f"Could not locate repository root from {path}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AnalysisError(f"Missing or empty required JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object in {path}")
    return value


def finite_float(value: Any, *, context: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise AnalysisError(f"Missing numeric value for {context}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"Invalid numeric value for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"Non-finite numeric value for {context}: {value!r}")
    return result


def optional_finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any, *, context: str) -> int:
    result = finite_float(value, context=context)
    if not result.is_integer():
        raise AnalysisError(f"Expected integer for {context}, got {value!r}")
    return int(result)


def config_seed(config: Mapping[str, Any]) -> int | None:
    mapping = config.get("seeds")
    value = mapping.get("env_seed") if isinstance(mapping, Mapping) else config.get("env_seed")
    parsed = optional_finite_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def effective_query_mode(config: Mapping[str, Any], method_key: str) -> str:
    value = config.get("support_query_mode")
    if value is None and method_key == MAIN_KEY:
        return "current_policy"
    if not isinstance(value, str):
        raise AnalysisError(
            f"{method_key}: config.support_query_mode must be a string; got {value!r}"
        )
    return value.lower()


def metrics_path_for(run_dir: Path) -> Path:
    for name in ("metrics_selected_10k.csv", "metrics.csv"):
        path = run_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise AnalysisError(f"Missing metrics_selected_10k.csv or metrics.csv in {run_dir}")


def _candidate_matches(config: Mapping[str, Any], method_key: str, seed: int) -> bool:
    try:
        mode = effective_query_mode(config, method_key)
    except AnalysisError:
        return False
    return (
        config.get("method") == METHOD
        and config.get("weight_scope") == "critic_only"
        and config.get("total_steps") == TOTAL_STEPS
        and config.get("smoke") is False
        and config_seed(config) == seed
        and mode == QUERY_MODES[method_key]
    )


def discover_run(root: Path, method_key: str, seed: int) -> Path:
    """Find one exact resolved-config match, ignoring other steps and seeds."""
    if method_key not in QUERY_MODES:
        raise AnalysisError(f"Unknown method key: {method_key!r}")
    if not root.is_dir():
        raise MissingRunError(f"Run root does not exist: {root}")
    matches: list[Path] = []
    for config_path in sorted(root.rglob("config.json")):
        try:
            config = read_json(config_path)
        except AnalysisError:
            continue
        run_dir = config_path.parent
        if _candidate_matches(config, method_key, seed) and (
            (run_dir / "summary.json").is_file()
            and (run_dir / "env_snapshot.json").is_file()
        ):
            try:
                metrics_path_for(run_dir)
            except AnalysisError:
                continue
            matches.append(run_dir)
    if not matches:
        raise MissingRunError(
            f"Missing {method_key} seed {seed} exact 1M non-smoke run under {root}"
        )
    if len(matches) != 1:
        raise AnalysisError(
            f"Expected exactly one {method_key} seed {seed} exact run under {root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _validate_expected_config(config: Mapping[str, Any], run_dir: Path) -> None:
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if isinstance(expected, float):
            try:
                equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
            except (TypeError, ValueError):
                equal = False
        else:
            equal = actual == expected
        if not equal:
            raise AnalysisError(f"{run_dir}: config.{key} expected {expected!r}, got {actual!r}")
    terminal = config.get("terminal_convention")
    if terminal != {"success_terminates": False, "timeout_bootstrap": True}:
        raise AnalysisError(f"{run_dir}: terminal_convention mismatch: {terminal!r}")


def _validate_seed_mapping(config: Mapping[str, Any], seed: int, run_dir: Path) -> None:
    mapping = config.get("seeds")
    if not isinstance(mapping, Mapping):
        raise AnalysisError(f"{run_dir}: config.seeds must be an object")
    expected = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "eval_seed": 10_000 + seed,
        "weight_seed": 20_000 + seed,
    }
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise AnalysisError(
                f"{run_dir}: config.seeds.{key} expected {value!r}, got {mapping.get(key)!r}"
            )


def validate_run_config(
    config: Mapping[str, Any], method_key: str, seed: int, run_dir: Path
) -> None:
    _validate_expected_config(config, run_dir)
    _validate_seed_mapping(config, seed, run_dir)
    if config.get("smoke") is not False:
        raise AnalysisError(f"{run_dir}: smoke results are inadmissible")
    mode = effective_query_mode(config, method_key)
    if mode != QUERY_MODES[method_key]:
        raise AnalysisError(
            f"{run_dir}: support_query_mode expected {QUERY_MODES[method_key]!r}, got {mode!r}"
        )
    run_name = config.get("run_name")
    if not isinstance(run_name, str) or not run_name or "smoke" in run_name.lower():
        raise AnalysisError(f"{run_dir}: invalid production run_name {run_name!r}")


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def validate_env_snapshot(snapshot: Mapping[str, Any], run_dir: Path) -> None:
    for path, expected in EXPECTED_ENV_CORE.items():
        actual = _nested_value(snapshot, path)
        if isinstance(expected, float):
            try:
                equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
            except (TypeError, ValueError):
                equal = False
        else:
            equal = actual == expected
        if not equal:
            raise AnalysisError(
                f"{run_dir}: env_snapshot.{'.'.join(path)} expected {expected!r}, got {actual!r}"
            )


def _success_for_row(row: Mapping[str, str]) -> tuple[float | None, str | None]:
    primary = optional_finite_float(row.get(PRIMARY_SUCCESS_FIELD))
    if primary is not None:
        return primary, PRIMARY_SUCCESS_FIELD
    fallback = optional_finite_float(row.get(FALLBACK_SUCCESS_FIELD))
    if fallback is not None:
        return fallback, FALLBACK_SUCCESS_FIELD
    return None, None


def load_exact_grid(
    path: Path,
) -> tuple[tuple[dict[str, str], ...], tuple[float, ...], tuple[str, ...]]:
    by_step: dict[int, tuple[dict[str, str], float, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise AnalysisError(f"{path}: missing required step column")
        if not ({PRIMARY_SUCCESS_FIELD, FALLBACK_SUCCESS_FIELD} & set(reader.fieldnames)):
            raise AnalysisError(
                f"{path}: missing both {PRIMARY_SUCCESS_FIELD!r} and {FALLBACK_SUCCESS_FIELD!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            success, source = _success_for_row(row)
            if success is None or source is None:
                continue
            step = integer(row.get("step"), context=f"{path}:{row_number}:step")
            if step not in GRID:
                raise AnalysisError(
                    f"{path}: evaluation at unexpected step {step}; no interpolation, "
                    "smoothing, extrapolation, or forward fill is allowed"
                )
            if step in by_step:
                raise AnalysisError(f"{path}: duplicate observed evaluation at step {step}")
            if not 0.0 <= success <= 1.0:
                raise AnalysisError(f"{path}: success {success} outside [0,1] at step {step}")
            by_step[step] = (dict(row), success, source)
    missing = [step for step in GRID if step not in by_step]
    if missing:
        raise AnalysisError(
            f"{path}: missing exact observed evaluation steps {missing}; no interpolation, "
            "smoothing, extrapolation, or forward fill is allowed"
        )
    return (
        tuple(by_step[step][0] for step in GRID),
        tuple(by_step[step][1] for step in GRID),
        tuple(by_step[step][2] for step in GRID),
    )


def normalized_trapezoid_auc(curve: Sequence[float]) -> float:
    if len(curve) != len(GRID):
        raise AnalysisError(f"AUC requires exactly {len(GRID)} observed values")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in curve):
        raise AnalysisError("AUC requires finite success values in [0,1]")
    area = sum(
        (GRID[index + 1] - GRID[index]) * (curve[index] + curve[index + 1]) / 2.0
        for index in range(len(GRID) - 1)
    )
    return area / float(AUC_DENOMINATOR)


def first_hit(curve: Sequence[float], threshold: float = 0.90) -> int | None:
    return next((step for step, value in zip(GRID, curve) if value >= threshold), None)


def _diagnostics(rows: Sequence[Mapping[str, str]]) -> dict[str, DiagnosticSummary]:
    output: dict[str, DiagnosticSummary] = {}
    for field in DIAGNOSTIC_FIELDS:
        fallback = DIAGNOSTIC_FALLBACKS.get(field)
        values: list[float] = []
        used_columns: set[str] = set()
        for row in rows:
            value = optional_finite_float(row.get(field))
            column = field
            if value is None and fallback:
                value = optional_finite_float(row.get(fallback))
                column = fallback
            if value is not None:
                values.append(value)
                used_columns.add(column)
        source = "+".join(sorted(used_columns)) if used_columns else None
        output[field] = DiagnosticSummary(
            mean=statistics.fmean(values) if values else None,
            finite_count=len(values),
            missing_count=len(rows) - len(values),
            source_column=source,
        )
    return output


def _summary_hit(value: Any, *, context: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return integer(value, context=context)


def _assert_close(actual: float, expected: Any, *, context: str) -> None:
    target = finite_float(expected, context=context)
    if not math.isclose(actual, target, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
        raise AnalysisError(f"{context}: recomputed={actual:.15g}, recorded={target:.15g}")


def load_run(run_dir: Path, method_key: str, seed: int) -> RunResult:
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    env_snapshot = read_json(run_dir / "env_snapshot.json")
    validate_run_config(config, method_key, seed, run_dir)
    validate_env_snapshot(env_snapshot, run_dir)
    if summary.get("method") != METHOD:
        raise AnalysisError(f"{run_dir}: summary.method mismatch: {summary.get('method')!r}")
    if summary.get("total_steps") != TOTAL_STEPS or summary.get("total_updates") != TOTAL_UPDATES:
        raise AnalysisError(
            f"{run_dir}: incomplete summary total_steps={summary.get('total_steps')!r}, "
            f"total_updates={summary.get('total_updates')!r}"
        )
    metrics_path = metrics_path_for(run_dir)
    rows, curve, success_sources = load_exact_grid(metrics_path)
    auc = normalized_trapezoid_auc(curve)
    final = curve[-1]
    best = max(curve)
    hit90 = first_hit(curve)
    native_auc = finite_float(summary.get("eval_AUC_success"), context=f"{run_dir}.eval_AUC_success")
    _assert_close(auc, native_auc, context=f"{run_dir}.eval_AUC_success")
    _assert_close(final, summary.get("final_eval_success"), context=f"{run_dir}.final_eval_success")
    _assert_close(best, summary.get("best_eval_success"), context=f"{run_dir}.best_eval_success")
    recorded_hit = _summary_hit(summary.get("first_hit_0.90"), context=f"{run_dir}.first_hit_0.90")
    if recorded_hit != hit90:
        raise AnalysisError(
            f"{run_dir}.first_hit_0.90: recomputed={hit90!r}, recorded={recorded_hit!r}"
        )
    diagnostics = _diagnostics(rows)
    if method_key == REPLAY_KEY:
        replay_match = diagnostics["query/query_bin_matches_replay_action_rate"]
        if replay_match.finite_count != len(GRID) or replay_match.missing_count != 0:
            raise AnalysisError(
                f"{run_dir}: replay-action match diagnostic must be finite at all {len(GRID)} evaluations"
            )
        values = [
            finite_float(
                row.get("query/query_bin_matches_replay_action_rate"),
                context=f"{run_dir}:query/query_bin_matches_replay_action_rate",
            )
            for row in rows
        ]
        if any(not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE) for value in values):
            raise AnalysisError(f"{run_dir}: query-bin/replay-action match rate is not identically 1")
        metric_modes = {
            row.get("config/support_query_mode")
            for row in rows
            if row.get("config/support_query_mode") not in (None, "")
        }
        if metric_modes and metric_modes != {"replay_action"}:
            raise AnalysisError(f"{run_dir}: metric query modes mismatch: {sorted(metric_modes)}")
    method = "DDPG+HER+SABER Main" if method_key == MAIN_KEY else "DDPG+HER+SABER Replay-Action Query"
    return RunResult(
        method_key=method_key,
        method=method,
        query_mode=effective_query_mode(config, method_key),
        seed=seed,
        run_dir=run_dir.resolve(),
        run_name=str(config["run_name"]),
        metrics_path=metrics_path.resolve(),
        metrics_rows=rows,
        curve=curve,
        success_sources=success_sources,
        auc=auc,
        native_auc=native_auc,
        final=final,
        best=best,
        first_hit_90=hit90,
        config=dict(config),
        summary=dict(summary),
        env_snapshot=dict(env_snapshot),
        diagnostics=diagnostics,
    )


def normalized_config(config: Mapping[str, Any], method_key: str) -> dict[str, Any]:
    result = dict(config)
    result["support_query_mode"] = effective_query_mode(config, method_key)
    return result


def config_differences(
    main_config: Mapping[str, Any], replay_config: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    main = normalized_config(main_config, MAIN_KEY)
    replay = normalized_config(replay_config, REPLAY_KEY)
    return [
        (key, main.get(key), replay.get(key))
        for key in sorted(set(main) | set(replay))
        if main.get(key) != replay.get(key)
    ]


def validate_pair_config_parity(main: RunResult, replay: RunResult) -> None:
    if main.seed != replay.seed:
        raise AnalysisError(f"Cannot pair different seeds: {main.seed} and {replay.seed}")
    differences = config_differences(main.config, replay.config)
    forbidden = [item for item in differences if item[0] not in PAIR_CONFIG_EXEMPT_KEYS]
    if forbidden:
        rendered = "; ".join(
            f"{key}: main={left!r}, replay={right!r}" for key, left, right in forbidden
        )
        raise AnalysisError(f"Seed {main.seed} non-query config parity failed: {rendered}")
    query_diffs = [item for item in differences if item[0] == "support_query_mode"]
    if query_diffs != [("support_query_mode", "current_policy", "replay_action")]:
        raise AnalysisError(
            f"Seed {main.seed}: expected only current_policy -> replay_action query change; "
            f"got query differences {query_diffs!r}"
        )


def sample_std(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisError("Cannot compute sample SD for an empty sequence")
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("Cannot compute percentile of empty values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if not differences or not all(math.isfinite(value) for value in differences):
        raise AnalysisError("Paired bootstrap requires finite differences")
    if resamples <= 0:
        raise AnalysisError("Bootstrap resamples must be positive")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    )
    return percentile(means, 0.025), percentile(means, 0.975)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    if not differences or not all(math.isfinite(value) for value in differences):
        raise AnalysisError("Exact sign-flip test requires finite differences")
    observed = abs(statistics.fmean(differences))
    extreme = 0
    assignments = 1 << len(differences)
    for mask in range(assignments):
        mean = statistics.fmean(
            value * (1.0 if mask & (1 << index) else -1.0)
            for index, value in enumerate(differences)
        )
        if abs(mean) >= observed - 1e-15:
            extreme += 1
    return extreme / assignments


def _per_seed_row(run: RunResult) -> dict[str, Any]:
    return {
        "method_key": run.method_key,
        "method": run.method,
        "support_query_mode": run.query_mode,
        "seed": run.seed,
        "run_name": run.run_name,
        "total_steps": TOTAL_STEPS,
        "total_updates": TOTAL_UPDATES,
        "eval_grid_points": len(GRID),
        "success_auc": run.auc,
        "native_success_auc": run.native_auc,
        "native_minus_recomputed_auc": run.native_auc - run.auc,
        "final_success": run.final,
        "best_success": run.best,
        "first_hit_0.90": run.first_hit_90,
        "hit90_reached": run.first_hit_90 is not None,
        "success_primary_checkpoint_count": sum(
            source == PRIMARY_SUCCESS_FIELD for source in run.success_sources
        ),
        "success_fallback_checkpoint_count": sum(
            source == FALLBACK_SUCCESS_FIELD for source in run.success_sources
        ),
    }


def _aggregate_row(runs: Sequence[RunResult]) -> dict[str, Any]:
    hits = [run.first_hit_90 for run in runs if run.first_hit_90 is not None]
    hit_values = [float(value) for value in hits]
    return {
        "method_key": runs[0].method_key,
        "method": runs[0].method,
        "support_query_mode": runs[0].query_mode,
        "n_seeds": len(runs),
        "success_auc_mean": statistics.fmean(run.auc for run in runs),
        "success_auc_sample_std": sample_std([run.auc for run in runs]),
        "final_success_mean": statistics.fmean(run.final for run in runs),
        "final_success_sample_std": sample_std([run.final for run in runs]),
        "best_success_mean": statistics.fmean(run.best for run in runs),
        "best_success_sample_std": sample_std([run.best for run in runs]),
        "hit90_reached_count": len(hits),
        "hit90_total": len(runs),
        "hit90_step_mean_reached_only": statistics.fmean(hit_values) if hit_values else None,
        "hit90_step_sample_std_reached_only": sample_std(hit_values) if hit_values else None,
        "hit90_step_median_reached_only": statistics.median(hit_values) if hit_values else None,
    }


def _paired_row(main: RunResult, replay: RunResult) -> dict[str, Any]:
    both_hit = main.first_hit_90 is not None and replay.first_hit_90 is not None
    return {
        "comparison": "Main SABER - Replay-Action Query",
        "delta_definition": "main_minus_replay",
        "seed": main.seed,
        "main_auc": main.auc,
        "replay_auc": replay.auc,
        "main_minus_replay_auc": main.auc - replay.auc,
        "main_final_success": main.final,
        "replay_final_success": replay.final,
        "main_minus_replay_final_success": main.final - replay.final,
        "main_best_success": main.best,
        "replay_best_success": replay.best,
        "main_minus_replay_best_success": main.best - replay.best,
        "main_first_hit_0.90": main.first_hit_90,
        "replay_first_hit_0.90": replay.first_hit_90,
        "main_minus_replay_hit90_step": (
            main.first_hit_90 - replay.first_hit_90 if both_hit else None
        ),
        "hit90_pair_complete": both_hit,
    }


def analyze_query_ablation(
    main_root: Path,
    replay_root: Path,
    seeds: Sequence[int] = SEEDS,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> AnalysisBundle:
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if normalized_seeds != SEEDS:
        raise AnalysisError(f"Confirmatory analysis requires exact ordered seeds {list(SEEDS)}")
    runs: dict[str, tuple[RunResult, ...]] = {}
    for method_key, root in ((MAIN_KEY, main_root), (REPLAY_KEY, replay_root)):
        runs[method_key] = tuple(
            load_run(discover_run(root, method_key, seed), method_key, seed)
            for seed in normalized_seeds
        )
    for main, replay in zip(runs[MAIN_KEY], runs[REPLAY_KEY]):
        validate_pair_config_parity(main, replay)

    per_seed = tuple(
        _per_seed_row(run)
        for method_key in (MAIN_KEY, REPLAY_KEY)
        for run in runs[method_key]
    )
    aggregate = tuple(_aggregate_row(runs[key]) for key in (MAIN_KEY, REPLAY_KEY))
    paired = tuple(
        _paired_row(main, replay)
        for main, replay in zip(runs[MAIN_KEY], runs[REPLAY_KEY])
    )
    differences = [float(row["main_minus_replay_auc"]) for row in paired]
    ci_low, ci_high = paired_bootstrap_ci(
        differences, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    inference = (
        {
            "comparison": "Main SABER - Replay-Action Query",
            "metric": "normalized_success_auc",
            "delta_definition": "main_minus_replay",
            "direction_favoring_main": "positive",
            "n_pairs": len(differences),
            "paired_seeds": ";".join(map(str, normalized_seeds)),
            "mean_delta": statistics.fmean(differences),
            "sample_std_delta": sample_std(differences),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "positive_count": sum(value > 0.0 for value in differences),
            "negative_count": sum(value < 0.0 for value in differences),
            "zero_count": sum(value == 0.0 for value in differences),
            "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(differences),
            "exact_sign_flip_assignments": 1 << len(differences),
        },
    )
    loso: list[dict[str, Any]] = []
    for omitted_seed in normalized_seeds:
        retained = [
            float(row["main_minus_replay_auc"])
            for row in paired
            if int(row["seed"]) != omitted_seed
        ]
        retained_seeds = [seed for seed in normalized_seeds if seed != omitted_seed]
        loso.append(
            {
                "comparison": "Main SABER - Replay-Action Query",
                "omitted_seed": omitted_seed,
                "retained_seeds": ";".join(map(str, retained_seeds)),
                "retained_seed_count": len(retained),
                "mean_main_minus_replay_auc": statistics.fmean(retained),
                "positive_count": sum(value > 0.0 for value in retained),
                "negative_count": sum(value < 0.0 for value in retained),
                "zero_count": sum(value == 0.0 for value in retained),
            }
        )
    mean_curves: list[dict[str, Any]] = []
    for method_key in (MAIN_KEY, REPLAY_KEY):
        method_runs = runs[method_key]
        for index, step in enumerate(GRID):
            values = [run.curve[index] for run in method_runs]
            mean_curves.append(
                {
                    "method_key": method_key,
                    "method": method_runs[0].method,
                    "support_query_mode": method_runs[0].query_mode,
                    "step": step,
                    "n_seeds": len(values),
                    "success_mean": statistics.fmean(values),
                    "success_sample_std": sample_std(values),
                }
            )
    diagnostic_per_seed: list[dict[str, Any]] = []
    for method_key in (MAIN_KEY, REPLAY_KEY):
        for run in runs[method_key]:
            for metric in DIAGNOSTIC_FIELDS:
                item = run.diagnostics[metric]
                diagnostic_per_seed.append(
                    {
                        "method_key": method_key,
                        "method": run.method,
                        "support_query_mode": run.query_mode,
                        "seed": run.seed,
                        "metric": metric,
                        "mean_over_finite_exact_eval_snapshots": item.mean,
                        "finite_checkpoint_count": item.finite_count,
                        "missing_or_nonfinite_checkpoint_count": item.missing_count,
                        "source_column": item.source_column,
                    }
                )
    diagnostic_aggregate: list[dict[str, Any]] = []
    for method_key in (MAIN_KEY, REPLAY_KEY):
        method_runs = runs[method_key]
        for metric in DIAGNOSTIC_FIELDS:
            items = [run.diagnostics[metric] for run in method_runs]
            values = [item.mean for item in items if item.mean is not None]
            diagnostic_aggregate.append(
                {
                    "method_key": method_key,
                    "method": method_runs[0].method,
                    "support_query_mode": method_runs[0].query_mode,
                    "metric": metric,
                    "available_seed_count": len(values),
                    "mean_of_seed_means": statistics.fmean(values) if values else None,
                    "sample_std_of_seed_means": sample_std(values) if values else None,
                    "total_finite_checkpoint_count": sum(item.finite_count for item in items),
                    "total_missing_or_nonfinite_checkpoint_count": sum(
                        item.missing_count for item in items
                    ),
                }
            )
    return AnalysisBundle(
        seeds=normalized_seeds,
        runs_by_method=runs,
        per_seed_rows=per_seed,
        aggregate_rows=aggregate,
        paired_rows=paired,
        inference_rows=inference,
        leave_one_seed_out_rows=tuple(loso),
        mean_curve_rows=tuple(mean_curves),
        diagnostic_per_seed_rows=tuple(diagnostic_per_seed),
        diagnostic_aggregate_rows=tuple(diagnostic_aggregate),
    )


def csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text(rows, fields), encoding="utf-8", newline="\n")


def analysis_json(bundle: AnalysisBundle) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "method_seed_coverage": {
            MAIN_KEY: list(bundle.seeds),
            REPLAY_KEY: list(bundle.seeds),
        },
        "evaluation": {
            "steps": list(GRID),
            "checkpoint_count": len(GRID),
            "primary_success_field": PRIMARY_SUCCESS_FIELD,
            "fallback_success_field": FALLBACK_SUCCESS_FIELD,
            "auc_method": "trapezoidal",
            "auc_normalization_denominator": AUC_DENOMINATOR,
            "interpolation": False,
            "smoothing": False,
            "extrapolation": False,
            "forward_fill": False,
        },
        "bootstrap": {
            "method": "paired percentile",
            "confidence_level": 0.95,
            "resamples": int(bundle.inference_rows[0]["bootstrap_resamples"]),
            "seed": int(bundle.inference_rows[0]["bootstrap_seed"]),
        },
        "exact_sign_flip": {"two_sided": True, "assignments": 32},
        "per_seed": list(bundle.per_seed_rows),
        "aggregate": list(bundle.aggregate_rows),
        "paired_inference": list(bundle.inference_rows),
    }


OUTPUT_FILES = {
    "per_seed_results.csv": ("per_seed_rows", PER_SEED_FIELDS),
    "aggregate_results.csv": ("aggregate_rows", AGGREGATE_FIELDS),
    "paired_differences.csv": ("paired_rows", PAIRED_FIELDS),
    "paired_inference.csv": ("inference_rows", INFERENCE_FIELDS),
    "leave_one_seed_out.csv": ("leave_one_seed_out_rows", LOSO_FIELDS),
    "mean_learning_curve.csv": ("mean_curve_rows", MEAN_CURVE_FIELDS),
    "query_diagnostics_per_seed.csv": (
        "diagnostic_per_seed_rows",
        DIAGNOSTIC_PER_SEED_FIELDS,
    ),
    "query_diagnostics_aggregate.csv": (
        "diagnostic_aggregate_rows",
        DIAGNOSTIC_AGGREGATE_FIELDS,
    ),
}


def report_markdown(bundle: AnalysisBundle) -> str:
    aggregate = {row["method_key"]: row for row in bundle.aggregate_rows}
    inference = bundle.inference_rows[0]
    diagnostic = {
        (row["method_key"], row["metric"]): row
        for row in bundle.diagnostic_aggregate_rows
    }
    lines = [
        "# FPP DDPG SABER Replay-Action Query results",
        "",
        "The endpoint is normalized trapezoidal success AUC on exact observed ",
        "10k, 20k, ..., 1M evaluations (100 points; denominator 990,000). No ",
        "interpolation, smoothing, extrapolation, or forward filling is used.",
        "",
        "Replay-action querying retains the sampled transition in the local replay pool and can therefore self-support.",
        "",
        "| Method | AUC mean ± sample SD | Final mean ± sample SD | Best mean ± sample SD | Hit90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (MAIN_KEY, REPLAY_KEY):
        row = aggregate[key]
        lines.append(
            f"| {row['method']} | {row['success_auc_mean']:.6f} ± "
            f"{row['success_auc_sample_std']:.6f} | {row['final_success_mean']:.6f} ± "
            f"{row['final_success_sample_std']:.6f} | {row['best_success_mean']:.6f} ± "
            f"{row['best_success_sample_std']:.6f} | {row['hit90_reached_count']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Paired Main − Replay AUC",
            "",
            f"Mean delta: {inference['mean_delta']:+.6f}; 95% paired percentile bootstrap CI "
            f"[{inference['bootstrap_ci95_low']:.6f}, {inference['bootstrap_ci95_high']:.6f}] "
            f"(10,000 resamples, seed 27027); positive seeds: {inference['positive_count']}/5; "
            f"exact two-sided sign-flip p={inference['exact_sign_flip_p_two_sided']:.6f} "
            "over 32 assignments.",
            "",
            "## Query/support diagnostics",
            "",
            "Blank legacy values are unavailable and are not imputed.",
            "",
            "| Method | Context eligible | Support positive | Query present given context |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in (MAIN_KEY, REPLAY_KEY):
        values = []
        for metric in (
            "query/context_count_eligible_rate",
            "query/support_positive_rate",
            "query/query_bin_present_given_context_rate",
        ):
            value = diagnostic[(key, metric)]["mean_of_seed_means"]
            values.append("" if value is None else f"{value:.6f}")
        lines.append(f"| {aggregate[key]['method']} | {values[0]} | {values[1]} | {values[2]} |")
    return "\n".join(lines) + "\n"


def write_outputs(bundle: AnalysisBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (attribute, fields) in OUTPUT_FILES.items():
        write_csv(output_dir / name, getattr(bundle, attribute), fields)
    (output_dir / "analysis_results.json").write_text(
        json.dumps(analysis_json(bundle), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "report.md").write_text(
        report_markdown(bundle), encoding="utf-8", newline="\n"
    )


def expected_rendered_outputs(bundle: AnalysisBundle) -> dict[str, str]:
    output = {
        name: csv_text(getattr(bundle, attribute), fields)
        for name, (attribute, fields) in OUTPUT_FILES.items()
    }
    output["analysis_results.json"] = (
        json.dumps(analysis_json(bundle), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    output["report.md"] = report_markdown(bundle)
    return output


def verify_rendered_outputs(
    bundle: AnalysisBundle,
    output_dir: Path,
    *,
    extra_allowed_files: set[str] | frozenset[str] = frozenset(),
) -> None:
    expected = expected_rendered_outputs(bundle)
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    allowed = set(expected) | set(extra_allowed_files)
    if actual_files != allowed:
        raise AnalysisError(
            f"Result file set mismatch: expected={sorted(allowed)}, actual={sorted(actual_files)}"
        )
    for name, content in expected.items():
        actual = (output_dir / name).read_text(encoding="utf-8")
        if actual != content:
            raise AnalysisError(f"Rendered result differs from exact recomputation: {name}")
