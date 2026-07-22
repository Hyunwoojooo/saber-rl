#!/usr/bin/env python3
"""Exact-grid analysis primitives for the FPP SABER weight-scope ablation.

The primary endpoint is success AUC on the observed evaluation grid
``10_000, 20_000, ..., 1_000_000``, normalized by 990_000.  Missing grid
points are errors: this module never interpolates, smooths, or forward-fills.

Only the Python standard library is required.  The functions in this module
are shared by the command-line analyzer, compact artifact builder, verifier,
and synthetic-fixture tests.
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


ARTIFACT_ID = "fpp_ddpg_saber_actor_only_v4_ls512_1m"
METHOD = "b3_support_runtime_opt"
ACTOR_KEY = "actor_only"
CRITIC_KEY = "critic_only"
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 1_000_000
TOTAL_UPDATES = 995_001
EVAL_FREQ = 10_000
GRID = tuple(range(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_012_027
SUMMARY_TOLERANCE = 1e-12

EXPECTED_CONFIG: Mapping[str, Any] = {
    "method": METHOD,
    "task_name": "fetch_pick_and_place",
    "env_id": "FetchPickAndPlace-v4",
    "resolved_env_id": "FetchPickAndPlace-v4",
    "reward_type": "sparse",
    "obs_mode": "state",
    "robot_uids": "panda",
    "control_mode": None,
    "num_envs": 1,
    "backend": None,
    "sim_backend": None,
    "total_steps": TOTAL_STEPS,
    "max_episode_steps": 50,
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
    "l2_action_penalty": 0.0,
    "eval_freq": EVAL_FREQ,
    "eval_episodes": 20,
    "checkpoint_every_steps": 50_000,
    "save_step_checkpoints": True,
    "log_every_steps": 1_000,
    "validate_steps": 200,
    "validate_random_policy_steps": 200,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "N_gap_min": 128,
    "local_search_max": 512,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "weighting_mode": "batch_norm",
    "policy_support_mode": "bin",
    "policy_action_radius": 0.35,
    "support_scan_mode": "deterministic_tail_v1",
    "support_cache_enabled": True,
    "support_cache_max_entries": 200_000,
    "local_scan_cache_enabled": True,
    "local_scan_cache_max_entries": 50_000,
    "support_short_circuit": True,
    "verify_runtime_optimization": False,
    "d_reach": 0.08,
    "d_goal_near": 0.05,
    "z_lift_min": 0.48,
    "grip_close_thresh": 0.04,
    "pick_hand_obj_bins": [0.03, 0.05, 0.08, 0.12, 0.2, 0.4],
    "pick_obj_goal_bins": [0.05, 0.1, 0.2, 0.4, 0.7],
    "pick_object_z_bins": [0.43, 0.48, 0.55, 0.65, 0.75],
    "pick_gripper_gap_bins": [0.02, 0.05, 0.08],
    "horizon_bins": [10, 25, 50, 100],
    "gap_window": 50_000,
    "lambda_P": 1.0,
    "U_P_max": 1.0,
    "lambda_TD": 1.0,
    "U_TD_max": 1.0,
    "per_alpha": 0.6,
    "eps_PER": 1e-6,
    "U_PER_max": 2.0,
    "unrestricted_candidate_std": 0.5,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "terminal_convention": {
        "success_terminates": False,
        "timeout_bootstrap": True,
    },
    "resume": False,
    "resume_path": None,
    "render": False,
    "device": "cuda",
}

EXPECTED_ENV_CORE: Mapping[tuple[str, ...], Any] = {
    ("train_env", "env_id"): "FetchPickAndPlace-v4",
    ("train_env", "entry_point"):
        "gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
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
    ("eval_env", "entry_point"):
        "gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
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
    ("task_spec", "d_reach"): 0.08,
    ("task_spec", "d_goal_near"): 0.05,
    ("task_spec", "z_lift_min"): 0.48,
    ("task_spec", "grip_close_thresh"): 0.04,
}

# A paired run is intended to differ only in where the already-computed SABER
# weight is applied.  These are the only run-config fields allowed to differ.
PAIR_CONFIG_EXEMPT_KEYS = frozenset({"weight_scope", "run_name", "output_root"})

PER_SEED_FIELDS = (
    "arm",
    "weight_scope",
    "seed",
    "run_name",
    "total_steps",
    "eval_grid_points",
    "success_auc",
    "final_success",
    "best_success",
    "first_hit_0.90",
    "hit90_reached",
)
AGGREGATE_FIELDS = (
    "arm",
    "weight_scope",
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
    "seed",
    "actor_auc",
    "critic_auc",
    "actor_minus_critic_auc",
    "actor_final_success",
    "critic_final_success",
    "actor_minus_critic_final_success",
    "actor_best_success",
    "critic_best_success",
    "actor_minus_critic_best_success",
    "actor_first_hit_0.90",
    "critic_first_hit_0.90",
    "actor_minus_critic_hit90_step",
    "hit90_pair_complete",
)
PAIRED_STAT_FIELDS = (
    "metric",
    "delta_definition",
    "direction_favoring_actor_only",
    "n_pairs",
    "paired_seeds",
    "missing_pair_seeds",
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
    "exact_sign_flip_permutations",
)
MEAN_CURVE_FIELDS = (
    "arm",
    "weight_scope",
    "step",
    "n_seeds",
    "success_mean",
    "success_sample_std",
)


class AnalysisError(ValueError):
    """Raised when a run or analysis artifact violates the frozen protocol."""


class MissingPairedRunsError(AnalysisError):
    """Raised when any requested actor/critic seed is absent."""

    def __init__(self, missing: Mapping[str, Sequence[int]]) -> None:
        normalized = {key: tuple(sorted(values)) for key, values in missing.items() if values}
        self.missing = normalized
        details = "; ".join(f"{key} seeds={list(values)}" for key, values in normalized.items())
        super().__init__(
            "Paired analysis is fail-closed and requires both arms for every requested seed; "
            f"missing {details}. No partial paired result was produced."
        )


@dataclass(frozen=True)
class RunResult:
    arm: str
    weight_scope: str
    seed: int
    run_dir: Path
    run_name: str
    metrics_path: Path
    metrics_rows: tuple[dict[str, str], ...]
    curve: tuple[float, ...]
    auc: float
    final: float
    best: float
    first_hit_90: int | None
    config: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class AnalysisBundle:
    seeds: tuple[int, ...]
    runs_by_arm: dict[str, tuple[RunResult, ...]]
    per_seed_rows: tuple[dict[str, Any], ...]
    aggregate_rows: tuple[dict[str, Any], ...]
    paired_rows: tuple[dict[str, Any], ...]
    paired_stat_rows: tuple[dict[str, Any], ...]
    mean_curve_rows: tuple[dict[str, Any], ...]


def find_repo_root(path: Path) -> Path:
    """Find the owning worktree, including sparse experiment-only worktrees."""
    start = path.resolve()
    for candidate in (start, *start.parents):
        has_experiments = (candidate / "experiment_FPP").is_dir()
        if has_experiments and (
            (candidate / ".git").exists() or (candidate / "version_v0_lite").is_dir()
        ):
            return candidate
    raise AnalysisError(f"Could not locate repository root from {path}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
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


def _integer(value: Any, *, context: str) -> int:
    number = finite_float(value, context=context)
    if not number.is_integer():
        raise AnalysisError(f"Expected integer for {context}, got {value!r}")
    return int(number)


def config_seed(config: Mapping[str, Any]) -> int | None:
    seeds = config.get("seeds")
    value = seeds.get("env_seed") if isinstance(seeds, dict) else config.get("env_seed")
    number = optional_finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _candidate_matches(config: Mapping[str, Any], weight_scope: str, seed: int) -> bool:
    return (
        config.get("method") == METHOD
        and config.get("weight_scope") == weight_scope
        and config_seed(config) == seed
        and config.get("total_steps") == TOTAL_STEPS
        and config.get("smoke") is False
    )


def discover_run(root: Path, weight_scope: str, seed: int, *, allow_missing: bool = False) -> Path | None:
    """Discover one completed-protocol candidate by config, never by a smoke name."""
    if not root.is_dir():
        if allow_missing:
            return None
        raise MissingPairedRunsError({weight_scope: [seed]})
    matches: list[Path] = []
    for config_path in sorted(root.rglob("config.json")):
        try:
            config = read_json(config_path)
        except AnalysisError:
            continue
        if _candidate_matches(config, weight_scope, seed):
            matches.append(config_path.parent)
    if not matches:
        if allow_missing:
            return None
        raise MissingPairedRunsError({weight_scope: [seed]})
    if len(matches) != 1:
        raise AnalysisError(
            f"Expected exactly one {weight_scope} seed {seed} 1M non-smoke run under {root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _validate_seed_mapping(config: Mapping[str, Any], seed: int, run_dir: Path) -> None:
    mapping = config.get("seeds")
    if not isinstance(mapping, dict):
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


def validate_run_config(config: Mapping[str, Any], weight_scope: str, seed: int, run_dir: Path) -> None:
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
    if config.get("weight_scope") != weight_scope:
        raise AnalysisError(
            f"{run_dir}: config.weight_scope expected {weight_scope!r}, got {config.get('weight_scope')!r}"
        )
    if config.get("smoke") is not False:
        raise AnalysisError(f"{run_dir}: only non-smoke runs are admissible")
    _validate_seed_mapping(config, seed, run_dir)
    run_name = str(config.get("run_name", ""))
    required_suffix = f"_s{seed}_{TOTAL_STEPS}_ls512"
    if not run_name.endswith(required_suffix) or "smoke" in run_name.lower():
        raise AnalysisError(
            f"{run_dir}: run_name must end with {required_suffix!r} and exclude smoke; got {run_name!r}"
        )


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def validate_env_snapshot(snapshot: Mapping[str, Any], run_dir: Path) -> None:
    """Validate the task/environment identity and dimensions used by each run."""
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
            dotted = ".".join(path)
            raise AnalysisError(
                f"{run_dir}: env_snapshot.{dotted} expected {expected!r}, got {actual!r}"
            )


def metrics_path_for(run_dir: Path) -> Path:
    # Prefer the untouched raw metrics when present.  Historical canonical
    # critic data in this repository is already compacted, so that format is
    # the explicit fallback.
    for name in ("metrics.csv", "metrics_selected_10k.csv"):
        path = run_dir / name
        if path.is_file() and path.stat().st_size:
            return path
    raise AnalysisError(f"Missing metrics.csv or metrics_selected_10k.csv in {run_dir}")


def load_exact_grid(path: Path) -> tuple[tuple[dict[str, str], ...], tuple[float, ...]]:
    """Load exactly one observed evaluation at each frozen grid point."""
    by_step: dict[int, tuple[dict[str, str], float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "eval/success_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AnalysisError(f"{path}: missing required columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            success = optional_finite_float(row.get("eval/success_mean"))
            if success is None:
                continue
            step = _integer(row.get("step"), context=f"{path}:{row_number}:step")
            if step not in GRID:
                raise AnalysisError(
                    f"{path}: observed evaluation at unexpected step {step}; "
                    "interpolation and off-grid evaluations are prohibited"
                )
            if step in by_step:
                raise AnalysisError(f"{path}: duplicate observed evaluation at step {step}")
            if not 0.0 <= success <= 1.0:
                raise AnalysisError(f"{path}: success {success} outside [0,1] at step {step}")
            by_step[step] = (dict(row), success)
    missing = [step for step in GRID if step not in by_step]
    if missing:
        raise AnalysisError(
            f"{path}: missing exact observed evaluation steps {missing}; "
            "no interpolation, smoothing, or forward-fill is allowed"
        )
    rows = tuple(by_step[step][0] for step in GRID)
    curve = tuple(by_step[step][1] for step in GRID)
    return rows, curve


def normalized_trapezoid_auc(curve: Sequence[float]) -> float:
    if len(curve) != len(GRID):
        raise AnalysisError(f"AUC requires exactly {len(GRID)} observed grid values")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in curve):
        raise AnalysisError("AUC requires finite success values in [0,1]")
    area = sum(
        (GRID[index + 1] - GRID[index]) * (curve[index] + curve[index + 1]) / 2.0
        for index in range(len(GRID) - 1)
    )
    return area / float(GRID[-1] - GRID[0])


def first_hit(curve: Sequence[float], threshold: float = 0.90) -> int | None:
    return next((step for step, value in zip(GRID, curve) if value >= threshold), None)


def _normalized_summary_hit(value: Any, *, context: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _integer(value, context=context)


def _assert_close(actual: float, expected: Any, *, context: str) -> None:
    target = finite_float(expected, context=context)
    if not math.isclose(actual, target, rel_tol=SUMMARY_TOLERANCE, abs_tol=SUMMARY_TOLERANCE):
        raise AnalysisError(f"{context}: recomputed={actual:.15g}, recorded={target:.15g}")


def load_run(run_dir: Path, arm: str, weight_scope: str, seed: int) -> RunResult:
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    env_snapshot = read_json(run_dir / "env_snapshot.json")
    validate_run_config(config, weight_scope, seed, run_dir)
    validate_env_snapshot(env_snapshot, run_dir)
    if (
        summary.get("method") != METHOD
        or summary.get("total_steps") != TOTAL_STEPS
        or summary.get("total_updates") != TOTAL_UPDATES
    ):
        raise AnalysisError(
            f"{run_dir}: incomplete/mismatched summary; method={summary.get('method')!r}, "
            f"total_steps={summary.get('total_steps')!r}, "
            f"total_updates={summary.get('total_updates')!r}"
        )
    metrics_path = metrics_path_for(run_dir)
    metric_rows, curve = load_exact_grid(metrics_path)
    auc = normalized_trapezoid_auc(curve)
    final = curve[-1]
    best = max(curve)
    hit90 = first_hit(curve)
    _assert_close(final, summary.get("final_eval_success"), context=f"{run_dir}.final_eval_success")
    _assert_close(best, summary.get("best_eval_success"), context=f"{run_dir}.best_eval_success")
    _assert_close(auc, summary.get("eval_AUC_success"), context=f"{run_dir}.eval_AUC_success")
    recorded_hit = _normalized_summary_hit(
        summary.get("first_hit_0.90"), context=f"{run_dir}.first_hit_0.90"
    )
    if hit90 != recorded_hit:
        raise AnalysisError(
            f"{run_dir}.first_hit_0.90: recomputed={hit90!r}, recorded={recorded_hit!r}"
        )
    return RunResult(
        arm=arm,
        weight_scope=weight_scope,
        seed=seed,
        run_dir=run_dir.resolve(),
        run_name=str(config["run_name"]),
        metrics_path=metrics_path.resolve(),
        metrics_rows=metric_rows,
        curve=curve,
        auc=auc,
        final=final,
        best=best,
        first_hit_90=hit90,
        config=dict(config),
        summary=dict(summary),
    )


def _config_differences(
    actor: Mapping[str, Any], critic: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    differences: list[tuple[str, Any, Any]] = []
    for key in sorted(set(actor) | set(critic)):
        if key in PAIR_CONFIG_EXEMPT_KEYS:
            continue
        if actor.get(key) != critic.get(key):
            differences.append((key, actor.get(key), critic.get(key)))
    return differences


def validate_pair_config_parity(actor: RunResult, critic: RunResult) -> None:
    if actor.seed != critic.seed:
        raise AnalysisError(f"Cannot pair different seeds: actor={actor.seed}, critic={critic.seed}")
    differences = _config_differences(actor.config, critic.config)
    if differences:
        rendered = "; ".join(
            f"{key}: actor={left!r}, critic={right!r}" for key, left, right in differences
        )
        raise AnalysisError(
            f"Seed {actor.seed} config parity failed. Only {sorted(PAIR_CONFIG_EXEMPT_KEYS)} "
            f"may differ; found {rendered}"
        )


def sample_std(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        raise AnalysisError("Cannot compute sample SD for an empty sequence")
    return statistics.stdev(data) if len(data) > 1 else 0.0


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("Cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise AnalysisError(f"Invalid quantile probability {probability}")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if not differences:
        raise AnalysisError("Paired bootstrap requires at least one complete pair")
    if resamples <= 0:
        raise AnalysisError("Bootstrap resamples must be positive")
    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_two_sided_sign_flip_p(differences: Sequence[float]) -> float:
    if not differences:
        raise AnalysisError("Exact sign-flip test requires at least one complete pair")
    if len(differences) > 20:
        raise AnalysisError("Exact sign-flip enumeration is limited to at most 20 pairs")
    observed = abs(statistics.mean(differences))
    extreme = 0
    for mask in range(1 << len(differences)):
        statistic = abs(
            sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(differences)
            )
            / len(differences)
        )
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / float(1 << len(differences))


def _aggregate(arm: str, rows: Sequence[RunResult]) -> dict[str, Any]:
    hit_steps = [row.first_hit_90 for row in rows if row.first_hit_90 is not None]
    return {
        "arm": arm,
        "weight_scope": rows[0].weight_scope,
        "n_seeds": len(rows),
        "success_auc_mean": statistics.mean(row.auc for row in rows),
        "success_auc_sample_std": sample_std(row.auc for row in rows),
        "final_success_mean": statistics.mean(row.final for row in rows),
        "final_success_sample_std": sample_std(row.final for row in rows),
        "best_success_mean": statistics.mean(row.best for row in rows),
        "best_success_sample_std": sample_std(row.best for row in rows),
        "hit90_reached_count": len(hit_steps),
        "hit90_total": len(rows),
        "hit90_step_mean_reached_only": statistics.mean(hit_steps) if hit_steps else None,
        "hit90_step_sample_std_reached_only": sample_std(hit_steps) if hit_steps else None,
        "hit90_step_median_reached_only": statistics.median(hit_steps) if hit_steps else None,
    }


def _paired_stat(
    metric: str,
    values: Sequence[tuple[int, float]],
    all_seeds: Sequence[int],
    direction: str,
) -> dict[str, Any]:
    paired_seeds = [seed for seed, _value in values]
    missing = [seed for seed in all_seeds if seed not in paired_seeds]
    differences = [value for _seed, value in values]
    if differences:
        ci_low, ci_high = paired_bootstrap_ci(differences)
        p_value = exact_two_sided_sign_flip_p(differences)
        mean_delta: float | None = statistics.mean(differences)
        std_delta: float | None = sample_std(differences)
    else:
        ci_low = ci_high = p_value = mean_delta = std_delta = None
    return {
        "metric": metric,
        "delta_definition": "actor_only - canonical_critic_only",
        "direction_favoring_actor_only": direction,
        "n_pairs": len(differences),
        "paired_seeds": ",".join(map(str, paired_seeds)),
        "missing_pair_seeds": ",".join(map(str, missing)),
        "mean_delta": mean_delta,
        "sample_std_delta": std_delta,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES if differences else 0,
        "bootstrap_seed": BOOTSTRAP_SEED if differences else None,
        "positive_count": sum(value > 0.0 for value in differences),
        "negative_count": sum(value < 0.0 for value in differences),
        "zero_count": sum(value == 0.0 for value in differences),
        "exact_sign_flip_p_two_sided": p_value,
        "exact_sign_flip_permutations": (1 << len(differences)) if differences else 0,
    }


def analyze_scope_ablation(
    actor_root: Path,
    critic_root: Path,
    seeds: Sequence[int] = SEEDS,
) -> AnalysisBundle:
    requested = tuple(int(seed) for seed in seeds)
    if not requested or len(requested) != len(set(requested)):
        raise AnalysisError(f"Seeds must be a non-empty unique sequence, got {requested}")

    roots = {ACTOR_KEY: Path(actor_root), CRITIC_KEY: Path(critic_root)}
    scopes = {ACTOR_KEY: ACTOR_KEY, CRITIC_KEY: CRITIC_KEY}
    discovered: dict[str, dict[int, Path]] = {ACTOR_KEY: {}, CRITIC_KEY: {}}
    missing: dict[str, list[int]] = {ACTOR_KEY: [], CRITIC_KEY: []}
    for arm in (ACTOR_KEY, CRITIC_KEY):
        for seed in requested:
            run_dir = discover_run(roots[arm], scopes[arm], seed, allow_missing=True)
            if run_dir is None:
                missing[arm].append(seed)
            else:
                discovered[arm][seed] = run_dir
    if any(missing.values()):
        raise MissingPairedRunsError(missing)

    runs_by_arm: dict[str, tuple[RunResult, ...]] = {}
    for arm in (ACTOR_KEY, CRITIC_KEY):
        runs_by_arm[arm] = tuple(
            load_run(discovered[arm][seed], arm, scopes[arm], seed) for seed in requested
        )

    actor_by_seed = {row.seed: row for row in runs_by_arm[ACTOR_KEY]}
    critic_by_seed = {row.seed: row for row in runs_by_arm[CRITIC_KEY]}
    for seed in requested:
        validate_pair_config_parity(actor_by_seed[seed], critic_by_seed[seed])

    per_seed_rows = tuple(
        {
            "arm": row.arm,
            "weight_scope": row.weight_scope,
            "seed": row.seed,
            "run_name": row.run_name,
            "total_steps": TOTAL_STEPS,
            "eval_grid_points": len(GRID),
            "success_auc": row.auc,
            "final_success": row.final,
            "best_success": row.best,
            "first_hit_0.90": row.first_hit_90,
            "hit90_reached": int(row.first_hit_90 is not None),
        }
        for arm in (ACTOR_KEY, CRITIC_KEY)
        for row in runs_by_arm[arm]
    )
    aggregate_rows = tuple(
        _aggregate(arm, runs_by_arm[arm]) for arm in (ACTOR_KEY, CRITIC_KEY)
    )

    paired_rows_list: list[dict[str, Any]] = []
    for seed in requested:
        actor = actor_by_seed[seed]
        critic = critic_by_seed[seed]
        hit_complete = actor.first_hit_90 is not None and critic.first_hit_90 is not None
        paired_rows_list.append(
            {
                "seed": seed,
                "actor_auc": actor.auc,
                "critic_auc": critic.auc,
                "actor_minus_critic_auc": actor.auc - critic.auc,
                "actor_final_success": actor.final,
                "critic_final_success": critic.final,
                "actor_minus_critic_final_success": actor.final - critic.final,
                "actor_best_success": actor.best,
                "critic_best_success": critic.best,
                "actor_minus_critic_best_success": actor.best - critic.best,
                "actor_first_hit_0.90": actor.first_hit_90,
                "critic_first_hit_0.90": critic.first_hit_90,
                "actor_minus_critic_hit90_step": (
                    actor.first_hit_90 - critic.first_hit_90 if hit_complete else None
                ),
                "hit90_pair_complete": int(hit_complete),
            }
        )
    paired_rows = tuple(paired_rows_list)

    paired_stat_rows = (
        _paired_stat(
            "success_auc",
            [(int(row["seed"]), float(row["actor_minus_critic_auc"])) for row in paired_rows],
            requested,
            "positive",
        ),
        _paired_stat(
            "final_success",
            [
                (int(row["seed"]), float(row["actor_minus_critic_final_success"]))
                for row in paired_rows
            ],
            requested,
            "positive",
        ),
        _paired_stat(
            "best_success",
            [
                (int(row["seed"]), float(row["actor_minus_critic_best_success"]))
                for row in paired_rows
            ],
            requested,
            "positive",
        ),
        _paired_stat(
            "first_hit_0.90_step",
            [
                (int(row["seed"]), float(row["actor_minus_critic_hit90_step"]))
                for row in paired_rows
                if row["actor_minus_critic_hit90_step"] is not None
            ],
            requested,
            "negative",
        ),
    )
    mean_curve_rows = tuple(
        {
            "arm": arm,
            "weight_scope": runs_by_arm[arm][0].weight_scope,
            "step": step,
            "n_seeds": len(runs_by_arm[arm]),
            "success_mean": statistics.mean(run.curve[index] for run in runs_by_arm[arm]),
            "success_sample_std": sample_std(run.curve[index] for run in runs_by_arm[arm]),
        }
        for arm in (ACTOR_KEY, CRITIC_KEY)
        for index, step in enumerate(GRID)
    )
    return AnalysisBundle(
        seeds=requested,
        runs_by_arm=runs_by_arm,
        per_seed_rows=per_seed_rows,
        aggregate_rows=aggregate_rows,
        paired_rows=paired_rows,
        paired_stat_rows=paired_stat_rows,
        mean_curve_rows=mean_curve_rows,
    )


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def render_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return buffer.getvalue()


def analysis_payload(bundle: AnalysisBundle) -> dict[str, Any]:
    auc_stats = next(row for row in bundle.paired_stat_rows if row["metric"] == "success_auc")
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "analysis_status": "complete",
        "arms": {
            ACTOR_KEY: {"weight_scope": ACTOR_KEY},
            CRITIC_KEY: {"weight_scope": CRITIC_KEY, "canonical": True},
        },
        "seeds": list(bundle.seeds),
        "n_pairs": len(bundle.seeds),
        "primary_endpoint": (
            "trapezoidal eval/success_mean AUC on observed 10k..1M grid, divided by 990k"
        ),
        "evaluation_grid": {
            "start": GRID[0],
            "end": GRID[-1],
            "frequency": EVAL_FREQ,
            "points": len(GRID),
            "normalization_denominator": GRID[-1] - GRID[0],
        },
        "processing": {
            "interpolation": False,
            "smoothing": False,
            "forward_fill": False,
            "missing_paired_runs": "error; no partial paired result",
            "hit90_missingness": "descriptive reached count; paired step inference uses complete pairs only",
            "mean_learning_curve": "per-arm mean and sample SD at each exact observed grid point",
        },
        "config_parity": {
            "status": "verified",
            "only_exempt_keys": sorted(PAIR_CONFIG_EXEMPT_KEYS),
        },
        "aggregate": list(bundle.aggregate_rows),
        "paired_statistics": list(bundle.paired_stat_rows),
        "primary_paired_auc": {
            "delta_definition": auc_stats["delta_definition"],
            "mean_delta": auc_stats["mean_delta"],
            "sample_std_delta": auc_stats["sample_std_delta"],
            "bootstrap_ci_95": [
                auc_stats["bootstrap_ci95_low"],
                auc_stats["bootstrap_ci95_high"],
            ],
            "bootstrap_resamples": auc_stats["bootstrap_resamples"],
            "bootstrap_seed": auc_stats["bootstrap_seed"],
            "positive_count": auc_stats["positive_count"],
            "negative_count": auc_stats["negative_count"],
            "zero_count": auc_stats["zero_count"],
            "exact_sign_flip_p_two_sided": auc_stats["exact_sign_flip_p_two_sided"],
            "exact_sign_flip_permutations": auc_stats["exact_sign_flip_permutations"],
        },
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_summary(bundle: AnalysisBundle) -> str:
    aggregate = {row["arm"]: row for row in bundle.aggregate_rows}
    auc_stats = next(row for row in bundle.paired_stat_rows if row["metric"] == "success_auc")
    lines = [
        "# FPP DDPG+HER+SABER Actor-Only Scope Ablation",
        "",
        "Primary endpoint: normalized trapezoidal `eval/success_mean` AUC on the exact observed ",
        "10k–1M grid (100 points; denominator 990k). No interpolation, smoothing, or forward-fill.",
        "",
        "| Arm | n | AUC mean +/- sample SD | Final mean +/- sample SD | Best mean +/- sample SD | Hit90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in (ACTOR_KEY, CRITIC_KEY):
        row = aggregate[arm]
        hit90 = f"{row['hit90_reached_count']}/{row['hit90_total']}"
        lines.append(
            f"| {arm} | {row['n_seeds']} | {_fmt(row['success_auc_mean'])} +/- "
            f"{_fmt(row['success_auc_sample_std'])} | {_fmt(row['final_success_mean'])} +/- "
            f"{_fmt(row['final_success_sample_std'])} | {_fmt(row['best_success_mean'])} +/- "
            f"{_fmt(row['best_success_sample_std'])} | {hit90} |"
        )
    lines.extend(
        [
            "",
            "## Paired primary result",
            "",
            "Delta is actor-only minus canonical critic-only on seeds 1–5.",
            "",
            f"- Mean AUC delta: {_fmt(auc_stats['mean_delta'])}",
            f"- Sample SD of paired deltas: {_fmt(auc_stats['sample_std_delta'])}",
            f"- 95% paired bootstrap CI (10,000 resamples, RNG seed {BOOTSTRAP_SEED}): "
            f"[{_fmt(auc_stats['bootstrap_ci95_low'])}, {_fmt(auc_stats['bootstrap_ci95_high'])}]",
            f"- Positive/negative/zero seed deltas: {auc_stats['positive_count']}/"
            f"{auc_stats['negative_count']}/{auc_stats['zero_count']}",
            f"- Exact two-sided sign-flip p: {_fmt(auc_stats['exact_sign_flip_p_two_sided'])} "
            f"({auc_stats['exact_sign_flip_permutations']} assignments)",
            "",
            "Hit90 means the first observed 10k-grid step with success >= 0.90. Unreached runs remain "
            "missing; Hit90 step inference is complete-case and reports the omitted seeds explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def render_outputs(bundle: AnalysisBundle) -> dict[str, str]:
    return {
        "per_seed_results.csv": render_csv(bundle.per_seed_rows, PER_SEED_FIELDS),
        "aggregate_results.csv": render_csv(bundle.aggregate_rows, AGGREGATE_FIELDS),
        "paired_results.csv": render_csv(bundle.paired_rows, PAIRED_FIELDS),
        "paired_statistics.csv": render_csv(bundle.paired_stat_rows, PAIRED_STAT_FIELDS),
        "mean_learning_curve.csv": render_csv(bundle.mean_curve_rows, MEAN_CURVE_FIELDS),
        "analysis_results.json": json.dumps(
            analysis_payload(bundle), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        "analysis_summary.md": render_summary(bundle),
    }


def write_outputs(bundle: AnalysisBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in render_outputs(bundle).items():
        (output_dir / name).write_text(content, encoding="utf-8", newline="\n")


def verify_rendered_outputs(bundle: AnalysisBundle, output_dir: Path) -> None:
    expected = render_outputs(bundle)
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(expected):
        raise AnalysisError(
            f"Analysis file-set mismatch in {output_dir}: expected={sorted(expected)}, "
            f"actual={sorted(actual_names)}"
        )
    for name, content in expected.items():
        actual = (output_dir / name).read_text(encoding="utf-8")
        if actual != content:
            raise AnalysisError(f"Analysis output does not match exact recomputation: {output_dir / name}")
