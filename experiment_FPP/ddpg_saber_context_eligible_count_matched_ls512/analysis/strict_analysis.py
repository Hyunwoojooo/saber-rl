#!/usr/bin/env python3
"""Standalone fail-closed exact analysis for DDPG Main versus Strict control."""

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


ARTIFACT_ID = "fpp_ddpg_saber_context_eligible_count_matched_ls512_1m"
SCHEMA_VERSION = 1
MAIN_KEY = "main"
CONTROL_KEY = "control"
METHOD = "b3_support_runtime_opt"
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 1_000_000
TOTAL_UPDATES = 995_001
GRID = tuple(range(10_000, 1_000_001, 10_000))
AUC_DENOMINATOR = 990_000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027
PRIMARY_SUCCESS = "eval/env_success_mean"
FALLBACK_SUCCESS = "eval/success_mean"
ASSIGNMENT_MODES = {MAIN_KEY: "support", CONTROL_KEY: "context_eligible_count_matched"}
OPTIONAL_NONFINITE_EXACT = {
    "query/query_bin_present_given_context_rate",
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "mean_w_P_reference", "reference_high_weight_rate",
    "assigned_high_weight_rate", "support_weight_alignment_rate",
    "support_cache_hit_rate", "short_circuit_rate",
    "lite/mean_w_P_reference", "lite/reference_high_weight_rate",
    "lite/assigned_high_weight_rate", "lite/support_weight_alignment_rate",
    "lite/support_cache_hit_rate", "lite/short_circuit_rate",
    "b2/gap_to_pos_progress_corr", "b2/gap_to_rot_progress_corr",
    "b2/high_gap_bad_progress_rate",
    "b3/support_to_pos_progress_corr", "b3/support_to_rot_progress_corr",
    "b3/support_to_success_corr", "b3/support_to_td_error_corr",
    "b4/B2_gap_minus_B4_gap", "b4/unsupported_gap_rate",
}
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
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "checkpoint_every_steps": 50_000,
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
DIAGNOSTIC_FIELDS = (
    "context_eligible_count",
    "main_positive_count",
    "assigned_positive_count",
    "context_eligible_rate",
    "main_positive_rate",
    "assigned_positive_rate",
    "main_minus_assigned_positive_count",
    "overlap_count",
    "overlap_over_k",
    "jaccard",
    "k_zero_update_count",
    "exact_k_mismatch_count",
    "raw_histogram_mismatch_count",
    "normalized_histogram_mismatch_count",
    "invariant_failure_count",
    "reference_raw_weight_ESS",
    "assigned_raw_weight_ESS",
    "reference_normalized_weight_ESS",
    "assigned_normalized_weight_ESS",
    "normalized_weight_sum_abs_diff",
)


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class RunResult:
    method_key: str
    seed: int
    run_dir: Path
    run_name: str
    config: dict[str, Any]
    env_snapshot: dict[str, Any]
    summary: dict[str, Any]
    metrics_path: Path
    metrics_rows: tuple[dict[str, str], ...]
    curve: tuple[float, ...]
    success_sources: tuple[str, ...]
    auc: float
    final: float
    best: float
    hit90: int | None
    diagnostics_path: Path | None
    diagnostics: dict[str, float] | None


@dataclass(frozen=True)
class AnalysisBundle:
    runs: dict[str, tuple[RunResult, ...]]
    per_seed_rows: tuple[dict[str, Any], ...]
    aggregate_rows: tuple[dict[str, Any], ...]
    paired_rows: tuple[dict[str, Any], ...]
    inference_rows: tuple[dict[str, Any], ...]
    loso_rows: tuple[dict[str, Any], ...]
    mean_curve_rows: tuple[dict[str, Any], ...]
    diagnostic_rows: tuple[dict[str, Any], ...]


def find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "experiment_FPP").is_dir():
            return candidate
    raise AnalysisError(f"could not find repository/artifact root from {path}")


def optional_nonfinite_metric(key: str) -> bool:
    return (
        key in OPTIONAL_NONFINITE_EXACT
        or (key.startswith("runtime/") and key not in {"runtime/steps_per_sec", "runtime/total_sec"})
        or key.startswith("td_error/")
        or key.startswith("per/")
    )


def _validate_json_numbers(
    value: Any, *, path: tuple[str, ...], allow_optional_summary_nonfinite: bool
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_json_numbers(
                child,
                path=(*path, str(key)),
                allow_optional_summary_nonfinite=allow_optional_summary_nonfinite,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_numbers(
                child,
                path=(*path, str(index)),
                allow_optional_summary_nonfinite=allow_optional_summary_nonfinite,
            )
    elif isinstance(value, float) and not math.isfinite(value):
        allowed = (
            allow_optional_summary_nonfinite
            and math.isnan(value)
            and len(path) == 2
            and path[0] == "final_lite_metrics"
            and optional_nonfinite_metric(path[1])
        )
        if not allowed:
            raise AnalysisError(f"non-finite JSON value at {'/'.join(path)}")


def read_json(
    path: Path, *, allow_optional_summary_nonfinite: bool = False
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"missing/empty JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid/non-finite JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    _validate_json_numbers(
        value,
        path=(),
        allow_optional_summary_nonfinite=allow_optional_summary_nonfinite,
    )
    return value


def finite(value: Any, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{context}: invalid numeric {value!r}") from exc
    if not math.isfinite(parsed):
        raise AnalysisError(f"{context}: non-finite numeric {value!r}")
    return parsed


def integer(value: Any, context: str) -> int:
    parsed = finite(value, context)
    if not parsed.is_integer():
        raise AnalysisError(f"{context}: expected integer, got {value!r}")
    return int(parsed)


def config_seed(config: Mapping[str, Any]) -> int | None:
    seeds = config.get("seeds")
    value = seeds.get("env_seed") if isinstance(seeds, Mapping) else None
    try:
        return integer(value, "config seed")
    except AnalysisError:
        return None


def query_mode(config: Mapping[str, Any]) -> str:
    return str(config.get("support_query_mode", "current_policy"))


def assignment_mode(config: Mapping[str, Any]) -> str:
    return str(config.get("support_assignment_mode", "support"))


def metrics_path_for(run_dir: Path) -> Path:
    for name in ("metrics_selected_10k.csv", "metrics.csv"):
        path = run_dir / name
        if path.is_file() and path.stat().st_size:
            return path
    raise AnalysisError(f"{run_dir}: missing metrics")


def discover_run(root: Path, method_key: str, seed: int) -> Path:
    matches: list[Path] = []
    if not root.is_dir():
        raise AnalysisError(f"run root missing: {root}")
    for config_path in sorted(root.rglob("config.json")):
        try:
            config = read_json(config_path)
        except AnalysisError:
            continue
        if (
            config.get("method") == METHOD
            and config.get("total_steps") == TOTAL_STEPS
            and config.get("smoke") is False
            and config_seed(config) == seed
            and query_mode(config) == "current_policy"
            and assignment_mode(config) == ASSIGNMENT_MODES[method_key]
            and (config_path.parent / "summary.json").is_file()
            and (config_path.parent / "env_snapshot.json").is_file()
        ):
            try:
                metrics_path_for(config_path.parent)
            except AnalysisError:
                continue
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise AnalysisError(f"expected one exact {method_key} seed {seed} run under {root}, found {matches}")
    return matches[0]


def validate_config(config: Mapping[str, Any], method_key: str, seed: int, run_dir: Path) -> None:
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15) if isinstance(expected, float) and isinstance(actual, (int, float)) else actual == expected
        if not equal:
            raise AnalysisError(f"{run_dir}: config.{key} expected {expected!r}, got {actual!r}")
    if query_mode(config) != "current_policy" or assignment_mode(config) != ASSIGNMENT_MODES[method_key]:
        raise AnalysisError(f"{run_dir}: query/assignment mode mismatch")
    seeds = config.get("seeds")
    if not isinstance(seeds, Mapping):
        raise AnalysisError(f"{run_dir}: missing seed mapping")
    for key, value in {
        "env_seed": seed, "torch_seed": seed, "numpy_seed": seed,
        "action_noise_seed": seed, "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }.items():
        if seeds.get(key) != value:
            raise AnalysisError(f"{run_dir}: seeds.{key} mismatch")
    if method_key == CONTROL_KEY:
        if seeds.get("assignment_seed") != 30_000 + seed:
            raise AnalysisError(f"{run_dir}: assignment_seed mismatch")
        if config.get("assignment_diagnostic_window_updates") != 1_000:
            raise AnalysisError(f"{run_dir}: assignment diagnostic window mismatch")
    terminal = config.get("terminal_convention")
    if terminal != {"success_terminates": False, "timeout_bootstrap": True}:
        raise AnalysisError(f"{run_dir}: terminal convention mismatch")


def validate_env(snapshot: Mapping[str, Any], run_dir: Path) -> None:
    for label in ("train_env", "eval_env"):
        env = snapshot.get(label)
        if not isinstance(env, Mapping):
            raise AnalysisError(f"{run_dir}: missing {label}")
        for key, expected in {"env_id": "FetchPickAndPlace-v4", "obs_dim": 25, "goal_dim": 3, "state_dim": 28, "action_dim": 4, "max_episode_steps": 50}.items():
            if env.get(key) != expected:
                raise AnalysisError(f"{run_dir}: {label}.{key} mismatch")


def load_grid(path: Path) -> tuple[tuple[dict[str, str], ...], tuple[float, ...], tuple[str, ...]]:
    found: dict[int, tuple[dict[str, str], float, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise AnalysisError(f"{path}: missing step column")
        for row_number, row in enumerate(reader, start=2):
            primary = row.get(PRIMARY_SUCCESS)
            if primary in (None, ""):
                continue
            value = finite(primary, f"{path}:{row_number}.{PRIMARY_SUCCESS}")
            source = PRIMARY_SUCCESS
            step = integer(row.get("step"), f"{path}:{row_number}.step")
            if step not in GRID:
                raise AnalysisError(f"{path}: unexpected evaluation step {step}")
            if step in found:
                raise AnalysisError(f"{path}: duplicate evaluation step {step}")
            if not 0.0 <= value <= 1.0:
                raise AnalysisError(f"{path}: success outside [0,1]")
            found[step] = (dict(row), value, source)
    missing = [step for step in GRID if step not in found]
    if missing:
        raise AnalysisError(f"{path}: missing exact evaluation steps {missing}; no filling allowed")
    return tuple(found[s][0] for s in GRID), tuple(found[s][1] for s in GRID), tuple(found[s][2] for s in GRID)


def auc(curve: Sequence[float]) -> float:
    if len(curve) != len(GRID):
        raise AnalysisError("AUC requires exact 100-point grid")
    area = sum((GRID[i + 1] - GRID[i]) * (curve[i] + curve[i + 1]) / 2 for i in range(len(GRID) - 1))
    return area / AUC_DENOMINATOR


def load_assignment_diagnostics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AnalysisError(f"{path}: no assignment windows")
    covered = 0
    count_fields = {
        "k_zero_update_count",
        "exact_k_mismatch_count",
        "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch_count",
        "invariant_failure_count",
    }
    totals = {field: 0.0 for field in DIAGNOSTIC_FIELDS if field not in count_fields}
    count_totals = {field: 0 for field in count_fields}
    for index, row in enumerate(rows, start=1):
        updates = integer(row.get("updates"), f"{path}:window{index}.updates")
        start = integer(row.get("update_start"), f"{path}:window{index}.start")
        end = integer(row.get("update_end"), f"{path}:window{index}.end")
        if integer(row.get("window_index"), f"{path}:window{index}.index") != index or start != covered + 1 or end != covered + updates:
            raise AnalysisError(f"{path}: non-contiguous diagnostic windows")
        if not 0 < updates <= 1_000 or (index < len(rows) and updates != 1_000):
            raise AnalysisError(f"{path}: invalid partial window")
        complete = integer(row.get("complete_window"), f"{path}:window{index}.complete")
        if complete != int(updates == 1_000):
            raise AnalysisError(f"{path}: complete_window flag disagrees with window size")
        eligible = finite(row.get("context_eligible_count_mean"), f"{path}:eligible")
        main = finite(row.get("main_positive_count_mean"), f"{path}:main")
        assigned = finite(row.get("assigned_positive_count_mean"), f"{path}:assigned")
        for name, value in (("eligible", eligible), ("main", main), ("assigned", assigned)):
            if not 0.0 <= value <= 256.0:
                raise AnalysisError(f"{path}: {name} mean outside [0,256]")
        if main > eligible + 1e-7 or not math.isclose(main, assigned, rel_tol=0.0, abs_tol=1e-7):
            raise AnalysisError(f"{path}: exact-k/context invariant failed")
        for prefix, mean in (
            ("context_eligible_count", eligible),
            ("main_positive_count", main),
            ("assigned_positive_count", assigned),
        ):
            minimum = finite(row.get(f"{prefix}_min"), f"{path}:{prefix}_min")
            maximum = finite(row.get(f"{prefix}_max"), f"{path}:{prefix}_max")
            if not 0.0 <= minimum <= mean + 1e-7 or not mean <= maximum + 1e-7 or maximum > 256.0:
                raise AnalysisError(f"{path}: invalid {prefix} min/mean/max")
        for family in ("raw_weight", "normalized_weight"):
            for statistic in ("mean", "std", "ESS"):
                reference = finite(
                    row.get(f"reference_{family}_{statistic}_mean"),
                    f"{path}:reference_{family}_{statistic}",
                )
                assigned_statistic = finite(
                    row.get(f"assigned_{family}_{statistic}_mean"),
                    f"{path}:assigned_{family}_{statistic}",
                )
                if not math.isclose(reference, assigned_statistic, rel_tol=0.0, abs_tol=1e-7):
                    raise AnalysisError(
                        f"{path}: reference/assigned {family} {statistic} mismatch"
                    )
                if statistic == "mean" and reference <= 0.0:
                    raise AnalysisError(f"{path}: nonpositive {family} mean")
                if statistic == "std" and reference < 0.0:
                    raise AnalysisError(f"{path}: negative {family} std")
                if statistic == "ESS" and not 0.0 < reference <= 256.0 + 1e-7:
                    raise AnalysisError(f"{path}: {family} ESS outside (0,256]")
        for side in ("reference", "assigned"):
            normalized_mean = finite(
                row.get(f"{side}_normalized_weight_mean_mean"),
                f"{path}:{side}_normalized_mean",
            )
            normalized_sum = finite(
                row.get(f"{side}_normalized_weight_sum_mean"),
                f"{path}:{side}_normalized_sum",
            )
            if not math.isclose(normalized_mean, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise AnalysisError(f"{path}: {side} normalized-weight mean is not one")
            if not math.isclose(normalized_sum, 256.0, rel_tol=0.0, abs_tol=1e-4):
                raise AnalysisError(f"{path}: {side} normalized-weight sum is not batch size")
        values = {
            "context_eligible_count": eligible,
            "main_positive_count": main,
            "assigned_positive_count": assigned,
            "context_eligible_rate": finite(row.get("context_eligible_rate_mean"), f"{path}:eligible_rate"),
            "main_positive_rate": finite(row.get("main_positive_rate_mean"), f"{path}:main_rate"),
            "assigned_positive_rate": finite(row.get("assigned_positive_rate_mean"), f"{path}:assigned_rate"),
            "main_minus_assigned_positive_count": main - assigned,
            "overlap_count": finite(row.get("overlap_count_mean"), f"{path}:overlap"),
            "overlap_over_k": finite(row.get("overlap_over_k_mean"), f"{path}:overlap/k"),
            "jaccard": finite(row.get("jaccard_mean"), f"{path}:jaccard"),
            "k_zero_update_count": float(integer(row.get("k_zero_update_count"), f"{path}:k_zero")),
            "exact_k_mismatch_count": float(integer(row.get("exact_k_mismatch_count"), f"{path}:exact_k_mismatch")),
            "raw_histogram_mismatch_count": finite(row.get("raw_histogram_mismatch_count"), f"{path}:raw_mismatch"),
            "normalized_histogram_mismatch_count": finite(row.get("normalized_histogram_mismatch_count"), f"{path}:norm_mismatch"),
            "invariant_failure_count": finite(row.get("invariant_failure_count"), f"{path}:invariant"),
            "reference_raw_weight_ESS": finite(row.get("reference_raw_weight_ESS_mean"), f"{path}:ref_raw_ess"),
            "assigned_raw_weight_ESS": finite(row.get("assigned_raw_weight_ESS_mean"), f"{path}:assigned_raw_ess"),
            "reference_normalized_weight_ESS": finite(row.get("reference_normalized_weight_ESS_mean"), f"{path}:ref_norm_ess"),
            "assigned_normalized_weight_ESS": finite(row.get("assigned_normalized_weight_ESS_mean"), f"{path}:assigned_norm_ess"),
            "normalized_weight_sum_abs_diff": finite(row.get("normalized_weight_sum_abs_diff_max"), f"{path}:weight_sum_diff"),
        }
        for field in (
            "exact_k_mismatch_count",
            "raw_histogram_mismatch_count",
            "normalized_histogram_mismatch_count",
            "invariant_failure_count",
        ):
            if values[field] != 0.0:
                raise AnalysisError(f"{path}: {field} is nonzero")
        for field in ("context_eligible_rate", "main_positive_rate", "assigned_positive_rate", "overlap_over_k", "jaccard"):
            if not 0.0 <= values[field] <= 1.0:
                raise AnalysisError(f"{path}: {field} outside [0,1]")
        if not 0.0 <= values["overlap_count"] <= min(main, assigned) + 1e-7:
            raise AnalysisError(f"{path}: overlap count outside [0,k]")
        if not 0.0 <= values["k_zero_update_count"] <= updates:
            raise AnalysisError(f"{path}: k-zero count outside window")
        overlap_sum = finite(row.get("overlap_count_sum"), f"{path}:overlap_sum")
        if not math.isclose(overlap_sum, values["overlap_count"] * updates, rel_tol=0.0, abs_tol=1e-5):
            raise AnalysisError(f"{path}: overlap sum/mean mismatch")
        if abs(values["main_minus_assigned_positive_count"]) > 1e-7 or values["normalized_weight_sum_abs_diff"] > 1e-4:
            raise AnalysisError(f"{path}: count/weight-sum invariant failed")
        for field, value in values.items():
            if field in count_fields:
                count_totals[field] += int(value)
            else:
                totals[field] += value * updates
        covered += updates
    if covered != TOTAL_UPDATES:
        raise AnalysisError(f"{path}: diagnostics cover {covered}, expected {TOTAL_UPDATES}")
    return (
        {field: value / covered for field, value in totals.items()}
        | {field: float(value) for field, value in count_totals.items()}
        | {"covered_updates": float(covered), "window_count": float(len(rows))}
    )


def normalized_pair_configs(main: Mapping[str, Any], control: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left, right = dict(main), dict(control)
    left["support_query_mode"] = left.get("support_query_mode", "current_policy")
    left["support_assignment_mode"] = left.get("support_assignment_mode", "support")
    right["support_query_mode"] = right.get("support_query_mode", "current_policy")
    right["support_assignment_mode"] = right.get("support_assignment_mode", "support")
    for value in (left, right):
        value.pop("resume", None)
        value.pop("resume_path", None)
        seeds = dict(value.get("seeds", {}))
        seeds.pop("assignment_seed", None)
        value["seeds"] = seeds
        value.pop("assignment_diagnostic_window_updates", None)
    return left, right


def config_differences(main: Mapping[str, Any], control: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    left, right = normalized_pair_configs(main, control)
    return [(key, left.get(key), right.get(key)) for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def actual_config_differences(
    main: Mapping[str, Any], control: Mapping[str, Any]
) -> list[dict[str, Any]]:
    left, right = dict(main), dict(control)
    left["support_query_mode"] = left.get("support_query_mode", "current_policy")
    left["support_assignment_mode"] = left.get("support_assignment_mode", "support")
    right["support_query_mode"] = right.get("support_query_mode", "current_policy")
    right["support_assignment_mode"] = right.get("support_assignment_mode", "support")
    for value in (left, right):
        value.pop("resume", None)
        value.pop("resume_path", None)

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            for key in sorted(value):
                child = f"{prefix}.{key}" if prefix else str(key)
                output.update(flatten(value[key], child))
            return output
        return {prefix: value}

    flat_left, flat_right = flatten(left), flatten(right)
    return [
        {"field": field, "main": flat_left.get(field), "control": flat_right.get(field)}
        for field in sorted(set(flat_left) | set(flat_right))
        if flat_left.get(field) != flat_right.get(field)
    ]


def validate_pair(main: RunResult, control: RunResult) -> None:
    if main.seed != control.seed:
        raise AnalysisError("paired seeds differ")
    differences = config_differences(main.config, control.config)
    allowed = {"run_name", "output_root", "support_assignment_mode"}
    forbidden = [item for item in differences if item[0] not in allowed]
    if forbidden:
        raise AnalysisError(f"seed {main.seed}: forbidden config differences {forbidden}")
    expected = ("support_assignment_mode", "support", "context_eligible_count_matched")
    if expected not in differences:
        raise AnalysisError(f"seed {main.seed}: missing strict assignment difference")
    actual_fields = {item["field"] for item in actual_config_differences(main.config, control.config)}
    expected_actual = {
        "assignment_diagnostic_window_updates",
        "output_root",
        "run_name",
        "seeds.assignment_seed",
        "support_assignment_mode",
    }
    if actual_fields != expected_actual:
        raise AnalysisError(
            f"seed {main.seed}: actual config differences {sorted(actual_fields)} "
            f"!= {sorted(expected_actual)}"
        )


def load_run(run_dir: Path, method_key: str, seed: int) -> RunResult:
    config = read_json(run_dir / "config.json")
    summary = read_json(
        run_dir / "summary.json",
        allow_optional_summary_nonfinite=True,
    )
    snapshot = read_json(run_dir / "env_snapshot.json")
    validate_config(config, method_key, seed, run_dir)
    validate_env(snapshot, run_dir)
    if summary.get("method") != METHOD or summary.get("total_steps") != TOTAL_STEPS or summary.get("total_updates") != TOTAL_UPDATES:
        raise AnalysisError(f"{run_dir}: incomplete summary")
    metrics_path = metrics_path_for(run_dir)
    rows, curve, sources = load_grid(metrics_path)
    computed_auc = auc(curve)
    for actual, recorded, field in (
        (computed_auc, summary.get("eval_AUC_success"), "AUC"),
        (curve[-1], summary.get("final_eval_success"), "final"),
        (max(curve), summary.get("best_eval_success"), "best"),
    ):
        if not math.isclose(actual, finite(recorded, f"{run_dir}:{field}"), rel_tol=1e-12, abs_tol=1e-12):
            raise AnalysisError(f"{run_dir}: recomputed {field} differs from summary")
    hit90 = next((step for step, value in zip(GRID, curve) if value >= 0.9), None)
    recorded_hit = summary.get("first_hit_0.90")
    recorded_hit = None if recorded_hit in (None, "") else integer(recorded_hit, f"{run_dir}:Hit90")
    if recorded_hit != hit90:
        raise AnalysisError(f"{run_dir}: Hit90 mismatch")
    diagnostics_path = None
    diagnostics = None
    if method_key == CONTROL_KEY:
        diagnostics_path = run_dir / "assignment_diagnostics_1000.csv"
        if not diagnostics_path.is_file():
            raise AnalysisError(f"{run_dir}: missing assignment diagnostics")
        diagnostics = load_assignment_diagnostics(diagnostics_path)
    return RunResult(method_key, seed, run_dir.resolve(), str(config.get("run_name")), dict(config), dict(snapshot), dict(summary), metrics_path.resolve(), rows, curve, sources, computed_auc, curve[-1], max(curve), hit90, diagnostics_path, diagnostics)


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def paired_bootstrap(differences: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    if resamples <= 0 or not differences or not all(math.isfinite(value) for value in differences):
        raise AnalysisError("invalid paired bootstrap input")
    rng = random.Random(seed)
    means = [statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences) for _ in range(resamples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def exact_sign_flip(differences: Sequence[float]) -> float:
    observed = abs(statistics.fmean(differences))
    extreme = 0
    for mask in range(1 << len(differences)):
        value = statistics.fmean(delta * (1 if mask & (1 << index) else -1) for index, delta in enumerate(differences))
        extreme += abs(value) >= observed - 1e-15
    return extreme / (1 << len(differences))


def analyze(main_root: Path, control_root: Path) -> AnalysisBundle:
    runs = {
        key: tuple(load_run(discover_run(root, key, seed), key, seed) for seed in SEEDS)
        for key, root in ((MAIN_KEY, main_root), (CONTROL_KEY, control_root))
    }
    for main, control in zip(runs[MAIN_KEY], runs[CONTROL_KEY]):
        validate_pair(main, control)
    per_seed = tuple({
        "method_key": run.method_key, "seed": run.seed, "run_name": run.run_name,
        "success_auc": run.auc, "final_success": run.final, "best_success": run.best,
        "first_hit_0.90": run.hit90, "hit90_reached": run.hit90 is not None,
        "total_steps": TOTAL_STEPS, "total_updates": TOTAL_UPDATES, "eval_grid_points": len(GRID),
    } for key in (MAIN_KEY, CONTROL_KEY) for run in runs[key])
    aggregates: list[dict[str, Any]] = []
    for key in (MAIN_KEY, CONTROL_KEY):
        values = runs[key]
        hits = [run.hit90 for run in values if run.hit90 is not None]
        aggregates.append({
            "method_key": key, "n_seeds": len(values),
            "success_auc_mean": statistics.fmean(run.auc for run in values),
            "success_auc_sample_std": sample_std([run.auc for run in values]),
            "final_success_mean": statistics.fmean(run.final for run in values),
            "final_success_sample_std": sample_std([run.final for run in values]),
            "best_success_mean": statistics.fmean(run.best for run in values),
            "best_success_sample_std": sample_std([run.best for run in values]),
            "hit90_reached_count": len(hits), "hit90_total": len(values),
            "hit90_step_median_reached_only": statistics.median(hits) if hits else None,
        })
    paired = tuple({
        "comparison": "Main SABER - Strict Context-Eligible Count-Matched Control",
        "seed": main.seed, "main_auc": main.auc, "control_auc": control.auc,
        "main_minus_control_auc": main.auc - control.auc,
        "main_final_success": main.final, "control_final_success": control.final,
        "main_best_success": main.best, "control_best_success": control.best,
        "main_first_hit_0.90": main.hit90, "control_first_hit_0.90": control.hit90,
    } for main, control in zip(runs[MAIN_KEY], runs[CONTROL_KEY]))
    differences = [row["main_minus_control_auc"] for row in paired]
    ci_low, ci_high = paired_bootstrap(
        differences, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    )
    inference = ({
        "comparison": paired[0]["comparison"], "metric": "normalized_success_auc",
        "n_pairs": len(differences), "paired_seeds": ";".join(map(str, SEEDS)),
        "mean_delta": statistics.fmean(differences), "sample_std_delta": sample_std(differences),
        "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "positive_count": sum(value > 0 for value in differences),
        "negative_count": sum(value < 0 for value in differences),
        "zero_count": sum(value == 0 for value in differences),
        "exact_sign_flip_p_two_sided": exact_sign_flip(differences),
        "exact_sign_flip_assignments": 32,
    },)
    loso = tuple({
        "omitted_seed": omitted,
        "retained_seeds": ";".join(str(seed) for seed in SEEDS if seed != omitted),
        "mean_main_minus_control_auc": statistics.fmean(row["main_minus_control_auc"] for row in paired if row["seed"] != omitted),
    } for omitted in SEEDS)
    curves = tuple({
        "method_key": key, "step": step, "n_seeds": 5,
        "success_mean": statistics.fmean(run.curve[index] for run in runs[key]),
        "success_sample_std": sample_std([run.curve[index] for run in runs[key]]),
    } for key in (MAIN_KEY, CONTROL_KEY) for index, step in enumerate(GRID))
    diagnostics = tuple({"seed": run.seed, **(run.diagnostics or {})} for run in runs[CONTROL_KEY])
    return AnalysisBundle(runs, per_seed, tuple(aggregates), paired, inference, loso, curves, diagnostics)


PER_SEED_FIELDS = ("method_key", "seed", "run_name", "total_steps", "total_updates", "eval_grid_points", "success_auc", "final_success", "best_success", "first_hit_0.90", "hit90_reached")
AGGREGATE_FIELDS = ("method_key", "n_seeds", "success_auc_mean", "success_auc_sample_std", "final_success_mean", "final_success_sample_std", "best_success_mean", "best_success_sample_std", "hit90_reached_count", "hit90_total", "hit90_step_median_reached_only")
PAIRED_FIELDS = ("comparison", "seed", "main_auc", "control_auc", "main_minus_control_auc", "main_final_success", "control_final_success", "main_best_success", "control_best_success", "main_first_hit_0.90", "control_first_hit_0.90")
INFERENCE_FIELDS = ("comparison", "metric", "n_pairs", "paired_seeds", "mean_delta", "sample_std_delta", "bootstrap_ci95_low", "bootstrap_ci95_high", "bootstrap_resamples", "bootstrap_seed", "positive_count", "negative_count", "zero_count", "exact_sign_flip_p_two_sided", "exact_sign_flip_assignments")
LOSO_FIELDS = ("omitted_seed", "retained_seeds", "mean_main_minus_control_auc")
CURVE_FIELDS = ("method_key", "step", "n_seeds", "success_mean", "success_sample_std")
DIAGNOSTIC_OUTPUT_FIELDS = ("seed", *DIAGNOSTIC_FIELDS, "covered_updates", "window_count")
OUTPUTS = {
    "per_seed_results.csv": ("per_seed_rows", PER_SEED_FIELDS),
    "aggregate_results.csv": ("aggregate_rows", AGGREGATE_FIELDS),
    "paired_differences.csv": ("paired_rows", PAIRED_FIELDS),
    "paired_inference.csv": ("inference_rows", INFERENCE_FIELDS),
    "leave_one_seed_out.csv": ("loso_rows", LOSO_FIELDS),
    "mean_learning_curve.csv": ("mean_curve_rows", CURVE_FIELDS),
    "assignment_diagnostic_summary.csv": ("diagnostic_rows", DIAGNOSTIC_OUTPUT_FIELDS),
}


def csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue()


def report(bundle: AnalysisBundle) -> str:
    aggregate = {row["method_key"]: row for row in bundle.aggregate_rows}
    inference = bundle.inference_rows[0]
    return (
        "# FPP DDPG SABER Strict Context-Eligible Count-Matched Control\n\n"
        "Primary endpoint: exact observed 10k--1M environment-success grid; trapezoidal AUC / 990,000. "
        "No smoothing, interpolation, extrapolation, or forward filling.\n\n"
        f"Main AUC: {aggregate[MAIN_KEY]['success_auc_mean']:.6f} ± {aggregate[MAIN_KEY]['success_auc_sample_std']:.6f}.\n\n"
        f"Strict AUC: {aggregate[CONTROL_KEY]['success_auc_mean']:.6f} ± {aggregate[CONTROL_KEY]['success_auc_sample_std']:.6f}.\n\n"
        f"Main − Strict: {inference['mean_delta']:+.6f}, paired percentile 95% CI "
        f"[{inference['bootstrap_ci95_low']:.6f}, {inference['bootstrap_ci95_high']:.6f}], "
        f"positive seeds {inference['positive_count']}/5, exact sign-flip p={inference['exact_sign_flip_p_two_sided']:.6f}.\n"
    )


def write_outputs(bundle: AnalysisBundle, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, (attribute, fields) in OUTPUTS.items():
        (output / name).write_text(csv_text(getattr(bundle, attribute), fields), encoding="utf-8", newline="\n")
    payload = {
        "schema_version": SCHEMA_VERSION, "artifact_id": ARTIFACT_ID,
        "seeds": list(SEEDS), "grid": list(GRID), "auc_denominator": AUC_DENOMINATOR,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "aggregate": list(bundle.aggregate_rows), "paired_inference": list(bundle.inference_rows),
    }
    (output / "analysis_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(bundle), encoding="utf-8")


def verify_outputs(bundle: AnalysisBundle, output: Path, extra: frozenset[str] = frozenset()) -> None:
    expected = {name: csv_text(getattr(bundle, attribute), fields) for name, (attribute, fields) in OUTPUTS.items()}
    expected["analysis_results.json"] = json.dumps({
        "schema_version": SCHEMA_VERSION, "artifact_id": ARTIFACT_ID,
        "seeds": list(SEEDS), "grid": list(GRID), "auc_denominator": AUC_DENOMINATOR,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "aggregate": list(bundle.aggregate_rows), "paired_inference": list(bundle.inference_rows),
    }, indent=2, sort_keys=True, allow_nan=False) + "\n"
    expected["report.md"] = report(bundle)
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != set(expected) | set(extra):
        raise AnalysisError(f"result file set mismatch: {actual}")
    for name, text in expected.items():
        if (output / name).read_text(encoding="utf-8") != text:
            raise AnalysisError(f"rendered output differs from recomputation: {name}")
