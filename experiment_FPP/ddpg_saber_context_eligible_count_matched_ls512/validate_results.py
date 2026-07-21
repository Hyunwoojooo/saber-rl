#!/usr/bin/env python3
"""Strict completion gates for DDPG Context-Eligible Count-Matched runs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from source_fingerprint import load_fingerprint, verify_current


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CANONICAL_ROOT = REPO_ROOT / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512"
RUN_SLUG = "ddpg_saber_context_eligible_count_matched_ls512"
WARMUP_STEPS = 5_000
BATCH_SIZE = 256
STAGES = {
    "smoke": {
        "steps": 5_000,
        "root": Path("experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_smoke_5k"),
        "slug": "ddpg_saber_context_eligible_count_matched_smoke",
        "seeds": (1,),
    },
    "gate": {
        "steps": 50_000,
        "root": Path("experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_gate_50k"),
        "slug": "ddpg_saber_context_eligible_count_matched_gate",
        "seeds": (1,),
    },
    "production": {
        "steps": 1_000_000,
        "root": Path("experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_ls512"),
        "slug": RUN_SLUG,
        "seeds": (1, 2, 3, 4, 5),
    },
}
FATAL_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|torch\.cuda\.OutOfMemoryError|"
    r"\bRuntimeError\b|\bAssertionError\b|\bFloatingPointError\b|"
    r"segmentation fault|\bKilled\b|\[failed\]|\bERROR\b",
    re.IGNORECASE,
)
REQUIRED_ALWAYS = {
    "config.json", "env_snapshot.json", "metrics.csv", "metrics.jsonl",
    "run.log", "checkpoint.pt", "summary.json", "assignment_diagnostics_1000.csv",
}
ASSIGNMENT_ZERO_FIELDS = (
    "exact_k_mismatch_count",
    "raw_histogram_mismatch_count",
    "normalized_histogram_mismatch_count",
    "invariant_failure_count",
)
ASSIGNMENT_TRACKED_FIELDS = (
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
    "reference_raw_weight_std_mean",
    "reference_raw_weight_ESS_mean",
    "assigned_raw_weight_mean_mean",
    "assigned_raw_weight_std_mean",
    "assigned_raw_weight_ESS_mean",
    "reference_normalized_weight_mean_mean",
    "reference_normalized_weight_std_mean",
    "reference_normalized_weight_ESS_mean",
    "assigned_normalized_weight_mean_mean",
    "assigned_normalized_weight_std_mean",
    "assigned_normalized_weight_ESS_mean",
    "reference_normalized_weight_sum_mean",
    "assigned_normalized_weight_sum_mean",
    "normalized_weight_sum_abs_diff_mean",
)
OPTIONAL_NONFINITE_EXACT = {
    "query/query_bin_present_given_context_rate",
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "mean_w_P_reference",
    "reference_high_weight_rate",
    "assigned_high_weight_rate",
    "support_weight_alignment_rate",
    "support_cache_hit_rate",
    "short_circuit_rate",
    "lite/mean_w_P_reference",
    "lite/reference_high_weight_rate",
    "lite/assigned_high_weight_rate",
    "lite/support_weight_alignment_rate",
    "lite/support_cache_hit_rate",
    "lite/short_circuit_rate",
    "b2/gap_to_pos_progress_corr",
    "b2/gap_to_rot_progress_corr",
    "b2/high_gap_bad_progress_rate",
    "b3/support_to_pos_progress_corr",
    "b3/support_to_rot_progress_corr",
    "b3/support_to_success_corr",
    "b3/support_to_td_error_corr",
    "b4/B2_gap_minus_B4_gap",
    "b4/unsupported_gap_rate",
}


class GateError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def resolved(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def optional_nonfinite_metric(key: str) -> bool:
    return (
        key in OPTIONAL_NONFINITE_EXACT
        or (key.startswith("runtime/") and key not in {"runtime/steps_per_sec", "runtime/total_sec"})
        or key.startswith("td_error/")
        or key.startswith("per/")
        # The shared trainer emits Push-T evaluation placeholders for every
        # task. They are structurally unavailable in FetchPickAndPlace and
        # therefore use the canonical NaN sentinel. Only NaN is accepted by
        # the caller; +/-Inf remains fatal.
        or key.startswith("pushT/")
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
        require(allowed, f"non-finite JSON value at {'/'.join(path)}")


def read_object(
    path: Path, *, allow_optional_summary_nonfinite: bool = False
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    _validate_json_numbers(
        value,
        path=(),
        allow_optional_summary_nonfinite=allow_optional_summary_nonfinite,
    )
    return value


def number(value: Any, label: str) -> float:
    require(not isinstance(value, bool), f"{label} is bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} is not numeric: {value!r}") from exc
    require(math.isfinite(parsed), f"{label} is non-finite: {value!r}")
    return parsed


def integer(value: Any, label: str) -> int:
    parsed = number(value, label)
    require(parsed.is_integer(), f"{label} is not integral: {value!r}")
    return int(parsed)


def rate(value: Any, label: str) -> float:
    parsed = number(value, label)
    require(0.0 <= parsed <= 1.0, f"{label} is outside [0,1]: {parsed}")
    return parsed


def expected_updates(steps: int) -> int:
    return max(0, steps - max(WARMUP_STEPS, BATCH_SIZE) + 1)


def expected_eval_steps(steps: int) -> list[int]:
    # The trainer always evaluates at the final transition, even when the
    # budget is shorter than or is not divisible by eval_freq.  In particular,
    # the 5k startup smoke has the exact observed grid [5000].
    grid = list(range(10_000, steps + 1, 10_000))
    if not grid or grid[-1] != steps:
        grid.append(steps)
    return grid


def canonical_run_config(seed: int) -> dict[str, Any]:
    path = CANONICAL_ROOT / (
        f"fetch_pick_and_place_b3_critic_only_runtime_opt_v4_ls512_"
        f"s{seed}_1000000_ls512/config.json"
    )
    require(path.is_file(), f"missing canonical config: {path}")
    return read_object(path)


def normalized_for_parity(config: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(config)
    output["support_query_mode"] = output.get("support_query_mode", "current_policy")
    output["support_assignment_mode"] = output.get("support_assignment_mode", "support")
    seeds = dict(output.get("seeds", {}))
    seeds.pop("assignment_seed", None)
    output["seeds"] = seeds
    output.pop("assignment_diagnostic_window_updates", None)
    return output


def validate_config_parity(
    config: Mapping[str, Any], *, run_name: str, output_root: Path,
    stage: str, seed: int, steps: int,
) -> list[str]:
    canonical = normalized_for_parity(canonical_run_config(seed))
    control = normalized_for_parity(config)
    allowed = {"run_name", "output_root", "support_assignment_mode"}
    if stage != "production":
        allowed.add("total_steps")
    # config.json is written when a run is created and need not be rewritten by
    # a completed-run resume.  Prove resume through run.log/checkpoint state,
    # not through a mutable config field.
    canonical.pop("resume", None)
    control.pop("resume", None)
    canonical.pop("resume_path", None)
    control.pop("resume_path", None)
    differences = sorted(
        key for key in set(canonical) | set(control)
        if canonical.get(key) != control.get(key)
    )
    require(set(differences) == allowed, f"{run_name}: forbidden resolved config differences {differences}; expected {sorted(allowed)}")
    require(config.get("run_name") == run_name, f"{run_name}: run_name mismatch")
    require(resolved(str(config.get("output_root", ""))) == output_root, f"{run_name}: output_root mismatch")
    require(config.get("total_steps") == steps, f"{run_name}: total_steps mismatch")
    require(config.get("support_query_mode", "current_policy") == "current_policy", f"{run_name}: query mode mismatch")
    require(config.get("support_assignment_mode") == "context_eligible_count_matched", f"{run_name}: assignment mode mismatch")
    require(config.get("assignment_diagnostic_window_updates") == 1_000, f"{run_name}: diagnostic window mismatch")
    seed_map = config.get("seeds")
    require(isinstance(seed_map, Mapping), f"{run_name}: missing seed map")
    expected_seed_map = {
        "env_seed": seed, "torch_seed": seed, "numpy_seed": seed,
        "action_noise_seed": seed, "weight_seed": 20_000 + seed,
        "assignment_seed": 30_000 + seed, "eval_seed": 10_000 + seed,
    }
    for key, value in expected_seed_map.items():
        require(seed_map.get(key) == value, f"{run_name}: seeds.{key} expected {value!r}, got {seed_map.get(key)!r}")
    return differences


def validate_fixed_config(config: Mapping[str, Any], run_name: str, stage: str) -> None:
    expected = {
        "method": "b3_support_runtime_opt",
        "task_name": "fetch_pick_and_place",
        "env_id": "FetchPickAndPlace-v4",
        "resolved_env_id": "FetchPickAndPlace-v4",
        "reward_type": "sparse",
        "max_episode_steps": 50,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
        "updates_per_step": 1,
        "batch_size": 256,
        "buffer_capacity": 300_000,
        "her_ratio": 0.8,
        "gamma": 0.98,
        "tau": 0.005,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "hidden_dim": 256,
        "noise_std": 0.2,
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
        "weighting_mode": "batch_norm",
        "weight_scope": "critic_only",
        "policy_support_mode": "bin",
        "support_query_mode": "current_policy",
        "support_assignment_mode": "context_eligible_count_matched",
        "support_scan_mode": "deterministic_tail_v1",
        "support_cache_enabled": True,
        "support_cache_max_entries": 200_000,
        "local_scan_cache_enabled": True,
        "local_scan_cache_max_entries": 50_000,
        "support_short_circuit": True,
        "success_terminates": False,
        "timeout_bootstrap": True,
        "device": "cuda",
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"{run_name}: config.{key} expected {value!r}, got {config.get(key)!r}")
    require(isinstance(config.get("resume", False), bool), f"{run_name}: resume flag is not boolean")
    terminal = config.get("terminal_convention")
    require(terminal == {"success_terminates": False, "timeout_bootstrap": True}, f"{run_name}: terminal convention mismatch")


def validate_env_snapshot(snapshot: Mapping[str, Any], run_name: str) -> None:
    for label in ("train_env", "eval_env"):
        env = snapshot.get(label)
        require(isinstance(env, Mapping), f"{run_name}: missing {label}")
        expected = {"env_id": "FetchPickAndPlace-v4", "obs_dim": 25, "goal_dim": 3, "state_dim": 28, "action_dim": 4, "max_episode_steps": 50}
        for key, value in expected.items():
            require(env.get(key) == value, f"{run_name}: {label}.{key} mismatch")
        require(env.get("action_low") == [-1.0] * 4 and env.get("action_high") == [1.0] * 4, f"{run_name}: action bounds mismatch")


def load_assignment_diagnostics(path: Path, total_updates: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path}: missing header")
        required = {
            "window_index", "update_start", "update_end", "updates", "complete_window",
            *ASSIGNMENT_ZERO_FIELDS, *ASSIGNMENT_TRACKED_FIELDS,
            "normalized_weight_sum_abs_diff_max", "overlap_count_sum", "k_zero_update_count",
            "context_eligible_count_min", "context_eligible_count_max",
            "main_positive_count_min", "main_positive_count_max",
            "assigned_positive_count_min", "assigned_positive_count_max",
        }
        require(required <= set(reader.fieldnames), f"{path}: missing diagnostic columns {sorted(required - set(reader.fieldnames))}")
        rows = list(reader)
    require(rows, f"{path}: no diagnostic windows")
    covered = 0
    weighted = {field: 0.0 for field in ASSIGNMENT_TRACKED_FIELDS}
    zero_totals = {field: 0 for field in ASSIGNMENT_ZERO_FIELDS}
    k_zero_total = 0
    max_weight_diff = 0.0
    for index, row in enumerate(rows, start=1):
        label = f"{path}:window {index}"
        require(integer(row["window_index"], f"{label}.window_index") == index, f"{label}: index mismatch")
        updates = integer(row["updates"], f"{label}.updates")
        start = integer(row["update_start"], f"{label}.update_start")
        end = integer(row["update_end"], f"{label}.update_end")
        require(start == covered + 1 and end == covered + updates, f"{label}: non-contiguous update coverage")
        complete = integer(row["complete_window"], f"{label}.complete_window")
        require(0 < updates <= 1_000, f"{label}: invalid window size")
        require(complete == int(updates == 1_000), f"{label}: complete_window mismatch")
        if index < len(rows):
            require(updates == 1_000, f"{label}: non-final partial window")
        eligible = number(row["context_eligible_count_mean"], f"{label}.eligible")
        main = number(row["main_positive_count_mean"], f"{label}.main")
        assigned = number(row["assigned_positive_count_mean"], f"{label}.assigned")
        for field, value in (("eligible", eligible), ("main", main), ("assigned", assigned)):
            require(0.0 <= value <= BATCH_SIZE, f"{label}: {field} mean outside [0,{BATCH_SIZE}]")
        for prefix, mean in (
            ("context_eligible_count", eligible),
            ("main_positive_count", main),
            ("assigned_positive_count", assigned),
        ):
            minimum = number(row[f"{prefix}_min"], f"{label}.{prefix}_min")
            maximum = number(row[f"{prefix}_max"], f"{label}.{prefix}_max")
            require(
                0.0 <= minimum <= mean + 1e-7
                and mean <= maximum + 1e-7
                and maximum <= BATCH_SIZE,
                f"{label}: invalid {prefix} min/mean/max",
            )
        require(-1e-12 <= main <= eligible + 1e-7, f"{label}: k exceeds eligible")
        require(math.isclose(main, assigned, rel_tol=0.0, abs_tol=1e-7), f"{label}: exact-k mean mismatch")
        for prefix in ("raw_weight", "normalized_weight"):
            for statistic in ("mean", "std", "ESS"):
                left = number(row[f"reference_{prefix}_{statistic}_mean"], f"{label}.reference_{prefix}_{statistic}")
                right = number(row[f"assigned_{prefix}_{statistic}_mean"], f"{label}.assigned_{prefix}_{statistic}")
                require(math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7), f"{label}: {prefix} {statistic} mismatch")
                if statistic == "mean":
                    require(left > 0.0, f"{label}: nonpositive {prefix} mean")
                elif statistic == "std":
                    require(left >= 0.0, f"{label}: negative {prefix} std")
                else:
                    require(0.0 < left <= BATCH_SIZE + 1e-7, f"{label}: {prefix} ESS outside (0,{BATCH_SIZE}]")
        for side in ("reference", "assigned"):
            mean = number(row[f"{side}_normalized_weight_mean_mean"], f"{label}.{side}.normalized_mean")
            total = number(row[f"{side}_normalized_weight_sum_mean"], f"{label}.{side}.normalized_sum")
            require(math.isclose(mean, 1.0, rel_tol=0.0, abs_tol=1e-6), f"{label}: normalized mean != 1")
            require(math.isclose(total, BATCH_SIZE, rel_tol=0.0, abs_tol=1e-4), f"{label}: normalized sum != batch")
        for field in ASSIGNMENT_ZERO_FIELDS:
            value = integer(row[field], f"{label}.{field}")
            require(value == 0, f"{label}: {field}={value}")
            zero_totals[field] += value
        for field in (
            "context_eligible_rate_mean",
            "main_positive_rate_mean",
            "assigned_positive_rate_mean",
            "overlap_over_k_mean",
            "jaccard_mean",
        ):
            rate(row[field], f"{label}.{field}")
        overlap = number(row["overlap_count_mean"], f"{label}.overlap_count")
        require(0.0 <= overlap <= min(main, assigned) + 1e-7, f"{label}: overlap outside [0,k]")
        overlap_sum = number(row["overlap_count_sum"], f"{label}.overlap_count_sum")
        require(
            math.isclose(overlap_sum, overlap * updates, rel_tol=0.0, abs_tol=1e-5),
            f"{label}: overlap mean/sum mismatch",
        )
        k_zero = integer(row["k_zero_update_count"], f"{label}.k_zero_update_count")
        require(0 <= k_zero <= updates, f"{label}: k-zero count outside window")
        k_zero_total += k_zero
        max_weight_diff = max(max_weight_diff, number(row["normalized_weight_sum_abs_diff_max"], f"{label}.weight_sum_diff"))
        for field in ASSIGNMENT_TRACKED_FIELDS:
            weighted[field] += number(row[field], f"{label}.{field}") * updates
        covered += updates
    require(covered == total_updates, f"{path}: diagnostics cover {covered}, expected {total_updates}")
    require(max_weight_diff <= 1e-4, f"{path}: normalized weight-sum mismatch {max_weight_diff}")
    return {
        "window_count": len(rows),
        "covered_updates": covered,
        **zero_totals,
        "k_zero_update_count": k_zero_total,
        "normalized_weight_sum_abs_diff_max": max_weight_diff,
        **{field: total / covered for field, total in weighted.items()},
    }


def inspect_metrics(path: Path, steps: int, run_name: str) -> dict[str, Any]:
    episode_steps: list[int] = []
    eval_steps: list[int] = []
    assignment_seen = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path}: missing header")
        required = {"step", "episode", "config/weight_scope", "config/support_query_mode", "config/support_assignment_mode", "loss/critic", "loss/actor", "runtime/steps_per_sec", "eval/env_success_mean"}
        require(required <= set(reader.fieldnames), f"{path}: missing columns {sorted(required - set(reader.fieldnames))}")
        for row_number, row in enumerate(reader, start=2):
            require(None not in row, f"{path}:{row_number}: malformed row")
            step = integer(row.get("step"), f"{path}:{row_number}.step")
            require(0 < step <= steps, f"{path}:{row_number}: step out of range")
            require(row.get("config/weight_scope") == "critic_only", f"{path}:{row_number}: scope drift")
            require(row.get("config/support_query_mode") == "current_policy", f"{path}:{row_number}: query drift")
            require(row.get("config/support_assignment_mode") == "context_eligible_count_matched", f"{path}:{row_number}: assignment drift")
            number(row.get("loss/critic"), f"{path}:{row_number}.critic")
            number(row.get("loss/actor"), f"{path}:{row_number}.actor")
            require(number(row.get("runtime/steps_per_sec"), f"{path}:{row_number}.steps_per_sec") > 0.0, f"{path}:{row_number}: nonpositive throughput")
            is_eval = row.get("eval/goal_success_mean") not in (None, "")
            if is_eval:
                rate(row.get("eval/env_success_mean"), f"{path}:{row_number}.eval_env_success")
                eval_steps.append(step)
            else:
                episode_steps.append(step)
            if row.get("assignment/invariant_failure_count") not in (None, ""):
                for field in (
                    "assignment/invariant_failure_count",
                    "assignment/raw_histogram_mismatch_count",
                    "assignment/normalized_histogram_mismatch_count",
                ):
                    require(number(row.get(field), f"{path}:{row_number}.{field}") == 0.0, f"{path}:{row_number}: {field} nonzero")
                assignment_seen = True
    require(episode_steps == list(range(50, steps + 1, 50)), f"{run_name}: episode metric grid mismatch")
    require(eval_steps == expected_eval_steps(steps), f"{run_name}: eval grid mismatch {eval_steps}")
    require(len(eval_steps) == len(set(eval_steps)), f"{run_name}: duplicate eval checkpoints")
    require(assignment_seen, f"{run_name}: no assignment diagnostics in metrics")
    return {"episode_rows": len(episode_steps), "evaluation_steps": eval_steps, "evaluation_points": len(eval_steps)}


def inspect_jsonl(path: Path, steps: int) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(line.strip() and "\x00" not in line, f"{path}:{line_number}: blank/NUL")
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise GateError(f"{path}:{line_number}: invalid/non-finite JSON: {exc}") from exc
            require(isinstance(row, Mapping), f"{path}:{line_number}: non-object")
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    require(
                        math.isnan(value) and optional_nonfinite_metric(str(key)),
                        f"{path}:{line_number}: critical non-finite metric {key}",
                    )
            step = integer(row.get("step"), f"{path}:{line_number}.step")
            require(0 < step <= steps, f"{path}:{line_number}: step out of range")
            count += 1
    require(count > 0, f"{path}: empty JSONL")
    return count


def validate_summary(summary: Mapping[str, Any], run_dir: Path, steps: int) -> dict[str, Any]:
    expected = {"method": "b3_support_runtime_opt", "total_steps": steps, "episodes": steps // 50, "total_updates": expected_updates(steps), "final_replay_size": min(steps, 300_000)}
    for key, value in expected.items():
        require(summary.get(key) == value, f"{run_dir.name}: summary.{key} expected {value!r}")
    require(resolved(str(summary.get("run_dir", ""))) == run_dir, f"{run_dir.name}: summary run_dir mismatch")
    runtime = number(summary.get("runtime_total_sec"), f"{run_dir.name}.runtime")
    throughput = number(summary.get("final_steps_per_sec"), f"{run_dir.name}.throughput")
    require(runtime > 0 and throughput > 0, f"{run_dir.name}: invalid runtime")
    losses = summary.get("final_losses")
    require(isinstance(losses, Mapping), f"{run_dir.name}: missing final losses")
    for key, value in losses.items():
        number(value, f"{run_dir.name}.final_losses.{key}")
    if expected_eval_steps(steps):
        for key in ("final_eval_success", "best_eval_success", "eval_AUC_success"):
            rate(summary.get(key), f"{run_dir.name}.{key}")
    return {"total_updates": expected_updates(steps), "runtime_total_sec": runtime, "steps_per_sec": throughput}


def validate_checkpoint(path: Path, *, run_name: str, stage: str, seed: int, steps: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise GateError("PyTorch is required for checkpoint validation") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        require(isinstance(payload, Mapping), f"{run_name}: checkpoint is not a mapping")
        require(payload.get("global_step") == steps, f"{run_name}: checkpoint step mismatch")
        require(payload.get("total_updates") == expected_updates(steps), f"{run_name}: checkpoint update mismatch")
        config = payload.get("config")
        require(isinstance(config, Mapping), f"{run_name}: checkpoint missing config")
        require(config.get("support_query_mode") == "current_policy", f"{run_name}: checkpoint query mode")
        require(config.get("support_assignment_mode") == "context_eligible_count_matched", f"{run_name}: checkpoint assignment mode")
        require(config.get("seeds", {}).get("assignment_seed") == 30_000 + seed, f"{run_name}: checkpoint assignment seed")
        require("assignment_rng_state" in payload, f"{run_name}: missing assignment_rng_state")
        diagnostic_state = payload.get("assignment_diagnostics_state")
        require(isinstance(diagnostic_state, Mapping), f"{run_name}: missing assignment diagnostic state")
        def inspect_diagnostic_numbers(value: Any, label: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    inspect_diagnostic_numbers(child, f"{label}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    inspect_diagnostic_numbers(child, f"{label}[{index}]")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                require(math.isfinite(float(value)), f"{run_name}: non-finite {label}")
        inspect_diagnostic_numbers(diagnostic_state, "assignment_diagnostics_state")
        require(diagnostic_state.get("last_update") == expected_updates(steps), f"{run_name}: diagnostic last_update mismatch")
        require(diagnostic_state.get("count") == 0, f"{run_name}: final diagnostic window was not flushed")
        require(diagnostic_state.get("minimums") == {}, f"{run_name}: empty diagnostic minimums must be empty")
        require(diagnostic_state.get("maximums") == {}, f"{run_name}: empty diagnostic maximums must be empty")
        agent = payload.get("agent")
        require(isinstance(agent, Mapping), f"{run_name}: missing agent state")
        tensor_count = 0
        def inspect(value: Any) -> None:
            nonlocal tensor_count
            if torch.is_tensor(value):
                if torch.is_floating_point(value) or torch.is_complex(value):
                    tensor_count += 1
                    require(bool(torch.isfinite(value).all()), f"{run_name}: non-finite checkpoint tensor")
            elif isinstance(value, Mapping):
                for child in value.values(): inspect(child)
            elif isinstance(value, (list, tuple)):
                for child in value: inspect(child)
        inspect(agent)
        require(tensor_count > 0, f"{run_name}: no model tensors")
        return {"assignment_rng_state_present": True, "diagnostic_state_present": True, "finite_agent_tensors": tensor_count}
    finally:
        del payload
        gc.collect()


def validate_log(path: Path, run_dir: Path, steps: int, stage: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    require(not FATAL_PATTERN.search(text), f"fatal pattern in {path}")
    require(f"step={steps}" in text, f"missing final-step marker in {path}")
    require("finished run at" in text, f"missing completion marker in {path}")
    resumed = "resumed from" in text
    if stage in {"smoke", "gate"}:
        require(resumed, f"{path}: {stage} resume path was not exercised")
    return {"resume_exercised": resumed}


def run_name(stage: str, seed: int, steps: int) -> str:
    return f"fetch_pick_and_place_{STAGES[stage]['slug']}_s{seed}_{steps}_ls512"


def validate_run(output_root: Path, stage: str, seed: int, launcher_log_dir: Path | None) -> dict[str, Any]:
    steps = int(STAGES[stage]["steps"])
    name = run_name(stage, seed, steps)
    run_dir = (output_root / name).resolve()
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    matches = sorted(path.resolve() for path in output_root.glob(f"{name}*") if path.is_dir())
    require(matches == [run_dir], f"duplicate/timestamped run directories: {matches}")
    required = set(REQUIRED_ALWAYS)
    if expected_eval_steps(steps):
        required |= {"best_checkpoint.pt", "best_checkpoint.json"}
    missing = sorted(item for item in required if not (run_dir / item).is_file() or (run_dir / item).stat().st_size == 0)
    require(not missing, f"{name}: missing/empty files {missing}")
    checkpoint_steps = list(range(50_000, steps + 1, 50_000))
    if not checkpoint_steps or checkpoint_steps[-1] != steps:
        checkpoint_steps.append(steps)
    actual_checkpoints = sorted(path.name for path in run_dir.glob("checkpoint_step_*.pt") if path.is_file())
    expected_checkpoints = sorted(f"checkpoint_step_{step}.pt" for step in checkpoint_steps)
    require(actual_checkpoints == expected_checkpoints, f"{name}: checkpoint grid mismatch")
    config = read_object(run_dir / "config.json")
    differences = validate_config_parity(config, run_name=name, output_root=output_root, stage=stage, seed=seed, steps=steps)
    validate_fixed_config(config, name, stage)
    validate_env_snapshot(read_object(run_dir / "env_snapshot.json"), name)
    summary = validate_summary(
        read_object(
            run_dir / "summary.json",
            allow_optional_summary_nonfinite=True,
        ),
        run_dir,
        steps,
    )
    metrics = inspect_metrics(run_dir / "metrics.csv", steps, name)
    jsonl_rows = inspect_jsonl(run_dir / "metrics.jsonl", steps)
    diagnostics = load_assignment_diagnostics(run_dir / "assignment_diagnostics_1000.csv", expected_updates(steps))
    checkpoint = validate_checkpoint(run_dir / "checkpoint.pt", run_name=name, stage=stage, seed=seed, steps=steps)
    log = validate_log(run_dir / "run.log", run_dir, steps, stage)
    if launcher_log_dir is not None:
        launcher = launcher_log_dir / f"seed_{seed}.log"
        require(launcher.is_file(), f"missing launcher log {launcher}")
        launcher_text = launcher.read_text(encoding="utf-8", errors="replace")
        require(not FATAL_PATTERN.search(launcher_text), f"fatal pattern in {launcher}")
    return {
        "seed": seed, "run_dir": str(run_dir), "stage": stage, "steps": steps,
        "config_differences": differences, **summary, **metrics,
        "jsonl_rows": jsonl_rows, "assignment_diagnostics": diagnostics,
        "checkpoint": checkpoint, "log": log,
    }


def write_selected_10k(run_dir: Path, steps: int) -> Path:
    source = run_dir / "metrics.csv"
    target = run_dir / "metrics_selected_10k.csv"
    temporary = target.with_suffix(".csv.tmp")
    count = 0
    with source.open("r", encoding="utf-8", newline="") as src, temporary.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        require(reader.fieldnames is not None, f"{source}: missing header")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if row.get("eval/goal_success_mean") not in (None, ""):
                writer.writerow(row)
                count += 1
    require(count == len(expected_eval_steps(steps)), f"{run_dir}: selected eval count mismatch")
    temporary.replace(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--launcher-log-dir", type=Path)
    parser.add_argument("--source-fingerprint", type=Path, required=True)
    parser.add_argument("--write-selected-10k", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from validate_config_parity import validate_config, validate_source
        validate_config()
        validate_source()
        spec = STAGES[args.stage]
        output_root = resolved(args.output_root or spec["root"])
        require(output_root == resolved(spec["root"]), f"{args.stage} output root must be {resolved(spec['root'])}")
        seeds = tuple(args.seeds or spec["seeds"])
        require(seeds == tuple(spec["seeds"]), f"{args.stage} seeds must be exactly {list(spec['seeds'])}")
        launcher_log = resolved(args.launcher_log_dir) if args.launcher_log_dir else None
        fingerprint_path = resolved(args.source_fingerprint)
        expected_fingerprint = output_root / "_execution/source_fingerprint.json"
        require(
            fingerprint_path == expected_fingerprint,
            f"{args.stage} fingerprint must be {expected_fingerprint}",
        )
        fingerprint, _ = load_fingerprint(
            fingerprint_path, expected_stage=args.stage
        )
        verify_current(fingerprint, expected_stage=args.stage)
        runs = [validate_run(output_root, args.stage, seed, launcher_log) for seed in seeds]
        selected: list[str] = []
        if args.write_selected_10k:
            selected = [str(write_selected_10k(Path(run["run_dir"]), int(spec["steps"]))) for run in runs]
        # Fail if source changed while validation inspected the raw run.
        verify_current(fingerprint, expected_stage=args.stage)
        payload = {
            "schema_version": 1, "kind": args.stage, "passed": True,
            "performance_used_for_gate_decision": False,
            "steps": spec["steps"], "seeds": list(seeds), "runs": runs,
            "selected_10k_csv": selected,
            "source_fingerprint": fingerprint,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"[gate-fail] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
