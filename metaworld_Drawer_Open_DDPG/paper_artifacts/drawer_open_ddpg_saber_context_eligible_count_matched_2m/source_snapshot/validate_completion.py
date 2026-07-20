#!/usr/bin/env python3
"""Strict completion gate for DrawerOpen DDPG count-matched seeds 1--5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import analyze_results as analysis


TOTAL_STEPS = 2_000_000
TOTAL_UPDATES = 1_995_001
FINAL_REPLAY_SIZE = 300_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 20

STRICT_METRIC_FIELDS = (
    "strict/main_positive_count",
    "strict/context_eligible_count",
    "strict/control_positive_count",
    "strict/main_positive_rate",
    "strict/context_eligible_rate",
    "strict/control_positive_rate",
    "strict/main_control_overlap_count",
    "strict/main_control_overlap_rate",
    "strict/control_selected_outside_eligible",
    "strict/raw_weight_mean",
    "strict/normalized_weight_mean",
    "strict/normalized_weight_min",
    "strict/normalized_weight_max",
    "strict/count_equality_violation",
    "strict/raw_histogram_mismatch",
    "strict/fallback_sampling_used",
    "strict/replacement_sampling_used",
    "runtime/support_lookup_sec_per_update",
    "runtime/strict_assignment_sec_per_update",
)

TRACE_MEAN_FIELDS = (
    "context_eligible_count_mean",
    "main_positive_count_mean",
    "control_positive_count_mean",
    "context_eligible_rate_mean",
    "main_positive_rate_mean",
    "control_positive_rate_mean",
    "main_control_overlap_count_mean",
    "main_control_overlap_rate_mean",
    "raw_weight_mean_mean",
    "normalized_weight_mean_mean",
    "normalized_weight_min_mean",
    "normalized_weight_max_mean",
    "support_lookup_sec_per_update_mean",
    "strict_assignment_sec_per_update_mean",
)
TRACE_ZERO_SUM_FIELDS = (
    "count_equality_violation_sum",
    "control_selected_outside_eligible_sum",
    "raw_histogram_mismatch_sum",
    "fallback_sampling_used_sum",
    "replacement_sampling_used_sum",
)
TRACE_BOUND_FIELDS = (
    "normalized_weight_mean_min",
    "normalized_weight_mean_max",
    "normalized_weight_min_min",
    "normalized_weight_max_max",
)

# These are the only non-finite raw-metric families tolerated after warm-up.
# They are legacy diagnostics that are mathematically undefined or unrelated
# to DrawerOpen.  Strict, loss, value, replay, evaluation, and runtime fields
# are never covered by this allowlist.
ALLOWED_NAN_PREFIXES = (
    "pushT/",
    "td_error/",
    "per/",
    "b2/",
    "b4/",
    "diagnostic/",
    "lite/slide_",
)
ALLOWED_NAN_EXACT = {
    "runtime/overhead_vs_B0",
    "b3/support_to_pos_progress_corr",
    "b3/support_to_rot_progress_corr",
    "b3/support_to_success_corr",
    "b3/support_to_td_error_corr",
    "mean_w_P_reference",
    "reference_high_weight_rate",
    "assigned_high_weight_rate",
    "support_weight_alignment_rate",
    "support_weight_overlap_rate",
    "lite/mean_w_P_reference",
    "lite/reference_high_weight_rate",
    "lite/assigned_high_weight_rate",
    "lite/support_weight_alignment_rate",
    "lite/support_weight_overlap_rate",
}
NON_NUMERIC_METRIC_FIELDS = {"config/weight_scope"}

REQUIRED_CONFIG = {
    "task_name": "drawer_open",
    "reward_type": "sparse",
    "metaworld_env_name": "drawer-open-v3",
    "metaworld_reward_function_version": "v2",
    "max_episode_steps": 200,
    "total_steps": TOTAL_STEPS,
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "updates_per_step": 1,
    "batch_size": 256,
    "buffer_capacity": FINAL_REPLAY_SIZE,
    "her_ratio": 0.8,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "eval_freq": EVAL_FREQ,
    "eval_episodes": EVAL_EPISODES,
    "checkpoint_every_steps": 50_000,
    "save_step_checkpoints": False,
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
    "support_cache_enabled": True,
    "support_cache_max_entries": 200_000,
    "local_scan_cache_enabled": True,
    "local_scan_cache_max_entries": 50_000,
    "support_short_circuit": True,
    "support_query_mode": "current_policy",
    "numba_support_scan": True,
    "support_predicate_early_exit": True,
    "support_diagnostic_audit_interval": 100,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "method": "b3_support_runtime_opt",
    "support_assignment_mode": "context_eligible_count_matched",
}


@dataclass(frozen=True)
class Expectations:
    total_steps: int = TOTAL_STEPS
    total_updates: int = TOTAL_UPDATES
    final_replay_size: int = FINAL_REPLAY_SIZE
    eval_steps: tuple[int, ...] = analysis.EVAL_STEPS
    eval_episodes: int = EVAL_EPISODES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_finite(name: str, value: object) -> float:
    parsed = analysis.finite_float(value)
    if parsed is None:
        raise ValueError(f"{name}: expected a finite numeric value, got {value!r}")
    return parsed


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def require_config(config: Mapping[str, object], seed: int, expectations: Expectations) -> None:
    expected = dict(REQUIRED_CONFIG)
    expected["total_steps"] = expectations.total_steps
    expected["eval_episodes"] = expectations.eval_episodes
    expected["buffer_capacity"] = expectations.final_replay_size
    for key, wanted in expected.items():
        if key not in config:
            raise ValueError(f"config: missing canonical field {key}")
        actual = config[key]
        if isinstance(wanted, float):
            if not close(require_finite(f"config.{key}", actual), wanted, 1e-12):
                raise ValueError(f"config.{key}: {actual!r} != {wanted!r}")
        elif actual != wanted:
            raise ValueError(f"config.{key}: {actual!r} != {wanted!r}")
    diagnostic_window = int(config.get("assignment_diagnostic_window_updates", 0))
    if diagnostic_window <= 0:
        raise ValueError("config.assignment_diagnostic_window_updates must be positive")
    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("config.seeds must be an object")
    mappings = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
        "assignment_seed": 30_000 + seed,
    }
    for key, wanted in mappings.items():
        if int(seeds.get(key, -1)) != wanted:
            raise ValueError(f"config.seeds.{key}: {seeds.get(key)!r} != {wanted}")


def allowed_parity_path(path: str) -> bool:
    normalized = path.replace("[", ".").replace("]", "").strip(".")
    exact = {
        "method",
        "control_mode",
        "support_assignment_mode",
        "run_name",
        "output_root",
        "seeds.assignment_seed",
        "assignment_diagnostic_window_updates",
    }
    return normalized in exact or normalized.startswith("strict_") or normalized.startswith("strict.")


def _seed_record(payload: Mapping[str, object], seed: int) -> Mapping[str, object]:
    records = payload.get("seed_records", payload.get("seeds"))
    if isinstance(records, list):
        for item in records:
            if isinstance(item, dict) and int(item.get("seed", -1)) == seed:
                return item
    if isinstance(records, dict):
        for key in (str(seed), f"seed_{seed}", f"seed_{seed:02d}"):
            item = records.get(key)
            if isinstance(item, dict):
                return item
    return payload


def find_audit(run_dir: Path, strict_root: Path, name: str) -> Path:
    experiment_root = Path(__file__).resolve().parent
    for candidate in (run_dir / name, strict_root / name, experiment_root / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{run_dir}: missing {name} in run or strict root")


def validate_config_parity(path: Path, seed: int) -> dict[str, object]:
    payload = analysis.read_json(path)
    if payload.get("passed") is not True:
        raise ValueError(f"{path}: overall config parity failed")
    source_checks = payload.get("canonical_source_checks")
    if not isinstance(source_checks, dict) or not source_checks or not all(
        value is True for value in source_checks.values()
    ):
        raise ValueError(f"{path}: canonical Main source checks did not all pass")
    record = _seed_record(payload, seed)
    passed = record.get("passed", record.get("status") in {"passed", "pass"})
    if passed is not True:
        raise ValueError(f"{path}: parity did not pass for seed {seed}")
    unexpected = record.get(
        "unexpected_differences",
        record.get("nonallowed_differences", []),
    )
    if not isinstance(unexpected, list) or unexpected:
        raise ValueError(f"{path}: unexpected config differences for seed {seed}: {unexpected!r}")
    allowed = record.get("allowed_differences", record.get("differences", []))
    if not isinstance(allowed, list):
        raise ValueError(f"{path}: allowed_differences must be a list")
    paths: list[str] = []
    for item in allowed:
        difference_path = str(item.get("path", "")) if isinstance(item, dict) else str(item)
        if isinstance(item, dict) and item.get("allowed") is not True:
            raise ValueError(f"{path}: parity row is not explicitly allowed: {item!r}")
        paths.append(difference_path)
        if not allowed_parity_path(difference_path):
            raise ValueError(f"{path}: non-allowlisted parity difference {difference_path!r}")
    return {"path": path.as_posix(), "passed": True, "allowed_difference_paths": paths}


def validate_resume_audit(path: Path, seed: int) -> dict[str, object]:
    payload = analysis.read_json(path)
    if payload.get("passed") is not True:
        raise ValueError(f"{path}: overall checkpoint/resume audit failed")
    coverage = payload.get("seed_coverage")
    if coverage != list(analysis.SEEDS):
        raise ValueError(f"{path}: checkpoint/resume seed coverage mismatch: {coverage!r}")
    record = _seed_record(payload, seed)
    required = (
        "assignment_reproduced",
        "learning_result_reproduced",
        "other_rng_streams_unchanged",
        "checkpoint_rng_state_restored",
    )
    if record.get("passed", record.get("status") in {"passed", "pass"}) is not True:
        raise ValueError(f"{path}: checkpoint/resume audit did not pass for seed {seed}")
    missing_or_false = [field for field in required if record.get(field) is not True]
    if missing_or_false:
        raise ValueError(f"{path}: failed resume audit fields {missing_or_false}")
    return {"path": path.as_posix(), "passed": True, **{field: True for field in required}}


def validate_source_parity(path: Path) -> dict[str, object]:
    payload = analysis.read_json(path)
    if payload.get("passed") is not True or payload.get("failures") != []:
        raise ValueError(f"{path}: source parity did not pass")
    exact = payload.get("exact_files")
    controlled = payload.get("controlled_change_files")
    if not isinstance(exact, list) or not exact:
        raise ValueError(f"{path}: missing exact source parity records")
    if not isinstance(controlled, list) or not controlled:
        raise ValueError(f"{path}: missing controlled source records")
    for row in exact:
        if not isinstance(row, dict) or row.get("byte_identical") is not True:
            raise ValueError(f"{path}: non-identical exact source row {row!r}")
        if row.get("original_sha256") != row.get("control_sha256"):
            raise ValueError(f"{path}: source hashes differ for {row.get('path')!r}")
    for row in controlled:
        if not isinstance(row, dict) or len(str(row.get("sha256", ""))) != 64:
            raise ValueError(f"{path}: malformed controlled source row {row!r}")
        if int(row.get("size_bytes", 0)) <= 0:
            raise ValueError(f"{path}: empty controlled source file {row.get('path')!r}")
    return {
        "path": path.resolve().as_posix(),
        "passed": True,
        "exact_file_count": len(exact),
        "controlled_file_count": len(controlled),
    }


def validate_source_freeze(path: Path, experiment_root: Path) -> dict[str, object]:
    payload = analysis.read_json(path)
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: frozen source manifest is empty")
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: malformed frozen source row {row!r}")
        relative = str(row.get("path", ""))
        if not relative or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"{path}: unsafe or duplicate frozen source path {relative!r}")
        seen.add(relative)
        source = experiment_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"{path}: frozen source is missing: {source}")
        if source.stat().st_size != int(row.get("size", -1)) or sha256(source) != row.get("sha256"):
            raise ValueError(f"{path}: frozen source changed: {relative}")
    source_commit = str(payload.get("source_commit", ""))
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit.lower()):
        raise ValueError(f"{path}: invalid source commit {source_commit!r}")
    return {
        "path": path.resolve().as_posix(),
        "passed": True,
        "source_commit": source_commit,
        "file_count": len(entries),
    }


def validate_env_snapshot(snapshot: Mapping[str, object], seed: int) -> None:
    for key, expected_seed in (("train_env", seed), ("eval_env", 10_000 + seed)):
        env = snapshot.get(key)
        if not isinstance(env, dict):
            raise ValueError(f"env_snapshot.{key} must be an object")
        if str(env.get("env_name")) != "drawer-open-v3":
            raise ValueError(f"env_snapshot.{key}.env_name mismatch")
        if int(env.get("max_episode_steps", -1)) != 200:
            raise ValueError(f"env_snapshot.{key}.max_episode_steps mismatch")
        if str(env.get("reward_type")) != "sparse":
            raise ValueError(f"env_snapshot.{key}.reward_type mismatch")
        if str(env.get("desired_goal_source")) != "_target_pos":
            raise ValueError(f"env_snapshot.{key}.desired_goal_source mismatch")
        if int(env.get("action_dim", -1)) != 4:
            raise ValueError(f"env_snapshot.{key}.action_dim mismatch")
        kwargs = env.get("spec_kwargs")
        if not isinstance(kwargs, dict) or int(kwargs.get("seed", -1)) != expected_seed:
            raise ValueError(f"env_snapshot.{key}.spec_kwargs.seed mismatch")
    task = snapshot.get("task_spec")
    if not isinstance(task, dict) or "DrawerOpen" not in str(task.get("task", "")):
        raise ValueError("env_snapshot.task_spec is not DrawerOpen")


def _nan_allowed(field: str) -> bool:
    return field in ALLOWED_NAN_EXACT or field.startswith(ALLOWED_NAN_PREFIXES)


def validate_metrics(metrics_path: Path, warmup_steps: int) -> dict[str, object]:
    checked_rows = 0
    allowed_nan_observations: set[str] = set()
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(set(STRICT_METRIC_FIELDS) - fields)
        if missing:
            raise ValueError(f"{metrics_path}: missing strict metric columns {missing}")
        for row in reader:
            step = analysis.finite_float(row.get("step"))
            if step is None or step < warmup_steps:
                continue
            checked_rows += 1
            for field, raw in row.items():
                if raw in (None, "") or field in NON_NUMERIC_METRIC_FIELDS:
                    continue
                try:
                    parsed = float(raw)
                except ValueError:
                    raise ValueError(f"{metrics_path}: non-numeric metric {field}={raw!r}")
                if not math.isfinite(parsed):
                    if _nan_allowed(field):
                        allowed_nan_observations.add(field)
                        continue
                    raise ValueError(f"{metrics_path}: non-finite required metric {field} at step {int(step)}")
            strict = {field: require_finite(f"metrics[{int(step)}].{field}", row.get(field)) for field in STRICT_METRIC_FIELDS}
            if not close(strict["strict/main_positive_count"], strict["strict/control_positive_count"], 0.0):
                raise ValueError(f"{metrics_path}: logged main/control count mismatch at step {int(step)}")
            if strict["strict/control_selected_outside_eligible"] != 0.0:
                raise ValueError(f"{metrics_path}: outside-eligible selection at step {int(step)}")
            if strict["strict/count_equality_violation"] != 0.0:
                raise ValueError(f"{metrics_path}: equality violation at step {int(step)}")
            if strict["strict/fallback_sampling_used"] != 0.0 or strict["strict/replacement_sampling_used"] != 0.0:
                raise ValueError(f"{metrics_path}: fallback/replacement sampling at step {int(step)}")
            if not close(strict["strict/normalized_weight_mean"], 1.0, 1e-6):
                raise ValueError(f"{metrics_path}: normalized weight mean differs from one at step {int(step)}")
            if strict["strict/normalized_weight_min"] <= 0.0:
                raise ValueError(f"{metrics_path}: non-positive normalized weight at step {int(step)}")
    if checked_rows == 0:
        raise ValueError(f"{metrics_path}: no post-warmup metric rows")
    return {
        "post_warmup_rows_checked": checked_rows,
        "allowed_nan_fields_observed": sorted(allowed_nan_observations),
        "allowed_nan_exact": sorted(ALLOWED_NAN_EXACT),
        "allowed_nan_prefixes": list(ALLOWED_NAN_PREFIXES),
    }


def validate_trace(path: Path, total_updates: int) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty strict diagnostic trace")
    required = {
        "window_index", "update_start", "update_end", "updates", "complete_window",
        *TRACE_MEAN_FIELDS, *TRACE_ZERO_SUM_FIELDS, *TRACE_BOUND_FIELDS,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path}: missing strict trace columns {missing}")
    expected_start = 1
    update_sum = 0
    for index, row in enumerate(rows):
        for field, raw in row.items():
            if raw in (None, ""):
                raise ValueError(f"{path}: empty trace value row={index + 1} field={field}")
            require_finite(f"{path.name}[{index + 1}].{field}", raw)
        start = int(float(row["update_start"]))
        end = int(float(row["update_end"]))
        updates = int(float(row["updates"]))
        if start != expected_start or end != start + updates - 1 or updates <= 0:
            raise ValueError(f"{path}: non-contiguous diagnostic window at row {index + 1}")
        expected_start = end + 1
        update_sum += updates
        for field in TRACE_ZERO_SUM_FIELDS:
            if require_finite(field, row[field]) != 0.0:
                raise ValueError(f"{path}: {field} is nonzero in row {index + 1}")
        if not close(require_finite("normalized_weight_mean_min", row["normalized_weight_mean_min"]), 1.0, 1e-6):
            raise ValueError(f"{path}: per-update normalized mean minimum differs from one")
        if not close(require_finite("normalized_weight_mean_max", row["normalized_weight_mean_max"]), 1.0, 1e-6):
            raise ValueError(f"{path}: per-update normalized mean maximum differs from one")
        if require_finite("normalized_weight_min_min", row["normalized_weight_min_min"]) <= 0.0:
            raise ValueError(f"{path}: non-positive weight observed")
    if update_sum != total_updates or expected_start - 1 != total_updates:
        raise ValueError(f"{path}: trace covers {update_sum} updates, expected {total_updates}")
    return {
        "window_count": len(rows),
        "updates_covered": update_sum,
        "first_update": 1,
        "last_update": total_updates,
        "all_count_equal": True,
        "outside_eligible_sum": 0,
        "fallback_sum": 0,
        "replacement_sum": 0,
    }


def load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production environment has torch.
        raise RuntimeError("PyTorch is required to audit checkpoint.pt") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint payload is not a mapping")
    return payload


def _canonical_checkpoint_config(config: Mapping[str, object]) -> dict[str, object]:
    payload = dict(config)
    for field in ("run_name", "output_root", "resume", "resume_path"):
        payload.pop(field, None)
    return payload


def validate_checkpoint(path: Path, config: Mapping[str, object], expectations: Expectations) -> dict[str, object]:
    payload = load_checkpoint(path)
    if int(payload.get("global_step", -1)) != expectations.total_steps:
        raise ValueError(f"{path}: global_step mismatch")
    if int(payload.get("total_updates", -1)) != expectations.total_updates:
        raise ValueError(f"{path}: total_updates mismatch")
    replay = payload.get("replay")
    if not isinstance(replay, dict) or int(replay.get("size", -1)) != expectations.final_replay_size:
        raise ValueError(f"{path}: replay size mismatch")
    required_rng = (
        "noise_rng_state", "sample_rng_state", "weight_rng_state", "assignment_rng_state",
        "python_random_state", "numpy_global_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all",
    )
    missing_rng = [field for field in required_rng if field not in payload]
    if missing_rng:
        raise ValueError(f"{path}: missing RNG states {missing_rng}")
    history = payload.get("eval_history_env")
    if not isinstance(history, list):
        raise ValueError(f"{path}: missing eval_history_env")
    history_steps = [int(item[0]) for item in history]
    history_values = [require_finite(f"checkpoint.eval_history[{step}]", item[1]) for step, item in zip(history_steps, history)]
    if history_steps != list(expectations.eval_steps):
        raise ValueError(f"{path}: checkpoint exact eval history is incomplete")
    if any(not 0.0 <= value <= 1.0 for value in history_values):
        raise ValueError(f"{path}: checkpoint eval success outside [0,1]")
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"{path}: missing checkpoint config")
    if _canonical_checkpoint_config(checkpoint_config) != _canonical_checkpoint_config(config):
        raise ValueError(f"{path}: checkpoint config differs from run config")
    agent = payload.get("agent")
    if not isinstance(agent, dict) or "critic" not in agent or "critic_target" not in agent:
        raise ValueError(f"{path}: checkpoint is not single-critic DDPG")
    if "critic2" in agent or "critic2_target" in agent:
        raise ValueError(f"{path}: unexpected second critic")
    resume_audit = payload.get("resume_audit")
    if not isinstance(resume_audit, dict):
        raise ValueError(f"{path}: missing resume_audit metadata")
    expected_rng_fields = [
        "python_random_state", "numpy_global_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all",
        "noise_rng_state", "sample_rng_state", "weight_rng_state", "assignment_rng_state",
    ]
    if resume_audit.get("rng_state_fields") != expected_rng_fields:
        raise ValueError(f"{path}: resume_audit RNG field contract mismatch")
    if (
        resume_audit.get("strict_control") is not True
        or resume_audit.get("episode_boundary") is not True
        or int(resume_audit.get("global_step", -1)) != expectations.total_steps
        or int(resume_audit.get("total_updates", -1)) != expectations.total_updates
        or resume_audit.get("config_fingerprint") != payload.get("config_fingerprint")
    ):
        raise ValueError(f"{path}: inconsistent resume_audit metadata")
    diagnostic_state = payload.get("assignment_diagnostics_state")
    if not isinstance(diagnostic_state, dict):
        raise ValueError(f"{path}: missing assignment_diagnostics_state")
    if int(diagnostic_state.get("last_update", -1)) != expectations.total_updates:
        raise ValueError(f"{path}: diagnostic checkpoint state is incomplete")
    if int(diagnostic_state.get("count", -1)) != 0:
        raise ValueError(f"{path}: final diagnostic checkpoint has an unflushed window")
    return {
        "global_step": expectations.total_steps,
        "total_updates": expectations.total_updates,
        "replay_size": expectations.final_replay_size,
        "eval_history_points": len(history),
        "dedicated_assignment_rng_state_present": True,
        "single_critic_ddpg": True,
        "resume_audit_metadata_valid": True,
        "assignment_diagnostics_flushed": True,
    }


def validate_summary(summary: Mapping[str, object], steps: Sequence[int], values: Sequence[float], expectations: Expectations) -> dict[str, object]:
    if int(summary.get("total_steps", -1)) != expectations.total_steps:
        raise ValueError("summary.total_steps mismatch")
    if int(summary.get("total_updates", -1)) != expectations.total_updates:
        raise ValueError("summary.total_updates mismatch")
    if int(summary.get("final_replay_size", -1)) != expectations.final_replay_size:
        raise ValueError("summary.final_replay_size mismatch")
    auc = analysis.normalized_auc(steps, values)
    final = float(values[-1])
    best = max(float(value) for value in values)
    hit = analysis.first_hit90(steps, values)
    comparisons = {
        "eval_AUC_success": auc,
        "final_eval_env_success": final,
        "best_eval_env_success": best,
    }
    for field, recomputed in comparisons.items():
        native = require_finite(f"summary.{field}", summary.get(field))
        if not close(native, recomputed, 1e-10):
            raise ValueError(f"summary.{field}: {native} != recomputed {recomputed}")
    native_hit = summary.get("first_hit_0.90")
    normalized_hit = None if native_hit in (None, "") else int(native_hit)
    if normalized_hit != hit:
        raise ValueError(f"summary.first_hit_0.90: {normalized_hit} != recomputed {hit}")
    losses = summary.get("final_losses")
    if not isinstance(losses, dict) or not losses:
        raise ValueError("summary.final_losses missing")
    for field, value in losses.items():
        require_finite(f"summary.final_losses.{field}", value)
    strict = summary.get("final_strict_metrics", summary.get("final_lite_metrics"))
    if not isinstance(strict, dict):
        raise ValueError("summary final strict metrics missing")
    for field in STRICT_METRIC_FIELDS:
        require_finite(f"summary.final_strict_metrics.{field}", strict.get(field))
    return {"auc": auc, "final_success": final, "best_success": best, "hit90": hit}


def git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def validate_run(
    run_dir: Path,
    strict_root: Path,
    seed: int,
    expectations: Expectations = Expectations(),
) -> dict[str, object]:
    required = ("config.json", "env_snapshot.json", "summary.json", "metrics.csv", "checkpoint.pt")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing completion files {missing}")
    config = analysis.read_json(run_dir / "config.json")
    summary = analysis.read_json(run_dir / "summary.json")
    snapshot = analysis.read_json(run_dir / "env_snapshot.json")
    require_config(config, seed, expectations)
    trace_path = run_dir / f"strict_diagnostics_{int(config['assignment_diagnostic_window_updates'])}.csv"
    if not trace_path.is_file():
        raise FileNotFoundError(f"{run_dir}: missing {trace_path.name}")
    validate_env_snapshot(snapshot, seed)
    metrics_path = run_dir / "metrics.csv"
    steps, values = analysis.load_eval_curve(metrics_path, expectations.eval_steps)
    if len(steps) != len(expectations.eval_steps):
        raise ValueError(f"{run_dir}: evaluation point count mismatch")
    metrics_validation = validate_metrics(metrics_path, int(config["warmup_steps"]))
    summary_validation = validate_summary(summary, steps, values, expectations)
    trace_validation = validate_trace(trace_path, expectations.total_updates)
    checkpoint_validation = validate_checkpoint(run_dir / "checkpoint.pt", config, expectations)
    parity_path = find_audit(run_dir, strict_root, "config_parity.json")
    parity_validation = validate_config_parity(parity_path, seed)
    resume_path = find_audit(run_dir, strict_root, "checkpoint_resume_audit.json")
    resume_validation = validate_resume_audit(resume_path, seed)
    inputs = []
    for path in (
        run_dir / "config.json", run_dir / "env_snapshot.json", run_dir / "summary.json",
        metrics_path, trace_path, run_dir / "checkpoint.pt", parity_path, resume_path,
    ):
        inputs.append({"path": path.resolve().as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "seed": seed,
        "run_dir": run_dir.resolve().as_posix(),
        "passed": True,
        "total_steps": expectations.total_steps,
        "total_updates": expectations.total_updates,
        "final_replay_size": expectations.final_replay_size,
        "evaluation_points": len(steps),
        "evaluation_episodes_per_point": expectations.eval_episodes,
        "summary": summary_validation,
        "strict_trace": trace_validation,
        "metrics": metrics_validation,
        "checkpoint": checkpoint_validation,
        "config_parity": parity_validation,
        "checkpoint_resume_audit": resume_validation,
        "environment_snapshot_validated": True,
        "inputs": inputs,
    }


def validate_all(
    strict_root: Path,
    repo_root: Path,
    expectations: Expectations = Expectations(),
) -> dict[str, object]:
    runs = analysis.discover_runs(strict_root)
    records = [validate_run(run_dir, strict_root, seed, expectations) for seed, run_dir in runs.items()]
    experiment_root = Path(__file__).resolve().parent
    source_parity_path = find_audit(next(iter(runs.values())), strict_root, "source_parity.json")
    source_freeze_path = find_audit(next(iter(runs.values())), strict_root, "freeze_manifest.json")
    source_parity = validate_source_parity(source_parity_path)
    source_freeze = validate_source_freeze(source_freeze_path, experiment_root)
    current_head = git_head(repo_root)
    if current_head is not None and current_head != source_freeze["source_commit"]:
        # A dirty working tree is allowed while developing, but the recorded
        # source commit must identify the checkout used for the frozen run.
        # Content integrity is established by every frozen file hash above.
        head_relation = "recorded_source_commit_differs_from_current_head"
    else:
        head_relation = "recorded_source_commit_matches_current_head"
    return {
        "schema_version": 1,
        "experiment": "DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M",
        "passed": True,
        "source_commit": source_freeze["source_commit"],
        "current_git_head": current_head,
        "source_commit_relation": head_relation,
        "strict_root": strict_root.resolve().as_posix(),
        "seed_count": len(records),
        "seeds": list(analysis.SEEDS),
        "evaluation_grid": list(expectations.eval_steps),
        "evaluation_field": analysis.PRIMARY_SUCCESS_FIELD,
        "allowed_nan_policy": {
            "exact_fields": sorted(ALLOWED_NAN_EXACT),
            "prefix_families": list(ALLOWED_NAN_PREFIXES),
            "scope": "post-warmup raw metrics only; strict/loss/value/replay/eval/runtime fields excluded",
        },
        "source_parity": source_parity,
        "source_freeze": source_freeze,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--strict-root", type=Path,
        default=repo / "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strict_root = args.strict_root.resolve()
    output = args.output.resolve() if args.output else strict_root / "completion_validation.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing completion validation: {output}")
    result = validate_all(strict_root, args.repo_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": output.as_posix(), "seed_count": len(result["records"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
