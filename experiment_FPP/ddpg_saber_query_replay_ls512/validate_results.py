#!/usr/bin/env python3
"""Strict completion gate for the FPP DDPG SABER replay-action query arm."""

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


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CANONICAL_RUN_ROOT = REPO_ROOT / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512"
RUN_SLUG = "ddpg_saber_query_replay_ls512"
RUN_PREFIX = f"fetch_pick_and_place_{RUN_SLUG}"
MAIN_STEPS = 1_000_000
SMOKE_STEPS = 20_000
WARMUP_STEPS = 5_000
MAIN_ROOT = Path("experiment_FPP/runs/ddpg_saber_query_replay_ls512")
SMOKE_ROOT = Path("experiment_FPP/runs/ddpg_saber_query_replay_ls512_smoke20k")
PHASES = {
    "smoke": (SMOKE_STEPS, SMOKE_ROOT),
    "main": (MAIN_STEPS, MAIN_ROOT),
}
REQUIRED_FILES = (
    "config.json",
    "env_snapshot.json",
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
    "checkpoint.pt",
    "best_checkpoint.pt",
    "best_checkpoint.json",
    "summary.json",
)
FATAL_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|torch\.cuda\.OutOfMemoryError|"
    r"\bRuntimeError\b|\bAssertionError\b|\bFloatingPointError\b|"
    r"segmentation fault|\bKilled\b|\[failed\]|\bERROR\b",
    re.IGNORECASE,
)
BASE_FINITE_FIELDS = (
    "train/episode_return",
    "train/episode_success",
    "train/episode_len",
    "train/replay_size",
    "train/her_relabel_ratio",
    "config/lambda_support",
    "loss/critic",
    "loss/actor",
    "loss/td_error_abs_mean",
    "loss/td_error_abs_p90",
    "value/q_mean",
    "value/target_mean",
    "lite/mean_w_P",
    "lite/max_w_P",
    "lite/std_w_P",
    "lite/ESS_w",
    "lite/ESS_w_normalized",
    "lite/support_available_rate",
    "lite/psi_q_rank_rate",
    "lite/q_candidate_fallback_rate",
    "runtime/steps_per_sec",
    "runtime/total_sec",
)
QUERY_FINITE_FIELDS = (
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/raw_weight_std",
    "query/raw_weight_ESS",
    "query/normalized_weight_mean",
    "query/normalized_weight_std",
    "query/normalized_weight_ESS",
)
RATE_FIELDS = (
    "train/episode_success",
    "train/her_relabel_ratio",
    "lite/support_available_rate",
    "lite/psi_q_rank_rate",
    "lite/q_candidate_fallback_rate",
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
)
EVAL_FINITE_FIELDS = (
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/goal_success_ever_mean",
    "eval/goal_success_ever_std",
    "eval/env_success_ever_mean",
    "eval/env_success_ever_std",
    "eval/goal_success_final_mean",
    "eval/goal_success_final_std",
    "eval/env_success_final_mean",
    "eval/env_success_final_std",
    "eval/episode_length_mean",
    "eval/final_success",
    "eval/best_success",
    "eval/AUC_success",
)
EVAL_RATE_FIELDS = (
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/goal_success_ever_mean",
    "eval/env_success_ever_mean",
    "eval/goal_success_final_mean",
    "eval/env_success_final_mean",
    "eval/final_success",
    "eval/best_success",
    "eval/AUC_success",
)


class GateError(AssertionError):
    """A result failed a preregistered experiment invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"could not read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def resolved(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def number(value: Any, label: str) -> float:
    require(not isinstance(value, bool), f"{label} must be numeric, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} is not numeric: {value!r}") from exc
    require(math.isfinite(parsed), f"{label} is non-finite: {value!r}")
    return parsed


def optional_rate(value: Any, label: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label} is not numeric: {value!r}") from exc
    if math.isnan(parsed):
        return None
    require(math.isfinite(parsed) and 0.0 <= parsed <= 1.0, f"{label} must be NaN or [0,1]: {value!r}")
    return parsed


def integer(value: Any, label: str) -> int:
    parsed = number(value, label)
    require(parsed.is_integer(), f"{label} must be an integer: {value!r}")
    return int(parsed)


def rate(value: Any, label: str) -> float:
    parsed = number(value, label)
    require(0.0 <= parsed <= 1.0, f"{label} must be in [0,1]: {parsed}")
    return parsed


def expected_eval_steps(steps: int) -> list[int]:
    return list(range(10_000, steps + 1, 10_000))


def expected_updates(steps: int) -> int:
    return max(0, steps - max(WARMUP_STEPS, 256) + 1)


def infer_phase(steps: int, requested: str | None) -> str:
    matches = [name for name, (budget, _root) in PHASES.items() if budget == steps]
    require(matches, f"steps must be exactly {SMOKE_STEPS} or {MAIN_STEPS}")
    phase = matches[0]
    require(requested in (None, phase), f"--phase {requested!r} conflicts with --steps {steps}")
    return phase


def validate_package_contract(package: Mapping[str, Any]) -> None:
    expected = {
        "trainer_method": "b3_support_runtime_opt",
        "support_query_mode": "replay_action",
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "run_slug": RUN_SLUG,
        "output_root": MAIN_ROOT.as_posix(),
        "total_steps": MAIN_STEPS,
        "local_search_max": 512,
        "default_local_search_max": 512,
        "lambda_support": 1.0,
    }
    for key, value in expected.items():
        require(package.get(key) == value, f"package config {key!r} must equal {value!r}")
    require(package.get("seeds") == [1, 2, 3, 4, 5], "package seeds must be exactly [1,2,3,4,5]")


def validate_output_root(output_root: Path, phase: str) -> None:
    registered = resolved(PHASES[phase][1])
    require(output_root == registered, f"{phase} output root must be {registered}, got {output_root}")
    require(resolved(MAIN_ROOT) != resolved(SMOKE_ROOT), "main and smoke roots alias")


def canonical_run_config(seed: int) -> dict[str, Any]:
    name = f"fetch_pick_and_place_b3_critic_only_runtime_opt_v4_ls512_s{seed}_1000000_ls512"
    path = CANONICAL_RUN_ROOT / name / "config.json"
    require(path.is_file(), f"missing canonical resolved config for seed {seed}: {path}")
    return read_object(path)


def validate_resolved_config_parity(
    cfg: Mapping[str, Any], run_name: str, output_root: Path, seed: int, steps: int
) -> None:
    canonical = canonical_run_config(seed)
    added = set(cfg) - set(canonical)
    missing = set(canonical) - set(cfg)
    require(added == {"support_query_mode"}, f"{run_name}: resolved config added keys {sorted(added)}")
    require(not missing, f"{run_name}: resolved config missing keys {sorted(missing)}")
    allowed_value_changes = {"run_name", "output_root"}
    if steps == SMOKE_STEPS:
        allowed_value_changes.add("total_steps")
    changed = {key for key in canonical if cfg[key] != canonical[key]}
    require(
        changed == allowed_value_changes,
        f"{run_name}: forbidden resolved config changes: expected={sorted(allowed_value_changes)}, actual={sorted(changed)}",
    )
    require(cfg.get("run_name") == run_name, f"{run_name}: run_name mismatch")
    require(resolved(str(cfg.get("output_root", ""))) == output_root, f"{run_name}: output_root mismatch")
    require(cfg.get("total_steps") == steps, f"{run_name}: total_steps mismatch")
    require(cfg.get("support_query_mode") == "replay_action", f"{run_name}: query mode mismatch")
    require(cfg.get("thread_hint") == canonical.get("thread_hint"), f"{run_name}: canonical thread_hint drift")


def validate_run_config(
    cfg: Mapping[str, Any], output_root: Path, run_name: str, seed: int, steps: int
) -> None:
    validate_resolved_config_parity(cfg, run_name, output_root, seed, steps)
    exact = {
        "method": "b3_support_runtime_opt",
        "task_name": "fetch_pick_and_place",
        "reward_type": "sparse",
        "env_id": "FetchPickAndPlace-v4",
        "resolved_env_id": "FetchPickAndPlace-v4",
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
        "save_step_checkpoints": True,
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
        "support_query_mode": "replay_action",
        "support_scan_mode": "deterministic_tail_v1",
        "support_cache_enabled": True,
        "support_cache_max_entries": 200_000,
        "local_scan_cache_enabled": True,
        "local_scan_cache_max_entries": 50_000,
        "support_short_circuit": True,
        "success_terminates": False,
        "timeout_bootstrap": True,
        "smoke": False,
        "resume": False,
        "resume_path": None,
        "render": False,
        "num_envs": 1,
        "device": "cuda",
    }
    for key, value in exact.items():
        require(cfg.get(key) == value, f"{run_name}: config {key!r} must be {value!r}, got {cfg.get(key)!r}")
    expected_seeds = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    require(cfg.get("seeds") == expected_seeds, f"{run_name}: seed mapping mismatch")
    require(
        cfg.get("terminal_convention") == {"success_terminates": False, "timeout_bootstrap": True},
        f"{run_name}: terminal convention mismatch",
    )


def validate_env_snapshot(snapshot: Mapping[str, Any], run_name: str) -> None:
    for label in ("train_env", "eval_env"):
        env = snapshot.get(label)
        require(isinstance(env, dict), f"{run_name}: missing {label}")
        expected = {
            "env_id": "FetchPickAndPlace-v4",
            "obs_dim": 25,
            "goal_dim": 3,
            "state_dim": 28,
            "action_dim": 4,
            "max_episode_steps": 50,
        }
        for key, value in expected.items():
            require(env.get(key) == value, f"{run_name}: {label}.{key} mismatch")
        require(env.get("action_low") == [-1.0] * 4, f"{run_name}: {label} action_low mismatch")
        require(env.get("action_high") == [1.0] * 4, f"{run_name}: {label} action_high mismatch")
        require(env.get("spec_kwargs", {}).get("reward_type") == "sparse", f"{run_name}: {label} reward mismatch")
    task = snapshot.get("task_spec")
    require(isinstance(task, dict), f"{run_name}: missing task_spec")
    require(task.get("task") == "FetchPickAndPlace", f"{run_name}: wrong task_spec")
    require(
        task.get("phase_labels") == {"0": "approach", "1": "grasp", "2": "lift", "3": "place"},
        f"{run_name}: phase labels mismatch",
    )


def inspect_metric_row(
    row: Mapping[str, Any], kind: str, source: str, row_number: int, steps: int
) -> tuple[int, dict[str, bool]]:
    label = f"{source} row {row_number}"
    step = integer(row.get("step"), f"{label} step")
    require(0 < step <= steps, f"{label}: step outside run range")
    require(integer(row.get("episode"), f"{label} episode") == step // 50, f"{label}: episode mismatch")
    require(row.get("config/weight_scope") == "critic_only", f"{label}: wrong weight_scope")
    require(row.get("config/support_query_mode") == "replay_action", f"{label}: wrong query mode")
    for key in (*BASE_FINITE_FIELDS, *QUERY_FINITE_FIELDS):
        number(row.get(key), f"{label} {key}")
    for key in RATE_FIELDS:
        rate(row.get(key), f"{label} {key}")
    require(math.isclose(number(row.get("config/lambda_support"), f"{label} lambda"), 1.0), f"{label}: lambda drift")
    require(integer(row.get("train/replay_size"), f"{label} replay") == min(step, 300_000), f"{label}: replay size mismatch")

    eligible = rate(row.get("query/context_count_eligible_rate"), f"{label} context eligible")
    present = optional_rate(row.get("query/query_bin_present_given_context_rate"), f"{label} query present")
    if eligible > 0.0:
        require(present is not None, f"{label}: query-present diagnostic undefined despite eligible contexts")

    evidence = {
        "postwarmup_query_match": False,
        "numba_checked": False,
        "support_positive_seen": rate(row.get("query/support_positive_rate"), f"{label} support") > 0.0,
    }
    if step >= WARMUP_STEPS:
        replay_match = number(
            row.get("query/query_bin_matches_replay_action_rate"),
            f"{label} query/replay bin match",
        )
        require(replay_match == 1.0, f"{label}: replay-query bin match must be exactly 1.0")
        rate(row.get("query/query_bin_matches_current_policy_rate"), f"{label} query/policy bin match")
        normalized_mean = number(row.get("query/normalized_weight_mean"), f"{label} normalized mean")
        require(math.isclose(normalized_mean, 1.0, rel_tol=1e-5, abs_tol=1e-5), f"{label}: normalized weight mean != 1")
        enabled = number(row.get("runtime/numba_support_scan_enabled"), f"{label} numba enabled")
        warmed = number(row.get("runtime/numba_scanner_warmed"), f"{label} numba warmed")
        fallback = number(row.get("runtime/numba_support_scan_fallback_rate"), f"{label} numba fallback")
        exact_path = number(row.get("lite/runtime_opt_exact_path"), f"{label} exact path")
        require(enabled == 1.0 and warmed == 1.0, f"{label}: Numba scanner disabled/unwarmed")
        require(math.isclose(fallback, 0.0, abs_tol=1e-12), f"{label}: Numba fallback observed")
        require(exact_path == 1.0, f"{label}: runtime exact path disabled")
        evidence["postwarmup_query_match"] = True
        evidence["numba_checked"] = True

    if kind == "episode":
        require(integer(row.get("train/episode_len"), f"{label} episode length") == 50, f"{label}: incomplete episode")
    elif kind == "eval":
        for key in EVAL_FINITE_FIELDS:
            number(row.get(key), f"{label} {key}")
        for key in EVAL_RATE_FIELDS:
            rate(row.get(key), f"{label} {key}")
        episode_length = number(row.get("eval/episode_length_mean"), f"{label} eval length")
        require(0.0 < episode_length <= 50.0, f"{label}: eval length out of range")
    else:
        raise GateError(f"{label}: unknown metric type {kind!r}")
    return step, evidence


def _validate_metric_evidence(
    path: Path,
    episode_steps: list[int],
    eval_steps: list[int],
    flags: Mapping[str, bool],
    steps: int,
) -> dict[str, Any]:
    require(episode_steps == list(range(50, steps + 1, 50)), f"episode grid mismatch in {path}")
    expected_evals = expected_eval_steps(steps)
    require(eval_steps == expected_evals, f"eval grid mismatch in {path}: {eval_steps}")
    require(len(set(eval_steps)) == len(eval_steps), f"duplicate eval step in {path}")
    require(flags["postwarmup_query_match"], f"no post-warmup query-match evidence in {path}")
    require(flags["numba_checked"], f"no exact Numba evidence in {path}")
    return {
        "rows": len(episode_steps) + len(eval_steps),
        "episode_rows": len(episode_steps),
        "eval_steps": eval_steps,
        "eval_points": len(eval_steps),
        "query_match_replay_exact": True,
        "support_positive_seen": bool(flags["support_positive_seen"]),
        "numba_exact_path": True,
    }


def scan_csv(path: Path, steps: int) -> dict[str, Any]:
    episode_steps: list[int] = []
    eval_steps: list[int] = []
    flags = {"postwarmup_query_match": False, "numba_checked": False, "support_positive_seen": False}

    def checked_lines(handle: Iterable[str]) -> Iterable[str]:
        for line in handle:
            require("\x00" not in line, f"NUL byte in {path}")
            yield line

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(checked_lines(handle))
        require(reader.fieldnames is not None, f"missing CSV header: {path}")
        required = set(BASE_FINITE_FIELDS) | set(QUERY_FINITE_FIELDS) | set(EVAL_FINITE_FIELDS) | {
            "step",
            "episode",
            "config/weight_scope",
            "config/support_query_mode",
            "query/query_bin_present_given_context_rate",
            "query/query_bin_matches_current_policy_rate",
            "query/query_bin_matches_replay_action_rate",
            "runtime/numba_support_scan_enabled",
            "runtime/numba_scanner_warmed",
            "runtime/numba_support_scan_fallback_rate",
            "lite/runtime_opt_exact_path",
        }
        require(required <= set(reader.fieldnames), f"missing CSV columns: {sorted(required - set(reader.fieldnames))}")
        for index, row in enumerate(reader, start=2):
            require(None not in row, f"malformed CSV row {index}: {path}")
            kind = "eval" if row.get("eval/goal_success_mean") not in (None, "") else "episode"
            step, evidence = inspect_metric_row(row, kind, str(path), index, steps)
            (eval_steps if kind == "eval" else episode_steps).append(step)
            for key, value in evidence.items():
                flags[key] = flags[key] or value
    return _validate_metric_evidence(path, episode_steps, eval_steps, flags, steps)


def scan_jsonl(path: Path, steps: int) -> dict[str, Any]:
    episode_steps: list[int] = []
    eval_steps: list[int] = []
    flags = {"postwarmup_query_match": False, "numba_checked": False, "support_positive_seen": False}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            require("\x00" not in line and line.strip(), f"blank/NUL JSONL line {index}: {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid JSONL {path} line {index}: {exc}") from exc
            require(isinstance(row, dict), f"non-object JSONL row {index}: {path}")
            kind = row.get("type")
            require(kind in {"episode", "eval"}, f"unknown metric type {kind!r}: {path} line {index}")
            step, evidence = inspect_metric_row(row, str(kind), str(path), index, steps)
            (eval_steps if kind == "eval" else episode_steps).append(step)
            for key, value in evidence.items():
                flags[key] = flags[key] or value
    return _validate_metric_evidence(path, episode_steps, eval_steps, flags, steps)


def validate_summary(summary: Mapping[str, Any], run_dir: Path, steps: int) -> dict[str, Any]:
    label = run_dir.name
    updates = expected_updates(steps)
    exact = {
        "method": "b3_support_runtime_opt",
        "total_steps": steps,
        "episodes": steps // 50,
        "total_updates": updates,
        "final_replay_size": min(steps, 300_000),
        "selection_metric": "eval_goal_success_mean",
        "posthoc_correlation_metrics_online_computed": False,
    }
    for key, value in exact.items():
        require(summary.get(key) == value, f"{label}: summary {key!r} must be {value!r}")
    require(resolved(str(summary.get("run_dir", ""))) == run_dir, f"{label}: summary run_dir mismatch")
    runtime = number(summary.get("runtime_total_sec"), f"{label} runtime_total_sec")
    throughput = number(summary.get("final_steps_per_sec"), f"{label} final_steps_per_sec")
    require(runtime > 0.0 and throughput > 0.0, f"{label}: runtime/throughput must be positive")
    require(math.isclose(throughput, steps / runtime, rel_tol=1e-9, abs_tol=1e-9), f"{label}: throughput inconsistent")

    losses = summary.get("final_losses")
    require(isinstance(losses, dict), f"{label}: missing final_losses")
    for key in (
        "critic", "actor", "q_mean", "target_mean", "q_std", "target_std",
        "q_abs_mean", "q_abs_max", "td_error_abs_mean", "td_error_abs_p90",
    ):
        number(losses.get(key), f"{label} final_losses.{key}")

    lite = summary.get("final_lite_metrics")
    require(isinstance(lite, dict), f"{label}: missing final_lite_metrics")
    for key in QUERY_FINITE_FIELDS:
        number(lite.get(key), f"{label} final_lite_metrics.{key}")
    rate(lite.get("query/context_count_eligible_rate"), f"{label} final context eligible")
    rate(lite.get("query/support_positive_rate"), f"{label} final support positive")
    require(
        number(lite.get("query/query_bin_matches_replay_action_rate"), f"{label} final replay match") == 1.0,
        f"{label}: final replay-query bin match must be exactly 1",
    )
    rate(lite.get("query/query_bin_matches_current_policy_rate"), f"{label} final policy match")
    optional_rate(lite.get("query/query_bin_present_given_context_rate"), f"{label} final query present")
    require(
        math.isclose(number(lite.get("query/normalized_weight_mean"), f"{label} normalized mean"), 1.0, rel_tol=1e-5, abs_tol=1e-5),
        f"{label}: final normalized mean != 1",
    )
    for key in (
        "runtime/numba_support_scan_enabled",
        "runtime/numba_scanner_warmed",
        "runtime/numba_support_scan_fallback_rate",
        "lite/runtime_opt_exact_path",
    ):
        number(lite.get(key), f"{label} final_lite_metrics.{key}")
    require(lite["runtime/numba_support_scan_enabled"] == 1.0, f"{label}: Numba disabled")
    require(lite["runtime/numba_scanner_warmed"] == 1.0, f"{label}: Numba unwarmed")
    require(math.isclose(lite["runtime/numba_support_scan_fallback_rate"], 0.0, abs_tol=1e-12), f"{label}: Numba fallback")
    require(lite["lite/runtime_opt_exact_path"] == 1.0, f"{label}: exact path disabled")

    for key in (
        "best_eval_success", "best_eval_selection_success", "best_eval_goal_success",
        "best_eval_env_success", "final_eval_success", "final_eval_goal_success",
        "final_eval_env_success", "final_eval_goal_success_ever", "final_eval_env_success_ever",
        "final_eval_goal_success_final", "final_eval_env_success_final", "eval_AUC_success",
    ):
        rate(summary.get(key), f"{label} {key}")
    number(summary.get("best_eval_return"), f"{label} best_eval_return")
    number(summary.get("final_eval_return"), f"{label} final_eval_return")
    best_step = integer(summary.get("best_eval_step"), f"{label} best_eval_step")
    require(best_step in expected_eval_steps(steps), f"{label}: best step off eval grid")
    return {
        "updates": updates,
        "episodes": steps // 50,
        "best_eval_step": best_step,
        "best_eval_goal_success": float(summary["best_eval_goal_success"]),
        "final_eval_goal_success": float(summary["final_eval_goal_success"]),
        "eval_AUC_success": float(summary["eval_AUC_success"]),
    }


def validate_best_metadata(metadata: Mapping[str, Any], summary: Mapping[str, Any], run_dir: Path, steps: int) -> None:
    label = run_dir.name
    require(Path(str(metadata.get("checkpoint", ""))).name == "best_checkpoint.pt", f"{label}: best checkpoint path")
    best_step = integer(metadata.get("step"), f"{label} best metadata step")
    require(best_step in expected_eval_steps(steps), f"{label}: best metadata off grid")
    require(best_step == summary.get("best_eval_step"), f"{label}: best metadata/summary mismatch")
    require(metadata.get("selection_metric") == "eval_goal_success_mean", f"{label}: selection metric mismatch")
    for key in (
        "eval_success_mean", "eval_goal_success_mean", "eval_env_success_mean",
        "eval_goal_success_ever_mean", "eval_env_success_ever_mean",
        "eval_goal_success_final_mean", "eval_env_success_final_mean",
    ):
        rate(metadata.get(key), f"{label} best metadata {key}")
    number(metadata.get("eval_return_mean"), f"{label} best eval return")


def validate_checkpoint(path: Path, run_name: str, seed: int, steps: int, output_root: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise GateError("PyTorch is required to validate checkpoint.pt") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GateError(f"could not load checkpoint {path}: {exc}") from exc
    try:
        require(isinstance(payload, dict), f"{run_name}: checkpoint is not a mapping")
        require(payload.get("checkpoint_kind") == "periodic", f"{run_name}: wrong checkpoint kind")
        require(payload.get("global_step") == steps, f"{run_name}: checkpoint step mismatch")
        require(payload.get("episode_idx") == steps // 50, f"{run_name}: checkpoint episode mismatch")
        require(payload.get("total_updates") == expected_updates(steps), f"{run_name}: checkpoint update mismatch")
        cfg = payload.get("config")
        require(isinstance(cfg, dict), f"{run_name}: checkpoint missing config")
        require(cfg.get("run_name") == run_name, f"{run_name}: checkpoint run_name mismatch")
        require(cfg.get("support_query_mode") == "replay_action", f"{run_name}: checkpoint query mode mismatch")
        require(cfg.get("weight_scope") == "critic_only", f"{run_name}: checkpoint scope mismatch")
        require(cfg.get("total_steps") == steps, f"{run_name}: checkpoint total_steps mismatch")
        require(cfg.get("seeds", {}).get("env_seed") == seed, f"{run_name}: checkpoint seed mismatch")
        require(resolved(str(cfg.get("output_root", ""))) == output_root, f"{run_name}: checkpoint output_root mismatch")
        agent = payload.get("agent")
        require(isinstance(agent, dict), f"{run_name}: checkpoint missing agent")
        for key in ("actor", "actor_target", "critic", "critic_target", "actor_opt", "critic_opt"):
            require(key in agent, f"{run_name}: checkpoint missing agent.{key}")
        tensor_count = 0

        def finite_tensors(value: Any, label: str) -> None:
            nonlocal tensor_count
            if torch.is_tensor(value):
                if torch.is_floating_point(value) or torch.is_complex(value):
                    tensor_count += 1
                    require(bool(torch.isfinite(value).all().item()), f"{run_name}: non-finite tensor in {label}")
                return
            if isinstance(value, Mapping):
                for child_key, child in value.items():
                    finite_tensors(child, f"{label}.{child_key}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    finite_tensors(child, f"{label}[{index}]")

        finite_tensors(agent, "agent")
        require(tensor_count > 0, f"{run_name}: no floating agent tensors")
        replay = payload.get("replay")
        require(isinstance(replay, dict), f"{run_name}: checkpoint missing replay")
        require(replay.get("size") == min(steps, 300_000), f"{run_name}: checkpoint replay size")
        require(replay.get("capacity") == 300_000, f"{run_name}: checkpoint replay capacity")
        require(isinstance(payload.get("gap_stats"), dict), f"{run_name}: checkpoint missing gap_stats")
        for key in ("noise_rng_state", "sample_rng_state", "weight_rng_state"):
            require(key in payload, f"{run_name}: checkpoint missing {key}")
    finally:
        del payload
        gc.collect()


def validate_log(path: Path, run_dir: Path, steps: int) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    require(not FATAL_PATTERN.search(text), f"fatal pattern in {path}")
    require(f"step={steps}" in text, f"missing final step marker in {path}")
    finished = [line for line in text.splitlines() if "finished run at" in line]
    require(finished and run_dir.name in finished[-1], f"missing/wrong finished marker in {path}")


def validate_launcher_log(log_dir: Path, seed: int, run_dir: Path, steps: int) -> None:
    require(log_dir.is_dir(), f"launcher log directory missing: {log_dir}")
    candidates = sorted(path for path in log_dir.glob(f"seed_{seed}.log") if path.is_file())
    require(len(candidates) == 1, f"expected one launcher log for seed {seed}, got {candidates}")
    validate_log(candidates[0], run_dir, steps)


def validate_run(
    output_root: Path,
    other_root: Path,
    package: Mapping[str, Any],
    seed: int,
    steps: int,
    launcher_log_dir: Path | None,
) -> dict[str, Any]:
    run_name = f"{RUN_PREFIX}_s{seed}_{steps}_ls{int(package['local_search_max'])}"
    run_dir = (output_root / run_name).resolve()
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    siblings = sorted(path.resolve() for path in output_root.glob(f"{run_name}*") if path.is_dir())
    require(siblings == [run_dir], f"duplicate/timestamped run directories for {run_name}: {siblings}")
    require(not (other_root / run_name).exists(), f"{run_name} leaked into other phase root")
    for name in REQUIRED_FILES:
        path = run_dir / name
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty required file: {path}")

    checkpoint_steps = list(range(50_000, steps + 1, 50_000))
    if not checkpoint_steps or checkpoint_steps[-1] != steps:
        checkpoint_steps.append(steps)
    expected_step_files = sorted(f"checkpoint_step_{step}.pt" for step in checkpoint_steps)
    actual_step_files = sorted(path.name for path in run_dir.glob("checkpoint_step_*.pt") if path.is_file())
    require(actual_step_files == expected_step_files, f"{run_name}: checkpoint step grid mismatch")
    for name in expected_step_files:
        require((run_dir / name).stat().st_size > 0, f"empty step checkpoint: {run_dir / name}")

    cfg = read_object(run_dir / "config.json")
    snapshot = read_object(run_dir / "env_snapshot.json")
    summary = read_object(run_dir / "summary.json")
    best = read_object(run_dir / "best_checkpoint.json")
    validate_run_config(cfg, output_root, run_name, seed, steps)
    validate_env_snapshot(snapshot, run_name)
    summary_report = validate_summary(summary, run_dir, steps)
    validate_best_metadata(best, summary, run_dir, steps)
    csv_report = scan_csv(run_dir / "metrics.csv", steps)
    jsonl_report = scan_jsonl(run_dir / "metrics.jsonl", steps)
    require(csv_report["eval_steps"] == jsonl_report["eval_steps"], f"{run_name}: CSV/JSONL eval mismatch")
    validate_log(run_dir / "run.log", run_dir, steps)
    if launcher_log_dir is not None:
        validate_launcher_log(launcher_log_dir, seed, run_dir, steps)
    validate_checkpoint(run_dir / "checkpoint.pt", run_name, seed, steps, output_root)
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "steps": steps,
        **summary_report,
        "eval_points": csv_report["eval_points"],
        "metric_rows": csv_report["rows"],
        "query_match_replay_exact": True,
        "support_positive_seen": csv_report["support_positive_seen"],
        "numba_exact_path": True,
    }


def write_selected_10k(run_dir: Path, steps: int) -> Path:
    source = run_dir / "metrics.csv"
    target = run_dir / "metrics_selected_10k.csv"
    temporary = target.with_suffix(".csv.tmp")
    selected = 0
    with source.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        require(reader.fieldnames is not None, f"missing CSV header: {source}")
        with temporary.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("eval/goal_success_mean") not in (None, ""):
                    writer.writerow(row)
                    selected += 1
    require(selected == len(expected_eval_steps(steps)), f"selected eval count mismatch for {run_dir}")
    temporary.replace(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict FPP DDPG SABER replay-query completion gate")
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--launcher-log-dir", "--log-dir", dest="launcher_log_dir", type=Path)
    parser.add_argument("--write-selected-10k", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from validate_config_parity import validate_config, validate_source

        validate_config()
        validate_source()
        phase = infer_phase(args.steps, args.phase)
        package = read_object(EXPERIMENT_DIR / "config.json")
        validate_package_contract(package)
        require(args.seeds, "--seeds must not be empty")
        require(len(args.seeds) == len(set(args.seeds)), "--seeds contains duplicates")
        require(all(seed in package["seeds"] for seed in args.seeds), f"seeds must be subset of {package['seeds']}")
        output_root = resolved(args.output_root)
        validate_output_root(output_root, phase)
        require(output_root.is_dir(), f"output root does not exist: {output_root}")
        other_phase = "main" if phase == "smoke" else "smoke"
        other_root = resolved(PHASES[other_phase][1])
        launcher_log_dir = resolved(args.launcher_log_dir) if args.launcher_log_dir else None
        runs = [
            validate_run(output_root, other_root, package, seed, args.steps, launcher_log_dir)
            for seed in args.seeds
        ]
        selected: list[str] = []
        if args.write_selected_10k:
            selected = [str(write_selected_10k(Path(run["run_dir"]), args.steps)) for run in runs]
        print(
            json.dumps(
                {
                    "ok": True,
                    "phase": phase,
                    "steps": args.steps,
                    "output_root": str(output_root),
                    "seeds": args.seeds,
                    "runs": runs,
                    "selected_10k_csv": selected,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"[gate-fail] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
