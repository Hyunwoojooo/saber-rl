#!/usr/bin/env python3
"""Strict completion gate for the FPP DDPG+HER+SABER actor-only arm."""

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
RUN_PREFIX = "fetch_pick_and_place"
MAIN_STEPS = 1_000_000
SMOKE_STEPS = 20_000
MAIN_ROOT = Path("experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512")
SMOKE_ROOT = Path("experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512_smoke20k")
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
RATE_FIELDS = (
    "train/episode_success",
    "train/her_relabel_ratio",
    "lite/support_available_rate",
    "lite/psi_q_rank_rate",
    "lite/q_candidate_fallback_rate",
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
    """A result failed a registered experiment invariant."""


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


def integer(value: Any, label: str) -> int:
    parsed = number(value, label)
    require(parsed.is_integer(), f"{label} must be an integer: {value!r}")
    return int(parsed)


def rate(value: Any, label: str) -> float:
    parsed = number(value, label)
    require(0.0 <= parsed <= 1.0, f"{label} must be in [0, 1]: {parsed}")
    return parsed


def expected_eval_steps(steps: int) -> list[int]:
    values = list(range(10_000, steps + 1, 10_000))
    if not values or values[-1] != steps:
        values.append(steps)
    return values


def expected_updates(steps: int) -> int:
    return max(0, steps - max(5_000, 256) + 1)


def infer_phase(steps: int, requested: str | None) -> str:
    matches = [name for name, (registered, _root) in PHASES.items() if steps == registered]
    require(matches, f"steps must be exactly {SMOKE_STEPS} (smoke) or {MAIN_STEPS} (main)")
    inferred = matches[0]
    require(requested in (None, inferred), f"--phase {requested!r} conflicts with --steps {steps}")
    return inferred


def validate_package_contract(package: Mapping[str, Any]) -> None:
    expected = {
        "trainer_method": "b3_support_runtime_opt",
        "weight_scope": "actor_only",
        "run_slug": "ddpg_saber_actor_only_v4_ls512",
        "output_root": MAIN_ROOT.as_posix(),
        "total_steps": MAIN_STEPS,
        "local_search_max": 512,
        "default_local_search_max": 512,
        "lambda_support": 1.0,
        "weighting_mode": "batch_norm",
    }
    for key, value in expected.items():
        require(package.get(key) == value, f"package config {key!r} must equal {value!r}")
    require(package.get("seeds") == [1, 2, 3, 4, 5], "package seeds must be exactly [1,2,3,4,5]")


def validate_output_root(output_root: Path, phase: str) -> None:
    registered = resolved(PHASES[phase][1])
    require(output_root == registered, f"{phase} output root must be {registered}, got {output_root}")
    require(resolved(MAIN_ROOT) != resolved(SMOKE_ROOT), "main and smoke roots unexpectedly alias")


def validate_run_config(
    cfg: Mapping[str, Any],
    output_root: Path,
    run_name: str,
    seed: int,
    steps: int,
) -> None:
    exact = {
        "run_name": run_name,
        "method": "b3_support_runtime_opt",
        "task_name": "fetch_pick_and_place",
        "reward_type": "sparse",
        "env_id": "FetchPickAndPlace-v4",
        "resolved_env_id": "FetchPickAndPlace-v4",
        "max_episode_steps": 50,
        "total_steps": steps,
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
        "weight_scope": "actor_only",
        "policy_support_mode": "bin",
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
    }
    for key, value in exact.items():
        require(cfg.get(key) == value, f"{run_name}: config {key!r} must equal {value!r}, got {cfg.get(key)!r}")
    require(resolved(str(cfg.get("output_root", ""))) == output_root, f"{run_name}: config output_root mismatch")
    seeds = cfg.get("seeds")
    require(isinstance(seeds, dict), f"{run_name}: missing seed mapping")
    expected_seeds = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    require(seeds == expected_seeds, f"{run_name}: seed mapping mismatch: {seeds!r}")
    require(
        cfg.get("terminal_convention")
        == {"success_terminates": False, "timeout_bootstrap": True},
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
    row: Mapping[str, Any],
    kind: str,
    source: str,
    row_number: int,
    steps: int,
) -> tuple[int, dict[str, bool]]:
    label = f"{source} row {row_number}"
    step = integer(row.get("step"), f"{label} step")
    require(0 < step <= steps, f"{label}: step outside run range")
    require(integer(row.get("episode"), f"{label} episode") == step // 50, f"{label}: episode mapping mismatch")
    require(row.get("config/weight_scope") == "actor_only", f"{label}: wrong weight_scope")
    for key in BASE_FINITE_FIELDS:
        number(row.get(key), f"{label} {key}")
    for key in RATE_FIELDS:
        rate(row.get(key), f"{label} {key}")
    require(math.isclose(number(row.get("config/lambda_support"), f"{label} lambda"), 1.0), f"{label}: lambda mismatch")
    require(
        integer(row.get("train/replay_size"), f"{label} replay")
        == min(step, 300_000),
        f"{label}: replay size mismatch",
    )
    evidence = {
        "std_positive": number(row.get("lite/std_w_P"), f"{label} std_w_P") > 0.0,
        "support_positive": rate(row.get("lite/support_available_rate"), f"{label} support rate") > 0.0,
        "mean_weight_above_one": number(row.get("lite/mean_w_P"), f"{label} mean_w_P") > 1.0,
        "numba_checked": False,
    }
    if step >= 5_000:
        enabled = number(row.get("runtime/numba_support_scan_enabled"), f"{label} numba enabled")
        warmed = number(row.get("runtime/numba_scanner_warmed"), f"{label} numba warmed")
        fallback = number(row.get("runtime/numba_support_scan_fallback_rate"), f"{label} numba fallback")
        exact_path = number(row.get("lite/runtime_opt_exact_path"), f"{label} exact path")
        require(math.isclose(enabled, 1.0), f"{label}: Numba support scan disabled")
        require(math.isclose(warmed, 1.0), f"{label}: Numba scanner not warmed")
        require(math.isclose(fallback, 0.0, abs_tol=1e-12), f"{label}: Numba fallback observed")
        require(math.isclose(exact_path, 1.0), f"{label}: runtime exact path disabled")
        evidence["numba_checked"] = True
    if kind == "episode":
        require(integer(row.get("train/episode_len"), f"{label} episode length") == 50, f"{label}: incomplete episode row")
    elif kind == "eval":
        for key in EVAL_FINITE_FIELDS:
            number(row.get(key), f"{label} {key}")
        for key in EVAL_RATE_FIELDS:
            rate(row.get(key), f"{label} {key}")
        episode_length = number(row.get("eval/episode_length_mean"), f"{label} eval length")
        require(0.0 < episode_length <= 50.0, f"{label}: eval episode length out of range")
    else:
        raise GateError(f"{label}: unknown metric type {kind!r}")
    return step, evidence


def scan_csv(path: Path, steps: int) -> dict[str, Any]:
    episode_steps: list[int] = []
    eval_steps: list[int] = []
    flags = {"std_positive": False, "support_positive": False, "mean_weight_above_one": False, "numba_checked": False}

    def checked_lines(handle: Iterable[str]) -> Iterable[str]:
        for line in handle:
            require("\x00" not in line, f"NUL byte in {path}")
            yield line

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(checked_lines(handle))
        require(reader.fieldnames is not None, f"missing CSV header: {path}")
        required = set(BASE_FINITE_FIELDS) | set(EVAL_FINITE_FIELDS) | {
            "step", "episode", "config/weight_scope",
            "runtime/numba_support_scan_enabled", "runtime/numba_scanner_warmed",
            "runtime/numba_support_scan_fallback_rate", "lite/runtime_opt_exact_path",
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
    flags = {"std_positive": False, "support_positive": False, "mean_weight_above_one": False, "numba_checked": False}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            require("\x00" not in line, f"NUL byte in {path} line {index}")
            require(line.strip(), f"blank JSONL line in {path} line {index}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid JSONL in {path} line {index}: {exc}") from exc
            require(isinstance(row, dict), f"non-object JSONL row in {path} line {index}")
            kind = row.get("type")
            require(kind in {"episode", "eval"}, f"unknown metric type in {path} line {index}: {kind!r}")
            step, evidence = inspect_metric_row(row, str(kind), str(path), index, steps)
            (eval_steps if kind == "eval" else episode_steps).append(step)
            for key, value in evidence.items():
                flags[key] = flags[key] or value
    return _validate_metric_evidence(path, episode_steps, eval_steps, flags, steps)


def _validate_metric_evidence(
    path: Path,
    episode_steps: list[int],
    eval_steps: list[int],
    flags: Mapping[str, bool],
    steps: int,
) -> dict[str, Any]:
    expected_episodes = list(range(50, steps + 1, 50))
    require(episode_steps == expected_episodes, f"episode grid mismatch in {path}")
    expected_evals = expected_eval_steps(steps)
    require(eval_steps == expected_evals, f"eval grid mismatch in {path}: {eval_steps}")
    require(flags["numba_checked"], f"no post-warmup Numba evidence in {path}")
    require(flags["std_positive"], f"support weights never varied in {path}")
    require(flags["support_positive"], f"support was never available in {path}")
    require(flags["mean_weight_above_one"], f"support weights never activated in {path}")
    return {
        "rows": len(episode_steps) + len(eval_steps),
        "episode_rows": len(episode_steps),
        "eval_steps": eval_steps,
        "support_activated": True,
        "numba_exact_path": True,
    }


def validate_summary(
    summary: Mapping[str, Any],
    run_dir: Path,
    steps: int,
) -> dict[str, Any]:
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
        require(summary.get(key) == value, f"{label}: summary {key!r} must equal {value!r}")
    require(resolved(str(summary.get("run_dir", ""))) == run_dir, f"{label}: summary run_dir mismatch")
    runtime = number(summary.get("runtime_total_sec"), f"{label} runtime_total_sec")
    throughput = number(summary.get("final_steps_per_sec"), f"{label} final_steps_per_sec")
    require(runtime > 0.0 and throughput > 0.0, f"{label}: runtime/throughput must be positive")
    require(
        math.isclose(throughput, steps / runtime, rel_tol=1e-9, abs_tol=1e-9),
        f"{label}: final_steps_per_sec is inconsistent with runtime",
    )
    losses = summary.get("final_losses")
    require(isinstance(losses, dict), f"{label}: missing final_losses")
    for key in (
        "critic", "actor", "q_mean", "target_mean", "q_std", "target_std",
        "q_abs_mean", "q_abs_max", "td_error_abs_mean", "td_error_abs_p90",
    ):
        number(losses.get(key), f"{label} final_losses.{key}")
    final_lite = summary.get("final_lite_metrics")
    require(isinstance(final_lite, dict), f"{label}: missing final_lite_metrics")
    for key in (
        "lite/mean_w_P",
        "lite/std_w_P",
        "lite/support_available_rate",
        "runtime/numba_support_scan_enabled",
        "runtime/numba_scanner_warmed",
        "runtime/numba_support_scan_fallback_rate",
        "lite/runtime_opt_exact_path",
    ):
        number(final_lite.get(key), f"{label} final_lite_metrics.{key}")
    require(
        math.isclose(float(final_lite["runtime/numba_support_scan_enabled"]), 1.0)
        and math.isclose(float(final_lite["runtime/numba_scanner_warmed"]), 1.0)
        and math.isclose(float(final_lite["runtime/numba_support_scan_fallback_rate"]), 0.0, abs_tol=1e-12)
        and math.isclose(float(final_lite["lite/runtime_opt_exact_path"]), 1.0),
        f"{label}: final runtime optimization evidence mismatch",
    )
    for key in (
        "best_eval_success",
        "best_eval_selection_success",
        "best_eval_goal_success",
        "best_eval_env_success",
        "final_eval_success",
        "final_eval_goal_success",
        "final_eval_env_success",
        "final_eval_goal_success_ever",
        "final_eval_env_success_ever",
        "final_eval_goal_success_final",
        "final_eval_env_success_final",
        "eval_AUC_success",
    ):
        rate(summary.get(key), f"{label} {key}")
    number(summary.get("best_eval_return"), f"{label} best_eval_return")
    number(summary.get("final_eval_return"), f"{label} final_eval_return")
    best_step = integer(summary.get("best_eval_step"), f"{label} best_eval_step")
    require(best_step in expected_eval_steps(steps), f"{label}: best step is off eval grid")
    return {
        "updates": updates,
        "episodes": steps // 50,
        "best_eval_step": best_step,
        "best_eval_goal_success": float(summary["best_eval_goal_success"]),
        "final_eval_goal_success": float(summary["final_eval_goal_success"]),
        "eval_AUC_success": float(summary["eval_AUC_success"]),
    }


def validate_best_metadata(
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_dir: Path,
    steps: int,
) -> None:
    label = run_dir.name
    checkpoint = Path(str(metadata.get("checkpoint", "")))
    require(checkpoint.name == "best_checkpoint.pt", f"{label}: best checkpoint path mismatch")
    best_step = integer(metadata.get("step"), f"{label} best metadata step")
    require(best_step in expected_eval_steps(steps), f"{label}: best metadata step is off eval grid")
    require(best_step == summary.get("best_eval_step"), f"{label}: best metadata/summary step mismatch")
    require(metadata.get("selection_metric") == "eval_goal_success_mean", f"{label}: best selection metric mismatch")
    for key in (
        "eval_success_mean",
        "eval_goal_success_mean",
        "eval_env_success_mean",
        "eval_goal_success_ever_mean",
        "eval_env_success_ever_mean",
        "eval_goal_success_final_mean",
        "eval_env_success_final_mean",
    ):
        rate(metadata.get(key), f"{label} best metadata {key}")
    number(metadata.get("eval_return_mean"), f"{label} best metadata eval_return_mean")
    require(
        math.isclose(float(metadata["eval_goal_success_mean"]), float(summary["best_eval_goal_success"])),
        f"{label}: best goal-success metadata mismatch",
    )


def validate_checkpoint(
    path: Path,
    run_name: str,
    seed: int,
    steps: int,
    output_root: Path,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise GateError("PyTorch is required to validate checkpoint.pt") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GateError(f"could not load checkpoint {path}: {exc}") from exc
    try:
        require(isinstance(payload, dict), f"{run_name}: checkpoint payload is not a mapping")
        require(payload.get("checkpoint_kind") == "periodic", f"{run_name}: wrong checkpoint kind")
        require(payload.get("global_step") == steps, f"{run_name}: checkpoint step mismatch")
        require(payload.get("episode_idx") == steps // 50, f"{run_name}: checkpoint episode mismatch")
        require(payload.get("total_updates") == expected_updates(steps), f"{run_name}: checkpoint updates mismatch")
        cfg = payload.get("config")
        require(isinstance(cfg, dict), f"{run_name}: checkpoint missing config")
        require(cfg.get("run_name") == run_name, f"{run_name}: checkpoint run_name mismatch")
        require(cfg.get("method") == "b3_support_runtime_opt", f"{run_name}: checkpoint method mismatch")
        require(cfg.get("weight_scope") == "actor_only", f"{run_name}: checkpoint scope mismatch")
        require(cfg.get("total_steps") == steps, f"{run_name}: checkpoint total_steps mismatch")
        require(cfg.get("seeds", {}).get("env_seed") == seed, f"{run_name}: checkpoint seed mismatch")
        require(resolved(str(cfg.get("output_root", ""))) == output_root, f"{run_name}: checkpoint output_root mismatch")
        agent = payload.get("agent")
        require(isinstance(agent, dict), f"{run_name}: checkpoint missing agent")
        for key in ("actor", "actor_target", "critic", "critic_target", "actor_opt", "critic_opt"):
            require(key in agent, f"{run_name}: checkpoint missing agent.{key}")
        tensor_count = 0

        def check_finite_tensors(value: Any, label: str) -> None:
            nonlocal tensor_count
            if torch.is_tensor(value):
                if torch.is_floating_point(value) or torch.is_complex(value):
                    tensor_count += 1
                    require(
                        bool(torch.isfinite(value).all().item()),
                        f"{run_name}: non-finite tensor in {label}",
                    )
                return
            if isinstance(value, Mapping):
                for child_key, child in value.items():
                    check_finite_tensors(child, f"{label}.{child_key}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    check_finite_tensors(child, f"{label}[{index}]")

        check_finite_tensors(agent, "agent")
        require(tensor_count > 0, f"{run_name}: no floating agent/optimizer tensors found")
        replay = payload.get("replay")
        require(isinstance(replay, dict), f"{run_name}: checkpoint missing replay")
        require(replay.get("size") == min(steps, 300_000), f"{run_name}: checkpoint replay size mismatch")
        require(replay.get("capacity") == 300_000, f"{run_name}: checkpoint replay capacity mismatch")
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
    finished_lines = [line for line in text.splitlines() if "finished run at" in line]
    require(finished_lines, f"missing finished marker in {path}")
    require(run_dir.name in finished_lines[-1], f"finished marker names wrong run in {path}")


def validate_launcher_log(log_dir: Path, seed: int, run_dir: Path, steps: int) -> None:
    require(log_dir.is_dir(), f"launcher log directory does not exist: {log_dir}")
    seed_pattern = re.compile(rf"(?<!\d)(?:seed|s)[_-]?{seed}(?!\d)", re.IGNORECASE)
    candidates = sorted(path for path in log_dir.glob("*.log") if seed_pattern.search(path.stem))
    require(len(candidates) == 1, f"expected one launcher log for seed {seed}, found {candidates}")
    validate_log(candidates[0], run_dir, steps)


def validate_run(
    output_root: Path,
    other_root: Path,
    package: Mapping[str, Any],
    seed: int,
    steps: int,
    launcher_log_dir: Path | None,
) -> dict[str, Any]:
    run_name = (
        f"{RUN_PREFIX}_{package['run_slug']}_s{seed}_{steps}_"
        f"ls{int(package['local_search_max'])}"
    )
    run_dir = (output_root / run_name).resolve()
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    siblings = sorted(path.resolve() for path in output_root.glob(f"{run_name}*") if path.is_dir())
    require(siblings == [run_dir], f"duplicate/timestamped run directories for {run_name}: {siblings}")
    require(not (other_root / run_name).exists(), f"{run_name} leaked into the other phase root")
    for name in REQUIRED_FILES:
        path = run_dir / name
        require(path.is_file(), f"missing required file: {path}")
        require(path.stat().st_size > 0, f"empty required file: {path}")
    checkpoint_steps = list(range(50_000, steps + 1, 50_000))
    if not checkpoint_steps or checkpoint_steps[-1] != steps:
        checkpoint_steps.append(steps)
    expected_step_files = [f"checkpoint_step_{step}.pt" for step in checkpoint_steps]
    actual_step_files = sorted(
        path.name for path in run_dir.glob("checkpoint_step_*.pt") if path.is_file()
    )
    require(
        actual_step_files == sorted(expected_step_files),
        f"{run_name}: checkpoint step grid mismatch: {actual_step_files}",
    )
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
        "eval_points": len(csv_report["eval_steps"]),
        "metric_rows": csv_report["rows"],
        "support_activated": True,
        "numba_exact_path": True,
    }


def write_selected_10k(run_dir: Path) -> Path:
    source = run_dir / "metrics.csv"
    target = run_dir / "metrics_selected_10k.csv"
    temporary = target.with_suffix(".csv.tmp")
    with source.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        require(reader.fieldnames is not None, f"missing CSV header: {source}")
        with temporary.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("eval/goal_success_mean") not in (None, ""):
                    writer.writerow(row)
    temporary.replace(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict FPP DDPG SABER actor-only completion gate")
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--launcher-log-dir", "--log-dir", dest="launcher_log_dir", type=Path)
    parser.add_argument(
        "--write-selected-10k",
        action="store_true",
        help="After every gate passes, write per-run metrics_selected_10k.csv files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        phase = infer_phase(args.steps, args.phase)
        package = read_object(EXPERIMENT_DIR / "config.json")
        validate_package_contract(package)
        registered_seeds = package["seeds"]
        require(args.seeds, "--seeds must not be empty")
        require(len(args.seeds) == len(set(args.seeds)), "--seeds contains duplicates")
        require(all(seed in registered_seeds for seed in args.seeds), f"seeds must be a subset of {registered_seeds}")
        output_root = resolved(args.output_root)
        validate_output_root(output_root, phase)
        require(output_root.is_dir(), f"output root does not exist: {output_root}")
        other_phase = "main" if phase == "smoke" else "smoke"
        other_root = resolved(PHASES[other_phase][1])
        launcher_log_dir = resolved(args.launcher_log_dir) if args.launcher_log_dir else None
        runs = [
            validate_run(
                output_root,
                other_root,
                package,
                seed,
                args.steps,
                launcher_log_dir,
            )
            for seed in args.seeds
        ]
        selected: list[str] = []
        if args.write_selected_10k:
            selected = [str(write_selected_10k(Path(run["run_dir"]))) for run in runs]
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
