#!/usr/bin/env python3
"""Correctness/smoke/production gate; never interprets short-run performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

import run as experiment_run


QUERY_NUMERIC_FIELDS = (
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/normalized_weight_mean",
)
QUERY_MODE_FIELD = "config/support_query_mode"
FINITE_TRAIN_FIELDS = (
    "train/replay_size",
    "train/her_relabel_ratio",
    "loss/critic",
    "loss/actor",
    "loss/td_error_abs_mean",
    "loss/td_error_abs_p90",
    "value/q_mean",
    "value/target_mean",
)
FINITE_EVAL_FIELDS = (
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/episode_length_mean",
)
CHECKPOINT_TOP_LEVEL_RNG_FIELDS = (
    # The generic checkpoint payload serializes assignment state even though
    # support assignment does not consume it in replay-query mode.
    "assignment_rng_state",
    "rng_state_schema_version",
    "python_random_state",
    "numpy_global_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state_all",
    "noise_rng_state",
    "sample_rng_state",
    "weight_rng_state",
)
REPLAY_RESUME_AUDIT_RNG_FIELDS = (
    # This list must exactly match trainer.replay_rng_fields. The unused
    # assignment RNG is deliberately absent from the replay-mode audit contract.
    "python_random_state",
    "numpy_global_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state_all",
    "noise_rng_state",
    "sample_rng_state",
    "weight_rng_state",
)


def _number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        raise RuntimeError(f"missing metric value {key!r}")
    value = float(raw)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric {key!r}: {raw!r}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_eval_grid(mode: experiment_run.Mode) -> list[int]:
    grid = list(range(mode.eval_freq, mode.steps + 1, mode.eval_freq))
    if mode.steps > 0 and (not grid or grid[-1] != mode.steps):
        grid.append(mode.steps)
    return grid


def _validate_config(config: dict[str, Any], mode: experiment_run.Mode, seed: int) -> None:
    expected = {
        "method": "b3_support_runtime_opt",
        "weight_scope": "critic_only",
        "support_query_mode": "replay_action",
        "support_assignment_mode": "support",
        "weighting_mode": "batch_norm",
        "task_name": "drawer_open",
        "reward_type": "sparse",
        "metaworld_env_name": "drawer-open-v3",
        "metaworld_reward_function_version": "v2",
        "total_steps": mode.steps,
        "start_steps": mode.start_steps,
        "warmup_steps": mode.warmup_steps,
        "eval_freq": mode.eval_freq,
        "eval_episodes": mode.eval_episodes,
        "checkpoint_every_steps": mode.checkpoint_every_steps,
        "save_step_checkpoints": False,
        "max_episode_steps": 200,
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
        "success_terminates": False,
        "timeout_bootstrap": True,
        "M_B": 16,
        "M_Q_min": 4,
        "N_local_min": 4,
        "N_gap_min": 128,
        "local_search_max": 512,
        "action_bins_per_dim": 5,
        "lambda_support": 1.0,
        "policy_support_mode": "bin",
        "support_scan_mode": "deterministic_tail_v1",
        "support_cache_enabled": True,
        "support_cache_max_entries": 200_000,
        "local_scan_cache_enabled": True,
        "local_scan_cache_max_entries": 50_000,
        "support_short_circuit": True,
        "numba_support_scan": True,
        "support_predicate_early_exit": True,
        "support_diagnostic_audit_interval": 100,
        "drawer_success_threshold": 0.05,
        "drawer_d_reach": 0.08,
        "drawer_d_goal_near": 0.05,
        "drawer_d_opening_mid": 0.12,
    }
    failures = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    expected_seeds = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "assignment_seed": 30_001,
        "eval_seed": 10_000 + seed,
    }
    seeds = config.get("seeds", {})
    for name, value in expected_seeds.items():
        actual = seeds.get(name) if isinstance(seeds, dict) else None
        if actual != value:
            failures[f"seeds.{name}"] = {"expected": value, "actual": actual}
    if failures:
        raise RuntimeError(f"config failures: {failures}")


def _validate_checkpoint(
    run_dir: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    mode: experiment_run.Mode,
    expected_updates: int,
    expected_replay: int,
    expected_grid: list[int],
    eval_success: list[float],
) -> dict[str, Any]:
    required = (
        "checkpoint_schema_version",
        "checkpoint_metadata",
        "config",
        "config_fingerprint",
        "global_step",
        "total_updates",
        "episode_state",
        "agent",
        "replay",
        "resume_audit",
        "eval_history_env",
        *CHECKPOINT_TOP_LEVEL_RNG_FIELDS,
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"{run_dir}: checkpoint fields missing {missing}")
    if int(checkpoint["global_step"]) != mode.steps:
        raise RuntimeError(f"{run_dir}: checkpoint global step mismatch")
    if int(checkpoint["total_updates"]) != expected_updates:
        raise RuntimeError(f"{run_dir}: checkpoint update count mismatch")
    checkpoint_config = checkpoint["config"]
    if not isinstance(checkpoint_config, dict) or checkpoint_config != config:
        raise RuntimeError(f"{run_dir}: checkpoint config differs from config.json")
    if checkpoint_config.get("support_query_mode") != "replay_action":
        raise RuntimeError(f"{run_dir}: checkpoint lost replay_action query mode")
    if checkpoint_config.get("support_assignment_mode") != "support":
        raise RuntimeError(f"{run_dir}: checkpoint enabled non-support assignment")

    replay = checkpoint["replay"]
    if not isinstance(replay, dict) or int(replay.get("size", -1)) != expected_replay:
        raise RuntimeError(f"{run_dir}: checkpoint replay size mismatch")
    agent = checkpoint["agent"]
    if not isinstance(agent, dict):
        raise RuntimeError(f"{run_dir}: checkpoint agent state is not a mapping")
    expected_agent = {
        "actor", "actor_target", "critic", "critic_target",
        "actor_opt", "critic_opt", "total_it",
    }
    missing_agent = sorted(expected_agent - set(agent))
    twin_keys = sorted(
        key for key in agent if key.startswith(("critic1", "critic2", "qf1", "qf2"))
    )
    if missing_agent or twin_keys:
        raise RuntimeError(
            f"{run_dir}: checkpoint is not canonical single-critic DDPG "
            f"missing={missing_agent} twin_keys={twin_keys}"
        )
    if int(agent.get("total_it", -1)) != expected_updates:
        raise RuntimeError(f"{run_dir}: DDPG update counter mismatch")

    history = list(checkpoint["eval_history_env"])
    history_steps = [int(item[0]) for item in history]
    history_values = [float(item[1]) for item in history]
    if history_steps != expected_grid:
        raise RuntimeError(f"{run_dir}: checkpoint evaluation grid mismatch")
    if len(history_values) != len(eval_success) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(history_values, eval_success)
    ):
        raise RuntimeError(f"{run_dir}: checkpoint evaluation history differs from metrics")

    resume_audit = checkpoint["resume_audit"]
    if not isinstance(resume_audit, dict):
        raise RuntimeError(f"{run_dir}: malformed checkpoint resume audit")
    if resume_audit.get("config_fingerprint") != checkpoint.get("config_fingerprint"):
        raise RuntimeError(f"{run_dir}: checkpoint config fingerprint mismatch")
    recorded_rng = resume_audit.get("rng_state_fields")
    if recorded_rng != list(REPLAY_RESUME_AUDIT_RNG_FIELDS):
        raise RuntimeError(
            f"{run_dir}: replay resume-audit RNG contract mismatch "
            f"actual={recorded_rng} expected={list(REPLAY_RESUME_AUDIT_RNG_FIELDS)}"
        )
    metadata = checkpoint["checkpoint_metadata"]
    if not isinstance(metadata, dict) or bool(metadata.get("strict_control", True)):
        raise RuntimeError(f"{run_dir}: checkpoint unexpectedly uses strict control")
    if int(metadata.get("global_step", -1)) != mode.steps:
        raise RuntimeError(f"{run_dir}: checkpoint metadata step mismatch")

    return {
        "checkpoint_schema_version": int(checkpoint["checkpoint_schema_version"]),
        "single_critic_ddpg": True,
        "checkpoint_top_level_rng_fields_present": True,
        "replay_resume_audit_rng_contract_matches": True,
        "checkpoint_config_matches": True,
    }


def validate_run(run_dir: Path, mode: experiment_run.Mode, seed: int) -> dict[str, Any]:
    required_files = (
        "config.json",
        "env_snapshot.json",
        "metrics.csv",
        "summary.json",
        "checkpoint.pt",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"{run_dir}: missing required files {missing}")
    step_checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if step_checkpoints:
        raise RuntimeError(f"{run_dir}: step checkpoints must be disabled")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    _validate_config(config, mode, seed)
    if int(summary.get("total_steps", -1)) != mode.steps:
        raise RuntimeError(f"{run_dir}: incomplete summary total_steps")
    expected_updates = mode.steps - mode.warmup_steps + 1
    if int(summary.get("total_updates", -1)) != expected_updates:
        raise RuntimeError(f"{run_dir}: total_updates mismatch")
    expected_replay = min(mode.steps, 300_000)
    if int(summary.get("final_replay_size", -1)) != expected_replay:
        raise RuntimeError(f"{run_dir}: final replay size mismatch")
    if summary.get("support_query_mode") != "replay_action":
        raise RuntimeError(f"{run_dir}: summary support_query_mode mismatch")
    if summary.get("support_assignment_mode") != "support":
        raise RuntimeError(f"{run_dir}: summary support_assignment_mode mismatch")
    if int(summary.get("assignment_seed", -1)) != 30_001:
        raise RuntimeError(f"{run_dir}: assignment seed is not canonical default")

    rows = _load_csv(run_dir / "metrics.csv")
    if not rows:
        raise RuntimeError(f"{run_dir}: empty metrics.csv")
    required_headers = (
        QUERY_MODE_FIELD,
        *QUERY_NUMERIC_FIELDS,
        *FINITE_TRAIN_FIELDS,
        *FINITE_EVAL_FIELDS,
    )
    missing_headers = [field for field in required_headers if field not in rows[0]]
    if missing_headers:
        raise RuntimeError(f"{run_dir}: missing metric columns {missing_headers}")

    checked_query_rows = 0
    for row in rows:
        step = int(float(row.get("step", -1)))
        if step < mode.warmup_steps:
            continue
        if row.get(QUERY_MODE_FIELD) != "replay_action":
            raise RuntimeError(f"{run_dir}: metrics row lost replay_action query mode")
        values = {field: _number(row, field) for field in QUERY_NUMERIC_FIELDS}
        for field in FINITE_TRAIN_FIELDS:
            _number(row, field)
        if not math.isclose(
            values["query/query_bin_matches_replay_action_rate"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{run_dir}: replay-action query-bin match is not one")
        for field in (
            "query/query_bin_matches_current_policy_rate",
            "query/context_count_eligible_rate",
            "query/support_positive_rate",
        ):
            if not 0.0 <= values[field] <= 1.0:
                raise RuntimeError(f"{run_dir}: invalid rate {field}={values[field]}")
        high_count = values["query/high_weight_count"]
        if not 0.0 <= high_count <= 256.0 or not high_count.is_integer():
            raise RuntimeError(f"{run_dir}: invalid high-weight count {high_count}")
        if not 1.0 <= values["query/raw_weight_mean"] <= 2.0:
            raise RuntimeError(f"{run_dir}: raw weight mean outside [1,2]")
        if not math.isclose(
            values["query/normalized_weight_mean"], 1.0,
            rel_tol=0.0, abs_tol=2e-6,
        ):
            raise RuntimeError(f"{run_dir}: normalized critic weight mean is not one")
        checked_query_rows += 1
    if checked_query_rows == 0:
        raise RuntimeError(f"{run_dir}: no finite post-warmup query rows were checked")

    eval_rows = [row for row in rows if row.get("eval/env_success_mean", "") != ""]
    expected_grid = _expected_eval_grid(mode)
    actual_grid = [int(float(row["step"])) for row in eval_rows]
    if actual_grid != expected_grid:
        raise RuntimeError(
            f"{run_dir}: evaluation grid mismatch actual={actual_grid} expected={expected_grid}"
        )
    eval_success: list[float] = []
    for row in eval_rows:
        for field in FINITE_EVAL_FIELDS:
            _number(row, field)
        value = _number(row, "eval/env_success_mean")
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{run_dir}: invalid evaluation success {value}")
        eval_success.append(value)

    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False,
    )
    checkpoint_report = _validate_checkpoint(
        run_dir, checkpoint, config, mode, expected_updates,
        expected_replay, expected_grid, eval_success,
    )
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "steps": mode.steps,
        "updates": expected_updates,
        "replay_size": expected_replay,
        "eval_points": len(eval_rows),
        "query_metric_rows_checked": checked_query_rows,
        "replay_query_match_rate_exact_one": True,
        "support_assignment_mode": "support",
        "weight_scope": "critic_only",
        **checkpoint_report,
        "passed": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(experiment_run.MODES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = experiment_run.MODES[args.mode]
    seeds = tuple(args.seeds) if args.seeds else mode.seeds
    reports = [
        validate_run(mode.output_root / experiment_run.run_name(mode, seed), mode, seed)
        for seed in seeds
    ]
    report = {"mode": args.mode, "seeds": reports, "passed": True}
    output = args.output or experiment_run.EXPERIMENT_DIR / "reports" / f"{args.mode}_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
