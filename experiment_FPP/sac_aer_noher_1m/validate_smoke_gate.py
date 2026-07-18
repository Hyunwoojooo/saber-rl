#!/usr/bin/env python3
"""Fail closed unless every seed-1..5 20k smoke triplet completed cleanly."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
ARMS = {
    "sac_er": "sac_er",
    "sac_annealing_er": "sac_annealing_er",
    "sac_aer": "sac_aer",
}
SEEDS = (1, 2, 3, 4, 5)
STEPS = 20_000
ERROR_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:|"
    r"^\s*(?:ERROR\b|\[failed\]|\[worker-failed\]|\[block-failed\])",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "experiment_FPP/runs/fpp_sac_aer_noher_smoke20k_s1_5",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT
        / "experiment_FPP/logs/fpp_sac_aer_noher_smoke20k_s1_5",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty required file: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(require_file(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def require_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return result


def scan_log(path: Path) -> str:
    text = require_file(path).read_text(encoding="utf-8", errors="replace")
    match = ERROR_PATTERN.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        raise ValueError(f"error marker in {path}:{line}: {match.group(0)!r}")
    return text


def validate_metrics(path: Path, seed: int, arm: str) -> None:
    eval_steps: list[int] = []
    eval_rows: dict[int, dict[str, Any]] = {}
    final_records: list[dict[str, Any]] = []
    with require_file(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            if row.get("type") == "eval":
                step = int(row["step"])
                eval_steps.append(step)
                eval_rows[step] = row
            elif row.get("type") == "final_fixed_eval":
                final_records.append(row)
            if "train/her_relabel_ratio" in row:
                require_equal(
                    float(row["train/her_relabel_ratio"]),
                    0.0,
                    f"{arm}/seed{seed} HER ratio at JSONL line {line_number}",
                )

    require_equal(eval_steps, [10_000, 20_000], f"{arm}/seed{seed} eval grid")
    require_equal(len(final_records), 1, f"{arm}/seed{seed} final eval count")
    final_record = final_records[0]
    require_equal(final_record.get("step"), STEPS, f"{arm}/seed{seed} final JSONL step")
    require_equal(final_record.get("episodes"), 100, f"{arm}/seed{seed} final JSONL episodes")
    for step, expected_lambda in ((10_000, 1.375), (20_000, 1.0)):
        row = eval_rows[step]
        alpha = require_finite(row.get("sac/alpha"), f"{arm}/seed{seed} SAC alpha at {step}")
        if alpha <= 0.0:
            raise ValueError(f"{arm}/seed{seed}: non-positive SAC alpha at {step}: {alpha}")
        require_finite(row.get("sac/alpha_loss"), f"{arm}/seed{seed} alpha loss at {step}")
        require_finite(row.get("sac/log_prob_mean"), f"{arm}/seed{seed} log prob at {step}")
        expected_schedule_lambda = expected_lambda if arm != "sac_er" else 1.0
        actual_lambda = require_finite(
            row.get("aer/lambda"), f"{arm}/seed{seed} lambda at {step}"
        )
        if not math.isclose(
            actual_lambda, expected_schedule_lambda, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"{arm}/seed{seed} lambda at {step}: "
                f"{actual_lambda} != {expected_schedule_lambda}"
            )
        expected_lr = 3e-4 * expected_schedule_lambda
        for key in ("aer/effective_actor_lr", "aer/effective_critic_lr"):
            actual = require_finite(row.get(key), f"{arm}/seed{seed} {key} at {step}")
            if not math.isclose(actual, expected_lr, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{arm}/seed{seed} {key} at {step}: {actual} != {expected_lr}")
        temperature_lr = require_finite(
            row.get("aer/effective_temperature_lr"),
            f"{arm}/seed{seed} fixed temperature LR at {step}",
        )
        if not math.isclose(temperature_lr, 3e-4, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{arm}/seed{seed} fixed temperature LR at {step}: "
                f"{temperature_lr} != 0.0003"
            )
        require_equal(
            float(row.get("aer/enabled")),
            1.0 if arm == "sac_aer" else 0.0,
            f"{arm}/seed{seed} attentive replay flag at {step}",
        )
        require_equal(
            float(row.get("aer/selected_count")),
            256.0,
            f"{arm}/seed{seed} selected count at {step}",
        )
        expected_candidates = 352.0 if arm == "sac_aer" and step == 10_000 else 256.0
        require_equal(
            float(row.get("aer/candidate_count")),
            expected_candidates,
            f"{arm}/seed{seed} candidate count at {step}",
        )


def validate_checkpoint(path: Path, seed: int, arm: str) -> None:
    try:
        import torch

        payload = torch.load(require_file(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{arm}/seed{seed}: checkpoint payload is not a dict")

    expected = {
        "checkpoint_kind": "periodic",
        "global_step": STEPS,
        "episode_idx": 400,
        "episode_step": 0,
        "total_updates": 15_001,
    }
    for key, value in expected.items():
        require_equal(payload.get(key), value, f"{arm}/seed{seed} checkpoint.{key}")

    agent = payload.get("agent")
    if not isinstance(agent, dict):
        raise ValueError(f"{arm}/seed{seed}: checkpoint agent state is missing")
    required = {
        "actor",
        "critic1",
        "critic2",
        "critic1_target",
        "critic2_target",
        "actor_opt",
        "critic_opt",
        "alpha_opt",
        "log_alpha",
        "fixed_alpha",
        "auto_alpha",
        "target_entropy",
        "total_it",
    }
    missing = required.difference(agent)
    if missing:
        raise ValueError(f"{arm}/seed{seed}: incomplete SAC checkpoint: {sorted(missing)}")
    if {"actor_target", "critic", "critic_target"}.intersection(agent):
        raise ValueError(f"{arm}/seed{seed}: checkpoint contains non-SAC network state")
    require_equal(agent.get("auto_alpha"), True, f"{arm}/seed{seed} auto alpha")
    require_equal(agent.get("target_entropy"), -4.0, f"{arm}/seed{seed} target entropy")
    require_finite(agent.get("log_alpha"), f"{arm}/seed{seed} log alpha")
    for optimizer_name in ("actor_opt", "critic_opt", "alpha_opt"):
        optimizer = agent.get(optimizer_name)
        if (
            not isinstance(optimizer, dict)
            or not optimizer.get("param_groups")
            or not optimizer.get("state")
        ):
            raise ValueError(f"{arm}/seed{seed}: malformed or empty {optimizer_name}")
        lr = require_finite(
            optimizer["param_groups"][0].get("lr"),
            f"{arm}/seed{seed} {optimizer_name} LR",
        )
        if not math.isclose(lr, 3e-4, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{arm}/seed{seed}: {optimizer_name} LR {lr} != 0.0003")

    for rng_key in (
        "noise_rng_state",
        "sample_rng_state",
        "weight_rng_state",
        "torch_rng_state",
        "torch_cuda_rng_state_all",
    ):
        if rng_key not in payload:
            raise ValueError(f"{arm}/seed{seed}: checkpoint missing {rng_key}")
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"{arm}/seed{seed}: checkpoint config is missing")
    require_equal(checkpoint_config.get("method"), ARMS[arm], f"{arm}/seed{seed} method")
    require_equal(checkpoint_config.get("total_steps"), STEPS, f"{arm}/seed{seed} steps")
    if str(checkpoint_config.get("device", "")).startswith("cuda") and not payload.get(
        "torch_cuda_rng_state_all"
    ):
        raise ValueError(f"{arm}/seed{seed}: CUDA checkpoint has no CUDA RNG state")


def validate_run(output_root: Path, log_dir: Path, arm: str, seed: int) -> None:
    method = ARMS[arm]
    run_name = f"fetch_pick_and_place_{arm}_noher_s{seed}_{STEPS}"
    run_dir = output_root / run_name
    if not run_dir.is_dir():
        raise ValueError(f"missing run directory: {run_dir}")

    config = read_json(run_dir / "config.json")
    expected_config = {
        "method": method,
        "task_name": "fetch_pick_and_place",
        "env_id": "FetchPickAndPlace-v4",
        "reward_type": "sparse",
        "total_steps": STEPS,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
        "updates_per_step": 1,
        "batch_size": 256,
        "buffer_capacity": 300_000,
        "her_ratio": 0.0,
        "sac_alpha": 0.2,
        "sac_auto_alpha": True,
        "sac_alpha_lr": 3e-4,
        "sac_target_entropy": None,
        "max_episode_steps": 50,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "final_eval_episodes": 100,
        "success_terminates": False,
        "lambda_support": 0.0,
        "weight_scope": "none",
        "support_cache_enabled": False,
        "local_scan_cache_enabled": False,
        "support_short_circuit": False,
        "numba_support_scan": False,
        "support_predicate_early_exit": False,
        "smoke": False,
        "resume": False,
    }
    for key, expected in expected_config.items():
        require_equal(config.get(key), expected, f"{arm}/seed{seed} config.{key}")
    require_equal(
        config.get("terminal_convention"),
        {"success_terminates": False, "timeout_bootstrap": True},
        f"{arm}/seed{seed} terminal convention",
    )
    require_equal(
        config.get("seeds"),
        {
            "env_seed": seed,
            "torch_seed": seed,
            "numpy_seed": seed,
            "action_noise_seed": seed,
            "weight_seed": 20_000 + seed,
            "eval_seed": 10_000 + seed,
        },
        f"{arm}/seed{seed} seed mapping",
    )

    summary = read_json(run_dir / "summary.json")
    require_equal(summary.get("method"), method, f"{arm}/seed{seed} summary.method")
    require_equal(summary.get("total_steps"), STEPS, f"{arm}/seed{seed} summary.total_steps")
    require_equal(summary.get("final_replay_size"), STEPS, f"{arm}/seed{seed} replay size")
    require_equal(summary.get("fixed_final_eval_episodes"), 100, f"{arm}/seed{seed} fixed eval episodes")
    require_equal(
        summary.get("fixed_final_eval_seed"),
        100_000_000 + 10_000 + seed,
        f"{arm}/seed{seed} fixed eval seed",
    )
    require_equal(summary.get("total_updates"), 15_001, f"{arm}/seed{seed} update count")
    require_finite(summary.get("runtime_total_sec"), f"{arm}/seed{seed} runtime")
    if require_finite(summary.get("final_steps_per_sec"), f"{arm}/seed{seed} throughput") <= 0:
        raise ValueError(f"{arm}/seed{seed}: non-positive throughput")
    final_losses = summary.get("final_losses")
    if not isinstance(final_losses, dict) or not {"actor", "critic"}.issubset(final_losses):
        raise ValueError(f"{arm}/seed{seed}: incomplete final losses")
    for name, value in final_losses.items():
        require_finite(value, f"{arm}/seed{seed} final loss {name}")
    for key in (
        "fixed_final_eval_return",
        "fixed_final_eval_goal_success",
        "fixed_final_eval_env_success",
    ):
        require_finite(summary.get(key), f"{arm}/seed{seed} summary.{key}")

    final_eval = read_json(run_dir / "final_evaluation_100.json")
    require_equal(final_eval.get("policy"), "final", f"{arm}/seed{seed} final policy")
    require_equal(final_eval.get("step"), STEPS, f"{arm}/seed{seed} final step")
    require_equal(final_eval.get("episodes"), 100, f"{arm}/seed{seed} final episodes")
    require_equal(
        final_eval.get("base_seed"),
        100_000_000 + 10_000 + seed,
        f"{arm}/seed{seed} final base seed",
    )
    for key in ("eval_return_mean", "eval_goal_success_mean", "eval_env_success_mean"):
        require_finite(final_eval.get(key), f"{arm}/seed{seed} final evaluation {key}")

    validate_metrics(run_dir / "metrics.jsonl", seed, arm)
    require_file(run_dir / "metrics.csv")
    validate_checkpoint(run_dir / "checkpoint.pt", seed, arm)

    run_log = scan_log(run_dir / "run.log")
    outer_log = scan_log(log_dir / f"{arm}_seed{seed}.log")
    for text, label in ((run_log, "run.log"), (outer_log, "launcher log")):
        if "step=20000" not in text:
            raise ValueError(f"{arm}/seed{seed} {label}: missing step=20000 marker")
        if "fixed final evaluation episodes=100" not in text:
            raise ValueError(f"{arm}/seed{seed} {label}: missing fixed final evaluation marker")
        if "finished run at" not in text:
            raise ValueError(f"{arm}/seed{seed} {label}: missing finished marker")


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    completed = 0
    for seed in SEEDS:
        for arm in ARMS:
            try:
                validate_run(args.output_root, args.log_dir, arm, seed)
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(str(exc))
            else:
                completed += 1
                print(f"[gate-ok] arm={arm} seed={seed} steps={STEPS}")

    if failures:
        for failure in failures:
            print(f"[gate-fail] {failure}")
        raise SystemExit(
            f"smoke gate failed: {len(failures)} run(s) invalid; main experiment will not start"
        )

    print(f"[gate-pass] validated {completed} completed smoke runs; safe to start main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
