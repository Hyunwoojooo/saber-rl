#!/usr/bin/env python3
"""Fail closed unless a five-seed 1M SAC arm completed cleanly."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
METHODS = {
    "sac_er": "sac_er",
    "sac_annealing_er": "sac_annealing_er",
    "sac_aer": "sac_aer",
}
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
STEPS = 1_000_000
EVAL_GRID = list(range(10_000, STEPS + 1, 10_000))
ERROR_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|RuntimeError:|AssertionError:|"
    r"FloatingPointError:|Segmentation fault|^\s*Killed\s*$|"
    r"^\s*(?:ERROR\b|\[failed\]|\[worker-failed\]|\[block-failed\])",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_seed_spec(value: str) -> tuple[int, ...]:
    seeds = tuple(int(token) for token in value.replace(",", " ").split())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a nonempty unique list")
    if any(seed not in range(1, 11) for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be within 1..10")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(METHODS), required=True)
    parser.add_argument("--seeds", type=parse_seed_spec, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiment_FPP/runs/fpp_sac_aer_noher_1m",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "experiment_FPP/logs/fpp_sac_aer_noher_1m",
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


def require_rate(value: Any, label: str) -> float:
    result = require_finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label}: expected rate in [0, 1], got {result!r}")
    return result


def scan_log(path: Path) -> str:
    text = require_file(path).read_text(encoding="utf-8", errors="replace")
    match = ERROR_PATTERN.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        raise ValueError(f"fatal marker in {path}:{line}: {match.group(0)!r}")
    return text


def validate_env_snapshot(path: Path, arm: str, seed: int) -> None:
    snapshot = read_json(path)
    for env_name in ("train_env", "eval_env"):
        env = snapshot.get(env_name)
        if not isinstance(env, dict):
            raise ValueError(f"{arm}/seed{seed}: missing {env_name} snapshot")
        expected = {
            "env_id": "FetchPickAndPlace-v4",
            "obs_dim": 25,
            "goal_dim": 3,
            "state_dim": 28,
            "action_dim": 4,
            "max_episode_steps": 50,
        }
        for key, value in expected.items():
            require_equal(env.get(key), value, f"{arm}/seed{seed} {env_name}.{key}")
        require_equal(
            env.get("spec_kwargs", {}).get("reward_type"),
            "sparse",
            f"{arm}/seed{seed} {env_name} reward type",
        )


def validate_metrics(path: Path, arm: str, seed: int) -> None:
    eval_steps: list[int] = []
    final_records: list[dict[str, Any]] = []
    episode_count = 0
    with require_file(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            row_type = row.get("type")
            if row_type == "episode":
                episode_count += 1
            elif row_type == "eval":
                eval_steps.append(int(row["step"]))
                require_rate(
                    row.get("eval/success_mean"),
                    f"{arm}/seed{seed} eval success at step {row.get('step')}",
                )
                alpha = require_finite(
                    row.get("sac/alpha"),
                    f"{arm}/seed{seed} SAC alpha at step {row.get('step')}",
                )
                if alpha <= 0.0:
                    raise ValueError(f"{arm}/seed{seed}: non-positive SAC alpha {alpha}")
                require_finite(
                    row.get("sac/alpha_loss"),
                    f"{arm}/seed{seed} alpha loss at step {row.get('step')}",
                )
                require_finite(
                    row.get("sac/log_prob_mean"),
                    f"{arm}/seed{seed} log prob at step {row.get('step')}",
                )
                require_equal(
                    float(row.get("aer/effective_temperature_lr")),
                    3e-4,
                    f"{arm}/seed{seed} temperature LR at step {row.get('step')}",
                )
            elif row_type == "final_fixed_eval":
                final_records.append(row)
            if "train/her_relabel_ratio" in row:
                require_equal(
                    float(row["train/her_relabel_ratio"]),
                    0.0,
                    f"{arm}/seed{seed} HER ratio at JSONL line {line_number}",
                )

    require_equal(episode_count, 20_000, f"{arm}/seed{seed} episode record count")
    require_equal(eval_steps, EVAL_GRID, f"{arm}/seed{seed} evaluation grid")
    require_equal(len(final_records), 1, f"{arm}/seed{seed} final evaluation count")
    final_record = final_records[0]
    require_equal(final_record.get("step"), STEPS, f"{arm}/seed{seed} final JSONL step")
    require_equal(final_record.get("episodes"), 100, f"{arm}/seed{seed} final JSONL episodes")
    require_equal(
        final_record.get("base_seed"),
        100_010_000 + seed,
        f"{arm}/seed{seed} final JSONL seed",
    )


def validate_checkpoint(path: Path, arm: str, seed: int) -> None:
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
        "episode_idx": 20_000,
        "episode_step": 0,
        "total_updates": 995_001,
    }
    for key, value in expected.items():
        require_equal(payload.get(key), value, f"{arm}/seed{seed} checkpoint.{key}")
    agent_state = payload.get("agent")
    if not isinstance(agent_state, dict):
        raise ValueError(f"{arm}/seed{seed}: checkpoint agent state is missing")
    required_sac_keys = {
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
    missing_sac_keys = required_sac_keys.difference(agent_state)
    if missing_sac_keys:
        raise ValueError(
            f"{arm}/seed{seed}: checkpoint is not a complete twin-critic SAC agent; "
            f"missing {sorted(missing_sac_keys)}"
        )
    forbidden_non_sac_keys = {
        "actor_target",
        "critic",
        "critic_target",
        "policy_delay",
        "target_policy_noise",
        "target_noise_clip",
    }
    present_non_sac_keys = forbidden_non_sac_keys.intersection(agent_state)
    if present_non_sac_keys:
        raise ValueError(
            f"{arm}/seed{seed}: checkpoint unexpectedly contains non-SAC state "
            f"{sorted(present_non_sac_keys)}"
        )
    require_equal(agent_state.get("auto_alpha"), True, f"{arm}/seed{seed} auto alpha")
    require_equal(agent_state.get("target_entropy"), -4.0, f"{arm}/seed{seed} target entropy")
    require_equal(agent_state.get("fixed_alpha"), 0.2, f"{arm}/seed{seed} initial alpha")
    require_finite(agent_state.get("log_alpha"), f"{arm}/seed{seed} log alpha")
    for optimizer_name in ("actor_opt", "critic_opt", "alpha_opt"):
        optimizer = agent_state.get(optimizer_name)
        if (
            not isinstance(optimizer, dict)
            or not optimizer.get("param_groups")
            or not optimizer.get("state")
        ):
            raise ValueError(f"{arm}/seed{seed}: malformed or empty {optimizer_name}")
        optimizer_lr = require_finite(
            optimizer["param_groups"][0].get("lr"),
            f"{arm}/seed{seed} {optimizer_name} LR",
        )
        if not math.isclose(optimizer_lr, 3e-4, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{arm}/seed{seed}: final {optimizer_name} LR "
                f"{optimizer_lr} != 0.0003"
            )
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
    if str(checkpoint_config.get("device", "")).startswith("cuda") and not payload.get(
        "torch_cuda_rng_state_all"
    ):
        raise ValueError(f"{arm}/seed{seed}: CUDA checkpoint has no CUDA RNG state")
    require_equal(checkpoint_config.get("method"), METHODS[arm], f"{arm}/seed{seed} checkpoint method")
    require_equal(checkpoint_config.get("total_steps"), STEPS, f"{arm}/seed{seed} checkpoint steps")
    require_equal(
        checkpoint_config.get("seeds", {}).get("env_seed"),
        seed,
        f"{arm}/seed{seed} checkpoint seed",
    )
    eval_history = payload.get("eval_history_env")
    if not isinstance(eval_history, list):
        raise ValueError(f"{arm}/seed{seed}: checkpoint eval history is missing")
    require_equal(
        [int(item[0]) for item in eval_history],
        EVAL_GRID,
        f"{arm}/seed{seed} checkpoint evaluation grid",
    )


def validate_run(output_root: Path, log_dir: Path, arm: str, seed: int) -> None:
    method = METHODS[arm]
    run_name = f"fetch_pick_and_place_{arm}_noher_s{seed}_{STEPS}"
    run_dir = output_root / run_name
    if not run_dir.is_dir():
        raise ValueError(f"missing run directory: {run_dir}")

    config = read_json(run_dir / "config.json")
    expected_config = {
        "run_name": run_name,
        "method": method,
        "task_name": "fetch_pick_and_place",
        "env_id": "FetchPickAndPlace-v4",
        "resolved_env_id": "FetchPickAndPlace-v4",
        "reward_type": "sparse",
        "total_steps": STEPS,
        "max_episode_steps": 50,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
        "updates_per_step": 1,
        "batch_size": 256,
        "buffer_capacity": 300_000,
        "her_ratio": 0.0,
        "gamma": 0.98,
        "tau": 0.005,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "sac_alpha": 0.2,
        "sac_auto_alpha": True,
        "sac_alpha_lr": 3e-4,
        "sac_target_entropy": None,
        "hidden_dim": 256,
        "noise_std": 0.2,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "final_eval_episodes": 100,
        "checkpoint_every_steps": 50_000,
        "success_terminates": False,
        "lambda_support": 0.0,
        "weight_scope": "none",
        "support_cache_enabled": False,
        "local_scan_cache_enabled": False,
        "support_short_circuit": False,
        "numba_support_scan": False,
        "support_predicate_early_exit": False,
        "aer_lambda0": 2.5,
        "aer_alpha": 2.0 / 3.0,
        "aer_similarity": "cosine",
        "aer_similarity_eps": 1e-8,
        "aer_candidate_rounding": "ceil",
        "smoke": False,
        "resume": False,
    }
    for key, expected in expected_config.items():
        require_equal(config.get(key), expected, f"{arm}/seed{seed} config.{key}")
    for key in (
        "td3_policy_delay",
        "td3_target_policy_noise",
        "td3_target_noise_clip",
    ):
        if key in config:
            raise ValueError(f"{arm}/seed{seed}: SAC config contains inactive TD3 field {key}")
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
    require_equal(summary.get("total_steps"), STEPS, f"{arm}/seed{seed} total steps")
    require_equal(summary.get("episodes"), 20_000, f"{arm}/seed{seed} episodes")
    require_equal(summary.get("total_updates"), 995_001, f"{arm}/seed{seed} updates")
    require_equal(summary.get("final_replay_size"), 300_000, f"{arm}/seed{seed} replay size")
    require_equal(summary.get("fixed_final_eval_policy"), "final", f"{arm}/seed{seed} final policy")
    require_equal(summary.get("fixed_final_eval_episodes"), 100, f"{arm}/seed{seed} fixed eval episodes")
    require_equal(
        summary.get("fixed_final_eval_seed"),
        100_010_000 + seed,
        f"{arm}/seed{seed} fixed eval seed",
    )
    if require_finite(summary.get("runtime_total_sec"), f"{arm}/seed{seed} runtime") <= 0:
        raise ValueError(f"{arm}/seed{seed}: non-positive runtime")
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
        "eval_AUC_success",
    ):
        value = summary.get(key)
        if "success" in key:
            require_rate(value, f"{arm}/seed{seed} summary.{key}")
        else:
            require_finite(value, f"{arm}/seed{seed} summary.{key}")
    if arm == "sac_aer":
        final_aer = summary.get("final_lite_metrics")
        if not isinstance(final_aer, dict):
            raise ValueError(f"{arm}/seed{seed}: missing final AER metrics")
        expected_aer = {
            "aer/enabled": 1.0,
            "aer/lambda": 1.0,
            "aer/candidate_count": 256.0,
            "aer/selected_count": 256.0,
            "aer/effective_actor_lr": 3e-4,
            "aer/effective_critic_lr": 3e-4,
            "aer/effective_temperature_lr": 3e-4,
        }
        for key, expected in expected_aer.items():
            actual = require_finite(final_aer.get(key), f"{arm}/seed{seed} {key}")
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"{arm}/seed{seed} {key}: expected {expected!r}, got {actual!r}"
                )

    final_eval = read_json(run_dir / "final_evaluation_100.json")
    require_equal(final_eval.get("policy"), "final", f"{arm}/seed{seed} final policy file")
    require_equal(final_eval.get("step"), STEPS, f"{arm}/seed{seed} final step")
    require_equal(final_eval.get("episodes"), 100, f"{arm}/seed{seed} final episodes")
    require_equal(
        final_eval.get("base_seed"),
        100_010_000 + seed,
        f"{arm}/seed{seed} final base seed",
    )
    require_finite(final_eval.get("eval_return_mean"), f"{arm}/seed{seed} final return")
    require_rate(final_eval.get("eval_goal_success_mean"), f"{arm}/seed{seed} final goal success")
    require_rate(final_eval.get("eval_env_success_mean"), f"{arm}/seed{seed} final env success")

    validate_env_snapshot(run_dir / "env_snapshot.json", arm, seed)
    validate_metrics(run_dir / "metrics.jsonl", arm, seed)
    for filename in ("metrics.csv", "best_checkpoint.pt", "best_checkpoint.json"):
        require_file(run_dir / filename)
    validate_checkpoint(run_dir / "checkpoint.pt", arm, seed)

    run_log = scan_log(run_dir / "run.log")
    outer_log = scan_log(log_dir / f"{arm}_seed{seed}.log")
    for text, label in ((run_log, "run.log"), (outer_log, "launcher log")):
        if "step=1000000" not in text:
            raise ValueError(f"{arm}/seed{seed} {label}: missing step=1000000 marker")
        if "fixed final evaluation episodes=100" not in text:
            raise ValueError(f"{arm}/seed{seed} {label}: missing fixed final evaluation marker")
        if f"finished run at {run_dir}" not in text:
            raise ValueError(f"{arm}/seed{seed} {label}: missing exact finished marker")


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    completed = 0
    for seed in args.seeds:
        try:
            validate_run(args.output_root, args.log_dir, args.arm, seed)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
        else:
            completed += 1
            print(f"[gate-ok] arm={args.arm} seed={seed} steps={STEPS}")

    if failures:
        for failure in failures:
            print(f"[gate-fail] {failure}")
        raise SystemExit(
            f"main arm gate failed: {len(failures)} run(s) invalid; successor will not start"
        )

    print(f"[gate-pass] arm={args.arm} completed cleanly for {completed} seed(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
