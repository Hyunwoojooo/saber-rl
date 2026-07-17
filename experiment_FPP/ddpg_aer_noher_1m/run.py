#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARM_NAMES = ("ddpg_er", "ddpg_annealing_er", "ddpg_aer")


def find_repo_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "version_v0_lite").is_dir() and (parent / ".venv").is_dir():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


def load_config() -> dict[str, Any]:
    return json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def phase_for_run(
    config: dict[str, Any], steps: int, seed: int
) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, phase)
        for name, phase in config["phases"].items()
        if int(phase["steps"]) == steps
        and seed in [int(value) for value in phase["seeds"]]
    ]
    if len(matches) != 1:
        allowed = sorted(
            (
                name,
                int(phase["steps"]),
                [int(value) for value in phase["seeds"]],
            )
            for name, phase in config["phases"].items()
        )
        raise ValueError(
            f"No unique frozen phase for steps={steps}, seed={seed}; "
            f"allowed phases={allowed}"
        )
    return matches[0]


def validate_protocol(
    config: dict[str, Any], arm_name: str, seed: int, steps: int
) -> tuple[str, dict[str, Any]]:
    if arm_name not in ARM_NAMES or arm_name not in config["arms"]:
        raise ValueError(f"Unknown arm: {arm_name}")

    phase_name, phase = phase_for_run(config, steps, seed)
    allowed_seeds = [int(value) for value in phase["seeds"]]
    if seed not in allowed_seeds:
        raise ValueError(
            f"Seed {seed} is outside frozen phase {phase_name}: {allowed_seeds}"
        )

    common = config["common"]
    expected_common = {
        "task_name": "fetch_pick_and_place",
        "env_id": "FetchPickAndPlace-v4",
        "reward_type": "sparse",
        "max_episode_steps": 50,
        "total_steps": 1_000_000,
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
        "hidden_dim": 256,
        "noise_std": 0.2,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "final_eval_episodes": 100,
        "checkpoint_every_steps": 50_000,
        "save_step_checkpoints": False,
        "success_terminates": False,
        "timeout_bootstrap": True,
    }
    actual_common = {key: common.get(key) for key in expected_common}
    if actual_common != expected_common:
        raise ValueError(
            "Frozen common FPP/DDPG protocol changed: "
            f"expected={expected_common}, actual={actual_common}"
        )

    expected_arms = {
        "ddpg_er": ("ddpg_er", "uniform", "fixed"),
        "ddpg_annealing_er": (
            "ddpg_annealing_er",
            "uniform",
            "aer_lambda_scaled",
        ),
        "ddpg_aer": (
            "ddpg_aer",
            "attentive_experience_replay",
            "aer_lambda_scaled",
        ),
    }
    arm = config["arms"][arm_name]
    actual_arm = (
        arm.get("method"),
        arm.get("replay_sampling"),
        arm.get("learning_rate_schedule"),
    )
    if actual_arm != expected_arms[arm_name]:
        raise ValueError(
            f"Frozen {arm_name} arm changed: "
            f"expected={expected_arms[arm_name]}, actual={actual_arm}"
        )

    disabled_support = (
        float(arm.get("lambda_support", -1.0)) == 0.0
        and arm.get("weight_scope") == "none"
        and not bool(arm.get("support_lookup"))
        and not bool(arm.get("support_cache_enabled"))
        and not bool(arm.get("local_scan_cache_enabled"))
        and not bool(arm.get("support_short_circuit"))
        and not bool(arm.get("numba_support_scan"))
        and not bool(arm.get("support_predicate_early_exit"))
        and int(arm.get("support_diagnostic_audit_interval", -1)) == 0
    )
    if not disabled_support:
        raise ValueError(f"SABER support path must stay disabled for {arm_name}")

    aer = config["aer"]
    expected_aer = {
        "lambda0": 2.5,
        "alpha": 2.0 / 3.0,
        "similarity": "cosine",
        "similarity_eps": 1e-8,
        "state_representation": "observation_plus_original_desired_goal",
        "candidate_rounding": "ceil",
        "importance_sampling_correction": False,
    }
    actual_aer = {key: aer.get(key) for key in expected_aer}
    if actual_aer != expected_aer:
        raise ValueError(
            f"Frozen AER protocol changed: expected={expected_aer}, actual={actual_aer}"
        )
    return phase_name, phase


def run_name(config: dict[str, Any], arm_name: str, seed: int, steps: int) -> str:
    return (
        f"fetch_pick_and_place_{config['arms'][arm_name]['run_slug']}_"
        f"s{seed}_{steps}"
    )


def build_command(
    arm_name: str,
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    validate_only: bool,
    resume: bool = False,
    resume_path: str | None = None,
) -> list[str]:
    config = load_config()
    common = config["common"]
    aer = config["aer"]
    arm = config["arms"][arm_name]
    n_steps = int(common["total_steps"] if steps is None else steps)
    _phase_name, phase = validate_protocol(config, arm_name, seed, n_steps)

    repo_root = find_repo_root(EXPERIMENT_DIR)
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    trainer = EXPERIMENT_DIR / "code" / "src" / "train_fetch_pick_and_place_lite_v0.py"
    resolved_output = (
        Path(output_root) if output_root else repo_root / str(phase["output_root"])
    )

    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", str(common["task_name"]),
        "--run_name", run_name(config, arm_name, seed, n_steps),
        "--output_root", str(resolved_output),
        "--method", str(arm["method"]),
        "--reward_type", str(common["reward_type"]),
        "--max_episode_steps", str(common["max_episode_steps"]),
        "--total_steps", str(n_steps),
        "--start_steps", str(common["start_steps"]),
        "--warmup_steps", str(common["warmup_steps"]),
        "--updates_per_step", str(common["updates_per_step"]),
        "--batch_size", str(common["batch_size"]),
        "--buffer_capacity", str(common["buffer_capacity"]),
        "--her_ratio", str(common["her_ratio"]),
        "--gamma", str(common["gamma"]),
        "--tau", str(common["tau"]),
        "--actor_lr", str(common["actor_lr"]),
        "--critic_lr", str(common["critic_lr"]),
        "--hidden_dim", str(common["hidden_dim"]),
        "--noise_std", str(common["noise_std"]),
        "--eval_freq", str(common["eval_freq"]),
        "--eval_episodes", str(common["eval_episodes"]),
        "--final_eval_episodes", str(common["final_eval_episodes"]),
        "--checkpoint_every_steps", str(common["checkpoint_every_steps"]),
        "--M_B", str(common["M_B"]),
        "--M_Q_min", str(common["M_Q_min"]),
        "--N_local_min", str(common["N_local_min"]),
        "--N_gap_min", str(common["N_gap_min"]),
        "--local_search_max", str(common["local_search_max"]),
        "--action_bins_per_dim", str(common["action_bins_per_dim"]),
        "--lambda_support", str(arm["lambda_support"]),
        "--per_alpha", str(common["per_alpha"]),
        "--eps_PER", str(common["eps_PER"]),
        "--U_PER_max", str(common["U_PER_max"]),
        "--weighting_mode", str(common["weighting_mode"]),
        "--weight_scope", str(arm["weight_scope"]),
        "--policy_support_mode", str(common["policy_support_mode"]),
        "--support_scan_mode", str(common["support_scan_mode"]),
        "--support_diagnostic_audit_interval",
        str(arm["support_diagnostic_audit_interval"]),
        "--aer_lambda0", str(aer["lambda0"]),
        "--aer_alpha", str(aer["alpha"]),
        "--aer_similarity", str(aer["similarity"]),
        "--aer_similarity_eps", str(aer["similarity_eps"]),
        "--aer_candidate_rounding", str(aer["candidate_rounding"]),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20_000 + seed),
        "--eval_seed", str(10_000 + seed),
        bool_flag("support_cache_enabled", bool(arm["support_cache_enabled"])),
        bool_flag("local_scan_cache_enabled", bool(arm["local_scan_cache_enabled"])),
        bool_flag("support_short_circuit", bool(arm["support_short_circuit"])),
        bool_flag("numba_support_scan", bool(arm["numba_support_scan"])),
        bool_flag(
            "support_predicate_early_exit",
            bool(arm["support_predicate_early_exit"]),
        ),
        "--no_save_step_checkpoints",
    ]
    if validate_only:
        cmd.append("--validate_only")
    if resume:
        cmd.append("--resume")
    if resume_path is not None:
        cmd.extend(["--resume_path", str(Path(resume_path).resolve())])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen FPP DDPG ER/annealing-ER/AER no-HER protocol."
    )
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_command(
        arm_name=args.arm,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        validate_only=args.validate_only,
        resume=bool(args.resume or args.resume_path),
        resume_path=args.resume_path,
    )
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(printable, flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    source_root = EXPERIMENT_DIR / "code" / "src"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
