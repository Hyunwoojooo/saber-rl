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


def repo_root_from(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text())


def run_name(cfg: dict[str, Any], seed: int, steps: int, smoke: bool = False) -> str:
    name = f"{cfg['task_name']}_{cfg['run_slug']}_s{int(seed)}_{int(steps)}"
    return f"{name}_smoke" if smoke else name


def _boolean_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def build_command(
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    smoke: bool,
    validate_only: bool,
    resume: bool = False,
    resume_path: str | None = None,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    trainer = folder / "code" / "src" / cfg["trainer_script"]
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    n_steps = int(cfg["total_steps"] if steps is None else steps)
    out_root = output_root or str(repo_root / cfg["output_root"])
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", cfg["task_name"],
        "--run_name", run_name(cfg, seed, n_steps, smoke=smoke),
        "--output_root", out_root,
        "--method", cfg["trainer_method"],
        "--reward_type", cfg["reward_type"],
        "--metaworld_env_name", cfg["metaworld_env_name"],
        "--metaworld_reward_function_version", cfg["metaworld_reward_function_version"],
        "--max_episode_steps", str(cfg["max_episode_steps"]),
        "--total_steps", str(n_steps),
        "--start_steps", str(cfg["start_steps"]),
        "--warmup_steps", str(cfg["warmup_steps"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
        "--batch_size", str(cfg["batch_size"]),
        "--buffer_capacity", str(cfg["buffer_capacity"]),
        "--her_ratio", str(cfg["her_ratio"]),
        "--gamma", str(cfg["gamma"]),
        "--tau", str(cfg["tau"]),
        "--actor_lr", str(cfg["actor_lr"]),
        "--critic_lr", str(cfg["critic_lr"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--noise_std", str(cfg["noise_std"]),
        "--policy_delay", str(cfg["policy_delay"]),
        "--target_policy_noise", str(cfg["target_policy_noise"]),
        "--target_noise_clip", str(cfg["target_noise_clip"]),
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--checkpoint_every_steps", str(cfg["checkpoint_every_steps"]),
        "--log_every_steps", str(cfg["log_every_steps"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--action_bins_per_dim", str(cfg["action_bins_per_dim"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--per_alpha", str(cfg["per_alpha"]),
        "--per_beta_start", str(cfg["per_beta_start"]),
        "--per_beta_end", str(cfg["per_beta_end"]),
        "--per_beta_steps", str(cfg["per_beta_steps"]),
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
        "--support_diagnostic_audit_interval", str(cfg["support_diagnostic_audit_interval"]),
        "--drawer_success_threshold", str(cfg["drawer_success_threshold"]),
        "--drawer_d_reach", str(cfg["drawer_d_reach"]),
        "--drawer_d_goal_near", str(cfg["drawer_d_goal_near"]),
        "--drawer_d_opening_mid", str(cfg["drawer_d_opening_mid"]),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + int(seed)),
        "--eval_seed", str(10000 + int(seed)),
    ]
    cmd.append(_boolean_flag("support_cache_enabled", cfg["support_cache_enabled"]))
    cmd.append(_boolean_flag("local_scan_cache_enabled", cfg["local_scan_cache_enabled"]))
    cmd.append(_boolean_flag("support_short_circuit", cfg["support_short_circuit"]))
    cmd.append(_boolean_flag("numba_support_scan", cfg["numba_support_scan"]))
    cmd.append(_boolean_flag("support_predicate_early_exit", cfg["support_predicate_early_exit"]))
    if not cfg["save_step_checkpoints"]:
        cmd.append("--no_save_step_checkpoints")
    if cfg["success_terminates"]:
        cmd.append("--success_terminates")
    if not cfg["timeout_bootstrap"]:
        cmd.append("--no_timeout_bootstrap")
    if smoke:
        cmd.append("--smoke")
    if validate_only:
        cmd.append("--validate_only")
    if resume:
        cmd.append("--resume")
    if resume_path is not None:
        cmd.extend(["--resume_path", resume_path])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a frozen Drawer HER + Full PER experiment.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        smoke=args.smoke,
        validate_only=args.validate_only,
        resume=args.resume,
        resume_path=args.resume_path,
    )
    printable = " ".join(shlex.quote(part) for part in command)
    if args.dry_run:
        print(printable)
        return 0
    print(printable, flush=True)
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent / "code" / "src"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
