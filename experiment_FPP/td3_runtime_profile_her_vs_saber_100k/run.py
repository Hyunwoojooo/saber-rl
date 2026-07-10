from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


METHODS = ("td3_her", "td3_saber_runtime_opt")


def repo_root_from(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / "experiment_FPP").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def run_name(method: str, seed: int, steps: int, local_search_max: int) -> str:
    if method == "td3_her":
        return f"fetch_pick_and_place_td3_her_runtime_profile_s{seed}_{steps}"
    return (
        "fetch_pick_and_place_td3_saber_runtime_profile_"
        f"ls{local_search_max}_s{seed}_{steps}"
    )


def build_command(args: argparse.Namespace) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    method_cfg = cfg["methods"][args.method]
    source_root = folder / "code" / "src"
    trainer = source_root / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    steps = int(cfg["total_steps"] if args.steps is None else args.steps)
    start_steps = int(cfg["start_steps"] if args.start_steps is None else args.start_steps)
    warmup_steps = int(cfg["warmup_steps"] if args.warmup_steps is None else args.warmup_steps)
    eval_freq = int(cfg["eval_freq"] if args.eval_freq is None else args.eval_freq)
    eval_episodes = int(cfg["eval_episodes"] if args.eval_episodes is None else args.eval_episodes)
    profile_start = int(cfg["profile_start_step"] if args.profile_start_step is None else args.profile_start_step)
    profile_end = int(steps if args.profile_end_step is None else args.profile_end_step)
    profile_every = int(
        cfg["profile_sample_every_updates"]
        if args.profile_sample_every_updates is None
        else args.profile_sample_every_updates
    )
    local_search_max = int(cfg["local_search_max"])
    output_root = args.output_root or str(repo_root / cfg["output_root"])

    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(args.method, args.seed, steps, local_search_max),
        "--output_root", output_root,
        "--method", args.method,
        "--reward_type", "sparse",
        "--total_steps", str(steps),
        "--start_steps", str(start_steps),
        "--warmup_steps", str(warmup_steps),
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
        "--eval_freq", str(eval_freq),
        "--eval_episodes", str(eval_episodes),
        "--checkpoint_every_steps", str(steps),
        "--no_save_step_checkpoints",
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(local_search_max),
        "--lambda_support", str(method_cfg["lambda_support"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", method_cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
        "--support_diagnostic_audit_interval", "0",
        "--td3_policy_delay", str(cfg["td3_policy_delay"]),
        "--td3_target_policy_noise", str(cfg["td3_target_policy_noise"]),
        "--td3_target_noise_clip", str(cfg["td3_target_noise_clip"]),
        "--profile_start_step", str(profile_start),
        "--profile_end_step", str(profile_end),
        "--profile_sample_every_updates", str(profile_every),
        "--device", args.device,
        "--env_seed", str(args.seed),
        "--torch_seed", str(args.seed),
        "--numpy_seed", str(args.seed),
        "--action_noise_seed", str(args.seed),
        "--weight_seed", str(20_000 + args.seed),
        "--eval_seed", str(10_000 + args.seed),
    ]
    if args.runtime_profile:
        cmd.append("--runtime_profile")
    cmd.append("--profile_sync_cuda" if args.profile_sync_cuda else "--no-profile_sync_cuda")
    cmd.append("--profile_events_jsonl" if args.profile_events_jsonl else "--no-profile_events_jsonl")
    cmd.append("--support_cache_enabled" if method_cfg["support_cache_enabled"] else "--no-support_cache_enabled")
    cmd.append(
        "--local_scan_cache_enabled"
        if method_cfg["local_scan_cache_enabled"]
        else "--no-local_scan_cache_enabled"
    )
    cmd.append("--support_short_circuit" if method_cfg["support_lookup"] else "--no-support_short_circuit")
    cmd.append("--numba_support_scan" if method_cfg["numba_support_scan"] else "--no-numba_support_scan")
    cmd.append("--support_predicate_early_exit" if method_cfg["support_lookup"] else "--no-support_predicate_early_exit")
    if args.validate_only:
        cmd.append("--validate_only")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile matched FPP TD3 HER and TD3 HER+SABER runs.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--start_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--runtime_profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile_start_step", type=int, default=None)
    parser.add_argument("--profile_end_step", type=int, default=None)
    parser.add_argument("--profile_sample_every_updates", type=int, default=None)
    parser.add_argument("--profile_sync_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile_events_jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_command(args)
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0
    print(printable, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "code" / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
