from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ARM_TO_METHOD = {
    "er": "ddpg_er",
    "annealing_er": "ddpg_annealing_er",
    "aer": "ddpg_aer",
}


def repo_root_from(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / ".venv").exists() and (parent / "metaworld_Drawer_Close_DDPG").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def build_command(
    arm: str,
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    smoke: bool,
    validate_only: bool,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    n_steps = int(cfg["total_steps"] if steps is None else steps)
    replay_method = ARM_TO_METHOD[arm]
    run_name = f"drawer_close_{replay_method}_s{seed}_{n_steps}"
    out_root = output_root or str(folder / "runs" / replay_method)
    cmd = [
        str(python_bin),
        str(folder / "code" / "src" / "train_drawer_close_lite_v0.py"),
        "--task_name", "drawer_close",
        "--run_name", run_name,
        "--output_root", out_root,
        "--method", "b0_ddpg_er",
        "--replay_method", replay_method,
        "--weight_scope", "none",
        "--reward_type", cfg["reward_type"],
        "--metaworld_env_name", "drawer-close-v3",
        "--metaworld_reward_function_version", "v2",
        "--max_episode_steps", str(cfg["max_episode_steps"]),
        "--total_steps", str(n_steps),
        "--start_steps", str(cfg["start_steps"]),
        "--warmup_steps", str(cfg["warmup_steps"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
        "--batch_size", str(cfg["batch_size"]),
        "--buffer_capacity", str(cfg["buffer_capacity"]),
        "--her_ratio", "0.0",
        "--gamma", str(cfg["gamma"]),
        "--tau", str(cfg["tau"]),
        "--actor_lr", str(cfg["actor_lr"]),
        "--critic_lr", str(cfg["critic_lr"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--noise_std", str(cfg["noise_std"]),
        "--policy_delay", str(cfg["policy_delay"]),
        "--target_policy_noise", str(cfg["target_policy_noise"]),
        "--target_noise_clip", str(cfg["target_noise_clip"]),
        "--aer_lambda0", str(cfg["aer_lambda0"]),
        "--aer_alpha", str(cfg["aer_alpha"]),
        "--aer_cosine_eps", str(cfg["aer_cosine_eps"]),
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--final_eval_episodes", str(cfg["final_eval_episodes"]),
        "--checkpoint_every_steps", "50000",
        "--log_every_steps", "1000",
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + seed),
        "--eval_seed", str(10000 + seed),
        "--final_eval_seed", str(40000 + seed),
        "--no_save_step_checkpoints",
    ]
    if smoke:
        cmd.append("--smoke")
    if validate_only:
        cmd.append("--validate_only")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="DrawerClose DDPG no-HER AER experiment runner")
    parser.add_argument("--arm", choices=sorted(ARM_TO_METHOD), default="aer")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd = build_command(
        arm=args.arm,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        smoke=args.smoke,
        validate_only=args.validate_only,
    )
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent / "code" / "src"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
