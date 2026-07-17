from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


FROZEN_EXPECTED: dict[str, Any] = {
    "task_name": "fetch_pick_and_place",
    "backbone": "SAC",
    "trainer_method": "sac_her_per_style_loss_weighting",
    "run_slug": "sac_her_per_style_loss_weighting_ls512",
    "output_root": "experiment_FPP/runs/sac_her_per_style_loss_weighting_ls512",
    "reward_type": "sparse",
    "max_episode_steps": 50,
    "total_steps": 1_000_000,
    "seeds": [1, 2, 3, 4, 5],
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
    "sac_alpha": 0.2,
    "sac_auto_alpha": True,
    "sac_alpha_lr": 3e-4,
    "sac_target_entropy": None,
    "sac_per_priority_reduction": "min_abs",
    "per_alpha": 0.6,
    "eps_PER": 1e-6,
    "U_PER_max": 2.0,
    "weighting_mode": "batch_norm",
    "weight_scope": "critic_only",
    "replay_sampling": "uniform_her",
    "priority_storage": False,
    "importance_sampling_beta": False,
    "eval_freq": 25_000,
    "eval_episodes": 20,
    "checkpoint_every_steps": 50_000,
    "save_step_checkpoints": False,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "N_gap_min": 128,
    "local_search_max": 512,
    "lambda_support": 0.0,
    "support_lookup": False,
    "policy_support_mode": "bin",
    "support_scan_mode": "deterministic_tail_v1",
    "support_cache_enabled": False,
    "local_scan_cache_enabled": False,
    "support_short_circuit": False,
    "numba_support_scan": False,
    "support_predicate_early_exit": False,
    "diagnostic_log_samples": False,
    "diagnostic_full_batch_metrics": False,
    "seed_mapping": {
        "env_seed": "seed",
        "torch_seed": "seed",
        "numpy_seed": "seed",
        "action_noise_seed": "seed",
        "weight_seed": "20000 + seed",
        "eval_seed": "10000 + seed",
    },
}


def repo_root_from(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / "experiment_FPP").is_dir() and (
            (parent / ".git").exists()
            or (parent / "version_v0_lite").is_dir()
        ):
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


def validate_frozen_config(cfg: dict[str, Any]) -> None:
    for key, value in FROZEN_EXPECTED.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Frozen config mismatch: {key}={cfg.get(key)!r}, expected {value!r}"
            )


def load_config(folder: Path) -> dict[str, Any]:
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    validate_frozen_config(cfg)
    return cfg


def run_name(cfg: dict[str, Any], seed: int, steps: int) -> str:
    return f"fetch_pick_and_place_{cfg['run_slug']}_s{int(seed)}_{int(steps)}"


def build_command(
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    validate_only: bool,
    resume: bool = False,
    resume_path: str | None = None,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    n_steps = int(cfg["total_steps"] if steps is None else steps)
    if seed <= 0:
        raise ValueError("seed must be positive")
    if n_steps <= 0 or n_steps % int(cfg["max_episode_steps"]) != 0:
        raise ValueError(
            f"steps must be a positive multiple of episode horizon {cfg['max_episode_steps']}"
        )

    trainer = folder / "code" / "src" / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    out_root = Path(output_root) if output_root else repo_root / cfg["output_root"]
    if not out_root.is_absolute():
        out_root = repo_root / out_root

    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(cfg, seed, n_steps),
        "--output_root", str(out_root),
        "--method", cfg["trainer_method"],
        "--reward_type", cfg["reward_type"],
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
        "--sac_alpha", str(cfg["sac_alpha"]),
        "--sac_alpha_lr", str(cfg["sac_alpha_lr"]),
        "--sac_auto_alpha" if cfg["sac_auto_alpha"] else "--no-sac_auto_alpha",
        "--sac_per_priority_reduction", cfg["sac_per_priority_reduction"],
        "--per_alpha", str(cfg["per_alpha"]),
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--checkpoint_every_steps", str(cfg["checkpoint_every_steps"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + seed),
        "--eval_seed", str(10000 + seed),
        "--no-support_cache_enabled",
        "--no-local_scan_cache_enabled",
        "--no-support_short_circuit",
        "--no-numba_support_scan",
        "--no-support_predicate_early_exit",
        "--no-diagnostic_log_samples",
        "--no-diagnostic_full_batch_metrics",
    ]
    if cfg["sac_target_entropy"] is not None:
        cmd.extend(["--sac_target_entropy", str(cfg["sac_target_entropy"])])
    if not cfg["save_step_checkpoints"]:
        cmd.append("--no_save_step_checkpoints")
    if cfg["success_terminates"]:
        cmd.append("--success_terminates")
    if not cfg["timeout_bootstrap"]:
        cmd.append("--no_timeout_bootstrap")
    if resume:
        cmd.append("--resume")
    if resume_path:
        cmd.extend(["--resume_path", str(resume_path)])
    if validate_only:
        cmd.append("--validate_only")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen FetchPickAndPlace SAC+HER+PER-style experiment."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        validate_only=args.validate_only,
        resume=args.resume,
        resume_path=args.resume_path,
    )
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0

    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    n_steps = int(cfg["total_steps"] if args.steps is None else args.steps)
    out_root = Path(args.output_root) if args.output_root else repo_root / cfg["output_root"]
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    target = out_root / run_name(cfg, args.seed, n_steps)
    if not args.validate_only:
        checkpoint = Path(args.resume_path) if args.resume_path else target / "checkpoint.pt"
        if args.resume and not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        if not args.resume and target.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {target}")

    print(printable, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(folder / "code" / "src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(name, "1")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
