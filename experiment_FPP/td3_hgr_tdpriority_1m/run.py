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
        if (parent / ".git").exists() and (parent / "experiment_FPP").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    expected = {
        "backbone": "TD3",
        "trainer_method": "td3_hgr",
        "her_ratio": 0.8,
        "updates_per_step": 1,
        "hgr_td3_priority_reduction": "min_abs",
        "success_terminates": False,
        "timeout_bootstrap": True,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Frozen config mismatch: {key}={cfg.get(key)!r}, expected {value!r}")
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
        "--td3_policy_delay", str(cfg["td3_policy_delay"]),
        "--td3_target_policy_noise", str(cfg["td3_target_policy_noise"]),
        "--td3_target_noise_clip", str(cfg["td3_target_noise_clip"]),
        "--hgr_priority_eps", str(cfg["hgr_priority_eps"]),
        "--hgr_priority_init", str(cfg["hgr_priority_init"]),
        "--hgr_priority_clip_max", str(cfg["hgr_priority_clip_max"]),
        "--hgr_alpha_episode", str(cfg["hgr_alpha_episode"]),
        "--hgr_alpha_goal", str(cfg["hgr_alpha_goal"]),
        "--hgr_beta_episode_initial", str(cfg["hgr_beta_episode_initial"]),
        "--hgr_beta_goal_initial", str(cfg["hgr_beta_goal_initial"]),
        "--hgr_beta_episode_final", str(cfg["hgr_beta_episode_final"]),
        "--hgr_beta_goal_final", str(cfg["hgr_beta_goal_final"]),
        "--hgr_beta_anneal_updates", str(cfg["hgr_beta_anneal_updates"]),
        "--hgr_uniform_mix", str(cfg["hgr_uniform_mix"]),
        "--hgr_td3_priority_reduction", cfg["hgr_td3_priority_reduction"],
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--checkpoint_every_steps", str(cfg["checkpoint_every_steps"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--per_alpha", str(cfg["per_alpha"]),
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
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
        description="Run the frozen FetchPickAndPlace TD3+HER+HGR experiment."
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
    env["PYTHONPATH"] = str(folder / "code" / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
