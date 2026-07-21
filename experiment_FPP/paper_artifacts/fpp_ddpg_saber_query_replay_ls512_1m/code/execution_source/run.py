from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


MAIN_STEPS = 1_000_000
SMOKE_STEPS = 20_000
RUN_SLUG = "ddpg_saber_query_replay_ls512"
MAIN_OUTPUT_ROOT = Path("experiment_FPP/runs/ddpg_saber_query_replay_ls512")
SMOKE_OUTPUT_ROOT = Path("experiment_FPP/runs/ddpg_saber_query_replay_ls512_smoke20k")

# Canonical values that are implicit trainer defaults and therefore deliberately
# absent from the compact package config.  They are still passed explicitly here
# so an upstream default change cannot silently alter this matched ablation.
FIXED = {
    "max_episode_steps": 50,
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "buffer_capacity": 300_000,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "l2_action_penalty": 0.0,
    "checkpoint_every_steps": 50_000,
    "log_every_steps": 1_000,
    "action_bins_per_dim": 5,
    "policy_action_radius": 0.35,
    "d_reach": 0.08,
    "d_goal_near": 0.05,
    "z_lift_min": 0.48,
    "grip_close_thresh": 0.04,
    "gap_window": 50_000,
    "lambda_P": 1.0,
    "U_P_max": 1.0,
    "lambda_TD": 1.0,
    "U_TD_max": 1.0,
    "per_alpha": 0.6,
    "eps_PER": 1e-6,
    "U_PER_max": 2.0,
    "unrestricted_candidate_std": 0.5,
}


def repo_root_from(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / ".git").exists() and (parent / "experiment_FPP").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    value = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config.json must contain an object")
    return value


def allowed_seeds(cfg: dict[str, Any]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in cfg.get("seeds", ()))
    if seeds != (1, 2, 3, 4, 5):
        raise ValueError(f"config seeds must be exactly (1,2,3,4,5), got {seeds}")
    return seeds


def resolve_python(repo_root: Path) -> Path:
    configured = os.environ.get("PYTHON_BIN", "").strip()
    if configured:
        return Path(configured).expanduser()
    repository_python = repo_root / ".venv" / "bin" / "python"
    return repository_python if repository_python.exists() else Path(sys.executable)


def run_name(cfg: dict[str, Any], seed: int, steps: int) -> str:
    return (
        f"fetch_pick_and_place_{cfg['run_slug']}_"
        f"s{int(seed)}_{int(steps)}_ls{int(cfg['local_search_max'])}"
    )


def registered_output_root(repo_root: Path, steps: int) -> Path:
    relative = SMOKE_OUTPUT_ROOT if steps == SMOKE_STEPS else MAIN_OUTPUT_ROOT
    return (repo_root / relative).resolve()


def resolve_output_root(repo_root: Path, steps: int, requested: str | None) -> Path:
    expected = registered_output_root(repo_root, steps)
    if requested is None:
        return expected
    supplied = Path(requested).expanduser()
    if not supplied.is_absolute():
        supplied = repo_root / supplied
    supplied = supplied.resolve()
    if supplied != expected:
        raise ValueError(
            f"{steps}-step runs must use isolated output root {expected}, got {supplied}"
        )
    return supplied


def require_new_target(output_root: Path, cfg: dict[str, Any], seed: int, steps: int) -> None:
    name = run_name(cfg, seed, steps)
    siblings = sorted(path for path in output_root.glob(f"{name}*") if path.exists())
    if siblings:
        raise FileExistsError(
            "Refusing to create or reuse a duplicate/timestamped run directory: "
            + ", ".join(str(path) for path in siblings)
        )


def build_command(
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    validate_only: bool,
    verify_runtime_optimization: bool,
    enforce_new_target: bool = True,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    if cfg.get("run_slug") != RUN_SLUG:
        raise ValueError(f"run_slug must be {RUN_SLUG!r}")
    if cfg.get("support_query_mode") != "replay_action":
        raise ValueError("support_query_mode must be replay_action")
    if cfg.get("weight_scope") != "critic_only":
        raise ValueError("weight_scope must be critic_only")
    if int(seed) not in allowed_seeds(cfg):
        raise ValueError(f"seed must be one of {allowed_seeds(cfg)}, got {seed}")
    n_steps = int(cfg["total_steps"] if steps is None else steps)
    if n_steps not in {SMOKE_STEPS, MAIN_STEPS}:
        raise ValueError(f"steps must be exactly {SMOKE_STEPS} or {MAIN_STEPS}, got {n_steps}")
    out_root = resolve_output_root(repo_root, n_steps, output_root)
    if enforce_new_target and not validate_only:
        require_new_target(out_root, cfg, seed, n_steps)

    source_root = folder / "code" / "src"
    trainer = source_root / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = resolve_python(repo_root)
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(cfg, seed, n_steps),
        "--output_root", str(out_root),
        "--method", str(cfg["trainer_method"]),
        "--reward_type", str(cfg["reward_type"]),
        "--max_episode_steps", str(FIXED["max_episode_steps"]),
        "--total_steps", str(n_steps),
        "--start_steps", str(FIXED["start_steps"]),
        "--warmup_steps", str(FIXED["warmup_steps"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
        "--batch_size", str(cfg["batch_size"]),
        "--buffer_capacity", str(FIXED["buffer_capacity"]),
        "--her_ratio", str(cfg["her_ratio"]),
        "--gamma", str(FIXED["gamma"]),
        "--tau", str(FIXED["tau"]),
        "--actor_lr", str(FIXED["actor_lr"]),
        "--critic_lr", str(FIXED["critic_lr"]),
        "--hidden_dim", str(FIXED["hidden_dim"]),
        "--noise_std", str(FIXED["noise_std"]),
        "--l2_action_penalty", str(FIXED["l2_action_penalty"]),
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--checkpoint_every_steps", str(FIXED["checkpoint_every_steps"]),
        "--log_every_steps", str(FIXED["log_every_steps"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--action_bins_per_dim", str(FIXED["action_bins_per_dim"]),
        "--policy_support_mode", str(cfg["policy_support_mode"]),
        "--policy_action_radius", str(FIXED["policy_action_radius"]),
        "--support_query_mode", str(cfg["support_query_mode"]),
        "--support_scan_mode", str(cfg["support_scan_mode"]),
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
        "--support_diagnostic_audit_interval", str(cfg["support_diagnostic_audit_interval"]),
        "--d_reach", str(FIXED["d_reach"]),
        "--d_goal_near", str(FIXED["d_goal_near"]),
        "--z_lift_min", str(FIXED["z_lift_min"]),
        "--grip_close_thresh", str(FIXED["grip_close_thresh"]),
        "--gap_window", str(FIXED["gap_window"]),
        "--lambda_P", str(FIXED["lambda_P"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--U_P_max", str(FIXED["U_P_max"]),
        "--lambda_TD", str(FIXED["lambda_TD"]),
        "--U_TD_max", str(FIXED["U_TD_max"]),
        "--per_alpha", str(FIXED["per_alpha"]),
        "--eps_PER", str(FIXED["eps_PER"]),
        "--U_PER_max", str(FIXED["U_PER_max"]),
        "--weighting_mode", str(cfg["weighting_mode"]),
        "--weight_scope", str(cfg["weight_scope"]),
        "--unrestricted_candidate_std", str(FIXED["unrestricted_candidate_std"]),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20_000 + int(seed)),
        "--eval_seed", str(10_000 + int(seed)),
    ]
    cmd.append("--support_cache_enabled" if cfg["support_cache_enabled"] else "--no-support_cache_enabled")
    cmd.append(
        "--local_scan_cache_enabled"
        if cfg["local_scan_cache_enabled"]
        else "--no-local_scan_cache_enabled"
    )
    cmd.append("--support_short_circuit" if cfg["support_short_circuit"] else "--no-support_short_circuit")
    cmd.append("--numba_support_scan" if cfg["numba_support_scan"] else "--no-numba_support_scan")
    cmd.append(
        "--support_predicate_early_exit"
        if cfg["support_predicate_early_exit"]
        else "--no-support_predicate_early_exit"
    )
    if validate_only:
        cmd.append("--validate_only")
    if verify_runtime_optimization:
        cmd.append("--verify_runtime_optimization")
    return cmd


def main() -> int:
    folder = Path(__file__).resolve().parent
    cfg = load_config(folder)
    parser = argparse.ArgumentParser(
        description="Run the strict matched FPP DDPG SABER replay-action-query LS512 ablation."
    )
    parser.add_argument("--seed", type=int, default=1, choices=allowed_seeds(cfg))
    parser.add_argument("--steps", type=int, default=MAIN_STEPS, choices=(SMOKE_STEPS, MAIN_STEPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--verify_runtime_optimization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        cmd = build_command(
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            output_root=args.output_root,
            validate_only=args.validate_only,
            verify_runtime_optimization=args.verify_runtime_optimization,
            enforce_new_target=not args.dry_run,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0
    print(printable, flush=True)
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent / "code" / "src"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
