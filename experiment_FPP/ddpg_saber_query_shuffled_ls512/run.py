from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any


SUPPORT_QUERY_MODES = ("current_policy", "shuffled_policy")


def repo_root_from(path: Path) -> Path:
    """Find this checkout/worktree without assuming a historical parent name."""

    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists() and (parent / "experiment_FPP").is_dir():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def resolve_output_root(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def run_name(cfg: dict[str, Any], mode: str, seed: int, steps: int) -> str:
    slug = cfg["run_slug"] if mode == "shuffled_policy" else "b3_critic_only_runtime_opt_v4_ls512"
    return (
        f"fetch_pick_and_place_{slug}_"
        f"s{int(seed)}_{int(steps)}_ls{int(cfg['local_search_max'])}"
    )


def build_command(
    *,
    seed: int,
    steps: int | None = None,
    device: str = "auto",
    output_root: str | None = None,
    support_query_mode: str = "shuffled_policy",
    smoke: bool = False,
    resume: bool = False,
    resume_path: str | None = None,
    validate_only: bool = False,
    verify_runtime_optimization: bool = False,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)

    if support_query_mode not in SUPPORT_QUERY_MODES:
        raise ValueError(
            f"support_query_mode must be one of {SUPPORT_QUERY_MODES}, got {support_query_mode!r}"
        )
    if int(seed) not in {int(value) for value in cfg["seeds"]}:
        raise ValueError(f"seed must be one of {cfg['seeds']}, got {seed}")
    if smoke and steps not in {None, int(cfg["smoke_steps"])}:
        raise ValueError(f"--smoke fixes --steps to {cfg['smoke_steps']}")
    if resume_path and not resume:
        raise ValueError("--resume-path requires --resume")

    n_steps = int(cfg["smoke_steps"] if smoke else cfg["total_steps"] if steps is None else steps)
    if n_steps <= 0:
        raise ValueError(f"steps must be positive, got {n_steps}")
    default_root = cfg["smoke_output_root"] if smoke else cfg["output_root"]
    resolved_output_root = resolve_output_root(repo_root, output_root or default_root)

    source_root = folder / "code" / "src"
    trainer = source_root / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.is_file():
        python_bin = Path(sys.executable)
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", str(cfg["task_name"]),
        "--run_name", run_name(cfg, support_query_mode, seed, n_steps),
        "--output_root", str(resolved_output_root),
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
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_query_mode", support_query_mode,
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
        "--d_reach", str(cfg["d_reach"]),
        "--d_goal_near", str(cfg["d_goal_near"]),
        "--z_lift_min", str(cfg["z_lift_min"]),
        "--grip_close_thresh", str(cfg["grip_close_thresh"]),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + int(seed)),
        "--eval_seed", str(10000 + int(seed)),
    ]
    cmd.append("--support_cache_enabled" if cfg.get("support_cache_enabled", True) else "--no-support_cache_enabled")
    cmd.append("--local_scan_cache_enabled" if cfg.get("local_scan_cache_enabled", True) else "--no-local_scan_cache_enabled")
    cmd.append("--support_short_circuit" if cfg.get("support_short_circuit", True) else "--no-support_short_circuit")
    cmd.append("--numba_support_scan" if cfg.get("numba_support_scan", True) else "--no-numba_support_scan")
    cmd.append(
        "--support_predicate_early_exit"
        if cfg.get("support_predicate_early_exit", True)
        else "--no-support_predicate_early_exit"
    )
    cmd.extend(["--support_diagnostic_audit_interval", str(cfg.get("support_diagnostic_audit_interval", 100))])
    if not cfg["save_step_checkpoints"]:
        cmd.append("--no_save_step_checkpoints")
    if cfg["success_terminates"]:
        cmd.append("--success_terminates")
    if not cfg["timeout_bootstrap"]:
        cmd.append("--no_timeout_bootstrap")
    if resume:
        cmd.append("--resume")
        if resume_path:
            cmd.extend(["--resume_path", str(Path(resume_path).expanduser())])
    if validate_only:
        cmd.append("--validate_only")
    if verify_runtime_optimization:
        cmd.append("--verify_runtime_optimization")
    return cmd


def command_option(command: list[str], name: str) -> str:
    try:
        index = command.index(name)
    except ValueError as exc:
        raise RuntimeError(f"command is missing required option {name}") from exc
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise RuntimeError(f"command option {name} has no value")
    return command[index + 1]


def execution_provenance(
    repo_root: Path,
    experiment_root: Path,
    command: list[str],
) -> dict[str, Any]:
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    relative_source = experiment_root.relative_to(repo_root)
    source_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(relative_source)],
        cwd=repo_root,
        text=True,
    ).splitlines()
    return {
        "command": command,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(command_option(command, "--env_seed")),
        "source_commit": source_commit,
        "source_root": str(relative_source),
        "source_tree_dirty": bool(source_status),
        "source_tree_status": source_status,
        "steps": int(command_option(command, "--total_steps")),
        "support_query_mode": command_option(command, "--support_query_mode"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FPP DDPG SABER shuffled-policy-query ablation."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=None)
    parser.add_argument(
        "--support-query-mode",
        "--support_query_mode",
        dest="support_query_mode",
        choices=SUPPORT_QUERY_MODES,
        default="shuffled_policy",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", "--resume_path", dest="resume_path", default=None)
    parser.add_argument("--validate-only", "--validate_only", dest="validate_only", action="store_true")
    parser.add_argument("--verify-runtime-optimization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd = build_command(
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        support_query_mode=args.support_query_mode,
        smoke=args.smoke,
        resume=args.resume,
        resume_path=args.resume_path,
        validate_only=args.validate_only,
        verify_runtime_optimization=args.verify_runtime_optimization,
    )
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0

    experiment_root = Path(__file__).resolve().parent
    repo_root = repo_root_from(experiment_root)
    output_root = Path(command_option(cmd, "--output_root"))
    run_dir = output_root / command_option(cmd, "--run_name")
    if not args.validate_only:
        if args.resume:
            if not run_dir.is_dir():
                raise RuntimeError(f"resume run directory does not exist: {run_dir}")
        elif run_dir.exists():
            raise RuntimeError(f"refusing to collide with existing run directory: {run_dir}")

    print(printable, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "code" / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.call(cmd, env=env)
    if result == 0 and not args.validate_only:
        if not run_dir.is_dir():
            raise RuntimeError(f"trainer succeeded but run directory is missing: {run_dir}")
        payload = execution_provenance(repo_root, experiment_root, cmd)
        (run_dir / "execution_provenance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
