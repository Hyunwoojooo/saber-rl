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
        if (parent / "version_v0_lite").exists():
            return parent
        if (parent / "experiment_FetchPush").exists() and (parent / "metaworld_Drawer_Open").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text())


def run_name(cfg: dict[str, Any], seed: int, steps: int) -> str:
    return (
        f"drawer_open_{cfg['run_slug']}_"
        f"s{int(seed)}_{int(steps)}_ls{int(cfg['local_search_max'])}"
    )


def build_command(
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    smoke: bool,
    validate_only: bool,
    verify_runtime_optimization: bool,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    source_root = folder / "code" / "src"
    trainer = source_root / "train_drawer_open_lite_v0.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    n_steps = int(cfg["total_steps"] if steps is None else steps)
    out_root = output_root or str(repo_root / cfg["output_root"])
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "drawer_open",
        "--run_name", run_name(cfg, seed, n_steps),
        "--output_root", out_root,
        "--method", cfg["trainer_method"],
        "--reward_type", cfg["reward_type"],
        "--metaworld_env_name", cfg.get("metaworld_env_name", "drawer-open-v3"),
        "--metaworld_reward_function_version", cfg.get("metaworld_reward_function_version", "v2"),
        "--max_episode_steps", str(cfg.get("max_episode_steps", 200)),
        "--total_steps", str(n_steps),
        "--batch_size", str(cfg["batch_size"]),
        "--her_ratio", str(cfg["her_ratio"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
        "--policy_delay", str(cfg.get("policy_delay", 2)),
        "--target_policy_noise", str(cfg.get("target_policy_noise", 0.2)),
        "--target_noise_clip", str(cfg.get("target_noise_clip", 0.5)),
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--support_rate_cap", str(cfg.get("support_rate_cap", 1.0)),
        "--drawer_success_threshold", str(cfg.get("drawer_success_threshold", 0.05)),
        "--drawer_d_reach", str(cfg.get("drawer_d_reach", 0.08)),
        "--drawer_d_goal_near", str(cfg.get("drawer_d_goal_near", 0.05)),
        "--drawer_d_opening_mid", str(cfg.get("drawer_d_opening_mid", 0.12)),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
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
    if smoke:
        cmd.append("--smoke")
    if validate_only:
        cmd.append("--validate_only")
    if verify_runtime_optimization:
        cmd.append("--verify_runtime_optimization")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SABER MetaWorld drawer-open-v3 B3 support-rate-cap 0.4 method.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--verify_runtime_optimization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd = build_command(
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        smoke=args.smoke,
        validate_only=args.validate_only,
        verify_runtime_optimization=args.verify_runtime_optimization,
    )
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0
    print(printable, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "code" / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
