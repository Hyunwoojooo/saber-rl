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
        if (parent / "version_v0_lite").exists() and (parent / "SABER").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text())


def run_name(cfg: dict[str, Any], seed: int, steps: int) -> str:
    return (
        f"fetch_pick_and_place_{cfg['run_slug']}_"
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
    trainer = source_root / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    n_steps = int(cfg["total_steps"] if steps is None else steps)
    out_root = output_root or str(repo_root / cfg["output_root"])
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(cfg, seed, n_steps),
        "--output_root", out_root,
        "--method", cfg["trainer_method"],
        "--reward_type", cfg["reward_type"],
        "--total_steps", str(n_steps),
        "--batch_size", str(cfg["batch_size"]),
        "--her_ratio", str(cfg["her_ratio"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
        "--runtime_predicate_chunk_size", str(cfg.get("runtime_predicate_chunk_size", 64)),
        "--runtime_exact_audit_interval", str(cfg.get("runtime_exact_audit_interval", 100)),
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
    cmd.append("--runtime_predicate_early_exit_enabled" if cfg.get("runtime_predicate_early_exit_enabled", True) else "--no-runtime_predicate_early_exit_enabled")
    if smoke:
        cmd.append("--smoke")
    if validate_only:
        cmd.append("--validate_only")
    if verify_runtime_optimization:
        cmd.append("--verify_runtime_optimization")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run patched SABER FetchPickAndPlace B3 runtime optimization.")
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
