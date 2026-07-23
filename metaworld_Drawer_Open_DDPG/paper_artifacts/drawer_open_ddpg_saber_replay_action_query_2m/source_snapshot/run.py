#!/usr/bin/env python3
"""Launch DrawerOpen replay-action-query correctness, smoke, and main runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXPERIMENT_NAME = "DDPG-HER-B3-Replay-Action-Query-2M"
RUN_PREFIX = "drawer_open_DDPG-HER-B3-Replay-Action-Query-2M"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DRAWER_ROOT = EXPERIMENT_DIR.parent
REPO_ROOT = DRAWER_ROOT.parent
TRAIN = EXPERIMENT_DIR / "code" / "src" / "train_drawer_open_lite_v0.py"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class Mode:
    steps: int
    start_steps: int
    warmup_steps: int
    eval_freq: int
    eval_episodes: int
    checkpoint_every_steps: int
    seeds: tuple[int, ...]
    output_root: Path
    log_root: Path
    tag: str


# Apart from total duration, the short runs retain the production optimization,
# replay, seeding, and evaluation settings. In particular the 5k correctness
# run keeps the 10k random-action phase and the 5k warm-up, so it performs one
# update at step 5,000 and evaluates once at its final step.
MODES = {
    "correctness5k": Mode(
        steps=5_000,
        start_steps=10_000,
        warmup_steps=5_000,
        eval_freq=10_000,
        eval_episodes=20,
        checkpoint_every_steps=50_000,
        seeds=(1,),
        output_root=EXPERIMENT_DIR / "runs" / "correctness_5k",
        log_root=EXPERIMENT_DIR / "logs" / "correctness_5k",
        tag="5k_correctness",
    ),
    "smoke20k_seed1": Mode(
        steps=20_000,
        start_steps=10_000,
        warmup_steps=5_000,
        eval_freq=10_000,
        eval_episodes=20,
        checkpoint_every_steps=50_000,
        seeds=(1,),
        output_root=EXPERIMENT_DIR / "runs" / "smoke_20k_seed1",
        log_root=EXPERIMENT_DIR / "logs" / "smoke_20k_seed1",
        tag="20k_smoke_seed1",
    ),
    "smoke20k_seeds1_5": Mode(
        steps=20_000,
        start_steps=10_000,
        warmup_steps=5_000,
        eval_freq=10_000,
        eval_episodes=20,
        checkpoint_every_steps=50_000,
        seeds=(1, 2, 3, 4, 5),
        output_root=EXPERIMENT_DIR / "runs" / "smoke_20k_seeds1_5",
        log_root=EXPERIMENT_DIR / "logs" / "smoke_20k_seeds1_5",
        tag="20k_smoke",
    ),
    "main2m": Mode(
        steps=2_000_000,
        start_steps=10_000,
        warmup_steps=5_000,
        eval_freq=10_000,
        eval_episodes=20,
        checkpoint_every_steps=50_000,
        seeds=(1, 2, 3, 4, 5),
        output_root=DRAWER_ROOT / "results_2m" / EXPERIMENT_NAME,
        log_root=DRAWER_ROOT / "logs_2m" / EXPERIMENT_NAME,
        tag="2m",
    ),
}


def common_args(mode: Mode, seed: int) -> list[str]:
    """Arguments shared exactly with the canonical DrawerOpen B3 Main."""
    return [
        "--task_name", "drawer_open",
        "--reward_type", "sparse",
        "--metaworld_env_name", "drawer-open-v3",
        "--metaworld_reward_function_version", "v2",
        "--max_episode_steps", "200",
        "--total_steps", str(mode.steps),
        "--start_steps", str(mode.start_steps),
        "--warmup_steps", str(mode.warmup_steps),
        "--updates_per_step", "1",
        "--batch_size", "256",
        "--buffer_capacity", "300000",
        "--her_ratio", "0.8",
        "--gamma", "0.98",
        "--tau", "0.005",
        "--actor_lr", "0.0003",
        "--critic_lr", "0.0003",
        "--hidden_dim", "256",
        "--noise_std", "0.2",
        "--policy_delay", "2",
        "--target_policy_noise", "0.2",
        "--target_noise_clip", "0.5",
        "--eval_freq", str(mode.eval_freq),
        "--eval_episodes", str(mode.eval_episodes),
        "--checkpoint_every_steps", str(mode.checkpoint_every_steps),
        "--log_every_steps", "1000",
        "--device", "cuda",
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20_000 + seed),
        "--eval_seed", str(10_000 + seed),
        # The periodic checkpoint is always refreshed and the final checkpoint
        # is always written; this suppresses only checkpoint_step_*.pt copies.
        "--no_save_step_checkpoints",
    ]


def saber_args() -> list[str]:
    """Canonical DrawerOpen SABER Main support configuration."""
    return [
        "--M_B", "16",
        "--M_Q_min", "4",
        "--N_local_min", "4",
        "--N_gap_min", "128",
        "--local_search_max", "512",
        "--action_bins_per_dim", "5",
        "--lambda_support", "1.0",
        "--drawer_success_threshold", "0.05",
        "--drawer_d_reach", "0.08",
        "--drawer_d_goal_near", "0.05",
        "--drawer_d_opening_mid", "0.12",
        "--weighting_mode", "batch_norm",
        "--policy_support_mode", "bin",
        "--support_scan_mode", "deterministic_tail_v1",
        "--support_cache_enabled",
        "--support_cache_max_entries", "200000",
        "--local_scan_cache_enabled",
        "--local_scan_cache_max_entries", "50000",
        "--support_short_circuit",
        "--numba_support_scan",
        "--support_predicate_early_exit",
        "--support_diagnostic_audit_interval", "100",
    ]


def replay_query_args() -> list[str]:
    """The one-factor query change relative to canonical SABER Main."""
    return [
        "--method", "b3_support_runtime_opt",
        "--weight_scope", "critic_only",
        "--support_query_mode", "replay_action",
        "--support_assignment_mode", "support",
    ]


def run_name(mode: Mode, seed: int) -> str:
    if mode is MODES["main2m"]:
        return f"{RUN_PREFIX}_s{seed}_2m_cuda"
    return f"{RUN_PREFIX}_s{seed}_{mode.tag}_cuda"


def train_args(mode: Mode, seed: int, *, resume: bool = False) -> list[str]:
    args = [*common_args(mode, seed), *saber_args(), *replay_query_args()]
    args += [
        "--run_name", run_name(mode, seed),
        "--output_root", str(mode.output_root),
    ]
    if resume:
        args.append("--resume")
    return args


def command(python: Path, mode: Mode, seed: int, *, resume: bool = False) -> list[str]:
    return [str(python), str(TRAIN), *train_args(mode, seed, resume=resume)]


def _completed(run_dir: Path, expected_steps: int) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return int(summary.get("total_steps", -1)) == expected_steps


def _preflight(mode: Mode, seeds: Iterable[int], resume: bool) -> tuple[list[int], dict[int, bool]]:
    """Reject every collision before starting the first process."""
    runnable: list[int] = []
    resume_by_seed: dict[int, bool] = {}
    for seed in seeds:
        target = mode.output_root / run_name(mode, seed)
        if _completed(target, mode.steps):
            if resume:
                print(f"SKIP complete seed={seed}: {target}")
                continue
            raise FileExistsError(f"refusing to overwrite completed run: {target}")
        if target.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite existing run: {target}")
            checkpoint = target / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"resume requested but checkpoint is missing: {checkpoint}")
            resume_by_seed[seed] = True
        else:
            if resume:
                raise FileNotFoundError(f"resume requested but run directory is missing: {target}")
            resume_by_seed[seed] = False
        runnable.append(seed)
    return runnable, resume_by_seed


def _runtime_env(cuda_device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EXPERIMENT_DIR / "code" / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _run_bounded(
    python: Path,
    mode: Mode,
    seeds: list[int],
    resume_by_seed: dict[int, bool],
    cuda_device: str,
    max_parallel: int,
) -> None:
    """Run fixed-size batches so no more than max_parallel jobs share a GPU."""
    mode.output_root.mkdir(parents=True, exist_ok=True)
    mode.log_root.mkdir(parents=True, exist_ok=True)
    env = _runtime_env(cuda_device)
    for offset in range(0, len(seeds), max_parallel):
        batch = seeds[offset : offset + max_parallel]
        processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
        try:
            for seed in batch:
                log_path = mode.log_root / f"{run_name(mode, seed)}.log"
                log_handle = log_path.open("ab")
                cmd = command(python, mode, seed, resume=resume_by_seed[seed])
                print(f"START seed={seed} log={log_path}", flush=True)
                try:
                    process = subprocess.Popen(
                        cmd,
                        cwd=EXPERIMENT_DIR,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log_handle.close()
                    raise
                processes.append((seed, process, log_handle))
            failed: list[tuple[int, int]] = []
            for seed, process, _handle in processes:
                return_code = process.wait()
                if return_code:
                    failed.append((seed, return_code))
            if failed:
                raise RuntimeError(f"one or more seeds failed: {failed}")
        finally:
            for _seed, process, handle in processes:
                if process.poll() is None:
                    process.terminate()
                handle.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="maximum jobs sharing CUDA_DEVICE (safe default: 1)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = MODES[args.mode]
    seeds = tuple(args.seeds) if args.seeds else mode.seeds
    if not seeds or any(seed not in {1, 2, 3, 4, 5} for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 1..5")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not allowed")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if not args.python.is_file():
        raise FileNotFoundError(f"python executable not found: {args.python}")
    if not TRAIN.is_file():
        raise FileNotFoundError(f"trainer not found: {TRAIN}")

    if args.validate_only:
        if args.resume:
            raise ValueError("--validate-only and --resume cannot be combined")
        seed = seeds[0]
        validate_command = [
            *command(args.python, mode, seed),
            "--validate_only",
            "--device",
            "cpu",
        ]
        if args.dry_run:
            print(shlex.join(validate_command))
            return 0
        subprocess.run(
            validate_command,
            cwd=EXPERIMENT_DIR,
            env=_runtime_env(args.cuda_device),
            check=True,
        )
        return 0

    if args.dry_run:
        for seed in seeds:
            print(shlex.join(command(args.python, mode, seed, resume=args.resume)))
        return 0

    runnable, resume_by_seed = _preflight(mode, seeds, args.resume)
    if not runnable:
        print("all requested runs are already complete")
        return 0
    _run_bounded(
        args.python,
        mode,
        runnable,
        resume_by_seed,
        args.cuda_device,
        args.max_parallel,
    )
    print(f"completed mode={args.mode} seeds={runnable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
