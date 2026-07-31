#!/usr/bin/env python3
"""Package-relative launcher for the seven fixed primary SABER presets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "presets"

REQUIRED_SETTINGS = {
    "task_name",
    "method",
    "reward_type",
    "max_episode_steps",
    "start_steps",
    "warmup_steps",
    "updates_per_step",
    "batch_size",
    "buffer_capacity",
    "her_ratio",
    "gamma",
    "tau",
    "actor_lr",
    "critic_lr",
    "hidden_dim",
    "eval_freq",
    "eval_episodes",
    "M_B",
    "M_Q_min",
    "N_local_min",
    "N_gap_min",
    "local_search_max",
    "action_bins_per_dim",
    "lambda_support",
    "weighting_mode",
    "weight_scope",
    "policy_support_mode",
    "support_scan_mode",
}

CRITICAL_VALUES = {
    "updates_per_step": 1,
    "batch_size": 256,
    "buffer_capacity": 300000,
    "her_ratio": 0.8,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "N_gap_min": 128,
    "local_search_max": 512,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "weighting_mode": "batch_norm",
    "weight_scope": "critic_only",
    "policy_support_mode": "bin",
    "support_scan_mode": "deterministic_tail_v1",
}


def load_presets() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        preset_id = str(data.get("preset_id", ""))
        if not preset_id or preset_id in presets:
            raise ValueError(f"Invalid or duplicate preset_id in {path}")
        validate_preset(data, path)
        presets[preset_id] = data
    if len(presets) != 7:
        raise ValueError(f"Expected seven primary presets, found {len(presets)}")
    return presets


def validate_preset(data: dict[str, Any], path: Path) -> None:
    for field in ("preset_id", "task_label", "backbone", "source", "entrypoint", "paper_steps"):
        if field not in data:
            raise ValueError(f"{path}: missing {field}")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: settings must be an object")
    missing = sorted(REQUIRED_SETTINGS - set(settings))
    if missing:
        raise ValueError(f"{path}: missing settings {missing}")
    for key, expected in CRITICAL_VALUES.items():
        if settings.get(key) != expected:
            raise ValueError(f"{path}: {key}={settings.get(key)!r}, expected {expected!r}")
    source = ROOT / str(data["source"])
    entrypoint = source / str(data["entrypoint"])
    if not source.is_dir() or not entrypoint.is_file():
        raise ValueError(f"{path}: missing source or entrypoint")


def setting_arguments(settings: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in settings.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
            else:
                arguments.append(f"--no-{key.replace('_', '-')}")
        elif value is not None:
            arguments.extend((flag, str(value)))
    return arguments


def build_command(
    preset: dict[str, Any],
    seed: int,
    steps: int | None,
    device: str,
    output_root: Path,
    smoke: bool,
    validate_only: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    source = ROOT / str(preset["source"])
    entrypoint = source / str(preset["entrypoint"])
    budget = int(preset["paper_steps"] if steps is None else steps)
    run_name = f"{preset['preset_id']}_s{seed}_{budget}"
    command = [
        sys.executable,
        str(entrypoint),
        *setting_arguments(dict(preset["settings"])),
        "--total_steps",
        str(budget),
        "--output_root",
        str(output_root.resolve()),
        "--run_name",
        run_name,
        "--device",
        device,
        "--env_seed",
        str(seed),
        "--torch_seed",
        str(seed),
        "--numpy_seed",
        str(seed),
        "--action_noise_seed",
        str(seed),
        "--weight_seed",
        str(20000 + seed),
        "--eval_seed",
        str(10000 + seed),
    ]
    if smoke:
        command.append("--smoke")
    if validate_only:
        command.append("--validate_only")
    env = dict(os.environ)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source) if not prior else f"{source}{os.pathsep}{prior}"
    return command, source, env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preset")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reviewer_runs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to start a process; otherwise the command is only printed.",
    )
    args = parser.parse_args()

    presets = load_presets()
    if args.list:
        for preset_id, preset in presets.items():
            print(
                f"{preset_id:20s} {preset['task_label']:12s} "
                f"{preset['backbone']:4s} steps={preset['paper_steps']}"
            )
        return 0
    if args.preset not in presets:
        parser.error("--preset must name one of: " + ", ".join(sorted(presets)))
    if args.seed < 1:
        parser.error("--seed must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")

    command, cwd, env = build_command(
        presets[args.preset],
        args.seed,
        args.steps,
        args.device,
        args.output_root,
        args.smoke,
        args.validate_only,
    )
    printable = {
        "preset": args.preset,
        "cwd": str(cwd.relative_to(ROOT)),
        "pythonpath": str(cwd.relative_to(ROOT)),
        "command": shlex.join(command),
        "paper_budget_overridden": args.steps is not None,
    }
    print(json.dumps(printable, indent=2))
    if args.dry_run or not args.execute:
        if not args.dry_run and not args.execute:
            print("Not executed. Add --execute to start the process.")
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
