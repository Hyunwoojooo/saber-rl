from __future__ import annotations

import argparse
import hashlib
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
    return json.loads((folder / "config.json").read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_production_freeze(folder: Path) -> None:
    manifest_path = folder / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "production is locked until smoke and gate validations are frozen: "
            f"missing {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("production_config_frozen") is not True:
        raise RuntimeError(f"invalid production freeze manifest: {manifest_path}")
    expected = manifest.get("frozen_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"freeze manifest has no source hashes: {manifest_path}")
    for relative, expected_digest in expected.items():
        path = folder / relative
        if not path.is_file():
            raise RuntimeError(f"frozen production source is missing: {relative}")
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                "frozen production source changed after the technical gate: "
                f"{relative}"
            )


def run_name(cfg: dict[str, Any], stage: str, seed: int, steps: int) -> str:
    return (
        f"fetch_pick_and_place_{cfg['stage_run_slugs'][stage]}_"
        f"s{int(seed)}_{int(steps)}_ls{int(cfg['local_search_max'])}"
    )


def build_command(
    stage: str,
    seed: int,
    device: str,
    output_root: str | None,
    validate_only: bool,
    resume: bool = False,
    resume_path: str | None = None,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    source_root = folder / "code" / "src"
    trainer = source_root / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    if stage not in cfg["stage_output_roots"]:
        raise ValueError(f"Unsupported stage={stage!r}")
    if int(seed) not in cfg["seeds"]:
        raise ValueError(f"seed must be one of {cfg['seeds']}, got {seed}")
    if stage in {"smoke", "gate"} and int(seed) != 1:
        raise ValueError(f"{stage} is fixed to seed 1")
    n_steps = int(cfg["stage_steps"][stage])
    out_root = output_root or str(repo_root / cfg["stage_output_roots"][stage])
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(cfg, stage, seed, n_steps),
        "--output_root", out_root,
        "--method", cfg["trainer_method"],
        "--reward_type", cfg["reward_type"],
        "--max_episode_steps", str(cfg["max_episode_steps"]),
        "--total_steps", str(n_steps),
        "--start_steps", str(cfg["start_steps"]),
        "--warmup_steps", str(cfg["warmup_steps"]),
        "--batch_size", str(cfg["batch_size"]),
        "--buffer_capacity", str(cfg["buffer_capacity"]),
        "--her_ratio", str(cfg["her_ratio"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
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
        "--support_query_mode", cfg["support_query_mode"],
        "--support_assignment_mode", cfg["support_assignment_mode"],
        "--assignment_diagnostic_window_updates",
        str(cfg["assignment_diagnostic_window_updates"]),
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg.get("support_cache_max_entries", 200000)),
        "--local_scan_cache_max_entries", str(cfg.get("local_scan_cache_max_entries", 50000)),
        "--td3_policy_delay", str(cfg.get("td3_policy_delay", 2)),
        "--td3_target_policy_noise", str(cfg.get("td3_target_policy_noise", 0.2)),
        "--td3_target_noise_clip", str(cfg.get("td3_target_noise_clip", 0.5)),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + int(seed)),
        "--assignment_seed", str(30000 + int(seed)),
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
    if stage == "smoke":
        cmd.append("--smoke")
    if resume:
        cmd.append("--resume")
        if resume_path is not None:
            cmd.extend(["--resume_path", resume_path])
    if validate_only:
        cmd.append("--validate_only")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FPP TD3 context-eligible count-matched control."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["smoke", "gate", "production"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    if args.stage == "production":
        verify_production_freeze(folder)
    cmd = build_command(
        stage=args.stage,
        seed=args.seed,
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
    print(printable, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "code" / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
