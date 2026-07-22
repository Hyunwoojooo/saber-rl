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


RUN_SLUG = "ddpg_saber_context_eligible_count_matched_ls512"
STAGES = {
    "smoke": {
        "steps": 5_000,
        "output": Path("experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_smoke_5k"),
        "slug": "ddpg_saber_context_eligible_count_matched_smoke",
    },
    "gate": {
        "steps": 50_000,
        "output": Path("experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_gate_50k"),
        "slug": "ddpg_saber_context_eligible_count_matched_gate",
    },
    "production": {
        "steps": 1_000_000,
        "output": Path("experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_ls512"),
        "slug": RUN_SLUG,
    },
}

# Canonical DDPG v4 LS512 values that are implicit in the reference package.
# Passing them explicitly prevents upstream default drift.
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_production_freeze(folder: Path) -> None:
    path = folder / "freeze_manifest.json"
    if not path.is_file():
        raise RuntimeError(
            "production is locked until smoke and technical-gate evidence is frozen: "
            f"missing {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("production_config_frozen") is not True:
        raise RuntimeError(f"invalid production freeze manifest: {path}")
    files = payload.get("frozen_files_sha256")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"freeze manifest has no source hashes: {path}")
    for relative, expected in files.items():
        source = folder / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise RuntimeError(f"frozen production source changed after gate: {relative}")


def allowed_seeds(cfg: dict[str, Any]) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in cfg.get("seeds", ()))
    if seeds != (1, 2, 3, 4, 5):
        raise ValueError(f"config seeds must be exactly (1,2,3,4,5), got {seeds}")
    return seeds


def validate_package(cfg: dict[str, Any]) -> None:
    expected = {
        "run_slug": RUN_SLUG,
        "trainer_method": "b3_support_runtime_opt",
        "support_query_mode": "current_policy",
        "support_assignment_mode": "context_eligible_count_matched",
        "assignment_seed_offset": 30_000,
        "assignment_diagnostic_window_updates": 1_000,
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "total_steps": 1_000_000,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"config {key!r} must equal {value!r}, got {cfg.get(key)!r}")
    allowed_seeds(cfg)


def resolve_python(repo_root: Path) -> Path:
    configured = os.environ.get("PYTHON_BIN", "").strip()
    if configured:
        return Path(configured).expanduser()
    candidate = repo_root / ".venv/bin/python"
    return candidate if candidate.exists() else Path(sys.executable)


def run_name(cfg: dict[str, Any], stage: str, seed: int) -> str:
    spec = STAGES[stage]
    return (
        f"fetch_pick_and_place_{spec['slug']}_s{int(seed)}_"
        f"{int(spec['steps'])}_ls{int(cfg['local_search_max'])}"
    )


def registered_output_root(repo_root: Path, stage: str) -> Path:
    return (repo_root / STAGES[stage]["output"]).resolve()


def resolve_output_root(repo_root: Path, stage: str, requested: str | None) -> Path:
    expected = registered_output_root(repo_root, stage)
    if requested is None:
        return expected
    supplied = Path(requested).expanduser()
    if not supplied.is_absolute():
        supplied = repo_root / supplied
    supplied = supplied.resolve()
    if supplied != expected:
        raise ValueError(f"{stage} must use isolated output root {expected}, got {supplied}")
    return supplied


def matching_targets(output_root: Path, cfg: dict[str, Any], stage: str, seed: int) -> list[Path]:
    name = run_name(cfg, stage, seed)
    return sorted(path for path in output_root.glob(f"{name}*") if path.exists())


def validate_target_policy(
    output_root: Path,
    cfg: dict[str, Any],
    stage: str,
    seed: int,
    *,
    resume: bool,
    validate_only: bool,
) -> None:
    if validate_only:
        return
    matches = matching_targets(output_root, cfg, stage, seed)
    if resume:
        exact = output_root / run_name(cfg, stage, seed)
        if matches != [exact] or not (exact / "checkpoint.pt").is_file():
            raise FileNotFoundError(
                f"resume requires exactly one registered run with checkpoint.pt: {exact}"
            )
    elif matches:
        raise FileExistsError(
            "Refusing to overwrite/reuse an exact or timestamped run: "
            + ", ".join(str(path) for path in matches)
        )


def build_command(
    *,
    stage: str,
    seed: int,
    device: str,
    output_root: str | None,
    validate_only: bool,
    verify_runtime_optimization: bool,
    resume: bool,
    resume_path: str | None,
    enforce_target: bool = True,
) -> list[str]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = load_config(folder)
    validate_package(cfg)
    if stage not in STAGES:
        raise ValueError(f"unsupported stage {stage!r}")
    if seed not in allowed_seeds(cfg):
        raise ValueError(f"seed must be one of {allowed_seeds(cfg)}, got {seed}")
    if stage != "production" and seed != 1:
        raise ValueError(f"{stage} is fixed to seed 1")
    if stage == "production":
        verify_production_freeze(folder)
    if resume_path and not resume:
        raise ValueError("--resume_path requires --resume")
    out_root = resolve_output_root(repo_root, stage, output_root)
    if enforce_target:
        validate_target_policy(
            out_root, cfg, stage, seed, resume=resume, validate_only=validate_only
        )

    trainer = folder / "code/src/train_fetch_pick_and_place_lite_v0.py"
    cmd = [
        str(resolve_python(repo_root)), str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(cfg, stage, seed),
        "--output_root", str(out_root),
        "--method", str(cfg["trainer_method"]),
        "--reward_type", str(cfg["reward_type"]),
        "--max_episode_steps", str(FIXED["max_episode_steps"]),
        "--total_steps", str(STAGES[stage]["steps"]),
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
        "--support_assignment_mode", str(cfg["support_assignment_mode"]),
        "--assignment_diagnostic_window_updates", str(cfg["assignment_diagnostic_window_updates"]),
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
        "--weight_seed", str(20_000 + seed),
        "--assignment_seed", str(int(cfg["assignment_seed_offset"]) + seed),
        "--eval_seed", str(10_000 + seed),
    ]
    cmd.append("--support_cache_enabled" if cfg["support_cache_enabled"] else "--no-support_cache_enabled")
    cmd.append("--local_scan_cache_enabled" if cfg["local_scan_cache_enabled"] else "--no-local_scan_cache_enabled")
    cmd.append("--support_short_circuit" if cfg["support_short_circuit"] else "--no-support_short_circuit")
    cmd.append("--numba_support_scan" if cfg["numba_support_scan"] else "--no-numba_support_scan")
    cmd.append("--support_predicate_early_exit" if cfg["support_predicate_early_exit"] else "--no-support_predicate_early_exit")
    if resume:
        cmd.append("--resume")
        if resume_path:
            cmd.extend(["--resume_path", resume_path])
    if validate_only:
        cmd.append("--validate_only")
    if verify_runtime_optimization:
        cmd.append("--verify_runtime_optimization")
    return cmd


def main() -> int:
    folder = Path(__file__).resolve().parent
    cfg = load_config(folder)
    parser = argparse.ArgumentParser(
        description="Run the FPP DDPG Strict Context-Eligible Count-Matched Control."
    )
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument("--seed", type=int, default=1, choices=allowed_seeds(cfg))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--verify_runtime_optimization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        cmd = build_command(
            stage=args.stage, seed=args.seed, device=args.device,
            output_root=args.output_root, validate_only=args.validate_only,
            verify_runtime_optimization=args.verify_runtime_optimization,
            resume=args.resume, resume_path=args.resume_path,
            enforce_target=not args.dry_run,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0
    print(printable, flush=True)
    env = os.environ.copy()
    source_root = folder / "code/src"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
