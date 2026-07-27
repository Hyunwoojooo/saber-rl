from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PRODUCTION_STEPS = 1_000_000
PRODUCTION_SEEDS = (1, 2, 3, 4, 5)


def repo_root_from(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists() and (parent / "experiment_FPP").is_dir():
            return parent
        if (parent / "version_v0_lite").exists() and (parent / "SABER").exists():
            return parent / "SABER"
    raise RuntimeError(f"Could not find repository root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def resolve_variant(base: dict[str, Any], variant: str) -> dict[str, Any]:
    variants = base.get("action_bin_variants")
    if not isinstance(variants, dict) or variant not in variants:
        choices = sorted(variants) if isinstance(variants, dict) else []
        raise ValueError(f"variant must be one of {choices}, got {variant!r}")
    cfg = copy.deepcopy(base)
    cfg.pop("action_bin_variants")
    bins = int(variants[variant])
    cfg["variant"] = variant
    cfg["action_bins_per_dim"] = bins
    cfg["run_slug"] = f"td3_saber_action_bins_{bins}_critic_only_v4_ls512"
    cfg["output_root"] = (
        "experiment_FPP/runs/"
        f"td3_saber_action_bin_sensitivity_ls512_1m/{variant}"
    )
    return cfg


def run_name(cfg: dict[str, Any], seed: int, steps: int) -> str:
    return (
        f"fetch_pick_and_place_{cfg['run_slug']}_"
        f"s{int(seed)}_{int(steps)}_ls{int(cfg['local_search_max'])}"
    )


def validate_run_request(
    seed: int,
    steps: int | None,
    *,
    smoke: bool,
    validate_only: bool,
    dry_run: bool,
) -> None:
    noncanonical_allowed = bool(smoke or validate_only or dry_run)
    requested_steps = PRODUCTION_STEPS if steps is None else int(steps)
    if requested_steps != PRODUCTION_STEPS and not noncanonical_allowed:
        raise ValueError(
            f"production runs require exactly {PRODUCTION_STEPS} steps; "
            "noncanonical --steps is allowed only with --smoke, "
            "--validate_only, or --dry-run"
        )
    if int(seed) not in PRODUCTION_SEEDS and not noncanonical_allowed:
        raise ValueError(
            f"production seed must be one of {list(PRODUCTION_SEEDS)}; "
            "noncanonical seeds are allowed only for validation"
        )


def python_executable(repo_root: Path) -> Path:
    override = os.environ.get("SABER_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    candidate = repo_root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def resolved_output_root(repo_root: Path, configured: str, override: str | None) -> str:
    path = Path(configured if override is None else override).expanduser()
    return str(path if path.is_absolute() else repo_root / path)


def bool_flag(cmd: list[str], name: str, enabled: bool) -> None:
    cmd.append(f"--{name}" if enabled else f"--no-{name}")


def build_command(
    variant: str,
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    smoke: bool,
    validate_only: bool,
) -> tuple[list[str], dict[str, Any]]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    cfg = resolve_variant(load_config(folder), variant)
    source_root = folder / "code" / "src"
    trainer = source_root / "train_fetch_pick_and_place_lite_v0.py"
    python_bin = python_executable(repo_root)
    n_steps = int(cfg["total_steps"] if steps is None else steps)
    out_root = resolved_output_root(repo_root, cfg["output_root"], output_root)

    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", str(cfg["task_name"]),
        "--run_name", run_name(cfg, seed, n_steps),
        "--output_root", out_root,
        "--method", str(cfg["trainer_method"]),
        "--reward_type", str(cfg["reward_type"]),
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
        "--l2_action_penalty", str(cfg["l2_action_penalty"]),
        "--td3_policy_delay", str(cfg["td3_policy_delay"]),
        "--td3_target_policy_noise", str(cfg["td3_target_policy_noise"]),
        "--td3_target_noise_clip", str(cfg["td3_target_noise_clip"]),
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
        "--policy_support_mode", str(cfg["policy_support_mode"]),
        "--policy_action_radius", str(cfg["policy_action_radius"]),
        "--support_scan_mode", str(cfg["support_scan_mode"]),
        "--support_cache_max_entries", str(cfg["support_cache_max_entries"]),
        "--local_scan_cache_max_entries", str(cfg["local_scan_cache_max_entries"]),
        "--support_diagnostic_audit_interval",
        str(cfg["support_diagnostic_audit_interval"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--per_alpha", str(cfg["per_alpha"]),
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", str(cfg["weighting_mode"]),
        "--weight_scope", str(cfg["weight_scope"]),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20_000 + int(seed)),
        "--eval_seed", str(10_000 + int(seed)),
    ]
    bool_flag(cmd, "support_cache_enabled", bool(cfg["support_cache_enabled"]))
    bool_flag(
        cmd,
        "local_scan_cache_enabled",
        bool(cfg["local_scan_cache_enabled"]),
    )
    bool_flag(cmd, "support_short_circuit", bool(cfg["support_short_circuit"]))
    bool_flag(cmd, "numba_support_scan", bool(cfg["numba_support_scan"]))
    bool_flag(
        cmd,
        "support_predicate_early_exit",
        bool(cfg["support_predicate_early_exit"]),
    )
    if not bool(cfg["save_step_checkpoints"]):
        cmd.append("--no_save_step_checkpoints")
    if bool(cfg["success_terminates"]):
        cmd.append("--success_terminates")
    if not bool(cfg["timeout_bootstrap"]):
        cmd.append("--no_timeout_bootstrap")
    if smoke:
        cmd.append("--smoke")
    if validate_only:
        cmd.append("--validate_only")
    return cmd, cfg


def require_numba(python_bin: str, env: dict[str, str]) -> None:
    result = subprocess.run(
        [python_bin, "-c", "import numba"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Numba import failed"
        raise RuntimeError(
            "This frozen experiment requires Numba support scanning; "
            f"selected interpreter {python_bin!r} cannot import numba: {detail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FPP TD3 SABER action-bin sensitivity (K=3 or K=7)."
    )
    parser.add_argument("--variant", required=True, choices=("k3", "k7"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_run_request(
        args.seed,
        args.steps,
        smoke=args.smoke,
        validate_only=args.validate_only,
        dry_run=args.dry_run,
    )
    cmd, cfg = build_command(
        variant=args.variant,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        smoke=args.smoke,
        validate_only=args.validate_only,
    )
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0

    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent / "code" / "src"
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    if bool(cfg["numba_support_scan"]):
        require_numba(cmd[0], env)
    print(printable, flush=True)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
