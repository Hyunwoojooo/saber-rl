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


CONDITIONS = ("probe", "diverse", "random_count")
PROBE_THRESHOLDS = (2, 4, 8)
PRODUCTION_STEPS = 1_000_000
PROBE_STEPS = 50_000


def repo_root_from(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists() and (parent / "experiment_FPP").is_dir():
            return parent
        if (parent / "version_v0_lite").exists() and (parent / "SABER").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {path}")


def load_config(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def load_selected_threshold(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("success_metrics_used") is not False:
        raise ValueError("threshold selection must explicitly record success_metrics_used=false")
    selected = int(payload["selected_dmin"])
    candidates = tuple(int(value) for value in payload.get("candidate_thresholds", ()))
    if candidates != PROBE_THRESHOLDS:
        raise ValueError(
            f"threshold candidates must be {PROBE_THRESHOLDS}, got {candidates}"
        )
    if selected not in PROBE_THRESHOLDS:
        raise ValueError(f"selected_dmin must be one of {PROBE_THRESHOLDS}, got {selected}")
    if int(payload.get("window_start_step", -1)) != 20_000:
        raise ValueError("threshold selection must use window_start_step=20000")
    if int(payload.get("window_end_step", -1)) != 50_000:
        raise ValueError("threshold selection must use window_end_step=50000")
    return selected


def resolve_condition_config(
    base: dict[str, Any],
    condition: str,
    selected_threshold_path: Path | None,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    cfg = copy.deepcopy(base)

    if condition == "probe":
        cfg.update(
            {
                "total_steps": PROBE_STEPS,
                "run_slug": "td3_saber_diversity_probe_d1_50k",
                "output_root": "experiment_FPP/runs/td3_saber_diversity_probe_d1_50k",
                "support_min_unique_action_bins": 1,
                "support_assignment_mode": "support",
                "support_diversity_probe_thresholds": list(PROBE_THRESHOLDS),
                "support_predicate_early_exit": False,
                "support_short_circuit": False,
            }
        )
        allowed_changes = {
            "total_steps",
            "run_slug",
            "output_root",
            "support_diversity_probe_thresholds",
            "support_predicate_early_exit",
            "support_short_circuit",
        }
    else:
        if selected_threshold_path is None:
            raise ValueError(
                f"{condition} requires --selected-threshold from the 50k probe"
            )
        selected = load_selected_threshold(selected_threshold_path)
        assignment = "support" if condition == "diverse" else "random_count"
        slug = (
            f"td3_saber_diverse_d{selected}_critic_only_v4"
            if condition == "diverse"
            else f"td3_saber_random_diversity_count_d{selected}_critic_only_v4"
        )
        cfg.update(
            {
                "total_steps": PRODUCTION_STEPS,
                "run_slug": slug,
                "output_root": f"experiment_FPP/runs/{slug}_ls512",
                "support_min_unique_action_bins": selected,
                "support_assignment_mode": assignment,
                "support_diversity_probe_thresholds": [],
                "support_predicate_early_exit": True,
            }
        )
        allowed_changes = {
            "run_slug",
            "output_root",
            "support_min_unique_action_bins",
        }
        if condition == "random_count":
            allowed_changes.add("support_assignment_mode")

    changed = {
        key
        for key in set(base) | set(cfg)
        if base.get(key) != cfg.get(key)
    }
    unexpected = changed - allowed_changes
    if unexpected:
        raise RuntimeError(
            f"condition {condition} changed non-allowlisted config keys: {sorted(unexpected)}"
        )
    return cfg


def run_name(cfg: dict[str, Any], seed: int, steps: int) -> str:
    return (
        f"fetch_pick_and_place_{cfg['run_slug']}_"
        f"s{int(seed)}_{int(steps)}_ls{int(cfg['local_search_max'])}"
    )


def validate_run_request(
    condition: str,
    seed: int,
    steps: int | None,
    *,
    smoke: bool,
    validate_only: bool,
    dry_run: bool,
) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    noncanonical_allowed = bool(smoke or validate_only or dry_run)
    expected_steps = PROBE_STEPS if condition == "probe" else PRODUCTION_STEPS
    requested_steps = expected_steps if steps is None else int(steps)
    if requested_steps != expected_steps and not noncanonical_allowed:
        raise ValueError(
            f"{condition} must run exactly {expected_steps} steps; "
            "noncanonical --steps is allowed only for smoke/validate/dry-run"
        )
    valid_seeds = {1} if condition == "probe" else set(range(1, 6))
    if int(seed) not in valid_seeds and not noncanonical_allowed:
        raise ValueError(
            f"{condition} requires seed in {sorted(valid_seeds)}; "
            "noncanonical seeds are allowed only for smoke/validate/dry-run"
        )


def build_command(
    condition: str,
    seed: int,
    steps: int | None,
    device: str,
    output_root: str | None,
    selected_threshold_path: Path | None,
    smoke: bool,
    validate_only: bool,
) -> tuple[list[str], dict[str, Any]]:
    folder = Path(__file__).resolve().parent
    repo_root = repo_root_from(folder)
    base = load_config(folder)
    cfg = resolve_condition_config(base, condition, selected_threshold_path)
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
        "--action_bins_per_dim", str(cfg.get("action_bins_per_dim", 5)),
        "--support_min_unique_action_bins",
        str(cfg.get("support_min_unique_action_bins", 1)),
        "--support_assignment_mode",
        str(cfg.get("support_assignment_mode", "support")),
        "--lambda_support", str(cfg["lambda_support"]),
        "--per_alpha", str(cfg["per_alpha"]),
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries",
        str(cfg.get("support_cache_max_entries", 200000)),
        "--local_scan_cache_max_entries",
        str(cfg.get("local_scan_cache_max_entries", 50000)),
        "--td3_policy_delay", str(cfg.get("td3_policy_delay", 2)),
        "--td3_target_policy_noise",
        str(cfg.get("td3_target_policy_noise", 0.2)),
        "--td3_target_noise_clip",
        str(cfg.get("td3_target_noise_clip", 0.5)),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + int(seed)),
        "--eval_seed", str(10000 + int(seed)),
    ]
    probe_thresholds = tuple(
        int(value) for value in cfg.get("support_diversity_probe_thresholds", ())
    )
    if probe_thresholds:
        cmd.append("--support_diversity_probe_thresholds")
        cmd.extend(str(value) for value in probe_thresholds)

    cmd.append(
        "--support_cache_enabled"
        if cfg.get("support_cache_enabled", True)
        else "--no-support_cache_enabled"
    )
    cmd.append(
        "--local_scan_cache_enabled"
        if cfg.get("local_scan_cache_enabled", True)
        else "--no-local_scan_cache_enabled"
    )
    cmd.append(
        "--support_short_circuit"
        if cfg.get("support_short_circuit", True)
        else "--no-support_short_circuit"
    )
    cmd.append(
        "--numba_support_scan"
        if cfg.get("numba_support_scan", True)
        else "--no-numba_support_scan"
    )
    cmd.append(
        "--support_predicate_early_exit"
        if cfg.get("support_predicate_early_exit", True)
        else "--no-support_predicate_early_exit"
    )
    cmd.extend(
        [
            "--support_diagnostic_audit_interval",
            str(cfg.get("support_diagnostic_audit_interval", 0)),
        ]
    )
    if smoke:
        cmd.append("--smoke")
    if validate_only:
        cmd.append("--validate_only")
    return cmd, cfg


def require_numba_environment(python_bin: str, env: dict[str, str]) -> None:
    result = subprocess.run(
        [python_bin, "-c", "import numba"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        final_line = detail[-1] if detail else "numba import failed"
        raise RuntimeError(
            "numba_support_scan=true but the selected Python environment cannot "
            f"import numba ({final_line}). Install requirements-numba.txt in "
            "a compatible environment or target and expose it through PYTHONPATH."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FetchPickAndPlace TD3 SABER action-diversity ablation."
    )
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--selected-threshold", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    args = parser.parse_args()

    validate_run_request(
        args.condition,
        args.seed,
        args.steps,
        smoke=args.smoke,
        validate_only=args.validate_only,
        dry_run=args.dry_run,
    )

    cmd, cfg = build_command(
        condition=args.condition,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        output_root=args.output_root,
        selected_threshold_path=args.selected_threshold,
        smoke=args.smoke,
        validate_only=args.validate_only,
    )
    if args.print_resolved_config:
        print(json.dumps(cfg, indent=2, sort_keys=True))
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0

    print(printable, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent / "code" / "src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    if cfg.get("numba_support_scan", True):
        require_numba_environment(cmd[0], env)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
