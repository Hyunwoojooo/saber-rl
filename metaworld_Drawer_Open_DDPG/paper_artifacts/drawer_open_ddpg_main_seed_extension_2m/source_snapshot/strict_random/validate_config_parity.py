#!/usr/bin/env python3
"""Generate a full recursive Main/control config diff with a strict allowlist."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import run as experiment_run


EXPERIMENT_DIR = Path(__file__).resolve().parent
DRAWER_ROOT = EXPERIMENT_DIR.parent
MAIN_SCRIPT = DRAWER_ROOT / "DDPG-HER-B3-Critic-Only-2M.sh"
CODE_SRC = EXPERIMENT_DIR / "code" / "src"

# Every other recursive config path must compare exactly.
ALLOWED_PATHS = {
    "method",
    "control_mode",
    "run_name",
    "output_root",
    "support_assignment_mode",
    "assignment_diagnostic_window_updates",
    "seeds.assignment_seed",
}

# Canonicalized values from DDPG-HER-B3-Critic-Only-2M.sh. Comparing Main and
# control to each other is insufficient if a shared runner field drifts, so the
# generated Main config must also satisfy this full learning/evaluation contract.
CANONICAL_MAIN_EXPECTED = {
    "task_name": "drawer_open",
    "reward_type": "sparse",
    "metaworld_env_name": "drawer-open-v3",
    "metaworld_reward_function_version": "v2",
    "max_episode_steps": 200,
    "total_steps": 2_000_000,
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "updates_per_step": 1,
    "batch_size": 256,
    "buffer_capacity": 300_000,
    "her_ratio": 0.8,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "l2_action_penalty": 0.0,
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "checkpoint_every_steps": 50_000,
    "save_step_checkpoints": False,
    "log_every_steps": 1_000,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "N_gap_min": 128,
    "local_search_max": 512,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "drawer_success_threshold": 0.05,
    "drawer_d_reach": 0.08,
    "drawer_d_goal_near": 0.05,
    "drawer_d_opening_mid": 0.12,
    "weighting_mode": "batch_norm",
    "weight_scope": "critic_only",
    "policy_support_mode": "bin",
    "support_query_mode": "current_policy",
    "support_assignment_mode": "support",
    "support_scan_mode": "deterministic_tail_v1",
    "support_cache_enabled": True,
    "support_cache_max_entries": 200_000,
    "local_scan_cache_enabled": True,
    "local_scan_cache_max_entries": 50_000,
    "support_short_circuit": True,
    "numba_support_scan": True,
    "support_predicate_early_exit": True,
    "support_diagnostic_audit_interval": 100,
    "method": "b3_support_runtime_opt",
    "success_terminates": False,
    "timeout_bootstrap": True,
    "device": "cuda",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_train_module():
    sys.path.insert(0, str(CODE_SRC))
    try:
        return importlib.import_module("train_drawer_open_lite_v0")
    finally:
        sys.path.pop(0)


def _parse_config(train_module: Any, cli_args: list[str]) -> dict[str, Any]:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(experiment_run.TRAIN), *cli_args]
        namespace = train_module.parse_args()
    finally:
        sys.argv = old_argv
    config = train_module.config_from_args(namespace)
    # Backward-compatible with the pre-transplant parser while making these
    # convention fields auditable once they are explicit dataclass members.
    for name in (
        "numba_support_scan",
        "support_predicate_early_exit",
        "support_diagnostic_audit_interval",
    ):
        if hasattr(namespace, name):
            setattr(config, name, getattr(namespace, name))
    config.normalize()
    return config.as_dict()


def _main_cli(mode: experiment_run.Mode, seed: int) -> list[str]:
    return [
        *experiment_run.common_args(mode, seed),
        *experiment_run.saber_args(),
        "--method", "b3_support_runtime_opt",
        "--weight_scope", "critic_only",
        "--run_name", f"drawer_open_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda",
        "--output_root", str(DRAWER_ROOT / "results_2m" / "DDPG-HER-B3-Critic-Only-2M"),
    ]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], path))
        return flattened
    return {prefix: value}


def _diff(main: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    left = _flatten(main)
    right = _flatten(control)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(left) | set(right)):
        main_value = left.get(path, "<missing>")
        control_value = right.get(path, "<missing>")
        if main_value != control_value:
            rows.append(
                {
                    "path": path,
                    "main": main_value,
                    "control": control_value,
                    "allowed": path in ALLOWED_PATHS,
                }
            )
    return rows


def _canonical_main_failures(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    failures = []
    for path, expected in CANONICAL_MAIN_EXPECTED.items():
        actual = config.get(path, "<missing>")
        if actual != expected:
            failures.append({"path": path, "expected": expected, "actual": actual})
    seed_expected = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    seed_config = config.get("seeds", {})
    if not isinstance(seed_config, dict):
        failures.append(
            {"path": "seeds", "expected": seed_expected, "actual": seed_config}
        )
        return failures
    for key, expected in seed_expected.items():
        actual = seed_config.get(key, "<missing>")
        if actual != expected:
            failures.append(
                {"path": f"seeds.{key}", "expected": expected, "actual": actual}
            )
    return failures


def _canonical_source_checks(text: str) -> dict[str, bool]:
    """Guard against silently comparing to a hand-edited non-Main recipe."""
    required = {
        "main_method": '"b3_support_runtime_opt" "critic_only"',
        "steps": '--total_steps "$STEPS"',
        "horizon": "--max_episode_steps 200",
        "replay": "--buffer_capacity 300000",
        "batch": "--batch_size 256",
        "her": "--her_ratio 0.8",
        "eval_grid": "--eval_freq 10000",
        "eval_episodes": "--eval_episodes 20",
        "M_B": "--M_B 16",
        "M_Q_min": "--M_Q_min 4",
        "N_local_min": "--N_local_min 4",
        "local_search_max": "--local_search_max 512",
        "lambda_support": "--lambda_support 1.0",
        "weight_scope": 'run_variant "DDPG+HER+B3-Critic-Only 2M"',
    }
    return {name: snippet in text for name, snippet in required.items()}


def build_report() -> dict[str, Any]:
    train_module = _load_train_module()
    mode = experiment_run.MODES["main2m"]
    source_text = MAIN_SCRIPT.read_text(encoding="utf-8")
    source_checks = _canonical_source_checks(source_text)
    seed_reports: list[dict[str, Any]] = []
    all_nonallowed: list[dict[str, Any]] = []
    all_canonical_failures: list[dict[str, Any]] = []
    for seed in mode.seeds:
        main_config = _parse_config(train_module, _main_cli(mode, seed))
        control_config = _parse_config(
            train_module,
            experiment_run.train_args(mode, seed),
        )
        differences = _diff(main_config, control_config)
        nonallowed = [row for row in differences if not row["allowed"]]
        canonical_failures = _canonical_main_failures(main_config, seed)
        all_nonallowed.extend({"seed": seed, **row} for row in nonallowed)
        all_canonical_failures.extend(
            {"seed": seed, **row} for row in canonical_failures
        )
        seed_reports.append(
            {
                "seed": seed,
                "passed": not nonallowed and not canonical_failures,
                "differences": differences,
                "nonallowed_differences": nonallowed,
                "canonical_main_failures": canonical_failures,
            }
        )
    return {
        "schema_version": 1,
        "main_source": str(MAIN_SCRIPT.relative_to(experiment_run.REPO_ROOT)),
        "main_source_sha256": _sha256(MAIN_SCRIPT),
        "canonical_source_checks": source_checks,
        "allowed_paths": sorted(ALLOWED_PATHS),
        "seeds": seed_reports,
        "nonallowed_differences": all_nonallowed,
        "canonical_main_expected": CANONICAL_MAIN_EXPECTED,
        "canonical_main_failures": all_canonical_failures,
        "passed": (
            all(source_checks.values())
            and not all_nonallowed
            and not all_canonical_failures
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "config_parity.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise RuntimeError("Main/control config parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
