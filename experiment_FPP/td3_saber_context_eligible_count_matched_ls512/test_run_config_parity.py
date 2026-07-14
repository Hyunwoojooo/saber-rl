#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("query_ablation_run", FOLDER / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)

STAGES = ("smoke", "gate", "production")
MAIN_PARITY_OPTIONS = {
    "--method": "method",
    "--reward_type": "reward_type",
    "--max_episode_steps": "max_episode_steps",
    "--total_steps": "total_steps",
    "--start_steps": "start_steps",
    "--warmup_steps": "warmup_steps",
    "--batch_size": "batch_size",
    "--buffer_capacity": "buffer_capacity",
    "--her_ratio": "her_ratio",
    "--updates_per_step": "updates_per_step",
    "--gamma": "gamma",
    "--tau": "tau",
    "--actor_lr": "actor_lr",
    "--critic_lr": "critic_lr",
    "--hidden_dim": "hidden_dim",
    "--noise_std": "noise_std",
    "--eval_freq": "eval_freq",
    "--eval_episodes": "eval_episodes",
    "--M_B": "M_B",
    "--M_Q_min": "M_Q_min",
    "--N_local_min": "N_local_min",
    "--N_gap_min": "N_gap_min",
    "--local_search_max": "local_search_max",
    "--action_bins_per_dim": "action_bins_per_dim",
    "--lambda_support": "lambda_support",
    "--weighting_mode": "weighting_mode",
    "--weight_scope": "weight_scope",
    "--policy_support_mode": "policy_support_mode",
    "--support_scan_mode": "support_scan_mode",
    "--td3_policy_delay": "td3_policy_delay",
    "--td3_target_policy_noise": "td3_target_policy_noise",
    "--td3_target_noise_clip": "td3_target_noise_clip",
}


def command_options(command: list[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise AssertionError(f"unexpected command token: {token}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[token] = command[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return options


def build(stage: str, seed: int) -> dict[str, str | bool]:
    command = RUN.build_command(
        stage=stage,
        seed=seed,
        device="cuda",
        output_root=None,
        validate_only=False,
    )
    return command_options(command)


def load_main_config(repo_root: Path, seed: int) -> dict[str, object]:
    root = repo_root / "experiment_FPP" / "runs" / "td3_saber_critic_only_v4_ls512"
    matches = sorted(root.glob(f"*s{seed}_1000000_ls512/config.json"))
    assert len(matches) == 1, (seed, matches)
    return json.loads(matches[0].read_text())


def assert_main_parity(
    options: dict[str, str | bool],
    main_config: dict[str, object],
    seed: int,
) -> None:
    for option, config_key in MAIN_PARITY_OPTIONS.items():
        assert str(options[option]) == str(main_config[config_key]), (
            option,
            options[option],
            main_config[config_key],
        )
    for option, config_key in (
        ("--support_cache_enabled", "support_cache_enabled"),
        ("--local_scan_cache_enabled", "local_scan_cache_enabled"),
        ("--support_short_circuit", "support_short_circuit"),
        ("--numba_support_scan", "numba_support_scan"),
        ("--support_predicate_early_exit", "support_predicate_early_exit"),
    ):
        assert bool(options[option]) is bool(main_config[config_key])
    main_seeds = dict(main_config["seeds"])
    for name, offset in (
        ("env_seed", 0),
        ("torch_seed", 0),
        ("numpy_seed", 0),
        ("action_noise_seed", 0),
        ("weight_seed", 20_000),
        ("eval_seed", 10_000),
    ):
        assert int(main_seeds[name]) == seed + offset
        assert options[f"--{name}"] == str(seed + offset)


def main() -> int:
    cfg = RUN.load_config(FOLDER)
    repo_root = RUN.repo_root_from(FOLDER)
    assert tuple(cfg["stage_steps"]) == STAGES
    for seed in cfg["seeds"]:
        options = build("production", seed)
        assert_main_parity(options, load_main_config(repo_root, seed), seed)
        assert options["--support_query_mode"] == "current_policy"
        assert (
            options["--support_assignment_mode"]
            == "context_eligible_count_matched"
        )
        assert options["--assignment_seed"] == str(30_000 + seed)
        assert options["--assignment_diagnostic_window_updates"] == "1000"
        assert "--smoke" not in options
        expected_root = str(repo_root / cfg["stage_output_roots"]["production"])
        assert options["--output_root"] == expected_root

    smoke = build("smoke", 1)
    assert smoke["--total_steps"] == "5000"
    assert smoke["--smoke"] is True
    gate = build("gate", 1)
    assert gate["--total_steps"] == "50000"
    assert "--smoke" not in gate

    resume_command = RUN.build_command(
        stage="gate",
        seed=1,
        device="cuda",
        output_root=None,
        validate_only=False,
        resume=True,
        resume_path="checkpoint.pt",
    )
    resume_options = command_options(resume_command)
    assert resume_options["--resume"] is True
    assert resume_options["--resume_path"] == "checkpoint.pt"

    print("test_run_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
