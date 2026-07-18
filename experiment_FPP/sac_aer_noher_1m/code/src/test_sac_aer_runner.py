#!/usr/bin/env python3
"""Frozen runner contract checks for FPP SAC ER/AnnealingER/AER."""

from __future__ import annotations

import importlib.util
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
RUNNER_PATH = EXPERIMENT_DIR / "run.py"
SPEC = importlib.util.spec_from_file_location("sac_aer_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {RUNNER_PATH}")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_frozen_three_arm_commands() -> None:
    expected = {
        "sac_er": "sac_er",
        "sac_annealing_er": "sac_annealing_er",
        "sac_aer": "sac_aer",
    }
    for arm, method in expected.items():
        command = runner.build_command(
            arm_name=arm,
            seed=101,
            steps=20_000,
            device="cpu",
            output_root=None,
            validate_only=False,
        )
        assert _option(command, "--method") == method
        assert _option(command, "--her_ratio") == "0.0"
        assert _option(command, "--sac_alpha") == "0.2"
        assert _option(command, "--sac_alpha_lr") == "0.0003"
        assert "--sac_auto_alpha" in command
        assert "--sac_target_entropy" not in command
        assert _option(command, "--aer_lambda0") == "2.5"
        assert _option(command, "--aer_candidate_rounding") == "ceil"
        assert "--no-support_cache_enabled" in command
        assert "--no-local_scan_cache_enabled" in command
        assert "--no-support_short_circuit" in command
        assert "--no-numba_support_scan" in command
        assert "--no-support_predicate_early_exit" in command
        assert "--no_save_step_checkpoints" in command
        assert not any(token.startswith("--td3_") for token in command)
        expected_name = f"fetch_pick_and_place_{arm}_noher_s101_20000"
        assert _option(command, "--run_name") == expected_name


def test_protocol_freezes_temperature_schedule() -> None:
    config = runner.load_config()
    assert config["common"]["sac_auto_alpha"] is True
    assert config["common"]["sac_target_entropy"] is None
    assert config["common"]["sac_alpha_lr"] == 3.0e-4
    assert config["aer"]["learning_rate_targets"] == ["actor", "critic"]
    assert config["aer"]["temperature_learning_rate_schedule"] == "fixed"
    assert config["aer"]["query_state"] == "current_online_state"
    assert config["aer"]["selected_batch_size"] == config["common"]["batch_size"]
    assert config["aer"]["annealing_horizon"] == "alpha_times_run_total_steps"
    assert config["aer"]["learning_rate_scale"] == "lambda_t"


def test_out_of_phase_requests_fail_closed() -> None:
    config = runner.load_config()
    for steps, seed in ((20_000, 999), (21_000, 101), (1_000_000, 101)):
        try:
            runner.validate_protocol(config, "sac_aer", seed, steps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"request unexpectedly accepted: steps={steps}, seed={seed}")


def main() -> int:
    for test in (
        test_frozen_three_arm_commands,
        test_protocol_freezes_temperature_schedule,
        test_out_of_phase_requests_fail_closed,
    ):
        test()
    print("test_sac_aer_runner.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
