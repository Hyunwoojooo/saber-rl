#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[2]


def load_runner():
    spec = importlib.util.spec_from_file_location("td3_hgr_runner", EXPERIMENT / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def main() -> int:
    runner = load_runner()
    cfg = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    command = runner.build_command(
        seed=3,
        steps=5_000,
        device="cpu",
        output_root="/tmp/fpp_td3_hgr_runner_test",
        validate_only=False,
    )
    assert value_after(command, "--method") == "td3_hgr"
    assert value_after(command, "--her_ratio") == "0.8"
    assert value_after(command, "--td3_policy_delay") == "2"
    assert value_after(command, "--td3_target_policy_noise") == "0.2"
    assert value_after(command, "--td3_target_noise_clip") == "0.5"
    assert value_after(command, "--hgr_alpha_episode") == "0.6"
    assert value_after(command, "--hgr_alpha_goal") == "0.6"
    assert value_after(command, "--hgr_td3_priority_reduction") == "min_abs"
    assert value_after(command, "--env_seed") == "3"
    assert value_after(command, "--eval_seed") == "10003"
    assert value_after(command, "--weight_seed") == "20003"
    assert "--no_save_step_checkpoints" in command
    assert "--smoke" not in command
    assert runner.run_name(cfg, 3, 5_000) == "fetch_pick_and_place_td3_hgr_tdpriority_s3_5000"

    try:
        runner.build_command(1, 5_001, "cpu", None, False)
    except ValueError as exc:
        assert "multiple of episode horizon" in str(exc)
    else:
        raise AssertionError("non-boundary step count was accepted")

    print("TD3-HGR frozen runner protocol tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
