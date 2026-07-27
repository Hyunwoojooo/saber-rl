#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "td3_saber_critic_only_v4_ls512"
MAIN_ARTIFACT_CONFIG = (
    HERE.parent
    / "paper_artifacts/fpp_td3_saber_query_ablation_ls512/data/saber/"
    "fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s1_1000000_ls512/"
    "config.json"
)


def load_run_module():
    spec = importlib.util.spec_from_file_location("action_bin_run", HERE / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def normalized_command(command: list[str]) -> list[str]:
    normalized = list(command)
    for option in ("--run_name", "--action_bins_per_dim"):
        index = normalized.index(option)
        del normalized[index : index + 2]
    return normalized


def expect_value_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    module = load_run_module()
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    base_cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    main_artifact_cfg = json.loads(
        MAIN_ARTIFACT_CONFIG.read_text(encoding="utf-8")
    )

    assert cfg["action_bin_variants"] == {"k3": 3, "k7": 7}
    assert cfg["seeds"] == [1, 2, 3, 4, 5]
    assert cfg["total_steps"] == 1_000_000
    assert file_hashes(HERE / "code") == file_hashes(BASE / "code")

    allowed_metadata = {
        "action_bin_variants",
        "actor_lr",
        "buffer_capacity",
        "checkpoint_every_steps",
        "critic_lr",
        "folder",
        "gamma",
        "hidden_dim",
        "l2_action_penalty",
        "log_every_steps",
        "max_episode_steps",
        "noise_std",
        "output_root",
        "policy_action_radius",
        "purpose",
        "reference",
        "run_slug",
        "save_step_checkpoints",
        "source_spec",
        "start_steps",
        "success_terminates",
        "tau",
        "timeout_bootstrap",
        "warmup_steps",
    }
    for key in set(cfg) & set(base_cfg):
        if key not in allowed_metadata:
            assert cfg[key] == base_cfg[key], (key, cfg[key], base_cfg[key])

    expected_defaults = {
        "actor_lr": 3e-4,
        "buffer_capacity": 300_000,
        "checkpoint_every_steps": 50_000,
        "critic_lr": 3e-4,
        "gamma": 0.98,
        "hidden_dim": 256,
        "l2_action_penalty": 0.0,
        "log_every_steps": 1_000,
        "max_episode_steps": 50,
        "noise_std": 0.2,
        "policy_action_radius": 0.35,
        "start_steps": 10_000,
        "tau": 0.005,
        "warmup_steps": 5_000,
    }
    for key, value in expected_defaults.items():
        assert cfg[key] == value, (key, cfg[key], value)

    resolved_k3 = module.resolve_variant(cfg, "k3")
    runtime_parity_keys = (
        "max_episode_steps",
        "total_steps",
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
        "noise_std",
        "l2_action_penalty",
        "td3_policy_delay",
        "td3_target_policy_noise",
        "td3_target_noise_clip",
        "eval_freq",
        "eval_episodes",
        "checkpoint_every_steps",
        "log_every_steps",
        "M_B",
        "M_Q_min",
        "N_local_min",
        "N_gap_min",
        "local_search_max",
        "policy_support_mode",
        "policy_action_radius",
        "support_scan_mode",
        "support_cache_enabled",
        "support_cache_max_entries",
        "local_scan_cache_enabled",
        "local_scan_cache_max_entries",
        "support_short_circuit",
        "numba_support_scan",
        "support_predicate_early_exit",
        "support_diagnostic_audit_interval",
        "lambda_support",
        "per_alpha",
        "eps_PER",
        "U_PER_max",
        "weighting_mode",
        "weight_scope",
        "save_step_checkpoints",
        "success_terminates",
        "timeout_bootstrap",
    )
    for key in runtime_parity_keys:
        assert resolved_k3[key] == main_artifact_cfg[key], (
            key,
            resolved_k3[key],
            main_artifact_cfg[key],
        )

    shared_output = "/tmp/saber-action-bin-contract"
    k3_cmd, k3_cfg = module.build_command(
        "k3", 1, None, "cpu", shared_output, False, False
    )
    k7_cmd, k7_cfg = module.build_command(
        "k7", 1, None, "cpu", shared_output, False, False
    )
    assert k3_cfg["action_bins_per_dim"] == 3
    assert k7_cfg["action_bins_per_dim"] == 7
    assert option_value(k3_cmd, "--action_bins_per_dim") == "3"
    assert option_value(k7_cmd, "--action_bins_per_dim") == "7"
    assert normalized_command(k3_cmd) == normalized_command(k7_cmd)

    for command in (k3_cmd, k7_cmd):
        assert option_value(command, "--method") == "td3_saber_runtime_opt"
        assert option_value(command, "--total_steps") == "1000000"
        assert option_value(command, "--her_ratio") == "0.8"
        assert option_value(command, "--updates_per_step") == "1"
        assert option_value(command, "--weight_scope") == "critic_only"
        assert option_value(command, "--weighting_mode") == "batch_norm"
        assert option_value(command, "--local_search_max") == "512"
        assert option_value(command, "--td3_policy_delay") == "2"

    module.validate_run_request(
        1, 1_000_000, smoke=False, validate_only=False, dry_run=False
    )
    expect_value_error(
        lambda: module.validate_run_request(
            6, 1_000_000, smoke=False, validate_only=False, dry_run=False
        )
    )
    expect_value_error(
        lambda: module.validate_run_request(
            1, 999_999, smoke=False, validate_only=False, dry_run=False
        )
    )
    print("action-bin sensitivity contract: PASS")


if __name__ == "__main__":
    main()
