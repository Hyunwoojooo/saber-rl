#!/usr/bin/env python3
from __future__ import annotations

import copy
import random
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from metacog_lite_v0.assignment_diagnostics import (
    StrictAssignmentDiagnosticsWindow,
)
from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    GLOBAL_RNG_STATE_FIELDS,
    capture_global_rng_state,
    checkpoint_config_fingerprint,
    restore_global_rng_state,
    validate_resume_output_tails,
    validate_resume_payload,
)
from train_drawer_open_lite_v0 import config_from_args, parse_args


def strict_config(**overrides: object) -> LiteConfig:
    values = {
        "task_name": "drawer_open",
        "method": "b3_support_runtime_opt",
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "support_query_mode": "current_policy",
        "support_assignment_mode": (
            "context_eligible_count_matched"
        ),
        "seeds": SeedConfig(assignment_seed=30_001),
    }
    values.update(overrides)
    config = LiteConfig(**values)
    config.normalize()
    return config


def minimal_payload(config: LiteConfig) -> dict[str, object]:
    fingerprint = checkpoint_config_fingerprint(config)
    global_states = capture_global_rng_state()
    diagnostic_state = StrictAssignmentDiagnosticsWindow(
        1_000
    ).state_dict()
    diagnostic_state["last_update"] = 3
    rng_fields = list(GLOBAL_RNG_STATE_FIELDS) + [
        "noise_rng_state",
        "sample_rng_state",
        "weight_rng_state",
        "assignment_rng_state",
    ]
    serialized_config = {
        **config.as_dict(),
        "thread_hint": {"OMP_NUM_THREADS": "1"},
        "terminal_convention": {
            "success_terminates": config.success_terminates,
            "timeout_bootstrap": config.timeout_bootstrap,
        },
    }
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_metadata": {
            "global_step": 200,
            "total_updates": 3,
            "at_episode_boundary": True,
            "config_fingerprint": fingerprint,
        },
        "config": serialized_config,
        "config_fingerprint": fingerprint,
        "global_step": 200,
        "episode_idx": 1,
        "total_updates": 3,
        "agent": {},
        "replay": {},
        "gap_stats": {},
        "noise_rng_state": np.random.default_rng(1).bit_generator.state,
        "sample_rng_state": np.random.default_rng(2).bit_generator.state,
        "weight_rng_state": np.random.default_rng(3).bit_generator.state,
        "assignment_rng_state": (
            np.random.default_rng(4).bit_generator.state
        ),
        "assignment_diagnostics_state": diagnostic_state,
        "last_eval_metrics": {},
        "eval_history_env": [],
        "last_loss": {},
        "last_lite": {},
        "last_her_ratio": 0.0,
        "episode_state": {
            "at_episode_boundary": True,
            "episode_step": 0,
        },
        "resume_audit": {
            "schema_version": 1,
            "strict_control": True,
            "rng_state_fields": rng_fields,
            "global_step": 200,
            "total_updates": 3,
            "episode_boundary": True,
            "config_fingerprint": fingerprint,
        },
        **global_states,
    }


def expect_failure(callback: object, text: str) -> None:
    try:
        callback()
    except (RuntimeError, ValueError) as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected failure containing {text!r}")


def test_strict_config_and_cli() -> None:
    config = strict_config()
    assert config.M_B == 16
    assert config.M_Q_min == 4
    assert config.N_local_min == 4
    assert config.local_search_max == 512
    assert config.action_bins_per_dim == 5
    assert config.numba_support_scan is True
    assert config.support_predicate_early_exit is True
    assert config.support_diagnostic_audit_interval == 100
    assert config.seeds.assignment_seed == 30_001
    payload = config.as_dict()
    for field in (
        "support_query_mode",
        "support_assignment_mode",
        "assignment_diagnostic_window_updates",
        "numba_support_scan",
        "support_predicate_early_exit",
        "support_diagnostic_audit_interval",
    ):
        assert field in payload

    args = parse_args(
        [
            "--method",
            "b3_support_runtime_opt",
            "--weight_scope",
            "critic_only",
            "--support_query_mode",
            "current_policy",
            "--support_assignment_mode",
            "context_eligible_count_matched",
            "--assignment_seed",
            "30005",
            "--assignment_diagnostic_window_updates",
            "250",
        ]
    )
    parsed = config_from_args(args)
    parsed.normalize()
    assert parsed.seeds.assignment_seed == 30_005
    assert parsed.assignment_diagnostic_window_updates == 250

    expect_failure(
        lambda: strict_config(method="b4_lite"),
        "must match DrawerOpen SABER Main",
    )
    expect_failure(
        lambda: strict_config(weight_scope="actor_critic"),
        "must match DrawerOpen SABER Main",
    )
    expect_failure(
        lambda: strict_config(support_query_mode="replay_action"),
        "support_query_mode=current_policy",
    )


def test_fingerprint_and_recursive_parity() -> None:
    config = strict_config()
    identity_only = replace(
        config,
        run_name="another_run",
        output_root="another_output",
        resume=True,
        resume_path="checkpoint.pt",
    )
    assert checkpoint_config_fingerprint(config) == (
        checkpoint_config_fingerprint(identity_only)
    )
    payload = minimal_payload(config)
    result = validate_resume_payload(
        payload,
        identity_only,
        torch.device("cpu"),
    )
    assert result["config_fingerprint_matched"] is True
    assert result["recursive_config_parity_passed"] is True
    # The checkpoint must retain the complete on-disk config for the
    # completion gate while treating derived reporting metadata as
    # non-training identity during resume validation.
    assert payload["config"]["thread_hint"] == {
        "OMP_NUM_THREADS": "1"
    }
    assert payload["config"]["terminal_convention"] == {
        "success_terminates": False,
        "timeout_bootstrap": True,
    }

    changed_seed = replace(
        config,
        seeds=replace(config.seeds, assignment_seed=30_999),
    )
    expect_failure(
        lambda: validate_resume_payload(
            payload,
            changed_seed,
            torch.device("cpu"),
        ),
        "training config differs",
    )


def test_strict_missing_and_boundary_failures() -> None:
    config = strict_config()
    payload = minimal_payload(config)
    missing = copy.deepcopy(payload)
    missing.pop("assignment_rng_state")
    expect_failure(
        lambda: validate_resume_payload(
            missing,
            config,
            torch.device("cpu"),
        ),
        "missing required state",
    )

    mid_episode = copy.deepcopy(payload)
    mid_episode["episode_state"] = {
        "at_episode_boundary": False,
        "episode_step": 17,
    }
    expect_failure(
        lambda: validate_resume_payload(
            mid_episode,
            config,
            torch.device("cpu"),
        ),
        "episode-boundary checkpoint",
    )


def test_output_tail_contract() -> None:
    config = strict_config()
    payload = minimal_payload(config)
    with tempfile.TemporaryDirectory(
        prefix="drawer_strict_tail_test_"
    ) as temp:
        root = Path(temp)
        metrics = root / "metrics.csv"
        diagnostics = root / "strict_diagnostics_1000.csv"
        metrics.write_text("step\n201\n", encoding="utf-8")
        diagnostics.write_text("update_end\n3\n", encoding="utf-8")
        expect_failure(
            lambda: validate_resume_output_tails(
                root,
                payload,
                strict=True,
                diagnostic_window_updates=1_000,
            ),
            "metrics.csv tail is newer",
        )
        metrics.write_text("step\n200\n", encoding="utf-8")
        diagnostics.write_text("update_end\n4\n", encoding="utf-8")
        expect_failure(
            lambda: validate_resume_output_tails(
                root,
                payload,
                strict=True,
                diagnostic_window_updates=1_000,
            ),
            "diagnostic tail is newer",
        )
        diagnostics.write_text("update_end\n3\n", encoding="utf-8")
        result = validate_resume_output_tails(
            root,
            payload,
            strict=True,
            diagnostic_window_updates=1_000,
        )
        assert result["metrics_tail_step"] == 200
        assert result["strict_diagnostic_tail_update"] == 3


def test_global_rng_round_trip() -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    state = capture_global_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    restore_global_rng_state(
        state,
        strict=True,
        device=torch.device("cpu"),
    )
    observed = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert expected[0] == observed[0]
    assert expected[1] == observed[1]
    assert torch.equal(expected[2], observed[2])


def main() -> int:
    test_strict_config_and_cli()
    test_fingerprint_and_recursive_parity()
    test_strict_missing_and_boundary_failures()
    test_output_tail_contract()
    test_global_rng_round_trip()
    print("test_strict_config_resume_contract.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
