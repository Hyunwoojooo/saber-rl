#!/usr/bin/env python3
from __future__ import annotations

import copy
import io
import sys
from unittest.mock import patch

import numpy as np

from metacog_lite_v0.assignment_diagnostics import AssignmentDiagnosticsWindow
from metacog_lite_v0.config import (
    COUNT_MATCHED_ASSIGNMENT_MODES,
    LiteConfig,
    SeedConfig,
)
from metacog_lite_v0.trainer import (
    restore_assignment_checkpoint_state,
    restore_training_progress,
    validate_resume_checkpoint_config,
)
from train_fetch_pick_and_place_lite_v0 import config_from_args, parse_args


def test_resume_checkpoint_config_contract() -> None:
    config = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        run_name="random_support_seed_1",
        output_root="experiment_FPP/runs/random_support",
        total_steps=1_000_000,
        support_query_mode="current_policy",
        support_assignment_mode="batch_wide_count_matched",
        weight_scope="critic_only",
        weighting_mode="batch_norm",
        seeds=SeedConfig(
            env_seed=1,
            torch_seed=1,
            numpy_seed=1,
            action_noise_seed=1,
            weight_seed=20_001,
            assignment_seed=30_001,
            eval_seed=10_001,
        ),
    )
    config.normalize()
    stored = config.as_dict()
    stored["resume"] = False
    stored["resume_path"] = None

    config.resume = True
    config.resume_path = "/tmp/selected-checkpoint.pt"
    validate_resume_checkpoint_config(config, {"config": stored})

    forbidden_mutations = (
        ("method", "b0_ddpg_her"),
        ("support_query_mode", "replay_action"),
        ("support_assignment_mode", "support"),
        ("run_name", "random_support_seed_2"),
        ("total_steps", 50_000),
    )
    for key, value in forbidden_mutations:
        poisoned = dict(stored)
        poisoned[key] = value
        try:
            validate_resume_checkpoint_config(config, {"config": poisoned})
        except RuntimeError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"resume accepted mismatched checkpoint field {key}")

    poisoned = dict(stored)
    poisoned["seeds"] = dict(stored["seeds"])
    poisoned["seeds"]["assignment_seed"] = 30_002
    try:
        validate_resume_checkpoint_config(config, {"config": poisoned})
    except RuntimeError as exc:
        assert "seeds" in str(exc)
    else:
        raise AssertionError("resume accepted a different assignment seed")

    for malformed in ({}, {"config": []}):
        try:
            validate_resume_checkpoint_config(config, malformed)
        except RuntimeError as exc:
            assert "resolved config" in str(exc)
        else:
            raise AssertionError("resume accepted a checkpoint without config")


def test_count_matched_assignment_resume_contract() -> None:
    assert COUNT_MATCHED_ASSIGNMENT_MODES == {
        "context_eligible_count_matched",
        "batch_wide_count_matched",
    }
    for mode in sorted(COUNT_MATCHED_ASSIGNMENT_MODES):
        config = LiteConfig(
            task_name="fetch_pick_and_place",
            method="b3_support_runtime_opt",
            support_query_mode="current_policy",
            support_assignment_mode=mode,
            weight_scope="critic_only",
            weighting_mode="batch_norm",
            assignment_diagnostic_window_updates=10,
            seeds=SeedConfig(assignment_seed=30_001),
        )
        config.normalize()

        assignment_rng = np.random.default_rng(config.seeds.assignment_seed)
        diagnostics = AssignmentDiagnosticsWindow(window_size=10)
        for payload, missing_key in (
            ({}, "assignment_rng_state"),
            (
                {"assignment_rng_state": assignment_rng.bit_generator.state},
                "assignment_diagnostics_state",
            ),
        ):
            try:
                restore_assignment_checkpoint_state(
                    config,
                    payload,
                    np.random.default_rng(1),
                    AssignmentDiagnosticsWindow(window_size=10),
                )
            except RuntimeError as exc:
                assert missing_key in str(exc)
            else:
                raise AssertionError(
                    f"{mode} resume accepted checkpoint missing {missing_key}"
                )

        # A valid checkpoint restores the dedicated stream exactly, without
        # falling back to a seed-derived restart.
        assignment_rng.choice(32, size=7, replace=False)
        saved_rng_state = copy.deepcopy(assignment_rng.bit_generator.state)
        saved_diagnostic_state = copy.deepcopy(diagnostics.state_dict())
        expected = assignment_rng.choice(32, size=7, replace=False)

        restored_rng = np.random.default_rng(999)
        restored_diagnostics = AssignmentDiagnosticsWindow(window_size=10)
        restore_assignment_checkpoint_state(
            config,
            {
                "assignment_rng_state": saved_rng_state,
                "assignment_diagnostics_state": saved_diagnostic_state,
            },
            restored_rng,
            restored_diagnostics,
        )
        np.testing.assert_array_equal(
            restored_rng.choice(32, size=7, replace=False), expected
        )
        assert restored_diagnostics.state_dict() == saved_diagnostic_state

        poisoned_state = copy.deepcopy(saved_diagnostic_state)
        first_sum = next(iter(poisoned_state["sums"]))
        poisoned_state["sums"][first_sum] = float("nan")
        try:
            restore_assignment_checkpoint_state(
                config,
                {
                    "assignment_rng_state": saved_rng_state,
                    "assignment_diagnostics_state": poisoned_state,
                },
                np.random.default_rng(999),
                AssignmentDiagnosticsWindow(window_size=10),
            )
        except RuntimeError as exc:
            assert "non-finite count-matched checkpoint state" in str(exc)
        else:
            raise AssertionError(f"{mode} resume accepted non-finite state")

    # Legacy/default support checkpoints are not retroactively required to
    # contain assignment-only state.
    support_config = LiteConfig(support_assignment_mode="support")
    support_config.normalize()
    restore_assignment_checkpoint_state(
        support_config,
        {},
        np.random.default_rng(1),
        AssignmentDiagnosticsWindow(),
    )


def test_assignment_mode_cli_contract() -> None:
    with patch.object(
        sys,
        "argv",
        [
            "train_fetch_pick_and_place_lite_v0.py",
            "--method",
            "b3_support_runtime_opt",
            "--weight_scope",
            "critic_only",
            "--support_assignment_mode",
            "batch_wide_count_matched",
            "--assignment_seed",
            "30005",
        ],
    ):
        args = parse_args()
    config = config_from_args(args)
    config.normalize()
    assert config.support_assignment_mode == "batch_wide_count_matched"
    assert config.seeds.assignment_seed == 30_005

    with (
        patch.object(
            sys,
            "argv",
            [
                "train_fetch_pick_and_place_lite_v0.py",
                "--support_assignment_mode",
                "random_fixed_count",
            ],
        ),
        patch.object(sys, "stderr", io.StringIO()),
    ):
        try:
            parse_args()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("CLI accepted an unsupported assignment mode")


def main() -> int:
    test_resume_checkpoint_config_contract()
    test_count_matched_assignment_resume_contract()
    test_assignment_mode_cli_contract()
    eval_history = [(step, step / 1_000_000.0) for step in range(10_000, 50_001, 10_000)]
    payload = {
        "global_step": 50_000,
        "episode_idx": 1_000,
        "total_updates": 45_001,
        "last_loss": {"critic": 0.25, "actor": -0.5},
        "last_lite": {"lite/mean_w_P": 1.0, "assignment/enabled": 1.0},
        "last_her_ratio": 0.8,
        "last_eval_metrics": {
            "eval_goal_success_mean": 0.4,
            "eval_env_success_mean": 0.4,
        },
        "eval_history_env": eval_history,
    }
    last_loss = {"critic": 0.0, "actor": 0.0, "q_mean": 0.0}
    last_lite = {"lite/mean_w_P": 0.0}
    (
        start_step,
        episode_idx,
        total_updates,
        last_her_ratio,
        last_eval_metrics,
        restored_history,
    ) = restore_training_progress(payload, last_loss, last_lite)

    # global_step is the number of completed transitions. Resume at exactly
    # 50k; the next loop iteration creates transition 50,001.
    assert start_step == 50_000
    assert episode_idx == 1_000
    assert total_updates == 45_001
    remaining_transitions = 1_000_000 - start_step
    assert total_updates + remaining_transitions == 995_001

    assert last_loss["critic"] == 0.25
    assert last_loss["actor"] == -0.5
    assert last_loss["q_mean"] == 0.0
    assert last_lite["lite/mean_w_P"] == 1.0
    assert last_lite["assignment/enabled"] == 1.0
    assert last_her_ratio == 0.8
    assert last_eval_metrics["eval_goal_success_mean"] == 0.4
    assert restored_history == eval_history
    assert [step for step, _ in restored_history] == [
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
    ]

    # Historical checkpoints without optional observability state remain
    # loadable for the default path, while count-matched checkpoints created
    # by this source persist the full state.
    fallback_loss = {"critic": 7.0}
    fallback_lite = {"lite/mean_w_P": 3.0}
    fallback = restore_training_progress(
        {"global_step": 5, "episode_idx": 1, "total_updates": 1},
        fallback_loss,
        fallback_lite,
    )
    assert fallback[0:3] == (5, 1, 1)
    assert fallback[3] == 0.0 and fallback[4] == {} and fallback[5] == []
    assert fallback_loss == {"critic": 7.0}
    assert fallback_lite == {"lite/mean_w_P": 3.0}

    print("test_trainer_resume_contract.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
