#!/usr/bin/env python3
from __future__ import annotations

from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.trainer import (
    restore_training_progress,
    validate_resume_checkpoint_config,
)


def test_resume_checkpoint_config_contract() -> None:
    config = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        run_name="strict_seed_1",
        output_root="experiment_FPP/runs/strict",
        total_steps=1_000_000,
        support_query_mode="current_policy",
        support_assignment_mode="context_eligible_count_matched",
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
        ("run_name", "strict_seed_2"),
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


def main() -> int:
    test_resume_checkpoint_config_contract()
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
    # loadable for the default path, while strict checkpoints created by this
    # source persist the full state.
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
