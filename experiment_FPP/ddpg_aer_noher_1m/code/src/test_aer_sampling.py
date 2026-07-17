#!/usr/bin/env python3
"""Contract tests for the no-HER Attentive Experience Replay path.

The implementation is intentionally developed separately from this test.  The
planned public API is:

* ``aer_lambda(global_step, total_steps, lambda0, alpha) -> float``;
* ``HERReplayBuffer.sample_attentive(batch_size, current_state,
  lambda_value, rng) -> (SampleBatch, stats)``; and
* DDPGAgent and TD3Agent expose matching learning-rate setters.

This file makes the following small parts of that contract explicit:

* ``current_state`` and replay states use the actor input convention
  ``concatenate([observation, desired_goal])``;
* attentive sampling never performs HER relabeling;
* at ``lambda_value == 1`` attentive sampling is an exact regression alias for
  ``sample(batch_size, her_ratio=0.0, rng=rng)``, including RNG consumption;
* stats may be either a mapping or an attribute/dataclass object, but expose
  ``candidate_count`` and ``candidate_similarity_min/mean/max``; and
* ``current_learning_rates`` may return ``(actor_lr, critic_lr)`` or a mapping
  with ``actor_lr`` and ``critic_lr`` keys.

For non-integral ``lambda_value * batch_size``, the implementation must
document its integer-rounding rule and whether a candidate draw is with
replacement.  These tests use integral candidate counts and therefore do not
couple correctness to those independent implementation choices.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Mapping

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent, TD3Agent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch, aer_lambda
from metacog_lite_v0.trainer import make_agent


def _add_transition(
    replay: HERReplayBuffer,
    observation: np.ndarray,
    desired_goal: np.ndarray,
    *,
    episode_id: int,
    episode_step: int,
    next_achieved_goal: np.ndarray | None = None,
) -> None:
    observation = np.asarray(observation, dtype=np.float32)
    desired_goal = np.asarray(desired_goal, dtype=np.float32)
    achieved_goal = np.zeros(replay.goal_dim, dtype=np.float32)
    if next_achieved_goal is None:
        next_achieved_goal = achieved_goal
    replay.add_transition(
        observation=observation,
        achieved_goal=achieved_goal,
        desired_goal=desired_goal,
        action_exec=np.zeros(replay.action_dim, dtype=np.float32),
        reward_ext=-1.0,
        next_observation=observation + np.float32(0.01),
        next_achieved_goal=np.asarray(next_achieved_goal, dtype=np.float32),
        done=False,
        timeout=False,
        h=50 - int(episode_step),
        episode_id=int(episode_id),
        episode_step=int(episode_step),
    )


def _direction_replay() -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=8, obs_dim=2, goal_dim=1, action_dim=1)
    observations = (
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([-1.0, 0.0], dtype=np.float32),
    )
    for step, observation in enumerate(observations):
        _add_transition(
            replay,
            observation,
            np.zeros(1, dtype=np.float32),
            episode_id=0,
            episode_step=step,
        )
    return replay


def _resume_replay() -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=32, obs_dim=2, goal_dim=1, action_dim=1)
    for step in range(12):
        observation = np.asarray(
            [math.cos(0.37 * step), math.sin(0.37 * step)],
            dtype=np.float32,
        )
        desired_goal = np.asarray([0.05 * (step - 5)], dtype=np.float32)
        _add_transition(
            replay,
            observation,
            desired_goal,
            episode_id=step // 4,
            episode_step=step % 4,
            next_achieved_goal=np.asarray([100.0 + step], dtype=np.float32),
        )
    return replay


def _stats_mapping(stats: object) -> Mapping[str, Any]:
    if isinstance(stats, Mapping):
        return stats
    if is_dataclass(stats):
        return asdict(stats)
    if hasattr(stats, "__dict__"):
        return vars(stats)
    raise AssertionError(
        "AER stats must be a mapping, dataclass, or attribute object; "
        f"got {type(stats)!r}"
    )


def _stat(stats: object, name: str) -> Any:
    mapping = _stats_mapping(stats)
    if name not in mapping:
        raise AssertionError(f"AER stats are missing required field {name!r}: {mapping}")
    return mapping[name]


def _assert_batches_equal(actual: SampleBatch, expected: SampleBatch) -> None:
    for field in fields(SampleBatch):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(
                actual_value,
                expected_value,
                err_msg=f"SampleBatch.{field.name} differs",
            )
        else:
            assert actual_value == expected_value, (
                f"SampleBatch.{field.name}: {actual_value!r} != {expected_value!r}"
            )


def _assert_stats_equal(actual: object, expected: object) -> None:
    actual_mapping = _stats_mapping(actual)
    expected_mapping = _stats_mapping(expected)
    assert actual_mapping.keys() == expected_mapping.keys()
    for name in actual_mapping:
        # Wall-clock diagnostics are intentionally not deterministic across a
        # checkpoint/resume boundary and are not part of RNG reproducibility.
        if name == "sampling_seconds":
            assert np.isfinite(float(actual_mapping[name]))
            assert np.isfinite(float(expected_mapping[name]))
            continue
        actual_value = actual_mapping[name]
        expected_value = expected_mapping[name]
        if isinstance(actual_value, np.ndarray) or isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=name)
        elif isinstance(actual_value, (float, np.floating)):
            assert float(actual_value) == float(expected_value), name
        else:
            assert actual_value == expected_value, name


def _learning_rates(agent: DDPGAgent | TD3Agent) -> tuple[float, float]:
    rates = agent.current_learning_rates()
    if isinstance(rates, Mapping):
        if "actor_lr" not in rates or "critic_lr" not in rates:
            raise AssertionError(
                "Mapping returned by current_learning_rates() must contain "
                "'actor_lr' and 'critic_lr'"
            )
        return float(rates["actor_lr"]), float(rates["critic_lr"])
    if isinstance(rates, (tuple, list)) and len(rates) == 2:
        return float(rates[0]), float(rates[1])
    raise AssertionError(
        "current_learning_rates() must return (actor_lr, critic_lr) or a mapping; "
        f"got {rates!r}"
    )


def test_aer_lambda_schedule_endpoints() -> None:
    total_steps = 300
    lambda0 = 2.5
    alpha = 2.0 / 3.0
    assert math.isclose(aer_lambda(0, total_steps, lambda0, alpha), 2.5)
    assert math.isclose(aer_lambda(-10, total_steps, lambda0, alpha), 2.5)
    assert math.isclose(aer_lambda(100, total_steps, lambda0, alpha), 1.75)
    assert math.isclose(aer_lambda(200, total_steps, lambda0, alpha), 1.0)
    assert math.isclose(aer_lambda(total_steps, total_steps, lambda0, alpha), 1.0)
    assert math.isclose(aer_lambda(10 * total_steps, total_steps, lambda0, alpha), 1.0)


def test_attentive_sampling_selects_known_cosine_top_k() -> None:
    replay = _direction_replay()
    current_state = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    # With four candidates and a four-row buffer, a without-replacement draw
    # sees the whole buffer.  Seed 8 also gives [2, 1, 0, 3] for NumPy's
    # replacement-based integer draw, so this test permits either documented
    # candidate-draw convention.
    batch, stats = replay.sample_attentive(
        batch_size=2,
        current_state=current_state,
        lambda_value=2.0,
        rng=np.random.default_rng(8),
    )

    assert set(batch.indices.tolist()) == {0, 1}
    assert int(_stat(stats, "candidate_count")) == 4
    assert batch.her_relabel_ratio == 0.0


def test_attentive_sampling_never_relabels_goals() -> None:
    replay = _resume_replay()
    batch, _stats = replay.sample_attentive(
        batch_size=4,
        current_state=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        lambda_value=2.0,
        rng=np.random.default_rng(41),
    )

    assert batch.her_relabel_ratio == 0.0
    np.testing.assert_array_equal(
        batch.desired_goal_train,
        replay.desired_goal[batch.indices],
    )
    # Future achieved goals were deliberately placed far outside the original
    # desired-goal range, making accidental HER relabeling easy to detect.
    assert np.all(np.abs(batch.desired_goal_train) < 1.0)


def test_lambda_one_is_exact_uniform_noher_regression() -> None:
    replay = _resume_replay()
    uniform_rng = np.random.default_rng(2026)
    attentive_rng = np.random.default_rng(2026)

    expected = replay.sample(batch_size=6, her_ratio=0.0, rng=uniform_rng)
    actual, stats = replay.sample_attentive(
        batch_size=6,
        current_state=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        lambda_value=1.0,
        rng=attentive_rng,
    )

    _assert_batches_equal(actual, expected)
    assert int(_stat(stats, "candidate_count")) == 6
    np.testing.assert_array_equal(
        attentive_rng.integers(0, 2**31, size=16, dtype=np.int64),
        uniform_rng.integers(0, 2**31, size=16, dtype=np.int64),
    )


def test_zero_norm_cosine_is_finite() -> None:
    replay = HERReplayBuffer(capacity=8, obs_dim=2, goal_dim=1, action_dim=1)
    for step in range(4):
        _add_transition(
            replay,
            np.zeros(2, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            episode_id=0,
            episode_step=step,
        )

    batch, stats = replay.sample_attentive(
        batch_size=2,
        current_state=np.zeros(3, dtype=np.float32),
        lambda_value=2.0,
        rng=np.random.default_rng(8),
    )

    for name in (
        "candidate_similarity_min",
        "candidate_similarity_mean",
        "candidate_similarity_max",
    ):
        assert np.isfinite(float(_stat(stats, name))), name
    assert np.all(np.isfinite(batch.observation))
    assert np.all(np.isfinite(batch.desired_goal_train))


def test_attentive_rng_state_resume_is_deterministic() -> None:
    replay = _resume_replay()
    resumed_replay = HERReplayBuffer(capacity=32, obs_dim=2, goal_dim=1, action_dim=1)
    resumed_replay.load_state_dict(copy.deepcopy(replay.state_dict()))
    assert resumed_replay.version == replay.version
    assert (
        resumed_replay.action_index.state_dict()["epoch"]
        == replay.action_index.state_dict()["epoch"]
    )

    rng = np.random.default_rng(903)
    rng.integers(0, 10_000, size=7)
    saved_rng_state = copy.deepcopy(rng.bit_generator.state)

    expected_batch, expected_stats = replay.sample_attentive(
        batch_size=4,
        current_state=np.asarray([0.25, 0.75, -0.10], dtype=np.float32),
        lambda_value=2.5,
        rng=rng,
    )
    expected_tail = rng.integers(0, 2**31, size=16, dtype=np.int64)

    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = copy.deepcopy(saved_rng_state)
    actual_batch, actual_stats = resumed_replay.sample_attentive(
        batch_size=4,
        current_state=np.asarray([0.25, 0.75, -0.10], dtype=np.float32),
        lambda_value=2.5,
        rng=resumed_rng,
    )
    actual_tail = resumed_rng.integers(0, 2**31, size=16, dtype=np.int64)

    _assert_batches_equal(actual_batch, expected_batch)
    _assert_stats_equal(actual_stats, expected_stats)
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_td3_optimizer_learning_rate_setter() -> None:
    agent = TD3Agent(
        state_dim=3,
        action_dim=2,
        hidden_dim=8,
        actor_lr=3.0e-4,
        critic_lr=3.0e-4,
        gamma=0.99,
        tau=0.005,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    )

    assert _learning_rates(agent) == (3.0e-4, 3.0e-4)
    agent.set_learning_rates(actor_lr=7.5e-4, critic_lr=6.0e-4)
    assert _learning_rates(agent) == (7.5e-4, 6.0e-4)
    assert all(group["lr"] == 7.5e-4 for group in agent.actor_opt.param_groups)
    assert all(group["lr"] == 6.0e-4 for group in agent.critic_opt.param_groups)


def _make_ddpg_agent() -> DDPGAgent:
    return DDPGAgent(
        state_dim=3,
        action_dim=2,
        hidden_dim=8,
        actor_lr=3.0e-4,
        critic_lr=3.0e-4,
        gamma=0.99,
        tau=0.005,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
    )


def test_ddpg_optimizer_learning_rate_setter_and_resume() -> None:
    agent = _make_ddpg_agent()
    assert _learning_rates(agent) == (3.0e-4, 3.0e-4)
    agent.set_learning_rates(actor_lr=7.5e-4, critic_lr=6.0e-4)
    assert _learning_rates(agent) == (7.5e-4, 6.0e-4)
    assert all(group["lr"] == 7.5e-4 for group in agent.actor_opt.param_groups)
    assert all(group["lr"] == 6.0e-4 for group in agent.critic_opt.param_groups)

    resumed = _make_ddpg_agent()
    resumed.load_state_dict(copy.deepcopy(agent.state_dict()))
    assert _learning_rates(resumed) == (7.5e-4, 6.0e-4)


def test_ddpg_aer_dispatch_is_single_critic_ddpg() -> None:
    config = LiteConfig(
        task_name="fetch_pick_and_place",
        method="ddpg_aer",
        her_ratio=0.0,
        lambda_support=0.0,
        weight_scope="none",
        device="cpu",
    )
    config.normalize()
    agent = make_agent(
        config=config,
        state_dim=3,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert isinstance(agent, DDPGAgent)
    assert not isinstance(agent, TD3Agent)
    serialized_config = config.as_dict()
    assert "td3_policy_delay" not in serialized_config
    assert "td3_target_policy_noise" not in serialized_config
    assert "td3_target_noise_clip" not in serialized_config
    state = agent.state_dict()
    assert {"critic", "critic_target"}.issubset(state)
    assert not {"critic1", "critic2", "policy_delay"}.intersection(state)


def main() -> int:
    tests = (
        test_aer_lambda_schedule_endpoints,
        test_attentive_sampling_selects_known_cosine_top_k,
        test_attentive_sampling_never_relabels_goals,
        test_lambda_one_is_exact_uniform_noher_regression,
        test_zero_norm_cosine_is_finite,
        test_attentive_rng_state_resume_is_deterministic,
        test_td3_optimizer_learning_rate_setter,
        test_ddpg_optimizer_learning_rate_setter_and_resume,
        test_ddpg_aer_dispatch_is_single_critic_ddpg,
    )
    for test in tests:
        test()
    print("test_aer_sampling.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
