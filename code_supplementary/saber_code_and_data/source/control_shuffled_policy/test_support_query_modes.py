#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec
from metacog_lite_v0.weights import GapStats, LiteWeightComputer, make_derangement


ACTION_A = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_B = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)


def make_task() -> FetchPickAndPlaceTaskSpec:
    return FetchPickAndPlaceTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )


def make_obs() -> np.ndarray:
    obs = np.zeros(25, dtype=np.float32)
    obs[0:3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    obs[3:6] = np.asarray([0.04, 0.0, 0.44], dtype=np.float32)
    obs[9:11] = np.asarray([0.03, 0.03], dtype=np.float32)
    return obs


class VectorAgent:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != self.actions.shape[0]:
            raise AssertionError("unexpected synthetic batch size")
        return self.actions.to(state.device)


def make_config(mode: str, cache_enabled: bool) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=4,
        M_B=16,
        M_Q_min=4,
        N_local_min=4,
        local_search_max=512,
        action_bins_per_dim=5,
        lambda_support=1.0,
        policy_support_mode="bin",
        support_query_mode=mode,
        support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=cache_enabled,
        local_scan_cache_enabled=cache_enabled,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.numba_support_scan = os.environ.get("QUERY_NUMBA", "0") == "1"
    cfg.support_predicate_early_exit = cfg.numba_support_scan
    cfg.support_diagnostic_audit_interval = 0
    return cfg


def make_replay(task: FetchPickAndPlaceTaskSpec) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    obs = make_obs()
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    action_bin = int(task.action_bin_batch(ACTION_A[None, :], 5)[0])
    for row in range(4):
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=ACTION_A,
            reward_ext=0.0,
            next_observation=obs,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=20,
            episode_id=row,
            episode_step=0,
            coarse_key=task.coarse_hash(obs, 20),
            action_bin=action_bin,
        )
    return replay


def make_batch() -> SampleBatch:
    obs = np.repeat(make_obs()[None, :], 4, axis=0)
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    goals = np.repeat(goal[None, :], 4, axis=0)
    replay_actions = np.stack([ACTION_B, ACTION_A, ACTION_B, ACTION_A])
    return SampleBatch(
        indices=np.arange(4, dtype=np.int64),
        observation=obs,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=obs.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=replay_actions,
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=np.full(4, 20, dtype=np.int16),
        her_relabel_ratio=1.0,
    )


def compute(mode: str, cache_enabled: bool, rng_seed: int = 27027):
    task = make_task()
    cfg = make_config(mode, cache_enabled)
    replay = make_replay(task)
    batch = make_batch()
    policy_actions = np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B])
    agent = VectorAgent(policy_actions)
    wb = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
        batch,
        replay,
        agent,
        np.random.default_rng(rng_seed),
        torch.device("cpu"),
    )
    return wb, task, policy_actions, batch


def expected_shuffled_mask(task: FetchPickAndPlaceTaskSpec, policy_actions: np.ndarray) -> np.ndarray:
    permutation = make_derangement(policy_actions.shape[0], np.random.default_rng(27027))
    assert np.all(permutation != np.arange(policy_actions.shape[0]))
    query_bins = task.action_bin_batch(policy_actions[permutation], 5)
    supported_bin = int(task.action_bin_batch(ACTION_A[None, :], 5)[0])
    return (query_bins == supported_bin).astype(np.float32)

def test_distinct_context_alignment() -> None:
    task = make_task()
    cfg = make_config("shuffled_policy", cache_enabled=False)
    replay = HERReplayBuffer(capacity=32, obs_dim=25, goal_dim=3, action_dim=4)
    obs = make_obs()
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    horizons = np.asarray([5, 20, 40, 75], dtype=np.int16)
    supported_actions = np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B])
    for sample_index, (horizon, supported_action) in enumerate(zip(horizons, supported_actions)):
        action_bin = int(task.action_bin_batch(supported_action[None, :], 5)[0])
        for local_index in range(4):
            replay.add_transition(
                observation=obs,
                achieved_goal=goal,
                desired_goal=goal,
                action_exec=supported_action,
                reward_ext=0.0,
                next_observation=obs,
                next_achieved_goal=goal,
                done=False,
                timeout=False,
                h=int(horizon),
                episode_id=10 * sample_index + local_index,
                episode_step=0,
                coarse_key=task.coarse_hash(obs, int(horizon)),
                action_bin=action_bin,
            )

    observations = np.repeat(obs[None, :], 4, axis=0)
    goals = np.repeat(goal[None, :], 4, axis=0)
    batch = SampleBatch(
        indices=np.asarray([0, 4, 8, 12], dtype=np.int64),
        observation=observations.copy(),
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=observations.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.stack([ACTION_B, ACTION_B, ACTION_A, ACTION_A]),
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=horizons.copy(),
        her_relabel_ratio=1.0,
    )
    policy_actions = np.stack([ACTION_A, ACTION_B, ACTION_A, ACTION_B])
    before_observation = batch.observation.copy()
    before_goal = batch.desired_goal_train.copy()
    before_horizon = batch.h.copy()
    permutation = make_derangement(4, np.random.default_rng(31415))
    query_actions = policy_actions[permutation]
    expected = (
        task.action_bin_batch(query_actions, 5)
        == task.action_bin_batch(supported_actions, 5)
    ).astype(np.float32)

    wb = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
        batch,
        replay,
        VectorAgent(policy_actions),
        np.random.default_rng(31415),
        torch.device("cpu"),
    )
    np.testing.assert_array_equal(wb.psi_q_rank, expected)
    np.testing.assert_array_equal(batch.observation, before_observation)
    np.testing.assert_array_equal(batch.desired_goal_train, before_goal)
    np.testing.assert_array_equal(batch.h, before_horizon)
    assert len(set(task.coarse_hash_batch(batch.observation, batch.h).tolist())) == 4


def main() -> int:
    test_distinct_context_alignment()
    expected = {
        "current_policy": np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    probe, task, policy_actions, _batch = compute("shuffled_policy", cache_enabled=True)
    expected["shuffled_policy"] = expected_shuffled_mask(task, policy_actions)
    np.testing.assert_array_equal(probe.psi_q_rank, expected["shuffled_policy"])

    for mode, expected_mask in expected.items():
        cached, _task, _policy, _batch = compute(mode, cache_enabled=True)
        uncached, _task, _policy, _batch = compute(mode, cache_enabled=False)
        np.testing.assert_array_equal(cached.psi_q_rank, expected_mask)
        np.testing.assert_array_equal(cached.psi_a_query, cached.psi_a_policy)
        for field in ("psi_q_rank", "psi_a_policy", "w_p_raw", "omega_critic", "omega_actor"):
            np.testing.assert_array_equal(getattr(cached, field), getattr(uncached, field))
        assert cached.extra_metrics["query/context_count_eligible_rate"] == 1.0
        assert cached.extra_metrics["query/support_positive_rate"] == float(np.mean(expected_mask))

    invalid = make_config("current_policy", cache_enabled=True)
    invalid.support_query_mode = "uniform_random"
    try:
        invalid.normalize()
    except ValueError as exc:
        assert "support_query_mode" in str(exc)
    else:
        raise AssertionError("invalid support_query_mode did not raise")

    for batch_size in (2, 3, 4, 17, 256):
        first = make_derangement(batch_size, np.random.default_rng(12345))
        second = make_derangement(batch_size, np.random.default_rng(12345))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.arange(batch_size))
        assert np.all(first != np.arange(batch_size))

    action_multiset = np.arange(24, dtype=np.float32).reshape(6, 4)
    action_permutation = make_derangement(6, np.random.default_rng(987))
    np.testing.assert_array_equal(
        np.sort(action_multiset[action_permutation], axis=0),
        np.sort(action_multiset, axis=0),
    )

    global_numpy_state = np.random.get_state()
    global_torch_state = torch.random.get_rng_state().clone()
    unrelated_rng = np.random.default_rng(88)
    unrelated_state = repr(unrelated_rng.bit_generator.state)
    make_derangement(32, np.random.default_rng(99))
    after_numpy_state = np.random.get_state()
    assert global_numpy_state[0] == after_numpy_state[0]
    np.testing.assert_array_equal(global_numpy_state[1], after_numpy_state[1])
    assert global_numpy_state[2:] == after_numpy_state[2:]
    assert torch.equal(global_torch_state, torch.random.get_rng_state())
    assert unrelated_state == repr(unrelated_rng.bit_generator.state)

    singleton_rng = np.random.default_rng(1)
    try:
        make_derangement(1, singleton_rng)
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("singleton derangement did not raise")

    print("test_support_query_modes.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
