#!/usr/bin/env python3
from __future__ import annotations

import copy
import json

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


def make_config(
    mode: str,
    cache_enabled: bool,
    *,
    batch_size: int = 4,
    n_local_min: int = 4,
    m_q_min: int = 4,
    m_b: int = 16,
) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=batch_size,
        M_B=m_b,
        M_Q_min=m_q_min,
        N_local_min=n_local_min,
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
    cfg.numba_support_scan = False
    cfg.support_predicate_early_exit = False
    cfg.support_diagnostic_audit_interval = 0
    return cfg


def add_transition(
    replay: HERReplayBuffer,
    task: FetchPickAndPlaceTaskSpec,
    *,
    action: np.ndarray,
    row: int,
) -> None:
    obs = make_obs()
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    action_bin = int(task.action_bin_batch(action[None, :], 5)[0])
    replay.add_transition(
        observation=obs,
        achieved_goal=goal,
        desired_goal=goal,
        action_exec=action,
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


def make_replay(task: FetchPickAndPlaceTaskSpec, count: int = 4) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    for row in range(count):
        add_transition(replay, task, action=ACTION_A, row=row)
    return replay


def make_batch(actions: np.ndarray | None = None) -> SampleBatch:
    if actions is None:
        actions = np.stack([ACTION_B, ACTION_A, ACTION_B, ACTION_A])
    actions = np.asarray(actions, dtype=np.float32)
    size = int(actions.shape[0])
    obs = np.repeat(make_obs()[None, :], size, axis=0)
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    goals = np.repeat(goal[None, :], size, axis=0)
    return SampleBatch(
        indices=np.arange(size, dtype=np.int64),
        observation=obs,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=obs.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=actions,
        reward_ext=np.zeros(size, dtype=np.float32),
        done=np.zeros(size, dtype=np.bool_),
        timeout=np.zeros(size, dtype=np.bool_),
        h=np.full(size, 20, dtype=np.int16),
        her_relabel_ratio=1.0,
    )


def compute(mode: str, cache_enabled: bool, rng_seed: int = 27027):
    task = make_task()
    cfg = make_config(mode, cache_enabled)
    replay = make_replay(task)
    batch = make_batch()
    policy_actions = np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B])
    agent = VectorAgent(policy_actions)
    rng = np.random.default_rng(rng_seed)
    before_rng = copy.deepcopy(rng.bit_generator.state)
    snapshots = {
        name: getattr(batch, name).copy()
        for name in (
            "indices",
            "observation",
            "desired_goal_train",
            "action_exec",
            "h",
            "reward_ext",
        )
    }
    wb = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
        batch,
        replay,
        agent,
        rng,
        torch.device("cpu"),
    )
    for name, expected in snapshots.items():
        np.testing.assert_array_equal(getattr(batch, name), expected)
    return wb, task, policy_actions, batch, before_rng, copy.deepcopy(rng.bit_generator.state)


def expected_shuffled_mask(task: FetchPickAndPlaceTaskSpec, policy_actions: np.ndarray) -> np.ndarray:
    permutation = make_derangement(policy_actions.shape[0], np.random.default_rng(27027))
    assert np.all(permutation != np.arange(policy_actions.shape[0]))
    query_bins = task.action_bin_batch(policy_actions[permutation], 5)
    supported_bin = int(task.action_bin_batch(ACTION_A[None, :], 5)[0])
    return (query_bins == supported_bin).astype(np.float32)


def assert_replay_query_identity_and_rng() -> None:
    task = make_task()
    cfg = make_config("replay_action", True)
    computer = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6))
    batch = make_batch()
    policy = np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B])
    rng = np.random.default_rng(711)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    replay_snapshot = batch.action_exec.copy()
    query, requires_membership, metrics = computer._support_query_actions(batch, policy, rng)
    assert requires_membership
    assert query is batch.action_exec
    assert np.shares_memory(query, batch.action_exec)
    np.testing.assert_array_equal(query, replay_snapshot)
    np.testing.assert_array_equal(batch.action_exec, replay_snapshot)
    assert metrics["query/query_bin_matches_replay_action_rate"] == 1.0
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == before


def assert_self_support_and_population_threshold() -> None:
    task = make_task()
    replay = make_replay(task, count=1)
    batch = make_batch(ACTION_A[None, :])
    agent = VectorAgent(ACTION_B[None, :])

    positive_cfg = make_config(
        "replay_action", True, batch_size=1, n_local_min=1, m_q_min=1, m_b=1
    )
    positive = LiteWeightComputer(
        positive_cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(batch, replay, agent, np.random.default_rng(4), torch.device("cpu"))
    assert batch.indices[0] == 0
    assert positive.psi_a_query[0] == 1.0
    assert positive.psi_q_rank[0] == 1.0, "sampled transition must remain eligible to self-support"

    negative_cfg = make_config(
        "replay_action", True, batch_size=1, n_local_min=2, m_q_min=2, m_b=2
    )
    negative = LiteWeightComputer(
        negative_cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(batch, replay, agent, np.random.default_rng(4), torch.device("cpu"))
    assert negative.psi_q_rank[0] == 0.0, "self-support does not bypass population thresholds"


def assert_cache_keys_separate_context_bin_and_mode() -> None:
    task = make_task()
    cfg = make_config("current_policy", True)
    computer = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6))
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    kwargs = dict(coarse_key=1, full_key=2, goal=goal, h=20, bucket_version=3)
    key_a = computer._support_cache_key(**kwargs, policy_bin=10)
    key_b = computer._support_cache_key(**kwargs, policy_bin=11)
    key_context = computer._support_cache_key(**{**kwargs, "full_key": 7}, policy_bin=10)
    cfg.support_query_mode = "replay_action"
    key_mode = computer._support_cache_key(**kwargs, policy_bin=10)
    assert len({key_a, key_b, key_context, key_mode}) == 4


def main() -> int:
    expected = {
        "current_policy": np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "replay_action": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "context_count_only": np.ones(4, dtype=np.float32),
    }
    probe, task, policy_actions, _batch, _before, _after = compute(
        "shuffled_policy", cache_enabled=True
    )
    expected["shuffled_policy"] = expected_shuffled_mask(task, policy_actions)
    np.testing.assert_array_equal(probe.psi_q_rank, expected["shuffled_policy"])

    for mode, expected_mask in expected.items():
        cached, _task, _policy, _batch, before_rng, after_rng = compute(mode, cache_enabled=True)
        uncached, _task, _policy, _batch, _before_rng, _after_rng = compute(
            mode, cache_enabled=False
        )
        np.testing.assert_array_equal(cached.psi_q_rank, expected_mask)
        np.testing.assert_array_equal(cached.psi_a_query, cached.psi_a_policy)
        for field in (
            "psi_q_rank",
            "psi_a_policy",
            "w_p_raw",
            "omega_critic",
            "omega_actor",
            "len_a_q_local",
            "len_a_q_sup",
        ):
            np.testing.assert_array_equal(getattr(cached, field), getattr(uncached, field))
        assert cached.extra_metrics["query/context_count_eligible_rate"] == 1.0
        assert cached.extra_metrics["query/support_positive_rate"] == float(np.mean(expected_mask))
        assert np.all(cached.omega_actor == 1.0)
        if mode in {"current_policy", "replay_action", "context_count_only"}:
            assert before_rng == after_rng, f"{mode} unexpectedly consumed weight RNG"
        if mode != "context_count_only":
            assert cached.extra_metrics["query/query_bin_present_given_context_rate"] == float(
                np.mean(expected_mask)
            )

    # Critic-only scope applies non-uniform query weights only to the critic.
    replay_query, _task, _policy, _batch, _before, _after = compute(
        "replay_action", cache_enabled=True
    )
    assert replay_query.extra_metrics["query/query_bin_matches_replay_action_rate"] == 1.0
    assert not np.allclose(replay_query.omega_critic, 1.0)
    assert np.all(replay_query.omega_actor == 1.0)

    context_only, _task, _policy, _batch, _before, _after = compute(
        "context_count_only", cache_enabled=True
    )
    assert np.isnan(context_only.extra_metrics["query/query_bin_present_given_context_rate"])
    assert np.isnan(context_only.extra_metrics["query/query_bin_matches_current_policy_rate"])
    assert context_only.extra_metrics["runtime/coarse_action_bin_count_lookup_rate"] == 0.0

    invalid = make_config("current_policy", cache_enabled=True)
    invalid.support_query_mode = "uniform_random"
    try:
        invalid.normalize()
    except ValueError as exc:
        assert "support_query_mode" in str(exc)
    else:
        raise AssertionError("invalid support_query_mode did not raise")

    try:
        make_derangement(1, np.random.default_rng(1))
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("singleton derangement did not raise")

    assert_replay_query_identity_and_rng()
    assert_self_support_and_population_threshold()
    assert_cache_keys_separate_context_bin_and_mode()
    print("test_support_query_modes.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
