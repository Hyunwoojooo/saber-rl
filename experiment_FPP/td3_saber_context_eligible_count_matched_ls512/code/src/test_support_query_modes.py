#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import replace

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
    assignment_mode: str = "support",
) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="td3_saber_runtime_opt",
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
        support_assignment_mode=assignment_mode,
        local_scan_cache_enabled=cache_enabled,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.numba_support_scan = False
    cfg.support_predicate_early_exit = False
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


def compute(
    mode: str,
    cache_enabled: bool,
    rng_seed: int = 27027,
    assignment_mode: str = "support",
    capture_assignment_state: bool = False,
):
    task = make_task()
    cfg = make_config(mode, cache_enabled, assignment_mode)
    replay = make_replay(task)
    batch = make_batch()
    policy_actions = np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B])
    agent = VectorAgent(policy_actions)
    assignment_rng = np.random.default_rng(30_001)
    assignment_before = copy.deepcopy(assignment_rng.bit_generator.state)
    wb = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
        batch,
        replay,
        agent,
        np.random.default_rng(rng_seed),
        torch.device("cpu"),
        assignment_rng=assignment_rng,
    )
    if capture_assignment_state:
        return (
            wb,
            task,
            policy_actions,
            batch,
            assignment_before,
            copy.deepcopy(assignment_rng.bit_generator.state),
        )
    return wb, task, policy_actions, batch


def expected_shuffled_mask(task: FetchPickAndPlaceTaskSpec, policy_actions: np.ndarray) -> np.ndarray:
    permutation = make_derangement(policy_actions.shape[0], np.random.default_rng(27027))
    assert np.all(permutation != np.arange(policy_actions.shape[0]))
    query_bins = task.action_bin_batch(policy_actions[permutation], 5)
    supported_bin = int(task.action_bin_batch(ACTION_A[None, :], 5)[0])
    return (query_bins == supported_bin).astype(np.float32)


def main() -> int:
    expected = {
        "current_policy": np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "replay_action": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "context_count_only": np.ones(4, dtype=np.float32),
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
        if mode != "context_count_only":
            assert cached.extra_metrics["query/query_bin_present_given_context_rate"] == float(
                np.mean(expected_mask)
            )

    context_only, _task, _policy, _batch = compute("context_count_only", cache_enabled=True)
    assert np.isnan(context_only.extra_metrics["query/query_bin_present_given_context_rate"])
    assert np.isnan(context_only.extra_metrics["query/query_bin_matches_current_policy_rate"])
    assert context_only.extra_metrics["runtime/coarse_action_bin_count_lookup_rate"] == 0.0

    replay_query, _task, _policy, _batch = compute("replay_action", cache_enabled=True)
    assert replay_query.extra_metrics["query/query_bin_matches_replay_action_rate"] == 1.0

    (
        main_wb,
        _task,
        _policy,
        _batch,
        main_assignment_before,
        main_assignment_after,
    ) = compute(
        "current_policy",
        cache_enabled=True,
        assignment_mode="support",
        capture_assignment_state=True,
    )
    assert main_assignment_before == main_assignment_after

    (
        control_wb,
        _task,
        _policy,
        _batch,
        control_assignment_before,
        control_assignment_after,
    ) = compute(
        "current_policy",
        cache_enabled=True,
        assignment_mode="context_eligible_count_matched",
        capture_assignment_state=True,
    )
    assert control_assignment_before != control_assignment_after
    np.testing.assert_array_equal(control_wb.psi_q_rank, main_wb.psi_q_rank)
    assert control_wb.extra_metrics["assignment/context_eligible_count"] == 4.0
    assert control_wb.extra_metrics["assignment/main_positive_count"] == 2.0
    assert control_wb.extra_metrics["assignment/assigned_positive_count"] == 2.0
    np.testing.assert_array_equal(
        np.sort(control_wb.w_p_raw),
        np.sort(main_wb.w_p_raw),
    )
    np.testing.assert_array_equal(
        np.sort(control_wb.omega_critic),
        np.sort(main_wb.omega_critic),
    )
    np.testing.assert_allclose(
        np.sum(control_wb.omega_critic, dtype=np.float64),
        4.0,
        rtol=0.0,
        atol=1e-6,
    )

    invalid = make_config("current_policy", cache_enabled=True)
    invalid.support_query_mode = "uniform_random"
    try:
        invalid.normalize()
    except ValueError as exc:
        assert "support_query_mode" in str(exc)
    else:
        raise AssertionError("invalid support_query_mode did not raise")

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
