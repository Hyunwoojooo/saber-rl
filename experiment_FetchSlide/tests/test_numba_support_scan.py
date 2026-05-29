#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "version_v0_lite" / "src"))

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_fetch_slide import FetchSlideTaskSpec
from metacog_lite_v0.weights import GapStats
from saber_fetch_slide.experiment import RuntimeOptimizedSupportComputer
from saber_fetch_slide.numba_support_scan import (
    NUMBA_AVAILABLE,
    fetch_slide_support_exact_nopython_signatures,
    fetch_slide_support_predicate_nopython_signatures,
    make_fetch_slide_support_scan_constants,
    scan_fetch_slide_support_exact_with_constants,
    scan_fetch_slide_support_predicate_with_constants,
)


ACTION_DIM = 4
OBS_DIM = 25
ACTION_BINS_PER_DIM = 5


def make_task() -> FetchSlideTaskSpec:
    return FetchSlideTaskSpec(
        action_low=-np.ones(ACTION_DIM, dtype=np.float32),
        action_high=np.ones(ACTION_DIM, dtype=np.float32),
        epsilon_g=0.05,
    )


def base_obs() -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0:3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    obs[3:6] = np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
    obs[14:17] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    obs[20:23] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return obs


def nonmatching_obs() -> np.ndarray:
    obs = base_obs()
    obs[3:6] = np.asarray([1.4, 0.0, 0.0], dtype=np.float32)
    obs[14:17] = np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
    return obs


def legacy_scan(
    task: FetchSlideTaskSpec,
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    query_policy_bin: int,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
) -> tuple[int, int, int, int, int]:
    full_key = int(
        task.full_context_hash_batch(
            query_obs[None, :],
            query_goal[None, :],
            np.asarray([query_h], dtype=np.int16),
        )[0]
    )
    ref_obs = replay_obs[scanned_indices]
    ref_goal = np.repeat(query_goal[None, :], scanned_indices.size, axis=0)
    ref_h = replay_h[scanned_indices]
    ref_hash = task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
    local_idx = scanned_indices[ref_hash == full_key]
    if local_idx.size > 0:
        local_bins = replay_action_bin[local_idx].copy()
        if local_bins.shape[0] != local_idx.size or np.any(local_bins < 0):
            local_bins = task.action_bin_batch(replay_action_exec[local_idx], ACTION_BINS_PER_DIM)
        action_bins = np.unique(local_bins.astype(np.int64, copy=False))
    else:
        action_bins = np.zeros((0,), dtype=np.int64)
    local_pool_size = int(local_idx.size)
    available_support_size = min(local_pool_size, int(m_b))
    psi_a_policy = int(query_policy_bin in set(int(x) for x in action_bins.tolist()))
    support_positive = int(
        local_pool_size >= n_local_min and available_support_size >= m_q_min and psi_a_policy == 1
    )
    fallback = 0 if support_positive else 1
    return local_pool_size, available_support_size, psi_a_policy, support_positive, fallback


def run_scanner_case(
    name: str,
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    query_policy_action: np.ndarray,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
) -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    constants = make_fetch_slide_support_scan_constants(task)
    full_key = int(
        task.full_context_hash_batch(
            query_obs[None, :],
            query_goal[None, :],
            np.asarray([query_h], dtype=np.int16),
        )[0]
    )
    policy_bin = int(task.action_bin_batch(query_policy_action[None, :], ACTION_BINS_PER_DIM)[0])
    expected = legacy_scan(
        task,
        replay_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        scanned_indices,
        query_obs,
        query_goal,
        query_h,
        policy_bin,
        n_local_min,
        m_q_min,
        m_b,
    )
    actual = scan_fetch_slide_support_exact_with_constants(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=full_key,
        query_policy_bin=policy_bin,
        constants=constants,
        n_local_min=n_local_min,
        m_q_min=m_q_min,
        m_b=m_b,
        action_bins_per_dim=ACTION_BINS_PER_DIM,
    )
    assert actual == expected, f"{name}: actual={actual} expected={expected}"


def run_predicate_case(
    name: str,
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    query_policy_action: np.ndarray,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    expect_early_exit: bool,
) -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    constants = make_fetch_slide_support_scan_constants(task)
    full_key = int(
        task.full_context_hash_batch(
            query_obs[None, :],
            query_goal[None, :],
            np.asarray([query_h], dtype=np.int16),
        )[0]
    )
    policy_bin = int(task.action_bin_batch(query_policy_action[None, :], ACTION_BINS_PER_DIM)[0])
    expected = legacy_scan(
        task,
        replay_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        scanned_indices,
        query_obs,
        query_goal,
        query_h,
        policy_bin,
        n_local_min,
        m_q_min,
        m_b,
    )
    actual = scan_fetch_slide_support_predicate_with_constants(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=full_key,
        query_policy_bin=policy_bin,
        constants=constants,
        n_local_min=n_local_min,
        m_q_min=m_q_min,
        m_b=m_b,
        action_bins_per_dim=ACTION_BINS_PER_DIM,
    )
    assert actual[2:5] == expected[2:5], f"{name}: semantic actual={actual} expected={expected}"
    assert (actual[6] != 0) == expect_early_exit, f"{name}: early_exit_code={actual[6]}"
    if expect_early_exit:
        assert actual[5] < scanned_indices.size, f"{name}: inspected={actual[5]} scan={scanned_indices.size}"


def test_scanner_cases() -> None:
    task = make_task()
    query_obs = base_obs()
    query_goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    policy_action = np.asarray([0.2, -0.2, 0.0, 0.4], dtype=np.float32)
    policy_bin = int(task.action_bin_batch(policy_action[None, :], ACTION_BINS_PER_DIM)[0])
    other_action = -policy_action
    other_bin = int(task.action_bin_batch(other_action[None, :], ACTION_BINS_PER_DIM)[0])
    assert other_bin != policy_bin

    match = base_obs()
    miss = nonmatching_obs()

    def arrays(obs_rows: list[np.ndarray], actions: list[np.ndarray], bins: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.stack(obs_rows).astype(np.float32),
            np.full(len(obs_rows), query_h, dtype=np.int16),
            np.asarray(bins, dtype=np.int64),
            np.stack(actions).astype(np.float32),
        )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [miss, miss, miss],
        [other_action, other_action, other_action],
        [other_bin, other_bin, other_bin],
    )
    run_scanner_case(
        "no local matches",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.arange(3, dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=2,
        m_q_min=2,
        m_b=4,
    )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [match, match, miss],
        [policy_action, other_action, other_action],
        [policy_bin, other_bin, other_bin],
    )
    run_scanner_case(
        "local_count below threshold",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.arange(3, dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=3,
        m_q_min=2,
        m_b=4,
    )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [match, match, match, match],
        [other_action, other_action, other_action, other_action],
        [other_bin, other_bin, other_bin, other_bin],
    )
    run_scanner_case(
        "policy bin absent",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.arange(4, dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=2,
        m_q_min=2,
        m_b=4,
    )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [match, match, match, match, miss],
        [other_action, policy_action, other_action, other_action, other_action],
        [other_bin, -1, other_bin, other_bin, other_bin],
    )
    run_scanner_case(
        "positive with negative cached action bin",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.arange(5, dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=2,
        m_q_min=2,
        m_b=3,
    )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [match, miss, match, miss, match, match],
        [policy_action, other_action, other_action, other_action, other_action, policy_action],
        [policy_bin, other_bin, other_bin, other_bin, other_bin, policy_bin],
    )
    run_scanner_case(
        "truncated scanned input",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.asarray([0, 2, 4], dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=2,
        m_q_min=2,
        m_b=4,
    )
    if NUMBA_AVAILABLE:
        assert fetch_slide_support_exact_nopython_signatures()


def test_predicate_scanner_cases() -> None:
    task = make_task()
    query_obs = base_obs()
    query_goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    policy_action = np.asarray([0.2, -0.2, 0.0, 0.4], dtype=np.float32)
    other_action = -policy_action
    policy_bin = int(task.action_bin_batch(policy_action[None, :], ACTION_BINS_PER_DIM)[0])
    other_bin = int(task.action_bin_batch(other_action[None, :], ACTION_BINS_PER_DIM)[0])
    match = base_obs()
    miss = nonmatching_obs()

    def arrays(obs_rows: list[np.ndarray], actions: list[np.ndarray], bins: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.stack(obs_rows).astype(np.float32),
            np.full(len(obs_rows), query_h, dtype=np.int16),
            np.asarray(bins, dtype=np.int64),
            np.stack(actions).astype(np.float32),
        )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [match, match, match, match, miss],
        [policy_action, other_action, other_action, other_action, other_action],
        [policy_bin, other_bin, other_bin, other_bin, other_bin],
    )
    run_predicate_case(
        "positive early exit",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.arange(5, dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=2,
        m_q_min=2,
        m_b=4,
        expect_early_exit=True,
    )

    replay_obs, replay_h, replay_bin, replay_action = arrays(
        [match, miss, miss, miss],
        [policy_action, other_action, other_action, other_action],
        [policy_bin, other_bin, other_bin, other_bin],
    )
    run_predicate_case(
        "negative full scan",
        replay_obs,
        replay_h,
        replay_bin,
        replay_action,
        np.arange(4, dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_action,
        n_local_min=3,
        m_q_min=3,
        m_b=4,
        expect_early_exit=False,
    )
    if NUMBA_AVAILABLE:
        assert fetch_slide_support_predicate_nopython_signatures()


class FixedPolicyAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(action[None, :], dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.to(state.device).repeat(state.shape[0], 1)


def test_runtime_computer_verify_with_numba() -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    policy_action = np.asarray([0.2, -0.2, 0.0, 0.4], dtype=np.float32)
    other_action = -policy_action
    query_obs = base_obs()
    query_goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    policy_bin = int(task.action_bin_batch(policy_action[None, :], ACTION_BINS_PER_DIM)[0])
    other_bin = int(task.action_bin_batch(other_action[None, :], ACTION_BINS_PER_DIM)[0])

    cfg = LiteConfig(
        task_name="fetch_slide",
        method="b3_support_only",
        batch_size=3,
        M_B=4,
        M_Q_min=2,
        N_local_min=2,
        local_search_max=10,
        action_bins_per_dim=ACTION_BINS_PER_DIM,
        policy_support_mode="bin",
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.verify_runtime_optimization = True

    replay = HERReplayBuffer(capacity=16, obs_dim=OBS_DIM, goal_dim=3, action_dim=ACTION_DIM)
    rows = [
        (base_obs(), policy_action, policy_bin),
        (base_obs(), other_action, other_bin),
        (base_obs(), other_action, other_bin),
        (nonmatching_obs(), other_action, other_bin),
    ]
    for step, (obs, action, action_bin) in enumerate(rows):
        replay.add_transition(
            observation=obs,
            achieved_goal=query_goal,
            desired_goal=query_goal,
            action_exec=action,
            reward_ext=0.0,
            next_observation=obs,
            next_achieved_goal=query_goal,
            done=False,
            timeout=False,
            h=query_h,
            episode_id=step,
            episode_step=0,
            coarse_key=int(task.coarse_hash(obs, query_h)),
            action_bin=int(action_bin),
        )

    obs_batch = np.stack([query_obs, query_obs, nonmatching_obs()]).astype(np.float32)
    goal_batch = np.stack([query_goal, query_goal, query_goal]).astype(np.float32)
    batch = SampleBatch(
        indices=np.arange(3, dtype=np.int64),
        observation=obs_batch,
        achieved_goal=goal_batch.copy(),
        desired_goal_train=goal_batch.copy(),
        next_observation=obs_batch.copy(),
        next_achieved_goal=goal_batch.copy(),
        action_exec=np.zeros((3, ACTION_DIM), dtype=np.float32),
        reward_ext=np.zeros(3, dtype=np.float32),
        done=np.zeros(3, dtype=np.bool_),
        timeout=np.zeros(3, dtype=np.bool_),
        h=np.full(3, query_h, dtype=np.int16),
        her_relabel_ratio=1.0,
    )
    computer = RuntimeOptimizedSupportComputer(
        cfg,
        task,
        GapStats(maxlen=100, n_min=10, q_low=0.6, q_high=0.95, eps_g=1e-6),
    )
    wb = computer.compute(
        batch=batch,
        replay=replay,
        agent=FixedPolicyAgent(policy_action),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    assert wb.extra_metrics["runtime/numba_support_scan_enabled"] == 1.0
    assert wb.extra_metrics["runtime/numba_support_queries"] > 0.0
    np.testing.assert_array_equal(wb.psi_q_rank, np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(wb.psi_a_policy, np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(wb.q_candidate_fallback, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def test_runtime_computer_verify_with_predicate_early_exit() -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    policy_action = np.asarray([0.2, -0.2, 0.0, 0.4], dtype=np.float32)
    other_action = -policy_action
    query_obs = base_obs()
    query_goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    policy_bin = int(task.action_bin_batch(policy_action[None, :], ACTION_BINS_PER_DIM)[0])
    other_bin = int(task.action_bin_batch(other_action[None, :], ACTION_BINS_PER_DIM)[0])

    cfg = LiteConfig(
        task_name="fetch_slide",
        method="b3_support_only",
        batch_size=2,
        M_B=4,
        M_Q_min=2,
        N_local_min=2,
        local_search_max=10,
        action_bins_per_dim=ACTION_BINS_PER_DIM,
        policy_support_mode="bin",
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.verify_runtime_optimization = True

    replay = HERReplayBuffer(capacity=16, obs_dim=OBS_DIM, goal_dim=3, action_dim=ACTION_DIM)
    rows = [
        (base_obs(), policy_action, policy_bin),
        (base_obs(), other_action, other_bin),
        (base_obs(), other_action, other_bin),
        (nonmatching_obs(), other_action, other_bin),
    ]
    for step, (obs, action, action_bin) in enumerate(rows):
        replay.add_transition(
            observation=obs,
            achieved_goal=query_goal,
            desired_goal=query_goal,
            action_exec=action,
            reward_ext=0.0,
            next_observation=obs,
            next_achieved_goal=query_goal,
            done=False,
            timeout=False,
            h=query_h,
            episode_id=step,
            episode_step=0,
            coarse_key=int(task.coarse_hash(obs, query_h)),
            action_bin=int(action_bin),
        )

    obs_batch = np.stack([query_obs, nonmatching_obs()]).astype(np.float32)
    goal_batch = np.stack([query_goal, query_goal]).astype(np.float32)
    batch = SampleBatch(
        indices=np.arange(2, dtype=np.int64),
        observation=obs_batch,
        achieved_goal=goal_batch.copy(),
        desired_goal_train=goal_batch.copy(),
        next_observation=obs_batch.copy(),
        next_achieved_goal=goal_batch.copy(),
        action_exec=np.zeros((2, ACTION_DIM), dtype=np.float32),
        reward_ext=np.zeros(2, dtype=np.float32),
        done=np.zeros(2, dtype=np.bool_),
        timeout=np.zeros(2, dtype=np.bool_),
        h=np.full(2, query_h, dtype=np.int16),
        her_relabel_ratio=1.0,
    )
    computer = RuntimeOptimizedSupportComputer(
        cfg,
        task,
        GapStats(maxlen=100, n_min=10, q_low=0.6, q_high=0.95, eps_g=1e-6),
        enable_predicate_early_exit=True,
    )
    wb = computer.compute(
        batch=batch,
        replay=replay,
        agent=FixedPolicyAgent(policy_action),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    assert wb.extra_metrics["runtime/predicate_early_exit_enabled"] == 1.0
    assert wb.extra_metrics["runtime/predicate_early_exit_rate"] > 0.0
    np.testing.assert_array_equal(wb.psi_q_rank, np.asarray([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(wb.q_candidate_fallback, np.asarray([0.0, 1.0], dtype=np.float32))


def main() -> None:
    test_scanner_cases()
    test_predicate_scanner_cases()
    test_runtime_computer_verify_with_numba()
    test_runtime_computer_verify_with_predicate_early_exit()


if __name__ == "__main__":
    main()
