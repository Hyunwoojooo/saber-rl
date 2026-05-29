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
    fetch_slide_support_quality_nopython_signatures,
    make_fetch_slide_support_scan_constants,
    scan_fetch_slide_support_quality_with_constants,
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


def obs_with(
    ee: tuple[float, float, float],
    obj: tuple[float, float, float],
    vel: tuple[float, float, float],
) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0:3] = np.asarray(ee, dtype=np.float32)
    obs[3:6] = np.asarray(obj, dtype=np.float32)
    obs[14:17] = np.asarray(vel, dtype=np.float32)
    obs[20:23] = np.zeros(3, dtype=np.float32)
    return obs


def policy_action() -> np.ndarray:
    return np.asarray([0.2, -0.2, 0.0, 0.4], dtype=np.float32)


def other_action() -> np.ndarray:
    return -policy_action()


def scan_quality_case(
    task: FetchSlideTaskSpec,
    replay_obs: np.ndarray,
    replay_next_obs: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    n_local_min: int = 1,
    m_q_min: int = 1,
    m_b: int = 4,
) -> tuple[int, int, int, int, int, np.float32, np.float32, int, np.float32, np.float32, np.float32, np.float32]:
    constants = make_fetch_slide_support_scan_constants(task)
    full_key = int(
        task.full_context_hash_batch(
            query_obs[None, :],
            query_goal[None, :],
            np.asarray([query_h], dtype=np.int16),
        )[0]
    )
    policy_bin = int(task.action_bin_batch(policy_action()[None, :], ACTION_BINS_PER_DIM)[0])
    replay_h = np.full(replay_obs.shape[0], query_h, dtype=np.int16)
    return scan_fetch_slide_support_quality_with_constants(
        replay_obs=replay_obs,
        replay_next_obs=replay_next_obs,
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
        quality_floor=0.25,
        approach_scale=0.02,
        goal_progress_scale=0.03,
        align_min=0.0,
        strike_speed_scale_mult=2.0,
        near_margin_mult=0.25,
        near_speed_mult=1.0,
    )


def test_phase_quality_cases() -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    p_bin = int(task.action_bin_batch(policy_action()[None, :], ACTION_BINS_PER_DIM)[0])

    approach = obs_with((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    approach_next_mid = obs_with((0.01, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = scan_quality_case(
        task,
        replay_obs=np.stack([approach]).astype(np.float32),
        replay_next_obs=np.stack([approach_next_mid]).astype(np.float32),
        replay_action_bin=np.asarray([p_bin], dtype=np.int64),
        replay_action_exec=policy_action()[None, :],
        scanned_indices=np.asarray([0], dtype=np.int64),
        query_obs=approach,
        query_goal=goal,
        query_h=query_h,
    )
    assert result[3] == 1
    assert 0.45 < float(result[5]) < 0.55

    strike = obs_with((0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    strike_next = obs_with((0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.05, 0.0, 0.0))
    result = scan_quality_case(
        task,
        np.stack([strike]).astype(np.float32),
        np.stack([strike_next]).astype(np.float32),
        np.asarray([p_bin], dtype=np.int64),
        policy_action()[None, :],
        np.asarray([0], dtype=np.int64),
        strike,
        goal,
        query_h,
    )
    assert 0.45 < float(result[5]) < 0.55
    assert float(result[9]) > 0.45
    assert float(result[10]) > 0.99

    slide = obs_with((0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.05, 0.0, 0.0))
    slide_next = obs_with((0.2, 0.0, 0.0), (0.23, 0.0, 0.0), (0.05, 0.0, 0.0))
    result = scan_quality_case(
        task,
        np.stack([slide]).astype(np.float32),
        np.stack([slide_next]).astype(np.float32),
        np.asarray([p_bin], dtype=np.int64),
        policy_action()[None, :],
        np.asarray([0], dtype=np.int64),
        slide,
        goal,
        query_h,
    )
    assert float(result[5]) > 0.99
    assert float(result[8]) > 0.99

    near = obs_with((0.56, 0.0, 0.0), (0.57, 0.0, 0.0), (0.0, 0.0, 0.0))
    near_next_good = obs_with((0.56, 0.0, 0.0), (0.57, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = scan_quality_case(
        task,
        np.stack([near]).astype(np.float32),
        np.stack([near_next_good]).astype(np.float32),
        np.asarray([p_bin], dtype=np.int64),
        policy_action()[None, :],
        np.asarray([0], dtype=np.int64),
        near,
        goal,
        query_h,
    )
    assert float(result[5]) > 0.99
    assert float(result[11]) > 0.99


def test_quality_scanner_cases() -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    p_bin = int(task.action_bin_batch(policy_action()[None, :], ACTION_BINS_PER_DIM)[0])
    other_bin = int(task.action_bin_batch(other_action()[None, :], ACTION_BINS_PER_DIM)[0])

    match = obs_with((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    miss = obs_with((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.2, 0.0, 0.0))
    low_next = match.copy()
    high_next = obs_with((0.04, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))

    result = scan_quality_case(
        task,
        np.stack([miss]).astype(np.float32),
        np.stack([miss]).astype(np.float32),
        np.asarray([p_bin], dtype=np.int64),
        policy_action()[None, :],
        np.asarray([0], dtype=np.int64),
        match,
        goal,
        query_h,
    )
    assert result[0] == 0
    assert result[2] == 0
    assert result[5] == 0.0

    result = scan_quality_case(
        task,
        np.stack([match, match]).astype(np.float32),
        np.stack([high_next, high_next]).astype(np.float32),
        np.asarray([other_bin, other_bin], dtype=np.int64),
        np.stack([other_action(), other_action()]).astype(np.float32),
        np.asarray([0, 1], dtype=np.int64),
        match,
        goal,
        query_h,
        n_local_min=2,
        m_q_min=2,
    )
    assert result[0] == 2
    assert result[2] == 0
    assert result[3] == 0
    assert result[7] == 0

    result = scan_quality_case(
        task,
        np.stack([match, match]).astype(np.float32),
        np.stack([low_next, high_next]).astype(np.float32),
        np.asarray([p_bin, p_bin], dtype=np.int64),
        np.stack([policy_action(), policy_action()]).astype(np.float32),
        np.asarray([0, 1], dtype=np.int64),
        match,
        goal,
        query_h,
        n_local_min=2,
        m_q_min=2,
    )
    assert result[0] == 2
    assert result[3] == 1
    assert result[7] == 2
    assert float(result[5]) > 0.99
    assert float(result[6]) > 0.99
    assert fetch_slide_support_quality_nopython_signatures()


class FixedPolicyAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(action[None, :], dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.to(state.device).repeat(state.shape[0], 1)


def test_runtime_quality_support_verify() -> None:
    if not NUMBA_AVAILABLE:
        return
    task = make_task()
    goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_h = 20
    p_action = policy_action()
    o_action = other_action()
    p_bin = int(task.action_bin_batch(p_action[None, :], ACTION_BINS_PER_DIM)[0])
    o_bin = int(task.action_bin_batch(o_action[None, :], ACTION_BINS_PER_DIM)[0])
    match = obs_with((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    mid_next = obs_with((0.01, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))

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
    cfg.support_quality_mode = "fetch_slide_phase_soft"
    cfg.support_quality_apply_to_weight = True
    cfg.support_quality_floor = 0.25
    cfg.support_quality_approach_scale = 0.02
    cfg.support_quality_goal_progress_scale = 0.03
    cfg.support_quality_align_min = 0.0
    cfg.support_quality_strike_speed_scale_mult = 2.0
    cfg.support_quality_near_margin_mult = 0.25
    cfg.support_quality_near_speed_mult = 1.0

    replay = HERReplayBuffer(capacity=16, obs_dim=OBS_DIM, goal_dim=3, action_dim=ACTION_DIM)
    rows = [
        (match, mid_next, p_action, p_bin),
        (match, match, o_action, o_bin),
    ]
    for step, (obs, next_obs, action, action_bin) in enumerate(rows):
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=action,
            reward_ext=0.0,
            next_observation=next_obs,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=query_h,
            episode_id=step,
            episode_step=0,
            coarse_key=int(task.coarse_hash(obs, query_h)),
            action_bin=int(action_bin),
        )

    obs_batch = np.stack([match, obs_with((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.2, 0.0, 0.0))]).astype(np.float32)
    goal_batch = np.stack([goal, goal]).astype(np.float32)
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
        enable_numba_scan=True,
        enable_predicate_early_exit=True,
    )
    wb = computer.compute(
        batch=batch,
        replay=replay,
        agent=FixedPolicyAgent(p_action),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    assert wb.extra_metrics["runtime/support_quality_enabled"] == 1.0
    assert wb.extra_metrics["runtime/support_quality_numba_enabled"] == 1.0
    np.testing.assert_array_equal(wb.psi_q_rank, np.asarray([1.0, 0.0], dtype=np.float32))
    assert 1.60 < float(wb.w_p_raw[0]) < 1.65
    assert float(wb.w_p_raw[1]) == 1.0
    np.testing.assert_array_equal(wb.omega_actor, np.ones_like(wb.omega_actor, dtype=np.float32))


def main() -> None:
    test_phase_quality_cases()
    test_quality_scanner_cases()
    test_runtime_quality_support_verify()


if __name__ == "__main__":
    main()
