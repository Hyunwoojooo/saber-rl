#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.numba_support_scan_fpp import (
    NUMBA_AVAILABLE,
    numba_support_quality_nopython_signatures,
    scan_fpp_support_quality_reference,
    scan_fpp_support_quality_with_constants,
)
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec
from metacog_lite_v0.weights import GapStats, LiteWeightComputer

BINS_PER_DIM = 5
N_LOCAL_MIN = 2
M_Q_MIN = 2
M_B = 4


def make_task() -> FetchPickAndPlaceTaskSpec:
    return FetchPickAndPlaceTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )


def make_obs(
    ee: tuple[float, float, float] = (0.0, 0.0, 0.0),
    obj: tuple[float, float, float] = (0.04, 0.0, 0.44),
    gap: float = 0.06,
) -> np.ndarray:
    obs = np.zeros(25, dtype=np.float32)
    obs[0:3] = np.asarray(ee, dtype=np.float32)
    obs[3:6] = np.asarray(obj, dtype=np.float32)
    obs[9:11] = np.asarray([0.5 * gap, 0.5 * gap], dtype=np.float32)
    return obs


def quality_kwargs(
    task: FetchPickAndPlaceTaskSpec,
    replay_obs: np.ndarray,
    replay_next_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    query_policy_bin: int,
    mode: str = "fpp_phase_soft",
) -> dict[str, object]:
    return dict(
        replay_obs=replay_obs,
        replay_next_obs=replay_next_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=task.full_context_hash(query_obs, query_goal, query_h),
        query_policy_bin=query_policy_bin,
        hand_obj_bins=np.asarray(task.hand_obj_bins, dtype=np.float32),
        obj_goal_bins=np.asarray(task.obj_goal_bins, dtype=np.float32),
        object_z_bins=np.asarray(task.object_z_bins, dtype=np.float32),
        gripper_gap_bins=np.asarray(task.gripper_gap_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        d_reach=task.d_reach,
        d_goal_near=task.d_goal_near,
        z_lift_min=task.z_lift_min,
        grip_close_thresh=task.grip_close_thresh,
        n_local_min=N_LOCAL_MIN,
        m_q_min=M_Q_MIN,
        m_b=M_B,
        ee_start=task.ee_slice[0],
        object_start=task.object_slice[0],
        gripper_start=task.gripper_slice[0],
        action_low=task.action_low,
        action_high=task.action_high,
        bins_per_dim=BINS_PER_DIM,
        quality_mode_id=1 if mode == "fpp_phase_soft" else 2,
        quality_floor=0.5,
        hand_progress_scale=0.03,
        goal_progress_scale=0.03,
        z_progress_scale=0.03,
        grip_progress_scale=0.02,
        near_margin_mult=0.5,
    )


def assert_quality_scan_case(name: str, kwargs: dict[str, object], expected_quality: float | None = None) -> None:
    reference = scan_fpp_support_quality_reference(**kwargs)
    if NUMBA_AVAILABLE:
        actual = scan_fpp_support_quality_with_constants(**kwargs)
        assert actual[:5] == reference[:5], f"{name}: coverage mismatch actual={actual} reference={reference}"
        np.testing.assert_allclose(actual[5:13], reference[5:13], atol=1e-6, err_msg=name)
        assert actual[13] == reference[13]
        assert len(numba_support_quality_nopython_signatures()) > 0
    if expected_quality is not None:
        assert abs(reference[5] - expected_quality) < 1e-5, f"{name}: quality={reference[5]} expected={expected_quality}"


def test_quality_scanner_cases() -> None:
    task = make_task()
    policy_action = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    other_action = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    policy_bin = int(task.action_bin_batch(policy_action[None, :], BINS_PER_DIM)[0])
    other_bin = int(task.action_bin_batch(other_action[None, :], BINS_PER_DIM)[0])
    query_goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    query_h = 20
    query_obs = make_obs()
    match = make_obs()
    no_match = make_obs(obj=(0.8, 0.0, 0.44))
    high_next = make_obs(ee=(0.04, 0.0, 0.42), obj=(0.04, 0.0, 0.44))
    low_next = match.copy()

    assert_quality_scan_case(
        "no local match",
        quality_kwargs(
            task,
            np.stack([no_match, no_match]),
            np.stack([no_match, no_match]),
            np.asarray([20, 20], dtype=np.int16),
            np.asarray([policy_bin, policy_bin], dtype=np.int64),
            np.stack([policy_action, policy_action]),
            np.asarray([0, 1], dtype=np.int64),
            query_obs,
            query_goal,
            query_h,
            policy_bin,
        ),
        expected_quality=0.0,
    )
    assert_quality_scan_case(
        "local match no same action bin",
        quality_kwargs(
            task,
            np.stack([match, match]),
            np.stack([high_next, high_next]),
            np.asarray([20, 20], dtype=np.int16),
            np.asarray([other_bin, other_bin], dtype=np.int64),
            np.stack([other_action, other_action]),
            np.asarray([0, 1], dtype=np.int64),
            query_obs,
            query_goal,
            query_h,
            policy_bin,
        ),
        expected_quality=0.0,
    )
    low_kwargs = quality_kwargs(
        task,
        np.stack([match, match]),
        np.stack([low_next, low_next]),
        np.asarray([20, 20], dtype=np.int16),
        np.asarray([policy_bin, policy_bin], dtype=np.int64),
        np.stack([policy_action, policy_action]),
        np.asarray([0, 1], dtype=np.int64),
        query_obs,
        query_goal,
        query_h,
        policy_bin,
    )
    assert_quality_scan_case("same bin low quality", low_kwargs, expected_quality=0.0)
    low = scan_fpp_support_quality_reference(**low_kwargs)
    assert low[3] == 1 and abs(low[6] - 0.5) < 1e-6

    assert_quality_scan_case(
        "same bin high quality",
        quality_kwargs(
            task,
            np.stack([match, match]),
            np.stack([high_next, high_next]),
            np.asarray([20, 20], dtype=np.int16),
            np.asarray([policy_bin, policy_bin], dtype=np.int64),
            np.stack([policy_action, policy_action]),
            np.asarray([0, 1], dtype=np.int64),
            query_obs,
            query_goal,
            query_h,
            policy_bin,
        ),
        expected_quality=1.0,
    )
    assert_quality_scan_case(
        "multiple same bin takes max quality",
        quality_kwargs(
            task,
            np.stack([match, match, match]),
            np.stack([low_next, high_next, low_next]),
            np.asarray([20, 20, 20], dtype=np.int16),
            np.asarray([policy_bin, policy_bin, policy_bin], dtype=np.int64),
            np.stack([policy_action, policy_action, policy_action]),
            np.asarray([0, 1, 2], dtype=np.int64),
            query_obs,
            query_goal,
            query_h,
            policy_bin,
        ),
        expected_quality=1.0,
    )

    place_obs = make_obs(ee=(0.04, 0.0, 0.70), obj=(0.04, 0.0, 0.70), gap=0.02)
    place_goal = np.asarray([0.045, 0.0, 0.70], dtype=np.float32)
    assert task.phase_batch(place_obs[None, :], place_goal[None, :])[0] == task.PHASE_PLACE
    assert_quality_scan_case(
        "near-goal no-worsen place quality",
        quality_kwargs(
            task,
            np.stack([place_obs, place_obs]),
            np.stack([place_obs, place_obs]),
            np.asarray([20, 20], dtype=np.int16),
            np.asarray([policy_bin, policy_bin], dtype=np.int64),
            np.stack([policy_action, policy_action]),
            np.asarray([0, 1], dtype=np.int64),
            place_obs,
            place_goal,
            query_h,
            policy_bin,
        ),
        expected_quality=1.0,
    )


class StaticAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(action[None, :], dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.to(state.device).repeat(state.shape[0], 1)


def make_config(mode: str = "none", apply: bool = False, numba_enabled: bool = True) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=4,
        M_B=4,
        M_Q_min=2,
        N_local_min=2,
        local_search_max=8,
        action_bins_per_dim=5,
        lambda_support=1.0,
        policy_support_mode="bin",
        support_scan_mode="deterministic_tail_v1",
        weighting_mode="batch_norm",
        weight_scope="critic_only",
        support_quality_mode=mode,
        support_quality_apply_to_weight=apply,
        support_quality_floor=0.5,
    )
    cfg.normalize()
    cfg.numba_support_scan = bool(numba_enabled)
    cfg.support_predicate_early_exit = True
    cfg.support_diagnostic_audit_interval = 1
    return cfg


def make_replay(task: FetchPickAndPlaceTaskSpec, cfg: LiteConfig, next_same: bool = True) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    policy_action = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = make_obs()
    next_obs = obs.copy() if next_same else make_obs(ee=(0.04, 0.0, 0.42), obj=(0.04, 0.0, 0.44))
    for step in range(4):
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=policy_action,
            reward_ext=0.0,
            next_observation=next_obs,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=20,
            episode_id=step,
            episode_step=0,
            coarse_key=task.coarse_hash(obs, 20),
            action_bin=int(task.action_bin_batch(policy_action[None, :], cfg.action_bins_per_dim)[0]),
        )
    return replay


def make_batch() -> SampleBatch:
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    obs = np.stack([make_obs(), make_obs(), make_obs(), make_obs()])
    goals = np.stack([goal, goal, goal, goal])
    return SampleBatch(
        indices=np.arange(4, dtype=np.int64),
        observation=obs,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=obs.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.zeros((4, 4), dtype=np.float32),
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=np.asarray([20, 20, 20, 20], dtype=np.int16),
        her_relabel_ratio=1.0,
    )


def compute(cfg: LiteConfig, replay: HERReplayBuffer, task: FetchPickAndPlaceTaskSpec):
    agent = StaticAgent(np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32))
    return LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
        make_batch(), replay, agent, np.random.default_rng(0), torch.device("cpu")
    )


def test_quality_integration_weights() -> None:
    task = make_task()
    cfg_none = make_config("none", False)
    cfg_diag = make_config("fpp_phase_soft", False)
    cfg_apply = make_config("fpp_phase_soft", True)
    replay = make_replay(task, cfg_none, next_same=True)

    wb_none = compute(cfg_none, replay, task)
    wb_diag = compute(cfg_diag, replay, task)
    np.testing.assert_array_equal(wb_none.psi_q_rank, wb_diag.psi_q_rank)
    np.testing.assert_array_equal(wb_none.w_p_raw, wb_diag.w_p_raw)
    np.testing.assert_array_equal(wb_none.omega_critic, wb_diag.omega_critic)
    assert wb_diag.extra_metrics["runtime/support_quality_enabled"] == 1.0
    assert wb_diag.extra_metrics["runtime/support_quality_apply_to_weight"] == 0.0
    assert np.allclose(wb_diag.support_quality[wb_diag.psi_q_rank > 0.5], 0.0)
    assert np.allclose(wb_diag.support_score[wb_diag.psi_q_rank > 0.5], 0.5)

    wb_apply = compute(cfg_apply, replay, task)
    assert wb_apply.extra_metrics["runtime/support_quality_apply_to_weight"] == 1.0
    assert np.allclose(wb_apply.w_p_raw[wb_apply.psi_q_rank > 0.5], 1.5)
    assert np.all(wb_none.w_p_raw[wb_none.psi_q_rank > 0.5] > wb_apply.w_p_raw[wb_apply.psi_q_rank > 0.5])
    assert np.allclose(wb_apply.omega_actor, 1.0)


def main() -> int:
    test_quality_scanner_cases()
    test_quality_integration_weights()
    print("test_fpp_support_quality.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
