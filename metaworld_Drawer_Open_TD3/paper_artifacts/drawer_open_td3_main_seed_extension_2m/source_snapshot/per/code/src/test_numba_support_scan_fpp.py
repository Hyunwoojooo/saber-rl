#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from metacog_lite_v0.numba_support_scan_fpp import (
    NUMBA_AVAILABLE,
    numba_support_predicate_nopython_signatures,
    numba_support_scan_nopython_signatures,
    scan_fetch_pick_and_place_support_exact,
    scan_fetch_pick_and_place_support_predicate,
)
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec


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
    grip: tuple[float, float] = (0.03, 0.03),
) -> np.ndarray:
    obs = np.zeros(25, dtype=np.float32)
    obs[0:3] = np.asarray(ee, dtype=np.float32)
    obs[3:6] = np.asarray(obj, dtype=np.float32)
    obs[9:11] = np.asarray(grip, dtype=np.float32)
    return obs


def reference_scan(
    task: FetchPickAndPlaceTaskSpec,
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    policy_bin: int,
) -> tuple[int, int, int, int, int, int]:
    full_key = task.full_context_hash(query_obs, query_goal, query_h)
    ref_obs = replay_obs[scanned_indices]
    ref_goal = np.repeat(query_goal[None, :], scanned_indices.size, axis=0)
    ref_h = replay_h[scanned_indices]
    ref_hash = task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
    local_idx = scanned_indices[ref_hash == int(full_key)]
    local_bins = replay_action_bin[local_idx].copy()
    if local_bins.shape[0] != local_idx.size or np.any(local_bins < 0):
        local_bins = task.action_bin_batch(replay_action_exec[local_idx], BINS_PER_DIM)
    action_bins = np.unique(local_bins.astype(np.int64, copy=False)) if local_idx.size else np.zeros(0, dtype=np.int64)
    local_count = int(local_idx.size)
    available = min(local_count, M_B)
    psi_a = int(np.any(action_bins == int(policy_bin)))
    psi = int(local_count >= N_LOCAL_MIN and available >= M_Q_MIN and psi_a == 1)
    fallback = 1 - psi
    return local_count, available, psi_a, psi, fallback, int(scanned_indices.size)


def numba_scan(
    task: FetchPickAndPlaceTaskSpec,
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    policy_bin: int,
) -> tuple[int, int, int, int, int, int]:
    return scan_fetch_pick_and_place_support_exact(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=task.full_context_hash(query_obs, query_goal, query_h),
        query_policy_bin=policy_bin,
        hand_obj_bins=np.asarray(task.hand_obj_bins, dtype=np.float32),
        obj_goal_bins=np.asarray(task.obj_goal_bins, dtype=np.float32),
        object_z_bins=np.asarray(task.object_z_bins, dtype=np.float32),
        gripper_gap_bins=np.asarray(task.gripper_gap_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        d_reach=task.d_reach,
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
    )


def numba_predicate_scan(
    task: FetchPickAndPlaceTaskSpec,
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_obs: np.ndarray,
    query_goal: np.ndarray,
    query_h: int,
    policy_bin: int,
) -> tuple[int, int, int, int, int, int, int]:
    return scan_fetch_pick_and_place_support_predicate(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=task.full_context_hash(query_obs, query_goal, query_h),
        query_policy_bin=policy_bin,
        hand_obj_bins=np.asarray(task.hand_obj_bins, dtype=np.float32),
        obj_goal_bins=np.asarray(task.obj_goal_bins, dtype=np.float32),
        object_z_bins=np.asarray(task.object_z_bins, dtype=np.float32),
        gripper_gap_bins=np.asarray(task.gripper_gap_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        d_reach=task.d_reach,
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
    )


def run_case(name: str, replay_obs: np.ndarray, replay_h: np.ndarray, action_bin: np.ndarray, action_exec: np.ndarray, scanned: np.ndarray) -> None:
    task = make_task()
    query_obs = make_obs()
    query_goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    query_h = 20
    policy_action = np.asarray([[0.6, 0.0, 0.0, 0.0]], dtype=np.float32)
    policy_bin = int(task.action_bin_batch(policy_action, BINS_PER_DIM)[0])
    expected = reference_scan(task, replay_obs, replay_h, action_bin, action_exec, scanned, query_obs, query_goal, query_h, policy_bin)
    actual = numba_scan(task, replay_obs, replay_h, action_bin, action_exec, scanned, query_obs, query_goal, query_h, policy_bin)
    assert actual == expected, f"{name}: actual={actual} expected={expected}"
    pred = numba_predicate_scan(task, replay_obs, replay_h, action_bin, action_exec, scanned, query_obs, query_goal, query_h, policy_bin)
    assert pred[2:5] == expected[2:5], f"{name}: predicate={pred} expected={expected}"
    if pred[6] == 0:
        assert pred[:6] == expected, f"{name}: predicate exact fallback={pred} expected={expected}"


def main() -> int:
    if not NUMBA_AVAILABLE:
        print("SKIP: numba is not installed")
        return 0
    task = make_task()
    policy_action = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    other_action = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    policy_bin = int(task.action_bin_batch(policy_action[None, :], BINS_PER_DIM)[0])
    other_bin = int(task.action_bin_batch(other_action[None, :], BINS_PER_DIM)[0])
    match_obs = make_obs()
    nonmatch_obs = make_obs(obj=(0.8, 0.0, 0.44))
    h_match = np.asarray([20], dtype=np.int16)

    run_case("no local matches", np.stack([nonmatch_obs, nonmatch_obs]), np.asarray([20, 20], dtype=np.int16), np.asarray([policy_bin, policy_bin], dtype=np.int64), np.stack([policy_action, policy_action]), np.asarray([0, 1], dtype=np.int64))
    run_case("local_count below threshold", np.stack([match_obs, nonmatch_obs]), np.asarray([20, 20], dtype=np.int16), np.asarray([policy_bin, policy_bin], dtype=np.int64), np.stack([policy_action, policy_action]), np.asarray([0, 1], dtype=np.int64))
    run_case("policy bin absent", np.stack([match_obs, match_obs, match_obs]), np.repeat(h_match, 3), np.asarray([other_bin, other_bin, other_bin], dtype=np.int64), np.stack([other_action, other_action, other_action]), np.asarray([0, 1, 2], dtype=np.int64))
    run_case("positive support", np.stack([match_obs, match_obs, match_obs]), np.repeat(h_match, 3), np.asarray([other_bin, policy_bin, other_bin], dtype=np.int64), np.stack([other_action, policy_action, other_action]), np.asarray([0, 1, 2], dtype=np.int64))
    run_case("truncated scanned input", np.stack([match_obs, nonmatch_obs, match_obs, nonmatch_obs, match_obs]), np.asarray([20, 20, 20, 20, 20], dtype=np.int16), np.asarray([policy_bin, other_bin, other_bin, other_bin, other_bin], dtype=np.int64), np.stack([policy_action, other_action, other_action, other_action, other_action]), np.asarray([0, 2, 4], dtype=np.int64))
    run_case("negative action_bin recompute", np.stack([match_obs, match_obs, match_obs]), np.repeat(h_match, 3), np.asarray([-1, other_bin, other_bin], dtype=np.int64), np.stack([policy_action, other_action, other_action]), np.asarray([0, 1, 2], dtype=np.int64))
    assert len(numba_support_scan_nopython_signatures()) > 0
    assert len(numba_support_predicate_nopython_signatures()) > 0
    print("test_numba_support_scan_fpp.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
