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
N_LOCAL_MIN = 4
M_Q_MIN = 4
M_B = 16


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


def action_set(include_policy: bool = True) -> np.ndarray:
    actions = [
        np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.6, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, -0.6, 0.0], dtype=np.float32),
    ]
    if not include_policy:
        actions[0] = np.asarray([0.0, 0.0, 0.0, 0.6], dtype=np.float32)
    return np.stack(actions)


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
    dmin: int,
) -> tuple[int, int, int, int, int, int, int]:
    full_key = task.full_context_hash(query_obs, query_goal, query_h)
    ref_obs = replay_obs[scanned_indices]
    ref_goal = np.repeat(query_goal[None, :], scanned_indices.size, axis=0)
    ref_h = replay_h[scanned_indices]
    ref_hash = task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
    local_idx = scanned_indices[ref_hash == int(full_key)]
    local_bins = replay_action_bin[local_idx].copy()
    if local_bins.shape[0] != local_idx.size or np.any(local_bins < 0):
        local_bins = task.action_bin_batch(
            replay_action_exec[local_idx],
            BINS_PER_DIM,
        )
    unique_bins = (
        np.unique(local_bins.astype(np.int64, copy=False))
        if local_idx.size
        else np.zeros(0, dtype=np.int64)
    )
    local_count = int(local_idx.size)
    available = min(local_count, M_B)
    policy_seen = int(np.any(unique_bins == int(policy_bin)))
    unique_count = int(unique_bins.size)
    support = int(
        local_count >= N_LOCAL_MIN
        and available >= M_Q_MIN
        and policy_seen == 1
        and unique_count >= int(dmin)
    )
    return (
        local_count,
        available,
        policy_seen,
        unique_count,
        support,
        1 - support,
        int(scanned_indices.size),
    )


def legacy_projection(result: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
    return result[0], result[1], result[2], result[4], result[5], result[6]


def scan_kwargs(
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
    dmin: int,
) -> dict[str, object]:
    return {
        "replay_obs": replay_obs,
        "replay_h": replay_h,
        "replay_action_bin": replay_action_bin,
        "replay_action_exec": replay_action_exec,
        "scanned_indices": scanned_indices,
        "query_goal": query_goal,
        "query_full_key": task.full_context_hash(query_obs, query_goal, query_h),
        "query_policy_bin": policy_bin,
        "hand_obj_bins": np.asarray(task.hand_obj_bins, dtype=np.float32),
        "obj_goal_bins": np.asarray(task.obj_goal_bins, dtype=np.float32),
        "object_z_bins": np.asarray(task.object_z_bins, dtype=np.float32),
        "gripper_gap_bins": np.asarray(task.gripper_gap_bins, dtype=np.float32),
        "horizon_bins": np.asarray(task.horizon_bins, dtype=np.float32),
        "d_reach": task.d_reach,
        "z_lift_min": task.z_lift_min,
        "grip_close_thresh": task.grip_close_thresh,
        "n_local_min": N_LOCAL_MIN,
        "m_q_min": M_Q_MIN,
        "m_b": M_B,
        "ee_start": task.ee_slice[0],
        "object_start": task.object_slice[0],
        "gripper_start": task.gripper_slice[0],
        "action_low": task.action_low,
        "action_high": task.action_high,
        "bins_per_dim": BINS_PER_DIM,
        "support_min_unique_action_bins": dmin,
    }


def make_case(
    actions: np.ndarray,
    stored_bins: np.ndarray | None = None,
) -> tuple[
    FetchPickAndPlaceTaskSpec,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    task = make_task()
    replay_obs = np.stack([make_obs() for _ in range(actions.shape[0])])
    replay_h = np.full(actions.shape[0], 20, dtype=np.int16)
    bins = task.action_bin_batch(actions, BINS_PER_DIM).astype(np.int64)
    if stored_bins is not None:
        bins = stored_bins.astype(np.int64)
    scanned = np.arange(actions.shape[0], dtype=np.int64)
    query_obs = make_obs()
    query_goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    policy_action = np.asarray([[0.6, 0.0, 0.0, 0.0]], dtype=np.float32)
    policy_bin = int(task.action_bin_batch(policy_action, BINS_PER_DIM)[0])
    return (
        task,
        replay_obs,
        replay_h,
        bins,
        actions,
        scanned,
        query_obs,
        query_goal,
        policy_bin,
    )


def assert_case(actions: np.ndarray, dmin: int, expected_support: int) -> tuple[int, ...]:
    (
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        policy_bin,
    ) = make_case(actions)
    kwargs = scan_kwargs(
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        20,
        policy_bin,
        dmin,
    )
    expected = reference_scan(
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        20,
        policy_bin,
        dmin,
    )
    exact = scan_fetch_pick_and_place_support_exact(**kwargs)
    predicate = scan_fetch_pick_and_place_support_predicate(**kwargs)
    assert exact == expected, (exact, expected)
    assert exact[4] == expected_support, exact
    assert predicate[2] == exact[2]
    assert predicate[4:6] == exact[4:6]
    if predicate[7] == 0:
        assert predicate[:7] == exact
    return exact


def test_dmin1_legacy_parity() -> None:
    actions = np.concatenate([action_set(True), action_set(False)], axis=0)
    (
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        policy_bin,
    ) = make_case(actions)
    kwargs = scan_kwargs(
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        20,
        policy_bin,
        1,
    )
    explicit = scan_fetch_pick_and_place_support_exact(**kwargs)
    kwargs.pop("support_min_unique_action_bins")
    defaulted = scan_fetch_pick_and_place_support_exact(**kwargs)
    expected = reference_scan(
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        20,
        policy_bin,
        1,
    )
    assert explicit == defaulted == expected
    legacy = (
        expected[0],
        expected[1],
        expected[2],
        int(
            expected[0] >= N_LOCAL_MIN
            and expected[1] >= M_Q_MIN
            and expected[2] == 1
        ),
        0,
        expected[6],
    )
    legacy = legacy[:4] + (1 - legacy[3], legacy[5])
    assert legacy_projection(explicit) == legacy


def test_same_bin_four() -> None:
    policy = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    actions = np.repeat(policy[None, :], 4, axis=0)
    main = assert_case(actions, dmin=1, expected_support=1)
    diverse = assert_case(actions, dmin=4, expected_support=0)
    assert main[3] == diverse[3] == 1


def test_four_distinct_with_policy() -> None:
    result = assert_case(action_set(True), dmin=4, expected_support=1)
    assert result[2] == 1
    assert result[3] == 4


def test_diverse_without_policy() -> None:
    result = assert_case(action_set(False), dmin=4, expected_support=0)
    assert result[2] == 0
    assert result[3] == 4


def test_negative_bin_recompute() -> None:
    actions = action_set(True)
    stored = np.full(actions.shape[0], -1, dtype=np.int64)
    (
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        policy_bin,
    ) = make_case(actions, stored)
    kwargs = scan_kwargs(
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        20,
        policy_bin,
        4,
    )
    result = scan_fetch_pick_and_place_support_exact(**kwargs)
    assert result[3] == 4
    assert result[4] == 1


def test_early_exit_parity() -> None:
    actions = np.concatenate([action_set(True), action_set(False)], axis=0)
    (
        task,
        obs,
        h,
        bins,
        action_exec,
        scanned,
        query_obs,
        query_goal,
        policy_bin,
    ) = make_case(actions)
    for dmin in (1, 2, 4, 8):
        kwargs = scan_kwargs(
            task,
            obs,
            h,
            bins,
            action_exec,
            scanned,
            query_obs,
            query_goal,
            20,
            policy_bin,
            dmin,
        )
        exact = scan_fetch_pick_and_place_support_exact(**kwargs)
        predicate = scan_fetch_pick_and_place_support_predicate(**kwargs)
        assert predicate[2] == exact[2]
        assert predicate[4:6] == exact[4:6]
        if predicate[7]:
            assert predicate[6] <= exact[6]
            assert predicate[3] >= dmin
        else:
            assert predicate[:7] == exact


def main() -> int:
    if not NUMBA_AVAILABLE:
        print("SKIP: numba is not installed")
        return 0
    test_dmin1_legacy_parity()
    test_same_bin_four()
    test_four_distinct_with_policy()
    test_diverse_without_policy()
    test_negative_bin_recompute()
    test_early_exit_parity()
    assert len(numba_support_scan_nopython_signatures()) > 0
    assert len(numba_support_predicate_nopython_signatures()) > 0
    print("test_numba_support_scan_fpp.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
