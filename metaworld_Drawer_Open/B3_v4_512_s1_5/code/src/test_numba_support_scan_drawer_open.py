from __future__ import annotations

import numpy as np

from metacog_lite_v0.numba_support_scan_drawer_open import (
    NUMBA_AVAILABLE,
    numba_support_predicate_nopython_signatures,
    numba_support_scan_nopython_signatures,
    scan_drawer_open_support_exact,
    scan_drawer_open_support_predicate,
)
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec


def main() -> None:
    if not NUMBA_AVAILABLE:
        print("numba unavailable; skipped")
        return

    task = MetaWorldDrawerOpenTaskSpec(
        action_low=np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        epsilon_g=0.05,
    )
    obs = np.zeros((8, 12), dtype=np.float32)
    goal = np.asarray([0.10, 0.20, 0.30], dtype=np.float32)
    obs[:, 0:3] = np.asarray([0.08, 0.18, 0.30], dtype=np.float32)
    obs[:, 3] = 0.0
    obs[:, 4:7] = np.asarray([0.09, 0.19, 0.30], dtype=np.float32)
    obs[4:, 4:7] = np.asarray([0.50, 0.50, 0.50], dtype=np.float32)
    h = np.asarray([20] * 8, dtype=np.int16)
    actions = np.zeros((8, 4), dtype=np.float32)
    action_bins = task.action_bin_batch(actions, bins_per_dim=5)
    scanned = np.arange(8, dtype=np.int64)
    query_full_key = int(task.full_context_hash_batch(obs[:1], goal[None, :], h[:1])[0])
    query_policy_bin = int(action_bins[0])

    exact = scan_drawer_open_support_exact(
        replay_obs=obs,
        replay_h=h,
        replay_action_bin=action_bins,
        replay_action_exec=actions,
        scanned_indices=scanned,
        query_goal=goal,
        query_full_key=query_full_key,
        query_policy_bin=query_policy_bin,
        hand_handle_bins=np.asarray(task.hand_handle_bins, dtype=np.float32),
        handle_goal_bins=np.asarray(task.handle_goal_bins, dtype=np.float32),
        handle_y_bins=np.asarray(task.handle_y_bins, dtype=np.float32),
        gripper_bins=np.asarray(task.gripper_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        d_reach=task.d_reach,
        d_goal_near=task.d_goal_near,
        d_opening_mid=task.d_opening_mid,
        n_local_min=4,
        m_q_min=4,
        m_b=16,
        ee_start=task.ee_slice[0],
        handle_start=task.handle_slice[0],
        gripper_start=task.gripper_slice[0],
        action_low=task.action_low,
        action_high=task.action_high,
        bins_per_dim=5,
    )
    predicate = scan_drawer_open_support_predicate(
        replay_obs=obs,
        replay_h=h,
        replay_action_bin=action_bins,
        replay_action_exec=actions,
        scanned_indices=scanned,
        query_goal=goal,
        query_full_key=query_full_key,
        query_policy_bin=query_policy_bin,
        hand_handle_bins=np.asarray(task.hand_handle_bins, dtype=np.float32),
        handle_goal_bins=np.asarray(task.handle_goal_bins, dtype=np.float32),
        handle_y_bins=np.asarray(task.handle_y_bins, dtype=np.float32),
        gripper_bins=np.asarray(task.gripper_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        d_reach=task.d_reach,
        d_goal_near=task.d_goal_near,
        d_opening_mid=task.d_opening_mid,
        n_local_min=4,
        m_q_min=4,
        m_b=16,
        ee_start=task.ee_slice[0],
        handle_start=task.handle_slice[0],
        gripper_start=task.gripper_slice[0],
        action_low=task.action_low,
        action_high=task.action_high,
        bins_per_dim=5,
    )
    assert exact[0] == 4, exact
    assert exact[1] == 4, exact
    assert exact[2] == 1, exact
    assert exact[3] == 1, exact
    assert predicate[0] >= 4, predicate
    assert predicate[2] == 1, predicate
    assert predicate[3] == 1, predicate
    assert len(numba_support_scan_nopython_signatures()) > 0
    assert len(numba_support_predicate_nopython_signatures()) > 0
    print("test_numba_support_scan_drawer_open.py passed")


if __name__ == "__main__":
    main()
