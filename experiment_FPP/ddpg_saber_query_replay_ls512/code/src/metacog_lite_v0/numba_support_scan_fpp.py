from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import numba as _numba
except Exception:  # pragma: no cover - used when numba is not installed.
    _numba = None


NUMBA_AVAILABLE = _numba is not None


if NUMBA_AVAILABLE:

    @_numba.njit
    def _bin_scalar(value: float, boundaries: np.ndarray) -> int:
        out = 0
        for j in range(boundaries.shape[0]):
            if value > boundaries[j]:
                out += 1
        return out

    @_numba.njit
    def _sqrt32(value: float) -> np.float32:
        if value <= 0.0:
            return np.float32(0.0)
        return np.float32(math.sqrt(np.float32(value)))

    @_numba.njit
    def _action_bin_one(
        action: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        bins_per_dim: int,
    ) -> int:
        acc = 0
        multiplier = 1
        bins_max = float(bins_per_dim - 1)
        for j in range(action.shape[0]):
            center = np.float32(0.5) * (action_high[j] + action_low[j])
            scale = max(np.float32(1e-6), np.float32(0.5) * (action_high[j] - action_low[j]))
            norm = (action[j] - center) / scale
            scaled = (norm + np.float32(1.0)) * np.float32(0.5) * float(bins_per_dim)
            if scaled < 0.0:
                clipped = 0.0
            elif scaled > bins_max:
                clipped = bins_max
            else:
                clipped = scaled
            acc += int(math.floor(clipped)) * multiplier
            multiplier *= int(bins_per_dim)
        return acc

    @_numba.njit
    def _scan_fetch_pick_and_place_support_exact_jit(
        replay_obs: np.ndarray,
        replay_h: np.ndarray,
        replay_action_bin: np.ndarray,
        replay_action_exec: np.ndarray,
        scanned_indices: np.ndarray,
        query_goal: np.ndarray,
        query_full_key: int,
        query_policy_bin: int,
        hand_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        object_z_bins: np.ndarray,
        gripper_gap_bins: np.ndarray,
        horizon_bins: np.ndarray,
        d_reach: float,
        z_lift_min: float,
        grip_close_thresh: float,
        n_local_min: int,
        m_q_min: int,
        m_b: int,
        ee_start: int,
        object_start: int,
        gripper_start: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        bins_per_dim: int,
        require_action_membership: bool,
    ) -> Tuple[int, int, int, int, int, int]:
        q_phase = query_full_key // 1_000_000
        rem = query_full_key - q_phase * 1_000_000
        q_hand_bin = rem // 100_000
        rem = rem - q_hand_bin * 100_000
        q_goal_bin = rem // 10_000
        rem = rem - q_goal_bin * 10_000
        q_z_bin = rem // 1_000
        rem = rem - q_z_bin * 1_000
        q_grip_bin = rem // 100
        q_h_bin = rem - q_grip_bin * 100

        local_count = 0
        policy_seen = 0 if require_action_membership else 1
        n_scanned = int(scanned_indices.shape[0])

        for pos in range(n_scanned):
            idx = int(scanned_indices[pos])

            ee_x = replay_obs[idx, ee_start]
            ee_y = replay_obs[idx, ee_start + 1]
            ee_z = replay_obs[idx, ee_start + 2]
            obj_x = replay_obs[idx, object_start]
            obj_y = replay_obs[idx, object_start + 1]
            obj_z = replay_obs[idx, object_start + 2]
            gripper_gap = replay_obs[idx, gripper_start] + replay_obs[idx, gripper_start + 1]

            dx_hand = ee_x - obj_x
            dy_hand = ee_y - obj_y
            dz_hand = ee_z - obj_z
            d_hand_obj = _sqrt32(dx_hand * dx_hand + dy_hand * dy_hand + dz_hand * dz_hand)

            dx_goal = obj_x - query_goal[0]
            dy_goal = obj_y - query_goal[1]
            dz_goal = obj_z - query_goal[2]
            d_obj_goal = _sqrt32(dx_goal * dx_goal + dy_goal * dy_goal + dz_goal * dz_goal)

            phase = 3
            if obj_z <= z_lift_min:
                phase = 2
            if obj_z <= z_lift_min and gripper_gap > grip_close_thresh:
                phase = 1
            if d_hand_obj > d_reach:
                phase = 0

            if phase != q_phase:
                continue
            if _bin_scalar(d_hand_obj, hand_obj_bins) != q_hand_bin:
                continue
            if _bin_scalar(d_obj_goal, obj_goal_bins) != q_goal_bin:
                continue
            if _bin_scalar(obj_z, object_z_bins) != q_z_bin:
                continue
            if _bin_scalar(gripper_gap, gripper_gap_bins) != q_grip_bin:
                continue
            if _bin_scalar(np.float32(replay_h[idx]), horizon_bins) != q_h_bin:
                continue

            local_count += 1
            if require_action_membership:
                action_bin = int(replay_action_bin[idx])
                if action_bin < 0:
                    action_bin = _action_bin_one(
                        replay_action_exec[idx],
                        action_low,
                        action_high,
                        bins_per_dim,
                    )
                if action_bin == query_policy_bin:
                    policy_seen = 1

        available_support_size = local_count
        if available_support_size > m_b:
            available_support_size = m_b
        support_positive = 0
        if local_count >= n_local_min and available_support_size >= m_q_min and policy_seen == 1:
            support_positive = 1
        fallback = 1 - support_positive
        return local_count, available_support_size, policy_seen, support_positive, fallback, n_scanned


    @_numba.njit
    def _scan_fetch_pick_and_place_support_predicate_jit(
        replay_obs: np.ndarray,
        replay_h: np.ndarray,
        replay_action_bin: np.ndarray,
        replay_action_exec: np.ndarray,
        scanned_indices: np.ndarray,
        query_goal: np.ndarray,
        query_full_key: int,
        query_policy_bin: int,
        hand_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        object_z_bins: np.ndarray,
        gripper_gap_bins: np.ndarray,
        horizon_bins: np.ndarray,
        d_reach: float,
        z_lift_min: float,
        grip_close_thresh: float,
        n_local_min: int,
        m_q_min: int,
        m_b: int,
        ee_start: int,
        object_start: int,
        gripper_start: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        bins_per_dim: int,
        require_action_membership: bool,
    ) -> Tuple[int, int, int, int, int, int, int]:
        q_phase = query_full_key // 1_000_000
        rem = query_full_key - q_phase * 1_000_000
        q_hand_bin = rem // 100_000
        rem = rem - q_hand_bin * 100_000
        q_goal_bin = rem // 10_000
        rem = rem - q_goal_bin * 10_000
        q_z_bin = rem // 1_000
        rem = rem - q_z_bin * 1_000
        q_grip_bin = rem // 100
        q_h_bin = rem - q_grip_bin * 100

        local_count = 0
        policy_seen = 0 if require_action_membership else 1
        n_scanned = int(scanned_indices.shape[0])
        required_count = n_local_min
        if m_q_min > required_count:
            required_count = m_q_min

        for pos in range(n_scanned):
            idx = int(scanned_indices[pos])

            ee_x = replay_obs[idx, ee_start]
            ee_y = replay_obs[idx, ee_start + 1]
            ee_z = replay_obs[idx, ee_start + 2]
            obj_x = replay_obs[idx, object_start]
            obj_y = replay_obs[idx, object_start + 1]
            obj_z = replay_obs[idx, object_start + 2]
            gripper_gap = replay_obs[idx, gripper_start] + replay_obs[idx, gripper_start + 1]

            dx_hand = ee_x - obj_x
            dy_hand = ee_y - obj_y
            dz_hand = ee_z - obj_z
            d_hand_obj = _sqrt32(dx_hand * dx_hand + dy_hand * dy_hand + dz_hand * dz_hand)

            dx_goal = obj_x - query_goal[0]
            dy_goal = obj_y - query_goal[1]
            dz_goal = obj_z - query_goal[2]
            d_obj_goal = _sqrt32(dx_goal * dx_goal + dy_goal * dy_goal + dz_goal * dz_goal)

            phase = 3
            if obj_z <= z_lift_min:
                phase = 2
            if obj_z <= z_lift_min and gripper_gap > grip_close_thresh:
                phase = 1
            if d_hand_obj > d_reach:
                phase = 0

            if phase != q_phase:
                continue
            if _bin_scalar(d_hand_obj, hand_obj_bins) != q_hand_bin:
                continue
            if _bin_scalar(d_obj_goal, obj_goal_bins) != q_goal_bin:
                continue
            if _bin_scalar(obj_z, object_z_bins) != q_z_bin:
                continue
            if _bin_scalar(gripper_gap, gripper_gap_bins) != q_grip_bin:
                continue
            if _bin_scalar(np.float32(replay_h[idx]), horizon_bins) != q_h_bin:
                continue

            local_count += 1
            if require_action_membership:
                action_bin = int(replay_action_bin[idx])
                if action_bin < 0:
                    action_bin = _action_bin_one(
                        replay_action_exec[idx],
                        action_low,
                        action_high,
                        bins_per_dim,
                    )
                if action_bin == query_policy_bin:
                    policy_seen = 1

            if local_count >= required_count and policy_seen == 1 and m_b >= m_q_min:
                available_support_size = local_count
                if available_support_size > m_b:
                    available_support_size = m_b
                return local_count, available_support_size, policy_seen, 1, 0, pos + 1, 1

        available_support_size = local_count
        if available_support_size > m_b:
            available_support_size = m_b
        support_positive = 0
        if local_count >= n_local_min and available_support_size >= m_q_min and policy_seen == 1:
            support_positive = 1
        fallback = 1 - support_positive
        return local_count, available_support_size, policy_seen, support_positive, fallback, n_scanned, 0

else:
    _scan_fetch_pick_and_place_support_exact_jit = None
    _scan_fetch_pick_and_place_support_predicate_jit = None


def scan_fetch_pick_and_place_support_exact(
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    hand_obj_bins: np.ndarray,
    obj_goal_bins: np.ndarray,
    object_z_bins: np.ndarray,
    gripper_gap_bins: np.ndarray,
    horizon_bins: np.ndarray,
    d_reach: float,
    z_lift_min: float,
    grip_close_thresh: float,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    ee_start: int = 0,
    object_start: int = 3,
    gripper_start: int = 9,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    bins_per_dim: int = 5,
    require_action_membership: bool = True,
) -> tuple[int, int, int, int, int, int]:
    if not NUMBA_AVAILABLE or _scan_fetch_pick_and_place_support_exact_jit is None:
        raise RuntimeError("numba is not available")
    if action_low is None or action_high is None:
        raise ValueError("action_low and action_high are required for exact negative action-bin handling")
    result = _scan_fetch_pick_and_place_support_exact_jit(
        np.asarray(replay_obs, dtype=np.float32),
        np.asarray(replay_h),
        np.asarray(replay_action_bin, dtype=np.int64),
        np.asarray(replay_action_exec, dtype=np.float32),
        np.asarray(scanned_indices, dtype=np.int64),
        np.asarray(query_goal, dtype=np.float32),
        int(query_full_key),
        int(query_policy_bin),
        np.asarray(hand_obj_bins, dtype=np.float32),
        np.asarray(obj_goal_bins, dtype=np.float32),
        np.asarray(object_z_bins, dtype=np.float32),
        np.asarray(gripper_gap_bins, dtype=np.float32),
        np.asarray(horizon_bins, dtype=np.float32),
        float(d_reach),
        float(z_lift_min),
        float(grip_close_thresh),
        int(n_local_min),
        int(m_q_min),
        int(m_b),
        int(ee_start),
        int(object_start),
        int(gripper_start),
        np.asarray(action_low, dtype=np.float32),
        np.asarray(action_high, dtype=np.float32),
        int(bins_per_dim),
        bool(require_action_membership),
    )
    return tuple(int(v) for v in result)



def scan_fetch_pick_and_place_support_predicate(
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    hand_obj_bins: np.ndarray,
    obj_goal_bins: np.ndarray,
    object_z_bins: np.ndarray,
    gripper_gap_bins: np.ndarray,
    horizon_bins: np.ndarray,
    d_reach: float,
    z_lift_min: float,
    grip_close_thresh: float,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    ee_start: int = 0,
    object_start: int = 3,
    gripper_start: int = 9,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    bins_per_dim: int = 5,
    require_action_membership: bool = True,
) -> tuple[int, int, int, int, int, int, int]:
    if not NUMBA_AVAILABLE or _scan_fetch_pick_and_place_support_predicate_jit is None:
        raise RuntimeError("numba is not available")
    if action_low is None or action_high is None:
        raise ValueError("action_low and action_high are required for exact negative action-bin handling")
    result = _scan_fetch_pick_and_place_support_predicate_jit(
        np.asarray(replay_obs, dtype=np.float32),
        np.asarray(replay_h),
        np.asarray(replay_action_bin, dtype=np.int64),
        np.asarray(replay_action_exec, dtype=np.float32),
        np.asarray(scanned_indices, dtype=np.int64),
        np.asarray(query_goal, dtype=np.float32),
        int(query_full_key),
        int(query_policy_bin),
        np.asarray(hand_obj_bins, dtype=np.float32),
        np.asarray(obj_goal_bins, dtype=np.float32),
        np.asarray(object_z_bins, dtype=np.float32),
        np.asarray(gripper_gap_bins, dtype=np.float32),
        np.asarray(horizon_bins, dtype=np.float32),
        float(d_reach),
        float(z_lift_min),
        float(grip_close_thresh),
        int(n_local_min),
        int(m_q_min),
        int(m_b),
        int(ee_start),
        int(object_start),
        int(gripper_start),
        np.asarray(action_low, dtype=np.float32),
        np.asarray(action_high, dtype=np.float32),
        int(bins_per_dim),
        bool(require_action_membership),
    )
    return tuple(int(v) for v in result)

def numba_support_scan_nopython_signatures() -> tuple[object, ...]:
    if not NUMBA_AVAILABLE or _scan_fetch_pick_and_place_support_exact_jit is None:
        return ()
    return tuple(_scan_fetch_pick_and_place_support_exact_jit.nopython_signatures)



def numba_support_predicate_nopython_signatures() -> tuple[object, ...]:
    if not NUMBA_AVAILABLE or _scan_fetch_pick_and_place_support_predicate_jit is None:
        return ()
    return tuple(_scan_fetch_pick_and_place_support_predicate_jit.nopython_signatures)
