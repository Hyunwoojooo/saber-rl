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
        policy_seen = 0
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
        policy_seen = 0
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


# Quality/progress-gated FPP support scanner. This is intentionally separate from
# the base exact/predicate scanners so support_quality_mode=none preserves the
# existing v4 path.
if NUMBA_AVAILABLE:

    @_numba.njit
    def _clip01(value: float) -> np.float32:
        if value <= 0.0:
            return np.float32(0.0)
        if value >= 1.0:
            return np.float32(1.0)
        return np.float32(value)

    @_numba.njit
    def _scan_fpp_support_quality_with_constants_jit(
        replay_obs: np.ndarray,
        replay_next_obs: np.ndarray,
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
        d_goal_near: float,
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
        quality_mode_id: int,
        quality_floor: float,
        hand_progress_scale: float,
        goal_progress_scale: float,
        z_progress_scale: float,
        grip_progress_scale: float,
        near_margin_mult: float,
    ):
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
        policy_seen = 0
        same_bin_match_count = 0
        n_scanned = int(scanned_indices.shape[0])

        max_quality = np.float32(0.0)
        max_q_hand = np.float32(0.0)
        max_q_grip = np.float32(0.0)
        max_q_lift = np.float32(0.0)
        max_q_place = np.float32(0.0)
        max_q_near_stable = np.float32(0.0)
        near_margin = float(near_margin_mult) * float(d_goal_near)
        if near_margin <= 1e-8:
            near_margin = 1e-8

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
            action_bin = int(replay_action_bin[idx])
            if action_bin < 0:
                action_bin = _action_bin_one(
                    replay_action_exec[idx],
                    action_low,
                    action_high,
                    bins_per_dim,
                )
            if action_bin != query_policy_bin:
                continue

            policy_seen = 1
            same_bin_match_count += 1

            next_ee_x = replay_next_obs[idx, ee_start]
            next_ee_y = replay_next_obs[idx, ee_start + 1]
            next_ee_z = replay_next_obs[idx, ee_start + 2]
            next_obj_x = replay_next_obs[idx, object_start]
            next_obj_y = replay_next_obs[idx, object_start + 1]
            next_obj_z = replay_next_obs[idx, object_start + 2]
            next_gripper_gap = replay_next_obs[idx, gripper_start] + replay_next_obs[idx, gripper_start + 1]

            ndx_hand = next_ee_x - next_obj_x
            ndy_hand = next_ee_y - next_obj_y
            ndz_hand = next_ee_z - next_obj_z
            d_hand_obj_after = _sqrt32(ndx_hand * ndx_hand + ndy_hand * ndy_hand + ndz_hand * ndz_hand)

            ndx_goal = next_obj_x - query_goal[0]
            ndy_goal = next_obj_y - query_goal[1]
            ndz_goal = next_obj_z - query_goal[2]
            d_obj_goal_after = _sqrt32(ndx_goal * ndx_goal + ndy_goal * ndy_goal + ndz_goal * ndz_goal)

            q_hand = _clip01((d_hand_obj - d_hand_obj_after) / hand_progress_scale)
            q_close = _clip01((gripper_gap - next_gripper_gap) / grip_progress_scale)
            q_near_hand = _clip01((d_reach - d_hand_obj_after) / d_reach)
            q_grip = q_hand
            q_close_near = q_close * q_near_hand
            if q_close_near > q_grip:
                q_grip = q_close_near

            q_lift_progress = _clip01((next_obj_z - obj_z) / z_progress_scale)
            q_high = np.float32(1.0) if next_obj_z >= z_lift_min else np.float32(0.0)
            q_lift = q_lift_progress
            q_lift_coupled = np.float32(0.5) * q_high * q_near_hand
            if q_lift_coupled > q_lift:
                q_lift = q_lift_coupled

            goal_progress = d_obj_goal - d_obj_goal_after
            q_place_progress = _clip01(goal_progress / goal_progress_scale)
            worsen = d_obj_goal_after - d_obj_goal
            if worsen < 0.0:
                worsen = 0.0
            q_no_worsen = _clip01((near_margin - worsen) / near_margin)
            q_near_stable = np.float32(0.0)
            q_place = q_place_progress
            if d_obj_goal <= d_goal_near:
                q_near_stable = q_no_worsen
                if q_no_worsen > q_place:
                    q_place = q_no_worsen

            if quality_mode_id == 1:
                q = q_place
                if q_phase == 0:
                    q = q_hand
                elif q_phase == 1:
                    q = q_grip
                elif q_phase == 2:
                    q = q_lift
            else:
                q = q_hand
                if q_grip > q:
                    q = q_grip
                if q_lift > q:
                    q = q_lift
                if q_place > q:
                    q = q_place

            if q > max_quality:
                max_quality = q
            if q_hand > max_q_hand:
                max_q_hand = q_hand
            if q_grip > max_q_grip:
                max_q_grip = q_grip
            if q_lift > max_q_lift:
                max_q_lift = q_lift
            if q_place > max_q_place:
                max_q_place = q_place
            if q_near_stable > max_q_near_stable:
                max_q_near_stable = q_near_stable

        available_support_size = local_count
        if available_support_size > m_b:
            available_support_size = m_b
        support_positive = 0
        if local_count >= n_local_min and available_support_size >= m_q_min and policy_seen == 1:
            support_positive = 1
        fallback = 1 - support_positive
        support_score = np.float32(0.0)
        if support_positive == 1:
            support_score = np.float32(quality_floor) + (np.float32(1.0) - np.float32(quality_floor)) * max_quality
        return (
            local_count,
            available_support_size,
            policy_seen,
            support_positive,
            fallback,
            max_quality,
            support_score,
            same_bin_match_count,
            max_q_hand,
            max_q_grip,
            max_q_lift,
            max_q_place,
            max_q_near_stable,
            n_scanned,
        )

else:
    _scan_fpp_support_quality_with_constants_jit = None


def _bin_scalar_py(value: float, boundaries: np.ndarray) -> int:
    out = 0
    for boundary in boundaries:
        if float(value) > float(boundary):
            out += 1
    return out


def _clip01_py(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _action_bin_one_py(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray, bins_per_dim: int) -> int:
    center = 0.5 * (action_high + action_low)
    scale = np.maximum(1e-6, 0.5 * (action_high - action_low))
    norm = (action - center) / scale
    scaled = (norm + 1.0) * 0.5 * float(bins_per_dim)
    bins = np.floor(np.clip(scaled, 0.0, float(bins_per_dim - 1))).astype(np.int64)
    multipliers = bins_per_dim ** np.arange(action.shape[0], dtype=np.int64)
    return int(np.sum(bins * multipliers))


def scan_fpp_support_quality_reference(
    replay_obs: np.ndarray,
    replay_next_obs: np.ndarray,
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
    d_goal_near: float,
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
    quality_mode_id: int = 1,
    quality_floor: float = 0.5,
    hand_progress_scale: float = 0.03,
    goal_progress_scale: float = 0.03,
    z_progress_scale: float = 0.03,
    grip_progress_scale: float = 0.02,
    near_margin_mult: float = 0.5,
) -> tuple[int, int, int, int, int, float, float, int, float, float, float, float, float, int]:
    if action_low is None or action_high is None:
        raise ValueError("action_low and action_high are required for quality action-bin handling")
    replay_obs = np.asarray(replay_obs, dtype=np.float32)
    replay_next_obs = np.asarray(replay_next_obs, dtype=np.float32)
    replay_h = np.asarray(replay_h)
    replay_action_bin = np.asarray(replay_action_bin, dtype=np.int64)
    replay_action_exec = np.asarray(replay_action_exec, dtype=np.float32)
    scanned_indices = np.asarray(scanned_indices, dtype=np.int64)
    query_goal = np.asarray(query_goal, dtype=np.float32)
    hand_obj_bins = np.asarray(hand_obj_bins, dtype=np.float32)
    obj_goal_bins = np.asarray(obj_goal_bins, dtype=np.float32)
    object_z_bins = np.asarray(object_z_bins, dtype=np.float32)
    gripper_gap_bins = np.asarray(gripper_gap_bins, dtype=np.float32)
    horizon_bins = np.asarray(horizon_bins, dtype=np.float32)
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)

    q_phase = int(query_full_key) // 1_000_000
    rem = int(query_full_key) - q_phase * 1_000_000
    q_hand_bin = rem // 100_000
    rem -= q_hand_bin * 100_000
    q_goal_bin = rem // 10_000
    rem -= q_goal_bin * 10_000
    q_z_bin = rem // 1_000
    rem -= q_z_bin * 1_000
    q_grip_bin = rem // 100
    q_h_bin = rem - q_grip_bin * 100

    local_count = 0
    policy_seen = 0
    same_bin_match_count = 0
    max_quality = max_q_hand = max_q_grip = max_q_lift = max_q_place = max_q_near_stable = 0.0
    near_margin = max(1e-8, float(near_margin_mult) * float(d_goal_near))

    for idx_raw in scanned_indices:
        idx = int(idx_raw)
        ee = replay_obs[idx, ee_start:ee_start + 3]
        obj = replay_obs[idx, object_start:object_start + 3]
        gap = float(replay_obs[idx, gripper_start] + replay_obs[idx, gripper_start + 1])
        d_hand_obj = float(np.linalg.norm(ee - obj))
        d_obj_goal = float(np.linalg.norm(obj - query_goal))
        obj_z = float(obj[2])
        phase = 3
        if obj_z <= z_lift_min:
            phase = 2
        if obj_z <= z_lift_min and gap > grip_close_thresh:
            phase = 1
        if d_hand_obj > d_reach:
            phase = 0
        if phase != q_phase:
            continue
        if _bin_scalar_py(d_hand_obj, hand_obj_bins) != q_hand_bin:
            continue
        if _bin_scalar_py(d_obj_goal, obj_goal_bins) != q_goal_bin:
            continue
        if _bin_scalar_py(obj_z, object_z_bins) != q_z_bin:
            continue
        if _bin_scalar_py(gap, gripper_gap_bins) != q_grip_bin:
            continue
        if _bin_scalar_py(float(replay_h[idx]), horizon_bins) != q_h_bin:
            continue
        local_count += 1
        action_bin = int(replay_action_bin[idx])
        if action_bin < 0:
            action_bin = _action_bin_one_py(replay_action_exec[idx], action_low, action_high, bins_per_dim)
        if action_bin != int(query_policy_bin):
            continue
        policy_seen = 1
        same_bin_match_count += 1

        next_ee = replay_next_obs[idx, ee_start:ee_start + 3]
        next_obj = replay_next_obs[idx, object_start:object_start + 3]
        next_gap = float(replay_next_obs[idx, gripper_start] + replay_next_obs[idx, gripper_start + 1])
        d_hand_obj_after = float(np.linalg.norm(next_ee - next_obj))
        d_obj_goal_after = float(np.linalg.norm(next_obj - query_goal))
        q_hand = _clip01_py((d_hand_obj - d_hand_obj_after) / float(hand_progress_scale))
        q_close = _clip01_py((gap - next_gap) / float(grip_progress_scale))
        q_near_hand = _clip01_py((float(d_reach) - d_hand_obj_after) / float(d_reach))
        q_grip = max(q_hand, q_close * q_near_hand)
        q_lift_progress = _clip01_py((float(next_obj[2]) - obj_z) / float(z_progress_scale))
        q_lift = max(q_lift_progress, 0.5 * (1.0 if float(next_obj[2]) >= z_lift_min else 0.0) * q_near_hand)
        q_place_progress = _clip01_py((d_obj_goal - d_obj_goal_after) / float(goal_progress_scale))
        worsen = max(0.0, d_obj_goal_after - d_obj_goal)
        q_no_worsen = _clip01_py((near_margin - worsen) / near_margin)
        q_near_stable = 0.0
        q_place = q_place_progress
        if d_obj_goal <= d_goal_near:
            q_near_stable = q_no_worsen
            q_place = max(q_place, q_no_worsen)
        if int(quality_mode_id) == 1:
            q = (q_hand, q_grip, q_lift, q_place)[min(max(q_phase, 0), 3)]
        else:
            q = max(q_hand, q_grip, q_lift, q_place)
        max_quality = max(max_quality, q)
        max_q_hand = max(max_q_hand, q_hand)
        max_q_grip = max(max_q_grip, q_grip)
        max_q_lift = max(max_q_lift, q_lift)
        max_q_place = max(max_q_place, q_place)
        max_q_near_stable = max(max_q_near_stable, q_near_stable)

    available = min(local_count, int(m_b))
    support_positive = int(local_count >= int(n_local_min) and available >= int(m_q_min) and policy_seen == 1)
    fallback = 1 - support_positive
    support_score = 0.0
    if support_positive:
        support_score = float(quality_floor) + (1.0 - float(quality_floor)) * max_quality
    return (
        int(local_count), int(available), int(policy_seen), int(support_positive), int(fallback),
        float(max_quality), float(support_score), int(same_bin_match_count),
        float(max_q_hand), float(max_q_grip), float(max_q_lift), float(max_q_place),
        float(max_q_near_stable), int(scanned_indices.shape[0]),
    )


def scan_fpp_support_quality_with_constants(
    replay_obs: np.ndarray,
    replay_next_obs: np.ndarray,
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
    d_goal_near: float,
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
    quality_mode_id: int = 1,
    quality_floor: float = 0.5,
    hand_progress_scale: float = 0.03,
    goal_progress_scale: float = 0.03,
    z_progress_scale: float = 0.03,
    grip_progress_scale: float = 0.02,
    near_margin_mult: float = 0.5,
) -> tuple[int, int, int, int, int, float, float, int, float, float, float, float, float, int]:
    if not NUMBA_AVAILABLE or _scan_fpp_support_quality_with_constants_jit is None:
        raise RuntimeError("numba is not available")
    if action_low is None or action_high is None:
        raise ValueError("action_low and action_high are required for quality action-bin handling")
    result = _scan_fpp_support_quality_with_constants_jit(
        np.asarray(replay_obs, dtype=np.float32),
        np.asarray(replay_next_obs, dtype=np.float32),
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
        float(d_goal_near),
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
        int(quality_mode_id),
        float(quality_floor),
        float(hand_progress_scale),
        float(goal_progress_scale),
        float(z_progress_scale),
        float(grip_progress_scale),
        float(near_margin_mult),
    )
    return (
        int(result[0]), int(result[1]), int(result[2]), int(result[3]), int(result[4]),
        float(result[5]), float(result[6]), int(result[7]),
        float(result[8]), float(result[9]), float(result[10]), float(result[11]),
        float(result[12]), int(result[13]),
    )


def numba_support_quality_nopython_signatures() -> tuple[object, ...]:
    if not NUMBA_AVAILABLE or _scan_fpp_support_quality_with_constants_jit is None:
        return ()
    return tuple(_scan_fpp_support_quality_with_constants_jit.nopython_signatures)
