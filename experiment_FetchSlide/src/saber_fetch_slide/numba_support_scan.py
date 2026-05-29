from __future__ import annotations

import math
import types as py_types
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    try:
        import coverage

        if not hasattr(coverage, "types"):
            coverage.types = py_types.SimpleNamespace(  # type: ignore[attr-defined]
                Tracer=object,
                TTraceData=Any,
                TShouldTraceFn=Any,
                TFileDisposition=Any,
                TShouldStartContextFn=Any,
                TWarnFn=Any,
                TTraceFn=Any,
            )
    except Exception:
        pass
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on machines without numba.
    njit = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class FetchSlideSupportScanConstants:
    ee_obj_bins: np.ndarray
    obj_goal_bins: np.ndarray
    obj_speed_bins: np.ndarray
    vel_align_bins: np.ndarray
    horizon_bins: np.ndarray
    ee_start: int
    object_start: int
    object_velocity_start: int
    action_low: np.ndarray
    action_high: np.ndarray
    d_contact: np.float32
    d_goal_near: np.float32
    v_slide_min: np.float32


def can_scan_fetch_slide_support_exact(task: Any) -> bool:
    return bool(
        NUMBA_AVAILABLE
        and getattr(task, "metadata", lambda: {})().get("task") == "FetchSlide"
        and hasattr(task, "ee_slice")
        and hasattr(task, "object_slice")
        and hasattr(task, "object_velocity_slice")
    )


def make_fetch_slide_support_scan_constants(task: Any) -> FetchSlideSupportScanConstants:
    return FetchSlideSupportScanConstants(
        ee_obj_bins=np.asarray(task.ee_obj_bins, dtype=np.float32),
        obj_goal_bins=np.asarray(task.obj_goal_bins, dtype=np.float32),
        obj_speed_bins=np.asarray(task.obj_speed_bins, dtype=np.float32),
        vel_align_bins=np.asarray(task.vel_align_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        ee_start=int(task.ee_slice[0]),
        object_start=int(task.object_slice[0]),
        object_velocity_start=int(task.object_velocity_slice[0]),
        action_low=np.asarray(task.action_low, dtype=np.float32),
        action_high=np.asarray(task.action_high, dtype=np.float32),
        d_contact=np.float32(task.d_contact),
        d_goal_near=np.float32(task.d_goal_near),
        v_slide_min=np.float32(task.v_slide_min),
    )


def scan_fetch_slide_support_exact(
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    task: Any,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    action_bins_per_dim: int,
) -> tuple[int, int, int, int, int]:
    """Exact FetchSlide support lookup without materializing local arrays.

    The wrapper extracts FetchSlide constants from the task object. The JIT
    implementation receives only NumPy arrays and primitive scalar values.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    constants = make_fetch_slide_support_scan_constants(task)
    return scan_fetch_slide_support_exact_with_constants(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=query_full_key,
        query_policy_bin=query_policy_bin,
        constants=constants,
        n_local_min=n_local_min,
        m_q_min=m_q_min,
        m_b=m_b,
        action_bins_per_dim=action_bins_per_dim,
    )


def scan_fetch_slide_support_exact_with_constants(
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    constants: FetchSlideSupportScanConstants,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    action_bins_per_dim: int,
) -> tuple[int, int, int, int, int]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    return _scan_fetch_slide_support_exact_jit(
        replay_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        np.asarray(scanned_indices, dtype=np.int64),
        np.asarray(query_goal, dtype=np.float32),
        int(query_full_key),
        int(query_policy_bin),
        constants.ee_obj_bins,
        constants.obj_goal_bins,
        constants.obj_speed_bins,
        constants.vel_align_bins,
        constants.horizon_bins,
        constants.ee_start,
        constants.object_start,
        constants.object_velocity_start,
        constants.action_low,
        constants.action_high,
        constants.d_contact,
        constants.d_goal_near,
        constants.v_slide_min,
        int(n_local_min),
        int(m_q_min),
        int(m_b),
        int(action_bins_per_dim),
    )


def scan_fetch_slide_support_predicate_with_constants(
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    constants: FetchSlideSupportScanConstants,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    action_bins_per_dim: int,
) -> tuple[int, int, int, int, int, int, int]:
    """Predicate-only support lookup.

    Returns approximate local-count diagnostics when an early exit occurs, but
    returns the exact support predicate for the fixed scanned_indices subset.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    return _scan_fetch_slide_support_predicate_jit(
        replay_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        np.asarray(scanned_indices, dtype=np.int64),
        np.asarray(query_goal, dtype=np.float32),
        int(query_full_key),
        int(query_policy_bin),
        constants.ee_obj_bins,
        constants.obj_goal_bins,
        constants.obj_speed_bins,
        constants.vel_align_bins,
        constants.horizon_bins,
        constants.ee_start,
        constants.object_start,
        constants.object_velocity_start,
        constants.action_low,
        constants.action_high,
        constants.d_contact,
        constants.d_goal_near,
        constants.v_slide_min,
        int(n_local_min),
        int(m_q_min),
        int(m_b),
        int(action_bins_per_dim),
    )


def scan_fetch_slide_support_quality_with_constants(
    replay_obs: np.ndarray,
    replay_next_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    constants: FetchSlideSupportScanConstants,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    action_bins_per_dim: int,
    quality_floor: float,
    approach_scale: float,
    goal_progress_scale: float,
    align_min: float,
    strike_speed_scale_mult: float,
    near_margin_mult: float,
    near_speed_mult: float,
) -> tuple[int, int, int, int, int, np.float32, np.float32, int, np.float32, np.float32, np.float32, np.float32]:
    """Exact FetchSlide support lookup with phase-conditioned quality.

    This scans every fixed candidate index because support_quality is a max over
    all same-bin full-context matches.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    return _scan_fetch_slide_support_quality_jit(
        replay_obs,
        replay_next_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        np.asarray(scanned_indices, dtype=np.int64),
        np.asarray(query_goal, dtype=np.float32),
        int(query_full_key),
        int(query_policy_bin),
        constants.ee_obj_bins,
        constants.obj_goal_bins,
        constants.obj_speed_bins,
        constants.vel_align_bins,
        constants.horizon_bins,
        constants.ee_start,
        constants.object_start,
        constants.object_velocity_start,
        constants.action_low,
        constants.action_high,
        constants.d_contact,
        constants.d_goal_near,
        constants.v_slide_min,
        int(n_local_min),
        int(m_q_min),
        int(m_b),
        int(action_bins_per_dim),
        np.float32(quality_floor),
        np.float32(approach_scale),
        np.float32(goal_progress_scale),
        np.float32(align_min),
        np.float32(strike_speed_scale_mult),
        np.float32(near_margin_mult),
        np.float32(near_speed_mult),
    )


def warmup_fetch_slide_support_exact(task: Any, action_dim: int, obs_dim: int) -> bool:
    if not can_scan_fetch_slide_support_exact(task):
        return False
    replay_obs = np.zeros((1, max(int(obs_dim), int(task.ee_velocity_slice[1]))), dtype=np.float32)
    replay_h = np.ones(1, dtype=np.int16)
    replay_action_bin = np.zeros(1, dtype=np.int64)
    replay_action_exec = np.zeros((1, int(action_dim)), dtype=np.float32)
    scanned_indices = np.zeros(1, dtype=np.int64)
    query_goal = np.zeros(3, dtype=np.float32)
    scan_fetch_slide_support_exact(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=0,
        query_policy_bin=0,
        task=task,
        n_local_min=1,
        m_q_min=1,
        m_b=1,
        action_bins_per_dim=5,
    )
    return bool(getattr(_scan_fetch_slide_support_exact_jit, "nopython_signatures", ()))


def warmup_fetch_slide_support_predicate(task: Any, action_dim: int, obs_dim: int) -> bool:
    if not can_scan_fetch_slide_support_exact(task):
        return False
    constants = make_fetch_slide_support_scan_constants(task)
    replay_obs = np.zeros((2, max(int(obs_dim), int(task.ee_velocity_slice[1]))), dtype=np.float32)
    replay_h = np.ones(2, dtype=np.int16)
    replay_action_bin = np.zeros(2, dtype=np.int64)
    replay_action_exec = np.zeros((2, int(action_dim)), dtype=np.float32)
    scanned_indices = np.arange(2, dtype=np.int64)
    query_goal = np.zeros(3, dtype=np.float32)
    scan_fetch_slide_support_predicate_with_constants(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=0,
        query_policy_bin=0,
        constants=constants,
        n_local_min=1,
        m_q_min=1,
        m_b=1,
        action_bins_per_dim=5,
    )
    return bool(getattr(_scan_fetch_slide_support_predicate_jit, "nopython_signatures", ()))


def warmup_fetch_slide_support_quality(task: Any, action_dim: int, obs_dim: int) -> bool:
    if not can_scan_fetch_slide_support_exact(task):
        return False
    constants = make_fetch_slide_support_scan_constants(task)
    replay_obs = np.zeros((2, max(int(obs_dim), int(task.ee_velocity_slice[1]))), dtype=np.float32)
    replay_next_obs = replay_obs.copy()
    replay_h = np.ones(2, dtype=np.int16)
    replay_action_bin = np.zeros(2, dtype=np.int64)
    replay_action_exec = np.zeros((2, int(action_dim)), dtype=np.float32)
    scanned_indices = np.arange(2, dtype=np.int64)
    query_goal = np.zeros(3, dtype=np.float32)
    scan_fetch_slide_support_quality_with_constants(
        replay_obs=replay_obs,
        replay_next_obs=replay_next_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=0,
        query_policy_bin=0,
        constants=constants,
        n_local_min=1,
        m_q_min=1,
        m_b=1,
        action_bins_per_dim=5,
        quality_floor=0.25,
        approach_scale=0.02,
        goal_progress_scale=0.03,
        align_min=0.0,
        strike_speed_scale_mult=2.0,
        near_margin_mult=0.25,
        near_speed_mult=1.0,
    )
    return bool(getattr(_scan_fetch_slide_support_quality_jit, "nopython_signatures", ()))


def fetch_slide_support_exact_nopython_signatures() -> tuple[Any, ...]:
    if not NUMBA_AVAILABLE:
        return ()
    return tuple(getattr(_scan_fetch_slide_support_exact_jit, "nopython_signatures", ()))


def fetch_slide_support_predicate_nopython_signatures() -> tuple[Any, ...]:
    if not NUMBA_AVAILABLE:
        return ()
    return tuple(getattr(_scan_fetch_slide_support_predicate_jit, "nopython_signatures", ()))


def fetch_slide_support_quality_nopython_signatures() -> tuple[Any, ...]:
    if not NUMBA_AVAILABLE:
        return ()
    return tuple(getattr(_scan_fetch_slide_support_quality_jit, "nopython_signatures", ()))


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _bin_value(value: np.float32, boundaries: np.ndarray) -> int:
        out = 0
        for i in range(boundaries.shape[0]):
            if value > boundaries[i]:
                out += 1
        return out

    @njit(cache=True)
    def _norm2_float32(x0: np.float32, x1: np.float32) -> np.float32:
        total = np.float32(x0 * x0 + x1 * x1)
        return np.float32(math.sqrt(total))

    @njit(cache=True)
    def _norm3_float32(x0: np.float32, x1: np.float32, x2: np.float32) -> np.float32:
        total = np.float32(x0 * x0 + x1 * x1 + x2 * x2)
        return np.float32(math.sqrt(total))

    @njit(cache=True)
    def _clip01_float32(value: np.float32) -> np.float32:
        if value < np.float32(0.0):
            return np.float32(0.0)
        if value > np.float32(1.0):
            return np.float32(1.0)
        return np.float32(value)

    @njit(cache=True)
    def _fetch_slide_phase_one(
        obs: np.ndarray,
        idx: int,
        query_goal: np.ndarray,
        ee_start: int,
        object_start: int,
        object_velocity_start: int,
        d_contact: np.float32,
        d_goal_near: np.float32,
        v_slide_min: np.float32,
    ) -> int:
        ee_x = obs[idx, ee_start]
        ee_y = obs[idx, ee_start + 1]
        ee_z = obs[idx, ee_start + 2]
        obj_x = obs[idx, object_start]
        obj_y = obs[idx, object_start + 1]
        obj_z = obs[idx, object_start + 2]
        vel_x = obs[idx, object_velocity_start]
        vel_y = obs[idx, object_velocity_start + 1]
        d_ee_obj = _norm3_float32(
            np.float32(ee_x - obj_x),
            np.float32(ee_y - obj_y),
            np.float32(ee_z - obj_z),
        )
        d_obj_goal = _norm3_float32(
            np.float32(obj_x - query_goal[0]),
            np.float32(obj_y - query_goal[1]),
            np.float32(obj_z - query_goal[2]),
        )
        obj_speed_xy = _norm2_float32(np.float32(vel_x), np.float32(vel_y))
        phase = 0
        if d_ee_obj <= d_contact:
            phase = 1
        if obj_speed_xy >= v_slide_min:
            phase = 2
        if d_obj_goal <= d_goal_near:
            phase = 3
        return phase

    @njit(cache=True)
    def _fetch_slide_goal_alignment_xy(
        vel_x: np.float32,
        vel_y: np.float32,
        obj_x: np.float32,
        obj_y: np.float32,
        query_goal: np.ndarray,
    ) -> np.float32:
        speed = _norm2_float32(vel_x, vel_y)
        if speed < np.float32(1e-8):
            return np.float32(0.0)
        goal_dx = np.float32(query_goal[0] - obj_x)
        goal_dy = np.float32(query_goal[1] - obj_y)
        goal_norm = _norm2_float32(goal_dx, goal_dy)
        if goal_norm < np.float32(1e-8):
            return np.float32(0.0)
        align = np.float32((vel_x * goal_dx + vel_y * goal_dy) / (speed * goal_norm))
        if align < np.float32(-1.0):
            return np.float32(-1.0)
        if align > np.float32(1.0):
            return np.float32(1.0)
        return align

    @njit(cache=True)
    def _fetch_slide_candidate_quality_jit(
        replay_obs: np.ndarray,
        replay_next_obs: np.ndarray,
        idx: int,
        query_goal: np.ndarray,
        ee_start: int,
        object_start: int,
        object_velocity_start: int,
        d_contact: np.float32,
        d_goal_near: np.float32,
        v_slide_min: np.float32,
        quality_approach_scale: np.float32,
        quality_goal_progress_scale: np.float32,
        quality_align_min: np.float32,
        quality_strike_speed_scale_mult: np.float32,
        quality_near_margin_mult: np.float32,
        quality_near_speed_mult: np.float32,
    ) -> tuple[np.float32, np.float32, np.float32, np.float32, np.float32]:
        phase = _fetch_slide_phase_one(
            replay_obs,
            idx,
            query_goal,
            ee_start,
            object_start,
            object_velocity_start,
            d_contact,
            d_goal_near,
            v_slide_min,
        )

        ee_x = replay_obs[idx, ee_start]
        ee_y = replay_obs[idx, ee_start + 1]
        ee_z = replay_obs[idx, ee_start + 2]
        obj_x = replay_obs[idx, object_start]
        obj_y = replay_obs[idx, object_start + 1]
        obj_z = replay_obs[idx, object_start + 2]
        next_ee_x = replay_next_obs[idx, ee_start]
        next_ee_y = replay_next_obs[idx, ee_start + 1]
        next_ee_z = replay_next_obs[idx, ee_start + 2]
        next_obj_x = replay_next_obs[idx, object_start]
        next_obj_y = replay_next_obs[idx, object_start + 1]
        next_obj_z = replay_next_obs[idx, object_start + 2]
        next_vel_x = replay_next_obs[idx, object_velocity_start]
        next_vel_y = replay_next_obs[idx, object_velocity_start + 1]

        d_obj_goal_before = _norm3_float32(
            np.float32(obj_x - query_goal[0]),
            np.float32(obj_y - query_goal[1]),
            np.float32(obj_z - query_goal[2]),
        )
        d_obj_goal_after = _norm3_float32(
            np.float32(next_obj_x - query_goal[0]),
            np.float32(next_obj_y - query_goal[1]),
            np.float32(next_obj_z - query_goal[2]),
        )
        speed_next = _norm2_float32(np.float32(next_vel_x), np.float32(next_vel_y))
        align_next = _fetch_slide_goal_alignment_xy(
            np.float32(next_vel_x),
            np.float32(next_vel_y),
            np.float32(obj_x),
            np.float32(obj_y),
            query_goal,
        )
        align_denom = np.float32(1.0 - quality_align_min)
        if align_denom < np.float32(1e-6):
            align_denom = np.float32(1e-6)
        q_align = _clip01_float32(np.float32((align_next - quality_align_min) / align_denom))
        speed_denom = np.float32(quality_strike_speed_scale_mult * v_slide_min)
        if speed_denom < np.float32(1e-6):
            speed_denom = np.float32(1e-6)
        q_speed = _clip01_float32(np.float32(speed_next / speed_denom))

        if phase == 0:
            d_ee_obj_before = _norm3_float32(
                np.float32(ee_x - obj_x),
                np.float32(ee_y - obj_y),
                np.float32(ee_z - obj_z),
            )
            d_ee_obj_after = _norm3_float32(
                np.float32(next_ee_x - next_obj_x),
                np.float32(next_ee_y - next_obj_y),
                np.float32(next_ee_z - next_obj_z),
            )
            denom = quality_approach_scale
            if denom < np.float32(1e-6):
                denom = np.float32(1e-6)
            q_progress = _clip01_float32(np.float32((d_ee_obj_before - d_ee_obj_after) / denom))
            return q_progress, q_progress, np.float32(0.0), np.float32(0.0), np.float32(0.0)

        if phase == 1:
            quality = np.float32(q_speed * q_align)
            return quality, np.float32(0.0), q_speed, q_align, np.float32(0.0)

        if phase == 2:
            denom = quality_goal_progress_scale
            if denom < np.float32(1e-6):
                denom = np.float32(1e-6)
            q_progress = _clip01_float32(np.float32((d_obj_goal_before - d_obj_goal_after) / denom))
            quality = q_progress
            if q_align > quality:
                quality = q_align
            return quality, q_progress, q_speed, q_align, np.float32(0.0)

        distance_delta = np.float32(d_obj_goal_after - d_obj_goal_before)
        near_margin = np.float32(quality_near_margin_mult * d_goal_near)
        if near_margin < np.float32(1e-6):
            near_margin = np.float32(1e-6)
        near_speed_limit = np.float32(quality_near_speed_mult * v_slide_min)
        if near_speed_limit < np.float32(1e-6):
            near_speed_limit = np.float32(1e-6)
        overshoot = distance_delta
        if overshoot < np.float32(0.0):
            overshoot = np.float32(0.0)
        q_no_overshoot = _clip01_float32(np.float32((near_margin - overshoot) / near_margin))
        q_slow = _clip01_float32(np.float32((near_speed_limit - speed_next) / near_speed_limit))
        q_near = np.float32(q_no_overshoot * q_slow)
        return q_near, np.float32(0.0), q_slow, np.float32(0.0), q_near

    @njit(cache=True)
    def _fetch_slide_full_context_key_one(
        replay_obs: np.ndarray,
        idx: int,
        query_goal: np.ndarray,
        h_value: int,
        ee_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        obj_speed_bins: np.ndarray,
        vel_align_bins: np.ndarray,
        horizon_bins: np.ndarray,
        ee_start: int,
        object_start: int,
        object_velocity_start: int,
        d_contact: np.float32,
        d_goal_near: np.float32,
        v_slide_min: np.float32,
    ) -> int:
        ee_x = replay_obs[idx, ee_start]
        ee_y = replay_obs[idx, ee_start + 1]
        ee_z = replay_obs[idx, ee_start + 2]
        obj_x = replay_obs[idx, object_start]
        obj_y = replay_obs[idx, object_start + 1]
        obj_z = replay_obs[idx, object_start + 2]
        vel_x = replay_obs[idx, object_velocity_start]
        vel_y = replay_obs[idx, object_velocity_start + 1]

        d_ee_obj = _norm3_float32(
            np.float32(ee_x - obj_x),
            np.float32(ee_y - obj_y),
            np.float32(ee_z - obj_z),
        )
        d_obj_goal = _norm3_float32(
            np.float32(obj_x - query_goal[0]),
            np.float32(obj_y - query_goal[1]),
            np.float32(obj_z - query_goal[2]),
        )
        obj_speed_xy = _norm2_float32(np.float32(vel_x), np.float32(vel_y))

        goal_dx = np.float32(query_goal[0] - obj_x)
        goal_dy = np.float32(query_goal[1] - obj_y)
        goal_norm = _norm2_float32(goal_dx, goal_dy)
        if goal_norm < np.float32(1e-8):
            goal_norm = np.float32(1e-8)
        denom_speed = obj_speed_xy
        if denom_speed < np.float32(1e-8):
            denom_speed = np.float32(1e-8)
        goal_dir_x = np.float32(goal_dx / goal_norm)
        goal_dir_y = np.float32(goal_dy / goal_norm)
        vel_align = np.float32(np.float32(vel_x * goal_dir_x + vel_y * goal_dir_y) / denom_speed)
        if vel_align < np.float32(-1.0):
            vel_align = np.float32(-1.0)
        elif vel_align > np.float32(1.0):
            vel_align = np.float32(1.0)

        phase = 0
        if d_ee_obj <= d_contact:
            phase = 1
        if obj_speed_xy >= v_slide_min:
            phase = 2
        if d_obj_goal <= d_goal_near:
            phase = 3

        ee_bin = _bin_value(d_ee_obj, ee_obj_bins)
        goal_bin = _bin_value(d_obj_goal, obj_goal_bins)
        speed_bin = _bin_value(obj_speed_xy, obj_speed_bins)
        align_bin = _bin_value(vel_align, vel_align_bins)
        h_bin = _bin_value(np.float32(h_value), horizon_bins)
        return phase * 1_000_000 + ee_bin * 100_000 + goal_bin * 10_000 + speed_bin * 1_000 + align_bin * 100 + h_bin

    @njit(cache=True)
    def _action_bin_one(
        replay_action_exec: np.ndarray,
        idx: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_bins_per_dim: int,
    ) -> int:
        out = 0
        multiplier = 1
        for dim in range(replay_action_exec.shape[1]):
            center = np.float32(np.float32(action_high[dim] + action_low[dim]) * np.float32(0.5))
            scale = np.float32(np.float32(action_high[dim] - action_low[dim]) * np.float32(0.5))
            if scale < np.float32(1e-6):
                scale = np.float32(1e-6)
            normed = np.float32(np.float32(replay_action_exec[idx, dim] - center) / scale)
            scaled = np.float32(np.float32(normed + np.float32(1.0)) * np.float32(0.5 * action_bins_per_dim))
            if scaled < np.float32(0.0):
                scaled = np.float32(0.0)
            max_bin = np.float32(action_bins_per_dim - 1)
            if scaled > max_bin:
                scaled = max_bin
            out += int(math.floor(scaled)) * multiplier
            multiplier *= action_bins_per_dim
        return out

    @njit(cache=True)
    def _scan_fetch_slide_support_exact_jit(
        replay_obs: np.ndarray,
        replay_h: np.ndarray,
        replay_action_bin: np.ndarray,
        replay_action_exec: np.ndarray,
        scanned_indices: np.ndarray,
        query_goal: np.ndarray,
        query_full_key: int,
        query_policy_bin: int,
        ee_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        obj_speed_bins: np.ndarray,
        vel_align_bins: np.ndarray,
        horizon_bins: np.ndarray,
        ee_start: int,
        object_start: int,
        object_velocity_start: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        d_contact: np.float32,
        d_goal_near: np.float32,
        v_slide_min: np.float32,
        n_local_min: int,
        m_q_min: int,
        m_b: int,
        action_bins_per_dim: int,
    ) -> tuple[int, int, int, int, int]:
        local_pool_size = 0
        psi_a_policy = 0
        for pos in range(scanned_indices.shape[0]):
            idx = int(scanned_indices[pos])
            candidate_key = _fetch_slide_full_context_key_one(
                replay_obs,
                idx,
                query_goal,
                int(replay_h[idx]),
                ee_obj_bins,
                obj_goal_bins,
                obj_speed_bins,
                vel_align_bins,
                horizon_bins,
                ee_start,
                object_start,
                object_velocity_start,
                d_contact,
                d_goal_near,
                v_slide_min,
            )
            if candidate_key == query_full_key:
                local_pool_size += 1
                candidate_bin = int(replay_action_bin[idx])
                if candidate_bin < 0:
                    candidate_bin = _action_bin_one(
                        replay_action_exec,
                        idx,
                        action_low,
                        action_high,
                        action_bins_per_dim,
                    )
                if candidate_bin == query_policy_bin:
                    psi_a_policy = 1

        available_support_size = local_pool_size
        if available_support_size > m_b:
            available_support_size = m_b
        support_positive = 0
        if local_pool_size >= n_local_min and available_support_size >= m_q_min and psi_a_policy == 1:
            support_positive = 1
        fallback = 1
        if support_positive == 1:
            fallback = 0
        return local_pool_size, available_support_size, psi_a_policy, support_positive, fallback

    @njit(cache=True)
    def _scan_fetch_slide_support_predicate_jit(
        replay_obs: np.ndarray,
        replay_h: np.ndarray,
        replay_action_bin: np.ndarray,
        replay_action_exec: np.ndarray,
        scanned_indices: np.ndarray,
        query_goal: np.ndarray,
        query_full_key: int,
        query_policy_bin: int,
        ee_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        obj_speed_bins: np.ndarray,
        vel_align_bins: np.ndarray,
        horizon_bins: np.ndarray,
        ee_start: int,
        object_start: int,
        object_velocity_start: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        d_contact: np.float32,
        d_goal_near: np.float32,
        v_slide_min: np.float32,
        n_local_min: int,
        m_q_min: int,
        m_b: int,
        action_bins_per_dim: int,
    ) -> tuple[int, int, int, int, int, int, int]:
        local_pool_size = 0
        psi_a_policy = 0
        support_positive = 0
        inspected_count = 0
        early_exit_code = 0
        for pos in range(scanned_indices.shape[0]):
            inspected_count = pos + 1
            idx = int(scanned_indices[pos])
            candidate_key = _fetch_slide_full_context_key_one(
                replay_obs,
                idx,
                query_goal,
                int(replay_h[idx]),
                ee_obj_bins,
                obj_goal_bins,
                obj_speed_bins,
                vel_align_bins,
                horizon_bins,
                ee_start,
                object_start,
                object_velocity_start,
                d_contact,
                d_goal_near,
                v_slide_min,
            )
            if candidate_key == query_full_key:
                local_pool_size += 1
                candidate_bin = int(replay_action_bin[idx])
                if candidate_bin < 0:
                    candidate_bin = _action_bin_one(
                        replay_action_exec,
                        idx,
                        action_low,
                        action_high,
                        action_bins_per_dim,
                    )
                if candidate_bin == query_policy_bin:
                    psi_a_policy = 1

            available_support_size = local_pool_size
            if available_support_size > m_b:
                available_support_size = m_b
            if (
                local_pool_size >= n_local_min
                and available_support_size >= m_q_min
                and psi_a_policy == 1
            ):
                support_positive = 1
                early_exit_code = 1
                return local_pool_size, available_support_size, psi_a_policy, support_positive, 0, inspected_count, early_exit_code

        available_support_size = local_pool_size
        if available_support_size > m_b:
            available_support_size = m_b
        if local_pool_size >= n_local_min and available_support_size >= m_q_min and psi_a_policy == 1:
            support_positive = 1
        fallback = 1
        if support_positive == 1:
            fallback = 0
        return local_pool_size, available_support_size, psi_a_policy, support_positive, fallback, inspected_count, early_exit_code

    @njit(cache=True)
    def _scan_fetch_slide_support_quality_jit(
        replay_obs: np.ndarray,
        replay_next_obs: np.ndarray,
        replay_h: np.ndarray,
        replay_action_bin: np.ndarray,
        replay_action_exec: np.ndarray,
        scanned_indices: np.ndarray,
        query_goal: np.ndarray,
        query_full_key: int,
        query_policy_bin: int,
        ee_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        obj_speed_bins: np.ndarray,
        vel_align_bins: np.ndarray,
        horizon_bins: np.ndarray,
        ee_start: int,
        object_start: int,
        object_velocity_start: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        d_contact: np.float32,
        d_goal_near: np.float32,
        v_slide_min: np.float32,
        n_local_min: int,
        m_q_min: int,
        m_b: int,
        action_bins_per_dim: int,
        quality_floor: np.float32,
        quality_approach_scale: np.float32,
        quality_goal_progress_scale: np.float32,
        quality_align_min: np.float32,
        quality_strike_speed_scale_mult: np.float32,
        quality_near_margin_mult: np.float32,
        quality_near_speed_mult: np.float32,
    ) -> tuple[int, int, int, int, int, np.float32, np.float32, int, np.float32, np.float32, np.float32, np.float32]:
        local_pool_size = 0
        psi_a_policy = 0
        same_bin_match_count = 0
        best_quality = np.float32(0.0)
        best_progress = np.float32(0.0)
        best_speed = np.float32(0.0)
        best_alignment = np.float32(0.0)
        best_near_stability = np.float32(0.0)

        for pos in range(scanned_indices.shape[0]):
            idx = int(scanned_indices[pos])
            candidate_key = _fetch_slide_full_context_key_one(
                replay_obs,
                idx,
                query_goal,
                int(replay_h[idx]),
                ee_obj_bins,
                obj_goal_bins,
                obj_speed_bins,
                vel_align_bins,
                horizon_bins,
                ee_start,
                object_start,
                object_velocity_start,
                d_contact,
                d_goal_near,
                v_slide_min,
            )
            if candidate_key == query_full_key:
                local_pool_size += 1
                candidate_bin = int(replay_action_bin[idx])
                if candidate_bin < 0:
                    candidate_bin = _action_bin_one(
                        replay_action_exec,
                        idx,
                        action_low,
                        action_high,
                        action_bins_per_dim,
                    )
                if candidate_bin == query_policy_bin:
                    psi_a_policy = 1
                    same_bin_match_count += 1
                    quality, q_progress, q_speed, q_alignment, q_near = _fetch_slide_candidate_quality_jit(
                        replay_obs,
                        replay_next_obs,
                        idx,
                        query_goal,
                        ee_start,
                        object_start,
                        object_velocity_start,
                        d_contact,
                        d_goal_near,
                        v_slide_min,
                        quality_approach_scale,
                        quality_goal_progress_scale,
                        quality_align_min,
                        quality_strike_speed_scale_mult,
                        quality_near_margin_mult,
                        quality_near_speed_mult,
                    )
                    if quality > best_quality:
                        best_quality = quality
                        best_progress = q_progress
                        best_speed = q_speed
                        best_alignment = q_alignment
                        best_near_stability = q_near

        available_support_size = local_pool_size
        if available_support_size > m_b:
            available_support_size = m_b
        support_positive = 0
        if local_pool_size >= n_local_min and available_support_size >= m_q_min and psi_a_policy == 1:
            support_positive = 1
        fallback = 1
        support_score = np.float32(0.0)
        if support_positive == 1:
            fallback = 0
            floor = _clip01_float32(quality_floor)
            support_score = np.float32(floor + np.float32(1.0 - floor) * best_quality)
        return (
            local_pool_size,
            available_support_size,
            psi_a_policy,
            support_positive,
            fallback,
            best_quality,
            support_score,
            same_bin_match_count,
            best_progress,
            best_speed,
            best_alignment,
            best_near_stability,
        )

else:

    def _scan_fetch_slide_support_exact_jit(*args: Any, **kwargs: Any) -> tuple[int, int, int, int, int]:
        raise RuntimeError("numba is not available")

    def _scan_fetch_slide_support_predicate_jit(*args: Any, **kwargs: Any) -> tuple[int, int, int, int, int, int, int]:
        raise RuntimeError("numba is not available")

    def _scan_fetch_slide_support_quality_jit(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[int, int, int, int, int, np.float32, np.float32, int, np.float32, np.float32, np.float32, np.float32]:
        raise RuntimeError("numba is not available")
