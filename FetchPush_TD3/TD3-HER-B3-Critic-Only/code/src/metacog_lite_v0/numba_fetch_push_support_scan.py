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
class FetchPushSupportScanConstants:
    ee_obj_bins: np.ndarray
    obj_goal_bins: np.ndarray
    horizon_bins: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    d_contact: np.float32
    d_goal_near: np.float32


def can_scan_fetch_push_support_exact(task: Any) -> bool:
    return bool(
        NUMBA_AVAILABLE
        and getattr(task, "metadata", lambda: {})().get("task") == "FetchPush"
        and hasattr(task, "action_low")
        and hasattr(task, "action_high")
        and hasattr(task, "ee_obj_bins")
        and hasattr(task, "obj_goal_bins")
        and hasattr(task, "horizon_bins")
    )


def make_fetch_push_support_scan_constants(task: Any) -> FetchPushSupportScanConstants:
    return FetchPushSupportScanConstants(
        ee_obj_bins=np.asarray(task.ee_obj_bins, dtype=np.float32),
        obj_goal_bins=np.asarray(task.obj_goal_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        action_low=np.asarray(task.action_low, dtype=np.float32),
        action_high=np.asarray(task.action_high, dtype=np.float32),
        d_contact=np.float32(task.d_contact),
        d_goal_near=np.float32(task.d_goal_near),
    )


def scan_fetch_push_support_exact(
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
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    constants = make_fetch_push_support_scan_constants(task)
    return scan_fetch_push_support_exact_with_constants(
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


def scan_fetch_push_support_exact_with_constants(
    replay_obs: np.ndarray,
    replay_h: np.ndarray,
    replay_action_bin: np.ndarray,
    replay_action_exec: np.ndarray,
    scanned_indices: np.ndarray,
    query_goal: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    constants: FetchPushSupportScanConstants,
    n_local_min: int,
    m_q_min: int,
    m_b: int,
    action_bins_per_dim: int,
) -> tuple[int, int, int, int, int]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    return _scan_fetch_push_support_exact_jit(
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
        constants.horizon_bins,
        constants.action_low,
        constants.action_high,
        constants.d_contact,
        constants.d_goal_near,
        int(n_local_min),
        int(m_q_min),
        int(m_b),
        int(action_bins_per_dim),
    )


def warmup_fetch_push_support_exact(task: Any, action_dim: int, obs_dim: int) -> bool:
    if not can_scan_fetch_push_support_exact(task):
        return False
    replay_obs = np.zeros((1, max(int(obs_dim), 6)), dtype=np.float32)
    replay_h = np.ones(1, dtype=np.int16)
    replay_action_bin = np.zeros(1, dtype=np.int64)
    replay_action_exec = np.zeros((1, int(action_dim)), dtype=np.float32)
    scanned_indices = np.zeros(1, dtype=np.int64)
    query_goal = np.zeros(3, dtype=np.float32)
    scan_fetch_push_support_exact(
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
    return bool(getattr(_scan_fetch_push_support_exact_jit, "nopython_signatures", ()))


def fetch_push_support_exact_nopython_signatures() -> tuple[Any, ...]:
    if not NUMBA_AVAILABLE:
        return ()
    return tuple(getattr(_scan_fetch_push_support_exact_jit, "nopython_signatures", ()))


if NUMBA_AVAILABLE:

    @njit(cache=False)
    def _bin_value(value: np.float32, boundaries: np.ndarray) -> int:
        out = 0
        for i in range(boundaries.shape[0]):
            if value > boundaries[i]:
                out += 1
        return out

    @njit(cache=False)
    def _norm3_float32(x0: np.float32, x1: np.float32, x2: np.float32) -> np.float32:
        total = np.float32(x0 * x0 + x1 * x1 + x2 * x2)
        return np.float32(math.sqrt(total))

    @njit(cache=False)
    def _fetch_push_full_context_key_one(
        replay_obs: np.ndarray,
        idx: int,
        query_goal: np.ndarray,
        h_value: int,
        ee_obj_bins: np.ndarray,
        obj_goal_bins: np.ndarray,
        horizon_bins: np.ndarray,
        d_contact: np.float32,
        d_goal_near: np.float32,
    ) -> int:
        ee_x = replay_obs[idx, 0]
        ee_y = replay_obs[idx, 1]
        ee_z = replay_obs[idx, 2]
        obj_x = replay_obs[idx, 3]
        obj_y = replay_obs[idx, 4]
        obj_z = replay_obs[idx, 5]

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

        phase = 0
        if d_ee_obj <= d_contact and d_obj_goal > d_goal_near:
            phase = 1
        if d_ee_obj <= d_contact and d_obj_goal <= d_goal_near:
            phase = 2

        ee_bin = _bin_value(d_ee_obj, ee_obj_bins)
        goal_bin = _bin_value(d_obj_goal, obj_goal_bins)
        h_bin = _bin_value(np.float32(h_value), horizon_bins)
        return phase * 10_000 + ee_bin * 1_000 + goal_bin * 100 + h_bin

    @njit(cache=False)
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

    @njit(cache=False)
    def _scan_fetch_push_support_exact_jit(
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
        horizon_bins: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        d_contact: np.float32,
        d_goal_near: np.float32,
        n_local_min: int,
        m_q_min: int,
        m_b: int,
        action_bins_per_dim: int,
    ) -> tuple[int, int, int, int, int]:
        local_pool_size = 0
        psi_a_policy = 0
        for pos in range(scanned_indices.shape[0]):
            idx = int(scanned_indices[pos])
            candidate_key = _fetch_push_full_context_key_one(
                replay_obs,
                idx,
                query_goal,
                int(replay_h[idx]),
                ee_obj_bins,
                obj_goal_bins,
                horizon_bins,
                d_contact,
                d_goal_near,
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

else:

    def _scan_fetch_push_support_exact_jit(*args: Any, **kwargs: Any) -> tuple[int, int, int, int, int]:
        raise RuntimeError("numba is not available")
