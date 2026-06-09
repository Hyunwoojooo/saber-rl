#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "version_v0_lite" / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "code" / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from metacog_lite_v0.task_fetch_slide import FetchSlideTaskSpec
from saber_fetch_slide.numba_support_scan import (
    NUMBA_AVAILABLE,
    fetch_slide_support_exact_nopython_signatures,
    fetch_slide_support_predicate_nopython_signatures,
    make_fetch_slide_support_scan_constants,
    scan_fetch_slide_support_exact_with_constants,
    scan_fetch_slide_support_predicate_with_constants,
)


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
    policy_bin: int,
    m_b: int,
    n_local_min: int,
    m_q_min: int,
    action_bins_per_dim: int,
) -> tuple[int, int, int, int, int]:
    full_key = int(task.full_context_hash_batch(query_obs[None, :], query_goal[None, :], np.asarray([query_h], dtype=np.int16))[0])
    ref_obs = replay_obs[scanned_indices]
    ref_goal = np.repeat(query_goal[None, :], scanned_indices.size, axis=0)
    ref_h = replay_h[scanned_indices]
    ref_hash = task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
    local_idx = scanned_indices[ref_hash == full_key]
    if local_idx.size > 0:
        local_bins = replay_action_bin[local_idx].copy()
        if local_bins.shape[0] != local_idx.size or np.any(local_bins < 0):
            local_bins = task.action_bin_batch(replay_action_exec[local_idx], action_bins_per_dim)
        action_bins = np.unique(local_bins.astype(np.int64, copy=False))
    else:
        action_bins = np.zeros((0,), dtype=np.int64)
    local_pool_size = int(local_idx.size)
    available_support_size = min(local_pool_size, m_b)
    psi_a_policy = int(policy_bin in set(int(x) for x in action_bins.tolist()))
    psi = int(local_pool_size >= n_local_min and available_support_size >= m_q_min and psi_a_policy == 1)
    fallback = 0 if psi else 1
    return local_pool_size, available_support_size, psi_a_policy, psi, fallback


def main() -> None:
    if not NUMBA_AVAILABLE:
        print("numba_available 0")
        return
    rng = np.random.default_rng(123)
    obs_dim = 25
    action_dim = 4
    scan_size = 2048
    repeat = 200
    action_bins_per_dim = 5
    n_local_min = 4
    m_q_min = 4
    m_b = 16
    task = FetchSlideTaskSpec(
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
        epsilon_g=0.05,
    )
    replay_obs = rng.normal(0.0, 0.25, size=(scan_size, obs_dim)).astype(np.float32)
    replay_obs[:, 3:6] = replay_obs[:, 0:3] + rng.normal(0.12, 0.04, size=(scan_size, 3)).astype(np.float32)
    replay_obs[:, 14:17] = rng.normal(0.0, 0.08, size=(scan_size, 3)).astype(np.float32)
    replay_h = rng.integers(1, 100, size=scan_size, dtype=np.int16)
    replay_action_exec = rng.uniform(-1.0, 1.0, size=(scan_size, action_dim)).astype(np.float32)
    replay_action_bin = task.action_bin_batch(replay_action_exec, action_bins_per_dim)
    replay_action_bin[::17] = -1
    scanned_indices = np.arange(scan_size, dtype=np.int64)
    query_obs = replay_obs[0].copy()
    query_goal = np.asarray([0.55, 0.0, 0.0], dtype=np.float32)
    query_h = int(replay_h[0])
    policy_action = replay_action_exec[0].copy()
    policy_bin = int(task.action_bin_batch(policy_action[None, :], action_bins_per_dim)[0])
    full_key = int(task.full_context_hash_batch(query_obs[None, :], query_goal[None, :], np.asarray([query_h], dtype=np.int16))[0])
    constants = make_fetch_slide_support_scan_constants(task)

    warmed = scan_fetch_slide_support_exact_with_constants(
        replay_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        scanned_indices,
        query_goal,
        full_key,
        policy_bin,
        constants,
        n_local_min,
        m_q_min,
        m_b,
        action_bins_per_dim,
    )
    legacy = legacy_scan(
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
        m_b,
        n_local_min,
        m_q_min,
        action_bins_per_dim,
    )
    assert warmed == legacy, (warmed, legacy)
    predicate_warmed = scan_fetch_slide_support_predicate_with_constants(
        replay_obs,
        replay_h,
        replay_action_bin,
        replay_action_exec,
        scanned_indices,
        query_goal,
        full_key,
        policy_bin,
        constants,
        n_local_min,
        m_q_min,
        m_b,
        action_bins_per_dim,
    )
    assert predicate_warmed[2:5] == legacy[2:5], (predicate_warmed, legacy)

    t0 = time.perf_counter()
    for _ in range(repeat):
        legacy = legacy_scan(
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
            m_b,
            n_local_min,
            m_q_min,
            action_bins_per_dim,
        )
    legacy_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeat):
        numba = scan_fetch_slide_support_exact_with_constants(
            replay_obs,
            replay_h,
            replay_action_bin,
            replay_action_exec,
            scanned_indices,
            query_goal,
            full_key,
            policy_bin,
            constants,
            n_local_min,
            m_q_min,
            m_b,
            action_bins_per_dim,
        )
    numba_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeat):
        predicate = scan_fetch_slide_support_predicate_with_constants(
            replay_obs,
            replay_h,
            replay_action_bin,
            replay_action_exec,
            scanned_indices,
            query_goal,
            full_key,
            policy_bin,
            constants,
            n_local_min,
            m_q_min,
            m_b,
            action_bins_per_dim,
        )
    predicate_time = time.perf_counter() - t0

    print(f"legacy_time_sec {legacy_time:.6f}")
    print(f"numba_time_sec {numba_time:.6f}")
    print(f"predicate_time_sec {predicate_time:.6f}")
    print(f"speedup {legacy_time / max(numba_time, 1e-12):.3f}")
    print(f"predicate_speedup {legacy_time / max(predicate_time, 1e-12):.3f}")
    print(f"local_count_match {int(legacy[0] == numba[0])}")
    print(f"psi_match {int(legacy[3] == numba[3])}")
    print(f"predicate_psi_match {int(legacy[3] == predicate[3])}")
    print(f"predicate_mean_inspected {predicate[5]}")
    print(f"predicate_early_exit_code {predicate[6]}")
    print(f"numba_nopython_signatures {len(fetch_slide_support_exact_nopython_signatures())}")
    print(f"predicate_nopython_signatures {len(fetch_slide_support_predicate_nopython_signatures())}")


if __name__ == "__main__":
    main()
