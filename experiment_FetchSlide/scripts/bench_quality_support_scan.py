#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "version_v0_lite" / "src"))

from metacog_lite_v0.task_fetch_slide import FetchSlideTaskSpec
from saber_fetch_slide.numba_support_scan import (
    NUMBA_AVAILABLE,
    make_fetch_slide_support_scan_constants,
    scan_fetch_slide_support_exact_with_constants,
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


def make_obs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[:, 0:3] = rng.normal(0.0, 0.01, size=(n, 3)).astype(np.float32)
    obs[:, 3:6] = np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
    obs[:, 14:17] = rng.normal(0.0, 0.02, size=(n, 3)).astype(np.float32)
    obs[:, 20:23] = 0.0

    next_obs = obs.copy()
    next_obs[:, 0] += rng.uniform(0.0, 0.04, size=n).astype(np.float32)
    next_obs[:, 14] = rng.uniform(0.0, 0.10, size=n).astype(np.float32)

    action = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    return obs, next_obs, action


def main() -> None:
    if not NUMBA_AVAILABLE:
        print("numba_available 0")
        return
    task = make_task()
    constants = make_fetch_slide_support_scan_constants(task)
    replay_obs, replay_next_obs, replay_action_exec = make_obs(4096)
    replay_h = np.full(replay_obs.shape[0], 20, dtype=np.int16)
    replay_action_bin = task.action_bin_batch(replay_action_exec, ACTION_BINS_PER_DIM)
    query_goal = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    query_obs = replay_obs[0].copy()
    query_full_key = int(task.full_context_hash_batch(query_obs[None, :], query_goal[None, :], replay_h[:1])[0])
    query_policy_bin = int(replay_action_bin[0])
    scanned_indices = np.arange(1024, dtype=np.int64)

    exact_kwargs = dict(
        replay_obs=replay_obs,
        replay_h=replay_h,
        replay_action_bin=replay_action_bin,
        replay_action_exec=replay_action_exec,
        scanned_indices=scanned_indices,
        query_goal=query_goal,
        query_full_key=query_full_key,
        query_policy_bin=query_policy_bin,
        constants=constants,
        n_local_min=4,
        m_q_min=4,
        m_b=16,
        action_bins_per_dim=ACTION_BINS_PER_DIM,
    )
    quality_kwargs = dict(
        replay_next_obs=replay_next_obs,
        quality_floor=0.25,
        approach_scale=0.02,
        goal_progress_scale=0.03,
        align_min=0.0,
        strike_speed_scale_mult=2.0,
        near_margin_mult=0.25,
        near_speed_mult=1.0,
        **exact_kwargs,
    )

    exact_result = scan_fetch_slide_support_exact_with_constants(**exact_kwargs)
    quality_result = scan_fetch_slide_support_quality_with_constants(**quality_kwargs)

    loops = 500
    t0 = time.perf_counter()
    for _ in range(loops):
        exact_result = scan_fetch_slide_support_exact_with_constants(**exact_kwargs)
    exact_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(loops):
        quality_result = scan_fetch_slide_support_quality_with_constants(**quality_kwargs)
    quality_time = time.perf_counter() - t1

    base_scan_time_sec = exact_time / float(loops)
    quality_scan_time_sec = quality_time / float(loops)
    print(f"base_scan_time_sec {base_scan_time_sec:.8f}")
    print(f"quality_scan_time_sec {quality_scan_time_sec:.8f}")
    print(f"quality_overhead_ratio {quality_scan_time_sec / max(base_scan_time_sec, 1e-12):.3f}")
    print(f"local_count_match {int(exact_result[0] == quality_result[0])}")
    print(f"psi_match {int(exact_result[3] == quality_result[3])}")


if __name__ == "__main__":
    main()
