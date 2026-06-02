#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

from metacog_lite_v0.numba_support_scan_fpp import (  # noqa: E402
    NUMBA_AVAILABLE,
    scan_fetch_pick_and_place_support_exact,
    scan_fpp_support_quality_reference,
    scan_fpp_support_quality_with_constants,
)
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec  # noqa: E402


def make_obs(ee=(0.0, 0.0, 0.0), obj=(0.04, 0.0, 0.44), gap=0.06) -> np.ndarray:
    obs = np.zeros(25, dtype=np.float32)
    obs[0:3] = np.asarray(ee, dtype=np.float32)
    obs[3:6] = np.asarray(obj, dtype=np.float32)
    obs[9:11] = np.asarray([0.5 * gap, 0.5 * gap], dtype=np.float32)
    return obs


def main() -> int:
    if not NUMBA_AVAILABLE:
        print("numba_available=false")
        return 0
    task = FetchPickAndPlaceTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )
    n = 4096
    scan_n = 512
    loops = 500
    query_goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    query_obs = make_obs()
    query_h = 20
    policy_action = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    other_action = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    policy_bin = int(task.action_bin_batch(policy_action[None, :], 5)[0])
    other_bin = int(task.action_bin_batch(other_action[None, :], 5)[0])
    obs = np.stack([make_obs() for _ in range(n)]).astype(np.float32)
    next_obs = obs.copy()
    next_obs[:, 0:3] = np.asarray([0.04, 0.0, 0.42], dtype=np.float32)
    h = np.full(n, query_h, dtype=np.int16)
    action_exec = np.tile(other_action[None, :], (n, 1)).astype(np.float32)
    action_bin = np.full(n, other_bin, dtype=np.int64)
    action_exec[::3] = policy_action
    action_bin[::3] = policy_bin
    scanned = np.arange(n - scan_n, n, dtype=np.int64)
    full_key = task.full_context_hash(query_obs, query_goal, query_h)
    common = dict(
        replay_obs=obs,
        replay_h=h,
        replay_action_bin=action_bin,
        replay_action_exec=action_exec,
        scanned_indices=scanned,
        query_goal=query_goal,
        query_full_key=full_key,
        query_policy_bin=policy_bin,
        hand_obj_bins=np.asarray(task.hand_obj_bins, dtype=np.float32),
        obj_goal_bins=np.asarray(task.obj_goal_bins, dtype=np.float32),
        object_z_bins=np.asarray(task.object_z_bins, dtype=np.float32),
        gripper_gap_bins=np.asarray(task.gripper_gap_bins, dtype=np.float32),
        horizon_bins=np.asarray(task.horizon_bins, dtype=np.float32),
        d_reach=task.d_reach,
        z_lift_min=task.z_lift_min,
        grip_close_thresh=task.grip_close_thresh,
        n_local_min=4,
        m_q_min=4,
        m_b=16,
        ee_start=task.ee_slice[0],
        object_start=task.object_slice[0],
        gripper_start=task.gripper_slice[0],
        action_low=task.action_low,
        action_high=task.action_high,
        bins_per_dim=5,
    )
    quality_common = dict(
        **common,
        replay_next_obs=next_obs,
        d_goal_near=task.d_goal_near,
        quality_mode_id=1,
        quality_floor=0.5,
        hand_progress_scale=0.03,
        goal_progress_scale=0.03,
        z_progress_scale=0.03,
        grip_progress_scale=0.02,
        near_margin_mult=0.5,
    )
    base = scan_fetch_pick_and_place_support_exact(**common)
    qual = scan_fpp_support_quality_with_constants(**quality_common)
    ref = scan_fpp_support_quality_reference(**quality_common)
    t0 = time.perf_counter()
    for _ in range(loops):
        base = scan_fetch_pick_and_place_support_exact(**common)
    base_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(loops):
        qual = scan_fpp_support_quality_with_constants(**quality_common)
    qual_t = time.perf_counter() - t0
    print(f"base_scan_time_sec={base_t:.6f}")
    print(f"quality_scan_time_sec={qual_t:.6f}")
    print(f"quality_overhead_ratio={qual_t / max(base_t, 1e-12):.3f}")
    print(f"psi_match={int(base[3] == qual[3])}")
    print(f"local_count_match={int(base[0] == qual[0])}")
    print(f"quality_reference_match={int(np.allclose(np.asarray(qual[5:13], dtype=np.float64), np.asarray(ref[5:13], dtype=np.float64), atol=1e-6))}")
    print(f"support_quality={qual[5]:.6f}")
    print(f"support_score={qual[6]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
