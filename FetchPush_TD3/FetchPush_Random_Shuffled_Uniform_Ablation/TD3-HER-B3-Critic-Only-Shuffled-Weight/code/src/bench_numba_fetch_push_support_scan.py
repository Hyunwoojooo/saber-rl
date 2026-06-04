#!/usr/bin/env python3
from __future__ import annotations

import json
import time

import numpy as np

from metacog_lite_v0.numba_fetch_push_support_scan import (
    NUMBA_AVAILABLE,
    fetch_push_support_exact_nopython_signatures,
    make_fetch_push_support_scan_constants,
    scan_fetch_push_support_exact_with_constants,
    warmup_fetch_push_support_exact,
)
from metacog_lite_v0.task_fetch_push import FetchPushTaskSpec


def main() -> int:
    if not NUMBA_AVAILABLE:
        print(json.dumps({"status": "skipped", "reason": "numba is not available"}, indent=2))
        return 0

    rng = np.random.default_rng(20260528)
    n = 4096
    k = 512
    action_dim = 4
    task = FetchPushTaskSpec(
        action_low=np.full(action_dim, -1.0, dtype=np.float32),
        action_high=np.full(action_dim, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )
    constants = make_fetch_push_support_scan_constants(task)

    replay_obs = np.zeros((n, 10), dtype=np.float32)
    replay_obs[:, 0:3] = rng.normal(0.30, 0.04, size=(n, 3)).astype(np.float32)
    replay_obs[:, 3:6] = rng.normal(0.31, 0.04, size=(n, 3)).astype(np.float32)
    replay_obs[:, 2] = 0.42
    replay_obs[:, 5] = 0.42
    replay_h = rng.integers(1, 50, size=n, endpoint=True, dtype=np.int16)
    replay_action_exec = rng.uniform(-1.0, 1.0, size=(n, action_dim)).astype(np.float32)
    replay_action_bin = task.action_bin_batch(replay_action_exec, 5).astype(np.int64)
    replay_action_bin[::7] = -1

    scanned_indices = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
    query_goal = np.asarray([0.50, 0.00, 0.42], dtype=np.float32)
    query_full_key = int(task.full_context_hash_batch(
        replay_obs[0:1],
        query_goal[None, :],
        replay_h[0:1],
    )[0])
    query_policy_bin = int(task.action_bin_batch(replay_action_exec[0:1], 5)[0])

    warmup_fetch_push_support_exact(task, action_dim=action_dim, obs_dim=replay_obs.shape[1])
    scan_fetch_push_support_exact_with_constants(
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
        m_b=8,
        action_bins_per_dim=5,
    )

    loops = 2000
    t0 = time.perf_counter()
    numba_result = None
    for _ in range(loops):
        numba_result = scan_fetch_push_support_exact_with_constants(
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
            m_b=8,
            action_bins_per_dim=5,
        )
    numba_time = time.perf_counter() - t0
    print(json.dumps(
        {
            "status": "ok",
            "loops": loops,
            "numba_time_sec": numba_time,
            "numba_us_per_lookup": 1_000_000.0 * numba_time / max(1, loops),
            "numba_result": numba_result,
            "nopython_signature_count": len(fetch_push_support_exact_nopython_signatures()),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
