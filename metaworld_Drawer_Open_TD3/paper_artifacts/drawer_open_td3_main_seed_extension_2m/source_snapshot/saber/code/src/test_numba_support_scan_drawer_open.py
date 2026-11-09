import numpy as np

from metacog_lite_v0.numba_support_scan_drawer_open import (
    NUMBA_AVAILABLE,
    numba_support_predicate_nopython_signatures,
    numba_support_scan_nopython_signatures,
    scan_drawer_open_support_exact,
    scan_drawer_open_support_predicate,
)
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec


def _task() -> MetaWorldDrawerOpenTaskSpec:
    return MetaWorldDrawerOpenTaskSpec(
        action_low=np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        epsilon_g=0.05,
    )


def _scan_exact(
    task: MetaWorldDrawerOpenTaskSpec,
    obs: np.ndarray,
    goal: np.ndarray,
    h: np.ndarray,
    actions: np.ndarray,
    action_bins: np.ndarray,
    scanned: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    n_local_min: int = 4,
    m_q_min: int = 4,
    m_b: int = 16,
) -> tuple[int, int, int, int, int, int]:
    return scan_drawer_open_support_exact(
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
        n_local_min=n_local_min,
        m_q_min=m_q_min,
        m_b=m_b,
        ee_start=task.ee_slice[0],
        handle_start=task.handle_slice[0],
        gripper_start=task.gripper_slice[0],
        action_low=task.action_low,
        action_high=task.action_high,
        bins_per_dim=5,
    )


def _python_exact(
    task: MetaWorldDrawerOpenTaskSpec,
    obs: np.ndarray,
    goal: np.ndarray,
    h: np.ndarray,
    actions: np.ndarray,
    action_bins: np.ndarray,
    scanned: np.ndarray,
    query_full_key: int,
    query_policy_bin: int,
    n_local_min: int = 4,
    m_q_min: int = 4,
    m_b: int = 16,
) -> tuple[int, int, int, int, int, int]:
    ref_hash = task.full_context_hash_batch(obs[scanned], goal[None, :], h[scanned])
    local_positions = np.flatnonzero(ref_hash == int(query_full_key))
    local_indices = scanned[local_positions]
    local_count = int(local_indices.size)
    bins = action_bins[local_indices].astype(np.int64, copy=True)
    missing = bins < 0
    if np.any(missing):
        bins[missing] = task.action_bin_batch(actions[local_indices[missing]], bins_per_dim=5)
    policy_seen = int(np.any(bins == int(query_policy_bin)))
    available = min(local_count, int(m_b))
    support = int(local_count >= n_local_min and available >= m_q_min and policy_seen == 1)
    fallback = 1 - support
    return local_count, available, policy_seen, support, fallback, int(scanned.size)


def _make_positive_fixture(task: MetaWorldDrawerOpenTaskSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.zeros((8, 12), dtype=np.float32)
    goal = np.asarray([0.10, 0.20, 0.30], dtype=np.float32)
    obs[:, 0:3] = np.asarray([0.08, 0.18, 0.30], dtype=np.float32)
    obs[:, 3] = 0.0
    obs[:, 4:7] = np.asarray([0.09, 0.19, 0.30], dtype=np.float32)
    obs[4:, 4:7] = np.asarray([0.50, 0.50, 0.50], dtype=np.float32)
    h = np.asarray([20] * 8, dtype=np.int16)
    actions = np.zeros((8, 4), dtype=np.float32)
    return obs, goal, h, actions


def _assert_exact_matches_python(
    task: MetaWorldDrawerOpenTaskSpec,
    obs: np.ndarray,
    goal: np.ndarray,
    h: np.ndarray,
    actions: np.ndarray,
    action_bins: np.ndarray,
    scanned: np.ndarray,
    query_idx: int,
    query_policy_bin: int,
) -> None:
    query_full_key = int(task.full_context_hash_batch(obs[query_idx : query_idx + 1], goal[None, :], h[query_idx : query_idx + 1])[0])
    exact = _scan_exact(task, obs, goal, h, actions, action_bins, scanned, query_full_key, query_policy_bin)
    expected = _python_exact(task, obs, goal, h, actions, action_bins, scanned, query_full_key, query_policy_bin)
    assert exact == expected, (exact, expected, query_idx, query_policy_bin)


def main() -> None:
    if not NUMBA_AVAILABLE:
        print("numba unavailable; skipped")
        return

    task = _task()
    obs, goal, h, actions = _make_positive_fixture(task)
    action_bins = task.action_bin_batch(actions, bins_per_dim=5)
    scanned = np.arange(8, dtype=np.int64)
    query_full_key = int(task.full_context_hash_batch(obs[:1], goal[None, :], h[:1])[0])
    query_policy_bin = int(action_bins[0])

    exact = _scan_exact(task, obs, goal, h, actions, action_bins, scanned, query_full_key, query_policy_bin)
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
    assert exact == _python_exact(task, obs, goal, h, actions, action_bins, scanned, query_full_key, query_policy_bin)
    assert exact[0] == 4, exact
    assert exact[1] == 4, exact
    assert exact[2] == 1, exact
    assert exact[3] == 1, exact
    assert predicate[0] >= 4, predicate
    assert predicate[2] == 1, predicate
    assert predicate[3] == 1, predicate

    # action_bin=-1 fallback must match Python action discretization.
    fallback_bins = action_bins.copy()
    fallback_bins[[0, 2, 4]] = -1
    _assert_exact_matches_python(task, obs, goal, h, actions, fallback_bins, scanned, 0, query_policy_bin)

    # Boundary cases around phase thresholds must mirror full_context_hash_batch.
    boundary_obs = np.zeros((6, 12), dtype=np.float32)
    boundary_goal = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    boundary_obs[:, 3] = np.asarray([-0.7, -0.2, 0.1, 0.4, 0.0, 0.2], dtype=np.float32)
    boundary_obs[:, 4:7] = np.asarray(
        [
            [task.d_reach, 0.0, 0.0],
            [0.0, task.d_opening_mid, 0.0],
            [0.0, task.d_goal_near, 0.0],
            [0.03, 0.03, 0.0],
            [0.12, 0.12, 0.0],
            [0.30, 0.30, 0.0],
        ],
        dtype=np.float32,
    )
    boundary_obs[:, 0:3] = boundary_obs[:, 4:7]
    boundary_obs[0, 0] += task.d_reach
    boundary_h = np.asarray([10, 25, 50, 100, 200, 500], dtype=np.int16)
    boundary_actions = np.linspace(-0.8, 0.8, 24, dtype=np.float32).reshape(6, 4)
    boundary_bins = task.action_bin_batch(boundary_actions, bins_per_dim=5)
    boundary_scanned = np.arange(boundary_obs.shape[0], dtype=np.int64)
    for idx in range(boundary_obs.shape[0]):
        _assert_exact_matches_python(
            task,
            boundary_obs,
            boundary_goal,
            boundary_h,
            boundary_actions,
            boundary_bins,
            boundary_scanned,
            idx,
            int(boundary_bins[idx]),
        )

    rng = np.random.default_rng(123)
    rand_obs = rng.normal(0.0, 0.20, size=(64, 12)).astype(np.float32)
    rand_obs[:, 2] += 0.25
    rand_obs[:, 6] += 0.25
    rand_goal = np.asarray([0.05, 0.12, 0.18], dtype=np.float32)
    rand_h = rng.integers(1, 500, size=64, dtype=np.int16)
    rand_actions = rng.uniform(-1.0, 1.0, size=(64, 4)).astype(np.float32)
    rand_bins = task.action_bin_batch(rand_actions, bins_per_dim=5)
    rand_bins[::7] = -1
    rand_scanned = rng.choice(np.arange(64, dtype=np.int64), size=31, replace=False).astype(np.int64)
    for query_idx in [0, 3, 7, 19, 41, 63]:
        _assert_exact_matches_python(
            task,
            rand_obs,
            rand_goal,
            rand_h,
            rand_actions,
            rand_bins,
            rand_scanned,
            query_idx,
            int(task.action_bin_batch(rand_actions[query_idx : query_idx + 1], bins_per_dim=5)[0]),
        )

    assert len(numba_support_scan_nopython_signatures()) > 0
    assert len(numba_support_predicate_nopython_signatures()) > 0
    print("test_numba_support_scan_drawer_open.py passed")


if __name__ == "__main__":
    main()
