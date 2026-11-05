#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from metacog_lite_v0.numba_support_scan_drawer_open import (
    NUMBA_AVAILABLE,
    scan_drawer_open_support_exact,
    scan_drawer_open_support_predicate,
)
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.weights import GapStats, LiteWeightComputer, LocalScan


def load_original_scanner():
    source = (
        Path(__file__).resolve().parents[2]
        / "upstream/code/src/metacog_lite_v0/numba_support_scan_drawer_open.py"
    )
    spec = importlib.util.spec_from_file_location("drawer_open_original_scanner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original scanner from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    task = MetaWorldDrawerOpenTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )
    observation = np.zeros((6, 7), dtype=np.float32)
    observation[:, 0:3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    observation[:, 3] = 0.2
    observation[:, 4:7] = np.asarray([0.25, 0.1, 0.0], dtype=np.float32)
    goal = np.asarray([0.6, 0.1, 0.0], dtype=np.float32)
    horizon = np.full(6, 20, dtype=np.int16)
    replay_action = np.full((6, 4), -0.9, dtype=np.float32)
    query_action = np.full((1, 4), 0.9, dtype=np.float32)
    replay_bins = task.action_bin_batch(replay_action, 5)
    query_bin = int(task.action_bin_batch(query_action, 5)[0])
    full_key = task.full_context_hash(observation[0], goal, int(horizon[0]))
    common = dict(
        replay_obs=observation,
        replay_h=horizon,
        replay_action_bin=replay_bins,
        replay_action_exec=replay_action,
        scanned_indices=np.arange(6, dtype=np.int64),
        query_goal=goal,
        query_full_key=full_key,
        query_policy_bin=query_bin,
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
        ee_start=0,
        handle_start=4,
        gripper_start=3,
        action_low=task.action_low,
        action_high=task.action_high,
        bins_per_dim=5,
    )
    return task, observation, goal, horizon, replay_action, replay_bins, query_bin, common


def test_scanner_membership_switch_preserves_default_main() -> None:
    if not NUMBA_AVAILABLE:
        return
    *_, common = fixture()

    default_main = scan_drawer_open_support_exact(**common)
    explicit_main = scan_drawer_open_support_exact(
        **common, require_action_membership=True
    )
    np.testing.assert_array_equal(default_main, explicit_main)
    assert default_main[0] == 6
    assert default_main[2] == 0
    assert default_main[3] == 0

    eligibility_only = scan_drawer_open_support_exact(
        **common, require_action_membership=False
    )
    assert eligibility_only[0] == 6
    assert eligibility_only[1] == 6
    assert eligibility_only[2] == 1
    assert eligibility_only[3] == 1

    predicate = scan_drawer_open_support_predicate(
        **common, require_action_membership=False
    )
    assert predicate[0] == 4
    assert predicate[1] == 4
    assert predicate[2] == 1
    assert predicate[3] == 1
    assert predicate[5] == 4
    assert predicate[6] == 1


def test_default_scanner_exact_tuple_parity_with_original_main() -> None:
    if not NUMBA_AVAILABLE:
        return
    original = load_original_scanner()
    task, _obs, _goal, _h, replay_action, replay_bins, _query_bin, common = fixture()
    cases = []
    cases.append(dict(common))

    present = dict(common)
    present["query_policy_bin"] = int(replay_bins[0])
    cases.append(present)

    insufficient = dict(common)
    insufficient["n_local_min"] = 7
    insufficient["m_q_min"] = 7
    cases.append(insufficient)

    negative_bins = dict(common)
    negative_bins["replay_action_bin"] = np.full(6, -1, dtype=np.int64)
    negative_bins["query_policy_bin"] = int(task.action_bin_batch(replay_action[:1], 5)[0])
    cases.append(negative_bins)

    for case in cases:
        expected_exact = original.scan_drawer_open_support_exact(**case)
        observed_exact = scan_drawer_open_support_exact(**case)
        assert observed_exact == expected_exact
        expected_predicate = original.scan_drawer_open_support_predicate(**case)
        observed_predicate = scan_drawer_open_support_predicate(**case)
        assert observed_predicate == expected_predicate


def test_absent_coarse_action_bin_still_computes_E() -> None:
    (
        task,
        observation,
        goal,
        horizon,
        replay_action,
        replay_bins,
        query_bin,
        _common,
    ) = fixture()
    cfg = SimpleNamespace(
        numba_support_scan=True,
        support_predicate_early_exit=True,
        N_local_min=4,
        M_Q_min=4,
        M_B=16,
        action_bins_per_dim=5,
        support_short_circuit=True,
        policy_support_mode="bin",
        support_assignment_mode="context_eligible_count_matched",
    )
    computer = LiteWeightComputer(
        config=cfg,
        task=task,
        gap_stats=GapStats(8, 128, 0.1, 0.9, 1e-6),
    )

    class Replay:
        def __init__(self) -> None:
            self.observation = observation
            self.h = horizon
            self.action_bin = replay_bins
            self.action_exec = replay_action

        @staticmethod
        def action_index_has_action_bin(coarse_key: int, action_bin: int) -> bool:
            del coarse_key, action_bin
            return False

    replay = Replay()
    coarse_key = task.coarse_hash(observation[0], int(horizon[0]))
    scan = LocalScan(
        indices=np.arange(6, dtype=np.int64),
        coarse_key=coarse_key,
        action_bins=replay_bins,
        bucket_version=1,
        num_coarse_candidates=6,
        local_scan_truncated=False,
    )
    context, _full_scan, exact_negative, *_ = computer._local_context_from_scan(
        full_key=task.full_context_hash(observation[0], goal, int(horizon[0])),
        goal=goal,
        replay=replay,
        scan=scan,
        policy_bins={query_bin},
        policy_bin=query_bin,
    )
    assert exact_negative
    assert context.psi_a_policy in {0.0, None}
    assert context.context_count_eligible == 1.0
    main_positive = bool(
        context.context_count_eligible
        and context.psi_a_policy is not None
        and context.psi_a_policy > 0.5
    )
    assert not main_positive

    cfg.support_assignment_mode = "support"
    main_computer = LiteWeightComputer(
        config=cfg,
        task=task,
        gap_stats=GapStats(8, 128, 0.1, 0.9, 1e-6),
    )
    main_context, main_full_scan, main_exact_negative, *_ = (
        main_computer._local_context_from_scan(
            full_key=task.full_context_hash(
                observation[0], goal, int(horizon[0])
            ),
            goal=goal,
            replay=replay,
            scan=scan,
            policy_bins={query_bin},
            policy_bin=query_bin,
        )
    )
    assert main_exact_negative
    assert not main_full_scan
    assert main_context.context_count_eligible is None


def main() -> int:
    test_scanner_membership_switch_preserves_default_main()
    test_default_scanner_exact_tuple_parity_with_original_main()
    test_absent_coarse_action_bin_still_computes_E()
    print("test_strict_drawer_open_scanner.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
