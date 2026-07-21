#!/usr/bin/env python3
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from metacog_lite_v0.assignment_diagnostics import AssignmentDiagnosticsWindow
from metacog_lite_v0.config import (
    COUNT_MATCHED_ASSIGNMENT_MODES,
    SUPPORT_ASSIGNMENT_MODES,
    LiteConfig,
    SeedConfig,
)
from metacog_lite_v0.weights import (
    GapStats,
    LiteWeightComputer,
    batch_wide_count_matched_assignment,
    context_eligible_count_matched_assignment,
)
from test_support_query_modes import (
    ACTION_A,
    ACTION_B,
    VectorAgent,
    make_batch,
    make_config,
    make_replay,
    make_task,
)


def _rng_state(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state, sort_keys=True)


def _normalized(weights: np.ndarray) -> np.ndarray:
    return (weights / float(np.mean(weights))).astype(np.float32)


def _ess(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(np.square(np.sum(values)) / np.sum(np.square(values)))


def _diagnostic_metrics(
    context_eligible: np.ndarray,
    main: np.ndarray,
    assigned: np.ndarray,
) -> dict[str, float]:
    batch_size = int(main.size)
    main_raw = np.ones(batch_size, dtype=np.float32)
    assigned_raw = np.ones(batch_size, dtype=np.float32)
    main_raw[main] = 2.0
    assigned_raw[assigned] = 2.0
    main_normalized = _normalized(main_raw)
    assigned_normalized = _normalized(assigned_raw)
    overlap = int(np.sum(np.logical_and(main, assigned)))
    union = int(np.sum(np.logical_or(main, assigned)))
    k = int(np.sum(main))
    main_sum = float(np.sum(main_normalized, dtype=np.float64))
    assigned_sum = float(np.sum(assigned_normalized, dtype=np.float64))
    return {
        "assignment/batch_size": float(batch_size),
        "assignment/context_eligible_count": float(np.sum(context_eligible)),
        "assignment/assignment_pool_count": float(batch_size),
        "assignment/assignment_pool_rate": 1.0,
        "assignment/main_positive_count": float(k),
        "assignment/assigned_positive_count": float(np.sum(assigned)),
        "assignment/context_eligible_rate": float(np.mean(context_eligible)),
        "assignment/main_positive_rate": float(np.mean(main)),
        "assignment/assigned_positive_rate": float(np.mean(assigned)),
        "assignment/overlap_count": float(overlap),
        "assignment/overlap_over_k": float(overlap / k) if k else 0.0,
        "assignment/jaccard": float(overlap / union) if union else 0.0,
        "assignment/raw_histogram_mismatch_count": 0.0,
        "assignment/normalized_histogram_mismatch_count": 0.0,
        "assignment/invariant_failure_count": 0.0,
        "assignment/reference_raw_weight_mean": float(np.mean(main_raw)),
        "assignment/reference_raw_weight_std": float(np.std(main_raw)),
        "assignment/reference_raw_weight_ESS": _ess(main_raw),
        "assignment/assigned_raw_weight_mean": float(np.mean(assigned_raw)),
        "assignment/assigned_raw_weight_std": float(np.std(assigned_raw)),
        "assignment/assigned_raw_weight_ESS": _ess(assigned_raw),
        "assignment/reference_normalized_weight_mean": float(
            np.mean(main_normalized)
        ),
        "assignment/reference_normalized_weight_std": float(
            np.std(main_normalized)
        ),
        "assignment/reference_normalized_weight_ESS": _ess(main_normalized),
        "assignment/assigned_normalized_weight_mean": float(
            np.mean(assigned_normalized)
        ),
        "assignment/assigned_normalized_weight_std": float(
            np.std(assigned_normalized)
        ),
        "assignment/assigned_normalized_weight_ESS": _ess(assigned_normalized),
        "assignment/reference_normalized_weight_sum": main_sum,
        "assignment/assigned_normalized_weight_sum": assigned_sum,
        "assignment/normalized_weight_sum_abs_diff": abs(
            main_sum - assigned_sum
        ),
    }


def _batch_wide_compute(cache_enabled: bool):
    task = make_task()
    cfg = make_config("current_policy", cache_enabled)
    cfg.support_assignment_mode = "batch_wide_count_matched"
    replay = make_replay(task)
    batch = make_batch()
    agent = VectorAgent(np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B]))
    weight_rng = np.random.default_rng(20_001)
    assignment_rng = np.random.default_rng(30_001)
    weight_before = _rng_state(weight_rng)
    assignment_before = _rng_state(assignment_rng)
    result = LiteWeightComputer(
        cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(
        batch,
        replay,
        agent,
        weight_rng,
        torch.device("cpu"),
        assignment_rng=assignment_rng,
    )
    return (
        result,
        weight_before,
        _rng_state(weight_rng),
        assignment_before,
        _rng_state(assignment_rng),
    )


def _main_compute(cache_enabled: bool):
    task = make_task()
    cfg = make_config("current_policy", cache_enabled)
    cfg.support_assignment_mode = "support"
    return LiteWeightComputer(
        cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(
        make_batch(),
        make_replay(task),
        VectorAgent(np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B])),
        np.random.default_rng(20_001),
        torch.device("cpu"),
    )


def test_exact_k_edges_and_no_replacement() -> None:
    batch_size = 8
    for k in (0, 1, 4, batch_size):
        main = np.zeros(batch_size, dtype=np.bool_)
        main[:k] = True
        rng = np.random.default_rng(30_001)
        before = _rng_state(rng)
        assigned = batch_wide_count_matched_assignment(main, rng)
        assert assigned.dtype == np.bool_
        assert assigned.shape == main.shape
        assert int(np.sum(assigned)) == k
        assert np.unique(np.flatnonzero(assigned)).size == k
        if k in (0, batch_size):
            assert _rng_state(rng) == before
        else:
            assert _rng_state(rng) != before

    try:
        batch_wide_count_matched_assignment(
            np.ones((2, 2), dtype=np.bool_), np.random.default_rng(1)
        )
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("two-dimensional Main mask was accepted")

    class RecordingRNG:
        def __init__(self) -> None:
            self.called = False

        def choice(self, candidates, *, size, replace):
            self.called = True
            np.testing.assert_array_equal(candidates, np.arange(batch_size))
            assert size == 3
            assert replace is False
            return np.asarray([0, 3, 7], dtype=np.int64)

    recording_rng = RecordingRNG()
    main = np.asarray([True, True, True, False, False, False, False, False])
    assigned = batch_wide_count_matched_assignment(main, recording_rng)
    assert recording_rng.called
    np.testing.assert_array_equal(np.flatnonzero(assigned), [0, 3, 7])


def test_identity_independence_full_batch_and_overlap() -> None:
    main_left = np.asarray([True, False, False, False])
    main_right = np.asarray([False, False, False, True])
    left = batch_wide_count_matched_assignment(
        main_left, np.random.default_rng(30_001)
    )
    right = batch_wide_count_matched_assignment(
        main_right, np.random.default_rng(30_001)
    )
    np.testing.assert_array_equal(left, right)
    # Seed 30001 selects position 3, proving a context-ineligible position can
    # remain a candidate when E contains only positions 0 and 1.
    context_eligible = np.asarray([True, True, False, False])
    assert bool(left[3]) and not bool(context_eligible[3])

    # Main-positive positions are not excluded: this seed selects position 0.
    overlapping = batch_wide_count_matched_assignment(
        main_left, np.random.default_rng(30_003)
    )
    assert bool(np.logical_and(overlapping, main_left).any())

    repeated = batch_wide_count_matched_assignment(
        np.asarray([True] * 3 + [False] * 17),
        np.random.default_rng(27027),
    )
    same = batch_wide_count_matched_assignment(
        np.asarray([False] * 17 + [True] * 3),
        np.random.default_rng(27027),
    )
    different = batch_wide_count_matched_assignment(
        np.asarray([True] * 3 + [False] * 17),
        np.random.default_rng(27028),
    )
    np.testing.assert_array_equal(repeated, same)
    assert not np.array_equal(repeated, different)


def test_rng_isolation() -> None:
    np.random.seed(1234)
    global_before = copy.deepcopy(np.random.get_state())
    torch.manual_seed(1234)
    torch_before = torch.get_rng_state().clone()
    protected = [
        np.random.default_rng(11),
        np.random.default_rng(12),
        np.random.default_rng(20_001),
    ]
    protected_before = [_rng_state(rng) for rng in protected]
    assignment_rng = np.random.default_rng(30_001)
    assignment_before = _rng_state(assignment_rng)
    main = np.asarray([True, True, False, False, False, False])
    batch_wide_count_matched_assignment(main, assignment_rng)

    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before)
    assert [_rng_state(rng) for rng in protected] == protected_before
    assert _rng_state(assignment_rng) != assignment_before


def test_diagnostics_checkpoint_and_histogram_invariants() -> None:
    context_eligible = np.asarray([True, True, False, False])
    main = np.asarray([True, False, False, False])
    rng = np.random.default_rng(30_001)
    window = AssignmentDiagnosticsWindow(window_size=2)
    assigned_1 = batch_wide_count_matched_assignment(main, rng)
    assert window.update(
        1, _diagnostic_metrics(context_eligible, main, assigned_1)
    ) is None
    rng_state = copy.deepcopy(rng.bit_generator.state)
    window_state = copy.deepcopy(window.state_dict())

    assigned_2 = batch_wide_count_matched_assignment(main, rng)
    expected_row = window.update(
        2, _diagnostic_metrics(context_eligible, main, assigned_2)
    )
    assert expected_row is not None
    assert expected_row["batch_size_mean"] == 4.0
    assert expected_row["batch_size_min"] == 4.0
    assert expected_row["batch_size_max"] == 4.0
    assert expected_row["assignment_pool_count_mean"] == 4.0
    assert expected_row["assignment_pool_count_min"] == 4.0
    assert expected_row["assignment_pool_count_max"] == 4.0
    assert expected_row["assignment_pool_rate_mean"] == 1.0
    assert expected_row["context_eligible_count_mean"] == 2.0
    assert expected_row["exact_k_mismatch_count"] == 0.0
    assert expected_row["raw_histogram_mismatch_count"] == 0.0
    assert expected_row["normalized_histogram_mismatch_count"] == 0.0

    restored_rng = np.random.default_rng(9)
    restored_rng.bit_generator.state = rng_state
    restored_window = AssignmentDiagnosticsWindow(window_size=2)
    restored_window.load_state_dict(window_state)
    replayed = batch_wide_count_matched_assignment(main, restored_rng)
    np.testing.assert_array_equal(replayed, assigned_2)
    replayed_row = restored_window.update(
        2, _diagnostic_metrics(context_eligible, main, replayed)
    )
    assert replayed_row == expected_row


def test_runtime_opt_parity_and_weight_invariants() -> None:
    cached, weight_before, weight_after, assignment_before, assignment_after = (
        _batch_wide_compute(True)
    )
    uncached, *_ = _batch_wide_compute(False)
    main = _main_compute(True)
    for field in (
        "psi_q_rank",
        "psi_a_policy",
        "len_a_q_local",
        "len_a_q_sup",
        "num_coarse_candidates",
        "num_scanned_candidates",
    ):
        np.testing.assert_array_equal(getattr(cached, field), getattr(uncached, field))
        np.testing.assert_array_equal(getattr(cached, field), getattr(main, field))

    main_mask = cached.psi_q_rank > 0.5
    assigned_mask = cached.w_p_raw > 1.5
    np.testing.assert_array_equal(main_mask, [True, True, False, False])
    np.testing.assert_array_equal(assigned_mask, [False, True, True, False])
    assert int(np.sum(main_mask)) == int(np.sum(assigned_mask)) == 2
    np.testing.assert_array_equal(
        np.sort(cached.w_p_raw), np.asarray([1.0, 1.0, 2.0, 2.0])
    )
    np.testing.assert_array_equal(np.sort(cached.w_p_raw), np.sort(main.w_p_raw))
    np.testing.assert_allclose(
        np.sort(cached.omega_critic), np.sort(main.omega_critic), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        np.mean(cached.omega_critic, dtype=np.float64), 1.0, rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        np.sum(cached.omega_critic, dtype=np.float64), 4.0, rtol=0.0, atol=1e-6
    )
    assert np.all(np.isfinite(cached.omega_critic))
    assert np.all(cached.omega_critic > 0.0)
    np.testing.assert_array_equal(cached.omega_actor, np.ones(4, dtype=np.float32))
    assert cached.extra_metrics["assignment/enabled"] == 1.0
    assert cached.extra_metrics["assignment/batch_size"] == 4.0
    assert cached.extra_metrics["assignment/context_eligible_count"] == 4.0
    assert cached.extra_metrics["assignment/assignment_pool_count"] == 4.0
    assert cached.extra_metrics["assignment/assignment_pool_rate"] == 1.0
    assert cached.extra_metrics["assignment/main_positive_count"] == 2.0
    assert cached.extra_metrics["assignment/assigned_positive_count"] == 2.0
    assert cached.extra_metrics["assignment/overlap_count"] == 1.0
    assert cached.extra_metrics["assignment/overlap_over_k"] == 0.5
    assert cached.extra_metrics["assignment/jaccard"] == 1.0 / 3.0
    assert cached.extra_metrics["assignment/raw_histogram_mismatch_count"] == 0.0
    assert (
        cached.extra_metrics["assignment/normalized_histogram_mismatch_count"]
        == 0.0
    )
    assert cached.extra_metrics["assignment/invariant_failure_count"] == 0.0
    assert weight_before == weight_after
    assert assignment_before != assignment_after


def test_context_ineligible_positions_are_assignment_candidates() -> None:
    task = make_task()
    cfg = make_config("current_policy", cache_enabled=True)
    cfg.support_assignment_mode = "batch_wide_count_matched"
    replay = make_replay(task)
    batch = make_batch()
    batch.observation[2:, 3:6] = np.asarray([0.8, 0.8, 0.8], dtype=np.float32)
    result = LiteWeightComputer(
        cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(
        batch,
        replay,
        VectorAgent(np.stack([ACTION_A, ACTION_B, ACTION_A, ACTION_A])),
        np.random.default_rng(20_001),
        torch.device("cpu"),
        assignment_rng=np.random.default_rng(30_001),
    )
    context_eligible = np.logical_and(
        result.len_a_q_local >= cfg.N_local_min,
        np.minimum(result.len_a_q_local, cfg.M_B) >= cfg.M_Q_min,
    )
    main = result.psi_q_rank > 0.5
    assigned = result.w_p_raw > 1.5
    np.testing.assert_array_equal(context_eligible, [True, True, False, False])
    np.testing.assert_array_equal(main, [True, False, False, False])
    np.testing.assert_array_equal(assigned, [False, False, False, True])
    assert np.all(~main | context_eligible)
    assert np.any(assigned & ~context_eligible)
    assert result.extra_metrics["assignment/context_eligible_count"] == 2.0
    assert result.extra_metrics["assignment/assignment_pool_count"] == 4.0
    assert result.extra_metrics["assignment/assignment_pool_rate"] == 1.0


def test_config_and_strict_mode_compatibility() -> None:
    assert COUNT_MATCHED_ASSIGNMENT_MODES == {
        "context_eligible_count_matched",
        "batch_wide_count_matched",
    }
    assert SUPPORT_ASSIGNMENT_MODES == {
        "support",
        "context_eligible_count_matched",
        "batch_wide_count_matched",
    }
    assert SeedConfig().assignment_seed == 30_001
    for mode in COUNT_MATCHED_ASSIGNMENT_MODES:
        cfg = LiteConfig(
            task_name="fetch_pick_and_place",
            method="b3_support_runtime_opt",
            support_query_mode="current_policy",
            support_assignment_mode=mode,
            weight_scope="critic_only",
            weighting_mode="batch_norm",
        )
        cfg.normalize()

    eligible = np.asarray([True, True, False, True])
    main = np.asarray([True, False, False, False])
    strict = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(30_001)
    )
    assert np.all(~strict | eligible)
    assert int(np.sum(strict)) == int(np.sum(main))

    invalid = [
        {"support_query_mode": "replay_action"},
        {"method": "b4_lite"},
        {"weight_scope": "actor_critic"},
        {"weighting_mode": "raw"},
    ]
    for override in invalid:
        kwargs = {
            "task_name": "fetch_pick_and_place",
            "method": "b3_support_runtime_opt",
            "support_query_mode": "current_policy",
            "support_assignment_mode": "batch_wide_count_matched",
            "weight_scope": "critic_only",
            "weighting_mode": "batch_norm",
            **override,
        }
        try:
            LiteConfig(**kwargs).normalize()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count-matched config was accepted: {override}")


def main() -> int:
    test_exact_k_edges_and_no_replacement()
    test_identity_independence_full_batch_and_overlap()
    test_rng_isolation()
    test_diagnostics_checkpoint_and_histogram_invariants()
    test_runtime_opt_parity_and_weight_invariants()
    test_context_ineligible_positions_are_assignment_candidates()
    test_config_and_strict_mode_compatibility()
    print("test_batch_wide_count_matched.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
