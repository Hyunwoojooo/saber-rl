#!/usr/bin/env python3
from __future__ import annotations

import copy

import numpy as np

from metacog_lite_v0.assignment_diagnostics import AssignmentDiagnosticsWindow
from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.weights import (
    context_eligible_count_matched_assignment,
    sample_exact_k_from_eligible,
)


def normalized(weights: np.ndarray) -> np.ndarray:
    return (weights / float(np.mean(weights))).astype(np.float32)


def assert_histograms_match(
    eligible: np.ndarray,
    main: np.ndarray,
    assigned: np.ndarray,
) -> None:
    assert np.all(np.logical_not(assigned) | eligible)
    assert int(np.sum(assigned)) == int(np.sum(main))
    main_raw = np.ones(main.size, dtype=np.float32)
    assigned_raw = np.ones(main.size, dtype=np.float32)
    main_raw[main] = 2.0
    assigned_raw[assigned] = 2.0
    np.testing.assert_array_equal(np.sort(main_raw), np.sort(assigned_raw))
    np.testing.assert_allclose(
        np.sort(normalized(main_raw)),
        np.sort(normalized(assigned_raw)),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.sum(normalized(main_raw), dtype=np.float64),
        float(main.size),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.sum(normalized(assigned_raw), dtype=np.float64),
        float(main.size),
        rtol=0.0,
        atol=1e-6,
    )


def diagnostic_metrics(
    eligible: np.ndarray,
    main: np.ndarray,
    assigned: np.ndarray,
) -> dict[str, float]:
    main_raw = np.ones(main.size, dtype=np.float32)
    assigned_raw = np.ones(main.size, dtype=np.float32)
    main_raw[main] = 2.0
    assigned_raw[assigned] = 2.0
    main_norm = normalized(main_raw)
    assigned_norm = normalized(assigned_raw)
    overlap = int(np.sum(np.logical_and(main, assigned)))
    union = int(np.sum(np.logical_or(main, assigned)))
    k = int(np.sum(main))
    ess = lambda values: float(
        np.square(np.sum(values, dtype=np.float64))
        / np.sum(np.square(values, dtype=np.float64))
    )
    return {
        "assignment/context_eligible_count": float(np.sum(eligible)),
        "assignment/main_positive_count": float(k),
        "assignment/assigned_positive_count": float(np.sum(assigned)),
        "assignment/context_eligible_rate": float(np.mean(eligible)),
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
        "assignment/reference_raw_weight_ESS": ess(main_raw),
        "assignment/assigned_raw_weight_mean": float(np.mean(assigned_raw)),
        "assignment/assigned_raw_weight_std": float(np.std(assigned_raw)),
        "assignment/assigned_raw_weight_ESS": ess(assigned_raw),
        "assignment/reference_normalized_weight_mean": float(np.mean(main_norm)),
        "assignment/reference_normalized_weight_std": float(np.std(main_norm)),
        "assignment/reference_normalized_weight_ESS": ess(main_norm),
        "assignment/assigned_normalized_weight_mean": float(
            np.mean(assigned_norm)
        ),
        "assignment/assigned_normalized_weight_std": float(np.std(assigned_norm)),
        "assignment/assigned_normalized_weight_ESS": ess(assigned_norm),
        "assignment/reference_normalized_weight_sum": float(
            np.sum(main_norm, dtype=np.float64)
        ),
        "assignment/assigned_normalized_weight_sum": float(
            np.sum(assigned_norm, dtype=np.float64)
        ),
        "assignment/normalized_weight_sum_abs_diff": float(
            abs(
                np.sum(main_norm, dtype=np.float64)
                - np.sum(assigned_norm, dtype=np.float64)
            )
        ),
    }


def test_edge_cases() -> None:
    eligible = np.asarray([True, True, False, True, False])
    empty_main = np.zeros(5, dtype=np.bool_)
    empty = context_eligible_count_matched_assignment(
        eligible, empty_main, np.random.default_rng(1)
    )
    np.testing.assert_array_equal(empty, np.zeros(5, dtype=np.bool_))
    assert_histograms_match(eligible, empty_main, empty)

    full = context_eligible_count_matched_assignment(
        eligible, eligible, np.random.default_rng(2)
    )
    np.testing.assert_array_equal(full, eligible)
    assert_histograms_match(eligible, eligible, full)

    partial_main = np.asarray([True, False, False, True, False])
    partial = context_eligible_count_matched_assignment(
        eligible, partial_main, np.random.default_rng(3)
    )
    assert_histograms_match(eligible, partial_main, partial)

    try:
        sample_exact_k_from_eligible(eligible, 4, np.random.default_rng(4))
    except RuntimeError as exc:
        assert "exceeds |E|" in str(exc)
    else:
        raise AssertionError("k > |E| did not fail")

    invalid_main = np.asarray([False, False, True, False, False])
    try:
        context_eligible_count_matched_assignment(
            eligible, invalid_main, np.random.default_rng(5)
        )
    except RuntimeError as exc:
        assert "M is not a subset of E" in str(exc)
    else:
        raise AssertionError("M outside E did not fail")


def test_seed_behavior_and_overlap() -> None:
    eligible = np.ones(20, dtype=np.bool_)
    main = np.zeros(20, dtype=np.bool_)
    main[:5] = True
    first = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(27027)
    )
    repeated = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(27027)
    )
    different = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(27028)
    )
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)

    overlap_seen = False
    eligible_small = np.ones(4, dtype=np.bool_)
    main_small = np.asarray([True, True, False, False])
    for seed in range(100):
        assigned = context_eligible_count_matched_assignment(
            eligible_small,
            main_small,
            np.random.default_rng(seed),
        )
        overlap_seen |= bool(np.any(np.logical_and(main_small, assigned)))
    assert overlap_seen, "sampling appears to force exclusion of Main positions"


def test_rng_isolation() -> None:
    np.random.seed(1234)
    global_before = np.random.get_state()
    replay_rng = np.random.default_rng(11)
    exploration_rng = np.random.default_rng(12)
    assignment_rng = np.random.default_rng(13)
    replay_before = copy.deepcopy(replay_rng.bit_generator.state)
    exploration_before = copy.deepcopy(exploration_rng.bit_generator.state)
    assignment_before = copy.deepcopy(assignment_rng.bit_generator.state)

    eligible = np.ones(16, dtype=np.bool_)
    main = np.zeros(16, dtype=np.bool_)
    main[:4] = True
    context_eligible_count_matched_assignment(eligible, main, assignment_rng)

    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    assert replay_before == replay_rng.bit_generator.state
    assert exploration_before == exploration_rng.bit_generator.state
    assert assignment_before != assignment_rng.bit_generator.state


def test_diagnostic_window_and_resume() -> None:
    eligible = np.ones(8, dtype=np.bool_)
    main = np.asarray([True, True, False, False, False, False, False, False])
    assigned_a = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(21)
    )
    assigned_b = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(22)
    )
    metrics_a = diagnostic_metrics(eligible, main, assigned_a)
    metrics_b = diagnostic_metrics(eligible, main, assigned_b)

    window = AssignmentDiagnosticsWindow(window_size=2)
    assert window.update(1, metrics_a) is None
    state = copy.deepcopy(window.state_dict())
    row = window.update(2, metrics_b)
    assert row is not None
    assert row["updates"] == 2.0
    assert row["complete_window"] == 1.0
    assert row["exact_k_mismatch_count"] == 0.0
    assert row["raw_histogram_mismatch_count"] == 0.0
    assert row["normalized_histogram_mismatch_count"] == 0.0

    resumed = AssignmentDiagnosticsWindow(window_size=2)
    resumed.load_state_dict(state)
    resumed_row = resumed.update(2, metrics_b)
    assert resumed_row == row

    partial = AssignmentDiagnosticsWindow(window_size=3)
    partial.update(1, metrics_a)
    partial_row = partial.flush_partial()
    assert partial_row is not None
    assert partial_row["updates"] == 1.0
    assert partial_row["complete_window"] == 0.0


def test_assignment_rng_resume() -> None:
    eligible = np.ones(32, dtype=np.bool_)
    main = np.zeros(32, dtype=np.bool_)
    main[:8] = True
    rng = np.random.default_rng(30_001)
    for _ in range(7):
        context_eligible_count_matched_assignment(eligible, main, rng)
    checkpoint_state = copy.deepcopy(rng.bit_generator.state)
    expected = [
        context_eligible_count_matched_assignment(eligible, main, rng)
        for _ in range(5)
    ]

    restored = np.random.default_rng(1)
    restored.bit_generator.state = checkpoint_state
    observed = [
        context_eligible_count_matched_assignment(
            eligible,
            main,
            restored,
        )
        for _ in range(5)
    ]
    for expected_mask, observed_mask in zip(expected, observed):
        np.testing.assert_array_equal(expected_mask, observed_mask)


def test_config_defaults() -> None:
    cfg = LiteConfig()
    seeds = SeedConfig()
    assert cfg.support_assignment_mode == "support"
    assert cfg.support_query_mode == "current_policy"
    assert seeds.assignment_seed == 30_001

    control = LiteConfig(
        task_name="fetch_pick_and_place",
        method="td3_saber_runtime_opt",
        support_query_mode="current_policy",
        support_assignment_mode="context_eligible_count_matched",
    )
    control.normalize()
    assert control.support_assignment_mode == "context_eligible_count_matched"

    invalid = LiteConfig(
        task_name="fetch_pick_and_place",
        method="td3_saber_runtime_opt",
        support_query_mode="replay_action",
        support_assignment_mode="context_eligible_count_matched",
    )
    try:
        invalid.normalize()
    except ValueError as exc:
        assert "support_query_mode=current_policy" in str(exc)
    else:
        raise AssertionError("mixed query and assignment axes were accepted")


def main() -> int:
    test_edge_cases()
    test_seed_behavior_and_overlap()
    test_rng_isolation()
    test_diagnostic_window_and_resume()
    test_assignment_rng_resume()
    test_config_defaults()
    print("test_context_eligible_count_matched.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

