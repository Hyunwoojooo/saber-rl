#!/usr/bin/env python3
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from metacog_lite_v0.assignment_diagnostics import AssignmentDiagnosticsWindow
from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.trainer import restore_assignment_checkpoint_state
from metacog_lite_v0.weights import (
    GapStats,
    LiteWeightComputer,
    context_eligible_count_matched_assignment,
    sample_exact_k_from_eligible,
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
    eligible: np.ndarray,
    main: np.ndarray,
    assigned: np.ndarray,
) -> dict[str, float]:
    main_raw = np.ones(main.size, dtype=np.float32)
    assigned_raw = np.ones(main.size, dtype=np.float32)
    main_raw[main] = 2.0
    assigned_raw[assigned] = 2.0
    main_norm = _normalized(main_raw)
    assigned_norm = _normalized(assigned_raw)
    overlap = int(np.sum(np.logical_and(main, assigned)))
    union = int(np.sum(np.logical_or(main, assigned)))
    k = int(np.sum(main))
    return {
        "assignment/batch_size": float(main.size),
        "assignment/context_eligible_count": float(np.sum(eligible)),
        "assignment/assignment_pool_count": float(np.sum(eligible)),
        "assignment/assignment_pool_rate": float(np.mean(eligible)),
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
        "assignment/reference_raw_weight_ESS": _ess(main_raw),
        "assignment/assigned_raw_weight_mean": float(np.mean(assigned_raw)),
        "assignment/assigned_raw_weight_std": float(np.std(assigned_raw)),
        "assignment/assigned_raw_weight_ESS": _ess(assigned_raw),
        "assignment/reference_normalized_weight_mean": float(np.mean(main_norm)),
        "assignment/reference_normalized_weight_std": float(np.std(main_norm)),
        "assignment/reference_normalized_weight_ESS": _ess(main_norm),
        "assignment/assigned_normalized_weight_mean": float(np.mean(assigned_norm)),
        "assignment/assigned_normalized_weight_std": float(np.std(assigned_norm)),
        "assignment/assigned_normalized_weight_ESS": _ess(assigned_norm),
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


def test_edge_cases_and_invalid_masks() -> None:
    eligible = np.asarray([True, True, False, True, False])
    empty_main = np.zeros(5, dtype=np.bool_)
    empty_rng = np.random.default_rng(1)
    empty_before = _rng_state(empty_rng)
    empty = context_eligible_count_matched_assignment(
        eligible, empty_main, empty_rng
    )
    np.testing.assert_array_equal(empty, np.zeros(5, dtype=np.bool_))
    assert _rng_state(empty_rng) == empty_before

    full_rng = np.random.default_rng(2)
    full_before = _rng_state(full_rng)
    full = context_eligible_count_matched_assignment(
        eligible, eligible, full_rng
    )
    np.testing.assert_array_equal(full, eligible)
    assert _rng_state(full_rng) == full_before

    partial_main = np.asarray([True, False, False, True, False])
    partial = context_eligible_count_matched_assignment(
        eligible, partial_main, np.random.default_rng(3)
    )
    assert int(np.sum(partial)) == int(np.sum(partial_main))
    assert np.all(np.logical_not(partial) | eligible)
    assert np.unique(np.flatnonzero(partial)).size == int(np.sum(partial))

    failures = [
        lambda: sample_exact_k_from_eligible(
            eligible, -1, np.random.default_rng(4)
        ),
        lambda: sample_exact_k_from_eligible(
            eligible, 4, np.random.default_rng(4)
        ),
        lambda: sample_exact_k_from_eligible(
            np.ones((2, 2), dtype=np.bool_), 1, np.random.default_rng(4)
        ),
        lambda: context_eligible_count_matched_assignment(
            eligible,
            np.asarray([False, False, True, False, False]),
            np.random.default_rng(5),
        ),
        lambda: context_eligible_count_matched_assignment(
            eligible,
            np.ones(4, dtype=np.bool_),
            np.random.default_rng(5),
        ),
        lambda: context_eligible_count_matched_assignment(
            eligible.reshape(1, -1),
            eligible.reshape(1, -1),
            np.random.default_rng(5),
        ),
    ]
    for failure in failures:
        try:
            failure()
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid strict-assignment input did not fail")


def test_distribution_identity_and_rng_isolation() -> None:
    eligible = np.ones(20, dtype=np.bool_)
    main_a = np.zeros(20, dtype=np.bool_)
    main_b = np.zeros(20, dtype=np.bool_)
    main_a[:5] = True
    main_b[-5:] = True
    first = context_eligible_count_matched_assignment(
        eligible, main_a, np.random.default_rng(27027)
    )
    repeated = context_eligible_count_matched_assignment(
        eligible, main_a, np.random.default_rng(27027)
    )
    different = context_eligible_count_matched_assignment(
        eligible, main_a, np.random.default_rng(27028)
    )
    identity_independent = context_eligible_count_matched_assignment(
        eligible, main_b, np.random.default_rng(27027)
    )
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, identity_independent)
    assert not np.array_equal(first, different)
    assert np.any(np.logical_and(first, main_a)), "Main positions were forcibly excluded"

    np.random.seed(1234)
    global_before = copy.deepcopy(np.random.get_state())
    torch.manual_seed(1234)
    torch_before = torch.get_rng_state().clone()
    replay_rng = np.random.default_rng(11)
    exploration_rng = np.random.default_rng(12)
    weight_rng = np.random.default_rng(13)
    assignment_rng = np.random.default_rng(14)
    protected = [replay_rng, exploration_rng, weight_rng]
    protected_before = [_rng_state(rng) for rng in protected]
    assignment_before = _rng_state(assignment_rng)
    context_eligible_count_matched_assignment(
        eligible, main_a, assignment_rng
    )
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before)
    assert [_rng_state(rng) for rng in protected] == protected_before
    assert _rng_state(assignment_rng) != assignment_before


def test_rng_and_diagnostic_checkpoint_resume() -> None:
    eligible = np.ones(32, dtype=np.bool_)
    main = np.zeros(32, dtype=np.bool_)
    main[:8] = True
    rng = np.random.default_rng(30_001)
    window = AssignmentDiagnosticsWindow(window_size=10)
    for update in range(1, 8):
        assigned = context_eligible_count_matched_assignment(eligible, main, rng)
        assert window.update(update, _diagnostic_metrics(eligible, main, assigned)) is None
    rng_state = copy.deepcopy(rng.bit_generator.state)
    window_state = copy.deepcopy(window.state_dict())

    expected_masks = []
    expected_rows = []
    for update in range(8, 15):
        assigned = context_eligible_count_matched_assignment(eligible, main, rng)
        expected_masks.append(assigned)
        row = window.update(update, _diagnostic_metrics(eligible, main, assigned))
        if row is not None:
            expected_rows.append(row)

    restored_rng = np.random.default_rng(1)
    restored_rng.bit_generator.state = rng_state
    restored_window = AssignmentDiagnosticsWindow(window_size=10)
    restored_window.load_state_dict(window_state)
    observed_rows = []
    for index, update in enumerate(range(8, 15)):
        assigned = context_eligible_count_matched_assignment(
            eligible, main, restored_rng
        )
        np.testing.assert_array_equal(assigned, expected_masks[index])
        row = restored_window.update(
            update, _diagnostic_metrics(eligible, main, assigned)
        )
        if row is not None:
            observed_rows.append(row)
    assert observed_rows == expected_rows
    assert restored_window.state_dict() == window.state_dict()

    partial = restored_window.flush_partial()
    assert partial is not None and partial["complete_window"] == 0.0
    empty_state = restored_window.state_dict()
    assert empty_state["count"] == 0
    assert empty_state["minimums"] == {}
    assert empty_state["maximums"] == {}
    for value in empty_state["sums"].values():
        assert np.isfinite(float(value))

    # Empty-window checkpoint state reconstructs the internal extrema
    # sentinels and still produces correct first-update min/max values.
    empty_restored = AssignmentDiagnosticsWindow(window_size=10)
    empty_restored.load_state_dict(copy.deepcopy(empty_state))
    first_assigned = context_eligible_count_matched_assignment(
        eligible, main, restored_rng
    )
    assert empty_restored.update(
        15, _diagnostic_metrics(eligible, main, first_assigned)
    ) is None
    post_update_state = empty_restored.state_dict()
    assert all(
        np.isfinite(float(value))
        for value in post_update_state["minimums"].values()
    )
    assert all(
        np.isfinite(float(value))
        for value in post_update_state["maximums"].values()
    )


def test_strict_checkpoint_restore_contract() -> None:
    strict = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        support_assignment_mode="context_eligible_count_matched",
        weight_scope="critic_only",
        weighting_mode="batch_norm",
    )
    strict.normalize()
    default = LiteConfig()
    target_rng = np.random.default_rng(1)
    target_window = AssignmentDiagnosticsWindow(window_size=3)

    restore_assignment_checkpoint_state(default, {}, target_rng, target_window)
    for missing_payload, expected_key in (
        ({}, "assignment_rng_state"),
        (
            {
                "assignment_rng_state": copy.deepcopy(
                    np.random.default_rng(30_001).bit_generator.state
                )
            },
            "assignment_diagnostics_state",
        ),
    ):
        try:
            restore_assignment_checkpoint_state(
                strict, missing_payload, target_rng, target_window
            )
        except RuntimeError as exc:
            assert expected_key in str(exc)
        else:
            raise AssertionError(f"strict resume accepted missing {expected_key}")

    source_rng = np.random.default_rng(30_001)
    source_rng.choice(np.arange(10), size=3, replace=False)
    source_window = AssignmentDiagnosticsWindow(window_size=3)
    eligible = np.ones(8, dtype=np.bool_)
    main = np.asarray([True, True, False, False, False, False, False, False])
    assigned = context_eligible_count_matched_assignment(
        eligible, main, source_rng
    )
    source_window.update(1, _diagnostic_metrics(eligible, main, assigned))
    payload = {
        "assignment_rng_state": copy.deepcopy(source_rng.bit_generator.state),
        "assignment_diagnostics_state": copy.deepcopy(source_window.state_dict()),
    }
    restored_rng = np.random.default_rng(9)
    restored_window = AssignmentDiagnosticsWindow(window_size=3)
    restore_assignment_checkpoint_state(
        strict, payload, restored_rng, restored_window
    )
    assert _rng_state(restored_rng) == _rng_state(source_rng)
    assert restored_window.state_dict() == source_window.state_dict()

    poisoned = copy.deepcopy(payload)
    poisoned["assignment_diagnostics_state"]["sums"][
        "context_eligible_count"
    ] = float("inf")
    try:
        restore_assignment_checkpoint_state(
            strict,
            poisoned,
            np.random.default_rng(9),
            AssignmentDiagnosticsWindow(window_size=3),
        )
    except RuntimeError as exc:
        assert "non-finite" in str(exc) and "checkpoint state" in str(exc)
    else:
        raise AssertionError("strict resume accepted non-finite diagnostic state")


def _strict_compute(cache_enabled: bool):
    task = make_task()
    cfg = make_config("current_policy", cache_enabled)
    cfg.support_assignment_mode = "context_eligible_count_matched"
    cfg.assignment_diagnostic_window_updates = 1_000
    replay = make_replay(task)
    batch = make_batch()
    agent = VectorAgent(np.stack([ACTION_A, ACTION_A, ACTION_B, ACTION_B]))
    weight_rng = np.random.default_rng(20_001)
    assignment_rng = np.random.default_rng(30_001)
    weight_before = _rng_state(weight_rng)
    assignment_before = _rng_state(assignment_rng)
    wb = LiteWeightComputer(
        cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(
        batch,
        replay,
        agent,
        weight_rng,
        torch.device("cpu"),
        assignment_rng=assignment_rng,
    )
    return wb, weight_before, _rng_state(weight_rng), assignment_before, _rng_state(assignment_rng)


def test_weight_computer_integration() -> None:
    cached, weight_before, weight_after, assignment_before, assignment_after = _strict_compute(True)
    uncached, *_ = _strict_compute(False)
    for field in (
        "psi_q_rank",
        "psi_a_policy",
        "w_p_raw",
        "omega_critic",
        "omega_actor",
        "len_a_q_local",
        "len_a_q_sup",
    ):
        np.testing.assert_array_equal(getattr(cached, field), getattr(uncached, field))

    main = cached.psi_q_rank > 0.5
    assigned = cached.w_p_raw > 1.5
    np.testing.assert_array_equal(main, np.asarray([True, True, False, False]))
    assert cached.extra_metrics["assignment/context_eligible_count"] == 4.0
    assert int(np.sum(assigned)) == int(np.sum(main)) == 2
    np.testing.assert_array_equal(np.sort(cached.w_p_raw), np.asarray([1.0, 1.0, 2.0, 2.0]))
    np.testing.assert_allclose(
        np.mean(cached.omega_critic, dtype=np.float64), 1.0, rtol=0.0, atol=1e-7
    )
    assert np.all(cached.omega_critic > 0.0)
    np.testing.assert_array_equal(cached.omega_actor, np.ones(4, dtype=np.float32))
    assert cached.extra_metrics["assignment/raw_histogram_mismatch_count"] == 0.0
    assert cached.extra_metrics["assignment/normalized_histogram_mismatch_count"] == 0.0
    assert cached.extra_metrics["assignment/invariant_failure_count"] == 0.0
    assert cached.extra_metrics["assignment/main_positive_count"] == 2.0
    assert cached.extra_metrics["assignment/assigned_positive_count"] == 2.0
    assert weight_before == weight_after
    assert assignment_before != assignment_after


def test_counterfactual_context_and_membership_semantics() -> None:
    task = make_task()
    cfg = make_config("current_policy", cache_enabled=True)
    cfg.support_assignment_mode = "context_eligible_count_matched"
    replay = make_replay(task)
    batch = make_batch()
    # Rows 2-3 have a different object-state context with no local replay
    # population. Rows 0-1 remain context-eligible, but only row 0 queries the
    # occupied action bin.
    batch.observation[2:, 3:6] = np.asarray(
        [0.8, 0.8, 0.8], dtype=np.float32
    )
    agent = VectorAgent(np.stack([ACTION_A, ACTION_B, ACTION_A, ACTION_A]))
    result = LiteWeightComputer(
        cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    ).compute(
        batch,
        replay,
        agent,
        np.random.default_rng(20_001),
        torch.device("cpu"),
        assignment_rng=np.random.default_rng(30_001),
    )
    eligible = np.logical_and(
        result.len_a_q_local >= cfg.N_local_min,
        np.minimum(result.len_a_q_local, cfg.M_B) >= cfg.M_Q_min,
    )
    main = result.psi_q_rank > 0.5
    assigned = result.w_p_raw > 1.5
    np.testing.assert_array_equal(
        eligible, np.asarray([True, True, False, False])
    )
    np.testing.assert_array_equal(
        main, np.asarray([True, False, False, False])
    )
    assert np.all(np.logical_not(main) | eligible)
    assert np.all(np.logical_not(assigned) | eligible)
    assert int(np.sum(assigned)) == int(np.sum(main)) == 1


def test_config_contract() -> None:
    cfg = LiteConfig()
    seeds = SeedConfig()
    assert cfg.support_query_mode == "current_policy"
    assert cfg.support_assignment_mode == "support"
    assert seeds.assignment_seed == 30_001

    control = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        support_query_mode="current_policy",
        support_assignment_mode="context_eligible_count_matched",
        weight_scope="critic_only",
        weighting_mode="batch_norm",
    )
    control.normalize()

    invalid_configs = [
        LiteConfig(
            task_name="fetch_pick_and_place",
            method="b3_support_runtime_opt",
            support_query_mode="replay_action",
            support_assignment_mode="context_eligible_count_matched",
            weight_scope="critic_only",
        ),
        LiteConfig(
            task_name="fetch_pick_and_place",
            method="b4_lite",
            support_query_mode="current_policy",
            support_assignment_mode="context_eligible_count_matched",
            weight_scope="critic_only",
        ),
        LiteConfig(
            task_name="fetch_pick_and_place",
            method="b3_support_runtime_opt",
            support_assignment_mode="context_eligible_count_matched",
            weight_scope="actor_critic",
        ),
        LiteConfig(
            task_name="fetch_pick_and_place",
            method="b3_support_runtime_opt",
            support_assignment_mode="context_eligible_count_matched",
            weight_scope="critic_only",
            weighting_mode="raw",
        ),
        LiteConfig(support_assignment_mode="random_fixed_count"),
    ]
    for invalid in invalid_configs:
        try:
            invalid.normalize()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid strict assignment configuration was accepted")


def main() -> int:
    test_edge_cases_and_invalid_masks()
    test_distribution_identity_and_rng_isolation()
    test_rng_and_diagnostic_checkpoint_resume()
    test_strict_checkpoint_restore_contract()
    test_weight_computer_integration()
    test_counterfactual_context_and_membership_semantics()
    test_config_contract()
    print("test_context_eligible_count_matched.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
