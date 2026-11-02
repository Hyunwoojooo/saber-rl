#!/usr/bin/env python3
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.assignment_diagnostics import StrictAssignmentDiagnosticsWindow
from metacog_lite_v0.replay import SampleBatch
from metacog_lite_v0.weights import (
    GapStats,
    LiteWeightComputer,
    LocalContextResult,
    context_eligible_count_matched_assignment,
    sample_exact_k_from_eligible,
)


def normalized(raw: np.ndarray) -> np.ndarray:
    return (raw / float(np.mean(raw))).astype(np.float32)


def strict_metrics(
    eligible: np.ndarray,
    main: np.ndarray,
    control: np.ndarray,
) -> dict[str, float]:
    raw = np.ones(main.size, dtype=np.float32)
    raw[control] = 2.0
    omega = normalized(raw)
    overlap = int(np.sum(np.logical_and(main, control)))
    k = int(np.sum(main))
    return {
        "strict/context_eligible_count": float(np.sum(eligible)),
        "strict/main_positive_count": float(k),
        "strict/control_positive_count": float(np.sum(control)),
        "strict/context_eligible_rate": float(np.mean(eligible)),
        "strict/main_positive_rate": float(np.mean(main)),
        "strict/control_positive_rate": float(np.mean(control)),
        "strict/main_control_overlap_count": float(overlap),
        "strict/main_control_overlap_rate": float(overlap / k) if k else 0.0,
        "strict/control_selected_outside_eligible": 0.0,
        "strict/raw_weight_mean": float(np.mean(raw)),
        "strict/normalized_weight_mean": float(np.mean(omega)),
        "strict/normalized_weight_min": float(np.min(omega)),
        "strict/normalized_weight_max": float(np.max(omega)),
        "strict/count_equality_violation": 0.0,
        "strict/raw_histogram_mismatch": 0.0,
        "strict/fallback_sampling_used": 0.0,
        "strict/replacement_sampling_used": 0.0,
        "runtime/support_lookup_sec_per_update": 0.001,
        "runtime/strict_assignment_sec_per_update": 0.0001,
    }


def assert_valid_assignment(
    eligible: np.ndarray,
    main: np.ndarray,
    control: np.ndarray,
) -> None:
    assert np.all(np.logical_not(main) | eligible)
    assert np.all(np.logical_not(control) | eligible)
    assert int(np.sum(control)) == int(np.sum(main))
    main_raw = np.ones(main.size, dtype=np.float32)
    control_raw = np.ones(main.size, dtype=np.float32)
    main_raw[main] = 2.0
    control_raw[control] = 2.0
    np.testing.assert_array_equal(np.sort(main_raw), np.sort(control_raw))
    omega = normalized(control_raw)
    np.testing.assert_allclose(np.mean(omega), 1.0, rtol=0.0, atol=1e-7)
    assert np.all(omega > 0.0)


def test_edges_and_subset_invariants() -> None:
    eligible = np.asarray([True, True, False, True, False])

    k_zero = context_eligible_count_matched_assignment(
        eligible,
        np.zeros(eligible.size, dtype=np.bool_),
        np.random.default_rng(1),
    )
    np.testing.assert_array_equal(k_zero, np.zeros_like(eligible))

    k_one_main = np.asarray([True, False, False, False, False])
    k_one = context_eligible_count_matched_assignment(
        eligible, k_one_main, np.random.default_rng(2)
    )
    assert_valid_assignment(eligible, k_one_main, k_one)

    k_full = context_eligible_count_matched_assignment(
        eligible, eligible, np.random.default_rng(3)
    )
    np.testing.assert_array_equal(k_full, eligible)

    invalid_main = np.asarray([False, False, True, False, False])
    try:
        context_eligible_count_matched_assignment(
            eligible, invalid_main, np.random.default_rng(4)
        )
    except RuntimeError as exc:
        assert "M is not a subset of E" in str(exc)
    else:
        raise AssertionError("Main positive outside E was accepted")

    try:
        sample_exact_k_from_eligible(eligible, 4, np.random.default_rng(5))
    except RuntimeError as exc:
        assert "exceeds |E|" in str(exc)
    else:
        raise AssertionError("k > |E| was accepted")


def test_uniform_without_replacement_call_contract() -> None:
    class ChoiceSpy:
        def __init__(self) -> None:
            self.called = False

        def choice(
            self,
            population: np.ndarray,
            size: int,
            replace: bool,
        ) -> np.ndarray:
            self.called = True
            np.testing.assert_array_equal(population, np.asarray([0, 2, 3, 5]))
            assert size == 2
            assert replace is False
            return np.asarray([2, 5])

    spy = ChoiceSpy()
    sampled = sample_exact_k_from_eligible(
        np.asarray([True, False, True, True, False, True]),
        2,
        spy,  # type: ignore[arg-type]
    )
    assert spy.called
    np.testing.assert_array_equal(
        sampled,
        np.asarray([False, False, True, False, False, True]),
    )


def test_natural_overlap_is_allowed() -> None:
    eligible = np.ones(4, dtype=np.bool_)
    main = np.asarray([True, True, False, False])
    overlaps = []
    for seed in range(100):
        control = context_eligible_count_matched_assignment(
            eligible, main, np.random.default_rng(seed)
        )
        assert_valid_assignment(eligible, main, control)
        overlaps.append(int(np.sum(np.logical_and(main, control))))
    assert max(overlaps) > 0
    assert min(overlaps) < int(np.sum(main))


def test_dedicated_rng_isolation_and_resume() -> None:
    np.random.seed(1_234)
    global_before = np.random.get_state()
    torch.manual_seed(4_321)
    torch_before = torch.random.get_rng_state().clone()
    replay_rng = np.random.default_rng(11)
    action_noise_rng = np.random.default_rng(12)
    evaluation_rng = np.random.default_rng(13)
    assignment_rng = np.random.default_rng(30_001)
    replay_before = copy.deepcopy(replay_rng.bit_generator.state)
    action_before = copy.deepcopy(action_noise_rng.bit_generator.state)
    evaluation_before = copy.deepcopy(evaluation_rng.bit_generator.state)

    eligible = np.ones(32, dtype=np.bool_)
    main = np.zeros(32, dtype=np.bool_)
    main[:8] = True
    for _ in range(7):
        context_eligible_count_matched_assignment(eligible, main, assignment_rng)
    checkpoint_state = copy.deepcopy(assignment_rng.bit_generator.state)
    expected = [
        context_eligible_count_matched_assignment(eligible, main, assignment_rng)
        for _ in range(5)
    ]

    restored = np.random.default_rng(1)
    restored.bit_generator.state = checkpoint_state
    observed = [
        context_eligible_count_matched_assignment(eligible, main, restored)
        for _ in range(5)
    ]
    for expected_mask, observed_mask in zip(expected, observed):
        np.testing.assert_array_equal(expected_mask, observed_mask)

    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert replay_before == replay_rng.bit_generator.state
    assert action_before == action_noise_rng.bit_generator.state
    assert evaluation_before == evaluation_rng.bit_generator.state


def test_diagnostic_window_checkpoint_resume() -> None:
    eligible = np.ones(8, dtype=np.bool_)
    main = np.asarray([True, True, False, False, False, False, False, False])
    first = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(21)
    )
    second = context_eligible_count_matched_assignment(
        eligible, main, np.random.default_rng(22)
    )
    metrics_first = strict_metrics(eligible, main, first)
    metrics_second = strict_metrics(eligible, main, second)

    window = StrictAssignmentDiagnosticsWindow(window_size=2)
    assert window.update(1, metrics_first) is None
    checkpoint = copy.deepcopy(window.state_dict())
    expected = window.update(2, metrics_second)
    assert expected is not None
    assert expected["count_equality_violation_sum"] == 0.0
    assert expected["control_selected_outside_eligible_sum"] == 0.0
    assert expected["replacement_sampling_used_sum"] == 0.0
    assert expected["normalized_weight_mean_min"] == 1.0
    assert expected["normalized_weight_mean_max"] == 1.0
    assert expected["normalized_weight_min_min"] > 0.0
    assert expected["normalized_weight_max_max"] > 0.0
    assert expected["normalized_weight_mean_abs_error_max"] == 0.0

    restored = StrictAssignmentDiagnosticsWindow(window_size=2)
    restored.load_state_dict(checkpoint)
    observed = restored.update(2, metrics_second)
    assert observed == expected


class StubWeightComputer(LiteWeightComputer):
    def __init__(self, cfg: SimpleNamespace, contexts: list[LocalContextResult]) -> None:
        self.cfg = cfg
        self.task = SimpleNamespace()
        self.gap_stats = GapStats(8, 128, 0.1, 0.9, 1e-6)
        self.contexts = contexts
        self.support_query_actions = None

    def _prepare_runtime_opt_contexts(self, batch, replay, a_policy):
        self.support_query_actions = a_policy.copy()
        assert not np.array_equal(a_policy, batch.action_exec)
        return (
            np.zeros(batch.observation.shape[0], dtype=np.int64),
            self.contexts,
            {},
            0.001,
        )


class StubAgent:
    def policy_actions(self, states: torch.Tensor) -> torch.Tensor:
        return torch.zeros((states.shape[0], 2), dtype=torch.float32, device=states.device)


def test_production_weight_path_is_critic_only_and_normalized() -> None:
    b = 6
    cfg = SimpleNamespace(
        method="b3_support_runtime_opt",
        M_B=16,
        M_Q_min=4,
        N_local_min=4,
        lambda_support=1.0,
        support_assignment_mode="context_eligible_count_matched",
        weighting_mode="batch_norm",
        weight_scope="critic_only",
        eps=1e-6,
    )
    contexts = []
    for i in range(b):
        eligible = i < 5
        contexts.append(
            LocalContextResult(
                indices=np.zeros(0, dtype=np.int64),
                action_bins=np.zeros(0, dtype=np.int64),
                num_coarse_candidates=8,
                num_scanned_candidates=8,
                local_scan_truncated=False,
                local_pool_size=4 if eligible else 3,
                available_support_size=4 if eligible else 3,
                psi_a_policy=1.0 if i < 2 else 0.0,
                context_count_eligible=float(eligible),
            )
        )
    computer = StubWeightComputer(cfg, contexts)
    zeros_obs = np.zeros((b, 3), dtype=np.float32)
    zeros_goal = np.zeros((b, 2), dtype=np.float32)
    batch = SampleBatch(
        indices=np.arange(b, dtype=np.int64),
        observation=zeros_obs,
        achieved_goal=zeros_goal,
        desired_goal_train=zeros_goal,
        next_observation=zeros_obs,
        next_achieved_goal=zeros_goal,
        action_exec=np.ones((b, 2), dtype=np.float32),
        reward_ext=np.zeros(b, dtype=np.float32),
        done=np.zeros(b, dtype=np.float32),
        timeout=np.zeros(b, dtype=np.float32),
        h=np.ones(b, dtype=np.int64),
        her_relabel_ratio=0.8,
    )
    replay = SimpleNamespace(action_dim=2)
    result = computer.compute(
        batch=batch,
        replay=replay,
        agent=StubAgent(),
        rng=np.random.default_rng(7),
        device=torch.device("cpu"),
        assignment_rng=np.random.default_rng(30_001),
    )
    assert result.extra_metrics["strict/context_eligible_count"] == 5.0
    assert result.extra_metrics["strict/main_positive_count"] == 2.0
    assert result.extra_metrics["strict/control_positive_count"] == 2.0
    assert result.extra_metrics["strict/control_selected_outside_eligible"] == 0.0
    np.testing.assert_array_equal(
        computer.support_query_actions,
        np.zeros((b, 2), dtype=np.float32),
    )
    np.testing.assert_allclose(np.mean(result.omega_critic), 1.0, atol=1e-7)
    assert np.all(result.omega_critic > 0.0)
    np.testing.assert_array_equal(result.omega_actor, np.ones(b, dtype=np.float32))

    main_cfg = SimpleNamespace(**vars(cfg))
    main_cfg.support_assignment_mode = "support"
    main_computer = StubWeightComputer(main_cfg, contexts)
    main_result = main_computer.compute(
        batch=batch,
        replay=replay,
        agent=StubAgent(),
        rng=np.random.default_rng(7),
        device=torch.device("cpu"),
        assignment_rng=None,
    )
    expected_raw = np.asarray([2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    expected_omega = normalized(expected_raw)
    np.testing.assert_array_equal(main_result.w_p_raw, expected_raw)
    np.testing.assert_array_equal(main_result.psi_q_rank, expected_raw - 1.0)
    np.testing.assert_array_equal(main_result.omega_critic, expected_omega)
    np.testing.assert_array_equal(main_result.omega_actor, np.ones(b, dtype=np.float32))
    assert not any(key.startswith("strict/") for key in main_result.extra_metrics)


def main() -> int:
    test_edges_and_subset_invariants()
    test_uniform_without_replacement_call_contract()
    test_natural_overlap_is_allowed()
    test_dedicated_rng_isolation_and_resume()
    test_diagnostic_window_checkpoint_resume()
    test_production_weight_path_is_critic_only_and_normalized()
    print("test_strict_context_eligible_count_matched.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
