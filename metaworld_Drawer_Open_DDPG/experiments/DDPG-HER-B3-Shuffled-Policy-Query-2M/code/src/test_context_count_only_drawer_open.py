#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.trainer import _restore_training_log_state, _resume_global_step
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


ACTION_REPLAY = np.full(4, -0.9, dtype=np.float32)
ACTION_QUERY_A = np.full(4, 0.9, dtype=np.float32)
ACTION_QUERY_B = np.asarray([0.9, -0.9, 0.9, -0.9], dtype=np.float32)


class VectorAgent:
    def __init__(self, action: np.ndarray, batch_size: int) -> None:
        self.actions = torch.as_tensor(np.repeat(action[None, :], batch_size, axis=0))

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.actions.to(state.device)


def task() -> MetaWorldDrawerOpenTaskSpec:
    return MetaWorldDrawerOpenTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )


def observation(handle_x: float = 0.25) -> np.ndarray:
    obs = np.zeros(7, dtype=np.float32)
    # Keep hand-handle distance fixed so every synthetic transition shares the
    # same coarse bucket while handle_x can change the relabeled-goal context.
    obs[0:3] = [handle_x - 0.25, 0.1, 0.0]
    obs[3] = 0.2
    obs[4:7] = [handle_x, 0.1, 0.0]
    return obs


def config(mode: str, cache: bool) -> LiteConfig:
    cfg = LiteConfig(
        task_name="drawer_open", method="b3_support_runtime_opt", batch_size=4,
        M_B=16, M_Q_min=4, N_local_min=4, local_search_max=512,
        action_bins_per_dim=5, lambda_support=1.0, policy_support_mode="bin",
        support_query_mode=mode, support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=cache, local_scan_cache_enabled=cache,
        weighting_mode="batch_norm", weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.support_diagnostic_audit_interval = 0
    return cfg


def fixture(local_count: int, mode: str, cache: bool, action: np.ndarray):
    spec = task()
    obs = observation()
    goal = np.asarray([0.6, 0.1, 0.0], dtype=np.float32)
    replay = HERReplayBuffer(capacity=32, obs_dim=7, goal_dim=3, action_dim=4)
    action_bin = int(spec.action_bin_batch(ACTION_REPLAY[None, :], 5)[0])
    for row in range(local_count):
        replay.add_transition(
            observation=obs, achieved_goal=goal, desired_goal=goal,
            action_exec=ACTION_REPLAY, reward_ext=0.0,
            next_observation=obs, next_achieved_goal=goal,
            done=False, timeout=False, h=20, episode_id=row, episode_step=0,
            coarse_key=spec.coarse_hash(obs, 20), action_bin=action_bin,
        )
    obs_batch = np.repeat(obs[None, :], 4, axis=0)
    goals = np.repeat(goal[None, :], 4, axis=0)
    batch = SampleBatch(
        indices=np.arange(4), observation=obs_batch, achieved_goal=goals,
        desired_goal_train=goals, next_observation=obs_batch.copy(),
        next_achieved_goal=goals.copy(), action_exec=np.repeat(ACTION_REPLAY[None, :], 4, axis=0),
        reward_ext=np.zeros(4, dtype=np.float32), done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_), h=np.full(4, 20, dtype=np.int16),
        her_relabel_ratio=1.0,
    )
    rng = np.random.default_rng(27027)
    before = rng.bit_generator.state.copy()
    wb = LiteWeightComputer(config(mode, cache), spec, GapStats(8, 128, 0.1, 0.9, 1e-6)).compute(
        batch, replay, VectorAgent(action, 4), rng, torch.device("cpu")
    )
    return wb, before, rng.bit_generator.state


def mixed_fixture(cache: bool, action: np.ndarray):
    """Return two eligible and two ineligible queries from one coarse bucket.

    With goal A, candidate handle x values 0.21, 0.24, 0.32, and 0.38
    share the query's exact full context (four matches).  With goal B, only
    0.15, 0.21, and 0.24 match (three matches).  Hand-handle distance,
    handle-y, gripper, and horizon are fixed, so all candidates and queries
    deliberately share one coarse replay bucket.
    """
    spec = task()
    query_obs = observation()
    goal_a = np.asarray([0.6, 0.1, 0.0], dtype=np.float32)
    goal_b = np.asarray([-0.1, 0.1, 0.0], dtype=np.float32)
    candidate_handle_x = (0.15, 0.21, 0.24, 0.32, 0.38)
    replay = HERReplayBuffer(capacity=32, obs_dim=7, goal_dim=3, action_dim=4)
    action_bin = int(spec.action_bin_batch(ACTION_REPLAY[None, :], 5)[0])
    coarse_key = spec.coarse_hash(query_obs, 20)
    candidate_observations = []
    for row, handle_x in enumerate(candidate_handle_x):
        candidate_obs = observation(handle_x)
        assert spec.coarse_hash(candidate_obs, 20) == coarse_key
        candidate_observations.append(candidate_obs)
        replay.add_transition(
            observation=candidate_obs, achieved_goal=goal_a, desired_goal=goal_a,
            action_exec=ACTION_REPLAY, reward_ext=0.0,
            next_observation=candidate_obs, next_achieved_goal=goal_a,
            done=False, timeout=False, h=20, episode_id=row, episode_step=0,
            coarse_key=coarse_key, action_bin=action_bin,
        )

    candidate_obs_batch = np.stack(candidate_observations).astype(np.float32)
    candidate_h = np.full(len(candidate_observations), 20, dtype=np.int16)
    for goal, expected_count in ((goal_a, 4), (goal_b, 3)):
        query_full_key = spec.full_context_hash(query_obs, goal, 20)
        candidate_full_keys = spec.full_context_hash_batch(
            candidate_obs_batch,
            np.repeat(goal[None, :], len(candidate_observations), axis=0),
            candidate_h,
        )
        assert int(np.sum(candidate_full_keys == query_full_key)) == expected_count

    obs_batch = np.repeat(query_obs[None, :], 4, axis=0)
    goals = np.stack([goal_a, goal_a, goal_b, goal_b]).astype(np.float32)
    batch = SampleBatch(
        indices=np.arange(4), observation=obs_batch,
        achieved_goal=goals.copy(), desired_goal_train=goals,
        next_observation=obs_batch.copy(), next_achieved_goal=goals.copy(),
        action_exec=np.repeat(ACTION_REPLAY[None, :], 4, axis=0),
        reward_ext=np.zeros(4, dtype=np.float32), done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_), h=np.full(4, 20, dtype=np.int16),
        her_relabel_ratio=1.0,
    )
    rng = np.random.default_rng(27027)
    before = rng.bit_generator.state.copy()
    wb = LiteWeightComputer(
        config("context_count_only", cache), spec, GapStats(8, 128, 0.1, 0.9, 1e-6)
    ).compute(batch, replay, VectorAgent(action, 4), rng, torch.device("cpu"))
    return wb, before, rng.bit_generator.state


def test_context_only() -> None:
    a, before, after = fixture(4, "context_count_only", True, ACTION_QUERY_A)
    b, _, _ = fixture(4, "context_count_only", True, ACTION_QUERY_B)
    uncached, _, _ = fixture(4, "context_count_only", False, ACTION_QUERY_A)
    for field in ("psi_q_rank", "w_p_raw", "omega_critic", "omega_actor"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        np.testing.assert_array_equal(getattr(a, field), getattr(uncached, field))
    np.testing.assert_array_equal(a.psi_q_rank, np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(a.w_p_raw, np.full(4, 2.0, dtype=np.float32))
    np.testing.assert_allclose(a.omega_critic, 1.0)
    np.testing.assert_array_equal(a.omega_actor, np.ones(4, dtype=np.float32))
    assert a.extra_metrics["query/context_count_eligible_rate"] == 1.0
    assert a.extra_metrics["query/support_positive_rate"] == 1.0
    assert a.extra_metrics["query/action_membership_lookup_count"] == 0.0
    assert a.extra_metrics["query/invariant_violation_count"] == 0.0
    assert np.isnan(a.extra_metrics["query/query_bin_present_given_context_rate"])
    assert before == after, "context-only consumed weight RNG"

    ineligible, _, _ = fixture(3, "context_count_only", True, ACTION_QUERY_A)
    np.testing.assert_array_equal(ineligible.psi_q_rank, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(ineligible.w_p_raw, np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(ineligible.omega_critic, np.ones(4, dtype=np.float32))


def test_context_only_mixed_minibatch() -> None:
    mixed, before, after = mixed_fixture(True, ACTION_QUERY_A)
    mixed_other_action, _, _ = mixed_fixture(True, ACTION_QUERY_B)
    mixed_uncached, _, _ = mixed_fixture(False, ACTION_QUERY_A)

    expected_mask = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    expected_raw = np.asarray([2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    expected_normalized = expected_raw / np.mean(expected_raw)
    for field in ("psi_q_rank", "w_p_raw", "omega_critic", "omega_actor"):
        np.testing.assert_array_equal(getattr(mixed, field), getattr(mixed_other_action, field))
        np.testing.assert_array_equal(getattr(mixed, field), getattr(mixed_uncached, field))

    np.testing.assert_array_equal(mixed.psi_q_rank, expected_mask)
    np.testing.assert_array_equal(mixed.w_p_raw, expected_raw)
    np.testing.assert_allclose(mixed.omega_critic, expected_normalized, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(np.mean(mixed.omega_critic), 1.0, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(mixed.omega_actor, np.ones(4, dtype=np.float32))
    assert mixed.extra_metrics["query/context_count_eligible_rate"] == 0.5
    assert mixed.extra_metrics["query/support_positive_rate"] == 0.5
    assert mixed.extra_metrics["query/high_weight_count"] == 2.0
    assert mixed.extra_metrics["query/mixed_weight_update"] == 1.0
    assert mixed.extra_metrics["query/action_membership_lookup_count"] == 0.0
    assert mixed.extra_metrics["query/invariant_violation_count"] == 0.0
    assert before == after, "mixed context-only minibatch consumed weight RNG"


def test_main_and_invalid_mode() -> None:
    supported_cached, _, _ = fixture(4, "current_policy", True, ACTION_REPLAY)
    supported_uncached, _, _ = fixture(4, "current_policy", False, ACTION_REPLAY)
    unsupported_cached, _, _ = fixture(4, "current_policy", True, ACTION_QUERY_A)
    unsupported_uncached, _, _ = fixture(4, "current_policy", False, ACTION_QUERY_A)
    for field in ("psi_q_rank", "w_p_raw", "omega_critic", "omega_actor"):
        np.testing.assert_array_equal(
            getattr(supported_cached, field), getattr(supported_uncached, field)
        )
        np.testing.assert_array_equal(
            getattr(unsupported_cached, field), getattr(unsupported_uncached, field)
        )
    np.testing.assert_array_equal(supported_cached.psi_q_rank, np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(supported_cached.w_p_raw, np.full(4, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(unsupported_cached.psi_q_rank, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(unsupported_cached.w_p_raw, np.ones(4, dtype=np.float32))
    bad = config("current_policy", True)
    bad.support_query_mode = "invalid"
    try:
        bad.normalize()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid support_query_mode did not fail")


def test_context_mode_checkpoint_resume_contract() -> None:
    cfg = config("context_count_only", True)
    assert cfg.as_dict()["support_query_mode"] == "context_count_only"

    payload = {
        "config": cfg.as_dict(),
        "global_step": 50_000,
        "last_loss": {"critic": 1.25, "actor": -0.5},
        "last_lite": {"query/invariant_violation_count": 0.0},
        "last_her_ratio": 0.8125,
        "last_eval_metrics": {"eval_env_success_mean": 0.75},
        "eval_history_env": [(10_000, 0.25), (20_000, 0.75)],
    }
    loss, lite, her_ratio, last_eval, history = _restore_training_log_state(
        payload, task().phase_names
    )
    assert payload["config"]["support_query_mode"] == "context_count_only"
    assert _resume_global_step(payload) == 50_000
    assert loss["critic"] == 1.25 and loss["actor"] == -0.5
    assert lite["query/invariant_violation_count"] == 0.0
    assert her_ratio == 0.8125
    assert last_eval == {"eval_env_success_mean": 0.75}
    assert history == [(10_000, 0.25), (20_000, 0.75)]

    # Checkpoints written before these fields were added remain loadable.
    old_loss, old_lite, old_her, old_eval, old_history = (
        _restore_training_log_state({}, task().phase_names)
    )
    assert old_loss["critic"] == 0.0
    assert old_lite["lite/mean_w_P"] == 1.0
    assert old_her == 0.0 and old_eval == {} and old_history == []


def main() -> int:
    test_context_only()
    test_context_only_mixed_minibatch()
    test_main_and_invalid_mode()
    test_context_mode_checkpoint_resume_contract()
    print("test_context_count_only_drawer_open.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
