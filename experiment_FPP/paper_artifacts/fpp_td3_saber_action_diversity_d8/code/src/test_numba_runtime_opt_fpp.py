#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.numba_support_scan_fpp import NUMBA_AVAILABLE
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


def make_task() -> FetchPickAndPlaceTaskSpec:
    return FetchPickAndPlaceTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )


def make_obs(obj_x: float = 0.04, obj_z: float = 0.44, gap: float = 0.06) -> np.ndarray:
    obs = np.zeros(25, dtype=np.float32)
    obs[0:3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    obs[3:6] = np.asarray([obj_x, 0.0, obj_z], dtype=np.float32)
    obs[9:11] = np.asarray([0.5 * gap, 0.5 * gap], dtype=np.float32)
    return obs


def diverse_actions() -> np.ndarray:
    return np.asarray(
        [
            [0.6, 0.0, 0.0, 0.0],
            [-0.6, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0],
            [0.0, 0.0, -0.6, 0.0],
        ],
        dtype=np.float32,
    )


class StaticAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(action[None, :], dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.to(state.device).repeat(state.shape[0], 1)


def make_config(
    numba_enabled: bool,
    predicate_enabled: bool,
    dmin: int = 4,
    cache_enabled: bool = True,
    assignment_mode: str = "support",
    probe_thresholds: tuple[int, ...] = (),
) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="td3_saber_runtime_opt",
        batch_size=4,
        M_B=16,
        M_Q_min=4,
        N_local_min=4,
        local_search_max=8,
        action_bins_per_dim=5,
        support_min_unique_action_bins=dmin,
        support_assignment_mode=assignment_mode,
        support_diversity_probe_thresholds=probe_thresholds,
        lambda_support=1.0,
        policy_support_mode="bin",
        support_scan_mode="deterministic_tail_v1",
        support_short_circuit=not bool(probe_thresholds),
        support_cache_enabled=cache_enabled,
        local_scan_cache_enabled=cache_enabled,
        numba_support_scan=numba_enabled,
        support_predicate_early_exit=predicate_enabled,
        support_diagnostic_audit_interval=1,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    return cfg


def make_replay(task: FetchPickAndPlaceTaskSpec, cfg: LiteConfig) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    rows = [(make_obs(), action) for action in diverse_actions()]
    rows.append((make_obs(obj_x=0.8), np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)))
    for step, (obs, action) in enumerate(rows):
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=action,
            reward_ext=0.0,
            next_observation=obs,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=20,
            episode_id=step,
            episode_step=0,
            coarse_key=task.coarse_hash(obs, 20),
            action_bin=int(
                task.action_bin_batch(action[None, :], cfg.action_bins_per_dim)[0]
            ),
        )
    return replay


def make_batch() -> SampleBatch:
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    obs_pos = make_obs()
    obs_neg = make_obs(obj_x=0.8)
    obs = np.stack([obs_pos, obs_pos, obs_neg, obs_pos])
    goals = np.stack([goal, goal, goal, goal])
    h = np.asarray([20, 20, 20, 20], dtype=np.int16)
    return SampleBatch(
        indices=np.arange(4, dtype=np.int64),
        observation=obs,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=obs.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.zeros((4, 4), dtype=np.float32),
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=h,
        her_relabel_ratio=1.0,
    )


def compute(cfg: LiteConfig, seed: int = 0, computer: LiteWeightComputer | None = None):
    task = make_task()
    replay = make_replay(task, cfg)
    batch = make_batch()
    agent = StaticAgent(np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32))
    if computer is None:
        computer = LiteWeightComputer(
            cfg,
            task,
            GapStats(100, 10, 0.6, 0.95, 1e-6),
        )
    wb = computer.compute(
        batch,
        replay,
        agent,
        np.random.default_rng(seed),
        torch.device("cpu"),
    )
    return wb, computer, replay, batch, agent


def assert_weight_batch_equal(left, right) -> None:
    for name in (
        "psi_q_rank",
        "psi_a_policy",
        "pre_diversity_support",
        "unique_action_bin_count",
        "diversity_pass",
        "q_candidate_fallback",
        "w_p_raw",
        "omega_critic",
        "omega_actor",
        "len_a_q_local",
        "len_a_q_sup",
        "num_scanned_candidates",
        "local_scan_truncated",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_python_numba_and_early_exit_parity() -> None:
    cfg_python = make_config(False, False, dmin=4)
    cfg_numba = make_config(True, False, dmin=4)
    cfg_early = make_config(True, True, dmin=4)
    wb_python, *_ = compute(cfg_python)
    wb_numba, *_ = compute(cfg_numba)
    wb_early, *_ = compute(cfg_early)
    assert_weight_batch_equal(wb_numba, wb_python)
    for name in (
        "psi_q_rank",
        "psi_a_policy",
        "pre_diversity_support",
        "diversity_pass",
        "q_candidate_fallback",
        "w_p_raw",
        "omega_critic",
        "omega_actor",
    ):
        np.testing.assert_array_equal(getattr(wb_early, name), getattr(wb_python, name))
    assert wb_numba.extra_metrics["runtime/numba_support_scan_enabled"] == 1.0
    assert wb_early.extra_metrics["runtime/predicate_early_exit_rate"] > 0.0


def test_cache_on_off_parity() -> None:
    cfg_cached = make_config(True, True, dmin=4, cache_enabled=True)
    cfg_uncached = make_config(True, True, dmin=4, cache_enabled=False)
    wb_cached, computer, replay, batch, agent = compute(cfg_cached)
    wb_cached_second = computer.compute(
        batch,
        replay,
        agent,
        np.random.default_rng(0),
        torch.device("cpu"),
    )
    wb_uncached, *_ = compute(cfg_uncached)
    assert_weight_batch_equal(wb_cached, wb_uncached)
    assert_weight_batch_equal(wb_cached_second, wb_uncached)
    assert wb_cached_second.extra_metrics["runtime/support_cache_hit_rate"] > 0.0


def test_random_count_control() -> None:
    cfg_reference = make_config(True, True, dmin=4, assignment_mode="support")
    cfg_random = make_config(True, True, dmin=4, assignment_mode="random_count")
    reference, *_ = compute(cfg_reference, seed=27027)
    random_a, *_ = compute(cfg_random, seed=27027)
    random_b, *_ = compute(cfg_random, seed=27027)
    reference_count = int(np.sum(reference.psi_q_rank > 0.5))
    assigned_a = random_a.w_p_raw > 1.5
    assigned_b = random_b.w_p_raw > 1.5
    assert int(np.sum(assigned_a)) == reference_count
    np.testing.assert_array_equal(assigned_a, assigned_b)
    assert random_a.extra_metrics["support/random_count_match"] == 1.0
    assert (
        random_a.extra_metrics["support/random_count_assigned_count"]
        == random_a.extra_metrics["support/random_count_reference_count"]
    )


def test_probe_cumulative_counts_and_serialization() -> None:
    cfg = make_config(
        True,
        False,
        dmin=1,
        probe_thresholds=(2, 4, 8),
    )
    wb_first, computer, replay, batch, agent = compute(cfg)
    wb_second = computer.compute(
        batch,
        replay,
        agent,
        np.random.default_rng(1),
        torch.device("cpu"),
    )
    assert wb_first.extra_metrics["support/probe_total_count"] == 4.0
    assert wb_second.extra_metrics["support/probe_total_count"] == 8.0
    assert wb_second.extra_metrics["support/probe_main_positive_count"] == 6.0
    assert wb_second.extra_metrics["support/probe_d4_positive_count"] == 6.0
    assert wb_second.extra_metrics["support/probe_d8_positive_count"] == 0.0
    payload = cfg.as_dict()
    assert payload["support_min_unique_action_bins"] == 1
    assert payload["support_assignment_mode"] == "support"
    assert payload["support_diversity_probe_thresholds"] == (2, 4, 8)


def test_cache_key_contains_dmin() -> None:
    task = make_task()
    cfg_d1 = make_config(True, True, dmin=1)
    cfg_d4 = make_config(True, True, dmin=4)
    comp_d1 = LiteWeightComputer(cfg_d1, task, GapStats(100, 10, 0.6, 0.95, 1e-6))
    comp_d4 = LiteWeightComputer(cfg_d4, task, GapStats(100, 10, 0.6, 0.95, 1e-6))
    kwargs = {
        "coarse_key": 1,
        "full_key": 2,
        "goal": np.zeros(3, dtype=np.float32),
        "h": 20,
        "bucket_version": 3,
        "policy_bin": 4,
    }
    assert comp_d1._support_cache_key(**kwargs) != comp_d4._support_cache_key(**kwargs)


def main() -> int:
    if not NUMBA_AVAILABLE:
        print("SKIP: numba is not installed")
        return 0
    test_python_numba_and_early_exit_parity()
    test_cache_on_off_parity()
    test_random_count_control()
    test_probe_cumulative_counts_and_serialization()
    test_cache_key_contains_dmin()
    print("test_numba_runtime_opt_fpp.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
