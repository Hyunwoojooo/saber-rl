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


class StaticAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(action[None, :], dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.to(state.device).repeat(state.shape[0], 1)


def make_config(
    numba_enabled: bool,
    predicate_enabled: bool = False,
    mode: str = "current_policy",
    cache_enabled: bool = True,
) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=4,
        M_B=4,
        M_Q_min=2,
        N_local_min=2,
        local_search_max=8,
        action_bins_per_dim=5,
        lambda_support=1.0,
        policy_support_mode="bin",
        support_query_mode=mode,
        support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=cache_enabled,
        local_scan_cache_enabled=cache_enabled,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.numba_support_scan = bool(numba_enabled)
    cfg.support_predicate_early_exit = bool(predicate_enabled)
    cfg.support_diagnostic_audit_interval = 1
    return cfg


def make_replay(task: FetchPickAndPlaceTaskSpec, cfg: LiteConfig) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    policy_action = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    other_action = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    rows = [
        (make_obs(), policy_action),
        (make_obs(), other_action),
        (make_obs(), other_action),
        (make_obs(obj_x=0.8), other_action),
    ]
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
            action_bin=int(task.action_bin_batch(action[None, :], cfg.action_bins_per_dim)[0]),
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


def main() -> int:
    if not NUMBA_AVAILABLE:
        print("SKIP: numba is not installed")
        return 0
    semantic_fields = (
        "psi_q_rank",
        "psi_a_policy",
        "q_candidate_fallback",
        "w_p_raw",
        "omega_critic",
        "omega_actor",
        "len_a_q_local",
        "len_a_q_sup",
        "num_scanned_candidates",
        "local_scan_truncated",
    )
    for mode in ("current_policy", "replay_action", "shuffled_policy", "context_count_only"):
        task = make_task()
        cfg_numba = make_config(True, predicate_enabled=False, mode=mode)
        replay = make_replay(task, cfg_numba)
        batch = make_batch()
        agent = StaticAgent(np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32))

        cfg_python = replace(cfg_numba)
        cfg_python.numba_support_scan = False
        wb_numba = LiteWeightComputer(cfg_numba, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
            batch, replay, agent, np.random.default_rng(0), torch.device("cpu")
        )
        wb_python = LiteWeightComputer(cfg_python, task, GapStats(100, 10, 0.6, 0.95, 1e-6)).compute(
            batch, replay, agent, np.random.default_rng(0), torch.device("cpu")
        )
        for name in semantic_fields:
            np.testing.assert_array_equal(getattr(wb_numba, name), getattr(wb_python, name))
        assert wb_numba.extra_metrics["runtime/numba_support_scan_enabled"] == 1.0
        assert wb_numba.extra_metrics["runtime/numba_support_scan_fallback_rate"] == 0.0

        cfg_predicate = make_config(True, predicate_enabled=True, mode=mode)
        wb_predicate = LiteWeightComputer(
            cfg_predicate, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
        ).compute(batch, replay, agent, np.random.default_rng(0), torch.device("cpu"))
        for name in (
            "psi_q_rank",
            "psi_a_policy",
            "q_candidate_fallback",
            "w_p_raw",
            "omega_critic",
            "omega_actor",
        ):
            np.testing.assert_array_equal(getattr(wb_predicate, name), getattr(wb_python, name))
        assert wb_predicate.extra_metrics["runtime/numba_support_scan_enabled"] == 1.0
        assert wb_predicate.extra_metrics["runtime/numba_support_scan_fallback_rate"] == 0.0
        assert "runtime/predicate_early_exit_rate" in wb_predicate.extra_metrics
        assert "runtime/diagnostic_audit_rate" in wb_predicate.extra_metrics

        cfg_no_cache = make_config(True, predicate_enabled=True, mode=mode, cache_enabled=False)
        wb_no_cache = LiteWeightComputer(
            cfg_no_cache, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
        ).compute(batch, replay, agent, np.random.default_rng(0), torch.device("cpu"))
        for name in semantic_fields:
            np.testing.assert_array_equal(getattr(wb_no_cache, name), getattr(wb_predicate, name))
    print("test_numba_runtime_opt_fpp.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
