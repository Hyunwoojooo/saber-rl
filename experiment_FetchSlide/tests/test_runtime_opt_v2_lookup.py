#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "version_v0_lite" / "src"))

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.weights import GapStats
from saber_fetch_slide.experiment_b3_critic_only_runtime_opt_v2 import B3RuntimeOptV2SupportComputer


class FakeTask:
    action_low = np.asarray([-1.0], dtype=np.float32)
    action_high = np.asarray([1.0], dtype=np.float32)
    phase_names = ("all",)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        del h
        return obs[:, 0].astype(np.int64)

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        del h
        return (obs[:, 0].astype(np.int64) * 100 + goal[:, 0].astype(np.int64)).astype(np.int64)

    def action_bin_batch(self, action: np.ndarray, bins_per_dim: int) -> np.ndarray:
        scaled = (np.clip(action[:, 0], -1.0, 1.0) + 1.0) * 0.5 * float(bins_per_dim)
        return np.floor(np.clip(scaled, 0.0, float(bins_per_dim - 1))).astype(np.int64)

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.zeros(obs.shape[0], dtype=np.int64)

    def diagnostic_metrics(self, obs: np.ndarray, goal: np.ndarray) -> dict[str, float]:
        return {}


class FakeAgent:
    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        # First two rows support action-bin 3, last two support action-bin 1.
        values = torch.where(
            state[:, :1] < 2.0,
            torch.full_like(state[:, :1], 0.35),
            torch.full_like(state[:, :1], -0.35),
        )
        return values


def make_config() -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_slide",
        method="b3_support_only",
        batch_size=5,
        M_B=4,
        M_Q_min=2,
        N_local_min=2,
        local_search_max=3,
        action_bins_per_dim=5,
        policy_support_mode="bin",
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.verify_runtime_optimization = True
    cfg.support_scan_mode = "deterministic_tail_v1"
    cfg.runtime_predicate_chunk_size = 1
    cfg.runtime_audit_interval_updates = 1
    return cfg


def make_replay(task: FakeTask, cfg: LiteConfig) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=1, goal_dim=1, action_dim=1)
    rows = [
        (1, 7, 0.40),
        (1, 7, 0.45),
        (1, 7, -0.40),
        (1, 8, -0.40),
        (2, 3, -0.40),
        (2, 3, -0.45),
        (2, 4, 0.40),
        (3, 9, 0.40),
    ]
    for step, (coarse, goal_value, action_value) in enumerate(rows):
        obs = np.asarray([coarse], dtype=np.float32)
        goal = np.asarray([goal_value], dtype=np.float32)
        action = np.asarray([action_value], dtype=np.float32)
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
            h=1,
            episode_id=step,
            episode_step=0,
            coarse_key=int(task.coarse_hash_batch(obs[None, :], np.asarray([1]))[0]),
            action_bin=int(task.action_bin_batch(action[None, :], cfg.action_bins_per_dim)[0]),
        )
    return replay


def assert_action_bin_counts_update(task: FakeTask, cfg: LiteConfig) -> None:
    replay = HERReplayBuffer(capacity=1, obs_dim=1, goal_dim=1, action_dim=1)
    obs = np.asarray([1], dtype=np.float32)
    goal = np.asarray([7], dtype=np.float32)
    first_action = np.asarray([0.40], dtype=np.float32)
    second_action = np.asarray([-0.40], dtype=np.float32)
    coarse_key = int(task.coarse_hash_batch(obs[None, :], np.asarray([1]))[0])
    first_bin = int(task.action_bin_batch(first_action[None, :], cfg.action_bins_per_dim)[0])
    second_bin = int(task.action_bin_batch(second_action[None, :], cfg.action_bins_per_dim)[0])
    for step, action, action_bin in (
        (0, first_action, first_bin),
        (1, second_action, second_bin),
    ):
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
            h=1,
            episode_id=step,
            episode_step=0,
            coarse_key=coarse_key,
            action_bin=action_bin,
        )
    assert replay.action_index.action_bin_count(coarse_key, first_bin) == 0
    assert replay.action_index.action_bin_count(coarse_key, second_bin) == 1
    assert not replay.action_index.has_action_bin(coarse_key, first_bin)
    assert replay.action_index.has_action_bin(coarse_key, second_bin)


def make_batch() -> SampleBatch:
    obs = np.asarray([[1], [1], [1], [2], [3]], dtype=np.float32)
    goals = np.asarray([[7], [7], [8], [3], [9]], dtype=np.float32)
    h = np.ones(5, dtype=np.int16)
    return SampleBatch(
        indices=np.arange(5, dtype=np.int64),
        observation=obs,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=obs.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.zeros((5, 1), dtype=np.float32),
        reward_ext=np.zeros(5, dtype=np.float32),
        done=np.zeros(5, dtype=np.bool_),
        timeout=np.zeros(5, dtype=np.bool_),
        h=h,
        her_relabel_ratio=1.0,
    )


def main() -> None:
    cfg = make_config()
    task = FakeTask()
    assert_action_bin_counts_update(task, cfg)
    replay = make_replay(task, cfg)
    assert replay.action_index.has_action_bin(1, 3)
    assert replay.action_index.action_bin_count(1, 3) == 2
    gap_stats = GapStats(maxlen=100, n_min=10, q_low=0.6, q_high=0.95, eps_g=1e-6)
    computer = B3RuntimeOptV2SupportComputer(cfg, task, gap_stats)
    batch = make_batch()
    wb = computer.compute(
        batch=batch,
        replay=replay,
        agent=FakeAgent(),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(wb.psi_q_rank, np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(wb.q_candidate_fallback, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert wb.extra_metrics["runtime/batch_query_coalescing_saving_rate"] > 0.0
    assert wb.extra_metrics["runtime/local_scan_cache_hit_rate"] > 0.0
    assert wb.extra_metrics["runtime/threshold_short_circuit_rate"] > 0.0
    assert wb.extra_metrics["runtime/coarse_count_short_circuit_rate"] > 0.0
    assert wb.extra_metrics["runtime/predicate_early_exit_rate"] > 0.0
    assert wb.extra_metrics["runtime/diagnostic_audit_rate"] == 1.0

    wb_cached = computer.compute(
        batch=batch,
        replay=replay,
        agent=FakeAgent(),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(wb_cached.psi_q_rank, wb.psi_q_rank)
    np.testing.assert_array_equal(wb_cached.omega_critic, wb.omega_critic)
    np.testing.assert_array_equal(wb_cached.omega_actor, wb.omega_actor)
    assert wb_cached.extra_metrics["runtime/support_cache_hit_rate"] > 0.0
    assert wb_cached.extra_metrics["runtime/full_context_scan_rate"] < 1.0


if __name__ == "__main__":
    main()
