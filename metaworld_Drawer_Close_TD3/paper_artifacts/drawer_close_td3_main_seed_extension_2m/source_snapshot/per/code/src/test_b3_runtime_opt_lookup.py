#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


class MiniTask:
    def __init__(self) -> None:
        self.action_low = np.asarray([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.asarray([1.0, 1.0], dtype=np.float32)

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.zeros(obs.shape[0], dtype=np.int64)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        obs_key = np.floor(obs[:, 0]).astype(np.int64)
        h_key = (h.astype(np.int64) > 10).astype(np.int64)
        return obs_key * 10 + h_key

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        obs_key = np.floor(obs[:, 0]).astype(np.int64)
        goal_key = np.floor(goal[:, 0] * 10.0).astype(np.int64)
        h_key = (h.astype(np.int64) > 10).astype(np.int64)
        return obs_key * 1000 + goal_key * 10 + h_key

    def full_context_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return int(self.full_context_hash_batch(obs[None, :], goal[None, :], np.asarray([h], dtype=np.int64))[0])

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32)

    def action_bin_batch(self, action: np.ndarray, bins_per_dim: int) -> np.ndarray:
        norm = self.normalize_action(action)
        scaled = (norm + 1.0) * 0.5 * float(bins_per_dim)
        bins = np.floor(np.clip(scaled, 0.0, float(bins_per_dim - 1))).astype(np.int64)
        multipliers = (bins_per_dim ** np.arange(action.shape[1], dtype=np.int64))[None, :]
        return np.sum(bins * multipliers, axis=1).astype(np.int64)


class FakeAgent:
    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((state.shape[0], 2), dtype=torch.float32, device=state.device)
        out[:, 0] = torch.where(state[:, 0] > 0.5, torch.tensor(0.55, device=state.device), torch.tensor(-0.25, device=state.device))
        out[:, 1] = 0.10
        return out


def make_config() -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=8,
        M_B=4,
        M_Q_min=3,
        N_local_min=3,
        local_search_max=7,
        action_bins_per_dim=5,
        lambda_support=1.0,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    return cfg


def make_replay(task: MiniTask, cfg: LiteConfig) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=64, obs_dim=2, goal_dim=1, action_dim=2)
    goal = np.asarray([0.0], dtype=np.float32)
    actions = [
        np.asarray([-0.25, 0.10], dtype=np.float32),
        np.asarray([0.55, 0.10], dtype=np.float32),
        np.asarray([0.90, -0.50], dtype=np.float32),
        np.asarray([-0.80, 0.80], dtype=np.float32),
    ]
    for slot in range(40):
        group = 0 if slot < 24 else 1
        obs = np.asarray([float(group), float(slot % 5)], dtype=np.float32)
        action = actions[slot % len(actions)]
        h = 5 if slot % 3 else 15
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=action,
            reward_ext=0.0,
            next_observation=obs + 0.01,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=h,
            episode_id=slot // 5,
            episode_step=slot % 5,
            coarse_key=task.coarse_hash(obs, h),
            action_bin=int(task.action_bin_batch(action[None, :], cfg.action_bins_per_dim)[0]),
        )
    return replay


def make_batch(replay: HERReplayBuffer) -> SampleBatch:
    indices = np.asarray([1, 2, 4, 5, 7, 25, 26, 28], dtype=np.int64)
    desired_goal_train = np.zeros((indices.size, 1), dtype=np.float32)
    return SampleBatch(
        indices=indices,
        observation=replay.observation[indices].copy(),
        achieved_goal=replay.achieved_goal[indices].copy(),
        desired_goal_train=desired_goal_train,
        next_observation=replay.next_observation[indices].copy(),
        next_achieved_goal=replay.next_achieved_goal[indices].copy(),
        action_exec=replay.action_exec[indices].copy(),
        reward_ext=replay.reward_ext[indices].copy(),
        done=replay.done[indices].copy(),
        timeout=replay.timeout[indices].copy(),
        h=replay.h[indices].copy(),
        her_relabel_ratio=1.0,
    )


def naive_deterministic(cfg: LiteConfig, task: MiniTask, batch: SampleBatch, replay: HERReplayBuffer, agent: FakeAgent):
    helper = LiteWeightComputer(replace(cfg, method="b3_support_only"), task, GapStats(16, 4, 0.6, 0.95, 1e-6))
    state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
    action_policy = agent.policy_actions(torch.as_tensor(state_np, dtype=torch.float32)).cpu().numpy().astype(np.float32)
    b = batch.observation.shape[0]
    psi = np.zeros(b, dtype=np.float32)
    psi_a = np.zeros(b, dtype=np.float32)
    fallback = np.ones(b, dtype=np.float32)
    len_local = np.zeros(b, dtype=np.float32)
    len_sup = np.zeros(b, dtype=np.float32)
    scanned_count = np.zeros(b, dtype=np.float32)
    truncated = np.zeros(b, dtype=np.float32)
    for i in range(b):
        coarse = task.coarse_hash(batch.observation[i], int(batch.h[i]))
        full = task.full_context_hash(batch.observation[i], batch.desired_goal_train[i], int(batch.h[i]))
        scanned, _n_candidates, was_truncated = replay.indexed_reference_indices_deterministic(coarse, cfg.local_search_max)
        if scanned.size:
            ref_goal = np.repeat(batch.desired_goal_train[i][None, :], scanned.size, axis=0)
            ref_hash = task.full_context_hash_batch(replay.observation[scanned], ref_goal, replay.h[scanned])
            local_idx = scanned[ref_hash == full]
        else:
            local_idx = np.zeros((0,), dtype=np.int64)
        local_bins = replay.action_bin[local_idx].copy() if local_idx.size else np.zeros((0,), dtype=np.int64)
        len_local[i] = float(local_idx.size)
        scanned_count[i] = float(scanned.size)
        truncated[i] = float(was_truncated)
        if local_idx.size < cfg.N_local_min:
            continue
        n_sup = min(int(local_idx.size), int(cfg.M_B))
        len_sup[i] = float(n_sup)
        psi_a[i] = helper._policy_action_supported_from_indices(action_policy[i], replay, local_idx, local_bins)
        if n_sup >= cfg.M_Q_min and psi_a[i] >= 0.5:
            psi[i] = 1.0
            fallback[i] = 0.0
    w = 1.0 + cfg.lambda_support * psi
    omega = (w / max(cfg.eps, float(np.mean(w)))).astype(np.float32)
    return {
        "psi_q_rank": psi,
        "psi_a_policy": psi_a,
        "q_candidate_fallback": fallback,
        "w_p_raw": w.astype(np.float32),
        "omega_critic": omega,
        "omega_actor": np.ones_like(omega, dtype=np.float32),
        "len_a_q_local": len_local,
        "len_a_q_sup": len_sup,
        "num_scanned_candidates": scanned_count,
        "local_scan_truncated": truncated,
    }


def main() -> int:
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = make_config()
    task = MiniTask()
    replay = make_replay(task, cfg)
    batch = make_batch(replay)
    agent = FakeAgent()
    expected = naive_deterministic(cfg, task, batch, replay, agent)
    computer = LiteWeightComputer(cfg, task, GapStats(16, 4, 0.6, 0.95, 1e-6))
    wb = computer.compute(batch, replay, agent, np.random.default_rng(123), torch.device("cpu"))
    wb_cached = computer.compute(batch, replay, agent, np.random.default_rng(456), torch.device("cpu"))
    actual = {
        "psi_q_rank": wb.psi_q_rank,
        "psi_a_policy": wb.psi_a_policy,
        "q_candidate_fallback": wb.q_candidate_fallback,
        "w_p_raw": wb.w_p_raw,
        "omega_critic": wb.omega_critic,
        "omega_actor": wb.omega_actor,
        "len_a_q_local": wb.len_a_q_local,
        "len_a_q_sup": wb.len_a_q_sup,
        "num_scanned_candidates": wb.num_scanned_candidates,
        "local_scan_truncated": wb.local_scan_truncated,
    }
    mismatches = []
    for key, exp in expected.items():
        if not np.allclose(exp, actual[key], atol=0.0, rtol=0.0):
            mismatches.append(key)
    cache_hit_rate_second = float(wb_cached.extra_metrics.get("runtime/support_cache_hit_rate", 0.0))
    result = {
        "mismatches": mismatches,
        "support_cache_hit_rate_second": cache_hit_rate_second,
        "batch_query_unique_rate": float(wb.extra_metrics.get("runtime/batch_query_unique_rate", 0.0)),
        "local_scan_cache_hit_rate_first": float(wb.extra_metrics.get("runtime/local_scan_cache_hit_rate", 0.0)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if mismatches or cache_hit_rate_second < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
