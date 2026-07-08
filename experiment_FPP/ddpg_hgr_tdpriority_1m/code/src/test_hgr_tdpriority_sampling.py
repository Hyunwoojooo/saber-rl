#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.hgr import HGRSampler
from metacog_lite_v0.replay import HERReplayBuffer


class FakeGoalEnv:
    distance_threshold = 0.05

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        del info
        dist = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(dist > self.distance_threshold).astype(np.float32)

    def success_from_goal(self, achieved_goal, desired_goal):
        dist = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return (dist <= self.distance_threshold).astype(np.float32)


def build_replay() -> HERReplayBuffer:
    rng = np.random.default_rng(7)
    replay = HERReplayBuffer(capacity=64, obs_dim=4, goal_dim=3, action_dim=2)
    for ep in range(3):
        desired = np.array([1.0 + ep, 0.5, -0.25], dtype=np.float32)
        for t in range(5):
            obs = rng.normal(size=4).astype(np.float32)
            next_obs = (obs + 0.01 * rng.normal(size=4)).astype(np.float32)
            achieved = np.array([ep, t, 0.1 * t], dtype=np.float32)
            next_achieved = np.array([ep, t + 1, 0.1 * (t + 1)], dtype=np.float32)
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            reward = float(FakeGoalEnv().compute_reward(next_achieved[None, :], desired[None, :])[0])
            replay.add_transition(
                observation=obs,
                achieved_goal=achieved,
                desired_goal=desired,
                action_exec=action,
                reward_ext=reward,
                next_observation=next_obs,
                next_achieved_goal=next_achieved,
                done=t == 4,
                timeout=t == 4,
                h=5 - t,
                episode_id=ep,
                episode_step=t,
                coarse_key=ep,
                action_bin=0,
            )
    return replay


def main() -> int:
    rng = np.random.default_rng(11)
    replay = build_replay()
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="hgr",
        batch_size=8,
        her_ratio=1.0,
        hgr_priority_eps=1e-6,
        hgr_priority_clip_max=50.0,
        hgr_alpha_episode=1.0,
        hgr_alpha_goal=1.0,
        hgr_max_future_candidates=50,
        timeout_bootstrap=True,
        success_terminates=False,
    )
    device = torch.device("cpu")
    agent = DDPGAgent(
        state_dim=7,
        action_dim=2,
        hidden_dim=16,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.98,
        tau=0.005,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=device,
        l2_action_penalty=0.0,
    )
    sampler = HGRSampler(cfg, FakeGoalEnv(), device)

    stats = replay.hgr_priority_stats(cfg.hgr_priority_eps, cfg.hgr_priority_clip_max)
    assert np.isfinite(stats["hgr/episode_priority_mean"])
    assert replay.future_slots(4).size == 0

    future = replay.future_slots(0, max_candidates=cfg.hgr_max_future_candidates, rng=rng)
    td = sampler.candidate_td_errors(replay, agent, 0, future)
    assert td.shape == (future.size,)
    assert np.all(np.isfinite(td))

    batch = sampler.sample_batch(replay, agent, batch_size=8, her_ratio=1.0, rng=rng)
    assert batch.observation.shape == (8, 4)
    assert batch.desired_goal_train.shape == (8, 3)
    assert batch.her_relabel is not None
    assert batch.hgr_future_slot is not None
    assert float(batch.her_relabel_ratio) > 0.0

    for batch_i, (replay_slot, future_slot, relabel) in enumerate(zip(batch.indices, batch.hgr_future_slot, batch.her_relabel)):
        if float(relabel) <= 0.5:
            continue
        future_slots = replay.future_slots(int(replay_slot), max_candidates=cfg.hgr_max_future_candidates, rng=None)
        assert int(future_slot) in set(int(v) for v in future_slots.tolist())
        np.testing.assert_allclose(batch.desired_goal_train[batch_i], replay.next_achieved_goal[int(future_slot)], atol=1e-6)

    reward = FakeGoalEnv().compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {})
    assert reward.shape == (8,)
    assert np.all(np.isfinite(reward))
    assert set(np.unique(reward)).issubset({-1.0, 0.0})

    replay.update_hgr_priorities(batch.indices, np.linspace(0.1, 2.0, num=8, dtype=np.float32), cfg.hgr_priority_eps, cfg.hgr_priority_clip_max)
    after = replay.hgr_priority_stats(cfg.hgr_priority_eps, cfg.hgr_priority_clip_max)
    assert np.isfinite(after["hgr/episode_priority_mean"])
    assert np.isfinite(after["hgr/episode_priority_max"])
    assert np.all(np.isfinite(replay.hgr_priority[: replay.size]))

    print("HGR fake-batch sampling test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
