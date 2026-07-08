#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch

from metacog_lite_v0.hgr import HGRSampler
from metacog_lite_v0.replay import HERReplayBuffer


class FakeEnv:
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info: dict) -> np.ndarray:
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        return np.where(dist <= 0.05, 0.0, -1.0).astype(np.float32)

    def success_from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(achieved_goal - desired_goal, axis=1) <= 0.05).astype(np.float32)


class FakeAgent:
    @torch.no_grad()
    def td_error_abs(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> torch.Tensor:
        del next_state, nonterminal
        goal = state[:, -3:]
        action_term = action_exec.pow(2).sum(dim=1)
        return (reward.reshape(-1) + goal.abs().sum(dim=1) + 0.1 * action_term).abs()


def build_replay() -> tuple[HERReplayBuffer, HGRSampler]:
    rng = np.random.default_rng(123)
    replay = HERReplayBuffer(capacity=64, obs_dim=6, goal_dim=3, action_dim=4)
    sampler = HGRSampler(capacity=64, max_future_candidates=50)
    for ep in range(3):
        for t in range(5):
            obs = rng.normal(size=6).astype(np.float32)
            ag = np.array([ep, t, 0.0], dtype=np.float32) * 0.02
            next_ag = np.array([ep, t + 1, 0.0], dtype=np.float32) * 0.02
            desired = np.array([ep, 10, 0.0], dtype=np.float32) * 0.02
            action = rng.normal(size=4).astype(np.float32)
            slot = replay.add_transition(
                observation=obs,
                achieved_goal=ag,
                desired_goal=desired,
                action_exec=action,
                reward_ext=-1.0,
                next_observation=obs + 0.01,
                next_achieved_goal=next_ag,
                done=(t == 4),
                timeout=(t == 4),
                h=5 - t,
                episode_id=ep,
                episode_step=t,
                coarse_key=None,
                action_bin=None,
            )
            sampler.on_replay_add(slot, ep, t)
    return replay, sampler


def main() -> int:
    rng = np.random.default_rng(321)
    replay, sampler = build_replay()
    env = FakeEnv()
    agent = FakeAgent()
    batch, info = sampler.sample(
        replay=replay,
        batch_size=8,
        her_ratio=1.0,
        rng=rng,
        agent=agent,
        env=env,
        device=torch.device("cpu"),
        success_terminates=False,
        timeout_bootstrap=True,
    )
    assert sampler.slot_priority.shape == (64,)
    assert batch.indices.shape == (8,)
    assert batch.desired_goal_train.shape == (8, 3)
    assert np.any(info.selected_future_slots >= 0)
    assert np.isfinite(info.metrics["hgr/candidate_td_error_mean"])
    for row_i, (idx, future_slot) in enumerate(zip(batch.indices, info.selected_future_slots)):
        assert future_slot >= 0
        assert int(replay.episode_id[idx]) == int(replay.episode_id[future_slot])
        assert int(replay.episode_step[future_slot]) > int(replay.episode_step[idx])
        assert np.allclose(batch.desired_goal_train[row_i], replay.next_achieved_goal[future_slot])
    reward = env.compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {})
    assert reward.shape == (8,)
    assert np.all(np.logical_or(np.isclose(reward, -1.0), np.isclose(reward, 0.0)))
    update = sampler.update_priorities(batch.indices, np.linspace(0.1, 2.0, num=8, dtype=np.float32), replay)
    assert np.all(np.isfinite(sampler.slot_priority[: replay.size]))
    assert np.isfinite(update["hgr/priority_clip_rate"])

    # Last-step future candidates are empty; the helper should report none.
    last_slot = int([slot for slot in range(replay.size) if replay.episode_step[slot] == 4][0])
    empty = sampler._candidate_td_errors(
        replay=replay,
        base_slot=last_slot,
        future_slots=np.zeros((0,), dtype=np.int64),
        agent=agent,
        env=env,
        device=torch.device("cpu"),
        success_terminates=False,
        timeout_bootstrap=True,
    )
    assert empty.shape == (0,)
    print("HGR fake-batch test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
