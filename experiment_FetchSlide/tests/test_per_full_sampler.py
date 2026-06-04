#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "version_v0_lite" / "src"))

from metacog_lite_v0.replay import HERReplayBuffer  # noqa: E402
from saber_fetch_slide.per_full_sampler import FullPERSampler  # noqa: E402


def _make_replay(n: int = 20, capacity: int = 64) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=capacity, obs_dim=4, goal_dim=3, action_dim=2)
    for i in range(n):
        obs = np.full(4, i, dtype=np.float32)
        ag = np.array([i, 0.0, 0.0], dtype=np.float32)
        dg = np.zeros(3, dtype=np.float32)
        action = np.array([0.1, -0.2], dtype=np.float32)
        next_obs = obs + 1.0
        next_ag = np.array([i + 1, 0.0, 0.0], dtype=np.float32)
        replay.add_transition(
            observation=obs,
            achieved_goal=ag,
            desired_goal=dg,
            action_exec=action,
            reward_ext=0.0,
            next_observation=next_obs,
            next_achieved_goal=next_ag,
            done=(i == n - 1),
            timeout=False,
            h=n - i,
            episode_id=0,
            episode_step=i,
        )
    return replay


def test_per_full_sample_preserves_batch_shapes_and_weights() -> None:
    replay = _make_replay()
    sampler = FullPERSampler(capacity=replay.capacity, alpha=0.6, beta_start=0.4, beta_final=1.0)
    rng = np.random.default_rng(123)
    batch = sampler.sample(replay, batch_size=8, her_ratio=0.8, rng=rng, step=10)

    assert batch.indices.shape == (8,)
    assert batch.observation.shape == (8, replay.obs_dim)
    assert batch.desired_goal_train.shape == (8, replay.goal_dim)
    assert hasattr(batch, "per_full_is_weight")
    weights = getattr(batch, "per_full_is_weight")
    assert weights.shape == (8,)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert 0.0 <= batch.her_relabel_ratio <= 1.0


def test_per_full_priority_update_changes_sampling_distribution() -> None:
    replay = _make_replay(n=300, capacity=512)
    sampler = FullPERSampler(capacity=replay.capacity, alpha=1.0, beta_start=0.4, beta_final=1.0)
    sampler.update_priorities(np.array([3, 7]), np.array([10.0, 0.1], dtype=np.float32))

    rng = np.random.default_rng(321)
    batch = sampler.sample(replay, batch_size=256, her_ratio=0.0, rng=rng, step=100)

    assert np.count_nonzero(batch.indices == 3) > np.count_nonzero(batch.indices == 7)
    metrics = getattr(batch, "per_full_sample_metrics")
    assert metrics["per_full/priority_max"] > metrics["per_full/priority_mean"]
    assert metrics["per_full/is_weight_max"] <= 1.0


if __name__ == "__main__":
    test_per_full_sample_preserves_batch_shapes_and_weights()
    test_per_full_priority_update_changes_sampling_distribution()
