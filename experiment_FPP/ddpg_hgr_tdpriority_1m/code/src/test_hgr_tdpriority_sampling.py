#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.hgr import HGRSampler
from metacog_lite_v0.replay import HERReplayBuffer


class FakeGoalEnv:
    distance_threshold = 0.05

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        del info
        dist = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(dist > self.distance_threshold).astype(np.float32)


def make_config(capacity: int, **overrides: object) -> LiteConfig:
    values: dict[str, object] = {
        "task_name": "fetch_pick_and_place",
        "method": "hgr",
        "total_steps": 1_000,
        "warmup_steps": 0,
        "updates_per_step": 1,
        "batch_size": 8,
        "buffer_capacity": capacity,
        "her_ratio": 0.8,
        "hgr_priority_eps": 1e-6,
        "hgr_priority_init": 1.0,
        "hgr_priority_clip_max": 0.0,
        "hgr_alpha_episode": 0.6,
        "hgr_alpha_goal": 0.6,
        "hgr_beta_episode_initial": 0.4,
        "hgr_beta_goal_initial": 0.4,
        "hgr_beta_episode_final": 1.0,
        "hgr_beta_goal_final": 1.0,
        "hgr_beta_anneal_updates": 100,
        "hgr_uniform_mix": 0.0,
        "timeout_bootstrap": True,
        "success_terminates": False,
    }
    values.update(overrides)
    config = LiteConfig(**values)
    config.normalize()
    return config


def make_replay(capacity: int) -> HERReplayBuffer:
    return HERReplayBuffer(capacity=capacity, obs_dim=4, goal_dim=3, action_dim=2)


def add_episode(
    replay: HERReplayBuffer,
    sampler: HGRSampler,
    episode_id: int,
    length: int,
) -> None:
    rng = np.random.default_rng(100 + int(episode_id))
    desired = np.array([10.0 + episode_id, 0.5, -0.25], dtype=np.float32)
    env = FakeGoalEnv()
    for step in range(length):
        obs = rng.normal(size=4).astype(np.float32)
        next_obs = (obs + 0.01 * rng.normal(size=4)).astype(np.float32)
        achieved = np.array([episode_id, step, 0.1 * step], dtype=np.float32)
        next_achieved = np.array(
            [episode_id, step + 1, 0.1 * (step + 1)], dtype=np.float32
        )
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        done = step == length - 1
        slot = replay.add_transition(
            observation=obs,
            achieved_goal=achieved,
            desired_goal=desired,
            action_exec=action,
            reward_ext=float(env.compute_reward(next_achieved[None], desired[None])[0]),
            next_observation=next_obs,
            next_achieved_goal=next_achieved,
            done=done,
            timeout=done,
            h=length - step,
            episode_id=episode_id,
            episode_step=step,
            coarse_key=episode_id,
            action_bin=0,
        )
        sampler.on_replay_add(
            slot,
            episode_id,
            step,
            replay,
            episode_done=done,
        )


def assert_future_pair_batch(
    replay: HERReplayBuffer,
    sampler: HGRSampler,
    batch_size: int = 8,
) -> None:
    batch = sampler.sample_batch(
        replay,
        batch_size=batch_size,
        her_ratio=1.0,
        rng=np.random.default_rng(11),
    )
    assert batch.her_relabel is not None
    assert batch.hgr_episode_id is not None
    assert batch.hgr_pair_index is not None
    assert batch.hgr_goal_slot is not None
    assert np.all(batch.her_relabel == 1.0)
    for out_position in range(batch_size):
        transition_slot = int(batch.indices[out_position])
        episode_id = int(batch.hgr_episode_id[out_position])
        pair_index = int(batch.hgr_pair_index[out_position])
        goal_slot = int(batch.hgr_goal_slot[out_position])
        table = sampler._tables[episode_id]
        transition_positions, goal_positions = sampler._pair_positions(table.slots.size)
        assert transition_slot == int(table.slots[int(transition_positions[pair_index])])
        assert goal_slot == int(table.slots[int(goal_positions[pair_index])])
        assert int(replay.episode_id[transition_slot]) == episode_id
        assert int(replay.episode_id[goal_slot]) == episode_id
        # next_achieved_goal[k] is g_{k+1}, so k >= j is exactly j < i.
        assert int(replay.episode_step[goal_slot]) >= int(replay.episode_step[transition_slot])
        np.testing.assert_allclose(
            batch.desired_goal_train[out_position],
            replay.next_achieved_goal[goal_slot],
            atol=1e-6,
        )

    reward = FakeGoalEnv().compute_reward(
        batch.next_achieved_goal, batch.desired_goal_train, {}
    )
    assert reward.shape == (batch_size,)
    assert np.all(np.isfinite(reward))
    assert set(np.unique(reward)).issubset({-1.0, 0.0})


def assert_empirical_probabilities(replay: HERReplayBuffer, sampler: HGRSampler) -> None:
    for episode_id, scale in ((0, 1.0), (1, 2.0)):
        table = sampler._tables[episode_id]
        values = scale * np.arange(1, table.pair_count + 1, dtype=np.float32)
        table.priorities[:] = values
        table.priority_sum = float(np.sum(values, dtype=np.float64))
        sampler.max_priority = max(sampler.max_priority, float(np.max(values)))

    episode_ids, episode_probs, _means = sampler.episode_distribution()
    expected_episode = {int(ep): float(p) for ep, p in zip(episode_ids, episode_probs)}
    expected_pair_zero = sampler.pair_distribution(0)
    episode_counts = {int(ep): 0 for ep in episode_ids}
    pair_zero_counts = np.zeros_like(expected_pair_zero, dtype=np.int64)
    rng = np.random.default_rng(17)
    total = 0
    for _ in range(1_500):
        batch = sampler.sample_batch(replay, batch_size=8, her_ratio=1.0, rng=rng)
        for episode_id, pair_index in zip(batch.hgr_episode_id, batch.hgr_pair_index):
            ep = int(episode_id)
            episode_counts[ep] += 1
            total += 1
            if ep == 0:
                pair_zero_counts[int(pair_index)] += 1

    for episode_id, expected in expected_episode.items():
        observed = episode_counts[episode_id] / float(total)
        assert abs(observed - expected) < 0.025, (episode_id, observed, expected)
    observed_pair_zero = pair_zero_counts / float(np.sum(pair_zero_counts))
    np.testing.assert_allclose(observed_pair_zero, expected_pair_zero, atol=0.025, rtol=0.0)


def assert_is_and_beta(replay: HERReplayBuffer, sampler: HGRSampler) -> None:
    rng = np.random.default_rng(23)
    batch = sampler.sample_batch(replay, batch_size=8, her_ratio=1.0, rng=rng)
    critic_weight, actor_weight = sampler.loss_weights(batch)
    assert np.all(np.isfinite(critic_weight))
    assert np.all(critic_weight > 0.0)
    assert float(np.max(critic_weight)) <= 1.0 + 1e-6
    assert np.isclose(float(np.max(critic_weight)), 1.0)
    np.testing.assert_allclose(actor_weight, 1.0)

    sampler.critic_updates = 0
    np.testing.assert_allclose(sampler.current_betas(), (0.4, 0.4), atol=1e-7)
    sampler.critic_updates = sampler.beta_anneal_updates // 2
    np.testing.assert_allclose(sampler.current_betas(), (0.7, 0.7), atol=1e-7)
    sampler.critic_updates = sampler.beta_anneal_updates
    np.testing.assert_allclose(sampler.current_betas(), (1.0, 1.0), atol=1e-7)
    sampler.critic_updates = 0


def assert_uniform_reduction() -> None:
    replay = make_replay(32)
    config = make_config(
        32,
        hgr_alpha_episode=0.0,
        hgr_alpha_goal=0.0,
        hgr_beta_episode_initial=0.0,
        hgr_beta_goal_initial=0.0,
        hgr_beta_episode_final=0.0,
        hgr_beta_goal_final=0.0,
    )
    sampler = HGRSampler(config)
    add_episode(replay, sampler, 0, 4)
    add_episode(replay, sampler, 1, 4)
    sampler._tables[0].priorities[:] = np.arange(1, 11, dtype=np.float32)
    sampler._tables[0].priority_sum = float(np.sum(sampler._tables[0].priorities))
    episode_ids, episode_probs, _means = sampler.episode_distribution()
    np.testing.assert_allclose(episode_probs, 1.0 / episode_ids.size)
    np.testing.assert_allclose(
        sampler.pair_distribution(0),
        1.0 / sampler._tables[0].pair_count,
    )
    batch = sampler.sample_batch(
        replay, batch_size=8, her_ratio=1.0, rng=np.random.default_rng(29)
    )
    critic_weight, actor_weight = sampler.loss_weights(batch)
    np.testing.assert_allclose(critic_weight, 1.0)
    np.testing.assert_allclose(actor_weight, 1.0)


def assert_priority_update_checkpoint_and_overwrite() -> None:
    replay = make_replay(8)
    config = make_config(8)
    sampler = HGRSampler(config)
    add_episode(replay, sampler, 0, 4)
    add_episode(replay, sampler, 1, 4)
    batch = sampler.sample_batch(
        replay, batch_size=8, her_ratio=1.0, rng=np.random.default_rng(31)
    )
    before = {ep: table.priorities.copy() for ep, table in sampler._tables.items()}
    sampled = {
        (int(ep), int(pair))
        for ep, pair in zip(batch.hgr_episode_id, batch.hgr_pair_index)
    }
    td_abs_before = np.linspace(2.0, 9.0, num=8, dtype=np.float32)
    sampler.update_priorities(batch, td_abs_before)
    for episode_id, table in sampler._tables.items():
        for pair_index in range(table.pair_count):
            if (episode_id, pair_index) not in sampled:
                assert table.priorities[pair_index] == before[episode_id][pair_index]
        assert np.isclose(table.priority_sum, np.sum(table.priorities, dtype=np.float64))

    replay_state = replay.state_dict()
    sampler_state = sampler.state_dict()
    restored_replay = make_replay(8)
    restored_replay.load_state_dict(replay_state)
    restored_sampler = HGRSampler(config)
    restored_sampler.load_state_dict(sampler_state, restored_replay)
    assert restored_sampler.critic_updates == sampler.critic_updates
    assert restored_sampler.beta_progress() == sampler.beta_progress()
    assert restored_sampler.max_priority == sampler.max_priority
    for episode_id, table in sampler._tables.items():
        np.testing.assert_array_equal(restored_sampler._tables[episode_id].slots, table.slots)
        np.testing.assert_allclose(
            restored_sampler._tables[episode_id].priorities, table.priorities
        )

    # Adding episode 2 overwrites episode 0 and must remove its entire pair table.
    add_episode(replay, sampler, 2, 4)
    assert 0 not in sampler._tables
    assert set(sampler._tables) == {1, 2}
    assert np.allclose(sampler._tables[2].priorities, sampler.max_priority)


def main() -> int:
    replay = make_replay(32)
    config = make_config(32)
    sampler = HGRSampler(config)
    add_episode(replay, sampler, 0, 4)
    add_episode(replay, sampler, 1, 4)
    assert sampler._tables[0].pair_count == 10
    assert sampler._tables[1].pair_count == 10
    assert_future_pair_batch(replay, sampler)
    assert_empirical_probabilities(replay, sampler)
    assert_is_and_beta(replay, sampler)
    assert_uniform_reduction()
    assert_priority_update_checkpoint_and_overwrite()
    print("HGR pair-priority, probability, IS, beta, overwrite, and checkpoint tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
