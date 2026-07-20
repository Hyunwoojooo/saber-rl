#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.trainer import _apply_per_full_loss_weights, per_beta_at_step
from metacog_lite_v0.weights import LiteWeightComputer


METHOD = "ddpg_her_per_full"
EPS = 1e-6


def make_replay(num_transitions: int = 8, episode_length: int = 4) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=2, goal_dim=1, action_dim=1)
    for index in range(num_transitions):
        episode_id = index // episode_length
        episode_step = index % episode_length
        replay.add_transition(
            observation=np.asarray([index, -index], dtype=np.float32),
            achieved_goal=np.asarray([index], dtype=np.float32),
            desired_goal=np.asarray([-100 - episode_id], dtype=np.float32),
            action_exec=np.asarray([0.1 * index], dtype=np.float32),
            reward_ext=-1.0,
            next_observation=np.asarray([index + 1, -index - 1], dtype=np.float32),
            next_achieved_goal=np.asarray([index + 1], dtype=np.float32),
            done=episode_step == episode_length - 1,
            timeout=episode_step == episode_length - 1,
            h=episode_length - episode_step,
            episode_id=episode_id,
            episode_step=episode_step,
            coarse_key=episode_id,
            action_bin=index,
        )
    return replay


class PrioritizedReplayTests(unittest.TestCase):
    def test_sampling_distribution_and_importance_weights(self) -> None:
        replay = make_replay(num_transitions=4)
        replay.priority[:4] = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        replay.max_priority = 8.0
        expected = np.power(replay.priority[:4].astype(np.float64), 0.6)
        expected /= expected.sum()
        counts = np.zeros(4, dtype=np.int64)
        rng = np.random.default_rng(1234)
        for _ in range(2_000):
            batch = replay.sample_prioritized(4, 0.0, rng, alpha=0.6, beta=0.4, eps=EPS)
            counts += np.bincount(batch.indices, minlength=4)
            np.testing.assert_allclose(batch.sampling_prob, expected[batch.indices], rtol=1e-6)
            expected_is = np.power(4 * expected[batch.indices], -0.4)
            expected_is /= expected_is.max()
            np.testing.assert_allclose(batch.is_weight, expected_is, rtol=1e-6)
        frequency = counts / counts.sum()
        np.testing.assert_allclose(frequency, expected, atol=0.02)

    def test_future_her_occurs_after_priority_slot_sampling(self) -> None:
        replay = make_replay()
        replay.priority[: replay.size] = np.arange(1, replay.size + 1, dtype=np.float32)
        batch = replay.sample_prioritized(
            8, 1.0, np.random.default_rng(8), alpha=0.6, beta=0.4, eps=EPS
        )
        for row, slot in enumerate(batch.indices):
            slot_i = int(slot)
            self.assertAlmostEqual(
                float(batch.sampling_priority[row]), float(replay.priority[slot_i])
            )
            episode_id = int(replay.episode_id[slot_i])
            episode_step = int(replay.episode_step[slot_i])
            future_slots = [
                candidate
                for candidate in replay._episode_to_slots[episode_id]
                if int(replay.episode_step[candidate]) > episode_step
            ]
            if future_slots:
                future_goals = {
                    float(replay.next_achieved_goal[candidate, 0]) for candidate in future_slots
                }
                self.assertIn(float(batch.desired_goal_train[row, 0]), future_goals)
            else:
                np.testing.assert_allclose(
                    batch.desired_goal_train[row], replay.desired_goal[slot_i]
                )

    def test_duplicate_priority_update_and_checkpoint_round_trip(self) -> None:
        replay = make_replay(num_transitions=4)
        replay.update_priorities(
            np.asarray([1, 1, 2], dtype=np.int64),
            np.asarray([0.2, 3.0, 1.0], dtype=np.float32),
            eps=EPS,
        )
        self.assertAlmostEqual(float(replay.priority[1]), 3.0 + EPS, places=5)
        self.assertAlmostEqual(float(replay.priority[2]), 1.0 + EPS, places=5)
        payload = copy.deepcopy(replay.state_dict())
        restored = HERReplayBuffer(capacity=16, obs_dim=2, goal_dim=1, action_dim=1)
        restored.load_state_dict(payload)
        np.testing.assert_array_equal(restored.priority, replay.priority)
        self.assertEqual(restored.max_priority, replay.max_priority)


class WeightAndConfigurationTests(unittest.TestCase):
    def make_config(self) -> LiteConfig:
        config = LiteConfig(
            method=METHOD,
            per_beta_start=0.4,
            per_beta_end=1.0,
            per_beta_steps=2_000_000,
            weighting_mode="batch_norm",
            weight_scope="critic_only",
        )
        config.normalize()
        return config

    def test_beta_schedule_and_critic_only_is_weights(self) -> None:
        config = self.make_config()
        self.assertAlmostEqual(per_beta_at_step(config, 0), 0.4)
        self.assertAlmostEqual(per_beta_at_step(config, 1_000_000), 0.7)
        self.assertAlmostEqual(per_beta_at_step(config, 2_000_000), 1.0)
        replay = make_replay(num_transitions=4)
        replay.priority[:4] = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        batch = replay.sample_prioritized(
            4, 0.0, np.random.default_rng(2), 0.6, 0.7, EPS
        )
        wb = SimpleNamespace(
            w_p_raw=np.ones(4, dtype=np.float32),
            omega_critic=np.ones(4, dtype=np.float32),
            omega_actor=np.ones(4, dtype=np.float32),
            extra_metrics={},
        )
        _apply_per_full_loss_weights(config, wb, batch)
        self.assertAlmostEqual(float(np.mean(wb.omega_critic)), 1.0, places=6)
        np.testing.assert_array_equal(wb.omega_actor, np.ones(4, dtype=np.float32))
        for key in (
            "per/beta",
            "per/sampling_prob_min",
            "per/sampling_prob_max",
            "per/priority_entropy",
            "per/weight_ESS",
            "per/sample_duplicate_rate",
        ):
            self.assertIn(key, wb.extra_metrics)
            self.assertTrue(np.isfinite(wb.extra_metrics[key]))

    def test_full_per_bypasses_support_query(self) -> None:
        config = self.make_config()
        computer = LiteWeightComputer.__new__(LiteWeightComputer)
        computer.cfg = config
        sentinel = object()
        computer._task_batch_diagnostics = lambda batch, size: (
            np.zeros(size, dtype=np.int64),
            {},
        )
        computer._constant_batch = lambda *args, **kwargs: sentinel
        batch = SimpleNamespace(observation=np.zeros((4, 2), dtype=np.float32))
        result = computer.compute(batch, None, None, np.random.default_rng(1), torch.device("cpu"))
        self.assertIs(result, sentinel)

    def test_invalid_full_per_scope_and_schedule_fail(self) -> None:
        with self.assertRaises(ValueError):
            LiteConfig(method=METHOD, weight_scope="actor_critic").normalize()
        with self.assertRaises(ValueError):
            LiteConfig(
                method=METHOD,
                weight_scope="critic_only",
                weighting_mode="raw",
            ).normalize()
        with self.assertRaises(ValueError):
            LiteConfig(
                method=METHOD,
                weight_scope="critic_only",
                per_beta_start=0.8,
                per_beta_end=0.4,
            ).normalize()


class AgentPriorityTests(unittest.TestCase):
    def test_priority_vector_matches_backbone_td_error(self) -> None:
        torch.manual_seed(19)
        agent = DDPGAgent(
            state_dim=3,
            action_dim=2,
            hidden_dim=16,
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.98,
            tau=0.005,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            device=torch.device("cpu"),
            l2_action_penalty=0.0,
            policy_delay=2,
            target_policy_noise=0.0,
            target_noise_clip=0.5,
        )
        state = torch.randn(5, 3)
        action = torch.tanh(torch.randn(5, 2))
        reward = torch.randn(5)
        next_state = torch.randn(5, 3)
        nonterminal = torch.ones(5)
        critic_weight = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5])
        actor_weight = torch.ones(5)

        with torch.no_grad():
            next_action = agent.actor_target(next_state)
            target = reward + agent.gamma * nonterminal * agent.critic_target(next_state, next_action)
            td_error = agent.critic(state, action) - target
            expected_priority = td_error.abs()
            expected_loss = (critic_weight * td_error.pow(2)).mean()
        update = agent.update(
            state,
            action,
            reward,
            next_state,
            nonterminal,
            critic_weight,
            actor_weight,
        )
        np.testing.assert_allclose(
            update.td_error_abs_values,
            expected_priority.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertAlmostEqual(update.critic_loss, float(expected_loss), places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
