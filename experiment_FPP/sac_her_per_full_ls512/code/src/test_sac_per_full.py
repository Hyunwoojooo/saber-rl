#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.agent import SACAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.trainer import _apply_per_full_loss_weights, make_agent, per_beta_at_step


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
    def test_sampling_probability_is_weights_and_bias(self) -> None:
        replay = make_replay(num_transitions=4)
        replay.priority[:4] = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        replay.max_priority = 8.0
        alpha = 0.6
        beta = 0.4
        expected_prob = np.power(replay.priority[:4].astype(np.float64), alpha)
        expected_prob /= expected_prob.sum()
        rng = np.random.default_rng(1234)
        counts = np.zeros(4, dtype=np.int64)
        saw_mixed_batch = False

        for _ in range(2000):
            batch = replay.sample_prioritized(4, 0.0, rng, alpha=alpha, beta=beta, eps=EPS)
            counts += np.bincount(batch.indices, minlength=4)
            np.testing.assert_allclose(batch.sampling_prob, expected_prob[batch.indices], rtol=1e-6)
            if np.unique(batch.indices).size > 1 and not saw_mixed_batch:
                expected_is = np.power(4 * expected_prob[batch.indices], -beta)
                expected_is /= expected_is.max()
                np.testing.assert_allclose(batch.is_weight, expected_is, rtol=1e-6)
                saw_mixed_batch = True

        self.assertTrue(saw_mixed_batch)
        frequency = counts / counts.sum()
        self.assertGreater(frequency[3], frequency[2])
        self.assertGreater(frequency[2], frequency[1])
        self.assertGreater(frequency[1], frequency[0])
        np.testing.assert_allclose(frequency, expected_prob, atol=0.02)

    def test_uniform_reduction_and_her_after_slot_sampling(self) -> None:
        replay = make_replay()
        rng = np.random.default_rng(8)
        uniform = replay.sample_prioritized(8, 0.0, rng, alpha=0.0, beta=0.0, eps=EPS)
        np.testing.assert_allclose(uniform.sampling_prob, np.full(8, 1.0 / replay.size))
        np.testing.assert_allclose(uniform.is_weight, np.ones(8))

        replay.priority[: replay.size] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        relabeled = replay.sample_prioritized(8, 1.0, rng, alpha=0.6, beta=0.4, eps=EPS)
        for out_index, slot in enumerate(relabeled.indices):
            slot_i = int(slot)
            expected_priority = replay.priority[slot_i]
            self.assertAlmostEqual(float(relabeled.sampling_priority[out_index]), float(expected_priority))
            episode_id = int(replay.episode_id[slot_i])
            episode_step = int(replay.episode_step[slot_i])
            future_slots = [
                candidate
                for candidate in replay._episode_to_slots[episode_id]
                if int(replay.episode_step[candidate]) > episode_step
            ]
            if future_slots:
                valid_goals = {float(replay.next_achieved_goal[candidate, 0]) for candidate in future_slots}
                self.assertIn(float(relabeled.desired_goal_train[out_index, 0]), valid_goals)
            else:
                np.testing.assert_allclose(
                    relabeled.desired_goal_train[out_index],
                    replay.desired_goal[slot_i],
                )

    def test_priority_update_insert_and_checkpoint_round_trip(self) -> None:
        replay = make_replay(num_transitions=4)
        replay.update_priorities(
            np.asarray([1, 1, 2], dtype=np.int64),
            np.asarray([0.2, 3.0, 1.0], dtype=np.float32),
            eps=EPS,
        )
        self.assertAlmostEqual(float(replay.priority[1]), 3.0 + EPS, places=5)
        self.assertAlmostEqual(float(replay.priority[2]), 1.0 + EPS, places=5)
        self.assertAlmostEqual(replay.max_priority, 3.0 + EPS, places=5)

        replay.add_transition(
            observation=np.zeros(2, dtype=np.float32),
            achieved_goal=np.zeros(1, dtype=np.float32),
            desired_goal=np.zeros(1, dtype=np.float32),
            action_exec=np.zeros(1, dtype=np.float32),
            reward_ext=-1.0,
            next_observation=np.ones(2, dtype=np.float32),
            next_achieved_goal=np.ones(1, dtype=np.float32),
            done=False,
            timeout=False,
            h=4,
            episode_id=2,
            episode_step=0,
            coarse_key=2,
            action_bin=0,
        )
        self.assertAlmostEqual(float(replay.priority[4]), replay.max_priority, places=5)
        with self.assertRaises(ValueError):
            replay.update_priorities(np.asarray([0, 1]), np.asarray([1.0]), eps=EPS)

        payload = copy.deepcopy(replay.state_dict())
        restored = HERReplayBuffer(capacity=16, obs_dim=2, goal_dim=1, action_dim=1)
        restored.load_state_dict(payload)
        np.testing.assert_array_equal(restored.priority, replay.priority)
        self.assertEqual(restored.ptr, replay.ptr)
        self.assertEqual(restored.size, replay.size)
        self.assertEqual(restored.version, replay.version)
        self.assertEqual(restored.max_priority, replay.max_priority)

        rng_a = np.random.default_rng(77)
        rng_b = np.random.default_rng()
        rng_b.bit_generator.state = copy.deepcopy(rng_a.bit_generator.state)
        batch_a = replay.sample_prioritized(4, 1.0, rng_a, 0.6, 0.4, EPS)
        batch_b = restored.sample_prioritized(4, 1.0, rng_b, 0.6, 0.4, EPS)
        np.testing.assert_array_equal(batch_a.indices, batch_b.indices)
        np.testing.assert_array_equal(batch_a.desired_goal_train, batch_b.desired_goal_train)
        np.testing.assert_array_equal(batch_a.is_weight, batch_b.is_weight)


class WeightAndScheduleTests(unittest.TestCase):
    def test_beta_schedule_and_batch_normalized_critic_weights(self) -> None:
        config = LiteConfig(
            method="sac_her_per_full",
            per_beta_start=0.4,
            per_beta_end=1.0,
            per_beta_steps=1_000_000,
            weighting_mode="batch_norm",
            weight_scope="critic_only",
        )
        config.normalize()
        self.assertAlmostEqual(per_beta_at_step(config, 0), 0.4)
        self.assertAlmostEqual(per_beta_at_step(config, 500_000), 0.7)
        self.assertAlmostEqual(per_beta_at_step(config, 1_000_000), 1.0)
        self.assertAlmostEqual(per_beta_at_step(config, 2_000_000), 1.0)

        replay = make_replay(num_transitions=4)
        replay.priority[:4] = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        batch = replay.sample_prioritized(
            4,
            0.0,
            np.random.default_rng(2),
            alpha=config.per_alpha,
            beta=0.7,
            eps=config.eps_PER,
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
        self.assertLessEqual(float(np.max(wb.w_p_raw)), 1.0)
        self.assertAlmostEqual(wb.extra_metrics["per/beta"], 0.7)


def make_sac_agent() -> SACAgent:
    return SACAgent(
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
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=3e-4,
        target_entropy=-2.0,
    )


class SACAgentTests(unittest.TestCase):
    def test_method_routes_to_sac_agent(self) -> None:
        config = LiteConfig(method="sac_her_per_full")
        config.normalize()
        agent = make_agent(
            config=config,
            state_dim=3,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            device=torch.device("cpu"),
        )
        self.assertIsInstance(agent, SACAgent)

    def test_same_stochastic_target_priority_loss_scale_and_checkpoint(self) -> None:
        torch.manual_seed(19)
        agent = make_sac_agent()
        state = torch.randn(5, 3)
        action = torch.tanh(torch.randn(5, 2))
        reward = torch.randn(5)
        next_state = torch.randn(5, 3)
        nonterminal = torch.ones(5)
        critic_weight = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5])
        actor_weight = torch.ones(5)

        rng_state = torch.get_rng_state()
        with torch.no_grad():
            next_action, next_log_prob = agent.actor.sample(
                next_state,
                agent.action_scale_t,
                agent.action_bias_t,
            )
            target_q = torch.minimum(
                agent.critic1_target(next_state, next_action),
                agent.critic2_target(next_state, next_action),
            ) - agent.alpha.detach() * next_log_prob
            target = reward + agent.gamma * nonterminal * target_q
            td1 = agent.critic1(state, action) - target
            td2 = agent.critic2(state, action) - target
            expected_priority = 0.5 * (td1.abs() + td2.abs())
            expected_diagnostic = torch.minimum(td1.abs(), td2.abs())
            expected_loss = (critic_weight * td1.pow(2)).mean() + (
                critic_weight * td2.pow(2)
            ).mean()
        torch.set_rng_state(rng_state)
        actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        alpha_before = float(agent.alpha.detach())
        update = agent.update(
            state,
            action,
            reward,
            next_state,
            nonterminal,
            critic_weight,
            actor_weight,
        )
        self.assertIsNotNone(update.priority_td_error_abs_values)
        np.testing.assert_allclose(
            update.priority_td_error_abs_values,
            expected_priority.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertAlmostEqual(update.critic_loss, float(expected_loss), places=5)
        self.assertAlmostEqual(update.td_error_abs_mean, float(expected_diagnostic.mean()), places=5)
        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(actor_before, agent.actor.parameters())
            )
        )
        self.assertNotEqual(alpha_before, float(agent.alpha.detach()))
        self.assertTrue(np.isfinite(update.sac_alpha))
        self.assertTrue(np.isfinite(update.sac_alpha_loss))
        self.assertTrue(np.isfinite(update.sac_log_prob_mean))

        restored = make_sac_agent()
        restored.load_state_dict(copy.deepcopy(agent.state_dict()))
        self.assertEqual(restored.total_it, agent.total_it)
        self.assertAlmostEqual(float(restored.alpha.detach()), float(agent.alpha.detach()), places=7)
        for expected, actual in zip(agent.actor.parameters(), restored.actor.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_actor_and_temperature_ignore_direct_is_actor_weights(self) -> None:
        torch.manual_seed(31)
        agent_a = make_sac_agent()
        agent_b = make_sac_agent()
        agent_b.load_state_dict(copy.deepcopy(agent_a.state_dict()))
        state = torch.randn(5, 3)
        action = torch.tanh(torch.randn(5, 2))
        reward = torch.randn(5)
        next_state = torch.randn(5, 3)
        nonterminal = torch.ones(5)
        critic_weight = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5])
        rng_state = torch.get_rng_state()
        update_a = agent_a.update(
            state, action, reward, next_state, nonterminal, critic_weight, torch.ones(5)
        )
        torch.set_rng_state(rng_state)
        update_b = agent_b.update(
            state,
            action,
            reward,
            next_state,
            nonterminal,
            critic_weight,
            torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0]),
        )
        self.assertTrue(
            all(torch.equal(left.detach(), right.detach()) for left, right in zip(agent_a.actor.parameters(), agent_b.actor.parameters()))
        )
        self.assertAlmostEqual(float(agent_a.alpha.detach()), float(agent_b.alpha.detach()), places=7)
        self.assertAlmostEqual(update_a.actor_loss, update_b.actor_loss, places=7)
        self.assertAlmostEqual(update_a.sac_alpha_loss, update_b.sac_alpha_loss, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
