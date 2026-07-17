from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "code" / "src"))

from metacog_lite_v0.aer import (
    TD3_AER,
    TD3_ANNEALING_ER,
    TD3_ER,
    attentive_lambda,
    candidate_count,
    effective_learning_rates,
)
from metacog_lite_v0.agent import TD3Agent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer


def make_replay() -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=128, obs_dim=2, goal_dim=1, action_dim=1)
    for i in range(100):
        observation = np.asarray([i / 100.0, (100 - i) / 100.0], dtype=np.float32)
        goal = np.asarray([(i % 5) / 5.0], dtype=np.float32)
        replay.add_transition(
            observation=observation,
            achieved_goal=np.asarray([i / 100.0], dtype=np.float32),
            desired_goal=goal,
            action_exec=np.asarray([(i % 7) / 7.0], dtype=np.float32),
            reward_ext=0.0 if i % 13 == 0 else -1.0,
            next_observation=observation + 0.001,
            next_achieved_goal=np.asarray([(i + 1) / 100.0], dtype=np.float32),
            done=False,
            timeout=False,
            h=200 - (i % 10),
            episode_id=i // 10,
            episode_step=i % 10,
        )
    return replay


class AERScheduleTest(unittest.TestCase):
    def test_schedule_and_candidate_count(self) -> None:
        total_steps = 2_000_000
        self.assertAlmostEqual(attentive_lambda(0, total_steps, 2.5, 2.0 / 3.0), 2.5)
        self.assertEqual(candidate_count(256, 2.5), 640)
        self.assertAlmostEqual(
            attentive_lambda(1_333_334, total_steps, 2.5, 2.0 / 3.0),
            1.0,
        )
        self.assertAlmostEqual(attentive_lambda(total_steps, total_steps, 2.5, 2.0 / 3.0), 1.0)

    def test_lr_control_separation(self) -> None:
        self.assertEqual(effective_learning_rates(TD3_ER, 3e-4, 3e-4, 2.5), (3e-4, 3e-4))
        for method in (TD3_ANNEALING_ER, TD3_AER):
            actor_lr, critic_lr = effective_learning_rates(method, 3e-4, 3e-4, 2.5)
            self.assertAlmostEqual(actor_lr, 7.5e-4)
            self.assertAlmostEqual(critic_lr, 7.5e-4)


class AERReplayTest(unittest.TestCase):
    def test_lambda_one_matches_uniform_sample_exactly(self) -> None:
        replay = make_replay()
        uniform_rng = np.random.default_rng(17)
        attentive_rng = np.random.default_rng(17)
        uniform = replay.sample(16, her_ratio=0.0, rng=uniform_rng)
        attentive, info = replay.sample_attentive(
            batch_size=16,
            her_ratio=0.0,
            query_state=np.asarray([0.4, 0.6, 0.0], dtype=np.float32),
            lambda_value=1.0,
            similarity_eps=1e-8,
            rng=attentive_rng,
        )
        np.testing.assert_array_equal(attentive.indices, uniform.indices)
        np.testing.assert_array_equal(attentive.observation, uniform.observation)
        np.testing.assert_array_equal(attentive.desired_goal_train, uniform.desired_goal_train)
        self.assertEqual(info.candidate_count, 16)
        self.assertEqual(attentive.her_relabel_ratio, 0.0)

    def test_raw_cosine_top_k_is_exact_and_no_her(self) -> None:
        replay = make_replay()
        query = np.asarray([0.9, 0.1, 0.2], dtype=np.float32)
        expected_rng = np.random.default_rng(23)
        candidate_indices = expected_rng.integers(0, replay.size, size=40, endpoint=False, dtype=np.int64)
        states = np.concatenate(
            [replay.observation[candidate_indices], replay.desired_goal[candidate_indices]],
            axis=1,
        )
        cosine = (states @ query) / (
            np.maximum(np.linalg.norm(states, axis=1), 1e-8) * max(np.linalg.norm(query), 1e-8)
        )
        expected_positions = np.argsort(-cosine, kind="stable")[:16]
        expected_indices = candidate_indices[expected_positions]

        batch, info = replay.sample_attentive(
            batch_size=16,
            her_ratio=0.0,
            query_state=query,
            lambda_value=2.5,
            similarity_eps=1e-8,
            rng=np.random.default_rng(23),
        )
        np.testing.assert_array_equal(info.candidate_indices, candidate_indices)
        np.testing.assert_array_equal(batch.indices, expected_indices)
        np.testing.assert_array_equal(batch.desired_goal_train, replay.desired_goal[batch.indices])
        self.assertGreaterEqual(float(np.mean(info.selected_similarity)), float(np.mean(info.candidate_similarity)))
        self.assertEqual(batch.her_relabel_ratio, 0.0)


class AERInvariantTest(unittest.TestCase):
    def test_config_rejects_her_and_loss_weighting(self) -> None:
        with self.assertRaisesRegex(ValueError, "her_ratio=0"):
            LiteConfig(
                task_name="drawer_close",
                method=TD3_AER,
                her_ratio=0.8,
                weight_scope="none",
            ).normalize()
        with self.assertRaisesRegex(ValueError, "weight_scope='none'"):
            LiteConfig(
                task_name="drawer_close",
                method=TD3_AER,
                her_ratio=0.0,
                weight_scope="critic_only",
            ).normalize()

    def test_td3_lr_setter_preserves_optimizer(self) -> None:
        agent = TD3Agent(
            state_dim=3,
            action_dim=1,
            hidden_dim=8,
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.98,
            tau=0.005,
            action_low=np.asarray([-1.0], dtype=np.float32),
            action_high=np.asarray([1.0], dtype=np.float32),
            device=torch.device("cpu"),
            l2_action_penalty=0.0,
        )
        actor_opt_id = id(agent.actor_opt)
        critic_opt_id = id(agent.critic_opt)
        agent.set_learning_rates(7.5e-4, 6e-4)
        self.assertEqual(agent.learning_rates(), (7.5e-4, 6e-4))
        self.assertEqual(id(agent.actor_opt), actor_opt_id)
        self.assertEqual(id(agent.critic_opt), critic_opt_id)


if __name__ == "__main__":
    unittest.main()
