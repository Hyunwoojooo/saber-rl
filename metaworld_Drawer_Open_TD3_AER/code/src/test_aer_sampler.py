from __future__ import annotations

import unittest

import numpy as np

from metacog_lite_v0.aer import (
    annealed_lambda,
    attentive_candidate_count,
    cosine_similarity,
    select_attentive_indices,
    top_k_positions,
)
from metacog_lite_v0.replay import HERReplayBuffer


def make_replay(size: int = 64) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=128, obs_dim=3, goal_dim=2, action_dim=1)
    for step in range(size):
        obs = np.asarray([step, step + 1, 1.0], dtype=np.float32)
        goal = np.asarray([0.25, -0.5], dtype=np.float32)
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=np.asarray([0.0], dtype=np.float32),
            reward_ext=-1.0,
            next_observation=obs + 0.1,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=50,
            episode_id=0,
            episode_step=step,
        )
    return replay


class AERSamplerTest(unittest.TestCase):
    def test_schedule_and_candidate_count(self) -> None:
        self.assertAlmostEqual(annealed_lambda(0, 2_000_000, 2.5, 2.0 / 3.0), 2.5)
        self.assertAlmostEqual(
            annealed_lambda(1_333_334, 2_000_000, 2.5, 2.0 / 3.0),
            1.0,
        )
        self.assertEqual(attentive_candidate_count(2.5, 256), 640)
        self.assertEqual(attentive_candidate_count(1.0, 256), 256)

    def test_top_k_and_zero_norm_are_finite(self) -> None:
        positions = top_k_positions(np.asarray([0.1, 0.9, 0.2, 0.8]), 2)
        np.testing.assert_array_equal(positions, np.asarray([1, 3]))
        similarity, norm = cosine_similarity(
            np.zeros(5, dtype=np.float32),
            np.zeros((4, 5), dtype=np.float32),
            1e-8,
        )
        self.assertEqual(norm, 0.0)
        self.assertTrue(np.all(np.isfinite(similarity)))

    def test_lambda_one_matches_uniform_sampler_and_rng(self) -> None:
        replay = make_replay()
        rng_uniform = np.random.default_rng(77)
        rng_aer = np.random.default_rng(77)
        uniform = replay.sample(16, 0.0, rng_uniform)
        selection = select_attentive_indices(
            replay,
            np.asarray([1.0, 2.0, 1.0, 0.25, -0.5], dtype=np.float32),
            16,
            1.0,
            rng_aer,
            1e-8,
        )
        attentive = replay.sample_from_indices(selection.selected_indices, 0.0, rng_aer)
        np.testing.assert_array_equal(uniform.indices, attentive.indices)
        np.testing.assert_array_equal(uniform.desired_goal_train, attentive.desired_goal_train)
        np.testing.assert_array_equal(
            rng_uniform.integers(0, 1_000_000, size=32),
            rng_aer.integers(0, 1_000_000, size=32),
        )
        self.assertEqual(attentive.her_relabel_ratio, 0.0)

    def test_ranked_selection_and_checkpoint_age(self) -> None:
        replay = make_replay()
        selection = select_attentive_indices(
            replay,
            np.asarray([63.0, 64.0, 1.0, 0.25, -0.5], dtype=np.float32),
            16,
            2.5,
            np.random.default_rng(12),
            1e-8,
        )
        selected = selection.selected_similarity
        unselected = np.delete(selection.candidate_similarity, selection.selected_positions)
        self.assertEqual(selection.candidate_indices.shape, (40,))
        self.assertEqual(selection.selected_indices.shape, (16,))
        self.assertGreaterEqual(float(np.min(selected)), float(np.max(unselected)))
        payload = replay.state_dict()
        restored = HERReplayBuffer(capacity=128, obs_dim=3, goal_dim=2, action_dim=1)
        restored.load_state_dict(payload)
        np.testing.assert_array_equal(
            replay.transition_ages(np.asarray([0, 63])),
            restored.transition_ages(np.asarray([0, 63])),
        )


if __name__ == "__main__":
    unittest.main()
