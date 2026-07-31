from __future__ import annotations

import unittest

import numpy as np

from _source_import import add_fpp_ddpg

add_fpp_ddpg()
from metacog_lite_v0.replay import HERReplayBuffer  # noqa: E402


class HERFutureIndexingTest(unittest.TestCase):
    def test_later_transition_next_goal_is_used(self) -> None:
        replay = HERReplayBuffer(capacity=8, obs_dim=1, goal_dim=1, action_dim=1)
        for step in range(4):
            replay.add_transition(
                observation=np.asarray([step], dtype=np.float32),
                achieved_goal=np.asarray([10 + step], dtype=np.float32),
                desired_goal=np.asarray([-1], dtype=np.float32),
                action_exec=np.asarray([0], dtype=np.float32),
                reward_ext=-1.0,
                next_observation=np.asarray([step + 1], dtype=np.float32),
                next_achieved_goal=np.asarray([20 + step], dtype=np.float32),
                done=step == 3,
                timeout=False,
                h=4 - step,
                episode_id=3,
                episode_step=step,
                coarse_key=0,
                action_bin=0,
            )

        first_candidates = {
            float(replay._sample_future_goal(0, np.random.default_rng(seed))[0])
            for seed in range(32)
        }
        self.assertTrue(first_candidates.issubset({21.0, 22.0, 23.0}))
        self.assertNotIn(20.0, first_candidates)
        self.assertIsNone(replay._sample_future_goal(3, np.random.default_rng(0)))


if __name__ == "__main__":
    unittest.main()
