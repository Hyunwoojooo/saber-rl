from __future__ import annotations

import inspect
import unittest

from _source_import import add_fpp_ddpg

add_fpp_ddpg()
from metacog_lite_v0.agent import DDPGAgent  # noqa: E402


class ReplayActionRegressionTest(unittest.TestCase):
    def test_critic_regresses_on_replay_action_argument(self) -> None:
        source = inspect.getsource(DDPGAgent.update)
        self.assertIn("q = self.critic(state, action_exec)", source)
        self.assertIn("pred_action = self.actor(state)", source)
        self.assertIn("self.critic(state, pred_action)", source)


if __name__ == "__main__":
    unittest.main()
