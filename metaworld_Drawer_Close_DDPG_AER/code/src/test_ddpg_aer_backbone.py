from __future__ import annotations

import unittest

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent


class DDPGAERBackboneTest(unittest.TestCase):
    def make_agent(self) -> DDPGAgent:
        return DDPGAgent(
            state_dim=5,
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
            policy_delay=1,
            target_policy_noise=0.0,
            target_noise_clip=0.0,
        )

    def test_single_critic_and_learning_rate_schedule(self) -> None:
        agent = self.make_agent()
        self.assertTrue(hasattr(agent, "critic"))
        self.assertTrue(hasattr(agent, "critic_target"))
        self.assertFalse(hasattr(agent, "critic2"))
        agent.set_learning_rates(actor_lr=7.5e-4, critic_lr=7.5e-4)
        actor_lr, critic_lr = agent.learning_rates()
        self.assertAlmostEqual(actor_lr, 7.5e-4)
        self.assertAlmostEqual(critic_lr, 7.5e-4)

    def test_actor_updates_on_each_critic_update(self) -> None:
        torch.manual_seed(7)
        agent = self.make_agent()
        state = torch.randn(8, 5)
        action = torch.tanh(torch.randn(8, 2))
        reward = torch.randn(8)
        next_state = torch.randn(8, 5)
        nonterminal = torch.ones(8)
        weights = torch.ones(8)
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        agent.update(state, action, reward, next_state, nonterminal, weights, weights)
        after = list(agent.actor.parameters())
        self.assertEqual(agent.total_it, 1)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
