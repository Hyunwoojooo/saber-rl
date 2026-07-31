#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent


def make_agent() -> DDPGAgent:
    return DDPGAgent(
        state_dim=5,
        action_dim=2,
        hidden_dim=16,
        actor_lr=0.0,
        critic_lr=0.0,
        gamma=0.98,
        tau=0.0,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
    )


def test_ddpg_actor_scope_loss_assignment() -> None:
    torch.manual_seed(9_103)
    baseline = make_agent()
    actor_only = make_agent()
    actor_critic = make_agent()
    actor_only.load_state_dict(baseline.state_dict())
    actor_critic.load_state_dict(baseline.state_dict())

    state = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [-0.2, 0.7, -0.1, 0.9, 0.3],
            [0.8, -0.4, 0.2, -0.5, 0.6],
            [-0.9, 0.1, 0.5, 0.2, -0.3],
        ],
        dtype=torch.float32,
    )
    replay_action = torch.tensor(
        [[-0.8, 0.7], [0.6, -0.4], [-0.3, -0.9], [0.9, 0.2]],
        dtype=torch.float32,
    )
    next_state = torch.flip(state, dims=[0])
    reward = torch.tensor([-1.0, 0.0, 0.5, -0.25], dtype=torch.float32)
    nonterminal = torch.ones(4, dtype=torch.float32)
    omega = torch.tensor([0.4, 0.8, 1.2, 1.6], dtype=torch.float32)
    ones = torch.ones(4, dtype=torch.float32)

    with torch.no_grad():
        next_action = baseline.actor_target(next_state)
        target = reward + baseline.gamma * nonterminal * baseline.critic_target(
            next_state, next_action
        )
        td_squared = (baseline.critic(state, replay_action) - target).pow(2)
        q_policy = baseline.critic(state, baseline.actor(state))
        expected_critic_unweighted = float(td_squared.mean().item())
        expected_critic_weighted = float((omega * td_squared).mean().item())
        expected_actor_unweighted = float(-q_policy.mean().item())
        expected_actor_weighted = float(-(omega * q_policy).mean().item())

    baseline_result = baseline.update(
        state,
        replay_action,
        reward,
        next_state,
        nonterminal,
        omega_critic=ones,
        omega_actor=ones,
    )
    actor_only_result = actor_only.update(
        state,
        replay_action,
        reward,
        next_state,
        nonterminal,
        omega_critic=ones,
        omega_actor=omega,
    )
    actor_critic_result = actor_critic.update(
        state,
        replay_action,
        reward,
        next_state,
        nonterminal,
        omega_critic=omega,
        omega_actor=omega,
    )

    np.testing.assert_allclose(baseline_result.critic_loss, expected_critic_unweighted)
    np.testing.assert_allclose(actor_only_result.critic_loss, expected_critic_unweighted)
    np.testing.assert_allclose(actor_critic_result.critic_loss, expected_critic_weighted)
    np.testing.assert_allclose(baseline_result.actor_loss, expected_actor_unweighted)
    np.testing.assert_allclose(actor_only_result.actor_loss, expected_actor_weighted)
    np.testing.assert_allclose(actor_critic_result.actor_loss, expected_actor_weighted)
    assert not np.isclose(expected_critic_weighted, expected_critic_unweighted)
    assert not np.isclose(expected_actor_weighted, expected_actor_unweighted)


def main() -> int:
    test_ddpg_actor_scope_loss_assignment()
    print("test_actor_scope_ddpg.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
