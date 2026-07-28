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


def test_context_only_ddpg_critic_only_weighting_and_replay_action_regression() -> None:
    torch.manual_seed(9_001)
    weighted = make_agent()
    baseline = make_agent()
    baseline.load_state_dict(weighted.state_dict())

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
    omega_control = torch.tensor([0.5, 0.75, 1.25, 1.5], dtype=torch.float32)
    omega_ones = torch.ones(4, dtype=torch.float32)

    with torch.no_grad():
        next_action = weighted.actor_target(next_state)
        target = reward + weighted.gamma * nonterminal * weighted.critic_target(
            next_state, next_action
        )
        q_replay = weighted.critic(state, replay_action)
        td_squared = (q_replay - target).pow(2)
        expected_weighted_critic = float((omega_control * td_squared).mean().item())
        expected_baseline_critic = float(td_squared.mean().item())
        policy_action = weighted.actor(state)
        expected_actor = float(-weighted.critic(state, policy_action).mean().item())

    critic_action_calls: list[torch.Tensor] = []

    def capture_critic_action(_module, inputs):
        critic_action_calls.append(inputs[1].detach().clone())

    hook = weighted.critic.register_forward_pre_hook(capture_critic_action)
    weighted_metrics = weighted.update(
        state=state,
        action_exec=replay_action,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=omega_control,
        omega_actor=omega_ones,
    )
    hook.remove()
    baseline_metrics = baseline.update(
        state=state,
        action_exec=replay_action,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=omega_ones,
        omega_actor=omega_ones,
    )

    assert weighted_metrics.critic_loss == expected_weighted_critic
    assert baseline_metrics.critic_loss == expected_baseline_critic
    assert weighted_metrics.critic_loss != baseline_metrics.critic_loss
    assert weighted_metrics.actor_loss == expected_actor
    assert baseline_metrics.actor_loss == expected_actor
    assert weighted_metrics.actor_loss == baseline_metrics.actor_loss

    assert len(critic_action_calls) >= 2
    torch.testing.assert_close(critic_action_calls[0], replay_action, rtol=0.0, atol=0.0)
    assert not torch.equal(critic_action_calls[1], replay_action)
    for weighted_parameter, baseline_parameter in zip(
        weighted.actor.parameters(), baseline.actor.parameters()
    ):
        torch.testing.assert_close(
            weighted_parameter,
            baseline_parameter,
            rtol=0.0,
            atol=0.0,
        )


def main() -> int:
    test_context_only_ddpg_critic_only_weighting_and_replay_action_regression()
    print("test_context_count_only_ddpg_loss_scope.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
