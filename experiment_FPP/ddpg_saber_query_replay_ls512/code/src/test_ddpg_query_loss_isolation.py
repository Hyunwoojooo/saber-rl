#!/usr/bin/env python3
from __future__ import annotations

import copy

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent


def make_agent(seed: int) -> DDPGAgent:
    torch.manual_seed(seed)
    return DDPGAgent(
        state_dim=28,
        action_dim=4,
        hidden_dim=32,
        actor_lr=0.0,
        critic_lr=0.0,
        gamma=0.98,
        tau=0.0,
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
    )


def run_update(
    agent: DDPGAgent,
    state: torch.Tensor,
    action_exec: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    nonterminal: torch.Tensor,
    omega_critic: torch.Tensor,
) -> tuple[object, list[torch.Tensor]]:
    critic_actions: list[torch.Tensor] = []

    def capture_critic_action(_module, args, _output) -> None:
        critic_actions.append(args[1].detach().clone())

    handle = agent.critic.register_forward_hook(capture_critic_action)
    try:
        metrics = agent.update(
            state=state,
            action_exec=action_exec,
            reward=reward,
            next_state=next_state,
            nonterminal=nonterminal,
            omega_critic=omega_critic,
            omega_actor=torch.ones_like(omega_critic),
        )
    finally:
        handle.remove()
    return metrics, critic_actions


def main() -> int:
    rng = np.random.default_rng(917)
    batch_size = 4
    state = torch.as_tensor(rng.normal(size=(batch_size, 28)), dtype=torch.float32)
    next_state = torch.as_tensor(rng.normal(size=(batch_size, 28)), dtype=torch.float32)
    action_exec = torch.as_tensor(
        np.asarray(
            [
                [-0.9, -0.4, 0.1, 0.8],
                [-0.2, 0.7, -0.6, 0.3],
                [0.5, -0.8, 0.9, -0.1],
                [0.95, 0.2, -0.3, -0.7],
            ],
            dtype=np.float32,
        )
    )
    reward = torch.as_tensor([-1.0, -0.4, 0.2, 1.3], dtype=torch.float32)
    nonterminal = torch.as_tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    uniform_agent = make_agent(1234)
    weighted_agent = make_agent(9999)
    weighted_agent.load_state_dict(copy.deepcopy(uniform_agent.state_dict()))

    uniform = torch.ones(batch_size, dtype=torch.float32)
    nonuniform = torch.as_tensor([0.1, 0.1, 0.1, 3.7], dtype=torch.float32)
    assert torch.isclose(nonuniform.mean(), torch.tensor(1.0))

    uniform_metrics, uniform_calls = run_update(
        uniform_agent, state, action_exec, reward, next_state, nonterminal, uniform
    )
    weighted_metrics, weighted_calls = run_update(
        weighted_agent, state, action_exec, reward, next_state, nonterminal, nonuniform
    )

    # The first online-critic forward is Bellman regression and must receive the
    # executed replay action exactly; the later call evaluates actor actions.
    assert len(uniform_calls) == 2
    assert len(weighted_calls) == 2
    torch.testing.assert_close(uniform_calls[0], action_exec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weighted_calls[0], action_exec, rtol=0.0, atol=0.0)
    assert not torch.equal(uniform_calls[1], action_exec)

    # Query-derived critic weights alter only Bellman regression. With both
    # optimizers frozen and omega_actor=1, the actor objective is identical.
    assert abs(uniform_metrics.critic_loss - weighted_metrics.critic_loss) > 1e-6
    assert uniform_metrics.actor_loss == weighted_metrics.actor_loss

    print("test_ddpg_query_loss_isolation.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
