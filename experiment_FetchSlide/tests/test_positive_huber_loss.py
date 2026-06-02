#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from saber_fetch_slide.positive_huber_loss import (  # noqa: E402
    td_loss_vector,
    update_ddpg_with_saber_loss,
)


class ConstantActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(state.shape[0], 1)


class LinearCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class TinyAgent:
    def __init__(self) -> None:
        torch.manual_seed(7)
        self.gamma = 0.98
        self.tau = 0.005
        self.l2_action_penalty = 0.0
        self.actor = ConstantActor()
        self.actor_target = ConstantActor()
        self.critic = LinearCritic()
        self.critic_target = LinearCritic()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=0.0)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=0.0)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)


def test_huber_vector_matches_manual_values() -> None:
    td = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32)
    expected = torch.tensor([1.5, 0.125, 0.0, 0.125, 1.5], dtype=torch.float32)
    actual = td_loss_vector(td, "huber", 1.0)
    assert torch.allclose(actual, expected)


def test_mse_batch_norm_vector_is_default_squared_td() -> None:
    td = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)
    assert torch.allclose(td_loss_vector(td, "mse", 1.0), td.pow(2))


def test_positive_bonus_manual_cases() -> None:
    td = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    loss_vec = td_loss_vector(td, "mse", 1.0)
    beta = 2.0
    base = loss_vec.mean()
    zeros = torch.zeros_like(loss_vec)
    ones = torch.ones_like(loss_vec)
    mixed = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    assert torch.allclose(base + beta * (zeros * loss_vec).mean(), base)
    assert torch.allclose(base + beta * (ones * loss_vec).mean(), base + beta * base)
    expected_mixed = base + beta * (mixed * loss_vec).mean()
    manual_mixed = torch.tensor((1.0 + 4.0 + 9.0) / 3.0 + 2.0 * (0.0 + 2.0 + 9.0) / 3.0)
    assert torch.allclose(expected_mixed, manual_mixed)


def test_positive_bonus_preserves_base_loss_for_unsupported_samples() -> None:
    td = torch.tensor([1.0, 10.0], dtype=torch.float32)
    rel = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss_vec = td_loss_vector(td, "huber", 1.0)
    total = loss_vec.mean() + (rel * loss_vec).mean()
    assert torch.allclose(loss_vec.mean(), torch.tensor((0.5 + 9.5) / 2.0))
    assert torch.allclose(total, torch.tensor(5.25))


def test_actor_loss_formula_does_not_use_reliability_score() -> None:
    state = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    next_state = state + 0.1
    action = torch.zeros(3, 1)
    reward = torch.zeros(3)
    nonterminal = torch.ones(3)
    omega = torch.ones(3)

    agent_a = TinyAgent()
    agent_b = TinyAgent()
    out_a = update_ddpg_with_saber_loss(
        agent_a,
        state,
        action,
        reward,
        next_state,
        nonterminal,
        omega,
        omega,
        torch.zeros(3),
        "huber",
        1.0,
        "positive_huber_bonus",
        1.0,
        True,
    )
    out_b = update_ddpg_with_saber_loss(
        agent_b,
        state,
        action,
        reward,
        next_state,
        nonterminal,
        omega,
        omega,
        torch.ones(3),
        "huber",
        1.0,
        "positive_huber_bonus",
        1.0,
        True,
    )
    assert abs(out_a.actor_loss - out_b.actor_loss) < 1e-7


if __name__ == "__main__":
    test_huber_vector_matches_manual_values()
    test_mse_batch_norm_vector_is_default_squared_td()
    test_positive_bonus_manual_cases()
    test_positive_bonus_preserves_base_loss_for_unsupported_samples()
    test_actor_loss_formula_does_not_use_reliability_score()
