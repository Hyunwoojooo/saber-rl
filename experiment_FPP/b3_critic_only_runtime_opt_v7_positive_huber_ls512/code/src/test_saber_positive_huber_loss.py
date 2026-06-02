from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if (ROOT / "metacog_lite_v0").exists():
    sys.path.insert(0, str(ROOT))
else:
    sys.path.insert(0, str(ROOT.parent / "code" / "src"))

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.losses import compute_saber_critic_loss, compute_td_loss_vector


def test_mse_batch_norm_default_formula() -> None:
    td = torch.tensor([1.0, -2.0, 3.0])
    omega = torch.tensor([1.0, 0.5, 2.0])
    parts = compute_saber_critic_loss(td, omega, None, critic_td_loss="mse", saber_loss_mode="batch_norm")
    expected = (omega * td.pow(2)).mean()
    assert torch.allclose(parts.total, expected)


def test_huber_vector_manual_values() -> None:
    td = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    actual = compute_td_loss_vector(td, "huber", 1.0)
    expected = torch.tensor([1.5, 0.125, 0.0, 0.125, 1.5])
    assert torch.allclose(actual, expected)


def test_positive_bonus_zero_reliability_keeps_base() -> None:
    td = torch.tensor([1.0, 2.0, 4.0])
    score = torch.zeros_like(td)
    parts = compute_saber_critic_loss(
        td,
        torch.ones_like(td),
        score,
        critic_td_loss="huber",
        huber_delta=1.0,
        saber_loss_mode="positive_huber_bonus",
        beta_eff=3.0,
    )
    assert torch.allclose(parts.total, parts.base)
    assert torch.allclose(parts.support_bonus, torch.tensor(0.0))


def test_positive_bonus_all_ones_scales_base() -> None:
    td = torch.tensor([0.5, 2.0, -3.0])
    score = torch.ones_like(td)
    beta = 0.75
    parts = compute_saber_critic_loss(
        td,
        torch.ones_like(td),
        score,
        critic_td_loss="huber",
        huber_delta=1.0,
        saber_loss_mode="positive_huber_bonus",
        beta_eff=beta,
    )
    assert torch.allclose(parts.support_bonus, parts.base)
    assert torch.allclose(parts.total, parts.base * (1.0 + beta))


def test_positive_bonus_mixed_reliability_manual() -> None:
    td = torch.tensor([0.5, 2.0, -3.0, 0.0])
    score = torch.tensor([1.0, 0.0, 0.25, 1.0])
    loss_vec = compute_td_loss_vector(td, "huber", 1.0)
    beta = 2.0
    parts = compute_saber_critic_loss(
        td,
        torch.ones_like(td),
        score,
        critic_td_loss="huber",
        huber_delta=1.0,
        saber_loss_mode="positive_huber_bonus",
        beta_eff=beta,
    )
    expected = loss_vec.mean() + beta * (score * loss_vec).mean()
    assert torch.allclose(parts.total, expected)


def test_unsupported_samples_are_in_base_loss() -> None:
    td = torch.tensor([10.0, 2.0])
    score = torch.tensor([0.0, 1.0])
    parts = compute_saber_critic_loss(
        td,
        torch.ones_like(td),
        score,
        critic_td_loss="mse",
        saber_loss_mode="positive_huber_bonus",
        beta_eff=1.0,
    )
    assert torch.allclose(parts.base, torch.tensor((100.0 + 4.0) / 2.0))
    assert torch.allclose(parts.support_bonus, torch.tensor(4.0 / 2.0))


def test_actor_loss_unchanged_when_critic_loss_equivalent() -> None:
    torch.manual_seed(7)
    action_low = np.full(2, -1.0, dtype=np.float32)
    action_high = np.full(2, 1.0, dtype=np.float32)
    agent_a = DDPGAgent(5, 2, 16, 1e-3, 1e-3, 0.98, 0.005, action_low, action_high, torch.device("cpu"), 0.0)
    agent_b = DDPGAgent(5, 2, 16, 1e-3, 1e-3, 0.98, 0.005, action_low, action_high, torch.device("cpu"), 0.0)
    agent_b.load_state_dict(copy.deepcopy(agent_a.state_dict()))
    state = torch.randn(4, 5)
    action = torch.randn(4, 2).clamp(-1.0, 1.0)
    reward = torch.randn(4)
    next_state = torch.randn(4, 5)
    nonterminal = torch.ones(4)
    omega = torch.ones(4)
    score = torch.tensor([1.0, 0.0, 0.5, 0.0])
    update_a = agent_a.update(state, action, reward, next_state, nonterminal, omega, omega)
    update_b = agent_b.update(
        state,
        action,
        reward,
        next_state,
        nonterminal,
        omega,
        omega,
        reliability_score=score,
        support_bonus_beta_eff=0.0,
        critic_td_loss="mse",
        saber_loss_mode="positive_huber_bonus",
    )
    assert abs(update_a.actor_loss - update_b.actor_loss) < 1e-6


if __name__ == "__main__":
    test_mse_batch_norm_default_formula()
    test_huber_vector_manual_values()
    test_positive_bonus_zero_reliability_keeps_base()
    test_positive_bonus_all_ones_scales_base()
    test_positive_bonus_mixed_reliability_manual()
    test_unsupported_samples_are_in_base_loss()
    test_actor_loss_unchanged_when_critic_loss_equivalent()
    print("test_saber_positive_huber_loss.py passed")
