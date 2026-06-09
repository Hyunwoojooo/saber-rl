#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "version_v0_lite" / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "code" / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from saber_fetch_slide.experiment import METHODS, RUNTIME_OPT_METHODS  # noqa: E402
from saber_fetch_slide.sac_agent import SACAgent  # noqa: E402


def _make_agent(auto_alpha: bool = True) -> SACAgent:
    torch.manual_seed(17)
    return SACAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=16,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.98,
        tau=0.005,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
        alpha=0.2,
        auto_alpha=auto_alpha,
        alpha_lr=1e-3,
        target_entropy=None,
    )


def test_sac_update_returns_finite_losses_and_alpha() -> None:
    agent = _make_agent(auto_alpha=True)
    state = torch.randn(8, 4)
    action = torch.randn(8, 2).clamp(-1.0, 1.0)
    reward = torch.zeros(8)
    next_state = torch.randn(8, 4)
    nonterminal = torch.ones(8)
    omega = torch.ones(8)

    out = agent.update(state, action, reward, next_state, nonterminal, omega, omega)
    assert out.critic_loss > 0.0
    assert np.isfinite(out.actor_loss)
    assert out.alpha > 0.0
    assert out.auto_alpha == 1.0
    assert out.target_entropy == -2.0


def test_sac_policy_actions_are_deterministic_mean_actions() -> None:
    agent = _make_agent(auto_alpha=False)
    state = torch.randn(5, 4)
    with torch.no_grad():
        a1 = agent.policy_actions(state)
        a2 = agent.policy_actions(state)
    assert torch.allclose(a1, a2)
    assert torch.all(a1 <= 1.0)
    assert torch.all(a1 >= -1.0)


def test_sac_priority_td_errors_are_finite_batch_values() -> None:
    agent = _make_agent(auto_alpha=True)
    state = torch.randn(7, 4)
    action = torch.randn(7, 2).clamp(-1.0, 1.0)
    reward = torch.randn(7)
    next_state = torch.randn(7, 4)
    nonterminal = torch.ones(7)
    td = agent.td_errors_for_priority(state, action, reward, next_state, nonterminal)
    assert td.shape == reward.shape
    assert torch.isfinite(td).all()
    assert (td >= 0.0).all()


def test_sac_fpp_compatible_td_error_abs_is_finite() -> None:
    agent = _make_agent(auto_alpha=True)
    state = torch.randn(7, 4)
    action = torch.randn(7, 2).clamp(-1.0, 1.0)
    reward = torch.randn(7)
    next_state = torch.randn(7, 4)
    nonterminal = torch.ones(7)
    td = agent.td_error_abs(state, action, reward, next_state, nonterminal)
    assert td.shape == reward.shape
    assert torch.isfinite(td).all()
    assert (td >= 0.0).all()


def test_sac_saber_runtime_aliases_match_fpp_contract() -> None:
    for method in ("sac_saber", "sac_saber_critic_only_v4_ls512"):
        spec = METHODS[method]
        assert spec.lite_method == "b3_support_only"
        assert spec.weight_scope == "critic_only"
        assert spec.local_search_max == 512
        assert spec.agent_kind == "sac"
        assert spec.public_name in RUNTIME_OPT_METHODS


if __name__ == "__main__":
    test_sac_update_returns_finite_losses_and_alpha()
    test_sac_policy_actions_are_deterministic_mean_actions()
    test_sac_priority_td_errors_are_finite_batch_values()
    test_sac_fpp_compatible_td_error_abs_is_finite()
    test_sac_saber_runtime_aliases_match_fpp_contract()
