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
from saber_fetch_slide.td3_agent import TD3Agent  # noqa: E402


def _make_agent(policy_delay: int = 2) -> TD3Agent:
    torch.manual_seed(11)
    return TD3Agent(
        state_dim=3,
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
        policy_delay=policy_delay,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    )


def test_td3_actor_update_is_delayed() -> None:
    agent = _make_agent(policy_delay=2)
    state = torch.randn(8, 3)
    action = torch.zeros(8, 2)
    reward = torch.zeros(8)
    next_state = torch.randn(8, 3)
    nonterminal = torch.ones(8)
    omega = torch.ones(8)

    first = agent.update(state, action, reward, next_state, nonterminal, omega, omega)
    second = agent.update(state, action, reward, next_state, nonterminal, omega, omega)

    assert first.actor_updated == 0.0
    assert second.actor_updated == 1.0
    assert second.policy_delay == 2.0
    assert second.critic_q1_loss > 0.0
    assert second.critic_q2_loss > 0.0


def test_td3_q_values_use_conservative_minimum() -> None:
    agent = _make_agent(policy_delay=2)
    state = torch.randn(5, 3)
    action = torch.randn(5, 2).clamp(-1.0, 1.0)
    with torch.no_grad():
        q1 = agent.critic(state, action)
        q2 = agent.critic2(state, action)
        q_min = torch.minimum(q1, q2)
        q_values = agent.q_values(state, action)
    assert torch.allclose(q_values, q_min)


def test_td3_priority_td_errors_are_finite_batch_values() -> None:
    agent = _make_agent(policy_delay=2)
    state = torch.randn(7, 3)
    action = torch.randn(7, 2).clamp(-1.0, 1.0)
    reward = torch.randn(7)
    next_state = torch.randn(7, 3)
    nonterminal = torch.ones(7)
    td = agent.td_errors_for_priority(state, action, reward, next_state, nonterminal)
    assert td.shape == reward.shape
    assert torch.isfinite(td).all()
    assert (td >= 0.0).all()


def test_td3_fpp_compatible_td_error_abs_is_finite() -> None:
    agent = _make_agent(policy_delay=2)
    state = torch.randn(7, 3)
    action = torch.randn(7, 2).clamp(-1.0, 1.0)
    reward = torch.randn(7)
    next_state = torch.randn(7, 3)
    nonterminal = torch.ones(7)
    td = agent.td_error_abs(state, action, reward, next_state, nonterminal)
    assert td.shape == reward.shape
    assert torch.isfinite(td).all()
    assert (td >= 0.0).all()


def test_td3_saber_runtime_aliases_match_fpp_contract() -> None:
    for method in ("td3_saber", "td3_saber_runtime_opt", "td3_saber_critic_only_v4_ls512"):
        spec = METHODS[method]
        assert spec.lite_method == "b3_support_only"
        assert spec.weight_scope == "critic_only"
        assert spec.local_search_max == 512
        assert spec.agent_kind == "td3"
        assert spec.public_name in RUNTIME_OPT_METHODS


if __name__ == "__main__":
    test_td3_actor_update_is_delayed()
    test_td3_q_values_use_conservative_minimum()
    test_td3_priority_td_errors_are_finite_batch_values()
    test_td3_fpp_compatible_td_error_abs_is_finite()
    test_td3_saber_runtime_aliases_match_fpp_contract()
