#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch

from metacog_lite_v0.agent import SACAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.hgr import HGRSampler
from metacog_lite_v0.trainer import make_agent

from test_hgr_tdpriority_sampling import FakeGoalEnv, add_episode, make_replay


def make_sac_config(**overrides: object) -> LiteConfig:
    values: dict[str, object] = {
        "task_name": "fetch_pick_and_place",
        "method": "sac_hgr",
        "total_steps": 1_000,
        "warmup_steps": 0,
        "updates_per_step": 1,
        "batch_size": 8,
        "buffer_capacity": 32,
        "her_ratio": 0.8,
        "hidden_dim": 32,
        "sac_alpha": 0.2,
        "sac_auto_alpha": True,
        "sac_alpha_lr": 3e-4,
        "sac_target_entropy": None,
        "hgr_priority_eps": 1e-6,
        "hgr_priority_init": 1.0,
        "hgr_priority_clip_max": 0.0,
        "hgr_alpha_episode": 0.6,
        "hgr_alpha_goal": 0.6,
        "hgr_beta_episode_initial": 0.4,
        "hgr_beta_goal_initial": 0.4,
        "hgr_beta_episode_final": 1.0,
        "hgr_beta_goal_final": 1.0,
        "hgr_beta_anneal_updates": 100,
        "hgr_uniform_mix": 0.0,
        "hgr_sac_priority_reduction": "min_abs",
        "timeout_bootstrap": True,
        "success_terminates": False,
    }
    values.update(overrides)
    cfg = LiteConfig(**values)
    cfg.normalize()
    return cfg


def make_agent_from_config(cfg: LiteConfig) -> SACAgent:
    agent = make_agent(
        config=cfg,
        state_dim=7,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert isinstance(agent, SACAgent)
    return agent


def parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [value.detach().clone() for value in module.parameters()]


def any_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def assert_sac_dispatch_priority_weighting_and_alpha() -> None:
    torch.manual_seed(7)
    cfg = make_sac_config()
    agent = make_agent_from_config(cfg)
    assert agent.td_priority_reduction == "min_abs"
    assert agent.critic1 is not agent.critic2
    assert agent.auto_alpha
    assert agent.target_entropy == -2.0

    rng = np.random.default_rng(11)
    batch_size = 8
    state = torch.as_tensor(rng.normal(size=(batch_size, 7)), dtype=torch.float32)
    action = torch.as_tensor(
        rng.uniform(-1.0, 1.0, size=(batch_size, 2)), dtype=torch.float32
    )
    reward = torch.as_tensor(rng.normal(size=batch_size), dtype=torch.float32)
    next_state = torch.as_tensor(
        rng.normal(size=(batch_size, 7)), dtype=torch.float32
    )
    nonterminal = torch.ones(batch_size, dtype=torch.float32)
    critic_weight = torch.linspace(0.2, 1.0, steps=batch_size)
    actor_weight = torch.linspace(10.0, 80.0, steps=batch_size)
    fixed_target = torch.as_tensor(rng.normal(size=batch_size), dtype=torch.float32)

    with torch.no_grad():
        q1 = agent.critic1(state, action)
        q2 = agent.critic2(state, action)
        expected_priority = torch.minimum(
            (q1 - fixed_target).abs(), (q2 - fixed_target).abs()
        )
        expected_loss = (
            (critic_weight * (q1 - fixed_target).pow(2)).mean()
            + (critic_weight * (q2 - fixed_target).pow(2)).mean()
        )

    actor_before = parameters(agent.actor)
    critic1_before = parameters(agent.critic1)
    critic2_before = parameters(agent.critic2)
    target_before = parameters(agent.critic1_target)
    alpha_before = agent.log_alpha.detach().clone()
    update = agent.update(
        state,
        action,
        reward,
        next_state,
        nonterminal,
        critic_weight,
        actor_weight,
        fixed_target=fixed_target,
    )

    assert update.td_error_abs_values is not None
    np.testing.assert_allclose(
        update.td_error_abs_values,
        expected_priority.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        update.critic_loss, expected_loss.item(), rtol=1e-6, atol=1e-6
    )
    assert agent.total_it == 1
    assert any_changed(actor_before, agent.actor)
    assert any_changed(critic1_before, agent.critic1)
    assert any_changed(critic2_before, agent.critic2)
    assert any_changed(target_before, agent.critic1_target)
    assert not torch.equal(alpha_before, agent.log_alpha.detach())
    assert np.isfinite(update.sac_alpha)
    assert np.isfinite(update.sac_alpha_loss)
    assert np.isfinite(update.sac_log_prob_mean)

    payload = agent.state_dict()
    restored = make_agent_from_config(cfg)
    restored.load_state_dict(payload)
    assert restored.total_it == agent.total_it
    assert restored.td_priority_reduction == "min_abs"
    assert restored.auto_alpha == agent.auto_alpha
    torch.testing.assert_close(restored.log_alpha, agent.log_alpha)
    for source, destination in (
        (agent.actor, restored.actor),
        (agent.critic1, restored.critic1),
        (agent.critic2, restored.critic2),
        (agent.critic1_target, restored.critic1_target),
        (agent.critic2_target, restored.critic2_target),
    ):
        assert all(
            torch.equal(left.detach(), right.detach())
            for left, right in zip(source.parameters(), destination.parameters())
        )


def assert_mean_abs_sensitivity_is_explicit() -> None:
    torch.manual_seed(13)
    cfg = make_sac_config(hgr_sac_priority_reduction="mean_abs")
    agent = make_agent_from_config(cfg)
    state = torch.zeros((2, 7), dtype=torch.float32)
    action = torch.zeros((2, 2), dtype=torch.float32)
    target = torch.tensor([1.0, -1.0], dtype=torch.float32)
    with torch.no_grad():
        q1 = agent.critic1(state, action)
        q2 = agent.critic2(state, action)
        expected = 0.5 * ((q1 - target).abs() + (q2 - target).abs())
    update = agent.update(
        state,
        action,
        torch.zeros(2),
        state,
        torch.ones(2),
        torch.ones(2),
        torch.ones(2),
        fixed_target=target,
    )
    assert update.td_error_abs_values is not None
    np.testing.assert_allclose(update.td_error_abs_values, expected.numpy(), atol=1e-6)


def assert_real_hgr_batch_updates_sac_priorities() -> None:
    torch.manual_seed(19)
    cfg = make_sac_config()
    replay = make_replay(32)
    sampler = HGRSampler(cfg)
    add_episode(replay, sampler, 0, 4)
    add_episode(replay, sampler, 1, 4)
    batch = sampler.sample_batch(
        replay,
        batch_size=8,
        her_ratio=1.0,
        rng=np.random.default_rng(23),
    )
    critic_weight, actor_weight = sampler.loss_weights(batch)
    np.testing.assert_allclose(actor_weight, 1.0)
    assert np.all(np.isfinite(critic_weight))
    assert np.all(critic_weight > 0.0)
    assert float(np.max(critic_weight)) <= 1.0 + 1e-6

    agent = make_agent_from_config(cfg)
    reward = FakeGoalEnv().compute_reward(
        batch.next_achieved_goal, batch.desired_goal_train, {}
    ).astype(np.float32)
    state = np.concatenate(
        [batch.observation, batch.desired_goal_train], axis=1
    ).astype(np.float32)
    next_state = np.concatenate(
        [batch.next_observation, batch.desired_goal_train], axis=1
    ).astype(np.float32)
    update = agent.update(
        torch.as_tensor(state),
        torch.as_tensor(batch.action_exec),
        torch.as_tensor(reward),
        torch.as_tensor(next_state),
        torch.ones(8),
        torch.as_tensor(critic_weight),
        torch.as_tensor(actor_weight),
    )
    assert update.td_error_abs_values is not None
    before_updates = sampler.critic_updates
    metrics = sampler.update_priorities(batch, update.td_error_abs_values)
    assert sampler.critic_updates == before_updates + 1
    assert np.isfinite(metrics["hgr/update_priority_mean"])
    assert metrics["hgr/update_priority_mean"] >= cfg.hgr_priority_eps


def main() -> int:
    assert_sac_dispatch_priority_weighting_and_alpha()
    assert_mean_abs_sensitivity_is_explicit()
    assert_real_hgr_batch_updates_sac_priorities()
    print("SAC-HGR dispatch, twin priority, IS, alpha, and checkpoint tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
