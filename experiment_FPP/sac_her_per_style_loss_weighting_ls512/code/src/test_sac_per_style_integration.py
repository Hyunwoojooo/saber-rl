#!/usr/bin/env python3
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.agent import SACAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.trainer import (
    _apply_td_or_per_loss_weights,
    make_agent,
    validate_resume_compatibility,
)


def make_config(**overrides: object) -> LiteConfig:
    values: dict[str, object] = {
        "task_name": "fetch_pick_and_place",
        "method": "sac_her_per_style_loss_weighting",
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
        "sac_per_priority_reduction": "min_abs",
        "per_alpha": 0.6,
        "eps_PER": 1e-6,
        "U_PER_max": 2.0,
        "weighting_mode": "batch_norm",
        "weight_scope": "critic_only",
        "timeout_bootstrap": True,
        "success_terminates": False,
    }
    values.update(overrides)
    cfg = LiteConfig(**values)
    cfg.normalize()
    return cfg


def make_sac_agent(cfg: LiteConfig) -> SACAgent:
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


def make_tensors() -> tuple[torch.Tensor, ...]:
    rng = np.random.default_rng(11)
    batch_size = 8
    return (
        torch.as_tensor(rng.normal(size=(batch_size, 7)), dtype=torch.float32),
        torch.as_tensor(
            rng.uniform(-1.0, 1.0, size=(batch_size, 2)),
            dtype=torch.float32,
        ),
        torch.as_tensor(rng.normal(size=batch_size), dtype=torch.float32),
        torch.as_tensor(rng.normal(size=(batch_size, 7)), dtype=torch.float32),
        torch.ones(batch_size, dtype=torch.float32),
    )


def new_weight_batch(batch_size: int) -> SimpleNamespace:
    ones = np.ones(batch_size, dtype=np.float32)
    return SimpleNamespace(
        w_p_raw=ones.copy(),
        omega_critic=ones.copy(),
        omega_actor=ones.copy(),
        extra_metrics={},
    )


def parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [value.detach().clone() for value in module.parameters()]


def any_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def assert_exact_priority_formula_and_target_reuse() -> None:
    torch.manual_seed(7)
    cfg = make_config()
    agent = make_sac_agent(cfg)
    state, action, reward, next_state, nonterminal = make_tensors()
    wb = new_weight_batch(state.shape[0])

    fixed_target = _apply_td_or_per_loss_weights(
        cfg,
        agent,
        wb,
        state,
        action,
        reward,
        next_state,
        nonterminal,
    )
    assert fixed_target is not None
    assert fixed_target.shape == reward.shape

    with torch.no_grad():
        q1 = agent.critic1(state, action)
        q2 = agent.critic2(state, action)
        reduced = torch.minimum(
            (q1 - fixed_target).abs(),
            (q2 - fixed_target).abs(),
        )
        priority = np.power(
            reduced.numpy() + cfg.eps_PER,
            cfg.per_alpha,
        ).astype(np.float32)
        expected_raw = np.clip(
            priority / max(cfg.eps, float(np.mean(priority))),
            0.0,
            cfg.U_PER_max,
        ).astype(np.float32)
        expected_omega = expected_raw / max(
            cfg.eps, float(np.mean(expected_raw))
        )
        expected_loss = (
            (
                torch.as_tensor(expected_omega)
                * (q1 - fixed_target).pow(2)
            ).mean()
            + (
                torch.as_tensor(expected_omega)
                * (q2 - fixed_target).pow(2)
            ).mean()
        )

    np.testing.assert_allclose(wb.w_p_raw, expected_raw, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        wb.omega_critic, expected_omega, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(wb.omega_actor, 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.mean(wb.omega_critic), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        wb.extra_metrics["per/critic_weight_mean"], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        wb.extra_metrics["per/actor_weight_mean"], 1.0, atol=1e-6
    )
    assert float(np.max(wb.w_p_raw)) <= cfg.U_PER_max + 1e-6
    for key in (
        "per/priority_entropy",
        "per/priority_ESS",
        "per/weight_mean",
        "per/weight_max",
        "per/weight_entropy",
        "per/weight_ESS",
        "per/critic_weight_mean",
        "per/critic_weight_max",
        "per/actor_weight_mean",
    ):
        assert key in wb.extra_metrics
        assert np.isfinite(wb.extra_metrics[key])

    actor_before = parameters(agent.actor)
    critic1_before = parameters(agent.critic1)
    critic2_before = parameters(agent.critic2)
    alpha_before = agent.log_alpha.detach().clone()

    # A different RNG state cannot change the critic target because the exact
    # target draw used to construct the weights is passed into update().
    torch.manual_seed(999)
    update = agent.update(
        state=state,
        action_exec=action,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=torch.as_tensor(wb.omega_critic),
        omega_actor=torch.as_tensor(wb.omega_actor),
        fixed_target=fixed_target,
    )
    np.testing.assert_allclose(
        update.critic_loss, expected_loss.item(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        update.target_mean, fixed_target.mean().item(), rtol=1e-6, atol=1e-6
    )
    assert update.td_error_abs_values is not None
    np.testing.assert_allclose(
        update.td_error_abs_values,
        reduced.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert any_changed(actor_before, agent.actor)
    assert any_changed(critic1_before, agent.critic1)
    assert any_changed(critic2_before, agent.critic2)
    assert not torch.equal(alpha_before, agent.log_alpha.detach())
    assert agent.total_it == 1

    payload = agent.state_dict()
    restored = make_sac_agent(cfg)
    restored.load_state_dict(payload)
    assert restored.total_it == agent.total_it
    assert restored.td_priority_reduction == "min_abs"
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


def assert_mean_abs_is_separate_sensitivity() -> None:
    torch.manual_seed(13)
    cfg = make_config(sac_per_priority_reduction="mean_abs")
    agent = make_sac_agent(cfg)
    state, action, reward, next_state, nonterminal = make_tensors()
    q1, q2, target, td1, td2, reduced = agent.td_error_components(
        state, action, reward, next_state, nonterminal
    )
    del q1, q2, target
    torch.testing.assert_close(reduced, 0.5 * (td1 + td2))
    assert agent.td_priority_reduction == "mean_abs"


def assert_zero_td_gives_uniform_weights() -> None:
    torch.manual_seed(17)
    cfg = make_config()
    agent = make_sac_agent(cfg)
    for critic in (agent.critic1, agent.critic2):
        for parameter in critic.parameters():
            parameter.data.zero_()
    state = torch.zeros((8, 7), dtype=torch.float32)
    action = torch.zeros((8, 2), dtype=torch.float32)
    reward = torch.zeros(8, dtype=torch.float32)
    next_state = torch.zeros((8, 7), dtype=torch.float32)
    nonterminal = torch.zeros(8, dtype=torch.float32)
    wb = new_weight_batch(8)
    fixed_target = _apply_td_or_per_loss_weights(
        cfg,
        agent,
        wb,
        state,
        action,
        reward,
        next_state,
        nonterminal,
    )
    assert fixed_target is not None
    torch.testing.assert_close(fixed_target, torch.zeros_like(fixed_target))
    np.testing.assert_allclose(wb.w_p_raw, 1.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(wb.omega_critic, 1.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(wb.omega_actor, 1.0, rtol=0.0, atol=0.0)


def assert_actor_and_temperature_ignore_replay_weights() -> None:
    torch.manual_seed(23)
    cfg = make_config()
    source = make_sac_agent(cfg)
    state, action, reward, next_state, nonterminal = make_tensors()
    fixed_target = torch.linspace(-1.0, 1.0, steps=state.shape[0])
    payload = copy.deepcopy(source.state_dict())
    left = make_sac_agent(cfg)
    right = make_sac_agent(cfg)
    left.load_state_dict(copy.deepcopy(payload))
    right.load_state_dict(copy.deepcopy(payload))
    critic_weight = torch.linspace(0.5, 1.5, steps=state.shape[0])

    torch.manual_seed(1234)
    left_update = left.update(
        state,
        action,
        reward,
        next_state,
        nonterminal,
        critic_weight,
        torch.ones(state.shape[0]),
        fixed_target=fixed_target,
    )
    torch.manual_seed(1234)
    right_update = right.update(
        state,
        action,
        reward,
        next_state,
        nonterminal,
        critic_weight,
        torch.full((state.shape[0],), 1e6),
        fixed_target=fixed_target,
    )
    for left_parameter, right_parameter in zip(
        left.actor.parameters(), right.actor.parameters()
    ):
        torch.testing.assert_close(left_parameter, right_parameter)
    torch.testing.assert_close(left.log_alpha, right.log_alpha)
    np.testing.assert_allclose(
        left_update.actor_loss, right_update.actor_loss, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        left_update.sac_alpha_loss,
        right_update.sac_alpha_loss,
        rtol=0.0,
        atol=0.0,
    )


def assert_resume_compatibility_gate() -> None:
    cfg = make_config(total_steps=1_000, max_episode_steps=50)
    checkpoint_config = cfg.as_dict()
    validate_resume_compatibility(cfg, checkpoint_config, 500)

    bad_method = copy.deepcopy(checkpoint_config)
    bad_method["method"] = "sac_hgr"
    try:
        validate_resume_compatibility(cfg, bad_method, 500)
    except ValueError as exc:
        assert "method:" in str(exc)
    else:
        raise AssertionError("cross-method resume was accepted")

    bad_seed = copy.deepcopy(checkpoint_config)
    bad_seed["seeds"]["torch_seed"] += 1
    try:
        validate_resume_compatibility(cfg, bad_seed, 500)
    except ValueError as exc:
        assert "seeds:" in str(exc)
    else:
        raise AssertionError("cross-seed resume was accepted")

    try:
        validate_resume_compatibility(cfg, checkpoint_config, 501)
    except ValueError as exc:
        assert "episode boundary" in str(exc)
    else:
        raise AssertionError("mid-episode resume was accepted")


def main() -> int:
    assert_exact_priority_formula_and_target_reuse()
    assert_mean_abs_is_separate_sensitivity()
    assert_zero_td_gives_uniform_weights()
    assert_actor_and_temperature_ignore_replay_weights()
    assert_resume_compatibility_gate()
    print(
        "SAC PER-style formula, exact target reuse, alpha, and checkpoint tests passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
