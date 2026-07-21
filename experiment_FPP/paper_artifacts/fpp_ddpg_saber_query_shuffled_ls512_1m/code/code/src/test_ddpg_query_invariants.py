#!/usr/bin/env python3
from __future__ import annotations

import copy
import inspect

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.trainer import _validate_resume_query_mode, run_training
from metacog_lite_v0.weights import make_derangement
from test_support_query_modes import compute


def make_agent() -> DDPGAgent:
    return DDPGAgent(
        state_dim=28,
        action_dim=4,
        hidden_dim=16,
        actor_lr=0.0,
        critic_lr=0.0,
        gamma=0.98,
        tau=0.005,
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
    )


def test_single_critic_and_replay_action_regression() -> None:
    torch.manual_seed(7)
    agent = make_agent()
    payload_keys = set(agent.state_dict())
    assert "critic" in payload_keys and "critic_target" in payload_keys
    assert all("critic2" not in key and "twin" not in key for key in payload_keys)

    batch_size = 4
    state = torch.linspace(-0.3, 0.3, batch_size * 28).reshape(batch_size, 28)
    next_state = torch.flip(state, dims=(0,))
    action_exec = torch.tensor(
        [
            [-0.8, -0.4, 0.2, 0.6],
            [0.7, 0.1, -0.3, -0.9],
            [0.4, -0.5, 0.8, -0.2],
            [-0.1, 0.9, -0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    reward = torch.tensor([[0.0], [-1.0], [0.5], [-0.25]], dtype=torch.float32)
    nonterminal = torch.tensor([[1.0], [1.0], [0.0], [1.0]], dtype=torch.float32)
    omega_critic = torch.tensor([[0.5], [1.0], [1.5], [1.0]], dtype=torch.float32)
    omega_actor = torch.ones((batch_size, 1), dtype=torch.float32)

    with torch.no_grad():
        next_action = agent.actor_target(next_state)
        target = reward + agent.gamma * nonterminal * agent.critic_target(next_state, next_action)
        q = agent.critic(state, action_exec)
        expected_critic_loss = float((omega_critic * (q - target).pow(2)).mean())

    recorded_actions: list[torch.Tensor] = []

    def record_action(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        recorded_actions.append(args[1].detach().clone())

    hook = agent.critic.register_forward_pre_hook(record_action)
    metrics = agent.update(
        state=state,
        action_exec=action_exec,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=omega_critic,
        omega_actor=omega_actor,
    )
    hook.remove()

    assert len(recorded_actions) >= 2
    torch.testing.assert_close(recorded_actions[0], action_exec)
    assert abs(metrics.critic_loss - expected_critic_loss) < 1e-6


def test_query_weights_are_critic_only_and_detached() -> None:
    wb, _task, _policy, _batch = compute("shuffled_policy", cache_enabled=True)
    np.testing.assert_array_equal(wb.omega_actor, np.ones_like(wb.omega_actor))
    assert np.all(wb.omega_critic > 0.0)
    assert abs(float(np.mean(wb.omega_critic)) - 1.0) < 1e-6
    for value in (wb.w_p_raw, wb.omega_critic, wb.omega_actor, wb.psi_q_rank):
        tensor = torch.as_tensor(value)
        assert not tensor.requires_grad


def test_shuffled_weights_apply_to_same_index_residual() -> None:
    wb, _task, _policy, _batch = compute("shuffled_policy", cache_enabled=True)
    omega_critic = torch.as_tensor(wb.omega_critic[:, None], dtype=torch.float32)
    assert not np.allclose(wb.omega_critic, wb.omega_critic[0])

    torch.manual_seed(19)
    agent = make_agent()
    state = torch.linspace(-1.0, 1.0, 4 * 28).reshape(4, 28)
    next_state = torch.zeros_like(state)
    action_exec = torch.tensor(
        [
            [-0.9, -0.7, -0.5, -0.3],
            [-0.1, 0.1, 0.3, 0.5],
            [0.7, 0.9, -0.8, -0.6],
            [0.8, 0.6, 0.4, 0.2],
        ],
        dtype=torch.float32,
    )
    reward = torch.tensor([[-2.0], [-0.25], [1.5], [3.0]], dtype=torch.float32)
    nonterminal = torch.zeros((4, 1), dtype=torch.float32)
    with torch.no_grad():
        q = agent.critic(state, action_exec)
        residual_sq = (q - reward).pow(2)
        aligned_loss = float((omega_critic * residual_sq).mean())
        reversed_loss = float((torch.flip(omega_critic, dims=(0,)) * residual_sq).mean())
    assert abs(aligned_loss - reversed_loss) > 1e-5

    metrics = agent.update(
        state=state,
        action_exec=action_exec,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=omega_critic,
        omega_actor=torch.ones((4, 1), dtype=torch.float32),
    )
    assert abs(metrics.critic_loss - aligned_loss) < 1e-6

    trainer_source = inspect.getsource(run_training)
    compute_pos = trainer_source.index("wb = weight_computer.compute(batch, replay, agent, weight_rng, device)")
    action_pos = trainer_source.index("action_t = torch.as_tensor(batch.action_exec")
    update_pos = trainer_source.index("omega_critic=torch.as_tensor(wb.omega_critic")
    assert compute_pos < action_pos < update_pos


def test_weight_rng_checkpoint_reproduces_assignment() -> None:
    rng = np.random.default_rng(20001)
    first = make_derangement(256, rng)
    checkpoint_state = copy.deepcopy(rng.bit_generator.state)
    expected_after_resume = make_derangement(256, rng)

    resumed_rng = np.random.default_rng(999999)
    resumed_rng.bit_generator.state = copy.deepcopy(checkpoint_state)
    actual_after_resume = make_derangement(256, resumed_rng)

    assert np.all(first != np.arange(256))
    np.testing.assert_array_equal(expected_after_resume, actual_after_resume)
    assert resumed_rng.bit_generator.state == rng.bit_generator.state


def test_resume_mode_validation() -> None:
    shuffled = LiteConfig(support_query_mode="shuffled_policy")
    shuffled.normalize()
    current = LiteConfig(support_query_mode="current_policy")
    current.normalize()

    _validate_resume_query_mode(
        {"config": {"support_query_mode": "shuffled_policy"}},
        shuffled,
    )
    _validate_resume_query_mode({"config": {}}, current)

    try:
        _validate_resume_query_mode(
            {"config": {"support_query_mode": "current_policy"}},
            shuffled,
        )
    except ValueError as exc:
        assert "support_query_mode" in str(exc)
    else:
        raise AssertionError("resume query-mode mismatch did not fail")


def test_resume_validation_precedes_provenance_writes() -> None:
    source = inspect.getsource(run_training)
    validation = source.index("_validate_resume_query_mode(resume_payload, config)")
    mkdir = source.index("run_dir.mkdir")
    config_write = source.index("run_dir / \"config.json\"")
    assert validation < mkdir < config_write


def main() -> int:
    test_single_critic_and_replay_action_regression()
    test_query_weights_are_critic_only_and_detached()
    test_shuffled_weights_apply_to_same_index_residual()
    test_weight_rng_checkpoint_reproduces_assignment()
    test_resume_mode_validation()
    test_resume_validation_precedes_provenance_writes()
    print("test_ddpg_query_invariants.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
