#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec
from metacog_lite_v0.weights import GapStats, LiteWeightComputer, WeightBatch


def make_task() -> FetchPickAndPlaceTaskSpec:
    return FetchPickAndPlaceTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )


def make_observation(obj_x: float = 0.04, obj_z: float = 0.44, gap: float = 0.06) -> np.ndarray:
    observation = np.zeros(25, dtype=np.float32)
    observation[0:3] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    observation[3:6] = np.asarray([obj_x, 0.0, obj_z], dtype=np.float32)
    observation[9:11] = np.asarray([0.5 * gap, 0.5 * gap], dtype=np.float32)
    return observation


class StaticAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = torch.as_tensor(action[None, :], dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.action.to(state.device).repeat(state.shape[0], 1)


def make_config(weight_scope: str) -> LiteConfig:
    config = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=4,
        M_B=4,
        M_Q_min=2,
        N_local_min=2,
        local_search_max=8,
        action_bins_per_dim=5,
        lambda_support=1.0,
        policy_support_mode="bin",
        support_scan_mode="deterministic_tail_v1",
        weighting_mode="batch_norm",
        weight_scope=weight_scope,
    )
    config.normalize()
    # Scope routing must not depend on the optional Numba implementation.
    config.numba_support_scan = False
    config.support_predicate_early_exit = False
    config.support_diagnostic_audit_interval = 0
    return config


def make_replay(task: FetchPickAndPlaceTaskSpec, config: LiteConfig) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    policy_action = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    other_action = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    rows = [
        (make_observation(), policy_action),
        (make_observation(), other_action),
        (make_observation(), other_action),
        (make_observation(obj_x=0.8), other_action),
    ]
    for step, (observation, action) in enumerate(rows):
        replay.add_transition(
            observation=observation,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=action,
            reward_ext=0.0,
            next_observation=observation,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=20,
            episode_id=step,
            episode_step=0,
            coarse_key=task.coarse_hash(observation, 20),
            action_bin=int(task.action_bin_batch(action[None, :], config.action_bins_per_dim)[0]),
        )
    return replay


def make_batch() -> SampleBatch:
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    positive_observation = make_observation()
    negative_observation = make_observation(obj_x=0.8)
    observations = np.stack(
        [positive_observation, positive_observation, negative_observation, positive_observation]
    )
    goals = np.stack([goal, goal, goal, goal])
    horizon = np.asarray([20, 20, 20, 20], dtype=np.int16)
    return SampleBatch(
        indices=np.arange(4, dtype=np.int64),
        observation=observations,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=observations.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.zeros((4, 4), dtype=np.float32),
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=horizon,
        her_relabel_ratio=1.0,
    )


def compute_weights(weight_scope: str) -> WeightBatch:
    config = make_config(weight_scope)
    task = make_task()
    replay = make_replay(task, config)
    batch = make_batch()
    agent = StaticAgent(np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32))
    computer = LiteWeightComputer(config, task, GapStats(100, 10, 0.6, 0.95, 1e-6))
    return computer.compute(
        batch,
        replay,
        agent,
        np.random.default_rng(0),
        torch.device("cpu"),
    )


def assert_non_grad_weight(weight: np.ndarray) -> torch.Tensor:
    assert isinstance(weight, np.ndarray)
    assert weight.dtype == np.float32
    assert weight.ndim == 1
    tensor = torch.as_tensor(weight, dtype=torch.float32)
    assert tensor.requires_grad is False
    assert tensor.grad_fn is None
    return tensor


def test_mixed_support_scope_routing() -> None:
    batches = {
        scope: compute_weights(scope)
        for scope in ("actor_only", "critic_only", "actor_critic")
    }
    expected_support = np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    expected_raw = np.asarray([2.0, 2.0, 1.0, 2.0], dtype=np.float32)
    expected_omega = expected_raw / np.mean(expected_raw)
    ones = np.ones(4, dtype=np.float32)

    for batch in batches.values():
        np.testing.assert_array_equal(batch.psi_q_rank, expected_support)
        np.testing.assert_array_equal(batch.w_p_raw, expected_raw)
        assert_non_grad_weight(batch.omega_critic)
        assert_non_grad_weight(batch.omega_actor)

    actor_only = batches["actor_only"]
    np.testing.assert_array_equal(actor_only.omega_critic, ones)
    np.testing.assert_allclose(actor_only.omega_actor, expected_omega, rtol=1e-7, atol=0.0)
    assert np.isclose(float(np.mean(actor_only.omega_actor)), 1.0, rtol=1e-7, atol=1e-7)
    assert float(np.ptp(actor_only.omega_actor)) > 0.0
    assert actor_only.omega_actor[0] > actor_only.omega_actor[2]

    critic_only = batches["critic_only"]
    np.testing.assert_allclose(critic_only.omega_critic, expected_omega, rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(critic_only.omega_actor, ones)

    actor_critic = batches["actor_critic"]
    np.testing.assert_allclose(actor_critic.omega_critic, expected_omega, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(actor_critic.omega_actor, expected_omega, rtol=1e-7, atol=0.0)


def make_ddpg_agent() -> DDPGAgent:
    return DDPGAgent(
        state_dim=5,
        action_dim=2,
        hidden_dim=16,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.98,
        tau=0.005,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
    )


def assert_module_parameters_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    assert left_parameters.keys() == right_parameters.keys()
    for name in left_parameters:
        torch.testing.assert_close(
            left_parameters[name],
            right_parameters[name],
            rtol=0.0,
            atol=0.0,
        )


def module_parameters_differ(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    return any(
        not torch.equal(left_parameter, right_parameter)
        for left_parameter, right_parameter in zip(left.parameters(), right.parameters())
    )


def test_one_update_actor_vs_critic_isolation() -> None:
    actor_only_weights = compute_weights("actor_only")
    omega_actor = assert_non_grad_weight(actor_only_weights.omega_actor)
    omega_critic = assert_non_grad_weight(actor_only_weights.omega_critic)
    omega_ones = torch.ones_like(omega_actor)

    torch.manual_seed(9_001)
    weighted_agent = make_ddpg_agent()
    control_agent = make_ddpg_agent()
    control_agent.load_state_dict(weighted_agent.state_dict())

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

    weighted_metrics = weighted_agent.update(
        state=state,
        action_exec=replay_action,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=omega_critic,
        omega_actor=omega_actor,
    )
    control_metrics = control_agent.update(
        state=state,
        action_exec=replay_action,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=omega_ones,
        omega_actor=omega_ones,
    )

    assert weighted_metrics.critic_loss == control_metrics.critic_loss
    assert not np.isclose(weighted_metrics.actor_loss, control_metrics.actor_loss, rtol=0.0, atol=1e-10)
    # Actor backward leaves temporary critic gradients, so the isolation contract
    # is checked on critic parameters and optimizer-updated target parameters.
    assert_module_parameters_equal(weighted_agent.critic, control_agent.critic)
    assert_module_parameters_equal(weighted_agent.critic_target, control_agent.critic_target)
    assert module_parameters_differ(weighted_agent.actor, control_agent.actor)
    assert module_parameters_differ(weighted_agent.actor_target, control_agent.actor_target)
    assert omega_actor.requires_grad is False and omega_actor.grad is None
    assert omega_critic.requires_grad is False and omega_critic.grad is None


def main() -> int:
    test_mixed_support_scope_routing()
    test_one_update_actor_vs_critic_isolation()
    print("test_actor_only_scope.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
