#!/usr/bin/env python3
from __future__ import annotations

import copy

import numpy as np
import torch

from test_batch_wide_count_matched import _batch_wide_compute
from test_ddpg_query_loss_isolation import make_agent, run_update


def main() -> int:
    weight_batch, *_ = _batch_wide_compute(cache_enabled=True)
    random_support_omega = torch.as_tensor(
        weight_batch.omega_critic, dtype=torch.float32
    )
    torch.testing.assert_close(
        torch.as_tensor(weight_batch.omega_actor),
        torch.ones_like(random_support_omega),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        random_support_omega, torch.ones_like(random_support_omega)
    )

    rng = np.random.default_rng(20260720)
    state = torch.as_tensor(rng.normal(size=(4, 28)), dtype=torch.float32)
    next_state = torch.as_tensor(rng.normal(size=(4, 28)), dtype=torch.float32)
    action_exec = torch.as_tensor(
        rng.uniform(-1.0, 1.0, size=(4, 4)), dtype=torch.float32
    )
    reward = torch.as_tensor([-1.0, -0.6, 0.4, 1.2], dtype=torch.float32)
    nonterminal = torch.as_tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    uniform_agent = make_agent(711)
    random_support_agent = make_agent(917)
    random_support_agent.load_state_dict(
        copy.deepcopy(uniform_agent.state_dict())
    )
    uniform_metrics, uniform_calls = run_update(
        uniform_agent,
        state,
        action_exec,
        reward,
        next_state,
        nonterminal,
        torch.ones(4, dtype=torch.float32),
    )
    random_support_metrics, random_support_calls = run_update(
        random_support_agent,
        state,
        action_exec,
        reward,
        next_state,
        nonterminal,
        random_support_omega,
    )
    torch.testing.assert_close(uniform_calls[0], action_exec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        random_support_calls[0], action_exec, rtol=0.0, atol=0.0
    )
    assert (
        abs(
            uniform_metrics.critic_loss
            - random_support_metrics.critic_loss
        )
        > 1e-7
    )
    assert uniform_metrics.actor_loss == random_support_metrics.actor_loss
    print("test_ddpg_random_support_loss_scope.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
