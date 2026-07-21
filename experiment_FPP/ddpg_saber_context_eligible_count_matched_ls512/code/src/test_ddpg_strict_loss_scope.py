#!/usr/bin/env python3
from __future__ import annotations

import copy

import numpy as np
import torch

from test_context_eligible_count_matched import _strict_compute
from test_ddpg_query_loss_isolation import make_agent, run_update


def main() -> int:
    weight_batch, *_ = _strict_compute(cache_enabled=True)
    strict_omega = torch.as_tensor(
        weight_batch.omega_critic, dtype=torch.float32
    )
    torch.testing.assert_close(
        torch.as_tensor(weight_batch.omega_actor),
        torch.ones_like(strict_omega),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(strict_omega, torch.ones_like(strict_omega))

    rng = np.random.default_rng(20260720)
    state = torch.as_tensor(rng.normal(size=(4, 28)), dtype=torch.float32)
    next_state = torch.as_tensor(rng.normal(size=(4, 28)), dtype=torch.float32)
    action_exec = torch.as_tensor(
        rng.uniform(-1.0, 1.0, size=(4, 4)), dtype=torch.float32
    )
    reward = torch.as_tensor([-1.0, -0.6, 0.4, 1.2], dtype=torch.float32)
    nonterminal = torch.as_tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    uniform_agent = make_agent(711)
    strict_agent = make_agent(917)
    strict_agent.load_state_dict(copy.deepcopy(uniform_agent.state_dict()))
    uniform_metrics, uniform_calls = run_update(
        uniform_agent,
        state,
        action_exec,
        reward,
        next_state,
        nonterminal,
        torch.ones(4, dtype=torch.float32),
    )
    strict_metrics, strict_calls = run_update(
        strict_agent,
        state,
        action_exec,
        reward,
        next_state,
        nonterminal,
        strict_omega,
    )
    torch.testing.assert_close(uniform_calls[0], action_exec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(strict_calls[0], action_exec, rtol=0.0, atol=0.0)
    assert abs(uniform_metrics.critic_loss - strict_metrics.critic_loss) > 1e-7
    assert uniform_metrics.actor_loss == strict_metrics.actor_loss
    print("test_ddpg_strict_loss_scope.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
