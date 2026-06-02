from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

from .networks import Actor, Critic


@dataclass
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_mean: float
    q_std: float
    target_std: float
    q_abs_mean: float
    q_abs_max: float
    td_error_abs_mean: float
    td_error_abs_p90: float


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        l2_action_penalty: float,
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.l2_action_penalty = float(l2_action_penalty)
        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float, noise_rng: np.random.Generator) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_t).squeeze(0).detach().cpu().numpy()
        if noise_std > 0.0:
            action = action + noise_rng.normal(0.0, noise_std, size=action.shape).astype(np.float32)
        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32)

    def update(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
        omega_critic: torch.Tensor,
        omega_actor: torch.Tensor,
    ) -> UpdateMetrics:
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target = reward + self.gamma * nonterminal * self.critic_target(next_state, next_action)

        q = self.critic(state, action_exec)
        td_abs = (q - target).detach().abs()
        critic_loss = (omega_critic * (q - target).pow(2)).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        pred_action = self.actor(state)
        actor_loss = -(omega_actor * self.critic(state, pred_action)).mean()
        if self.l2_action_penalty > 0:
            actor_loss = actor_loss + self.l2_action_penalty * pred_action.pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q.mean().item()),
            target_mean=float(target.mean().item()),
            q_std=float(q.std(unbiased=False).item()),
            target_std=float(target.std(unbiased=False).item()),
            q_abs_mean=float(q.detach().abs().mean().item()),
            q_abs_max=float(q.detach().abs().max().item()),
            td_error_abs_mean=float(td_abs.mean().item()),
            td_error_abs_p90=float(torch.quantile(td_abs, 0.90).item()),
        )

    @torch.no_grad()
    def td_error_abs(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> torch.Tensor:
        next_action = self.actor_target(next_state)
        target = reward + self.gamma * nonterminal * self.critic_target(next_state, next_action)
        q = self.critic(state, action_exec)
        return (q - target).abs()

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic(state, action)

    @torch.no_grad()
    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.gamma = float(payload.get("gamma", self.gamma))
        self.tau = float(payload.get("tau", self.tau))
