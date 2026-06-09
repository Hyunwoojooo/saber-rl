from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

from metacog_lite_v0.networks import Actor, Critic


@dataclass
class TD3UpdateMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_mean: float
    q1_mean: float
    q2_mean: float
    critic_q1_loss: float
    critic_q2_loss: float
    actor_updated: float
    policy_delay: float
    target_policy_noise: float
    target_noise_clip: float


class TD3Agent:
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
        policy_delay: int,
        target_policy_noise: float,
        target_noise_clip: float,
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.l2_action_penalty = float(l2_action_penalty)
        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)
        self.action_low = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=device)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_params = list(self.critic.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)

        self.update_step = 0
        self.last_actor_loss = 0.0

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
    ) -> TD3UpdateMetrics:
        self.update_step += 1
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(next_action) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_action = next_action + noise
            next_action = torch.max(torch.min(next_action, self.action_high), self.action_low)
            target_q1 = self.critic_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            target = reward + self.gamma * nonterminal * target_q

        q1 = self.critic(state, action_exec)
        q2 = self.critic2(state, action_exec)
        q1_loss = (omega_critic * (q1 - target).pow(2)).mean()
        q2_loss = (omega_critic * (q2 - target).pow(2)).mean()
        critic_loss = q1_loss + q2_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        self.critic_opt.step()

        actor_updated = self.update_step % self.policy_delay == 0
        if actor_updated:
            pred_action = self.actor(state)
            actor_loss_t = -(omega_actor * self.critic(state, pred_action)).mean()
            if self.l2_action_penalty > 0.0:
                actor_loss_t = actor_loss_t + self.l2_action_penalty * pred_action.pow(2).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss_t.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.critic2, self.critic2_target)
            self.last_actor_loss = float(actor_loss_t.item())

        return TD3UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(self.last_actor_loss),
            q_mean=float(torch.minimum(q1, q2).mean().item()),
            target_mean=float(target.mean().item()),
            q1_mean=float(q1.mean().item()),
            q2_mean=float(q2.mean().item()),
            critic_q1_loss=float(q1_loss.item()),
            critic_q2_loss=float(q2_loss.item()),
            actor_updated=float(actor_updated),
            policy_delay=float(self.policy_delay),
            target_policy_noise=float(self.target_policy_noise),
            target_noise_clip=float(self.target_noise_clip),
        )

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.critic(state, action), self.critic2(state, action))

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
        if self.target_policy_noise > 0.0:
            noise = torch.randn_like(next_action) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = next_action + noise
        next_action = torch.max(torch.min(next_action, self.action_high), self.action_low)
        target_q1 = self.critic_target(next_state, next_action)
        target_q2 = self.critic2_target(next_state, next_action)
        target = reward + self.gamma * nonterminal * torch.minimum(target_q1, target_q2)
        q1 = self.critic(state, action_exec)
        q2 = self.critic2(state, action_exec)
        return torch.minimum((q1 - target).abs(), (q2 - target).abs())

    @torch.no_grad()
    def td_errors_for_priority(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> torch.Tensor:
        next_action = self.actor_target(next_state)
        if self.target_policy_noise > 0.0:
            noise = torch.randn_like(next_action) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = next_action + noise
        next_action = torch.max(torch.min(next_action, self.action_high), self.action_low)
        target_q1 = self.critic_target(next_state, next_action)
        target_q2 = self.critic2_target(next_state, next_action)
        target = reward + self.gamma * nonterminal * torch.minimum(target_q1, target_q2)
        q = torch.minimum(self.critic(state, action_exec), self.critic2(state, action_exec))
        return torch.abs(q - target)

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
            "critic2": self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "policy_delay": self.policy_delay,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
            "update_step": self.update_step,
            "last_actor_loss": self.last_actor_loss,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.gamma = float(payload.get("gamma", self.gamma))
        self.tau = float(payload.get("tau", self.tau))
        self.policy_delay = int(payload.get("policy_delay", self.policy_delay))
        self.target_policy_noise = float(payload.get("target_policy_noise", self.target_policy_noise))
        self.target_noise_clip = float(payload.get("target_noise_clip", self.target_noise_clip))
        self.update_step = int(payload.get("update_step", self.update_step))
        self.last_actor_loss = float(payload.get("last_actor_loss", self.last_actor_loss))
