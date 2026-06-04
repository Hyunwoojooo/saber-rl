from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

from metacog_lite_v0.networks import Critic, mlp


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LOG_PROB_EPS = 1e-6


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = mlp(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + LOG_PROB_EPS)
        log_prob = log_prob.sum(dim=-1)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


@dataclass
class SACUpdateMetrics:
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
    alpha: float
    alpha_loss: float
    log_prob_mean: float
    target_entropy: float
    auto_alpha: float


class SACAgent:
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
        alpha: float,
        auto_alpha: bool,
        alpha_lr: float,
        target_entropy: float | None,
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.l2_action_penalty = float(l2_action_penalty)
        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)
        self.action_low = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=device)
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_params = list(self.critic.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)
        init_alpha = max(float(alpha), 1e-8)
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.fixed_alpha = init_alpha

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.as_tensor(self.fixed_alpha, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float, noise_rng: np.random.Generator) -> np.ndarray:
        del noise_rng
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if noise_std > 0.0:
            action, _, _ = self.actor.sample(state_t)
        else:
            mean, _ = self.actor(state_t)
            action = torch.tanh(mean)
        return action.squeeze(0).detach().cpu().numpy().clip(self.action_low_np, self.action_high_np).astype(np.float32)

    def update(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
        omega_critic: torch.Tensor,
        omega_actor: torch.Tensor,
    ) -> SACUpdateMetrics:
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1 = self.critic_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
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

        policy_action, log_prob, _ = self.actor.sample(state)
        q_pi = torch.minimum(self.critic(state, policy_action), self.critic2(state, policy_action))
        actor_loss = (omega_actor * (self.alpha.detach() * log_prob - q_pi)).mean()
        if self.l2_action_penalty > 0.0:
            actor_loss = actor_loss + self.l2_action_penalty * policy_action.pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss_t = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss_t.backward()
            self.alpha_opt.step()
        else:
            alpha_loss_t = torch.zeros((), dtype=torch.float32, device=self.device)

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.critic2, self.critic2_target)

        return SACUpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(torch.minimum(q1, q2).mean().item()),
            target_mean=float(target.mean().item()),
            q1_mean=float(q1.mean().item()),
            q2_mean=float(q2.mean().item()),
            critic_q1_loss=float(q1_loss.item()),
            critic_q2_loss=float(q2_loss.item()),
            actor_updated=1.0,
            policy_delay=1.0,
            target_policy_noise=0.0,
            target_noise_clip=0.0,
            alpha=float(self.alpha.detach().item()),
            alpha_loss=float(alpha_loss_t.detach().item()),
            log_prob_mean=float(log_prob.detach().mean().item()),
            target_entropy=float(self.target_entropy),
            auto_alpha=float(self.auto_alpha),
        )

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.critic(state, action), self.critic2(state, action))

    @torch.no_grad()
    def td_errors_for_priority(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> torch.Tensor:
        next_action, next_log_prob, _ = self.actor.sample(next_state)
        target_q1 = self.critic_target(next_state, next_action)
        target_q2 = self.critic2_target(next_state, next_action)
        target_q = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
        target = reward + self.gamma * nonterminal * target_q
        q = torch.minimum(self.critic(state, action_exec), self.critic2(state, action_exec))
        return torch.abs(q - target)

    @torch.no_grad()
    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor(state)
        return torch.tanh(mean)

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "fixed_alpha": self.fixed_alpha,
            "auto_alpha": self.auto_alpha,
            "target_entropy": self.target_entropy,
            "gamma": self.gamma,
            "tau": self.tau,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        if "alpha_opt" in payload:
            self.alpha_opt.load_state_dict(payload["alpha_opt"])
        self.log_alpha.data.copy_(torch.as_tensor(payload.get("log_alpha", self.log_alpha.detach()), device=self.device))
        self.fixed_alpha = float(payload.get("fixed_alpha", self.fixed_alpha))
        self.auto_alpha = bool(payload.get("auto_alpha", self.auto_alpha))
        self.target_entropy = float(payload.get("target_entropy", self.target_entropy))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.tau = float(payload.get("tau", self.tau))
