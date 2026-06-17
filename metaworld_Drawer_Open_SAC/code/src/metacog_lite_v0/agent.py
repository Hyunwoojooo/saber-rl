from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .networks import Critic, mlp

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_mean: float
    td_error_abs_mean: float
    td_error_abs_p90: float
    critic_grad_norm: float
    actor_grad_norm: float
    q_policy_mean: float
    q_policy_std: float
    q_target_mean: float
    q_target_std: float
    q_abs_mean: float
    q_abs_max: float
    td_error_abs_values: np.ndarray


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, 2 * action_dim)
        self.action_dim = int(action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob.sum(dim=-1)

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean)


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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        entropy_alpha: float = 0.2,
    ) -> None:
        del policy_delay, target_policy_noise, target_noise_clip
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.l2_action_penalty = float(l2_action_penalty)
        self.entropy_alpha = float(entropy_alpha)
        self.total_it = 0

        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float, noise_rng: np.random.Generator) -> np.ndarray:
        del noise_rng
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if noise_std > 0.0:
            action, _ = self.actor.sample(state_t)
        else:
            action = self.actor.deterministic(state_t)
        return np.clip(action.squeeze(0).detach().cpu().numpy(), self.action_low_np, self.action_high_np).astype(np.float32)

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
        self.total_it += 1
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next = self.critic1_target(next_state, next_action)
            q2_next = self.critic2_target(next_state, next_action)
            target_q = torch.minimum(q1_next, q2_next) - self.entropy_alpha * next_log_prob
            target = reward + self.gamma * nonterminal * target_q

        q1 = self.critic1(state, action_exec)
        q2 = self.critic2(state, action_exec)
        td_error1 = q1 - target
        td_error2 = q2 - target
        td_abs = 0.5 * (td_error1.detach().abs() + td_error2.detach().abs())
        critic_loss = (omega_critic * (td_error1.pow(2) + td_error2.pow(2)) * 0.5).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm_t = nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            max_norm=10.0,
        )
        self.critic_opt.step()

        pi_action, log_pi = self.actor.sample(state)
        q1_pi = self.critic1(state, pi_action)
        q2_pi = self.critic2(state, pi_action)
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (omega_actor * (self.entropy_alpha * log_pi - q_pi)).mean()
        if self.l2_action_penalty > 0:
            actor_loss = actor_loss + self.l2_action_penalty * pi_action.pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm_t = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        with torch.no_grad():
            policy_action = self.actor.deterministic(state)
            q_policy_eval = torch.minimum(
                self.critic1(state, policy_action),
                self.critic2(state, policy_action),
            )
            q_min = torch.minimum(q1, q2)

        return UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q_min.mean().item()),
            target_mean=float(target.mean().item()),
            td_error_abs_mean=float(td_abs.mean().item()),
            td_error_abs_p90=float(torch.quantile(td_abs.flatten(), 0.9).item()),
            critic_grad_norm=float(critic_grad_norm_t.detach().cpu().item()),
            actor_grad_norm=float(actor_grad_norm_t.detach().cpu().item()),
            q_policy_mean=float(q_policy_eval.detach().mean().item()),
            q_policy_std=float(q_policy_eval.detach().std(unbiased=False).item()),
            q_target_mean=float(target.detach().mean().item()),
            q_target_std=float(target.detach().std(unbiased=False).item()),
            q_abs_mean=float(q_min.detach().abs().mean().item()),
            q_abs_max=float(q_min.detach().abs().max().item()),
            td_error_abs_values=td_abs.detach().cpu().numpy().astype(np.float32),
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
        next_action, next_log_prob = self.actor.sample(next_state)
        q1_next = self.critic1_target(next_state, next_action)
        q2_next = self.critic2_target(next_state, next_action)
        target_q = torch.minimum(q1_next, q2_next) - self.entropy_alpha * next_log_prob
        target = reward + self.gamma * nonterminal * target_q
        q1 = self.critic1(state, action_exec)
        q2 = self.critic2(state, action_exec)
        return 0.5 * ((q1 - target).abs() + (q2 - target).abs())

    @torch.no_grad()
    def target_q_values(self, next_state: torch.Tensor) -> torch.Tensor:
        next_action, next_log_prob = self.actor.sample(next_state)
        q1_t = self.critic1_target(next_state, next_action)
        q2_t = self.critic2_target(next_state, next_action)
        return torch.minimum(q1_t, q2_t) - self.entropy_alpha * next_log_prob

    @torch.no_grad()
    def critic_target(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1_t = self.critic1_target(state, action)
        q2_t = self.critic2_target(state, action)
        return torch.minimum(q1_t, q2_t)

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return torch.minimum(q1, q2)

    @torch.no_grad()
    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(state)

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "entropy_alpha": self.entropy_alpha,
            "total_it": self.total_it,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic1_target.load_state_dict(payload["critic1_target"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.gamma = float(payload.get("gamma", self.gamma))
        self.tau = float(payload.get("tau", self.tau))
        self.entropy_alpha = float(payload.get("entropy_alpha", self.entropy_alpha))
        self.total_it = int(payload.get("total_it", self.total_it))


DDPGAgent = SACAgent
TD3Agent = SACAgent
