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
    td_error_abs_values: np.ndarray | None = None


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
        fixed_target: torch.Tensor | None = None,
    ) -> UpdateMetrics:
        with torch.no_grad():
            if fixed_target is None:
                next_action = self.actor_target(next_state)
                target = reward + self.gamma * nonterminal * self.critic_target(next_state, next_action)
            else:
                target = fixed_target.detach()

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
            td_error_abs_values=td_abs.cpu().numpy().reshape(-1).astype(np.float32),
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
    def td_error_components(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_action = self.actor_target(next_state)
        target = reward + self.gamma * nonterminal * self.critic_target(next_state, next_action)
        q = self.critic(state, action_exec)
        return q, target, (q - target).abs()

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic(state, action)

    @torch.no_grad()
    def q_value_components(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        td_priority_reduction: str = "min_abs",
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.l2_action_penalty = float(l2_action_penalty)
        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.td_priority_reduction = str(td_priority_reduction).lower()
        if self.td_priority_reduction not in {"min_abs", "mean_abs"}:
            raise ValueError("td_priority_reduction must be min_abs|mean_abs")
        self.total_it = 0
        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)
        self.action_low_t = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=device)
        self.action_high_t = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=device)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
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
        self._last_actor_loss = 0.0

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
        fixed_target: torch.Tensor | None = None,
    ) -> UpdateMetrics:
        self.total_it += 1
        with torch.no_grad():
            if fixed_target is None:
                target = self._target_value(action_exec, reward, next_state, nonterminal)
            else:
                target = fixed_target.detach()

        q1 = self.critic1(state, action_exec)
        q2 = self.critic2(state, action_exec)
        td1 = q1 - target
        td2 = q2 - target
        # Use the same pre-update residuals and smoothed target as this optimizer step.
        # Recomputing later would draw different target-policy smoothing noise.
        td_abs = self._reduce_twin_td(td1.detach().abs(), td2.detach().abs())
        critic_loss = (omega_critic * td1.pow(2)).mean() + (omega_critic * td2.pow(2)).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=10.0)
        self.critic_opt.step()

        actor_loss_value = self._last_actor_loss
        if self.total_it % self.policy_delay == 0:
            pred_action = self.actor(state)
            actor_loss = -(omega_actor * self.critic1(state, pred_action)).mean()
            if self.l2_action_penalty > 0:
                actor_loss = actor_loss + self.l2_action_penalty * pred_action.pow(2).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            actor_loss_value = float(actor_loss.item())
            self._last_actor_loss = actor_loss_value

        q_min = torch.minimum(q1, q2)
        return UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss_value),
            q_mean=float(q_min.mean().item()),
            target_mean=float(target.mean().item()),
            q_std=float(q_min.std(unbiased=False).item()),
            target_std=float(target.std(unbiased=False).item()),
            q_abs_mean=float(q_min.detach().abs().mean().item()),
            q_abs_max=float(q_min.detach().abs().max().item()),
            td_error_abs_mean=float(td_abs.mean().item()),
            td_error_abs_p90=float(torch.quantile(td_abs, 0.90).item()),
            td_error_abs_values=td_abs.cpu().numpy().reshape(-1).astype(np.float32),
        )

    @torch.no_grad()
    def _target_value(
        self,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_exec) * self.target_policy_noise
        noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
        next_action = self.actor_target(next_state) + noise
        next_action = torch.max(torch.min(next_action, self.action_high_t), self.action_low_t)
        target_q1 = self.critic1_target(next_state, next_action)
        target_q2 = self.critic2_target(next_state, next_action)
        return reward + self.gamma * nonterminal * torch.minimum(target_q1, target_q2)

    @torch.no_grad()
    def td_error_abs(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> torch.Tensor:
        target = self._target_value(action_exec, reward, next_state, nonterminal)
        q1 = self.critic1(state, action_exec)
        q2 = self.critic2(state, action_exec)
        return self._reduce_twin_td((q1 - target).abs(), (q2 - target).abs())

    def _reduce_twin_td(
        self, td1_abs: torch.Tensor, td2_abs: torch.Tensor
    ) -> torch.Tensor:
        if self.td_priority_reduction == "mean_abs":
            return 0.5 * (td1_abs + td2_abs)
        return torch.minimum(td1_abs, td2_abs)

    @torch.no_grad()
    def td_error_components(
        self,
        state: torch.Tensor,
        action_exec: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        nonterminal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self._target_value(action_exec, reward, next_state, nonterminal)
        q1 = self.critic1(state, action_exec)
        q2 = self.critic2(state, action_exec)
        td1_abs = (q1 - target).abs()
        td2_abs = (q2 - target).abs()
        td_mean = 0.5 * (td1_abs + td2_abs)
        return q1, q2, target, td1_abs, td2_abs, td_mean

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.critic1(state, action), self.critic2(state, action))

    @torch.no_grad()
    def q_value_components(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(state, action), self.critic2(state, action)

    @torch.no_grad()
    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

    def state_dict(self) -> Dict[str, object]:
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_delay': self.policy_delay,
            'target_policy_noise': self.target_policy_noise,
            'target_noise_clip': self.target_noise_clip,
            'td_priority_reduction': self.td_priority_reduction,
            'total_it': self.total_it,
            'last_actor_loss': self._last_actor_loss,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload['actor'])
        self.actor_target.load_state_dict(payload['actor_target'])
        self.critic1.load_state_dict(payload['critic1'])
        self.critic2.load_state_dict(payload['critic2'])
        self.critic1_target.load_state_dict(payload['critic1_target'])
        self.critic2_target.load_state_dict(payload['critic2_target'])
        self.actor_opt.load_state_dict(payload['actor_opt'])
        self.critic_opt.load_state_dict(payload['critic_opt'])
        self.gamma = float(payload.get('gamma', self.gamma))
        self.tau = float(payload.get('tau', self.tau))
        self.policy_delay = int(payload.get('policy_delay', self.policy_delay))
        self.target_policy_noise = float(payload.get('target_policy_noise', self.target_policy_noise))
        self.target_noise_clip = float(payload.get('target_noise_clip', self.target_noise_clip))
        self.td_priority_reduction = str(
            payload.get("td_priority_reduction", self.td_priority_reduction)
        )
        self.total_it = int(payload.get('total_it', self.total_it))
        self._last_actor_loss = float(payload.get('last_actor_loss', self._last_actor_loss))
