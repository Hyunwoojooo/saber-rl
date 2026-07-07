from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

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
    alpha_loss: float
    entropy_alpha: float
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


class REDQAgent:
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
        sac_auto_alpha: bool = False,
        target_entropy: float | None = None,
        alpha_lr: float | None = None,
        ensemble_size: int = 5,
        target_subset_size: int = 2,
        target_q_bounding: bool = False,
        target_q_min: float | None = None,
        target_q_max: float | None = None,
    ) -> None:
        del policy_delay, target_policy_noise, target_noise_clip
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.l2_action_penalty = float(l2_action_penalty)
        self.entropy_alpha = float(entropy_alpha)
        self.sac_auto_alpha = bool(sac_auto_alpha)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)
        self.alpha_lr = float(alpha_lr) if alpha_lr is not None else float(actor_lr)
        self.ensemble_size = int(ensemble_size)
        self.target_subset_size = int(target_subset_size)
        if self.ensemble_size < 2:
            raise ValueError("REDQ requires ensemble_size >= 2")
        if not 1 <= self.target_subset_size <= self.ensemble_size:
            raise ValueError("target_subset_size must be in [1, ensemble_size]")
        self.target_q_bounding = bool(target_q_bounding)
        self.target_q_min = float(target_q_min) if target_q_min is not None else -1.0 / (1.0 - self.gamma)
        self.target_q_max = float(target_q_max) if target_q_max is not None else 0.0
        if self.target_q_bounding and self.target_q_min > self.target_q_max:
            raise ValueError("target_q_min must be <= target_q_max")
        self.total_it = 0

        self.action_low_np = np.asarray(action_low, dtype=np.float32)
        self.action_high_np = np.asarray(action_high, dtype=np.float32)

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(device)
        self.critics = nn.ModuleList(
            [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(self.ensemble_size)]
        )
        self.critic_targets = nn.ModuleList(
            [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(self.ensemble_size)]
        )
        for target, source in zip(self.critic_targets, self.critics):
            target.load_state_dict(source.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)
        self.log_entropy_alpha = torch.tensor(
            math.log(max(self.entropy_alpha, 1e-8)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_opt = (
            torch.optim.Adam([self.log_entropy_alpha], lr=self.alpha_lr) if self.sac_auto_alpha else None
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

    def _sample_target_indices(self) -> list[int]:
        return torch.randperm(self.ensemble_size)[: self.target_subset_size].tolist()

    def _alpha(self) -> torch.Tensor:
        if self.sac_auto_alpha:
            return self.log_entropy_alpha.exp()
        return torch.as_tensor(self.entropy_alpha, dtype=torch.float32, device=self.device)

    def _bound_target_q(self, target_q: torch.Tensor) -> torch.Tensor:
        if not self.target_q_bounding:
            return target_q
        return torch.clamp(target_q, min=self.target_q_min, max=self.target_q_max)

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
            subset = self._sample_target_indices()
            q_next_stack = torch.stack(
                [self.critic_targets[idx](next_state, next_action) for idx in subset],
                dim=0,
            )
            target_q = q_next_stack.min(dim=0).values - self._alpha().detach() * next_log_prob
            target_q = self._bound_target_q(target_q)
            target = reward + self.gamma * nonterminal * target_q

        q_stack = torch.stack([critic(state, action_exec) for critic in self.critics], dim=0)
        td_errors = q_stack - target.unsqueeze(0)
        td_abs = td_errors.detach().abs().mean(dim=0)
        critic_loss = (omega_critic.unsqueeze(0) * td_errors.pow(2)).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm_t = nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=10.0)
        self.critic_opt.step()

        pi_action, log_pi = self.actor.sample(state)
        q_pi_stack = torch.stack([critic(state, pi_action) for critic in self.critics], dim=0)
        q_pi = q_pi_stack.mean(dim=0)
        actor_loss = (omega_actor * (self._alpha().detach() * log_pi - q_pi)).mean()
        if self.l2_action_penalty > 0:
            actor_loss = actor_loss + self.l2_action_penalty * pi_action.pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm_t = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        alpha_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.sac_auto_alpha:
            alpha_loss = -(self.log_entropy_alpha * (log_pi.detach() + self.target_entropy)).mean()
            assert self.alpha_opt is not None
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self.entropy_alpha = float(self._alpha().detach().cpu().item())

        for source, target_net in zip(self.critics, self.critic_targets):
            self._soft_update(source, target_net)

        with torch.no_grad():
            policy_action = self.actor.deterministic(state)
            q_policy_stack = torch.stack([critic(state, policy_action) for critic in self.critics], dim=0)
            q_policy_eval = q_policy_stack.mean(dim=0)
            q_eval = q_stack.detach().mean(dim=0)

        return UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q_eval.mean().item()),
            target_mean=float(target.mean().item()),
            td_error_abs_mean=float(td_abs.mean().item()),
            td_error_abs_p90=float(torch.quantile(td_abs.flatten(), 0.9).item()),
            critic_grad_norm=float(critic_grad_norm_t.detach().cpu().item()),
            actor_grad_norm=float(actor_grad_norm_t.detach().cpu().item()),
            q_policy_mean=float(q_policy_eval.detach().mean().item()),
            q_policy_std=float(q_policy_eval.detach().std(unbiased=False).item()),
            q_target_mean=float(target.detach().mean().item()),
            q_target_std=float(target.detach().std(unbiased=False).item()),
            q_abs_mean=float(q_eval.detach().abs().mean().item()),
            q_abs_max=float(q_eval.detach().abs().max().item()),
            alpha_loss=float(alpha_loss.detach().cpu().item()),
            entropy_alpha=float(self.entropy_alpha),
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
        q_next_stack = torch.stack(
            [target(next_state, next_action) for target in self.critic_targets],
            dim=0,
        )
        target_q = q_next_stack.min(dim=0).values - self._alpha().detach() * next_log_prob
        target_q = self._bound_target_q(target_q)
        target = reward + self.gamma * nonterminal * target_q
        q_stack = torch.stack([critic(state, action_exec) for critic in self.critics], dim=0)
        return (q_stack - target.unsqueeze(0)).abs().mean(dim=0)

    @torch.no_grad()
    def target_q_values(self, next_state: torch.Tensor) -> torch.Tensor:
        next_action, next_log_prob = self.actor.sample(next_state)
        q_stack = torch.stack([target(next_state, next_action) for target in self.critic_targets], dim=0)
        target_q = q_stack.min(dim=0).values - self._alpha().detach() * next_log_prob
        return self._bound_target_q(target_q)

    @torch.no_grad()
    def critic_target(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_stack = torch.stack([target(state, action) for target in self.critic_targets], dim=0)
        return q_stack.min(dim=0).values

    @torch.no_grad()
    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_stack = torch.stack([critic(state, action) for critic in self.critics], dim=0)
        return q_stack.mean(dim=0)

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
            "critics": self.critics.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
            "gamma": self.gamma,
            "tau": self.tau,
            "entropy_alpha": self.entropy_alpha,
            "sac_auto_alpha": self.sac_auto_alpha,
            "target_entropy": self.target_entropy,
            "alpha_lr": self.alpha_lr,
            "log_entropy_alpha": self.log_entropy_alpha.detach().cpu(),
            "ensemble_size": self.ensemble_size,
            "target_subset_size": self.target_subset_size,
            "target_q_bounding": self.target_q_bounding,
            "target_q_min": self.target_q_min,
            "target_q_max": self.target_q_max,
            "total_it": self.total_it,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critics.load_state_dict(payload["critics"])
        self.critic_targets.load_state_dict(payload["critic_targets"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.gamma = float(payload.get("gamma", self.gamma))
        self.tau = float(payload.get("tau", self.tau))
        self.entropy_alpha = float(payload.get("entropy_alpha", self.entropy_alpha))
        self.sac_auto_alpha = bool(payload.get("sac_auto_alpha", self.sac_auto_alpha))
        self.target_entropy = float(payload.get("target_entropy", self.target_entropy))
        self.alpha_lr = float(payload.get("alpha_lr", self.alpha_lr))
        log_alpha_payload = payload.get("log_entropy_alpha")
        with torch.no_grad():
            if log_alpha_payload is not None:
                self.log_entropy_alpha.copy_(torch.as_tensor(log_alpha_payload, dtype=torch.float32, device=self.device))
            else:
                self.log_entropy_alpha.fill_(math.log(max(self.entropy_alpha, 1e-8)))
        if self.sac_auto_alpha and self.alpha_opt is None:
            self.alpha_opt = torch.optim.Adam([self.log_entropy_alpha], lr=self.alpha_lr)
        if self.alpha_opt is not None and payload.get("alpha_opt") is not None:
            self.alpha_opt.load_state_dict(payload["alpha_opt"])
        self.ensemble_size = int(payload.get("ensemble_size", self.ensemble_size))
        self.target_subset_size = int(payload.get("target_subset_size", self.target_subset_size))
        self.target_q_bounding = bool(payload.get("target_q_bounding", self.target_q_bounding))
        self.target_q_min = float(payload.get("target_q_min", self.target_q_min))
        self.target_q_max = float(payload.get("target_q_max", self.target_q_max))
        self.total_it = int(payload.get("total_it", self.total_it))


SACAgent = REDQAgent
DDPGAgent = REDQAgent
TD3Agent = REDQAgent
