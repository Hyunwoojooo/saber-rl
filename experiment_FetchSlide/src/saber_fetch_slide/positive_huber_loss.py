from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class SaberLossUpdateMetrics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_mean: float
    critic_base_loss: float
    support_bonus_loss: float
    critic_total_loss: float
    huber_delta: float
    td_loss_mode_huber: float
    saber_loss_mode_positive_huber_bonus: float
    reliability_score_mean: float
    reliability_score_p50: float
    reliability_score_p90: float
    support_bonus_beta_eff: float


def td_loss_vector(td_error: torch.Tensor, mode: str, huber_delta: float) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "mse":
        return td_error.pow(2)
    if mode != "huber":
        raise ValueError(f"Unsupported critic_td_loss={mode!r}")
    delta = float(huber_delta)
    if delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    abs_td = td_error.abs()
    quadratic = 0.5 * td_error.pow(2)
    linear = delta * (abs_td - 0.5 * delta)
    return torch.where(abs_td <= delta, quadratic, linear)


def reliability_score_from_weight_batch(wb: Any, source: str) -> np.ndarray:
    source = str(source).lower()
    if source not in {"auto", "psi", "support_score"}:
        raise ValueError(f"Unsupported support_bonus_score_source={source!r}")
    if source in {"auto", "support_score"} and hasattr(wb, "support_score"):
        score = np.asarray(getattr(wb, "support_score"), dtype=np.float32)
    elif source in {"auto", "support_score"} and hasattr(wb, "reliability_score"):
        score = np.asarray(getattr(wb, "reliability_score"), dtype=np.float32)
    else:
        score = np.asarray(wb.psi_q_rank, dtype=np.float32)
    if score.shape != wb.psi_q_rank.shape:
        score = np.reshape(score, wb.psi_q_rank.shape)
    return np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)


def reliability_metrics(score: np.ndarray, beta_eff: float) -> dict[str, float]:
    if score.size == 0:
        return {
            "lite/reliability_score_mean": 0.0,
            "lite/reliability_score_p50": 0.0,
            "lite/reliability_score_p90": 0.0,
            "lite/support_bonus_beta_eff": float(beta_eff),
        }
    return {
        "lite/reliability_score_mean": float(np.mean(score)),
        "lite/reliability_score_p50": float(np.percentile(score, 50.0)),
        "lite/reliability_score_p90": float(np.percentile(score, 90.0)),
        "lite/support_bonus_beta_eff": float(beta_eff),
    }


def update_ddpg_with_saber_loss(
    agent: Any,
    state: torch.Tensor,
    action_exec: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    nonterminal: torch.Tensor,
    omega_critic: torch.Tensor,
    omega_actor: torch.Tensor,
    reliability_score: torch.Tensor,
    critic_td_loss: str,
    huber_delta: float,
    saber_loss_mode: str,
    support_bonus_beta_eff: float,
    support_bonus_detach_score: bool,
) -> SaberLossUpdateMetrics:
    with torch.no_grad():
        next_action = agent.actor_target(next_state)
        target = reward + agent.gamma * nonterminal * agent.critic_target(next_state, next_action)

    q = agent.critic(state, action_exec)
    td_error = q - target
    td_loss = td_loss_vector(td_error, critic_td_loss, huber_delta)
    mode = str(saber_loss_mode).lower()
    rel = reliability_score.to(device=td_loss.device, dtype=td_loss.dtype).view_as(td_loss)
    rel = torch.clamp(rel, min=0.0, max=1.0)
    if mode == "positive_huber_bonus":
        score = rel.detach() if support_bonus_detach_score else rel
        critic_base_loss = td_loss.mean()
        support_bonus_loss = (score * td_loss).mean()
        critic_loss = critic_base_loss + float(support_bonus_beta_eff) * support_bonus_loss
    else:
        critic_loss = (omega_critic * td_loss).mean()
        critic_base_loss = td_loss.mean()
        support_bonus_loss = torch.zeros((), dtype=td_loss.dtype, device=td_loss.device)

    agent.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=10.0)
    agent.critic_opt.step()

    pred_action = agent.actor(state)
    actor_loss = -(omega_actor * agent.critic(state, pred_action)).mean()
    if agent.l2_action_penalty > 0:
        actor_loss = actor_loss + agent.l2_action_penalty * pred_action.pow(2).mean()
    agent.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=10.0)
    agent.actor_opt.step()

    agent._soft_update(agent.actor, agent.actor_target)
    agent._soft_update(agent.critic, agent.critic_target)

    rel_detached = rel.detach().flatten()
    return SaberLossUpdateMetrics(
        critic_loss=float(critic_loss.item()),
        actor_loss=float(actor_loss.item()),
        q_mean=float(q.mean().item()),
        target_mean=float(target.mean().item()),
        critic_base_loss=float(critic_base_loss.detach().item()),
        support_bonus_loss=float(support_bonus_loss.detach().item()),
        critic_total_loss=float(critic_loss.detach().item()),
        huber_delta=float(huber_delta),
        td_loss_mode_huber=float(str(critic_td_loss).lower() == "huber"),
        saber_loss_mode_positive_huber_bonus=float(mode == "positive_huber_bonus"),
        reliability_score_mean=float(rel_detached.mean().item()),
        reliability_score_p50=float(torch.quantile(rel_detached, 0.5).item()),
        reliability_score_p90=float(torch.quantile(rel_detached, 0.9).item()),
        support_bonus_beta_eff=float(support_bonus_beta_eff),
    )
