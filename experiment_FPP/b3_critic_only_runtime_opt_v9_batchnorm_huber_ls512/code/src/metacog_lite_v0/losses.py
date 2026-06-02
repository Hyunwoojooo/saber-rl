from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SaberCriticLossParts:
    total: torch.Tensor
    base: torch.Tensor
    support_bonus: torch.Tensor
    beta_eff: float


def compute_td_loss_vector(td_error: torch.Tensor, mode: str, huber_delta: float) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "mse":
        return td_error.pow(2)
    if mode != "huber":
        raise ValueError(f"critic_td_loss must be mse|huber, got {mode!r}")
    delta = float(huber_delta)
    if delta <= 0.0:
        raise ValueError("huber_delta must be > 0")
    abs_td = td_error.abs()
    quadratic = 0.5 * td_error.pow(2)
    linear = delta * (abs_td - 0.5 * delta)
    return torch.where(abs_td <= delta, quadratic, linear)


def compute_saber_critic_loss(
    td_error: torch.Tensor,
    omega_critic: torch.Tensor,
    reliability_score: torch.Tensor | None,
    *,
    critic_td_loss: str = "mse",
    huber_delta: float = 1.0,
    saber_loss_mode: str = "batch_norm",
    beta_eff: float = 0.0,
    detach_score: bool = True,
) -> SaberCriticLossParts:
    td_loss_vec = compute_td_loss_vector(td_error, critic_td_loss, huber_delta)
    mode = str(saber_loss_mode).lower()
    if mode == "batch_norm":
        total = (omega_critic.view_as(td_loss_vec) * td_loss_vec).mean()
        zero = td_loss_vec.new_tensor(0.0)
        return SaberCriticLossParts(total=total, base=total.detach(), support_bonus=zero, beta_eff=0.0)
    if mode != "positive_huber_bonus":
        raise ValueError(f"saber_loss_mode must be batch_norm|positive_huber_bonus, got {mode!r}")
    if reliability_score is None:
        raise ValueError("positive_huber_bonus requires reliability_score")
    score = reliability_score.to(device=td_loss_vec.device, dtype=td_loss_vec.dtype).view_as(td_loss_vec)
    score = torch.clamp(score, min=0.0, max=1.0)
    if detach_score:
        score = score.detach()
    base = td_loss_vec.mean()
    support_bonus = (score * td_loss_vec).mean()
    total = base + float(beta_eff) * support_bonus
    return SaberCriticLossParts(total=total, base=base, support_bonus=support_bonus, beta_eff=float(beta_eff))
