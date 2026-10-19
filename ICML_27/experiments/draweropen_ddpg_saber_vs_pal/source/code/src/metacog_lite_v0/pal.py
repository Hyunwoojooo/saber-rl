"""Single-critic adaptation of the author's PAL loss, not TD-error top-k.

Reference: third_party/lap_pal/PAL_TD3.py and its MIT license. The original
piecewise constants (including the boundary value) are intentionally retained.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class PALLoss:
    loss: torch.Tensor
    normalizer: torch.Tensor
    base_loss: torch.Tensor
    large_error_fraction: torch.Tensor


def prioritized_approximation_loss(
    delta: torch.Tensor, alpha: float = 0.4, min_priority: float = 1.0,
) -> PALLoss:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("PAL alpha must be finite and in [0, 1]")
    if not math.isfinite(min_priority) or min_priority <= 0:
        raise ValueError("PAL min_priority must be finite and positive")
    if not delta.is_floating_point() or delta.numel() == 0:
        raise ValueError("PAL requires nonempty floating-point residuals")
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite PAL residual")
    magnitude = delta.abs()
    base_loss = torch.where(
        magnitude < min_priority,
        (min_priority ** alpha) * 0.5 * delta.pow(2),
        min_priority * magnitude.pow(1.0 + alpha) / (1.0 + alpha),
    ).mean()
    # DDPG has one critic: the official max(|delta1|, |delta2|) becomes |delta|.
    normalizer = magnitude.clamp(min=min_priority).pow(alpha).mean().detach()
    loss = base_loss / normalizer
    if not torch.isfinite(loss) or not torch.isfinite(normalizer):
        raise FloatingPointError("Non-finite PAL loss or normalizer")
    return PALLoss(loss, normalizer, base_loss,
                   (magnitude.detach() >= min_priority).to(delta.dtype).mean())
