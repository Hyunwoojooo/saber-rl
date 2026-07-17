from __future__ import annotations

import math


TD3_ER = "td3_er"
TD3_ANNEALING_ER = "td3_annealing_er"
TD3_AER = "td3_aer"

AER_METHODS = frozenset({TD3_ER, TD3_ANNEALING_ER, TD3_AER})
AER_LR_METHODS = frozenset({TD3_ANNEALING_ER, TD3_AER})


def attentive_lambda(
    global_step: int,
    total_steps: int,
    lambda0: float,
    anneal_fraction: float,
) -> float:
    """Linear AER bias schedule from lambda0 to one."""
    if lambda0 < 1.0:
        raise ValueError(f"AER lambda0 must be >= 1, got {lambda0}")
    if not 0.0 < anneal_fraction <= 1.0:
        raise ValueError(
            f"AER anneal_fraction must be in (0, 1], got {anneal_fraction}"
        )
    horizon = max(1.0, float(total_steps) * float(anneal_fraction))
    remaining = max(0.0, 1.0 - float(global_step) / horizon)
    return float(1.0 + (float(lambda0) - 1.0) * remaining)


def candidate_count(batch_size: int, lambda_value: float) -> int:
    """Paper-faithful lambda*B oversampling with explicit integer rounding."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lambda_value < 1.0:
        raise ValueError(f"lambda_value must be >= 1, got {lambda_value}")
    return max(int(batch_size), int(math.ceil(float(lambda_value) * int(batch_size))))


def method_lambda(
    method: str,
    global_step: int,
    total_steps: int,
    lambda0: float,
    anneal_fraction: float,
) -> float:
    if method not in AER_METHODS:
        raise ValueError(f"Unsupported AER method: {method!r}")
    if method == TD3_ER:
        return 1.0
    return attentive_lambda(global_step, total_steps, lambda0, anneal_fraction)


def effective_learning_rates(
    method: str,
    base_actor_lr: float,
    base_critic_lr: float,
    lambda_value: float,
) -> tuple[float, float]:
    scale = float(lambda_value) if method in AER_LR_METHODS else 1.0
    return float(base_actor_lr) * scale, float(base_critic_lr) * scale
