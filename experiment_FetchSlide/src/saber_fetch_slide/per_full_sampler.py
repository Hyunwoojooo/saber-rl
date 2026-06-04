from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch


@dataclass
class FullPERSampleInfo:
    beta: float
    priority_mean: float
    priority_max: float
    sample_prob_mean: float
    sample_prob_max: float
    is_weight_mean: float
    is_weight_min: float
    is_weight_max: float
    priority_ess: float

    def as_metrics(self) -> Dict[str, float]:
        return {
            "per_full/beta": self.beta,
            "per_full/priority_mean": self.priority_mean,
            "per_full/priority_max": self.priority_max,
            "per_full/sample_prob_mean": self.sample_prob_mean,
            "per_full/sample_prob_max": self.sample_prob_max,
            "per_full/is_weight_mean": self.is_weight_mean,
            "per_full/is_weight_min": self.is_weight_min,
            "per_full/is_weight_max": self.is_weight_max,
            "per_full/priority_ESS": self.priority_ess,
        }


class FullPERSampler:
    """Transition-level PER sampler that preserves the existing HER relabel path."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_final: float = 1.0,
        beta_anneal_steps: int = 1_000_000,
        eps: float = 1e-6,
        initial_priority: float = 1.0,
        min_priority: float = 1e-6,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_final = float(beta_final)
        self.beta_anneal_steps = max(1, int(beta_anneal_steps))
        self.eps = float(eps)
        self.initial_priority = float(initial_priority)
        self.min_priority = float(min_priority)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.beta_start < 0.0 or self.beta_final < 0.0:
            raise ValueError("PER beta values must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        self.priority = np.full(self.capacity, self.initial_priority, dtype=np.float32)
        self.max_priority = float(self.initial_priority)

    def beta(self, step: int | None) -> float:
        step_i = 0 if step is None else max(0, int(step))
        frac = min(1.0, step_i / float(self.beta_anneal_steps))
        return float(self.beta_start + frac * (self.beta_final - self.beta_start))

    def on_transition_added(self, slot: int) -> None:
        slot_i = int(slot)
        if 0 <= slot_i < self.capacity:
            self.priority[slot_i] = max(self.max_priority, self.initial_priority, self.min_priority)

    def sample(
        self,
        replay: HERReplayBuffer,
        batch_size: int,
        her_ratio: float,
        rng: np.random.Generator,
        step: int | None,
    ) -> SampleBatch:
        if replay.size < batch_size:
            raise ValueError(f"Not enough replay samples: {replay.size} < {batch_size}")
        valid_priority = np.maximum(self.priority[: replay.size], self.min_priority)
        powered = np.power(valid_priority, self.alpha, dtype=np.float32)
        total = float(np.sum(powered))
        if not np.isfinite(total) or total <= 0.0:
            powered.fill(1.0)
            total = float(powered.size)

        cdf = np.cumsum(powered, dtype=np.float64)
        draws = rng.random(int(batch_size)) * total
        indices = np.searchsorted(cdf, draws, side="right").astype(np.int64)
        np.minimum(indices, replay.size - 1, out=indices)

        desired_goal_train = replay.desired_goal[indices].copy()
        relabel_mask = rng.random(int(batch_size)) < float(her_ratio)
        relabeled = 0
        for out_i, slot in enumerate(indices):
            if not relabel_mask[out_i]:
                continue
            goal = replay._sample_future_goal(int(slot), rng)
            if goal is None:
                continue
            desired_goal_train[out_i] = goal
            relabeled += 1

        prob = powered[indices].astype(np.float64) / max(total, 1e-12)
        beta = self.beta(step)
        weights = np.power(np.maximum(replay.size * prob, 1e-12), -beta)
        min_powered = float(np.min(powered))
        min_prob = min_powered / max(total, 1e-12)
        max_weight = float(np.power(max(replay.size * min_prob, 1e-12), -beta))
        if max_weight > 0.0 and np.isfinite(max_weight):
            weights = weights / max_weight
        weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

        batch = SampleBatch(
            indices=indices,
            observation=replay.observation[indices].copy(),
            achieved_goal=replay.achieved_goal[indices].copy(),
            desired_goal_train=desired_goal_train,
            next_observation=replay.next_observation[indices].copy(),
            next_achieved_goal=replay.next_achieved_goal[indices].copy(),
            action_exec=replay.action_exec[indices].copy(),
            reward_ext=replay.reward_ext[indices].copy(),
            done=replay.done[indices].copy(),
            timeout=replay.timeout[indices].copy(),
            h=replay.h[indices].copy(),
            her_relabel_ratio=float(relabeled / max(1, int(batch_size))),
        )
        setattr(batch, "per_full_is_weight", weights)
        setattr(batch, "per_full_sample_prob", prob.astype(np.float32))
        setattr(batch, "per_full_beta", float(beta))
        setattr(batch, "per_full_sample_metrics", self._sample_metrics(replay.size, valid_priority, powered, prob, weights, beta))
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> Dict[str, float]:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        td = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        n = min(idx.size, td.size)
        if n <= 0:
            return {
                "per_full/priority_update_abs_mean": 0.0,
                "per_full/priority_update_abs_p90": 0.0,
                "per_full/priority_update_count": 0.0,
            }
        updated = np.maximum(np.abs(td[:n]) + self.eps, self.min_priority).astype(np.float32)
        for slot, prio in zip(idx[:n], updated):
            slot_i = int(slot)
            if 0 <= slot_i < self.capacity:
                self.priority[slot_i] = float(prio)
        self.max_priority = max(self.max_priority, float(np.max(updated)))
        return {
            "per_full/priority_update_abs_mean": float(np.mean(np.abs(td[:n]))),
            "per_full/priority_update_abs_p90": float(np.percentile(np.abs(td[:n]), 90.0)),
            "per_full/priority_update_count": float(n),
        }

    def _sample_metrics(
        self,
        size: int,
        valid_priority: np.ndarray,
        powered: np.ndarray,
        sampled_prob: np.ndarray,
        weights: np.ndarray,
        beta: float,
    ) -> Dict[str, float]:
        total = float(np.sum(powered))
        sumsq = float(np.sum(np.square(powered, dtype=np.float64)))
        ess = (total * total / max(sumsq, 1e-12)) if total > 0.0 else 0.0
        info = FullPERSampleInfo(
            beta=float(beta),
            priority_mean=float(np.mean(valid_priority)) if size > 0 else 0.0,
            priority_max=float(np.max(valid_priority)) if size > 0 else 0.0,
            sample_prob_mean=float(np.mean(sampled_prob)) if sampled_prob.size else 0.0,
            sample_prob_max=float(np.max(sampled_prob)) if sampled_prob.size else 0.0,
            is_weight_mean=float(np.mean(weights)) if weights.size else 1.0,
            is_weight_min=float(np.min(weights)) if weights.size else 1.0,
            is_weight_max=float(np.max(weights)) if weights.size else 1.0,
            priority_ess=float(ess),
        )
        metrics = info.as_metrics()
        metrics["per_full/alpha"] = float(self.alpha)
        metrics["per_full/replay_size"] = float(size)
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_final": self.beta_final,
            "beta_anneal_steps": self.beta_anneal_steps,
            "eps": self.eps,
            "initial_priority": self.initial_priority,
            "min_priority": self.min_priority,
            "priority": self.priority,
            "max_priority": self.max_priority,
        }

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        priority = np.asarray(payload.get("priority", []), dtype=np.float32)
        n = min(priority.size, self.capacity)
        if n > 0:
            self.priority[:n] = priority[:n]
        self.max_priority = float(payload.get("max_priority", float(np.max(self.priority[: max(1, n)]))))
