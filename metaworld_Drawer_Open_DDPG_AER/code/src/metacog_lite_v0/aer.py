from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class AERSelection:
    candidate_indices: np.ndarray
    selected_positions: np.ndarray
    candidate_similarity: np.ndarray
    lambda_value: float
    query_norm: float

    @property
    def selected_indices(self) -> np.ndarray:
        return self.candidate_indices[self.selected_positions]

    @property
    def selected_similarity(self) -> np.ndarray:
        return self.candidate_similarity[self.selected_positions]


def annealed_lambda(step: int, total_steps: int, lambda0: float, alpha: float) -> float:
    horizon = float(alpha) * float(total_steps)
    if horizon <= 0.0:
        raise ValueError("AER annealing horizon must be positive")
    fraction = max(0.0, 1.0 - float(step) / horizon)
    return float(1.0 + (float(lambda0) - 1.0) * fraction)


def attentive_candidate_count(lambda_value: float, batch_size: int) -> int:
    return max(int(batch_size), int(np.ceil(float(lambda_value) * int(batch_size))))


def cosine_similarity(query: np.ndarray, candidates: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    query_arr = np.asarray(query, dtype=np.float32).reshape(-1)
    candidate_arr = np.asarray(candidates, dtype=np.float32)
    if candidate_arr.ndim != 2 or candidate_arr.shape[1] != query_arr.shape[0]:
        raise ValueError(
            f"AER cosine shape mismatch: query={query_arr.shape}, candidates={candidate_arr.shape}"
        )
    query_norm = float(np.linalg.norm(query_arr))
    candidate_norm = np.linalg.norm(candidate_arr, axis=1)
    denom = np.maximum(float(eps), candidate_norm * max(float(eps), query_norm))
    similarity = (candidate_arr @ query_arr) / denom
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return similarity, query_norm


def top_k_positions(similarity: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(similarity, dtype=np.float32).reshape(-1)
    if k <= 0 or k > values.size:
        raise ValueError(f"Invalid top-k request: k={k}, candidates={values.size}")
    if k == values.size:
        return np.arange(values.size, dtype=np.int64)
    positions = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((positions, -values))
    return order[:k].astype(np.int64)


def select_attentive_indices(
    replay: Any,
    query_state: np.ndarray,
    batch_size: int,
    lambda_value: float,
    rng: np.random.Generator,
    eps: float,
) -> AERSelection:
    candidate_n = attentive_candidate_count(lambda_value, batch_size)
    candidate_indices = rng.integers(
        0, replay.size, size=candidate_n, endpoint=False, dtype=np.int64
    )
    candidate_states = np.concatenate(
        [replay.observation[candidate_indices], replay.desired_goal[candidate_indices]],
        axis=1,
    ).astype(np.float32)
    similarity, query_norm = cosine_similarity(query_state, candidate_states, eps)
    return AERSelection(
        candidate_indices=candidate_indices,
        selected_positions=top_k_positions(similarity, int(batch_size)),
        candidate_similarity=similarity,
        lambda_value=float(lambda_value),
        query_norm=query_norm,
    )


def describe_uniform_indices(
    replay: Any,
    query_state: np.ndarray,
    indices: np.ndarray,
    eps: float,
    lambda_value: float,
) -> AERSelection:
    candidate_indices = np.asarray(indices, dtype=np.int64).copy()
    candidate_states = np.concatenate(
        [replay.observation[candidate_indices], replay.desired_goal[candidate_indices]],
        axis=1,
    ).astype(np.float32)
    similarity, query_norm = cosine_similarity(query_state, candidate_states, eps)
    return AERSelection(
        candidate_indices=candidate_indices,
        selected_positions=np.arange(candidate_indices.size, dtype=np.int64),
        candidate_similarity=similarity,
        lambda_value=float(lambda_value),
        query_norm=query_norm,
    )


def _duplicate_rate(indices: np.ndarray) -> float:
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return 0.0
    return float(1.0 - np.unique(values).size / values.size)


def selection_metrics(
    selection: AERSelection,
    replay: Any,
    task: Any,
    sampling_sec: float,
    effective_actor_lr: float,
    effective_critic_lr: float,
    buffer_success_rate: float,
) -> Dict[str, float]:
    candidate_indices = selection.candidate_indices
    selected_indices = selection.selected_indices
    candidate_similarity = selection.candidate_similarity
    selected_similarity = selection.selected_similarity
    candidate_age = replay.transition_ages(candidate_indices)
    selected_age = replay.transition_ages(selected_indices)
    candidate_success = replay.reward_ext[candidate_indices] >= -1e-8
    selected_success = replay.reward_ext[selected_indices] >= -1e-8
    selected_phase = task.phase_batch(
        replay.observation[selected_indices], replay.desired_goal[selected_indices]
    )
    metrics: Dict[str, float] = {
        "aer/lambda": float(selection.lambda_value),
        "aer/candidate_count": float(candidate_indices.size),
        "aer/candidate_similarity_mean": float(np.mean(candidate_similarity)),
        "aer/candidate_similarity_std": float(np.std(candidate_similarity)),
        "aer/candidate_similarity_min": float(np.min(candidate_similarity)),
        "aer/candidate_similarity_max": float(np.max(candidate_similarity)),
        "aer/selected_similarity_mean": float(np.mean(selected_similarity)),
        "aer/selected_similarity_std": float(np.std(selected_similarity)),
        "aer/selected_similarity_min": float(np.min(selected_similarity)),
        "aer/selected_similarity_max": float(np.max(selected_similarity)),
        "aer/selected_similarity_uplift": float(
            np.mean(selected_similarity) - np.mean(candidate_similarity)
        ),
        "aer/candidate_duplicate_rate": _duplicate_rate(candidate_indices),
        "aer/selected_duplicate_rate": _duplicate_rate(selected_indices),
        "aer/candidate_transition_age_mean": float(np.mean(candidate_age)),
        "aer/selected_transition_age_mean": float(np.mean(selected_age)),
        "aer/candidate_success_transition_rate": float(np.mean(candidate_success)),
        "aer/selected_success_transition_rate": float(np.mean(selected_success)),
        "aer/buffer_success_transition_rate": float(buffer_success_rate),
        "aer/query_norm": float(selection.query_norm),
        "aer/query_zero_norm_rate": float(selection.query_norm <= 1e-8),
        "aer/effective_actor_lr": float(effective_actor_lr),
        "aer/effective_critic_lr": float(effective_critic_lr),
        "runtime/aer_sampling_sec_per_update": float(sampling_sec),
    }
    for phase_id, phase_name in enumerate(task.phase_names):
        metrics[f"aer/selected_phase_{phase_name}_rate"] = float(
            np.mean(selected_phase == phase_id)
        )
    return metrics
