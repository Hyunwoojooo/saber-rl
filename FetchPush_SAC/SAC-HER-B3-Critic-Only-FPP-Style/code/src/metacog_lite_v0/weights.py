from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from .agent import DDPGAgent
from .config import LiteConfig
from .replay import HERReplayBuffer, SampleBatch

try:
    from .numba_fetch_push_support_scan import (
        FetchPushSupportScanConstants,
        can_scan_fetch_push_support_exact,
        make_fetch_push_support_scan_constants,
        scan_fetch_push_support_exact_with_constants,
        warmup_fetch_push_support_exact,
    )
except Exception:  # pragma: no cover - optional optimization import guard.
    FetchPushSupportScanConstants = Any  # type: ignore[misc, assignment]

    def can_scan_fetch_push_support_exact(task: Any) -> bool:
        del task
        return False

    def make_fetch_push_support_scan_constants(task: Any) -> Any:
        del task
        return None

    def scan_fetch_push_support_exact_with_constants(*args: Any, **kwargs: Any) -> tuple[int, int, int, int, int]:
        del args, kwargs
        raise RuntimeError("FetchPush numba support scanner is unavailable")

    def warmup_fetch_push_support_exact(task: Any, action_dim: int, obs_dim: int) -> bool:
        del task, action_dim, obs_dim
        return False


@dataclass
class WeightBatch:
    omega_critic: np.ndarray
    omega_actor: np.ndarray
    w_p_raw: np.ndarray
    w_p_reference: np.ndarray
    assigned_high_weight: np.ndarray
    reference_high_weight: np.ndarray
    support_weight_overlap: np.ndarray
    g_sup: np.ndarray
    kappa: np.ndarray
    psi_q_rank: np.ndarray
    q_candidate_fallback: np.ndarray
    psi_a_policy: np.ndarray
    len_a_q_sup: np.ndarray
    len_a_q_local: np.ndarray
    num_coarse_candidates: np.ndarray
    num_scanned_candidates: np.ndarray
    local_scan_truncated: np.ndarray
    q_imp_minus_q_policy: np.ndarray
    unrestricted_gap: np.ndarray
    phase: np.ndarray
    timing: Dict[str, float]
    gap_ready: bool
    gap_tau: float
    gap_max: float
    extra_metrics: Dict[str, float]


@dataclass
class LocalScan:
    indices: np.ndarray
    action_bins: np.ndarray
    bucket_version: int
    num_coarse_candidates: int
    local_scan_truncated: bool


@dataclass
class LocalContextResult:
    indices: np.ndarray
    action_bins: np.ndarray
    num_coarse_candidates: int
    num_scanned_candidates: int
    local_scan_truncated: bool
    local_pool_size_direct: int | None = None
    available_support_size_direct: int | None = None
    psi_a_policy_direct: float | None = None
    psi_q_rank_direct: float | None = None
    fallback_direct: float | None = None


class GapStats:
    def __init__(self, maxlen: int, n_min: int, q_low: float, q_high: float, eps_g: float) -> None:
        self.values: deque[float] = deque(maxlen=int(maxlen))
        self.n_min = int(n_min)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps_g = float(eps_g)

    def update(self, gaps: np.ndarray, psi_q_rank: np.ndarray) -> None:
        mask = np.logical_and(gaps > 0.0, psi_q_rank > 0.5)
        for value in gaps[mask]:
            self.values.append(float(value))

    def thresholds(self) -> tuple[bool, float, float]:
        if len(self.values) < self.n_min:
            return False, 0.0, 0.0
        arr = np.asarray(self.values, dtype=np.float32)
        tau_g = float(np.quantile(arr, self.q_low))
        g_max = float(np.quantile(arr, self.q_high))
        ready = (g_max - tau_g) >= self.eps_g
        return bool(ready), tau_g, g_max

    def state_dict(self) -> Dict[str, object]:
        return {"values": list(self.values)}

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.values.clear()
        for value in payload.get("values", []):
            self.values.append(float(value))


class LiteWeightComputer:
    def __init__(self, config: LiteConfig, task: Any, gap_stats: GapStats) -> None:
        self.cfg = config
        self.task = task
        self.gap_stats = gap_stats
        self._support_context_cache: Dict[tuple[Any, ...], LocalContextResult] = {}
        self._local_scan_cache: Dict[tuple[Any, ...], LocalScan] = {}
        self._numba_support_scan_enabled = False
        self._numba_scanner_warmed = False
        self._numba_support_scan_constants: FetchPushSupportScanConstants | None = None
        if bool(getattr(config, "support_numba_scan_enabled", False)):
            try:
                if can_scan_fetch_push_support_exact(task):
                    self._numba_support_scan_constants = make_fetch_push_support_scan_constants(task)
                    if bool(getattr(config, "support_numba_scan_warmup", True)):
                        action_dim = int(np.asarray(task.action_low).shape[0])
                        self._numba_scanner_warmed = warmup_fetch_push_support_exact(
                            task,
                            action_dim=action_dim,
                            obs_dim=6,
                        )
                    self._numba_support_scan_enabled = True
            except Exception:
                self._numba_support_scan_enabled = False
                self._numba_support_scan_constants = None
                self._numba_scanner_warmed = False

    @staticmethod
    def _goal_cache_key(goal: np.ndarray) -> tuple[int, ...]:
        arr = np.asarray(goal, dtype=np.float32)
        return tuple(arr.view(np.uint32).reshape(-1).tolist())

    def _scan_is_deterministic(self) -> bool:
        mode = str(self.cfg.support_scan_mode)
        return mode.startswith("deterministic_") or mode in {"sticky_random_v1", "cached_random_v1"}

    def _can_use_numba_support_scan(self) -> bool:
        return bool(
            self._numba_support_scan_enabled
            and self._numba_support_scan_constants is not None
            and self.cfg.policy_support_mode == "bin"
            and self.cfg.method in {"b3_support_only", "b3_support_random", "b3_support_shuffled"}
        )

    @torch.no_grad()
    def compute(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        rng: np.random.Generator,
        device: torch.device,
        reward_train: np.ndarray | None = None,
        nonterminal: np.ndarray | None = None,
        state_np: np.ndarray | None = None,
        next_state_np: np.ndarray | None = None,
    ) -> WeightBatch:
        t0 = time.perf_counter()
        b = int(batch.observation.shape[0])
        phase, extra_metrics = self._task_batch_diagnostics(batch, b)
        if self.cfg.method == "b0_ddpg_her":
            return self._constant_batch(b, t0, t0, phase=phase, extra_metrics=extra_metrics)

        state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        a_policy = agent.policy_actions(state_t).detach().cpu().numpy().astype(np.float32)
        t_actor = time.perf_counter()

        ready, tau_g, g_max = self.gap_stats.thresholds()
        t_lookup0 = time.perf_counter()
        if self.cfg.method == "b2_unrestricted_gap":
            query_coarse = np.zeros(b, dtype=np.int64)
            query_full = np.zeros(b, dtype=np.int64)
            local_lookup: Dict[int, LocalScan] = {}
        else:
            query_coarse, local_lookup, local_scan_stats = self._prepare_local_lookup(batch, replay, rng)
            query_full = self.task.full_context_hash_batch(
                batch.observation,
                batch.desired_goal_train,
                batch.h,
            ).astype(np.int64, copy=False)
        local_lookup_time = time.perf_counter() - t_lookup0
        if self.cfg.method == "b2_unrestricted_gap":
            local_scan_stats = {
                "local_scan_cache_hit": 0,
                "local_scan_cache_miss": 0,
                "local_scan_cache_attempt": 0,
                "replay_index_direct_scan": 0,
                "replay_index_candidates_materialized": 0,
                "replay_index_live_count_lookup": 0,
                "action_bin_metadata_query": 0,
                "action_bin_unique_compute": 0,
                "action_bin_metadata_size": 0,
                "numba_support_queries": 0,
                "numba_support_fallback": 0,
                "numba_support_scan_time": 0.0,
            }
        local_context_cache: Dict[tuple[Any, ...], LocalContextResult] = {}
        policy_action_bins = (
            self.task.action_bin_batch(a_policy, self.cfg.action_bins_per_dim).astype(np.int64, copy=False)
            if self.cfg.policy_support_mode == "bin"
            else np.full(b, -1, dtype=np.int64)
        )
        if self.cfg.method == "b2_unrestricted_gap":
            query_count = 0
            unique_query_count = 0
        else:
            query_count = int(b)
            goal_cache_keys = [self._goal_cache_key(goal) for goal in batch.desired_goal_train]
            if self._can_use_numba_support_scan():
                support_keys = [
                    (
                        int(query_coarse[j]),
                        int(query_full[j]),
                        goal_cache_keys[j],
                        int(policy_action_bins[j]),
                        "numba_fetch_push_exact_v1",
                    )
                    for j in range(b)
                ]
            else:
                support_keys = [
                    (int(query_coarse[j]), int(query_full[j]), goal_cache_keys[j])
                    for j in range(b)
                ]
            unique_query_count = len(set(support_keys))
        lookup_stats = {
            "query_count": query_count,
            "unique_query_count": unique_query_count,
            "batch_cache_hit": 0,
            "versioned_cache_hit": 0,
            "versioned_cache_miss": 0,
            "versioned_cache_attempt": 0,
            "short_circuit": 0,
            "full_context_scan": 0,
            "full_context_filter_count": 0,
            "local_scan_cache_hit": int(local_scan_stats.get("local_scan_cache_hit", 0)),
            "local_scan_cache_miss": int(local_scan_stats.get("local_scan_cache_miss", 0)),
            "local_scan_cache_attempt": int(local_scan_stats.get("local_scan_cache_attempt", 0)),
            "replay_index_direct_scan": int(local_scan_stats.get("replay_index_direct_scan", 0)),
            "replay_index_candidates_materialized": int(
                local_scan_stats.get("replay_index_candidates_materialized", 0)
            ),
            "replay_index_live_count_lookup": int(local_scan_stats.get("replay_index_live_count_lookup", 0)),
            "action_bin_metadata_query": int(local_scan_stats.get("action_bin_metadata_query", 0)),
            "action_bin_unique_compute": int(local_scan_stats.get("action_bin_unique_compute", 0)),
            "action_bin_metadata_size": int(local_scan_stats.get("action_bin_metadata_size", 0)),
            "numba_support_queries": 0,
            "numba_support_fallback": 0,
            "numba_support_scan_time": 0.0,
        }
        candidate_construction_time = 0.0
        q_policy_time = 0.0
        q_candidate_time = 0.0
        weight_computation_time = 0.0

        g_sup = np.zeros(b, dtype=np.float32)
        unrestricted_gap = np.zeros(b, dtype=np.float32)
        kappa = np.zeros(b, dtype=np.float32)
        psi_q_rank = np.zeros(b, dtype=np.float32)
        fallback = np.ones(b, dtype=np.float32)
        psi_a_policy = np.zeros(b, dtype=np.float32)
        len_sup = np.zeros(b, dtype=np.float32)
        len_local = np.zeros(b, dtype=np.float32)
        coarse_count = np.zeros(b, dtype=np.float32)
        scanned_count = np.zeros(b, dtype=np.float32)
        scan_truncated = np.zeros(b, dtype=np.float32)
        q_delta = np.zeros(b, dtype=np.float32)
        w_raw = np.ones(b, dtype=np.float32)
        max_candidates = int(self.cfg.M_B)
        candidate_actions = np.zeros((b, max_candidates, replay.action_dim), dtype=np.float32)
        candidate_mask = np.zeros((b, max_candidates), dtype=np.bool_)

        for i in range(b):
            if self.cfg.method == "b2_unrestricted_gap":
                t_cand0 = time.perf_counter()
                cand_actions = self._unrestricted_actions(a_policy[i], rng)
                candidate_construction_time += time.perf_counter() - t_cand0
                psi_a_policy[i] = 1.0
                psi_q_rank[i] = 1.0
                fallback[i] = 0.0
            else:
                t_local0 = time.perf_counter()
                local_result = self._local_indices(
                    i,
                    batch,
                    replay,
                    query_coarse,
                    query_full,
                    local_lookup,
                    local_context_cache,
                    lookup_stats,
                    a_policy[i],
                    int(policy_action_bins[i]),
                    goal_cache_keys[i],
                )
                local_lookup_time += time.perf_counter() - t_local0
                coarse_count[i] = float(local_result.num_coarse_candidates)
                scanned_count[i] = float(local_result.num_scanned_candidates)
                scan_truncated[i] = float(local_result.local_scan_truncated)
                if local_result.local_pool_size_direct is not None:
                    local_count = int(local_result.local_pool_size_direct)
                else:
                    local_count = int(local_result.indices.size)
                len_local[i] = float(local_count)
                if local_count < self.cfg.N_local_min:
                    continue
                if local_result.available_support_size_direct is not None:
                    n_sup = int(local_result.available_support_size_direct)
                else:
                    n_sup = min(local_count, int(self.cfg.M_B))
                len_sup[i] = float(n_sup)
                if local_result.psi_a_policy_direct is not None:
                    psi_a_policy[i] = float(local_result.psi_a_policy_direct)
                else:
                    t_support0 = time.perf_counter()
                    psi_a_policy[i] = self._policy_action_supported_from_indices(
                        a_policy[i],
                        replay,
                        local_result.indices,
                        local_result.action_bins,
                        int(policy_action_bins[i]),
                    )
                    candidate_construction_time += time.perf_counter() - t_support0
                if n_sup < self.cfg.M_Q_min or psi_a_policy[i] < 0.5:
                    continue
                psi_q_rank[i] = (
                    float(local_result.psi_q_rank_direct)
                    if local_result.psi_q_rank_direct is not None
                    else 1.0
                )
                fallback[i] = (
                    float(local_result.fallback_direct)
                    if local_result.fallback_direct is not None
                    else 0.0
                )
                if self.cfg.method in {"b3_support_only", "b3_support_random", "b3_support_shuffled"}:
                    w_raw[i] = 1.0 + self.cfg.lambda_support
                    continue

                t_cand0 = time.perf_counter()
                candidate_indices = self._candidate_indices(local_result.indices, rng)
                cand_actions = replay.action_exec[candidate_indices].copy().astype(np.float32)
                candidate_construction_time += time.perf_counter() - t_cand0

            n_cand = min(int(cand_actions.shape[0]), max_candidates)
            len_sup[i] = float(n_cand)
            if n_cand > 0:
                candidate_actions[i, :n_cand] = cand_actions[:n_cand]
                candidate_mask[i, :n_cand] = True

        if self.cfg.method not in {"b3_support_only", "b3_support_random", "b3_support_shuffled"}:
            q_rows = np.logical_and(psi_q_rank > 0.5, np.any(candidate_mask, axis=1))
            if np.any(q_rows):
                t_q_policy0 = time.perf_counter()
                q_policy_np = agent.q_values(
                    state_t,
                    torch.as_tensor(a_policy, dtype=torch.float32, device=device),
                ).detach().cpu().numpy().astype(np.float32)
                q_policy_time = time.perf_counter() - t_q_policy0

                t_q_candidate0 = time.perf_counter()
                x_repeat_flat = np.repeat(state_np[:, None, :], max_candidates, axis=1).reshape(
                    b * max_candidates,
                    state_np.shape[1],
                )
                candidate_flat = candidate_actions.reshape(b * max_candidates, replay.action_dim)
                q_cand_matrix = agent.q_values(
                    torch.as_tensor(x_repeat_flat, dtype=torch.float32, device=device),
                    torch.as_tensor(candidate_flat, dtype=torch.float32, device=device),
                ).detach().cpu().numpy().astype(np.float32).reshape(b, max_candidates)
                q_cand_matrix[np.logical_not(candidate_mask)] = -np.inf
                q_cand_matrix[np.logical_not(q_rows), :] = -np.inf
                q_imp_np = np.max(q_cand_matrix, axis=1)
                q_candidate_time = time.perf_counter() - t_q_candidate0

                t_weight0 = time.perf_counter()
                delta_raw = q_imp_np - q_policy_np
                valid_delta = delta_raw[q_rows]
                q_delta[q_rows] = valid_delta.astype(np.float32)
                delta = np.maximum(0.0, valid_delta).astype(np.float32)
                g_sup[q_rows] = delta
                if self.cfg.method == "b2_unrestricted_gap":
                    unrestricted_gap[q_rows] = delta
                if ready:
                    kappa_vals = np.clip(
                        (delta - tau_g) / (g_max - tau_g + self.cfg.eps),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    kappa[q_rows] = kappa_vals
                    w_raw[q_rows] = 1.0 + self.cfg.lambda_P * np.minimum(self.cfg.U_P_max, kappa_vals)
                weight_computation_time += time.perf_counter() - t_weight0

        t_weight0 = time.perf_counter()
        self.gap_stats.update(g_sup, psi_q_rank)
        local_context_time = local_lookup_time + candidate_construction_time
        reference_w_raw = w_raw.astype(np.float32, copy=True)

        if self.cfg.method == "b3_support_shuffled":
            w_raw = w_raw[rng.permutation(b)].astype(np.float32, copy=True)
        elif self.cfg.method == "b3_support_random":
            positive_count = int(np.count_nonzero(reference_w_raw > (1.0 + self.cfg.eps)))
            randomized = np.ones(b, dtype=np.float32)
            if positive_count > 0:
                chosen = rng.choice(b, size=positive_count, replace=False)
                randomized[chosen] = 1.0 + self.cfg.lambda_support
            w_raw = randomized
        assigned_high = (w_raw > (1.0 + self.cfg.eps)).astype(np.float32)
        reference_high = (reference_w_raw > (1.0 + self.cfg.eps)).astype(np.float32)
        support_weight_overlap = np.logical_and(
            assigned_high > 0.5,
            reference_high > 0.5,
        ).astype(np.float32)

        if self.cfg.method == "b1_uniform_scale":
            # Uniform scale control matches the final Omega scale of the selected
            # weighting mode. Under batch_norm this is 1, under raw it is mean(w_P).
            scale = 1.0 if self.cfg.weighting_mode == "batch_norm" else float(np.mean(w_raw))
            omega = np.full(b, scale, dtype=np.float32)
        elif self.cfg.weighting_mode == "raw":
            omega = w_raw.astype(np.float32)
        else:
            omega = (w_raw / max(self.cfg.eps, float(np.mean(w_raw)))).astype(np.float32)

        if self.cfg.weight_scope == "actor_critic":
            omega_critic = omega
            omega_actor = omega
        elif self.cfg.weight_scope == "critic_only":
            omega_critic = omega
            omega_actor = np.ones(b, dtype=np.float32)
        else:
            omega_critic = np.ones(b, dtype=np.float32)
            omega_actor = omega
        weight_computation_time += time.perf_counter() - t_weight0

        extra_metrics_out = dict(extra_metrics)
        lookup_count = (
            lookup_stats["versioned_cache_hit"]
            + lookup_stats["versioned_cache_miss"]
            + lookup_stats["short_circuit"]
        )
        lookup_denom = max(1, int(lookup_count))
        cache_attempts = lookup_stats["versioned_cache_hit"] + lookup_stats["versioned_cache_miss"]
        cache_denom = max(1, int(cache_attempts))
        support_cache_attempt_denom = max(1, int(lookup_stats["query_count"]))
        local_scan_cache_attempts = (
            lookup_stats["local_scan_cache_hit"] + lookup_stats["local_scan_cache_miss"]
        )
        local_scan_cache_denom = max(1, int(local_scan_cache_attempts))
        query_denom = max(1, int(lookup_stats["query_count"]))
        unique_rate = float(lookup_stats["unique_query_count"] / query_denom)
        coalescing_saving_rate = (
            max(0.0, 1.0 - unique_rate) if self.cfg.support_batch_query_coalescing else 0.0
        )
        exact_short_circuit_rate = float(lookup_stats["short_circuit"] / query_denom)
        full_context_scan_rate = float(lookup_stats["full_context_scan"] / query_denom)
        replay_index_lookup_denom = max(1, int(lookup_stats["replay_index_direct_scan"]))
        action_bin_query_denom = max(1, int(lookup_stats["action_bin_metadata_query"]))
        numba_query_denom = max(1, int(lookup_stats["numba_support_queries"]))
        if self.cfg.method != "b0_ddpg_her":
            extra_metrics_out.update(
                {
                    "lite/support_lookup_optimization_enabled": float(self.cfg.support_lookup_optimization),
                    "lite/support_cache_enabled": float(self.cfg.support_cache_enabled),
                    "lite/support_short_circuit_enabled": float(self.cfg.support_short_circuit),
                    "lite/support_batch_query_coalescing_enabled": float(
                        self.cfg.support_batch_query_coalescing
                    ),
                    "lite/support_deterministic_scan_enabled": float(self._scan_is_deterministic()),
                    "lite/support_scan_mode_is_sticky_random": float(
                        str(self.cfg.support_scan_mode) == "sticky_random_v1"
                    ),
                    "lite/support_scan_seed": float(self.cfg.support_scan_seed),
                    "lite/support_query_count": float(lookup_stats["query_count"]),
                    "lite/support_unique_query_count": float(lookup_stats["unique_query_count"]),
                    "lite/support_batch_cache_hit_count": float(lookup_stats["batch_cache_hit"]),
                    "lite/support_versioned_cache_hit_count": float(lookup_stats["versioned_cache_hit"]),
                    "lite/support_versioned_cache_miss_count": float(lookup_stats["versioned_cache_miss"]),
                    "lite/support_versioned_cache_attempt_count": float(
                        lookup_stats["versioned_cache_attempt"]
                    ),
                    "lite/support_short_circuit_count": float(lookup_stats["short_circuit"]),
                    "lite/support_full_context_scan_count": float(lookup_stats["full_context_scan"]),
                    "lite/support_full_context_filter_count": float(lookup_stats["full_context_filter_count"]),
                    "lite/local_scan_cache_hit_count": float(lookup_stats["local_scan_cache_hit"]),
                    "lite/local_scan_cache_miss_count": float(lookup_stats["local_scan_cache_miss"]),
                    "lite/local_scan_cache_attempt_count": float(
                        lookup_stats["local_scan_cache_attempt"]
                    ),
                    "lite/replay_index_direct_scan_count": float(lookup_stats["replay_index_direct_scan"]),
                    "lite/replay_index_candidates_materialized_count": float(
                        lookup_stats["replay_index_candidates_materialized"]
                    ),
                    "lite/replay_index_live_count_lookup_count": float(
                        lookup_stats["replay_index_live_count_lookup"]
                    ),
                    "lite/action_bin_metadata_query_count": float(
                        lookup_stats["action_bin_metadata_query"]
                    ),
                    "lite/action_bin_unique_compute_count": float(
                        lookup_stats["action_bin_unique_compute"]
                    ),
                    "lite/action_bin_metadata_size": float(lookup_stats["action_bin_metadata_size"]),
                    "lite/support_versioned_cache_hit_rate": float(
                        lookup_stats["versioned_cache_hit"] / lookup_denom
                    ),
                    "lite/support_short_circuit_rate": float(lookup_stats["short_circuit"] / lookup_denom),
                    "runtime/exact_negative_short_circuit_rate": exact_short_circuit_rate,
                    "runtime/full_context_scan_rate": full_context_scan_rate,
                    "runtime/support_cache_hit_rate": float(
                        lookup_stats["versioned_cache_hit"] / cache_denom
                    ),
                    "runtime/support_cache_miss_rate": float(
                        lookup_stats["versioned_cache_miss"] / cache_denom
                    ),
                    "runtime/support_cache_attempt_rate": float(
                        lookup_stats["versioned_cache_attempt"] / support_cache_attempt_denom
                    ),
                    "runtime/support_context_cache_size": float(len(self._support_context_cache)),
                    "runtime/batch_query_unique_rate": unique_rate,
                    "runtime/batch_query_coalescing_saving_rate": coalescing_saving_rate,
                    "runtime/local_scan_cache_hit_rate": float(
                        lookup_stats["local_scan_cache_hit"] / local_scan_cache_denom
                    ),
                    "runtime/local_scan_cache_miss_rate": float(
                        lookup_stats["local_scan_cache_miss"] / local_scan_cache_denom
                    ),
                    "runtime/local_scan_cache_attempt_rate": float(
                        lookup_stats["local_scan_cache_attempt"] / max(1, int(lookup_stats["unique_query_count"]))
                    ),
                    "runtime/local_scan_cache_size": float(len(self._local_scan_cache)),
                    "runtime/full_context_filter_count_mean": float(
                        lookup_stats["full_context_filter_count"] / max(1, int(lookup_stats["full_context_scan"]))
                    ),
                    "runtime/replay_index_direct_scan_count": float(
                        lookup_stats["replay_index_direct_scan"]
                    ),
                    "runtime/replay_index_candidates_materialized_count": float(
                        lookup_stats["replay_index_candidates_materialized"]
                    ),
                    "runtime/replay_index_candidates_materialized_rate": float(
                        lookup_stats["replay_index_candidates_materialized"] / replay_index_lookup_denom
                    ),
                    "runtime/replay_index_live_count_lookup_rate": float(
                        lookup_stats["replay_index_live_count_lookup"] / replay_index_lookup_denom
                    ),
                    "runtime/action_bin_metadata_query_count": float(
                        lookup_stats["action_bin_metadata_query"]
                    ),
                    "runtime/action_bin_unique_compute_count": float(
                        lookup_stats["action_bin_unique_compute"]
                    ),
                    "runtime/action_bin_unique_compute_rate": float(
                        lookup_stats["action_bin_unique_compute"] / action_bin_query_denom
                    ),
                    "runtime/action_bin_metadata_size": float(
                        lookup_stats["action_bin_metadata_size"]
                    ),
                    "runtime/numba_support_scan_enabled": float(self._can_use_numba_support_scan()),
                    "runtime/numba_support_scan_fallback_rate": float(
                        lookup_stats["numba_support_fallback"] / numba_query_denom
                    ),
                    "runtime/numba_support_queries": float(lookup_stats["numba_support_queries"]),
                    "runtime/numba_scanner_warmed": float(self._numba_scanner_warmed),
                    "time/numba_support_scan": float(lookup_stats["numba_support_scan_time"]),
                }
            )

        return WeightBatch(
            omega_critic=omega_critic.astype(np.float32),
            omega_actor=omega_actor.astype(np.float32),
            w_p_raw=w_raw,
            w_p_reference=reference_w_raw,
            assigned_high_weight=assigned_high,
            reference_high_weight=reference_high,
            support_weight_overlap=support_weight_overlap,
            g_sup=g_sup,
            kappa=kappa,
            psi_q_rank=psi_q_rank,
            q_candidate_fallback=fallback,
            psi_a_policy=psi_a_policy,
            len_a_q_sup=len_sup,
            len_a_q_local=len_local,
            num_coarse_candidates=coarse_count,
            num_scanned_candidates=scanned_count,
            local_scan_truncated=scan_truncated,
            q_imp_minus_q_policy=q_delta,
            unrestricted_gap=unrestricted_gap,
            phase=phase,
            timing={
                "actor": t_actor - t0,
                "local_context_lookup": local_lookup_time,
                "candidate_construction": candidate_construction_time,
                "local_context": local_context_time,
                "q_policy_eval": q_policy_time,
                "q_candidate_eval": q_candidate_time,
                "weight_compute": weight_computation_time,
                "weight_computation": weight_computation_time,
                "numba_support_scan": float(lookup_stats["numba_support_scan_time"]),
            },
            gap_ready=ready,
            gap_tau=tau_g,
            gap_max=g_max,
            extra_metrics=extra_metrics_out,
        )

    def _task_batch_diagnostics(self, batch: SampleBatch, b: int) -> tuple[np.ndarray, Dict[str, float]]:
        if hasattr(self.task, "phase_batch"):
            phase = self.task.phase_batch(batch.observation, batch.desired_goal_train).astype(np.int64)
        else:
            phase = np.zeros(b, dtype=np.int64)
        if hasattr(self.task, "diagnostic_metrics"):
            extra_metrics = self.task.diagnostic_metrics(batch.observation, batch.desired_goal_train)
        else:
            extra_metrics = {}
        return phase, extra_metrics

    def _constant_batch(
        self,
        b: int,
        t0: float,
        t_actor: float,
        phase: np.ndarray | None = None,
        extra_metrics: Dict[str, float] | None = None,
    ) -> WeightBatch:
        ones = np.ones(b, dtype=np.float32)
        zeros = np.zeros(b, dtype=np.float32)
        if phase is None:
            phase = zeros.astype(np.int64)
        return WeightBatch(
            omega_critic=ones.copy(),
            omega_actor=ones.copy(),
            w_p_raw=ones.copy(),
            w_p_reference=ones.copy(),
            assigned_high_weight=zeros.copy(),
            reference_high_weight=zeros.copy(),
            support_weight_overlap=zeros.copy(),
            g_sup=zeros.copy(),
            kappa=zeros.copy(),
            psi_q_rank=zeros.copy(),
            q_candidate_fallback=ones.copy(),
            psi_a_policy=zeros.copy(),
            len_a_q_sup=zeros.copy(),
            len_a_q_local=zeros.copy(),
            num_coarse_candidates=zeros.copy(),
            num_scanned_candidates=zeros.copy(),
            local_scan_truncated=zeros.copy(),
            q_imp_minus_q_policy=zeros.copy(),
            unrestricted_gap=zeros.copy(),
            phase=phase.astype(np.int64, copy=False),
            timing={
                "actor": t_actor - t0,
                "local_context_lookup": 0.0,
                "candidate_construction": 0.0,
                "local_context": 0.0,
                "q_policy_eval": 0.0,
                "q_candidate_eval": 0.0,
                "weight_compute": 0.0,
                "weight_computation": 0.0,
                "numba_support_scan": 0.0,
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics={} if extra_metrics is None else extra_metrics,
        )

    def _prepare_local_lookup(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, Dict[int, LocalScan], Dict[str, int]]:
        query_coarse = self.task.coarse_hash_batch(batch.observation, batch.h)
        stats = {
            "local_scan_cache_hit": 0,
            "local_scan_cache_miss": 0,
            "local_scan_cache_attempt": 0,
            "replay_index_direct_scan": 0,
            "replay_index_candidates_materialized": 0,
            "replay_index_live_count_lookup": 0,
            "action_bin_metadata_query": 0,
            "action_bin_unique_compute": 0,
            "action_bin_metadata_size": 0,
        }
        if replay.size == 0:
            return query_coarse, {}, stats

        lookup: Dict[int, LocalScan] = {}
        scan_is_deterministic = self._scan_is_deterministic()
        use_scan_cache = (
            self.cfg.support_local_scan_cache_enabled
            and self.cfg.support_local_scan_cache_max_entries > 0
            and scan_is_deterministic
        )
        for key in np.unique(query_coarse):
            key_i = int(key)
            bucket_version = replay.action_index.version(key_i)
            scan_cache_key = (
                key_i,
                int(bucket_version),
                int(self.cfg.local_search_max),
                int(self.cfg.action_bins_per_dim),
                str(self.cfg.policy_support_mode),
                str(self.cfg.support_scan_mode),
                int(self.cfg.support_scan_seed),
            )
            if use_scan_cache:
                stats["local_scan_cache_attempt"] += 1
                cached = self._local_scan_cache.get(scan_cache_key)
                if cached is not None:
                    stats["local_scan_cache_hit"] += 1
                    lookup[key_i] = cached
                    continue
                stats["local_scan_cache_miss"] += 1
            scanned, n_candidates, truncated = replay.indexed_reference_indices(
                coarse_key=key_i,
                max_n=self.cfg.local_search_max,
                rng=rng,
                deterministic=scan_is_deterministic,
                scan_mode=str(self.cfg.support_scan_mode),
                scan_seed=int(self.cfg.support_scan_seed),
            )
            stats["replay_index_direct_scan"] += 1
            stats["replay_index_live_count_lookup"] += 1
            if self.cfg.support_short_circuit and self.cfg.policy_support_mode == "bin":
                action_bins = replay.action_index.action_bins(key_i)
                stats["action_bin_metadata_query"] += 1
                stats["action_bin_metadata_size"] += int(action_bins.size)
            else:
                action_bins = np.zeros((0,), dtype=np.int64)
            scan = LocalScan(
                indices=scanned,
                action_bins=action_bins,
                bucket_version=bucket_version,
                num_coarse_candidates=n_candidates,
                local_scan_truncated=bool(truncated),
            )
            lookup[key_i] = scan
            if use_scan_cache:
                if len(self._local_scan_cache) >= self.cfg.support_local_scan_cache_max_entries:
                    self._local_scan_cache.clear()
                    self._support_context_cache.clear()
                self._local_scan_cache[scan_cache_key] = scan
        return query_coarse, lookup, stats

    def _local_indices(
        self,
        i: int,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        query_coarse: np.ndarray,
        query_full: np.ndarray,
        local_lookup: Dict[int, LocalScan],
        local_context_cache: Dict[tuple[Any, ...], LocalContextResult],
        lookup_stats: Dict[str, int],
        action_policy: np.ndarray,
        policy_action_bin: int,
        goal_key: tuple[int, ...],
    ) -> LocalContextResult:
        coarse_key = int(query_coarse[i])
        goal = batch.desired_goal_train[i]
        full_key = int(query_full[i])
        use_numba_scan = self._can_use_numba_support_scan()
        scanner_tag = "numba_fetch_push_exact_v1"
        cache_key = (
            (coarse_key, full_key, goal_key, int(policy_action_bin), scanner_tag)
            if use_numba_scan
            else (coarse_key, full_key, goal_key)
        )
        if self.cfg.support_batch_query_coalescing:
            cached = local_context_cache.get(cache_key)
            if cached is not None:
                lookup_stats["batch_cache_hit"] += 1
                return cached

        scan = local_lookup.get(coarse_key)
        if scan is None or scan.indices.size == 0:
            empty_result = LocalContextResult(
                indices=np.zeros((0,), dtype=np.int64),
                action_bins=np.zeros((0,), dtype=np.int64),
                num_coarse_candidates=0,
                num_scanned_candidates=0,
                local_scan_truncated=False,
            )
            if self.cfg.support_batch_query_coalescing:
                local_context_cache[cache_key] = empty_result
            return empty_result
        if (
            self.cfg.support_short_circuit
            and self.cfg.policy_support_mode == "bin"
            and scan.action_bins.size > 0
        ):
            p_bin = int(policy_action_bin)
            if not bool(np.any(scan.action_bins == p_bin)):
                lookup_stats["short_circuit"] += 1
                empty = np.zeros((0,), dtype=np.int64)
                return LocalContextResult(
                    indices=empty,
                    action_bins=empty,
                    num_coarse_candidates=scan.num_coarse_candidates,
                    num_scanned_candidates=int(scan.indices.size),
                    local_scan_truncated=scan.local_scan_truncated,
                )

        scan_cacheable = self._scan_is_deterministic() or not scan.local_scan_truncated
        can_persist = (
            self.cfg.support_cache_enabled
            and self.cfg.support_cache_max_entries > 0
            and scan_cacheable
        )
        persistent_key = (
            (
                coarse_key,
                full_key,
                goal_key,
                int(policy_action_bin),
                int(scan.bucket_version),
                int(self.cfg.local_search_max),
                int(self.cfg.action_bins_per_dim),
                str(self.cfg.policy_support_mode),
                str(self.cfg.support_scan_mode),
                int(self.cfg.support_scan_seed),
                scanner_tag,
            )
            if use_numba_scan
            else (
                coarse_key,
                full_key,
                goal_key,
                int(scan.bucket_version),
                int(self.cfg.local_search_max),
                int(self.cfg.action_bins_per_dim),
                str(self.cfg.policy_support_mode),
                str(self.cfg.support_scan_mode),
                int(self.cfg.support_scan_seed),
            )
        )
        if can_persist:
            lookup_stats["versioned_cache_attempt"] += 1
            persistent = self._support_context_cache.get(persistent_key)
            if persistent is not None:
                lookup_stats["versioned_cache_hit"] += 1
                if self.cfg.support_batch_query_coalescing:
                    local_context_cache[cache_key] = persistent
                return persistent
            lookup_stats["versioned_cache_miss"] += 1

        if use_numba_scan:
            constants = self._numba_support_scan_constants
            if constants is not None:
                lookup_stats["numba_support_queries"] += 1
                t_numba0 = time.perf_counter()
                try:
                    (
                        local_pool_size,
                        available_support_size,
                        psi_a_policy_value,
                        support_positive,
                        fallback_value,
                    ) = scan_fetch_push_support_exact_with_constants(
                        replay_obs=replay.observation,
                        replay_h=replay.h,
                        replay_action_bin=replay.action_bin,
                        replay_action_exec=replay.action_exec,
                        scanned_indices=scan.indices,
                        query_goal=goal,
                        query_full_key=full_key,
                        query_policy_bin=int(policy_action_bin),
                        constants=constants,
                        n_local_min=int(self.cfg.N_local_min),
                        m_q_min=int(self.cfg.M_Q_min),
                        m_b=int(self.cfg.M_B),
                        action_bins_per_dim=int(self.cfg.action_bins_per_dim),
                    )
                except Exception:
                    lookup_stats["numba_support_fallback"] += 1
                    self._numba_support_scan_enabled = False
                    self._numba_support_scan_constants = None
                else:
                    lookup_stats["numba_support_scan_time"] += time.perf_counter() - t_numba0
                    lookup_stats["full_context_scan"] += 1
                    lookup_stats["full_context_filter_count"] += int(scan.indices.size)
                    result = LocalContextResult(
                        indices=np.zeros((0,), dtype=np.int64),
                        action_bins=np.zeros((0,), dtype=np.int64),
                        num_coarse_candidates=scan.num_coarse_candidates,
                        num_scanned_candidates=int(scan.indices.size),
                        local_scan_truncated=scan.local_scan_truncated,
                        local_pool_size_direct=int(local_pool_size),
                        available_support_size_direct=int(available_support_size),
                        psi_a_policy_direct=float(psi_a_policy_value),
                        psi_q_rank_direct=float(support_positive),
                        fallback_direct=float(fallback_value),
                    )
                    if self.cfg.support_batch_query_coalescing:
                        local_context_cache[cache_key] = result
                    if can_persist:
                        if len(self._support_context_cache) >= self.cfg.support_cache_max_entries:
                            self._support_context_cache.clear()
                        self._support_context_cache[persistent_key] = result
                    return result

        ref_obs = replay.observation[scan.indices]
        ref_goal = np.repeat(goal[None, :], scan.indices.size, axis=0)
        ref_h = replay.h[scan.indices]
        lookup_stats["full_context_scan"] += 1
        lookup_stats["full_context_filter_count"] += int(scan.indices.size)
        ref_hash = self.task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
        local_idx = scan.indices[ref_hash == full_key]
        local_bins = (
            replay.action_bin[local_idx].copy()
            if local_idx.size > 0
            else np.zeros((0,), dtype=np.int64)
        )
        result = LocalContextResult(
            indices=local_idx.astype(np.int64, copy=True),
            action_bins=local_bins.astype(np.int64, copy=False),
            num_coarse_candidates=scan.num_coarse_candidates,
            num_scanned_candidates=int(scan.indices.size),
            local_scan_truncated=scan.local_scan_truncated,
        )
        if self.cfg.support_batch_query_coalescing:
            local_context_cache[cache_key] = result
        if can_persist:
            if len(self._support_context_cache) >= self.cfg.support_cache_max_entries:
                self._support_context_cache.clear()
            self._support_context_cache[persistent_key] = result
        return result

    def _candidate_indices(self, local_indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if local_indices.size <= self.cfg.M_B:
            return local_indices.astype(np.int64, copy=True)
        pos = rng.choice(local_indices.size, size=self.cfg.M_B, replace=False)
        return local_indices[pos].astype(np.int64, copy=True)

    def _unrestricted_actions(self, action_policy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(
            0.0,
            self.cfg.unrestricted_candidate_std,
            size=(self.cfg.M_B, action_policy.shape[0]),
        ).astype(np.float32)
        cand = action_policy[None, :] + noise
        return np.clip(cand, self.task.action_low, self.task.action_high).astype(np.float32)

    def _policy_action_supported(
        self,
        action_policy: np.ndarray,
        local_actions: np.ndarray,
        local_action_bins: np.ndarray,
    ) -> float:
        if local_actions.shape[0] == 0:
            return 0.0
        if self.cfg.policy_support_mode == "radius":
            p = self.task.normalize_action(action_policy[None, :])
            local = self.task.normalize_action(local_actions)
            dist = np.linalg.norm(local - p, axis=1) / np.sqrt(float(local.shape[1]))
            return float(np.min(dist) <= self.cfg.policy_action_radius)
        p_bin = int(self.task.action_bin_batch(action_policy[None, :], self.cfg.action_bins_per_dim)[0])
        if local_action_bins.shape[0] == local_actions.shape[0] and np.all(local_action_bins >= 0):
            local_bins = local_action_bins
        else:
            local_bins = self.task.action_bin_batch(local_actions, self.cfg.action_bins_per_dim)
        return float(np.any(local_bins == p_bin))

    def _policy_action_supported_from_indices(
        self,
        action_policy: np.ndarray,
        replay: HERReplayBuffer,
        local_indices: np.ndarray,
        local_action_bins: np.ndarray,
        policy_action_bin: int,
    ) -> float:
        if local_indices.size == 0:
            return 0.0
        if self.cfg.policy_support_mode == "radius":
            local_actions = replay.action_exec[local_indices]
            return self._policy_action_supported(action_policy, local_actions, local_action_bins)

        p_bin = int(policy_action_bin)
        if local_action_bins.shape[0] == local_indices.size and np.all(local_action_bins >= 0):
            local_bins = local_action_bins
        else:
            local_bins = self.task.action_bin_batch(
                replay.action_exec[local_indices],
                self.cfg.action_bins_per_dim,
            )
        return float(np.any(local_bins == p_bin))

    def _q_gap(
        self,
        agent: DDPGAgent,
        state: np.ndarray,
        action_policy: np.ndarray,
        cand_actions: np.ndarray,
        device: torch.device,
    ) -> tuple[float, float]:
        state_rep = np.repeat(state[None, :], cand_actions.shape[0], axis=0)
        q_cand = agent.q_values(
            torch.as_tensor(state_rep, dtype=torch.float32, device=device),
            torch.as_tensor(cand_actions, dtype=torch.float32, device=device),
        )
        q_imp = float(q_cand.max().item())
        q_policy = float(
            agent.q_values(
                torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(action_policy, dtype=torch.float32, device=device).unsqueeze(0),
            ).item()
        )
        return q_imp, q_policy
