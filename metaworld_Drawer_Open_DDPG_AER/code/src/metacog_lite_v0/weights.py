from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from .agent import DDPGAgent
from .config import LiteConfig
from .replay import HERReplayBuffer, SampleBatch
from .numba_support_scan_drawer_open import (
    NUMBA_AVAILABLE,
    numba_support_predicate_nopython_signatures,
    numba_support_scan_nopython_signatures,
    scan_drawer_open_support_exact,
    scan_drawer_open_support_predicate,
)


@dataclass
class WeightBatch:
    omega_critic: np.ndarray
    omega_actor: np.ndarray
    w_p_raw: np.ndarray
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
    coarse_key: int
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
    local_pool_size: int | None = None
    available_support_size: int | None = None
    psi_a_policy: float | None = None
    threshold_short_circuit: bool = False
    coarse_count_short_circuit: bool = False
    coarse_support_size_short_circuit: bool = False
    scanned_count_short_circuit: bool = False
    coarse_action_bin_count_lookup: bool = False
    predicate_early_exit: bool = False


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
        self._support_cache: OrderedDict[tuple[Any, ...], LocalContextResult] = OrderedDict()
        self._local_scan_cache: OrderedDict[tuple[Any, ...], LocalScan] = OrderedDict()
        self._runtime_context_calls = 0
        self._numba_scan_params = self._build_numba_scan_params(task)
        self._numba_support_scan_enabled = bool(
            getattr(config, "numba_support_scan", True)
            and NUMBA_AVAILABLE
            and self._numba_scan_params is not None
        )
        self._numba_scanner_warmed = False
        if self._numba_support_scan_enabled:
            self._numba_scanner_warmed = self._warmup_numba_scanner()
            self._numba_support_scan_enabled = bool(self._numba_scanner_warmed)

    @torch.no_grad()
    def compute(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        rng: np.random.Generator,
        device: torch.device,
    ) -> WeightBatch:
        t0 = time.perf_counter()
        b = int(batch.observation.shape[0])
        phase, extra_metrics = self._task_batch_diagnostics(batch, b)
        if self.cfg.method in {
            "b0_ddpg_er",
            "b0_ddpg_her",
            "td_error",
            "per_style_loss_weighting",
        }:
            return self._constant_batch(b, t0, t0, phase=phase, extra_metrics=extra_metrics)
        if self.cfg.method == "b3_fully_random_fixed_count":
            return self._fully_random_fixed_count_batch(b, t0, rng, phase=phase, extra_metrics=extra_metrics)

        state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        a_policy = agent.policy_actions(state_t).detach().cpu().numpy().astype(np.float32)
        t_actor = time.perf_counter()

        ready, tau_g, g_max = self.gap_stats.thresholds()
        runtime_opt_contexts: list[LocalContextResult] | None = None
        if self.cfg.method == "b2_unrestricted_gap":
            t_lookup0 = time.perf_counter()
            query_coarse = np.zeros(b, dtype=np.int64)
            local_lookup: Dict[int, LocalScan] = {}
            local_lookup_time = time.perf_counter() - t_lookup0
        elif self.cfg.method in {
            "b3_support_runtime_opt",
            "b3_support_random_runtime_opt",
            "b3_support_shuffled_runtime_opt",
        }:
            query_coarse, runtime_opt_contexts, runtime_metrics, local_lookup_time = self._prepare_runtime_opt_contexts(
                batch, replay, a_policy
            )
            local_lookup = {}
            extra_metrics.update(runtime_metrics)
        else:
            t_lookup0 = time.perf_counter()
            query_coarse, local_lookup = self._prepare_local_lookup(batch, replay, rng)
            local_lookup_time = time.perf_counter() - t_lookup0
        local_context_cache: Dict[tuple[int, int], LocalContextResult] = {}
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
        candidate_action_norm = np.zeros(b, dtype=np.float32)
        action_delta_norm = np.zeros(b, dtype=np.float32)

        for i in range(b):
            if self.cfg.method == "b2_unrestricted_gap":
                t_cand0 = time.perf_counter()
                cand_actions = self._unrestricted_actions(a_policy[i], rng)
                candidate_construction_time += time.perf_counter() - t_cand0
                cand_norm = self.task.normalize_action(cand_actions)
                policy_norm = self.task.normalize_action(a_policy[i][None, :])
                candidate_action_norm[i] = float(np.mean(np.linalg.norm(cand_norm, axis=1)))
                action_delta_norm[i] = float(np.mean(np.linalg.norm(cand_norm - policy_norm, axis=1)))
                psi_a_policy[i] = 1.0
                psi_q_rank[i] = 1.0
                fallback[i] = 0.0
            else:
                if runtime_opt_contexts is not None:
                    local_context = runtime_opt_contexts[i]
                    local_indices = local_context.indices
                    local_action_bins = local_context.action_bins
                    n_coarse = local_context.num_coarse_candidates
                    n_scanned = local_context.num_scanned_candidates
                    was_truncated = local_context.local_scan_truncated
                else:
                    t_local0 = time.perf_counter()
                    local_indices, local_action_bins, n_coarse, n_scanned, was_truncated = self._local_indices(
                        i,
                        batch,
                        replay,
                        query_coarse,
                        local_lookup,
                        local_context_cache,
                    )
                    local_lookup_time += time.perf_counter() - t_local0
                coarse_count[i] = float(n_coarse)
                scanned_count[i] = float(n_scanned)
                scan_truncated[i] = float(was_truncated)
                if runtime_opt_contexts is not None and local_context.local_pool_size is not None:
                    local_count = int(local_context.local_pool_size)
                else:
                    local_count = int(local_indices.size)
                len_local[i] = float(local_count)
                if local_count < self.cfg.N_local_min:
                    continue
                if runtime_opt_contexts is not None and local_context.available_support_size is not None:
                    n_sup = int(local_context.available_support_size)
                else:
                    n_sup = min(local_count, int(self.cfg.M_B))
                len_sup[i] = float(n_sup)
                t_support0 = time.perf_counter()
                if runtime_opt_contexts is not None and local_context.psi_a_policy is not None:
                    psi_a_policy[i] = float(local_context.psi_a_policy)
                else:
                    psi_a_policy[i] = self._policy_action_supported_from_indices(
                        a_policy[i],
                        replay,
                        local_indices,
                        local_action_bins,
                    )
                candidate_construction_time += time.perf_counter() - t_support0
                if n_sup < self.cfg.M_Q_min or psi_a_policy[i] < 0.5:
                    continue
                psi_q_rank[i] = 1.0
                fallback[i] = 0.0
                if self.cfg.method in {
                    "b3_support_only",
                    "b3_support_random",
                    "b3_support_random_runtime_opt",
                    "b3_support_shuffled",
                    "b3_support_shuffled_runtime_opt",
                    "b3_support_runtime_opt",
                }:
                    w_raw[i] = 1.0 + self.cfg.lambda_support
                    continue

                t_cand0 = time.perf_counter()
                candidate_indices = self._candidate_indices(local_indices, rng)
                cand_actions = replay.action_exec[candidate_indices].copy().astype(np.float32)
                candidate_construction_time += time.perf_counter() - t_cand0

            n_cand = min(int(cand_actions.shape[0]), max_candidates)
            len_sup[i] = float(n_cand)
            if n_cand > 0:
                candidate_actions[i, :n_cand] = cand_actions[:n_cand]
                candidate_mask[i, :n_cand] = True

        if self.cfg.method not in {
            "b3_support_only",
            "b3_support_random",
            "b3_support_random_runtime_opt",
            "b3_support_shuffled",
            "b3_support_shuffled_runtime_opt",
            "b3_support_runtime_opt",
        }:
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
        assigned_w_raw = reference_w_raw.copy()
        if self.cfg.method in {"b3_support_random", "b3_support_random_runtime_opt"}:
            k_positive = int(np.sum(psi_q_rank > 0.5))
            assigned_w_raw = np.ones(b, dtype=np.float32)
            if k_positive > 0:
                random_idx = rng.choice(b, size=k_positive, replace=False)
                assigned_w_raw[random_idx] = 1.0 + self.cfg.lambda_support
        elif self.cfg.method in {"b3_support_shuffled", "b3_support_shuffled_runtime_opt"}:
            assigned_w_raw = rng.permutation(reference_w_raw).astype(np.float32)

        if self.cfg.method in {
            "b3_support_random",
            "b3_support_random_runtime_opt",
            "b3_support_shuffled",
            "b3_support_shuffled_runtime_opt",
        }:
            reference_high = reference_w_raw > (1.0 + 0.5 * self.cfg.lambda_support)
            assigned_high = assigned_w_raw > (1.0 + 0.5 * self.cfg.lambda_support)
            if np.any(assigned_high):
                alignment = float(np.mean(reference_high[assigned_high]))
            else:
                alignment = 0.0
            overlap = float(np.mean(np.logical_and(reference_high, assigned_high))) if b else 0.0
            extra_metrics.update(
                {
                    "mean_w_P_reference": float(np.mean(reference_w_raw)) if b else 1.0,
                    "reference_high_weight_rate": float(np.mean(reference_high)) if b else 0.0,
                    "assigned_high_weight_rate": float(np.mean(assigned_high)) if b else 0.0,
                    "support_weight_alignment_rate": alignment,
                    "support_weight_overlap_rate": overlap,
                    "lite/mean_w_P_reference": float(np.mean(reference_w_raw)) if b else 1.0,
                    "lite/reference_high_weight_rate": float(np.mean(reference_high)) if b else 0.0,
                    "lite/assigned_high_weight_rate": float(np.mean(assigned_high)) if b else 0.0,
                    "lite/support_weight_alignment_rate": alignment,
                    "lite/support_weight_overlap_rate": overlap,
                }
            )

        if self.cfg.method in {
            "b3_support_runtime_opt",
            "b3_support_random_runtime_opt",
            "b3_support_shuffled_runtime_opt",
        }:
            cache_hit_rate = float(extra_metrics.get("runtime/support_cache_hit_rate", 0.0))
            extra_metrics.setdefault("support_cache_hit_rate", cache_hit_rate)
            extra_metrics.setdefault("short_circuit_rate", 0.0)
            extra_metrics.setdefault("lite/support_cache_hit_rate", cache_hit_rate)
            extra_metrics.setdefault("lite/short_circuit_rate", 0.0)
            extra_metrics.setdefault("lite/runtime_opt_exact_path", 1.0)

        if self.cfg.method == "b1_uniform_scale":
            # Uniform scale control matches the final Omega scale of the selected
            # weighting mode. Under batch_norm this is 1, under raw it is mean(w_P).
            scale = 1.0 if self.cfg.weighting_mode == "batch_norm" else float(np.mean(assigned_w_raw))
            omega = np.full(b, scale, dtype=np.float32)
        elif self.cfg.weighting_mode == "raw":
            omega = assigned_w_raw.astype(np.float32)
        else:
            omega = (assigned_w_raw / max(self.cfg.eps, float(np.mean(assigned_w_raw)))).astype(np.float32)

        if self.cfg.weight_scope == "actor_critic":
            omega_critic = omega
            omega_actor = omega
        elif self.cfg.weight_scope == "critic_only":
            omega_critic = omega
            omega_actor = np.ones(b, dtype=np.float32)
        elif self.cfg.weight_scope == "actor_only":
            omega_critic = np.ones(b, dtype=np.float32)
            omega_actor = omega
        else:
            omega_critic = np.ones(b, dtype=np.float32)
            omega_actor = np.ones(b, dtype=np.float32)
        weight_computation_time += time.perf_counter() - t_weight0
        if self.cfg.method == "b2_unrestricted_gap":
            extra_metrics["b2/candidate_action_norm_mean"] = float(np.mean(candidate_action_norm)) if b else 0.0
            extra_metrics["b2/action_delta_norm_mean"] = float(np.mean(action_delta_norm)) if b else 0.0

        return WeightBatch(
            omega_critic=omega_critic.astype(np.float32),
            omega_actor=omega_actor.astype(np.float32),
            w_p_raw=assigned_w_raw.astype(np.float32),
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
                "numba_support_scan": float(extra_metrics.get("time/numba_support_scan", 0.0)),
                "time/numba_support_scan": float(extra_metrics.get("time/numba_support_scan", 0.0)),
            },
            gap_ready=ready,
            gap_tau=tau_g,
            gap_max=g_max,
            extra_metrics=extra_metrics,
        )

    def _fully_random_fixed_count_batch(
        self,
        b: int,
        t0: float,
        rng: np.random.Generator,
        phase: np.ndarray | None = None,
        extra_metrics: Dict[str, float] | None = None,
    ) -> WeightBatch:
        k_cfg = int(getattr(self.cfg, "random_fixed_count_k", 0))
        if k_cfg <= 0:
            raise ValueError("random_fixed_count_k must be positive for b3_fully_random_fixed_count")
        k = min(k_cfg, b)
        ones = np.ones(b, dtype=np.float32)
        zeros = np.zeros(b, dtype=np.float32)
        w_raw = ones.copy()
        if k > 0:
            chosen = rng.choice(b, size=k, replace=False)
            w_raw[chosen] = 1.0 + float(self.cfg.lambda_support)
        if self.cfg.weighting_mode == "raw":
            omega = w_raw.astype(np.float32)
        else:
            omega = (w_raw / max(self.cfg.eps, float(np.mean(w_raw)))).astype(np.float32)
        if self.cfg.weight_scope == "actor_critic":
            omega_critic = omega
            omega_actor = omega
        elif self.cfg.weight_scope == "critic_only":
            omega_critic = omega
            omega_actor = ones.copy()
        elif self.cfg.weight_scope == "actor_only":
            omega_critic = ones.copy()
            omega_actor = omega
        else:
            omega_critic = ones.copy()
            omega_actor = ones.copy()
        if phase is None:
            phase = zeros.astype(np.int64)
        metrics = dict(extra_metrics or {})
        high = (w_raw > (1.0 + self.cfg.eps)).astype(np.float32)
        high_rate = float(np.mean(high)) if b else 0.0
        metrics.update(
            {
                "fully_random_fixed_count_k": float(k),
                "fully_random_fixed_count_config_k": float(k_cfg),
                "fully_random_high_weight_rate": high_rate,
                "fully_random_support_lookup_used": 0.0,
                "assigned_high_weight_rate": high_rate,
                "support_weight_alignment_rate": 0.0,
                "lite/fully_random_fixed_count_k": float(k),
                "lite/fully_random_fixed_count_config_k": float(k_cfg),
                "lite/fully_random_high_weight_rate": high_rate,
                "lite/fully_random_support_lookup_used": 0.0,
                "lite/assigned_high_weight_rate": high_rate,
                "lite/support_weight_alignment_rate": 0.0,
            }
        )
        return WeightBatch(
            omega_critic=omega_critic.astype(np.float32),
            omega_actor=omega_actor.astype(np.float32),
            w_p_raw=w_raw.astype(np.float32),
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
                "actor": 0.0,
                "local_context_lookup": 0.0,
                "candidate_construction": 0.0,
                "local_context": 0.0,
                "q_policy_eval": 0.0,
                "q_candidate_eval": 0.0,
                "weight_compute": time.perf_counter() - t0,
                "weight_computation": time.perf_counter() - t0,
                "numba_support_scan": 0.0,
                "time/numba_support_scan": 0.0,
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics=metrics,
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
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics={} if extra_metrics is None else extra_metrics,
        )

    @staticmethod
    def _build_numba_scan_params(task: Any) -> Dict[str, Any] | None:
        required_attrs = (
            "full_context_hash_batch",
            "ee_slice",
            "object_slice",
            "gripper_slice",
            "hand_obj_bins",
            "obj_goal_bins",
            "object_z_bins",
            "gripper_gap_bins",
            "horizon_bins",
            "d_reach",
            "z_lift_min",
            "grip_close_thresh",
            "action_low",
            "action_high",
        )
        if any(not hasattr(task, name) for name in required_attrs):
            return None
        ee_slice = tuple(getattr(task, "ee_slice"))
        object_slice = tuple(getattr(task, "object_slice"))
        gripper_slice = tuple(getattr(task, "gripper_slice"))
        if len(ee_slice) != 2 or len(object_slice) != 2 or len(gripper_slice) != 2:
            return None
        if ee_slice[1] - ee_slice[0] < 3 or object_slice[1] - object_slice[0] < 3:
            return None
        if gripper_slice[1] - gripper_slice[0] < 1:
            return None
        return {
            "hand_obj_bins": np.asarray(getattr(task, "hand_obj_bins"), dtype=np.float32),
            "obj_goal_bins": np.asarray(getattr(task, "obj_goal_bins"), dtype=np.float32),
            "object_z_bins": np.asarray(getattr(task, "object_z_bins"), dtype=np.float32),
            "gripper_gap_bins": np.asarray(getattr(task, "gripper_gap_bins"), dtype=np.float32),
            "horizon_bins": np.asarray(getattr(task, "horizon_bins"), dtype=np.float32),
            "d_reach": float(getattr(task, "d_reach")),
            "d_goal_near": float(getattr(task, "d_goal_near", getattr(task, "z_lift_min"))),
            "d_opening_mid": float(getattr(task, "d_opening_mid", getattr(task, "grip_close_thresh"))),
            "z_lift_min": float(getattr(task, "z_lift_min")),
            "grip_close_thresh": float(getattr(task, "grip_close_thresh")),
            "ee_start": int(ee_slice[0]),
            "object_start": int(object_slice[0]),
            "gripper_start": int(gripper_slice[0]),
            "obs_dim_min": int(max(ee_slice[1], object_slice[1], gripper_slice[1])),
            "action_low": np.asarray(getattr(task, "action_low"), dtype=np.float32),
            "action_high": np.asarray(getattr(task, "action_high"), dtype=np.float32),
        }

    def _warmup_numba_scanner(self) -> bool:
        params = self._numba_scan_params
        if params is None:
            return False
        try:
            obs_dim = max(
                int(params["ee_start"]) + 3,
                int(params["object_start"]) + 3,
                int(params["gripper_start"]) + 2,
                int(params["obs_dim_min"]),
            )
            action_dim = int(params["action_low"].shape[0])
            dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
            dummy_goal = np.zeros(3, dtype=np.float32)
            dummy_h = np.asarray([1], dtype=np.int16)
            dummy_action_bin = np.asarray([0], dtype=np.int64)
            dummy_action = np.zeros((1, action_dim), dtype=np.float32)
            dummy_indices = np.asarray([0], dtype=np.int64)
            full_key = int(
                self.task.full_context_hash_batch(
                    dummy_obs,
                    dummy_goal[None, :],
                    np.asarray([1], dtype=np.int64),
                )[0]
            )
            kwargs = dict(
                replay_obs=dummy_obs,
                replay_h=dummy_h,
                replay_action_bin=dummy_action_bin,
                replay_action_exec=dummy_action,
                scanned_indices=dummy_indices,
                query_goal=dummy_goal,
                query_full_key=full_key,
                query_policy_bin=0,
                hand_handle_bins=params["hand_obj_bins"],
                handle_goal_bins=params["obj_goal_bins"],
                handle_y_bins=params["object_z_bins"],
                gripper_bins=params["gripper_gap_bins"],
                horizon_bins=params["horizon_bins"],
                d_reach=params["d_reach"],
                d_goal_near=params["d_goal_near"],
                d_opening_mid=params["d_opening_mid"],
                n_local_min=int(self.cfg.N_local_min),
                m_q_min=int(self.cfg.M_Q_min),
                m_b=int(self.cfg.M_B),
                ee_start=params["ee_start"],
                handle_start=params["object_start"],
                gripper_start=params["gripper_start"],
                action_low=params["action_low"],
                action_high=params["action_high"],
                bins_per_dim=int(self.cfg.action_bins_per_dim),
            )
            scan_drawer_open_support_exact(**kwargs)
            exact_ready = len(numba_support_scan_nopython_signatures()) > 0
            if bool(getattr(self.cfg, "support_predicate_early_exit", True)):
                scan_drawer_open_support_predicate(**kwargs)
                return exact_ready and len(numba_support_predicate_nopython_signatures()) > 0
            return exact_ready
        except Exception:
            return False

    def _prepare_local_lookup(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, Dict[int, LocalScan]]:
        query_coarse = self.task.coarse_hash_batch(batch.observation, batch.h)
        if replay.size == 0:
            return query_coarse, {}

        lookup: Dict[int, LocalScan] = {}
        for key in np.unique(query_coarse):
            scanned, n_candidates, truncated = replay.indexed_reference_indices(
                coarse_key=int(key),
                max_n=self.cfg.local_search_max,
                rng=rng,
            )
            lookup[int(key)] = LocalScan(
                indices=scanned,
                coarse_key=int(key),
                action_bins=self._coarse_action_bins(replay, int(key)),
                bucket_version=self._bucket_version(replay, int(key)),
                num_coarse_candidates=n_candidates,
                local_scan_truncated=bool(truncated),
            )
        return query_coarse, lookup

    def _prepare_runtime_opt_contexts(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        a_policy: np.ndarray,
    ) -> tuple[np.ndarray, list[LocalContextResult], Dict[str, float], float]:
        t0 = time.perf_counter()
        self._runtime_context_calls += 1
        audit_interval = int(getattr(self.cfg, "support_diagnostic_audit_interval", 100))
        audit_this = audit_interval > 0 and (self._runtime_context_calls % audit_interval == 0)
        b = int(batch.observation.shape[0])
        query_coarse = self.task.coarse_hash_batch(batch.observation, batch.h)
        full_keys = self.task.full_context_hash_batch(batch.observation, batch.desired_goal_train, batch.h)
        policy_bins = self.task.action_bin_batch(a_policy, self.cfg.action_bins_per_dim)

        support_keys: list[tuple[Any, ...]] = []
        unique_queries: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        for i in range(b):
            coarse_key = int(query_coarse[i])
            full_key = int(full_keys[i])
            bucket_version = self._bucket_version(replay, coarse_key)
            policy_bin = int(policy_bins[i])
            support_key = self._support_cache_key(
                coarse_key=coarse_key,
                full_key=full_key,
                goal=batch.desired_goal_train[i],
                h=int(batch.h[i]),
                bucket_version=bucket_version,
                policy_bin=policy_bin if self._numba_support_scan_enabled else None,
            )
            support_keys.append(support_key)
            query = unique_queries.get(support_key)
            if query is None:
                query = {
                    "coarse_key": coarse_key,
                    "full_key": full_key,
                    "goal": batch.desired_goal_train[i].astype(np.float32, copy=True),
                    "policy_bin": policy_bin,
                    "policy_bins": set(),
                }
                unique_queries[support_key] = query
            query["policy_bins"].add(policy_bin)

        scan_hits = 0
        scan_misses = 0
        support_hits = 0
        support_misses = 0
        exact_negative = 0
        threshold_short_circuit = 0
        coarse_count_short_circuit = 0
        coarse_support_size_short_circuit = 0
        scanned_count_short_circuit = 0
        coarse_action_bin_count_lookup = 0
        full_context_scans = 0
        full_context_filter_count = 0
        numba_queries = 0
        numba_fallbacks = 0
        numba_scan_time = 0.0
        predicate_early_exit = 0
        predicate_scanned_count = 0
        diagnostic_audits = 0
        audit_len_local_abs_diff = 0.0
        audit_len_sup_abs_diff = 0.0
        audit_psi_mismatch = 0
        context_by_key: Dict[tuple[Any, ...], LocalContextResult] = {}
        support_cache_enabled = bool(self.cfg.support_cache_enabled and self.cfg.support_cache_max_entries > 0)

        for support_key, query in unique_queries.items():
            cached = self._lru_get(self._support_cache, support_key) if support_cache_enabled else None
            if cached is not None:
                support_hits += 1
                predicate_early_exit += int(cached.predicate_early_exit)
                if cached.predicate_early_exit:
                    predicate_scanned_count += int(cached.num_scanned_candidates)
                context_by_key[support_key] = cached
                continue

            support_misses += 1
            coarse_key = int(query["coarse_key"])
            scan, scan_hit = self._runtime_opt_local_scan(coarse_key, replay)
            if scan_hit:
                scan_hits += 1
            else:
                scan_misses += 1

            context, did_full_scan, did_exact_negative, filter_count, numba_used, numba_fallback, scan_time = self._local_context_from_scan(
                full_key=int(query["full_key"]),
                goal=query["goal"],
                replay=replay,
                scan=scan,
                policy_bins=query["policy_bins"],
                policy_bin=int(query["policy_bin"]),
            )
            full_context_scans += int(did_full_scan)
            exact_negative += int(did_exact_negative)
            threshold_short_circuit += int(context.threshold_short_circuit)
            coarse_count_short_circuit += int(context.coarse_count_short_circuit)
            coarse_support_size_short_circuit += int(context.coarse_support_size_short_circuit)
            scanned_count_short_circuit += int(context.scanned_count_short_circuit)
            coarse_action_bin_count_lookup += int(context.coarse_action_bin_count_lookup)
            full_context_filter_count += int(filter_count)
            numba_queries += int(numba_used)
            numba_fallbacks += int(numba_fallback)
            numba_scan_time += float(scan_time)
            predicate_early_exit += int(context.predicate_early_exit)
            if context.predicate_early_exit:
                predicate_scanned_count += int(context.num_scanned_candidates)
            if audit_this and context.predicate_early_exit:
                exact_local, exact_sup, exact_psi = self._audit_local_context_exact(
                    full_key=int(query["full_key"]),
                    goal=query["goal"],
                    replay=replay,
                    scan=scan,
                    policy_bin=int(query["policy_bin"]),
                )
                diagnostic_audits += 1
                audit_len_local_abs_diff += abs(float(exact_local) - float(context.local_pool_size or 0))
                audit_len_sup_abs_diff += abs(float(exact_sup) - float(context.available_support_size or 0))
                audit_psi_mismatch += int(float(exact_psi) != float(context.psi_a_policy or 0.0))
            if support_cache_enabled and not did_exact_negative:
                self._lru_put(self._support_cache, support_key, context, int(self.cfg.support_cache_max_entries))
            context_by_key[support_key] = context

        empty_context = LocalContextResult(
            indices=np.zeros((0,), dtype=np.int64),
            action_bins=np.zeros((0,), dtype=np.int64),
            num_coarse_candidates=0,
            num_scanned_candidates=0,
            local_scan_truncated=False,
        )
        contexts = [context_by_key.get(key, empty_context) for key in support_keys]
        support_total = max(1, support_hits + support_misses)
        scan_total = max(1, scan_hits + scan_misses)
        unique_count = len(unique_queries)
        coarse_unique = len(set(int(v) for v in query_coarse.tolist()))
        numba_attempts = max(1, numba_queries + numba_fallbacks)
        audit_total = max(1, diagnostic_audits)
        metrics = {
            "runtime/support_cache_hit_rate": float(support_hits / support_total),
            "runtime/support_cache_miss_rate": float(support_misses / support_total),
            "runtime/support_cache_attempt_rate": float((support_hits + support_misses) / max(1, b)),
            "runtime/support_context_cache_size": float(len(self._support_cache)),
            "runtime/batch_query_unique_rate": float(unique_count / max(1, b)),
            "runtime/batch_query_coalescing_saving_rate": float(1.0 - unique_count / max(1, b)),
            "runtime/coarse_query_unique_rate": float(coarse_unique / max(1, b)),
            "runtime/local_scan_cache_hit_rate": float(scan_hits / scan_total),
            "runtime/local_scan_cache_miss_rate": float(scan_misses / scan_total),
            "runtime/local_scan_cache_attempt_rate": float((scan_hits + scan_misses) / max(1, b)),
            "runtime/local_scan_cache_size": float(len(self._local_scan_cache)),
            "runtime/full_context_scan_rate": float(full_context_scans / max(1, b)),
            "runtime/exact_negative_short_circuit_rate": float(exact_negative / max(1, b)),
            "runtime/count_short_circuit_rate": float(threshold_short_circuit / max(1, b)),
            "runtime/threshold_short_circuit_rate": float(threshold_short_circuit / max(1, b)),
            "runtime/coarse_count_short_circuit_rate": float(coarse_count_short_circuit / max(1, b)),
            "runtime/coarse_support_size_short_circuit_rate": float(coarse_support_size_short_circuit / max(1, b)),
            "runtime/scanned_count_short_circuit_rate": float(scanned_count_short_circuit / max(1, b)),
            "runtime/coarse_action_bin_count_lookup_rate": float(coarse_action_bin_count_lookup / max(1, b)),
            "runtime/full_context_filter_count_mean": float(full_context_filter_count / max(1, full_context_scans)),
            "runtime/numba_support_scan_enabled": float(self._numba_support_scan_enabled),
            "runtime/numba_scanner_warmed": float(self._numba_scanner_warmed),
            "runtime/numba_support_scan_fallback_rate": float(numba_fallbacks / numba_attempts),
            "runtime/numba_support_queries": float(numba_queries),
            "runtime/predicate_early_exit_rate": float(predicate_early_exit / max(1, b)),
            "runtime/predicate_scanned_count_mean": float(predicate_scanned_count / max(1, predicate_early_exit)),
            "runtime/diagnostic_audit_rate": float(diagnostic_audits / max(1, b)),
            "runtime/audit_len_A_Q_local_abs_diff_mean": float(audit_len_local_abs_diff / audit_total),
            "runtime/audit_len_A_Q_sup_abs_diff_mean": float(audit_len_sup_abs_diff / audit_total),
            "runtime/audit_psi_A_policy_mismatch_rate": float(audit_psi_mismatch / audit_total),
            "time/numba_support_scan": float(numba_scan_time),
            "support_cache_hit_rate": float(support_hits / support_total),
            "short_circuit_rate": float(exact_negative / max(1, unique_count)),
            "lite/support_cache_hit_rate": float(support_hits / support_total),
            "lite/short_circuit_rate": float(exact_negative / max(1, unique_count)),
            "lite/runtime_opt_exact_path": 1.0,
        }
        return query_coarse, contexts, metrics, time.perf_counter() - t0

    def _runtime_opt_local_scan(
        self,
        coarse_key: int,
        replay: HERReplayBuffer,
    ) -> tuple[LocalScan, bool]:
        bucket_version = self._bucket_version(replay, int(coarse_key))
        cache_key = self._scan_cache_key(int(coarse_key), bucket_version)
        cache_enabled = bool(self.cfg.local_scan_cache_enabled and self.cfg.local_scan_cache_max_entries > 0)
        if cache_enabled:
            cached = self._lru_get(self._local_scan_cache, cache_key)
            if cached is not None:
                return cached, True

        scanned, n_candidates, truncated = replay.indexed_reference_indices_deterministic(
            coarse_key=int(coarse_key),
            max_n=int(self.cfg.local_search_max),
            mode=str(self.cfg.support_scan_mode),
        )
        scan = LocalScan(
            indices=scanned.astype(np.int64, copy=True),
            coarse_key=int(coarse_key),
            action_bins=self._coarse_action_bins(replay, int(coarse_key)),
            bucket_version=bucket_version,
            num_coarse_candidates=int(n_candidates),
            local_scan_truncated=bool(truncated),
        )
        if cache_enabled:
            self._lru_put(self._local_scan_cache, cache_key, scan, int(self.cfg.local_scan_cache_max_entries))
        return scan, False

    def _audit_local_context_exact(
        self,
        full_key: int,
        goal: np.ndarray,
        replay: HERReplayBuffer,
        scan: LocalScan,
        policy_bin: int,
    ) -> tuple[int, int, float]:
        params = self._numba_scan_params
        if params is not None and NUMBA_AVAILABLE:
            try:
                local_count, available, psi_a, _support, _fallback, _num_scanned = scan_drawer_open_support_exact(
                    replay_obs=replay.observation,
                    replay_h=replay.h,
                    replay_action_bin=replay.action_bin,
                    replay_action_exec=replay.action_exec,
                    scanned_indices=scan.indices,
                    query_goal=goal,
                    query_full_key=int(full_key),
                    query_policy_bin=int(policy_bin),
                    hand_handle_bins=params["hand_obj_bins"],
                    handle_goal_bins=params["obj_goal_bins"],
                    handle_y_bins=params["object_z_bins"],
                    gripper_bins=params["gripper_gap_bins"],
                    horizon_bins=params["horizon_bins"],
                    d_reach=params["d_reach"],
                    d_goal_near=params["d_goal_near"],
                    d_opening_mid=params["d_opening_mid"],
                    n_local_min=int(self.cfg.N_local_min),
                    m_q_min=int(self.cfg.M_Q_min),
                    m_b=int(self.cfg.M_B),
                    ee_start=params["ee_start"],
                    handle_start=params["object_start"],
                    gripper_start=params["gripper_start"],
                    action_low=params["action_low"],
                    action_high=params["action_high"],
                    bins_per_dim=int(self.cfg.action_bins_per_dim),
                )
                return int(local_count), int(available), float(psi_a)
            except Exception:
                pass

        ref_obs = replay.observation[scan.indices]
        ref_goal = np.repeat(goal[None, :], scan.indices.size, axis=0)
        ref_h = replay.h[scan.indices]
        ref_hash = self.task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
        local_idx = scan.indices[ref_hash == int(full_key)]
        local_bins = replay.action_bin[local_idx].copy() if local_idx.size else np.zeros((0,), dtype=np.int64)
        if local_bins.shape[0] != local_idx.size or np.any(local_bins < 0):
            local_bins = self.task.action_bin_batch(replay.action_exec[local_idx], self.cfg.action_bins_per_dim)
        psi_a = float(np.any(local_bins.astype(np.int64, copy=False) == int(policy_bin))) if local_bins.size else 0.0
        local_count = int(local_idx.size)
        return local_count, min(local_count, int(self.cfg.M_B)), psi_a

    def _local_context_from_scan(
        self,
        full_key: int,
        goal: np.ndarray,
        replay: HERReplayBuffer,
        scan: LocalScan,
        policy_bins: set[int],
        policy_bin: int,
    ) -> tuple[LocalContextResult, bool, bool, int, int, int, float]:
        if int(self.cfg.M_B) < int(self.cfg.M_Q_min):
            return (
                LocalContextResult(
                    indices=np.zeros((0,), dtype=np.int64),
                    action_bins=np.zeros((0,), dtype=np.int64),
                    num_coarse_candidates=scan.num_coarse_candidates,
                    num_scanned_candidates=0,
                    local_scan_truncated=scan.local_scan_truncated,
                    local_pool_size=0,
                    available_support_size=0,
                    psi_a_policy=0.0 if self._numba_support_scan_enabled else None,
                    threshold_short_circuit=True,
                    coarse_support_size_short_circuit=True,
                ),
                False,
                True,
                0,
                0,
                0,
                0.0,
            )
        required_count = max(int(self.cfg.N_local_min), int(self.cfg.M_Q_min))
        n_coarse = int(scan.num_coarse_candidates)
        if self.cfg.support_short_circuit and n_coarse < int(self.cfg.N_local_min):
            return (
                LocalContextResult(
                    indices=np.zeros((0,), dtype=np.int64),
                    action_bins=np.zeros((0,), dtype=np.int64),
                    num_coarse_candidates=scan.num_coarse_candidates,
                    num_scanned_candidates=0,
                    local_scan_truncated=scan.local_scan_truncated,
                    local_pool_size=0,
                    available_support_size=0,
                    psi_a_policy=0.0 if self._numba_support_scan_enabled else None,
                    threshold_short_circuit=True,
                    coarse_count_short_circuit=True,
                ),
                False,
                True,
                0,
                0,
                0,
                0.0,
            )
        if self.cfg.support_short_circuit and min(n_coarse, int(self.cfg.M_B)) < int(self.cfg.M_Q_min):
            return (
                LocalContextResult(
                    indices=np.zeros((0,), dtype=np.int64),
                    action_bins=np.zeros((0,), dtype=np.int64),
                    num_coarse_candidates=scan.num_coarse_candidates,
                    num_scanned_candidates=0,
                    local_scan_truncated=scan.local_scan_truncated,
                    local_pool_size=0,
                    available_support_size=0,
                    psi_a_policy=0.0 if self._numba_support_scan_enabled else None,
                    threshold_short_circuit=True,
                    coarse_support_size_short_circuit=True,
                ),
                False,
                True,
                0,
                0,
                0,
                0.0,
            )
        if scan.indices.size == 0:
            return (
                LocalContextResult(
                    indices=np.zeros((0,), dtype=np.int64),
                    action_bins=np.zeros((0,), dtype=np.int64),
                    num_coarse_candidates=scan.num_coarse_candidates,
                    num_scanned_candidates=0,
                    local_scan_truncated=scan.local_scan_truncated,
                    local_pool_size=0,
                    available_support_size=0,
                    psi_a_policy=0.0 if self._numba_support_scan_enabled else None,
                    threshold_short_circuit=True,
                    scanned_count_short_circuit=True,
                ),
                False,
                True,
                0,
                0,
                0,
                0.0,
            )
        if self.cfg.support_short_circuit and scan.indices.size < required_count:
            return (
                LocalContextResult(
                    indices=np.zeros((0,), dtype=np.int64),
                    action_bins=np.zeros((0,), dtype=np.int64),
                    num_coarse_candidates=scan.num_coarse_candidates,
                    num_scanned_candidates=int(scan.indices.size),
                    local_scan_truncated=scan.local_scan_truncated,
                    local_pool_size=0,
                    available_support_size=0,
                    psi_a_policy=0.0 if self._numba_support_scan_enabled else None,
                    threshold_short_circuit=True,
                    scanned_count_short_circuit=True,
                ),
                False,
                True,
                0,
                0,
                0,
                0.0,
            )

        action_bin_count_lookup = False
        if self.cfg.support_short_circuit and self.cfg.policy_support_mode == "bin" and policy_bins:
            if hasattr(replay, "action_index_has_action_bin"):
                action_bin_count_lookup = True
                policy_bin_in_coarse = any(
                    bool(replay.action_index_has_action_bin(int(scan.coarse_key), int(pb)))
                    for pb in policy_bins
                )
            elif scan.action_bins.size > 0:
                coarse_bins = set(int(v) for v in scan.action_bins.tolist())
                policy_bin_in_coarse = any(int(pb) in coarse_bins for pb in policy_bins)
            else:
                policy_bin_in_coarse = True
            if not policy_bin_in_coarse:
                return (
                    LocalContextResult(
                        indices=np.zeros((0,), dtype=np.int64),
                        action_bins=np.zeros((0,), dtype=np.int64),
                        num_coarse_candidates=scan.num_coarse_candidates,
                        num_scanned_candidates=int(scan.indices.size),
                        local_scan_truncated=scan.local_scan_truncated,
                        local_pool_size=0,
                        available_support_size=0,
                        psi_a_policy=0.0 if self._numba_support_scan_enabled else None,
                        coarse_action_bin_count_lookup=action_bin_count_lookup,
                    ),
                    False,
                    True,
                    0,
                    0,
                    0,
                    0.0,
                )

        if self._numba_support_scan_enabled and self._numba_scan_params is not None and self.cfg.policy_support_mode == "bin":
            params = self._numba_scan_params
            use_predicate = bool(getattr(self.cfg, "support_predicate_early_exit", True))
            try:
                t_numba0 = time.perf_counter()
                common_kwargs = dict(
                    replay_obs=replay.observation,
                    replay_h=replay.h,
                    replay_action_bin=replay.action_bin,
                    replay_action_exec=replay.action_exec,
                    scanned_indices=scan.indices,
                    query_goal=goal,
                    query_full_key=int(full_key),
                    query_policy_bin=int(policy_bin),
                    hand_handle_bins=params["hand_obj_bins"],
                    handle_goal_bins=params["obj_goal_bins"],
                    handle_y_bins=params["object_z_bins"],
                    gripper_bins=params["gripper_gap_bins"],
                    horizon_bins=params["horizon_bins"],
                    d_reach=params["d_reach"],
                    d_goal_near=params["d_goal_near"],
                    d_opening_mid=params["d_opening_mid"],
                    n_local_min=int(self.cfg.N_local_min),
                    m_q_min=int(self.cfg.M_Q_min),
                    m_b=int(self.cfg.M_B),
                    ee_start=params["ee_start"],
                    handle_start=params["object_start"],
                    gripper_start=params["gripper_start"],
                    action_low=params["action_low"],
                    action_high=params["action_high"],
                    bins_per_dim=int(self.cfg.action_bins_per_dim),
                )
                if use_predicate:
                    (
                        local_pool_size,
                        available_support_size,
                        psi_a_policy_int,
                        _support_positive,
                        _fallback,
                        num_scanned_candidates,
                        early_exit_int,
                    ) = scan_drawer_open_support_predicate(**common_kwargs)
                else:
                    (
                        local_pool_size,
                        available_support_size,
                        psi_a_policy_int,
                        _support_positive,
                        _fallback,
                        num_scanned_candidates,
                    ) = scan_drawer_open_support_exact(**common_kwargs)
                    early_exit_int = 0
                numba_time = time.perf_counter() - t_numba0
                early_exit = bool(early_exit_int)
                return (
                    LocalContextResult(
                        indices=np.zeros((0,), dtype=np.int64),
                        action_bins=np.zeros((0,), dtype=np.int64),
                        num_coarse_candidates=scan.num_coarse_candidates,
                        num_scanned_candidates=int(num_scanned_candidates),
                        local_scan_truncated=scan.local_scan_truncated,
                        local_pool_size=int(local_pool_size),
                        available_support_size=int(available_support_size),
                        psi_a_policy=float(psi_a_policy_int),
                        coarse_action_bin_count_lookup=bool(action_bin_count_lookup),
                        predicate_early_exit=early_exit,
                    ),
                    not early_exit,
                    False,
                    0 if early_exit else int(scan.indices.size),
                    1,
                    0,
                    numba_time,
                )
            except Exception:
                self._numba_support_scan_enabled = False

        ref_obs = replay.observation[scan.indices]
        ref_goal = np.repeat(goal[None, :], scan.indices.size, axis=0)
        ref_h = replay.h[scan.indices]
        ref_hash = self.task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
        local_idx = scan.indices[ref_hash == int(full_key)]
        local_bins = (
            replay.action_bin[local_idx].copy()
            if local_idx.size > 0
            else np.zeros((0,), dtype=np.int64)
        )
        return (
            LocalContextResult(
                indices=local_idx.astype(np.int64, copy=True),
                action_bins=local_bins.astype(np.int64, copy=False),
                num_coarse_candidates=scan.num_coarse_candidates,
                num_scanned_candidates=int(scan.indices.size),
                local_scan_truncated=scan.local_scan_truncated,
                coarse_action_bin_count_lookup=bool(action_bin_count_lookup),
            ),
            True,
            False,
            int(scan.indices.size),
            0,
            int(self._numba_scanner_warmed),
            0.0,
        )

    def _support_cache_key(
        self,
        coarse_key: int,
        full_key: int,
        goal: np.ndarray,
        h: int,
        bucket_version: int,
        policy_bin: int | None = None,
    ) -> tuple[Any, ...]:
        key: tuple[Any, ...] = (
            int(coarse_key),
            int(full_key),
            np.ascontiguousarray(np.asarray(goal, dtype=np.float32)).tobytes(),
            int(h),
            int(bucket_version),
            int(self.cfg.local_search_max),
            int(self.cfg.action_bins_per_dim),
            str(self.cfg.policy_support_mode),
            str(self.cfg.support_scan_mode),
        )
        if policy_bin is not None:
            key = key + ("policy_bin", int(policy_bin))
        return key

    def _scan_cache_key(self, coarse_key: int, bucket_version: int) -> tuple[Any, ...]:
        return (
            int(coarse_key),
            int(bucket_version),
            int(self.cfg.local_search_max),
            int(self.cfg.action_bins_per_dim),
            str(self.cfg.support_scan_mode),
        )

    @staticmethod
    def _bucket_version(replay: HERReplayBuffer, coarse_key: int) -> int:
        if hasattr(replay, "action_index_version"):
            return int(replay.action_index_version(int(coarse_key)))
        action_index = getattr(replay, "action_index", None)
        if action_index is not None and hasattr(action_index, "version"):
            return int(action_index.version(int(coarse_key)))
        return int(getattr(replay, "version", replay.size))

    @staticmethod
    def _coarse_action_bins(replay: HERReplayBuffer, coarse_key: int) -> np.ndarray:
        if hasattr(replay, "action_index_bins"):
            return replay.action_index_bins(int(coarse_key)).astype(np.int64, copy=False)
        action_index = getattr(replay, "action_index", None)
        if action_index is not None and hasattr(action_index, "action_bins"):
            return action_index.action_bins(int(coarse_key), replay.size, replay.action_bin).astype(np.int64, copy=False)
        return np.zeros((0,), dtype=np.int64)

    @staticmethod
    def _lru_get(cache: OrderedDict, key: tuple[Any, ...]) -> Any | None:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _lru_put(cache: OrderedDict, key: tuple[Any, ...], value: Any, max_size: int) -> None:
        if max_size <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _local_indices(
        self,
        i: int,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        query_coarse: np.ndarray,
        local_lookup: Dict[int, LocalScan],
        local_context_cache: Dict[tuple[int, int], LocalContextResult],
    ) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
        coarse_key = int(query_coarse[i])
        goal = batch.desired_goal_train[i]
        full_key = self.task.full_context_hash(batch.observation[i], goal, int(batch.h[i]))
        cache_key = (coarse_key, full_key)
        cached = local_context_cache.get(cache_key)
        if cached is not None:
            return (
                cached.indices,
                cached.action_bins,
                cached.num_coarse_candidates,
                cached.num_scanned_candidates,
                cached.local_scan_truncated,
            )

        scan = local_lookup.get(coarse_key)
        if scan is None or scan.indices.size == 0:
            local_context_cache[cache_key] = LocalContextResult(
                indices=np.zeros((0,), dtype=np.int64),
                action_bins=np.zeros((0,), dtype=np.int64),
                num_coarse_candidates=0,
                num_scanned_candidates=0,
                local_scan_truncated=False,
            )
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                0,
                0,
                False,
            )
        ref_obs = replay.observation[scan.indices]
        ref_goal = np.repeat(goal[None, :], scan.indices.size, axis=0)
        ref_h = replay.h[scan.indices]
        ref_hash = self.task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
        local_idx = scan.indices[ref_hash == full_key]
        local_bins = (
            replay.action_bin[local_idx].copy()
            if local_idx.size > 0
            else np.zeros((0,), dtype=np.int64)
        )
        local_context_cache[cache_key] = LocalContextResult(
            indices=local_idx.astype(np.int64, copy=True),
            action_bins=local_bins.astype(np.int64, copy=False),
            num_coarse_candidates=scan.num_coarse_candidates,
            num_scanned_candidates=int(scan.indices.size),
            local_scan_truncated=scan.local_scan_truncated,
        )
        if local_idx.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                scan.num_coarse_candidates,
                int(scan.indices.size),
                scan.local_scan_truncated,
            )
        return (
            local_idx.astype(np.int64, copy=False),
            local_bins,
            scan.num_coarse_candidates,
            int(scan.indices.size),
            scan.local_scan_truncated,
        )

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
    ) -> float:
        if local_indices.size == 0:
            return 0.0
        if self.cfg.policy_support_mode == "radius":
            local_actions = replay.action_exec[local_indices]
            return self._policy_action_supported(action_policy, local_actions, local_action_bins)

        p_bin = int(self.task.action_bin_batch(action_policy[None, :], self.cfg.action_bins_per_dim)[0])
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
