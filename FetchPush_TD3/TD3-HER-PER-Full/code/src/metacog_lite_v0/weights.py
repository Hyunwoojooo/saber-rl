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
    num_coarse_candidates: int
    local_scan_truncated: bool


@dataclass
class LocalContextResult:
    indices: np.ndarray
    action_bins: np.ndarray
    num_coarse_candidates: int
    num_scanned_candidates: int
    local_scan_truncated: bool


def _entropy_np(w: np.ndarray, eps: float) -> float:
    if w.size == 0:
        return 0.0
    p = np.maximum(w.astype(np.float64), float(eps))
    p = p / max(float(eps), float(np.sum(p)))
    return float(-(p * np.log(p + float(eps))).sum())


def _effective_sample_size_np(w: np.ndarray, eps: float = 1e-8) -> float:
    if w.size == 0:
        return 0.0
    w64 = np.maximum(w.astype(np.float64), 0.0)
    denom = float(np.sum(w64 * w64))
    if denom <= eps:
        return 0.0
    return float((float(np.sum(w64)) ** 2) / denom)


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

        if self.cfg.method == "per_full_baseline":
            return self._per_full_batch(
                batch=batch,
                replay=replay,
                t0=t0,
                phase=phase,
                extra_metrics=extra_metrics,
            )

        if self.cfg.method in {"td_error_baseline", "per_style_baseline"}:
            if reward_train is None or nonterminal is None or state_np is None or next_state_np is None:
                raise ValueError("TD/PER baselines require reward_train, nonterminal, state_np, and next_state_np")
            return self._td_priority_batch(
                batch=batch,
                replay=replay,
                agent=agent,
                device=device,
                reward_train=reward_train,
                nonterminal=nonterminal,
                state_np=state_np,
                next_state_np=next_state_np,
                t0=t0,
                phase=phase,
                extra_metrics=extra_metrics,
            )

        state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        a_policy = agent.policy_actions(state_t).detach().cpu().numpy().astype(np.float32)
        t_actor = time.perf_counter()

        ready, tau_g, g_max = self.gap_stats.thresholds()
        t_lookup0 = time.perf_counter()
        if self.cfg.method == "b2_unrestricted_gap":
            query_coarse = np.zeros(b, dtype=np.int64)
            local_lookup: Dict[int, LocalScan] = {}
        else:
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
                local_count = int(local_indices.size)
                len_local[i] = float(local_count)
                if local_count < self.cfg.N_local_min:
                    continue
                n_sup = min(local_count, int(self.cfg.M_B))
                len_sup[i] = float(n_sup)
                t_support0 = time.perf_counter()
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
                if self.cfg.method in {"b3_support_only", "b3_support_random", "b3_support_shuffled"}:
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
            },
            gap_ready=ready,
            gap_tau=tau_g,
            gap_max=g_max,
            extra_metrics=extra_metrics,
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
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics={} if extra_metrics is None else extra_metrics,
        )

    def _per_full_batch(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        t0: float,
        phase: np.ndarray,
        extra_metrics: Dict[str, float],
    ) -> WeightBatch:
        t_weight0 = time.perf_counter()
        b = int(batch.observation.shape[0])
        is_weight = np.asarray(getattr(batch, "is_weight", np.ones(b, dtype=np.float32)), dtype=np.float32)
        if is_weight.shape[0] != b:
            raise ValueError(f"PER-Full IS weight shape mismatch: {is_weight.shape[0]} != {b}")
        w_raw = np.clip(is_weight, 0.0, float(self.cfg.U_PER_max)).astype(np.float32)
        if self.cfg.weighting_mode == "raw":
            omega = w_raw.astype(np.float32)
        else:
            omega = (w_raw / max(self.cfg.eps, float(np.mean(w_raw)))).astype(np.float32)

        priority = np.asarray(
            getattr(batch, "sampling_priority", np.ones(b, dtype=np.float32)),
            dtype=np.float32,
        )
        sample_prob = np.asarray(
            getattr(batch, "sampling_prob", np.full(b, 1.0 / max(1, replay.size), dtype=np.float32)),
            dtype=np.float32,
        )
        zeros = np.zeros(b, dtype=np.float32)
        sample_age = ((int(replay.ptr) - batch.indices.astype(np.int64)) % max(1, int(replay.capacity))).astype(np.float32)
        duplicate_rate = 1.0 - float(np.unique(batch.indices).size / max(1, b))
        priority_ess = _effective_sample_size_np(priority)
        metrics = {
            **extra_metrics,
            "td_error/abs_mean": 0.0,
            "td_error/abs_p90": 0.0,
            "td_error/weight_mean": float(np.mean(w_raw)) if w_raw.size else 0.0,
            "td_error/weight_max": float(np.max(w_raw)) if w_raw.size else 0.0,
            "td_error/weight_entropy": _entropy_np(w_raw, self.cfg.eps),
            "td_error/weight_ESS": _effective_sample_size_np(w_raw),
            "per/beta": float(getattr(batch, "per_beta", 0.0)),
            "per/sampling_prob_mean": float(np.mean(sample_prob)) if sample_prob.size else 0.0,
            "per/sampling_prob_max": float(np.max(sample_prob)) if sample_prob.size else 0.0,
            "per/priority_entropy": _entropy_np(priority, self.cfg.eps),
            "per/priority_ESS": priority_ess,
            "per/IS_weight_mean": float(np.mean(w_raw)) if w_raw.size else 0.0,
            "per/IS_weight_max": float(np.max(w_raw)) if w_raw.size else 0.0,
            "per/replay_sample_age_mean": float(np.mean(sample_age)) if sample_age.size else 0.0,
            "per/replay_sample_age_p90": float(np.percentile(sample_age, 90)) if sample_age.size else 0.0,
            "per/sample_duplicate_rate": duplicate_rate,
        }
        weight_time = time.perf_counter() - t_weight0
        return WeightBatch(
            omega_critic=omega.astype(np.float32),
            omega_actor=np.ones(b, dtype=np.float32),
            w_p_raw=w_raw,
            w_p_reference=w_raw.astype(np.float32, copy=True),
            assigned_high_weight=(w_raw > (1.0 + self.cfg.eps)).astype(np.float32),
            reference_high_weight=(w_raw > (1.0 + self.cfg.eps)).astype(np.float32),
            support_weight_overlap=zeros.copy(),
            g_sup=zeros.copy(),
            kappa=zeros.copy(),
            psi_q_rank=zeros.copy(),
            q_candidate_fallback=zeros.copy(),
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
                "weight_compute": weight_time,
                "weight_computation": weight_time,
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics=metrics,
        )

    @torch.no_grad()
    def _td_priority_batch(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        device: torch.device,
        reward_train: np.ndarray,
        nonterminal: np.ndarray,
        state_np: np.ndarray,
        next_state_np: np.ndarray,
        t0: float,
        phase: np.ndarray,
        extra_metrics: Dict[str, float],
    ) -> WeightBatch:
        b = int(batch.observation.shape[0])
        t_td0 = time.perf_counter()
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        next_state_t = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
        action_t = torch.as_tensor(batch.action_exec, dtype=torch.float32, device=device)
        reward_t = torch.as_tensor(reward_train, dtype=torch.float32, device=device)
        nonterminal_t = torch.as_tensor(nonterminal, dtype=torch.float32, device=device)

        if hasattr(agent, "target_q_values"):
            target_q = agent.target_q_values(next_state_t)
        else:
            next_action = agent.actor_target(next_state_t)
            target_q = agent.critic_target(next_state_t, next_action)
        target = reward_t + agent.gamma * nonterminal_t * target_q
        q = agent.q_values(state_t, action_t)
        td_abs = torch.abs(q - target).detach().cpu().numpy().astype(np.float32)

        if self.cfg.method == "td_error_baseline":
            td_norm = td_abs / max(self.cfg.eps, float(np.mean(td_abs)))
            u = np.clip(td_norm, 0.0, float(self.cfg.U_TD_max)).astype(np.float32)
            w_raw = (1.0 + float(self.cfg.lambda_TD) * u).astype(np.float32)
            priority = td_abs.astype(np.float32, copy=True)
        elif self.cfg.method == "per_style_baseline":
            priority = np.power(td_abs + float(self.cfg.eps_PER), float(self.cfg.per_alpha)).astype(np.float32)
            p_norm = priority / max(self.cfg.eps, float(np.mean(priority)))
            w_raw = np.clip(p_norm, 0.0, float(self.cfg.U_PER_max)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported TD priority method={self.cfg.method!r}")

        if self.cfg.weighting_mode == "raw":
            omega = w_raw.astype(np.float32)
        else:
            omega = (w_raw / max(self.cfg.eps, float(np.mean(w_raw)))).astype(np.float32)

        omega_critic = omega.astype(np.float32)
        omega_actor = np.ones(b, dtype=np.float32)
        zeros = np.zeros(b, dtype=np.float32)
        assigned_high = (w_raw > (1.0 + self.cfg.eps)).astype(np.float32)
        sample_age = ((int(replay.ptr) - batch.indices.astype(np.int64)) % max(1, int(replay.capacity))).astype(np.float32)
        duplicate_rate = 1.0 - float(np.unique(batch.indices).size / max(1, b))
        td_weight_ess = _effective_sample_size_np(w_raw)
        priority_ess = _effective_sample_size_np(priority)
        metrics = {
            **extra_metrics,
            "td_error/abs_mean": float(np.mean(td_abs)) if td_abs.size else 0.0,
            "td_error/abs_p90": float(np.percentile(td_abs, 90)) if td_abs.size else 0.0,
            "td_error/weight_mean": float(np.mean(w_raw)) if w_raw.size else 0.0,
            "td_error/weight_max": float(np.max(w_raw)) if w_raw.size else 0.0,
            "td_error/weight_entropy": _entropy_np(w_raw, self.cfg.eps),
            "td_error/weight_ESS": td_weight_ess,
            "per/priority_entropy": _entropy_np(priority, self.cfg.eps),
            "per/priority_ESS": priority_ess,
            "per/IS_weight_mean": float(np.mean(w_raw)) if w_raw.size else 0.0,
            "per/IS_weight_max": float(np.max(w_raw)) if w_raw.size else 0.0,
            "per/replay_sample_age_mean": float(np.mean(sample_age)) if sample_age.size else 0.0,
            "per/replay_sample_age_p90": float(np.percentile(sample_age, 90)) if sample_age.size else 0.0,
            "per/sample_duplicate_rate": duplicate_rate,
        }
        weight_time = time.perf_counter() - t_td0
        return WeightBatch(
            omega_critic=omega_critic,
            omega_actor=omega_actor,
            w_p_raw=w_raw,
            w_p_reference=w_raw.astype(np.float32, copy=True),
            assigned_high_weight=assigned_high,
            reference_high_weight=assigned_high.astype(np.float32, copy=True),
            support_weight_overlap=zeros.copy(),
            g_sup=zeros.copy(),
            kappa=zeros.copy(),
            psi_q_rank=zeros.copy(),
            q_candidate_fallback=zeros.copy(),
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
                "weight_compute": weight_time,
                "weight_computation": weight_time,
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics=metrics,
        )

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
                num_coarse_candidates=n_candidates,
                local_scan_truncated=bool(truncated),
            )
        return query_coarse, lookup

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
