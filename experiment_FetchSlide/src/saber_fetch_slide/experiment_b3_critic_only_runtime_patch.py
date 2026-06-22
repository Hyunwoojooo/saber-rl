#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SABER_REPO_ROOT = EXPERIMENT_ROOT.parent
WORKSPACE_ROOT = SABER_REPO_ROOT.parent
REPO_ROOT = WORKSPACE_ROOT
LITE_SRC_CANDIDATES = (
    WORKSPACE_ROOT / "version_v0_lite" / "src",
    SABER_REPO_ROOT / "version_v0_lite" / "src",
    EXPERIMENT_ROOT / "code" / "src",
)
LITE_SRC = next((path for path in LITE_SRC_CANDIDATES if path.exists()), LITE_SRC_CANDIDATES[0])
if str(LITE_SRC) not in sys.path:
    sys.path.insert(0, str(LITE_SRC))

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.io_utils import (
    MetricsCSVWriter,
    MetricsJSONLWriter,
    env_thread_hint,
    select_device,
    set_all_seeds,
    setup_logger,
    write_json,
)
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.trainer import load_checkpoint, make_env, make_task_spec, save_checkpoint
from metacog_lite_v0.weights import GapStats, LiteWeightComputer, WeightBatch


FETCH_SLIDE_DEFAULT_LOCAL_SEARCH_MAX = 1024
FETCH_SLIDE_FAST_SWEEP = (512, 2048)
DEFAULT_THRESHOLDS = (0.25, 0.50, 0.75, 0.90, 1.00)
RUNTIME_OPT_METHOD = "b3_critic_only_runtime_optimization"
PATCHED_B3_METHOD = "b3_critic_only"
DEFAULT_SUPPORT_SCAN_MODE = "deterministic_tail_v1"
RUNTIME_SUPPORT_CACHE_MAX = 200_000
RUNTIME_LOCAL_SCAN_CACHE_MAX = 50_000


@dataclass(frozen=True)
class MethodSpec:
    public_name: str
    lite_method: str
    weight_scope: str
    local_search_max: int | None
    control_mode: str = "none"
    lambda_td: float = 1.0
    u_td_max: float = 1.0
    per_alpha: float = 0.6
    per_eps: float = 1e-6
    u_per_max: float = 2.0


METHODS: Dict[str, MethodSpec] = {
    "b0": MethodSpec("b0", "b0_ddpg_her", "none", None),
    "b3_original": MethodSpec(
        "b3_original",
        "b3_support_only",
        "actor_critic",
        FETCH_SLIDE_DEFAULT_LOCAL_SEARCH_MAX,
    ),
    "b3_critic_only": MethodSpec(
        "b3_critic_only",
        "b3_support_only",
        "critic_only",
        FETCH_SLIDE_DEFAULT_LOCAL_SEARCH_MAX,
    ),
    "b3_critic_only_random": MethodSpec(
        "b3_critic_only_random",
        "b3_support_only",
        "critic_only",
        FETCH_SLIDE_DEFAULT_LOCAL_SEARCH_MAX,
        control_mode="random",
    ),
    "b3_critic_only_shuffled": MethodSpec(
        "b3_critic_only_shuffled",
        "b3_support_only",
        "critic_only",
        FETCH_SLIDE_DEFAULT_LOCAL_SEARCH_MAX,
        control_mode="shuffled",
    ),
    "b3_critic_only_runtime_optimization": MethodSpec(
        "b3_critic_only_runtime_optimization",
        "b3_support_only",
        "critic_only",
        FETCH_SLIDE_DEFAULT_LOCAL_SEARCH_MAX,
    ),
    "b3_critic_only_fast_512": MethodSpec(
        "b3_critic_only_fast_512",
        "b3_support_only",
        "critic_only",
        512,
    ),
    "b3_critic_only_fast_2048": MethodSpec(
        "b3_critic_only_fast_2048",
        "b3_support_only",
        "critic_only",
        2048,
    ),
    "td_error": MethodSpec(
        "td_error",
        "b0_ddpg_her",
        "critic_only",
        None,
        control_mode="td_error",
    ),
    "per_style": MethodSpec(
        "per_style",
        "b0_ddpg_her",
        "critic_only",
        None,
        control_mode="per_style",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SABER FetchSlide experiment runner.")
    parser.add_argument("--method", choices=[PATCHED_B3_METHOD], default=PATCHED_B3_METHOD)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(EXPERIMENT_ROOT / "experiment" / "FetchSlideRuntimePatch"),
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--reward_type", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify_runtime_optimization", action="store_true")
    parser.add_argument(
        "--support_scan_mode",
        choices=["deterministic_tail_v1"],
        default=DEFAULT_SUPPORT_SCAN_MODE,
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_capacity", type=int, default=300_000)
    parser.add_argument("--her_ratio", type=float, default=0.8)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=5_000)
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    parser.add_argument("--log_every_steps", type=int, default=1_000)
    parser.add_argument("--save_step_checkpoints", action="store_true")

    parser.add_argument("--local_search_max", type=int, default=None)
    parser.add_argument("--M_B", type=int, default=16)
    parser.add_argument("--M_Q_min", type=int, default=4)
    parser.add_argument("--N_local_min", type=int, default=4)
    parser.add_argument("--lambda_support", type=float, default=1.0)
    parser.add_argument("--action_bins_per_dim", type=int, default=5)
    parser.add_argument("--policy_support_mode", choices=["bin", "radius", "auto"], default="bin")
    parser.add_argument("--policy_action_radius", type=float, default=0.35)

    parser.add_argument("--lambda_td", type=float, default=1.0)
    parser.add_argument("--U_TD_max", type=float, default=1.0)
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--eps_PER", type=float, default=1e-6)
    parser.add_argument("--U_PER_max", type=float, default=2.0)

    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--l2_action_penalty", type=float, default=0.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace, spec: MethodSpec, run_name: str, output_root: Path) -> LiteConfig:
    local_search_max = args.local_search_max
    if local_search_max is None:
        local_search_max = spec.local_search_max if spec.local_search_max is not None else 0
    config_weight_scope = spec.weight_scope if spec.weight_scope != "none" else "critic_only"
    cfg = LiteConfig(
        task_name="fetch_slide",
        run_name=run_name,
        output_root=str(output_root),
        method=spec.lite_method,
        reward_type=args.reward_type,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        warmup_steps=args.warmup_steps,
        updates_per_step=args.updates_per_step,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        her_ratio=args.her_ratio,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        noise_std=args.noise_std,
        l2_action_penalty=args.l2_action_penalty,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_every_steps=args.checkpoint_every_steps,
        save_step_checkpoints=args.save_step_checkpoints,
        log_every_steps=args.log_every_steps,
        smoke=args.smoke,
        M_B=args.M_B,
        M_Q_min=args.M_Q_min,
        N_local_min=args.N_local_min,
        local_search_max=local_search_max,
        lambda_support=args.lambda_support,
        action_bins_per_dim=args.action_bins_per_dim,
        policy_support_mode=args.policy_support_mode,
        policy_action_radius=args.policy_action_radius,
        weighting_mode="batch_norm",
        weight_scope=config_weight_scope,
        timeout_bootstrap=True,
        success_terminates=False,
        device=args.device,
        seeds=SeedConfig(
            env_seed=args.seed,
            torch_seed=args.seed,
            numpy_seed=args.seed,
            action_noise_seed=args.seed,
            weight_seed=20_000 + args.seed,
            eval_seed=10_000 + args.seed,
        ),
    )
    cfg.normalize()
    cfg.verify_runtime_optimization = bool(args.verify_runtime_optimization)
    cfg.support_scan_mode = str(args.support_scan_mode)
    if args.smoke:
        cfg.total_steps = min(cfg.total_steps, 1_000)
        cfg.start_steps = min(cfg.start_steps, 200)
        cfg.warmup_steps = min(cfg.warmup_steps, 200)
        cfg.batch_size = min(cfg.batch_size, 64)
        cfg.buffer_capacity = min(cfg.buffer_capacity, 20_000)
        cfg.eval_freq = min(cfg.eval_freq, 500)
        cfg.eval_episodes = min(cfg.eval_episodes, 3)
        cfg.checkpoint_every_steps = min(cfg.checkpoint_every_steps, cfg.total_steps)
        cfg.local_search_max = min(cfg.local_search_max, 1_024)
        cfg.N_gap_min = min(cfg.N_gap_min, 32)
        cfg.gap_window = min(cfg.gap_window, 2_000)
    return cfg


def default_run_name(method: str, seed: int, steps: int, local_search_max: int | None) -> str:
    ls = f"ls{local_search_max}" if local_search_max is not None else "lsnone"
    return f"fetch_slide_{method}_s{seed}_{steps}_{ls}"


def make_run_dir(output_root: Path, method: str, run_name: str, resume: bool) -> Path:
    method_root = output_root / method
    run_dir = method_root / run_name
    if resume or not run_dir.exists():
        return run_dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return method_root / f"{run_name}_{stamp}"


def uses_support_lookup(spec: MethodSpec) -> bool:
    return spec.lite_method == "b3_support_only"


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pt"


def _best_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "best_checkpoint.pt"


def _step_checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"checkpoint_step_{int(step)}.pt"


@dataclass
class RuntimeScanResult:
    indices: np.ndarray
    action_bins: np.ndarray
    num_coarse_candidates: int
    local_scan_truncated: bool


@dataclass
class RuntimeSupportEntry:
    local_pool_size: int
    available_support_size: int
    action_bins: np.ndarray
    num_coarse_candidates: int
    num_scanned_candidates: int
    local_scan_truncated: bool
    full_context_scanned: bool
    exact_negative_short_circuit: bool
    full_context_filter_count: int


class PatchedB3SupportComputer:
    """Patched B3 critic-only support lookup.

    This path keeps B3/SABER support semantics fixed while reducing repeated
    lookup work with deterministic bounded scans, local scan cache, support
    context cache, batch query coalescing, and exact-negative short-circuiting.
    It does not cache Q values, targets, gaps, losses, or gradients.
    """

    def __init__(self, cfg: LiteConfig, task: Any, gap_stats: GapStats) -> None:
        self.cfg = cfg
        self.task = task
        self.gap_stats = gap_stats
        self.support_cache: OrderedDict[tuple[Any, ...], RuntimeSupportEntry] = OrderedDict()
        self.local_scan_cache: OrderedDict[tuple[Any, ...], RuntimeScanResult] = OrderedDict()

    @torch.no_grad()
    def compute(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        rng: np.random.Generator,
        device: torch.device,
    ) -> WeightBatch:
        del rng
        t0 = time.perf_counter()
        b = int(batch.observation.shape[0])
        phase, extra_metrics = self._task_batch_diagnostics(batch, b)

        state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        a_policy = agent.policy_actions(state_t).detach().cpu().numpy().astype(np.float32)
        t_actor = time.perf_counter()

        support = self._support_arrays(
            batch=batch,
            replay=replay,
            a_policy=a_policy,
            use_caches=True,
            coalesce=True,
        )
        extra_metrics.update(support["extra_metrics"])

        w_raw = support["w_raw"]
        omega_critic, omega_actor = self._omegas(w_raw)
        if getattr(self.cfg, "verify_runtime_optimization", False):
            reference = self._support_arrays(
                batch=batch,
                replay=replay,
                a_policy=a_policy,
                use_caches=False,
                coalesce=False,
            )
            ref_critic, ref_actor = self._omegas(reference["w_raw"])
            self._assert_equivalent(support, reference, omega_critic, omega_actor, ref_critic, ref_actor)

        t_weight0 = time.perf_counter()
        self.gap_stats.update(support["g_sup"], support["psi_q_rank"])
        weight_computation_time = time.perf_counter() - t_weight0
        weight_computation_time += support["weight_compute_time"]

        local_context_time = support["local_context_lookup_time"] + support["candidate_construction_time"]
        return WeightBatch(
            omega_critic=omega_critic.astype(np.float32),
            omega_actor=omega_actor.astype(np.float32),
            w_p_raw=w_raw.astype(np.float32),
            g_sup=support["g_sup"],
            kappa=support["kappa"],
            psi_q_rank=support["psi_q_rank"],
            q_candidate_fallback=support["q_candidate_fallback"],
            psi_a_policy=support["psi_a_policy"],
            len_a_q_sup=support["len_a_q_sup"],
            len_a_q_local=support["len_a_q_local"],
            num_coarse_candidates=support["num_coarse_candidates"],
            num_scanned_candidates=support["num_scanned_candidates"],
            local_scan_truncated=support["local_scan_truncated"],
            q_imp_minus_q_policy=support["q_imp_minus_q_policy"],
            unrestricted_gap=support["unrestricted_gap"],
            phase=phase,
            timing={
                "actor": t_actor - t0,
                "local_context_lookup": support["local_context_lookup_time"],
                "candidate_construction": support["candidate_construction_time"],
                "local_context": local_context_time,
                "q_policy_eval": 0.0,
                "q_candidate_eval": 0.0,
                "weight_compute": weight_computation_time,
                "weight_computation": weight_computation_time,
            },
            gap_ready=False,
            gap_tau=0.0,
            gap_max=0.0,
            extra_metrics=extra_metrics,
        )

    def _support_arrays(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        a_policy: np.ndarray,
        use_caches: bool,
        coalesce: bool,
    ) -> Dict[str, Any]:
        t_lookup0 = time.perf_counter()
        b = int(batch.observation.shape[0])
        w_raw = np.ones(b, dtype=np.float32)
        g_sup = np.zeros(b, dtype=np.float32)
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
        unrestricted_gap = np.zeros(b, dtype=np.float32)

        if self.cfg.policy_support_mode != "bin":
            raise ValueError(
                "patched B3 runtime path preserves the action-bin support rule only; "
                f"got policy_support_mode={self.cfg.policy_support_mode!r}"
            )
        query_coarse = self.task.coarse_hash_batch(batch.observation, batch.h)
        full_keys = self.task.full_context_hash_batch(
            batch.observation,
            batch.desired_goal_train,
            batch.h,
        )
        policy_bins = self.task.action_bin_batch(a_policy, self.cfg.action_bins_per_dim)
        support_keys = [
            self._support_cache_key(
                coarse_key=int(query_coarse[i]),
                full_key=int(full_keys[i]),
                goal=batch.desired_goal_train[i],
                h=int(batch.h[i]),
                replay=replay,
            )
            for i in range(b)
        ]
        lookup_keys = [(support_keys[i], int(policy_bins[i])) for i in range(b)]

        support_hits = 0
        support_misses = 0
        support_attempts = 0
        scan_hits = 0
        scan_misses = 0
        full_context_scans = 0
        full_context_filter_count = 0
        exact_negative_short_circuits = 0
        if coalesce:
            first_index: Dict[tuple[Any, ...], int] = {}
            for i, key in enumerate(lookup_keys):
                if key not in first_index:
                    first_index[key] = i
            unique_items = list(first_index.items())
        else:
            unique_items = [((i, lookup_keys[i]), i) for i in range(b)]

        lookup_results: Dict[Any, RuntimeSupportEntry] = {}
        for key, i in unique_items:
            cache_key = key[0] if coalesce else key[1][0]
            (
                entry,
                hit,
                support_attempted,
                scan_hit,
                scan_miss,
                did_full_context_scan,
                did_exact_negative,
                filter_count,
            ) = (
                self._lookup_support_entry(
                    cache_key=cache_key,
                    coarse_key=int(query_coarse[i]),
                    full_key=int(full_keys[i]),
                    goal=batch.desired_goal_train[i],
                    policy_bin=int(policy_bins[i]),
                    replay=replay,
                    use_caches=use_caches,
                )
            )
            if use_caches and support_attempted:
                support_attempts += 1
                support_hits += int(hit)
                support_misses += int(not hit)
            if use_caches:
                scan_hits += scan_hit
                scan_misses += scan_miss
            full_context_scans += int(did_full_context_scan)
            full_context_filter_count += int(filter_count)
            exact_negative_short_circuits += int(did_exact_negative)
            lookup_results[key] = entry

        local_context_lookup_time = time.perf_counter() - t_lookup0
        t_support0 = time.perf_counter()
        for i, key in enumerate(lookup_keys):
            entry_key = key if coalesce else (i, key)
            entry = lookup_results[entry_key]
            coarse_count[i] = float(entry.num_coarse_candidates)
            scanned_count[i] = float(entry.num_scanned_candidates)
            scan_truncated[i] = float(entry.local_scan_truncated)
            local_count = int(entry.local_pool_size)
            len_local[i] = float(local_count)
            if local_count < self.cfg.N_local_min:
                continue
            n_sup = int(entry.available_support_size)
            len_sup[i] = float(n_sup)
            psi_a_policy[i] = self._bin_supported(entry.action_bins, int(policy_bins[i]))
            if n_sup < self.cfg.M_Q_min or psi_a_policy[i] < 0.5:
                continue
            psi_q_rank[i] = 1.0
            fallback[i] = 0.0
            w_raw[i] = 1.0 + self.cfg.lambda_support
        candidate_construction_time = time.perf_counter() - t_support0

        unique_count = len(unique_items)
        support_queries = max(1, support_attempts)
        scan_queries = max(1, scan_hits + scan_misses)
        extra_metrics = {
            "runtime/full_context_scan_rate": float(full_context_scans / max(1, unique_count)),
            "runtime/exact_negative_short_circuit_rate": float(
                exact_negative_short_circuits / max(1, unique_count)
            ),
            "runtime/support_cache_hit_rate": float(support_hits / support_queries),
            "runtime/support_cache_miss_rate": float(support_misses / support_queries),
            "runtime/support_cache_attempt_rate": float(support_queries / max(1, unique_count)),
            "runtime/support_context_cache_size": float(len(self.support_cache)),
            "runtime/batch_query_unique_rate": float(unique_count / max(1, b)),
            "runtime/batch_query_coalescing_saving_rate": float(1.0 - unique_count / max(1, b)),
            "runtime/local_scan_cache_hit_rate": float(scan_hits / scan_queries),
            "runtime/local_scan_cache_miss_rate": float(scan_misses / scan_queries),
            "runtime/local_scan_cache_attempt_rate": float(scan_queries / max(1, unique_count)),
            "runtime/local_scan_cache_size": float(len(self.local_scan_cache)),
            "runtime/full_context_filter_count_mean": float(full_context_filter_count / max(1, unique_count)),
        }
        return {
            "w_raw": w_raw,
            "g_sup": g_sup,
            "kappa": kappa,
            "psi_q_rank": psi_q_rank,
            "q_candidate_fallback": fallback,
            "psi_a_policy": psi_a_policy,
            "len_a_q_sup": len_sup,
            "len_a_q_local": len_local,
            "num_coarse_candidates": coarse_count,
            "num_scanned_candidates": scanned_count,
            "local_scan_truncated": scan_truncated,
            "q_imp_minus_q_policy": q_delta,
            "unrestricted_gap": unrestricted_gap,
            "local_context_lookup_time": local_context_lookup_time,
            "candidate_construction_time": candidate_construction_time,
            "weight_compute_time": 0.0,
            "extra_metrics": extra_metrics,
        }

    def _lookup_support_entry(
        self,
        cache_key: tuple[Any, ...],
        coarse_key: int,
        full_key: int,
        goal: np.ndarray,
        policy_bin: int,
        replay: HERReplayBuffer,
        use_caches: bool,
    ) -> tuple[RuntimeSupportEntry, bool, bool, int, int, bool, bool, int]:
        scan, scan_hit, scan_miss = self._scan_for_key(coarse_key, replay, use_caches)
        if scan.indices.size == 0:
            entry = RuntimeSupportEntry(
                local_pool_size=0,
                available_support_size=0,
                action_bins=np.zeros((0,), dtype=np.int64),
                num_coarse_candidates=scan.num_coarse_candidates,
                num_scanned_candidates=0,
                local_scan_truncated=scan.local_scan_truncated,
                full_context_scanned=False,
                exact_negative_short_circuit=False,
                full_context_filter_count=0,
            )
            if use_caches:
                self._lru_put(self.support_cache, cache_key, entry, RUNTIME_SUPPORT_CACHE_MAX)
            return entry, False, False, scan_hit, scan_miss, False, False, 0

        if not self._bin_supported(scan.action_bins, int(policy_bin)):
            entry = RuntimeSupportEntry(
                local_pool_size=0,
                available_support_size=0,
                action_bins=np.zeros((0,), dtype=np.int64),
                num_coarse_candidates=scan.num_coarse_candidates,
                num_scanned_candidates=int(scan.indices.size),
                local_scan_truncated=scan.local_scan_truncated,
                full_context_scanned=False,
                exact_negative_short_circuit=True,
                full_context_filter_count=0,
            )
            return entry, False, False, scan_hit, scan_miss, False, True, 0

        if use_caches:
            cached = self._lru_get(self.support_cache, cache_key)
            if cached is not None:
                return cached, True, True, scan_hit, scan_miss, False, False, 0

        ref_obs = replay.observation[scan.indices]
        ref_goal = np.repeat(np.asarray(goal, dtype=np.float32)[None, :], scan.indices.size, axis=0)
        ref_h = replay.h[scan.indices]
        ref_hash = self.task.full_context_hash_batch(ref_obs, ref_goal, ref_h)
        local_idx = scan.indices[ref_hash == int(full_key)]
        if local_idx.size > 0:
            local_bins = replay.action_bin[local_idx].copy()
            if local_bins.shape[0] != local_idx.size or np.any(local_bins < 0):
                local_bins = self.task.action_bin_batch(
                    replay.action_exec[local_idx],
                    self.cfg.action_bins_per_dim,
                )
            action_bins = np.unique(local_bins.astype(np.int64, copy=False))
        else:
            action_bins = np.zeros((0,), dtype=np.int64)
        local_pool_size = int(local_idx.size)
        entry = RuntimeSupportEntry(
            local_pool_size=local_pool_size,
            available_support_size=min(local_pool_size, int(self.cfg.M_B)),
            action_bins=action_bins,
            num_coarse_candidates=scan.num_coarse_candidates,
            num_scanned_candidates=int(scan.indices.size),
            local_scan_truncated=scan.local_scan_truncated,
            full_context_scanned=True,
            exact_negative_short_circuit=False,
            full_context_filter_count=int(scan.indices.size),
        )
        if use_caches:
            self._lru_put(self.support_cache, cache_key, entry, RUNTIME_SUPPORT_CACHE_MAX)
        return entry, False, bool(use_caches), scan_hit, scan_miss, True, False, int(scan.indices.size)

    def _scan_for_key(
        self,
        coarse_key: int,
        replay: HERReplayBuffer,
        use_caches: bool,
    ) -> tuple[RuntimeScanResult, int, int]:
        cache_key = self._scan_cache_key(coarse_key, replay)
        if use_caches:
            cached = self._lru_get(self.local_scan_cache, cache_key)
            if cached is not None:
                return cached, 1, 0

        candidates = replay.action_index.candidates(int(coarse_key), replay.size)
        n_candidates = int(candidates.size)
        max_scan = int(self.cfg.local_search_max)
        truncated = max_scan > 0 and n_candidates > max_scan
        if truncated:
            scanned = candidates[-max_scan:].astype(np.int64, copy=True)
        else:
            scanned = candidates.astype(np.int64, copy=True)
        action_bins = self._action_bins_for_indices(replay, scanned)
        scan = RuntimeScanResult(
            indices=scanned,
            action_bins=action_bins,
            num_coarse_candidates=n_candidates,
            local_scan_truncated=bool(truncated),
        )
        if use_caches:
            self._lru_put(self.local_scan_cache, cache_key, scan, RUNTIME_LOCAL_SCAN_CACHE_MAX)
        return scan, 0, int(use_caches)

    def _support_cache_key(
        self,
        coarse_key: int,
        full_key: int,
        goal: np.ndarray,
        h: int,
        replay: HERReplayBuffer,
    ) -> tuple[Any, ...]:
        return (
            int(coarse_key),
            int(full_key),
            np.asarray(goal, dtype=np.float32).tobytes(),
            int(h),
            int(self.cfg.local_search_max),
            int(self.cfg.action_bins_per_dim),
            str(self.cfg.policy_support_mode),
            self._scan_mode(),
            self._replay_version_for_key(replay, int(coarse_key)),
        )

    def _scan_cache_key(self, coarse_key: int, replay: HERReplayBuffer) -> tuple[Any, ...]:
        return (
            int(coarse_key),
            int(self.cfg.local_search_max),
            int(self.cfg.action_bins_per_dim),
            self._scan_mode(),
            self._replay_version_for_key(replay, int(coarse_key)),
        )

    def _scan_mode(self) -> str:
        return str(getattr(self.cfg, "support_scan_mode", DEFAULT_SUPPORT_SCAN_MODE))

    def _action_bins_for_indices(self, replay: HERReplayBuffer, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.zeros((0,), dtype=np.int64)
        bins = replay.action_bin[indices].copy()
        if bins.shape[0] != indices.size or np.any(bins < 0):
            bins = self.task.action_bin_batch(
                replay.action_exec[indices],
                self.cfg.action_bins_per_dim,
            )
        return np.unique(bins.astype(np.int64, copy=False))

    @staticmethod
    def _replay_version_for_key(replay: HERReplayBuffer, coarse_key: int) -> int:
        if hasattr(replay, "action_index_version"):
            return int(replay.action_index_version(int(coarse_key)))
        return int(getattr(replay, "version", replay.size))

    @staticmethod
    def _lru_get(cache: OrderedDict, key: tuple[Any, ...]) -> Any | None:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _lru_put(cache: OrderedDict, key: tuple[Any, ...], value: Any, max_size: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    @staticmethod
    def _bin_supported(action_bins: np.ndarray, policy_bin: int) -> float:
        if action_bins.size == 0:
            return 0.0
        pos = int(np.searchsorted(action_bins, int(policy_bin)))
        return float(pos < action_bins.size and int(action_bins[pos]) == int(policy_bin))

    def _omegas(self, w_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.weighting_mode == "raw":
            omega = w_raw.astype(np.float32)
        else:
            omega = (w_raw / max(self.cfg.eps, float(np.mean(w_raw)))).astype(np.float32)
        if self.cfg.weight_scope == "actor_critic":
            return omega, omega
        if self.cfg.weight_scope == "critic_only":
            return omega, np.ones_like(omega, dtype=np.float32)
        return np.ones_like(omega, dtype=np.float32), omega

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

    @staticmethod
    def _assert_equivalent(
        optimized: Dict[str, Any],
        reference: Dict[str, Any],
        opt_critic: np.ndarray,
        opt_actor: np.ndarray,
        ref_critic: np.ndarray,
        ref_actor: np.ndarray,
    ) -> None:
        exact_fields = [
            "psi_q_rank",
            "psi_a_policy",
            "q_candidate_fallback",
            "w_raw",
            "len_a_q_local",
            "len_a_q_sup",
            "num_coarse_candidates",
            "num_scanned_candidates",
            "local_scan_truncated",
        ]
        for field in exact_fields:
            if not np.array_equal(optimized[field], reference[field]):
                raise AssertionError(f"runtime optimization mismatch in {field}")
        if not np.allclose(opt_critic, ref_critic, rtol=0.0, atol=0.0):
            raise AssertionError("runtime optimization mismatch in Omega_critic")
        if not np.allclose(opt_actor, ref_actor, rtol=0.0, atol=0.0):
            raise AssertionError("runtime optimization mismatch in Omega_actor")


class SaberWeightComputer:
    def __init__(self, cfg: LiteConfig, spec: MethodSpec, task: Any, gap_stats: GapStats) -> None:
        self.cfg = cfg
        self.spec = spec
        self.task = task
        self.base = LiteWeightComputer(cfg, task, gap_stats)
        self.runtime_optimized = (
            PatchedB3SupportComputer(cfg, task, gap_stats)
            if spec.public_name == PATCHED_B3_METHOD
            else None
        )

    def compute(
        self,
        batch: SampleBatch,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        rng: np.random.Generator,
        device: torch.device,
        reward_train: np.ndarray,
        nonterminal: np.ndarray,
    ) -> WeightBatch:
        if self.runtime_optimized is not None:
            return self.runtime_optimized.compute(batch, replay, agent, rng, device)

        if self.spec.control_mode in {"td_error", "per_style"}:
            wb = self.base.compute(batch, replay, agent, rng, device)
            weights, extras = self._td_or_per_weights(
                batch=batch,
                agent=agent,
                device=device,
                reward_train=reward_train,
                nonterminal=nonterminal,
            )
            wb.w_p_raw[:] = weights
            wb.omega_critic[:] = _batch_norm(weights, self.cfg.eps)
            wb.omega_actor[:] = 1.0
            wb.extra_metrics.update(extras)
            return wb

        wb = self.base.compute(batch, replay, agent, rng, device)
        if self.spec.control_mode == "random":
            assigned = np.ones_like(wb.w_p_raw, dtype=np.float32)
            high = wb.w_p_raw > 1.0 + 1e-8
            k = int(np.count_nonzero(high))
            if k > 0:
                chosen = rng.choice(wb.w_p_raw.size, size=k, replace=False)
                assigned[chosen] = 1.0 + self.cfg.lambda_support
            self._assign_control_weights(wb, assigned, "random")
        elif self.spec.control_mode == "shuffled":
            assigned = wb.w_p_raw.copy()
            rng.shuffle(assigned)
            self._assign_control_weights(wb, assigned, "shuffled")
        return wb

    def _assign_control_weights(self, wb: WeightBatch, assigned: np.ndarray, control_name: str) -> None:
        reference_support = wb.w_p_raw > 1.0 + 1e-8
        assigned_high = assigned > 1.0 + 1e-8
        if np.any(assigned_high):
            alignment = float(np.mean(reference_support[assigned_high]))
        else:
            alignment = 0.0
        wb.extra_metrics.update(
            {
                f"{control_name}/mean_w_P_reference": float(np.mean(wb.w_p_raw)) if wb.w_p_raw.size else 1.0,
                f"{control_name}/reference_high_weight_rate": float(np.mean(reference_support))
                if reference_support.size
                else 0.0,
                f"{control_name}/assigned_high_weight_rate": float(np.mean(assigned_high))
                if assigned_high.size
                else 0.0,
                f"{control_name}/support_weight_alignment_rate": alignment,
            }
        )
        wb.w_p_raw[:] = assigned.astype(np.float32)
        wb.omega_critic[:] = _batch_norm(assigned, self.cfg.eps)
        wb.omega_actor[:] = 1.0

    def _td_or_per_weights(
        self,
        batch: SampleBatch,
        agent: DDPGAgent,
        device: torch.device,
        reward_train: np.ndarray,
        nonterminal: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, float]]:
        state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        next_state_np = np.concatenate([batch.next_observation, batch.desired_goal_train], axis=1).astype(np.float32)
        state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
        action = torch.as_tensor(batch.action_exec, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_train, dtype=torch.float32, device=device)
        nt = torch.as_tensor(nonterminal, dtype=torch.float32, device=device)
        with torch.no_grad():
            target_action = agent.actor_target(next_state)
            target = reward + self.cfg.gamma * nt * agent.critic_target(next_state, target_action)
            q = agent.critic(state, action)
            td = torch.abs(q - target).detach().cpu().numpy().astype(np.float32)

        if self.spec.control_mode == "td_error":
            td_norm = td / max(float(np.mean(td)), self.cfg.eps)
            u = np.clip(td_norm, 0.0, self.spec.u_td_max).astype(np.float32)
            weights = (1.0 + self.spec.lambda_td * u).astype(np.float32)
            prefix = "td_error"
        else:
            priority = np.power(td + self.spec.per_eps, self.spec.per_alpha).astype(np.float32)
            p_norm = priority / max(float(np.mean(priority)), self.cfg.eps)
            weights = np.clip(p_norm, 0.0, self.spec.u_per_max).astype(np.float32)
            prefix = "per"

        extras = {
            f"{prefix}/abs_mean": float(np.mean(td)) if td.size else 0.0,
            f"{prefix}/abs_p90": _percentile(td, 90.0),
            f"{prefix}/weight_mean": float(np.mean(weights)) if weights.size else 1.0,
            f"{prefix}/weight_max": float(np.max(weights)) if weights.size else 1.0,
            f"{prefix}/weight_entropy": _entropy(weights, self.cfg.eps),
            f"{prefix}/weight_ESS": _ess(weights, self.cfg.eps),
        }
        if prefix == "per":
            extras["per/priority_entropy"] = _entropy(priority, self.cfg.eps)
            extras["per/priority_ESS"] = _ess(priority, self.cfg.eps)
        return weights, extras


def run(args: argparse.Namespace) -> Path:
    spec = METHODS[args.method]
    effective_ls = args.local_search_max if args.local_search_max is not None else spec.local_search_max
    run_name = args.run_name or default_run_name(args.method, args.seed, args.total_steps, effective_ls)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    cfg = build_config(args, spec, run_name, output_root / args.method)
    run_dir = make_run_dir(output_root, args.method, run_name, args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_root = str(run_dir.parent)
    cfg.run_name = run_dir.name

    logger = setup_logger(run_dir / "run.log")
    set_all_seeds(cfg.seeds.torch_seed, cfg.seeds.numpy_seed)
    device = select_device(cfg.device)
    logger.info(
        "method=%s lite_method=%s effective_weight_scope=%s config_weight_scope=%s support_lookup=%s device=%s",
        args.method,
        cfg.method,
        spec.weight_scope,
        cfg.weight_scope,
        uses_support_lookup(spec),
        device,
    )

    train_env = make_env(cfg, cfg.seeds.env_seed, False)
    eval_env = make_env(cfg, cfg.seeds.eval_seed, False)
    task = make_task_spec(cfg, train_env)
    phase_names = tuple(getattr(task, "phase_names", ("approach", "strike", "slide", "near_goal")))

    agent = DDPGAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        hidden_dim=cfg.hidden_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        action_low=train_env.action_low,
        action_high=train_env.action_high,
        device=device,
        l2_action_penalty=cfg.l2_action_penalty,
    )
    replay = HERReplayBuffer(
        capacity=cfg.buffer_capacity,
        obs_dim=train_env.obs_dim,
        goal_dim=train_env.goal_dim,
        action_dim=train_env.action_dim,
    )
    gap_stats = GapStats(
        maxlen=cfg.gap_window,
        n_min=cfg.N_gap_min,
        q_low=cfg.gap_q_low,
        q_high=cfg.gap_q_high,
        eps_g=cfg.eps_G,
    )
    weight_computer = SaberWeightComputer(cfg, spec, task, gap_stats)

    write_json(
        run_dir / "config.json",
        {
            **cfg.as_dict(),
            "saber_method": args.method,
            "method_spec": spec.__dict__,
            "effective_weight_scope": spec.weight_scope,
            "support_lookup_enabled": uses_support_lookup(spec),
            "runtime_optimization_enabled": args.method == PATCHED_B3_METHOD,
            "runtime_optimization_patch": "b3_critic_only_deterministic_tail_v1_local_scan_cache",
            "runtime_optimization_verify": bool(args.verify_runtime_optimization),
            "support_scan_mode": str(args.support_scan_mode),
            "thread_hint": env_thread_hint(),
            "result_layout": "experiment_FetchSlide/experiment/FetchSlideRuntimePatch/{method}/{run_name}",
        },
    )
    write_json(
        run_dir / "env_snapshot.json",
        {
            "train_env": train_env.snapshot(),
            "eval_env": eval_env.snapshot(),
            "task_spec": task.metadata(),
        },
    )

    fields = metric_fields(phase_names, spec.control_mode)
    csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", fields)
    jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")

    noise_rng = np.random.default_rng(cfg.seeds.action_noise_seed)
    sample_rng = np.random.default_rng(cfg.seeds.numpy_seed + 17)
    weight_rng = np.random.default_rng(cfg.seeds.weight_seed)

    start_step = 0
    episode_idx = 0
    total_updates = 0
    best_eval_success = -1.0
    best_eval_step = -1
    best_eval_return = float("-inf")
    eval_history: list[tuple[int, float]] = []
    eval_rows: list[Dict[str, Any]] = []
    last_eval_metrics: Dict[str, float] = {}
    last_loss = {"critic": 0.0, "actor": 0.0, "q_mean": 0.0, "target_mean": 0.0}
    last_lite = empty_metrics(phase_names, spec.control_mode)
    last_her_ratio = 0.0

    ckpt_path = _checkpoint_path(run_dir)
    if args.resume and ckpt_path.exists():
        payload = load_checkpoint(ckpt_path)
        agent.load_state_dict(payload["agent"])
        replay.load_state_dict(payload["replay"])
        if uses_support_lookup(spec) and replay.indexed_slot_count() < replay.size:
            replay.rebuild_action_index(task.coarse_hash_batch(replay.observation[: replay.size], replay.h[: replay.size]))
        if uses_support_lookup(spec):
            replay.rebuild_action_bins(task.action_bin_batch(replay.action_exec[: replay.size], cfg.action_bins_per_dim))
        gap_stats.load_state_dict(payload.get("gap_stats", {}))
        start_step = int(payload["global_step"]) + 1
        episode_idx = int(payload["episode_idx"])
        total_updates = int(payload.get("total_updates", 0))
        best_eval_success = float(payload.get("best_eval_success", -1.0))
        best_eval_return = float(payload.get("best_eval_return", float("-inf")))
        best_eval_step = int(payload.get("best_eval_step", -1))
        noise_rng.bit_generator.state = payload["noise_rng_state"]
        sample_rng.bit_generator.state = payload["sample_rng_state"]
        weight_rng.bit_generator.state = payload["weight_rng_state"]
        logger.info("resumed from %s at step=%d", ckpt_path, start_step)

    obs = train_env.reset(seed=cfg.seeds.env_seed + episode_idx)
    ep_return = 0.0
    ep_success = 0.0
    ep_step = 0
    wall_start = time.perf_counter()
    global_step = start_step

    def checkpoint_payload(step_value: int, kind: str) -> Dict[str, Any]:
        return {
            "checkpoint_kind": kind,
            "config": cfg.as_dict(),
            "saber_method": args.method,
            "global_step": int(step_value),
            "episode_idx": episode_idx,
            "total_updates": total_updates,
            "best_eval_success": best_eval_success,
            "best_eval_return": best_eval_return,
            "best_eval_step": best_eval_step,
            "agent": agent.state_dict(),
            "replay": replay.state_dict(),
            "gap_stats": gap_stats.state_dict(),
            "noise_rng_state": noise_rng.bit_generator.state,
            "sample_rng_state": sample_rng.bit_generator.state,
            "weight_rng_state": weight_rng.bit_generator.state,
        }

    while global_step < cfg.total_steps:
        h_t = train_env.max_episode_steps - ep_step
        if global_step < cfg.start_steps:
            action = noise_rng.uniform(train_env.action_low, train_env.action_high).astype(np.float32)
        else:
            action = agent.act(obs.flat_state(), noise_std=cfg.noise_std, noise_rng=noise_rng)

        obs_next, reward_ext, terminated, truncated, info = train_env.step(action)
        done = bool(terminated or truncated)
        timeout = bool(truncated)
        ep_return += float(reward_ext)
        ep_success = max(ep_success, float(info.get("goal_success", info.get("is_success", 0.0))))

        if uses_support_lookup(spec):
            coarse_key = task.coarse_hash(obs.observation, h_t)
            action_bin = int(task.action_bin_batch(action[None, :], cfg.action_bins_per_dim)[0])
        else:
            coarse_key = None
            action_bin = None

        replay.add_transition(
            observation=obs.observation,
            achieved_goal=obs.achieved_goal,
            desired_goal=obs.desired_goal,
            action_exec=action,
            reward_ext=float(reward_ext),
            next_observation=obs_next.observation,
            next_achieved_goal=obs_next.achieved_goal,
            done=done,
            timeout=timeout,
            h=h_t,
            episode_id=episode_idx,
            episode_step=ep_step,
            coarse_key=coarse_key,
            action_bin=action_bin,
        )

        obs = obs_next
        ep_step += 1
        global_step += 1

        if replay.size >= cfg.batch_size and global_step >= cfg.warmup_steps:
            for _ in range(cfg.updates_per_step):
                batch = replay.sample(cfg.batch_size, cfg.her_ratio, sample_rng)
                last_her_ratio = batch.her_relabel_ratio
                reward_train = train_env.compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {})
                success_train = train_env.success_from_goal(batch.next_achieved_goal, batch.desired_goal_train)
                env_terminal = batch.done.astype(np.float32) * (1.0 - batch.timeout.astype(np.float32))
                nonterminal = (1.0 - np.maximum(env_terminal, success_train.astype(np.float32) * 0.0)).astype(np.float32)

                wb = weight_computer.compute(batch, replay, agent, weight_rng, device, reward_train, nonterminal)
                state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
                next_state_np = np.concatenate([batch.next_observation, batch.desired_goal_train], axis=1).astype(np.float32)

                t_update = time.perf_counter()
                update = agent.update(
                    state=torch.as_tensor(state_np, dtype=torch.float32, device=device),
                    action_exec=torch.as_tensor(batch.action_exec, dtype=torch.float32, device=device),
                    reward=torch.as_tensor(reward_train, dtype=torch.float32, device=device),
                    next_state=torch.as_tensor(next_state_np, dtype=torch.float32, device=device),
                    nonterminal=torch.as_tensor(nonterminal, dtype=torch.float32, device=device),
                    omega_critic=torch.as_tensor(wb.omega_critic, dtype=torch.float32, device=device),
                    omega_actor=torch.as_tensor(wb.omega_actor, dtype=torch.float32, device=device),
                )
                actor_critic_time = time.perf_counter() - t_update
                last_loss = {
                    "critic": update.critic_loss,
                    "actor": update.actor_loss,
                    "q_mean": update.q_mean,
                    "target_mean": update.target_mean,
                }
                last_lite = metrics_from_weight_batch(wb, phase_names, spec.control_mode)
                last_lite["time/actor_critic_update"] = actor_critic_time
                total_updates += 1

        if done:
            episode_idx += 1
            row = metric_row(
                global_step,
                episode_idx,
                ep_return,
                ep_success,
                ep_step,
                replay.size,
                last_her_ratio,
                last_loss,
                last_lite,
                wall_start,
            )
            csv_writer.write(row)
            jsonl_writer.write({"type": "episode", **row})
            obs = train_env.reset(seed=cfg.seeds.env_seed + episode_idx)
            ep_return = 0.0
            ep_success = 0.0
            ep_step = 0

        do_eval = (global_step % cfg.eval_freq == 0) or (global_step == cfg.total_steps)
        if do_eval and global_step > 0:
            eval_metrics = evaluate_fetch_slide_agent(
                agent,
                eval_env,
                eval_seed=cfg.seeds.eval_seed + global_step * 10,
                eval_episodes=cfg.eval_episodes,
                task=task,
            )
            eval_success = float(eval_metrics["eval_env_success_mean"])
            eval_history.append((global_step, eval_success))
            last_eval_metrics = dict(eval_metrics)

            row = metric_row(
                global_step,
                episode_idx,
                ep_return,
                ep_success,
                ep_step,
                replay.size,
                last_her_ratio,
                last_loss,
                last_lite,
                wall_start,
            )
            apply_eval_metrics(row, eval_metrics, eval_history)
            eval_rows.append(row.copy())
            csv_writer.write(row)
            jsonl_writer.write({"type": "eval", **row, **eval_metrics})

            is_best = (eval_success > best_eval_success) or (
                math.isclose(eval_success, best_eval_success) and eval_metrics["eval_return_mean"] > best_eval_return
            )
            if is_best:
                best_eval_success = eval_success
                best_eval_return = float(eval_metrics["eval_return_mean"])
                best_eval_step = global_step
                save_checkpoint(_best_checkpoint_path(run_dir), checkpoint_payload(global_step, "best_eval"))
                write_json(
                    run_dir / "best_checkpoint.json",
                    {
                        "checkpoint": str(_best_checkpoint_path(run_dir)),
                        "step": global_step,
                        "eval_success_mean": best_eval_success,
                        "eval_goal_success_mean": eval_metrics["eval_goal_success_mean"],
                        "eval_env_success_mean": eval_metrics["eval_env_success_mean"],
                        "eval_return_mean": best_eval_return,
                    },
                )
            logger.info(
                "step=%d episode=%d replay=%d eval=%.3f ret=%.2f wP=%.3f psi=%.3f fallback=%.3f",
                global_step,
                episode_idx,
                replay.size,
                eval_success,
                eval_metrics["eval_return_mean"],
                last_lite["lite/mean_w_P"],
                last_lite["lite/psi_q_rank_rate"],
                last_lite["lite/q_candidate_fallback_rate"],
            )

        do_ckpt = (global_step % cfg.checkpoint_every_steps == 0 and global_step > 0) or (
            global_step == cfg.total_steps
        )
        if do_ckpt:
            payload = checkpoint_payload(global_step, "periodic")
            save_checkpoint(_checkpoint_path(run_dir), payload)
            if cfg.save_step_checkpoints:
                save_checkpoint(_step_checkpoint_path(run_dir, global_step), payload)

        if global_step % cfg.log_every_steps == 0 and global_step > 0:
            logger.info(
                "step=%d replay=%d updates=%d loss_q=%.4f loss_pi=%.4f",
                global_step,
                replay.size,
                total_updates,
                last_loss["critic"],
                last_loss["actor"],
            )

    runtime_total_sec = float(time.perf_counter() - wall_start)
    summary = build_summary(
        run_dir=run_dir,
        args=args,
        cfg=cfg,
        spec=spec,
        episode_idx=episode_idx,
        total_updates=total_updates,
        replay_size=replay.size,
        replay_indexed_slots=replay.indexed_slot_count(),
        best_eval_success=best_eval_success,
        best_eval_return=best_eval_return,
        best_eval_step=best_eval_step,
        last_loss=last_loss,
        last_lite=last_lite,
        last_eval_metrics=last_eval_metrics,
        eval_history=eval_history,
        eval_rows=eval_rows,
        runtime_total_sec=runtime_total_sec,
    )
    write_json(run_dir / "summary.json", summary)
    csv_writer.close()
    jsonl_writer.close()
    train_env.close()
    eval_env.close()
    logger.info("finished run at %s", run_dir)
    return run_dir


def metric_fields(phase_names: Iterable[str], control_mode: str) -> list[str]:
    fields = [
        "step",
        "episode",
        "train/episode_return",
        "train/episode_success",
        "train/episode_len",
        "train/replay_size",
        "train/her_relabel_ratio",
        "loss/critic",
        "loss/actor",
        "critic/q_policy_mean",
        "critic/q_target_mean",
        "lite/mean_w_P",
        "lite/max_w_P",
        "lite/std_w_P",
        "lite/entropy_w_P",
        "lite/ESS_w",
        "lite/ESS_w_normalized",
        "lite/support_available_rate",
        "lite/psi_q_rank_rate",
        "lite/q_candidate_fallback_rate",
        "lite/local_scan_truncated_rate",
        "lite/local_support_count_mean",
        "lite/local_support_count_p50",
        "lite/local_support_count_p90",
        "lite/mean_len_A_Q_replay_local",
        "lite/mean_len_A_Q_sup",
        "lite/mean_num_coarse_candidates",
        "lite/mean_num_scanned_candidates",
        "time/local_context_lookup",
        "time/candidate_construction",
        "time/weight_compute",
        "time/actor_critic_update",
        "time/q_candidate_eval",
        "time/q_policy_eval",
        "runtime/full_context_scan_rate",
        "runtime/exact_negative_short_circuit_rate",
        "runtime/support_cache_hit_rate",
        "runtime/support_cache_miss_rate",
        "runtime/support_cache_attempt_rate",
        "runtime/support_context_cache_size",
        "runtime/batch_query_unique_rate",
        "runtime/batch_query_coalescing_saving_rate",
        "runtime/local_scan_cache_hit_rate",
        "runtime/local_scan_cache_miss_rate",
        "runtime/local_scan_cache_attempt_rate",
        "runtime/local_scan_cache_size",
        "runtime/full_context_filter_count_mean",
        "runtime/steps_per_sec",
        "runtime/total_sec",
        "eval/success_mean",
        "eval/goal_success_mean",
        "eval/env_success_mean",
        "eval/return_mean",
        "eval/final_success",
        "eval/best_success",
        "eval/AUC_success",
        "fetch_slide/object_goal_dist_mean",
        "fetch_slide/object_goal_dist_p50",
        "fetch_slide/object_goal_dist_p90",
        "fetch_slide/object_speed_mean",
        "fetch_slide/object_speed_p90",
        "fetch_slide/object_velocity_alignment_mean",
        "fetch_slide/strike_contact_like_rate",
        "fetch_slide/progress_rate",
        "fetch_slide/overshoot_rate",
    ]
    for threshold in DEFAULT_THRESHOLDS:
        fields.append(f"eval/first_hit_{threshold:.2f}")
    for phase in phase_names:
        fields.extend(
            [
                f"phase/{phase}/count_rate",
                f"phase/{phase}/mean_w_P",
                f"phase/{phase}/support_rate",
                f"phase/{phase}/fallback_rate",
                f"phase/{phase}/local_scan_truncated_rate",
                f"phase/{phase}/td_error_abs_mean",
                f"fetch_slide/phase_distribution_{phase}",
            ]
        )
    if control_mode == "random":
        fields.extend(
            [
                "random/mean_w_P_reference",
                "random/reference_high_weight_rate",
                "random/assigned_high_weight_rate",
                "random/support_weight_alignment_rate",
            ]
        )
    if control_mode == "shuffled":
        fields.extend(
            [
                "shuffled/mean_w_P_reference",
                "shuffled/reference_high_weight_rate",
                "shuffled/assigned_high_weight_rate",
                "shuffled/support_weight_alignment_rate",
            ]
        )
    if control_mode == "td_error":
        fields.extend(
            [
                "td_error/abs_mean",
                "td_error/abs_p90",
                "td_error/weight_mean",
                "td_error/weight_max",
                "td_error/weight_entropy",
                "td_error/weight_ESS",
            ]
        )
    if control_mode == "per_style":
        fields.extend(
            [
                "per/priority_entropy",
                "per/priority_ESS",
                "per/weight_mean",
                "per/weight_max",
                "per/weight_entropy",
                "per/weight_ESS",
            ]
        )
    return list(dict.fromkeys(fields))


def empty_metrics(phase_names: Iterable[str], control_mode: str) -> Dict[str, float]:
    wb = np.ones(1, dtype=np.float32)
    metrics = {
        "lite/mean_w_P": 1.0,
        "lite/max_w_P": 1.0,
        "lite/std_w_P": 0.0,
        "lite/entropy_w_P": 0.0,
        "lite/ESS_w": 1.0,
        "lite/ESS_w_normalized": 1.0,
        "lite/support_available_rate": 0.0,
        "lite/psi_q_rank_rate": 0.0,
        "lite/q_candidate_fallback_rate": 1.0,
        "lite/local_scan_truncated_rate": 0.0,
        "lite/local_support_count_mean": 0.0,
        "lite/local_support_count_p50": 0.0,
        "lite/local_support_count_p90": 0.0,
        "lite/mean_len_A_Q_replay_local": 0.0,
        "lite/mean_len_A_Q_sup": 0.0,
        "lite/mean_num_coarse_candidates": 0.0,
        "lite/mean_num_scanned_candidates": 0.0,
        "time/local_context_lookup": 0.0,
        "time/candidate_construction": 0.0,
        "time/weight_compute": 0.0,
        "time/actor_critic_update": 0.0,
        "time/q_candidate_eval": 0.0,
        "time/q_policy_eval": 0.0,
    }
    for phase in phase_names:
        metrics[f"phase/{phase}/count_rate"] = 0.0
        metrics[f"phase/{phase}/mean_w_P"] = 0.0
        metrics[f"phase/{phase}/support_rate"] = 0.0
        metrics[f"phase/{phase}/fallback_rate"] = 0.0
        metrics[f"phase/{phase}/local_scan_truncated_rate"] = 0.0
        metrics[f"phase/{phase}/td_error_abs_mean"] = 0.0
        metrics[f"fetch_slide/phase_distribution_{phase}"] = 0.0
    metrics.update({key: 0.0 for key in metric_fields(phase_names, control_mode) if "/" in key and key not in metrics})
    metrics["lite/ESS_w"] = _ess(wb, 1e-8)
    return metrics


def metrics_from_weight_batch(wb: WeightBatch, phase_names: Iterable[str], control_mode: str) -> Dict[str, float]:
    w = wb.w_p_raw.astype(np.float32)
    metrics = empty_metrics(phase_names, control_mode)
    metrics.update(
        {
            "lite/mean_w_P": float(np.mean(w)) if w.size else 1.0,
            "lite/max_w_P": float(np.max(w)) if w.size else 1.0,
            "lite/std_w_P": float(np.std(w)) if w.size else 0.0,
            "lite/entropy_w_P": _entropy(w, 1e-8),
            "lite/ESS_w": _ess(w, 1e-8),
            "lite/ESS_w_normalized": _ess(w, 1e-8) / max(1, int(w.size)),
            "lite/support_available_rate": _mean(wb.psi_q_rank),
            "lite/psi_q_rank_rate": _mean(wb.psi_q_rank),
            "lite/q_candidate_fallback_rate": _mean(wb.q_candidate_fallback),
            "lite/local_scan_truncated_rate": _mean(wb.local_scan_truncated),
            "lite/local_support_count_mean": _mean(wb.len_a_q_local),
            "lite/local_support_count_p50": _percentile(wb.len_a_q_local, 50.0),
            "lite/local_support_count_p90": _percentile(wb.len_a_q_local, 90.0),
            "lite/mean_len_A_Q_replay_local": _mean(wb.len_a_q_local),
            "lite/mean_len_A_Q_sup": _mean(wb.len_a_q_sup),
            "lite/mean_num_coarse_candidates": _mean(wb.num_coarse_candidates),
            "lite/mean_num_scanned_candidates": _mean(wb.num_scanned_candidates),
            "time/local_context_lookup": float(wb.timing.get("local_context_lookup", 0.0)),
            "time/candidate_construction": float(wb.timing.get("candidate_construction", 0.0)),
            "time/weight_compute": float(wb.timing.get("weight_compute", 0.0)),
            "time/q_candidate_eval": float(wb.timing.get("q_candidate_eval", 0.0)),
            "time/q_policy_eval": float(wb.timing.get("q_policy_eval", 0.0)),
        }
    )
    phases = getattr(wb, "phase", np.zeros_like(w, dtype=np.int64))
    for phase_id, phase in enumerate(phase_names):
        mask = phases == phase_id
        if np.any(mask):
            metrics[f"phase/{phase}/count_rate"] = float(np.mean(mask))
            metrics[f"phase/{phase}/mean_w_P"] = float(np.mean(w[mask]))
            metrics[f"phase/{phase}/support_rate"] = float(np.mean(wb.psi_q_rank[mask]))
            metrics[f"phase/{phase}/fallback_rate"] = float(np.mean(wb.q_candidate_fallback[mask]))
            metrics[f"phase/{phase}/local_scan_truncated_rate"] = float(np.mean(wb.local_scan_truncated[mask]))
    for key, value in wb.extra_metrics.items():
        metrics[key] = float(value)
    return metrics


def metric_row(
    step: int,
    episode: int,
    ep_return: float,
    ep_success: float,
    ep_len: int,
    replay_size: int,
    her_ratio: float,
    loss: Dict[str, float],
    lite: Dict[str, float],
    wall_start: float,
) -> Dict[str, Any]:
    total_sec = max(0.0, float(time.perf_counter() - wall_start))
    return {
        "step": int(step),
        "episode": int(episode),
        "train/episode_return": float(ep_return),
        "train/episode_success": float(ep_success),
        "train/episode_len": int(ep_len),
        "train/replay_size": int(replay_size),
        "train/her_relabel_ratio": float(her_ratio),
        "loss/critic": loss["critic"],
        "loss/actor": loss["actor"],
        "critic/q_policy_mean": loss["q_mean"],
        "critic/q_target_mean": loss["target_mean"],
        **lite,
        "runtime/total_sec": total_sec,
        "runtime/steps_per_sec": float(step) / max(total_sec, 1e-8),
    }


def apply_eval_metrics(row: Dict[str, Any], eval_metrics: Dict[str, float], history: list[tuple[int, float]]) -> None:
    row["eval/success_mean"] = eval_metrics["eval_success_mean"]
    row["eval/goal_success_mean"] = eval_metrics["eval_goal_success_mean"]
    row["eval/env_success_mean"] = eval_metrics["eval_env_success_mean"]
    row["eval/return_mean"] = eval_metrics["eval_return_mean"]
    row["eval/final_success"] = eval_metrics["eval_env_success_final_mean"]
    row["eval/best_success"] = max(value for _, value in history) if history else eval_metrics["eval_env_success_mean"]
    row["eval/AUC_success"] = auc(history)
    for threshold in DEFAULT_THRESHOLDS:
        row[f"eval/first_hit_{threshold:.2f}"] = first_hit(history, threshold)
    slide_metrics = fetch_slide_eval_metrics(eval_metrics)
    row.update(slide_metrics)


@torch.no_grad()
def evaluate_fetch_slide_agent(
    agent: DDPGAgent,
    env: Any,
    eval_seed: int,
    eval_episodes: int,
    task: Any,
) -> Dict[str, float]:
    returns = []
    goal_successes_ever = []
    env_successes_ever = []
    goal_successes_final = []
    env_successes_final = []
    episode_lengths = []
    obj_goal_dist = []
    obj_speed = []
    vel_align = []
    strike_contact = []
    progress = []
    overshoot = []
    phase_values = []
    dummy_rng = np.random.default_rng(0)

    for ep in range(eval_episodes):
        obs = env.reset(seed=eval_seed + ep)
        ep_ret = 0.0
        ep_goal_ever = 0.0
        ep_env_ever = 0.0
        ep_goal_final = 0.0
        ep_env_final = 0.0
        ep_len = 0
        prev_dist = None
        for _ in range(env.max_episode_steps):
            action = agent.act(obs.flat_state(), noise_std=0.0, noise_rng=dummy_rng)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

            goal_succ = float(info.get("goal_success", info.get("is_success", 0.0)))
            env_succ = float(info.get("env_success", info.get("is_success", goal_succ)))
            ep_goal_ever = max(ep_goal_ever, goal_succ)
            ep_env_ever = max(ep_env_ever, env_succ)
            ep_goal_final = goal_succ
            ep_env_final = env_succ

            _, p_obj, _, _, d_ee_obj, speed_xy, align = task.parse_features(
                obs.observation[None, :],
                obs.desired_goal[None, :],
            )
            dist = float(np.linalg.norm(p_obj[0] - obs.desired_goal))
            obj_goal_dist.append(dist)
            obj_speed.append(float(speed_xy[0]))
            vel_align.append(float(align[0]))
            strike_contact.append(float(d_ee_obj[0] <= task.d_contact))
            if prev_dist is not None:
                progress.append(float(dist < prev_dist))
            prev_dist = dist
            overshoot.append(float(speed_xy[0] >= task.v_slide_min and align[0] < 0.0))
            phase_values.append(int(task.phase_batch(obs.observation[None, :], obs.desired_goal[None, :])[0]))

            if terminated or truncated:
                break
        returns.append(ep_ret)
        goal_successes_ever.append(ep_goal_ever)
        env_successes_ever.append(ep_env_ever)
        goal_successes_final.append(ep_goal_final)
        env_successes_final.append(ep_env_final)
        episode_lengths.append(ep_len)

    metrics = {
        "eval_return_mean": _mean_array(returns),
        "eval_success_mean": _mean_array(env_successes_ever),
        "eval_goal_success_mean": _mean_array(goal_successes_ever),
        "eval_env_success_mean": _mean_array(env_successes_ever),
        "eval_goal_success_ever_mean": _mean_array(goal_successes_ever),
        "eval_env_success_ever_mean": _mean_array(env_successes_ever),
        "eval_goal_success_final_mean": _mean_array(goal_successes_final),
        "eval_env_success_final_mean": _mean_array(env_successes_final),
        "eval_episode_length_mean": _mean_array(episode_lengths),
        "fetch_slide/object_goal_dist_mean": _mean_array(obj_goal_dist),
        "fetch_slide/object_goal_dist_p50": _percentile_list(obj_goal_dist, 50.0),
        "fetch_slide/object_goal_dist_p90": _percentile_list(obj_goal_dist, 90.0),
        "fetch_slide/object_speed_mean": _mean_array(obj_speed),
        "fetch_slide/object_speed_p90": _percentile_list(obj_speed, 90.0),
        "fetch_slide/object_velocity_alignment_mean": _mean_array(vel_align),
        "fetch_slide/strike_contact_like_rate": _mean_array(strike_contact),
        "fetch_slide/progress_rate": _mean_array(progress),
        "fetch_slide/overshoot_rate": _mean_array(overshoot),
    }
    phases = np.asarray(phase_values, dtype=np.int64)
    for phase_id, phase_name in enumerate(getattr(task, "phase_names", ())):
        metrics[f"fetch_slide/phase_distribution_{phase_name}"] = (
            float(np.mean(phases == phase_id)) if phases.size else 0.0
        )
    return metrics


def fetch_slide_eval_metrics(eval_metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in eval_metrics.items():
        if str(key).startswith("fetch_slide/"):
            out[str(key)] = float(value)
    key_map = {
        "lite/slide_mean_obj_speed": "fetch_slide/object_speed_mean",
        "lite/slide_mean_vel_align": "fetch_slide/object_velocity_alignment_mean",
        "lite/slide_phase_distribution_approach": "fetch_slide/phase_distribution_approach",
        "lite/slide_phase_distribution_strike": "fetch_slide/phase_distribution_strike",
        "lite/slide_phase_distribution_slide": "fetch_slide/phase_distribution_slide",
        "lite/slide_phase_distribution_near_goal": "fetch_slide/phase_distribution_near_goal",
    }
    for src, dst in key_map.items():
        if src in eval_metrics:
            out[dst] = float(eval_metrics[src])
    out.setdefault("fetch_slide/object_goal_dist_mean", float("nan"))
    out.setdefault("fetch_slide/object_goal_dist_p50", float("nan"))
    out.setdefault("fetch_slide/object_goal_dist_p90", float("nan"))
    out.setdefault("fetch_slide/object_speed_p90", float("nan"))
    out.setdefault("fetch_slide/strike_contact_like_rate", out.get("fetch_slide/phase_distribution_strike", 0.0))
    out.setdefault("fetch_slide/progress_rate", float("nan"))
    out.setdefault("fetch_slide/overshoot_rate", float("nan"))
    return out


def build_summary(
    run_dir: Path,
    args: argparse.Namespace,
    cfg: LiteConfig,
    spec: MethodSpec,
    episode_idx: int,
    total_updates: int,
    replay_size: int,
    replay_indexed_slots: int,
    best_eval_success: float,
    best_eval_return: float,
    best_eval_step: int,
    last_loss: Dict[str, float],
    last_lite: Dict[str, float],
    last_eval_metrics: Dict[str, float],
    eval_history: list[tuple[int, float]],
    eval_rows: list[Dict[str, Any]],
    runtime_total_sec: float,
) -> Dict[str, Any]:
    successes = np.asarray([value for _, value in eval_history], dtype=np.float32)
    final_success = float(successes[-1]) if successes.size else float("nan")
    best_success = float(np.max(successes)) if successes.size else float("nan")
    final_window = successes[-5:] if successes.size else np.asarray([], dtype=np.float32)
    total_steps = int(cfg.total_steps)
    last_quarter = [(s, v) for s, v in eval_history if s >= int(0.75 * total_steps)]
    summary = {
        "run_dir": str(run_dir),
        "task": "FetchSlide",
        "method": args.method,
        "lite_method": cfg.method,
        "weight_scope": spec.weight_scope,
        "config_weight_scope": cfg.weight_scope,
        "control_mode": spec.control_mode,
        "support_lookup_enabled": uses_support_lookup(spec),
        "runtime_optimization_enabled": args.method == PATCHED_B3_METHOD,
        "runtime_optimization_patch": "b3_critic_only_deterministic_tail_v1_local_scan_cache",
        "support_scan_mode": str(getattr(cfg, "support_scan_mode", DEFAULT_SUPPORT_SCAN_MODE)),
        "seed": int(args.seed),
        "total_steps": total_steps,
        "local_search_max": int(cfg.local_search_max) if uses_support_lookup(spec) else None,
        "episodes": int(episode_idx),
        "total_updates": int(total_updates),
        "final_replay_size": int(replay_size),
        "final_replay_indexed_slots": int(replay_indexed_slots),
        "final_success": final_success,
        "best_success": best_success,
        "best_step": int(best_eval_step),
        "best_eval_return": float(best_eval_return),
        "auc_eval_mean": auc(eval_history),
        "auc_trapz_0_to_total": auc(eval_history, total_steps=total_steps),
        "drop_from_best_to_final": best_success - final_success if np.isfinite(best_success) else float("nan"),
        "last_quarter_auc": auc(last_quarter, total_steps=max(1, total_steps // 4)),
        "final_window_success_mean": float(np.mean(final_window)) if final_window.size else float("nan"),
        "final_window_success_std": float(np.std(final_window)) if final_window.size else float("nan"),
        "final_window_n": int(final_window.size),
        "runtime_total_sec": runtime_total_sec,
        "steps_per_sec": total_steps / max(runtime_total_sec, 1e-8),
        "final_losses": last_loss,
        "final_lite_metrics": last_lite,
        "final_eval_return": last_eval_metrics.get("eval_return_mean", float("nan")),
        "final_eval_goal_success": last_eval_metrics.get("eval_goal_success_mean", float("nan")),
        "final_eval_env_success": last_eval_metrics.get("eval_env_success_mean", float("nan")),
    }
    for threshold in DEFAULT_THRESHOLDS:
        summary[f"first_hit_{threshold:.2f}"] = first_hit(eval_history, threshold) or None
    return summary


def _batch_norm(weights: np.ndarray, eps: float) -> np.ndarray:
    arr = weights.astype(np.float32)
    return (arr / max(float(np.mean(arr)), eps)).astype(np.float32)


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _mean_array(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    return float(np.mean(arr)) if arr.size else 0.0


def _percentile_list(values: Iterable[float], pct: float) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, pct))


def _percentile(arr: np.ndarray, pct: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr.astype(np.float32), pct))


def _entropy(weights: np.ndarray, eps: float) -> float:
    if weights.size == 0:
        return 0.0
    p = np.maximum(weights.astype(np.float64), eps)
    p = p / max(float(p.sum()), eps)
    return float(-(p * np.log(p + eps)).sum())


def _ess(weights: np.ndarray, eps: float) -> float:
    if weights.size == 0:
        return 0.0
    w = np.maximum(weights.astype(np.float64), eps)
    return float((w.sum() ** 2) / max(float(np.sum(w * w)), eps))


def auc(history: list[tuple[int, float]], total_steps: int | None = None) -> float:
    if not history:
        return float("nan")
    if len(history) == 1:
        return float(history[0][1])
    steps = np.asarray([s for s, _ in history], dtype=np.float64)
    values = np.asarray([v for _, v in history], dtype=np.float64)
    span = float(total_steps) if total_steps is not None else float(steps[-1] - steps[0])
    span = max(1.0, span)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, steps) / span)


def first_hit(history: list[tuple[int, float]], threshold: float) -> str:
    for step, value in history:
        if value >= threshold:
            return str(int(step))
    return ""


def main() -> int:
    args = parse_args()
    run_dir = run(args)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
