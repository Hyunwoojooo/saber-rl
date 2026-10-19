"""Frozen scientific and analysis contract. All paths are experiment-local."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "src"))
from metacog_lite_v0.config import LiteConfig, SeedConfig

ARMS = ("support", "random", "td_topk")
SEEDS = tuple(range(1, 11))
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_RESAMPLES = 10_000
ANALYSIS_VERSION = "assignment-v1"


def make_config(arm, seed, output_root, profile="production", device="cuda"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Arm/seed outside the preregistered cohort")
    if profile not in {"production", "smoke"}:
        raise ValueError(profile)
    cfg = LiteConfig(
        task_name="drawer_open", method="b3_support_runtime_opt",
        reward_type="sparse", metaworld_env_name="drawer-open-v3",
        metaworld_reward_function_version="v2", max_episode_steps=200,
        total_steps=2_000_000, start_steps=10_000, warmup_steps=5_000,
        updates_per_step=1, batch_size=256, buffer_capacity=300_000, her_ratio=0.8,
        gamma=0.98, tau=0.005, actor_lr=0.0003, critic_lr=0.0003,
        hidden_dim=256, noise_std=0.2, l2_action_penalty=0.0,
        eval_freq=10_000, eval_episodes=20, checkpoint_every_steps=50_000,
        save_step_checkpoints=False, log_every_steps=1_000,
        M_B=16, M_Q_min=4, N_local_min=4, N_gap_min=128,
        local_search_max=512, action_bins_per_dim=5, lambda_support=1.0,
        drawer_success_threshold=0.05, drawer_d_reach=0.08,
        drawer_d_goal_near=0.05, drawer_d_opening_mid=0.12,
        weight_scope="critic_only", weighting_mode="batch_norm",
        policy_support_mode="bin", support_query_mode="current_policy",
        support_assignment_mode="support", support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=True, support_cache_max_entries=200_000,
        local_scan_cache_enabled=True, local_scan_cache_max_entries=50_000,
        support_short_circuit=True, numba_support_scan=True,
        support_predicate_early_exit=True, support_diagnostic_audit_interval=100,
        success_terminates=False, timeout_bootstrap=True,
        assignment_experiment=True, assignment_rule=arm,
        assignment_diagnostic_window_updates=1_000,
        run_name="run", output_root=str(output_root), device=device,
        seeds=SeedConfig(env_seed=seed, torch_seed=seed, numpy_seed=seed,
                         action_noise_seed=seed, weight_seed=20_000 + seed,
                         assignment_seed=30_000 + seed, eval_seed=10_000 + seed),
    )
    # Smoke is a separate, ineligible-for-results profile. Production is immutable.
    if profile == "smoke":
        cfg.total_steps = 6_000
        cfg.start_steps = 1_000
        cfg.eval_freq = 2_000
        cfg.eval_episodes = 2
        cfg.checkpoint_every_steps = 6_000
    cfg.normalize()
    return cfg


def evaluation_seed(seed, checkpoint, episode):
    return 10_000 + seed + 10 * checkpoint + episode


def scientific_config(config):
    ignored = {"assignment_rule", "run_name", "output_root", "device", "seeds",
               "thread_hint", "terminal_convention", "assignment_log_enabled"}
    return {k: v for k, v in config.items() if k not in ignored}
