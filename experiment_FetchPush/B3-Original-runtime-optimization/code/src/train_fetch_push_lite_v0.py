#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaCog-Lite v0 FetchPush trainer.")
    parser.add_argument("--run_name", type=str, default="fetch_push_lite_v0")
    parser.add_argument("--output_root", type=str, default="version_v0_lite/results")
    parser.add_argument(
        "--method",
        type=str,
        default="b4_lite",
        choices=[
            "b0_ddpg_her",
            "b1_uniform_scale",
            "b2_unrestricted_gap",
            "b3_support_only",
            "b3_support_random",
            "b3_support_shuffled",
            "b4_lite",
        ],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--reward_type", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=5_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_capacity", type=int, default=300_000)
    parser.add_argument("--her_ratio", type=float, default=0.8)

    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--l2_action_penalty", type=float, default=0.0)

    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    parser.add_argument("--no_save_step_checkpoints", action="store_false", dest="save_step_checkpoints")
    parser.set_defaults(save_step_checkpoints=True)
    parser.add_argument("--log_every_steps", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--M_B", type=int, default=32)
    parser.add_argument("--M_Q_min", type=int, default=8)
    parser.add_argument("--N_local_min", type=int, default=8)
    parser.add_argument("--local_search_max", type=int, default=8192)
    parser.add_argument("--action_bins_per_dim", type=int, default=5)
    parser.add_argument("--policy_support_mode", type=str, default="bin", choices=["bin", "radius"])
    parser.add_argument("--policy_action_radius", type=float, default=0.35)
    parser.add_argument("--support_lookup_optimization", action="store_true")
    parser.add_argument("--support_cache_enabled", action="store_true")
    parser.add_argument("--support_short_circuit", action="store_true")
    parser.add_argument("--support_cache_max_entries", type=int, default=200_000)

    parser.add_argument("--d_contact", type=float, default=0.08)
    parser.add_argument("--d_goal_near", type=float, default=0.05)
    parser.add_argument("--N_gap_min", type=int, default=256)
    parser.add_argument("--gap_window", type=int, default=50_000)
    parser.add_argument("--lambda_P", type=float, default=1.0)
    parser.add_argument("--lambda_support", type=float, default=1.0)
    parser.add_argument("--U_P_max", type=float, default=1.0)
    parser.add_argument("--weighting_mode", type=str, default="batch_norm", choices=["batch_norm", "raw"])
    parser.add_argument("--weight_scope", type=str, default="actor_critic", choices=["actor_critic", "critic_only", "actor_only"])
    parser.add_argument("--unrestricted_candidate_std", type=float, default=0.5)
    parser.add_argument("--success_terminates", action="store_true")
    parser.add_argument("--no_timeout_bootstrap", action="store_false", dest="timeout_bootstrap")
    parser.set_defaults(timeout_bootstrap=True)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--env_seed", type=int, default=1)
    parser.add_argument("--torch_seed", type=int, default=1)
    parser.add_argument("--numpy_seed", type=int, default=1)
    parser.add_argument("--action_noise_seed", type=int, default=1)
    parser.add_argument("--weight_seed", type=int, default=31)
    parser.add_argument("--eval_seed", type=int, default=10001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = LiteConfig(
        run_name=args.run_name,
        output_root=args.output_root,
        method=args.method,
        resume=args.resume,
        resume_path=args.resume_path,
        reward_type=args.reward_type,
        max_episode_steps=args.max_episode_steps,
        render=args.render,
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
        local_search_max=args.local_search_max,
        action_bins_per_dim=args.action_bins_per_dim,
        policy_support_mode=args.policy_support_mode,
        policy_action_radius=args.policy_action_radius,
        support_lookup_optimization=args.support_lookup_optimization,
        support_cache_enabled=args.support_cache_enabled,
        support_short_circuit=args.support_short_circuit,
        support_cache_max_entries=args.support_cache_max_entries,
        d_contact=args.d_contact,
        d_goal_near=args.d_goal_near,
        N_gap_min=args.N_gap_min,
        gap_window=args.gap_window,
        lambda_P=args.lambda_P,
        lambda_support=args.lambda_support,
        U_P_max=args.U_P_max,
        weighting_mode=args.weighting_mode,
        weight_scope=args.weight_scope,
        unrestricted_candidate_std=args.unrestricted_candidate_std,
        success_terminates=args.success_terminates,
        timeout_bootstrap=args.timeout_bootstrap,
        device=args.device,
        seeds=SeedConfig(
            env_seed=args.env_seed,
            torch_seed=args.torch_seed,
            numpy_seed=args.numpy_seed,
            action_noise_seed=args.action_noise_seed,
            weight_seed=args.weight_seed,
            eval_seed=args.eval_seed,
        ),
    )
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = run_training(cfg, repo_root=repo_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
