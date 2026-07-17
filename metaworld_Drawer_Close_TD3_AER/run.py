#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parent
CODE_SRC = EXPERIMENT_ROOT / "code" / "src"
sys.path.insert(0, str(CODE_SRC))

from metacog_lite_v0.aer import AER_METHODS, TD3_AER, TD3_ANNEALING_ER, TD3_ER
from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.trainer import make_env, run_training


RUN_LABELS = {
    TD3_ER: "TD3-ER-2M",
    TD3_ANNEALING_ER: "TD3-AnnealingER-2M",
    TD3_AER: "TD3-AER-2M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched no-HER TD3 ER/AnnealingER/AER experiment on MetaWorld DrawerClose."
    )
    parser.add_argument("--method", choices=sorted(AER_METHODS), default=TD3_AER)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        default="metaworld_Drawer_Close_TD3_AER/results_2m",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument(
        "--save-step-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LiteConfig:
    seed = int(args.seed)
    run_name = args.run_name or f"{RUN_LABELS[args.method]}-seed{seed}"
    return LiteConfig(
        run_name=run_name,
        output_root=args.output_root,
        resume=bool(args.resume),
        resume_path=args.resume_path,
        task_name="drawer_close",
        method=args.method,
        reward_type="sparse",
        metaworld_env_name="drawer-close-v3",
        metaworld_reward_function_version="v2",
        max_episode_steps=200,
        total_steps=int(args.steps),
        start_steps=10_000,
        warmup_steps=5_000,
        updates_per_step=1,
        batch_size=256,
        buffer_capacity=300_000,
        her_ratio=0.0,
        gamma=0.98,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        hidden_dim=256,
        noise_std=0.2,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        l2_action_penalty=0.0,
        aer_lambda0=2.5,
        aer_anneal_fraction=2.0 / 3.0,
        aer_similarity_eps=1e-8,
        eval_freq=10_000,
        eval_episodes=int(args.eval_episodes),
        final_eval_episodes=int(args.final_eval_episodes),
        checkpoint_every_steps=int(args.checkpoint_every),
        save_step_checkpoints=bool(args.save_step_checkpoints),
        log_every_steps=1_000,
        smoke=bool(args.smoke),
        weighting_mode="batch_norm",
        weight_scope="none",
        success_terminates=False,
        timeout_bootstrap=True,
        device=args.device,
        seeds=SeedConfig(
            env_seed=seed,
            torch_seed=seed,
            numpy_seed=seed,
            action_noise_seed=seed,
            replay_seed=30_000 + seed,
            weight_seed=20_000 + seed,
            eval_seed=10_000 + seed,
            final_eval_seed=50_000 + seed,
        ),
    )


def validate_config(config: LiteConfig) -> None:
    config.normalize()
    checks = {
        "task": config.task_name == "drawer_close",
        "environment": config.metaworld_env_name == "drawer-close-v3",
        "sparse_reward": config.reward_type == "sparse",
        "no_her": config.her_ratio == 0.0,
        "td3_utd1": config.updates_per_step == 1 and config.policy_delay == 2,
        "unit_loss_weights": config.weight_scope == "none",
        "batch_256": config.batch_size == 256 or config.smoke,
        "aer_schedule": config.aer_lambda0 == 2.5 and config.aer_anneal_fraction == 2.0 / 3.0,
        "evaluation": config.eval_freq <= 10_000 and config.eval_episodes > 0,
    }
    env = make_env(config, config.seeds.env_seed, False)
    try:
        snapshot = env.snapshot()
        checks.update(
            {
                "state_is_observation_plus_goal": env.state_dim == env.obs_dim + env.goal_dim,
                "horizon_200": env.max_episode_steps == 200,
                "goal_source_safe": snapshot.get("desired_goal_source")
                not in {None, "unknown", "obs_tail", "synthetic_fallback"},
            }
        )
    finally:
        env.close()
    print(json.dumps({"checks": checks, "config": config.as_dict()}, indent=2, sort_keys=True))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"AER configuration validation failed: {failed}")


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.validate_only:
        validate_config(config)
        return 0
    run_dir = run_training(config, REPO_ROOT)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
