#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.trainer import make_env, make_task_spec, run_training
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaCog-Lite v0 ManiSkill PickCube trainer.")
    parser.add_argument("--task_name", type=str, default="pick_cube", choices=["pick_cube"])
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--run_name", type=str, default="pick_cube_lite_v0")
    parser.add_argument("--output_root", type=str, default="version_v0_lite/results")
    parser.add_argument(
        "--method",
        type=str,
        default="b4_lite",
        choices=["b0_ddpg_her", "b1_uniform_scale", "b2_unrestricted_gap", "b3_support_only", "b4_lite"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--validate_only", action="store_true")

    parser.add_argument("--reward_type", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--obs_mode", type=str, default="state")
    parser.add_argument("--robot_uids", type=str, default="panda")
    parser.add_argument("--control_mode", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--sim_backend", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--total_steps", type=int, default=100_000)
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
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    parser.add_argument("--no_save_step_checkpoints", action="store_false", dest="save_step_checkpoints")
    parser.set_defaults(save_step_checkpoints=True)
    parser.add_argument("--log_every_steps", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--M_B", type=int, default=16)
    parser.add_argument("--M_Q_min", type=int, default=4)
    parser.add_argument("--N_local_min", type=int, default=4)
    parser.add_argument("--local_search_max", type=int, default=1024)
    parser.add_argument("--action_bins_per_dim", type=int, default=5)
    parser.add_argument("--policy_support_mode", type=str, default="auto", choices=["auto", "bin", "radius"])
    parser.add_argument("--policy_action_radius", type=float, default=0.35)

    parser.add_argument("--d_reach", type=float, default=0.08)
    parser.add_argument("--d_goal_near", type=float, default=0.025)
    parser.add_argument("--pick_cube_z_lift_min", type=float, default=None)
    parser.add_argument("--N_gap_min", type=int, default=128)
    parser.add_argument("--gap_window", type=int, default=50_000)
    parser.add_argument("--lambda_P", type=float, default=1.0)
    parser.add_argument("--lambda_support", type=float, default=1.0)
    parser.add_argument("--U_P_max", type=float, default=1.0)
    parser.add_argument("--weighting_mode", type=str, default="batch_norm", choices=["batch_norm", "raw"])
    parser.add_argument(
        "--weight_scope",
        type=str,
        default="actor_critic",
        choices=["actor_critic", "critic_only", "actor_only"],
    )
    parser.add_argument("--unrestricted_candidate_std", type=float, default=0.5)
    parser.add_argument("--success_terminates", action="store_true")
    parser.add_argument("--no_timeout_bootstrap", action="store_false", dest="timeout_bootstrap")
    parser.set_defaults(timeout_bootstrap=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_seed", type=int, default=1)
    parser.add_argument("--torch_seed", type=int, default=1)
    parser.add_argument("--numpy_seed", type=int, default=1)
    parser.add_argument("--action_noise_seed", type=int, default=1)
    parser.add_argument("--weight_seed", type=int, default=20001)
    parser.add_argument("--eval_seed", type=int, default=10001)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LiteConfig:
    return LiteConfig(
        task_name=args.task_name,
        env_id=args.env_id,
        run_name=args.run_name,
        output_root=args.output_root,
        method=args.method,
        resume=args.resume,
        resume_path=args.resume_path,
        reward_type=args.reward_type,
        obs_mode=args.obs_mode,
        robot_uids=args.robot_uids,
        control_mode=args.control_mode,
        backend=args.backend,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
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
        d_reach=args.d_reach,
        d_goal_near=args.d_goal_near,
        pick_cube_z_lift_min=args.pick_cube_z_lift_min,
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


def validate_pick_cube_setup(config: LiteConfig) -> None:
    config.normalize()
    env = make_env(config, config.seeds.env_seed, False)
    if config.policy_support_mode == "auto":
        config.policy_support_mode = "bin" if int(env.action_dim) <= 4 else "radius"
    task = make_task_spec(config, env)
    rng = np.random.default_rng(config.seeds.numpy_seed)
    device = torch.device("cpu")
    obs = env.reset(seed=config.seeds.env_seed)
    h0 = env.max_episode_steps
    alt_goal = obs.desired_goal + np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    phase_ids = []
    cube_z_values = []
    goal_z_values = []
    d_tcp_cube_values = []
    d_cube_goal_values = []
    checks = {
        "env_id": env.snapshot()["env_id"],
        "robot_uids": env.snapshot().get("robot_uids"),
        "reset_finite": _finite_goal_obs(obs),
        "flat_state_is_concat_obs_goal": bool(
            np.allclose(obs.flat_state(), np.concatenate([obs.observation, obs.desired_goal], axis=0))
        ),
        "obs_dim": int(env.obs_dim),
        "goal_dim": int(env.goal_dim),
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "policy_support_mode_resolved": config.policy_support_mode,
        "distance_threshold": float(env.distance_threshold),
        "z_lift_min": float(getattr(env, "z_lift_min", 0.0)),
        "phase_names": list(getattr(task, "phase_names", ())),
        "action_low_finite": bool(np.all(np.isfinite(env.action_low))),
        "action_high_finite": bool(np.all(np.isfinite(env.action_high))),
        "coarse_key_goal_independent": True,
        "full_context_hash_changes_with_goal": bool(
            task.full_context_hash(obs.observation, obs.desired_goal, h0)
            != task.full_context_hash(obs.observation, alt_goal, h0)
        ),
    }
    replay = HERReplayBuffer(
        capacity=512,
        obs_dim=env.obs_dim,
        goal_dim=env.goal_dim,
        action_dim=env.action_dim,
    )
    ep_id = 0
    ep_step = 0
    for _ in range(30):
        h_t = env.max_episode_steps - ep_step
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        obs_next, reward, terminated, truncated, info = env.step(action)
        replay.add_transition(
            observation=obs.observation,
            achieved_goal=obs.achieved_goal,
            desired_goal=obs.desired_goal,
            action_exec=action,
            reward_ext=float(reward),
            next_observation=obs_next.observation,
            next_achieved_goal=obs_next.achieved_goal,
            done=bool(terminated or truncated),
            timeout=bool(truncated),
            h=h_t,
            episode_id=ep_id,
            episode_step=ep_step,
            coarse_key=task.coarse_hash(obs.observation, h_t),
            action_bin=int(task.action_bin_batch(action[None, :], config.action_bins_per_dim)[0]),
        )
        phase_ids.append(int(task.phase_batch(obs.observation[None, :], obs.desired_goal[None, :])[0]))
        tcp, cube, cube_z, _is_grasped = task.parse_features(obs.observation[None, :])
        cube_z_values.append(float(cube_z[0]))
        goal_z_values.append(float(obs.desired_goal[2]))
        d_tcp_cube_values.append(float(np.linalg.norm(tcp[0] - cube[0])))
        d_cube_goal_values.append(float(np.linalg.norm(cube[0] - obs.desired_goal)))
        obs = obs_next
        ep_step += 1
        if terminated or truncated:
            ep_id += 1
            ep_step = 0
            obs = env.reset(seed=config.seeds.env_seed + ep_id)

    batch_size = min(16, replay.size)
    batch = replay.sample(batch_size, her_ratio=1.0, rng=rng)
    original_goal = replay.desired_goal[batch.indices]
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=min(config.hidden_dim, 256),
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
        l2_action_penalty=config.l2_action_penalty,
    )
    gap_stats = GapStats(
        maxlen=config.gap_window,
        n_min=min(config.N_gap_min, 4),
        q_low=config.gap_q_low,
        q_high=config.gap_q_high,
        eps_g=config.eps_G,
    )
    b0_wb = LiteWeightComputer(replace(config, method="b0_ddpg_her"), task, gap_stats).compute(
        batch, replay, agent, rng, device
    )
    b2_wb = LiteWeightComputer(replace(config, method="b2_unrestricted_gap"), task, gap_stats).compute(
        batch, replay, agent, rng, device
    )
    b3_wb = LiteWeightComputer(replace(config, method="b3_support_only"), task, gap_stats).compute(
        batch, replay, agent, rng, device
    )
    b4_wb = LiteWeightComputer(replace(config, method="b4_lite"), task, gap_stats).compute(
        batch, replay, agent, rng, device
    )
    checks.update(
        {
            "random_step_30": True,
            "goal_success_logged_separately": "goal_success" in info,
            "env_success_logged_separately": "env_success" in info,
            "reward_shape": list(env.compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {}).shape),
            "success_shape": list(env.success_from_goal(batch.next_achieved_goal, batch.desired_goal_train).shape),
            "replay_size": int(replay.size),
            "her_relabel_ratio": float(batch.her_relabel_ratio),
            "her_relabel_changed_goal": bool(
                np.any(np.linalg.norm(batch.desired_goal_train - original_goal, axis=1) > 1e-6)
            ),
            "indexed_replay_slots": int(replay.indexed_slot_count()),
            "raw_reward_finite": bool(np.all(np.isfinite(replay.reward_ext[: replay.size]))),
            "b0_omega_all_one": bool(np.allclose(b0_wb.omega_critic, 1.0) and np.allclose(b0_wb.omega_actor, 1.0)),
            "b2_unrestricted_gap_finite": bool(np.all(np.isfinite(b2_wb.unrestricted_gap))),
            "b3_omega_finite": bool(np.all(np.isfinite(b3_wb.omega_critic))),
            "b4_omega_finite": bool(np.all(np.isfinite(b4_wb.omega_critic))),
            "b4_g_sup_finite": bool(np.all(np.isfinite(b4_wb.g_sup))),
            "phase_unique_count_random": int(len(set(phase_ids))),
            "phase_ids_random": sorted(set(phase_ids)),
            "cube_z_min_mean_max": _min_mean_max(cube_z_values),
            "goal_z_min_mean_max": _min_mean_max(goal_z_values),
            "d_tcp_cube_min_mean_max": _min_mean_max(d_tcp_cube_values),
            "d_cube_goal_min_mean_max": _min_mean_max(d_cube_goal_values),
            "time_q_candidate_eval_logged": "q_candidate_eval" in b4_wb.timing,
            "time_local_context_lookup_logged": "local_context_lookup" in b4_wb.timing,
        }
    )
    env.close()
    print(json.dumps(checks, indent=2, sort_keys=True))


def _finite_goal_obs(obs: object) -> bool:
    return bool(
        np.all(np.isfinite(obs.observation))
        and np.all(np.isfinite(obs.achieved_goal))
        and np.all(np.isfinite(obs.desired_goal))
    )


def _min_mean_max(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))]


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)
    repo_root = Path(__file__).resolve().parents[2]
    if args.validate_only:
        validate_pick_cube_setup(cfg)
        return 0
    run_dir = run_training(cfg, repo_root=repo_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
