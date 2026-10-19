#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.agent import SACAgent, TD3Agent
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.hgr import HGRSampler
from metacog_lite_v0.trainer import make_agent, make_env, make_task_spec, run_training
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaCog-Lite v0 FetchPickAndPlace trainer.")
    parser.add_argument("--task_name", type=str, default="fetch_pick_and_place", choices=["fetch_pick_and_place"])
    parser.add_argument("--run_name", type=str, default="fetch_pick_and_place_lite_v0")
    parser.add_argument("--output_root", type=str, default="version_v0_lite/results")
    parser.add_argument(
        "--method",
        type=str,
        default="b4_lite",
        choices=[
            "b0_ddpg_her",
            "hgr",
            "b1_uniform_scale",
            "b2_unrestricted_gap",
            "b3_support_only",
            "b3_support_random",
            "b3_support_shuffled",
            "b3_support_runtime_opt",
            "b4_lite",
            "td_error",
            "per_style_loss_weighting",
            "td3_her",
            "td3_hgr",
            "sac_hgr",
            "td3_her_high_rr_x4",
            "td3_saber_runtime_opt",
        ],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--validate_only", action="store_true")

    parser.add_argument("--reward_type", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=5_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_capacity", type=int, default=300_000)
    parser.add_argument("--her_ratio", type=float, default=0.8)
    parser.add_argument("--hgr_priority_eps", type=float, default=1e-6)
    parser.add_argument("--hgr_priority_init", type=float, default=1.0)
    parser.add_argument("--hgr_priority_clip_max", type=float, default=0.0)
    parser.add_argument("--hgr_alpha_episode", type=float, default=0.6)
    parser.add_argument("--hgr_alpha_goal", type=float, default=0.6)
    parser.add_argument("--hgr_beta_episode_initial", type=float, default=0.4)
    parser.add_argument("--hgr_beta_goal_initial", type=float, default=0.4)
    parser.add_argument("--hgr_beta_episode_final", type=float, default=1.0)
    parser.add_argument("--hgr_beta_goal_final", type=float, default=1.0)
    parser.add_argument("--hgr_beta_anneal_updates", type=int, default=0)
    parser.add_argument("--hgr_uniform_mix", type=float, default=0.0)
    parser.add_argument(
        "--hgr_sac_priority_reduction",
        type=str,
        default="min_abs",
        choices=["min_abs", "mean_abs"],
    )

    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--l2_action_penalty", type=float, default=0.0)
    parser.add_argument("--td3_policy_delay", type=int, default=2)
    parser.add_argument("--td3_target_policy_noise", type=float, default=0.2)
    parser.add_argument("--td3_target_noise_clip", type=float, default=0.5)
    parser.add_argument("--sac_alpha", type=float, default=0.2)
    parser.add_argument("--sac_auto_alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sac_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--sac_target_entropy", type=float, default=None)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    parser.add_argument("--no_save_step_checkpoints", action="store_false", dest="save_step_checkpoints")
    parser.set_defaults(save_step_checkpoints=True)
    parser.add_argument("--log_every_steps", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnostic_log_samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic_freq", type=int, default=10_000)
    parser.add_argument("--diagnostic_samples_per_log", type=int, default=128)
    parser.add_argument("--diagnostic_full_batch_metrics", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--M_B", type=int, default=16)
    parser.add_argument("--M_Q_min", type=int, default=4)
    parser.add_argument("--N_local_min", type=int, default=4)
    parser.add_argument("--local_search_max", type=int, default=1024)
    parser.add_argument("--action_bins_per_dim", type=int, default=5)
    parser.add_argument("--policy_support_mode", type=str, default="bin", choices=["bin", "radius"])
    parser.add_argument("--policy_action_radius", type=float, default=0.35)
    parser.add_argument(
        "--support_scan_mode",
        type=str,
        default="deterministic_tail_v1",
        choices=["deterministic_tail_v1", "deterministic_head_v1", "deterministic_stride_v1"],
    )
    parser.add_argument("--support_cache_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support_cache_max_entries", type=int, default=200_000)
    parser.add_argument("--local_scan_cache_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_scan_cache_max_entries", type=int, default=50_000)
    parser.add_argument("--support_short_circuit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--numba_support_scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support_predicate_early_exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support_diagnostic_audit_interval", type=int, default=100)
    parser.add_argument("--verify_runtime_optimization", action="store_true")

    parser.add_argument("--d_reach", type=float, default=0.08)
    parser.add_argument("--d_goal_near", type=float, default=0.05)
    parser.add_argument("--z_lift_min", type=float, default=0.48)
    parser.add_argument("--grip_close_thresh", type=float, default=0.04)
    parser.add_argument("--N_gap_min", type=int, default=128)
    parser.add_argument("--gap_window", type=int, default=50_000)
    parser.add_argument("--lambda_P", type=float, default=1.0)
    parser.add_argument("--lambda_support", type=float, default=1.0)
    parser.add_argument("--U_P_max", type=float, default=1.0)
    parser.add_argument("--lambda_TD", type=float, default=1.0)
    parser.add_argument("--U_TD_max", type=float, default=1.0)
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--eps_PER", type=float, default=1e-6)
    parser.add_argument("--U_PER_max", type=float, default=2.0)
    parser.add_argument("--weighting_mode", type=str, default="batch_norm", choices=["batch_norm", "raw"])
    parser.add_argument("--weight_scope", type=str, default="actor_critic", choices=["actor_critic", "critic_only", "actor_only", "none"])
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
        hgr_priority_eps=args.hgr_priority_eps,
        hgr_priority_init=args.hgr_priority_init,
        hgr_priority_clip_max=args.hgr_priority_clip_max,
        hgr_alpha_episode=args.hgr_alpha_episode,
        hgr_alpha_goal=args.hgr_alpha_goal,
        hgr_beta_episode_initial=args.hgr_beta_episode_initial,
        hgr_beta_goal_initial=args.hgr_beta_goal_initial,
        hgr_beta_episode_final=args.hgr_beta_episode_final,
        hgr_beta_goal_final=args.hgr_beta_goal_final,
        hgr_beta_anneal_updates=args.hgr_beta_anneal_updates,
        hgr_uniform_mix=args.hgr_uniform_mix,
        hgr_sac_priority_reduction=args.hgr_sac_priority_reduction,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        noise_std=args.noise_std,
        l2_action_penalty=args.l2_action_penalty,
        td3_policy_delay=args.td3_policy_delay,
        td3_target_policy_noise=args.td3_target_policy_noise,
        td3_target_noise_clip=args.td3_target_noise_clip,
        sac_alpha=args.sac_alpha,
        sac_auto_alpha=args.sac_auto_alpha,
        sac_alpha_lr=args.sac_alpha_lr,
        sac_target_entropy=args.sac_target_entropy,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_every_steps=args.checkpoint_every_steps,
        save_step_checkpoints=args.save_step_checkpoints,
        log_every_steps=args.log_every_steps,
        smoke=args.smoke,
        diagnostic_log_samples=args.diagnostic_log_samples,
        diagnostic_freq=args.diagnostic_freq,
        diagnostic_samples_per_log=args.diagnostic_samples_per_log,
        diagnostic_full_batch_metrics=args.diagnostic_full_batch_metrics,
        M_B=args.M_B,
        M_Q_min=args.M_Q_min,
        N_local_min=args.N_local_min,
        local_search_max=args.local_search_max,
        action_bins_per_dim=args.action_bins_per_dim,
        policy_support_mode=args.policy_support_mode,
        policy_action_radius=args.policy_action_radius,
        support_scan_mode=args.support_scan_mode,
        support_cache_enabled=args.support_cache_enabled,
        support_cache_max_entries=args.support_cache_max_entries,
        local_scan_cache_enabled=args.local_scan_cache_enabled,
        local_scan_cache_max_entries=args.local_scan_cache_max_entries,
        support_short_circuit=args.support_short_circuit,
        verify_runtime_optimization=args.verify_runtime_optimization,
        d_reach=args.d_reach,
        d_goal_near=args.d_goal_near,
        z_lift_min=args.z_lift_min,
        grip_close_thresh=args.grip_close_thresh,
        N_gap_min=args.N_gap_min,
        gap_window=args.gap_window,
        lambda_P=args.lambda_P,
        lambda_support=args.lambda_support,
        U_P_max=args.U_P_max,
        lambda_TD=args.lambda_TD,
        U_TD_max=args.U_TD_max,
        per_alpha=args.per_alpha,
        eps_PER=args.eps_PER,
        U_PER_max=args.U_PER_max,
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


def validate_fetch_pick_and_place_setup(config: LiteConfig) -> None:
    config.normalize()
    if not hasattr(config, "numba_support_scan"):
        config.numba_support_scan = True
    if not hasattr(config, "support_predicate_early_exit"):
        config.support_predicate_early_exit = True
    if not hasattr(config, "support_diagnostic_audit_interval"):
        config.support_diagnostic_audit_interval = 100
    env = make_env(config, config.seeds.env_seed, False)
    task = make_task_spec(config, env)
    rng = np.random.default_rng(config.seeds.numpy_seed)
    device = torch.device("cpu")
    obs = env.reset(seed=config.seeds.env_seed)
    checks = {
        "env_id": env.snapshot()["env_id"],
        "reset_finite": _finite_goal_obs(obs),
        "flat_state_is_concat_obs_goal": bool(
            np.allclose(obs.flat_state(), np.concatenate([obs.observation, obs.desired_goal], axis=0))
        ),
        "obs_dim": int(env.obs_dim),
        "goal_dim": int(env.goal_dim),
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "distance_threshold": float(env.distance_threshold),
        "phase_names": list(getattr(task, "phase_names", ())),
        "action_low_finite": bool(np.all(np.isfinite(env.action_low))),
        "action_high_finite": bool(np.all(np.isfinite(env.action_high))),
    }
    replay = HERReplayBuffer(
        capacity=max(256, 3 * int(env.max_episode_steps)),
        obs_dim=env.obs_dim,
        goal_dim=env.goal_dim,
        action_dim=env.action_dim,
    )
    hgr_sampler = HGRSampler(config, env, device)
    ep_id = 0
    ep_step = 0
    rollout_steps = 0
    for _ in range(max(64, 2 * int(env.max_episode_steps))):
        h_t = env.max_episode_steps - ep_step
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        obs_next, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        slot = replay.add_transition(
            observation=obs.observation,
            achieved_goal=obs.achieved_goal,
            desired_goal=obs.desired_goal,
            action_exec=action,
            reward_ext=float(reward),
            next_observation=obs_next.observation,
            next_achieved_goal=obs_next.achieved_goal,
            done=done,
            timeout=bool(truncated),
            h=h_t,
            episode_id=ep_id,
            episode_step=ep_step,
            coarse_key=task.coarse_hash(obs.observation, h_t),
            action_bin=int(task.action_bin_batch(action[None, :], config.action_bins_per_dim)[0]),
        )
        hgr_sampler.on_replay_add(
            slot,
            ep_id,
            ep_step,
            replay,
            episode_done=done,
        )
        obs = obs_next
        ep_step += 1
        rollout_steps += 1
        if done:
            ep_id += 1
            ep_step = 0
            obs = env.reset(seed=config.seeds.env_seed + ep_id)
            if hgr_sampler.pair_table_stats()[0] > 0:
                break

    batch_size = min(8, replay.size)
    batch = replay.sample(batch_size, her_ratio=1.0, rng=rng)
    original_goal = replay.desired_goal[batch.indices]
    agent = make_agent(
        config=config,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
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
    b4_wb = LiteWeightComputer(replace(config, method="b4_lite"), task, gap_stats).compute(
        batch, replay, agent, rng, device
    )
    hgr_batch = hgr_sampler.sample_batch(
        replay,
        batch_size=batch_size,
        her_ratio=1.0,
        rng=rng,
    )
    future_ok = True
    if hgr_batch.hgr_goal_slot is not None:
        for replay_slot, goal_slot, relabel_flag, desired_goal in zip(
            hgr_batch.indices,
            hgr_batch.hgr_goal_slot,
            hgr_batch.her_relabel,
            hgr_batch.desired_goal_train,
        ):
            if float(relabel_flag) <= 0.5:
                continue
            replay_slot_i = int(replay_slot)
            goal_slot_i = int(goal_slot)
            future_ok = future_ok and (
                int(replay.episode_id[replay_slot_i]) == int(replay.episode_id[goal_slot_i])
                and int(replay.episode_step[replay_slot_i]) <= int(replay.episode_step[goal_slot_i])
                and np.allclose(desired_goal, replay.next_achieved_goal[goal_slot_i])
            )
    hgr_critic_weight, hgr_actor_weight = hgr_sampler.loss_weights(hgr_batch)
    table_count, pair_count, _memory_mb = hgr_sampler.pair_table_stats()
    hgr_reward = env.compute_reward(hgr_batch.next_achieved_goal, hgr_batch.desired_goal_train, {})
    hgr_state = torch.as_tensor(
        np.concatenate(
            [hgr_batch.observation, hgr_batch.desired_goal_train], axis=1
        ),
        dtype=torch.float32,
        device=device,
    )
    hgr_next_state = torch.as_tensor(
        np.concatenate(
            [hgr_batch.next_observation, hgr_batch.desired_goal_train], axis=1
        ),
        dtype=torch.float32,
        device=device,
    )
    critic_updates_before = hgr_sampler.critic_updates
    sac_update = agent.update(
        state=hgr_state,
        action_exec=torch.as_tensor(
            hgr_batch.action_exec, dtype=torch.float32, device=device
        ),
        reward=torch.as_tensor(hgr_reward, dtype=torch.float32, device=device),
        next_state=hgr_next_state,
        nonterminal=torch.ones(batch_size, dtype=torch.float32, device=device),
        omega_critic=torch.as_tensor(
            hgr_critic_weight, dtype=torch.float32, device=device
        ),
        omega_actor=torch.as_tensor(
            hgr_actor_weight, dtype=torch.float32, device=device
        ),
    )
    if sac_update.td_error_abs_values is None:
        raise RuntimeError("SAC HGR validation did not expose priority values")
    hgr_sampler.update_priorities(
        hgr_batch, sac_update.td_error_abs_values
    )
    checks.update(
        {
            "random_rollout_steps": int(rollout_steps),
            "standalone_agent_is_sac": isinstance(agent, SACAgent),
            "sac_twin_critics_present": bool(
                hasattr(agent, "critic1") and hasattr(agent, "critic2")
            ),
            "sac_priority_reduction": getattr(
                agent, "td_priority_reduction", None
            ),
            "sac_priority_vector_shape": list(
                sac_update.td_error_abs_values.shape
            ),
            "sac_priority_vector_finite_nonnegative": bool(
                np.all(np.isfinite(sac_update.td_error_abs_values))
                and np.all(sac_update.td_error_abs_values >= 0.0)
            ),
            "hgr_priority_update_incremented": bool(
                hgr_sampler.critic_updates == critic_updates_before + 1
            ),
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
            "hgr_batch_shape": list(hgr_batch.observation.shape),
            "hgr_relabel_ratio": float(hgr_batch.her_relabel_ratio),
            "hgr_future_goal_selected_from_future": bool(future_ok),
            "hgr_pair_table_count": int(table_count),
            "hgr_pair_priority_count": int(pair_count),
            "hgr_is_weight_finite_positive": bool(
                np.all(np.isfinite(hgr_critic_weight)) and np.all(hgr_critic_weight > 0.0)
            ),
            "hgr_is_weight_max_at_most_one": bool(np.max(hgr_critic_weight) <= 1.0 + 1e-6),
            "hgr_actor_weight_all_one": bool(np.allclose(hgr_actor_weight, 1.0)),
            "hgr_episode_probability_sum_one": bool(
                np.isclose(np.sum(hgr_sampler.episode_distribution()[1]), 1.0)
            ),
            "hgr_reward_shape": list(hgr_reward.shape),
            "hgr_reward_finite": bool(np.all(np.isfinite(hgr_reward))),
            "hgr_reward_min": float(np.min(hgr_reward)),
            "hgr_reward_max": float(np.max(hgr_reward)),
            "hgr_metrics_finite_sampling_time": bool(np.isfinite(hgr_sampler.latest_metrics["runtime/hgr_sampling_sec_per_update"])),
            "b4_omega_finite": bool(np.all(np.isfinite(b4_wb.omega_critic))),
            "b4_g_sup_finite": bool(np.all(np.isfinite(b4_wb.g_sup))),
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


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.numba_support_scan = bool(args.numba_support_scan)
    cfg.support_predicate_early_exit = bool(args.support_predicate_early_exit)
    cfg.support_diagnostic_audit_interval = int(args.support_diagnostic_audit_interval)
    repo_root = Path(__file__).resolve().parents[2]
    if args.validate_only:
        validate_fetch_pick_and_place_setup(cfg)
        return 0
    run_dir = run_training(cfg, repo_root=repo_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
