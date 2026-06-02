from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .agent import DDPGAgent
from .config import LiteConfig
from .envs import FetchGoalEnv
from .io_utils import (
    MetricsCSVWriter,
    MetricsJSONLWriter,
    env_thread_hint,
    make_run_dir,
    select_device,
    set_all_seeds,
    setup_logger,
    write_json,
)
from .replay import HERReplayBuffer
from .task_fetch_push import FetchPushTaskSpec
from .weights import GapStats, LiteWeightComputer


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pt"


def _best_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "best_checkpoint.pt"


def _step_checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"checkpoint_step_{int(step)}.pt"


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def make_env(config: LiteConfig, seed: int, render: bool) -> Any:
    if config.task_name != "fetch_push":
        raise ValueError(
            f"Unsupported task_name={config.task_name!r}; "
            "this FetchPush experiment snapshot only supports 'fetch_push'"
        )
    return FetchGoalEnv(config.env_id, config.max_episode_steps, seed, render)

def make_task_spec(config: LiteConfig, env: Any) -> Any:
    if config.task_name != "fetch_push":
        raise ValueError(
            f"Unsupported task_name={config.task_name!r}; "
            "this FetchPush experiment snapshot only supports 'fetch_push'"
        )
    return FetchPushTaskSpec(
        action_low=env.action_low,
        action_high=env.action_high,
        epsilon_g=env.distance_threshold,
        d_contact=config.d_contact,
        d_goal_near=config.d_goal_near,
        ee_obj_bins=config.ee_obj_bins,
        obj_goal_bins=config.obj_goal_bins,
        horizon_bins=config.horizon_bins,
    )

@torch.no_grad()
def evaluate_agent(agent: DDPGAgent, env: Any, eval_seed: int, eval_episodes: int) -> Dict[str, float]:
    returns = []
    goal_successes = []
    env_successes = []
    dummy_rng = np.random.default_rng(0)
    for ep in range(eval_episodes):
        obs = env.reset(seed=eval_seed + ep)
        ep_ret = 0.0
        ep_goal_succ = 0.0
        ep_env_succ = 0.0
        for _ in range(env.max_episode_steps):
            action = agent.act(obs.flat_state(), noise_std=0.0, noise_rng=dummy_rng)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            goal_succ = float(info.get("goal_success", info.get("is_success", 0.0)))
            env_succ = float(info.get("env_success", info.get("is_success", goal_succ)))
            ep_goal_succ = max(ep_goal_succ, goal_succ)
            ep_env_succ = max(ep_env_succ, env_succ)
            if terminated or truncated:
                break
        returns.append(ep_ret)
        goal_successes.append(ep_goal_succ)
        env_successes.append(ep_env_succ)
    env_mean = float(np.mean(env_successes) if env_successes else 0.0)
    goal_mean = float(np.mean(goal_successes) if goal_successes else 0.0)
    return {
        "eval_return_mean": float(np.mean(returns) if returns else 0.0),
        "eval_return_std": float(np.std(returns) if returns else 0.0),
        "eval_success_mean": env_mean,
        "eval_success_std": float(np.std(env_successes) if env_successes else 0.0),
        "eval_goal_success_mean": goal_mean,
        "eval_goal_success_std": float(np.std(goal_successes) if goal_successes else 0.0),
        "eval_env_success_mean": env_mean,
        "eval_env_success_std": float(np.std(env_successes) if env_successes else 0.0),
    }


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size else 0.0


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _entropy_weight(w: np.ndarray, eps: float) -> float:
    if w.size == 0:
        return 0.0
    p = np.maximum(w.astype(np.float64), eps)
    p = p / max(eps, float(p.sum()))
    return float(-(p * np.log(p + eps)).sum())


def _effective_sample_size(w: np.ndarray, eps: float = 1e-8) -> float:
    if w.size == 0:
        return 0.0
    w64 = np.maximum(w.astype(np.float64), 0.0)
    denom = float(np.sum(w64 * w64))
    if denom <= eps:
        return 0.0
    return float((float(np.sum(w64)) ** 2) / denom)


def _phase_td_error_metrics(
    phases: np.ndarray,
    td_error_abs: np.ndarray,
    phase_names: tuple[str, ...],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if phases.size == 0 or td_error_abs.size == 0:
        return {f"phase/{name}/td_error_abs_mean": 0.0 for name in phase_names}
    for phase_id, phase_name in enumerate(phase_names):
        mask = phases == phase_id
        metrics[f"phase/{phase_name}/td_error_abs_mean"] = (
            float(np.mean(td_error_abs[mask])) if np.any(mask) else 0.0
        )
    return metrics


def _lite_with_runtime(lite: Dict[str, float], step: int, start_time: float) -> Dict[str, float]:
    runtime_sec = max(0.0, time.perf_counter() - start_time)
    out = dict(lite)
    out["runtime/total_sec"] = runtime_sec
    out["runtime/steps_per_sec"] = float(step / max(runtime_sec, 1e-8))
    return out


def _eval_curve_summary(eval_history: list[Dict[str, float]], total_steps: int) -> Dict[str, Any]:
    if not eval_history:
        return {
            "final_success": None,
            "best_success": None,
            "best_step": None,
            "auc_eval_mean": None,
            "auc_trapz_0_to_total": None,
            "first_hit_0.25": None,
            "first_hit_0.50": None,
            "first_hit_0.75": None,
            "first_hit_0.90": None,
            "first_hit_1.00": None,
            "drop_from_best_to_final": None,
            "last_quarter_auc": None,
            "final_window_success_mean": None,
            "final_window_success_std": None,
            "final_window_n": 0,
        }
    ordered = sorted(eval_history, key=lambda item: int(item["step"]))
    eval_steps = np.asarray([int(item["step"]) for item in ordered], dtype=np.float64)
    successes = np.asarray([float(item["eval_goal_success_mean"]) for item in ordered], dtype=np.float64)
    final_success = float(successes[-1])
    best_idx = int(np.argmax(successes))
    best_success = float(successes[best_idx])
    best_step = int(eval_steps[best_idx])

    auc_steps = np.concatenate([np.asarray([0.0]), eval_steps])
    auc_values = np.concatenate([np.asarray([0.0]), successes])
    if auc_steps[-1] < float(total_steps):
        auc_steps = np.concatenate([auc_steps, np.asarray([float(total_steps)])])
        auc_values = np.concatenate([auc_values, np.asarray([auc_values[-1]])])
    auc = float(np.trapz(auc_values, auc_steps) / max(1.0, float(total_steps)))

    out: Dict[str, Any] = {
        "final_success": final_success,
        "best_success": best_success,
        "best_step": best_step,
        "auc_eval_mean": auc,
        "auc_trapz_0_to_total": auc,
        "drop_from_best_to_final": float(best_success - final_success),
    }
    for threshold in (0.25, 0.50, 0.75, 0.90, 1.00):
        hit_steps = eval_steps[successes >= threshold]
        out[f"first_hit_{threshold:.2f}"] = int(hit_steps[0]) if hit_steps.size else None

    quarter_start = 0.75 * float(total_steps)
    q_steps = eval_steps[eval_steps >= quarter_start]
    q_success = successes[eval_steps >= quarter_start]
    if q_steps.size >= 2:
        out["last_quarter_auc"] = float(np.trapz(q_success, q_steps) / max(1.0, float(total_steps) - quarter_start))
    elif q_success.size == 1:
        out["last_quarter_auc"] = float(q_success[0])
    else:
        out["last_quarter_auc"] = None

    final_window = successes[-min(5, successes.size) :]
    out["final_window_success_mean"] = float(np.mean(final_window))
    out["final_window_success_std"] = float(np.std(final_window))
    out["final_window_n"] = int(final_window.size)
    return out


def _task_phase_names(task: Any) -> tuple[str, ...]:
    names = getattr(task, "phase_names", None)
    if callable(names):
        names = names()
    if names is None:
        metadata = task.metadata() if hasattr(task, "metadata") else {}
        labels = metadata.get("phase_labels", {}) if isinstance(metadata, dict) else {}
        if labels:
            names = [name for _, name in sorted(labels.items(), key=lambda item: int(item[0]))]
    if names is None:
        names = ("approach", "grasp", "transport", "release_place")
    return tuple(str(name) for name in names)


def run_training(config: LiteConfig, repo_root: Path) -> Path:
    config.normalize()
    output_root = config.output_root_path(repo_root)
    run_dir = make_run_dir(output_root, config.run_name, config.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")

    set_all_seeds(config.seeds.torch_seed, config.seeds.numpy_seed)
    device = select_device(config.device)
    logger.info("device=%s", device)

    train_env = make_env(config, config.seeds.env_seed, config.render)
    eval_env = make_env(config, config.seeds.eval_seed, False)
    if config.policy_support_mode == "auto":
        config.policy_support_mode = "bin" if int(train_env.action_dim) <= 4 else "radius"
        logger.info(
            "resolved policy_support_mode=%s for action_dim=%d",
            config.policy_support_mode,
            train_env.action_dim,
        )
    task = make_task_spec(config, train_env)
    phase_names = _task_phase_names(task)
    agent = DDPGAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        hidden_dim=config.hidden_dim,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_low=train_env.action_low,
        action_high=train_env.action_high,
        device=device,
        l2_action_penalty=config.l2_action_penalty,
        policy_delay=config.policy_delay,
        target_policy_noise=config.target_policy_noise,
        target_noise_clip=config.target_noise_clip,
    )
    replay = HERReplayBuffer(
        capacity=config.buffer_capacity,
        obs_dim=train_env.obs_dim,
        goal_dim=train_env.goal_dim,
        action_dim=train_env.action_dim,
    )
    gap_stats = GapStats(
        maxlen=config.gap_window,
        n_min=config.N_gap_min,
        q_low=config.gap_q_low,
        q_high=config.gap_q_high,
        eps_g=config.eps_G,
    )
    weight_computer = LiteWeightComputer(config, task, gap_stats)

    write_json(
        run_dir / "config.json",
        {
            **config.as_dict(),
            "thread_hint": env_thread_hint(),
            "terminal_convention": {
                "success_terminates": config.success_terminates,
                "timeout_bootstrap": config.timeout_bootstrap,
            },
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
        "loss/td_error_abs_mean",
        "loss/td_error_abs_p90",
        "loss/critic_grad_norm",
        "loss/actor_grad_norm",
        "value/q_mean",
        "value/target_mean",
        "critic/q_policy_mean",
        "critic/q_policy_std",
        "critic/q_target_mean",
        "critic/q_target_std",
        "critic/q_abs_mean",
        "critic/q_abs_max",
        "lite/mean_G_sup",
        "lite/positive_G_sup_rate",
        "lite/mean_kappa",
        "lite/mean_w_P",
        "lite/max_w_P",
        "lite/std_w_P",
        "lite/mean_w_norm",
        "lite/entropy_w_P",
        "lite/ESS_w",
        "lite/ESS_w_normalized",
        "lite/mean_w_P_reference",
        "lite/reference_high_weight_rate",
        "lite/assigned_high_weight_rate",
        "lite/support_weight_overlap_rate",
        "lite/assigned_high_on_support_rate",
        "lite/support_weight_alignment_rate",
        "lite/support_available_rate",
        "lite/psi_q_rank_rate",
        "lite/q_candidate_fallback_rate",
        "lite/psi_A_policy_rate",
        "lite/local_support_count_mean",
        "lite/local_support_count_p50",
        "lite/local_support_count_p90",
        "lite/mean_len_A_Q_sup",
        "lite/mean_len_A_Q_replay_local",
        "lite/mean_num_coarse_candidates",
        "lite/mean_num_scanned_candidates",
        "lite/local_scan_truncated_rate",
        "lite/mean_Q_imp_minus_Q_policy",
        "lite/unrestricted_gap_mean",
        "lite/gap_ready",
        "lite/gap_tau",
        "lite/gap_max",
        "time/local_context_lookup",
        "time/candidate_construction",
        "time/local_context",
        "time/q_policy_eval",
        "time/q_candidate_eval",
        "time/weight_compute",
        "time/actor_critic_update",
        "time_local_context_lookup",
        "time_candidate_construction",
        "time_q_policy_eval",
        "time_q_candidate_eval",
        "time_weight_computation",
        "time_actor_critic_update",
        "runtime/steps_per_sec",
        "runtime/total_sec",
        "runtime/exact_negative_short_circuit_rate",
        "runtime/full_context_scan_rate",
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
        "runtime/replay_index_direct_scan_count",
        "runtime/replay_index_candidates_materialized_count",
        "runtime/replay_index_candidates_materialized_rate",
        "runtime/replay_index_live_count_lookup_rate",
        "runtime/action_bin_metadata_query_count",
        "runtime/action_bin_unique_compute_count",
        "runtime/action_bin_unique_compute_rate",
        "runtime/action_bin_metadata_size",
        "runtime/numba_support_scan_enabled",
        "runtime/numba_support_scan_fallback_rate",
        "runtime/numba_support_queries",
        "runtime/numba_scanner_warmed",
        "time/numba_support_scan",
        "lite/support_scan_mode_is_sticky_random",
        "lite/support_scan_seed",
        "td_error/abs_mean",
        "td_error/abs_p90",
        "td_error/weight_mean",
        "td_error/weight_max",
        "td_error/weight_entropy",
        "td_error/weight_ESS",
        "per/priority_entropy",
        "per/priority_ESS",
        "per/IS_weight_mean",
        "per/IS_weight_max",
        "per/replay_sample_age_mean",
        "per/replay_sample_age_p90",
        "per/sample_duplicate_rate",
        "lite/slide_mean_obj_speed",
        "lite/slide_mean_vel_align",
        "lite/slide_phase_distribution_approach",
        "lite/slide_phase_distribution_strike",
        "lite/slide_phase_distribution_slide",
        "lite/slide_phase_distribution_near_goal",
        "eval/return_mean",
        "eval/success_mean",
        "eval/goal_success_mean",
        "eval/env_success_mean",
    ]
    for phase_name in phase_names:
        fields.extend(
            [
                f"lite/phase_{phase_name}_mean_w_P",
                f"lite/phase_{phase_name}_fallback_rate",
                f"lite/phase_{phase_name}_psi_q_rank_rate",
                f"lite/phase_{phase_name}_positive_G_sup_rate",
                f"phase/{phase_name}/count_rate",
                f"phase/{phase_name}/mean_w_P",
                f"phase/{phase_name}/support_rate",
                f"phase/{phase_name}/fallback_rate",
                f"phase/{phase_name}/local_scan_truncated_rate",
                f"phase/{phase_name}/td_error_abs_mean",
            ]
        )
    csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", fields)
    jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")

    noise_rng = np.random.default_rng(config.seeds.action_noise_seed)
    sample_rng = np.random.default_rng(config.seeds.numpy_seed + 17)
    weight_rng = np.random.default_rng(config.seeds.weight_seed)
    start_step = 0
    episode_idx = 0
    total_updates = 0
    best_eval_success = -1.0
    best_eval_return = float("-inf")
    best_eval_step = -1
    eval_history: list[Dict[str, float]] = []
    wall_clock_start = time.perf_counter()

    ckpt_path = Path(config.resume_path) if config.resume_path else _checkpoint_path(run_dir)
    if config.resume and ckpt_path.exists():
        payload = load_checkpoint(ckpt_path)
        agent.load_state_dict(payload["agent"])
        replay.load_state_dict(payload["replay"])
        if replay.indexed_slot_count() < replay.size:
            replay.rebuild_action_index(
                task.coarse_hash_batch(replay.observation[:replay.size], replay.h[:replay.size]),
                task.action_bin_batch(replay.action_exec[:replay.size], config.action_bins_per_dim),
            )
            logger.info("rebuilt replay action index slots=%d", replay.indexed_slot_count())
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

    obs = train_env.reset(seed=config.seeds.env_seed + episode_idx)
    ep_return = 0.0
    ep_success = 0.0
    ep_step = 0
    global_step = start_step
    last_loss = {
        "critic": 0.0,
        "actor": 0.0,
        "q_mean": 0.0,
        "target_mean": 0.0,
        "td_error_abs_mean": 0.0,
        "td_error_abs_p90": 0.0,
        "critic_grad_norm": 0.0,
        "actor_grad_norm": 0.0,
        "q_policy_mean": 0.0,
        "q_policy_std": 0.0,
        "q_target_mean": 0.0,
        "q_target_std": 0.0,
        "q_abs_mean": 0.0,
        "q_abs_max": 0.0,
    }
    last_lite = _empty_lite_metrics(phase_names)
    last_her_ratio = 0.0

    def checkpoint_payload(step_value: int, kind: str) -> Dict[str, Any]:
        return {
            "checkpoint_kind": kind,
            "config": config.as_dict(),
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

    while global_step < config.total_steps:
        h_t = train_env.max_episode_steps - ep_step
        if global_step < config.start_steps:
            action = noise_rng.uniform(train_env.action_low, train_env.action_high).astype(np.float32)
        else:
            action = agent.act(obs.flat_state(), noise_std=config.noise_std, noise_rng=noise_rng)

        obs_next, reward_ext, terminated, truncated, info = train_env.step(action)
        done = bool(terminated or truncated)
        timeout = bool(truncated)
        ep_return += float(reward_ext)
        ep_success = max(ep_success, float(info.get("goal_success", info.get("is_success", 0.0))))

        replay.add_transition(
            observation=obs.observation,
            achieved_goal=obs.achieved_goal,
            desired_goal=obs.desired_goal,
            action_exec=action,
            reward_ext=reward_ext,
            next_observation=obs_next.observation,
            next_achieved_goal=obs_next.achieved_goal,
            done=done,
            timeout=timeout,
            h=h_t,
            episode_id=episode_idx,
            episode_step=ep_step,
            coarse_key=task.coarse_hash(obs.observation, h_t),
            action_bin=int(task.action_bin_batch(action[None, :], config.action_bins_per_dim)[0]),
        )

        obs = obs_next
        ep_step += 1
        global_step += 1

        if replay.size >= config.batch_size and global_step >= config.warmup_steps:
            for _ in range(config.updates_per_step):
                batch = replay.sample(config.batch_size, config.her_ratio, sample_rng)
                last_her_ratio = batch.her_relabel_ratio
                reward_train = train_env.compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {})
                success_train = train_env.success_from_goal(batch.next_achieved_goal, batch.desired_goal_train)
                env_terminal = batch.done.astype(np.float32) * (1.0 - batch.timeout.astype(np.float32))
                if not config.timeout_bootstrap:
                    env_terminal = np.maximum(env_terminal, batch.timeout.astype(np.float32))
                terminal = env_terminal.copy()
                if config.success_terminates:
                    terminal = np.maximum(terminal, success_train.astype(np.float32))
                nonterminal = (1.0 - terminal).astype(np.float32)

                state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
                next_state_np = np.concatenate([batch.next_observation, batch.desired_goal_train], axis=1).astype(np.float32)
                wb = weight_computer.compute(
                    batch,
                    replay,
                    agent,
                    weight_rng,
                    device,
                    reward_train=reward_train,
                    nonterminal=nonterminal,
                    state_np=state_np,
                    next_state_np=next_state_np,
                )

                t_update0 = time.perf_counter()
                update = agent.update(
                    state=torch.as_tensor(state_np, dtype=torch.float32, device=device),
                    action_exec=torch.as_tensor(batch.action_exec, dtype=torch.float32, device=device),
                    reward=torch.as_tensor(reward_train, dtype=torch.float32, device=device),
                    next_state=torch.as_tensor(next_state_np, dtype=torch.float32, device=device),
                    nonterminal=torch.as_tensor(nonterminal, dtype=torch.float32, device=device),
                    omega_critic=torch.as_tensor(wb.omega_critic, dtype=torch.float32, device=device),
                    omega_actor=torch.as_tensor(wb.omega_actor, dtype=torch.float32, device=device),
                )
                t_update1 = time.perf_counter()
                last_loss = {
                    "critic": update.critic_loss,
                    "actor": update.actor_loss,
                    "q_mean": update.q_mean,
                    "target_mean": update.target_mean,
                    "td_error_abs_mean": update.td_error_abs_mean,
                    "td_error_abs_p90": update.td_error_abs_p90,
                    "critic_grad_norm": update.critic_grad_norm,
                    "actor_grad_norm": update.actor_grad_norm,
                    "q_policy_mean": update.q_policy_mean,
                    "q_policy_std": update.q_policy_std,
                    "q_target_mean": update.q_target_mean,
                    "q_target_std": update.q_target_std,
                    "q_abs_mean": update.q_abs_mean,
                    "q_abs_max": update.q_abs_max,
                }
                last_lite = _lite_metrics_from_weight_batch(wb, phase_names)
                last_lite.update(_phase_td_error_metrics(wb.phase, update.td_error_abs_values, phase_names))
                last_lite["time/actor_critic_update"] = float(t_update1 - t_update0)
                last_lite["time_actor_critic_update"] = float(t_update1 - t_update0)
                total_updates += 1

        if done:
            episode_idx += 1
            row = _metric_row(
                global_step,
                episode_idx,
                ep_return,
                ep_success,
                ep_step,
                replay.size,
                last_her_ratio,
                last_loss,
                _lite_with_runtime(last_lite, global_step, wall_clock_start),
            )
            csv_writer.write(row)
            jsonl_writer.write({"type": "episode", **row})
            obs = train_env.reset(seed=config.seeds.env_seed + episode_idx)
            ep_return = 0.0
            ep_success = 0.0
            ep_step = 0

        do_eval = (global_step % config.eval_freq == 0) or (global_step == config.total_steps)
        if do_eval and global_step > 0:
            eval_metrics = evaluate_agent(
                agent,
                eval_env,
                eval_seed=config.seeds.eval_seed + global_step * 10,
                eval_episodes=config.eval_episodes,
            )
            eval_history.append({"step": float(global_step), **eval_metrics})
            row = _metric_row(
                global_step,
                episode_idx,
                ep_return,
                ep_success,
                ep_step,
                replay.size,
                last_her_ratio,
                last_loss,
                _lite_with_runtime(last_lite, global_step, wall_clock_start),
            )
            row["eval/return_mean"] = eval_metrics["eval_return_mean"]
            row["eval/success_mean"] = eval_metrics["eval_success_mean"]
            row["eval/goal_success_mean"] = eval_metrics["eval_goal_success_mean"]
            row["eval/env_success_mean"] = eval_metrics["eval_env_success_mean"]
            csv_writer.write(row)
            jsonl_writer.write({"type": "eval", **row, **eval_metrics})
            is_best = (eval_metrics["eval_goal_success_mean"] > best_eval_success) or (
                np.isclose(eval_metrics["eval_goal_success_mean"], best_eval_success)
                and eval_metrics["eval_return_mean"] > best_eval_return
            )
            if is_best:
                best_eval_success = eval_metrics["eval_goal_success_mean"]
                best_eval_return = eval_metrics["eval_return_mean"]
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
                "step=%d episode=%d replay=%d eval_goal=%.3f eval_env=%.3f eval_ret=%.2f wP=%.3f psiQ=%.3f fallback=%.3f",
                global_step,
                episode_idx,
                replay.size,
                eval_metrics["eval_goal_success_mean"],
                eval_metrics["eval_env_success_mean"],
                eval_metrics["eval_return_mean"],
                last_lite["lite/mean_w_P"],
                last_lite["lite/psi_q_rank_rate"],
                last_lite["lite/q_candidate_fallback_rate"],
            )

        do_ckpt = (global_step % config.checkpoint_every_steps == 0 and global_step > 0) or (
            global_step == config.total_steps
        )
        if do_ckpt:
            payload = checkpoint_payload(global_step, "periodic")
            save_checkpoint(_checkpoint_path(run_dir), payload)
            if config.save_step_checkpoints:
                save_checkpoint(_step_checkpoint_path(run_dir, global_step), payload)

        if global_step % config.log_every_steps == 0 and global_step > 0:
            logger.info(
                "step=%d replay=%d updates=%d loss_q=%.4f loss_pi=%.4f wP=%.3f psiQ=%.3f fallback=%.3f",
                global_step,
                replay.size,
                total_updates,
                last_loss["critic"],
                last_loss["actor"],
                last_lite["lite/mean_w_P"],
                last_lite["lite/psi_q_rank_rate"],
                last_lite["lite/q_candidate_fallback_rate"],
            )

    runtime_total_sec = max(0.0, time.perf_counter() - wall_clock_start)
    final_lite_metrics = _lite_with_runtime(last_lite, global_step, wall_clock_start)
    eval_summary = _eval_curve_summary(eval_history, config.total_steps)
    summary = {
        "run_dir": str(run_dir),
        "method": config.method,
        "total_steps": config.total_steps,
        "episodes": episode_idx,
        "total_updates": total_updates,
        "best_eval_success": best_eval_success,
        "best_eval_return": best_eval_return,
        "best_eval_step": best_eval_step,
        "final_success": eval_summary["final_success"],
        "best_success": eval_summary["best_success"],
        "best_step": eval_summary["best_step"],
        "auc_eval_mean": eval_summary["auc_eval_mean"],
        "auc_trapz_0_to_total": eval_summary["auc_trapz_0_to_total"],
        "first_hit_0.25": eval_summary["first_hit_0.25"],
        "first_hit_0.50": eval_summary["first_hit_0.50"],
        "first_hit_0.75": eval_summary["first_hit_0.75"],
        "first_hit_0.90": eval_summary["first_hit_0.90"],
        "first_hit_1.00": eval_summary["first_hit_1.00"],
        "drop_from_best_to_final": eval_summary["drop_from_best_to_final"],
        "last_quarter_auc": eval_summary["last_quarter_auc"],
        "final_window_success_mean": eval_summary["final_window_success_mean"],
        "final_window_success_std": eval_summary["final_window_success_std"],
        "final_window_n": eval_summary["final_window_n"],
        "runtime_total_sec": runtime_total_sec,
        "steps_per_sec": float(global_step / max(runtime_total_sec, 1e-8)),
        "final_replay_size": replay.size,
        "final_losses": last_loss,
        "final_lite_metrics": final_lite_metrics,
    }
    write_json(run_dir / "summary.json", summary)
    csv_writer.close()
    jsonl_writer.close()
    train_env.close()
    eval_env.close()
    logger.info("finished run at %s", run_dir)
    return run_dir


def _empty_lite_metrics(phase_names: tuple[str, ...] = ("approach", "grasp", "transport", "release_place")) -> Dict[str, float]:
    metrics = {
        "lite/mean_G_sup": 0.0,
        "lite/positive_G_sup_rate": 0.0,
        "lite/mean_kappa": 0.0,
        "lite/mean_w_P": 1.0,
        "lite/max_w_P": 1.0,
        "lite/std_w_P": 0.0,
        "lite/mean_w_norm": 1.0,
        "lite/entropy_w_P": 0.0,
        "lite/ESS_w": 1.0,
        "lite/ESS_w_normalized": 1.0,
        "lite/mean_w_P_reference": 1.0,
        "lite/reference_high_weight_rate": 0.0,
        "lite/assigned_high_weight_rate": 0.0,
        "lite/support_weight_overlap_rate": 0.0,
        "lite/assigned_high_on_support_rate": 0.0,
        "lite/support_weight_alignment_rate": 0.0,
        "lite/support_available_rate": 0.0,
        "lite/psi_q_rank_rate": 0.0,
        "lite/q_candidate_fallback_rate": 1.0,
        "lite/psi_A_policy_rate": 0.0,
        "lite/local_support_count_mean": 0.0,
        "lite/local_support_count_p50": 0.0,
        "lite/local_support_count_p90": 0.0,
        "lite/mean_len_A_Q_sup": 0.0,
        "lite/mean_len_A_Q_replay_local": 0.0,
        "lite/mean_num_coarse_candidates": 0.0,
        "lite/mean_num_scanned_candidates": 0.0,
        "lite/local_scan_truncated_rate": 0.0,
        "lite/mean_Q_imp_minus_Q_policy": 0.0,
        "lite/unrestricted_gap_mean": 0.0,
        "lite/gap_ready": 0.0,
        "lite/gap_tau": 0.0,
        "lite/gap_max": 0.0,
        "time/local_context_lookup": 0.0,
        "time/candidate_construction": 0.0,
        "time/local_context": 0.0,
        "time/q_policy_eval": 0.0,
        "time/q_candidate_eval": 0.0,
        "time/weight_compute": 0.0,
        "time/actor_critic_update": 0.0,
        "time_local_context_lookup": 0.0,
        "time_candidate_construction": 0.0,
        "time_q_policy_eval": 0.0,
        "time_q_candidate_eval": 0.0,
        "time_weight_computation": 0.0,
        "time_actor_critic_update": 0.0,
        "runtime/steps_per_sec": 0.0,
        "runtime/total_sec": 0.0,
        "runtime/exact_negative_short_circuit_rate": 0.0,
        "runtime/full_context_scan_rate": 0.0,
        "runtime/support_cache_hit_rate": 0.0,
        "runtime/support_cache_miss_rate": 0.0,
        "runtime/support_cache_attempt_rate": 0.0,
        "runtime/support_context_cache_size": 0.0,
        "runtime/batch_query_unique_rate": 0.0,
        "runtime/batch_query_coalescing_saving_rate": 0.0,
        "runtime/local_scan_cache_hit_rate": 0.0,
        "runtime/local_scan_cache_miss_rate": 0.0,
        "runtime/local_scan_cache_attempt_rate": 0.0,
        "runtime/local_scan_cache_size": 0.0,
        "runtime/full_context_filter_count_mean": 0.0,
        "runtime/replay_index_direct_scan_count": 0.0,
        "runtime/replay_index_candidates_materialized_count": 0.0,
        "runtime/replay_index_candidates_materialized_rate": 0.0,
        "runtime/replay_index_live_count_lookup_rate": 0.0,
        "runtime/action_bin_metadata_query_count": 0.0,
        "runtime/action_bin_unique_compute_count": 0.0,
        "runtime/action_bin_unique_compute_rate": 0.0,
        "runtime/action_bin_metadata_size": 0.0,
        "runtime/numba_support_scan_enabled": 0.0,
        "runtime/numba_support_scan_fallback_rate": 0.0,
        "runtime/numba_support_queries": 0.0,
        "runtime/numba_scanner_warmed": 0.0,
        "time/numba_support_scan": 0.0,
        "lite/support_scan_mode_is_sticky_random": 0.0,
        "lite/support_scan_seed": 0.0,
        "td_error/abs_mean": 0.0,
        "td_error/abs_p90": 0.0,
        "td_error/weight_mean": 0.0,
        "td_error/weight_max": 0.0,
        "td_error/weight_entropy": 0.0,
        "td_error/weight_ESS": 0.0,
        "per/priority_entropy": 0.0,
        "per/priority_ESS": 0.0,
        "per/IS_weight_mean": 0.0,
        "per/IS_weight_max": 0.0,
        "per/replay_sample_age_mean": 0.0,
        "per/replay_sample_age_p90": 0.0,
        "per/sample_duplicate_rate": 0.0,
    }
    for phase_name in phase_names:
        metrics[f"lite/phase_{phase_name}_mean_w_P"] = 0.0
        metrics[f"lite/phase_{phase_name}_fallback_rate"] = 0.0
        metrics[f"lite/phase_{phase_name}_psi_q_rank_rate"] = 0.0
        metrics[f"lite/phase_{phase_name}_positive_G_sup_rate"] = 0.0
        metrics[f"phase/{phase_name}/count_rate"] = 0.0
        metrics[f"phase/{phase_name}/mean_w_P"] = 0.0
        metrics[f"phase/{phase_name}/support_rate"] = 0.0
        metrics[f"phase/{phase_name}/fallback_rate"] = 0.0
        metrics[f"phase/{phase_name}/local_scan_truncated_rate"] = 0.0
        metrics[f"phase/{phase_name}/td_error_abs_mean"] = 0.0
    metrics["lite/slide_mean_obj_speed"] = 0.0
    metrics["lite/slide_mean_vel_align"] = 0.0
    metrics["lite/slide_phase_distribution_approach"] = 0.0
    metrics["lite/slide_phase_distribution_strike"] = 0.0
    metrics["lite/slide_phase_distribution_slide"] = 0.0
    metrics["lite/slide_phase_distribution_near_goal"] = 0.0
    return metrics


def _lite_metrics_from_weight_batch(
    wb: Any,
    phase_names: tuple[str, ...] = ("approach", "grasp", "transport", "release_place"),
) -> Dict[str, float]:
    assigned_high_rate = _mean(wb.assigned_high_weight)
    overlap_rate = _mean(wb.support_weight_overlap)
    assigned_high_on_support = (
        float(np.sum(wb.support_weight_overlap) / max(1.0, float(np.sum(wb.assigned_high_weight))))
        if wb.assigned_high_weight.size
        else 0.0
    )
    ess_w = _effective_sample_size(wb.w_p_raw)
    metrics = {
        "lite/mean_G_sup": _mean(wb.g_sup),
        "lite/positive_G_sup_rate": float(np.mean(wb.g_sup > 0.0)),
        "lite/mean_kappa": _mean(wb.kappa),
        "lite/mean_w_P": _mean(wb.w_p_raw),
        "lite/max_w_P": float(np.max(wb.w_p_raw)) if wb.w_p_raw.size else 1.0,
        "lite/std_w_P": _std(wb.w_p_raw),
        "lite/mean_w_norm": _mean(wb.omega_critic),
        "lite/entropy_w_P": _entropy_weight(wb.w_p_raw, 1e-8),
        "lite/ESS_w": ess_w,
        "lite/ESS_w_normalized": float(ess_w / max(1, int(wb.w_p_raw.size))),
        "lite/mean_w_P_reference": _mean(wb.w_p_reference),
        "lite/reference_high_weight_rate": _mean(wb.reference_high_weight),
        "lite/assigned_high_weight_rate": assigned_high_rate,
        "lite/support_weight_overlap_rate": overlap_rate,
        "lite/assigned_high_on_support_rate": assigned_high_on_support,
        "lite/support_weight_alignment_rate": assigned_high_on_support,
        "lite/support_available_rate": float(np.mean(wb.len_a_q_local > 0.0)) if wb.len_a_q_local.size else 0.0,
        "lite/psi_q_rank_rate": _mean(wb.psi_q_rank),
        "lite/q_candidate_fallback_rate": _mean(wb.q_candidate_fallback),
        "lite/psi_A_policy_rate": _mean(wb.psi_a_policy),
        "lite/local_support_count_mean": _mean(wb.len_a_q_local),
        "lite/local_support_count_p50": _percentile(wb.len_a_q_local, 50),
        "lite/local_support_count_p90": _percentile(wb.len_a_q_local, 90),
        "lite/mean_len_A_Q_sup": _mean(wb.len_a_q_sup),
        "lite/mean_len_A_Q_replay_local": _mean(wb.len_a_q_local),
        "lite/mean_num_coarse_candidates": _mean(wb.num_coarse_candidates),
        "lite/mean_num_scanned_candidates": _mean(wb.num_scanned_candidates),
        "lite/local_scan_truncated_rate": _mean(wb.local_scan_truncated),
        "lite/mean_Q_imp_minus_Q_policy": _mean(wb.q_imp_minus_q_policy),
        "lite/unrestricted_gap_mean": _mean(wb.unrestricted_gap),
        "lite/gap_ready": float(wb.gap_ready),
        "lite/gap_tau": float(wb.gap_tau),
        "lite/gap_max": float(wb.gap_max),
        "time/local_context_lookup": float(wb.timing.get("local_context_lookup", 0.0)),
        "time/candidate_construction": float(wb.timing.get("candidate_construction", 0.0)),
        "time/local_context": float(wb.timing.get("local_context", 0.0)),
        "time/q_policy_eval": float(wb.timing.get("q_policy_eval", 0.0)),
        "time/q_candidate_eval": float(wb.timing.get("q_candidate_eval", 0.0)),
        "time/weight_compute": float(wb.timing.get("weight_compute", 0.0)),
        "time/actor_critic_update": 0.0,
        "time_local_context_lookup": float(wb.timing.get("local_context_lookup", 0.0)),
        "time_candidate_construction": float(wb.timing.get("candidate_construction", 0.0)),
        "time_q_policy_eval": float(wb.timing.get("q_policy_eval", 0.0)),
        "time_q_candidate_eval": float(wb.timing.get("q_candidate_eval", 0.0)),
        "time_weight_computation": float(wb.timing.get("weight_computation", wb.timing.get("weight_compute", 0.0))),
        "time_actor_critic_update": 0.0,
    }
    phases = getattr(wb, "phase", np.zeros_like(wb.w_p_raw, dtype=np.int64))
    for phase_id, phase_name in enumerate(phase_names):
        mask = phases == phase_id
        metrics[f"phase/{phase_name}/count_rate"] = float(np.mean(mask)) if mask.size else 0.0
        if np.any(mask):
            metrics[f"lite/phase_{phase_name}_mean_w_P"] = float(np.mean(wb.w_p_raw[mask]))
            metrics[f"lite/phase_{phase_name}_fallback_rate"] = float(np.mean(wb.q_candidate_fallback[mask]))
            metrics[f"lite/phase_{phase_name}_psi_q_rank_rate"] = float(np.mean(wb.psi_q_rank[mask]))
            metrics[f"lite/phase_{phase_name}_positive_G_sup_rate"] = float(np.mean(wb.g_sup[mask] > 0.0))
            metrics[f"phase/{phase_name}/mean_w_P"] = float(np.mean(wb.w_p_raw[mask]))
            metrics[f"phase/{phase_name}/support_rate"] = float(np.mean(wb.psi_q_rank[mask]))
            metrics[f"phase/{phase_name}/fallback_rate"] = float(np.mean(wb.q_candidate_fallback[mask]))
            metrics[f"phase/{phase_name}/local_scan_truncated_rate"] = float(np.mean(wb.local_scan_truncated[mask]))
            metrics[f"phase/{phase_name}/td_error_abs_mean"] = 0.0
        else:
            metrics[f"lite/phase_{phase_name}_mean_w_P"] = 0.0
            metrics[f"lite/phase_{phase_name}_fallback_rate"] = 0.0
            metrics[f"lite/phase_{phase_name}_psi_q_rank_rate"] = 0.0
            metrics[f"lite/phase_{phase_name}_positive_G_sup_rate"] = 0.0
            metrics[f"phase/{phase_name}/mean_w_P"] = 0.0
            metrics[f"phase/{phase_name}/support_rate"] = 0.0
            metrics[f"phase/{phase_name}/fallback_rate"] = 0.0
            metrics[f"phase/{phase_name}/local_scan_truncated_rate"] = 0.0
            metrics[f"phase/{phase_name}/td_error_abs_mean"] = 0.0
    for key, value in getattr(wb, "extra_metrics", {}).items():
        metrics[key] = float(value)
    return metrics


def _metric_row(
    step: int,
    episode: int,
    ep_return: float,
    ep_success: float,
    ep_len: int,
    replay_size: int,
    her_ratio: float,
    loss: Dict[str, float],
    lite: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "step": step,
        "episode": episode,
        "train/episode_return": ep_return,
        "train/episode_success": ep_success,
        "train/episode_len": ep_len,
        "train/replay_size": replay_size,
        "train/her_relabel_ratio": her_ratio,
        "loss/critic": loss["critic"],
        "loss/actor": loss["actor"],
        "loss/td_error_abs_mean": loss.get("td_error_abs_mean", 0.0),
        "loss/td_error_abs_p90": loss.get("td_error_abs_p90", 0.0),
        "loss/critic_grad_norm": loss.get("critic_grad_norm", 0.0),
        "loss/actor_grad_norm": loss.get("actor_grad_norm", 0.0),
        "value/q_mean": loss["q_mean"],
        "value/target_mean": loss["target_mean"],
        "critic/q_policy_mean": loss.get("q_policy_mean", 0.0),
        "critic/q_policy_std": loss.get("q_policy_std", 0.0),
        "critic/q_target_mean": loss.get("q_target_mean", 0.0),
        "critic/q_target_std": loss.get("q_target_std", 0.0),
        "critic/q_abs_mean": loss.get("q_abs_mean", 0.0),
        "critic/q_abs_max": loss.get("q_abs_max", 0.0),
        **lite,
        "eval/return_mean": "",
        "eval/success_mean": "",
        "eval/goal_success_mean": "",
        "eval/env_success_mean": "",
    }

