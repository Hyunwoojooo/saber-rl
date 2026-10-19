from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .agent import DDPGAgent, SACAgent, TD3Agent
from .config import LiteConfig
from .envs import FetchGoalEnv
from .envs_maniskill import ManiSkillPickCubeGoalEnv, ManiSkillPushTGoalEnv, UnitreeAppleGoalEnv
from .hgr import HGRSampler, HGR_STATE_SCHEMA
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
from .task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec
from .task_fetch_push import FetchPushTaskSpec
from .task_fetch_slide import FetchSlideTaskSpec
from .task_pick_cube import PickCubeTaskSpec
from .task_push_t import PushTTaskSpec
from .task_unitree_apple import UnitreeAppleTaskSpec
from .weights import GapStats, LiteWeightComputer


def make_agent(
    config: LiteConfig,
    state_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: torch.device,
) -> DDPGAgent | TD3Agent | SACAgent:
    if config.method == "sac_hgr":
        return SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            action_low=action_low,
            action_high=action_high,
            device=device,
            l2_action_penalty=config.l2_action_penalty,
            alpha=config.sac_alpha,
            auto_alpha=config.sac_auto_alpha,
            alpha_lr=config.sac_alpha_lr,
            target_entropy=config.sac_target_entropy,
            td_priority_reduction=config.hgr_sac_priority_reduction,
        )
    if config.method in {"td3_her", "td3_hgr", "td3_her_high_rr_x4", "td3_saber_runtime_opt"}:
        return TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            action_low=action_low,
            action_high=action_high,
            device=device,
            l2_action_penalty=config.l2_action_penalty,
            policy_delay=config.td3_policy_delay,
            target_policy_noise=config.td3_target_policy_noise,
            target_noise_clip=config.td3_target_noise_clip,
            td_priority_reduction=config.hgr_td3_priority_reduction,
        )
    return DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        action_low=action_low,
        action_high=action_high,
        device=device,
        l2_action_penalty=config.l2_action_penalty,
    )


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
    if config.task_name in {"fetch_push", "fetch_pick_and_place", "fetch_slide"}:
        return FetchGoalEnv(config.env_id, config.max_episode_steps, seed, render)
    if config.task_name == "unitree_apple":
        return UnitreeAppleGoalEnv(
            env_id=config.env_id,
            max_episode_steps=config.max_episode_steps,
            seed=seed,
            render=render,
            obs_mode=config.obs_mode,
            reward_type=config.reward_type,
            control_mode=config.control_mode,
            num_envs=config.num_envs,
            backend=config.backend,
            sim_backend=config.sim_backend,
        )
    if config.task_name == "pick_cube":
        return ManiSkillPickCubeGoalEnv(
            env_id=config.env_id,
            max_episode_steps=config.max_episode_steps,
            seed=seed,
            render=render,
            obs_mode=config.obs_mode,
            reward_type=config.reward_type,
            robot_uids=config.robot_uids,
            control_mode=config.control_mode,
            num_envs=config.num_envs,
            backend=config.backend,
            sim_backend=config.sim_backend,
        )
    if config.task_name == "push_t":
        return ManiSkillPushTGoalEnv(
            env_id=config.env_id,
            max_episode_steps=config.max_episode_steps,
            seed=seed,
            render=render,
            obs_mode=config.obs_mode,
            reward_type=config.reward_type,
            robot_uids=config.robot_uids,
            control_mode=config.control_mode,
            num_envs=config.num_envs,
            backend=config.backend,
            sim_backend=config.sim_backend,
            pos_success_thresh=config.push_t_pos_success_thresh,
            rot_success_thresh=config.push_t_rot_success_thresh,
            rot_weight=config.push_t_rot_weight,
            contact_thresh=config.push_t_contact_thresh,
        )
    raise ValueError(f"Unsupported task_name={config.task_name!r}")


def make_task_spec(config: LiteConfig, env: Any) -> Any:
    if config.task_name == "fetch_push":
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
    if config.task_name == "fetch_pick_and_place":
        return FetchPickAndPlaceTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            epsilon_g=env.distance_threshold,
            d_reach=config.d_reach,
            d_goal_near=config.d_goal_near,
            z_lift_min=config.z_lift_min,
            grip_close_thresh=config.grip_close_thresh,
            hand_obj_bins=config.pick_hand_obj_bins,
            obj_goal_bins=config.pick_obj_goal_bins,
            object_z_bins=config.pick_object_z_bins,
            gripper_gap_bins=config.pick_gripper_gap_bins,
            horizon_bins=config.horizon_bins,
        )
    if config.task_name == "fetch_slide":
        return FetchSlideTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            epsilon_g=env.distance_threshold,
            d_contact=config.slide_d_contact,
            d_goal_near=config.slide_d_goal_near,
            v_slide_min=config.slide_v_slide_min,
            v_fast=config.slide_v_fast,
            ee_obj_bins=config.slide_ee_obj_bins,
            obj_goal_bins=config.slide_obj_goal_bins,
            obj_speed_bins=config.slide_obj_speed_bins,
            vel_align_bins=config.slide_vel_align_bins,
            horizon_bins=config.horizon_bins,
        )
    if config.task_name == "unitree_apple":
        return UnitreeAppleTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            epsilon_g=env.distance_threshold,
            d_reach=config.d_reach,
            d_goal_near=config.d_goal_near,
            hand_apple_bins=config.hand_apple_bins,
            apple_goal_bins=config.apple_goal_bins,
            apple_z_bins=config.apple_z_bins,
            horizon_bins=config.horizon_bins,
        )
    if config.task_name == "pick_cube":
        z_lift_min = (
            float(config.pick_cube_z_lift_min)
            if config.pick_cube_z_lift_min is not None
            else float(getattr(env, "z_lift_min", 0.07))
        )
        return PickCubeTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            epsilon_g=env.distance_threshold,
            d_reach=config.d_reach,
            d_goal_near=env.distance_threshold,
            z_lift_min=z_lift_min,
            tcp_cube_bins=config.pick_cube_tcp_cube_bins,
            cube_goal_bins=config.pick_cube_goal_bins,
            cube_z_bins=config.pick_cube_z_bins,
            horizon_bins=config.horizon_bins,
            z_lift_min_source=str(getattr(env, "z_lift_min_source", "config")),
        )
    if config.task_name == "push_t":
        return PushTTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            pos_success_thresh=config.push_t_pos_success_thresh,
            rot_success_thresh=config.push_t_rot_success_thresh,
            contact_thresh=config.push_t_contact_thresh,
            pos_near_thresh=config.push_t_pos_near_thresh,
            rot_near_thresh=config.push_t_rot_near_thresh,
            rot_large_thresh=config.push_t_rot_large_thresh,
            speed_thresh=config.push_t_speed_thresh,
            ang_vel_thresh=config.push_t_ang_vel_thresh,
            tcp_obj_bins=config.push_t_tcp_obj_bins,
            obj_goal_dist_bins=config.push_t_obj_goal_dist_bins,
            rot_err_bins_rad=config.push_t_rot_err_bins_rad,
            obj_speed_bins=config.push_t_obj_speed_bins,
            ang_vel_bins=config.push_t_ang_vel_bins,
            horizon_bins=config.horizon_bins,
        )
    raise ValueError(f"Unsupported task_name={config.task_name!r}")


@torch.no_grad()
def evaluate_agent(
    agent: DDPGAgent,
    env: Any,
    eval_seed: int,
    eval_episodes: int,
    task: Any | None = None,
) -> Dict[str, float]:
    returns = []
    goal_successes_ever = []
    env_successes_ever = []
    goal_successes_final = []
    env_successes_final = []
    episode_lengths = []
    push_t_series: Dict[str, list[float]] = {}
    pos_progress = []
    rot_progress = []
    eval_obs = []
    eval_goals = []
    dummy_rng = np.random.default_rng(0)
    old_intersection_flag = getattr(env, "compute_intersection_ratio", None)
    if hasattr(env, "set_compute_intersection_ratio"):
        env.set_compute_intersection_ratio(True)
    try:
        for ep in range(eval_episodes):
            obs = env.reset(seed=eval_seed + ep)
            ep_ret = 0.0
            ep_goal_succ_ever = 0.0
            ep_env_succ_ever = 0.0
            ep_goal_succ_final = 0.0
            ep_env_succ_final = 0.0
            ep_len = 0
            prev_pos_dist = None
            prev_rot_err = None
            for _ in range(env.max_episode_steps):
                action = agent.act(obs.flat_state(), noise_std=0.0, noise_rng=dummy_rng)
                obs, reward, terminated, truncated, info = env.step(action)
                eval_obs.append(obs.observation.copy())
                eval_goals.append(obs.desired_goal.copy())
                ep_ret += float(reward)
                ep_len += 1
                goal_succ = float(info.get("goal_success", info.get("is_success", 0.0)))
                env_succ = float(info.get("env_success", info.get("is_success", goal_succ)))
                ep_goal_succ_ever = max(ep_goal_succ_ever, goal_succ)
                ep_env_succ_ever = max(ep_env_succ_ever, env_succ)
                ep_goal_succ_final = goal_succ
                ep_env_succ_final = env_succ
                for key, value in info.items():
                    if not str(key).startswith("pushT/"):
                        continue
                    try:
                        scalar = float(np.asarray(value).reshape(-1)[0])
                    except (TypeError, ValueError, IndexError):
                        continue
                    push_t_series.setdefault(str(key), []).append(scalar)
                pos_dist = info.get("pushT/obj_goal_xy_dist")
                rot_err = info.get("pushT/obj_goal_rot_err")
                if pos_dist is not None:
                    pos_value = float(np.asarray(pos_dist).reshape(-1)[0])
                    if prev_pos_dist is not None and np.isfinite(pos_value) and np.isfinite(prev_pos_dist):
                        pos_progress.append(float(pos_value < prev_pos_dist))
                    prev_pos_dist = pos_value
                if rot_err is not None:
                    rot_value = float(np.asarray(rot_err).reshape(-1)[0])
                    if prev_rot_err is not None and np.isfinite(rot_value) and np.isfinite(prev_rot_err):
                        rot_progress.append(float(rot_value < prev_rot_err))
                    prev_rot_err = rot_value
                if terminated or truncated:
                    break
            returns.append(ep_ret)
            goal_successes_ever.append(ep_goal_succ_ever)
            env_successes_ever.append(ep_env_succ_ever)
            goal_successes_final.append(ep_goal_succ_final)
            env_successes_final.append(ep_env_succ_final)
            episode_lengths.append(ep_len)
    finally:
        if old_intersection_flag is not None and hasattr(env, "set_compute_intersection_ratio"):
            env.set_compute_intersection_ratio(bool(old_intersection_flag))

    env_ever_mean = float(np.mean(env_successes_ever) if env_successes_ever else 0.0)
    goal_ever_mean = float(np.mean(goal_successes_ever) if goal_successes_ever else 0.0)
    env_final_mean = float(np.mean(env_successes_final) if env_successes_final else 0.0)
    goal_final_mean = float(np.mean(goal_successes_final) if goal_successes_final else 0.0)
    metrics = {
        "eval_return_mean": float(np.mean(returns) if returns else 0.0),
        "eval_return_std": float(np.std(returns) if returns else 0.0),
        "eval_success_mean": env_ever_mean,
        "eval_success_std": float(np.std(env_successes_ever) if env_successes_ever else 0.0),
        "eval_goal_success_mean": goal_ever_mean,
        "eval_goal_success_std": float(np.std(goal_successes_ever) if goal_successes_ever else 0.0),
        "eval_env_success_mean": env_ever_mean,
        "eval_env_success_std": float(np.std(env_successes_ever) if env_successes_ever else 0.0),
        "eval_goal_success_ever_mean": goal_ever_mean,
        "eval_goal_success_ever_std": float(np.std(goal_successes_ever) if goal_successes_ever else 0.0),
        "eval_env_success_ever_mean": env_ever_mean,
        "eval_env_success_ever_std": float(np.std(env_successes_ever) if env_successes_ever else 0.0),
        "eval_goal_success_final_mean": goal_final_mean,
        "eval_goal_success_final_std": float(np.std(goal_successes_final) if goal_successes_final else 0.0),
        "eval_env_success_final_mean": env_final_mean,
        "eval_env_success_final_std": float(np.std(env_successes_final) if env_successes_final else 0.0),
        "eval_episode_length_mean": float(np.mean(episode_lengths) if episode_lengths else 0.0),
    }
    push_specs = {
        "pushT/obj_goal_xy_dist": ("pushT/obj_goal_xy_dist_mean", "pushT/obj_goal_xy_dist_p50", "pushT/obj_goal_xy_dist_p90"),
        "pushT/obj_goal_rot_err": ("pushT/obj_goal_rot_err_mean", "pushT/obj_goal_rot_err_p50", "pushT/obj_goal_rot_err_p90"),
        "pushT/intersection_ratio": (
            "pushT/intersection_ratio_mean",
            "pushT/intersection_ratio_p50",
            "pushT/intersection_ratio_p90",
        ),
    }
    for raw_key, out_keys in push_specs.items():
        arr = np.asarray(push_t_series.get(raw_key, []), dtype=np.float32)
        metrics[out_keys[0]] = _nanmean(arr)
        metrics[out_keys[1]] = _nanpercentile(arr, 50.0)
        metrics[out_keys[2]] = _nanpercentile(arr, 90.0)
    for raw_key, out_key in {
        "pushT/success_margin": "pushT/success_margin_mean",
        "pushT/tcp_obj_dist": "pushT/tcp_obj_dist_mean",
        "pushT/contact_like": "pushT/contact_like_rate",
        "pushT/obj_speed_xy": "pushT/obj_speed_xy_mean",
        "pushT/obj_ang_vel_z_abs": "pushT/obj_ang_vel_z_abs_mean",
        "pushT/vel_align_to_goal": "pushT/vel_align_to_goal_mean",
    }.items():
        metrics[out_key] = _nanmean(np.asarray(push_t_series.get(raw_key, []), dtype=np.float32))
    metrics["pushT/pos_progress_rate"] = _nanmean(np.asarray(pos_progress, dtype=np.float32))
    metrics["pushT/rot_progress_rate"] = _nanmean(np.asarray(rot_progress, dtype=np.float32))
    if task is not None and hasattr(task, "diagnostic_metrics") and eval_obs:
        task_metrics = task.diagnostic_metrics(
            np.asarray(eval_obs, dtype=np.float32),
            np.asarray(eval_goals, dtype=np.float32),
        )
        for key, value in task_metrics.items():
            if str(key).startswith("pushT/phase_"):
                metrics[key] = float(value)
            elif key not in metrics or not np.isfinite(float(metrics[key])):
                metrics[key] = float(value)
    return metrics


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _nanmean(arr: np.ndarray) -> float:
    values = np.asarray(arr, dtype=np.float32)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _nanpercentile(arr: np.ndarray, percentile: float) -> float:
    values = np.asarray(arr, dtype=np.float32)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanpercentile(values, percentile))


def _auc_success(history: list[tuple[int, float]]) -> float:
    if not history:
        return float("nan")
    if len(history) == 1:
        return float(history[0][1])
    steps = np.asarray([step for step, _ in history], dtype=np.float64)
    values = np.asarray([value for _, value in history], dtype=np.float64)
    span = max(1.0, float(steps[-1] - steps[0]))
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, steps) / span)


def _first_hit(history: list[tuple[int, float]], threshold: float) -> str:
    for step, value in history:
        if value >= threshold:
            return str(int(step))
    return ""


def _entropy_weight(w: np.ndarray, eps: float) -> float:
    if w.size == 0:
        return 0.0
    p = np.maximum(w.astype(np.float64), eps)
    p = p / max(eps, float(p.sum()))
    return float(-(p * np.log(p + eps)).sum())


def _ess_weight(w: np.ndarray, eps: float) -> float:
    values = np.maximum(np.asarray(w, dtype=np.float64), eps)
    if values.size == 0:
        return 0.0
    return float((values.sum() ** 2) / max(eps, float(np.sum(values ** 2))))


def _apply_td_or_per_loss_weights(
    config: LiteConfig,
    agent: DDPGAgent,
    wb: Any,
    state: torch.Tensor,
    action_exec: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    nonterminal: torch.Tensor,
) -> None:
    td_abs = agent.td_error_abs(state, action_exec, reward, next_state, nonterminal).detach().cpu().numpy().astype(np.float32)
    eps = float(config.eps)
    if config.method == "td_error":
        td_norm = td_abs / max(eps, float(np.mean(td_abs)))
        clipped = np.clip(td_norm, 0.0, float(config.U_TD_max)).astype(np.float32)
        w_raw = (1.0 + float(config.lambda_TD) * clipped).astype(np.float32)
        wb.extra_metrics.update(
            {
                "td_error/abs_mean": float(np.mean(td_abs)) if td_abs.size else 0.0,
                "td_error/abs_p90": _nanpercentile(td_abs, 90.0),
                "td_error/weight_mean": float(np.mean(w_raw)) if w_raw.size else 1.0,
                "td_error/weight_max": float(np.max(w_raw)) if w_raw.size else 1.0,
                "td_error/weight_entropy": _entropy_weight(w_raw, eps),
                "td_error/weight_ESS": _ess_weight(w_raw, eps),
            }
        )
    elif config.method == "per_style_loss_weighting":
        priority = np.power(td_abs + float(config.eps_PER), float(config.per_alpha)).astype(np.float32)
        p_norm = priority / max(eps, float(np.mean(priority)))
        w_raw = np.clip(p_norm, 0.0, float(config.U_PER_max)).astype(np.float32)
        wb.extra_metrics.update(
            {
                "per/priority_entropy": _entropy_weight(priority, eps),
                "per/priority_ESS": _ess_weight(priority, eps),
                "per/weight_mean": float(np.mean(w_raw)) if w_raw.size else 1.0,
                "per/weight_max": float(np.max(w_raw)) if w_raw.size else 1.0,
                "per/weight_entropy": _entropy_weight(w_raw, eps),
                "per/weight_ESS": _ess_weight(w_raw, eps),
            }
        )
    else:
        return
    omega = (w_raw / max(eps, float(np.mean(w_raw)))).astype(np.float32)
    wb.w_p_raw = w_raw.astype(np.float32)
    wb.omega_critic = omega
    wb.omega_actor = np.ones_like(omega, dtype=np.float32)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if int(np.sum(mask)) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))) / (x_std * y_std))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * float(i + j - 1)
        i = j
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if int(np.sum(mask)) < 2:
        return float("nan")
    return _safe_pearson(_rankdata_average(x[mask]), _rankdata_average(y[mask]))


def _top_td_support_overlap(td_abs: np.ndarray, support: np.ndarray, frac: float) -> float:
    td = np.asarray(td_abs, dtype=np.float64).reshape(-1)
    support_mask = np.asarray(support, dtype=np.float64).reshape(-1) > 0.5
    if td.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(frac) * td.size)))
    top_idx = np.argsort(-td, kind="mergesort")[:k]
    return float(np.mean(support_mask[top_idx]))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask_arr):
        return float("nan")
    return float(np.mean(arr[mask_arr]))


def _masked_var(values: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask_arr):
        return float("nan")
    return float(np.var(arr[mask_arr]))


def _to_flat32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _write_td3_saber_sample_diagnostic(
    samples_writer: MetricsJSONLWriter,
    metrics_writer: MetricsJSONLWriter,
    *,
    run_name: str,
    step: int,
    update_index: int,
    seed: int,
    batch: Any,
    wb: Any,
    q1_before: np.ndarray,
    q2_before: np.ndarray,
    target: np.ndarray,
    td1_before: np.ndarray,
    td2_before: np.ndarray,
    td_mean_before: np.ndarray,
    q1_after: np.ndarray,
    q2_after: np.ndarray,
    policy_action: np.ndarray,
    task: Any,
    action_bins_per_dim: int,
    rng: np.random.Generator,
    sample_count: int,
) -> None:
    q1_before_arr = _to_flat32(q1_before)
    q2_before_arr = _to_flat32(q2_before)
    target_arr = _to_flat32(target)
    td1_before_arr = _to_flat32(td1_before)
    td2_before_arr = _to_flat32(td2_before)
    td_mean_before_arr = _to_flat32(td_mean_before)
    q1_after_arr = _to_flat32(q1_after)
    q2_after_arr = _to_flat32(q2_after)
    td1_after_arr = np.abs(q1_after_arr - target_arr).astype(np.float32)
    td2_after_arr = np.abs(q2_after_arr - target_arr).astype(np.float32)
    td_mean_after_arr = (0.5 * (td1_after_arr + td2_after_arr)).astype(np.float32)
    reduction_mean = (td_mean_before_arr - td_mean_after_arr).astype(np.float32)
    q_before_mean = (0.5 * (q1_before_arr + q2_before_arr)).astype(np.float32)
    q_after_mean = (0.5 * (q1_after_arr + q2_after_arr)).astype(np.float32)
    psi = np.asarray(wb.psi_q_rank, dtype=np.float32).reshape(-1)
    support_mask = psi > 0.5
    n = int(min(td_mean_before_arr.size, psi.size, batch.action_exec.shape[0]))
    if n <= 0:
        return
    eps = np.float32(1e-8)
    omega_critic_arr = np.asarray(wb.omega_critic, dtype=np.float32).reshape(-1)[:n]
    reduction_over_omega = (reduction_mean[:n] / np.maximum(omega_critic_arr, eps)).astype(np.float32)
    td_after_over_before = (td_mean_after_arr[:n] / np.maximum(td_mean_before_arr[:n], eps)).astype(np.float32)
    phase_arr = np.asarray(wb.phase, dtype=np.int64).reshape(-1)[:n]
    if hasattr(task, "full_context_hash_batch"):
        full_context_hash = task.full_context_hash_batch(
            batch.observation[:n], batch.desired_goal_train[:n], batch.h[:n]
        ).astype(np.int64)
    else:
        full_context_hash = np.full(n, -1, dtype=np.int64)
    if hasattr(task, "coarse_hash_batch"):
        coarse_context_hash = task.coarse_hash_batch(batch.observation[:n], batch.h[:n]).astype(np.int64)
    else:
        coarse_context_hash = np.full(n, -1, dtype=np.int64)
    replay_action_bin = task.action_bin_batch(batch.action_exec[:n], int(action_bins_per_dim)).astype(np.int64)
    policy_action_bin = task.action_bin_batch(policy_action[:n], int(action_bins_per_dim)).astype(np.int64)
    support_count = int(np.sum(support_mask[:n]))
    support_rate = float(support_count / max(1, n))
    top10_overlap = _top_td_support_overlap(td_mean_before_arr[:n], psi[:n], 0.10)
    top20_overlap = _top_td_support_overlap(td_mean_before_arr[:n], psi[:n], 0.20)
    phase_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(phase_arr, return_counts=True))}
    support_phase = phase_arr[support_mask[:n]]
    support_phase_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(support_phase, return_counts=True))}
    metrics_writer.write(
        {
            "type": "diagnostic_batch",
            "critic_family": "td3_twin",
            "run_name": run_name,
            "step": int(step),
            "update_index": int(update_index),
            "seed": int(seed),
            "batch_size": n,
            "logged_sample_count": int(min(n, int(sample_count))),
            "td_target_fixed_from_pre_update": True,
            "diagnostic_rng_isolated": True,
            "support_positive_count": support_count,
            "support_positive_rate": support_rate,
            "psi_td_abs_pearson": _safe_pearson(psi[:n], td_mean_before_arr[:n]),
            "psi_td_abs_spearman": _safe_spearman(psi[:n], td_mean_before_arr[:n]),
            "top10_td_support_overlap": top10_overlap,
            "top20_td_support_overlap": top20_overlap,
            "top10_td_support_enrichment": float(top10_overlap / support_rate) if support_rate > 0.0 else float("nan"),
            "top20_td_support_enrichment": float(top20_overlap / support_rate) if support_rate > 0.0 else float("nan"),
            "td_abs_before_q1_mean": float(np.mean(td1_before_arr[:n])),
            "td_abs_before_q2_mean": float(np.mean(td2_before_arr[:n])),
            "td_abs_before_mean": float(np.mean(td_mean_before_arr[:n])),
            "td_abs_after_q1_mean": float(np.mean(td1_after_arr[:n])),
            "td_abs_after_q2_mean": float(np.mean(td2_after_arr[:n])),
            "td_abs_after_mean": float(np.mean(td_mean_after_arr[:n])),
            "td_abs_reduction_mean": float(np.mean(reduction_mean[:n])),
            "support_positive_td_abs_before_mean": _masked_mean(td_mean_before_arr[:n], support_mask[:n]),
            "support_negative_td_abs_before_mean": _masked_mean(td_mean_before_arr[:n], ~support_mask[:n]),
            "support_positive_td_abs_after_mean": _masked_mean(td_mean_after_arr[:n], support_mask[:n]),
            "support_negative_td_abs_after_mean": _masked_mean(td_mean_after_arr[:n], ~support_mask[:n]),
            "support_positive_td_abs_reduction_mean": _masked_mean(reduction_mean[:n], support_mask[:n]),
            "support_negative_td_abs_reduction_mean": _masked_mean(reduction_mean[:n], ~support_mask[:n]),
            "support_positive_td_abs_reduction_over_omega_mean": _masked_mean(reduction_over_omega, support_mask[:n]),
            "support_negative_td_abs_reduction_over_omega_mean": _masked_mean(reduction_over_omega, ~support_mask[:n]),
            "support_positive_td_abs_after_over_before_mean": _masked_mean(td_after_over_before, support_mask[:n]),
            "support_negative_td_abs_after_over_before_mean": _masked_mean(td_after_over_before, ~support_mask[:n]),
            "target_variance_support_positive": _masked_var(target_arr[:n], support_mask[:n]),
            "target_variance_support_negative": _masked_var(target_arr[:n], ~support_mask[:n]),
            "phase_counts": phase_counts,
            "support_positive_phase_counts": support_phase_counts,
            "w_raw_mean": float(np.mean(wb.w_p_raw[:n])) if n else float("nan"),
            "omega_critic_mean": float(np.mean(omega_critic_arr)) if n else float("nan"),
        }
    )
    all_idx = np.arange(n, dtype=np.int64)
    if int(sample_count) < n:
        chosen = np.sort(rng.choice(all_idx, size=int(sample_count), replace=False).astype(np.int64))
    else:
        chosen = all_idx
    her_relabel = getattr(batch, "her_relabel", None)
    if her_relabel is None:
        her_relabel = np.zeros(n, dtype=np.float32)
    for out_i, batch_i in enumerate(chosen):
        i = int(batch_i)
        samples_writer.write(
            {
                "type": "diagnostic_sample",
                "critic_family": "td3_twin",
                "run_name": run_name,
                "step": int(step),
                "update_index": int(update_index),
                "seed": int(seed),
                "sample_id": int(out_i),
                "batch_index": i,
                "replay_index": int(batch.indices[i]),
                "psi_i": float(psi[i]),
                "psi_q_rank": float(psi[i]),
                "support_positive": bool(support_mask[i]),
                "td_abs_before_q1": float(td1_before_arr[i]),
                "td_abs_before_q2": float(td2_before_arr[i]),
                "td_abs_before_mean": float(td_mean_before_arr[i]),
                "td_abs_after_q1": float(td1_after_arr[i]),
                "td_abs_after_q2": float(td2_after_arr[i]),
                "td_abs_after_mean": float(td_mean_after_arr[i]),
                "td_abs_reduction_mean": float(reduction_mean[i]),
                "td_abs_before": float(td_mean_before_arr[i]),
                "td_abs_after": float(td_mean_after_arr[i]),
                "td_abs_reduction": float(reduction_mean[i]),
                "td_abs_reduction_over_omega": float(reduction_over_omega[i]),
                "td_abs_after_over_before": float(td_after_over_before[i]),
                "target_value": float(target_arr[i]),
                "q_before_q1": float(q1_before_arr[i]),
                "q_before_q2": float(q2_before_arr[i]),
                "q_before_mean": float(q_before_mean[i]),
                "q_after_q1": float(q1_after_arr[i]),
                "q_after_q2": float(q2_after_arr[i]),
                "q_after_mean": float(q_after_mean[i]),
                "q_before": float(q_before_mean[i]),
                "q_after": float(q_after_mean[i]),
                "w_raw": float(wb.w_p_raw[i]),
                "omega_critic": float(omega_critic_arr[i]),
                "local_pool_size": int(wb.len_a_q_local[i]),
                "support_candidate_count": int(wb.len_a_q_sup[i]),
                "phase_id": int(phase_arr[i]),
                "full_context_hash": int(full_context_hash[i]),
                "coarse_context_hash": int(coarse_context_hash[i]),
                "policy_action_bin": int(policy_action_bin[i]),
                "replay_action_bin": int(replay_action_bin[i]),
                "her_relabel": bool(float(her_relabel[i]) > 0.5),
            }
        )


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
    agent = make_agent(
        config=config,
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        action_low=train_env.action_low,
        action_high=train_env.action_high,
        device=device,
    )
    replay = HERReplayBuffer(
        capacity=config.buffer_capacity,
        obs_dim=train_env.obs_dim,
        goal_dim=train_env.goal_dim,
        action_dim=train_env.action_dim,
    )
    hgr_enabled = config.method in {"hgr", "td3_hgr", "sac_hgr"}
    hgr_sampler = HGRSampler(config, train_env, device) if hgr_enabled else None
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
            "hgr_convention": {
                "enabled": hgr_enabled,
                "backbone": (
                    "SAC" if isinstance(agent, SACAgent)
                    else "TD3" if isinstance(agent, TD3Agent)
                    else "DDPG"
                ),
                "two_step_ranking": True,
                "support_lookup": False,
                "priority_table": "completed-episode triangular transition/future-goal pair table",
                "episode_priority": "mean pair priority",
                "pair_priority": (
                    f"pre-update {config.hgr_sac_priority_reduction} twin-critic entropy-regularized TD error plus epsilon"
                    if isinstance(agent, SACAgent)
                    else f"pre-update {config.hgr_td3_priority_reduction} twin-critic TD error plus epsilon"
                    if isinstance(agent, TD3Agent)
                    else "pre-update absolute TD error plus epsilon"
                ),
                "twin_priority_reduction": (
                    config.hgr_sac_priority_reduction
                    if isinstance(agent, SACAgent)
                    else config.hgr_td3_priority_reduction
                    if isinstance(agent, TD3Agent)
                    else None
                ),
                "new_pair_priority": "current global maximum",
                "episode_alpha": config.hgr_alpha_episode,
                "pair_alpha": config.hgr_alpha_goal,
                "episode_beta_initial": config.hgr_beta_episode_initial,
                "pair_beta_initial": config.hgr_beta_goal_initial,
                "episode_beta_final": config.hgr_beta_episode_final,
                "pair_beta_final": config.hgr_beta_goal_final,
                "beta_schedule": "linear by critic-update progress",
                "beta_anneal_updates_resolved": (
                    hgr_sampler.beta_anneal_updates if hgr_sampler is not None else 0
                ),
                "uniform_mix": config.hgr_uniform_mix,
                "priority_clip_max": config.hgr_priority_clip_max,
                "checkpoint_state_schema": HGR_STATE_SCHEMA,
                "critic_loss_weight": (
                    "same normalized episode*pair IS weight applied to both twin critics on relabeled samples"
                    if isinstance(agent, (TD3Agent, SACAgent))
                    else "normalized product of episode and pair IS weights on relabeled samples"
                ),
                "actor_loss_weight": "all_ones",
                "original_goal_branch_weight": "one",
            },
        },
    )
    train_env_snapshot = train_env.snapshot()
    eval_env_snapshot = eval_env.snapshot()
    task_metadata = task.metadata()
    write_json(
        run_dir / "env_snapshot.json",
        {
            "train_env": train_env_snapshot,
            "eval_env": eval_env_snapshot,
            "task_spec": task_metadata,
        },
    )
    if config.task_name == "push_t":
        logger.info(
            "PushT assumptions: official_success=eval_env_success pose_goal_success=eval_goal_success "
            "goal_theta_source=%s velocity_available=%s intersection_train=%s intersection_eval=%s "
            "phase_names=%s max_episode_steps=%d action_dim=%d obs_dim=%d goal_dim=%d",
            train_env_snapshot.get("goal_theta_source"),
            train_env_snapshot.get("velocity_available"),
            train_env_snapshot.get("compute_intersection_ratio_default"),
            train_env_snapshot.get("compute_intersection_ratio_eval_enabled_by_trainer"),
            list(phase_names),
            train_env.max_episode_steps,
            train_env.action_dim,
            train_env.obs_dim,
            train_env.goal_dim,
        )

    fields = [
        "step",
        "episode",
        "train/episode_return",
        "train/episode_success",
        "train/episode_len",
        "train/replay_size",
        "train/her_relabel_ratio",
        "config/lambda_support",
        "config/weight_scope",
        "loss/critic",
        "loss/actor",
        "loss/td_error_abs_mean",
        "loss/td_error_abs_p90",
        "sac/alpha",
        "sac/alpha_loss",
        "sac/log_prob_mean",
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
        "lite/entropy_w_P",
        "lite/ESS_w",
        "lite/ESS_w_normalized",
        "lite/support_available_rate",
        "lite/psi_q_rank_rate",
        "lite/q_candidate_fallback_rate",
        "lite/psi_A_policy_rate",
        "lite/mean_len_A_Q_sup",
        "lite/mean_len_A_Q_replay_local",
        "lite/local_support_count_mean",
        "lite/local_support_count_p50",
        "lite/local_support_count_p90",
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
        "time/numba_support_scan",
        "time_local_context_lookup",
        "time_candidate_construction",
        "time_q_policy_eval",
        "time_q_candidate_eval",
        "time_weight_computation",
        "time_actor_critic_update",
        "runtime/steps_per_sec",
        "runtime/total_sec",
        "runtime/overhead_vs_B0",
        "runtime/support_cache_hit_rate",
        "runtime/support_cache_miss_rate",
        "runtime/batch_query_unique_rate",
        "runtime/batch_query_coalescing_saving_rate",
        "runtime/coarse_query_unique_rate",
        "runtime/local_scan_cache_hit_rate",
        "runtime/local_scan_cache_miss_rate",
        "runtime/full_context_scan_rate",
        "runtime/exact_negative_short_circuit_rate",
        "runtime/count_short_circuit_rate",
        "runtime/threshold_short_circuit_rate",
        "runtime/coarse_count_short_circuit_rate",
        "runtime/coarse_support_size_short_circuit_rate",
        "runtime/scanned_count_short_circuit_rate",
        "runtime/coarse_action_bin_count_lookup_rate",
        "runtime/support_cache_attempt_rate",
        "runtime/support_context_cache_size",
        "runtime/local_scan_cache_attempt_rate",
        "runtime/local_scan_cache_size",
        "runtime/full_context_filter_count_mean",
        "runtime/numba_support_scan_enabled",
        "runtime/numba_scanner_warmed",
        "runtime/numba_support_scan_fallback_rate",
        "runtime/numba_support_queries",
        "runtime/predicate_early_exit_rate",
        "runtime/predicate_scanned_count_mean",
        "runtime/diagnostic_audit_rate",
        "runtime/audit_len_A_Q_local_abs_diff_mean",
        "runtime/audit_len_A_Q_sup_abs_diff_mean",
        "runtime/audit_psi_A_policy_mismatch_rate",
        "hgr/episode_priority_mean",
        "hgr/episode_priority_std",
        "hgr/episode_priority_max",
        "hgr/eligible_episode_count",
        "hgr/pair_table_count",
        "hgr/pair_priority_count",
        "hgr/pair_priority_memory_mb",
        "hgr/goal_priority_entropy_mean",
        "hgr/goal_selected_rank_mean",
        "hgr/candidate_td_error_mean",
        "hgr/candidate_td_error_max",
        "hgr/relabel_fraction",
        "hgr/fallback_fraction",
        "hgr/episode_probability_mean",
        "hgr/pair_probability_mean",
        "hgr/is_weight_raw_mean",
        "hgr/is_weight_raw_max",
        "hgr/is_weight_mean",
        "hgr/is_weight_max",
        "hgr/is_weight_min",
        "hgr/actor_weight_mean",
        "hgr/beta_episode",
        "hgr/beta_goal",
        "hgr/beta_progress",
        "hgr/beta_episode_next",
        "hgr/beta_goal_next",
        "hgr/beta_progress_next",
        "hgr/max_priority",
        "hgr/priority_clip_enabled",
        "hgr/priority_clip_rate",
        "hgr/priority_safety_replacement_rate",
        "hgr/update_priority_mean",
        "runtime/hgr_sampling_sec_per_update",
        "mean_w_P_reference",
        "reference_high_weight_rate",
        "assigned_high_weight_rate",
        "support_weight_alignment_rate",
        "support_cache_hit_rate",
        "short_circuit_rate",
        "lite/mean_w_P_reference",
        "lite/reference_high_weight_rate",
        "lite/assigned_high_weight_rate",
        "lite/support_weight_alignment_rate",
        "lite/support_cache_hit_rate",
        "lite/short_circuit_rate",
        "lite/runtime_opt_exact_path",
        "td_error/abs_mean",
        "td_error/abs_p90",
        "td_error/weight_mean",
        "td_error/weight_max",
        "td_error/weight_entropy",
        "td_error/weight_ESS",
        "per/priority_entropy",
        "per/priority_ESS",
        "per/weight_mean",
        "per/weight_max",
        "per/weight_entropy",
        "per/weight_ESS",
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
        "eval/goal_success_ever_mean",
        "eval/goal_success_ever_std",
        "eval/env_success_ever_mean",
        "eval/env_success_ever_std",
        "eval/goal_success_final_mean",
        "eval/goal_success_final_std",
        "eval/env_success_final_mean",
        "eval/env_success_final_std",
        "eval/episode_length_mean",
        "eval/final_success",
        "eval/best_success",
        "eval/AUC_success",
        "eval/first_hit_0.25",
        "eval/first_hit_0.50",
        "eval/first_hit_0.75",
        "eval/first_hit_0.90",
        "pushT/obj_goal_xy_dist_mean",
        "pushT/obj_goal_xy_dist_p50",
        "pushT/obj_goal_xy_dist_p90",
        "pushT/obj_goal_rot_err_mean",
        "pushT/obj_goal_rot_err_p50",
        "pushT/obj_goal_rot_err_p90",
        "pushT/intersection_ratio_mean",
        "pushT/intersection_ratio_p50",
        "pushT/intersection_ratio_p90",
        "pushT/success_margin_mean",
        "pushT/tcp_obj_dist_mean",
        "pushT/contact_like_rate",
        "pushT/obj_speed_xy_mean",
        "pushT/obj_ang_vel_z_abs_mean",
        "pushT/vel_align_to_goal_mean",
        "pushT/pos_progress_rate",
        "pushT/rot_progress_rate",
        "b2/unrestricted_gap_mean",
        "b2/unrestricted_gap_p95",
        "b2/candidate_action_norm_mean",
        "b2/action_delta_norm_mean",
        "b2/q_imp_minus_q_policy_mean",
        "b2/high_gap_rate",
        "b2/gap_to_pos_progress_corr",
        "b2/gap_to_rot_progress_corr",
        "b2/high_gap_bad_progress_rate",
        "b3/support_available_rate",
        "b3/local_support_count_mean",
        "b3/local_support_count_p50",
        "b3/local_support_count_p90",
        "b3/support_to_pos_progress_corr",
        "b3/support_to_rot_progress_corr",
        "b3/support_to_success_corr",
        "b3/support_to_td_error_corr",
        "b4/G_sup_mean",
        "b4/G_sup_positive_rate",
        "b4/support_candidate_count_mean",
        "b4/support_candidate_count_p50",
        "b4/support_candidate_count_p90",
        "b4/fallback_rate",
        "b4/local_scan_truncated_rate",
        "b4/B2_gap_minus_B4_gap",
        "b4/unsupported_gap_rate",
    ]
    for phase_name in phase_names:
        fields.extend(
            [
                f"lite/phase_{phase_name}_mean_w_P",
                f"lite/phase_{phase_name}_fallback_rate",
                f"lite/phase_{phase_name}_psi_q_rank_rate",
                f"lite/phase_{phase_name}_positive_G_sup_rate",
                f"lite/phase_{phase_name}_count_rate",
                f"pushT/phase_{phase_name}_obj_goal_xy_dist_mean",
                f"pushT/phase_{phase_name}_obj_goal_rot_err_mean",
                f"pushT/phase_{phase_name}_contact_like_rate",
                f"pushT/phase_{phase_name}_intersection_ratio_mean",
                f"b3/support_by_phase_{phase_name}",
                f"b3/mean_w_P_by_phase_{phase_name}",
                f"b3/fallback_by_phase_{phase_name}",
            ]
        )
    fields = list(dict.fromkeys(fields))
    csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", fields)
    jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")
    diagnostic_samples_writer = (
        MetricsJSONLWriter(run_dir / "diagnostic_samples.jsonl") if config.diagnostic_log_samples else None
    )
    diagnostic_metrics_writer = (
        MetricsJSONLWriter(run_dir / "diagnostic_metrics.jsonl") if config.diagnostic_log_samples else None
    )

    noise_rng = np.random.default_rng(config.seeds.action_noise_seed)
    sample_rng = np.random.default_rng(config.seeds.numpy_seed + 17)
    weight_rng = np.random.default_rng(config.seeds.weight_seed)
    diagnostic_rng = np.random.default_rng(config.seeds.numpy_seed + 17003)
    last_diagnostic_step = -1
    start_step = 0
    episode_idx = 0
    total_updates = 0
    selection_metric = "eval_env_success_mean" if config.task_name == "push_t" else "eval_goal_success_mean"
    best_eval_success = -1.0
    best_eval_selection_success = -1.0
    best_eval_goal_success = -1.0
    best_eval_env_success = -1.0
    best_eval_return = float("-inf")
    best_eval_step = -1
    last_eval_metrics: Dict[str, float] = {}
    eval_history_env: list[tuple[int, float]] = []

    def restore_eval_history_from_metrics(limit_step: int) -> list[tuple[int, float]]:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return []
        history: list[tuple[int, float]] = []
        seen_steps: set[int] = set()
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as f:
                cleaned = (line.replace("\x00", "") for line in f)
                for row in csv.DictReader(cleaned):
                    step_raw = row.get("step", "")
                    value_raw = row.get("eval/env_success_mean", "")
                    if not step_raw or value_raw in ("", None):
                        continue
                    step = int(float(step_raw))
                    if step > int(limit_step):
                        continue
                    value = float(value_raw)
                    if step in seen_steps:
                        history = [(s, v) for s, v in history if s != step]
                    seen_steps.add(step)
                    history.append((step, value))
        except Exception as exc:
            logger.warning("failed to restore eval history from %s: %s", metrics_path, exc)
            return []
        history.sort(key=lambda item: item[0])
        return history

    ckpt_path = Path(config.resume_path) if config.resume_path else _checkpoint_path(run_dir)
    if config.resume and ckpt_path.exists():
        payload = load_checkpoint(ckpt_path)
        agent.load_state_dict(payload["agent"])
        replay.load_state_dict(payload["replay"])
        if replay.indexed_slot_count() < replay.size:
            replay.rebuild_action_index(task.coarse_hash_batch(replay.observation[:replay.size], replay.h[:replay.size]))
            logger.info("rebuilt replay action index slots=%d", replay.indexed_slot_count())
        replay.rebuild_action_bins(task.action_bin_batch(replay.action_exec[:replay.size], config.action_bins_per_dim))
        gap_stats.load_state_dict(payload.get("gap_stats", {}))
        # global_step is the number of transitions already stored.
        start_step = int(payload["global_step"])
        episode_idx = int(payload["episode_idx"])
        total_updates = int(payload.get("total_updates", 0))
        best_eval_success = float(payload.get("best_eval_success", -1.0))
        best_eval_selection_success = float(payload.get("best_eval_selection_success", best_eval_success))
        best_eval_goal_success = float(payload.get("best_eval_goal_success", -1.0))
        best_eval_env_success = float(payload.get("best_eval_env_success", -1.0))
        best_eval_return = float(payload.get("best_eval_return", float("-inf")))
        best_eval_step = int(payload.get("best_eval_step", -1))
        noise_rng.bit_generator.state = payload["noise_rng_state"]
        sample_rng.bit_generator.state = payload["sample_rng_state"]
        weight_rng.bit_generator.state = payload["weight_rng_state"]
        if "diagnostic_rng_state" in payload:
            diagnostic_rng.bit_generator.state = payload["diagnostic_rng_state"]
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        if hgr_sampler is not None:
            hgr_payload = payload.get("hgr_sampler")
            if hgr_payload:
                hgr_sampler.load_state_dict(hgr_payload, replay)
                logger.info("restored HGR priority cache from checkpoint")
            else:
                hgr_sampler.rebuild_from_replay(replay)
                logger.info("rebuilt HGR priority cache from replay")
        last_diagnostic_step = int(payload.get("last_diagnostic_step", -1))
        eval_history_env = restore_eval_history_from_metrics(int(payload["global_step"]))
        logger.info("restored %d eval history points from metrics.csv", len(eval_history_env))
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
        "q_std": 0.0,
        "target_std": 0.0,
        "q_abs_mean": 0.0,
        "q_abs_max": 0.0,
        "td_error_abs_mean": 0.0,
        "td_error_abs_p90": 0.0,
    }
    last_lite = _empty_lite_metrics(phase_names)
    last_her_ratio = 0.0
    wall_start = time.perf_counter()

    def checkpoint_payload(step_value: int, kind: str) -> Dict[str, Any]:
        return {
            "checkpoint_kind": kind,
            "config": config.as_dict(),
            "global_step": int(step_value),
            "episode_idx": episode_idx,
            "total_updates": total_updates,
            "selection_metric": selection_metric,
            "best_eval_success": best_eval_success,
            "best_eval_selection_success": best_eval_selection_success,
            "best_eval_goal_success": best_eval_goal_success,
            "best_eval_env_success": best_eval_env_success,
            "best_eval_return": best_eval_return,
            "best_eval_step": best_eval_step,
            "agent": agent.state_dict(),
            "replay": replay.state_dict(),
            "hgr_sampler": hgr_sampler.state_dict() if hgr_sampler is not None else None,
            "gap_stats": gap_stats.state_dict(),
            "noise_rng_state": noise_rng.bit_generator.state,
            "sample_rng_state": sample_rng.bit_generator.state,
            "weight_rng_state": weight_rng.bit_generator.state,
            "diagnostic_rng_state": diagnostic_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "last_diagnostic_step": last_diagnostic_step,
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

        slot_added = replay.add_transition(
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
        if hgr_sampler is not None:
            hgr_sampler.on_replay_add(
                slot_added,
                episode_idx,
                ep_step,
                replay,
                episode_done=done,
            )

        obs = obs_next
        ep_step += 1
        global_step += 1

        if replay.size >= config.batch_size and global_step >= config.warmup_steps:
            for _ in range(config.updates_per_step):
                if hgr_sampler is not None:
                    batch = hgr_sampler.sample_batch(
                        replay,
                        config.batch_size,
                        config.her_ratio,
                        sample_rng,
                    )
                    hgr_metrics = dict(hgr_sampler.latest_metrics)
                else:
                    batch = replay.sample(config.batch_size, config.her_ratio, sample_rng)
                    hgr_metrics = {}
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

                wb = weight_computer.compute(batch, replay, agent, weight_rng, device)
                if hgr_sampler is not None:
                    hgr_critic_weight, hgr_actor_weight = hgr_sampler.loss_weights(batch)
                    wb.omega_critic = hgr_critic_weight
                    wb.omega_actor = hgr_actor_weight
                state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
                next_state_np = np.concatenate([batch.next_observation, batch.desired_goal_train], axis=1).astype(np.float32)
                state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
                action_t = torch.as_tensor(batch.action_exec, dtype=torch.float32, device=device)
                reward_t = torch.as_tensor(reward_train, dtype=torch.float32, device=device)
                next_state_t = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
                nonterminal_t = torch.as_tensor(nonterminal, dtype=torch.float32, device=device)
                if config.method in {"td_error", "per_style_loss_weighting"}:
                    _apply_td_or_per_loss_weights(
                        config=config,
                        agent=agent,
                        wb=wb,
                        state=state_t,
                        action_exec=action_t,
                        reward=reward_t,
                        next_state=next_state_t,
                        nonterminal=nonterminal_t,
                    )

                diagnostic_before = None
                should_log_diagnostic = (
                    config.diagnostic_log_samples
                    and diagnostic_samples_writer is not None
                    and diagnostic_metrics_writer is not None
                    and global_step > 0
                    and global_step % config.diagnostic_freq == 0
                    and last_diagnostic_step != global_step
                    and isinstance(agent, TD3Agent)
                )
                if should_log_diagnostic:
                    q1_before, q2_before, target_before, td1_before, td2_before, td_mean_before = agent.td_error_components(
                        state_t, action_t, reward_t, next_state_t, nonterminal_t
                    )
                    policy_action_before = agent.policy_actions(state_t)
                    diagnostic_before = {
                        "target_tensor": target_before.detach(),
                        "q1_before": q1_before.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        "q2_before": q2_before.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        "target": target_before.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        "td1_before": td1_before.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        "td2_before": td2_before.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        "td_mean_before": td_mean_before.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        "policy_action": policy_action_before.detach().cpu().numpy().astype(np.float32),
                    }

                t_update0 = time.perf_counter()
                update = agent.update(
                    state=state_t,
                    action_exec=action_t,
                    reward=reward_t,
                    next_state=next_state_t,
                    nonterminal=nonterminal_t,
                    omega_critic=torch.as_tensor(wb.omega_critic, dtype=torch.float32, device=device),
                    omega_actor=torch.as_tensor(wb.omega_actor, dtype=torch.float32, device=device),
                    fixed_target=diagnostic_before["target_tensor"] if diagnostic_before is not None else None,
                )
                t_update1 = time.perf_counter()
                last_loss = {
                    "critic": update.critic_loss,
                    "actor": update.actor_loss,
                    "q_mean": update.q_mean,
                    "target_mean": update.target_mean,
                    "q_std": update.q_std,
                    "target_std": update.target_std,
                    "q_abs_mean": update.q_abs_mean,
                    "q_abs_max": update.q_abs_max,
                    "td_error_abs_mean": update.td_error_abs_mean,
                    "td_error_abs_p90": update.td_error_abs_p90,
                    "sac_alpha": update.sac_alpha,
                    "sac_alpha_loss": update.sac_alpha_loss,
                    "sac_log_prob_mean": update.sac_log_prob_mean,
                }
                last_lite = _lite_metrics_from_weight_batch(wb, phase_names)
                last_lite.update(hgr_metrics)
                if hgr_sampler is not None:
                    if update.td_error_abs_values is None:
                        raise RuntimeError("HGR backbone update did not expose pre-update TD errors")
                    last_lite.update(
                        hgr_sampler.update_priorities(batch, update.td_error_abs_values)
                    )
                last_lite["time/actor_critic_update"] = float(t_update1 - t_update0)
                last_lite["time_actor_critic_update"] = float(t_update1 - t_update0)
                if diagnostic_before is not None:
                    q1_after, q2_after = agent.q_value_components(state_t, action_t)
                    _write_td3_saber_sample_diagnostic(
                        samples_writer=diagnostic_samples_writer,
                        metrics_writer=diagnostic_metrics_writer,
                        run_name=config.run_name,
                        step=global_step,
                        update_index=total_updates + 1,
                        seed=config.seeds.env_seed,
                        batch=batch,
                        wb=wb,
                        q1_before=diagnostic_before["q1_before"],
                        q2_before=diagnostic_before["q2_before"],
                        target=diagnostic_before["target"],
                        td1_before=diagnostic_before["td1_before"],
                        td2_before=diagnostic_before["td2_before"],
                        td_mean_before=diagnostic_before["td_mean_before"],
                        q1_after=q1_after.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        q2_after=q2_after.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        policy_action=diagnostic_before["policy_action"],
                        task=task,
                        action_bins_per_dim=config.action_bins_per_dim,
                        rng=diagnostic_rng,
                        sample_count=config.diagnostic_samples_per_log,
                    )
                    last_diagnostic_step = global_step
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
                last_lite,
                config,
            )
            _add_runtime(row, wall_start, global_step)
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
                task=task,
            )
            row = _metric_row(
                global_step,
                episode_idx,
                ep_return,
                ep_success,
                ep_step,
                replay.size,
                last_her_ratio,
                last_loss,
                last_lite,
                config,
            )
            _add_runtime(row, wall_start, global_step)
            row["eval/return_mean"] = eval_metrics["eval_return_mean"]
            row["eval/success_mean"] = eval_metrics["eval_success_mean"]
            row["eval/goal_success_mean"] = eval_metrics["eval_goal_success_mean"]
            row["eval/env_success_mean"] = eval_metrics["eval_env_success_mean"]
            row["eval/goal_success_ever_mean"] = eval_metrics["eval_goal_success_ever_mean"]
            row["eval/goal_success_ever_std"] = eval_metrics["eval_goal_success_ever_std"]
            row["eval/env_success_ever_mean"] = eval_metrics["eval_env_success_ever_mean"]
            row["eval/env_success_ever_std"] = eval_metrics["eval_env_success_ever_std"]
            row["eval/goal_success_final_mean"] = eval_metrics["eval_goal_success_final_mean"]
            row["eval/goal_success_final_std"] = eval_metrics["eval_goal_success_final_std"]
            row["eval/env_success_final_mean"] = eval_metrics["eval_env_success_final_mean"]
            row["eval/env_success_final_std"] = eval_metrics["eval_env_success_final_std"]
            row["eval/episode_length_mean"] = eval_metrics["eval_episode_length_mean"]
            last_eval_metrics = dict(eval_metrics)
            eval_history_env.append((global_step, float(eval_metrics["eval_env_success_mean"])))
            row["eval/final_success"] = eval_metrics["eval_env_success_final_mean"]
            row["eval/best_success"] = max(value for _, value in eval_history_env)
            row["eval/AUC_success"] = _auc_success(eval_history_env)
            for threshold in (0.25, 0.50, 0.75, 0.90):
                row[f"eval/first_hit_{threshold:.2f}"] = _first_hit(eval_history_env, threshold)
            for key, value in eval_metrics.items():
                if str(key).startswith("pushT/"):
                    row[key] = value
            csv_writer.write(row)
            jsonl_writer.write({"type": "eval", **row, **eval_metrics})
            selection_success = float(eval_metrics[selection_metric])
            is_best = (selection_success > best_eval_selection_success) or (
                np.isclose(selection_success, best_eval_selection_success)
                and eval_metrics["eval_return_mean"] > best_eval_return
            )
            if is_best:
                best_eval_selection_success = selection_success
                best_eval_success = selection_success
                best_eval_goal_success = float(eval_metrics["eval_goal_success_mean"])
                best_eval_env_success = float(eval_metrics["eval_env_success_mean"])
                best_eval_return = eval_metrics["eval_return_mean"]
                best_eval_step = global_step
                save_checkpoint(_best_checkpoint_path(run_dir), checkpoint_payload(global_step, "best_eval"))
                write_json(
                    run_dir / "best_checkpoint.json",
                    {
                        "checkpoint": str(_best_checkpoint_path(run_dir)),
                        "step": global_step,
                        "selection_metric": selection_metric,
                        "eval_success_mean": best_eval_success,
                        "eval_goal_success_mean": eval_metrics["eval_goal_success_mean"],
                        "eval_env_success_mean": eval_metrics["eval_env_success_mean"],
                        "eval_goal_success_ever_mean": eval_metrics["eval_goal_success_ever_mean"],
                        "eval_env_success_ever_mean": eval_metrics["eval_env_success_ever_mean"],
                        "eval_goal_success_final_mean": eval_metrics["eval_goal_success_final_mean"],
                        "eval_env_success_final_mean": eval_metrics["eval_env_success_final_mean"],
                        "eval_return_mean": best_eval_return,
                    },
                )
            if hgr_sampler is not None:
                table_count, pair_count, _memory_mb = hgr_sampler.pair_table_stats()
                beta_next = hgr_sampler.current_betas()[0]
                logger.info(
                    "step=%d episode=%d replay=%d eval_goal=%.3f eval_env=%.3f eval_ret=%.2f "
                    "hgr_beta_next=%.4f hgr_is=%.4f hgr_relabel=%.3f hgr_fallback=%.3f "
                    "hgr_tables=%d hgr_pairs=%d hgr_sample_ms=%.3f",
                    global_step,
                    episode_idx,
                    replay.size,
                    eval_metrics["eval_goal_success_mean"],
                    eval_metrics["eval_env_success_mean"],
                    eval_metrics["eval_return_mean"],
                    beta_next,
                    last_lite["hgr/is_weight_mean"],
                    last_lite["hgr/relabel_fraction"],
                    last_lite["hgr/fallback_fraction"],
                    table_count,
                    pair_count,
                    1_000.0 * last_lite["runtime/hgr_sampling_sec_per_update"],
                )
            else:
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
            if hgr_sampler is not None:
                table_count, pair_count, _memory_mb = hgr_sampler.pair_table_stats()
                beta_next = hgr_sampler.current_betas()[0]
                logger.info(
                    "step=%d replay=%d updates=%d loss_q=%.4f loss_pi=%.4f "
                    "hgr_beta_next=%.4f hgr_is=%.4f hgr_relabel=%.3f hgr_fallback=%.3f "
                    "hgr_tables=%d hgr_pairs=%d hgr_sample_ms=%.3f",
                    global_step,
                    replay.size,
                    total_updates,
                    last_loss["critic"],
                    last_loss["actor"],
                    beta_next,
                    last_lite["hgr/is_weight_mean"],
                    last_lite["hgr/relabel_fraction"],
                    last_lite["hgr/fallback_fraction"],
                    table_count,
                    pair_count,
                    1_000.0 * last_lite["runtime/hgr_sampling_sec_per_update"],
                )
            else:
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

    runtime_total_sec = float(time.perf_counter() - wall_start)
    summary = {
        "run_dir": str(run_dir),
        "method": config.method,
        "total_steps": config.total_steps,
        "episodes": episode_idx,
        "total_updates": total_updates,
        "best_eval_success": best_eval_success,
        "best_eval_selection_success": best_eval_selection_success,
        "best_eval_goal_success": best_eval_goal_success,
        "best_eval_env_success": best_eval_env_success,
        "best_eval_return": best_eval_return,
        "best_eval_step": best_eval_step,
        "selection_metric": selection_metric,
        "final_replay_size": replay.size,
        "final_losses": last_loss,
        "final_lite_metrics": last_lite,
        "final_eval_return": last_eval_metrics.get("eval_return_mean", float("nan")),
        "final_eval_success": last_eval_metrics.get("eval_success_mean", float("nan")),
        "final_eval_goal_success": last_eval_metrics.get("eval_goal_success_mean", float("nan")),
        "final_eval_env_success": last_eval_metrics.get("eval_env_success_mean", float("nan")),
        "final_eval_goal_success_ever": last_eval_metrics.get("eval_goal_success_ever_mean", float("nan")),
        "final_eval_env_success_ever": last_eval_metrics.get("eval_env_success_ever_mean", float("nan")),
        "final_eval_goal_success_final": last_eval_metrics.get("eval_goal_success_final_mean", float("nan")),
        "final_eval_env_success_final": last_eval_metrics.get("eval_env_success_final_mean", float("nan")),
        "eval_AUC_success": _auc_success(eval_history_env),
        "first_hit_0.25": _first_hit(eval_history_env, 0.25),
        "first_hit_0.50": _first_hit(eval_history_env, 0.50),
        "first_hit_0.75": _first_hit(eval_history_env, 0.75),
        "first_hit_0.90": _first_hit(eval_history_env, 0.90),
        "runtime_total_sec": runtime_total_sec,
        "runtime_hours": runtime_total_sec / 3600.0,
        "final_steps_per_sec": float(global_step) / max(runtime_total_sec, 1e-8),
        "eval_AUC_success_per_hour": _auc_success(eval_history_env) / max(runtime_total_sec / 3600.0, 1e-8),
        "posthoc_correlation_metrics_online_computed": False,
        "diagnostic_log_samples": bool(config.diagnostic_log_samples),
        "diagnostic_freq": int(config.diagnostic_freq),
        "diagnostic_samples_per_log": int(config.diagnostic_samples_per_log),
        "diagnostic_full_batch_metrics": bool(config.diagnostic_full_batch_metrics),
        "diagnostic_samples_path": str(run_dir / "diagnostic_samples.jsonl") if config.diagnostic_log_samples else "",
        "diagnostic_metrics_path": str(run_dir / "diagnostic_metrics.jsonl") if config.diagnostic_log_samples else "",
        "diagnostic_target_convention": "TD3 td_abs_after uses the pre-update target y fixed before critic update; target actor/critic are not recomputed for after residual.",
        "diagnostic_rng_convention": "diagnostic sample subset uses an isolated diagnostic RNG; full-batch diagnostic_metrics.jsonl is preferred for paper metrics.",
    }
    if config.task_name == "push_t":
        summary.update(
            {
                "final_pushT_obj_goal_xy_dist_mean": last_eval_metrics.get(
                    "pushT/obj_goal_xy_dist_mean", float("nan")
                ),
                "final_pushT_obj_goal_rot_err_mean": last_eval_metrics.get(
                    "pushT/obj_goal_rot_err_mean", float("nan")
                ),
                "final_pushT_intersection_ratio_mean": last_eval_metrics.get(
                    "pushT/intersection_ratio_mean", float("nan")
                ),
                "final_pushT_contact_like_rate": last_eval_metrics.get("pushT/contact_like_rate", float("nan")),
                "final_pushT_pos_progress_rate": last_eval_metrics.get("pushT/pos_progress_rate", float("nan")),
                "final_pushT_rot_progress_rate": last_eval_metrics.get("pushT/rot_progress_rate", float("nan")),
            }
        )
    write_json(run_dir / "summary.json", summary)
    csv_writer.close()
    jsonl_writer.close()
    if diagnostic_samples_writer is not None:
        diagnostic_samples_writer.close()
    if diagnostic_metrics_writer is not None:
        diagnostic_metrics_writer.close()
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
        "lite/entropy_w_P": 0.0,
        "lite/ESS_w": 0.0,
        "lite/ESS_w_normalized": 0.0,
        "lite/support_available_rate": 0.0,
        "lite/psi_q_rank_rate": 0.0,
        "lite/q_candidate_fallback_rate": 1.0,
        "lite/psi_A_policy_rate": 0.0,
        "lite/mean_len_A_Q_sup": 0.0,
        "lite/mean_len_A_Q_replay_local": 0.0,
        "lite/local_support_count_mean": 0.0,
        "lite/local_support_count_p50": 0.0,
        "lite/local_support_count_p90": 0.0,
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
        "time/numba_support_scan": 0.0,
        "time_local_context_lookup": 0.0,
        "time_candidate_construction": 0.0,
        "time_q_policy_eval": 0.0,
        "time_q_candidate_eval": 0.0,
        "time_weight_computation": 0.0,
        "time_actor_critic_update": 0.0,
        "runtime/steps_per_sec": 0.0,
        "runtime/total_sec": 0.0,
        "runtime/overhead_vs_B0": np.nan,
        "runtime/support_cache_hit_rate": np.nan,
        "runtime/support_cache_miss_rate": np.nan,
        "runtime/batch_query_unique_rate": np.nan,
        "runtime/batch_query_coalescing_saving_rate": np.nan,
        "runtime/coarse_query_unique_rate": np.nan,
        "runtime/local_scan_cache_hit_rate": np.nan,
        "runtime/local_scan_cache_miss_rate": np.nan,
        "runtime/full_context_scan_rate": np.nan,
        "runtime/exact_negative_short_circuit_rate": np.nan,
        "runtime/count_short_circuit_rate": np.nan,
        "runtime/threshold_short_circuit_rate": np.nan,
        "runtime/coarse_count_short_circuit_rate": np.nan,
        "runtime/coarse_support_size_short_circuit_rate": np.nan,
        "runtime/scanned_count_short_circuit_rate": np.nan,
        "runtime/coarse_action_bin_count_lookup_rate": np.nan,
        "runtime/support_cache_attempt_rate": np.nan,
        "runtime/support_context_cache_size": np.nan,
        "runtime/local_scan_cache_attempt_rate": np.nan,
        "runtime/local_scan_cache_size": np.nan,
        "runtime/full_context_filter_count_mean": np.nan,
        "runtime/numba_support_scan_enabled": np.nan,
        "runtime/numba_scanner_warmed": np.nan,
        "runtime/numba_support_scan_fallback_rate": np.nan,
        "runtime/numba_support_queries": np.nan,
        "runtime/predicate_early_exit_rate": np.nan,
        "runtime/predicate_scanned_count_mean": np.nan,
        "runtime/diagnostic_audit_rate": np.nan,
        "runtime/audit_len_A_Q_local_abs_diff_mean": np.nan,
        "runtime/audit_len_A_Q_sup_abs_diff_mean": np.nan,
        "runtime/audit_psi_A_policy_mismatch_rate": np.nan,
        "mean_w_P_reference": np.nan,
        "reference_high_weight_rate": np.nan,
        "assigned_high_weight_rate": np.nan,
        "support_weight_alignment_rate": np.nan,
        "support_cache_hit_rate": np.nan,
        "short_circuit_rate": np.nan,
        "lite/mean_w_P_reference": np.nan,
        "lite/reference_high_weight_rate": np.nan,
        "lite/assigned_high_weight_rate": np.nan,
        "lite/support_weight_alignment_rate": np.nan,
        "lite/support_cache_hit_rate": np.nan,
        "lite/short_circuit_rate": np.nan,
        "lite/runtime_opt_exact_path": 0.0,
        "td_error/abs_mean": np.nan,
        "td_error/abs_p90": np.nan,
        "td_error/weight_mean": np.nan,
        "td_error/weight_max": np.nan,
        "td_error/weight_entropy": np.nan,
        "td_error/weight_ESS": np.nan,
        "per/priority_entropy": np.nan,
        "per/priority_ESS": np.nan,
        "per/weight_mean": np.nan,
        "per/weight_max": np.nan,
        "per/weight_entropy": np.nan,
        "per/weight_ESS": np.nan,
        "hgr/episode_priority_mean": np.nan,
        "hgr/episode_priority_std": np.nan,
        "hgr/episode_priority_max": np.nan,
        "hgr/eligible_episode_count": 0.0,
        "hgr/pair_table_count": 0.0,
        "hgr/pair_priority_count": 0.0,
        "hgr/pair_priority_memory_mb": 0.0,
        "hgr/goal_priority_entropy_mean": np.nan,
        "hgr/goal_selected_rank_mean": np.nan,
        "hgr/candidate_td_error_mean": np.nan,
        "hgr/candidate_td_error_max": np.nan,
        "hgr/relabel_fraction": 0.0,
        "hgr/fallback_fraction": 0.0,
        "hgr/episode_probability_mean": np.nan,
        "hgr/pair_probability_mean": np.nan,
        "hgr/is_weight_raw_mean": 1.0,
        "hgr/is_weight_raw_max": 1.0,
        "hgr/is_weight_mean": 1.0,
        "hgr/is_weight_max": 1.0,
        "hgr/is_weight_min": 1.0,
        "hgr/actor_weight_mean": 1.0,
        "hgr/beta_episode": np.nan,
        "hgr/beta_goal": np.nan,
        "hgr/beta_progress": 0.0,
        "hgr/beta_episode_next": np.nan,
        "hgr/beta_goal_next": np.nan,
        "hgr/beta_progress_next": 0.0,
        "hgr/max_priority": np.nan,
        "hgr/priority_clip_enabled": 0.0,
        "hgr/priority_clip_rate": 0.0,
        "hgr/priority_safety_replacement_rate": 0.0,
        "hgr/update_priority_mean": np.nan,
        "runtime/hgr_sampling_sec_per_update": 0.0,
        "b2/unrestricted_gap_mean": 0.0,
        "b2/unrestricted_gap_p95": 0.0,
        "b2/candidate_action_norm_mean": 0.0,
        "b2/action_delta_norm_mean": 0.0,
        "b2/q_imp_minus_q_policy_mean": 0.0,
        "b2/high_gap_rate": 0.0,
        "b2/gap_to_pos_progress_corr": np.nan,
        "b2/gap_to_rot_progress_corr": np.nan,
        "b2/high_gap_bad_progress_rate": np.nan,
        "b3/support_available_rate": 0.0,
        "b3/local_support_count_mean": 0.0,
        "b3/local_support_count_p50": 0.0,
        "b3/local_support_count_p90": 0.0,
        "b3/support_to_pos_progress_corr": np.nan,
        "b3/support_to_rot_progress_corr": np.nan,
        "b3/support_to_success_corr": np.nan,
        "b3/support_to_td_error_corr": np.nan,
        "b4/G_sup_mean": 0.0,
        "b4/G_sup_positive_rate": 0.0,
        "b4/support_candidate_count_mean": 0.0,
        "b4/support_candidate_count_p50": 0.0,
        "b4/support_candidate_count_p90": 0.0,
        "b4/fallback_rate": 1.0,
        "b4/local_scan_truncated_rate": 0.0,
        "b4/B2_gap_minus_B4_gap": np.nan,
        "b4/unsupported_gap_rate": np.nan,
    }
    for phase_name in phase_names:
        metrics[f"lite/phase_{phase_name}_mean_w_P"] = 0.0
        metrics[f"lite/phase_{phase_name}_fallback_rate"] = 0.0
        metrics[f"lite/phase_{phase_name}_psi_q_rank_rate"] = 0.0
        metrics[f"lite/phase_{phase_name}_positive_G_sup_rate"] = 0.0
        metrics[f"lite/phase_{phase_name}_count_rate"] = 0.0
        metrics[f"b3/support_by_phase_{phase_name}"] = 0.0
        metrics[f"b3/mean_w_P_by_phase_{phase_name}"] = 0.0
        metrics[f"b3/fallback_by_phase_{phase_name}"] = 0.0
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
    metrics = {
        "lite/mean_G_sup": _mean(wb.g_sup),
        "lite/positive_G_sup_rate": float(np.mean(wb.g_sup > 0.0)),
        "lite/mean_kappa": _mean(wb.kappa),
        "lite/mean_w_P": _mean(wb.w_p_raw),
        "lite/max_w_P": float(np.max(wb.w_p_raw)) if wb.w_p_raw.size else 1.0,
        "lite/std_w_P": float(np.std(wb.w_p_raw)) if wb.w_p_raw.size else 0.0,
        "lite/entropy_w_P": _entropy_weight(wb.w_p_raw, 1e-8),
        "lite/ESS_w": _ess_weight(wb.w_p_raw, 1e-8),
        "lite/ESS_w_normalized": _ess_weight(wb.w_p_raw, 1e-8) / max(1, int(wb.w_p_raw.size)),
        "lite/support_available_rate": _mean(wb.psi_q_rank),
        "lite/psi_q_rank_rate": _mean(wb.psi_q_rank),
        "lite/q_candidate_fallback_rate": _mean(wb.q_candidate_fallback),
        "lite/psi_A_policy_rate": _mean(wb.psi_a_policy),
        "lite/mean_len_A_Q_sup": _mean(wb.len_a_q_sup),
        "lite/mean_len_A_Q_replay_local": _mean(wb.len_a_q_local),
        "lite/local_support_count_mean": _mean(wb.len_a_q_local),
        "lite/local_support_count_p50": _nanpercentile(wb.len_a_q_local, 50.0),
        "lite/local_support_count_p90": _nanpercentile(wb.len_a_q_local, 90.0),
        "lite/mean_num_coarse_candidates": _mean(wb.num_coarse_candidates),
        "lite/mean_num_scanned_candidates": _mean(wb.num_scanned_candidates),
        "lite/local_scan_truncated_rate": _mean(wb.local_scan_truncated),
        "lite/mean_Q_imp_minus_Q_policy": _mean(wb.q_imp_minus_q_policy),
        "lite/unrestricted_gap_mean": _mean(wb.unrestricted_gap),
        "lite/gap_ready": float(wb.gap_ready),
        "lite/gap_tau": float(wb.gap_tau),
        "lite/gap_max": float(wb.gap_max),
        "b2/unrestricted_gap_mean": _mean(wb.unrestricted_gap),
        "b2/unrestricted_gap_p95": _nanpercentile(wb.unrestricted_gap, 95.0),
        "b2/candidate_action_norm_mean": float(getattr(wb, "extra_metrics", {}).get("b2/candidate_action_norm_mean", 0.0)),
        "b2/action_delta_norm_mean": float(getattr(wb, "extra_metrics", {}).get("b2/action_delta_norm_mean", 0.0)),
        "b2/q_imp_minus_q_policy_mean": _mean(wb.q_imp_minus_q_policy),
        "b2/high_gap_rate": float(np.mean(wb.unrestricted_gap > max(float(wb.gap_tau), 0.0))) if wb.unrestricted_gap.size else 0.0,
        "b2/gap_to_pos_progress_corr": np.nan,
        "b2/gap_to_rot_progress_corr": np.nan,
        "b2/high_gap_bad_progress_rate": np.nan,
        "b3/support_available_rate": _mean(wb.psi_q_rank),
        "b3/local_support_count_mean": _mean(wb.len_a_q_local),
        "b3/local_support_count_p50": _nanpercentile(wb.len_a_q_local, 50.0),
        "b3/local_support_count_p90": _nanpercentile(wb.len_a_q_local, 90.0),
        "b3/support_to_pos_progress_corr": np.nan,
        "b3/support_to_rot_progress_corr": np.nan,
        "b3/support_to_success_corr": np.nan,
        "b3/support_to_td_error_corr": np.nan,
        "b4/G_sup_mean": _mean(wb.g_sup),
        "b4/G_sup_positive_rate": float(np.mean(wb.g_sup > 0.0)) if wb.g_sup.size else 0.0,
        "b4/support_candidate_count_mean": _mean(wb.len_a_q_sup),
        "b4/support_candidate_count_p50": _nanpercentile(wb.len_a_q_sup, 50.0),
        "b4/support_candidate_count_p90": _nanpercentile(wb.len_a_q_sup, 90.0),
        "b4/fallback_rate": _mean(wb.q_candidate_fallback),
        "b4/local_scan_truncated_rate": _mean(wb.local_scan_truncated),
        "b4/B2_gap_minus_B4_gap": np.nan,
        "b4/unsupported_gap_rate": np.nan,
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
        if np.any(mask):
            metrics[f"lite/phase_{phase_name}_mean_w_P"] = float(np.mean(wb.w_p_raw[mask]))
            metrics[f"lite/phase_{phase_name}_fallback_rate"] = float(np.mean(wb.q_candidate_fallback[mask]))
            metrics[f"lite/phase_{phase_name}_psi_q_rank_rate"] = float(np.mean(wb.psi_q_rank[mask]))
            metrics[f"lite/phase_{phase_name}_positive_G_sup_rate"] = float(np.mean(wb.g_sup[mask] > 0.0))
            metrics[f"lite/phase_{phase_name}_count_rate"] = float(np.mean(mask))
            metrics[f"b3/support_by_phase_{phase_name}"] = float(np.mean(wb.psi_q_rank[mask]))
            metrics[f"b3/mean_w_P_by_phase_{phase_name}"] = float(np.mean(wb.w_p_raw[mask]))
            metrics[f"b3/fallback_by_phase_{phase_name}"] = float(np.mean(wb.q_candidate_fallback[mask]))
        else:
            metrics[f"lite/phase_{phase_name}_mean_w_P"] = 0.0
            metrics[f"lite/phase_{phase_name}_fallback_rate"] = 0.0
            metrics[f"lite/phase_{phase_name}_psi_q_rank_rate"] = 0.0
            metrics[f"lite/phase_{phase_name}_positive_G_sup_rate"] = 0.0
            metrics[f"lite/phase_{phase_name}_count_rate"] = 0.0
            metrics[f"b3/support_by_phase_{phase_name}"] = 0.0
            metrics[f"b3/mean_w_P_by_phase_{phase_name}"] = 0.0
            metrics[f"b3/fallback_by_phase_{phase_name}"] = 0.0
    for key, value in getattr(wb, "extra_metrics", {}).items():
        metrics[key] = float(value)
    return metrics


def _add_runtime(row: Dict[str, Any], wall_start: float, step: int) -> None:
    total_sec = max(0.0, float(time.perf_counter() - wall_start))
    row["runtime/total_sec"] = total_sec
    row["runtime/steps_per_sec"] = float(step) / max(total_sec, 1e-8)
    row["runtime/overhead_vs_B0"] = ""


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
    config: LiteConfig,
) -> Dict[str, Any]:
    return {
        "step": step,
        "episode": episode,
        "train/episode_return": ep_return,
        "train/episode_success": ep_success,
        "train/episode_len": ep_len,
        "train/replay_size": replay_size,
        "train/her_relabel_ratio": her_ratio,
        "config/lambda_support": config.lambda_support,
        "config/weight_scope": config.weight_scope,
        "loss/critic": loss["critic"],
        "loss/actor": loss["actor"],
        "loss/td_error_abs_mean": loss.get("td_error_abs_mean", 0.0),
        "loss/td_error_abs_p90": loss.get("td_error_abs_p90", 0.0),
        "sac/alpha": loss.get("sac_alpha", 0.0),
        "sac/alpha_loss": loss.get("sac_alpha_loss", 0.0),
        "sac/log_prob_mean": loss.get("sac_log_prob_mean", 0.0),
        "value/q_mean": loss["q_mean"],
        "value/target_mean": loss["target_mean"],
        "critic/q_policy_mean": loss.get("q_mean", 0.0),
        "critic/q_policy_std": loss.get("q_std", 0.0),
        "critic/q_target_mean": loss.get("target_mean", 0.0),
        "critic/q_target_std": loss.get("target_std", 0.0),
        "critic/q_abs_mean": loss.get("q_abs_mean", 0.0),
        "critic/q_abs_max": loss.get("q_abs_max", 0.0),
        **lite,
        "eval/return_mean": "",
        "eval/success_mean": "",
        "eval/goal_success_mean": "",
        "eval/env_success_mean": "",
        "eval/goal_success_ever_mean": "",
        "eval/goal_success_ever_std": "",
        "eval/env_success_ever_mean": "",
        "eval/env_success_ever_std": "",
        "eval/goal_success_final_mean": "",
        "eval/goal_success_final_std": "",
        "eval/env_success_final_mean": "",
        "eval/env_success_final_std": "",
        "eval/episode_length_mean": "",
        "eval/final_success": "",
        "eval/best_success": "",
        "eval/AUC_success": "",
        "eval/first_hit_0.25": "",
        "eval/first_hit_0.50": "",
        "eval/first_hit_0.75": "",
        "eval/first_hit_0.90": "",
    }
