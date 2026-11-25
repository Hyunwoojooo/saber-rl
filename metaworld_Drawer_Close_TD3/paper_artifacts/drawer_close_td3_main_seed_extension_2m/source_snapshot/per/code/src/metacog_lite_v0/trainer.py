from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .agent import DDPGAgent
from .config import LiteConfig
from .envs import FetchGoalEnv, MetaWorldDrawerCloseGoalEnv, MetaWorldDrawerOpenGoalEnv
from .envs_maniskill import ManiSkillPickCubeGoalEnv, ManiSkillPushTGoalEnv, UnitreeAppleGoalEnv
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
from .task_drawer_open import MetaWorldDrawerOpenTaskSpec
from .task_drawer_close import MetaWorldDrawerCloseTaskSpec
from .task_pick_cube import PickCubeTaskSpec
from .task_push_t import PushTTaskSpec
from .task_unitree_apple import UnitreeAppleTaskSpec
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
    if config.task_name in {"fetch_push", "fetch_pick_and_place", "fetch_slide"}:
        return FetchGoalEnv(config.env_id, config.max_episode_steps, seed, render)
    if config.task_name == "drawer_open":
        return MetaWorldDrawerOpenGoalEnv(
            env_id=config.env_id,
            env_name=config.metaworld_env_name,
            max_episode_steps=config.max_episode_steps,
            seed=seed,
            render=render,
            reward_type=config.reward_type,
            reward_function_version=config.metaworld_reward_function_version,
            distance_threshold=config.drawer_success_threshold,
            terminate_on_success=config.success_terminates,
        )
    if config.task_name == "drawer_close":
        return MetaWorldDrawerCloseGoalEnv(
            env_id=config.env_id,
            env_name=config.metaworld_env_name,
            max_episode_steps=config.max_episode_steps,
            seed=seed,
            render=render,
            reward_type=config.reward_type,
            reward_function_version=config.metaworld_reward_function_version,
            distance_threshold=config.drawer_success_threshold,
            terminate_on_success=config.success_terminates,
        )
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
    if config.task_name == "drawer_open":
        return MetaWorldDrawerOpenTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            epsilon_g=env.distance_threshold,
            d_reach=config.drawer_d_reach,
            d_goal_near=config.drawer_d_goal_near,
            d_opening_mid=config.drawer_d_opening_mid,
            hand_handle_bins=config.drawer_hand_handle_bins,
            handle_goal_bins=config.drawer_handle_goal_bins,
            handle_y_bins=config.drawer_handle_y_bins,
            gripper_bins=config.drawer_gripper_bins,
            horizon_bins=config.drawer_horizon_bins,
        )
    if config.task_name == "drawer_close":
        return MetaWorldDrawerCloseTaskSpec(
            action_low=env.action_low,
            action_high=env.action_high,
            epsilon_g=env.distance_threshold,
            d_reach=config.drawer_d_reach,
            d_goal_near=config.drawer_d_goal_near,
            d_opening_mid=config.drawer_d_opening_mid,
            hand_handle_bins=config.drawer_hand_handle_bins,
            handle_goal_bins=config.drawer_handle_goal_bins,
            handle_y_bins=config.drawer_handle_y_bins,
            gripper_bins=config.drawer_gripper_bins,
            horizon_bins=config.drawer_horizon_bins,
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


def per_beta_at_step(config: LiteConfig, global_step: int) -> float:
    progress = min(1.0, max(0.0, float(global_step) / max(1.0, float(config.per_beta_steps))))
    return float(config.per_beta_start) + (float(config.per_beta_end) - float(config.per_beta_start)) * progress


def _apply_per_full_loss_weights(config: LiteConfig, wb: Any, batch: Any) -> None:
    batch_size = int(batch.indices.shape[0])
    if batch.is_weight is None or batch.sampling_priority is None or batch.sampling_prob is None:
        raise ValueError("Full PER requires prioritized replay batch metadata")

    eps = float(config.eps)
    is_weight = np.asarray(batch.is_weight, dtype=np.float32).reshape(-1)
    priority = np.asarray(batch.sampling_priority, dtype=np.float32).reshape(-1)
    sample_prob = np.asarray(batch.sampling_prob, dtype=np.float32).reshape(-1)
    if is_weight.size != batch_size or priority.size != batch_size or sample_prob.size != batch_size:
        raise ValueError("Full PER batch metadata shape mismatch")
    if not (
        np.all(np.isfinite(is_weight))
        and np.all(np.isfinite(priority))
        and np.all(np.isfinite(sample_prob))
        and np.all(is_weight > 0.0)
        and np.all(priority > 0.0)
        and np.all(sample_prob > 0.0)
    ):
        raise ValueError("Full PER batch metadata must be finite and positive")

    w_raw = np.clip(is_weight, 0.0, float(config.U_PER_max)).astype(np.float32)
    if config.weighting_mode == "raw":
        omega = w_raw.copy()
    else:
        omega = (w_raw / max(eps, float(np.mean(w_raw)))).astype(np.float32)
    wb.w_p_raw = w_raw
    wb.omega_critic = omega
    wb.omega_actor = np.ones(batch_size, dtype=np.float32)
    wb.extra_metrics.update(
        {
            "per/beta": float(batch.per_beta),
            "per/sampling_prob_mean": float(np.mean(sample_prob)),
            "per/sampling_prob_min": float(np.min(sample_prob)),
            "per/sampling_prob_max": float(np.max(sample_prob)),
            "per/priority_mean": float(np.mean(priority)),
            "per/priority_max": float(np.max(priority)),
            "per/priority_entropy": _entropy_weight(priority, eps),
            "per/priority_ESS": _ess_weight(priority, eps),
            "per/weight_mean": float(np.mean(w_raw)),
            "per/weight_max": float(np.max(w_raw)),
            "per/weight_entropy": _entropy_weight(w_raw, eps),
            "per/weight_ESS": _ess_weight(w_raw, eps),
            "per/sample_duplicate_rate": 1.0
            - float(np.unique(batch.indices).size / max(1, batch_size)),
        }
    )


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
        "mean_w_P_reference",
        "reference_high_weight_rate",
        "assigned_high_weight_rate",
        "support_weight_alignment_rate",
        "support_weight_overlap_rate",
        "support_cache_hit_rate",
        "short_circuit_rate",
        "lite/mean_w_P_reference",
        "lite/reference_high_weight_rate",
        "lite/assigned_high_weight_rate",
        "lite/support_weight_alignment_rate",
        "lite/support_weight_overlap_rate",
        "lite/support_cache_hit_rate",
        "lite/short_circuit_rate",
        "lite/runtime_opt_exact_path",
        "td_error/abs_mean",
        "td_error/abs_p90",
        "td_error/weight_mean",
        "td_error/weight_max",
        "td_error/weight_entropy",
        "td_error/weight_ESS",
        "per/beta",
        "per/sampling_prob_mean",
        "per/sampling_prob_min",
        "per/sampling_prob_max",
        "per/priority_mean",
        "per/priority_max",
        "per/priority_entropy",
        "per/priority_ESS",
        "per/weight_mean",
        "per/weight_max",
        "per/weight_entropy",
        "per/weight_ESS",
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
        "hand_handle_dist_mean",
        "handle_goal_dist_mean",
        "phase_approach_frac",
        "phase_contact_frac",
        "phase_opening_frac",
        "phase_open_frac",
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

    noise_rng = np.random.default_rng(config.seeds.action_noise_seed)
    sample_rng = np.random.default_rng(config.seeds.numpy_seed + 17)
    weight_rng = np.random.default_rng(config.seeds.weight_seed)
    start_step = 0
    episode_idx = 0
    total_updates = 0
    selection_metric = (
        "eval_env_success_mean" if config.task_name in {"push_t", "drawer_open", "drawer_close"} else "eval_goal_success_mean"
    )
    best_eval_success = -1.0
    best_eval_selection_success = -1.0
    best_eval_goal_success = -1.0
    best_eval_env_success = -1.0
    best_eval_return = float("-inf")
    best_eval_step = -1
    last_eval_metrics: Dict[str, float] = {}
    eval_history_env: list[tuple[int, float]] = []

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
        start_step = int(payload["global_step"]) + 1
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
                if config.method == "td3_her_per_full":
                    batch = replay.sample_prioritized(config.batch_size, config.her_ratio, sample_rng, alpha=config.per_alpha, beta=per_beta_at_step(config, global_step), eps=config.eps_PER)
                else:
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

                wb = weight_computer.compute(batch, replay, agent, weight_rng, device)
                state_np = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
                next_state_np = np.concatenate([batch.next_observation, batch.desired_goal_train], axis=1).astype(np.float32)
                state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
                action_t = torch.as_tensor(batch.action_exec, dtype=torch.float32, device=device)
                reward_t = torch.as_tensor(reward_train, dtype=torch.float32, device=device)
                next_state_t = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)
                nonterminal_t = torch.as_tensor(nonterminal, dtype=torch.float32, device=device)
                if config.method == "td3_her_per_full":
                    _apply_per_full_loss_weights(config, wb, batch)
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

                t_update0 = time.perf_counter()
                update = agent.update(
                    state=state_t,
                    action_exec=action_t,
                    reward=reward_t,
                    next_state=next_state_t,
                    nonterminal=nonterminal_t,
                    omega_critic=torch.as_tensor(wb.omega_critic, dtype=torch.float32, device=device),
                    omega_actor=torch.as_tensor(wb.omega_actor, dtype=torch.float32, device=device),
                )
                if config.method == "td3_her_per_full":
                    replay.update_priorities(batch.indices, update.td_error_abs_values, eps=config.eps_PER)
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
                }
                last_lite = _lite_metrics_from_weight_batch(wb, phase_names)
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

    runtime_total_sec = float(time.perf_counter() - wall_start)
    summary = {
        "run_dir": str(run_dir),
        "method": config.method,
        "full_per": {
            "enabled": config.method in {"ddpg_her_per_full", "td3_her_per_full"},
            "sampling_order": "slot_priority_then_future_goal_relabeling",
            "priority_update": "mean_absolute_twin_critic_td_error",
            "alpha": config.per_alpha,
            "beta_start": config.per_beta_start,
            "beta_end": config.per_beta_end,
            "beta_steps": config.per_beta_steps,
            "priority_epsilon": config.eps_PER,
            "importance_sampling_scope": "critic_only",
            "minibatch_mean_normalized": config.weighting_mode == "batch_norm",
            "support_lookup": False,
        },
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
        "final_steps_per_sec": float(global_step) / max(runtime_total_sec, 1e-8),
        "posthoc_correlation_metrics_online_computed": False,
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
        "support_weight_overlap_rate": np.nan,
        "support_cache_hit_rate": np.nan,
        "short_circuit_rate": np.nan,
        "lite/mean_w_P_reference": np.nan,
        "lite/reference_high_weight_rate": np.nan,
        "lite/assigned_high_weight_rate": np.nan,
        "lite/support_weight_alignment_rate": np.nan,
        "lite/support_weight_overlap_rate": np.nan,
        "lite/support_cache_hit_rate": np.nan,
        "lite/short_circuit_rate": np.nan,
        "lite/runtime_opt_exact_path": 0.0,
        "td_error/abs_mean": np.nan,
        "td_error/abs_p90": np.nan,
        "td_error/weight_mean": np.nan,
        "td_error/weight_max": np.nan,
        "td_error/weight_entropy": np.nan,
        "td_error/weight_ESS": np.nan,
        "per/beta": np.nan,
        "per/sampling_prob_mean": np.nan,
        "per/sampling_prob_min": np.nan,
        "per/sampling_prob_max": np.nan,
        "per/priority_mean": np.nan,
        "per/priority_max": np.nan,
        "per/priority_entropy": np.nan,
        "per/priority_ESS": np.nan,
        "per/weight_mean": np.nan,
        "per/weight_max": np.nan,
        "per/weight_entropy": np.nan,
        "per/weight_ESS": np.nan,
        "per/sample_duplicate_rate": np.nan,
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
    metrics["hand_handle_dist_mean"] = 0.0
    metrics["handle_goal_dist_mean"] = 0.0
    metrics["phase_approach_frac"] = 0.0
    metrics["phase_contact_frac"] = 0.0
    metrics["phase_opening_frac"] = 0.0
    metrics["phase_open_frac"] = 0.0
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
