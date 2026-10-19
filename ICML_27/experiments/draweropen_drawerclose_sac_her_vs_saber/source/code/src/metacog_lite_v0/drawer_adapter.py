"""Drawer semantics from the frozen DDPG wrapper, independent of the SAC update."""
import numpy as np

from .envs import MetaWorldDrawerOpenGoalEnv, MetaWorldDrawerCloseGoalEnv
from .task_drawer_open import MetaWorldDrawerOpenTaskSpec
from .task_drawer_close import MetaWorldDrawerCloseTaskSpec


def make_drawer_env(config, seed, render):
    cls = MetaWorldDrawerOpenGoalEnv if config.task_name == "drawer_open" else MetaWorldDrawerCloseGoalEnv
    env = cls(env_id=config.env_id, env_name=config.metaworld_env_name,
              max_episode_steps=config.max_episode_steps, seed=seed, render=render,
              reward_type=config.reward_type,
              reward_function_version=config.metaworld_reward_function_version,
              distance_threshold=config.drawer_success_threshold,
              terminate_on_success=config.success_terminates)
    if env.desired_goal_source not in {"_target_pos", "target_pos", "_goal", "goal"}:
        env.close()
        raise RuntimeError("Drawer requires an environment-provided target; no synthetic goal fallback")
    if env.action_dim != 4 or env.max_episode_steps != config.max_episode_steps:
        env.close()
        raise RuntimeError("Drawer action dimension or horizon does not match the protocol")
    return env


def make_drawer_task(config, env):
    cls = MetaWorldDrawerOpenTaskSpec if config.task_name == "drawer_open" else MetaWorldDrawerCloseTaskSpec
    return cls(action_low=env.action_low, action_high=env.action_high,
               epsilon_g=env.distance_threshold, d_reach=config.drawer_d_reach,
               d_goal_near=config.drawer_d_goal_near, d_opening_mid=config.drawer_d_opening_mid,
               hand_handle_bins=config.drawer_hand_handle_bins,
               handle_goal_bins=config.drawer_handle_goal_bins, handle_y_bins=config.drawer_handle_y_bins,
               gripper_bins=config.drawer_gripper_bins, horizon_bins=config.drawer_horizon_bins)


def nonterminal_mask(config, batch, success):
    terminal = batch.done.astype(np.float32) * (1.0 - batch.timeout.astype(np.float32))
    if not config.timeout_bootstrap:
        terminal = np.maximum(terminal, batch.timeout.astype(np.float32))
    if config.success_terminates:
        terminal = np.maximum(terminal, success.astype(np.float32))
    return (1.0 - terminal).astype(np.float32)
