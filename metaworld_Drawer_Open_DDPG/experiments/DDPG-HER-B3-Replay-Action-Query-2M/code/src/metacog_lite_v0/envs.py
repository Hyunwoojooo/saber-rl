from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np


@dataclass
class GoalObservation:
    observation: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray
    is_success: float

    def flat_state(self) -> np.ndarray:
        return np.concatenate([self.observation, self.desired_goal], axis=0).astype(np.float32)


class FetchGoalEnv:
    def __init__(
        self,
        env_id: str,
        max_episode_steps: int,
        seed: Optional[int],
        render: bool,
    ) -> None:
        gym.register_envs(gymnasium_robotics)
        render_mode = "human" if render else None
        self._env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
        self._seed = seed
        obs, info = self._env.reset(seed=seed)
        goal_obs = self._parse_obs(obs, info)
        self.obs_dim = int(goal_obs.observation.shape[0])
        self.goal_dim = int(goal_obs.desired_goal.shape[0])
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = int(self._env.action_space.shape[0])
        self.max_episode_steps = int(getattr(self._env, "_max_episode_steps", max_episode_steps))
        self.distance_threshold = float(getattr(self._env.unwrapped, "distance_threshold", 0.05))

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self._env.action_space.low, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self._env.action_space.high, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> GoalObservation:
        use_seed = self._seed if seed is None else seed
        obs, info = self._env.reset(seed=use_seed)
        return self._parse_obs(obs, info)

    def step(self, action: np.ndarray) -> Tuple[GoalObservation, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._parse_obs(obs, info), float(reward), bool(terminated), bool(truncated), info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        return np.asarray(
            self._env.unwrapped.compute_reward(achieved_goal, desired_goal, info or {}),
            dtype=np.float32,
        )

    def success_from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (dist <= self.distance_threshold).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        spec = gym.spec(self._env.unwrapped.spec.id)
        return {
            "env_id": self._env.unwrapped.spec.id,
            "entry_point": spec.entry_point if spec else None,
            "spec_kwargs": spec.kwargs if spec else {},
            "max_episode_steps": self.max_episode_steps,
            "distance_threshold": self.distance_threshold,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "render_mode": self._env.render_mode,
        }

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _parse_obs(obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> GoalObservation:
        return GoalObservation(
            observation=np.asarray(obs["observation"], dtype=np.float32).copy(),
            achieved_goal=np.asarray(obs["achieved_goal"], dtype=np.float32).copy(),
            desired_goal=np.asarray(obs["desired_goal"], dtype=np.float32).copy(),
            is_success=float(info.get("is_success", 0.0)),
        )


class MetaWorldDrawerOpenGoalEnv:
    """Goal-conditioned wrapper for MetaWorld drawer-open-v3.

    MetaWorld does not expose a Fetch-style GoalEnv API. For SABER/B3 we keep
    the same goal-conditioned replay/HER interface by treating the drawer
    handle position as achieved_goal and the task target position as
    desired_goal. Environment success is still read from MetaWorld's info dict
    when available.
    """

    def __init__(
        self,
        env_id: str,
        env_name: str,
        max_episode_steps: int,
        seed: Optional[int],
        render: bool,
        reward_type: str = "sparse",
        reward_function_version: str = "v2",
        distance_threshold: float = 0.05,
        terminate_on_success: bool = False,
    ) -> None:
        try:
            import metaworld
        except Exception as exc:  # pragma: no cover - depends on local install.
            raise ImportError(
                "MetaWorld is required for drawer_open. Install metaworld before running this experiment."
            ) from exc
        try:
            gym.register_envs(metaworld)
        except Exception:
            pass

        render_mode = "human" if render else None
        make_kwargs: Dict[str, Any] = {
            "env_name": env_name,
            "seed": seed,
            "max_episode_steps": max_episode_steps,
            "terminate_on_success": terminate_on_success,
        }
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        if reward_function_version:
            make_kwargs["reward_function_version"] = reward_function_version
        try:
            self._env = gym.make(env_id, **make_kwargs)
        except TypeError:
            make_kwargs.pop("reward_function_version", None)
            self._env = gym.make(env_id, **make_kwargs)

        self._seed = seed
        self.env_name = str(env_name)
        self.reward_type = str(reward_type).lower()
        self.distance_threshold = float(distance_threshold)
        self.desired_goal_source = "unknown"
        obs, info = self._reset_raw(seed=seed)
        goal_obs = self._parse_obs(obs, info)
        self.obs_dim = int(goal_obs.observation.shape[0])
        self.goal_dim = int(goal_obs.desired_goal.shape[0])
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = int(self._env.action_space.shape[0])
        self.max_episode_steps = int(getattr(self._env, "_max_episode_steps", max_episode_steps))

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self._env.action_space.low, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self._env.action_space.high, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> GoalObservation:
        use_seed = self._seed if seed is None else seed
        obs, info = self._reset_raw(seed=use_seed)
        return self._parse_obs(obs, info)

    def step(self, action: np.ndarray) -> Tuple[GoalObservation, float, bool, bool, Dict[str, Any]]:
        result = self._env.step(action)
        if len(result) == 5:
            obs, env_reward, terminated, truncated, info = result
        else:  # pragma: no cover - old MetaWorld API fallback.
            obs, env_reward, done, info = result
            terminated, truncated = bool(done), False
        goal_obs = self._parse_obs(obs, info)
        goal_success = float(self.success_from_goal(goal_obs.achieved_goal[None, :], goal_obs.desired_goal[None, :])[0])
        env_success = float(info.get("success", info.get("is_success", goal_success)))
        info = dict(info)
        info["goal_success"] = goal_success
        info["env_success"] = env_success
        info["is_success"] = env_success
        info["desired_goal_source"] = self.desired_goal_source
        if self.reward_type == "sparse":
            reward = float(self.compute_reward(goal_obs.achieved_goal[None, :], goal_obs.desired_goal[None, :], info)[0])
        else:
            reward = float(env_reward)
        return goal_obs, reward, bool(terminated), bool(truncated), info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        del info
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        dist = np.linalg.norm(achieved - desired, axis=-1)
        if self.reward_type == "dense":
            return (-dist).astype(np.float32)
        return np.where(dist <= self.distance_threshold, 0.0, -1.0).astype(np.float32)

    def success_from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return (dist <= self.distance_threshold).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        spec = getattr(getattr(self._env, "unwrapped", self._env), "spec", None)
        return {
            "env_id": getattr(spec, "id", "Meta-World/MT1"),
            "env_name": self.env_name,
            "entry_point": getattr(spec, "entry_point", None),
            "spec_kwargs": getattr(spec, "kwargs", {}),
            "max_episode_steps": self.max_episode_steps,
            "distance_threshold": self.distance_threshold,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "render_mode": getattr(self._env, "render_mode", None),
            "reward_type": self.reward_type,
            "desired_goal_source": self.desired_goal_source,
        }

    def close(self) -> None:
        self._env.close()

    def _reset_raw(self, seed: Optional[int]) -> tuple[np.ndarray, Dict[str, Any]]:
        result = self._env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:  # pragma: no cover - old MetaWorld API fallback.
            obs, info = result, {}
        return np.asarray(obs, dtype=np.float32), dict(info)

    def _parse_obs(self, obs: np.ndarray, info: Dict[str, Any]) -> GoalObservation:
        obs_arr = np.asarray(obs, dtype=np.float32).copy()
        achieved = self._handle_position(obs_arr)
        desired, desired_goal_source = self._target_position(obs_arr)
        self.desired_goal_source = desired_goal_source
        info["desired_goal_source"] = desired_goal_source
        env_success = float(info.get("success", info.get("is_success", 0.0)))
        return GoalObservation(
            observation=obs_arr,
            achieved_goal=achieved.astype(np.float32, copy=True),
            desired_goal=desired.astype(np.float32, copy=True),
            is_success=env_success,
        )

    @staticmethod
    def _handle_position(obs: np.ndarray) -> np.ndarray:
        if obs.shape[0] >= 7:
            return np.asarray(obs[4:7], dtype=np.float32)
        if obs.shape[0] >= 6:
            return np.asarray(obs[3:6], dtype=np.float32)
        raise ValueError(f"MetaWorld drawer-open observation too small for handle position: obs_dim={obs.shape[0]}")

    def _target_position(self, obs: np.ndarray) -> tuple[np.ndarray, str]:
        unwrapped = getattr(self._env, "unwrapped", self._env)
        for name in ("_target_pos", "target_pos", "_goal", "goal"):
            value = getattr(unwrapped, name, None)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                return arr[:3].astype(np.float32, copy=True), name
        if obs.shape[0] >= 3:
            tail = np.asarray(obs[-3:], dtype=np.float32)
            if np.all(np.isfinite(tail)) and np.linalg.norm(tail) > 1e-8:
                return tail.astype(np.float32, copy=True), "obs_tail"
        # Last resort: a synthetic "opened" target slightly behind the handle.
        handle = self._handle_position(obs)
        fallback = handle.copy()
        fallback[1] += 0.15
        return fallback.astype(np.float32), "synthetic_fallback"
