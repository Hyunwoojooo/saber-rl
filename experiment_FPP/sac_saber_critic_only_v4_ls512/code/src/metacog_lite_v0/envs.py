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
