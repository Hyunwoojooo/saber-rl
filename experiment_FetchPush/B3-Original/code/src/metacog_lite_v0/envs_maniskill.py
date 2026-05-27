from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np

from .envs import GoalObservation


@dataclass
class UnitreeAppleFeatures:
    right_tcp_pos: np.ndarray
    apple_pos: np.ndarray
    bowl_pos: np.ndarray
    is_grasped: float
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass
class PickCubeFeatures:
    tcp_pos: np.ndarray
    cube_pos: np.ndarray
    goal_pos: np.ndarray
    is_grasped: float


class UnitreeAppleGoalEnv:
    """Goal wrapper for ManiSkill UnitreeG1PlaceAppleInBowl-v1.

    The actor/critic observation produced by this wrapper is goal-independent:
    [right_tcp_pos, apple_pos, tcp_to_apple_pos, is_grasped, qpos, qvel].
    The desired goal is appended only by GoalObservation.flat_state().
    """

    def __init__(
        self,
        env_id: str,
        max_episode_steps: int,
        seed: Optional[int],
        render: bool,
        obs_mode: str = "state",
        reward_type: str = "sparse",
        control_mode: Optional[str] = None,
        num_envs: int = 1,
        backend: Optional[str] = None,
        sim_backend: Optional[str] = None,
    ) -> None:
        if int(num_envs) != 1:
            raise ValueError("UnitreeAppleGoalEnv first implementation supports num_envs=1 only")
        try:
            import mani_skill.envs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mani_skill is required for UnitreeG1PlaceAppleInBowl-v1. "
                "Install ManiSkill on the DGX Spark environment before running this task."
            ) from exc

        render_mode = "human" if render else None
        reward_mode = "dense" if reward_type.lower() == "dense" else "sparse"
        kwargs: Dict[str, Any] = {
            "obs_mode": obs_mode,
            "reward_mode": reward_mode,
            "num_envs": 1,
            "render_mode": render_mode,
            "max_episode_steps": int(max_episode_steps),
        }
        if control_mode:
            kwargs["control_mode"] = control_mode
        if sim_backend:
            kwargs["sim_backend"] = sim_backend
        elif backend:
            kwargs["sim_backend"] = backend

        self._env = gym.make(env_id, **kwargs)
        self._seed = seed
        self.reward_type = reward_type.lower()
        self.distance_threshold = 0.05
        self.max_episode_steps = int(getattr(self._env, "_max_episode_steps", max_episode_steps))
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.backend = sim_backend or backend

        if not hasattr(self._env.action_space, "shape") or self._env.action_space.shape is None:
            raise TypeError("UnitreeAppleGoalEnv requires a Box action space; set a concrete control_mode")

        obs, info = self._env.reset(seed=seed)
        goal_obs = self._parse_obs(obs, info)
        self.obs_dim = int(goal_obs.observation.shape[0])
        self.goal_dim = int(goal_obs.desired_goal.shape[0])
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = int(np.prod(self._env.action_space.shape))

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self._env.action_space.low, dtype=np.float32).reshape(-1)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self._env.action_space.high, dtype=np.float32).reshape(-1)

    def reset(self, seed: Optional[int] = None) -> GoalObservation:
        use_seed = self._seed if seed is None else seed
        obs, info = self._env.reset(seed=use_seed)
        return self._parse_obs(obs, info)

    def step(self, action: np.ndarray) -> Tuple[GoalObservation, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action.reshape(self._env.action_space.shape))
        parsed = self._parse_obs(obs, info)
        info_out = dict(info)
        goal_success = float(parsed.is_success)
        env_success = _scalar_from_info(info, ("success", "env_success"), default=goal_success)
        info_out["goal_success"] = goal_success
        info_out["env_success"] = env_success
        info_out["is_success"] = goal_success
        return (
            parsed,
            float(_to_numpy(reward).reshape(-1)[0]),
            bool(_to_numpy(terminated).reshape(-1)[0]),
            bool(_to_numpy(truncated).reshape(-1)[0]),
            info_out,
        )

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return (-dist).astype(np.float32)
        return np.where(dist <= self.distance_threshold, 0.0, -1.0).astype(np.float32)

    def success_from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (dist <= self.distance_threshold).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "env_id": self._env.unwrapped.spec.id if self._env.unwrapped.spec else None,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "backend": self.backend,
            "max_episode_steps": self.max_episode_steps,
            "distance_threshold": self.distance_threshold,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "observation_layout": {
                "right_tcp_pos": [0, 3],
                "apple_pos": [3, 6],
                "tcp_to_apple_pos": [6, 9],
                "is_grasped": [9, 10],
                "qpos_start": 10,
                "qvel_after_qpos": True,
            },
        }

    def close(self) -> None:
        self._env.close()

    def _parse_obs(self, obs: Any, info: Dict[str, Any]) -> GoalObservation:
        features = self._extract_features(obs, info)
        tcp_to_apple = features.apple_pos - features.right_tcp_pos
        compact_obs = np.concatenate(
            [
                features.right_tcp_pos,
                features.apple_pos,
                tcp_to_apple,
                np.asarray([features.is_grasped], dtype=np.float32),
                features.qpos,
                features.qvel,
            ],
            axis=0,
        ).astype(np.float32)
        goal_success = float(
            np.linalg.norm(features.apple_pos - features.bowl_pos) <= self.distance_threshold
        )
        return GoalObservation(
            observation=compact_obs,
            achieved_goal=features.apple_pos.astype(np.float32),
            desired_goal=features.bowl_pos.astype(np.float32),
            is_success=goal_success,
        )

    def _extract_features(self, obs: Any, info: Dict[str, Any]) -> UnitreeAppleFeatures:
        flat = _flatten_obs(obs)
        right_tcp_pos = _first_not_none(
            _find_named_vector(flat, ("right_tcp", "tcp", "right_hand", "ee"), ("pos", "pose", "p")),
            _pose_from_attrs(self._env.unwrapped, ("right_tcp", "tcp", "right_hand_tcp", "right_ee")),
            _pose_from_attrs(getattr(self._env.unwrapped, "agent", None), ("right_tcp", "tcp", "right_ee")),
        )
        apple_pos = _first_not_none(
            _find_named_vector(flat, ("apple", "obj", "object"), ("pos", "pose", "p")),
            _pose_from_attrs(self._env.unwrapped, ("apple", "apple_actor", "obj", "object")),
        )
        bowl_pos = _first_not_none(
            _find_named_vector(flat, ("bowl", "goal", "target"), ("pos", "pose", "p"), exclude=("to", "delta")),
            _pose_from_attrs(self._env.unwrapped, ("bowl", "bowl_actor", "goal_site", "target")),
        )
        if right_tcp_pos is None or apple_pos is None or bowl_pos is None:
            missing = [
                name
                for name, value in (
                    ("right_tcp_pos", right_tcp_pos),
                    ("apple_pos", apple_pos),
                    ("bowl_pos", bowl_pos),
                )
                if value is None
            ]
            raise KeyError(
                "Could not parse UnitreeApple observation fields: "
                f"{missing}. Use obs_mode=state_dict for debugging and expose tcp/apple/bowl positions."
            )

        is_grasped = _scalar_from_info(info, ("is_grasped", "grasped"), default=np.nan)
        if not np.isfinite(is_grasped):
            grasp_value = _find_named_scalar(flat, ("is_grasped", "grasped", "grasp"))
            is_grasped = 0.0 if grasp_value is None else float(grasp_value)

        qpos = _first_not_none(_find_named_vector(flat, ("qpos",), ()), np.zeros((0,), dtype=np.float32))
        qvel = _first_not_none(_find_named_vector(flat, ("qvel",), ()), np.zeros((0,), dtype=np.float32))
        return UnitreeAppleFeatures(
            right_tcp_pos=np.asarray(right_tcp_pos, dtype=np.float32).reshape(-1)[:3],
            apple_pos=np.asarray(apple_pos, dtype=np.float32).reshape(-1)[:3],
            bowl_pos=np.asarray(bowl_pos, dtype=np.float32).reshape(-1)[:3],
            is_grasped=float(is_grasped),
            qpos=np.asarray(qpos, dtype=np.float32).reshape(-1),
            qvel=np.asarray(qvel, dtype=np.float32).reshape(-1),
        )


class ManiSkillPickCubeGoalEnv:
    """Goal wrapper for ManiSkill PickCube-v1.

    The actor/critic observation is compact and goal-independent:
    [tcp_pos(3), cube_pos(3), tcp_to_cube_pos(3), is_grasped(1)].
    The desired goal is appended only by GoalObservation.flat_state().
    """

    def __init__(
        self,
        env_id: str,
        max_episode_steps: int,
        seed: Optional[int],
        render: bool,
        obs_mode: str = "state",
        reward_type: str = "sparse",
        robot_uids: str = "panda",
        control_mode: Optional[str] = None,
        num_envs: int = 1,
        backend: Optional[str] = None,
        sim_backend: Optional[str] = None,
    ) -> None:
        if int(num_envs) != 1:
            raise ValueError("ManiSkillPickCubeGoalEnv first implementation supports num_envs=1 only")
        try:
            import mani_skill.envs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mani_skill is required for PickCube-v1. Install ManiSkill before running this task."
            ) from exc

        render_mode = "human" if render else None
        reward_mode = "dense" if reward_type.lower() == "dense" else "sparse"
        kwargs: Dict[str, Any] = {
            "obs_mode": obs_mode,
            "reward_mode": reward_mode,
            "num_envs": 1,
            "render_mode": render_mode,
            "max_episode_steps": int(max_episode_steps),
            "robot_uids": robot_uids,
        }
        if control_mode:
            kwargs["control_mode"] = control_mode
        if sim_backend:
            kwargs["sim_backend"] = sim_backend
        elif backend:
            kwargs["sim_backend"] = backend

        self._env = gym.make(env_id, **kwargs)
        self._seed = seed
        self.reward_type = reward_type.lower()
        self.obs_mode = obs_mode
        self.robot_uids = robot_uids
        self.control_mode = control_mode
        self.backend = sim_backend or backend
        self.max_episode_steps = int(getattr(self._env, "_max_episode_steps", max_episode_steps))
        self.distance_threshold = self._resolve_goal_threshold(default=0.025)

        if not hasattr(self._env.action_space, "shape") or self._env.action_space.shape is None:
            raise TypeError("ManiSkillPickCubeGoalEnv requires a Box action space; set a concrete control_mode")

        obs, info = self._env.reset(seed=seed)
        goal_obs = self._parse_obs(obs, info)
        self.initial_cube_z = float(goal_obs.achieved_goal[2])
        self.z_lift_min = self.initial_cube_z + 0.05
        self.z_lift_min_source = "initial_cube_z_plus_0.05"
        self.obs_dim = int(goal_obs.observation.shape[0])
        self.goal_dim = int(goal_obs.desired_goal.shape[0])
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = int(np.prod(self._env.action_space.shape))

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self._env.action_space.low, dtype=np.float32).reshape(-1)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self._env.action_space.high, dtype=np.float32).reshape(-1)

    def reset(self, seed: Optional[int] = None) -> GoalObservation:
        use_seed = self._seed if seed is None else seed
        obs, info = self._env.reset(seed=use_seed)
        return self._parse_obs(obs, info)

    def step(self, action: np.ndarray) -> Tuple[GoalObservation, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action.reshape(self._env.action_space.shape))
        parsed = self._parse_obs(obs, info)
        info_out = dict(info)
        goal_success = float(parsed.is_success)
        env_success = _scalar_from_info(info, ("success", "env_success"), default=goal_success)
        info_out["goal_success"] = goal_success
        info_out["env_success"] = env_success
        info_out["is_success"] = goal_success
        return (
            parsed,
            float(_to_numpy(reward).reshape(-1)[0]),
            bool(_to_numpy(terminated).reshape(-1)[0]),
            bool(_to_numpy(truncated).reshape(-1)[0]),
            info_out,
        )

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return (-dist).astype(np.float32)
        return np.where(dist <= self.distance_threshold, 0.0, -1.0).astype(np.float32)

    def success_from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (dist <= self.distance_threshold).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "env_id": self._env.unwrapped.spec.id if self._env.unwrapped.spec else None,
            "robot_uids": self.robot_uids,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "backend": self.backend,
            "max_episode_steps": self.max_episode_steps,
            "distance_threshold": self.distance_threshold,
            "initial_cube_z": self.initial_cube_z,
            "z_lift_min": self.z_lift_min,
            "z_lift_min_source": self.z_lift_min_source,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "observation_layout": {
                "tcp_pos": [0, 3],
                "cube_pos": [3, 6],
                "tcp_to_cube_pos": [6, 9],
                "is_grasped": [9, 10],
            },
            "observation_goal_independent": True,
        }

    def close(self) -> None:
        self._env.close()

    def _parse_obs(self, obs: Any, info: Dict[str, Any]) -> GoalObservation:
        features = self._extract_features(obs, info)
        tcp_to_cube = features.cube_pos - features.tcp_pos
        compact_obs = np.concatenate(
            [
                features.tcp_pos,
                features.cube_pos,
                tcp_to_cube,
                np.asarray([features.is_grasped], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        goal_success = float(np.linalg.norm(features.cube_pos - features.goal_pos) <= self.distance_threshold)
        return GoalObservation(
            observation=compact_obs,
            achieved_goal=features.cube_pos.astype(np.float32),
            desired_goal=features.goal_pos.astype(np.float32),
            is_success=goal_success,
        )

    def _extract_features(self, obs: Any, info: Dict[str, Any]) -> PickCubeFeatures:
        flat = _flatten_obs(obs)
        tcp_pos = _first_not_none(
            _find_named_vector(flat, ("tcp", "ee", "end_effector"), ("pos", "pose", "p")),
            _pose_from_attrs(self._env.unwrapped, ("tcp", "tcp_link", "ee_link", "end_effector")),
            _pose_from_attrs(getattr(self._env.unwrapped, "agent", None), ("tcp", "tcp_link", "ee_link")),
        )
        cube_pos = _first_not_none(
            _find_named_vector(flat, ("cube", "obj", "object"), ("pos", "pose", "p"), exclude=("to", "delta")),
            _pose_from_attrs(self._env.unwrapped, ("cube", "obj", "object")),
        )
        goal_pos = _first_not_none(
            _find_named_vector(
                flat,
                ("goal", "target", "goal_site"),
                ("pos", "pose", "p"),
                exclude=("to", "delta", "obj_to_goal", "cube_to_goal"),
            ),
            _pose_from_attrs(self._env.unwrapped, ("goal_site", "goal", "target")),
        )
        if tcp_pos is None or cube_pos is None or goal_pos is None:
            missing = [
                name
                for name, value in (
                    ("tcp_pos", tcp_pos),
                    ("cube_pos", cube_pos),
                    ("goal_pos", goal_pos),
                )
                if value is None
            ]
            keys = sorted(flat.keys())
            raise KeyError(
                "Could not parse PickCube observation fields: "
                f"{missing}. Available flattened observation keys={keys[:40]}"
            )

        is_grasped = _scalar_from_info(info, ("is_grasped", "grasped"), default=np.nan)
        if not np.isfinite(is_grasped):
            grasp_value = _find_named_scalar(flat, ("is_grasped", "grasped", "grasp"))
            is_grasped = 0.0 if grasp_value is None else float(grasp_value)

        return PickCubeFeatures(
            tcp_pos=np.asarray(tcp_pos, dtype=np.float32).reshape(-1)[:3],
            cube_pos=np.asarray(cube_pos, dtype=np.float32).reshape(-1)[:3],
            goal_pos=np.asarray(goal_pos, dtype=np.float32).reshape(-1)[:3],
            is_grasped=float(is_grasped),
        )

    def _resolve_goal_threshold(self, default: float) -> float:
        for root in (self._env.unwrapped, self._env):
            for name in ("goal_thresh", "goal_threshold", "cube_goal_thresh", "distance_threshold"):
                value = getattr(root, name, None)
                if value is None:
                    continue
                arr = _to_numpy(value).reshape(-1)
                if arr.size:
                    return float(arr[0])
        return float(default)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _squeeze_batch(value: Any) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.reshape(-1)


def _flatten_obs(obs: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        out: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten_obs(value, child))
        return out
    return {prefix: _squeeze_batch(obs)}


def _find_named_vector(
    flat: Dict[str, np.ndarray],
    name_tokens: Iterable[str],
    value_tokens: Iterable[str],
    exclude: Iterable[str] = (),
) -> Optional[np.ndarray]:
    names = tuple(t.lower() for t in name_tokens)
    values = tuple(t.lower() for t in value_tokens)
    excludes = tuple(t.lower() for t in exclude)
    for key, value in flat.items():
        key_l = key.lower()
        if excludes and any(token in key_l for token in excludes):
            continue
        if not any(token in key_l for token in names):
            continue
        if values and not any(token in key_l for token in values):
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    return None


def _find_named_scalar(flat: Dict[str, np.ndarray], name_tokens: Iterable[str]) -> Optional[float]:
    for key, value in flat.items():
        key_l = key.lower()
        if any(token in key_l for token in name_tokens):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size:
                return float(arr[0])
    return None


def _scalar_from_info(info: Dict[str, Any], keys: Iterable[str], default: float) -> float:
    for key in keys:
        if key in info:
            arr = _to_numpy(info[key]).reshape(-1)
            if arr.size:
                return float(arr[0])
    return float(default)


def _first_not_none(*values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    for value in values:
        if value is not None:
            return value
    return None


def _pose_from_attrs(root: Any, names: Iterable[str]) -> Optional[np.ndarray]:
    if root is None:
        return None
    for name in names:
        obj = getattr(root, name, None)
        if obj is None:
            continue
        pose = getattr(obj, "pose", None)
        if pose is None and hasattr(obj, "get_pose"):
            pose = obj.get_pose()
        pos = getattr(pose, "p", None) if pose is not None else getattr(obj, "p", None)
        if pos is not None:
            arr = _squeeze_batch(pos)
            if arr.size >= 3:
                return arr[:3]
    return None
