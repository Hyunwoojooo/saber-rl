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


@dataclass
class PushTFeatures:
    tcp_pos: np.ndarray
    obj_pos: np.ndarray
    goal_pos: np.ndarray
    obj_theta: float
    goal_theta: float
    obj_velocity_xy: np.ndarray
    obj_ang_vel_z: float
    tcp_velocity_xy: np.ndarray
    intersection_ratio: float
    goal_theta_source: str
    goal_theta_warning: str
    velocity_available: bool


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


class ManiSkillPushTGoalEnv:
    """Goal wrapper for ManiSkill PushT-v1.

    The actor/critic observation is compact and goal-independent:
    [tcp_pos(3), obj_pos(3), obj_cos(theta), obj_sin(theta),
     tcp_to_obj_pos(3), obj_velocity_xy(2), obj_ang_vel_z(1), tcp_velocity_xy(2)].
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
        robot_uids: str = "panda_stick",
        control_mode: Optional[str] = None,
        num_envs: int = 1,
        backend: Optional[str] = None,
        sim_backend: Optional[str] = None,
        pos_success_thresh: float = 0.05,
        rot_success_thresh: float = 0.2617993877991494,
        rot_weight: float = 0.25,
        contact_thresh: float = 0.06,
    ) -> None:
        if int(num_envs) != 1:
            raise ValueError("ManiSkillPushTGoalEnv first implementation supports num_envs=1 only")
        try:
            import mani_skill.envs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mani_skill is required for PushT-v1. Install ManiSkill before running this task."
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
        self.pos_success_thresh = float(pos_success_thresh)
        self.rot_success_thresh = float(rot_success_thresh)
        self.rot_weight = float(rot_weight)
        self.contact_thresh = float(contact_thresh)
        self.distance_threshold = self.pos_success_thresh
        self._goal_theta_source = "unknown"
        self._goal_theta_warning = ""
        self._velocity_available = False
        self._last_obs_keys: list[str] = []
        self.compute_intersection_ratio = False

        if not hasattr(self._env.action_space, "shape") or self._env.action_space.shape is None:
            raise TypeError("ManiSkillPushTGoalEnv requires a Box action space; set a concrete control_mode")

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

    def set_compute_intersection_ratio(self, enabled: bool) -> None:
        self.compute_intersection_ratio = bool(enabled)

    def reset(self, seed: Optional[int] = None) -> GoalObservation:
        use_seed = self._seed if seed is None else seed
        obs, info = self._env.reset(seed=use_seed)
        return self._parse_obs(obs, info)

    def step(self, action: np.ndarray) -> Tuple[GoalObservation, float, bool, bool, Dict[str, Any]]:
        obs, _env_reward, terminated, truncated, info = self._env.step(action.reshape(self._env.action_space.shape))
        parsed = self._parse_obs(obs, info)
        reward = float(self.compute_reward(parsed.achieved_goal[None, :], parsed.desired_goal[None, :], info)[0])
        info_out = dict(info)
        goal_success = float(parsed.is_success)
        env_success = _scalar_from_info(info, ("success", "env_success"), default=goal_success)
        info_out.update(
            self._push_t_info(
                parsed.observation,
                parsed.desired_goal,
                info,
                compute_intersection=self.compute_intersection_ratio,
            )
        )
        info_out["goal_success"] = goal_success
        info_out["env_success"] = env_success
        info_out["is_success"] = goal_success
        return (
            parsed,
            reward,
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
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        pos_dist = np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1)
        rot_err = _angle_error_from_cossin(
            achieved[..., 2],
            achieved[..., 3],
            desired[..., 2],
            desired[..., 3],
        )
        if self.reward_type == "dense":
            return (-pos_dist - self.rot_weight * rot_err).astype(np.float32)
        success = np.logical_and(pos_dist <= self.pos_success_thresh, rot_err <= self.rot_success_thresh)
        return np.where(success, 0.0, -1.0).astype(np.float32)

    def success_from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        pos_dist = np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1)
        rot_err = _angle_error_from_cossin(
            achieved[..., 2],
            achieved[..., 3],
            desired[..., 2],
            desired[..., 3],
        )
        return np.logical_and(pos_dist <= self.pos_success_thresh, rot_err <= self.rot_success_thresh).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "env_id": self._env.unwrapped.spec.id if self._env.unwrapped.spec else None,
            "robot_uids": self.robot_uids,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "backend": self.backend,
            "max_episode_steps": self.max_episode_steps,
            "pos_success_thresh": self.pos_success_thresh,
            "rot_success_thresh_rad": self.rot_success_thresh,
            "rot_success_thresh_deg": float(np.degrees(self.rot_success_thresh)),
            "rot_weight_dense": self.rot_weight,
            "contact_thresh": self.contact_thresh,
            "goal_theta_source": self._goal_theta_source,
            "goal_theta_warning": self._goal_theta_warning,
            "velocity_available": self._velocity_available,
            "compute_intersection_ratio_default": False,
            "compute_intersection_ratio_eval_enabled_by_trainer": True,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "observation_layout": {
                "tcp_pos": [0, 3],
                "obj_pos": [3, 6],
                "obj_cos_theta": [6, 7],
                "obj_sin_theta": [7, 8],
                "tcp_to_obj_pos": [8, 11],
                "obj_velocity_xy": [11, 13],
                "obj_ang_vel_z": [13, 14],
                "tcp_velocity_xy": [14, 16],
            },
            "goal_layout": {
                "xy": [0, 2],
                "cos_theta": [2, 3],
                "sin_theta": [3, 4],
            },
            "raw_state_flat_tail_layout": {
                "enabled_when_obs_is_flat": True,
                "tcp_pose": [-17, -10],
                "goal_pos": [-10, -7],
                "obj_pose": [-7, 0],
                "source": "ManiSkill PushT _get_obs_extra order: tcp_pose, goal_pos, obj_pose",
            },
            "intersection_ratio_source": (
                "info intersection/overlap ratio during train/eval; "
                "env.pseudo_render_intersection() under no_grad only when eval enables compute_intersection_ratio"
            ),
            "observation_goal_independent": True,
            "flattened_obs_key_sample": self._last_obs_keys[:80],
        }

    def close(self) -> None:
        self._env.close()

    def _parse_obs(self, obs: Any, info: Dict[str, Any]) -> GoalObservation:
        features = self._extract_features(obs, info)
        obj_cos, obj_sin = np.cos(features.obj_theta), np.sin(features.obj_theta)
        goal_cos, goal_sin = np.cos(features.goal_theta), np.sin(features.goal_theta)
        tcp_to_obj = features.obj_pos - features.tcp_pos
        compact_obs = np.concatenate(
            [
                features.tcp_pos,
                features.obj_pos,
                np.asarray([obj_cos, obj_sin], dtype=np.float32),
                tcp_to_obj,
                features.obj_velocity_xy,
                np.asarray([features.obj_ang_vel_z], dtype=np.float32),
                features.tcp_velocity_xy,
            ],
            axis=0,
        ).astype(np.float32)
        achieved_goal = np.asarray([features.obj_pos[0], features.obj_pos[1], obj_cos, obj_sin], dtype=np.float32)
        desired_goal = np.asarray([features.goal_pos[0], features.goal_pos[1], goal_cos, goal_sin], dtype=np.float32)
        goal_success = float(self.success_from_goal(achieved_goal[None, :], desired_goal[None, :])[0])
        self._goal_theta_source = features.goal_theta_source
        self._goal_theta_warning = features.goal_theta_warning
        self._velocity_available = features.velocity_available
        return GoalObservation(
            observation=compact_obs,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            is_success=goal_success,
        )

    def _extract_features(self, obs: Any, info: Dict[str, Any]) -> PushTFeatures:
        flat = _flatten_obs(obs)
        self._last_obs_keys = sorted(flat.keys())
        root = self._env.unwrapped
        agent = getattr(root, "agent", None)
        obj_tokens = ("tee", "t_block", "tblock", "block", "obj", "object")
        flat_state = flat.get("")
        tail_tcp_pose: Optional[np.ndarray] = None
        tail_goal_pos: Optional[np.ndarray] = None
        tail_obj_pose: Optional[np.ndarray] = None
        if flat_state is not None:
            flat_arr = np.asarray(flat_state, dtype=np.float32).reshape(-1)
            # ManiSkill `state` is the flattened `state_dict`. PushT's `_get_obs_extra`
            # appends tcp_pose(7), goal_pos(3), obj_pose(7), so these are the final
            # 17 values after robot proprioception.
            if flat_arr.size >= 17:
                extra_tail = flat_arr[-17:]
                tail_tcp_pose = extra_tail[0:7]
                tail_goal_pos = extra_tail[7:10]
                tail_obj_pose = extra_tail[10:17]

        tcp_pos = _first_not_none(
            _pos3(tail_tcp_pose),
            _find_named_vector(flat, ("tcp", "ee", "end_effector"), ("pos", "pose", "p"), exclude=("to", "delta")),
            _pose_from_attrs(root, ("tcp", "tcp_link", "ee_link", "end_effector")),
            _pose_from_attrs(agent, ("tcp", "tcp_link", "ee_link", "end_effector")),
        )
        obj_pose = _first_not_none(
            tail_obj_pose,
            _find_named_array(flat, obj_tokens, ("pose",), exclude=("to", "delta"), min_size=3),
            _pose_vector_from_attrs(root, ("tee", "t_block", "tblock", "block", "obj", "object")),
        )
        obj_pos_raw = _first_not_none(
            _pos3(obj_pose),
            _find_named_vector(flat, obj_tokens, ("pos", "p"), exclude=("to", "delta", "goal")),
            _pose_from_attrs(root, ("tee", "t_block", "tblock", "block", "obj", "object")),
        )
        goal_pose = _first_not_none(
            _find_named_array(
                flat,
                ("goal", "target"),
                ("pose",),
                exclude=("to", "delta", "obj_to_goal", "object_to_goal"),
                min_size=2,
            ),
            _pose_vector_from_attrs(root, ("goal_pose", "goal_site", "goal_tee", "target_pose", "target", "goal_region")),
        )
        goal_pos_raw = _first_not_none(
            tail_goal_pos,
            _pos3(goal_pose),
            _find_named_array(
                flat,
                ("goal", "target"),
                ("pos", "p", "xy"),
                exclude=("to", "delta", "obj_to_goal", "object_to_goal"),
                min_size=2,
            ),
            _pose_from_attrs(root, ("goal_site", "goal_tee", "goal", "target", "goal_region")),
        )
        if tcp_pos is None or obj_pos_raw is None or goal_pos_raw is None:
            missing = [
                name
                for name, value in (
                    ("tcp_pos", tcp_pos),
                    ("obj_pos", obj_pos_raw),
                    ("goal_pos", goal_pos_raw),
                )
                if value is None
            ]
            keys = sorted(flat.keys())
            raise KeyError(
                "Could not parse PushT observation fields: "
                f"{missing}. Available flattened observation keys={keys[:80]}"
            )

        obj_pos = _as_pos3(obj_pos_raw)
        goal_pos = _as_pos3(goal_pos_raw)
        obj_theta = _first_finite_scalar(
            _yaw_from_pose_vector(obj_pose),
            _find_named_scalar_matching(flat, obj_tokens, ("theta", "yaw", "angle", "z_rot", "rot")),
            _yaw_from_pose_vector(_pose_vector_from_attrs(root, ("tee", "t_block", "tblock", "block", "obj", "object"))),
        )
        if obj_theta is None:
            obj_theta = 0.0

        goal_theta_source = "env_attr"
        goal_theta_warning = ""
        goal_theta = _first_finite_scalar(
            _scalar_from_attrs(root, ("goal_z_rot", "goal_theta", "target_theta", "target_angle", "goal_angle")),
            _find_named_scalar_matching(flat, ("goal", "target"), ("theta", "yaw", "angle", "z_rot", "rot")),
        )
        if goal_theta is None:
            goal_theta_source = "goal_pose_quaternion"
            goal_theta = _yaw_from_pose_vector(goal_pose)
        if goal_theta is None:
            goal_theta_source = "assumed_zero"
            goal_theta = 0.0
            goal_theta_warning = (
                "goal theta was not exposed by PushT observation/env attrs; "
                "using 0 rad for pose-context desired_goal"
            )

        obj_vel_raw = _find_named_array(
            flat,
            obj_tokens,
            ("velocity", "vel", "linvel", "linear_velocity"),
            exclude=("ang", "angular", "goal"),
            min_size=2,
        )
        tcp_vel_raw = _find_named_array(
            flat,
            ("tcp", "ee", "end_effector"),
            ("velocity", "vel", "linvel", "linear_velocity"),
            exclude=("ang", "angular"),
            min_size=2,
        )
        obj_ang_raw = _first_finite_scalar(
            _find_named_scalar_matching(flat, obj_tokens, ("angular_velocity", "ang_vel", "omega", "wz")),
            _find_named_scalar_matching(flat, obj_tokens, ("avel", "angvel")),
        )
        velocity_available = obj_vel_raw is not None or tcp_vel_raw is not None or obj_ang_raw is not None
        obj_velocity_xy = (
            np.asarray(obj_vel_raw, dtype=np.float32).reshape(-1)[:2]
            if obj_vel_raw is not None
            else np.zeros(2, dtype=np.float32)
        )
        tcp_velocity_xy = (
            np.asarray(tcp_vel_raw, dtype=np.float32).reshape(-1)[:2]
            if tcp_vel_raw is not None
            else np.zeros(2, dtype=np.float32)
        )
        obj_ang_vel_z = float(obj_ang_raw) if obj_ang_raw is not None else 0.0
        intersection_ratio = _scalar_from_info(info, ("intersection_ratio", "overlap_ratio"), default=np.nan)
        if not np.isfinite(intersection_ratio):
            intersection_ratio = _find_named_scalar_matching(
                _flatten_obs(info),
                ("intersection", "overlap"),
                ("ratio",),
            )
            if intersection_ratio is None:
                intersection_ratio = np.nan

        return PushTFeatures(
            tcp_pos=np.asarray(tcp_pos, dtype=np.float32).reshape(-1)[:3],
            obj_pos=obj_pos.astype(np.float32),
            goal_pos=goal_pos.astype(np.float32),
            obj_theta=float(obj_theta),
            goal_theta=float(goal_theta),
            obj_velocity_xy=obj_velocity_xy.astype(np.float32),
            obj_ang_vel_z=obj_ang_vel_z,
            tcp_velocity_xy=tcp_velocity_xy.astype(np.float32),
            intersection_ratio=float(intersection_ratio),
            goal_theta_source=goal_theta_source,
            goal_theta_warning=goal_theta_warning,
            velocity_available=bool(velocity_available),
        )

    def _push_t_info(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        info: Dict[str, Any],
        compute_intersection: bool | None = None,
    ) -> Dict[str, float]:
        if compute_intersection is None:
            compute_intersection = self.compute_intersection_ratio
        tcp = observation[0:3]
        obj = observation[3:6]
        obj_cos = observation[6]
        obj_sin = observation[7]
        obj_vel_xy = observation[11:13]
        obj_ang_vel_z = observation[13]
        pos_vec = goal[:2] - obj[:2]
        pos_dist = float(np.linalg.norm(pos_vec))
        rot_err = float(_angle_error_from_cossin(obj_cos, obj_sin, goal[2], goal[3]))
        tcp_obj_dist = float(np.linalg.norm(tcp - obj))
        obj_speed_xy = float(np.linalg.norm(obj_vel_xy))
        if obj_speed_xy > 1e-8 and pos_dist > 1e-8:
            vel_align = float(np.clip(np.dot(obj_vel_xy, pos_vec / pos_dist) / obj_speed_xy, -1.0, 1.0))
        else:
            vel_align = 0.0
        intersection_ratio = _scalar_from_info(info, ("intersection_ratio", "overlap_ratio"), default=np.nan)
        if not np.isfinite(intersection_ratio):
            nested_ratio = _find_named_scalar_matching(_flatten_obs(info), ("intersection", "overlap"), ("ratio",))
            intersection_ratio = float(nested_ratio) if nested_ratio is not None else np.nan
        if bool(compute_intersection) and not np.isfinite(intersection_ratio):
            intersection_ratio = _pseudo_render_intersection_ratio(self._env.unwrapped, default=np.nan)
        success_margin = float(intersection_ratio - 0.90) if np.isfinite(intersection_ratio) else np.nan
        return {
            "pushT/obj_goal_xy_dist": pos_dist,
            "pushT/obj_goal_rot_err": rot_err,
            "pushT/intersection_ratio": float(intersection_ratio),
            "pushT/success_margin": success_margin,
            "pushT/tcp_obj_dist": tcp_obj_dist,
            "pushT/contact_like": float(tcp_obj_dist <= self.contact_thresh),
            "pushT/obj_speed_xy": obj_speed_xy,
            "pushT/obj_ang_vel_z_abs": float(abs(obj_ang_vel_z)),
            "pushT/vel_align_to_goal": vel_align,
        }


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


def _find_named_array(
    flat: Dict[str, np.ndarray],
    name_tokens: Iterable[str],
    value_tokens: Iterable[str],
    exclude: Iterable[str] = (),
    min_size: int = 1,
) -> Optional[np.ndarray]:
    names = tuple(t.lower() for t in name_tokens)
    values = tuple(t.lower() for t in value_tokens)
    excludes = tuple(t.lower() for t in exclude)
    for key, value in flat.items():
        key_l = key.lower()
        if excludes and any(token in key_l for token in excludes):
            continue
        if names and not any(token in key_l for token in names):
            continue
        if values and not any(token in key_l for token in values):
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= int(min_size):
            return arr
    return None


def _find_named_scalar(flat: Dict[str, np.ndarray], name_tokens: Iterable[str]) -> Optional[float]:
    for key, value in flat.items():
        key_l = key.lower()
        if any(token in key_l for token in name_tokens):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size:
                return float(arr[0])
    return None


def _find_named_scalar_matching(
    flat: Dict[str, np.ndarray],
    name_tokens: Iterable[str],
    value_tokens: Iterable[str],
    exclude: Iterable[str] = (),
) -> Optional[float]:
    arr = _find_named_array(flat, name_tokens, value_tokens, exclude=exclude, min_size=1)
    if arr is None:
        return None
    flat_arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat_arr.size == 0:
        return None
    return float(flat_arr[-1] if flat_arr.size >= 3 else flat_arr[0])


def _scalar_from_info(info: Dict[str, Any], keys: Iterable[str], default: float) -> float:
    for key in keys:
        if key in info:
            arr = _to_numpy(info[key]).reshape(-1)
            if arr.size:
                return float(arr[0])
    return float(default)


def _pseudo_render_intersection_ratio(root: Any, default: float) -> float:
    if root is None or not hasattr(root, "pseudo_render_intersection"):
        return float(default)
    try:
        import torch

        with torch.no_grad():
            value = root.pseudo_render_intersection()
        arr = _to_numpy(value).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        return float(default)
    return float(default)


def _scalar_from_attrs(root: Any, names: Iterable[str]) -> Optional[float]:
    if root is None:
        return None
    for name in names:
        value = getattr(root, name, None)
        if value is None:
            continue
        arr = _to_numpy(value).reshape(-1)
        if arr.size:
            return float(arr[0])
    return None


def _first_finite_scalar(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        value_f = float(value)
        if np.isfinite(value_f):
            return value_f
    return None


def _first_not_none(*values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    for value in values:
        if value is not None:
            return value
    return None


def _pos3(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    return _as_pos3(arr)


def _as_pos3(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected at least x/y position components, got shape={arr.shape}")
    out = np.zeros(3, dtype=np.float32)
    out[: min(3, arr.size)] = arr[: min(3, arr.size)]
    return out


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


def _pose_vector_from_attrs(root: Any, names: Iterable[str]) -> Optional[np.ndarray]:
    if root is None:
        return None
    for name in names:
        obj = getattr(root, name, None)
        if obj is None:
            continue
        pose = getattr(obj, "pose", None)
        if pose is None and hasattr(obj, "get_pose"):
            pose = obj.get_pose()
        pose_obj = pose if pose is not None else obj
        pos = getattr(pose_obj, "p", None)
        quat = getattr(pose_obj, "q", None)
        if pos is None:
            continue
        pos_arr = _squeeze_batch(pos)
        if quat is None:
            return pos_arr[:3]
        quat_arr = _squeeze_batch(quat)
        if pos_arr.size >= 3 and quat_arr.size >= 4:
            return np.concatenate([pos_arr[:3], quat_arr[:4]], axis=0).astype(np.float32)
        if pos_arr.size >= 3:
            return pos_arr[:3]
    return None


def _yaw_from_pose_vector(pose: Optional[np.ndarray]) -> Optional[float]:
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size >= 7:
        return _quat_wxyz_to_yaw(arr[3:7])
    if arr.size == 4:
        return float(arr[3])
    return None


def _quat_wxyz_to_yaw(quat: np.ndarray) -> float:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size < 4:
        return 0.0
    norm = np.linalg.norm(q[:4])
    if norm <= 1e-12:
        return 0.0
    w, x, y, z = (q[:4] / norm).tolist()
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _angle_error_from_cossin(
    obj_cos: np.ndarray | float,
    obj_sin: np.ndarray | float,
    goal_cos: np.ndarray | float,
    goal_sin: np.ndarray | float,
) -> np.ndarray:
    obj_c = np.asarray(obj_cos, dtype=np.float32)
    obj_s = np.asarray(obj_sin, dtype=np.float32)
    goal_c = np.asarray(goal_cos, dtype=np.float32)
    goal_s = np.asarray(goal_sin, dtype=np.float32)
    cross = obj_s * goal_c - obj_c * goal_s
    dot = obj_c * goal_c + obj_s * goal_s
    return np.asarray(np.abs(np.arctan2(cross, dot)), dtype=np.float32)
