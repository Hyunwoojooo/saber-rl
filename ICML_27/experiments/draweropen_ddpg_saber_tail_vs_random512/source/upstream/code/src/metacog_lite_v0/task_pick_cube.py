from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class PickCubeTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    epsilon_g: float = 0.025
    d_reach: float = 0.08
    d_goal_near: float = 0.025
    z_lift_min: float = 0.07
    tcp_cube_bins: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40)
    cube_goal_bins: Tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 0.40, 0.70)
    cube_z_bins: Tuple[float, ...] = (0.04, 0.07, 0.10, 0.15, 0.25, 0.40)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100)
    z_lift_min_source: str = "config"

    tcp_slice: Tuple[int, int] = (0, 3)
    cube_slice: Tuple[int, int] = (3, 6)
    tcp_to_cube_slice: Tuple[int, int] = (6, 9)
    is_grasped_index: int = 9

    PHASE_APPROACH: int = 0
    PHASE_GRASP: int = 1
    PHASE_LIFT: int = 2
    PHASE_PLACE: int = 3
    phase_names: Tuple[str, ...] = ("approach", "grasp", "lift", "place")

    def make_state(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal], axis=-1).astype(np.float32)

    def parse_features(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim != 2:
            raise ValueError(f"Expected batched observation with ndim=2, got shape={obs_arr.shape}")
        if obs_arr.shape[1] <= self.is_grasped_index:
            raise ValueError(
                "PickCube compact observation must expose is_grasped at index 9; "
                f"got obs_dim={obs_arr.shape[1]}"
            )
        tcp = obs_arr[:, self.tcp_slice[0]:self.tcp_slice[1]]
        cube = obs_arr[:, self.cube_slice[0]:self.cube_slice[1]]
        cube_z = cube[:, 2]
        is_grasped = obs_arr[:, self.is_grasped_index]
        return tcp, cube, cube_z, is_grasped

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        tcp, cube, cube_z, is_grasped = self.parse_features(obs)
        d_tcp_cube = np.linalg.norm(tcp - cube, axis=1)
        phase = np.full(obs.shape[0], self.PHASE_PLACE, dtype=np.int64)
        phase[cube_z <= self.z_lift_min] = self.PHASE_LIFT
        phase[is_grasped < 0.5] = self.PHASE_GRASP
        phase[d_tcp_cube > self.d_reach] = self.PHASE_APPROACH
        return phase

    def bucket_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.full_context_hash_batch(obs, goal, h)

    def bucket_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return self.full_context_hash(obs, goal, h)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        tcp, cube, cube_z, is_grasped = self.parse_features(obs)
        d_tcp_cube = np.linalg.norm(tcp - cube, axis=1)
        tcp_cube_bin = self._bin_array(d_tcp_cube, self.tcp_cube_bins)
        cube_z_bin = self._bin_array(cube_z, self.cube_z_bins)
        grasp_bin = (is_grasped >= 0.5).astype(np.int64)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (tcp_cube_bin * 1_000 + cube_z_bin * 100 + grasp_bin * 10 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        tcp, cube, cube_z, is_grasped = self.parse_features(obs)
        d_tcp_cube = np.linalg.norm(tcp - cube, axis=1)
        d_cube_goal = np.linalg.norm(cube - goal, axis=1)
        phase = self.phase_batch(obs, goal)
        tcp_cube_bin = self._bin_array(d_tcp_cube, self.tcp_cube_bins)
        cube_goal_bin = self._bin_array(d_cube_goal, self.cube_goal_bins)
        cube_z_bin = self._bin_array(cube_z, self.cube_z_bins)
        grasp_bin = (is_grasped >= 0.5).astype(np.int64)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (
            phase * 100_000
            + tcp_cube_bin * 10_000
            + cube_goal_bin * 1_000
            + cube_z_bin * 100
            + grasp_bin * 10
            + h_bin
        ).astype(np.int64)

    def full_context_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return int(
            self.full_context_hash_batch(
                obs[None, :],
                goal[None, :],
                np.asarray([h], dtype=np.int64),
            )[0]
        )

    def action_bin_batch(self, action: np.ndarray, bins_per_dim: int) -> np.ndarray:
        norm = self.normalize_action(action)
        scaled = (norm + 1.0) * 0.5 * float(bins_per_dim)
        bins = np.floor(np.clip(scaled, 0.0, float(bins_per_dim - 1))).astype(np.int64)
        out = np.zeros(action.shape[0], dtype=np.int64)
        for dim in range(bins.shape[1]):
            out = (out * 131 + bins[:, dim] + 1) % 2_147_483_647
        return out.astype(np.int64)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        center = 0.5 * (self.action_high + self.action_low)
        scale = np.maximum(1e-6, 0.5 * (self.action_high - self.action_low))
        return ((action - center) / scale).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        return {
            "task": "ManiSkillPickCube",
            "env_id": "PickCube-v1",
            "robot_uids": "panda",
            "coarse_index_key": "tcp-cube distance bin, cube-z bin, grasp bin, horizon bin",
            "full_context_key": "phase, tcp-cube distance bin, cube-goal distance bin, cube-z bin, grasp bin, horizon bin",
            "phase_labels": {idx: name for idx, name in enumerate(self.phase_names)},
            "phase_definition": (
                "approach if d_tcp_cube>d_reach; grasp if not grasped; "
                "lift if grasped and cube_z<=z_lift_min; else place"
            ),
            "observation_layout": {
                "tcp_pos": list(self.tcp_slice),
                "cube_pos": list(self.cube_slice),
                "tcp_to_cube_pos": list(self.tcp_to_cube_slice),
                "is_grasped_index": self.is_grasped_index,
            },
            "epsilon_g": self.epsilon_g,
            "d_reach": self.d_reach,
            "d_goal_near": self.d_goal_near,
            "z_lift_min": self.z_lift_min,
            "z_lift_min_source": self.z_lift_min_source,
            "tcp_cube_bins": list(self.tcp_cube_bins),
            "cube_goal_bins": list(self.cube_goal_bins),
            "cube_z_bins": list(self.cube_z_bins),
            "horizon_bins": list(self.horizon_bins),
            "goal_dependent_coarse_key": False,
        }

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out
