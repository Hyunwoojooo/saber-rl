from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FetchPickAndPlaceTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    epsilon_g: float
    d_reach: float = 0.08
    d_goal_near: float = 0.05
    z_lift_min: float = 0.48
    grip_close_thresh: float = 0.04
    hand_obj_bins: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40)
    obj_goal_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70)
    object_z_bins: Tuple[float, ...] = (0.43, 0.48, 0.55, 0.65, 0.75)
    gripper_gap_bins: Tuple[float, ...] = (0.02, 0.05, 0.08)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100)

    ee_slice: Tuple[int, int] = (0, 3)
    object_slice: Tuple[int, int] = (3, 6)
    gripper_slice: Tuple[int, int] = (9, 11)

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
        if obs_arr.shape[1] < self.gripper_slice[1]:
            raise ValueError(
                "FetchPickAndPlace observation must expose gripper state at indices 9:11; "
                f"got obs_dim={obs_arr.shape[1]}"
            )
        p_ee = obs_arr[:, self.ee_slice[0]:self.ee_slice[1]]
        p_obj = obs_arr[:, self.object_slice[0]:self.object_slice[1]]
        object_z = p_obj[:, 2]
        gripper_gap = obs_arr[:, self.gripper_slice[0]] + obs_arr[:, self.gripper_slice[0] + 1]
        return p_ee, p_obj, object_z, gripper_gap

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        p_ee, p_obj, object_z, gripper_gap = self.parse_features(obs)
        d_hand_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        phase = np.full(obs.shape[0], self.PHASE_PLACE, dtype=np.int64)
        phase[object_z <= self.z_lift_min] = self.PHASE_LIFT
        grasp_mask = np.logical_and(object_z <= self.z_lift_min, gripper_gap > self.grip_close_thresh)
        phase[grasp_mask] = self.PHASE_GRASP
        phase[d_hand_obj > self.d_reach] = self.PHASE_APPROACH
        return phase

    def bucket_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.full_context_hash_batch(obs, goal, h)

    def bucket_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return self.full_context_hash(obs, goal, h)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        p_ee, p_obj, object_z, gripper_gap = self.parse_features(obs)
        d_hand_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        hand_bin = self._bin_array(d_hand_obj, self.hand_obj_bins)
        z_bin = self._bin_array(object_z, self.object_z_bins)
        grip_bin = self._bin_array(gripper_gap, self.gripper_gap_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (hand_bin * 10_000 + z_bin * 1_000 + grip_bin * 100 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        p_ee, p_obj, object_z, gripper_gap = self.parse_features(obs)
        d_hand_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        d_obj_goal = np.linalg.norm(p_obj - goal, axis=1)
        phase = self.phase_batch(obs, goal)
        hand_bin = self._bin_array(d_hand_obj, self.hand_obj_bins)
        goal_bin = self._bin_array(d_obj_goal, self.obj_goal_bins)
        z_bin = self._bin_array(object_z, self.object_z_bins)
        grip_bin = self._bin_array(gripper_gap, self.gripper_gap_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (
            phase * 1_000_000
            + hand_bin * 100_000
            + goal_bin * 10_000
            + z_bin * 1_000
            + grip_bin * 100
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
        multipliers = (bins_per_dim ** np.arange(action.shape[1], dtype=np.int64))[None, :]
        return np.sum(bins * multipliers, axis=1).astype(np.int64)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        center = 0.5 * (self.action_high + self.action_low)
        scale = np.maximum(1e-6, 0.5 * (self.action_high - self.action_low))
        return ((action - center) / scale).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        return {
            "task": "FetchPickAndPlace",
            "coarse_index_key": "hand-object distance bin, object-z bin, gripper-gap bin, horizon bin",
            "full_context_key": (
                "phase, hand-object distance bin, object-goal distance bin, "
                "object-z bin, gripper-gap bin, horizon bin"
            ),
            "phase_labels": {idx: name for idx, name in enumerate(self.phase_names)},
            "phase_definition": (
                "approach if d_hand_obj>d_reach; grasp if object_z<=z_lift_min "
                "and gripper_gap>grip_close_thresh; lift if object_z<=z_lift_min; else place"
            ),
            "observation_layout": {
                "ee_pos": list(self.ee_slice),
                "object_pos": list(self.object_slice),
                "object_z_index": self.object_slice[0] + 2,
                "gripper_gap": list(self.gripper_slice),
            },
            "epsilon_g": self.epsilon_g,
            "d_reach": self.d_reach,
            "d_goal_near": self.d_goal_near,
            "z_lift_min": self.z_lift_min,
            "grip_close_thresh": self.grip_close_thresh,
            "hand_obj_bins": list(self.hand_obj_bins),
            "obj_goal_bins": list(self.obj_goal_bins),
            "object_z_bins": list(self.object_z_bins),
            "gripper_gap_bins": list(self.gripper_gap_bins),
            "horizon_bins": list(self.horizon_bins),
        }

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out
