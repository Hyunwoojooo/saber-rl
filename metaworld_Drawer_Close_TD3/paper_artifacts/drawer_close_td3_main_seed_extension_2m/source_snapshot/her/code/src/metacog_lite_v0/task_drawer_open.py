from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class MetaWorldDrawerOpenTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    epsilon_g: float
    d_reach: float = 0.08
    d_goal_near: float = 0.05
    d_opening_mid: float = 0.12
    hand_handle_bins: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40)
    handle_goal_bins: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40, 0.70)
    handle_y_bins: Tuple[float, ...] = (-0.05, 0.00, 0.05, 0.10, 0.15, 0.25)
    gripper_bins: Tuple[float, ...] = (-0.50, 0.0, 0.50)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100, 200, 500)

    ee_slice: Tuple[int, int] = (0, 3)
    handle_slice: Tuple[int, int] = (4, 7)
    gripper_slice: Tuple[int, int] = (3, 4)

    PHASE_APPROACH: int = 0
    PHASE_CONTACT: int = 1
    PHASE_OPENING: int = 2
    PHASE_OPEN: int = 3
    phase_names: Tuple[str, ...] = ("approach", "contact", "opening", "open")

    # Compatibility names consumed by the v4 runtime scanner interface.
    @property
    def object_slice(self) -> Tuple[int, int]:
        return self.handle_slice

    @property
    def hand_obj_bins(self) -> Tuple[float, ...]:
        return self.hand_handle_bins

    @property
    def obj_goal_bins(self) -> Tuple[float, ...]:
        return self.handle_goal_bins

    @property
    def object_z_bins(self) -> Tuple[float, ...]:
        return self.handle_y_bins

    @property
    def gripper_gap_bins(self) -> Tuple[float, ...]:
        return self.gripper_bins

    @property
    def z_lift_min(self) -> float:
        return self.d_goal_near

    @property
    def grip_close_thresh(self) -> float:
        return self.d_opening_mid

    def make_state(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal], axis=-1).astype(np.float32)

    def parse_features(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim != 2:
            raise ValueError(f"Expected batched observation with ndim=2, got shape={obs_arr.shape}")
        if obs_arr.shape[1] < self.handle_slice[1]:
            raise ValueError(
                "MetaWorld drawer-open observation must expose handle position at indices 4:7; "
                f"got obs_dim={obs_arr.shape[1]}"
            )
        p_ee = obs_arr[:, self.ee_slice[0] : self.ee_slice[1]]
        p_handle = obs_arr[:, self.handle_slice[0] : self.handle_slice[1]]
        handle_y = p_handle[:, 1]
        if obs_arr.shape[1] > self.gripper_slice[0]:
            gripper = obs_arr[:, self.gripper_slice[0]]
        else:
            gripper = np.zeros(obs_arr.shape[0], dtype=np.float32)
        return p_ee, p_handle, handle_y, gripper

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        p_ee, p_handle, _handle_y, _gripper = self.parse_features(obs)
        goal_arr = np.asarray(goal, dtype=np.float32)
        d_hand_handle = np.linalg.norm(p_ee - p_handle, axis=1)
        d_handle_goal = np.linalg.norm(p_handle - goal_arr, axis=1)
        phase = np.full(obs.shape[0], self.PHASE_APPROACH, dtype=np.int64)
        contact = d_hand_handle <= self.d_reach
        phase[contact] = self.PHASE_CONTACT
        phase[np.logical_and(contact, d_handle_goal <= self.d_opening_mid)] = self.PHASE_OPENING
        phase[d_handle_goal <= self.d_goal_near] = self.PHASE_OPEN
        return phase

    def bucket_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.full_context_hash_batch(obs, goal, h)

    def bucket_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return self.full_context_hash(obs, goal, h)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        p_ee, p_handle, handle_y, gripper = self.parse_features(obs)
        d_hand_handle = np.linalg.norm(p_ee - p_handle, axis=1)
        hand_bin = self._bin_array(d_hand_handle, self.hand_handle_bins)
        handle_y_bin = self._bin_array(handle_y, self.handle_y_bins)
        grip_bin = self._bin_array(gripper, self.gripper_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (hand_bin * 10_000 + handle_y_bin * 1_000 + grip_bin * 100 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        p_ee, p_handle, handle_y, gripper = self.parse_features(obs)
        goal_arr = np.asarray(goal, dtype=np.float32)
        d_hand_handle = np.linalg.norm(p_ee - p_handle, axis=1)
        d_handle_goal = np.linalg.norm(p_handle - goal_arr, axis=1)
        phase = self.phase_batch(obs, goal_arr)
        hand_bin = self._bin_array(d_hand_handle, self.hand_handle_bins)
        goal_bin = self._bin_array(d_handle_goal, self.handle_goal_bins)
        handle_y_bin = self._bin_array(handle_y, self.handle_y_bins)
        grip_bin = self._bin_array(gripper, self.gripper_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (
            phase * 1_000_000
            + hand_bin * 100_000
            + goal_bin * 10_000
            + handle_y_bin * 1_000
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

    def diagnostic_metrics(self, obs: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
        p_ee, p_handle, _handle_y, _gripper = self.parse_features(obs)
        goal_arr = np.asarray(goal, dtype=np.float32)
        d_hand_handle = np.linalg.norm(p_ee - p_handle, axis=1)
        d_handle_goal = np.linalg.norm(p_handle - goal_arr, axis=1)
        phase = self.phase_batch(obs, goal_arr)
        metrics: Dict[str, float] = {
            "hand_handle_dist_mean": float(np.mean(d_hand_handle)) if d_hand_handle.size else 0.0,
            "handle_goal_dist_mean": float(np.mean(d_handle_goal)) if d_handle_goal.size else 0.0,
        }
        for phase_id, phase_name in enumerate(self.phase_names):
            metrics[f"phase_{phase_name}_frac"] = float(np.mean(phase == phase_id)) if phase.size else 0.0
        return metrics

    def metadata(self) -> Dict[str, object]:
        return {
            "task": "MetaWorldDrawerOpen-v3",
            "coarse_index_key": "hand-handle distance bin, handle-y bin, gripper bin, horizon bin",
            "full_context_key": (
                "phase, hand-handle distance bin, handle-goal distance bin, "
                "handle-y bin, gripper bin, horizon bin"
            ),
            "phase_labels": {idx: name for idx, name in enumerate(self.phase_names)},
            "phase_definition": (
                "approach if d_hand_handle>d_reach; contact if hand is near handle; "
                "opening if hand is near and handle-goal distance is below d_opening_mid; "
                "open if handle-goal distance is below d_goal_near"
            ),
            "observation_layout": {
                "ee_pos": list(self.ee_slice),
                "handle_pos": list(self.handle_slice),
                "gripper": list(self.gripper_slice),
            },
            "epsilon_g": self.epsilon_g,
            "d_reach": self.d_reach,
            "d_goal_near": self.d_goal_near,
            "d_opening_mid": self.d_opening_mid,
            "hand_handle_bins": list(self.hand_handle_bins),
            "handle_goal_bins": list(self.handle_goal_bins),
            "handle_y_bins": list(self.handle_y_bins),
            "gripper_bins": list(self.gripper_bins),
            "horizon_bins": list(self.horizon_bins),
        }

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out
