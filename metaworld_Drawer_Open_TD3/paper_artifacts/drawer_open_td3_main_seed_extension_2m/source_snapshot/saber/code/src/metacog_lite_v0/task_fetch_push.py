from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FetchPushTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    epsilon_g: float
    d_contact: float = 0.08
    d_goal_near: float = 0.05
    ee_obj_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    obj_goal_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100)
    phase_names: Tuple[str, ...] = ("approach", "push", "near_goal")

    def make_state(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal], axis=-1).astype(np.float32)

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        p_ee = obs[:, 0:3]
        p_obj = obs[:, 3:6]
        d_ee_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        d_obj_goal = np.linalg.norm(p_obj - goal, axis=1)
        # Spec order is approach -> push -> near_goal. Near-goal should not
        # override approach when the gripper is far from the object.
        phase = np.zeros(obs.shape[0], dtype=np.int64)
        contact = d_ee_obj <= self.d_contact
        phase[np.logical_and(contact, d_obj_goal > self.d_goal_near)] = 1
        phase[np.logical_and(contact, d_obj_goal <= self.d_goal_near)] = 2
        return phase

    def bucket_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.full_context_hash_batch(obs, goal, h)

    def bucket_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return self.full_context_hash(obs, goal, h)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        p_ee = obs[:, 0:3]
        p_obj = obs[:, 3:6]
        d_ee_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        ee_bin = self._bin_array(d_ee_obj, self.ee_obj_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (ee_bin * 100 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        p_ee = obs[:, 0:3]
        p_obj = obs[:, 3:6]
        d_ee_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        d_obj_goal = np.linalg.norm(p_obj - goal, axis=1)
        phase = self.phase_batch(obs, goal)
        ee_bin = self._bin_array(d_ee_obj, self.ee_obj_bins)
        goal_bin = self._bin_array(d_obj_goal, self.obj_goal_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        # Compact deterministic hash; all components are small non-negative bins.
        return (phase * 10_000 + ee_bin * 1_000 + goal_bin * 100 + h_bin).astype(np.int64)

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
            "task": "FetchPush",
            "coarse_index_key": "ee-object distance bin, horizon bin",
            "full_context_key": "phase, ee-object distance bin, object-goal distance bin, horizon bin",
            "phase_labels": {idx: name for idx, name in enumerate(self.phase_names)},
            "d_contact": self.d_contact,
            "d_goal_near": self.d_goal_near,
            "ee_obj_bins": list(self.ee_obj_bins),
            "obj_goal_bins": list(self.obj_goal_bins),
            "horizon_bins": list(self.horizon_bins),
            "epsilon_g": self.epsilon_g,
        }

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out
