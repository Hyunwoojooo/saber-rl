from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SampleBatch:
    indices: np.ndarray
    observation: np.ndarray
    achieved_goal: np.ndarray
    desired_goal_train: np.ndarray
    next_observation: np.ndarray
    next_achieved_goal: np.ndarray
    action_exec: np.ndarray
    reward_ext: np.ndarray
    done: np.ndarray
    timeout: np.ndarray
    h: np.ndarray
    her_relabel_ratio: float


class ReplayActionIndex:
    """Coarse context -> replay slots for replay-action support lookup.

    The index stores slots only. Full context filtering remains train-time because
    FetchPush phase depends on the HER-relabeled goal.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._slots_by_key: Dict[int, List[int]] = {}
        self._key_by_slot = np.full(self.capacity, -1, dtype=np.int64)
        self._action_bin_by_slot = np.full(self.capacity, -1, dtype=np.int64)
        self._action_bin_counts_by_key: Dict[int, Dict[int, int]] = {}
        self._version_by_key: Dict[int, int] = {}
        self._epoch = 0

    def clear(self) -> None:
        self._slots_by_key.clear()
        self._key_by_slot.fill(-1)
        self._action_bin_by_slot.fill(-1)
        self._action_bin_counts_by_key.clear()
        self._version_by_key.clear()
        self._epoch += 1

    def update(self, slot: int, key: int, action_bin: Optional[int] = None) -> None:
        slot_i = int(slot)
        key_i = int(key)
        bin_i = -1 if action_bin is None else int(action_bin)
        if slot_i < 0 or slot_i >= self.capacity:
            raise IndexError(f"ReplayActionIndex slot out of range: {slot_i}")
        old_key = int(self._key_by_slot[slot_i])
        old_bin = int(self._action_bin_by_slot[slot_i])
        if old_key >= 0 and (old_key != key_i or old_bin != bin_i):
            self._decrement_action_bin(old_key, old_bin)
            if old_key != key_i:
                self._bump_version(old_key)
        self._key_by_slot[slot_i] = key_i
        self._action_bin_by_slot[slot_i] = bin_i
        if old_key != key_i or old_bin != bin_i:
            self._increment_action_bin(key_i, bin_i)
        self._slots_by_key.setdefault(key_i, []).append(slot_i)
        self._bump_version(key_i)

    def clear_slot(self, slot: int) -> None:
        slot_i = int(slot)
        if 0 <= slot_i < self.capacity:
            old_key = int(self._key_by_slot[slot_i])
            old_bin = int(self._action_bin_by_slot[slot_i])
            if old_key >= 0:
                self._decrement_action_bin(old_key, old_bin)
                self._bump_version(old_key)
            self._key_by_slot[slot_i] = -1
            self._action_bin_by_slot[slot_i] = -1

    def _increment_action_bin(self, key: int, action_bin: int) -> None:
        bin_i = int(action_bin)
        if bin_i < 0:
            return
        key_i = int(key)
        counts = self._action_bin_counts_by_key.setdefault(key_i, {})
        counts[bin_i] = int(counts.get(bin_i, 0)) + 1

    def _decrement_action_bin(self, key: int, action_bin: int) -> None:
        bin_i = int(action_bin)
        if bin_i < 0:
            return
        key_i = int(key)
        counts = self._action_bin_counts_by_key.get(key_i)
        if not counts:
            return
        next_count = int(counts.get(bin_i, 0)) - 1
        if next_count > 0:
            counts[bin_i] = next_count
        else:
            counts.pop(bin_i, None)
        if not counts:
            self._action_bin_counts_by_key.pop(key_i, None)

    def _bump_version(self, key: int) -> None:
        key_i = int(key)
        self._version_by_key[key_i] = int(self._version_by_key.get(key_i, 0)) + 1

    def version(self, key: int) -> int:
        key_i = int(key)
        return int((self._epoch << 32) + self._version_by_key.get(key_i, 0))

    def candidates(self, key: int, size: int) -> np.ndarray:
        key_i = int(key)
        slots = self._slots_by_key.get(key_i)
        if not slots:
            return np.zeros((0,), dtype=np.int64)

        size_i = min(int(size), self.capacity)
        valid: List[int] = []
        seen = set()
        for slot in slots:
            slot_i = int(slot)
            if slot_i in seen:
                continue
            if 0 <= slot_i < size_i and int(self._key_by_slot[slot_i]) == key_i:
                valid.append(slot_i)
                seen.add(slot_i)

        if len(valid) != len(slots):
            if valid:
                self._slots_by_key[key_i] = valid
            else:
                self._slots_by_key.pop(key_i, None)

        if not valid:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(valid, dtype=np.int64)

    def sample_candidates(
        self,
        key: int,
        size: int,
        max_scan: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, int, bool]:
        candidates = self.candidates(key, size)
        n_candidates = int(candidates.size)
        truncated = int(max_scan) > 0 and n_candidates > int(max_scan)
        if truncated:
            scanned = rng.choice(candidates, size=int(max_scan), replace=False).astype(np.int64)
        else:
            scanned = candidates.astype(np.int64)
        return scanned, n_candidates, bool(truncated)

    def deterministic_candidates(
        self,
        key: int,
        size: int,
        max_scan: int,
        mode: str = "deterministic_tail_v1",
    ) -> Tuple[np.ndarray, int, bool]:
        candidates = self.candidates(key, size)
        n_candidates = int(candidates.size)
        max_scan_i = int(max_scan)
        truncated = max_scan_i > 0 and n_candidates > max_scan_i
        if not truncated:
            scanned = candidates.astype(np.int64, copy=True)
        elif mode == "deterministic_head_v1":
            scanned = candidates[:max_scan_i].astype(np.int64, copy=True)
        elif mode == "deterministic_stride_v1":
            positions = np.linspace(0, n_candidates - 1, num=max_scan_i, dtype=np.int64)
            scanned = candidates[positions].astype(np.int64, copy=True)
        else:
            scanned = candidates[-max_scan_i:].astype(np.int64, copy=True)
        return scanned, n_candidates, bool(truncated)

    def action_bins(self, key: int, size: int, action_bin: np.ndarray | None = None) -> np.ndarray:
        del size, action_bin
        counts = self._action_bin_counts_by_key.get(int(key))
        if not counts:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(sorted(bin_i for bin_i, count in counts.items() if int(count) > 0), dtype=np.int64)

    def has_action_bin(self, key: int, action_bin: int) -> bool:
        counts = self._action_bin_counts_by_key.get(int(key))
        return bool(counts and int(counts.get(int(action_bin), 0)) > 0)

    def rebuild(self, coarse_keys: np.ndarray, size: int, action_bins: Optional[np.ndarray] = None) -> None:
        self.clear()
        for slot in range(min(int(size), self.capacity)):
            bin_i = None if action_bins is None else int(action_bins[slot])
            self.update(slot, int(coarse_keys[slot]), bin_i)

    def rebuild_action_bins(self, action_bins: np.ndarray, size: int) -> None:
        self._action_bin_by_slot.fill(-1)
        self._action_bin_counts_by_key.clear()
        for slot in range(min(int(size), self.capacity)):
            key_i = int(self._key_by_slot[slot])
            if key_i < 0:
                continue
            bin_i = int(action_bins[slot])
            self._action_bin_by_slot[slot] = bin_i
            self._increment_action_bin(key_i, bin_i)
        self._epoch += 1

    def indexed_slot_count(self, size: int) -> int:
        size_i = min(int(size), self.capacity)
        if size_i <= 0:
            return 0
        return int(np.count_nonzero(self._key_by_slot[:size_i] >= 0))

    def compact(self, size: int) -> None:
        for key in list(self._slots_by_key.keys()):
            self.candidates(key, size)

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "key_by_slot": self._key_by_slot,
            "action_bin_by_slot": self._action_bin_by_slot,
            "slots_by_key": self._slots_by_key,
            "action_bin_counts_by_key": self._action_bin_counts_by_key,
            "version_by_key": self._version_by_key,
            "epoch": self._epoch,
        }

    def load_state_dict(self, payload: Dict[str, object], size: int) -> None:
        self.clear()
        key_by_slot = np.asarray(payload.get("key_by_slot", []), dtype=np.int64)
        n = min(key_by_slot.shape[0], self.capacity)
        if n > 0:
            self._key_by_slot[:n] = key_by_slot[:n]
        bin_by_slot = np.asarray(payload.get("action_bin_by_slot", []), dtype=np.int64)
        if bin_by_slot.shape[0] > 0:
            self._action_bin_by_slot[: min(bin_by_slot.shape[0], self.capacity)] = bin_by_slot[: self.capacity]
        self._slots_by_key = {
            int(k): [int(v) for v in vals] for k, vals in payload.get("slots_by_key", {}).items()
        }
        self._action_bin_counts_by_key = {
            int(k): {int(bin_i): int(count) for bin_i, count in vals.items()}
            for k, vals in payload.get("action_bin_counts_by_key", {}).items()
        }
        self._version_by_key = {
            int(k): int(v) for k, v in payload.get("version_by_key", {}).items()
        }
        self._epoch = int(payload.get("epoch", self._epoch))
        self.compact(size)


class HERReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, goal_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.ptr = 0
        self.size = 0

        self.observation = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.achieved_goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.desired_goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_observation = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_achieved_goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.action_exec = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_ext = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.timeout = np.zeros(capacity, dtype=np.bool_)
        self.h = np.zeros(capacity, dtype=np.int16)
        self.action_bin = np.full(capacity, -1, dtype=np.int64)
        self.episode_id = np.full(capacity, -1, dtype=np.int64)
        self.episode_step = np.full(capacity, -1, dtype=np.int16)

        self._episode_to_slots: Dict[int, List[int]] = {}
        self._episode_order = deque()
        self.action_index = ReplayActionIndex(capacity)
        self.version = 0

    def add_transition(
        self,
        observation: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        action_exec: np.ndarray,
        reward_ext: float,
        next_observation: np.ndarray,
        next_achieved_goal: np.ndarray,
        done: bool,
        timeout: bool,
        h: int,
        episode_id: int,
        episode_step: int,
        coarse_key: Optional[int] = None,
        action_bin: Optional[int] = None,
    ) -> None:
        slot = self.ptr
        old_ep = int(self.episode_id[slot])
        if old_ep in self._episode_to_slots:
            self._episode_to_slots[old_ep] = [s for s in self._episode_to_slots[old_ep] if s != slot]

        self.observation[slot] = observation
        self.achieved_goal[slot] = achieved_goal
        self.desired_goal[slot] = desired_goal
        self.action_exec[slot] = action_exec
        self.reward_ext[slot] = float(reward_ext)
        self.next_observation[slot] = next_observation
        self.next_achieved_goal[slot] = next_achieved_goal
        self.done[slot] = bool(done)
        self.timeout[slot] = bool(timeout)
        self.h[slot] = int(h)
        self.action_bin[slot] = -1 if action_bin is None else int(action_bin)
        self.episode_id[slot] = int(episode_id)
        self.episode_step[slot] = int(episode_step)
        if coarse_key is None:
            self.action_index.clear_slot(slot)
        else:
            self.action_index.update(slot, int(coarse_key), int(self.action_bin[slot]))

        if episode_id not in self._episode_to_slots:
            self._episode_to_slots[episode_id] = []
            self._episode_order.append(episode_id)
        self._episode_to_slots[episode_id].append(slot)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.version += 1

    def sample(self, batch_size: int, her_ratio: float, rng: np.random.Generator) -> SampleBatch:
        if self.size < batch_size:
            raise ValueError(f"Not enough replay samples: {self.size} < {batch_size}")
        indices = rng.integers(0, self.size, size=batch_size, endpoint=False, dtype=np.int64)
        desired_goal_train = self.desired_goal[indices].copy()
        relabel_mask = rng.random(batch_size) < float(her_ratio)
        relabeled = 0
        for out_i, slot in enumerate(indices):
            if not relabel_mask[out_i]:
                continue
            goal = self._sample_future_goal(int(slot), rng)
            if goal is None:
                continue
            desired_goal_train[out_i] = goal
            relabeled += 1
        return SampleBatch(
            indices=indices,
            observation=self.observation[indices].copy(),
            achieved_goal=self.achieved_goal[indices].copy(),
            desired_goal_train=desired_goal_train,
            next_observation=self.next_observation[indices].copy(),
            next_achieved_goal=self.next_achieved_goal[indices].copy(),
            action_exec=self.action_exec[indices].copy(),
            reward_ext=self.reward_ext[indices].copy(),
            done=self.done[indices].copy(),
            timeout=self.timeout[indices].copy(),
            h=self.h[indices].copy(),
            her_relabel_ratio=float(relabeled / max(1, batch_size)),
        )

    def reference_indices(self, max_n: int, rng: np.random.Generator) -> np.ndarray:
        if self.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if max_n <= 0 or max_n >= self.size:
            return np.arange(self.size, dtype=np.int64)
        return rng.choice(self.size, size=max_n, replace=False).astype(np.int64)

    def indexed_reference_indices(
        self,
        coarse_key: int,
        max_n: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, int, bool]:
        return self.action_index.sample_candidates(
            key=int(coarse_key),
            size=self.size,
            max_scan=int(max_n),
            rng=rng,
        )

    def indexed_reference_indices_deterministic(
        self,
        coarse_key: int,
        max_n: int,
        mode: str = "deterministic_tail_v1",
    ) -> Tuple[np.ndarray, int, bool]:
        return self.action_index.deterministic_candidates(
            key=int(coarse_key),
            size=self.size,
            max_scan=int(max_n),
            mode=str(mode),
        )

    def action_index_version(self, coarse_key: int) -> int:
        return self.action_index.version(int(coarse_key))

    def action_index_bins(self, coarse_key: int) -> np.ndarray:
        return self.action_index.action_bins(int(coarse_key), self.size, self.action_bin)

    def action_index_has_action_bin(self, coarse_key: int, action_bin: int) -> bool:
        return self.action_index.has_action_bin(int(coarse_key), int(action_bin))

    def rebuild_action_index(self, coarse_keys: np.ndarray) -> None:
        if coarse_keys.shape[0] < self.size:
            raise ValueError(f"Need at least {self.size} coarse keys, got {coarse_keys.shape[0]}")
        self.action_index.rebuild(coarse_keys, self.size, self.action_bin[: self.size])
        self.version += 1

    def rebuild_action_bins(self, action_bins: np.ndarray) -> None:
        if action_bins.shape[0] < self.size:
            raise ValueError(f"Need at least {self.size} action bins, got {action_bins.shape[0]}")
        self.action_bin[:self.size] = action_bins[:self.size].astype(np.int64)
        self.action_index.rebuild_action_bins(self.action_bin[: self.size], self.size)
        self.version += 1

    def indexed_slot_count(self) -> int:
        return self.action_index.indexed_slot_count(self.size)

    def _sample_future_goal(self, slot: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        ep = int(self.episode_id[slot])
        slots = self._episode_to_slots.get(ep)
        if not slots:
            return None
        step = int(self.episode_step[slot])
        candidates = [
            s for s in slots
            if 0 <= s < self.size and int(self.episode_step[s]) > step and int(self.episode_id[s]) == ep
        ]
        if not candidates:
            return None
        future_slot = int(candidates[rng.integers(0, len(candidates), endpoint=False)])
        return self.next_achieved_goal[future_slot].copy()

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "action_dim": self.action_dim,
            "ptr": self.ptr,
            "size": self.size,
            "observation": self.observation,
            "achieved_goal": self.achieved_goal,
            "desired_goal": self.desired_goal,
            "next_observation": self.next_observation,
            "next_achieved_goal": self.next_achieved_goal,
            "action_exec": self.action_exec,
            "reward_ext": self.reward_ext,
            "done": self.done,
            "timeout": self.timeout,
            "h": self.h,
            "action_bin": self.action_bin,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "episode_to_slots": self._episode_to_slots,
            "episode_order": list(self._episode_order),
            "action_index": self.action_index.state_dict(),
            "version": self.version,
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.ptr = int(payload["ptr"])
        self.size = int(payload["size"])
        self.observation[:] = payload["observation"]
        self.achieved_goal[:] = payload["achieved_goal"]
        self.desired_goal[:] = payload["desired_goal"]
        self.next_observation[:] = payload["next_observation"]
        self.next_achieved_goal[:] = payload["next_achieved_goal"]
        self.action_exec[:] = payload["action_exec"]
        self.reward_ext[:] = payload["reward_ext"]
        self.done[:] = payload["done"]
        self.timeout[:] = payload["timeout"]
        self.h[:] = payload["h"]
        if "action_bin" in payload:
            self.action_bin[:] = payload["action_bin"]
        else:
            self.action_bin.fill(-1)
        self.episode_id[:] = payload["episode_id"]
        self.episode_step[:] = payload["episode_step"]
        self._episode_to_slots = {
            int(k): [int(v) for v in vals] for k, vals in payload["episode_to_slots"].items()
        }
        self._episode_order = deque(int(v) for v in payload["episode_order"])
        if "action_index" in payload:
            self.action_index.load_state_dict(payload["action_index"], self.size)
            self.action_index.rebuild_action_bins(self.action_bin[: self.size], self.size)
        else:
            self.action_index.clear()
        self.version = int(payload.get("version", 0))
