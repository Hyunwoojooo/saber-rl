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
    sampling_prob: np.ndarray
    is_weight: np.ndarray
    sampling_priority: np.ndarray
    per_beta: float


class ReplayActionIndex:
    """Coarse context -> replay slots for replay-action support lookup.

    The index stores slots only. Full context filtering remains train-time because
    FetchPush phase depends on the HER-relabeled goal.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._slots_by_key: Dict[int, List[int]] = {}
        self._key_by_slot = np.full(self.capacity, -1, dtype=np.int64)

    def clear(self) -> None:
        self._slots_by_key.clear()
        self._key_by_slot.fill(-1)

    def update(self, slot: int, key: int) -> None:
        slot_i = int(slot)
        key_i = int(key)
        if slot_i < 0 or slot_i >= self.capacity:
            raise IndexError(f"ReplayActionIndex slot out of range: {slot_i}")
        self._key_by_slot[slot_i] = key_i
        self._slots_by_key.setdefault(key_i, []).append(slot_i)

    def clear_slot(self, slot: int) -> None:
        slot_i = int(slot)
        if 0 <= slot_i < self.capacity:
            self._key_by_slot[slot_i] = -1

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

    def rebuild(self, coarse_keys: np.ndarray, size: int) -> None:
        self.clear()
        for slot in range(min(int(size), self.capacity)):
            self.update(slot, int(coarse_keys[slot]))

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
            "slots_by_key": self._slots_by_key,
        }

    def load_state_dict(self, payload: Dict[str, object], size: int) -> None:
        self.clear()
        key_by_slot = np.asarray(payload.get("key_by_slot", []), dtype=np.int64)
        n = min(key_by_slot.shape[0], self.capacity)
        if n > 0:
            self._key_by_slot[:n] = key_by_slot[:n]
        self._slots_by_key = {
            int(k): [int(v) for v in vals] for k, vals in payload.get("slots_by_key", {}).items()
        }
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
        self.priority = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.episode_id = np.full(capacity, -1, dtype=np.int64)
        self.episode_step = np.full(capacity, -1, dtype=np.int16)

        self._episode_to_slots: Dict[int, List[int]] = {}
        self._episode_order = deque()
        self.action_index = ReplayActionIndex(capacity)

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
        self.priority[slot] = float(max(self.max_priority, 1.0))
        self.episode_id[slot] = int(episode_id)
        self.episode_step[slot] = int(episode_step)
        if coarse_key is None:
            self.action_index.clear_slot(slot)
        else:
            self.action_index.update(slot, int(coarse_key))

        if episode_id not in self._episode_to_slots:
            self._episode_to_slots[episode_id] = []
            self._episode_order.append(episode_id)
        self._episode_to_slots[episode_id].append(slot)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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
            sampling_prob=np.full(batch_size, 1.0 / max(1, self.size), dtype=np.float32),
            is_weight=np.ones(batch_size, dtype=np.float32),
            sampling_priority=self.priority[indices].copy(),
            per_beta=0.0,
        )

    def sample_prioritized(
        self,
        batch_size: int,
        her_ratio: float,
        rng: np.random.Generator,
        alpha: float,
        beta: float,
        eps: float,
    ) -> SampleBatch:
        if self.size < batch_size:
            raise ValueError(f"Not enough replay samples: {self.size} < {batch_size}")
        priorities = np.asarray(self.priority[: self.size], dtype=np.float64)
        priorities = np.where(np.isfinite(priorities) & (priorities > 0.0), priorities, float(eps))
        scaled = np.power(priorities + float(eps), float(alpha))
        total = float(np.sum(scaled))
        if not np.isfinite(total) or total <= 0.0:
            prob = np.full(self.size, 1.0 / max(1, self.size), dtype=np.float64)
        else:
            prob = scaled / total
        indices = rng.choice(self.size, size=batch_size, replace=True, p=prob).astype(np.int64)

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

        sample_prob = prob[indices].astype(np.float32)
        beta_f = float(beta)
        is_weight = np.power(np.maximum(self.size * sample_prob.astype(np.float64), float(eps)), -beta_f)
        max_weight = float(np.max(is_weight)) if is_weight.size else 1.0
        if max_weight > 0.0 and np.isfinite(max_weight):
            is_weight = is_weight / max_weight
        else:
            is_weight = np.ones_like(is_weight)

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
            sampling_prob=sample_prob,
            is_weight=is_weight.astype(np.float32),
            sampling_priority=self.priority[indices].copy(),
            per_beta=beta_f,
        )

    def update_priorities(self, indices: np.ndarray, td_abs: np.ndarray, eps: float) -> None:
        if self.size <= 0:
            return
        idx = np.asarray(indices, dtype=np.int64)
        values = np.asarray(td_abs, dtype=np.float32).reshape(-1)
        if idx.size != values.size:
            raise ValueError(f"Priority update shape mismatch: {idx.size} indices vs {values.size} values")
        pending: Dict[int, float] = {}
        for slot, value in zip(idx, values):
            slot_i = int(slot)
            if slot_i < 0 or slot_i >= self.size:
                continue
            priority = float(abs(float(value)) + float(eps))
            prev = pending.get(slot_i)
            if prev is None or priority > prev:
                pending[slot_i] = priority
        for slot_i, priority in pending.items():
            self.priority[slot_i] = np.float32(priority)
        if pending:
            current_max = float(np.max(self.priority[: self.size]))
            self.max_priority = max(1.0, current_max)

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

    def rebuild_action_index(self, coarse_keys: np.ndarray) -> None:
        if coarse_keys.shape[0] < self.size:
            raise ValueError(f"Need at least {self.size} coarse keys, got {coarse_keys.shape[0]}")
        self.action_index.rebuild(coarse_keys, self.size)

    def rebuild_action_bins(self, action_bins: np.ndarray) -> None:
        if action_bins.shape[0] < self.size:
            raise ValueError(f"Need at least {self.size} action bins, got {action_bins.shape[0]}")
        self.action_bin[:self.size] = action_bins[:self.size].astype(np.int64)

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
            "priority": self.priority,
            "max_priority": self.max_priority,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "episode_to_slots": self._episode_to_slots,
            "episode_order": list(self._episode_order),
            "action_index": self.action_index.state_dict(),
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
        if "priority" in payload:
            self.priority[:] = payload["priority"]
        else:
            self.priority.fill(1.0)
        self.max_priority = float(payload.get("max_priority", float(np.max(self.priority[: max(1, self.size)]))))
        self.max_priority = max(1.0, self.max_priority)
        self.episode_id[:] = payload["episode_id"]
        self.episode_step[:] = payload["episode_step"]
        self._episode_to_slots = {
            int(k): [int(v) for v in vals] for k, vals in payload["episode_to_slots"].items()
        }
        self._episode_order = deque(int(v) for v in payload["episode_order"])
        if "action_index" in payload:
            self.action_index.load_state_dict(payload["action_index"], self.size)
        else:
            self.action_index.clear()
