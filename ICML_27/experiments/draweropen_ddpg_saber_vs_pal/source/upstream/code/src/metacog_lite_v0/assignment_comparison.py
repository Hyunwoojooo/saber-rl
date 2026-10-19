"""Assignment-only controls; no replay, target, actor, or environment mutations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


RULES = ("support", "random", "td_topk")


def validate_masks(eligible, support):
    e = np.asarray(eligible, dtype=bool)
    m = np.asarray(support, dtype=bool)
    if e.ndim != 1 or e.shape != m.shape or not e.size:
        raise ValueError("E and M must be nonempty 1D masks of the same shape")
    if np.any(m & ~e):
        raise ValueError("M must be a subset of E")
    return e, m, int(m.sum())


def random_mask(eligible, k, rng):
    ids = np.flatnonzero(eligible)
    if not 0 <= k <= len(ids):
        raise ValueError("k outside [0, |E|]")
    mask = np.zeros(len(eligible), dtype=bool)
    if k == len(ids):
        return np.asarray(eligible, dtype=bool).copy()
    if k:
        mask[rng.choice(ids, size=k, replace=False)] = True
    return mask


def td_mask(eligible, k, td_abs, tie_rng):
    scores = np.asarray(td_abs)
    if scores.shape != np.shape(eligible):
        raise ValueError("TD scores and E must have the same shape")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise FloatingPointError("Non-finite or negative absolute TD error")
    ids = np.flatnonzero(eligible)
    if not 0 <= k <= len(ids):
        raise ValueError("k outside [0, |E|]")
    mask = np.zeros(len(eligible), dtype=bool)
    if k == 0:
        return mask
    if k == len(ids):
        return np.asarray(eligible, dtype=bool).copy()
    # Only ties at the cutoff require randomization; strictly larger scores win.
    cutoff = np.partition(scores[ids], len(ids) - k)[len(ids) - k]
    mask[ids[scores[ids] > cutoff]] = True
    tied = ids[scores[ids] == cutoff]
    remaining = k - int(mask.sum())
    chosen = tied if remaining == len(tied) else tie_rng.choice(tied, remaining, replace=False)
    mask[chosen] = True
    return mask


def mean_one_weights(mask):
    raw = np.float32(1.0) + np.asarray(mask, dtype=np.float32)
    return raw, (raw / np.mean(raw)).astype(np.float32)


def overlap_statistics(e, support, random, td, assigned):
    k, m, b = int(support.sum()), int(e.sum()), len(e)
    out = {
        "eligible_count": m, "support_count": k,
        "eligible_rate": m / b, "support_rate": k / b,
        "empty_update": int(k == 0),
        "saturated_nonempty_update": int(k > 0 and k == m),
        "informative_update": int(0 < k < m),
        "random_expected_intersection": k * k / m if m else 0.0,
        "random_expected_overlap_fraction": k / m if k else None,
    }
    for name, mask in (("random", random), ("td", td), ("assigned", assigned)):
        inter = int((support & mask).sum())
        union = int((support | mask).sum())
        out[f"{name}_intersection"] = inter
        out[f"{name}_overlap_fraction"] = inter / k if k else None
        out[f"{name}_jaccard"] = inter / union if union else None
    return out


class AssignmentSelector:
    def __init__(self, weight_batch, rule, assignment_rng, tie_rng):
        if rule not in RULES:
            raise ValueError(rule)
        self.wb = weight_batch
        self.rule = rule
        self.assignment_rng = assignment_rng
        self.tie_rng = tie_rng
        self.stats = None
        self.masks = None

    def __call__(self, td_abs):
        e, support, k = validate_masks(self.wb.context_eligible_mask, self.wb.support_mask)
        random = random_mask(e, k, self.assignment_rng)
        td = td_mask(e, k, td_abs, self.tie_rng)
        masks = {"support": support, "random": random, "td_topk": td}
        assigned = masks[self.rule]
        for mask in masks.values():
            if int(mask.sum()) != k or np.any(mask & ~e):
                raise RuntimeError("Assignment count/eligibility invariant failed")
        raw, omega = mean_one_weights(assigned)
        if not np.isclose(omega.mean(dtype=np.float64), 1.0, atol=2e-6):
            raise RuntimeError("Weights do not have mean one")
        self.wb.w_p_raw = raw
        self.wb.omega_critic = omega
        self.wb.omega_actor = np.ones(len(e), dtype=np.float32)
        self.stats = overlap_statistics(e, support, random, td, assigned)
        self.masks = masks
        return omega


class AssignmentWindows:
    """Per-update means with explicit valid denominators, not zero-filled ratios."""
    def __init__(self, path: Path, window=1000):
        self.file = path.open("x", newline="", encoding="utf-8")
        self.writer = None
        self.window = window
        self.last_update = 0
        self.start_update = 1
        self.sums = {"all": {}, "informative": {}}
        self.counts = {"all": {}, "informative": {}}
        self.n = {"all": 0, "informative": 0}

    def add(self, update, step, stats):
        if update != self.last_update + 1:
            raise RuntimeError("Missing or repeated assignment update")
        self.last_update, self.step = update, step
        for group in ("all", "informative"):
            if group == "informative" and not stats["informative_update"]:
                continue
            self.n[group] += 1
            for key, value in stats.items():
                if value is not None:
                    if not np.isfinite(value):
                        raise FloatingPointError(f"Non-finite assignment diagnostic: {key}")
                    self.sums[group][key] = self.sums[group].get(key, 0.0) + value
                    self.counts[group][key] = self.counts[group].get(key, 0) + 1
        self.keys = list(stats)
        if update % self.window == 0:
            self.flush()

    def flush(self):
        if not self.n["all"]:
            return
        for group in ("all", "informative"):
            row = dict(start_update=self.start_update, end_update=self.last_update,
                       step=self.step, subset=group, n_updates=self.n[group])
            for key in self.keys:
                n = self.counts[group].get(key, 0)
                row[key + "_mean"] = self.sums[group].get(key, 0.0) / n if n else None
                row[key + "_n"] = n
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
        self.file.flush()
        self.start_update = self.last_update + 1
        self.sums = {"all": {}, "informative": {}}
        self.counts = {"all": {}, "informative": {}}
        self.n = {"all": 0, "informative": 0}

    def close(self):
        self.flush()
        self.file.close()
