#!/usr/bin/env python3
"""Replay-window statistics D1-D5 for every n20 HER/SABER checkpoint, 100k-1M every 50k (PROTOCOL_ko.md).

  .venv/bin/python <this dir>/extract.py [--workers 6]

Writes outputs/replay_stats/<arm>_s<seed>_<step>.json; existing files are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOU = HERE.parent / "draweropen_ddpg_first_order_usefulness_20261109"
OUT = HERE / "outputs" / "replay_stats"
SEEDS = tuple(range(11, 31))
STAGES = tuple(range(100_000, 1_000_001, 50_000))
WINDOW = 50_000
N_STATES = 4096
STATE_RNG_SEED = 20261118
MOVED = 0.02


def window_slots(r) -> np.ndarray:
    """Slots written in the last WINDOW env steps (one slot per env step, ring buffer)."""
    if r.size == r.capacity:
        newest = (r.ptr - 1) % r.capacity
        age = (newest - np.arange(r.capacity)) % r.capacity
    else:
        newest = r.size - 1
        age = newest - np.arange(r.size)
    return np.flatnonzero(age < WINDOW)


def stats(arm: str, seed: int, stage: int) -> dict:
    import torch
    sys.path.insert(0, str(FOU))
    from fou import protocol as P
    from run import load
    torch.set_num_threads(1)
    src = load("fpp_n20", P.fpp_path("fpp_n20", arm, seed, stage))
    assert (src.arm, src.seed, src.step) == (arm, seed, stage), (src.arm, src.seed, src.step)
    r, task = src.replay, src.task
    idx = window_slots(r)
    assert len(idx) == WINDOW, len(idx)
    obs, goal = r.observation[idx], r.desired_goal[idx]
    ph = task.phase_batch(obs, goal)
    d1 = float(np.mean(ph >= 2))                                   # phase names: approach, grasp, lift, place
    keys = task.full_context_hash_batch(obs, goal, r.h[idx])
    d2 = float(len(np.unique(keys)) / (len(idx) / 1000))
    # episodes that start inside the window: object displacement from its start position
    ep, st, ag = r.episode_id[idx], r.episode_step[idx], r.achieved_goal[idx]
    starts = {int(e): ag[i] for i, (e, s) in enumerate(zip(ep, st)) if s == 0}
    maxd = {}
    for e, g in zip(ep, ag):
        e = int(e)
        if e in starts:
            maxd[e] = max(maxd.get(e, 0.0), float(np.linalg.norm(g - starts[e])))
    d5 = float(np.mean([v > MOVED for v in maxd.values()]))
    pick = np.random.default_rng(STATE_RNG_SEED).choice(idx, N_STATES, replace=False)
    S = torch.as_tensor(np.concatenate([r.observation[pick], r.desired_goal[pick]], 1), dtype=torch.float32)
    with torch.no_grad():
        a = src.agent.actor(S)
    d3 = float((a.abs() > 0.99).float().mean())
    a = a.clone().requires_grad_(True)
    with torch.enable_grad():
        g = torch.autograd.grad(src.agent.critic1(S, a).sum(), a)[0]
    d4 = float(g.norm(dim=1).median())
    src.verify_unchanged()
    return dict(arm=arm, seed=seed, stage=stage, D1=d1, D2=d2, D3=d3, D4=d4, D5=d5, n_episodes=len(maxd),
                last_eval_success=src.last_eval_success)


def job(t):
    arm, seed, stage = t
    dest = OUT / f"{arm}_s{seed}_{stage}.json"
    if dest.exists():
        return f"skip {dest.name}"
    row = stats(arm, seed, stage)
    dest.write_text(json.dumps(row) + "\n")
    return f"done {dest.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke test)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(arm, s, st) for st in STAGES for s in SEEDS for arm in ("saber", "her")][: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(job, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
