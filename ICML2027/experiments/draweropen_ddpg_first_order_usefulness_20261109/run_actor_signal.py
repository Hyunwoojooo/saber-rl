#!/usr/bin/env python3
"""Actor-signal measurement on saved FPP TD3 checkpoints (no training). See PROTOCOL_actor_signal_ko.md.

  .venv/bin/python <this dir>/run_actor_signal.py measure --set fpp_n20 --out outputs/actor_signal_n20 \
        [--arms saber her] [--seeds 11 ...] [--stages 50000 ...] [--worker 0 --n-workers 5]

Each checkpoint writes <out>/<arm>/seed_<s>/step_<n>/{arms.csv, signal.npz, meta.json}; existing
directories are skipped, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fou import actor_signal as AS, core, protocol as P  # noqa: E402
from run import load, write_csv  # noqa: E402


def measure_one(path: Path, out: Path, set_name: str, expect: tuple[str, int, int]) -> dict:
    t0 = time.perf_counter()
    src = load(set_name, path)
    core.require((src.arm, src.seed, src.step) == expect, f"checkpoint content does not match {expect}: {path}")
    core.require(src.family == "fpp_td3", "actor-signal measurement is FPP only")
    sim = core.FetchSim(src)
    q = AS.build_queries(src, sim)
    Z = AS.directions(len(q.a_pi))
    Ap, Am = AS.perturbed_actions(q, Z)
    truth = AS.ground_truth(src, sim, q, Ap, Am)
    t_truth = time.perf_counter() - t0
    batches = [core.make_batch(src, P.KSTEP_SEED0 + k, check_production=False) for k in range(AS.K_BRANCH)]
    critics = {"K0": src.heads}
    critics.update(AS.branch_critics(src, batches, P.BRANCH_RNG_SEED))
    t_branch = time.perf_counter() - t0 - t_truth
    rows, arrays = [], {}
    for name, critic in critics.items():
        sig = AS.critic_signal(critic, q, Ap, Am)
        Aa = AS.ascent_actions(q, sig["grad"])
        Ga = np.array([AS.rollout_return(src, sim, q.snaps[i], q.goals[i], Aa[i]) for i in range(len(Aa))])
        gain = Ga - truth["G0"]
        rows.append({"critic": name, **AS.summarise(sig, truth, Z, gain)})
        arrays[f"{name}_dq"] = sig["dq"].astype(np.float32)
        arrays[f"{name}_grad"] = sig["grad"].astype(np.float32)
        arrays[f"{name}_Q0"] = sig["Q0"].astype(np.float32)
        arrays[f"{name}_ascent_gain"] = gain.astype(np.float32)
    src.verify_unchanged()
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "arms.csv", rows)
    np.savez_compressed(out / "signal.npz", Z=Z, a_pi=q.a_pi, S=q.S.numpy(), phase=q.phase, episode=q.episode,
                        t=q.t, G0=truth["G0"], Gp=truth["Gp"], Gm=truth["Gm"], **arrays)
    meta = dict(checkpoint=str(path), set=set_name, trained_arm=src.arm, seed=src.seed, step=src.step,
                last_eval_success=src.last_eval_success, eval_tail_mean=src.eval_tail_mean,
                n_queries=int(len(q.a_pi)), G0_mean=float(truth["G0"].mean()),
                saturated_frac=float(np.mean(np.abs(q.a_pi) > 0.99)),
                seconds=dict(truth=t_truth, branches=t_branch, total=time.perf_counter() - t0),
                source_state_hash=src.frozen_hash)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    m.add_argument("--set", default="fpp_n20", choices=[s for s in P.SETS if s.startswith("fpp")])
    m.add_argument("--arms", nargs="+", default=None)
    m.add_argument("--seeds", nargs="+", type=int, default=None)
    m.add_argument("--stages", nargs="+", type=int, default=None)
    m.add_argument("--worker", type=int, default=0)
    m.add_argument("--n-workers", type=int, default=1)
    m.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    todo = [t for t in P.checkpoints(a.set) if (a.arms is None or t[0] in a.arms)
            and (a.seeds is None or t[1] in a.seeds) and (a.stages is None or t[2] in a.stages)]
    # SABER-trained checkpoints (primary) first, then the rest; workers take every n-th item
    todo.sort(key=lambda t: (t[0] != "saber", t[2], t[1]))
    todo = todo[a.worker::a.n_workers]
    core.require(len(todo) > 0, "no checkpoint matches the given filters")
    torch.set_num_threads(a.threads)
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    env_file = out / f"run_environment_worker{a.worker}.json"
    env_file.write_text(json.dumps(dict(
        argv=sys.argv, set=a.set, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
        code_sha256={str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in sorted([HERE / "run.py", HERE / "run_actor_signal.py", *(HERE / "fou").glob("*.py")])},
        versions={k: importlib.metadata.version(k) for k in ("torch", "numpy", "mujoco", "gymnasium")},
        python=sys.version), indent=2) + "\n")
    for arm, s, st, path in todo:
        dest = out / arm / f"seed_{s}" / f"step_{st:07d}"
        if dest.exists():
            print(f"skip (exists) {dest}", flush=True)
            continue
        meta = measure_one(path, dest, a.set, (arm, s, st))
        print(f"done {arm} seed {s} step {st}: {meta['seconds']['total']:.0f}s "
              f"(eval {meta['last_eval_success']:.2f}, G0 {meta['G0_mean']:.1f})", flush=True)


if __name__ == "__main__":
    main()
