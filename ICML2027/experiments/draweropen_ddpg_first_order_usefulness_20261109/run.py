#!/usr/bin/env python3
"""Measure first-order usefulness on saved checkpoints (no training).

  .venv/bin/python <this dir>/run.py plan [--set a1|mature|ext2m]
  .venv/bin/python <this dir>/run.py measure --out outputs/<name> [--set a1] [--arms saber her] [--seeds 101]
                                             [--stages 10000 ...] [--batches 64] [--no-r2] [--no-ksteps]

Each checkpoint writes <out>/<arm>/seed_<s>/step_<n>/ and is never overwritten.
Use one --out per set (e.g. outputs/main for a1, outputs/late_mature, outputs/late_ext2m).
"""
from __future__ import annotations

import argparse
import csv
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
from fou import core, protocol as P  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load(set_name: str, path: Path) -> core.Source:
    info = P.set_info(set_name)
    if info["family"] == "drawer_ddpg":
        return core.load_source(path)
    src = core.load_source_fpp(path, info["code_src"])
    core.require(src.her_inclusive == info["her_inclusive"],
                 f"HER convention of the code ({src.her_inclusive}) differs from the protocol ({info['her_inclusive']})")
    return src


def measure_one(path: Path, out: Path, n_batches: int, use_r2: bool, use_ksteps: bool,
                expect: tuple[str, int, int] | None = None, set_name: str = "a1") -> dict:
    t0 = time.perf_counter()
    src = load(set_name, path)
    if expect is not None:
        core.require((src.arm, src.seed, src.step) == expect,
                     f"checkpoint content {(src.arm, src.seed, src.step)} does not match its path label {expect}: {path}")
    params = core.Params(src.heads)
    r1_episodes, r2_episodes, r2_queries = P.bank_settings(src.family)
    banks = [core.bank_r1(src, r1_episodes)]
    if use_r2:
        banks.append(core.bank_r2(src, r2_episodes, r2_queries))
    t_bank = time.perf_counter() - t0
    gR0 = {b.name: core.risk_grad(params, params.flat0, b) for b in banks}
    with torch.no_grad():
        R0 = {b.name: float(core.risk(params, params.flat0, b)) for b in banks}
    rng = np.random.default_rng(P.MASK_RNG_SEED)
    rows, samples = [], []
    for b in range(n_batches):
        bt = core.make_batch(src, P.BATCH_SEED0 + b, check_production=True)
        row, ps = core.analyse_batch(src, params, bt, banks, gR0, R0, rng)
        rows.append(row)
        samples.append(ps)
    t_first = time.perf_counter() - t0 - t_bank
    krows = []
    if use_ksteps:
        kb = [core.make_batch(src, P.KSTEP_SEED0 + k, check_production=False) for k in range(max(P.K_RECORD))]
        krows = core.run_branches(src, params, kb, banks, P.K_RECORD, P.BRANCH_RNG_SEED)
    src.verify_unchanged()
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "batches.csv", rows)
    if krows:
        write_csv(out / "ksteps.csv", krows)
    np.savez_compressed(out / "per_sample.npz", **{k: np.stack([s[k] for s in samples]) for k in samples[0]})
    meta = dict(checkpoint=str(path), set=set_name, family=src.family, her_inclusive=src.her_inclusive,
                phase_names=list(src.task.phase_names), trained_arm=src.arm, seed=src.seed, step=src.step,
                checkpoint_kind=src.kind, total_updates=src.total_updates,
                last_eval_success=src.last_eval_success, eval_tail_mean=src.eval_tail_mean,
                banks={b.name: dict(n=int(len(b.y)), R0=R0[b.name], grad_norm=float(gR0[b.name].norm()),
                                    phase_frac=np.bincount(b.phase, minlength=4).tolist()) for b in banks},
                n_batches=n_batches, k_record=list(P.K_RECORD) if use_ksteps else [],
                seconds=dict(banks=t_bank, first_order=t_first, total=time.perf_counter() - t0),
                source_state_hash=src.frozen_hash)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--set", default="a1", choices=P.SETS)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    m.add_argument("--set", default="a1", choices=P.SETS)
    m.add_argument("--arms", nargs="+", default=None)
    m.add_argument("--seeds", nargs="+", type=int, default=None)
    m.add_argument("--stages", nargs="+", type=int, default=None)
    m.add_argument("--batches", type=int, default=P.N_BATCHES)
    m.add_argument("--no-r2", action="store_true")
    m.add_argument("--no-ksteps", action="store_true")
    m.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    todo = P.checkpoints(a.set)
    if a.cmd == "plan":
        for arm, s, st, p in todo:
            print(("ok   " if p.exists() else "MISS ") + f"{arm:8s} seed {s:3d} step {st:>9,d}  {p}")
        print(f"{sum(p.exists() for *_, p in todo)}/{len(todo)} checkpoints present")
        return
    arms = a.arms if a.arms is not None else (list(P.PRIMARY_ARMS) if a.set == "a1" else None)
    todo = [t for t in todo if (arms is None or t[0] in arms) and (a.seeds is None or t[1] in a.seeds)
            and (a.stages is None or t[2] in a.stages)]
    core.require(len(todo) > 0, "no checkpoint matches the given filters")
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    run_meta = out / "run_environment.json"
    if not run_meta.exists():
        run_meta.write_text(json.dumps(dict(
            argv=sys.argv, set=a.set,
            git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
            # the measurement code may be uncommitted, so record the files themselves
            code_sha256={str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in sorted([HERE / "run.py", *(HERE / "fou").glob("*.py")])},
            versions={k: importlib.metadata.version(k) for k in ("torch", "numpy", "metaworld", "mujoco", "gymnasium")},
            python=sys.version), indent=2) + "\n")
    else:
        core.require(json.loads(run_meta.read_text()).get("set", "a1") == a.set,
                     f"{out} already holds another checkpoint set; use a separate --out per set")
    for arm, s, st, path in todo:
        dest = out / arm / f"seed_{s}" / f"step_{st:07d}"
        if dest.exists():
            print(f"skip (exists) {dest}", flush=True)
            continue
        meta = measure_one(path, dest, a.batches, not a.no_r2, not a.no_ksteps, expect=(arm, s, st), set_name=a.set)
        print(f"done {a.set} {arm} seed {s} step {st}: {meta['seconds']['total']:.1f}s "
              f"(last eval success {meta['last_eval_success']:.2f})", flush=True)


if __name__ == "__main__":
    main()
