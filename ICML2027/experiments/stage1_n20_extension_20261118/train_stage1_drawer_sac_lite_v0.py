#!/usr/bin/env python3
"""Stage 1 Drawer SAC run: the executed FPP-v2 Drawer SAC source (origin/main 5facb92,
ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source, extracted to sources/drawer_sac).

The config comes from that source's own protocol.make_config; only the allowed seed list is widened
in memory and the output root is set here. Before training, the config is checked against the
recorded config of an executed run of the same task/method (templates/), allowing differences only in
seeds, run_name, output_root and experiment_id. Training then follows train.py's production path
(thread pinning, run_training, validate_run) without its seed-1-5 production gates and claim folder.

  .venv/bin/python <this file> --task drawer_open --method saber --seed 11 --output-root <dir>
  (--smoke-steps N: short check run, never used for production)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources" / "drawer_sac"
TEMPLATES = {("drawer_open", m): HERE / "templates" / f"draweropen_sac_{m}_seed6_launch_record.json" for m in ("her", "saber")}
TEMPLATES.update({("drawer_close", m): HERE / "templates" / f"drawerclose_sac_{m}_seed1_config.json" for m in ("her", "saber")})
ALLOWED_DIFF = {"seeds", "run_name", "output_root", "experiment_id"}


def normalised(d: dict) -> dict:
    return json.loads(json.dumps(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["drawer_open", "drawer_close"], required=True)
    ap.add_argument("--method", choices=["her", "saber"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--smoke-steps", type=int, default=0)
    a = ap.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(SRC / ".cache" / "numba"))
    sys.path.insert(0, str(SRC))
    import protocol                                    # also puts sources/drawer_sac/code/src on sys.path
    protocol.SPEC["seeds"] = sorted(set(protocol.SPEC["seeds"]) | {a.seed})
    cfg = protocol.make_config(a.task, a.method, a.seed, a.device, False, a.output_root)
    ref = json.loads(TEMPLATES[(a.task, a.method)].read_text())
    ref = ref.get("config", ref)
    mine = normalised(cfg.as_dict())
    bad = sorted(k for k in set(mine) | set(ref) if k not in ALLOWED_DIFF and k in mine and mine.get(k) != ref.get(k))
    if bad:
        raise SystemExit(f"config differs from the executed reference run in {bad}")
    if a.smoke_steps:
        cfg.total_steps = a.smoke_steps
        cfg.eval_freq = cfg.checkpoint_every_steps = cfg.log_every_steps = min(cfg.eval_freq, a.smoke_steps)
        cfg.run_name += f"_smoke{a.smoke_steps}"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from metacog_lite_v0.trainer import run_training
    run_dir = run_training(cfg, SRC)
    if not a.smoke_steps:
        from analyze import validate_run
        print(json.dumps(validate_run(run_dir, cfg), default=str)[:2000])
    print(run_dir)


if __name__ == "__main__":
    main()
