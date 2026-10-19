"""Bounded parallel launcher for explicit subsets of the fixed 30-run plan."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from protocol import ROOT, ARMS, SEEDS
from freeze import verify
from analyze import list_attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--profile", choices=("production", "smoke"), default="production")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-infrastructure", metavar="REASON")
    args = ap.parse_args()
    if args.parallel < 1 or len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        ap.error("Parallelism must be positive; arms/seeds must be unique")
    snapshot = verify()
    if args.profile == "production" and not args.dry_run:
        gate = json.loads((ROOT / "validation" / "gate.json").read_text())
        if gate.get("source_manifest_sha256") != snapshot or gate.get("passed") is not True:
            raise SystemExit("Production requires tests and all three smoke runs on this snapshot")
    parent = args.output_root.resolve() / args.profile
    plan = {"arms": list(ARMS), "seeds": list(SEEDS), "source_manifest_sha256": snapshot,
            "profile": args.profile, "historical_reuse": "not approved",
            "primary_interval_resamples": 10_000}
    jobs = []
    for seed in args.seeds:
        for arm in args.arms:
            seed_dir = parent / arm / f"seed_{seed:02d}"
            attempts = list_attempts(seed_dir)
            if attempts:
                state = json.loads((attempts[-1] / "status.json").read_text())
                if not args.retry_infrastructure or state["status"] not in {"failed_infrastructure", "interrupted"}:
                    raise SystemExit(f"Refusing overwrite/replacement: {seed_dir} ({state['status']})")
            attempt = seed_dir / f"attempt_{len(attempts) + 1:02d}"
            command = [sys.executable, str(ROOT / "train.py"), "--arm", arm, "--seed", str(seed),
                       "--profile", args.profile, "--device", args.device, "--attempt-dir", str(attempt)]
            if args.retry_infrastructure:
                command += ["--retry-reason", args.retry_infrastructure]
            jobs.append((command, attempt))
    if args.dry_run:
        print(json.dumps({"plan": plan, "parallel": args.parallel,
                          "commands": [j[0] for j in jobs]}, indent=2))
        return
    parent.mkdir(parents=True, exist_ok=True)
    plan_path = parent / "plan.json"
    try:
        with plan_path.open("x") as f:
            json.dump(plan, f, indent=2)
    except FileExistsError:
        if json.loads(plan_path.read_text()) != plan:
            raise SystemExit("Output root already belongs to a different plan")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl")
    active, failures = [], []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while jobs or active:
            while jobs and len(active) < args.parallel:
                command, attempt = jobs.pop(0)
                attempt.parent.mkdir(parents=True, exist_ok=True)
                log = (attempt.parent / (attempt.name + ".log")).open("x")
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                active.append((child, log, attempt))
                print(f"START pid={child.pid} {attempt}", flush=True)
            for job in active[:]:
                child, log, attempt = job
                code = child.poll()
                if code is not None:
                    log.close()
                    active.remove(job)
                    if code:
                        failures.append(str(attempt))
                    print(f"EXIT code={code} {attempt}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for child, _, _ in active:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child, log, _ in active:
            child.wait()
            log.close()
    if failures:
        raise SystemExit(f"Failed attempts retained (not automatically rerun): {failures}")


if __name__ == "__main__":
    main()
