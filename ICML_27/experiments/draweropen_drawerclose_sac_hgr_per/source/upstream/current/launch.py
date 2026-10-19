"""Explicitly requested production only; failed/claimed seeds are never skipped."""
import argparse
import json
import os
import subprocess
import sys
import time
import signal
import fcntl

from protocol import ROOT, SPEC, make_config, validate_parity
from manifest import verify
from gates import require_production_gates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--tasks", nargs="+", choices=SPEC["tasks"], default=SPEC["tasks"])
    p.add_argument("--methods", nargs="+", choices=SPEC["methods"], default=SPEC["methods"])
    p.add_argument("--seeds", nargs="+", type=int, choices=SPEC["seeds"], default=SPEC["seeds"])
    args = p.parse_args()
    if args.parallel < 1 or any(len(v) != len(set(v)) for v in (args.tasks, args.methods, args.seeds)):
        p.error("Positive parallelism and unique tasks/methods/seeds are required")
    validate_parity()
    jobs = [make_config(t, m, s) for t in args.tasks for s in args.seeds for m in args.methods]
    print(json.dumps({"runs": len(jobs), "steps": sum(c.total_steps for c in jobs),
                      "parallel": args.parallel, "names": [c.run_name for c in jobs]}, indent=2), flush=True)
    if not args.execute:
        return
    digest = verify()
    require_production_gates(digest)
    with (ROOT / ".production_launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_jobs(jobs, args.parallel)


def execute_jobs(jobs, parallel):
    for cfg in jobs:
        if (ROOT / ".production_claims" / cfg.run_name).exists() or (ROOT / "runs" / cfg.run_name).exists():
            raise FileExistsError(f"Already reserved: {cfg.run_name}; no automatic reuse/resume")
    log_root = ROOT / "launch_logs"
    log_root.mkdir(exist_ok=True)
    active = []
    failed = False
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def interrupted(signum, _frame):
        raise InterruptedError(f"Launcher signal {signum}; no automatic resume")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while jobs or active:
            while jobs and len(active) < parallel and not failed:
                cfg = jobs.pop(0)
                method = "her" if cfg.method == "sac" else "saber"
                handle = (log_root / f"{cfg.run_name}.log").open("x")
                cmd = [sys.executable, "-B", str(ROOT / "train.py"), "--task", cfg.task_name,
                       "--method", method, "--seed", str(cfg.seeds.env_seed), "--execute"]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active.append((proc, handle, cfg.run_name))
            remaining = []
            for proc, handle, name in active:
                code = proc.poll()
                if code is None:
                    remaining.append((proc, handle, name))
                else:
                    handle.close()
                    print(f"{name}: exit={code}", flush=True)
                    failed |= code != 0
            active = remaining
            if failed and not active:
                raise RuntimeError("Cohort halted after failure; unstarted jobs were not replaced or skipped")
            if active:
                time.sleep(5)
    finally:
        for proc, handle, _ in active:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            handle.close()


if __name__ == "__main__":
    main()
