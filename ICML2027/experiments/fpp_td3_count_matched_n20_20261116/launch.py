#!/usr/bin/env python3
"""Train the FPP TD3 context-eligible count-matched control on seeds 11-30, paired with the n20 HER/SABER runs.

The frozen control package (experiment_FPP/td3_saber_context_eligible_count_matched_ls512) only allows
seeds 1-5 in its run.py. This launcher reuses that package's own build_command and freeze check
unchanged, only widening the allowed seed list in memory. It never edits the frozen folder.

  .venv/bin/python <this dir>/launch.py plan
  .venv/bin/python <this dir>/launch.py run --device cuda --threads 1 --max-parallel 10
  .venv/bin/python <this dir>/launch.py status
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PKG = REPO / "experiment_FPP" / "td3_saber_context_eligible_count_matched_ls512"
PYTHON = REPO / ".venv" / "bin" / "python"
SEEDS = tuple(range(11, 31))
STEPS = 1_000_000
OUT = HERE / "outputs"
BASE = OUT / "runs"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def frozen_run_module():
    spec = importlib.util.spec_from_file_location("cm_frozen_run", PKG / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.verify_production_freeze(PKG)          # same check the frozen run.py does for production
    original = mod.load_config

    def load_config(folder):
        cfg = original(folder)
        cfg["seeds"] = sorted(set(cfg["seeds"]) | set(SEEDS))   # the only change: allow seeds 11-30
        return cfg

    mod.load_config = load_config
    return mod


def build_command(mod, seed: int, device: str) -> list[str]:
    cmd = mod.build_command(stage="production", seed=seed, device=device, output_root=str(BASE), validate_only=False)
    assert cmd[0] == str(PYTHON) or Path(cmd[0]).resolve() == PYTHON.resolve(), cmd[0]
    return cmd


def run_dir(seed: int) -> Path:
    return BASE / f"fetch_pick_and_place_td3_saber_context_eligible_count_matched_s{seed}_{STEPS}_ls512"


def child_env(threads: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PKG / "code" / "src")
    for v in THREAD_VARS:
        env[v] = str(threads)
    return env


def tree_sha256(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(str(f.relative_to(folder)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def cmd_plan(_a) -> None:
    mod = frozen_run_module()
    for s in SEEDS:
        print(shlex.join(build_command(mod, s, "<device>")))
    print("code sha256:", tree_sha256(PKG / "code" / "src")[:16])


def cmd_run(a) -> None:
    mod = frozen_run_module()
    seeds = tuple(a.seeds) if a.seeds else SEEDS
    for s in seeds:
        if run_dir(s).exists():
            raise SystemExit(f"Refusing to overwrite existing run: {run_dir(s)}")
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    meta = dict(started=datetime.now().isoformat(timespec="seconds"), device=a.device, threads=a.threads,
                max_parallel=a.max_parallel, steps=STEPS, seeds=list(seeds),
                git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
                code_sha256=tree_sha256(PKG / "code" / "src"), python=sys.version, host=os.uname().nodename,
                commands={f"seed_{s}": build_command(mod, s, a.device) for s in seeds})
    (OUT / "launch_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT / "pip_freeze.txt").write_text(subprocess.run([str(PYTHON), "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    running, pending, failed = {}, list(seeds), []
    status_path = OUT / "launcher_status.json"

    def write_status():
        status_path.write_text(json.dumps(dict(
            updated=datetime.now().isoformat(timespec="seconds"), pending=pending,
            running={str(s): p.pid for s, p in running.items()}, failed=failed), indent=2) + "\n")

    while pending or running:
        while pending and len(running) < a.max_parallel and not failed:
            s = pending.pop(0)
            fh = open(logs / f"count_matched_seed_{s}.log", "w")
            running[s] = subprocess.Popen(build_command(mod, s, a.device), env=child_env(a.threads),
                                          stdout=fh, stderr=subprocess.STDOUT)
            write_status()
        time.sleep(20)
        for s, proc in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                del running[s]
                if rc != 0:
                    failed.append(f"seed_{s} (exit {rc})")       # no retry; stop starting new runs
                write_status()
    write_status()
    (OUT / ("FAILED" if failed else "DONE")).write_text(datetime.now().isoformat(timespec="seconds") + "\n")


def cmd_status(_a) -> None:
    st = OUT / "launcher_status.json"
    if st.exists():
        s = json.loads(st.read_text())
        print(f"launcher: updated {s['updated']}, running {len(s['running'])}, pending {len(s['pending'])}, failed {s['failed']}")
    for s in SEEDS:
        d = run_dir(s)
        log = d / "run.log"
        step = -1
        if log.exists():
            lines = [ln for ln in log.read_text(errors="replace").splitlines() if "step=" in ln]
            step = int(lines[-1].split("step=")[1].split()[0]) if lines else 0
        ck = len(list(d.glob("checkpoint_step_*.pt"))) if d.exists() else 0
        line = f"count_matched seed {s}: step {step:>9,d}  checkpoints {ck:2d}/20"
        summ = d / "summary.json"
        if summ.exists():
            line += f"  | AUC {json.loads(summ.read_text()).get('eval_AUC_success'):.4f}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--device", required=True, choices=["cpu", "cuda"])
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-parallel", type=int, default=10)
    r.add_argument("--seeds", type=int, nargs="+", default=None)
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
