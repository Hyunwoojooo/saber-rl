"""Actual-environment gates only. This command never starts production training."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

from protocol import ROOT, ORIGIN
from freeze import verify
from train import write_json


def main():
    snapshot = verify()
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    status_path = folder / "preflight_status.json"
    if status_path.exists():
        raise FileExistsError("Preflight already attempted; retain and inspect its evidence")
    state = {"status": "running", "source_manifest_sha256": snapshot, "stages": [],
             "started_utc": datetime.now(timezone.utc).isoformat()}
    old = folder / "upstream_tail_smoke/attempt_01"
    new = ROOT / "outputs/smoke/tail/seed_01/attempt_01"
    random_run = ROOT / "outputs/smoke/random512/seed_01/attempt_01"
    steps = [
        ("tests", [str(ROOT / "validate.py")]),
        ("historical_reuse", [str(ROOT / "audit_reuse.py")]),
        ("upstream_tail_smoke", [str(ORIGIN / "train.py"), "--arm", "support", "--seed", "1",
                                 "--profile", "smoke", "--device", "cuda", "--attempt-dir", str(old)]),
        ("new_smokes", [str(ROOT / "launch.py"), "--profile", "smoke", "--seeds", "1",
                        "--parallel", "2", "--device", "cuda"]),
        ("tail_parity", [str(ROOT / "check_tail_parity.py"), "--old", str(old), "--new", str(new)]),
        ("retrieval_pilot", [str(ROOT / "final_retrieval.py"), "--pilot-attempt", str(random_run)]),
        ("production_gate", [str(ROOT / "validate.py"), "--smoke-root", str(ROOT / "outputs")]),
    ]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1",
               NUMBA_CACHE_DIR=str(ROOT / "outputs/numba_cache"))
    try:
        for name, command in steps:
            state["stage"] = name
            write_json(status_path, state)
            print(f"START {name}", flush=True)
            old.parent.mkdir(parents=True, exist_ok=True)
            with (folder / f"preflight_{name}.log").open("x") as log:
                result = subprocess.run([sys.executable, "-B", *command], cwd=ROOT,
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            state["stages"].append({"stage": name, "returncode": result.returncode})
            print(f"EXIT {name}: {result.returncode}", flush=True)
            if result.returncode:
                raise RuntimeError(f"Preflight stopped at {name}; inspect its log, no production started")
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="blocked", error=str(exc))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, state)


if __name__ == "__main__":
    main()
