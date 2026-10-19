"""Run the approved production queue, then strict analysis. No retries."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from protocol import ROOT
from freeze import verify
from audit_reuse import verify_catalog
from train import write_json


def main():
    snapshot = verify()
    verify_catalog(ROOT / "validation/reuse_catalog.json")
    gate = json.loads((ROOT / "validation/gate.json").read_text())
    if not gate.get("passed") or gate["source_manifest_sha256"] != snapshot:
        raise ValueError("Production gate not valid")
    folder = ROOT / "outputs/automation"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "status.json").exists():
        raise FileExistsError("Queue already has an attempt; inspect it rather than replacing it")
    state = {"status": "running", "pid": os.getpid(), "source_manifest_sha256": snapshot,
             "started_utc": datetime.now(timezone.utc).isoformat(), "stages": []}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1",
               NUMBA_CACHE_DIR=str(ROOT / "outputs/numba_cache"))
    stages = [
        ("training", ["launch.py", "--arms", "random512", "--parallel", "5", "--device", "cuda"]),
        ("analysis", ["analyze.py"]),
        ("final_retrieval", ["final_retrieval.py"]),
    ]
    try:
        for name, arguments in stages:
            state["stage"] = name
            write_json(folder / "status.json", state)
            print(f"START {name}", flush=True)
            stage_env = env.copy()
            if name != "training":
                stage_env["CUDA_VISIBLE_DEVICES"] = ""
            with (folder / f"{name}.log").open("x") as log:
                result = subprocess.run([sys.executable, "-B", str(ROOT / arguments[0]), *arguments[1:]],
                                        cwd=ROOT, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
            state["stages"].append({"stage": name, "returncode": result.returncode,
                                     "finished_utc": datetime.now(timezone.utc).isoformat()})
            if result.returncode:
                raise RuntimeError(f"{name} failed: {result.returncode}; retained, not retried")
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="blocked", error=str(exc))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(folder / "status.json", state)


if __name__ == "__main__":
    main()
