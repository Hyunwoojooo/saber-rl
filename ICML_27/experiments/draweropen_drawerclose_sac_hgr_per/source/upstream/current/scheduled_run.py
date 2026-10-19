"""One-shot, fail-closed queue. Waiting uses only stdlib; no Torch/CUDA imports."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from protocol import ROOT, SCHEDULE
from manifest import verify
from gates import require_cpu_gate, require_production_gates, write_atomic


def alive(pid, expected_script=None):
    try:
        state = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state == "Z":
            return False
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return expected_script is None or str(expected_script) in command
    except (OSError, ValueError, TypeError):
        return False


def predecessor_state(prior=None, expected_digest=None):
    prior = Path(prior) if prior is not None else (ROOT / SCHEDULE["predecessor"]).resolve()
    expected_digest = expected_digest or SCHEDULE["predecessor_source_manifest_sha256"]
    status_path = prior / "outputs/automation/status.json"
    try:
        state = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return {"state": "waiting", "reason": "predecessor status file is being written"}
    if state.get("source_manifest_sha256") != expected_digest:
        raise RuntimeError("Unexpected predecessor experiment snapshot")
    if state.get("status") == "running":
        if not alive(state.get("pid"), prior / "after_training.py"):
            raise RuntimeError("Predecessor reports running but its orchestrator is absent; inspect manually")
        return {"state": "waiting", "reason": "predecessor active", "stage": state.get("stage")}
    if state.get("status") != "complete":
        raise RuntimeError(f"Predecessor did not complete successfully: {state.get('status')}")
    stages = state.get("stages", [])
    if [s.get("stage") for s in stages] != SCHEDULE["predecessor_required_stages"] or any(
            s.get("returncode") != 0 for s in stages):
        raise RuntimeError("Predecessor training/analysis/retrieval audit did not all succeed")
    manifest_path = prior / "source_manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_digest:
        raise RuntimeError("Predecessor source manifest changed")
    manifest = json.loads(manifest_path.read_text())["sha256"]
    for name, digest in manifest.items():
        if hashlib.sha256((prior / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Predecessor source changed: {name}")
    for seed in range(1, 11):
        seed_root = prior / f"outputs/production/random512/seed_{seed:02d}"
        attempts = sorted(p for p in seed_root.glob("attempt_*") if p.is_dir())
        if len(attempts) != 1:
            raise RuntimeError(f"Unexpected attempt count for predecessor seed {seed}")
        run = json.loads((attempts[0] / "status.json").read_text())
        summary = json.loads((attempts[0] / "run/summary.json").read_text())
        if run.get("status") != "complete" or run.get("source_manifest_sha256") != expected_digest:
            raise RuntimeError(f"Predecessor seed {seed} is not a valid completed run")
        if summary.get("total_steps") != 2000000 or summary.get("total_updates") != 1995001:
            raise RuntimeError(f"Predecessor seed {seed} has incomplete training")
    analysis = json.loads((prior / "outputs/analysis/report.json").read_text())
    audit = json.loads((prior / "outputs/final_retrieval/report.json").read_text())
    if analysis.get("failures_or_missing") != [] or len(analysis.get("runs", [])) != 20 or not analysis.get("inference"):
        raise RuntimeError("Predecessor primary analysis is incomplete")
    if audit.get("complete") is not True or audit.get("training_checkpoints") != 20 or audit.get("tail_parity_mismatches") != 0:
        raise RuntimeError("Predecessor final retrieval diagnostic is incomplete")
    if alive(state.get("pid"), prior / "after_training.py"):
        return {"state": "waiting", "reason": "predecessor finishing shutdown"}
    return {"state": "ready", "reason": "all 10 runs and both post-analyses complete"}


def resource_state():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
                             "--id=" + SCHEDULE["cuda_visible_devices"]], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect GPU ownership: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError(f"Unexpected GPU process report: {lines}")
    pids = sorted({int(line) for line in lines})
    if pids:
        return {"state": "waiting", "reason": "GPU has active compute processes", "pids": pids}
    disk = shutil.disk_usage(ROOT).free / 1024 ** 3
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    ram = int(meminfo["MemAvailable"].split()[0]) / 1024 ** 2
    if disk < SCHEDULE["minimum_free_disk_gib"] or ram < SCHEDULE["minimum_available_ram_gib"]:
        raise RuntimeError(f"Insufficient disk/RAM: {disk:.1f}/{ram:.1f} GiB")
    return {"state": "ready", "free_disk_gib": disk, "available_ram_gib": ram}


def stage_plan():
    stages = [("cpu_revalidation", ["validate.py"]), ("gpu_validation", ["gpu_validate.py", "--execute"])]
    for task in SCHEDULE["production_task_order"]:
        stages.append((f"production_{task}", ["launch.py", "--execute", "--parallel",
                        str(SCHEDULE["production_parallel"]), "--tasks", task]))
        stages.append((f"analysis_{task}", ["analyze.py", "--tasks", task,
                        "--output", str(ROOT / f"analysis/{task}.json")]))
    stages.append(("final_analysis", ["analyze.py"]))
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.execute and args.check:
        parser.error("Choose --execute or --check")
    digest = verify()
    require_cpu_gate(digest)
    if not args.execute:
        result = {"source_manifest_sha256": digest, "stages": stage_plan(), "specification": SCHEDULE}
        if args.check:
            result["predecessor"] = predecessor_state()
        print(json.dumps(result, indent=2))
        return
    destination = ROOT / "automation"
    destination.mkdir(exist_ok=True)
    with (destination / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_queue(destination, digest)


def execute_queue(destination, digest):
    status_path = destination / "status.json"
    with status_path.open("x") as handle:
        json.dump({"status": "initializing"}, handle)
    state = {"status": "waiting", "pid": os.getpid(), "source_manifest_sha256": digest,
             "started_utc": datetime.now(timezone.utc).isoformat(), "stage": "waiting_for_predecessor", "stages": []}
    child = None

    def save():
        state["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
        write_atomic(status_path, state)

    def stop(signum, _frame):
        raise InterruptedError(f"Queue stopped by signal {signum}; no automatic resume")

    signal.signal(signal.SIGTERM, stop)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl",
               CUDA_VISIBLE_DEVICES=SCHEDULE["cuda_visible_devices"], NUMBA_CACHE_DIR=str(ROOT / ".cache/numba"))
    save()
    try:
        while True:
            if verify() != digest:
                raise RuntimeError("Scheduled source changed while waiting")
            prior = predecessor_state()
            state["predecessor"] = prior
            if prior["state"] == "ready":
                state["resources"] = resource_state()
                if state["resources"]["state"] == "ready":
                    break
                state["stage"] = "waiting_for_gpu"
            save()
            time.sleep(SCHEDULE["poll_seconds"])
        for name, args in stage_plan():
            verify()
            if name.startswith("production_"):
                require_production_gates(digest)
            state.update(status="running", stage=name, stage_started_utc=datetime.now(timezone.utc).isoformat())
            save()
            stage_env = env.copy()
            if name == "cpu_revalidation" or name.startswith("analysis_") or name == "final_analysis":
                stage_env["CUDA_VISIBLE_DEVICES"] = ""
            print(f"START {name}", flush=True)
            with (destination / f"{name}.log").open("x") as log:
                child = subprocess.Popen([sys.executable, "-B", str(ROOT / args[0]), *args[1:]],
                                         cwd=ROOT, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                save()
                while child.poll() is None:
                    time.sleep(10)
                    save()
                code = child.returncode
                child = None
            state.pop("child_pid", None)
            state["stages"].append({"name": name, "returncode": code,
                                     "finished_utc": datetime.now(timezone.utc).isoformat()})
            save()
            if code:
                raise RuntimeError(f"{name} failed (exit {code}); subsequent stages were not started")
        state.update(status="complete", stage="finished")
    except BaseException as exc:
        state.update(status="blocked", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
