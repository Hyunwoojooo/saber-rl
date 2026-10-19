"""One-shot PAL completion barrier. The waiting process imports only stdlib."""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (ROOT, PAL, PAL_SHA, ASSIGNMENT, ASSIGNMENT_SHA, SEEDS,
                    completed_attempt, read_json, write_json, utcnow, verify_source)
from freeze import verify


def inspect_pal(pal=PAL):
    pal = Path(pal)
    state = read_json(pal / "outputs/automation/status.json")
    if state.get("source_manifest_sha256") != PAL_SHA:
        raise ValueError("PAL source identity changed")
    authorized = state.get("authorized_production", {})
    if (authorized.get("arm") != "pal" or authorized.get("seeds") != list(SEEDS)
            or authorized.get("steps_per_seed") != 2_000_000 or authorized.get("parallel") != 5):
        raise ValueError("Unexpected PAL execution plan")
    if state.get("status") in {"blocked", "failed", "cancelled"}:
        raise RuntimeError("PAL did not finish successfully; diagnostics withheld")
    counts = {"complete": 0, "running": 0, "pending": 0}
    for seed in SEEDS:
        parent = pal / "outputs/production/pal" / f"seed_{seed:02d}"
        attempts = sorted(p for p in parent.glob("attempt_*") if p.is_dir())
        if not attempts:
            counts["pending"] += 1
            continue
        if len(attempts) != 1 or attempts[0].name != "attempt_01":
            raise ValueError("PAL retries require manual review")
        path = attempts[0] / "status.json"
        if not path.exists():
            counts["running"] += 1
            continue
        run = read_json(path)
        required = {"arm": "pal", "seed": seed, "source_manifest_sha256": PAL_SHA,
                    "profile": "production", "checkpoint_continuation": False}
        if any(run.get(k) != v for k, v in required.items()):
            raise ValueError(f"PAL identity mismatch for seed {seed}")
        if run.get("status") not in {"running", "complete"}:
            raise RuntimeError(f"PAL seed {seed} failed; no automatic retry")
        counts[run["status"]] += 1
    ready = state.get("status") == "complete"
    if ready:
        if counts["complete"] != 10:
            raise ValueError("PAL queue complete but runs are missing")
        stages = state.get("stages", [])
        if not stages or stages[-1].get("name") != "analysis":
            raise ValueError("PAL final analysis was not run")
        if any(s.get("returncode") != 0 for s in stages):
            raise ValueError("A PAL stage did not pass")
        report = read_json(pal / "outputs/analysis/report.json")
        if (report.get("source_manifest_sha256") != PAL_SHA or report.get("problems") != []
                or not report.get("reference_reuse_audit_passed") or report.get("inference") is None
                or len(report.get("runs", [])) != 10):
            raise ValueError("PAL final result verification failed")
        for seed in SEEDS:
            completed_attempt(pal, "pal", seed, PAL_SHA)
    return {"ready": ready, "pal_stage": state.get("stage"), "pal_status": state.get("status"), **counts}


def upstream_processes(roots=(PAL, ASSIGNMENT), proc=Path("/proc")):
    roots = [Path(root).resolve() for root in roots]
    scripts = {"train.py", "launch.py", "analyze.py", "final_checks.py", "validate.py"}
    found = []
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            args = (item / "cmdline").read_bytes().decode().split("\0")
            if any(str(root / name) in args for root in roots for name in scripts):
                found.append(int(item.name))
            elif any(name in args for name in scripts) and (item / "cwd").resolve() in roots:
                found.append(int(item.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError("Cannot verify that upstream jobs have stopped") from exc
    return sorted(found)


def gpu_compute_pids():
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not value.isdigit() for value in values):
        raise RuntimeError("Cannot establish an idle GPU for uncontended diagnostics")
    return [int(value) for value in values]


def pipeline():
    python = [sys.executable, "-B"]
    return [
        ("tests", python + ["-m", "unittest", "discover", "-s", "tests", "-v"], 1800),
        ("failure_analysis", python + ["analyze_failures.py", "--output-dir", "outputs/failure_analysis"], 3600),
        ("retrieval_pilot", python + ["diagnose_retrieval.py", "--profile", "pilot",
                                     "--output-dir", "outputs/retrieval_pilot"], 3600),
        ("retrieval", python + ["diagnose_retrieval.py", "--profile", "full",
                               "--output-dir", "outputs/retrieval"], 12 * 3600),
        ("summary", python + ["summarize.py"], 600),
    ]


def run_pipeline(run_stage):
    for stage in pipeline():
        run_stage(*stage)


class Runner:
    def __init__(self, snapshot):
        self.folder = ROOT / "outputs/automation"
        self.snapshot = snapshot
        self.state = {"started_utc": utcnow(), "pid": os.getpid(), "stages": [],
                      "source_manifest_sha256": snapshot, "upstream_pal_sha256": PAL_SHA,
                      "new_training_authorized": False}

    def save(self, status, **fields):
        self.state.update(status=status, checked_utc=utcnow(), **fields)
        write_json(self.folder / "status.json", self.state)

    def check_stop(self):
        if (self.folder / "STOP").exists():
            raise InterruptedError("STOP requested; no automatic restart")

    def run_stage(self, name, command, timeout):
        self.check_stop()
        if verify() != self.snapshot or not inspect_pal()["ready"]:
            raise RuntimeError("Sources or PAL completion barrier changed")
        if upstream_processes() or gpu_compute_pids():
            raise RuntimeError("Compute became occupied; stage withheld")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONNOUSERSITE="1",
                   PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(ROOT / "outputs/numba_cache"))
        log_path = self.folder / f"{name}.log"
        stage = {"name": name, "command": command, "started_utc": utcnow(), "log": str(log_path)}
        self.state["stages"].append(stage)
        self.save("running", stage=name)
        print(f"{utcnow()} START {name}", flush=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            stage["pid"] = child.pid
            self.save("running")
            start = time.monotonic()
            try:
                while child.poll() is None:
                    self.check_stop()
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"{name} exceeded its fixed time limit; no query subsampling")
                    time.sleep(2)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        stage.update(finished_utc=utcnow(), returncode=child.returncode)
        self.save("running")
        if child.returncode:
            raise RuntimeError(f"{name} failed; see {log_path}")
        print(f"{utcnow()} PASS {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if not 30 <= args.poll_seconds <= 3600:
        parser.error("poll-seconds must be 30..3600")
    snapshot = verify()
    if not args.execute:
        print({"source_manifest_sha256": snapshot, "pal": inspect_pal(),
               "upstream_pids": upstream_processes(), "pipeline": [s[0] for s in pipeline()],
               "will_execute": False})
        return
    folder = ROOT / "outputs/automation"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "status.json").exists():
            raise RuntimeError("One-shot queue already has a record; manual review required")
        if any((ROOT / "outputs" / name).exists() for name in
               ("failure_analysis", "retrieval_pilot", "retrieval", "final")):
            raise RuntimeError("Retained outputs exist; will not overwrite")
        runner = Runner(snapshot)

        def interrupt(signum, frame):
            raise InterruptedError(f"Queue received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        deadline = time.monotonic() + 7 * 86400
        try:
            while True:
                runner.check_stop()
                state = inspect_pal()
                pids = upstream_processes()
                runner.save("waiting_for_pal", pal=state, upstream_pids=pids)
                print(f"{utcnow()} WAIT {state} pids={pids}", flush=True)
                if state["ready"] and not pids:
                    if not gpu_compute_pids():
                        break
                    runner.save("waiting_for_idle_compute")
                if time.monotonic() >= deadline:
                    raise TimeoutError("PAL/idle barrier not satisfied within seven days")
                time.sleep(args.poll_seconds)
            verify_source(PAL, PAL_SHA)
            verify_source(ASSIGNMENT, ASSIGNMENT_SHA)
            run_pipeline(runner.run_stage)
            runner.save("complete", finished_utc=utcnow())
        except BaseException as exc:
            runner.save("blocked", reason=str(exc), finished_utc=utcnow())
            (folder / "failure_traceback.txt").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
