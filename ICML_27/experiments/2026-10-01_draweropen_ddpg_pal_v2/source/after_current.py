"""One-shot local queue. Waiting imports only the standard library."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from freeze import verify

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "2026-09-25_draweropen_ddpg_assignment"
REFERENCE_SHA = "d281263084807ec6010986dc762301dce420b004e022c88985e51dcc50ccf496"
ARMS = ("support", "random", "td_topk")
SMOKE = "outputs/automatic_cuda_smoke"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def inspect_reference(reference=REFERENCE):
    root = Path(reference) / "outputs/production"
    plan = read_json(root / "plan.json")
    if (plan.get("source_manifest_sha256") != REFERENCE_SHA
            or plan.get("arms") != list(ARMS) or plan.get("seeds") != list(range(1, 11))
            or plan.get("profile") != "production"):
        raise ValueError("Assignment cohort plan changed; manual review required")
    result = {"complete": 0, "running": 0, "pending": 0, "total": 30}
    for seed in range(1, 11):
        for arm in ARMS:
            attempts = sorted(p for p in (root / arm / f"seed_{seed:02d}").glob("attempt_*")
                              if p.is_dir())
            if not attempts:
                result["pending"] += 1
                continue
            if len(attempts) != 1:
                raise ValueError(f"Multiple attempts require manual review: {arm}/{seed}")
            status_path = attempts[0] / "status.json"
            if not status_path.exists():
                result["running"] += 1
                continue
            status = read_json(status_path)
            if (status.get("source_manifest_sha256") != REFERENCE_SHA
                    or status.get("arm") != arm or status.get("seed") != seed
                    or status.get("profile") != "production"
                    or status.get("checkpoint_continuation") is not False):
                raise ValueError(f"Assignment identity/continuation mismatch: {arm}/{seed}")
            state = status.get("status")
            if state not in {"complete", "running"}:
                raise ValueError(f"Assignment run {arm}/{seed} is {state}; PAL withheld")
            result[state] += 1
    return result


def reference_processes(reference=REFERENCE, proc=Path("/proc")):
    found = []
    reference = Path(reference).resolve()
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / "cmdline").read_bytes().decode().split("\0")
            absolute = any(str(reference / name) in args for name in ("train.py", "launch.py"))
            relative = any(name in args for name in ("train.py", "launch.py"))
            if absolute or (relative and (directory / "cwd").resolve() == reference):
                found.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError("Cannot establish that the reference processes stopped") from exc
    return sorted(found)


def require_idle_gpu():
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if lines:
        raise RuntimeError(f"GPU is not idle or its process list is unsupported: {lines}")


def audit_early_checkpoints(production, accepted):
    accepted = dict(accepted)
    for seed in range(1, 11):
        if str(seed) in accepted:
            continue
        path = Path(production) / f"pal/seed_{seed:02d}/attempt_01/run/metrics.csv"
        if not path.exists():
            continue
        points = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                # A concurrently appended final row may not have finished yet.
                if None in row or None in row.values():
                    continue
                value = row.get("eval/env_success_mean", "")
                if value == "":
                    continue
                step = int(row["step"])
                if step > 20000:
                    break
                success = float(value)
                if not math.isfinite(success) or not 0 <= success <= 1:
                    raise ValueError(f"Invalid early evaluation for PAL seed {seed}")
                fields = ("loss/critic", "loss/actor", "pal/base_loss", "pal/normalizer",
                          "critic/q_policy_mean", "critic/q_policy_std")
                if not all(math.isfinite(float(row[key])) for key in fields):
                    raise ValueError(f"Invalid early loss/logs for PAL seed {seed}")
                z, base = float(row["pal/normalizer"]), float(row["pal/base_loss"])
                if z < 1 or not math.isclose(float(row["loss/critic"]), base / z, rel_tol=1e-5, abs_tol=1e-7):
                    raise ValueError(f"PAL normalization mismatch for seed {seed}")
                points.append({"step": step, "success": success})
                if step == 20000:
                    break
        if [p["step"] for p in points] not in ([], [10000], [10000, 20000]):
            raise ValueError(f"Unexpected early evaluation grid for PAL seed {seed}")
        if len(points) == 2:
            accepted[str(seed)] = {"passed": True, "points": points, "checked_utc": utcnow(),
                                   "performance_threshold": None}
    return accepted


def pipeline():
    python = [sys.executable, "-B"]
    return [
        ("static", python + ["static_check.py"], 300, True),
        ("cpu_tests", python + ["validate.py", "--run-tests"], 1200, True),
        ("reference", python + ["final_checks.py", "--stage", "reference"], 1800, False),
        ("cuda_loss", python + ["final_checks.py", "--stage", "cuda_loss"], 600, False),
        ("cuda_smoke", python + ["launch.py", "--profile", "smoke", "--device", "cuda",
          "--seeds", "1", "--parallel", "1", "--output-root", SMOKE, "--execute"], 5400, False),
        ("smoke_gate", python + ["validate.py", "--run-tests", "--smoke-root", SMOKE], 1800, False),
        ("readiness", python + ["final_checks.py", "--stage", "readiness", "--smoke-root", SMOKE], 1800, False),
        ("preview", python + ["launch.py", "--profile", "production", "--device", "cuda",
          "--parallel", "5", "--dry-run"], 300, True),
        ("production", python + ["launch.py", "--profile", "production", "--device", "cuda",
          "--parallel", "5", "--execute"], 7 * 86400, False),
        ("analysis", python + ["analyze.py", "--output-root", "outputs", "--saber-root",
          str(REFERENCE)], 3600, True),
    ]


def run_pipeline(run_stage):
    for stage in pipeline():
        run_stage(*stage)


class Runner:
    def __init__(self, folder, snapshot):
        self.folder = Path(folder)
        self.snapshot = snapshot
        self.state = {"started_utc": utcnow(), "pid": os.getpid(),
                      "source_manifest_sha256": snapshot, "reference": str(REFERENCE),
                      "authorized_production": {"arm": "pal", "seeds": list(range(1, 11)),
                                                "steps_per_seed": 2_000_000, "parallel": 5},
                      "stages": []}

    def save(self, status, **fields):
        self.state.update(status=status, checked_utc=utcnow(), **fields)
        write_json(self.folder / "status.json", self.state)

    def stop_requested(self):
        if (self.folder / "STOP").exists():
            raise InterruptedError("STOP file requested cancellation; no automatic restart")

    def run_stage(self, name, command, timeout, cpu_only):
        self.stop_requested()
        if verify() != self.snapshot or reference_processes():
            raise RuntimeError("Source changed or reference experiment is active again")
        if name in {"cuda_loss", "cuda_smoke", "readiness", "production"}:
            require_idle_gpu()
        if name == "production" and shutil.disk_usage(ROOT).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free; production withheld")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMBA_NUM_THREADS="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                   MUJOCO_GL="egl")
        if cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        log_path = self.folder / f"{name}.log"
        stage = {"name": name, "started_utc": utcnow(), "command": command, "log": str(log_path)}
        self.state["stages"].append(stage)
        self.save("production_running" if name == "production" else "validating", stage=name)
        print(f"{utcnow()} START {name}", flush=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            stage["pid"] = child.pid
            self.save(self.state["status"])
            start = time.monotonic()
            next_early_check = start
            early = {}
            try:
                while child.poll() is None:
                    self.stop_requested()
                    if name == "production" and time.monotonic() >= next_early_check:
                        early = audit_early_checkpoints(ROOT / "outputs/production", early)
                        write_json(self.folder / "early_checkpoints.json", early)
                        next_early_check = time.monotonic() + 60
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"{name} exceeded {timeout} seconds")
                    time.sleep(2)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        stage.update(finished_utc=utcnow(), returncode=child.returncode)
        self.save(self.state["status"])
        if child.returncode:
            raise RuntimeError(f"{name} failed ({child.returncode}); see {log_path}")
        if name == "production":
            early = audit_early_checkpoints(ROOT / "outputs/production", early)
            write_json(self.folder / "early_checkpoints.json", early)
            if len(early) != 10:
                raise ValueError("Production finished without ten validated early-checkpoint records")
        print(f"{utcnow()} PASS {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.execute and args.check:
        parser.error("Choose --check or --execute")
    if not 30 <= args.poll_seconds <= 3600:
        parser.error("Polling interval must be 30--3600 seconds")
    snapshot = verify()
    if not args.execute:
        print(json.dumps({"source_manifest_sha256": snapshot,
                          "reference": inspect_reference(), "reference_pids": reference_processes(),
                          "will_execute": False, "pipeline": [s[0] for s in pipeline()]}, indent=2))
        return
    folder = ROOT / "outputs/automation"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "status.json").exists():
            raise RuntimeError("One-shot queue already has a record; manual review required")
        for path in (ROOT / "outputs/production", ROOT / SMOKE):
            if path.exists():
                raise RuntimeError(f"Refusing to reuse output directory: {path}")
        runner = Runner(folder, snapshot)

        def interrupt(signum, frame):
            raise InterruptedError(f"Queue received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        deadline = time.monotonic() + 72 * 3600
        try:
            while True:
                runner.stop_requested()
                counts = inspect_reference()
                active = reference_processes()
                runner.save("waiting", reference_progress=counts, reference_pids=active)
                print(f"{utcnow()} WAIT {counts} pids={active}", flush=True)
                if counts["complete"] == 30 and not active:
                    break
                if not active:
                    raise RuntimeError("Reference stopped without 30 completed runs")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Reference has not finished within 72 hours")
                time.sleep(args.poll_seconds)
            run_pipeline(runner.run_stage)
            runner.save("complete", finished_utc=utcnow())
        except BaseException as exc:
            runner.save("blocked", reason=str(exc), finished_utc=utcnow())
            (folder / "failure_traceback.txt").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
