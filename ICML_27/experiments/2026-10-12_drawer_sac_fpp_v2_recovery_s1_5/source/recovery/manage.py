"""Audited fresh attempts after power loss; never mutate or resume parent runs."""
import argparse
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "recovery/plan.json").read_text())
PARENT = ROOT.parent / PLAN["parent"]
STATE = ROOT / "recovery_state"
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def name(job):
    task, method, seed = job
    return f"{task}_sac_{method}_s{seed}_2m"


def all_jobs():
    return [(t, m, s) for t in ("drawer_open", "drawer_close")
            for s in range(1, 6) for m in ("her", "saber")]


def check_plan(plan=PLAN):
    groups = [list(map(tuple, plan[k])) for k in ("reuse", "rerun_from_zero", "fresh_unstarted")]
    flat = sum(groups, [])
    if len(flat) != 20 or len(set(flat)) != 20 or set(flat) != set(all_jobs()):
        raise ValueError("Recovery must partition all twenty original jobs exactly once")
    if set(groups[0]) != {("drawer_open", "her", s) for s in (1, 2, 3)}:
        raise ValueError("Only the three completed HER runs may be reused")
    expected_retry = {("drawer_open", "saber", s) for s in range(1, 5)} | {("drawer_open", "her", 4)}
    if set(groups[1]) != expected_retry:
        raise ValueError("The five interrupted jobs must all be retried")
    if plan["checkpoint_continuation"] or plan["automatic_seed_extension"] or plan["performance_based_replacement"]:
        raise ValueError("No continuation, extension, or performance selection is authorized")
    if (plan["new_runs"], plan["new_environment_steps"], plan["steps_per_run"], plan["parallel"]) != (17, 34000000, 2000000, 5):
        raise ValueError("Unauthorized recovery budget")
    if plan["task_order"] != ["drawer_open", "drawer_close"]:
        raise ValueError("DrawerOpen must be completed first")


def pending_jobs(task):
    reuse = set(map(tuple, PLAN["reuse"]))
    return [job for job in all_jobs() if job[0] == task and job not in reuse]


def recovery_files():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "recovery").rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json", ".md"}}


def verify_source():
    from manifest import verify
    check_plan()
    digest = verify()
    if digest != PLAN["source_manifest_sha256"] or sha(PARENT / "source_manifest.json") != digest:
        raise RuntimeError("Scientific source snapshot changed")
    original = read(PARENT / "source_manifest.json")["sha256"]
    for path, expected in original.items():
        if sha(PARENT / path) != expected:
            raise RuntimeError(f"Parent source changed: {path}")
    if read(ROOT / "recovery_manifest.json")["sha256"] != recovery_files():
        raise RuntimeError("Recovery source snapshot changed")
    return digest


def current_environment():
    import torch
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
                            check=True, capture_output=True, text=True, timeout=20)
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": "cuda", "gpu": result.stdout.strip(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                         if d.metadata["Name"]}}


def require_environment(expected, actual):
    differences = [k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)]
    if differences:
        raise RuntimeError(f"Recorded execution environment differs: {differences}")


def scan_partial_csv(path):
    raw = Path(path).read_bytes()
    first_null = raw.find(b"\x00")
    prefix = raw if first_null < 0 else raw[:first_null]
    if first_null >= 0 and raw[first_null:] != b"\x00" * (len(raw) - first_null):
        raise ValueError("Unexpected corruption inside the retained CSV")
    end = prefix.rfind(b"\n") + 1
    rows = list(csv.DictReader(io.StringIO(prefix[:end].decode())))
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed complete CSV record")
    return {"last_complete_step": max(int(row["step"]) for row in rows),
            "complete_records": len(rows), "null_padding_bytes": len(raw) - len(prefix),
            "partial_record_bytes": len(prefix) - end}


def assert_no_parent_workers():
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = p.read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        if any(str(PARENT / script).encode() in argv for script in ("train.py", "launch.py", "scheduled_run.py")):
            raise RuntimeError("The original experiment still has active workers")


def copy_frozen_source():
    check_plan()
    expected = PLAN["source_manifest_sha256"]
    if sha(PARENT / "source_manifest.json") != expected:
        raise RuntimeError("Unexpected parent source manifest")
    files = dict(read(PARENT / "source_manifest.json")["sha256"])
    files.update({r["path"]: r["sha256"] for r in read(PARENT / "upstream_manifest.json")["files"]})
    files["source_manifest.json"] = expected
    for rel, digest in files.items():
        src, dst = PARENT / rel, ROOT / rel
        if not src.resolve().is_relative_to(PARENT.resolve()) or sha(src) != digest:
            raise RuntimeError(f"Invalid frozen input: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha(dst) != digest:
                raise RuntimeError(f"Refusing to replace a different file: {dst}")
        else:
            shutil.copy2(src, dst)
    return files


def prepare():
    if STATE.exists() or (ROOT / "runs").exists() or (ROOT / "recovery_manifest.json").exists():
        raise FileExistsError("Recovery is already prepared; do not overwrite its provenance")
    assert_no_parent_workers()
    files = copy_frozen_source()
    from manifest import verify
    from gates import require_production_gates, write_atomic
    from protocol import make_config
    from analyze import validate_run
    digest = verify()
    require_production_gates(digest, PARENT)
    environment = current_environment()
    catalog = {"created_utc": now(), "boot_id": boot_id(), "source_manifest_sha256": digest,
               "environment": environment, "original_attempts": [], "preserved_files": {}}
    for job in all_jobs():
        run_name = name(job)
        claim = PARENT / ".production_claims" / run_name
        folder = PARENT / "runs" / run_name
        row = {"job": list(job), "run_name": run_name}
        if not claim.exists():
            if folder.exists() or list(job) not in PLAN["fresh_unstarted"]:
                raise RuntimeError(f"Unexpected unstarted-run state: {run_name}")
            row["disposition"] = "not_started"
        else:
            status = read(claim / "status.json")
            if status["source_manifest_sha256"] != digest:
                raise RuntimeError("Original run source does not match the frozen protocol")
            require_environment(read(claim / "environment.json"), environment)
            row.update(recorded_status=status["status"], **scan_partial_csv(folder / "metrics.csv"))
            if list(job) in PLAN["reuse"]:
                if status["status"] != "completed":
                    raise RuntimeError("An incomplete run cannot be reused")
                cfg = make_config(*job, output_root=PARENT / "runs")
                row.update(disposition="reused_completed", metrics=validate_run(folder, cfg))
            elif list(job) in PLAN["rerun_from_zero"]:
                if status["status"] not in {"running", "starting"} or (folder / "summary.json").exists():
                    raise RuntimeError("Interrupted-run state differs from the approved inventory")
                row.update(disposition="interrupted_power_loss", retry_from_step=0,
                           checkpoint_continuation=False)
            else:
                raise RuntimeError("A planned unstarted run already has an execution")
            paths = [p for p in folder.iterdir() if p.is_file()]
            paths += [p for p in claim.iterdir() if p.is_file()]
            for p in paths:
                catalog["preserved_files"][str(p.relative_to(PARENT))] = sha(p)
        catalog["original_attempts"].append(row)
    for rel in ("automation/status.json", "automation/production_drawer_open.log",
                "validation/validation_report.json", "validation/gpu_validation_report.json"):
        catalog["preserved_files"][rel] = sha(PARENT / rel)
    recovery_manifest = {"recovery_id": PLAN["recovery_id"], "scientific_source_manifest_sha256": digest,
                         "sha256": recovery_files()}
    write_atomic(ROOT / "recovery_manifest.json", recovery_manifest)
    STATE.mkdir(exist_ok=False)
    write_atomic(STATE / "catalog.json", catalog)
    write_atomic(STATE / "catalog_integrity.json", {"sha256": sha(STATE / "catalog.json")})
    verify_all()
    print(json.dumps({"source_files_copied": len(files), "preserved_output_files": len(catalog["preserved_files"]),
                      "reuse": 3, "fresh_reruns": 5, "unstarted": 12, "environment_matches": True}, indent=2))


def verify_all(check_environment=True):
    digest = verify_source()
    if sha(STATE / "catalog.json") != read(STATE / "catalog_integrity.json")["sha256"]:
        raise RuntimeError("Recovery catalog changed")
    catalog = read(STATE / "catalog.json")
    if catalog["source_manifest_sha256"] != digest:
        raise RuntimeError("Catalog scientific snapshot mismatch")
    for rel, expected in catalog["preserved_files"].items():
        if sha(PARENT / rel) != expected:
            raise RuntimeError(f"Retained original file changed: {rel}")
    if check_environment:
        require_environment(catalog["environment"], current_environment())
    return digest, catalog


def child_environment():
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl",
                CUDA_VISIBLE_DEVICES="0", NUMBA_CACHE_DIR=str(ROOT / ".cache/numba"))


def gpu_gate():
    digest, catalog = verify_all()
    from gates import require_cpu_gate, require_production_gates, validate_gpu_report, write_atomic
    from protocol import SPEC, make_gpu_validation_config
    from scheduled_run import resource_state
    require_cpu_gate(digest)
    if resource_state()["state"] != "ready":
        raise RuntimeError("GPU must be idle for recovery verification")
    old = require_production_gates(digest, PARENT)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from gpu_validate import cuda_update_parity, inspect_run
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dest = ROOT / "validation" / f"recovery_gpu_{stamp}"
    dest.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_manifest_sha256": digest, "started_utc": now(),
              "boot_id": boot_id(), "recovery_manifest_sha256": sha(ROOT / "recovery_manifest.json"),
              "runs": [], "performance_threshold": None,
              "inherited_stability_source": str(PARENT / "validation/gpu_validation_report.json"),
              "inherited_stability_report_sha256": sha(PARENT / "validation/gpu_validation_report.json")}

    def save():
        write_atomic(dest / "report.json", report)
        write_atomic(ROOT / "validation/gpu_validation_report.json", report)

    save()
    try:
        report["cuda_update_parity"] = cuda_update_parity()
        for task in SPEC["tasks"]:
            for method in SPEC["methods"]:
                cfg = make_gpu_validation_config(task, method, 1, "gpu_smoke", dest / "runs")
                print(f"START fresh GPU smoke {task} {method}", flush=True)
                with (dest / f"{cfg.run_name}.log").open("x") as log:
                    subprocess.run([sys.executable, "-B", str(ROOT / "train.py"), "--task", task,
                        "--method", method, "--seed", "1", "--gpu-validation", "gpu_smoke", "--execute",
                        "--output-root", cfg.output_root], cwd=ROOT, env=child_environment(),
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
                folder = Path(cfg.output_root) / cfg.run_name
                result = inspect_run(folder, cfg)
                report["runs"].append({"profile": "gpu_smoke", "task": task, "method": method,
                    "seed": 1, "run_dir": str(folder), "evidence_origin": "fresh_after_reboot", **result})
                save()
                print(f"PASS fresh GPU smoke {task} {method}", flush=True)
        for row in old["runs"]:
            if row["profile"] != "stability":
                continue
            src = Path(row["run_dir"])
            folder = dest / "inherited_stability" / "runs" / src.name
            shutil.copytree(src, folder)
            shutil.copytree(src.parent / ".claims" / src.name, folder.parent / ".claims" / src.name)
            for filename, expected in row["sha256"].items():
                if sha(folder / filename) != expected:
                    raise RuntimeError("Inherited stability evidence changed while copying")
            cfg = make_gpu_validation_config(row["task"], row["method"], 1, "stability", src.parent)
            result = inspect_run(folder, cfg)
            report["runs"].append({**row, **result, "run_dir": str(folder),
                "evidence_origin": "inherited_same_source_and_environment", "original_run_dir": str(src)})
            print(f"VERIFIED inherited stability {row['task']} {row['method']}", flush=True)
        verify_all()
        report["status"] = "passed"
        validate_gpu_report(report, digest)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["finished_utc"] = now()
        save()
    require_production_gates(digest)
    print("PASS fresh CUDA parity/smokes and explicitly inherited stability evidence", flush=True)


def require_recovery_gate():
    digest, catalog = verify_all()
    from gates import require_production_gates
    report = require_production_gates(digest)
    if report.get("boot_id") != boot_id() or report.get("recovery_manifest_sha256") != sha(ROOT / "recovery_manifest.json"):
        raise RuntimeError("Recovery GPU validation must belong to this boot and recovery snapshot")
    return digest, catalog


def launch_task(task):
    require_recovery_gate()
    if task == "drawer_close":
        previous = read(ROOT / "analysis/drawer_open.json")
        if previous.get("status") != "completed" or len(previous.get("runs", [])) != 10:
            raise RuntimeError("DrawerOpen must validate before DrawerClose starts")
    from protocol import make_config
    from launch import execute_jobs
    with (ROOT / ".production_launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [make_config(*job) for job in pending_jobs(task)]
        print(json.dumps({"task": task, "new_runs": len(jobs), "names": [c.run_name for c in jobs],
                          "checkpoint_continuation": False, "parallel": PLAN["parallel"]}), flush=True)
        execute_jobs(jobs, PLAN["parallel"])


def validate_result(job, digest, environment):
    from protocol import make_config
    from analyze import validate_run
    reused = list(job) in PLAN["reuse"]
    origin = PARENT if reused else ROOT
    cfg = make_config(*job, output_root=origin / "runs")
    claim = origin / ".production_claims" / cfg.run_name
    status = read(claim / "status.json")
    if status.get("status") != "completed" or status.get("source_manifest_sha256") != digest:
        raise ValueError("Run is incomplete or has a different scientific source")
    require_environment(read(claim / "environment.json"), environment)
    metrics = validate_run(origin / "runs" / cfg.run_name, cfg)
    return {"task": job[0], "method": job[1], "seed": job[2], "reused_completed": reused,
            "run_dir": str(origin / "runs" / cfg.run_name), "checkpoint_continuation": False, **metrics}


def analyze_tasks(tasks):
    digest, catalog = verify_all()
    from analyze import paired_intervals
    from gates import write_atomic
    runs, incomplete = [], []
    for job in all_jobs():
        if job[0] not in tasks:
            continue
        try:
            runs.append(validate_result(job, digest, catalog["environment"]))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            incomplete.append({"job": job, "reason": str(exc)})
    payload = {"status": "incomplete" if incomplete else "completed", "tasks": tasks, "seeds": list(range(1, 6)),
               "cohort_stage": "initial_five_seed_assessment", "automatic_seed_extension": False,
               "inference_scope": "exploratory_initial_five_seed_cohort", "recovery_id": PLAN["recovery_id"],
               "original_attempts": catalog["original_attempts"], "runs": runs, "incomplete": incomplete,
               "source_manifest_sha256": digest, "recovery_manifest_sha256": sha(ROOT / "recovery_manifest.json")}
    if not incomplete:
        payload["comparisons"], payload["aggregates"] = {}, []
        lookup = {(r["task"], r["method"], r["seed"]): r for r in runs}
        for task in tasks:
            differences = [lookup[task, "saber", s]["auc"] - lookup[task, "her", s]["auc"] for s in range(1, 6)]
            payload["comparisons"][task] = paired_intervals(differences)
            for method in ("her", "saber"):
                rows = [lookup[task, method, s] for s in range(1, 6)]
                payload["aggregates"].append({"task": task, "method": method,
                    **{f"{metric}_{stat}": fn([r[metric] for r in rows])
                       for metric in ("auc", "fq_auc", "final", "runtime_seconds", "steps_per_second")
                       for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev))},
                    "hit90_count": sum(r["hit90"] is not None for r in rows)})
    output = ROOT / "analysis" / (f"{tasks[0]}.json" if len(tasks) == 1 else "final.json")
    write_atomic(output, payload)
    print(f"{len(runs)}/{len(tasks) * 10} validated; {output}", flush=True)
    if incomplete:
        raise RuntimeError("Incomplete cohort; inference withheld")


def run_queue(execute):
    if not execute:
        print(json.dumps({"plan": PLAN, "jobs": {t: pending_jobs(t) for t in PLAN["task_order"]}}, indent=2))
        return
    digest, catalog = require_recovery_gate()
    assert_no_parent_workers()
    from scheduled_run import resource_state
    from gates import write_atomic
    if resource_state()["state"] != "ready":
        raise RuntimeError("GPU is occupied; refusing to compete with another experiment")
    stages = []
    for task in PLAN["task_order"]:
        stages.extend([(f"production_{task}", ["launch", "--task", task, "--execute"]),
                       (f"analysis_{task}", ["analyze", "--task", task])])
    stages.append(("final_analysis", ["analyze", "--task", "all"]))
    with (STATE / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = STATE / "status.json"
        with path.open("x") as handle:
            json.dump({"status": "initializing"}, handle)
        state = {"status": "running", "pid": os.getpid(), "started_utc": now(), "boot_id": boot_id(),
                 "source_manifest_sha256": digest, "recovery_manifest_sha256": sha(ROOT / "recovery_manifest.json"),
                 "reused_completed_runs": 3, "fresh_runs_scheduled": 17, "stages": []}
        child = None

        def save():
            state["heartbeat_utc"] = now()
            write_atomic(path, state)

        def stop(signum, _frame):
            raise InterruptedError(f"Recovery stopped by signal {signum}; no automatic resume")

        signal.signal(signal.SIGTERM, stop)
        try:
            for stage, args in stages:
                verify_source()
                state.update(stage=stage, stage_started_utc=now())
                save()
                env = child_environment()
                if "analysis" in stage:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                with (STATE / f"{stage}.log").open("x") as log:
                    child = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), *args],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state["child_pid"] = child.pid
                    save()
                    while child.poll() is None:
                        time.sleep(10)
                        save()
                    code = child.returncode
                    child = None
                state.pop("child_pid", None)
                state["stages"].append({"name": stage, "returncode": code, "finished_utc": now()})
                save()
                if code:
                    raise RuntimeError(f"{stage} failed; later stages were not started")
            state.update(status="complete", stage="finished")
        except BaseException as exc:
            state.update(status="blocked", error=repr(exc))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            state["finished_utc"] = now()
            save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify", "gpu-gate", "run", "launch", "analyze"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task", choices=["drawer_open", "drawer_close", "all"], default="all")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        digest, catalog = verify_all()
        print(json.dumps({"status": "passed", "source_manifest_sha256": digest,
                          "preserved_files": len(catalog["preserved_files"])}))
    elif args.command == "gpu-gate":
        gpu_gate()
    elif args.command == "run":
        run_queue(args.execute)
    elif args.command == "launch":
        if args.task == "all":
            parser.error("Launch one task at a time")
        if args.execute:
            launch_task(args.task)
        else:
            print(json.dumps({"task": args.task, "jobs": pending_jobs(args.task)}, indent=2))
    elif args.command == "analyze":
        analyze_tasks(PLAN["task_order"] if args.task == "all" else [args.task])


if __name__ == "__main__":
    main()
