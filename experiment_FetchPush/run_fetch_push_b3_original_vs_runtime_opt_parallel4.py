from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\SS\Projects\metacog-rl\metacog-rl")
EXPERIMENT_ROOT = PROJECT_ROOT / "version5.0" / "metacog-lite-v0" / "experiment"
LAUNCHER_LOG = EXPERIMENT_ROOT / "fetch_push_b3_original_vs_runtime_opt_launcher.log"
SUMMARY_PATH = EXPERIMENT_ROOT / "fetch_push_b3_original_vs_runtime_opt_main_summary.json"


@dataclass
class Job:
    group_name: str
    code_root: Path
    script: Path
    output_root: Path
    log_root: Path
    run_name: str
    seed: int
    args: list[str]
    summary_path: Path
    out_log: Path
    err_log: Path
    process: subprocess.Popen | None = None


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LAUNCHER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def completed_summary(path: Path, total_steps: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        return int(data.get("total_steps", -1)) >= total_steps
    except Exception:
        return False


def build_job(
    *,
    group_name: str,
    code_root: Path,
    output_root: Path,
    log_root: Path,
    prefix: str,
    seed: int,
    total_steps: int,
    device: str,
    extra_args: list[str],
) -> Job:
    run_name = f"{prefix}_s{seed}_{total_steps}_ls512"
    script = code_root / "train_fetch_push_lite_v0.py"
    summary_path = output_root / run_name / "summary.json"
    out_log = log_root / f"{run_name}.out.log"
    err_log = log_root / f"{run_name}.err.log"
    args = [
        str(script),
        "--run_name",
        run_name,
        "--output_root",
        str(output_root),
        "--method",
        "b3_support_only",
        "--reward_type",
        "sparse",
        "--total_steps",
        str(total_steps),
        "--start_steps",
        "10000",
        "--warmup_steps",
        "5000",
        "--updates_per_step",
        "1",
        "--batch_size",
        "256",
        "--buffer_capacity",
        "300000",
        "--her_ratio",
        "0.8",
        "--M_B",
        "32",
        "--M_Q_min",
        "8",
        "--N_local_min",
        "8",
        "--local_search_max",
        "512",
        "--eval_freq",
        "25000",
        "--eval_episodes",
        "20",
        "--checkpoint_every_steps",
        "50000",
        "--log_every_steps",
        "1000",
        "--N_gap_min",
        "256",
        "--weighting_mode",
        "batch_norm",
        "--weight_scope",
        "actor_critic",
        "--device",
        device,
        "--env_seed",
        str(seed),
        "--torch_seed",
        str(seed),
        "--numpy_seed",
        str(seed),
        "--action_noise_seed",
        str(seed),
        "--weight_seed",
        str(20000 + seed),
        "--eval_seed",
        str(10000 + seed),
        "--no_save_step_checkpoints",
        *extra_args,
    ]
    return Job(
        group_name=group_name,
        code_root=code_root,
        script=script,
        output_root=output_root,
        log_root=log_root,
        run_name=run_name,
        seed=seed,
        args=args,
        summary_path=summary_path,
        out_log=out_log,
        err_log=err_log,
    )


def start_job(job: Job) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(job.code_root)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    job.log_root.mkdir(parents=True, exist_ok=True)
    job.output_root.mkdir(parents=True, exist_ok=True)

    out = job.out_log.open("w", encoding="utf-8")
    err = job.err_log.open("w", encoding="utf-8")
    job.process = subprocess.Popen(
        [sys.executable, *job.args],
        cwd=str(PROJECT_ROOT),
        stdout=out,
        stderr=err,
        env=env,
    )


def write_summary(
    *,
    total_steps: int,
    max_parallel: int,
    device: str,
    completed: list[Job],
    skipped: list[Job],
    failed: list[Job],
    jobs: list[Job],
) -> None:
    data = {
        "task": "fetch_push_b3_original_vs_runtime_optimization",
        "total_steps": total_steps,
        "local_search_max": 512,
        "max_parallel": max_parallel,
        "device": device,
        "groups": ["B3-Original", "B3-Original-runtime-optimization"],
        "seeds": [1, 2, 3, 4, 5],
        "completed_runs": [j.run_name for j in completed],
        "skipped_runs": [j.run_name for j in skipped],
        "failed_runs": [j.run_name for j in failed],
        "summary_paths": [str(j.summary_path) for j in jobs],
        "log_paths": [str(j.out_log) for j in jobs],
        "status": "failed" if failed else "completed",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    SUMMARY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total_steps", type=int, default=1000000)
    args = parser.parse_args()

    LAUNCHER_LOG.write_text("", encoding="utf-8")
    log("FetchPush B3 original vs runtime optimization Python launcher started")
    log(f"ProjectRoot={PROJECT_ROOT}")
    log(f"ExperimentRoot={EXPERIMENT_ROOT}")
    log(f"MaxParallel={args.max_parallel}")
    log(f"TotalSteps={args.total_steps}")
    log(f"Device={args.device}")
    log("local_search_max=512")

    group_specs = [
        {
            "group_name": "B3-Original",
            "code_root": EXPERIMENT_ROOT / "B3-Original" / "code" / "src",
            "output_root": EXPERIMENT_ROOT / "B3-Original" / "results",
            "log_root": EXPERIMENT_ROOT / "B3-Original" / "logs",
            "prefix": "fetch_push_b3_original",
            "extra_args": [],
        },
        {
            "group_name": "B3-Original-runtime-optimization",
            "code_root": EXPERIMENT_ROOT
            / "B3-Original-runtime-optimization"
            / "code"
            / "src",
            "output_root": EXPERIMENT_ROOT
            / "B3-Original-runtime-optimization"
            / "results",
            "log_root": EXPERIMENT_ROOT / "B3-Original-runtime-optimization" / "logs",
            "prefix": "fetch_push_b3_runtime_opt",
            "extra_args": ["--support_lookup_optimization"],
        },
    ]

    jobs: list[Job] = []
    for spec in group_specs:
        for seed in [1, 2, 3, 4, 5]:
            jobs.append(
                build_job(
                    group_name=spec["group_name"],
                    code_root=spec["code_root"],
                    output_root=spec["output_root"],
                    log_root=spec["log_root"],
                    prefix=spec["prefix"],
                    seed=seed,
                    total_steps=args.total_steps,
                    device=args.device,
                    extra_args=spec["extra_args"],
                )
            )

    pending: list[Job] = []
    skipped: list[Job] = []
    for job in jobs:
        if completed_summary(job.summary_path, args.total_steps):
            skipped.append(job)
            log(f"SKIP completed: {job.run_name}")
        else:
            pending.append(job)

    running: list[Job] = []
    completed: list[Job] = []
    failed: list[Job] = []

    def reap_finished() -> None:
        still_running: list[Job] = []
        for active in running:
            assert active.process is not None
            code = active.process.poll()
            if code is None:
                still_running.append(active)
            elif code == 0:
                completed.append(active)
                log(f"DONE: {active.run_name}")
            else:
                failed.append(active)
                log(f"FAILED: {active.run_name}, exit={code}, stderr={active.err_log}")
        running[:] = still_running

    for job in pending:
        while len(running) >= args.max_parallel:
            time.sleep(30)
            reap_finished()
            if failed:
                break
        if failed:
            break

        log(f"START: {job.run_name}")
        log(f"  CodeRoot={job.code_root}")
        log(f"  OutputRoot={job.output_root}")
        log(f"  OutLog={job.out_log}")
        log(f"  ErrLog={job.err_log}")
        start_job(job)
        running.append(job)

    while running:
        time.sleep(30)
        reap_finished()

    if failed:
        for active in running:
            if active.process is not None and active.process.poll() is None:
                active.process.terminate()

    write_summary(
        total_steps=args.total_steps,
        max_parallel=args.max_parallel,
        device=args.device,
        completed=completed,
        skipped=skipped,
        failed=failed,
        jobs=jobs,
    )
    log(f"Launcher summary written: {SUMMARY_PATH}")
    if failed:
        log("Launcher failed")
        return 1
    log("Launcher completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
