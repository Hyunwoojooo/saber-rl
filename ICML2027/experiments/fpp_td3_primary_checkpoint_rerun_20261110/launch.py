#!/usr/bin/env python3
"""Re-train the primary FetchPickAndPlace TD3 HER / SABER runs (seeds 1-5), keeping checkpoints.

The original runs (experiment_FPP/runs/td3_{her_baseline,saber_critic_only_v4}_ls512) kept only
summaries. This launcher runs the same frozen code with the same command line and writes to
this folder's outputs/. It never touches the original run folders.

  .venv/bin/python <this dir>/launch.py plan
  .venv/bin/python <this dir>/launch.py bench --device cpu --threads 1
  .venv/bin/python <this dir>/launch.py run --device cpu --threads 1 --max-parallel 10
  .venv/bin/python <this dir>/launch.py run --device cuda --threads 1 --max-parallel 10 --tag cuda
  .venv/bin/python <this dir>/launch.py status [--tag cuda]

The first run (2026-11-10, cpu) has no tag and lives in outputs/runs. A tagged run writes to
outputs/runs_<tag> with its own logs, metadata and DONE/FAILED marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FPP = REPO / "experiment_FPP"
PYTHON = REPO / ".venv" / "bin" / "python"
ARMS = {"her": FPP / "td3_her_baseline_ls512", "saber": FPP / "td3_saber_critic_only_v4_ls512"}
SEEDS = (1, 2, 3, 4, 5)
STEPS = 1_000_000
OUT = HERE / "outputs"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def cfg_of(arm: str) -> dict:
    return json.loads((ARMS[arm] / "config.json").read_text())


def run_name(arm: str, seed: int, steps: int) -> str:
    c = cfg_of(arm)
    return f"fetch_pick_and_place_{c['run_slug']}_s{int(seed)}_{int(steps)}_ls{int(c['local_search_max'])}"


def out_root(arm: str, base: Path) -> Path:
    return base / cfg_of(arm)["run_slug"]


def build_command(arm: str, seed: int, steps: int, device: str, base: Path) -> list[str]:
    """Same arguments, in the same order, as build_command in the frozen run.py of each arm."""
    cfg, folder = cfg_of(arm), ARMS[arm]
    cmd = [
        str(PYTHON), str(folder / "code" / "src" / "train_fetch_pick_and_place_lite_v0.py"),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(arm, seed, steps),
        "--output_root", str(out_root(arm, base)),
        "--method", cfg["trainer_method"],
        "--reward_type", cfg["reward_type"],
        "--total_steps", str(steps),
        "--batch_size", str(cfg["batch_size"]),
        "--her_ratio", str(cfg["her_ratio"]),
        "--updates_per_step", str(cfg["updates_per_step"]),
        "--eval_freq", str(cfg["eval_freq"]),
        "--eval_episodes", str(cfg["eval_episodes"]),
        "--M_B", str(cfg["M_B"]),
        "--M_Q_min", str(cfg["M_Q_min"]),
        "--N_local_min", str(cfg["N_local_min"]),
        "--N_gap_min", str(cfg["N_gap_min"]),
        "--local_search_max", str(cfg["local_search_max"]),
        "--lambda_support", str(cfg["lambda_support"]),
        "--per_alpha", str(cfg["per_alpha"]),
        "--eps_PER", str(cfg["eps_PER"]),
        "--U_PER_max", str(cfg["U_PER_max"]),
        "--weighting_mode", cfg["weighting_mode"],
        "--weight_scope", cfg["weight_scope"],
        "--policy_support_mode", cfg["policy_support_mode"],
        "--support_scan_mode", cfg["support_scan_mode"],
        "--support_cache_max_entries", str(cfg.get("support_cache_max_entries", 200000)),
        "--local_scan_cache_max_entries", str(cfg.get("local_scan_cache_max_entries", 50000)),
        "--td3_policy_delay", str(cfg.get("td3_policy_delay", 2)),
        "--td3_target_policy_noise", str(cfg.get("td3_target_policy_noise", 0.2)),
        "--td3_target_noise_clip", str(cfg.get("td3_target_noise_clip", 0.5)),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + int(seed)),
        "--eval_seed", str(10000 + int(seed)),
    ]
    cmd.append("--support_cache_enabled" if cfg.get("support_cache_enabled", True) else "--no-support_cache_enabled")
    cmd.append("--local_scan_cache_enabled" if cfg.get("local_scan_cache_enabled", True) else "--no-local_scan_cache_enabled")
    cmd.append("--support_short_circuit" if cfg.get("support_short_circuit", True) else "--no-support_short_circuit")
    cmd.append("--numba_support_scan" if cfg.get("numba_support_scan", True) else "--no-numba_support_scan")
    cmd.append("--support_predicate_early_exit" if cfg.get("support_predicate_early_exit", True)
               else "--no-support_predicate_early_exit")
    cmd.extend(["--support_diagnostic_audit_interval", str(cfg.get("support_diagnostic_audit_interval", 100))])
    return cmd


def child_env(arm: str, threads: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARMS[arm] / "code" / "src")
    for v in THREAD_VARS:
        env[v] = str(threads)
    return env


def tree_sha256(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(str(f.relative_to(folder)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def log_progress(run_dir: Path) -> tuple[int, str, str]:
    """(last logged step, first timestamp, last timestamp) from run.log."""
    log = run_dir / "run.log"
    if not log.exists():
        return -1, "", ""
    lines = [ln for ln in log.read_text(errors="replace").splitlines() if "step=" in ln]
    if not lines:
        return 0, "", ""
    step = int(lines[-1].split("step=")[1].split()[0])
    return step, lines[0][:19], lines[-1][:19]


def original_dir(arm: str, seed: int) -> Path:
    return FPP / "runs" / cfg_of(arm)["run_slug"] / run_name(arm, seed, STEPS)


def config_differences(arm: str, seed: int, base: Path) -> dict:
    new, old = out_root(arm, base) / run_name(arm, seed, STEPS) / "config.json", original_dir(arm, seed) / "config.json"
    if not (new.exists() and old.exists()):
        return {"missing": True}
    a, b = json.loads(new.read_text()), json.loads(old.read_text())
    return {k: (b.get(k, "<absent>"), a.get(k, "<absent>")) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def cmd_plan(_a) -> None:
    for arm in ARMS:
        for seed in SEEDS:
            print(shlex.join(build_command(arm, seed, STEPS, "<device>", OUT / "runs")))
    print("code sha256:", {arm: tree_sha256(ARMS[arm] / "code" / "src")[:16] for arm in ARMS})


def cmd_bench(a) -> None:
    """Short SABER run (updates start at 5k); reports env steps/s between 6k and the end."""
    base = OUT / "bench" / f"{a.device}_t{a.threads}"
    if base.exists():
        shutil.rmtree(base)
    steps = a.steps
    cmd = build_command("saber", 1, steps, a.device, base)
    t0 = time.time()
    subprocess.run(cmd, env=child_env("saber", a.threads), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = out_root("saber", base) / run_name("saber", 1, steps) / "run.log"
    rows = {}
    for ln in log.read_text().splitlines():
        if "step=" in ln and "updates=" in ln:
            rows[int(ln.split("step=")[1].split()[0])] = datetime.strptime(ln[:23], "%Y-%m-%d %H:%M:%S,%f")
    last = max(rows)
    rate = (last - 6000) / (rows[last] - rows[6000]).total_seconds()
    print(f"bench device={a.device} threads={a.threads}: {rate:.1f} env steps/s with updates "
          f"(6k->{last // 1000}k), wall {time.time() - t0:.0f}s; 1M steps ~ {1e6 / rate / 3600:.1f} h")
    shutil.rmtree(base)


def names(tag: str) -> dict:
    sfx = f"_{tag}" if tag else ""
    return dict(base=OUT / f"runs{sfx}", logs=OUT / f"logs{sfx}", meta=OUT / f"launch_metadata{sfx}.json",
                freeze=OUT / f"pip_freeze{sfx}.txt", status=OUT / f"launcher_status{sfx}.json",
                done=OUT / f"DONE{sfx}", failed=OUT / f"FAILED{sfx}")


def cmd_run(a) -> None:
    n = names(a.tag)
    base = n["base"]
    jobs = [(arm, seed) for seed in SEEDS for arm in ("saber", "her")]       # SABER (slower) starts first per seed
    for arm, seed in jobs:
        target = out_root(arm, base) / run_name(arm, seed, STEPS)
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing run: {target}")
    logs = n["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    meta = dict(started=datetime.now().isoformat(timespec="seconds"), device=a.device, threads=a.threads,
                max_parallel=a.max_parallel, steps=STEPS, seeds=list(SEEDS),
                git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
                code_sha256={arm: tree_sha256(ARMS[arm] / "code" / "src") for arm in ARMS},
                python=sys.version, host=os.uname().nodename,
                commands={f"{arm}_seed_{seed}": build_command(arm, seed, STEPS, a.device, base) for arm, seed in jobs})
    n["meta"].write_text(json.dumps(meta, indent=2) + "\n")
    n["freeze"].write_text(subprocess.run([str(PYTHON), "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    running, pending, failed = {}, list(jobs), []
    status_path = n["status"]

    def write_status():
        status_path.write_text(json.dumps(dict(
            updated=datetime.now().isoformat(timespec="seconds"), pending=[f"{x}_{s}" for x, s in pending],
            running={f"{x}_{s}": p.pid for (x, s), p in running.items()}, failed=failed), indent=2) + "\n")

    while pending or running:
        while pending and len(running) < a.max_parallel and not failed:
            arm, seed = pending.pop(0)
            out_root(arm, base).mkdir(parents=True, exist_ok=True)
            fh = open(logs / f"{arm}_seed_{seed}.log", "w")
            running[(arm, seed)] = subprocess.Popen(build_command(arm, seed, STEPS, a.device, base),
                                                    env=child_env(arm, a.threads), stdout=fh, stderr=subprocess.STDOUT)
            write_status()
        time.sleep(20)
        for key, proc in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                del running[key]
                if rc != 0:
                    failed.append(f"{key[0]}_{key[1]} (exit {rc})")      # no automatic retry; stop starting new runs
                write_status()
    write_status()
    (n["failed"] if failed else n["done"]).write_text(datetime.now().isoformat(timespec="seconds") + "\n")


def cmd_status(a) -> None:
    n = names(a.tag)
    base = n["base"]
    st = n["status"]
    if st.exists():
        s = json.loads(st.read_text())
        print(f"launcher: updated {s['updated']}, running {len(s['running'])}, pending {len(s['pending'])}, failed {s['failed']}")
    for arm in ARMS:
        for seed in SEEDS:
            d = out_root(arm, base) / run_name(arm, seed, STEPS)
            step, first, last = log_progress(d)
            ck = len(list(d.glob("checkpoint_step_*.pt"))) if d.exists() else 0
            line = f"{arm:5s} seed {seed}: step {step:>9,d}  checkpoints {ck:2d}/20"
            if step > 0 and first and last:
                dt = (datetime.strptime(last, "%Y-%m-%d %H:%M:%S") - datetime.strptime(first, "%Y-%m-%d %H:%M:%S")).total_seconds()
                if dt > 0 and step < STEPS:
                    line += f"  elapsed {dt / 3600:.1f} h, about {(STEPS - step) / (step / dt) / 3600:.1f} h left"
            diff = config_differences(arm, seed, base)
            if "missing" not in diff:
                line += f"  | config keys differing from the original run: {sorted(diff)}"
            summ = d / "summary.json"
            if summ.exists():
                new = json.loads(summ.read_text()).get("eval_AUC_success")
                old = json.loads((original_dir(arm, seed) / "summary.json").read_text()).get("eval_AUC_success")
                line += f"  | AUC {new:.4f} (original {old:.4f})"
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    b = sub.add_parser("bench")
    b.add_argument("--device", required=True, choices=["cpu", "cuda"])
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--steps", type=int, default=9000)
    b.set_defaults(fn=cmd_bench)
    r = sub.add_parser("run")
    r.add_argument("--device", required=True, choices=["cpu", "cuda"])
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-parallel", type=int, default=10)
    r.add_argument("--tag", default="")
    r.set_defaults(fn=cmd_run)
    st = sub.add_parser("status")
    st.add_argument("--tag", default="")
    st.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
