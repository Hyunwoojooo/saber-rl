#!/usr/bin/env python3
"""Stage 1: new paired HER / SABER seeds 11-30 for each task x backbone cell (PROTOCOL_ko.md).

Each arm runs the command its own frozen package builds (package run.py build_command, unchanged);
only the allowed seed list is widened in memory and the output root is redirected here.

  .venv/bin/python <this dir>/launch.py plan --cells fpp_ddpg fpp_sac
  .venv/bin/python <this dir>/launch.py run  --cells fpp_ddpg fpp_sac --max-total 14
  .venv/bin/python <this dir>/launch.py status

Scheduling: one launcher per call; before starting a job it counts every training process on the
machine (any launcher, any project) and checks MemAvailable, so several launchers share one cap.
A failed run is recorded and never retried; no new jobs start after a failure in the same launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
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
FPP = REPO / "experiment_FPP"
PYTHON = REPO / ".venv" / "bin" / "python"
SEEDS = tuple(range(11, 31))
OUT = HERE / "outputs"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
MIN_AVAILABLE_GB = 8.0
TRAINER_PATTERN = "train_.*_lite_v0.py"     # every SABER-family trainer on this machine

SOURCES = HERE / "sources"            # Drawer sources extracted from origin/main 5facb92 (templates/SOURCE_COMMIT)
# cell -> arm -> (package folder or Drawer spec, expected hours per run used only for ordering longest-first)
CELLS = {
    "fpp_ddpg": {"her": (FPP / "b0_same_env_v4", 5.0), "saber": (FPP / "b3_critic_only_runtime_opt_v4_ls512", 7.0)},
    "fpp_sac": {"her": (FPP / "sac_her_baseline_ls512", 9.5), "saber": (FPP / "sac_saber_critic_only_v4_ls512", 9.5)},
    "draweropen_sac": {"her": ({"kind": "drawer_sac", "task": "drawer_open"}, 23.0),
                       "saber": ({"kind": "drawer_sac", "task": "drawer_open"}, 34.0)},
    "drawerclose_sac": {"her": ({"kind": "drawer_sac", "task": "drawer_close"}, 23.0),
                        "saber": ({"kind": "drawer_sac", "task": "drawer_close"}, 34.0)},
    "drawerclose_ddpg": {"her": ({"kind": "dc_ddpg"}, 20.0), "saber": ({"kind": "dc_ddpg"}, 30.0)},
}
DDPG_SEED_ARGS = {"--env_seed": 0, "--torch_seed": 0, "--numpy_seed": 0, "--action_noise_seed": 0,
                  "--weight_seed": 20000, "--eval_seed": 10000}


def code_dir(cell: str, arm: str) -> Path:
    spec = CELLS[cell][arm][0]
    if isinstance(spec, Path):
        return spec / "code" / "src"
    if spec["kind"] == "drawer_sac":
        return SOURCES / "drawer_sac" / "code" / "src"
    return SOURCES / "drawerclose_ddpg" / arm / "code" / "src"


def package_module(folder: Path):
    spec = importlib.util.spec_from_file_location(f"pkg_{folder.name}", folder / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = mod.load_config

    def load_config(f):
        cfg = original(f)
        cfg["seeds"] = sorted(set(cfg.get("seeds", [])) | set(SEEDS))
        return cfg

    mod.load_config = load_config
    # the packages look for the pre-July repo layout (version_v0_lite/ + SABER/); here the root is only
    # used for .venv/bin/python and the default output root, which build() overrides
    mod.repo_root_from = lambda _path: REPO
    return mod


def build(cell: str, arm: str, seed: int, device: str) -> tuple[list[str], Path]:
    folder = CELLS[cell][arm][0]
    root = OUT / "runs" / cell / arm
    if isinstance(folder, dict) and folder["kind"] == "drawer_sac":
        # executed FPP-v2 Drawer SAC source via its own protocol.make_config (see the runner's docstring)
        cmd = [str(PYTHON), str(HERE / "train_stage1_drawer_sac_lite_v0.py"), "--task", folder["task"], "--method", arm,
               "--seed", str(seed), "--output-root", str(root), "--device", device]
        return cmd, root / f"{folder['task']}_sac_{arm}_s{seed}_2m"
    if isinstance(folder, dict):
        # DrawerClose DDPG: the recorded argv of the executed seed-6 extension run, with only seed values,
        # run name and output root replaced
        argv = list(json.loads((HERE / "templates" / f"drawerclose_ddpg_{arm}_seed6_launch_record.json").read_text())["argv"])
        for flag, offset in DDPG_SEED_ARGS.items():
            argv[argv.index(flag) + 1] = str(offset + seed)
        argv[argv.index("--run_name") + 1] = f"seed_{seed:02d}"
        argv[argv.index("--output_root") + 1] = str(root)
        argv[argv.index("--device") + 1] = device
        return [str(PYTHON), str(code_dir(cell, arm) / "train_drawer_close_lite_v0.py"), *argv], root / f"seed_{seed:02d}"
    mod = package_module(folder)
    kw = dict(seed=seed, steps=None, device=device, output_root=str(root), smoke=False, validate_only=False)
    if "verify_runtime_optimization" in inspect.signature(mod.build_command).parameters:
        kw["verify_runtime_optimization"] = False
    cmd = mod.build_command(**kw)
    run_name = cmd[cmd.index("--run_name") + 1]
    return cmd, root / run_name


def tree_sha256(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(str(f.relative_to(folder)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def jobs(cells) -> list[tuple[str, str, int]]:
    todo = [(c, a, s) for c in cells for a in CELLS[c] for s in SEEDS]
    return sorted(todo, key=lambda t: (-CELLS[t[0]][t[1]][1], t[2], t[0], t[1]))     # longest first


def trainers_running() -> int:
    r = subprocess.run(["pgrep", "-fc", TRAINER_PATTERN], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def child_env(src: Path, threads: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    for v in THREAD_VARS:
        env[v] = str(threads)
    return env


def cmd_plan(a) -> None:
    for c, arm, s in jobs(a.cells):
        cmd, d = build(c, arm, s, "<device>")
        print(f"{c} {arm} s{s}: {d.relative_to(HERE)}\n   {shlex.join(cmd)[:240]} ...")
    for c in a.cells:
        for arm in CELLS[c]:
            print(c, arm, code_dir(c, arm).relative_to(REPO), "code sha256", tree_sha256(code_dir(c, arm))[:16])


def cmd_run(a) -> None:
    todo, adopted = jobs(a.cells), {}
    if a.adopt_existing:
        # take over from a stopped launcher: runs whose folder exists are never restarted; they are
        # watched until summary.json appears (done) or their process is gone without it (failed)
        for job in list(todo):
            _, d = build(*job, a.device)
            if d.exists():
                todo.remove(job)
                adopted["/".join((job[0], job[1], f"s{job[2]}"))] = d
    for c, arm, s in todo:
        _, d = build(c, arm, s, a.device)
        if d.exists():
            raise SystemExit(f"Refusing to overwrite existing run: {d}")
    tag = "_".join(a.cells) + ("_adopt" if a.adopt_existing else "")
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    meta = dict(started=datetime.now().isoformat(timespec="seconds"), cells=a.cells, device=a.device, threads=a.threads,
                max_total=a.max_total, min_available_gb=MIN_AVAILABLE_GB, seeds=list(SEEDS), adopted=sorted(adopted),
                git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
                packages={f"{c}/{arm}": dict(code=str(code_dir(c, arm).relative_to(REPO)), code_sha256=tree_sha256(code_dir(c, arm)))
                          for c in a.cells for arm in CELLS[c]},
                python=sys.version, host=os.uname().nodename,
                commands={f"{c}/{arm}/s{s}": build(c, arm, s, a.device)[0] for c, arm, s in todo})
    (OUT / f"launch_metadata_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not (OUT / "pip_freeze.txt").exists():
        (OUT / "pip_freeze.txt").write_text(subprocess.run([str(PYTHON), "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    pending, running, failed, done = list(todo), {}, [], []
    status = OUT / f"launcher_status_{tag}.json"

    def write_status():
        status.write_text(json.dumps(dict(updated=datetime.now().isoformat(timespec="seconds"),
                                          pending=[f"{c}/{arm}/s{s}" for c, arm, s in pending],
                                          running={k: p.pid for k, p in running.items()}, adopted_running=sorted(adopted),
                                          done=done, failed=failed), indent=2) + "\n")
    def poll_adopted():
        for k, d in list(adopted.items()):
            if (d / "summary.json").exists():
                del adopted[k]
                done.append(k)
            elif subprocess.run(["pgrep", "-f", "--", f"--run_name {d.name} "], capture_output=True).returncode != 0:
                time.sleep(5)
                if not (d / "summary.json").exists():
                    del adopted[k]
                    failed.append(f"{k} (adopted, exited without summary.json)")
    while pending or running or adopted:
        poll_adopted()
        while (pending and not failed and trainers_running() < a.max_total and mem_available_gb() >= MIN_AVAILABLE_GB):
            c, arm, s = pending.pop(0)
            cmd, d = build(c, arm, s, a.device)
            d.parent.mkdir(parents=True, exist_ok=True)
            fh = open(logs / f"{c}_{arm}_s{s}.log", "w")
            running[f"{c}/{arm}/s{s}"] = subprocess.Popen(cmd, env=child_env(code_dir(c, arm), a.threads),
                                                          stdout=fh, stderr=subprocess.STDOUT)
            write_status()
            time.sleep(30)                       # let the new process allocate before re-checking memory
        time.sleep(30)
        for k, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[k]
                (done if rc == 0 else failed).append(k if rc == 0 else f"{k} (exit {rc})")
                write_status()
        write_status()
    (OUT / (f"FAILED_{tag}" if failed else f"DONE_{tag}")).write_text(datetime.now().isoformat(timespec="seconds") + "\n")


def cmd_status(_a) -> None:
    for f in sorted(OUT.glob("launcher_status_*.json")):
        s = json.loads(f.read_text())
        print(f"{f.name}: updated {s['updated']}, running {len(s['running']) + len(s.get('adopted_running', []))}, pending {len(s['pending'])}, "
              f"done {len(s['done'])}, failed {s['failed']}")
    for c in CELLS:
        for arm in CELLS[c]:
            root = OUT / "runs" / c / arm
            if not root.exists():
                continue
            fin = sorted(root.glob("*/summary.json"))
            steps = []
            for d in root.iterdir():
                log = d / "run.log"
                if log.exists() and not (d / "summary.json").exists():
                    lines = [ln for ln in log.read_text(errors="replace").splitlines()[-50:] if "step=" in ln]
                    if lines:
                        steps.append(int(lines[-1].split("step=")[1].split()[0]))
            print(f"{c:9s} {arm:6s} finished {len(fin):2d}/20  running steps {sorted(steps)}")
    print(f"trainers on machine: {trainers_running()}, MemAvailable {mem_available_gb():.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--cells", nargs="+", choices=list(CELLS), required=True)
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--cells", nargs="+", choices=list(CELLS), required=True)
    r.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-total", type=int, default=14)
    r.add_argument("--adopt-existing", action="store_true",
                   help="skip runs whose folder exists and watch them instead (after stopping an earlier launcher)")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
