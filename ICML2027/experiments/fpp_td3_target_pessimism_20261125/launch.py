#!/usr/bin/env python3
"""FPP TD3 target-pessimism test: HER and SABER with a less pessimistic TD3 target, seeds 11-30 (PROTOCOL_ko.md).

target = min(Q1', Q2') + alpha * |Q1' - Q2'|   (alpha 0 = TD3, 0.5 = mean of the two target critics)

The code in sources/<arm>/ is the frozen experiment_FPP TD3 package source plus one option
(--td3_target_mix_alpha, sources/PATCH_<arm>.diff). Each command is the n20 command
(fpp_td3_primary_checkpoint_rerun_20261110/launch.py build_command) with the trainer path, output root
and that option changed. The alpha = 0 arms are the existing n20 runs and are not re-run.

  .venv/bin/python <this dir>/launch.py plan
  .venv/bin/python <this dir>/launch.py verify            # alpha 0 of the patched code == frozen code, bit for bit
  .venv/bin/python <this dir>/launch.py run --max-total 18
  .venv/bin/python <this dir>/launch.py status
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
N20 = HERE.parent / "fpp_td3_primary_checkpoint_rerun_20261110"
PYTHON = REPO / ".venv" / "bin" / "python"
SEEDS = tuple(range(11, 31))
STEPS = 1_000_000
ALPHAS = {"a025": 0.25, "a050": 0.5}
ARMS = ("saber", "her")
OUT = HERE / "outputs"
MIN_AVAILABLE_GB = 8.0
TRAINER_PATTERN = "train_.*_lite_v0.py"     # every SABER-family trainer on this machine (shared cap)

_spec = importlib.util.spec_from_file_location("n20_launch", N20 / "launch.py")
L = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(L)                         # type: ignore[union-attr]


def src_dir(arm: str) -> Path:
    return HERE / "sources" / arm / "code" / "src"


def build(arm: str, tag: str, seed: int, device: str, base: Path = OUT / "runs", steps: int = STEPS,
          alpha: float | None = None) -> tuple[list[str], Path]:
    root = base / f"{arm}_{tag}"
    cmd = L.build_command(arm, seed, steps, device, base)
    cmd[1] = str(src_dir(arm) / "train_fetch_pick_and_place_lite_v0.py")
    cmd[cmd.index("--output_root") + 1] = str(root)
    cmd += ["--td3_target_mix_alpha", str(ALPHAS[tag] if alpha is None else alpha)]
    return cmd, root / L.run_name(arm, seed, steps)


def child_env(arm: str, threads: int, frozen: bool = False) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(L.ARMS[arm] / "code" / "src") if frozen else str(src_dir(arm))
    for v in L.THREAD_VARS:
        env[v] = str(threads)
    return env


def jobs() -> list[tuple[str, str, int]]:
    # SABER (slower) first; within an arm the two doses alternate so both progress together
    return [(arm, tag, s) for arm in ARMS for s in SEEDS for tag in ALPHAS]


def trainers_running() -> int:
    r = subprocess.run(["pgrep", "-fc", TRAINER_PATTERN], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def cmd_plan(_a) -> None:
    for arm, tag, s in jobs():
        cmd, d = build(arm, tag, s, "<device>")
        print(f"{arm} {tag} s{s}: {d.relative_to(HERE)}\n   {shlex.join(cmd[1:])[:200]} ... {shlex.join(cmd[-4:])}")
    for arm in ARMS:
        print(arm, "patched code sha256", L.tree_sha256(src_dir(arm))[:16], "| frozen", L.tree_sha256(L.ARMS[arm] / "code" / "src")[:16])


def cmd_verify(a) -> None:
    """Patched code with alpha 0 must reproduce the frozen code exactly (cpu, short run with updates),
    and alpha > 0 must change the result and be recorded in config.json."""
    import torch
    base = OUT / "verify"
    if base.exists():
        shutil.rmtree(base)
    report = {}
    for arm in ARMS:
        runs = {}
        frozen = L.build_command(arm, 11, a.steps, "cpu", base / "frozen")
        patched0, d0 = build(arm, "a050", 11, "cpu", base / "patched0", a.steps, alpha=0.0)
        patched5, d5 = build(arm, "a050", 11, "cpu", base / "patched5", a.steps)
        dirs = dict(frozen=L.out_root(arm, base / "frozen") / L.run_name(arm, 11, a.steps), patched0=d0, patched5=d5)
        procs = [("frozen", subprocess.Popen(frozen, env=child_env(arm, 1, frozen=True), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)),
                 ("patched0", subprocess.Popen(patched0, env=child_env(arm, 1), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)),
                 ("patched5", subprocess.Popen(patched5, env=child_env(arm, 1), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))]
        for name, p in procs:
            if p.wait() != 0:
                raise SystemExit(f"verify run failed: {arm} {name}")
            d = dirs[name]
            runs[name] = (torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False), json.loads((d / "config.json").read_text()))

        def tensors(ck):
            ag = ck["agent"] if "agent" in ck else ck
            return {f"{k}.{n}": t for k in ("actor", "critic1", "critic2", "critic1_target", "critic2_target") for n, t in ag[k].items()}

        t_f, t_0, t_5 = (tensors(runs[k][0]) for k in ("frozen", "patched0", "patched5"))
        same0 = all(torch.equal(t_f[k], t_0[k]) for k in t_f)
        differs5 = any(not torch.equal(t_f[k], t_5[k]) for k in t_f)
        cfg_diff = {k: (runs["frozen"][1].get(k, "<absent>"), v) for k, v in runs["patched0"][1].items()
                    if runs["frozen"][1].get(k, "<absent>") != v and k not in ("output_root",)}
        assert runs["frozen"][1]["method"] == runs["patched0"][1]["method"] == L.cfg_of(arm)["trainer_method"]
        report[arm] = dict(alpha0_bit_identical=same0, alpha05_differs=differs5, n_tensors=len(t_f),
                           config_keys_differing_alpha0_vs_frozen=cfg_diff,
                           alpha_recorded=runs["patched5"][1].get("td3_target_mix_alpha"))
        print(arm, json.dumps(report[arm]))
    ok = all(r["alpha0_bit_identical"] and r["alpha05_differs"] and r["alpha_recorded"] == 0.5
             and set(r["config_keys_differing_alpha0_vs_frozen"]) == {"td3_target_mix_alpha"} for r in report.values())
    (OUT / "verify_report.json").write_text(json.dumps(dict(steps=a.steps, ok=ok, arms=report,
                                                           checked=datetime.now().isoformat(timespec="seconds")), indent=2) + "\n")
    shutil.rmtree(base)
    print("VERIFY", "OK" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def cmd_run(a) -> None:
    rep = OUT / "verify_report.json"
    if not (rep.exists() and json.loads(rep.read_text())["ok"]):
        raise SystemExit("run `verify` first")
    todo = jobs()
    for job in todo:
        _, d = build(*job, a.device)
        if d.exists():
            raise SystemExit(f"Refusing to overwrite existing run: {d}")
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    meta = dict(started=datetime.now().isoformat(timespec="seconds"), device=a.device, threads=a.threads, max_total=a.max_total,
                min_available_gb=MIN_AVAILABLE_GB, seeds=list(SEEDS), alphas=ALPHAS, steps=STEPS,
                git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
                code_sha256={arm: dict(patched=L.tree_sha256(src_dir(arm)), frozen=L.tree_sha256(L.ARMS[arm] / "code" / "src")) for arm in ARMS},
                patch_sha256={arm: hashlib.sha256((HERE / "sources" / f"PATCH_{arm}.diff").read_bytes()).hexdigest() for arm in ARMS},
                python=sys.version, host=os.uname().nodename,
                commands={f"{arm}_{tag}/s{s}": build(arm, tag, s, a.device)[0] for arm, tag, s in todo})
    (OUT / "launch_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT / "pip_freeze.txt").write_text(subprocess.run([str(PYTHON), "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    pending, running, failed, done = list(todo), {}, [], []
    status = OUT / "launcher_status.json"

    def write_status():
        status.write_text(json.dumps(dict(updated=datetime.now().isoformat(timespec="seconds"),
                                          pending=[f"{arm}_{tag}/s{s}" for arm, tag, s in pending],
                                          running={k: p.pid for k, p in running.items()}, done=done, failed=failed), indent=2) + "\n")
    while pending or running:
        while pending and not failed and trainers_running() < a.max_total and mem_available_gb() >= MIN_AVAILABLE_GB:
            arm, tag, s = pending.pop(0)
            cmd, d = build(arm, tag, s, a.device)
            d.parent.mkdir(parents=True, exist_ok=True)
            fh = open(logs / f"{arm}_{tag}_s{s}.log", "w")
            running[f"{arm}_{tag}/s{s}"] = subprocess.Popen(cmd, env=child_env(arm, a.threads), stdout=fh, stderr=subprocess.STDOUT)
            write_status()
            time.sleep(20)                       # let the new process allocate before re-checking memory
        time.sleep(30)
        for k, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[k]
                (done if rc == 0 else failed).append(k if rc == 0 else f"{k} (exit {rc})")   # no retry; stop starting new runs
        write_status()
    (OUT / ("FAILED" if failed else "DONE")).write_text(datetime.now().isoformat(timespec="seconds") + "\n")


def cmd_status(_a) -> None:
    st = OUT / "launcher_status.json"
    if st.exists():
        s = json.loads(st.read_text())
        print(f"launcher: updated {s['updated']}, running {len(s['running'])}, pending {len(s['pending'])}, done {len(s['done'])}, failed {s['failed']}")
    for arm in ARMS:
        for tag in ALPHAS:
            root = OUT / "runs" / f"{arm}_{tag}"
            fin = len(list(root.glob("*/summary.json"))) if root.exists() else 0
            steps = []
            for d in (root.iterdir() if root.exists() else []):
                if not (d / "summary.json").exists():
                    steps.append(L.log_progress(d)[0])
            print(f"{arm:5s} {tag}: finished {fin:2d}/20  running steps {sorted(steps)}")
    print(f"trainers on machine: {trainers_running()}, MemAvailable {mem_available_gb():.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    v = sub.add_parser("verify")
    v.add_argument("--steps", type=int, default=8000)
    v.set_defaults(fn=cmd_verify)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-total", type=int, default=18)
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
