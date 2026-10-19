"""Post-predecessor GPU gate: four short smokes, then four 100k-step soaks."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from protocol import ROOT, SPEC, SCHEDULE, make_gpu_validation_config
from manifest import verify
from gates import require_cpu_gate, write_atomic
from analyze import validate_run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cuda_update_parity():
    import numpy as np
    import torch
    from metacog_lite_v0.agent import SACAgent
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
    folder = ROOT / "upstream/fpp/code/src/metacog_lite_v0"
    spec = importlib.util.spec_from_file_location("fpp_gpu_reference", folder / "__init__.py",
                                                submodule_search_locations=[str(folder)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    original = importlib.import_module("fpp_gpu_reference.agent").SACAgent
    kwargs = dict(state_dim=42, action_dim=4, hidden_dim=256, actor_lr=3e-4, critic_lr=3e-4,
                  gamma=.98, tau=.005, action_low=-np.ones(4, np.float32), action_high=np.ones(4, np.float32),
                  device=torch.device("cuda"), l2_action_penalty=0., alpha=.2, auto_alpha=True,
                  alpha_lr=3e-4, target_entropy=-4.)
    current = SACAgent(**kwargs)
    reference = original(**kwargs)
    reference.load_state_dict(copy.deepcopy(current.state_dict()))
    torch.manual_seed(20261008)
    data = dict(state=torch.randn(256, 42, device="cuda"), action_exec=torch.rand(256, 4, device="cuda") * 2 - 1,
                reward=-torch.ones(256, device="cuda"), next_state=torch.randn(256, 42, device="cuda"),
                nonterminal=torch.ones(256, device="cuda"), omega_actor=torch.ones(256, device="cuda"))

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise AssertionError("State keys differ")
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise AssertionError("State lengths differ")
            for x, y in zip(a, b):
                compare(x, y)
        elif a != b:
            if not isinstance(a, float) or not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7):
                raise AssertionError(f"Reference/current state differs: {a} != {b}")

    for i in range(3):
        raw = torch.ones(256, device="cuda")
        if i == 1:
            raw[:128] = 2
        data["omega_critic"] = raw / raw.mean()
        torch.manual_seed(700 + i)
        a = current.update(**data)
        torch.manual_seed(700 + i)
        b = reference.update(**data)
        compare(vars(a), vars(b))
        compare(current.state_dict(), reference.state_dict())
    torch.cuda.synchronize()
    result = {"status": "passed", "updates": 3, "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cuda": torch.version.cuda, "rtol": 1e-6, "atol": 1e-7}
    del reference, current, data
    torch.cuda.empty_cache()
    return result


def inspect_run(folder, cfg, expected_device="cuda"):
    import numpy as np
    import torch
    from metacog_lite_v0.trainer import make_agent, make_task_spec
    from types import SimpleNamespace
    metrics = validate_run(folder, cfg)
    claim = folder.parent / ".claims" / cfg.run_name
    status = json.loads((claim / "status.json").read_text())
    environment = json.loads((claim / "environment.json").read_text())
    if status.get("status") != "completed" or environment.get("device") != expected_device:
        raise RuntimeError("Validation run did not complete on the expected device")
    if expected_device == "cuda" and not environment.get("gpu"):
        raise RuntimeError("CUDA validation did not record a GPU")
    active_rows = []
    with (folder / "metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["step"]) < cfg.warmup_steps:
                continue
            for key in ("loss/critic", "loss/actor", "loss/td_error_abs_mean", "sac/alpha",
                        "sac/alpha_loss", "sac/log_prob_mean", "value/q_mean", "value/target_mean",
                        "runtime/total_sec", "runtime/steps_per_sec"):
                if not math.isfinite(float(row[key])):
                    raise RuntimeError(f"Non-finite logged {key} at step {row['step']}")
            if float(row["sac/alpha"]) <= 0:
                raise RuntimeError("Non-positive alpha")
            if cfg.method == "sac_saber" and (
                    float(row["runtime/numba_support_scan_enabled"]) != 1 or
                    float(row["runtime/numba_support_scan_fallback_rate"]) != 0):
                raise RuntimeError("SABER scanner unexpectedly fell back")
            active_rows.append(row)
    if not active_rows:
        raise RuntimeError("No updates were validated")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["global_step"] != cfg.total_steps or checkpoint["total_updates"] != metrics["updates"]:
        raise RuntimeError("Checkpoint budget mismatch")
    if checkpoint["agent"]["total_it"] != metrics["updates"]:
        raise RuntimeError("SAC iteration counter mismatch")
    recorded = json.loads((folder / "config.json").read_text())
    for key, value in json.loads(json.dumps(checkpoint["config"])).items():
        if recorded.get(key) != value:
            raise RuntimeError(f"Checkpoint config mismatch: {key}")

    def finite_tensors(value):
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise RuntimeError("Non-finite saved model/optimizer tensor")
        if isinstance(value, dict):
            for v in value.values():
                finite_tensors(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                finite_tensors(v)

    finite_tensors(checkpoint["agent"])
    rp = checkpoint["replay"]
    if rp["size"] != min(cfg.total_steps, cfg.buffer_capacity):
        raise RuntimeError("Replay size mismatch")
    for key in ("observation", "next_observation", "achieved_goal", "next_achieved_goal", "desired_goal", "action_exec"):
        if not np.isfinite(rp[key][:rp["size"]]).all():
            raise RuntimeError(f"Non-finite saved replay {key}")
    env = json.loads((folder / "env_snapshot.json").read_text())["train_env"]
    low, high = np.asarray(env["action_low"], np.float32), np.asarray(env["action_high"], np.float32)
    task = make_task_spec(cfg, SimpleNamespace(action_low=low, action_high=high, distance_threshold=env["distance_threshold"]))
    np.testing.assert_array_equal(rp["action_bin"][:rp["size"]],
                                  task.action_bin_batch(rp["action_exec"][:rp["size"]], cfg.action_bins_per_dim))
    agent = make_agent(cfg, env["state_dim"], env["action_dim"], low, high, torch.device("cpu"))
    agent.load_state_dict(checkpoint["agent"])
    states = np.concatenate([rp["observation"][:32], rp["desired_goal"][:32]], axis=1)
    actions = agent.policy_actions(torch.from_numpy(states)).numpy()
    if not np.isfinite(actions).all() or np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
        raise RuntimeError("Invalid policy action after checkpoint reload")
    return {"status": "passed", "device": expected_device, "steps": cfg.total_steps, "checkpoint_verified": True,
            "metrics": metrics, "logged_updates_checked": len(active_rows),
            "alpha_final": float(agent.alpha.detach()),
            "support_rate_final": float(active_rows[-1]["lite/psi_q_rank_rate"]),
            "sha256": {name: sha(folder / name) for name in
                       ("config.json", "summary.json", "metrics.csv", "checkpoint.pt")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"profiles": ["gpu_smoke", "stability"], "tasks": SPEC["tasks"],
                          "methods": SPEC["methods"], "seed": SCHEDULE["validation_seed"],
                          "parallel": SCHEDULE["validation_parallel"], "stability_steps": SCHEDULE["stability_steps"]}, indent=2))
        return
    digest = verify()
    require_cpu_gate(digest)
    from scheduled_run import predecessor_state, resource_state
    prior = predecessor_state()
    if prior["state"] != "ready":
        raise RuntimeError(f"Predecessor not complete: {prior}")
    resources = resource_state()
    if resources["state"] != "ready":
        raise RuntimeError(f"Resources not idle: {resources}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = SCHEDULE["cuda_visible_devices"]
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / ".cache/numba")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dest = ROOT / "validation" / f"gpu_{stamp}"
    dest.mkdir(exist_ok=False)
    report = {"status": "running", "source_manifest_sha256": digest, "started_utc": stamp,
              "runs": [], "directory": str(dest.relative_to(ROOT)), "performance_threshold": None}
    active = []

    def save():
        report["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
        report["active_pids"] = [p.pid for p, _, _ in active]
        write_atomic(dest / "report.json", report)
        write_atomic(ROOT / "validation/gpu_validation_report.json", report)

    def stop(signum, _frame):
        raise InterruptedError(f"GPU validation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    save()
    try:
        report["cuda_update_parity"] = cuda_update_parity()
        for profile in ("gpu_smoke", "stability"):
            report["stage"] = profile
            jobs = [(task, method) for task in SPEC["tasks"] for method in SPEC["methods"]]
            started = time.monotonic()
            while jobs or active:
                if time.monotonic() - started > SCHEDULE["validation_timeout_seconds"]:
                    raise TimeoutError(f"{profile} exceeded fixed validation timeout")
                while jobs and len(active) < SCHEDULE["validation_parallel"]:
                    task, method = jobs.pop(0)
                    cfg = make_gpu_validation_config(task, method, SCHEDULE["validation_seed"], profile, dest / "runs")
                    log = (dest / f"{cfg.run_name}.log").open("x")
                    cmd = [sys.executable, "-B", str(ROOT / "train.py"), "--task", task, "--method", method,
                           "--seed", str(SCHEDULE["validation_seed"]), "--gpu-validation", profile, "--execute",
                           "--output-root", cfg.output_root]
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=os.environ, stdout=log, stderr=subprocess.STDOUT)
                    active.append((proc, log, (cfg, task, method)))
                    print(f"START {profile} {task} {method}: pid={proc.pid}", flush=True)
                for item in active[:]:
                    proc, log, (cfg, task, method) = item
                    code = proc.poll()
                    if code is None:
                        continue
                    active.remove(item)
                    log.close()
                    if code:
                        raise RuntimeError(f"{cfg.run_name} failed with exit={code}; no retry")
                    result = inspect_run(Path(cfg.output_root) / cfg.run_name, cfg)
                    report["runs"].append({"profile": profile, "task": task, "method": method,
                                           "seed": cfg.seeds.env_seed, "run_dir": str(Path(cfg.output_root) / cfg.run_name), **result})
                    print(f"PASS {cfg.run_name}", flush=True)
                save()
                if active:
                    time.sleep(10)
            verify()
        report["status"] = "passed"
        from gates import validate_gpu_report
        validate_gpu_report(report, digest)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        for proc, log, _ in active:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()
        active.clear()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
