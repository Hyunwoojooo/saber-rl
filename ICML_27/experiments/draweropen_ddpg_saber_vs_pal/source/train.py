"""One fresh attempt, including durable failure and environment records."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from protocol import ARMS, ROOT, make_config
from freeze import verify
from records import local_output, require_production_gate


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    import torch
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                if d.metadata.get("Name")}
    gpu = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        gpu.append({"name": prop.name, "cuda_addressable_bytes": prop.total_memory})
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "host": platform.node(), "cpu_model": platform.processor(),
            "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "nvidia_driver": driver.stdout.strip() if driver and driver.returncode == 0 else None,
            "architecture": platform.machine(), "cpu_count": os.cpu_count(),
            "torch_cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "packages": packages, "gpu": gpu,
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, choices=range(1, 11), required=True)
    ap.add_argument("--attempt-dir", type=Path, required=True)
    ap.add_argument("--profile", choices=("production", "smoke"), default="production")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--retry-reason")
    ap.add_argument("--execute", action="store_true", help="Explicitly authorize this attempt")
    args = ap.parse_args()
    if not args.execute:
        raise SystemExit("Not started. Use --execute only after execution is authorized.")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    snapshot = verify()
    gate = None
    if args.profile == "production":
        gate = require_production_gate(snapshot, args.device)
    root = local_output(args.attempt_dir)
    root.mkdir(parents=True, exist_ok=False)
    state = dict(arm=args.arm, seed=args.seed, profile=args.profile, status="running",
                 source_manifest_sha256=snapshot, checkpoint_continuation=False,
                 retry_reason=args.retry_reason, pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat())
    status_path = root / "status.json"
    write_json(status_path, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        import torch
        torch.set_num_threads(1)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.set_num_interop_threads(1)
        observed_environment = environment()
        write_json(root / "environment.json", observed_environment)
        if gate is not None:
            from analyze import audit_environment
            audit_environment(gate["environment"], observed_environment)
        result = subprocess.run([os.sys.executable, "-m", "pip", "freeze", "--all"],
                                capture_output=True, text=True, check=True)
        (root / "pip_freeze.txt").write_text(result.stdout)
        cfg = make_config(args.arm, args.seed, root, args.profile, args.device)
        write_json(root / "requested_config.json", cfg.as_dict())
        from metacog_lite_v0.trainer import run_training
        run_training(cfg, ROOT)
        from analyze import validate_run
        metrics = validate_run(root, args.profile, require_status=False)
        state.update(status="complete", metrics=metrics)
    except FloatingPointError as exc:
        state.update(status="failed_learning", error=str(exc))
        raise
    except KeyboardInterrupt as exc:
        state.update(status="interrupted", error=str(exc))
        raise
    except OSError as exc:
        state.update(status="failed_infrastructure", error=str(exc))
        raise
    except BaseException as exc:
        state.update(status="failed_unclassified", error=str(exc))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if state["status"] != "complete":
            (root / "failure_traceback.txt").write_text(traceback.format_exc())
        write_json(status_path, state)


if __name__ == "__main__":
    main()
