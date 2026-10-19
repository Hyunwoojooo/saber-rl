"""Explicit, deferred tests/smoke certification. No work is done by default."""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from protocol import ROOT
from freeze import verify
from records import accepted_attempt, local_output, read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--smoke-root", type=Path)
    args = ap.parse_args()
    if not args.run_tests:
        raise SystemExit("No tests run. Use --run-tests only when execution is authorized.")
    snapshot = verify()
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    # A failed revalidation must not leave a usable old gate.
    for device in ("cpu", "cuda"):
        write_json(folder / f"gate_{device}.json", {
            "passed": False, "source_manifest_sha256": snapshot,
            "device": device, "reason": "revalidation in progress"})
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    (folder / "tests.log").write_text(test.stdout + test.stderr)
    print(test.stdout + test.stderr)
    write_json(folder / "tests.json", {"passed": test.returncode == 0,
               "source_manifest_sha256": snapshot, "checked_utc": datetime.now(timezone.utc).isoformat(),
               "device": "cpu", "returncode": test.returncode})
    if test.returncode:
        raise SystemExit(test.returncode)
    if args.smoke_root:
        from analyze import validate_run, audit_environment
        from train import environment
        import torch
        torch.set_num_threads(1)
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        attempt = accepted_attempt(local_output(args.smoke_root) / "smoke/pal/seed_01")
        if read_json(attempt / "status.json")["source_manifest_sha256"] != snapshot:
            raise ValueError("Smoke source differs from the current snapshot")
        cfg = read_json(attempt / "run/config.json")
        device = cfg["device"]
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif device != "cuda" or not torch.cuda.is_available():
            raise ValueError("GPU gate requires a GPU smoke and available CUDA")
        metrics = validate_run(attempt, "smoke")
        observed_environment = environment()
        audit_environment(read_json(attempt / "environment.json"), observed_environment)
        gate = {"passed": True, "tests_passed": True, "source_manifest_sha256": snapshot,
                "device": device, "smoke_attempt": str(attempt), "smoke_metrics": metrics,
                "environment": observed_environment,
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "production_started": False, "reference_reuse_approved": False}
        write_json(folder / f"gate_{device}.json", gate)
        print(f"{device} gate passed. No production training has been started.")


if __name__ == "__main__":
    main()
