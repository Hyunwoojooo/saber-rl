"""CPU-only sequential verification. Never launches a production run."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import ROOT, SPEC, SCHEDULE, make_config, validate_parity
from manifest import verify, verify_upstream
from gates import write_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    args = parser.parse_args()
    if hasattr(os, "nice"):
        os.nice(15)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / ".cache/numba")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "validation" / stamp
    output.mkdir(parents=True, exist_ok=False)
    digest = verify()
    report = {"status": "running", "source_manifest_sha256": digest, "device": "cpu",
              "threads": 1, "started_utc": stamp, "output": str(output.relative_to(ROOT)), "checks": []}
    start = time.monotonic()

    def save():
        write_atomic(output / "report.json", report)
        write_atomic(ROOT / "validation/validation_report.json", report)

    def command(name, argv):
        log = output / f"{name}.log"
        print(f"START {name}", flush=True)
        with log.open("x") as stream:
            result = subprocess.run([sys.executable, "-B", *argv], cwd=ROOT, env=os.environ,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
        report["checks"].append({"name": name, "exit_code": result.returncode, "log": str(log.relative_to(ROOT))})
        save()
        if result.returncode:
            print(log.read_text()[-14000:], flush=True)
            raise RuntimeError(f"Validation failed: {name}")
        print(f"PASS {name}", flush=True)

    save()
    try:
        verify_upstream(check_origins=True)
        origins = json.loads((ROOT / "upstream_manifest.json").read_text())["files"]
        report["original_source_files_rechecked"] = sum((ROOT.parents[1] / r["origin"]).exists() for r in origins)
        report["paired_configs"] = validate_parity()
        for path in sorted((ROOT / "code").rglob("*.py")) + list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py")):
            ast.parse(path.read_text(), filename=str(path))
        prior = (ROOT / SCHEDULE["predecessor"]).resolve()
        prior_manifest = prior / "source_manifest.json"
        prior_rows = json.loads(prior_manifest.read_text())["sha256"] if prior_manifest.exists() else {}
        prior_hashes = {p: hashlib.sha256((prior / p).read_bytes()).hexdigest() for p in prior_rows}
        report["existing_sac_source_files"] = len(prior_hashes)
        command("unit_and_environment_tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
        command("production_plan_dry_run", [str(ROOT / "launch.py")])
        if not args.unit_only:
            from gpu_validate import inspect_run
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            report["smoke_results"] = []
            for task in SPEC["tasks"]:
                for method in SPEC["methods"]:
                    command(f"smoke_{task}_{method}", [str(ROOT / "train.py"), "--task", task,
                        "--method", method, "--seed", "1", "--device", "cpu", "--smoke", "--execute",
                        "--output-root", str(output / "smoke")])
                    cfg = make_config(task, method, 1, "cpu", True, output / "smoke")
                    result = inspect_run(output / "smoke" / cfg.run_name, cfg, expected_device="cpu")
                    report["smoke_results"].append({"task": task, "method": method, **result})
        if any(hashlib.sha256((prior / p).read_bytes()).hexdigest() != h for p, h in prior_hashes.items()):
            raise RuntimeError("Existing SAC source changed during validation")
        verify_upstream(check_origins=True)
        if verify() != digest:
            raise RuntimeError("Source changed while tests were running")
        report["status"] = "unit_only_passed" if args.unit_only else "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["duration_seconds"] = time.monotonic() - start
        save()
    print(json.dumps({k: v for k, v in report.items() if k not in {"paired_configs", "checks"}}, indent=2))


if __name__ == "__main__":
    main()
