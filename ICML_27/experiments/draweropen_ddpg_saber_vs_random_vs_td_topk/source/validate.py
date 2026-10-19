"""Run regressions and optionally certify three completed real-environment smokes."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from protocol import ROOT, ARMS
from freeze import verify
from analyze import accepted_attempt, validate_run, read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-root", type=Path)
    args = ap.parse_args()
    snapshot = verify()
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    test = subprocess.run([sys.executable, str(ROOT / "tests/test_assignment.py"), "-v"],
                          capture_output=True, text=True)
    (folder / "tests.log").write_text(test.stdout + test.stderr)
    print(test.stdout + test.stderr)
    if test.returncode:
        raise SystemExit(test.returncode)
    if args.smoke_root:
        runs = {}
        for arm in ARMS:
            attempt = accepted_attempt(args.smoke_root.resolve() / "smoke" / arm / "seed_01")
            if read_json(attempt / "status.json")["source_manifest_sha256"] != snapshot:
                raise ValueError("Smoke source differs from current snapshot")
            runs[arm] = validate_run(attempt, "smoke")
        gate = {"passed": True, "source_manifest_sha256": snapshot, "smoke_metrics": runs,
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "production_results": False, "historical_reuse_approved": False}
        (folder / "gate.json").write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n")
        print("Production gate passed; full training has not been started.")


if __name__ == "__main__":
    main()
