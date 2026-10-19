"""Compare independently executed old/new Tail smoke trajectories exactly."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from protocol import ROOT
from freeze import verify
from audit_reuse import file_hash
from analyze import validate_run
from train import write_json


def identical(a, b, name="root"):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Different keys: {name}")
        for key in a:
            identical(a[key], b[key], name + "." + str(key))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"Different lengths: {name}")
        for i, (x, y) in enumerate(zip(a, b)):
            identical(x, y, name + f"[{i}]")
    elif torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=name)
    elif a != b:
        raise AssertionError(f"Different values: {name}: {a} != {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    args = ap.parse_args()
    validate_run(args.old, "smoke", historical=True)
    validate_run(args.new, "smoke")
    old_path, new_path = args.old / "run/checkpoint.pt", args.new / "run/checkpoint.pt"
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    fields = ["agent", "replay", "eval_history_env", "noise_rng_state", "sample_rng_state",
              "weight_rng_state", "assignment_rng_state", "td_tie_rng_state", "gap_stats",
              "last_loss", "total_updates", "global_step", "episode_state"]
    for field in fields:
        identical(old[field], new[field], field)
    for path in (args.old, args.new):
        with (path / "run/metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        signatures = [float(row["runtime/numba_support_scan_enabled"]) for row in rows
                      if int(row["step"]) >= 5000 and row.get("runtime/numba_support_scan_enabled", "") != ""]
        # The stronger mask/kernel regression also checks the compiled scanner.
        if not signatures or not all(v == 1 for v in signatures):
            raise ValueError("Numba fallback during smoke")
    traces = [path / "run/assignment_windows.csv" for path in (args.old, args.new)]
    if file_hash(traces[0]) != file_hash(traces[1]):
        raise ValueError("All-update assignment diagnostics differ")
    report = {"passed": True, "source_manifest_sha256": verify(),
              "checked_fields_exact": fields, "assignment_trace_identical": True,
              "old_attempt": str(args.old.resolve()), "new_attempt": str(args.new.resolve()),
              "checkpoint_sha256": {"old": file_hash(old_path), "new": file_hash(new_path)},
              "updates": old["total_updates"], "steps": old["global_step"]}
    write_json(ROOT / "validation/tail_parity.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
