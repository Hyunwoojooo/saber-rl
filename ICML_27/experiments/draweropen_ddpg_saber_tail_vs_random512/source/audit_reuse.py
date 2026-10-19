"""Fail-closed historical-tail catalog; no outcome-dependent reuse decisions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from protocol import ROOT, ORIGIN, SEEDS, ENV_FIELDS
from analyze import accepted_attempt, validate_run, read_json
from freeze import verify
from train import environment, write_json


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_origin():
    manifest = ORIGIN / "source_manifest.json"
    recorded = read_json(manifest)
    for name, expected in recorded["sha256"].items():
        if file_hash(ORIGIN / name) != expected:
            raise ValueError(f"Historical source changed: {name}")
    for path in (ROOT / "upstream/code").rglob("*.py"):
        if file_hash(path) != file_hash(ORIGIN / "code" / path.relative_to(ROOT / "upstream/code")):
            raise ValueError(f"Upstream copy differs: {path}")
    if file_hash(ROOT / "upstream/source_manifest.json") != file_hash(manifest):
        raise ValueError("Copied source manifest differs")
    return file_hash(manifest)


def verify_catalog(path):
    catalog = read_json(path)
    if catalog.get("passed") is not True or catalog["source_manifest_sha256"] != verify():
        raise ValueError("Historical reuse not approved for this snapshot")
    if catalog["origin_source_manifest_sha256"] != verify_origin():
        raise ValueError("Historical source manifest changed")
    if set(catalog["runs"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("Historical catalog must retain every seed")
    for seed, row in catalog["runs"].items():
        expected = ORIGIN / "outputs/production/support" / f"seed_{int(seed):02d}" / "attempt_01"
        if Path(row["attempt"]) != expected:
            raise ValueError("Unexpected historical attempt")
        for name, expected_hash in row["sha256"].items():
            if file_hash(Path(row["attempt"]) / name) != expected_hash:
                raise ValueError(f"Historical artifact changed: seed{seed}/{name}")
    return catalog


def main():
    snapshot = verify()
    origin_hash = verify_origin()
    current = environment()
    runs = {}
    for seed in SEEDS:
        attempt = accepted_attempt(ORIGIN / "outputs/production/support" / f"seed_{seed:02d}")
        state = read_json(attempt / "status.json")
        if state["source_manifest_sha256"] != origin_hash or state["seed"] != seed:
            raise ValueError("Historical source/seed mismatch")
        metrics = validate_run(attempt, historical=True)
        old = read_json(attempt / "environment.json")
        for field in ENV_FIELDS:
            if current[field] != old[field]:
                raise ValueError(f"Cannot reuse seed {seed}: environment mismatch {field}")
        files = ["status.json", "environment.json", "pip_freeze.txt", "requested_config.json",
                 "run/config.json", "run/summary.json", "run/metrics.csv",
                 "run/assignment_windows.csv", "run/checkpoint.pt"]
        runs[str(seed)] = {"attempt": str(attempt), "metrics": metrics,
                          "sha256": {f: file_hash(attempt / f) for f in files}}
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    path = folder / "reuse_catalog.json"
    if path.exists():
        raise FileExistsError("Reuse catalog already exists; do not replace evidence")
    report = {"passed": True, "source_manifest_sha256": snapshot,
              "origin_source_manifest_sha256": origin_hash, "current_environment": current,
              "runs": runs, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "note": "Historical baseline already observed; all ten seeds retained, no score filter."}
    write_json(path, report)
    print("Historical source, configs, environment and all ten completed Tail runs verified.")


if __name__ == "__main__":
    main()
