"""Read-only inputs and experiment-local outputs; no ML imports here."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT = ROOT.parent / "2026-09-25_draweropen_ddpg_assignment"
PAL = ROOT.parent / "2026-10-01_draweropen_ddpg_pal_v2"
ASSIGNMENT_SHA = "d281263084807ec6010986dc762301dce420b004e022c88985e51dcc50ccf496"
PAL_SHA = "05e08b06dd1938a839146cdd5ad0ccde2c09c309a017c3c77d8fe524061c8860"
SEEDS = tuple(range(1, 11))
ARMS = ("support", "random", "td_topk")
QUERIES = 1024
RANDOM_REPEATS = 5
SCAN_LIMIT = 512
QUERY_SEED = 2026100200
RETRIEVAL_SEED = 2026100300
TOTAL_STEPS = 2_000_000
TOTAL_UPDATES = 1_995_001


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(text)
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text)
        temp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing an empty unlabelled table: {path}")
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_directory(path):
    path = Path(path).resolve()
    if path == ROOT / "outputs" or not path.is_relative_to(ROOT / "outputs"):
        raise ValueError("Outputs must be in a new subdirectory of this experiment's outputs")
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_source(root, expected_sha):
    root = Path(root).resolve()
    path = root / "source_manifest.json"
    if digest(path) != expected_sha:
        raise ValueError(f"Unexpected source snapshot: {root.name}")
    for name, sha in read_json(path)["sha256"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or digest(source) != sha:
            raise ValueError(f"Frozen source changed: {root.name}/{name}")


def completed_attempt(root, arm, seed, source_sha):
    parent = Path(root) / "outputs/production" / arm / f"seed_{seed:02d}"
    attempts = sorted(p for p in parent.glob("attempt_*") if p.is_dir())
    if len(attempts) != 1 or attempts[0].name != "attempt_01":
        raise ValueError(f"Expected one fresh attempt: {arm}/{seed}")
    attempt = attempts[0]
    status = read_json(attempt / "status.json")
    expected = {"arm": arm, "seed": seed, "status": "complete", "profile": "production",
                "checkpoint_continuation": False, "source_manifest_sha256": source_sha}
    if any(status.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Incomplete or changed run: {arm}/{seed}")
    summary = read_json(attempt / "run/summary.json")
    if summary.get("total_steps") != TOTAL_STEPS or summary.get("total_updates") != TOTAL_UPDATES:
        raise ValueError(f"Incomplete training budget: {arm}/{seed}")
    return attempt


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite measurement: {value}")
    return result


def finish_artifact(folder, inputs, metadata):
    """Fail if an input changed while being read; seal all generated outputs."""
    for path, sha in inputs.items():
        if digest(path) != sha:
            raise ValueError(f"Input changed during analysis: {path}")
    from freeze import verify
    manifest = {"created_utc": utcnow(), "source_manifest_sha256": verify(),
                "inputs_sha256": inputs,
                "outputs_sha256": {str(p.relative_to(folder)): digest(p)
                                   for p in sorted(folder.rglob("*")) if p.is_file()},
                **metadata}
    write_json(folder / "manifest.json", manifest, exclusive=True)
    return manifest
