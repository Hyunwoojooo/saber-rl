"""Experiment-local IO and attempt rules; no simulator or tensor imports."""
from __future__ import annotations

import json
from pathlib import Path

from protocol import ROOT


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def local_output(path):
    path = Path(path).resolve()
    allowed = (ROOT / "outputs").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError(f"PAL may write results only below {allowed}")
    return path


def list_attempts(seed_dir):
    return sorted(p for p in Path(seed_dir).glob("attempt_*") if p.is_dir())


def accepted_attempt(seed_dir):
    attempts = list_attempts(seed_dir)
    if not attempts:
        raise ValueError("Missing run")
    for old in attempts[:-1]:
        if read_json(old / "status.json")["status"] not in {"failed_infrastructure", "interrupted"}:
            raise ValueError("Unapproved replacement of a scientific run")
    current = attempts[-1]
    if len(attempts) > 1 and not read_json(current / "status.json").get("retry_reason"):
        raise ValueError("Missing infrastructure retry reason")
    return current


def require_production_gate(snapshot, device):
    gate = read_json(ROOT / "validation" / f"gate_{device}.json")
    if (gate.get("passed") is not True or gate.get("source_manifest_sha256") != snapshot
            or gate.get("device") != device or gate.get("tests_passed") is not True):
        raise ValueError("Production requires passing tests and same-device smoke on this snapshot")
    return gate
