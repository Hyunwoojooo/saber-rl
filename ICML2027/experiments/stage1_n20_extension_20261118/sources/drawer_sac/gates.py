"""Fail-closed gates shared by the scheduler, launcher, and individual trainer."""
import hashlib
import json
from pathlib import Path

from protocol import ROOT, SPEC, SCHEDULE


def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def require_cpu_gate(digest, root=ROOT):
    report = json.loads((root / "validation/validation_report.json").read_text())
    if report.get("status") != "passed" or report.get("source_manifest_sha256") != digest:
        raise RuntimeError("CPU validation has not passed on this frozen source")
    return report


def validate_gpu_report(report, digest):
    if report.get("status") != "passed" or report.get("source_manifest_sha256") != digest:
        raise RuntimeError("GPU smoke and stability validation have not both passed on this source")
    parity = report.get("cuda_update_parity", {})
    if parity.get("status") != "passed" or parity.get("updates") != 3:
        raise RuntimeError("FPP-reference CUDA update parity has not passed")
    expected = {(profile, task, method) for profile in ("gpu_smoke", "stability")
                for task in SPEC["tasks"] for method in SPEC["methods"]}
    rows = report.get("runs", [])
    actual = {(r.get("profile"), r.get("task"), r.get("method")) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("GPU gate requires every task/method/profile exactly once")
    for row in rows:
        if row.get("status") != "passed" or row.get("seed") != SCHEDULE["validation_seed"]:
            raise RuntimeError("A GPU validation run is missing or failed")
        if row.get("steps") != SCHEDULE[f"{row['profile']}_steps"]:
            raise RuntimeError("GPU validation training budget mismatch")
        if row.get("device") != "cuda" or row.get("checkpoint_verified") is not True:
            raise RuntimeError("CUDA execution and checkpoint verification are required")
        hashes = row.get("sha256", {})
        if set(hashes) != {"config.json", "summary.json", "metrics.csv", "checkpoint.pt"}:
            raise RuntimeError("Validation evidence hashes are incomplete")
        for value in hashes.values():
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RuntimeError("Invalid validation SHA256")


def require_production_gates(digest, root=ROOT):
    require_cpu_gate(digest, root)
    gpu = json.loads((root / "validation/gpu_validation_report.json").read_text())
    validate_gpu_report(gpu, digest)
    for row in gpu["runs"]:
        folder = Path(row["run_dir"]).resolve()
        if not folder.is_relative_to((root / "validation").resolve()):
            raise RuntimeError("GPU validation evidence must be experiment-local")
        for name, expected in row["sha256"].items():
            with (folder / name).open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"GPU validation evidence changed: {folder / name}")
    return gpu
