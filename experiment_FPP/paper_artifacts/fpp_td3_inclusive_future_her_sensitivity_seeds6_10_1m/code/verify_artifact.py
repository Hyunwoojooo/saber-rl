#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(6, 11))
GRID = tuple(range(10_000, 1_000_001, 10_000))
EXPECTED = {
    "auc": (0.011363636363636364, -0.050656565656565664, 0.095),
    "final_quarter_auc": (-0.0264, -0.1422, 0.0724),
    "final_success": (-0.09, -0.35, 0.13),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: str, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: {actual} != {expected}")


def main() -> None:
    manifest = json.loads(
        (ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"{path}: size mismatch")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch")

    validation = json.loads(
        (ROOT / "validation_summary.json").read_text(encoding="utf-8")
    )
    if validation.get("status") != "PASS":
        raise ValueError("artifact validation status is not PASS")
    if validation.get("complete_runs") != 10:
        raise ValueError("artifact does not contain ten complete runs")

    contracts = json.loads(
        (ROOT / "provenance/run_contracts.json").read_text(encoding="utf-8")
    )
    if tuple(contracts.get("seeds", ())) != SEEDS:
        raise ValueError("run-contract seed cohort mismatch")
    if contracts.get("execution_source_commit") != (
        "70fb7041f4b5142ce91bfd207142e361cced17b8"
    ):
        raise ValueError("run-contract execution commit mismatch")

    with (ROOT / "data/evaluation_curves.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        curves = list(csv.DictReader(handle))
    if len(curves) != 1_000:
        raise ValueError(f"expected 1000 curve rows, got {len(curves)}")
    if any("/home/" in json.dumps(row) for row in curves):
        raise ValueError("absolute path leaked into evaluation curves")

    by_pair: dict[tuple[str, int], dict[int, float]] = {}
    for row in curves:
        key = (row["arm"], int(row["seed"]))
        step = int(row["step"])
        success = float(row["success"])
        if not math.isfinite(success) or not 0.0 <= success <= 1.0:
            raise ValueError(f"{key} step {step}: invalid success")
        bucket = by_pair.setdefault(key, {})
        if step in bucket:
            raise ValueError(f"{key}: duplicate evaluation step {step}")
        bucket[step] = success

    expected_pairs = {(arm, seed) for arm in ("her", "saber") for seed in SEEDS}
    if set(by_pair) != expected_pairs:
        raise ValueError("arm/seed coverage mismatch")
    for pair, curve in by_pair.items():
        if tuple(sorted(curve)) != GRID:
            raise ValueError(f"{pair}: exact evaluation grid mismatch")

    with (ROOT / "results/paired_statistics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        paired = {row["metric"]: row for row in csv.DictReader(handle)}
    if set(paired) != set(EXPECTED):
        raise ValueError("paired metric set mismatch")
    for metric, (mean, low, high) in EXPECTED.items():
        close(paired[metric]["mean_difference"], mean, f"{metric}.mean")
        close(paired[metric]["ci_low"], low, f"{metric}.ci_low")
        close(paired[metric]["ci_high"], high, f"{metric}.ci_high")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}
    )
    if "/home/" in serialized or "/tmp/" in serialized:
        raise ValueError("local absolute path leaked into artifact")
    print("inclusive-future sensitivity artifact: PASS")


if __name__ == "__main__":
    main()
