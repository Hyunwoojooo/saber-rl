#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "auc": (0.06742424242424243, -0.006767676767676767, 0.1416161616161616),
    "final_quarter_auc": (0.1854, -0.0212, 0.392),
    "final_success": (0.22, -0.2, 0.52),
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

    with (ROOT / "data/evaluation_curves.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        curves = list(csv.DictReader(handle))
    if len(curves) != 1_000:
        raise ValueError(f"expected 1000 curve rows, got {len(curves)}")
    if any("/home/" in json.dumps(row) for row in curves):
        raise ValueError("absolute path leaked into evaluation curves")

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
