#!/usr/bin/env python3
"""Validate compact Supplement artifacts without original raw run directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "results/derived"
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_manifest_path(artifact_dir: Path, stored_path: str) -> Path:
    parts = Path(stored_path).parts
    try:
        index = parts.index(artifact_dir.name)
    except ValueError:
        return artifact_dir / Path(stored_path).name
    return artifact_dir.joinpath(*parts[index + 1 :])


def verify_manifest(path: Path) -> list[str]:
    artifact_dir = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in payload.get("files", []):
        target = local_manifest_path(artifact_dir, str(item["path"]))
        if not target.is_file():
            errors.append(f"{target.relative_to(ROOT)}: missing")
            continue
        if target.stat().st_size != int(item["bytes"]):
            errors.append(f"{target.relative_to(ROOT)}: byte-size mismatch")
        if sha256(target) != item["sha256"]:
            errors.append(f"{target.relative_to(ROOT)}: sha256 mismatch")
    return errors


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    if value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def aggregate_errors(artifact_dir: Path) -> tuple[list[str], int]:
    per_path = artifact_dir / "per_seed_metrics.csv"
    aggregate_path = artifact_dir / "aggregate_metrics.csv"
    if not per_path.is_file() or not aggregate_path.is_file():
        return [], 0
    per_seed = read_csv(per_path)
    aggregate = read_csv(aggregate_path)
    if not per_seed or not aggregate:
        return [], 0

    aggregate_fields = tuple(aggregate[0])
    per_fields = set(per_seed[0])
    metric_bases = [
        field[: -len("_mean")]
        for field in aggregate_fields
        if field.endswith("_mean") and field[: -len("_mean")] in per_fields
    ]
    excluded = {
        "n_seeds",
        "hit90_reached",
        "hit90_total",
        "hit90_median_reached",
    }
    group_fields = [
        field
        for field in aggregate_fields
        if field in per_fields
        and field not in excluded
        and not field.endswith("_mean")
        and not field.endswith("_sample_sd")
        and field not in metric_bases
    ]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in per_seed:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)

    errors: list[str] = []
    for stored in aggregate:
        key = tuple(stored[field] for field in group_fields)
        rows = grouped.get(key, [])
        label = f"{artifact_dir.name}/{key}"
        if len(rows) != int(stored["n_seeds"]):
            errors.append(f"{label}: n_seeds mismatch")
            continue
        for metric in metric_bases:
            values = [as_float(row[metric]) for row in rows]
            if any(value is None for value in values):
                continue
            numeric = [float(value) for value in values if value is not None]
            mean = statistics.fmean(numeric)
            sd = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
            for suffix, computed in (("mean", mean), ("sample_sd", sd)):
                field = f"{metric}_{suffix}"
                if field not in stored or stored[field] == "":
                    continue
                expected = float(stored[field])
                if not math.isclose(computed, expected, rel_tol=0.0, abs_tol=TOLERANCE):
                    errors.append(
                        f"{label} {field}: recomputed={computed:.17g}, "
                        f"stored={expected:.17g}"
                    )
        if "hit90_step" in per_fields and "hit90_reached" in stored:
            hits = [int(row["hit90_step"]) for row in rows if row["hit90_step"]]
            if len(hits) != int(stored["hit90_reached"]):
                errors.append(f"{label}: hit90_reached mismatch")
            if len(rows) != int(stored["hit90_total"]):
                errors.append(f"{label}: hit90_total mismatch")
            if hits and stored["hit90_median_reached"]:
                median = statistics.median(hits)
                if not math.isclose(
                    float(median),
                    float(stored["hit90_median_reached"]),
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                ):
                    errors.append(f"{label}: hit90 median mismatch")
    return errors, len(aggregate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifests = sorted(DERIVED.glob("*/artifact_manifest.json"))
    errors: list[str] = []
    csv_rows = 0
    aggregate_artifacts = 0
    aggregate_rows = 0
    for manifest in manifests:
        errors.extend(verify_manifest(manifest))
    for artifact_dir in sorted(path for path in DERIVED.iterdir() if path.is_dir()):
        artifact_errors, rows = aggregate_errors(artifact_dir)
        errors.extend(artifact_errors)
        if rows:
            aggregate_artifacts += 1
            aggregate_rows += rows
    for path in DERIVED.glob("*/*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            csv_rows += sum(1 for _ in csv.DictReader(handle))

    report = {
        "artifact_manifests": len(manifests),
        "derived_csv_rows": csv_rows,
        "recomputed_aggregate_artifacts": aggregate_artifacts,
        "recomputed_aggregate_rows": aggregate_rows,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "supplement_reproduction_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
