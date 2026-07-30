#!/usr/bin/env python3
"""Build a compact all-seed DrawerOpen fixed-target diagnostic artifact.

Seeds 1--5 are read from their canonical completed run directories. Seeds
6--10 are read from the committed, deterministically compressed artifact.
The output contains only derived seed/cohort summaries and source hashes; it
does not duplicate the full metrics files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


SEEDS = tuple(range(1, 11))
PRIMARY_SEEDS = tuple(range(1, 6))
EXTENSION_SEEDS = tuple(range(6, 11))
PHASES = ("approach", "contact", "opening", "open")
EXPECTED_STEPS = tuple(range(5_000, 2_000_001, 200))
EXPECTED_TOP10_COUNT = 26
EXECUTION_COMMIT = "22015c0a3743d4a10f72197752022ed0e08ee730"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION = (
    ROOT
    / "metaworld_Drawer_Open_DDPG"
    / "paper_artifacts"
    / "drawer_open_ddpg_saber_fixed_target_diagnostic_s6_10_2m"
)
DEFAULT_OUTPUT = (
    ROOT
    / "metaworld_Drawer_Open_DDPG"
    / "paper_artifacts"
    / "drawer_open_ddpg_saber_fixed_target_diagnostic_all_10_2m"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-root",
        type=Path,
        required=True,
        help="Directory containing the completed seeds 1--5 run directories.",
    )
    parser.add_argument("--extension-artifact", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: Any) -> float:
    return float(value) if finite(value) else float("nan")


def finite_mean(values: Iterable[Any]) -> float:
    items = [number(value) for value in values if finite(value)]
    return statistics.fmean(items) if items else float("nan")


def sample_sd(values: Iterable[Any]) -> float:
    items = [number(value) for value in values if finite(value)]
    return statistics.stdev(items) if len(items) > 1 else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def locate_primary_run(root: Path, seed: int) -> Path:
    matches: list[Path] = []
    for config_path in root.glob("*/config.json"):
        config = read_json(config_path)
        if int(config.get("seeds", {}).get("env_seed", -1)) != seed:
            continue
        if config.get("task_name") != "drawer_open":
            continue
        if config.get("method") != "b3_support_runtime_opt":
            continue
        if config.get("weight_scope") != "critic_only":
            continue
        matches.append(config_path.parent)
    if len(matches) != 1:
        raise AssertionError(f"seed {seed}: expected one primary run, got {matches}")
    return matches[0]


def required_columns() -> set[str]:
    columns = {
        "step",
        "eval/success_mean",
        "diagnostic/fixed_target_from_pre_update",
        "diagnostic/same_minibatch_one_update",
        "diagnostic/snapshot_only_collection",
        "diagnostic/support_positive_rate",
        "diagnostic/support_to_td_error_corr",
        "diagnostic/td_error_top10_count",
        "diagnostic/td_error_top10_enrichment_ratio",
        "diagnostic/td_error_support_pos_mean",
        "diagnostic/td_error_support_neg_mean",
        "diagnostic/residual_reduction_support_pos_mean",
        "diagnostic/residual_reduction_support_neg_mean",
    }
    for phase in PHASES:
        columns.add(f"phase_{phase}_frac")
        columns.add(f"b3/support_by_phase_{phase}")
    return columns


def compact_snapshots(handle: TextIO, seed: int) -> list[dict[str, str]]:
    reader = csv.DictReader(handle)
    missing = sorted(required_columns() - set(reader.fieldnames or ()))
    if missing:
        raise ValueError(f"seed {seed}: missing columns {missing}")
    by_step: dict[int, dict[str, str]] = {}
    for row in reader:
        if number(row.get("diagnostic/fixed_target_from_pre_update")) != 1.0:
            continue
        # Evaluation rows copy the latest diagnostics and duplicate exact 10k
        # steps. Retain only the dedicated diagnostic snapshot row.
        if finite(row.get("eval/success_mean")):
            continue
        step = int(float(row["step"]))
        if step in by_step:
            raise AssertionError(f"seed {seed}: duplicate diagnostic step {step}")
        for key in (
            "diagnostic/same_minibatch_one_update",
            "diagnostic/snapshot_only_collection",
        ):
            if number(row.get(key)) != 1.0:
                raise AssertionError(f"seed {seed}: {key} != 1 at step {step}")
        if number(row.get("diagnostic/td_error_top10_count")) != EXPECTED_TOP10_COUNT:
            raise AssertionError(f"seed {seed}: invalid top-10 count at step {step}")
        by_step[step] = row
    if tuple(sorted(by_step)) != EXPECTED_STEPS:
        raise AssertionError(
            f"seed {seed}: fixed-target grid mismatch "
            f"({len(by_step)} vs {len(EXPECTED_STEPS)})"
        )
    return [by_step[step] for step in EXPECTED_STEPS]


def weighted_phase_support(rows: list[Mapping[str, str]], phase: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        fraction = number(row.get(f"phase_{phase}_frac"))
        support = number(row.get(f"b3/support_by_phase_{phase}"))
        if math.isfinite(fraction) and fraction > 0.0 and math.isfinite(support):
            numerator += fraction * support
            denominator += fraction
    return numerator / denominator if denominator else float("nan")


def summarize_seed(seed: int, rows: list[dict[str, str]], source: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seed": seed,
        "source": source,
        "snapshot_count": len(rows),
        "first_step": EXPECTED_STEPS[0],
        "last_step": EXPECTED_STEPS[-1],
        "support_positive_rate": finite_mean(
            row.get("diagnostic/support_positive_rate") for row in rows
        ),
        "support_td_corr": finite_mean(
            row.get("diagnostic/support_to_td_error_corr") for row in rows
        ),
        "td_top10_enrichment_ratio": finite_mean(
            row.get("diagnostic/td_error_top10_enrichment_ratio") for row in rows
        ),
        "td_support_pos_mean": finite_mean(
            row.get("diagnostic/td_error_support_pos_mean") for row in rows
        ),
        "td_support_neg_mean": finite_mean(
            row.get("diagnostic/td_error_support_neg_mean") for row in rows
        ),
        "fixed_reduction_pos_mean": finite_mean(
            row.get("diagnostic/residual_reduction_support_pos_mean") for row in rows
        ),
        "fixed_reduction_neg_mean": finite_mean(
            row.get("diagnostic/residual_reduction_support_neg_mean") for row in rows
        ),
    }
    for phase in PHASES:
        result[f"phase_{phase}_batch_fraction"] = finite_mean(
            row.get(f"phase_{phase}_frac") for row in rows
        )
        result[f"phase_{phase}_support_rate"] = weighted_phase_support(rows, phase)
    return result


def cohort_summary(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    td_pos = finite_mean(row["td_support_pos_mean"] for row in rows)
    td_neg = finite_mean(row["td_support_neg_mean"] for row in rows)
    reduction_pos = finite_mean(row["fixed_reduction_pos_mean"] for row in rows)
    reduction_neg = finite_mean(row["fixed_reduction_neg_mean"] for row in rows)
    return {
        "cohort": label,
        "seeds": f"{rows[0]['seed']}-{rows[-1]['seed']}",
        "n_seeds": len(rows),
        "support_rate_mean": finite_mean(row["support_positive_rate"] for row in rows),
        "support_rate_sample_sd": sample_sd(
            row["support_positive_rate"] for row in rows
        ),
        "pearson_r_mean": finite_mean(row["support_td_corr"] for row in rows),
        "top10_enrichment_mean": finite_mean(
            row["td_top10_enrichment_ratio"] for row in rows
        ),
        "td_support_pos_mean": td_pos,
        "td_support_neg_mean": td_neg,
        "td_pos_neg_ratio": td_pos / td_neg,
        "td_positive_seed_count": sum(
            row["td_support_pos_mean"] > row["td_support_neg_mean"] for row in rows
        ),
        "fixed_reduction_pos_mean": reduction_pos,
        "fixed_reduction_neg_mean": reduction_neg,
        "fixed_reduction_pos_neg_ratio": reduction_pos / reduction_neg,
        "fixed_positive_seed_count": sum(
            row["fixed_reduction_pos_mean"] > row["fixed_reduction_neg_mean"]
            for row in rows
        ),
    }


def phase_summaries(label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for phase in PHASES:
        fractions = [row[f"phase_{phase}_batch_fraction"] for row in rows]
        support = [
            row[f"phase_{phase}_support_rate"]
            for row in rows
            if finite(row[f"phase_{phase}_support_rate"])
        ]
        result.append(
            {
                "cohort": label,
                "seeds": f"{rows[0]['seed']}-{rows[-1]['seed']}",
                "phase": phase,
                "n_seeds": len(rows),
                "n_support_seeds": len(support),
                "batch_fraction_mean": finite_mean(fractions),
                "batch_fraction_sample_sd": sample_sd(fractions),
                "support_rate_mean": finite_mean(support),
                "support_rate_sample_sd": sample_sd(support),
            }
        )
    return result


def artifact_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(file_record(path, path.relative_to(output_root).as_posix()))
    write_json(
        output_root / "artifact_manifest.json",
        {
            "artifact_id": "drawer_open_ddpg_saber_fixed_target_diagnostic_all_10_2m",
            "schema_version": 1,
            "execution_commit": EXECUTION_COMMIT,
            "seeds": list(SEEDS),
            "raw_metrics_duplicated": False,
            "manifest_self_excluded": True,
            "files": files,
        },
    )


def main() -> int:
    args = parse_args()
    primary_root = args.primary_root.resolve()
    extension = args.extension_artifact.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_root}")

    extension_manifest = read_json(extension / "artifact_manifest.json")
    if extension_manifest.get("execution_commit") != EXECUTION_COMMIT:
        raise AssertionError("extension execution commit mismatch")
    if extension_manifest.get("seeds") != list(EXTENSION_SEEDS):
        raise AssertionError("extension seed set mismatch")

    seed_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed in PRIMARY_SEEDS:
        run_dir = locate_primary_run(primary_root, seed)
        with (run_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = compact_snapshots(handle, seed)
        seed_rows.append(summarize_seed(seed, rows, run_dir.name))
        sources.append(
            {
                "seed": seed,
                "source_kind": "canonical_completed_run",
                "source_run": run_dir.name,
                "files": [
                    file_record(run_dir / name, f"{run_dir.name}/{name}")
                    for name in ("metrics.csv", "config.json", "summary.json")
                ],
            }
        )

    extension_rel = extension.relative_to(ROOT)
    for seed in EXTENSION_SEEDS:
        metrics_path = extension / "data" / "seeds" / f"seed_{seed:02d}" / "metrics.csv.gz"
        with gzip.open(metrics_path, "rt", encoding="utf-8", newline="") as handle:
            rows = compact_snapshots(handle, seed)
        seed_rows.append(summarize_seed(seed, rows, f"seed_{seed:02d}/metrics.csv.gz"))
        sources.append(
            {
                "seed": seed,
                "source_kind": "committed_extension_artifact",
                "files": [
                    file_record(metrics_path, metrics_path.relative_to(ROOT).as_posix())
                ],
            }
        )

    primary_rows = seed_rows[:5]
    extension_rows = seed_rows[5:]
    cohort_rows = [
        cohort_summary("primary", primary_rows),
        cohort_summary("extension", extension_rows),
        cohort_summary("pooled", seed_rows),
    ]
    phase_rows = phase_summaries("primary", primary_rows)
    phase_rows.extend(phase_summaries("extension", extension_rows))
    phase_rows.extend(phase_summaries("pooled", seed_rows))

    write_csv(output_root / "results" / "diagnostic_by_seed.csv", seed_rows)
    write_csv(output_root / "results" / "diagnostic_cohort_summary.csv", cohort_rows)
    write_csv(output_root / "results" / "phase_cohort_summary.csv", phase_rows)
    write_json(
        output_root / "validation" / "validation_summary.json",
        {
            "all_runs_validated": True,
            "seed_count": len(seed_rows),
            "seeds": list(SEEDS),
            "snapshots_per_seed": len(EXPECTED_STEPS),
            "first_step": EXPECTED_STEPS[0],
            "last_step": EXPECTED_STEPS[-1],
            "cadence": 200,
            "fixed_target_from_pre_update": True,
            "same_minibatch_one_update": True,
            "exact_top10_count": EXPECTED_TOP10_COUNT,
        },
    )
    write_json(
        output_root / "provenance" / "source_manifest.json",
        {
            "execution_commit": EXECUTION_COMMIT,
            "raw_metrics_duplicated": False,
            "extension_artifact": {
                "path": extension_rel.as_posix(),
                "manifest_sha256": sha256(extension / "artifact_manifest.json"),
            },
            "sources": sources,
        },
    )
    (output_root / "README.md").write_text(
        "# DrawerOpen DDPG SABER fixed-target diagnostics, seeds 1--10\n\n"
        "Compact seed- and cohort-level summaries for the fixed-target "
        "diagnostic. Seeds 1--5 form the primary diagnostic cohort; seeds "
        "6--10 are a separate extension. The pooled row is descriptive. "
        "Every seed contains 9,976 snapshots on the 5k--2M/200-step grid, "
        "uses the pre-update Bellman target after one critic step, and uses "
        "an exact top-decile count of 26. Full raw metrics are not duplicated "
        "in this derived artifact; source hashes are recorded under "
        "`provenance/`.\n",
        encoding="utf-8",
    )
    artifact_manifest(output_root)
    print(f"built DrawerOpen all-10 fixed-target summary: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
