#!/usr/bin/env python3
"""Build a summary-only DrawerClose diagnostic artifact for seeds 1--10."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import build_diagnostic_artifact as base


SOURCE_SPECS = (
    {
        "name": "seeds_1_5",
        "directory": "diagnostic_artifact",
        "seeds": list(range(1, 6)),
        "source_commit": "d2038c7c964e73d0a8862ff3a5d361c21ae2f046",
    },
    {
        "name": "seeds_6_10",
        "directory": "diagnostic_artifact_s6_10",
        "seeds": list(range(6, 11)),
        "source_commit": "e990acdd30923bc8249f08b53d3abb7f7ae52ce9",
    },
)
EXPECTED_SEEDS = list(range(1, 11))
WINDOW_ORDER = {
    "early_0_500k": 0,
    "mid_500k_1m": 1,
    "late_1m_2m": 2,
}
SEED_FIELDS = (
    "seed",
    "source_run",
    "snapshot_count",
    "first_step",
    "final_step",
    "support_positive_rate",
    "support_td_corr",
    "td_top10_support_rate",
    "td_top10_enrichment_ratio",
    "td_support_pos_mean",
    "td_support_neg_mean",
    "residual_support_pos_mean",
    "residual_support_neg_mean",
    "support_residual_corr",
    "live_residual_support_pos_mean",
    "live_residual_support_neg_mean",
    "live_support_residual_corr",
    "td_support_difference",
    "residual_support_difference",
    "live_residual_support_difference",
    "phase_approach_batch_rate",
    "phase_approach_support_rate",
    "phase_approach_td_mean",
    "phase_approach_residual_mean",
    "phase_contact_batch_rate",
    "phase_contact_support_rate",
    "phase_contact_td_mean",
    "phase_contact_residual_mean",
    "phase_closing_batch_rate",
    "phase_closing_support_rate",
    "phase_closing_td_mean",
    "phase_closing_residual_mean",
    "phase_closed_batch_rate",
    "phase_closed_support_rate",
    "phase_closed_td_mean",
    "phase_closed_residual_mean",
)
WINDOW_FIELDS = (
    "seed",
    "window",
    "snapshot_count",
    "support_positive_rate",
    "support_td_corr",
    "td_top10_enrichment_ratio",
    "residual_support_difference",
)


def parse_args() -> argparse.Namespace:
    experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=experiment_root,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_root / "diagnostic_artifact_all_10",
    )
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repository_relative_path(experiment_root: Path, path: Path) -> str:
    repo_root = experiment_root.parents[2]
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def normalized_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("run_name", None)
    normalized.pop("seeds", None)
    return normalized


def validate_source_manifest(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "artifact_manifest.json"
    manifest = base.read_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"invalid source manifest: {manifest_path}")

    listed_paths: set[str] = set()
    total_bytes = 0
    for entry in entries:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        relative_text = relative.as_posix()
        if relative_text in listed_paths:
            raise ValueError(f"duplicate manifest path: {relative_text}")
        listed_paths.add(relative_text)
        source_file = source_root / relative
        if not source_file.is_file():
            raise FileNotFoundError(f"manifest file missing: {source_file}")
        if source_file.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"manifest byte mismatch: {source_file}")
        if base.sha256(source_file) != entry["sha256"]:
            raise ValueError(f"manifest hash mismatch: {source_file}")
        total_bytes += source_file.stat().st_size

    actual_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != listed_paths:
        missing = sorted(listed_paths - actual_paths)
        unlisted = sorted(actual_paths - listed_paths)
        raise ValueError(
            f"source manifest coverage mismatch: missing={missing}, unlisted={unlisted}"
        )

    return {
        "artifact_manifest_sha256": base.sha256(manifest_path),
        "manifest_file_count": len(entries),
        "manifest_total_bytes": total_bytes,
    }


def numeric_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> list[float]:
    values = base.finite_values(row.get(field) for row in rows)
    if len(values) != len(rows):
        raise ValueError(f"non-finite or missing values for {field}")
    return values


def mean_sd(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
    }


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator == 0.0:
        raise ValueError("ratio denominator must be finite and nonzero")
    return numerator / denominator


def build_paper_summary(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support = numeric_values(seed_rows, "support_positive_rate")
    support_td_corr = numeric_values(seed_rows, "support_td_corr")
    top10_enrichment = numeric_values(seed_rows, "td_top10_enrichment_ratio")
    td_positive = numeric_values(seed_rows, "td_support_pos_mean")
    td_negative = numeric_values(seed_rows, "td_support_neg_mean")
    residual_positive = numeric_values(seed_rows, "residual_support_pos_mean")
    residual_negative = numeric_values(seed_rows, "residual_support_neg_mean")

    phases: dict[str, Any] = {}
    for phase in base.PHASES:
        phases[phase] = {
            "batch_fraction": mean_sd(
                numeric_values(seed_rows, f"phase_{phase}_batch_rate")
            ),
            "support_positive_rate": mean_sd(
                numeric_values(seed_rows, f"phase_{phase}_support_rate")
            ),
            "pre_update_absolute_bellman_residual": mean_sd(
                numeric_values(seed_rows, f"phase_{phase}_td_mean")
            ),
            "fixed_target_one_update_residual_reduction": mean_sd(
                numeric_values(seed_rows, f"phase_{phase}_residual_mean")
            ),
        }

    return {
        "cohort": "DrawerClose DDPG HER+SABER diagnostic",
        "n_seeds": len(seed_rows),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "aggregation_unit": "seed",
        "support_positive_rate": mean_sd(support),
        "support_td_correlation": mean_sd(support_td_corr),
        "top10_td_error_support_enrichment": mean_sd(top10_enrichment),
        "pre_update_absolute_bellman_residual": {
            "support_positive_mean": mean(td_positive),
            "support_negative_mean": mean(td_negative),
            "positive_to_negative_ratio": ratio(
                mean(td_positive), mean(td_negative)
            ),
            "positive_greater_seed_count": sum(
                positive > negative
                for positive, negative in zip(td_positive, td_negative)
            ),
        },
        "fixed_target_one_update_residual_reduction": {
            "support_positive_mean": mean(residual_positive),
            "support_negative_mean": mean(residual_negative),
            "positive_to_negative_ratio": ratio(
                mean(residual_positive), mean(residual_negative)
            ),
            "positive_greater_seed_count": sum(
                positive > negative
                for positive, negative in zip(
                    residual_positive, residual_negative
                )
            ),
        },
        "phase_conditioned": phases,
    }


def read_source(
    experiment_root: Path,
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    source_root = experiment_root / str(spec["directory"])
    manifest_info = validate_source_manifest(source_root)
    validation = base.read_json(
        source_root / "validation" / "validation_summary.json"
    )
    expected_seeds = [int(seed) for seed in spec["seeds"]]
    validation_seeds = sorted(int(seed) for seed in validation["seeds"])
    if not validation.get("all_four_diagnostics_complete"):
        raise ValueError(f"incomplete source diagnostics: {source_root}")
    if validation_seeds != expected_seeds:
        raise ValueError(
            f"source validation seed mismatch: {validation_seeds} != {expected_seeds}"
        )

    seed_rows = base.read_csv(
        source_root / "results" / "diagnostic_by_seed.csv"
    )
    row_seeds = sorted(int(row["seed"]) for row in seed_rows)
    if row_seeds != expected_seeds:
        raise ValueError(
            f"source result seed mismatch: {row_seeds} != {expected_seeds}"
        )
    if tuple(seed_rows[0]) != SEED_FIELDS:
        raise ValueError(f"unexpected seed-result schema in {source_root}")

    window_rows = base.read_csv(
        source_root / "results" / "diagnostic_by_phase_window.csv"
    )
    if tuple(window_rows[0]) != WINDOW_FIELDS:
        raise ValueError(f"unexpected window-result schema in {source_root}")
    expected_window_pairs = {
        (seed, window)
        for seed in expected_seeds
        for window in WINDOW_ORDER
    }
    actual_window_pairs = {
        (int(row["seed"]), row["window"])
        for row in window_rows
    }
    if actual_window_pairs != expected_window_pairs:
        raise ValueError(f"source phase-window coverage mismatch in {source_root}")

    configs = []
    seed_sources: dict[str, Any] = {}
    for seed in expected_seeds:
        seed_root = source_root / "data" / "seeds" / f"seed_{seed:02d}"
        config = base.read_json(seed_root / "config.json")
        if int(config["seeds"]["env_seed"]) != seed:
            raise ValueError(f"environment seed mismatch for seed {seed}")
        configs.append(normalized_training_config(config))
        row = next(row for row in seed_rows if int(row["seed"]) == seed)
        seed_sources[str(seed)] = {
            "source_artifact": spec["name"],
            "source_run": row["source_run"],
        }

    config_hashes = {canonical_sha256(config) for config in configs}
    if len(config_hashes) != 1:
        raise ValueError(f"training config mismatch within {source_root}")

    source_info = {
        "name": spec["name"],
        "seeds": expected_seeds,
        "repository_path": repository_relative_path(
            experiment_root, source_root
        ),
        "source_commit": spec["source_commit"],
        "training_config_sha256_excluding_run_and_seeds": next(
            iter(config_hashes)
        ),
        "validation_all_four_diagnostics_complete": True,
        **manifest_info,
        "seed_sources": seed_sources,
    }
    return seed_rows, window_rows, source_info


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def artifact_readme(source_infos: Sequence[Mapping[str, Any]]) -> str:
    source_lines = "\n".join(
        f"- `{source['name']}`: seeds {source['seeds'][0]}--{source['seeds'][-1]}, "
        f"`{source['repository_path']}` at `{source['source_commit']}`"
        for source in source_infos
    )
    return f"""# DrawerClose DDPG SABER Diagnostic: Seeds 1--10

This summary-only artifact combines the two validated DrawerClose DDPG
HER+SABER diagnostic cohorts. Seeds are the independent aggregation units.

## Source artifacts

{source_lines}

The source artifacts retain all per-seed compressed diagnostic snapshots and
sanitized metadata. This directory deliberately does not duplicate those raw
files. `provenance.json` records each source commit, repository path, source
manifest hash, training-configuration hash, and seed-to-source mapping.

## Contents

- `results/diagnostic_by_seed.csv`: the ten source seed summaries.
- `results/diagnostic_aggregate.csv`: seed-level means, sample standard
  deviations, and 10,000-resample bootstrap confidence intervals.
- `results/diagnostic_by_phase_window.csv`: seed-level temporal summaries.
- `results/paper_summary.json`: manuscript-facing support, TD-error, residual,
  and phase aggregates.
- `validation/validation_summary.json`: source integrity, coverage, and config
  parity checks.
- `artifact_manifest.json`: byte counts and SHA-256 hashes for this artifact.

## Rebuild

From the experiment directory, run:

```bash
python analysis/build_combined_diagnostic_artifact.py
```

The builder verifies both source manifests before recomputing the aggregate.
It rejects missing seeds, duplicate seeds, incomplete diagnostics, schema
changes, source hash mismatches, and training-setting differences beyond run
names and seed values.
"""


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.unlink(missing_ok=True)

    all_seed_rows: list[dict[str, str]] = []
    all_window_rows: list[dict[str, str]] = []
    source_infos: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        seed_rows, window_rows, source_info = read_source(
            experiment_root, spec
        )
        all_seed_rows.extend(seed_rows)
        all_window_rows.extend(window_rows)
        source_infos.append(source_info)

    all_seed_rows.sort(key=lambda row: int(row["seed"]))
    all_window_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            WINDOW_ORDER[row["window"]],
        )
    )
    seeds = [int(row["seed"]) for row in all_seed_rows]
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"combined seeds must be 1--10, found {seeds}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in combined results: {seeds}")

    config_hashes = {
        source["training_config_sha256_excluding_run_and_seeds"]
        for source in source_infos
    }
    if len(config_hashes) != 1:
        raise ValueError("training configs differ between source cohorts")

    for row in all_seed_rows:
        if int(row["snapshot_count"]) != 9_976:
            raise ValueError(f"unexpected snapshot count for seed {row['seed']}")
        if int(row["first_step"]) != 5_000:
            raise ValueError(f"unexpected first step for seed {row['seed']}")
        if int(row["final_step"]) != 2_000_000:
            raise ValueError(f"unexpected final step for seed {row['seed']}")

    write_csv(
        output_root / "results" / "diagnostic_by_seed.csv",
        all_seed_rows,
        SEED_FIELDS,
    )
    write_csv(
        output_root / "results" / "diagnostic_aggregate.csv",
        base.aggregate_seed_stats(all_seed_rows),
        (
            "metric",
            "n_seeds",
            "mean",
            "sample_sd",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ),
    )
    write_csv(
        output_root / "results" / "diagnostic_by_phase_window.csv",
        all_window_rows,
        WINDOW_FIELDS,
    )
    paper_summary = build_paper_summary(all_seed_rows)
    write_json(output_root / "results" / "paper_summary.json", paper_summary)

    seed_sources = {
        seed: source
        for source_info in source_infos
        for seed, source in source_info["seed_sources"].items()
    }
    public_source_infos = [
        {
            key: value
            for key, value in source_info.items()
            if key != "seed_sources"
        }
        for source_info in source_infos
    ]
    provenance = {
        "artifact_schema_version": 1,
        "artifact_type": "summary-only combined diagnostic",
        "aggregation_unit": "seed",
        "builder_repository_path": repository_relative_path(
            experiment_root,
            experiment_root / "analysis" / Path(__file__).name,
        ),
        "raw_snapshot_policy": (
            "Referenced from source artifacts; not duplicated in this artifact."
        ),
        "source_artifacts": public_source_infos,
        "seed_sources": seed_sources,
    }
    write_json(output_root / "provenance.json", provenance)

    validation = {
        "artifact_schema_version": 1,
        "all_checks_passed": True,
        "aggregation_unit": "seed",
        "expected_seeds": EXPECTED_SEEDS,
        "observed_seeds": seeds,
        "n_seeds": len(seeds),
        "source_manifest_integrity": True,
        "source_diagnostics_complete": True,
        "training_config_parity_excluding_run_and_seeds": True,
        "training_config_sha256_excluding_run_and_seeds": next(
            iter(config_hashes)
        ),
        "snapshot_rows_per_seed": 9_976,
        "first_diagnostic_step": 5_000,
        "final_diagnostic_step": 2_000_000,
        "phase_window_rows": len(all_window_rows),
        "raw_snapshot_files_duplicated": False,
        "paper_summary_direction_counts": {
            "pre_update_td_positive_greater": paper_summary[
                "pre_update_absolute_bellman_residual"
            ]["positive_greater_seed_count"],
            "one_update_reduction_positive_greater": paper_summary[
                "fixed_target_one_update_residual_reduction"
            ]["positive_greater_seed_count"],
        },
    }
    write_json(
        output_root / "validation" / "validation_summary.json",
        validation,
    )
    (output_root / "README.md").write_text(
        artifact_readme(public_source_infos),
        encoding="utf-8",
    )

    duplicated_snapshots = list(
        output_root.rglob("diagnostic_snapshots.csv.gz")
    )
    if duplicated_snapshots:
        raise RuntimeError(
            f"combined artifact duplicates source snapshots: {duplicated_snapshots}"
        )
    for json_path in sorted(output_root.rglob("*.json")):
        leaked = base.absolute_json_strings(base.read_json(json_path))
        if leaked:
            raise RuntimeError(
                f"absolute local path leaked into {json_path}: {leaked[:3]}"
            )

    manifest = []
    for artifact_path in sorted(output_root.rglob("*")):
        if artifact_path.is_file() and artifact_path != manifest_path:
            manifest.append(
                {
                    "path": artifact_path.relative_to(output_root).as_posix(),
                    "bytes": artifact_path.stat().st_size,
                    "sha256": base.sha256(artifact_path),
                }
            )
    write_json(manifest_path, {"files": manifest})
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
