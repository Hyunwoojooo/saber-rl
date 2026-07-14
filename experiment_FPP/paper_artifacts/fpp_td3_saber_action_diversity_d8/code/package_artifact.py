#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 1_000_001, 10_000))
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
FALLBACK_SUCCESS_FIELD = "eval/success_mean"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027
EXPERIMENT_REL = Path("experiment_FPP/td3_saber_action_diversity_ablation")
SOURCE_RESULTS_REL = Path("experiment_FPP/reports/td3_saber_action_diversity_d8")
DEFAULT_OUTPUT_REL = Path(
    "experiment_FPP/paper_artifacts/fpp_td3_saber_action_diversity_d8"
)
PROBE_ROOT_REL = Path("experiment_FPP/runs/td3_saber_diversity_probe_d1_50k")
METHODS = (
    (
        "her",
        "TD3+HER",
        Path("experiment_FPP/runs/td3_her_baseline_ls512"),
    ),
    (
        "saber",
        "TD3+HER+SABER",
        Path("experiment_FPP/runs/td3_saber_critic_only_v4_ls512"),
    ),
    (
        "diverse_d8",
        "TD3+HER+SABER-Diverse",
        Path("experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512"),
    ),
    (
        "random_count_d8",
        "Random-Diversity-Count",
        Path(
            "experiment_FPP/runs/"
            "td3_saber_random_diversity_count_d8_critic_only_v4_ls512"
        ),
    ),
)
RESULT_FILES = (
    "per_seed_results.csv",
    "aggregate_results.csv",
    "paired_differences.csv",
    "paired_bootstrap_ci.csv",
)
RUN_JSON_FILES = ("config.json", "env_snapshot.json", "summary.json")
GATE_ACTIVITY_FIELDS = (
    (
        "pre_diversity_support_rate",
        "support/pre_diversity_support_rate",
        0.199484,
    ),
    ("d8_positive_rate", "support/diverse_positive_rate", 0.195250),
    (
        "d8_pass_given_main",
        "support/diversity_pass_given_main_rate",
        0.984199,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def row_step(row: Mapping[str, str]) -> int | None:
    for key in ("step", "global_step", "env_step"):
        value = finite_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def row_success(row: Mapping[str, str]) -> float | None:
    primary = finite_float(row.get(PRIMARY_SUCCESS_FIELD))
    if primary is not None:
        return primary
    return finite_float(row.get(FALLBACK_SUCCESS_FIELD))


def metrics_path_for(run_dir: Path) -> Path:
    for name in ("metrics_selected_10k.csv", "metrics.csv"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{run_dir}: missing metrics_selected_10k.csv and metrics.csv"
    )


def seed_for_run(run_dir: Path) -> int:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        return int(config["seeds"]["env_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{config_path}: missing seeds.env_seed") from exc


def discover_runs(root: Path, expected_seeds: Sequence[int]) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing run root: {root}")
    candidates = [root] if (root / "summary.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir()
    )
    found: dict[int, Path] = {}
    for run_dir in candidates:
        if not (run_dir / "summary.json").is_file():
            continue
        for name in RUN_JSON_FILES:
            if not (run_dir / name).is_file():
                raise FileNotFoundError(f"{run_dir}: missing {name}")
        metrics_path_for(run_dir)
        seed = seed_for_run(run_dir)
        if seed in found:
            raise ValueError(f"{root}: duplicate seed {seed}")
        found[seed] = run_dir
    if set(found) != set(expected_seeds):
        raise ValueError(
            f"{root}: expected seeds {list(expected_seeds)}, found {sorted(found)}"
        )
    return {seed: found[seed] for seed in expected_seeds}


def validate_eval_grid(metrics_path: Path, required_steps: Sequence[int]) -> None:
    values: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = row_step(row)
            success = row_success(row)
            if step is None or success is None:
                continue
            prior = values.get(step)
            if prior is not None and not math.isclose(
                prior, success, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{metrics_path}: disagreeing evaluation rows at step {step}"
                )
            values[step] = success
    missing = [step for step in required_steps if step not in values]
    if missing:
        raise ValueError(f"{metrics_path}: missing exact evaluation steps {missing}")


def compact_metrics(source: Path, destination: Path, max_step: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: missing CSV header")
        fieldnames = list(reader.fieldnames)
        selected = []
        for row in reader:
            step = row_step(row)
            if step is None or step <= 0 or step > max_step or step % 10_000:
                continue
            selected.append(row)
    with destination.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    if not selected:
        raise ValueError(f"{source}: compact selection produced no rows")
    return len(selected)


def sanitize_string(
    value: str,
    replacements: Sequence[tuple[str, str]],
    home: str,
    hostname: str,
) -> str:
    result = value
    for source, replacement in replacements:
        result = result.replace(source, replacement)
    if home:
        result = result.replace(home, "<HOME>")
    if hostname:
        result = result.replace(hostname, "<HOST>")
    home_pattern = "/" + "home" + r"/[^/\s]+"
    result = re.sub(home_pattern, "<HOME>", result)
    return re.sub(r"https?://[^\s]+", "<URL>", result)


def sanitize_json(
    value: object,
    replacements: Sequence[tuple[str, str]],
    home: str,
    hostname: str,
) -> object:
    if isinstance(value, str):
        return sanitize_string(value, replacements, home, hostname)
    if isinstance(value, list):
        return [sanitize_json(item, replacements, home, hostname) for item in value]
    if isinstance(value, dict):
        return {
            key: sanitize_json(item, replacements, home, hostname)
            for key, item in value.items()
        }
    return value


def git_output(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def source_commit(repo_root: Path, path: Path) -> str | None:
    value = git_output(
        repo_root, "log", "-1", "--format=%H", "--", path.as_posix()
    )
    return value or None


def input_file_record(repo_root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def copy_sanitized_json(
    source: Path,
    destination: Path,
    replacements: Sequence[tuple[str, str]],
    home: str,
    hostname: str,
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    write_json(destination, sanitize_json(payload, replacements, home, hostname))


def copy_run(
    repo_root: Path,
    package_root: Path,
    source_run: Path,
    package_run_rel: Path,
    max_step: int,
    input_paths: dict[str, Path],
) -> dict[str, object]:
    destination = package_root / package_run_rel
    source_metrics = metrics_path_for(source_run)
    replacements = (
        (source_run.resolve().as_posix(), package_run_rel.as_posix()),
        (repo_root.resolve().as_posix() + "/", ""),
        (repo_root.resolve().as_posix(), "."),
    )
    home = Path.home().as_posix()
    hostname = socket.gethostname()
    files: dict[str, object] = {}
    for name in RUN_JSON_FILES:
        source = source_run / name
        source_rel = source.relative_to(repo_root).as_posix()
        input_paths[source_rel] = source
        files[name] = input_file_record(repo_root, source)
        copy_sanitized_json(source, destination / name, replacements, home, hostname)
    metrics_rel = source_metrics.relative_to(repo_root).as_posix()
    input_paths[metrics_rel] = source_metrics
    files[source_metrics.name] = input_file_record(repo_root, source_metrics)
    compact_path = destination / "metrics_selected_10k.csv"
    compact_rows = compact_metrics(source_metrics, compact_path, max_step)
    required_steps = tuple(range(10_000, max_step + 1, 10_000))
    validate_eval_grid(compact_path, required_steps)
    return {
        "seed": seed_for_run(source_run),
        "source_run_dir": source_run.relative_to(repo_root).as_posix(),
        "packaged_run_dir": package_run_rel.as_posix(),
        "source_files": files,
        "source_metrics_file": source_metrics.name,
        "packaged_metrics_file": "metrics_selected_10k.csv",
        "packaged_metrics_rows": compact_rows,
        "evaluation_checkpoint_count": len(required_steps),
    }


def copy_code(repo_root: Path, package_root: Path) -> list[dict[str, object]]:
    source_root = repo_root / EXPERIMENT_REL
    root_files = (
        "analyze_results.py",
        "test_analyze_results.py",
        "select_diversity_threshold.py",
        "test_select_diversity_threshold.py",
        "test_packaged_config_parity.py",
        "run.py",
        "config.json",
        "requirements-numba.txt",
        "package_artifact.py",
    )
    sources = [source_root / name for name in root_files]
    sources.extend(
        sorted(
            path
            for path in (source_root / "code" / "src").rglob("*.py")
            if path.name != "test_b3_runtime_opt_lookup.py"
        )
    )
    records = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = (
            Path(source.name)
            if source.parent == source_root
            else source.relative_to(source_root / "code")
        )
        destination = package_root / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "source_path": source.relative_to(repo_root).as_posix(),
                "packaged_path": (Path("code") / relative).as_posix(),
                "sha256": sha256(source),
            }
        )
    return records


def run_checked(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_result_csv(source: Path, packaged: Path) -> None:
    source_rows = csv_rows(source)
    packaged_rows = csv_rows(packaged)
    if len(source_rows) != len(packaged_rows):
        raise ValueError(f"result row count differs: {source} versus {packaged}")
    ignored = {"run_dir", "metrics_path", "summary_path"}
    for index, (left, right) in enumerate(zip(source_rows, packaged_rows, strict=True)):
        for key in set(left) | set(right):
            if key in ignored:
                continue
            if left.get(key) != right.get(key):
                raise ValueError(
                    f"{source.name} row {index} field {key} differs: "
                    f"{left.get(key)!r} versus {right.get(key)!r}"
                )


def random_count_diagnostics(root: Path) -> dict[str, object]:
    observations = 0
    for metrics_path in sorted(root.glob("*/metrics_selected_10k.csv")):
        for row in csv_rows(metrics_path):
            match = finite_float(row.get("support/random_count_match"))
            if match is None:
                continue
            reference = finite_float(row.get("support/random_count_reference_count"))
            assigned = finite_float(row.get("support/random_count_assigned_count"))
            if not math.isclose(match, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{metrics_path}: random-count match is not one")
            if reference is None or assigned is None or not math.isclose(
                reference, assigned, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{metrics_path}: random-count cardinalities disagree")
            observations += 1
    if observations == 0:
        raise ValueError("no finite Random-Diversity-Count diagnostics found")
    return {
        "finite_observation_count": observations,
        "all_match_flags_one": True,
        "all_reference_and_assigned_counts_equal": True,
    }


def gate_activity_summary(
    diverse_root: Path,
    selected_threshold: Mapping[str, object],
) -> dict[str, object]:
    runs = discover_runs(diverse_root, SEEDS)
    aggregate: dict[str, list[float]] = {
        key: [] for key, _column, _expected in GATE_ACTIVITY_FIELDS
    }
    per_seed: dict[str, dict[str, float]] = {}
    for seed, run_dir in runs.items():
        snapshots: dict[str, dict[int, float]] = {
            key: {} for key, _column, _expected in GATE_ACTIVITY_FIELDS
        }
        metrics_path = metrics_path_for(run_dir)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                step = row_step(row)
                if step not in EVAL_STEPS:
                    continue
                for key, column, _expected in GATE_ACTIVITY_FIELDS:
                    value = finite_float(row.get(column))
                    if value is None:
                        continue
                    previous = snapshots[key].get(step)
                    if previous is not None and not math.isclose(
                        previous, value, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"{metrics_path}: {column} rows disagree at step {step}"
                        )
                    snapshots[key][step] = value

        seed_values: dict[str, float] = {}
        for key, column, _expected in GATE_ACTIVITY_FIELDS:
            missing = [step for step in EVAL_STEPS if step not in snapshots[key]]
            if missing:
                raise ValueError(
                    f"{metrics_path}: {column} missing exact checkpoints {missing}"
                )
            values = [snapshots[key][step] for step in EVAL_STEPS]
            seed_values[key] = float(sum(values) / len(values))
            aggregate[key].extend(values)
        per_seed[str(seed)] = seed_values

    metrics: dict[str, object] = {}
    for key, column, expected_6dp in GATE_ACTIVITY_FIELDS:
        values = aggregate[key]
        mean = float(sum(values) / len(values))
        rounded = round(mean, 6)
        if not math.isclose(rounded, expected_6dp, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{column}: expected six-decimal audit value {expected_6dp}, "
                f"computed {rounded}"
            )
        metrics[key] = {
            "source_column": column,
            "mean": mean,
            "mean_6dp": rounded,
            "per_seed_mean": {
                str(seed): per_seed[str(seed)][key] for seed in SEEDS
            },
        }

    return {
        "schema_version": "1.0",
        "method": "TD3+HER+SABER-Diverse",
        "dmin": 8,
        "grid": list(EVAL_STEPS),
        "checkpoint_count_per_seed": len(EVAL_STEPS),
        "seed_count": len(SEEDS),
        "observation_count": len(SEEDS) * len(EVAL_STEPS),
        "aggregation": (
            "Arithmetic mean across seeds 1-5 and exact 10k checkpoints; "
            "duplicate train/eval rows at one seed-step must agree and count once."
        ),
        "metrics": metrics,
        "threshold_metadata": {
            "target_interval_hit": bool(
                selected_threshold["target_interval_hit"]
            ),
            "fallback_used": bool(selected_threshold["fallback_used"]),
            "selected_dmin": int(selected_threshold["selected_dmin"]),
            "selected_ratio_to_d1": float(
                selected_threshold["selected_ratio_to_d1"]
            ),
            "selected_ratio_to_d1_6dp": round(
                float(selected_threshold["selected_ratio_to_d1"]), 6
            ),
        },
        "interpretation": (
            "D8 retained about 98.4% of main-positive samples on the packaged "
            "exact-10k diagnostic snapshots, so it was a weak additional gate."
        ),
    }


def assert_condition_configs(package_root: Path) -> None:
    expected = {
        "her": (1, "support"),
        "saber": (1, "support"),
        "diverse_d8": (8, "support"),
        "random_count_d8": (8, "random_count"),
    }
    for method, (dmin, assignment) in expected.items():
        runs = discover_runs(package_root / "data" / method, SEEDS)
        for run in runs.values():
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            if int(config.get("support_min_unique_action_bins", 1)) != dmin:
                raise ValueError(f"{run}: unexpected support_min_unique_action_bins")
            if str(config.get("support_assignment_mode", "support")) != assignment:
                raise ValueError(f"{run}: unexpected support_assignment_mode")


def build_readme(
    head: str,
    branch: str,
    selected: Mapping[str, object],
    aggregate: Sequence[Mapping[str, str]],
    bootstrap: Sequence[Mapping[str, str]],
    probe_metrics_rel: str,
    gate_activity: Mapping[str, object],
) -> str:
    aggregate_lines = []
    native_lines = []
    for row in aggregate:
        median_hit = row["hit90_median_reached"] or "not reached"
        aggregate_lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} +/- "
            f"{float(row['auc_std']):.6f} | {float(row['final_success_mean']):.6f} "
            f"+/- {float(row['final_success_std']):.6f} | "
            f"{row['hit90_reached_seed_count']}/{row['n']} | {median_hit} |"
        )
        delta = float(row["native_auc_mean"]) - float(row["auc_mean"])
        native_lines.append(
            f"| {row['method']} | {float(row['native_auc_mean']):.6f} | "
            f"{float(row['auc_mean']):.6f} | {delta:+.6f} |"
        )
    bootstrap_lines = [
        f"| {row['comparison']} | {float(row['mean_auc_difference']):.6f} | "
        f"[{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}] |"
        for row in bootstrap
    ]
    rates = selected["rates"]
    if not isinstance(rates, dict):
        raise TypeError("selected threshold rates must be a mapping")
    gate_metrics = gate_activity["metrics"]
    if not isinstance(gate_metrics, dict):
        raise TypeError("gate activity metrics must be a mapping")
    return f"""# FPP TD3 SABER Action-Diversity D8 Ablation

## Purpose

This artifact evaluates whether SABER's exact-context support predicate benefits
from requiring multiple distinct replay-action bins. The production variant keeps
the legacy policy-bin membership condition and additionally requires `Dmin=8` unique
replay-action bins. Random-Diversity-Count assigns the same number of high weights
uniformly at random within each batch as a cardinality-matched mechanism control.

No training is run by the packaging or analysis commands in this artifact.

## Threshold Selection

The seed-1 Dmin=1 probe ran for 50k steps. Dmin was selected without using success
or AUC, from counterfactual support-positive counters over the exact 20k-50k window.
The pre-registered target was `r_D/r_1` in `[0.4, 0.7]` with `r_D >= 0.03`; when no
candidate entered that interval, the pre-registered nearest-interval fallback chose
the closest candidate, then the closest midpoint, then the smallest Dmin.

| Dmin | Positive rate | Ratio to Dmin=1 |
|---:|---:|---:|
| 1 | {float(rates['1']['positive_rate']):.6f} | 1.000000 |
| 2 | {float(rates['2']['positive_rate']):.6f} | {float(rates['2']['ratio_to_d1']):.6f} |
| 4 | {float(rates['4']['positive_rate']):.6f} | {float(rates['4']['ratio_to_d1']):.6f} |
| 8 | {float(rates['8']['positive_rate']):.6f} | {float(rates['8']['ratio_to_d1']):.6f} |

Selected threshold: `Dmin={selected['selected_dmin']}`. The packaged probe input is
`{probe_metrics_rel}`.

Threshold metadata records `target_interval_hit={str(selected['target_interval_hit']).lower()}`
and `fallback_used={str(selected['fallback_used']).lower()}` with selected ratio
`{float(selected['selected_ratio_to_d1']):.6f}`.

## Gate Activity

The following rates are arithmetic means over seeds 1-5 and the 100 exact 10k
diagnostic checkpoints per seed. Duplicate train/eval rows at one seed-step are
required to agree and are counted once.

| Diagnostic | Mean |
|---|---:|
| Pre-diversity support rate | {float(gate_metrics['pre_diversity_support_rate']['mean']):.6f} |
| D8-positive rate | {float(gate_metrics['d8_positive_rate']['mean']):.6f} |
| D8 pass given main | {float(gate_metrics['d8_pass_given_main']['mean']):.6f} |

D8 therefore retained about 98.4% of main-positive samples on these snapshots;
it was a weak additional gate rather than a highly selective filter. Full values
and per-seed means are in `results/gate_activity_summary.json`.

## Evaluation

Methods are TD3+HER, TD3+HER+SABER (legacy Dmin=1),
TD3+HER+SABER-Diverse (Dmin=8), and Random-Diversity-Count (Dmin=8), each with
the preserved seed mapping 1-5. Evaluation uses exact checkpoints
`10k, 20k, ..., 1M` (100 points). It does not interpolate, smooth, extrapolate, or
forward-fill. Success uses `{PRIMARY_SUCCESS_FIELD}` and falls back to
`{FALLBACK_SUCCESS_FIELD}` only when the primary value is absent or non-finite.

AUC is trapezoidal area divided by `1,000,000 - 10,000 = 990,000`. Hit90 is the
first exact 10k-grid checkpoint with success at least 0.90. Means are arithmetic
means and standard deviations are sample standard deviations (`ddof=1`). Paired
AUC intervals use 10,000 paired percentile bootstrap resamples with RNG seed 27027;
paired seed differences, not methods independently, are resampled.

| Method | AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|
{os.linesep.join(aggregate_lines)}

AUC ranking is TD3+HER+SABER > TD3+HER > Random-Diversity-Count >
TD3+HER+SABER-Diverse.

| Method | Native AUC mean | Recomputed 10k AUC mean | Native - recomputed |
|---|---:|---:|---:|
{os.linesep.join(native_lines)}

| Paired comparison | Mean AUC delta | 95% percentile CI |
|---|---:|---:|
{os.linesep.join(bootstrap_lines)}

These intervals are uncertainty summaries; this artifact makes no additional
statistical-significance claim. Diverse D8 is lower than main SABER in this run set,
while its intervals against HER and the cardinality-matched random control include
zero. The random control therefore does not support attributing a difference to
action diversity alone.

## Contents

- `data/`: sanitized configs, environment snapshots, summaries, and compact exact-10k CSVs.
- `code/`: training implementation snapshot, scanner tests, threshold selector, analyzer, and tests.
- `results/`: recomputed tables, gate activity, paired intervals, report, and validation record.
- `input_manifest.json`: source artifact paths, provenance, sizes, and SHA256 digests.
- `artifact_manifest.json`: packaged-file hashes, coverage, dependencies, and generation metadata.

Checkpoints, full per-update metrics, JSONL logs, and run logs are intentionally
excluded. No PDF or PNG is included because the original analysis produced tabular
outputs only.

## Reproduction

From this artifact directory, with NumPy installed from `requirements.txt`:

```bash
python code/test_analyze_results.py
python code/test_select_diversity_threshold.py
python code/test_packaged_config_parity.py
python code/select_diversity_threshold.py \\
  --metrics {probe_metrics_rel} \\
  --output /tmp/selected_threshold.json
python code/analyze_results.py \\
  --repo-root . \\
  --her-root data/her \\
  --saber-root data/saber \\
  --diverse-root data/diverse_d8 \\
  --random-count-root data/random_count_d8 \\
  --output-dir /tmp/td3_saber_action_diversity_d8
```

Numba scanner tests can be run with the training dependencies in
`code/requirements-numba.txt`:

```bash
python code/src/test_numba_support_scan_fpp.py
python code/src/test_numba_runtime_opt_fpp.py
```

The repository-only `test_run_config_parity.py` is intentionally not packaged
because it references a sibling baseline source tree. The packaged parity test
uses only this artifact's `data/` directories.

## Provenance

- Source repository ID: `ARTIFACT_REPO`
- Source branch: `{branch}`
- Source repository commit at packaging: `{head}`
- Evaluation source commit for tracked HER and main-SABER inputs is recorded per method in `input_manifest.json`.
- Diverse D8, Random-Diversity-Count D8, and probe files were generated from an uncommitted experiment implementation snapshot; their source commit is `null`, with exact source and packaged hashes retained in the manifests.
"""


def package_files(package_root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        relative = path.relative_to(package_root).as_posix()
        if relative == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def content_digest(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_blind_safe(package_root: Path) -> None:
    forbidden = (
        b"/" + b"home" + b"/",
        b"mimic" + b"lab",
        b"Hyunwoo" + b"jooo",
        b"github" + b".com",
        b"http" + b"://",
        b"https" + b"://",
        socket.gethostname().encode("utf-8"),
    )
    violations = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        matched = [
            token.decode("utf-8", errors="replace")
            for token in forbidden
            if token and token in data
        ]
        if matched:
            violations.append((path.relative_to(package_root).as_posix(), matched))
    if violations:
        raise ValueError(f"blind-safety violations: {violations}")


def build_package(repo_root: Path, output: Path) -> None:
    repo_root = repo_root.resolve()
    output = output if output.is_absolute() else repo_root / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; move or remove it explicitly before repackaging"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    head = git_output(repo_root, "rev-parse", "HEAD")
    branch = git_output(repo_root, "branch", "--show-current")
    source_results = repo_root / SOURCE_RESULTS_REL
    for name in (*RESULT_FILES, "report.md"):
        if not (source_results / name).is_file():
            raise FileNotFoundError(source_results / name)

    method_runs = {
        key: discover_runs(repo_root / root_rel, SEEDS)
        for key, _label, root_rel in METHODS
    }
    probe_runs = discover_runs(repo_root / PROBE_ROOT_REL, (1,))
    source_selected_path = repo_root / PROBE_ROOT_REL / "selected_threshold.json"
    if not source_selected_path.is_file():
        raise FileNotFoundError(source_selected_path)

    input_paths: dict[str, Path] = {}
    stage = Path(tempfile.mkdtemp(prefix=".fpp_d8_artifact_", dir=output.parent))
    try:
        code_sources = copy_code(repo_root, stage)
        method_manifest: dict[str, object] = {}
        for key, label, root_rel in METHODS:
            run_records = []
            for seed, run_dir in method_runs[key].items():
                package_run_rel = Path("data") / key / run_dir.name
                record = copy_run(
                    repo_root,
                    stage,
                    run_dir,
                    package_run_rel,
                    1_000_000,
                    input_paths,
                )
                if record["seed"] != seed:
                    raise ValueError(f"{run_dir}: seed mapping changed during packaging")
                run_records.append(record)
            commit = source_commit(repo_root, root_rel)
            method_manifest[key] = {
                "label": label,
                "source_root": root_rel.as_posix(),
                "source_commit": commit,
                "provenance_note": (
                    None
                    if commit is not None
                    else "Run artifacts are uncommitted; exact source SHA256 values are recorded."
                ),
                "seeds": list(SEEDS),
                "runs": run_records,
            }

        probe_run = probe_runs[1]
        probe_run_rel = Path("data") / "probe" / probe_run.name
        probe_record = copy_run(
            repo_root,
            stage,
            probe_run,
            probe_run_rel,
            50_000,
            input_paths,
        )
        selected_rel = source_selected_path.relative_to(repo_root).as_posix()
        input_paths[selected_rel] = source_selected_path
        source_selected_record = input_file_record(repo_root, source_selected_path)
        baseline_hashes = {
            relative: sha256(path) for relative, path in input_paths.items()
        }

        probe_metrics_rel = (probe_run_rel / "metrics_selected_10k.csv").as_posix()
        packaged_selected_rel = Path("data/probe/selected_threshold.json")
        run_checked(
            (
                sys.executable,
                "code/select_diversity_threshold.py",
                "--metrics",
                probe_metrics_rel,
                "--output",
                packaged_selected_rel.as_posix(),
            ),
            stage,
        )
        source_selected = json.loads(source_selected_path.read_text(encoding="utf-8"))
        packaged_selected = json.loads(
            (stage / packaged_selected_rel).read_text(encoding="utf-8")
        )
        for key in (
            "window_start_step",
            "window_end_step",
            "candidate_thresholds",
            "minimum_positive_rate",
            "target_ratio_interval",
            "window_sample_count",
            "d1_positive_count",
            "rates",
            "selected_dmin",
            "selected_ratio_to_d1",
        ):
            if source_selected[key] != packaged_selected[key]:
                raise ValueError(f"threshold selection field changed: {key}")
        if int(packaged_selected["selected_dmin"]) != 8:
            raise ValueError("packaged probe did not select Dmin=8")
        if packaged_selected["target_interval_hit"] is not False:
            raise ValueError("packaged threshold must record target_interval_hit=false")
        if packaged_selected["fallback_used"] is not True:
            raise ValueError("packaged threshold must record fallback_used=true")

        assert_condition_configs(stage)
        random_validation = random_count_diagnostics(stage / "data/random_count_d8")

        results_dir = Path("results")
        gate_activity = gate_activity_summary(
            stage / "data/diverse_d8", packaged_selected
        )
        write_json(
            stage / results_dir / "gate_activity_summary.json", gate_activity
        )
        run_checked(
            (
                sys.executable,
                "code/analyze_results.py",
                "--repo-root",
                ".",
                "--her-root",
                "data/her",
                "--saber-root",
                "data/saber",
                "--diverse-root",
                "data/diverse_d8",
                "--random-count-root",
                "data/random_count_d8",
                "--output-dir",
                results_dir.as_posix(),
                "--bootstrap-resamples",
                str(BOOTSTRAP_RESAMPLES),
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
            ),
            stage,
        )
        for name in RESULT_FILES:
            compare_result_csv(source_results / name, stage / results_dir / name)

        current_hashes = {
            relative: sha256(path) for relative, path in input_paths.items()
        }
        if current_hashes != baseline_hashes:
            raise ValueError("an input artifact changed during packaging")

        validation = {
            "schema_version": "1.0",
            "analysis_matches_existing_results": True,
            "input_artifacts_unchanged": True,
            "threshold_selection_matches_source": True,
            "selected_dmin": 8,
            "evaluation_grid": list(EVAL_STEPS),
            "evaluation_checkpoint_count_per_seed": len(EVAL_STEPS),
            "method_seed_coverage": {key: list(SEEDS) for key, *_ in METHODS},
            "random_count_diagnostics": random_validation,
            "gate_activity_summary": {
                "path": "results/gate_activity_summary.json",
                "pre_diversity_support_rate": gate_activity["metrics"]
                ["pre_diversity_support_rate"]["mean_6dp"],
                "d8_positive_rate": gate_activity["metrics"]
                ["d8_positive_rate"]["mean_6dp"],
                "d8_pass_given_main": gate_activity["metrics"]
                ["d8_pass_given_main"]["mean_6dp"],
            },
            "interpolation_used": False,
            "smoothing_used": False,
            "forward_fill_used": False,
        }
        write_json(stage / results_dir / "validation.json", validation)

        input_manifest = {
            "schema_version": "1.0",
            "repository": {"id": "ARTIFACT_REPO", "branch": branch, "commit": head},
            "input_file_count": len(input_paths),
            "transform": {
                "metrics": "Rows at positive exact multiples of 10,000 only; no value transformation.",
                "json": "Machine-specific paths and URLs sanitized; numeric values unchanged.",
                "checkpoints_copied": False,
                "raw_metrics_copied": False,
            },
            "probe": {
                "source_root": PROBE_ROOT_REL.as_posix(),
                "source_commit": source_commit(repo_root, PROBE_ROOT_REL),
                "provenance_note": "Probe artifacts are uncommitted; exact source SHA256 values are recorded.",
                "run": probe_record,
                "source_selected_threshold": source_selected_record,
                "packaged_selected_threshold": packaged_selected_rel.as_posix(),
            },
            "methods": method_manifest,
            "implementation_source": {
                "repository_parent_commit": head,
                "source_commit": None,
                "provenance_note": "The experiment implementation was uncommitted at packaging time; exact file SHA256 values follow.",
                "files": code_sources,
            },
        }
        write_json(stage / "input_manifest.json", input_manifest)
        (stage / "requirements.txt").write_text(
            f"numpy=={np.__version__}\n", encoding="utf-8"
        )

        aggregate = csv_rows(stage / results_dir / "aggregate_results.csv")
        bootstrap = csv_rows(stage / results_dir / "paired_bootstrap_ci.csv")
        (stage / "README.md").write_text(
            build_readme(
                head,
                branch,
                packaged_selected,
                aggregate,
                bootstrap,
                probe_metrics_rel,
                gate_activity,
            ),
            encoding="utf-8",
        )

        assert_blind_safe(stage)
        packaged_files = package_files(stage)
        artifact_manifest = {
            "schema_version": "1.0",
            "artifact_id": "fpp_td3_saber_action_diversity_d8",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_script": "code/package_artifact.py",
            "repository": {"id": "ARTIFACT_REPO", "branch": branch, "commit": head},
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "evaluation": {
                "steps": list(EVAL_STEPS),
                "checkpoint_count": len(EVAL_STEPS),
                "primary_success_column": PRIMARY_SUCCESS_FIELD,
                "fallback_success_column": FALLBACK_SUCCESS_FIELD,
                "auc_normalization_span": 990_000,
                "hit90_threshold": 0.90,
                "interpolation": False,
                "smoothing": False,
                "forward_fill": False,
            },
            "bootstrap": {
                "method": "paired percentile",
                "confidence_level": 0.95,
                "resamples": BOOTSTRAP_RESAMPLES,
                "rng_seed": BOOTSTRAP_SEED,
            },
            "selected_dmin": 8,
            "threshold_metadata": gate_activity["threshold_metadata"],
            "gate_activity_summary": "results/gate_activity_summary.json",
            "input_file_count": len(input_paths),
            "method_seed_coverage": {key: list(SEEDS) for key, *_ in METHODS},
            "packaged_file_count_excluding_manifest": len(packaged_files),
            "package_content_sha256_excluding_manifest": content_digest(packaged_files),
            "packaged_files": packaged_files,
            "self_hash_note": "artifact_manifest.json is excluded to avoid a recursive self-hash.",
        }
        write_json(stage / "artifact_manifest.json", artifact_manifest)
        assert_blind_safe(stage)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package the completed FPP TD3 SABER action-diversity D8 artifact."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_REL)
    args = parser.parse_args()
    build_package(args.repo_root, args.output_dir)
    print(f"packaged {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
