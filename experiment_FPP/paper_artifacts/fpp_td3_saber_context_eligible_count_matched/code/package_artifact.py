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
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 1_000_001, 10_000))
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
FALLBACK_SUCCESS_FIELD = "eval/success_mean"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027
EXPERIMENT_REL = Path("experiment_FPP/td3_saber_context_eligible_count_matched_ls512")
SOURCE_RESULTS_REL = Path("experiment_FPP/reports/td3_saber_context_eligible_count_matched_ls512")
DEFAULT_OUTPUT_REL = Path(
    "experiment_FPP/paper_artifacts/fpp_td3_saber_context_eligible_count_matched"
)
METHODS = (
    (
        "main",
        "TD3+HER+SABER (Main)",
        Path("experiment_FPP/runs/td3_saber_critic_only_v4_ls512"),
        "current_policy",
        "support",
    ),
    (
        "control",
        "Context-eligible count-matched",
        Path("experiment_FPP/runs/td3_saber_context_eligible_count_matched_ls512"),
        "current_policy",
        "context_eligible_count_matched",
    ),
    (
        "context_only",
        "Context-count-only (count-unmatched reference)",
        Path("experiment_FPP/runs/td3_saber_context_count_only_ls512"),
        "context_count_only",
        "support",
    ),
)
RESULT_CSV_FILES = (
    "per_seed_results.csv",
    "aggregate_results.csv",
    "paired_differences.csv",
    "paired_inference.csv",
    "leave_one_seed_out.csv",
    "assignment_diagnostic_summary.csv",
)
RUN_JSON_FILES = ("config.json", "env_snapshot.json", "summary.json")


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
    return primary if primary is not None else finite_float(
        row.get(FALLBACK_SUCCESS_FIELD)
    )


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


def discover_runs(root: Path) -> dict[int, Path]:
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
    if set(found) != set(SEEDS):
        raise ValueError(
            f"{root}: expected seeds {list(SEEDS)}, found {sorted(found)}"
        )
    return {seed: found[seed] for seed in SEEDS}


def validate_eval_grid(metrics_path: Path) -> None:
    values: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = row_step(row)
            success = row_success(row)
            if step is None or success is None:
                continue
            previous = values.get(step)
            if previous is not None and not math.isclose(
                previous, success, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{metrics_path}: evaluation rows disagree at step {step}"
                )
            values[step] = success
    if tuple(sorted(values)) != EVAL_STEPS:
        missing = sorted(set(EVAL_STEPS) - set(values))
        extra = sorted(set(values) - set(EVAL_STEPS))
        raise ValueError(f"{metrics_path}: missing={missing}, extra={extra}")


def compact_metrics(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: missing CSV header")
        fieldnames = list(reader.fieldnames)
        selected = []
        for row in reader:
            step = row_step(row)
            if step in EVAL_STEPS:
                selected.append(row)
    if not selected:
        raise ValueError(f"{source}: compact selection produced no rows")
    with destination.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(
            output_handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)
    validate_eval_grid(destination)
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
    input_paths: dict[str, Path],
    include_diagnostics: bool,
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
    source_files: dict[str, object] = {}
    for name in RUN_JSON_FILES:
        source = source_run / name
        relative = source.relative_to(repo_root).as_posix()
        input_paths[relative] = source
        source_files[name] = input_file_record(repo_root, source)
        copy_sanitized_json(
            source, destination / name, replacements, home, hostname
        )
    metrics_relative = source_metrics.relative_to(repo_root).as_posix()
    input_paths[metrics_relative] = source_metrics
    source_files[source_metrics.name] = input_file_record(repo_root, source_metrics)
    selected_destination = destination / "metrics_selected_10k.csv"
    if source_metrics.name == "metrics_selected_10k.csv":
        shutil.copy2(source_metrics, selected_destination)
        compact_rows = len(csv_rows(selected_destination))
        selected_reference_sha256 = sha256(source_metrics)
    else:
        selected_reference = destination / ".metrics_selected_reference.csv"
        compact_rows = compact_metrics(source_metrics, selected_reference)
        selected_reference_sha256 = sha256(selected_reference)
        shutil.copy2(selected_reference, selected_destination)
        selected_reference.unlink()
    if sha256(selected_destination) != selected_reference_sha256:
        raise ValueError(f"{source_run}: selected metrics changed while packaging")
    diagnostics_copied = False
    if include_diagnostics:
        source_diagnostics = source_run / "assignment_diagnostics_1000.csv"
        if not source_diagnostics.is_file():
            raise FileNotFoundError(source_diagnostics)
        diagnostics_relative = source_diagnostics.relative_to(repo_root).as_posix()
        input_paths[diagnostics_relative] = source_diagnostics
        source_files[source_diagnostics.name] = input_file_record(repo_root, source_diagnostics)
        diagnostics_destination = destination / source_diagnostics.name
        shutil.copy2(source_diagnostics, diagnostics_destination)
        if sha256(source_diagnostics) != sha256(diagnostics_destination):
            raise ValueError(f"{source_run}: diagnostics changed while packaging")
        diagnostics_copied = True
    return {
        "seed": seed_for_run(source_run),
        "source_run_dir": source_run.relative_to(repo_root).as_posix(),
        "packaged_run_dir": package_run_rel.as_posix(),
        "source_files": source_files,
        "source_metrics_file": source_metrics.name,
        "packaged_metrics_file": "metrics_selected_10k.csv",
        "packaged_metrics_rows": compact_rows,
        "selected_metrics_byte_identical": True,
        "selected_metrics_sha256": selected_reference_sha256,
        "assignment_diagnostics_copied": diagnostics_copied,
        "evaluation_checkpoint_count": len(EVAL_STEPS),
    }


def copy_code(repo_root: Path, package_root: Path) -> list[dict[str, object]]:
    source_root = repo_root / EXPERIMENT_REL
    root_files = (
        "analyze_results.py",
        "freeze_experiment.py",
        "freeze_manifest.json",
        "validate_technical_gate.py",
        "run_all_seeds.sh",
        "test_analyze_results.py",
        "test_packaged_config_parity.py",
        "run.py",
        "config.json",
        "requirements-numba.txt",
        "package_artifact.py",
    )
    sources = [source_root / name for name in root_files]
    excluded = {
        "test_b3_runtime_opt_lookup.py",
        "test_current_policy_legacy_parity.py",
        "weight_parity_fixture.py",
    }
    sources.extend(
        sorted(
            path
            for path in (source_root / "code" / "src").rglob("*.py")
            if path.name not in excluded
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


def no_bytecode_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_checked(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(
        list(command), cwd=cwd, env=no_bytecode_env(), check=True
    )

def run_test_record(command: Sequence[str], cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=no_bytecode_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"packaged test failed: {' '.join(command[1:])}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "command": ["python", *command[1:]],
        "passed": True,
        "stdout": result.stdout.strip(),
    }

def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_result_csv(source: Path, packaged: Path) -> None:
    source_rows = csv_rows(source)
    packaged_rows = csv_rows(packaged)
    if len(source_rows) != len(packaged_rows):
        raise ValueError(f"result row count differs: {source} versus {packaged}")
    ignored = {"run_dir", "metrics_path", "summary_path", "diagnostics_path"}
    for index, (left, right) in enumerate(zip(source_rows, packaged_rows, strict=True)):
        for key in set(left) | set(right):
            if key in ignored:
                continue
            if left.get(key) != right.get(key):
                raise ValueError(
                    f"{source.name} row {index} field {key} differs: "
                    f"{left.get(key)!r} versus {right.get(key)!r}"
                )


def build_readme(
    head: str,
    branch: str,
    aggregate: Sequence[Mapping[str, str]],
    inference: Sequence[Mapping[str, str]],
    diagnostics: Sequence[Mapping[str, str]],
) -> str:
    ranking = sorted(aggregate, key=lambda row: float(row["auc_mean"]), reverse=True)
    aggregate_lines = []
    native_lines = []
    for row in aggregate:
        hit90 = (
            "not reached"
            if not row["hit90_median_reached"]
            else f"{float(row['hit90_median_reached']):.0f}"
        )
        aggregate_lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} +/- "
            f"{float(row['auc_std']):.6f} | "
            f"{float(row['final_success_mean']):.6f} +/- "
            f"{float(row['final_success_std']):.6f} | "
            f"{float(row['best_success_mean']):.6f} +/- "
            f"{float(row['best_success_std']):.6f} | "
            f"{row['hit90_reached_seed_count']}/{row['n']} | {hit90} |"
        )
        native_delta = float(row["native_auc_mean"]) - float(row["auc_mean"])
        native_lines.append(
            f"| {row['method']} | {float(row['native_auc_mean']):.6f} | "
            f"{float(row['auc_mean']):.6f} | {native_delta:+.6f} |"
        )
    paired_lines = [
        f"| {row['comparison']} | {float(row['mean_auc_difference']):+.6f} | "
        f"[{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}] | "
        f"{float(row['exact_sign_flip_pvalue_two_sided']):.6f} |"
        for row in inference
    ]
    diagnostic_lines = []
    for row in diagnostics:
        diagnostic_lines.append(
            f"| {row['seed']} | {float(row['context_eligible_rate_mean']):.6f} | "
            f"{float(row['main_positive_rate_mean']):.6f} | "
            f"{float(row['assigned_positive_rate_mean']):.6f} | "
            f"{float(row['overlap_over_k_mean']):.6f} | "
            f"{float(row['jaccard_mean']):.6f} |"
        )
    primary = inference[0]
    ci_lower = float(primary["ci_lower"])
    ci_upper = float(primary["ci_upper"])
    if ci_lower > 0.0:
        interpretation = "The CI lies above zero, supporting added value from membership-aligned assignment in FPP TD3."
    elif ci_lower <= 0.0 <= ci_upper:
        interpretation = "The CI includes zero; an independent membership-assignment contribution is not established."
    else:
        interpretation = "The control is similar or higher; SABER should be interpreted as context-gated, count-preserving critic reweighting here."
    ranking_text = " > ".join(row["method"] for row in ranking)
    return f"""# FPP TD3 SABER Context-Eligible Count-Matched Control

## Purpose

This artifact isolates membership-aligned sample assignment from context-gated,
count-preserving critic reweighting in FetchPickAndPlace-v4 TD3:

- E is the local-count/context-eligible mask immediately before action membership.
- M is E intersected with current-policy action-bin membership (Main SABER).
- k is sum(M).
- Z uniformly samples exactly k indices without replacement from the true indices
  of E (the control assignment).

The control recomputes E, counterfactual M, and k from the current trajectory at
every update. Z uses only E, k, and a dedicated assignment RNG; it does not use
M locations, policy actions, or action bins. Natural overlap between M and Z is
allowed. k=0 yields an all-false Z, k=|E| yields Z=E, and k>|E| fails immediately.

Both Main and control retain support_query_mode=current_policy. Positive count,
context eligibility, and raw/normalized weight histograms are matched per update;
only membership-aligned assignment differs. Context-count-only is included as a
count-unmatched descriptive reference. Random-support is excluded because its
generation provenance is not established here. Packaging and analysis run no
training.

## Evaluation

The package contains Main SABER, the context-eligible count-matched control, and
the count-unmatched context-only reference, each with seeds 1-5. Evaluation uses
the exact grid 10k, 20k, ..., 1M (100 points).
No interpolation, smoothing, extrapolation, or forward filling is used. Success
uses '{PRIMARY_SUCCESS_FIELD}' and falls back to
'{FALLBACK_SUCCESS_FIELD}' only when the primary value is absent or non-finite.

AUC is trapezoidal area divided by 1,000,000 - 10,000 = 990,000. Hit90 is the
first exact-grid checkpoint with success at least 0.90. Means are arithmetic means
and standard deviations are sample standard deviations (ddof=1).

| Method | AUC mean +/- std | Final mean +/- std | Best mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|---:|
{os.linesep.join(aggregate_lines)}

AUC ranking: {ranking_text}.

| Method | Native AUC mean | Recomputed AUC mean | Native - recomputed |
|---|---:|---:|---:|
{os.linesep.join(native_lines)}

## Primary Paired Comparison

Intervals are 95% paired percentile bootstrap CIs from 10,000 resamples using RNG
seed 27027. Paired seed differences are resampled directly. Exact sign-flip
p-values are two-sided over all 32 sign assignments for five paired seeds.

| Comparison | Mean AUC delta | 95% percentile CI | Exact sign-flip p |
|---|---:|---:|---:|
{os.linesep.join(paired_lines)}

The intervals, exact sign-flip values, and leave-one-seed-out summaries are
reported as generated. This artifact makes no additional statistical-significance
claim.

{interpretation}
This is an FPP TD3 result only. It does not establish a universal mechanism or
task-agnostic causality.

## Assignment Diagnostics

Values aggregate non-overlapping 1,000-update windows over each control seed.

| Seed | E rate | Main rate | Assigned rate | Overlap/k | Jaccard |
|---:|---:|---:|---:|---:|---:|
{os.linesep.join(diagnostic_lines)}

## Contents

- 'data/': sanitized configs, environment snapshots, summaries, and compact
  exact-10k metrics for all 15 runs plus control assignment diagnostics.
- 'code/': implementation snapshot, analyzer, unit tests, and standalone package
  parity test.
- 'results/': recomputed result tables, report, and validation record.
- 'input_manifest.json': source paths, provenance, sizes, and SHA256 digests.
- 'artifact_manifest.json': packaged-file hashes, coverage, dependencies, and
  generation metadata.

Checkpoints, full per-update metrics, JSONL files, and logs are intentionally
excluded. No PDF or PNG is included because the analysis is tabular.

## Reproduction

Run from this artifact directory:

~~~bash
python code/test_analyze_results.py
python code/test_packaged_config_parity.py
python code/analyze_results.py \\
  --repo-root . \\
  --main-root data/main \\
  --control-root data/control \\
  --context-only-root data/context_only \\
  --source-parity-evidence code/source_parity_evidence.json \\
  --output-dir /tmp/fpp_td3_saber_context_eligible_count_matched
~~~

Assignment and scanner tests require the dependencies in
'code/requirements-numba.txt':

~~~bash
python code/src/test_context_eligible_count_matched.py
python code/src/test_support_query_modes.py
python code/src/test_numba_support_scan_fpp.py
python code/src/test_numba_runtime_opt_fpp.py
~~~

The current-policy legacy parity test requires the full repository and is not
packaged. Its captured passing evidence is included in
'code/source_parity_evidence.json'.

## Provenance

- Source repository ID: ARTIFACT_REPO
- Source branch: {branch}
- Source repository commit at packaging: {head}
- Per-method source commits and exact input SHA256 values are recorded in
  'input_manifest.json'. Unknown run provenance is recorded as null, not inferred.
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


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for name in ("numba", "llvmlite", "torch", "gymnasium", "gymnasium-robotics"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_package(repo_root: Path, output: Path) -> None:
    repo_root = repo_root.resolve()
    output = output if output.is_absolute() else repo_root / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; move or remove it before repackaging"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    head = git_output(repo_root, "rev-parse", "HEAD")
    branch = git_output(repo_root, "branch", "--show-current")
    source_results = repo_root / SOURCE_RESULTS_REL
    for name in (*RESULT_CSV_FILES, "report.md"):
        if not (source_results / name).is_file():
            raise FileNotFoundError(source_results / name)

    method_runs = {
        key: discover_runs(repo_root / root_rel)
        for key, _label, root_rel, _query_mode, _assignment_mode in METHODS
    }
    input_paths: dict[str, Path] = {}
    stage = Path(tempfile.mkdtemp(prefix=".fpp_count_matched_artifact_", dir=output.parent))
    try:
        code_sources = copy_code(repo_root, stage)
        parity_test = (
            repo_root
            / EXPERIMENT_REL
            / "code/src/test_current_policy_legacy_parity.py"
        )
        parity_env = os.environ.copy()
        parity_env["PARITY_NUMBA"] = "0"
        parity_result = subprocess.run(
            [sys.executable, str(parity_test)],
            cwd=repo_root,
            env=parity_env,
            text=True,
            capture_output=True,
        )
        if parity_result.returncode != 0:
            raise RuntimeError(
                "current-policy legacy parity failed:\n"
                + parity_result.stdout
                + parity_result.stderr
            )
        parity_evidence = {
            "passed": True,
            "test": (EXPERIMENT_REL / "code/src/test_current_policy_legacy_parity.py").as_posix(),
            "stdout": parity_result.stdout.strip(),
            "numba_predicate": False,
        }
        write_json(stage / "code/source_parity_evidence.json", parity_evidence)
        technical_gate_records: dict[str, object] = {}
        for kind, relative in {
            "smoke": Path("experiment_FPP/runs/td3_saber_context_eligible_count_matched_smoke_5k/fetch_pick_and_place_td3_saber_context_eligible_count_matched_smoke_s1_5000_ls512/smoke_validation.json"),
            "gate": Path("experiment_FPP/runs/td3_saber_context_eligible_count_matched_gate_50k/fetch_pick_and_place_td3_saber_context_eligible_count_matched_gate_s1_50000_ls512/gate_validation.json"),
        }.items():
            source = repo_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            gate_payload = json.loads(source.read_text(encoding="utf-8"))
            if gate_payload.get("passed") is not True:
                raise ValueError(f"{source}: technical validation did not pass")
            if gate_payload.get("performance_used_for_gate_decision") is not False:
                raise ValueError(f"{source}: performance was used for gate decision")
            packaged = Path("technical_gate") / f"{kind}_validation.json"
            input_paths[relative.as_posix()] = source
            copy_sanitized_json(
                source,
                stage / packaged,
                ((repo_root.resolve().as_posix(), "."),),
                Path.home().as_posix(),
                socket.gethostname(),
            )
            technical_gate_records[kind] = {
                "source": input_file_record(repo_root, source),
                "packaged_path": packaged.as_posix(),
                "sha256_after_sanitization": sha256(stage / packaged),
            }
        method_manifest: dict[str, object] = {}
        for key, label, root_rel, query_mode, assignment_mode in METHODS:
            run_records = []
            for seed, run_dir in method_runs[key].items():
                package_run_rel = Path("data") / key / run_dir.name
                record = copy_run(
                    repo_root, stage, run_dir, package_run_rel, input_paths,
                    include_diagnostics=(key == "control"),
                )
                if record["seed"] != seed:
                    raise ValueError(f"{run_dir}: seed mapping changed during packaging")
                run_records.append(record)
            commit = source_commit(repo_root, root_rel)
            method_manifest[key] = {
                "label": label,
                "support_query_mode": query_mode,
                "support_assignment_mode": assignment_mode,
                "source_root": root_rel.as_posix(),
                "source_commit": commit,
                "provenance_note": (
                    None
                    if commit is not None
                    else (
                        "Run artifacts are untracked; exact source SHA256 values "
                        "are recorded without inferring a commit."
                    )
                ),
                "seeds": list(SEEDS),
                "runs": run_records,
            }

        baseline_hashes = {
            relative: sha256(path) for relative, path in input_paths.items()
        }
        results_dir = Path("results")
        run_checked(
            (
                sys.executable,
                "code/analyze_results.py",
                "--repo-root",
                ".",
                "--main-root",
                "data/main",
                "--control-root",
                "data/control",
                "--source-parity-evidence",
                "code/source_parity_evidence.json",
                "--context-only-root",
                "data/context_only",
                "--output-dir",
                results_dir.as_posix(),
                "--bootstrap-resamples",
                str(BOOTSTRAP_RESAMPLES),
                "--bootstrap-seed",
                str(BOOTSTRAP_SEED),
            ),
            stage,
        )
        for name in RESULT_CSV_FILES:
            compare_result_csv(source_results / name, stage / results_dir / name)

        test_commands = (
            (sys.executable, "code/test_analyze_results.py"),
            (sys.executable, "code/test_packaged_config_parity.py"),
            (sys.executable, "code/src/test_context_eligible_count_matched.py"),
            (sys.executable, "code/src/test_support_query_modes.py"),
            (sys.executable, "code/src/test_numba_runtime_opt_fpp.py"),
            (sys.executable, "code/src/test_numba_support_scan_fpp.py"),
        )
        test_records = [
            run_test_record(command, stage)
            for command in test_commands
        ]
        write_json(
            stage / "tests/test_results.json",
            {"schema_version": 1, "tests": test_records},
        )
        current_hashes = {
            relative: sha256(path) for relative, path in input_paths.items()
        }
        if current_hashes != baseline_hashes:
            raise ValueError("an input artifact changed during packaging")

        validation_path = stage / results_dir / "validation.json"
        validation = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        validation["packaging"] = {
            "analysis_matches_existing_results": True,
            "compared_result_csv_files": list(RESULT_CSV_FILES),
            "input_artifacts_unchanged": True,
            "standalone_config_parity_passed": True,
            "packaged_test_count": len(test_records),
            "current_policy_legacy_parity_passed_before_packaging": True,
            "selected_metrics_byte_identical_for_all_runs": True,
            "technical_gate_validations": {"smoke": True, "gate": True},
        }
        write_json(validation_path, validation)

        reference_results = [
            input_file_record(repo_root, source_results / name)
            for name in (*RESULT_CSV_FILES, "report.md")
        ]
        input_manifest = {
            "schema_version": "1.0",
            "repository": {"id": "ARTIFACT_REPO", "branch": branch, "commit": head},
            "input_file_count": len(input_paths),
            "transform": {
                "metrics": (
                    "Rows at exact positive multiples of 10,000 through 1M only; "
                    "no numeric value transformation."
                ),
                "json": (
                    "Machine-specific paths and URLs sanitized; numeric values "
                    "unchanged."
                ),
                "checkpoints_copied": False,
                "raw_metrics_copied": False,
                "jsonl_copied": False,
                "logs_copied": False,
            },
            "methods": method_manifest,
            "technical_gate": technical_gate_records,
            "source_parity_evidence": parity_evidence,
            "reference_results": reference_results,
            "implementation_source": {
                "repository_parent_commit": head,
                "source_commit": source_commit(repo_root, EXPERIMENT_REL),
                "files": code_sources,
            },
        }
        write_json(stage / "input_manifest.json", input_manifest)
        (stage / "requirements.txt").write_text(
            f"numpy=={np.__version__}\n", encoding="utf-8"
        )

        aggregate = csv_rows(stage / results_dir / "aggregate_results.csv")
        inference = csv_rows(stage / results_dir / "paired_inference.csv")
        diagnostics = csv_rows(stage / results_dir / "assignment_diagnostic_summary.csv")
        (stage / "README.md").write_text(
            build_readme(head, branch, aggregate, inference, diagnostics),
            encoding="utf-8",
        )

        assert_blind_safe(stage)
        packaged_files = package_files(stage)
        artifact_manifest = {
            "schema_version": "1.0",
            "artifact_id": "fpp_td3_saber_context_eligible_count_matched",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_script": "code/package_artifact.py",
            "repository": {"id": "ARTIFACT_REPO", "branch": branch, "commit": head},
            "runtime": dependency_versions(),
            "evaluation": {
                "steps": list(EVAL_STEPS),
                "checkpoint_count": len(EVAL_STEPS),
                "primary_success_column": PRIMARY_SUCCESS_FIELD,
                "fallback_success_column": FALLBACK_SUCCESS_FIELD,
                "auc_normalization_span": 990_000,
                "hit90_threshold": 0.90,
                "interpolation": False,
                "smoothing": False,
                "extrapolation": False,
                "forward_fill": False,
            },
            "assignment": {
                "support_query_mode": "current_policy",
                "control_mode": "context_eligible_count_matched",
                "diagnostic_window_updates": 1_000,
                "dedicated_seed_offset": 30_000,
                "exact_k_per_update": True,
                "raw_histogram_match_per_update": True,
                "normalized_histogram_match_per_update": True,
                "selected_metrics_byte_identical": True,
            },
            "technical_gate": {
                "smoke_5k_passed": True,
                "gate_50k_passed": True,
            },
            "tests": {
                "count": len(test_records),
                "all_passed": all(record["passed"] for record in test_records),
            },
            "bootstrap": {
                "method": "paired percentile",
                "confidence_level": 0.95,
                "resamples": BOOTSTRAP_RESAMPLES,
                "rng_seed": BOOTSTRAP_SEED,
            },
            "exact_sign_flip": {
                "two_sided": True,
                "assignment_count": 32,
            },
            "input_file_count": len(input_paths),
            "method_seed_coverage": {
                key: list(SEEDS) for key, *_rest in METHODS
            },
            "packaged_file_count_excluding_manifest": len(packaged_files),
            "package_content_sha256_excluding_manifest": content_digest(
                packaged_files
            ),
            "packaged_files": packaged_files,
            "self_hash_note": (
                "artifact_manifest.json is excluded to avoid a recursive self-hash."
            ),
        }
        write_json(stage / "artifact_manifest.json", artifact_manifest)
        assert_blind_safe(stage)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package the FPP TD3 SABER context-eligible count-matched control."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_REL)
    args = parser.parse_args()
    build_package(args.repo_root, args.output_dir)
    print(f"packaged {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
