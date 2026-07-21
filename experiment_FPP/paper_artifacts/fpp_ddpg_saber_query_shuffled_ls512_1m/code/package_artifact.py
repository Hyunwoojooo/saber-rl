#!/usr/bin/env python3
"""Build the compact paper artifact for the FPP DDPG shuffled-query experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import analyze_results as analysis
import source_parity as source_parity_tools


EXPERIMENT_REL = Path("experiment_FPP/ddpg_saber_query_shuffled_ls512")
MAIN_SOURCE_REL = Path("experiment_FPP/b3_critic_only_runtime_opt_v4_ls512")
MAIN_RUNS_REL = Path("experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512")
SHUFFLED_RUNS_REL = Path("experiment_FPP/runs/ddpg_saber_query_shuffled_ls512")
DEFAULT_ARTIFACT_REL = Path(
    "experiment_FPP/paper_artifacts/fpp_ddpg_saber_query_shuffled_ls512_1m"
)
RUN_JSON_FILES = ("config.json", "env_snapshot.json", "summary.json")
SOURCE_SUFFIXES = {".py", ".sh", ".json", ".md", ".txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def source_commit(repo_root: Path, relative_path: Path) -> str | None:
    value = git_output(repo_root, "log", "-1", "--format=%H", "--", relative_path.as_posix())
    return value or None


def sanitize_string(value: str, replacements: Sequence[tuple[str, str]]) -> str:
    result = value
    for source, replacement in replacements:
        result = result.replace(source, replacement)
    result = result.replace(Path.home().as_posix(), "<HOME>")
    result = result.replace(socket.gethostname(), "<HOST>")
    result = re.sub(r"/home/[^/\s]+", "<HOME>", result)
    return result


def sanitize_json(value: object, replacements: Sequence[tuple[str, str]]) -> object:
    if isinstance(value, str):
        return sanitize_string(value, replacements)
    if isinstance(value, list):
        return [sanitize_json(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_json(item, replacements) for key, item in value.items()}
    return value


def file_record(path: Path, relative_to: Path | None = None) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def compact_metrics(source: Path, destination: Path) -> int:
    """Write exactly one primary evaluation row for each exact-grid checkpoint."""

    selected = analysis.evaluation_rows(source)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: missing CSV header")
        fieldnames = list(reader.fieldnames)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for step in analysis.EVAL_STEPS:
            writer.writerow(selected[step])
    compact = analysis.evaluation_rows(destination)
    if len(compact) != 100:
        raise ValueError(f"{destination}: compact metrics did not produce 100 evaluations")
    return len(compact)


def copy_run(
    repo_root: Path,
    artifact_root: Path,
    method_key: str,
    seed: int,
    source_run: Path,
) -> dict[str, object]:
    destination_rel = Path("data") / method_key / f"seed_{seed}"
    destination = artifact_root / destination_rel
    destination.mkdir(parents=True, exist_ok=False)
    replacements = (
        (source_run.resolve().as_posix(), destination_rel.as_posix()),
        (repo_root.resolve().as_posix() + "/", ""),
        (repo_root.resolve().as_posix(), "."),
    )
    source_files: list[dict[str, object]] = []
    for name in RUN_JSON_FILES:
        source = source_run / name
        if not source.is_file():
            raise FileNotFoundError(f"{source_run}: missing {name}")
        packaged = destination / name
        payload = json.loads(source.read_text(encoding="utf-8"))
        write_json(packaged, sanitize_json(payload, replacements))
        source_files.append(
            {
                "source": file_record(source, repo_root),
                "packaged_path": packaged.relative_to(artifact_root).as_posix(),
                "packaged_size_bytes": packaged.stat().st_size,
                "packaged_sha256": sha256(packaged),
            }
        )
    metrics = analysis.metrics_path_for(source_run)
    packaged_metrics = destination / "metrics_selected_10k.csv"
    row_count = compact_metrics(metrics, packaged_metrics)
    source_files.append(
        {
            "source": file_record(metrics, repo_root),
            "packaged_path": packaged_metrics.relative_to(artifact_root).as_posix(),
            "packaged_size_bytes": packaged_metrics.stat().st_size,
            "packaged_sha256": sha256(packaged_metrics),
        }
    )
    return {
        "seed": seed,
        "source_run_dir": source_run.relative_to(repo_root).as_posix(),
        "packaged_run_dir": destination_rel.as_posix(),
        "evaluation_checkpoint_count": row_count,
        "files": source_files,
    }


def source_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(source_root)
        if any(part in {"__pycache__", ".pytest_cache", "smoke", "runs"} for part in relative.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source_root).as_posix())


def frozen_tree_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["packaged_path"])):
        digest.update(
            f"{record['packaged_path']}\0{record['sha256']}\0{record['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def copy_source_snapshot(source_root: Path, artifact_root: Path) -> tuple[list[dict[str, object]], str]:
    records: list[dict[str, object]] = []
    for source in source_files(source_root):
        relative = source.relative_to(source_root)
        destination = artifact_root / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "source_path": relative.as_posix(),
                "packaged_path": (Path("code") / relative).as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        if source.name.startswith("test_") and source.suffix == ".py":
            test_destination = artifact_root / "tests" / relative.name
            test_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, test_destination)
    digest = frozen_tree_digest(records)
    (artifact_root / "frozen_source.sha256").write_text(f"{digest}  code\n", encoding="utf-8")
    return records, digest


def render_readme(analysis_payload: Mapping[str, object], source_sha: str) -> str:
    methods = {str(row["method_key"]): row for row in analysis_payload["methods"]}  # type: ignore[index]
    inference = analysis_payload["paired_inference"]  # type: ignore[index]
    main = methods["main"]
    shuffled = methods["shuffled"]
    return f"""# FPP DDPG SABER Shuffled-Policy Query Ablation

## Purpose

This artifact tests whether sample-aligned current-policy querying is important to
FPP DDPG SABER. In the shuffled arm, a fixed-point-free minibatch derangement
preserves the complete current-policy-action multiset but assigns each action to a
different sample's unchanged context. Only `support_query_mode` changes from
`current_policy` to `shuffled_policy`.

This is **shuffled-policy query**, not shuffled-weight and not random-support.
The shuffled action is used only by support membership. Critic Bellman regression
still uses the sample's replay action, the resulting critic weight stays on that
sample's residual, and the actor remains unweighted DDPG.

## Evaluation protocol

- Seeds: 1–5, paired by seed.
- Exact evaluation grid: 10k, 20k, ..., 1M (100 points per seed).
- Primary success field only: `eval/env_success_mean`.
- No smoothing, interpolation, forward filling, or extrapolation.
- Normalized AUC: trapezoidal integral over 10k–1M divided by 990,000.
- Standard deviations are sample SD (`ddof=1`).
- Paired bootstrap: 10,000 percentile resamples, RNG seed 27027.
- Exact two-sided sign flip: all 32 assignments.

| Arm | AUC mean ± SD | Final mean ± SD | Best mean ± SD | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|---:|
| Main SABER | {float(main['auc_mean']):.6f} ± {float(main['auc_sample_std']):.6f} | {float(main['final_success_mean']):.6f} ± {float(main['final_success_sample_std']):.6f} | {float(main['best_success_mean']):.6f} ± {float(main['best_success_sample_std']):.6f} | {int(main['hit90_reached_seed_count'])}/5 | {main['hit90_median_reached']} |
| Shuffled-policy query | {float(shuffled['auc_mean']):.6f} ± {float(shuffled['auc_sample_std']):.6f} | {float(shuffled['final_success_mean']):.6f} ± {float(shuffled['final_success_sample_std']):.6f} | {float(shuffled['best_success_mean']):.6f} ± {float(shuffled['best_success_sample_std']):.6f} | {int(shuffled['hit90_reached_seed_count'])}/5 | {shuffled['hit90_median_reached']} |

Main − Shuffled paired AUC is {float(inference['mean_auc_difference']):+.6f}
(sample SD {float(inference['sample_sd_auc_difference']):.6f}, 95% paired-bootstrap CI
[{float(inference['bootstrap_ci_lower']):+.6f}, {float(inference['bootstrap_ci_upper']):+.6f}],
exact sign-flip p={float(inference['exact_two_sided_sign_flip_pvalue']):.6f}).
All five seeds and all paired statistics are retained regardless of direction.
If the interval includes zero, this artifact does not claim that an independent
alignment effect has been established.

## Contents and provenance

- `data/main/` and `data/shuffled/`: sanitized config, environment snapshot,
  summary, and one-row-per-checkpoint exact-grid metrics for seeds 1–5.
- `results/`: deterministic analysis tables and validation record.
- `code/`: frozen execution/analysis source snapshot.
- `tests/`: packaged analysis/artifact tests.
- `input_manifest.json`: original input paths, sizes, SHA256, and source commits.
- `artifact_manifest.json`: every packaged file's size and SHA256.
- `frozen_source.sha256`: aggregate SHA256 of the frozen `code/` tree.

Execution source commit: `{source_sha}`.

Checkpoints, full per-update metrics, JSONL, large logs, smoke outputs, caches, and
temporary files are intentionally excluded.

## Verification

From this directory:

```bash
python verify_artifact.py --artifact-dir .
```
"""


def package_file_records(artifact_root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(file_record(path, artifact_root))
    return records


def package_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['path']}\0{record['sha256']}\0{record['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def build_artifact(
    repo_root: Path,
    main_root: Path,
    shuffled_root: Path,
    output_dir: Path,
    execution_source_commit: str | None = None,
    source_root: Path | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    main_root = main_root.resolve()
    shuffled_root = shuffled_root.resolve()
    output_dir = output_dir.resolve()
    source_root = (source_root or (repo_root / EXPERIMENT_REL)).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_dir}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing source root: {source_root}")
    execution_source_commit = execution_source_commit or source_commit(repo_root, EXPERIMENT_REL)
    if not execution_source_commit:
        raise ValueError("execution source has no committed SHA; pass --execution-source-commit")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        source_records, source_digest = copy_source_snapshot(source_root, staging)
        verifier_source = source_root / "verify_artifact.py"
        if not verifier_source.is_file():
            raise FileNotFoundError(verifier_source)
        shutil.copy2(verifier_source, staging / "verify_artifact.py")
        (staging / "requirements.txt").write_text("numpy>=1.24\n", encoding="utf-8")

        input_methods: dict[str, object] = {}
        for method_key, root, expected_mode in (
            ("main", main_root, "current_policy"),
            ("shuffled", shuffled_root, "shuffled_policy"),
        ):
            runs = analysis.discover_runs(root, method_key)
            input_methods[method_key] = {
                "expected_support_query_mode": expected_mode,
                "source_run_root": root.relative_to(repo_root).as_posix(),
                "source_commit": (
                    source_commit(repo_root, MAIN_SOURCE_REL)
                    if method_key == "main"
                    else execution_source_commit
                ),
                "runs": [copy_run(repo_root, staging, method_key, seed, run) for seed, run in runs.items()],
            }

        input_manifest = {
            "schema_version": 1,
            "repository_head_at_packaging": git_output(repo_root, "rev-parse", "HEAD"),
            "execution_source_commit": execution_source_commit,
            "frozen_source_tree_sha256": source_digest,
            "implementation_source": source_records,
            "methods": input_methods,
        }
        write_json(staging / "input_manifest.json", input_manifest)

        analysis_payload = analysis.run_analysis(
            staging / "data" / "main",
            staging / "data" / "shuffled",
            staging / "results",
        )
        source_parity_tools.build_report(
            repo_root / MAIN_SOURCE_REL,
            source_root,
            staging / "results" / "source_parity.json",
        )
        (staging / "README.md").write_text(
            render_readme(analysis_payload, execution_source_commit), encoding="utf-8"
        )

        records = package_file_records(staging)
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_script": "code/package_artifact.py",
            "execution_source_commit": execution_source_commit,
            "method_seed_coverage": {"main": list(analysis.SEEDS), "shuffled": list(analysis.SEEDS)},
            "production_run_count": 10,
            "evaluation_protocol": {
                "primary_success_field": analysis.PRIMARY_SUCCESS_FIELD,
                "steps": list(analysis.EVAL_STEPS),
                "checkpoint_count_per_seed": 100,
                "auc_denominator": 990_000,
            },
            "paired_bootstrap": {"resamples": analysis.BOOTSTRAP_RESAMPLES, "rng_seed": analysis.BOOTSTRAP_SEED},
            "exact_sign_flip": {"assignment_count": 32, "two_sided": True},
            "frozen_source_tree_sha256": source_digest,
            "packaged_file_count_excluding_manifest": len(records),
            "package_content_sha256_excluding_manifest": package_digest(records),
            "packaged_files": records,
        }
        write_json(staging / "artifact_manifest.json", manifest)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--shuffled-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execution-source-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = build_artifact(
        repo_root,
        (args.main_root or (repo_root / MAIN_RUNS_REL)),
        (args.shuffled_root or (repo_root / SHUFFLED_RUNS_REL)),
        (args.output_dir or (repo_root / DEFAULT_ARTIFACT_REL)),
        args.execution_source_commit,
    )
    print(json.dumps({"artifact_dir": str(output), "status": "built"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
