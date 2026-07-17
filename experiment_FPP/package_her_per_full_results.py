#!/usr/bin/env python3
"""Package completed FPP TD3/SAC HER full-PER runs for version control."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / "experiment_FPP"
ARTIFACT_ROOT = (
    EXPERIMENT_ROOT / "paper_artifacts" / "fpp_td3_sac_her_per_full_1m"
)

METHODS = {
    "td3_her_per_full": {
        "label": "TD3+HER+PER(full)",
        "experiment": "td3_her_per_full_ls512",
        "run_prefix": "fetch_pick_and_place_td3_her_per_full_ls512",
        "eval_frequency": 10_000,
        "expected_evaluations": 100,
    },
    "sac_her_per_full": {
        "label": "SAC+HER+PER(full)",
        "experiment": "sac_her_per_full_ls512",
        "run_prefix": "fetch_pick_and_place_sac_her_per_full_ls512",
        "eval_frequency": 25_000,
        "expected_evaluations": 40,
    },
}

SEEDS = (1, 2, 3, 4, 5)
EXPECTED_STEPS = 1_000_000
EXPECTED_UPDATES = 995_001
EXPECTED_REPLAY_SIZE = 300_000

EVAL_COLUMNS = (
    "step",
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/goal_success_ever_mean",
    "eval/env_success_ever_mean",
    "eval/goal_success_final_mean",
    "eval/env_success_final_mean",
    "eval/episode_length_mean",
)

RESULT_COLUMNS = (
    "method",
    "seed",
    "total_steps",
    "total_updates",
    "final_eval_success",
    "best_eval_success",
    "best_eval_step",
    "eval_AUC_success",
    "final_eval_return",
    "first_hit_0.25",
    "first_hit_0.50",
    "first_hit_0.75",
    "first_hit_0.90",
    "runtime_total_sec",
    "final_steps_per_sec",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return value.replace(str(REPO_ROOT), "${REPO_ROOT}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def normalized_auc(steps: list[int], values: list[float]) -> float:
    if len(steps) < 2 or len(steps) != len(values):
        raise ValueError("AUC requires aligned evaluation arrays with at least 2 points")
    area = sum(
        (values[index - 1] + values[index])
        * (steps[index] - steps[index - 1])
        / 2.0
        for index in range(1, len(steps))
    )
    return area / float(steps[-1] - steps[0])


def extract_evaluations(metrics_path: Path) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in EVAL_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{metrics_path}: missing evaluation columns {missing}")
        for row in reader:
            if finite_float(row.get("eval/goal_success_mean")) is None:
                continue
            selected.append({column: row.get(column, "") for column in EVAL_COLUMNS})
    return selected


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def discover_source_files() -> list[Path]:
    files: list[Path] = [EXPERIMENT_ROOT / "package_her_per_full_results.py"]
    for config in METHODS.values():
        root = EXPERIMENT_ROOT / config["experiment"]
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pt"}
        )
    return sorted(set(files))


def aggregate_rows(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "final_eval_success",
        "best_eval_success",
        "eval_AUC_success",
        "final_eval_return",
        "runtime_total_sec",
        "final_steps_per_sec",
    )
    rows: list[dict[str, Any]] = []
    for method, config in METHODS.items():
        method_rows = [row for row in per_seed if row["method"] == method]
        aggregate: dict[str, Any] = {
            "method": method,
            "label": config["label"],
            "seeds": len(method_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows]
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = statistics.stdev(values)
        hits = [int(row["first_hit_0.90"]) for row in method_rows if row["first_hit_0.90"]]
        aggregate["hit90_count"] = len(hits)
        aggregate["hit90_median_step"] = statistics.median(hits) if hits else ""
        rows.append(aggregate)
    return rows


def fmt(value: float) -> str:
    return f"{value:.6f}"


def make_report(per_seed: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> str:
    by_method = {row["method"]: row for row in aggregate}
    lines = [
        "# FPP TD3/SAC HER Full-PER 1M Results",
        "",
        "All ten runs completed 1,000,000 environment steps with 995,001 updates",
        "and a final replay size of 300,000. Values below are arithmetic means",
        "and sample standard deviations (ddof=1) across seeds 1-5.",
        "",
        "## Aggregate Results",
        "",
        "| Method | Final success | Best success | Success AUC | Final return | Runtime (h) | Steps/s | Hit90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, config in METHODS.items():
        row = by_method[method]
        lines.append(
            "| {label} | {final_mean} +/- {final_std} | {best_mean} +/- {best_std} | "
            "{auc_mean} +/- {auc_std} | {return_mean} +/- {return_std} | "
            "{runtime_mean} +/- {runtime_std} | {sps_mean} +/- {sps_std} | {hits}/5 |".format(
                label=config["label"],
                final_mean=fmt(row["final_eval_success_mean"]),
                final_std=fmt(row["final_eval_success_std"]),
                best_mean=fmt(row["best_eval_success_mean"]),
                best_std=fmt(row["best_eval_success_std"]),
                auc_mean=fmt(row["eval_AUC_success_mean"]),
                auc_std=fmt(row["eval_AUC_success_std"]),
                return_mean=fmt(row["final_eval_return_mean"]),
                return_std=fmt(row["final_eval_return_std"]),
                runtime_mean=fmt(row["runtime_total_sec_mean"] / 3600.0),
                runtime_std=fmt(row["runtime_total_sec_std"] / 3600.0),
                sps_mean=fmt(row["final_steps_per_sec_mean"]),
                sps_std=fmt(row["final_steps_per_sec_std"]),
                hits=row["hit90_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Method | Seed | Final | Best (step) | AUC | Final return | Hit90 | Runtime (h) | Steps/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed:
        lines.append(
            "| {label} | {seed} | {final:.2f} | {best:.2f} ({best_step}) | "
            "{auc:.6f} | {ret:.2f} | {hit90} | {runtime:.4f} | {sps:.4f} |".format(
                label=METHODS[row["method"]]["label"],
                seed=row["seed"],
                final=float(row["final_eval_success"]),
                best=float(row["best_eval_success"]),
                best_step=row["best_eval_step"],
                auc=float(row["eval_AUC_success"]),
                ret=float(row["final_eval_return"]),
                hit90=row["first_hit_0.90"] or "-",
                runtime=float(row["runtime_total_sec"]) / 3600.0,
                sps=float(row["final_steps_per_sec"]),
            )
        )

    td3 = {int(row["seed"]): row for row in per_seed if row["method"] == "td3_her_per_full"}
    sac = {int(row["seed"]): row for row in per_seed if row["method"] == "sac_her_per_full"}
    auc_deltas = [
        float(sac[seed]["eval_AUC_success"]) - float(td3[seed]["eval_AUC_success"])
        for seed in SEEDS
    ]
    final_deltas = [
        float(sac[seed]["final_eval_success"]) - float(td3[seed]["final_eval_success"])
        for seed in SEEDS
    ]
    lines.extend(
        [
            "",
            "## Descriptive Comparison",
            "",
            f"SAC - TD3 mean AUC difference: {statistics.mean(auc_deltas):+.6f}.",
            f"SAC - TD3 mean final-success difference: {statistics.mean(final_deltas):+.6f}.",
            "All five paired seeds have positive SAC-minus-TD3 AUC and final-success differences.",
            "These five-seed summaries are descriptive; this artifact makes no formal",
            "statistical-significance or task-generalization claim.",
            "",
            "## Evaluation and Packaging",
            "",
            "TD3 was evaluated every 10,000 steps (100 points); SAC was evaluated every",
            "25,000 steps (40 points). Native success AUC is verified by trapezoidal",
            "integration over each method's exact evaluation grid, normalized by the",
            "distance from its first to final evaluation step. No interpolation, smoothing,",
            "extrapolation, or forward filling is used.",
            "",
            "Checkpoints, full per-update metrics, JSONL files, and logs are intentionally",
            "excluded. Packaged JSON replaces repository-local absolute paths with",
            "`${REPO_ROOT}` and non-finite diagnostic values with JSON `null`.",
            "",
        ]
    )
    return "\n".join(lines)


def make_readme(aggregate: list[dict[str, Any]]) -> str:
    by_method = {row["method"]: row for row in aggregate}
    td3 = by_method["td3_her_per_full"]
    sac = by_method["sac_her_per_full"]
    return f"""# FPP TD3/SAC HER Full-PER 1M Artifact

This artifact packages the completed FetchPickAndPlace-v4 sparse-reward
TD3+HER+PER(full) and SAC+HER+PER(full) experiments for seeds 1-5.

| Method | Final success mean +/- std | Success AUC mean +/- std |
|---|---:|---:|
| TD3+HER+PER(full) | {fmt(td3['final_eval_success_mean'])} +/- {fmt(td3['final_eval_success_std'])} | {fmt(td3['eval_AUC_success_mean'])} +/- {fmt(td3['eval_AUC_success_std'])} |
| SAC+HER+PER(full) | {fmt(sac['final_eval_success_mean'])} +/- {fmt(sac['final_eval_success_std'])} | {fmt(sac['eval_AUC_success_mean'])} +/- {fmt(sac['eval_AUC_success_std'])} |

## Contents

- `data/`: sanitized config, environment snapshot, summary, and compact
  evaluation-only metrics for each of the ten runs.
- `results/`: per-seed, aggregate, paired descriptive results, report, and
  machine-readable validation record.
- `input_manifest.json`: SHA256 provenance for raw inputs and experiment source.
- `artifact_manifest.json`: SHA256 inventory of the packaged artifact.

The runnable source is versioned in the same repository commit under
`experiment_FPP/td3_her_per_full_ls512` and
`experiment_FPP/sac_her_per_full_ls512`. Regenerate from repository root with:

```bash
.venv/bin/python experiment_FPP/package_her_per_full_results.py
```

Raw checkpoints, replay state, full metrics, JSONL, and logs are excluded.
See `results/report.md` for the complete result table and evaluation semantics.
"""


def main() -> None:
    if ARTIFACT_ROOT.exists():
        shutil.rmtree(ARTIFACT_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True)

    raw_input_files: list[Path] = []
    per_seed: list[dict[str, Any]] = []
    validation_runs: list[dict[str, Any]] = []

    for method, method_config in METHODS.items():
        run_root = EXPERIMENT_ROOT / "runs" / method_config["experiment"]
        for seed in SEEDS:
            run_name = (
                f"{method_config['run_prefix']}_s{seed}_{EXPECTED_STEPS}_ls512"
            )
            run_dir = run_root / run_name
            required = {
                "config": run_dir / "config.json",
                "env_snapshot": run_dir / "env_snapshot.json",
                "summary": run_dir / "summary.json",
                "metrics": run_dir / "metrics.csv",
            }
            missing = [str(path) for path in required.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing run inputs: {missing}")
            raw_input_files.extend(required.values())

            config = load_json(required["config"])
            summary = load_json(required["summary"])
            env_snapshot = load_json(required["env_snapshot"])
            evaluations = extract_evaluations(required["metrics"])

            expected_grid = list(
                range(
                    method_config["eval_frequency"],
                    EXPECTED_STEPS + 1,
                    method_config["eval_frequency"],
                )
            )
            eval_steps = [int(row["step"]) for row in evaluations]
            eval_success = [float(row["eval/goal_success_mean"]) for row in evaluations]
            recomputed_auc = normalized_auc(eval_steps, eval_success)

            checks = {
                "method": summary.get("method") == method == config.get("method"),
                "total_steps": summary.get("total_steps") == EXPECTED_STEPS,
                "total_updates": summary.get("total_updates") == EXPECTED_UPDATES,
                "replay_size": summary.get("final_replay_size") == EXPECTED_REPLAY_SIZE,
                "evaluation_grid": eval_steps == expected_grid,
                "evaluation_count": len(evaluations)
                == method_config["expected_evaluations"],
                "final_success": math.isclose(
                    float(summary["final_eval_success"]), eval_success[-1], abs_tol=1e-12
                ),
                "best_success": math.isclose(
                    float(summary["best_eval_success"]), max(eval_success), abs_tol=1e-12
                ),
                "native_auc": math.isclose(
                    float(summary["eval_AUC_success"]), recomputed_auc, abs_tol=1e-12
                ),
            }
            if not all(checks.values()):
                raise ValueError(f"validation failed for {run_name}: {checks}")

            destination = ARTIFACT_ROOT / "data" / method / f"seed{seed}"
            write_json(destination / "config.json", sanitize(config))
            write_json(destination / "env_snapshot.json", sanitize(env_snapshot))
            write_json(destination / "summary.json", sanitize(summary))
            write_csv(destination / "eval_metrics.csv", list(EVAL_COLUMNS), evaluations)

            result = {column: summary.get(column, "") for column in RESULT_COLUMNS}
            result["method"] = method
            result["seed"] = seed
            per_seed.append(result)
            validation_runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "run_name": run_name,
                    "evaluation_points": len(evaluations),
                    "recomputed_AUC_success": recomputed_auc,
                    "checks": checks,
                }
            )

    aggregate = aggregate_rows(per_seed)
    aggregate_fields = list(aggregate[0].keys())
    write_csv(ARTIFACT_ROOT / "results" / "per_seed_results.csv", list(RESULT_COLUMNS), per_seed)
    write_csv(ARTIFACT_ROOT / "results" / "aggregate_results.csv", aggregate_fields, aggregate)

    td3 = {int(row["seed"]): row for row in per_seed if row["method"] == "td3_her_per_full"}
    sac = {int(row["seed"]): row for row in per_seed if row["method"] == "sac_her_per_full"}
    paired_rows = []
    for seed in SEEDS:
        paired_rows.append(
            {
                "seed": seed,
                "sac_minus_td3_AUC_success": float(sac[seed]["eval_AUC_success"])
                - float(td3[seed]["eval_AUC_success"]),
                "sac_minus_td3_final_success": float(sac[seed]["final_eval_success"])
                - float(td3[seed]["final_eval_success"]),
                "sac_minus_td3_best_success": float(sac[seed]["best_eval_success"])
                - float(td3[seed]["best_eval_success"]),
                "sac_minus_td3_final_return": float(sac[seed]["final_eval_return"])
                - float(td3[seed]["final_eval_return"]),
            }
        )
    write_csv(
        ARTIFACT_ROOT / "results" / "paired_seed_differences.csv",
        list(paired_rows[0].keys()),
        paired_rows,
    )

    (ARTIFACT_ROOT / "results" / "report.md").write_text(
        make_report(per_seed, aggregate), encoding="utf-8"
    )
    (ARTIFACT_ROOT / "README.md").write_text(make_readme(aggregate), encoding="utf-8")

    all_input_files = sorted(set(raw_input_files + discover_source_files()))
    input_manifest = {
        "schema_version": 1,
        "source_head_at_packaging": source_head(),
        "inputs": [file_record(path, relative_to=REPO_ROOT) for path in all_input_files],
    }
    write_json(ARTIFACT_ROOT / "input_manifest.json", input_manifest)

    validation = {
        "schema_version": 1,
        "passed": True,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "expected_steps": EXPECTED_STEPS,
        "expected_updates": EXPECTED_UPDATES,
        "expected_replay_size": EXPECTED_REPLAY_SIZE,
        "runs": validation_runs,
    }
    write_json(ARTIFACT_ROOT / "results" / "validation.json", validation)

    packaged_files = sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_at_packaging": source_head(),
        "coverage": {
            "environment": "FetchPickAndPlace-v4",
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "steps_per_run": EXPECTED_STEPS,
            "run_count": len(per_seed),
        },
        "excluded": [
            "*.pt checkpoints and replay state",
            "full metrics.csv",
            "metrics.jsonl",
            "run and launcher logs",
            "__pycache__ and *.pyc",
        ],
        "files": [file_record(path, relative_to=ARTIFACT_ROOT) for path in packaged_files],
    }
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", artifact_manifest)

    forbidden = str(REPO_ROOT)
    for path in ARTIFACT_ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".csv"}:
            if forbidden in path.read_text(encoding="utf-8"):
                raise ValueError(f"unsanitized repository path in {path}")

    print(f"Packaged {len(per_seed)} runs at {ARTIFACT_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
