#!/usr/bin/env python3
"""Package completed FPP TD3+HER+PER-style 1M runs for version control."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / "experiment_FPP"
EXPERIMENT_NAME = "td3_her_per_style_loss_weighting_ls512"
METHOD = "td3_her_per_style_loss_weighting"
METHOD_LABEL = "TD3+HER+PER-style"
RUN_PREFIX = "fetch_pick_and_place_td3_her_per_style_loss_weighting_ls512"
ARTIFACT_ROOT = (
    EXPERIMENT_ROOT
    / "paper_artifacts"
    / "fpp_td3_her_per_style_loss_weighting_ls512_1m"
)

SEEDS = (1, 2, 3, 4, 5)
EXPECTED_STEPS = 1_000_000
EXPECTED_UPDATES = 995_001
EXPECTED_REPLAY_SIZE = 300_000
EVAL_FREQUENCY = 10_000
EXPECTED_EVALUATIONS = 100

CONFIG_CONTRACT: dict[str, Any] = {
    "method": METHOD,
    "total_steps": EXPECTED_STEPS,
    "eval_freq": EVAL_FREQUENCY,
    "buffer_capacity": EXPECTED_REPLAY_SIZE,
    "updates_per_step": 1,
    "her_ratio": 0.8,
    "weight_scope": "critic_only",
    "weighting_mode": "batch_norm",
    "per_alpha": 0.6,
    "U_PER_max": 2.0,
}

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

NONFINITE_TOKEN = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?infinity)(?![A-Za-z])", re.I)
FORBIDDEN_SUFFIXES = {".pt", ".jsonl", ".log", ".pyc"}


def fail(message: str) -> NoReturn:
    raise ValueError(message)


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


def load_json_strict(path: Path) -> Any:
    def reject_constant(token: str) -> NoReturn:
        fail(f"{path}: forbidden non-finite JSON constant {token}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def sanitize_string(value: str) -> str:
    sanitized = value.replace(str(REPO_ROOT), "${REPO_ROOT}")
    if sanitized.startswith("/"):
        return f"${{ABSOLUTE_PATH_REDACTED}}/{Path(sanitized).name}"
    return sanitized


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return sanitize_string(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def close(value: Any, expected: float, *, tolerance: float = 1e-12) -> bool:
    parsed = finite_float(value)
    return parsed is not None and math.isclose(
        parsed, expected, rel_tol=0.0, abs_tol=tolerance
    )


def normalized_auc(steps: list[int], values: list[float]) -> float:
    if len(steps) < 2 or len(steps) != len(values):
        fail("AUC requires aligned evaluation arrays with at least two points")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        fail("AUC evaluation steps must be strictly increasing")
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
        fieldnames = reader.fieldnames or []
        missing = [column for column in EVAL_COLUMNS if column not in fieldnames]
        if missing:
            fail(f"{metrics_path}: missing evaluation columns {missing}")
        for row_number, row in enumerate(reader, start=2):
            if finite_float(row.get("eval/goal_success_mean")) is None:
                continue
            compact: dict[str, str] = {}
            for column in EVAL_COLUMNS:
                value = row.get(column)
                if finite_float(value) is None:
                    fail(
                        f"{metrics_path}:{row_number}: non-finite or missing {column}"
                    )
                compact[column] = str(value)
            selected.append(compact)
    return selected


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, str):
        if NONFINITE_TOKEN.fullmatch(value.strip()):
            return ""
        return sanitize_string(value)
    return value


def write_csv(
    path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


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
    files = [EXPERIMENT_ROOT / "package_td3_her_per_style_results.py"]
    source_root = EXPERIMENT_ROOT / EXPERIMENT_NAME
    files.extend(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in FORBIDDEN_SUFFIXES
    )
    return sorted(set(files))


def expected_seed_config(seed: int, run_name: str) -> dict[str, Any]:
    return {
        **CONFIG_CONTRACT,
        "run_name": run_name,
        "seeds": {
            "action_noise_seed": seed,
            "env_seed": seed,
            "eval_seed": 10_000 + seed,
            "numpy_seed": seed,
            "torch_seed": seed,
            "weight_seed": 20_000 + seed,
        },
    }


def check_config(config: dict[str, Any], seed: int, run_name: str) -> dict[str, bool]:
    expected = expected_seed_config(seed, run_name)
    checks: dict[str, bool] = {}
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            checks[f"config_{key}"] = close(actual, value)
        else:
            checks[f"config_{key}"] = actual == value
    return checks


def canonical_hit(value: Any) -> int | str:
    if value in (None, ""):
        return ""
    parsed = finite_float(value)
    if parsed is None or not parsed.is_integer():
        fail(f"invalid first-hit value: {value!r}")
    return int(parsed)


def first_hit(steps: list[int], successes: list[float], threshold: float) -> int | str:
    return next(
        (step for step, success in zip(steps, successes) if success >= threshold),
        "",
    )


def validate_run(
    *,
    seed: int,
    run_name: str,
    config: dict[str, Any],
    env_snapshot: dict[str, Any],
    summary: dict[str, Any],
    evaluations: list[dict[str, str]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    expected_grid = list(range(EVAL_FREQUENCY, EXPECTED_STEPS + 1, EVAL_FREQUENCY))
    steps = [int(float(row["step"])) for row in evaluations]
    successes = [float(row["eval/goal_success_mean"]) for row in evaluations]
    returns = [float(row["eval/return_mean"]) for row in evaluations]
    native_successes = [float(row["eval/success_mean"]) for row in evaluations]
    best_success = max(successes)
    best_step = steps[successes.index(best_success)]
    auc = normalized_auc(steps, successes)
    hit_values = {
        f"first_hit_{threshold:.2f}": first_hit(steps, successes, threshold)
        for threshold in (0.25, 0.50, 0.75, 0.90)
    }

    checks = check_config(config, seed, run_name)
    checks.update(
        {
            "method": summary.get("method") == METHOD,
            "total_steps": summary.get("total_steps") == EXPECTED_STEPS,
            "total_updates": summary.get("total_updates") == EXPECTED_UPDATES,
            "replay_size": summary.get("final_replay_size")
            == EXPECTED_REPLAY_SIZE,
            "evaluation_grid": steps == expected_grid,
            "evaluation_count": len(evaluations) == EXPECTED_EVALUATIONS,
            "native_goal_success_alias": all(
                math.isclose(native, goal, rel_tol=0.0, abs_tol=1e-12)
                for native, goal in zip(native_successes, successes)
            ),
            "final_success": close(summary.get("final_eval_success"), successes[-1]),
            "final_goal_success": close(
                summary.get("final_eval_goal_success"), successes[-1]
            ),
            "best_success": close(summary.get("best_eval_success"), best_success),
            "best_goal_success": close(
                summary.get("best_eval_goal_success"), best_success
            ),
            "best_step": summary.get("best_eval_step") == best_step,
            "native_auc": close(summary.get("eval_AUC_success"), auc),
            "final_return": close(summary.get("final_eval_return"), returns[-1]),
            "selection_metric": summary.get("selection_metric")
            == "eval_goal_success_mean",
            "train_env": env_snapshot.get("train_env", {}).get("env_id")
            == "FetchPickAndPlace-v4",
            "eval_env": env_snapshot.get("eval_env", {}).get("env_id")
            == "FetchPickAndPlace-v4",
        }
    )
    for threshold in (0.25, 0.50, 0.75, 0.90):
        key = f"first_hit_{threshold:.2f}"
        checks[key] = canonical_hit(summary.get(key)) == hit_values[key]

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        fail(f"validation failed for {run_name}: {failed}")

    recomputed = {
        "final_eval_success": successes[-1],
        "best_eval_success": best_success,
        "best_eval_step": best_step,
        "eval_AUC_success": auc,
        "final_eval_return": returns[-1],
        **hit_values,
    }
    return checks, recomputed


def aggregate_rows(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if [int(row["seed"]) for row in per_seed] != list(SEEDS):
        fail("aggregate requires exactly seeds 1-5 in order")
    aggregate: dict[str, Any] = {
        "method": METHOD,
        "label": METHOD_LABEL,
        "seeds": len(per_seed),
    }
    for metric in (
        "final_eval_success",
        "best_eval_success",
        "eval_AUC_success",
        "final_eval_return",
        "runtime_total_sec",
        "final_steps_per_sec",
    ):
        values = [float(row[metric]) for row in per_seed]
        aggregate[f"{metric}_mean"] = statistics.mean(values)
        aggregate[f"{metric}_std"] = statistics.stdev(values)
    for threshold in ("0.25", "0.50", "0.75", "0.90"):
        values = [
            int(row[f"first_hit_{threshold}"])
            for row in per_seed
            if row[f"first_hit_{threshold}"] not in (None, "")
        ]
        aggregate[f"hit{threshold}_count"] = len(values)
        aggregate[f"hit{threshold}_median_step"] = (
            statistics.median(values) if values else ""
        )
    return [aggregate]


def fmt(value: float) -> str:
    return f"{value:.6f}"


def make_report(per_seed: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "# FPP TD3 HER PER-style Loss-Weighting 1M Results",
        "",
        "All five runs completed 1,000,000 environment steps with 995,001 updates",
        "and a final replay size of 300,000. Aggregate values are arithmetic means",
        "and sample standard deviations (ddof=1) across seeds 1-5.",
        "",
        "## Aggregate Results",
        "",
        "| Method | Final success | Best success | Success AUC | Final return | Runtime (h) | Steps/s | Hit90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| {label} | {final_mean} +/- {final_std} | {best_mean} +/- {best_std} | "
        "{auc_mean} +/- {auc_std} | {return_mean} +/- {return_std} | "
        "{runtime_mean} +/- {runtime_std} | {sps_mean} +/- {sps_std} | {hits}/5 |".format(
            label=METHOD_LABEL,
            final_mean=fmt(aggregate["final_eval_success_mean"]),
            final_std=fmt(aggregate["final_eval_success_std"]),
            best_mean=fmt(aggregate["best_eval_success_mean"]),
            best_std=fmt(aggregate["best_eval_success_std"]),
            auc_mean=fmt(aggregate["eval_AUC_success_mean"]),
            auc_std=fmt(aggregate["eval_AUC_success_std"]),
            return_mean=fmt(aggregate["final_eval_return_mean"]),
            return_std=fmt(aggregate["final_eval_return_std"]),
            runtime_mean=fmt(aggregate["runtime_total_sec_mean"] / 3600.0),
            runtime_std=fmt(aggregate["runtime_total_sec_std"] / 3600.0),
            sps_mean=fmt(aggregate["final_steps_per_sec_mean"]),
            sps_std=fmt(aggregate["final_steps_per_sec_std"]),
            hits=aggregate["hit0.90_count"],
        ),
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Final | Best (step) | AUC | Final return | Hit90 | Runtime (h) | Steps/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            "| {seed} | {final:.2f} | {best:.2f} ({best_step}) | {auc:.6f} | "
            "{ret:.2f} | {hit90} | {runtime:.4f} | {sps:.4f} |".format(
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
    lines.extend(
        [
            "",
            "## Evaluation and Packaging",
            "",
            "Evaluation used the native 10,000-step grid from 10,000 through 1,000,000",
            "(100 points). Success AUC is verified by trapezoidal integration on that",
            "exact grid and normalized by the distance from the first to final evaluation.",
            "No interpolation, smoothing, extrapolation, or forward filling is used.",
            "",
            "The fail-closed validation also verifies the exact method, run/seed identity,",
            "1M-step/update/replay counts, HER ratio 0.8, critic-only batch-normalized",
            "PER-style loss weighting, PER alpha 0.6, and U_PER_max 2.0.",
            "",
            "Checkpoints, full per-update metrics, JSONL files, and logs are intentionally",
            "excluded. Packaged JSON replaces repository-local absolute paths with",
            "`${REPO_ROOT}`, redacts any other absolute path, and emits non-finite values",
            "as strict JSON `null`.",
            "",
        ]
    )
    return "\n".join(lines)


def make_readme(aggregate: dict[str, Any]) -> str:
    return f"""# FPP TD3 HER PER-style Loss-Weighting 1M Artifact

This artifact packages completed FetchPickAndPlace-v4 sparse-reward
TD3+HER+PER-style loss-weighting runs for seeds 1-5.

| Method | Final success mean +/- std | Success AUC mean +/- std |
|---|---:|---:|
| {METHOD_LABEL} | {fmt(aggregate['final_eval_success_mean'])} +/- {fmt(aggregate['final_eval_success_std'])} | {fmt(aggregate['eval_AUC_success_mean'])} +/- {fmt(aggregate['eval_AUC_success_std'])} |

## Contents

- `data/`: sanitized config, environment snapshot, summary, and compact
  evaluation-only metrics (100 points) for every seed.
- `results/`: per-seed and aggregate CSVs, report, and machine-readable validation.
- `input_manifest.json`: SHA256 provenance for raw inputs and experiment source.
- `artifact_manifest.json`: SHA256 inventory of the packaged artifact.

The runnable source is versioned in the same repository under
`experiment_FPP/{EXPERIMENT_NAME}`. Regenerate from the repository root with:

```bash
.venv/bin/python experiment_FPP/package_td3_her_per_style_results.py
```

Raw checkpoints, replay state, full metrics, JSONL, and logs are excluded.
See `results/report.md` for tables and evaluation semantics.
"""


def assert_no_absolute_strings(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_absolute_strings(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_absolute_strings(item, context=f"{context}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        fail(f"absolute path survived sanitization at {context}: {value}")


def verify_artifact() -> None:
    expected_data_files = {
        f"data/{METHOD}/seed{seed}/{name}"
        for seed in SEEDS
        for name in ("config.json", "env_snapshot.json", "summary.json", "eval_metrics.csv")
    }
    actual_data_files = {
        path.relative_to(ARTIFACT_ROOT).as_posix()
        for path in (ARTIFACT_ROOT / "data").rglob("*")
        if path.is_file()
    }
    if actual_data_files != expected_data_files:
        fail(
            "packaged data file set mismatch: "
            f"missing={sorted(expected_data_files - actual_data_files)}, "
            f"extra={sorted(actual_data_files - expected_data_files)}"
        )

    for seed in SEEDS:
        eval_path = (
            ARTIFACT_ROOT / "data" / METHOD / f"seed{seed}" / "eval_metrics.csv"
        )
        with eval_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        steps = [int(row["step"]) for row in rows]
        if len(rows) != EXPECTED_EVALUATIONS or steps != list(
            range(EVAL_FREQUENCY, EXPECTED_STEPS + 1, EVAL_FREQUENCY)
        ):
            fail(f"packaged evaluation grid failed for seed {seed}")

    json_files = sorted(ARTIFACT_ROOT.rglob("*.json"))
    for path in json_files:
        value = load_json_strict(path)
        assert_no_absolute_strings(
            value, context=path.relative_to(ARTIFACT_ROOT).as_posix()
        )

    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ARTIFACT_ROOT).as_posix()
        if (
            path.suffix in FORBIDDEN_SUFFIXES
            or path.name == "metrics.csv"
            or "checkpoint" in path.name.lower()
        ):
            fail(f"forbidden raw artifact was packaged: {relative}")
        if path.suffix in {".json", ".md", ".csv"}:
            text = path.read_text(encoding="utf-8")
            if str(REPO_ROOT) in text or NONFINITE_TOKEN.search(text):
                fail(f"unsanitized absolute path or non-finite token in {relative}")

    input_manifest = load_json_strict(ARTIFACT_ROOT / "input_manifest.json")
    for record in input_manifest["inputs"]:
        source = REPO_ROOT / record["path"]
        if not source.is_file():
            fail(f"input manifest source is missing: {record['path']}")
        if source.stat().st_size != record["size_bytes"] or sha256(source) != record["sha256"]:
            fail(f"input manifest digest mismatch: {record['path']}")

    manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    manifest = load_json_strict(manifest_path)
    listed = {record["path"]: record for record in manifest["files"]}
    actual = {
        path.relative_to(ARTIFACT_ROOT).as_posix(): path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(listed) != set(actual):
        fail("artifact manifest file inventory mismatch")
    for relative, path in actual.items():
        record = listed[relative]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            fail(f"artifact manifest digest mismatch: {relative}")

    validation = load_json_strict(ARTIFACT_ROOT / "results" / "validation.json")
    if validation.get("passed") is not True or len(validation.get("runs", [])) != len(SEEDS):
        fail("validation record is not a passing five-run record")


def main() -> None:
    if ARTIFACT_ROOT.exists():
        shutil.rmtree(ARTIFACT_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True)

    raw_input_files: list[Path] = []
    per_seed: list[dict[str, Any]] = []
    validation_runs: list[dict[str, Any]] = []
    run_root = EXPERIMENT_ROOT / "runs" / EXPERIMENT_NAME

    for seed in SEEDS:
        run_name = f"{RUN_PREFIX}_s{seed}_{EXPECTED_STEPS}_ls512"
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
        env_snapshot = load_json(required["env_snapshot"])
        summary = load_json(required["summary"])
        evaluations = extract_evaluations(required["metrics"])
        checks, recomputed = validate_run(
            seed=seed,
            run_name=run_name,
            config=config,
            env_snapshot=env_snapshot,
            summary=summary,
            evaluations=evaluations,
        )

        destination = ARTIFACT_ROOT / "data" / METHOD / f"seed{seed}"
        write_json(destination / "config.json", config)
        write_json(destination / "env_snapshot.json", env_snapshot)
        write_json(destination / "summary.json", summary)
        write_csv(destination / "eval_metrics.csv", list(EVAL_COLUMNS), evaluations)

        result = {column: summary.get(column, "") for column in RESULT_COLUMNS}
        result["method"] = METHOD
        result["seed"] = seed
        per_seed.append(result)
        validation_runs.append(
            {
                "method": METHOD,
                "seed": seed,
                "run_name": run_name,
                "evaluation_points": len(evaluations),
                "recomputed": recomputed,
                "checks": checks,
            }
        )

    aggregate = aggregate_rows(per_seed)
    write_csv(
        ARTIFACT_ROOT / "results" / "per_seed_results.csv",
        list(RESULT_COLUMNS),
        per_seed,
    )
    write_csv(
        ARTIFACT_ROOT / "results" / "aggregate_results.csv",
        list(aggregate[0]),
        aggregate,
    )
    (ARTIFACT_ROOT / "results" / "report.md").write_text(
        make_report(per_seed, aggregate[0]), encoding="utf-8"
    )
    (ARTIFACT_ROOT / "README.md").write_text(
        make_readme(aggregate[0]), encoding="utf-8"
    )

    all_input_files = sorted(set(raw_input_files + discover_source_files()))
    write_json(
        ARTIFACT_ROOT / "input_manifest.json",
        {
            "schema_version": 1,
            "source_head_at_packaging": source_head(),
            "inputs": [
                file_record(path, relative_to=REPO_ROOT) for path in all_input_files
            ],
        },
    )
    write_json(
        ARTIFACT_ROOT / "results" / "validation.json",
        {
            "schema_version": 1,
            "passed": True,
            "method": METHOD,
            "seeds": list(SEEDS),
            "expected_steps": EXPECTED_STEPS,
            "expected_updates": EXPECTED_UPDATES,
            "expected_replay_size": EXPECTED_REPLAY_SIZE,
            "expected_evaluations": EXPECTED_EVALUATIONS,
            "config_contract": CONFIG_CONTRACT,
            "runs": validation_runs,
        },
    )

    packaged_files = sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(
        ARTIFACT_ROOT / "artifact_manifest.json",
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_head_at_packaging": source_head(),
            "coverage": {
                "environment": "FetchPickAndPlace-v4",
                "method": METHOD,
                "seeds": list(SEEDS),
                "steps_per_run": EXPECTED_STEPS,
                "evaluations_per_run": EXPECTED_EVALUATIONS,
                "run_count": len(per_seed),
            },
            "excluded": [
                "*.pt checkpoints and replay state",
                "full metrics.csv",
                "metrics.jsonl",
                "run and launcher logs",
                "__pycache__ and *.pyc",
            ],
            "files": [
                file_record(path, relative_to=ARTIFACT_ROOT)
                for path in packaged_files
            ],
        },
    )

    verify_artifact()
    print(
        f"Packaged and verified {len(per_seed)} runs at "
        f"{ARTIFACT_ROOT.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
