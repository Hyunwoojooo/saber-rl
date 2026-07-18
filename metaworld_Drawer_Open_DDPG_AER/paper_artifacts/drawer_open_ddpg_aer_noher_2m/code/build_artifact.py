#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ARTIFACT_NAME = "drawer_open_ddpg_aer_noher_2m"
TOTAL_STEPS = 2_000_000
EVAL_FREQ = 10_000
EVAL_GRID = tuple(range(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ))
LAST_QUARTER_START = 1_510_000
SEEDS = (1, 2, 3, 4, 5)

SCRIPT_PATH = Path(__file__).resolve()
ARTIFACT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "metaworld_Drawer_Open_DDPG_AER"


@dataclass(frozen=True)
class Arm:
    key: str
    label: str

    @property
    def run_root(self) -> Path:
        return EXPERIMENT_ROOT / "runs" / self.key

    def run_name(self, seed: int) -> str:
        return f"drawer_open_{self.key}_s{seed}_{TOTAL_STEPS}"


ARMS = (
    Arm("ddpg_er", "DDPG-ER"),
    Arm("ddpg_annealing_er", "DDPG-AnnealingER"),
    Arm("ddpg_aer", "DDPG-AER"),
)

SELECTED_METRIC_FIELDS = (
    "step",
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/final_success",
    "eval/best_success",
    "eval/AUC_success",
    "eval/first_hit_0.25",
    "eval/first_hit_0.50",
    "eval/first_hit_0.75",
    "eval/first_hit_0.90",
    "aer/lambda",
    "aer/candidate_count",
    "aer/candidate_similarity_mean",
    "aer/selected_similarity_mean",
    "aer/selected_similarity_uplift",
    "aer/candidate_duplicate_rate",
    "aer/selected_duplicate_rate",
    "aer/candidate_transition_age_mean",
    "aer/selected_transition_age_mean",
    "aer/candidate_success_transition_rate",
    "aer/selected_success_transition_rate",
    "aer/buffer_success_transition_rate",
    "aer/effective_actor_lr",
    "aer/effective_critic_lr",
    "runtime/aer_sampling_sec_per_update",
    "aer/selected_phase_approach_rate",
    "aer/selected_phase_contact_rate",
    "aer/selected_phase_opening_rate",
    "aer/selected_phase_open_rate",
)

COMMON_CONFIG = {
    "task_name": "drawer_open",
    "reward_type": "sparse",
    "metaworld_env_name": "drawer-open-v3",
    "metaworld_reward_function_version": "v2",
    "max_episode_steps": 200,
    "total_steps": TOTAL_STEPS,
    "batch_size": 256,
    "buffer_capacity": 300_000,
    "her_ratio": 0.0,
    "updates_per_step": 1,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "policy_delay": 1,
    "target_policy_noise": 0.0,
    "target_noise_clip": 0.0,
    "eval_freq": EVAL_FREQ,
    "eval_episodes": 20,
    "final_eval_episodes": 100,
    "aer_lambda0": 2.5,
    "aer_alpha": 2.0 / 3.0,
}


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number for {context}: {value!r}")
    return result


def optional_number(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{context}: recomputed={actual}, recorded={expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str):
        repo = str(REPO_ROOT)
        if value == repo:
            return "."
        if value.startswith(repo + "/"):
            return value[len(repo) + 1 :]
    return value


def validate_config(config: dict[str, object], arm: Arm, seed: int, run_name: str) -> None:
    for field, expected in COMMON_CONFIG.items():
        actual = config.get(field)
        if isinstance(expected, float):
            close(number(actual, f"{run_name}.{field}"), expected, f"{run_name}.{field}")
        elif actual != expected:
            raise ValueError(f"{run_name}.{field}: {actual!r} != {expected!r}")
    if config.get("replay_method") != arm.key:
        raise ValueError(f"Replay method mismatch for {run_name}")
    if config.get("run_name") != run_name:
        raise ValueError(f"Run-name mismatch for {run_name}")
    seeds = config.get("seeds")
    if not isinstance(seeds, dict) or int(number(seeds.get("env_seed"), "env_seed")) != seed:
        raise ValueError(f"Seed mismatch for {run_name}")


def read_evaluations(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        fieldnames = list(reader.fieldnames)
        missing = [field for field in SELECTED_METRIC_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"Missing fields in {path}: {missing}")
        by_step: dict[int, dict[str, str]] = {}
        for row_index, row in enumerate(reader, start=2):
            success = optional_number(row.get("eval/env_success_mean"))
            if not math.isfinite(success):
                continue
            step = int(number(row.get("step"), f"{path}:{row_index}:step"))
            if step not in EVAL_GRID:
                raise ValueError(f"Evaluation outside exact grid at {step}: {path}")
            if step in by_step:
                raise ValueError(f"Duplicate evaluation at {step}: {path}")
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"Success outside [0, 1] at {step}: {path}")
            by_step[step] = row
    missing_steps = [step for step in EVAL_GRID if step not in by_step]
    if missing_steps:
        raise ValueError(f"Missing evaluation steps in {path}: {missing_steps}")
    return fieldnames, [by_step[step] for step in EVAL_GRID]


def trapezoid(steps: Sequence[int], values: Sequence[float]) -> float:
    if len(steps) < 2 or len(steps) != len(values):
        raise ValueError("Trapezoidal AUC requires aligned sequences")
    area = sum(
        0.5 * (values[index] + values[index + 1]) * (steps[index + 1] - steps[index])
        for index in range(len(steps) - 1)
    )
    return area / float(steps[-1] - steps[0])


def first_hit(steps: Sequence[int], values: Sequence[float], threshold: float = 0.90) -> int | None:
    return next((step for step, value in zip(steps, values) if value >= threshold), None)


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_bootstrap(deltas: Sequence[float]) -> tuple[float, float, int]:
    n = len(deltas)
    means = [
        statistics.mean(deltas[index] for index in indices)
        for indices in itertools.product(range(n), repeat=n)
    ]
    return quantile(means, 0.025), quantile(means, 0.975), len(means)


def exact_sign_flip_p(deltas: Sequence[float]) -> tuple[float, int]:
    nonzero = [delta for delta in deltas if not math.isclose(delta, 0.0, abs_tol=1e-15)]
    if not nonzero:
        return 1.0, 1
    observed = abs(statistics.mean(nonzero))
    distribution = [
        abs(statistics.mean(sign * value for sign, value in zip(signs, nonzero)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero))
    ]
    extreme = sum(value >= observed - 1e-15 for value in distribution)
    return extreme / len(distribution), len(distribution)


def fmt(value: object) -> object:
    return format(value, ".17g") if isinstance(value, float) else value


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def load_run(arm: Arm, seed: int) -> dict[str, object]:
    run_name = arm.run_name(seed)
    run_dir = arm.run_root / run_name
    paths = {
        "config": run_dir / "config.json",
        "env": run_dir / "env_snapshot.json",
        "summary": run_dir / "summary.json",
        "final": run_dir / "final_evaluation_fixed.json",
        "metrics": run_dir / "metrics.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    config = read_json(paths["config"])
    summary = read_json(paths["summary"])
    final_eval = read_json(paths["final"])
    validate_config(config, arm, seed, run_name)
    if int(number(summary.get("total_steps"), f"{run_name}.total_steps")) != TOTAL_STEPS:
        raise ValueError(f"Incomplete source run: {run_name}")

    _fields, rows = read_evaluations(paths["metrics"])
    steps = [int(number(row["step"], f"{run_name}.step")) for row in rows]
    values = [number(row["eval/env_success_mean"], f"{run_name}@{step}") for row, step in zip(rows, steps)]
    auc = trapezoid(steps, values)
    quarter = [(step, value) for step, value in zip(steps, values) if step >= LAST_QUARTER_START]
    last_quarter_auc = trapezoid([step for step, _ in quarter], [value for _, value in quarter])
    hit90 = first_hit(steps, values)
    final_success = values[-1]
    best_success = max(values)
    fixed100 = number(final_eval.get("eval_success_mean"), f"{run_name}.fixed100")

    close(auc, number(summary.get("eval_AUC_success"), f"{run_name}.summary.auc"), f"{run_name}.auc")
    close(final_success, number(summary.get("final_eval_success"), f"{run_name}.summary.final"), f"{run_name}.final")
    close(best_success, number(summary.get("best_eval_success"), f"{run_name}.summary.best"), f"{run_name}.best")
    recorded_hit_raw = summary.get("first_hit_0.90")
    recorded_hit = None if recorded_hit_raw in (None, "") else int(number(recorded_hit_raw, "Hit90"))
    if recorded_hit != hit90:
        raise ValueError(f"{run_name}.Hit90: recomputed={hit90}, recorded={recorded_hit}")

    return {
        "arm": arm,
        "seed": seed,
        "run_name": run_name,
        "run_dir": run_dir,
        "paths": paths,
        "config": config,
        "env": read_json(paths["env"]),
        "summary": summary,
        "final_eval": final_eval,
        "rows": rows,
        "steps": steps,
        "values": values,
        "success_auc": auc,
        "last_quarter_auc": last_quarter_auc,
        "final_success": final_success,
        "best_success": best_success,
        "first_hit_0.90": hit90,
        "fixed100_success": fixed100,
        "steps_per_sec": number(summary.get("final_steps_per_sec"), f"{run_name}.steps_per_sec"),
        "runtime_hours": number(summary.get("runtime_total_sec"), f"{run_name}.runtime") / 3600.0,
    }


def copy_seed_record(run: dict[str, object]) -> None:
    arm = run["arm"]
    assert isinstance(arm, Arm)
    output = ARTIFACT_ROOT / "run_records" / arm.key / str(run["run_name"])
    output.mkdir(parents=True, exist_ok=True)
    for source_key, filename in (
        ("config", "config.json"),
        ("env", "env_snapshot.json"),
        ("summary", "summary.json"),
        ("final_eval", "final_evaluation_fixed.json"),
    ):
        write_json(output / filename, sanitize_paths(run[source_key]))
    selected_rows = [
        {field: row.get(field, "") for field in SELECTED_METRIC_FIELDS}
        for row in run["rows"]  # type: ignore[union-attr]
    ]
    write_csv(output / "metrics_selected_10k.csv", SELECTED_METRIC_FIELDS, selected_rows)


def build() -> None:
    for generated_name in ("runs", "run_records"):
        generated_runs = ARTIFACT_ROOT / generated_name
        if generated_runs.exists():
            shutil.rmtree(generated_runs)
    for filename in (
        "evaluation_curves_10k.csv",
        "per_seed_results.csv",
        "aggregate_results.csv",
        "paired_comparisons.csv",
        "source_provenance.json",
        "artifact_manifest.json",
    ):
        path = ARTIFACT_ROOT / filename
        if path.exists():
            path.unlink()

    runs = [load_run(arm, seed) for arm in ARMS for seed in SEEDS]
    for run in runs:
        copy_seed_record(run)

    curve_fields = ("method_key", "paper_label", "seed", "step", "eval_env_success_mean")
    curve_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []
    for run in runs:
        arm = run["arm"]
        assert isinstance(arm, Arm)
        seed = int(run["seed"])
        for step, success in zip(run["steps"], run["values"]):  # type: ignore[arg-type]
            curve_rows.append(
                {
                    "method_key": arm.key,
                    "paper_label": arm.label,
                    "seed": seed,
                    "step": step,
                    "eval_env_success_mean": success,
                }
            )
        per_seed_rows.append(
            {
                "method_key": arm.key,
                "paper_label": arm.label,
                "seed": seed,
                "run_name": run["run_name"],
                "final_success": run["final_success"],
                "best_success": run["best_success"],
                "success_auc": run["success_auc"],
                "last_quarter_auc": run["last_quarter_auc"],
                "first_hit_0.90": run["first_hit_0.90"] if run["first_hit_0.90"] is not None else "",
                "hit90_reached": int(run["first_hit_0.90"] is not None),
                "fixed100_success": run["fixed100_success"],
                "steps_per_sec": run["steps_per_sec"],
                "runtime_hours": run["runtime_hours"],
            }
        )
    write_csv(ARTIFACT_ROOT / "evaluation_curves_10k.csv", curve_fields, curve_rows)

    per_seed_fields = (
        "method_key",
        "paper_label",
        "seed",
        "run_name",
        "final_success",
        "best_success",
        "success_auc",
        "last_quarter_auc",
        "first_hit_0.90",
        "hit90_reached",
        "fixed100_success",
        "steps_per_sec",
        "runtime_hours",
    )
    write_csv(ARTIFACT_ROOT / "per_seed_results.csv", per_seed_fields, per_seed_rows)

    metric_fields = (
        "final_success",
        "best_success",
        "success_auc",
        "last_quarter_auc",
        "fixed100_success",
        "steps_per_sec",
        "runtime_hours",
    )
    aggregate_rows: list[dict[str, object]] = []
    for arm in ARMS:
        source = [row for row in per_seed_rows if row["method_key"] == arm.key]
        aggregate: dict[str, object] = {
            "method_key": arm.key,
            "paper_label": arm.label,
            "n": len(source),
        }
        for field in metric_fields:
            values = [float(row[field]) for row in source]
            aggregate[f"{field}_mean"] = statistics.mean(values)
            aggregate[f"{field}_sample_std"] = sample_std(values)
        hits = [int(row["first_hit_0.90"]) for row in source if row["first_hit_0.90"] != ""]
        aggregate["hit90_count"] = len(hits)
        aggregate["hit90_median_step_among_hits"] = statistics.median(hits) if hits else ""
        aggregate_rows.append(aggregate)
    aggregate_fields = (
        "method_key",
        "paper_label",
        "n",
        *(f"{field}_{suffix}" for field in metric_fields for suffix in ("mean", "sample_std")),
        "hit90_count",
        "hit90_median_step_among_hits",
    )
    write_csv(ARTIFACT_ROOT / "aggregate_results.csv", aggregate_fields, aggregate_rows)

    by_key = {(str(row["method_key"]), int(row["seed"])): row for row in per_seed_rows}
    comparisons = (
        ("AER-minus-ER", "ddpg_aer", "ddpg_er"),
        ("AER-minus-AnnealingER", "ddpg_aer", "ddpg_annealing_er"),
        ("AnnealingER-minus-ER", "ddpg_annealing_er", "ddpg_er"),
    )
    paired_rows: list[dict[str, object]] = []
    for comparison, method, reference in comparisons:
        for metric in ("success_auc", "final_success", "best_success", "last_quarter_auc", "fixed100_success"):
            deltas = [
                float(by_key[(method, seed)][metric]) - float(by_key[(reference, seed)][metric])
                for seed in SEEDS
            ]
            low, high, bootstrap_size = exact_bootstrap(deltas)
            sign_flip_p, sign_flip_size = exact_sign_flip_p(deltas)
            paired_rows.append(
                {
                    "comparison": comparison,
                    "method_key": method,
                    "reference_key": reference,
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "paired_seeds": ";".join(map(str, SEEDS)),
                    "mean_delta": statistics.mean(deltas),
                    "sample_std_delta": sample_std(deltas),
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_method": "exact_n_to_n_paired_resampling",
                    "bootstrap_distribution_size": bootstrap_size,
                    "positive_pairs": sum(delta > 0 for delta in deltas),
                    "zero_pairs": sum(math.isclose(delta, 0.0, abs_tol=1e-15) for delta in deltas),
                    "negative_pairs": sum(delta < 0 for delta in deltas),
                    "exact_sign_flip_p_two_sided": sign_flip_p,
                    "sign_flip_distribution_size": sign_flip_size,
                }
            )
    paired_fields = (
        "comparison",
        "method_key",
        "reference_key",
        "metric",
        "n_pairs",
        "paired_seeds",
        "mean_delta",
        "sample_std_delta",
        "ci95_low",
        "ci95_high",
        "bootstrap_method",
        "bootstrap_distribution_size",
        "positive_pairs",
        "zero_pairs",
        "negative_pairs",
        "exact_sign_flip_p_two_sided",
        "sign_flip_distribution_size",
    )
    write_csv(ARTIFACT_ROOT / "paired_comparisons.csv", paired_fields, paired_rows)

    provenance_runs = []
    for run in runs:
        arm = run["arm"]
        assert isinstance(arm, Arm)
        paths = run["paths"]
        assert isinstance(paths, dict)
        provenance_runs.append(
            {
                "method_key": arm.key,
                "seed": run["seed"],
                "run_name": run["run_name"],
                "source_run_path": Path(run["run_dir"]).relative_to(REPO_ROOT).as_posix(),
                "source_files": {
                    key: {
                        "path": Path(path).relative_to(REPO_ROOT).as_posix(),
                        "size_bytes": Path(path).stat().st_size,
                        "sha256": sha256(Path(path)),
                    }
                    for key, path in paths.items()
                },
            }
        )
    write_json(
        ARTIFACT_ROOT / "source_provenance.json",
        {
            "artifact": ARTIFACT_NAME,
            "source_runs": provenance_runs,
            "note": "Source checkpoints, raw metrics.jsonl, full metrics.csv, and run.log are intentionally excluded.",
        },
    )

    manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    payload_files = sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    manifest = {
        "artifact": ARTIFACT_NAME,
        "manifest_self_excluded": True,
        "evaluation_grid": {
            "start": EVAL_GRID[0],
            "end": EVAL_GRID[-1],
            "frequency": EVAL_FREQ,
            "points": len(EVAL_GRID),
        },
        "arms": [arm.key for arm in ARMS],
        "seeds": list(SEEDS),
        "file_count": len(payload_files),
        "total_size_bytes": sum(path.stat().st_size for path in payload_files),
        "files": [
            {
                "path": path.relative_to(ARTIFACT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in payload_files
        ],
    }
    write_json(manifest_path, manifest)

    print(f"built {ARTIFACT_NAME}: {len(runs)} runs, {len(curve_rows)} evaluation points")


if __name__ == "__main__":
    build()
