#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Iterable, Sequence


ARTIFACT_NAME = "drawer_open_ddpg_hgr_pair_priority_is_2m"
PAPER_LABEL = "HGR (pair priority + IS)"
TOTAL_STEPS = 2_000_000
EVAL_FREQ = 10_000
EVAL_GRID = tuple(range(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ))
LAST_QUARTER_START = 1_510_000
SEEDS = (1, 2, 3, 4, 5)

SCRIPT_PATH = Path(__file__).resolve()
ARTIFACT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]

HGR_LOGGED_FIELDS = (
    "hgr/episode_priority_mean",
    "hgr/episode_priority_std",
    "hgr/episode_priority_max",
    "hgr/pair_table_count",
    "hgr/pair_priority_count",
    "hgr/pair_priority_memory_mb",
    "hgr/goal_priority_entropy_mean",
    "hgr/goal_selected_rank_mean",
    "hgr/candidate_td_error_mean",
    "hgr/candidate_td_error_max",
    "hgr/relabel_fraction",
    "hgr/is_weight_mean",
    "hgr/is_weight_min",
    "hgr/beta_episode",
    "hgr/beta_goal",
    "hgr/beta_progress",
    "hgr/max_priority",
    "hgr/priority_clip_rate",
    "runtime/hgr_sampling_sec_per_update",
)

HGR_CONFIG_EXPECTED = {
    "task_name": "drawer_open",
    "reward_type": "sparse",
    "max_episode_steps": 200,
    "total_steps": TOTAL_STEPS,
    "method": "hgr",
    "weight_scope": "none",
    "batch_size": 256,
    "buffer_capacity": 300_000,
    "her_ratio": 0.8,
    "updates_per_step": 1,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "eval_freq": EVAL_FREQ,
    "eval_episodes": 20,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "hgr_alpha_episode": 0.6,
    "hgr_alpha_goal": 0.6,
    "hgr_beta_episode_initial": 0.4,
    "hgr_beta_episode_final": 1.0,
    "hgr_beta_goal_initial": 0.4,
    "hgr_beta_goal_final": 1.0,
    "hgr_uniform_mix": 0.0,
    "hgr_priority_clip_max": 0.0,
}

METHODS = (
    {
        "key": "hgr",
        "label": PAPER_LABEL,
        "root": REPO_ROOT
        / "metaworld_Drawer_Open_DDPG_HGR/runs/ddpg_hgr_tdpriority_original_pair_is_2m/DDPG-HER-HGR-original-pair-is-2m",
        "run_name": lambda seed: f"drawer_open_ddpg_hgr_original_pair_is_s{seed}_2m_cuda",
        "metrics_name": "metrics_selected_10k.csv",
        "logical_root": "metaworld_Drawer_Open_DDPG_HGR/runs/ddpg_hgr_tdpriority_original_pair_is_2m/DDPG-HER-HGR-original-pair-is-2m",
    },
    {
        "key": "her",
        "label": "DDPG+HER",
        "root": REPO_ROOT / "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B0-2M",
        "run_name": lambda seed: f"drawer_open_DDPG-HER-B0-2M_s{seed}_2m_cuda",
        "metrics_name": "metrics.csv",
        "logical_root": "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B0-2M",
    },
    {
        "key": "saber",
        "label": "SABER (critic-only DDPG+HER)",
        "root": REPO_ROOT
        / "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M",
        "run_name": lambda seed: f"drawer_open_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda",
        "metrics_name": "metrics.csv",
        "logical_root": "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M",
    },
)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


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


def read_eval_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        fieldnames = list(reader.fieldnames)
        if "step" not in fieldnames or "eval/env_success_mean" not in fieldnames:
            raise ValueError(f"Missing evaluation fields: {path}")
        by_step: dict[int, dict[str, str]] = {}
        for row_index, row in enumerate(reader, start=2):
            success = optional_number(row.get("eval/env_success_mean"))
            if not math.isfinite(success):
                continue
            step = int(number(row.get("step"), f"{path}:{row_index}:step"))
            if step not in EVAL_GRID:
                raise ValueError(f"Evaluation outside exact 10k grid at {step}: {path}")
            if step in by_step:
                raise ValueError(f"Duplicate evaluation at {step}: {path}")
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"Success outside [0, 1] at {step}: {path}")
            by_step[step] = row
    missing = [step for step in EVAL_GRID if step not in by_step]
    if missing:
        raise ValueError(f"Missing evaluation steps in {path}: {missing}")
    return fieldnames, [by_step[step] for step in EVAL_GRID]


def trapezoid(steps: Sequence[int], values: Sequence[float]) -> float:
    if len(steps) < 2 or len(steps) != len(values):
        raise ValueError("Trapezoidal AUC requires aligned sequences with at least two points")
    area = sum(
        0.5 * (values[index] + values[index + 1]) * (steps[index + 1] - steps[index])
        for index in range(len(steps) - 1)
    )
    return area / float(steps[-1] - steps[0])


def first_hit(steps: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    return next((step for step, value in zip(steps, values) if value >= threshold), None)


def validate_hgr_config(config: dict[str, object], seed: int, run_name: str) -> None:
    for field, expected in HGR_CONFIG_EXPECTED.items():
        actual = config.get(field)
        if actual != expected:
            raise ValueError(
                f"HGR protocol mismatch for {run_name}.{field}: {actual!r} != {expected!r}"
            )
    seeds = config.get("seeds")
    if not isinstance(seeds, dict) or int(number(seeds.get("env_seed"), "env_seed")) != seed:
        raise ValueError(f"HGR seed mismatch for {run_name}")
    if config.get("run_name") != run_name:
        raise ValueError(f"HGR run-name mismatch for {run_name}")


def load_run(spec: dict[str, object], seed: int) -> dict[str, object]:
    run_name = spec["run_name"](seed)  # type: ignore[operator]
    run_dir = Path(spec["root"]) / run_name
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / str(spec["metrics_name"])
    for required in (summary_path, metrics_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    summary = read_json(summary_path)
    if int(number(summary.get("total_steps"), f"{run_name}.total_steps")) != TOTAL_STEPS:
        raise ValueError(f"Incomplete source run: {run_name}")

    fieldnames, rows = read_eval_rows(metrics_path)
    steps = [int(number(row["step"], f"{run_name}.step")) for row in rows]
    values = [number(row["eval/env_success_mean"], f"{run_name}@{step}") for row, step in zip(rows, steps)]
    auc = trapezoid(steps, values)
    quarter_steps = [step for step in steps if step >= LAST_QUARTER_START]
    quarter_values = [value for step, value in zip(steps, values) if step >= LAST_QUARTER_START]
    last_quarter_auc = trapezoid(quarter_steps, quarter_values)
    final = values[-1]
    best = max(values)
    hit90 = first_hit(steps, values, 0.90)

    close(final, number(summary.get("final_eval_env_success"), f"{run_name}.summary.final"), f"{run_name}.final")
    close(best, number(summary.get("best_eval_env_success"), f"{run_name}.summary.best"), f"{run_name}.best")
    close(auc, number(summary.get("eval_AUC_success"), f"{run_name}.summary.auc"), f"{run_name}.auc")
    summary_hit = summary.get("first_hit_0.90")
    recorded_hit = None if summary_hit in (None, "") else int(number(summary_hit, "first_hit_0.90"))
    if hit90 != recorded_hit:
        raise ValueError(f"{run_name}.Hit90: recomputed={hit90}, recorded={recorded_hit}")
    recorded_quarter = optional_number(summary.get("eval_last_quarter_AUC_success"))
    if math.isfinite(recorded_quarter):
        close(last_quarter_auc, recorded_quarter, f"{run_name}.last_quarter_auc")

    config: dict[str, object] | None = None
    hgr_final: dict[str, float] = {}
    if spec["key"] == "hgr":
        config_path = run_dir / "config.json"
        env_path = run_dir / "env_snapshot.json"
        if not config_path.is_file() or not env_path.is_file():
            raise FileNotFoundError(f"Missing HGR config/environment snapshot: {run_name}")
        config = read_json(config_path)
        validate_hgr_config(config, seed, run_name)
        missing_fields = [field for field in HGR_LOGGED_FIELDS if field not in fieldnames]
        if missing_fields:
            raise ValueError(f"Missing HGR diagnostics for {run_name}: {missing_fields}")
        hgr_final = {
            field: number(rows[-1].get(field), f"{run_name}.final.{field}")
            for field in HGR_LOGGED_FIELDS
        }

    return {
        "method_key": spec["key"],
        "paper_label": spec["label"],
        "seed": seed,
        "run_name": run_name,
        "source_run_path": f"{spec['logical_root']}/{run_name}",
        "run_dir": run_dir,
        "summary": summary,
        "config": config,
        "fieldnames": fieldnames,
        "rows": rows,
        "steps": steps,
        "values": values,
        "final_success": final,
        "best_success": best,
        "success_auc": auc,
        "last_quarter_auc": last_quarter_auc,
        "first_hit_0.90": hit90,
        "runtime_total_sec": number(summary.get("runtime_total_sec"), f"{run_name}.runtime"),
        "steps_per_sec": number(summary.get("final_steps_per_sec"), f"{run_name}.speed"),
        "hgr_final": hgr_final,
    }


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def sample_std(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.stdev(items) if len(items) > 1 else 0.0


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
    bootstrap_means = [
        statistics.mean(deltas[index] for index in indices)
        for indices in itertools.product(range(n), repeat=n)
    ]
    return quantile(bootstrap_means, 0.025), quantile(bootstrap_means, 0.975), len(bootstrap_means)


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def build_result_rows(results: Sequence[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_seed: list[dict[str, object]] = []
    for result in results:
        per_seed.append(
            {
                "method_key": result["method_key"],
                "paper_label": result["paper_label"],
                "seed": result["seed"],
                "run_name": result["run_name"],
                "source_run_path": result["source_run_path"],
                "final_success": result["final_success"],
                "best_success": result["best_success"],
                "success_auc": result["success_auc"],
                "last_quarter_auc": result["last_quarter_auc"],
                "first_hit_0.90": result["first_hit_0.90"],
                "hit90_reached": int(result["first_hit_0.90"] is not None),
                "runtime_total_sec": result["runtime_total_sec"],
                "runtime_hours": float(result["runtime_total_sec"]) / 3600.0,
                "steps_per_sec": result["steps_per_sec"],
            }
        )

    aggregate: list[dict[str, object]] = []
    for method_key in ("hgr", "her", "saber"):
        group = [row for row in per_seed if row["method_key"] == method_key]
        hits = [int(row["first_hit_0.90"]) for row in group if row["first_hit_0.90"] is not None]
        aggregate.append(
            {
                "method_key": method_key,
                "paper_label": group[0]["paper_label"],
                "n_seeds": len(group),
                "final_success_mean": mean(float(row["final_success"]) for row in group),
                "final_success_sample_std": sample_std(float(row["final_success"]) for row in group),
                "best_success_mean": mean(float(row["best_success"]) for row in group),
                "best_success_sample_std": sample_std(float(row["best_success"]) for row in group),
                "success_auc_mean": mean(float(row["success_auc"]) for row in group),
                "success_auc_sample_std": sample_std(float(row["success_auc"]) for row in group),
                "last_quarter_auc_mean": mean(float(row["last_quarter_auc"]) for row in group),
                "last_quarter_auc_sample_std": sample_std(float(row["last_quarter_auc"]) for row in group),
                "hit90_count": len(hits),
                "hit90_total": len(group),
                "hit90_median_step_among_hits": statistics.median(hits) if hits else None,
                "steps_per_sec_mean": mean(float(row["steps_per_sec"]) for row in group),
                "steps_per_sec_sample_std": sample_std(float(row["steps_per_sec"]) for row in group),
                "runtime_hours_mean": mean(float(row["runtime_hours"]) for row in group),
                "runtime_hours_sample_std": sample_std(float(row["runtime_hours"]) for row in group),
            }
        )

    by_key = {(str(row["method_key"]), int(row["seed"])): row for row in per_seed}
    paired: list[dict[str, object]] = []
    metrics = (
        ("final_success", "higher"),
        ("success_auc", "higher"),
        ("last_quarter_auc", "higher"),
        ("hit90_reached", "higher"),
        ("first_hit_0.90_both_reached", "lower"),
    )
    for reference in ("her", "saber"):
        for metric, direction in metrics:
            pairs: list[tuple[int, float]] = []
            for seed in SEEDS:
                hgr = by_key[("hgr", seed)]
                other = by_key[(reference, seed)]
                field = "first_hit_0.90" if metric == "first_hit_0.90_both_reached" else metric
                if hgr[field] is None or other[field] is None:
                    continue
                pairs.append((seed, float(hgr[field]) - float(other[field])))
            deltas = [delta for _, delta in pairs]
            low, high, size = exact_bootstrap(deltas)
            paired.append(
                {
                    "comparison": f"HGR-minus-{reference.upper()}",
                    "metric": metric,
                    "direction_favoring_hgr": direction,
                    "n_pairs": len(pairs),
                    "paired_seeds": ";".join(str(seed) for seed, _ in pairs),
                    "seed_deltas": ";".join(f"s{seed}:{format(delta, '.15g')}" for seed, delta in pairs),
                    "positive_delta_count": sum(delta > 0 for delta in deltas),
                    "mean_delta": mean(deltas),
                    "sample_std_delta": sample_std(deltas),
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_method": "paired percentile bootstrap; exhaustive n^n enumeration",
                    "bootstrap_distribution_size": size,
                }
            )
    return per_seed, aggregate, paired


def copy_hgr_payload(result: dict[str, object]) -> None:
    run_name = str(result["run_name"])
    source = Path(result["run_dir"])
    destination = ARTIFACT_ROOT / "runs" / run_name
    destination.mkdir(parents=True, exist_ok=True)

    config = dict(result["config"])  # type: ignore[arg-type]
    config["output_root"] = str(Path(str(result["source_run_path"])).parent)
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = dict(result["summary"])  # type: ignore[arg-type]
    summary["run_dir"] = str(result["source_run_path"])
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(source / "env_snapshot.json", destination / "env_snapshot.json")
    write_csv(
        destination / "metrics_selected_10k.csv",
        result["fieldnames"],  # type: ignore[arg-type]
        result["rows"],  # type: ignore[arg-type]
    )


def markdown_number(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_readme(aggregate: Sequence[dict[str, object]], paired: Sequence[dict[str, object]], hgr_results: Sequence[dict[str, object]]) -> None:
    aggregate_lines = []
    for row in aggregate:
        median = int(float(row["hit90_median_step_among_hits"])) // 1000
        aggregate_lines.append(
            f"| {row['paper_label']} | {markdown_number(row['final_success_mean'])} +/- {markdown_number(row['final_success_sample_std'])} | "
            f"{markdown_number(row['best_success_mean'])} +/- {markdown_number(row['best_success_sample_std'])} | "
            f"{markdown_number(row['success_auc_mean'])} +/- {markdown_number(row['success_auc_sample_std'])} | "
            f"{markdown_number(row['last_quarter_auc_mean'])} +/- {markdown_number(row['last_quarter_auc_sample_std'])} | "
            f"{row['hit90_count']}/{row['hit90_total']} | {median}k | {markdown_number(row['steps_per_sec_mean'], 2)} |"
        )
    paired_lines = []
    for row in paired:
        if row["metric"] not in {"final_success", "success_auc", "last_quarter_auc", "hit90_reached"}:
            continue
        paired_lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['n_pairs']} | {markdown_number(row['mean_delta'])} | "
            f"[{markdown_number(row['ci95_low'])}, {markdown_number(row['ci95_high'])}] |"
        )

    diagnostics = []
    for field in HGR_LOGGED_FIELDS:
        values = [float(result["hgr_final"][field]) for result in hgr_results]  # type: ignore[index]
        diagnostics.append(f"| `{field}` | {mean(values):.6f} | {sample_std(values):.6f} |")

    text = f"""# DrawerOpen DDPG HGR Pair-Priority + IS 2M Artifact

Paper label: **{PAPER_LABEL}**

> This is a full two-stage HGR transfer under the repository's matched
> DrawerOpen DDPG+HER backbone: episode ranking, future transition-goal-pair
> ranking, and importance correction are all enabled. It is not an exact
> reproduction of the original HGR paper's task or training configuration.

## Scope

This artifact contains the official DrawerOpen DDPG Full HGR result. It covers
seeds 1-5, 2,000,000 environment steps, deterministic evaluation every 10,000
steps with 20 episodes, and the exact recorded evaluation grid. The matched
DDPG+HER and critic-only SABER rows are recomputed from their same-seed curves.

## Aggregate Results

Values after `+/-` are sample standard deviations over five seeds.

| Method | Final success | Best success | AUC | Last-quarter AUC | Hit90 | Hit90 median | Steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(aggregate_lines)}

## Paired Comparisons

Each delta is HGR minus the same-seed reference. The 95% intervals are paired
nonparametric percentile-bootstrap intervals. For five pairs, all 3,125
resamples are enumerated.

| Comparison | Metric | Pairs | Mean delta | 95% CI |
|---|---|---:|---:|---:|
{chr(10).join(paired_lines)}

The full HGR result is substantially stronger than matched HER. SABER has the
higher mean AUC, but the paired HGR-minus-SABER AUC interval includes zero;
these five seeds do not establish superiority of either method. HGR changes
HER sampling and applies sampling-probability correction, whereas SABER keeps
the sampled HER minibatch fixed and changes critic-loss allocation. The result
therefore supports a matched complete-method comparison, not a claim that the
two methods isolate the same mechanism.

## HGR Protocol

- completed-episode pair table over all valid transition/future-goal pairs
- episode and pair priorities with `alpha_episode=alpha_goal=0.6`
- episode and pair IS exponents annealed from 0.4 to 1.0
- HGR relabel fraction 0.8; original-goal uniform branch 0.2
- critic-only IS correction for HGR-relabeled samples; actor weights remain 1
- pre-update absolute TD error updates sampled pair priorities
- no priority clipping, no uniform mixture, UTD 1

## Final HGR Diagnostics

These values are taken from each seed's final 2M evaluation row.

| Metric | Mean | Sample std |
|---|---:|---:|
{chr(10).join(diagnostics)}

## Metric Definitions

- AUC is trapezoidal normalized success area on exactly `10k, ..., 2M`, divided
  by `2,000,000 - 10,000`; no smoothing, interpolation, or carry-forward is used.
- Last-quarter AUC follows the training summary convention: the final 50
  evaluation points, exactly `1.51M, ..., 2M`, divided by 490,000.
- Hit90 is the first recorded evaluation with success at least 0.90; the median
  is over reached seeds and always accompanies coverage.
- Summary-native final, best, AUC, and Hit90 values are checked against the
  exact-grid recomputation before outputs are written.

## Included Files

- `evaluation_curves_10k.csv`: all 15 exact-grid curves
- `per_seed_results.csv`, `aggregate_results.csv`, `paired_comparisons.csv`
- `runs/`: sanitized HGR config, environment snapshot, summary, and selected
  10k metrics for each seed
- `code/build_artifact.py` and `code/verify_artifact.py`
- `artifact_manifest.json`: SHA256 and size of every payload file except itself

Checkpoints, raw `metrics.csv`, JSONL, and run logs are excluded. Host-specific
absolute output paths in copied HGR config/summary files are replaced by their
repository-relative source paths.

## Rebuild

From the repository root, while the source runs are locally available:

```bash
python3 metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/{ARTIFACT_NAME}/code/build_artifact.py
python3 metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/{ARTIFACT_NAME}/code/verify_artifact.py
```
"""
    (ARTIFACT_ROOT / "README.md").write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files() -> list[Path]:
    manifest = ARTIFACT_ROOT / "artifact_manifest.json"
    return sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def write_manifest(results: Sequence[dict[str, object]]) -> None:
    files = [
        {
            "path": path.relative_to(ARTIFACT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload_files()
    ]
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT_NAME,
        "paper_label": PAPER_LABEL,
        "source_revision": "HGR source and artifact are co-committed",
        "evaluation_grid": {"start": EVAL_FREQ, "end": TOTAL_STEPS, "frequency": EVAL_FREQ, "points": len(EVAL_GRID)},
        "manifest_scope": "all payload files except artifact_manifest.json and Python bytecode",
        "manifest_self_excluded": True,
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "source_runs": [
            {
                "method_key": result["method_key"],
                "seed": result["seed"],
                "source_run_path": result["source_run_path"],
            }
            for result in results
        ],
        "excluded": ["checkpoints", "raw metrics.csv", "JSONL", "run logs"],
    }
    (ARTIFACT_ROOT / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    results = [load_run(spec, seed) for spec in METHODS for seed in SEEDS]
    hgr_results = [result for result in results if result["method_key"] == "hgr"]
    for result in hgr_results:
        copy_hgr_payload(result)

    per_seed, aggregate, paired = build_result_rows(results)
    write_csv(
        ARTIFACT_ROOT / "per_seed_results.csv",
        (
            "method_key", "paper_label", "seed", "run_name", "source_run_path",
            "final_success", "best_success", "success_auc", "last_quarter_auc",
            "first_hit_0.90", "hit90_reached", "runtime_total_sec", "runtime_hours",
            "steps_per_sec",
        ),
        per_seed,
    )
    write_csv(
        ARTIFACT_ROOT / "aggregate_results.csv",
        (
            "method_key", "paper_label", "n_seeds", "final_success_mean",
            "final_success_sample_std", "best_success_mean", "best_success_sample_std",
            "success_auc_mean", "success_auc_sample_std", "last_quarter_auc_mean",
            "last_quarter_auc_sample_std", "hit90_count", "hit90_total",
            "hit90_median_step_among_hits", "steps_per_sec_mean", "steps_per_sec_sample_std",
            "runtime_hours_mean", "runtime_hours_sample_std",
        ),
        aggregate,
    )
    write_csv(
        ARTIFACT_ROOT / "paired_comparisons.csv",
        (
            "comparison", "metric", "direction_favoring_hgr", "n_pairs", "paired_seeds",
            "seed_deltas", "positive_delta_count", "mean_delta", "sample_std_delta",
            "ci95_low", "ci95_high", "bootstrap_method", "bootstrap_distribution_size",
        ),
        paired,
    )
    curve_rows = [
        {
            "method_key": result["method_key"],
            "paper_label": result["paper_label"],
            "seed": result["seed"],
            "step": step,
            "eval_env_success_mean": value,
        }
        for result in results
        for step, value in zip(result["steps"], result["values"])
    ]
    write_csv(
        ARTIFACT_ROOT / "evaluation_curves_10k.csv",
        ("method_key", "paper_label", "seed", "step", "eval_env_success_mean"),
        curve_rows,
    )
    write_readme(aggregate, paired, hgr_results)
    write_manifest(results)
    print(f"artifact={ARTIFACT_ROOT}")
    print(f"verified_runs={len(results)}")
    print(f"curve_rows={len(curve_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
