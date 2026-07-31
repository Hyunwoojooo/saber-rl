#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = tuple(range(6, 11))
GRID = tuple(range(10_000, 1_000_001, 10_000))
QUARTER_GRID = tuple(step for step in GRID if step >= 750_000)
BOOTSTRAP_SEED = 27_027
BOOTSTRAP_RESAMPLES = 10_000
EXECUTION_COMMIT = "70fb7041f4b5142ce91bfd207142e361cced17b8"
VALIDATED_SOURCE_COMMIT = "47343f4cb295f354df16733dca17f3a882a2b34a"
INCLUSIVE_REPLAY_BLOB = "05ec87717586e6bc0823ff72b779515829580b84"
LAUNCHER_BLOB = "2b4a6d7f960eb56d9013f4694abe0e6d14d2c441"
ARMS = {
    "her": "fetch_pick_and_place_td3_her_inclusive_future_ls512_s{seed}_1000000_ls512",
    "saber": "fetch_pick_and_place_td3_saber_inclusive_future_v4_ls512_s{seed}_1000000_ls512",
}
PAIR_CONFIG_DIFFERENCES = {
    "lambda_support",
    "method",
    "run_name",
    "support_diagnostic_audit_interval",
    "weight_scope",
}
SEED_CONFIG_FIELDS = {"run_name", "seeds"}
ADMIN_CONFIG_FIELDS = {"output_root", "resume_path"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_auc(curve: dict[int, float], grid: tuple[int, ...]) -> float:
    xs = np.asarray(grid, dtype=np.float64)
    ys = np.asarray([curve[step] for step in grid], dtype=np.float64)
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def load_curve(path: Path) -> dict[int, float]:
    curve: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            value = row.get("eval/success_mean", "")
            if value in ("", None):
                continue
            step = int(row["step"])
            if step in curve:
                raise ValueError(f"{path}: duplicate evaluation step {step}")
            curve[step] = float(value)
    if tuple(sorted(curve)) != GRID:
        raise ValueError(
            f"{path}: expected exact {len(GRID)}-point grid, got {len(curve)} points"
        )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in curve.values()):
        raise ValueError(f"{path}: success values must be finite rates")
    return curve


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number")
    return number


def paired_bootstrap(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        differences.size,
        size=(BOOTSTRAP_RESAMPLES, differences.size),
    )
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    null_means = [
        float(np.mean(differences * np.asarray(signs, dtype=np.float64)))
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
    ]
    return sum(abs(value) >= observed - 1e-15 for value in null_means) / len(
        null_means
    )


def sanitize_launch_metadata(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        key, value = raw_line.split("=", 1)
        payload[key] = value
    if payload.get("repo_head") != EXECUTION_COMMIT:
        raise ValueError(
            f"{path}: execution commit mismatch: {payload.get('repo_head')}"
        )
    if payload.get("source_reference_commit") != VALIDATED_SOURCE_COMMIT:
        raise ValueError(f"{path}: validated source commit mismatch")
    if payload.get("inclusive_replay_blob") != INCLUSIVE_REPLAY_BLOB:
        raise ValueError(f"{path}: inclusive replay blob mismatch")
    if payload.get("launcher_blob") != LAUNCHER_BLOB:
        raise ValueError(f"{path}: launcher blob mismatch")
    if payload.get("seeds") != "6,7,8,9,10":
        raise ValueError(f"{path}: seed cohort mismatch")
    if payload.get("paired_arms") != "her,saber":
        raise ValueError(f"{path}: paired arm mismatch")
    payload.pop("output_root", None)
    payload["artifact_note"] = (
        "Absolute run and log paths are intentionally omitted from the artifact."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()

    curve_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    records: dict[tuple[str, int], dict[str, Any]] = {}
    configs: dict[tuple[str, int], dict[str, Any]] = {}
    source_hashes: list[dict[str, Any]] = []

    for arm, pattern in ARMS.items():
        for seed in SEEDS:
            run_name = pattern.format(seed=seed)
            run_dir = run_root / run_name
            required = {
                "config": run_dir / "config.json",
                "metrics": run_dir / "metrics.jsonl",
                "summary": run_dir / "summary.json",
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{run_name}: missing {missing}")
            config = json.loads(required["config"].read_text(encoding="utf-8"))
            summary = json.loads(required["summary"].read_text(encoding="utf-8"))
            curve = load_curve(required["metrics"])
            auc = normalized_auc(curve, GRID)
            quarter_auc = normalized_auc(curve, QUARTER_GRID)
            native_auc = finite_number(
                summary.get("eval_AUC_success"),
                f"{run_name}.eval_AUC_success",
            )
            if not math.isclose(auc, native_auc, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{run_name}: recomputed AUC {auc} != summary {native_auc}"
                )
            if int(summary.get("total_steps", -1)) != 1_000_000:
                raise ValueError(f"{run_name}: incomplete total_steps")
            seed_config = config.get("seeds")
            if not isinstance(seed_config, dict):
                raise ValueError(f"{run_name}: missing nested seed configuration")
            for key in (
                "env_seed",
                "torch_seed",
                "numpy_seed",
                "action_noise_seed",
            ):
                if int(seed_config.get(key, -1)) != seed:
                    raise ValueError(f"{run_name}: {key} does not match seed")
            if int(seed_config.get("eval_seed", -1)) != 10_000 + seed:
                raise ValueError(f"{run_name}: eval_seed mismatch")
            if int(seed_config.get("weight_seed", -1)) != 20_000 + seed:
                raise ValueError(f"{run_name}: weight_seed mismatch")

            final_success = curve[1_000_000]
            best_success = max(curve.values())
            hit90 = next(
                (step for step in GRID if curve[step] >= 0.9),
                None,
            )
            record = {
                "arm": arm,
                "seed": seed,
                "auc": auc,
                "final_quarter_auc": quarter_auc,
                "final_success": final_success,
                "best_success": best_success,
                "hit90": "" if hit90 is None else hit90,
                "runtime_seconds": finite_number(
                    summary.get("runtime_total_sec"),
                    f"{run_name}.runtime_total_sec",
                ),
                "steps_per_second": finite_number(
                    summary.get("final_steps_per_sec"),
                    f"{run_name}.final_steps_per_sec",
                ),
            }
            records[(arm, seed)] = record
            configs[(arm, seed)] = config
            per_seed_rows.append(record)
            for step in GRID:
                curve_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "step": step,
                        "success": curve[step],
                    }
                )
            for source_type, path in required.items():
                source_hashes.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "source_type": source_type,
                        "sha256": sha256(path),
                    }
                )

    for seed in SEEDS:
        her = configs[("her", seed)]
        saber = configs[("saber", seed)]
        differences = {
            key for key in set(her) | set(saber) if her.get(key) != saber.get(key)
        }
        if differences != PAIR_CONFIG_DIFFERENCES:
            raise ValueError(
                f"seed {seed}: unexpected HER/SABER config differences {differences}"
            )
    for arm in ARMS:
        reference = configs[(arm, SEEDS[0])]
        for seed in SEEDS[1:]:
            candidate = configs[(arm, seed)]
            differences = {
                key
                for key in set(reference) | set(candidate)
                if reference.get(key) != candidate.get(key)
            }
            if not differences <= SEED_CONFIG_FIELDS:
                raise ValueError(
                    f"{arm} seed {seed}: non-seed config differences {differences}"
                )

    aggregate_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for metric in ("auc", "final_quarter_auc", "final_success"):
            values = [float(records[(arm, seed)][metric]) for seed in SEEDS]
            aggregate_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                }
            )

    paired_seed_rows: list[dict[str, Any]] = []
    paired_statistics_rows: list[dict[str, Any]] = []
    for metric in ("auc", "final_quarter_auc", "final_success"):
        differences = np.asarray(
            [
                float(records[("saber", seed)][metric])
                - float(records[("her", seed)][metric])
                for seed in SEEDS
            ],
            dtype=np.float64,
        )
        for seed, difference in zip(SEEDS, differences):
            paired_seed_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "her": records[("her", seed)][metric],
                    "saber": records[("saber", seed)][metric],
                    "saber_minus_her": float(difference),
                }
            )
        low, high = paired_bootstrap(differences)
        tolerance = 1e-12
        paired_statistics_rows.append(
            {
                "metric": metric,
                "n": differences.size,
                "mean_difference": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "exact_sign_flip_p": exact_sign_flip_p(differences),
                "wins": int(np.sum(differences > tolerance)),
                "ties": int(np.sum(np.abs(differences) <= tolerance)),
                "losses": int(np.sum(differences < -tolerance)),
            }
        )

    write_csv(
        artifact_root / "data/evaluation_curves.csv",
        ["arm", "seed", "step", "success"],
        curve_rows,
    )
    write_csv(
        artifact_root / "results/per_seed_metrics.csv",
        [
            "arm",
            "seed",
            "auc",
            "final_quarter_auc",
            "final_success",
            "best_success",
            "hit90",
            "runtime_seconds",
            "steps_per_second",
        ],
        per_seed_rows,
    )
    write_csv(
        artifact_root / "results/paired_seed_metrics.csv",
        ["seed", "metric", "her", "saber", "saber_minus_her"],
        paired_seed_rows,
    )
    write_csv(
        artifact_root / "results/aggregate_metrics.csv",
        ["arm", "metric", "n", "mean", "sample_sd"],
        aggregate_rows,
    )
    write_csv(
        artifact_root / "results/paired_statistics.csv",
        [
            "metric",
            "n",
            "mean_difference",
            "ci_low",
            "ci_high",
            "bootstrap_resamples",
            "bootstrap_seed",
            "exact_sign_flip_p",
            "wins",
            "ties",
            "losses",
        ],
        paired_statistics_rows,
    )

    shared_config = {
        key: value
        for key, value in configs[("her", SEEDS[0])].items()
        if key
        not in SEED_CONFIG_FIELDS | PAIR_CONFIG_DIFFERENCES | ADMIN_CONFIG_FIELDS
    }
    write_json(
        artifact_root / "provenance/run_contracts.json",
        {
            "schema_version": 1,
            "task": "FetchPickAndPlace",
            "backbone": "TD3",
            "seeds": list(SEEDS),
            "total_steps": 1_000_000,
            "evaluation_grid": {
                "start": GRID[0],
                "stop": GRID[-1],
                "stride": 10_000,
                "count": len(GRID),
            },
            "relabeling_convention": (
                "Candidate transition indices satisfy u >= t and relabeling "
                "uses the candidate transition's next_achieved_goal."
            ),
            "execution_source_commit": EXECUTION_COMMIT,
            "shared_config": shared_config,
            "intended_arm_differences": {
                key: {
                    "her": configs[("her", SEEDS[0])].get(key),
                    "saber": configs[("saber", SEEDS[0])].get(key),
                }
                for key in sorted(PAIR_CONFIG_DIFFERENCES - {"run_name"})
            },
            "source_hashes": source_hashes,
        },
    )
    write_json(
        artifact_root / "provenance/execution.json",
        sanitize_launch_metadata(run_root / "_execution/launch_metadata.txt"),
    )
    write_json(
        artifact_root / "validation_summary.json",
        {
            "status": "PASS",
            "complete_runs": 10,
            "expected_runs": 10,
            "exact_grid_runs": 10,
            "checkpoints_per_run": len(GRID),
            "native_auc_matches": 10,
            "seed_contracts_valid": 10,
            "pair_config_contracts_valid": 5,
            "within_arm_config_contracts_valid": 8,
            "absolute_paths_in_generated_data": False,
        },
    )

    manifest_files = [
        path
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    ]
    write_json(
        artifact_root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact": artifact_root.name,
            "files": [
                {
                    "path": path.relative_to(artifact_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in manifest_files
            ],
        },
    )
    print(f"built {artifact_root}")


if __name__ == "__main__":
    main()
