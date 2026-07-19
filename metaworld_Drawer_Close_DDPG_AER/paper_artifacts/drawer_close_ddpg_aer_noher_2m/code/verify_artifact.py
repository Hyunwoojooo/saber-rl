#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Sequence


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
GRID = tuple(range(10_000, 2_000_001, 10_000))
SEEDS = (1, 2, 3, 4, 5)
ARMS = ("ddpg_er", "ddpg_annealing_er", "ddpg_aer")
EXPECTED_AUC_MEANS = {
    "ddpg_er": 0.14967336683417085,
    "ddpg_annealing_er": 0.19927135678391958,
    "ddpg_aer": 0.1935427135678392,
}
SEED_FILES = {
    "config.json",
    "env_snapshot.json",
    "summary.json",
    "final_evaluation_fixed.json",
    "metrics_selected_10k.csv",
}
FORBIDDEN_NAMES = {
    "checkpoint.pt",
    "best_checkpoint.pt",
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def required_number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Invalid number for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise AssertionError(f"Non-finite number for {context}: {value!r}")
    return result


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files() -> dict[str, Path]:
    manifest = ARTIFACT_ROOT / "artifact_manifest.json"
    return {
        path.relative_to(ARTIFACT_ROOT).as_posix(): path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def verify_manifest() -> None:
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact"] == "drawer_close_ddpg_aer_noher_2m"
    assert manifest["manifest_self_excluded"] is True
    assert manifest["evaluation_grid"] == {
        "start": 10_000,
        "end": 2_000_000,
        "frequency": 10_000,
        "points": 200,
    }
    assert manifest["arms"] == list(ARMS)
    assert manifest["seeds"] == list(SEEDS)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    actual = payload_files()
    assert set(entries) == set(actual), (
        f"Manifest mismatch: missing={sorted(set(actual) - set(entries))}, "
        f"stale={sorted(set(entries) - set(actual))}"
    )
    for relative, path in actual.items():
        assert int(entries[relative]["size_bytes"]) == path.stat().st_size, relative
        assert entries[relative]["sha256"] == sha256(path), relative
    assert int(manifest["file_count"]) == len(actual)
    assert int(manifest["total_size_bytes"]) == sum(path.stat().st_size for path in actual.values())


def trapezoid(steps: Sequence[int], values: Sequence[float]) -> float:
    area = sum(
        0.5 * (values[index] + values[index + 1]) * (steps[index + 1] - steps[index])
        for index in range(len(steps) - 1)
    )
    return area / float(steps[-1] - steps[0])


def first_hit(steps: Sequence[int], values: Sequence[float]) -> int | None:
    return next((step for step, value in zip(steps, values) if value >= 0.90), None)


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def verify_seed_payloads() -> None:
    for arm in ARMS:
        arm_dir = ARTIFACT_ROOT / "run_records" / arm
        assert arm_dir.is_dir(), arm_dir
        run_dirs = sorted(path for path in arm_dir.iterdir() if path.is_dir())
        assert len(run_dirs) == 5
        for seed, run_dir in zip(SEEDS, run_dirs):
            assert f"_s{seed}_" in run_dir.name, run_dir
            assert {path.name for path in run_dir.iterdir() if path.is_file()} == SEED_FILES
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            final_eval = json.loads((run_dir / "final_evaluation_fixed.json").read_text(encoding="utf-8"))
            json.loads((run_dir / "env_snapshot.json").read_text(encoding="utf-8"))
            assert config["replay_method"] == arm
            assert config["her_ratio"] == 0.0
            assert config["total_steps"] == 2_000_000
            assert config["policy_delay"] == 1
            assert config["target_policy_noise"] == 0.0
            assert config["target_noise_clip"] == 0.0
            assert config["seeds"]["env_seed"] == seed
            assert not str(config["output_root"]).startswith("/")
            assert not str(summary["run_dir"]).startswith("/")
            assert summary["total_steps"] == 2_000_000
            assert final_eval["episodes"] == 100
            fields, rows = read_csv(run_dir / "metrics_selected_10k.csv")
            assert len(fields) >= 30
            assert len(rows) == 200
            assert tuple(int(row["step"]) for row in rows) == GRID


def load_curves() -> dict[tuple[str, int], tuple[list[int], list[float]]]:
    fields, rows = read_csv(ARTIFACT_ROOT / "evaluation_curves_10k.csv")
    assert fields == ["method_key", "paper_label", "seed", "step", "eval_env_success_mean"]
    assert len(rows) == len(ARMS) * len(SEEDS) * len(GRID)
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["method_key"], int(row["seed"]))
        grouped.setdefault(key, []).append(
            (int(row["step"]), required_number(row["eval_env_success_mean"], f"curve.{key}"))
        )
    assert set(grouped) == {(arm, seed) for arm in ARMS for seed in SEEDS}
    curves: dict[tuple[str, int], tuple[list[int], list[float]]] = {}
    for key, points in grouped.items():
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        assert tuple(steps) == GRID, key
        assert all(0.0 <= value <= 1.0 for value in values), key
        curves[key] = (steps, values)
    return curves


def verify_per_seed(curves: dict[tuple[str, int], tuple[list[int], list[float]]]) -> list[dict[str, str]]:
    _fields, rows = read_csv(ARTIFACT_ROOT / "per_seed_results.csv")
    assert len(rows) == 15
    by_key = {(row["method_key"], int(row["seed"])): row for row in rows}
    assert set(by_key) == set(curves)
    for key, (steps, values) in curves.items():
        row = by_key[key]
        quarter = [(step, value) for step, value in zip(steps, values) if step >= 1_510_000]
        hit = first_hit(steps, values)
        close(required_number(row["final_success"], "final"), values[-1], f"{key}.final")
        close(required_number(row["best_success"], "best"), max(values), f"{key}.best")
        close(required_number(row["success_auc"], "auc"), trapezoid(steps, values), f"{key}.auc")
        close(
            required_number(row["last_quarter_auc"], "last_quarter"),
            trapezoid([step for step, _ in quarter], [value for _, value in quarter]),
            f"{key}.last_quarter",
        )
        recorded_hit = None if row["first_hit_0.90"] == "" else int(float(row["first_hit_0.90"]))
        assert recorded_hit == hit, key
        assert int(row["hit90_reached"]) == int(hit is not None), key
    return rows


def verify_aggregate(per_seed: Sequence[dict[str, str]]) -> None:
    _fields, rows = read_csv(ARTIFACT_ROOT / "aggregate_results.csv")
    assert len(rows) == 3
    by_arm = {row["method_key"]: row for row in rows}
    for arm in ARMS:
        source = [row for row in per_seed if row["method_key"] == arm]
        aggregate = by_arm[arm]
        assert len(source) == int(aggregate["n"]) == 5
        for field in (
            "final_success",
            "best_success",
            "success_auc",
            "last_quarter_auc",
            "fixed100_success",
            "steps_per_sec",
            "runtime_hours",
        ):
            values = [required_number(row[field], f"{arm}.{field}") for row in source]
            close(required_number(aggregate[f"{field}_mean"], "mean"), statistics.mean(values), f"{arm}.{field}.mean")
            close(required_number(aggregate[f"{field}_sample_std"], "std"), sample_std(values), f"{arm}.{field}.std")
        hits = [int(float(row["first_hit_0.90"])) for row in source if row["first_hit_0.90"]]
        assert int(aggregate["hit90_count"]) == len(hits)
        expected_median = "" if not hits else statistics.median(hits)
        if hits:
            close(required_number(aggregate["hit90_median_step_among_hits"], "Hit90"), float(expected_median), f"{arm}.Hit90")
        else:
            assert aggregate["hit90_median_step_among_hits"] == ""
        close(required_number(aggregate["success_auc_mean"], "auc.mean"), EXPECTED_AUC_MEANS[arm], f"{arm}.auc.mean")


def verify_paired() -> None:
    _fields, rows = read_csv(ARTIFACT_ROOT / "paired_comparisons.csv")
    assert len(rows) == 15
    assert {row["comparison"] for row in rows} == {
        "AER-minus-ER",
        "AER-minus-AnnealingER",
        "AnnealingER-minus-ER",
    }
    for row in rows:
        assert int(row["n_pairs"]) == 5
        assert row["paired_seeds"] == "1;2;3;4;5"
        assert row["bootstrap_method"] == "exact_n_to_n_paired_resampling"
        assert int(row["bootstrap_distribution_size"]) == 3125
        assert int(row["positive_pairs"]) + int(row["zero_pairs"]) + int(row["negative_pairs"]) == 5


def verify_exclusions() -> None:
    names = {path.name for path in ARTIFACT_ROOT.rglob("*") if path.is_file()}
    assert not names.intersection(FORBIDDEN_NAMES), sorted(names.intersection(FORBIDDEN_NAMES))
    oversized = [path for path in ARTIFACT_ROOT.rglob("*") if path.is_file() and path.stat().st_size > 5_000_000]
    assert not oversized, oversized
    provenance = json.loads((ARTIFACT_ROOT / "source_provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["source_runs"]) == 15


def main() -> None:
    verify_manifest()
    verify_seed_payloads()
    curves = load_curves()
    per_seed = verify_per_seed(curves)
    verify_aggregate(per_seed)
    verify_paired()
    verify_exclusions()
    print("artifact verification passed: 15 complete runs, exact 10k grid, no raw/checkpoint files")


if __name__ == "__main__":
    main()
