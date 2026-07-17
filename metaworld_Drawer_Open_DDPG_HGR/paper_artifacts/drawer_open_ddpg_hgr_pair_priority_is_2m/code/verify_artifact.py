#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Sequence


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
GRID = tuple(range(10_000, 2_000_001, 10_000))
SEEDS = (1, 2, 3, 4, 5)
METHODS = ("hgr", "her", "saber")
EXPECTED_HGR_AUCS = (
    0.799748743718593,
    0.7045226130653266,
    0.6922110552763819,
    0.7949748743718593,
    0.4494974874371859,
)
SEED_FILES = {
    "config.json",
    "env_snapshot.json",
    "summary.json",
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
    assert manifest["artifact"] == "drawer_open_ddpg_hgr_pair_priority_is_2m"
    assert manifest["manifest_self_excluded"] is True
    assert manifest["evaluation_grid"] == {
        "start": 10_000,
        "end": 2_000_000,
        "frequency": 10_000,
        "points": 200,
    }
    entries = {entry["path"]: entry for entry in manifest["files"]}
    actual = payload_files()
    assert set(entries) == set(actual), (
        f"Manifest mismatch: missing={sorted(set(actual) - set(entries))}, "
        f"stale={sorted(set(entries) - set(actual))}"
    )
    for relative, path in actual.items():
        entry = entries[relative]
        assert int(entry["size_bytes"]) == path.stat().st_size, relative
        assert entry["sha256"] == sha256(path), relative
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


def verify_seed_payloads() -> None:
    for seed in SEEDS:
        run_name = f"drawer_open_ddpg_hgr_original_pair_is_s{seed}_2m_cuda"
        run_dir = ARTIFACT_ROOT / "runs" / run_name
        assert run_dir.is_dir(), run_dir
        assert {path.name for path in run_dir.iterdir() if path.is_file()} == SEED_FILES
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        json.loads((run_dir / "env_snapshot.json").read_text(encoding="utf-8"))
        assert config["method"] == "hgr"
        assert config["weight_scope"] == "none"
        assert config["total_steps"] == 2_000_000
        assert config["her_ratio"] == 0.8
        assert config["hgr_alpha_episode"] == config["hgr_alpha_goal"] == 0.6
        assert config["hgr_beta_episode_initial"] == config["hgr_beta_goal_initial"] == 0.4
        assert config["hgr_beta_episode_final"] == config["hgr_beta_goal_final"] == 1.0
        assert config["hgr_priority_clip_max"] == 0.0
        assert config["seeds"]["env_seed"] == seed
        assert not str(config["output_root"]).startswith("/")
        assert not str(summary["run_dir"]).startswith("/")
        fields, rows = read_csv(run_dir / "metrics_selected_10k.csv")
        assert len(rows) == 200
        assert "hgr/is_weight_mean" in fields
        assert "hgr/pair_priority_count" in fields
        assert tuple(int(row["step"]) for row in rows) == GRID


def load_curves() -> dict[tuple[str, int], tuple[list[int], list[float]]]:
    fields, rows = read_csv(ARTIFACT_ROOT / "evaluation_curves_10k.csv")
    assert fields == ["method_key", "paper_label", "seed", "step", "eval_env_success_mean"]
    assert len(rows) == len(METHODS) * len(SEEDS) * len(GRID)
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["method_key"], int(row["seed"]))
        grouped.setdefault(key, []).append(
            (int(row["step"]), required_number(row["eval_env_success_mean"], f"curve.{key}"))
        )
    assert set(grouped) == {(method, seed) for method in METHODS for seed in SEEDS}
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
        quarter_points = [(step, value) for step, value in zip(steps, values) if step >= 1_510_000]
        quarter_steps = [step for step, _ in quarter_points]
        quarter_values = [value for _, value in quarter_points]
        hit = first_hit(steps, values)
        close(required_number(row["final_success"], "final"), values[-1], f"{key}.final")
        close(required_number(row["best_success"], "best"), max(values), f"{key}.best")
        close(required_number(row["success_auc"], "auc"), trapezoid(steps, values), f"{key}.auc")
        close(
            required_number(row["last_quarter_auc"], "last_quarter"),
            trapezoid(quarter_steps, quarter_values),
            f"{key}.last_quarter",
        )
        recorded_hit = None if row["first_hit_0.90"] == "" else int(float(row["first_hit_0.90"]))
        assert recorded_hit == hit, key
        assert int(row["hit90_reached"]) == int(hit is not None), key
    for seed, expected in zip(SEEDS, EXPECTED_HGR_AUCS):
        close(required_number(by_key[("hgr", seed)]["success_auc"], "hgr.auc"), expected, f"hgr.seed{seed}.auc")
    return rows


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def verify_aggregate(per_seed: Sequence[dict[str, str]]) -> None:
    _fields, rows = read_csv(ARTIFACT_ROOT / "aggregate_results.csv")
    assert len(rows) == 3
    by_method = {row["method_key"]: row for row in rows}
    for method in METHODS:
        source = [row for row in per_seed if row["method_key"] == method]
        assert len(source) == 5
        aggregate = by_method[method]
        for field in ("final_success", "best_success", "success_auc", "last_quarter_auc"):
            values = [required_number(row[field], f"{method}.{field}") for row in source]
            close(required_number(aggregate[f"{field}_mean"], "mean"), statistics.mean(values), f"{method}.{field}.mean")
            close(required_number(aggregate[f"{field}_sample_std"], "std"), sample_std(values), f"{method}.{field}.std")
        hits = [int(float(row["first_hit_0.90"])) for row in source if row["first_hit_0.90"]]
        assert int(aggregate["hit90_count"]) == len(hits)
        close(required_number(aggregate["hit90_median_step_among_hits"], "median"), statistics.median(hits), f"{method}.Hit90")
    close(required_number(by_method["hgr"]["success_auc_mean"], "hgr.mean_auc"), 0.6881909547738694, "hgr.mean_auc")


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap(deltas: Sequence[float]) -> tuple[float, float, int]:
    n = len(deltas)
    means = [
        statistics.mean(deltas[index] for index in indices)
        for indices in itertools.product(range(n), repeat=n)
    ]
    return quantile(means, 0.025), quantile(means, 0.975), len(means)


def verify_paired(per_seed: Sequence[dict[str, str]]) -> None:
    _fields, rows = read_csv(ARTIFACT_ROOT / "paired_comparisons.csv")
    assert len(rows) == 10
    by_key = {(row["method_key"], int(row["seed"])): row for row in per_seed}
    for row in rows:
        reference = "her" if row["comparison"] == "HGR-minus-HER" else "saber"
        field = "first_hit_0.90" if row["metric"] == "first_hit_0.90_both_reached" else row["metric"]
        pairs: list[tuple[int, float]] = []
        for seed in SEEDS:
            hgr_value = by_key[("hgr", seed)][field]
            other_value = by_key[(reference, seed)][field]
            if hgr_value == "" or other_value == "":
                continue
            pairs.append((seed, float(hgr_value) - float(other_value)))
        deltas = [delta for _, delta in pairs]
        assert int(row["n_pairs"]) == len(pairs)
        assert row["paired_seeds"] == ";".join(str(seed) for seed, _ in pairs)
        close(required_number(row["mean_delta"], "paired.mean"), statistics.mean(deltas), "paired.mean")
        close(required_number(row["sample_std_delta"], "paired.std"), sample_std(deltas), "paired.std")
        low, high, size = bootstrap(deltas)
        close(required_number(row["ci95_low"], "paired.low"), low, "paired.low")
        close(required_number(row["ci95_high"], "paired.high"), high, "paired.high")
        assert int(row["bootstrap_distribution_size"]) == size


def verify_scope_and_exclusions() -> None:
    readme = (ARTIFACT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "full two-stage HGR transfer" in readme
    assert "paired HGR-minus-SABER AUC interval includes zero" in readme
    assert "not an exact" in readme
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.name not in FORBIDDEN_NAMES, path
        assert path.suffix not in {".pt", ".jsonl"}, path


def main() -> int:
    verify_manifest()
    verify_seed_payloads()
    curves = load_curves()
    per_seed = verify_per_seed(curves)
    verify_aggregate(per_seed)
    verify_paired(per_seed)
    verify_scope_and_exclusions()
    print("artifact verification passed")
    print("hgr_seed_payloads=5")
    print("exact_curve_rows=3000")
    print("paired_rows=10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
