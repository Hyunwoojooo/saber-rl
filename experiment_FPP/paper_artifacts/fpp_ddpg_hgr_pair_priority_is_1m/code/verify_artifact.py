#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ARTIFACT_ROOT.parents[2]
SOURCE_COMMIT = "79d3347"
PAPER_LABEL = "HGR (matched DDPG+HER)"
SEEDS = (1, 2, 3, 4, 5)
GRID = tuple(range(10_000, 1_000_001, 10_000))
HGR_REQUIRED = (
    "hgr/relabel_fraction",
    "hgr/fallback_fraction",
    "hgr/beta_episode",
    "hgr/beta_goal",
    "hgr/is_weight_mean",
    "hgr/is_weight_min",
    "hgr/episode_priority_mean",
    "hgr/pair_priority_count",
    "hgr/priority_safety_replacement_rate",
    "runtime/hgr_sampling_sec_per_update",
)
SEED_FILES = {
    "summary.json",
    "config.json",
    "env_snapshot.json",
    "metrics_selected_10k.csv",
    "hgr_metrics_selected.csv",
}
FORBIDDEN_NAMES = {"metrics.csv", "metrics.jsonl", "run.log", "checkpoint.pt", "best_checkpoint.pt"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _number(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _required_number(value: object, context: str) -> float:
    result = _number(value)
    if not math.isfinite(result):
        raise AssertionError(f"Missing/non-finite numeric value for {context}: {value!r}")
    return result


def _close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest() -> None:
    manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["paper_label"] == PAPER_LABEL
    assert payload["manifest_self_excluded"] is True
    entries = {entry["path"]: entry for entry in payload["files"]}
    actual_files = {
        path.relative_to(ARTIFACT_ROOT).as_posix(): path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert set(entries) == set(actual_files), (
        f"Manifest file-set mismatch: missing={sorted(set(actual_files) - set(entries))}, "
        f"stale={sorted(set(entries) - set(actual_files))}"
    )
    for relative, path in actual_files.items():
        entry = entries[relative]
        assert int(entry["size_bytes"]) == path.stat().st_size, relative
        assert entry["sha256"] == _sha256(path), relative
    assert int(payload["file_count"]) == len(actual_files)
    assert int(payload["total_size_bytes"]) == sum(path.stat().st_size for path in actual_files.values())


def _eval_curve(path: Path) -> tuple[list[int], list[float]]:
    _fields, rows = _read_csv(path)
    eval_rows = [row for row in rows if math.isfinite(_number(row.get("eval/env_success_mean")))]
    steps = [int(_required_number(row.get("step"), f"{path}.step")) for row in eval_rows]
    values = [
        _required_number(row.get("eval/env_success_mean"), f"{path}@{step}.success")
        for step, row in zip(steps, eval_rows)
    ]
    assert tuple(steps) == GRID, f"Non-exact evaluation grid: {path}"
    return steps, values


def _auc(steps: Sequence[int], values: Sequence[float]) -> float:
    assert tuple(steps) == GRID
    area = sum(
        0.5 * (values[index] + values[index + 1]) * (steps[index + 1] - steps[index])
        for index in range(len(steps) - 1)
    )
    return area / float(steps[-1] - steps[0])


def _first_hit(steps: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    return next((step for step, value in zip(steps, values) if value >= threshold), None)


def _source_curve_path(row: dict[str, str]) -> Path:
    if row["method_key"] == "hgr":
        return ARTIFACT_ROOT / row["run_name"] / "metrics_selected_10k.csv"
    return REPO_ROOT / row["source_run_path"] / "metrics_selected_10k.csv"


def _verify_seed_payloads_and_results() -> list[dict[str, str]]:
    for seed in SEEDS:
        run_name = f"fetch_pick_and_place_ddpg_hgr_tdpriority_s{seed}_1000000"
        run_dir = ARTIFACT_ROOT / run_name
        assert run_dir.is_dir(), run_dir
        actual = {path.name for path in run_dir.iterdir() if path.is_file()}
        assert actual == SEED_FILES, f"Unexpected seed files for {run_name}: {sorted(actual)}"
        for json_name in ("summary.json", "config.json", "env_snapshot.json"):
            json.loads((run_dir / json_name).read_text(encoding="utf-8"))

        selected_fields, selected_rows = _read_csv(run_dir / "metrics_selected_10k.csv")
        assert len(selected_rows) == 200
        assert "eval/env_success_mean" in selected_fields
        selected_counts = {step: 0 for step in GRID}
        for row in selected_rows:
            step = int(_required_number(row.get("step"), f"{run_name}.selected.step"))
            assert step in selected_counts
            selected_counts[step] += 1
        assert set(selected_counts.values()) == {2}

        hgr_fields, hgr_rows = _read_csv(run_dir / "hgr_metrics_selected.csv")
        assert len(hgr_rows) == 100
        assert all(metric in hgr_fields for metric in HGR_REQUIRED)
        hgr_steps = [int(_required_number(row.get("step"), f"{run_name}.hgr.step")) for row in hgr_rows]
        assert tuple(hgr_steps) == GRID
        for row, step in zip(hgr_rows, hgr_steps):
            for metric in HGR_REQUIRED:
                _required_number(row.get(metric), f"{run_name}@{step}.{metric}")

    fields, rows = _read_csv(ARTIFACT_ROOT / "per_seed_results.csv")
    assert len(rows) == 15
    assert {row["method_key"] for row in rows} == {"hgr", "her", "saber"}
    assert {int(row["seed"]) for row in rows} == set(SEEDS)
    assert "success_auc" in fields
    for row in rows:
        steps, values = _eval_curve(_source_curve_path(row))
        final_success = values[-1]
        best_success = max(values)
        auc = _auc(steps, values)
        hit90 = _first_hit(steps, values, 0.90)
        _close(_required_number(row["final_success"], "per_seed.final"), final_success, "final")
        _close(_required_number(row["best_success"], "per_seed.best"), best_success, "best")
        _close(_required_number(row["success_auc"], "per_seed.auc"), auc, "auc")
        _close(
            _required_number(row["best_final_drop"], "per_seed.drop"),
            best_success - final_success,
            "drop",
        )
        recorded_hit = None if row["first_hit_0.90"] == "" else int(float(row["first_hit_0.90"]))
        assert recorded_hit == hit90
        assert int(row["hit90_reached"]) == int(hit90 is not None)
    return rows


def _sample_std(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(deltas: Sequence[float]) -> tuple[float, float, int]:
    n = len(deltas)
    means = [
        statistics.mean(deltas[index] for index in indices)
        for indices in itertools.product(range(n), repeat=n)
    ]
    return _quantile(means, 0.025), _quantile(means, 0.975), len(means)


def _verify_aggregates(per_seed_rows: list[dict[str, str]]) -> None:
    _fields, aggregate_rows = _read_csv(ARTIFACT_ROOT / "aggregate_results.csv")
    assert len(aggregate_rows) == 3
    for aggregate in aggregate_rows:
        method_rows = [row for row in per_seed_rows if row["method_key"] == aggregate["method_key"]]
        assert len(method_rows) == 5
        for source_field, mean_field, std_field in (
            ("final_success", "final_success_mean", "final_success_sample_std"),
            ("best_success", "best_success_mean", "best_success_sample_std"),
            ("success_auc", "success_auc_mean", "success_auc_sample_std"),
            ("best_final_drop", "best_final_drop_mean", "best_final_drop_sample_std"),
        ):
            values = [_required_number(row[source_field], source_field) for row in method_rows]
            _close(_required_number(aggregate[mean_field], mean_field), statistics.mean(values), mean_field)
            _close(_required_number(aggregate[std_field], std_field), _sample_std(values), std_field)
        hits = [
            int(float(row["first_hit_0.90"]))
            for row in method_rows
            if row["first_hit_0.90"] != ""
        ]
        assert int(aggregate["hit90_count"]) == len(hits)
        expected_median = statistics.median(hits) if hits else math.nan
        actual_median = _number(aggregate["hit90_median_step_among_hits"])
        if hits:
            _close(actual_median, float(expected_median), "hit90_median")
        else:
            assert not math.isfinite(actual_median)


def _verify_paired(per_seed_rows: list[dict[str, str]]) -> None:
    _fields, paired_rows = _read_csv(ARTIFACT_ROOT / "paired_vs_her_saber.csv")
    assert len(paired_rows) == 12
    by_key = {(row["method_key"], int(row["seed"])): row for row in per_seed_rows}
    getters = {
        "final_success": lambda row: _number(row["final_success"]),
        "best_success": lambda row: _number(row["best_success"]),
        "success_auc": lambda row: _number(row["success_auc"]),
        "best_final_drop": lambda row: _number(row["best_final_drop"]),
        "hit90_reached": lambda row: _number(row["hit90_reached"]),
        "first_hit_0.90_steps_both_hit": lambda row: _number(row["first_hit_0.90"]),
    }
    for row in paired_rows:
        reference = "her" if row["comparison"] == "HGR-minus-HER" else "saber"
        getter = getters[row["metric"]]
        pairs = []
        seeds = []
        for seed in SEEDS:
            hgr_value = getter(by_key[("hgr", seed)])
            reference_value = getter(by_key[(reference, seed)])
            if math.isfinite(hgr_value) and math.isfinite(reference_value):
                seeds.append(seed)
                pairs.append(hgr_value - reference_value)
        assert int(row["n_pairs"]) == len(pairs)
        assert row["paired_seeds"] == ";".join(str(seed) for seed in seeds)
        _close(_required_number(row["mean_delta"], "paired.mean"), statistics.mean(pairs), "paired.mean")
        _close(_required_number(row["sample_std_delta"], "paired.std"), _sample_std(pairs), "paired.std")
        low, high, size = _bootstrap(pairs)
        _close(_required_number(row["ci95_low"], "paired.low"), low, "paired.low")
        _close(_required_number(row["ci95_high"], "paired.high"), high, "paired.high")
        assert int(row["bootstrap_distribution_size"]) == size


def _verify_readme_and_exclusions() -> None:
    readme = (ARTIFACT_ROOT / "README.md").read_text(encoding="utf-8")
    boundary = (
        "This is an algorithmically faithful HGR pair-priority and importance-correction\n"
        "> implementation under the paper’s matched DDPG+HER backbone. It is not an exact\n"
        "> reproduction of the original HGR paper’s environment and training configuration."
    )
    assert boundary in readme
    assert f"Paper label: **{PAPER_LABEL}**" in readme
    assert "canonical" not in readme.lower()
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.name not in FORBIDDEN_NAMES, path
        assert path.suffix not in {".pt", ".jsonl"}, path


def main() -> int:
    _verify_manifest()
    per_seed_rows = _verify_seed_payloads_and_results()
    _verify_aggregates(per_seed_rows)
    _verify_paired(per_seed_rows)
    _verify_readme_and_exclusions()
    print("artifact verification passed")
    print("hgr_seed_payloads=5")
    print("exact_eval_grid_points=100")
    print("paired_comparison_rows=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
