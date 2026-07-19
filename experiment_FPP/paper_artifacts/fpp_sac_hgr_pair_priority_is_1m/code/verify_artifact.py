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
ARTIFACT_NAME = "fpp_sac_hgr_pair_priority_is_1m"
SOURCE_COMMIT = "c2be4e8"
SOURCE_COMMIT_FULL = "c2be4e85a451af61dabba8e54517e9d6d4c1cbf2"
FROZEN_MANIFEST_SHA256 = "3aa0c06576d9a5ecae26e538792d2be6f577bf72ff8af173b528dac408f89772"
PAPER_LABEL = "HGR (matched SAC+HER)"
SEEDS = (1, 2, 3, 4, 5)
PRIMARY_GRID = tuple(range(25_000, 1_000_001, 25_000))
COMMON_GRID = tuple(range(50_000, 1_000_001, 50_000))
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
    "metrics_selected_25k.csv",
    "hgr_metrics_selected.csv",
}
FORBIDDEN_NAMES = {
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
    "checkpoint.pt",
    "best_checkpoint.pt",
    "best_checkpoint.json",
}


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


def _reject_json_constant(token: str) -> None:
    raise AssertionError(f"Non-standard JSON constant: {token}")


def _read_strict_json(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    assert "/home/mimiclab/" not in text, f"Unsanitized local path: {path}"
    return json.loads(text, parse_constant=_reject_json_constant)


def _verify_manifest() -> None:
    assert ARTIFACT_ROOT.name == ARTIFACT_NAME
    manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    payload = _read_strict_json(manifest_path)
    assert isinstance(payload, dict)
    assert payload["artifact"] == ARTIFACT_NAME
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["paper_label"] == PAPER_LABEL
    assert payload["sample_std_ddof"] == 1
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
    assert int(payload["total_size_bytes"]) == sum(
        path.stat().st_size for path in actual_files.values()
    )


def _eval_curve(path: Path, expected_grid: Sequence[int]) -> tuple[list[int], list[float]]:
    _fields, rows = _read_csv(path)
    eval_rows = [row for row in rows if math.isfinite(_number(row.get("eval/env_success_mean")))]
    steps = [int(_required_number(row.get("step"), f"{path}.step")) for row in eval_rows]
    values = [
        _required_number(row.get("eval/env_success_mean"), f"{path}@{step}.success")
        for step, row in zip(steps, eval_rows)
    ]
    assert tuple(steps) == tuple(expected_grid), f"Non-exact evaluation grid: {path}"
    assert all(0.0 <= value <= 1.0 for value in values), path
    return steps, values


def _auc(steps: Sequence[int], values: Sequence[float], expected_grid: Sequence[int]) -> float:
    assert tuple(steps) == tuple(expected_grid)
    assert len(values) == len(expected_grid)
    area = sum(
        0.5 * (values[index] + values[index + 1]) * (steps[index + 1] - steps[index])
        for index in range(len(steps) - 1)
    )
    return area / float(steps[-1] - steps[0])


def _first_hit(steps: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    return next((step for step, value in zip(steps, values) if value >= threshold), None)


def _curve_metrics(
    steps: Sequence[int], values: Sequence[float], expected_grid: Sequence[int]
) -> dict[str, float | int | None]:
    best = max(values)
    hit90 = _first_hit(steps, values, 0.90)
    return {
        "final_success": values[-1],
        "best_success": best,
        "first_best_step": steps[values.index(best)],
        "success_auc": _auc(steps, values, expected_grid),
        "best_final_drop": best - values[-1],
        "first_hit_0.90": hit90,
        "hit90_reached": int(hit90 is not None),
    }


def _native_grid(method_key: str) -> tuple[int, ...]:
    return PRIMARY_GRID if method_key == "hgr" else COMMON_GRID


def _source_curve_path(row: dict[str, str]) -> Path:
    if row["method_key"] == "hgr":
        return ARTIFACT_ROOT / row["run_name"] / "metrics_selected_25k.csv"
    return REPO_ROOT / row["source_run_path"] / "metrics_selected_10k.csv"


def _assert_config(config: dict[str, object], seed: int) -> None:
    expected_numbers = {
        "max_episode_steps": 50,
        "total_steps": 1_000_000,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
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
        "eval_freq": 25_000,
        "eval_episodes": 20,
        "sac_alpha": 0.2,
        "sac_alpha_lr": 3e-4,
        "hgr_alpha_episode": 0.6,
        "hgr_alpha_goal": 0.6,
        "hgr_beta_episode_initial": 0.4,
        "hgr_beta_episode_final": 1.0,
        "hgr_beta_goal_initial": 0.4,
        "hgr_beta_goal_final": 1.0,
        "hgr_uniform_mix": 0.0,
        "hgr_priority_eps": 1e-6,
    }
    assert config["method"] == "sac_hgr"
    assert config["task_name"] == "fetch_pick_and_place"
    assert config["env_id"] == "FetchPickAndPlace-v4"
    assert config["reward_type"] == "sparse"
    for key, expected in expected_numbers.items():
        _close(_required_number(config.get(key), f"config.{key}"), float(expected), f"config.{key}")
    assert config["sac_auto_alpha"] is True
    assert config["sac_target_entropy"] is None
    assert config["hgr_sac_priority_reduction"] == "min_abs"
    assert config["success_terminates"] is False
    assert config["timeout_bootstrap"] is True
    seeds = config["seeds"]
    assert isinstance(seeds, dict) and int(seeds["env_seed"]) == seed
    convention = config["hgr_convention"]
    assert isinstance(convention, dict)
    assert convention["enabled"] is True and convention["backbone"] == "SAC"
    assert convention["twin_priority_reduction"] == "min_abs"


def _verify_seed_payloads_and_results() -> list[dict[str, str]]:
    for seed in SEEDS:
        run_name = f"fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000"
        run_dir = ARTIFACT_ROOT / run_name
        assert run_dir.is_dir(), run_dir
        actual = {path.name for path in run_dir.iterdir() if path.is_file()}
        assert actual == SEED_FILES, f"Unexpected seed files for {run_name}: {sorted(actual)}"
        for json_name in ("summary.json", "config.json", "env_snapshot.json"):
            payload = _read_strict_json(run_dir / json_name)
            assert isinstance(payload, dict)
        config = _read_strict_json(run_dir / "config.json")
        summary = _read_strict_json(run_dir / "summary.json")
        assert isinstance(config, dict) and isinstance(summary, dict)
        _assert_config(config, seed)
        assert summary["method"] == "sac_hgr"
        assert summary["total_steps"] == 1_000_000
        assert summary["total_updates"] == 995_001
        assert summary["episodes"] == 20_000
        assert summary["final_replay_size"] == 300_000

        selected_path = run_dir / "metrics_selected_25k.csv"
        selected_fields, selected_rows = _read_csv(selected_path)
        assert len(selected_rows) == 80
        assert "eval/env_success_mean" in selected_fields
        selected_counts = {step: 0 for step in PRIMARY_GRID}
        for row in selected_rows:
            step = int(_required_number(row.get("step"), f"{run_name}.selected.step"))
            assert step in selected_counts
            selected_counts[step] += 1
        assert set(selected_counts.values()) == {2}
        steps, values = _eval_curve(selected_path, PRIMARY_GRID)
        metrics = _curve_metrics(steps, values, PRIMARY_GRID)
        _close(_required_number(summary["final_eval_env_success"], "summary.final"), float(metrics["final_success"]), "summary.final")
        _close(_required_number(summary["eval_AUC_success"], "summary.auc"), float(metrics["success_auc"]), "summary.auc")

        hgr_fields, hgr_rows = _read_csv(run_dir / "hgr_metrics_selected.csv")
        assert len(hgr_rows) == 40
        assert all(metric in hgr_fields for metric in HGR_REQUIRED)
        hgr_steps = [int(_required_number(row.get("step"), f"{run_name}.hgr.step")) for row in hgr_rows]
        assert tuple(hgr_steps) == PRIMARY_GRID
        for row, step in zip(hgr_rows, hgr_steps):
            for metric in HGR_REQUIRED:
                _required_number(row.get(metric), f"{run_name}@{step}.{metric}")

    fields, rows = _read_csv(ARTIFACT_ROOT / "per_seed_results.csv")
    assert len(rows) == 15
    assert {row["method_key"] for row in rows} == {"hgr", "her", "saber"}
    assert {int(row["seed"]) for row in rows} == set(SEEDS)
    assert "success_auc" in fields
    seen: set[tuple[str, int]] = set()
    for row in rows:
        method_key = row["method_key"]
        seed = int(row["seed"])
        assert (method_key, seed) not in seen
        seen.add((method_key, seed))
        expected_grid = _native_grid(method_key)
        steps, values = _eval_curve(_source_curve_path(row), expected_grid)
        metrics = _curve_metrics(steps, values, expected_grid)
        assert int(row["total_steps"]) == 1_000_000
        assert int(row["eval_grid_points"]) == len(expected_grid)
        if "eval_grid_frequency" in fields:
            assert int(row["eval_grid_frequency"]) == expected_grid[1] - expected_grid[0]
        if "eval_grid_start" in fields:
            assert int(row["eval_grid_start"]) == expected_grid[0]
        if "eval_grid_end" in fields:
            assert int(row["eval_grid_end"]) == expected_grid[-1]
        for column in ("final_success", "best_success", "success_auc", "best_final_drop"):
            _close(_required_number(row[column], f"per_seed.{column}"), float(metrics[column]), column)
        assert int(row["first_best_step"]) == metrics["first_best_step"]
        recorded_hit = None if row["first_hit_0.90"] == "" else int(float(row["first_hit_0.90"]))
        assert recorded_hit == metrics["first_hit_0.90"]
        assert int(row["hit90_reached"]) == metrics["hit90_reached"]
        runtime_sec = _required_number(row["runtime_total_sec"], "per_seed.runtime_total_sec")
        _close(_required_number(row["runtime_hours"], "per_seed.runtime_hours"), runtime_sec / 3600.0, "runtime_hours")
    assert seen == {(method, seed) for method in ("hgr", "her", "saber") for seed in SEEDS}
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
    assert {row["method_key"] for row in aggregate_rows} == {"hgr", "her", "saber"}
    for aggregate in aggregate_rows:
        method_key = aggregate["method_key"]
        expected_grid = _native_grid(method_key)
        method_rows = [row for row in per_seed_rows if row["method_key"] == method_key]
        assert len(method_rows) == 5
        assert int(aggregate["n_seeds"]) == 5
        assert int(aggregate["total_steps"]) == 1_000_000
        assert int(aggregate["eval_grid_start"]) == expected_grid[0]
        assert int(aggregate["eval_grid_end"]) == expected_grid[-1]
        assert int(aggregate["eval_grid_frequency"]) == expected_grid[1] - expected_grid[0]
        assert int(aggregate["eval_grid_points"]) == len(expected_grid)
        for source_field, mean_field, std_field in (
            ("final_success", "final_success_mean", "final_success_sample_std"),
            ("best_success", "best_success_mean", "best_success_sample_std"),
            ("success_auc", "success_auc_mean", "success_auc_sample_std"),
            ("best_final_drop", "best_final_drop_mean", "best_final_drop_sample_std"),
            ("runtime_hours", "runtime_hours_mean", "runtime_hours_sample_std"),
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
        assert int(aggregate["hit90_total"]) == 5
        expected_median = statistics.median(hits) if hits else math.nan
        actual_median = _number(aggregate["hit90_median_step_among_hits"])
        if hits:
            _close(actual_median, float(expected_median), "hit90_median")
        else:
            assert not math.isfinite(actual_median)
        for metric in HGR_REQUIRED:
            mean_field = f"final_{metric}_mean"
            std_field = f"final_{metric}_sample_std"
            values = [_number(row[f"final_{metric}"]) for row in method_rows]
            if method_key == "hgr":
                assert all(math.isfinite(value) for value in values)
                _close(_required_number(aggregate[mean_field], mean_field), statistics.mean(values), mean_field)
                _close(_required_number(aggregate[std_field], std_field), _sample_std(values), std_field)
            else:
                assert not any(math.isfinite(value) for value in values)
                assert not math.isfinite(_number(aggregate[mean_field]))
                assert not math.isfinite(_number(aggregate[std_field]))


def _common_curve_metrics(row: dict[str, str]) -> dict[str, float | int | None]:
    native_grid = _native_grid(row["method_key"])
    steps, values = _eval_curve(_source_curve_path(row), native_grid)
    by_step = dict(zip(steps, values))
    common_values = [by_step[step] for step in COMMON_GRID]
    return _curve_metrics(COMMON_GRID, common_values, COMMON_GRID)


def _verify_paired(per_seed_rows: list[dict[str, str]]) -> None:
    _fields, paired_rows = _read_csv(ARTIFACT_ROOT / "paired_vs_her_saber.csv")
    assert len(paired_rows) == 12
    by_key = {(row["method_key"], int(row["seed"])): row for row in per_seed_rows}
    common = {key: _common_curve_metrics(row) for key, row in by_key.items()}
    getter_names = {
        "final_success": "final_success",
        "best_success": "best_success",
        "success_auc": "success_auc",
        "best_final_drop": "best_final_drop",
        "hit90_reached": "hit90_reached",
        "first_hit_0.90_steps_both_hit": "first_hit_0.90",
    }
    expected_keys = {
        (comparison, metric)
        for comparison in ("HGR-minus-HER", "HGR-minus-SABER")
        for metric in getter_names
    }
    assert {(row["comparison"], row["metric"]) for row in paired_rows} == expected_keys
    for row in paired_rows:
        reference = "her" if row["comparison"] == "HGR-minus-HER" else "saber"
        metric_name = getter_names[row["metric"]]
        pairs: list[float] = []
        seeds: list[int] = []
        for seed in SEEDS:
            hgr_value = common[("hgr", seed)][metric_name]
            reference_value = common[(reference, seed)][metric_name]
            if hgr_value is not None and reference_value is not None:
                hgr_float = float(hgr_value)
                reference_float = float(reference_value)
                if math.isfinite(hgr_float) and math.isfinite(reference_float):
                    seeds.append(seed)
                    pairs.append(hgr_float - reference_float)
        assert pairs
        assert int(row["n_pairs"]) == len(pairs)
        assert row["paired_seeds"] == ";".join(str(seed) for seed in seeds)
        expected_deltas = ";".join(
            f"s{seed}:{format(delta, '.15g')}" for seed, delta in zip(seeds, pairs)
        )
        assert row["seed_deltas"] == expected_deltas
        _close(_required_number(row["mean_delta"], "paired.mean"), statistics.mean(pairs), "paired.mean")
        _close(_required_number(row["sample_std_delta"], "paired.std"), _sample_std(pairs), "paired.std")
        low, high, size = _bootstrap(pairs)
        _close(_required_number(row["ci95_low"], "paired.low"), low, "paired.low")
        _close(_required_number(row["ci95_high"], "paired.high"), high, "paired.high")
        assert int(row["bootstrap_distribution_size"]) == size


def _expected_inputs() -> dict[str, tuple[str, int]]:
    output: dict[str, tuple[str, int]] = {}
    specs = {
        "hgr": (
            "experiment_FPP/runs/sac_hgr_tdpriority_1m",
            lambda seed: f"fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000",
            "metrics.csv",
        ),
        "her": (
            "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
            "sac_her_baseline_ls512",
            lambda seed: f"fetch_pick_and_place_sac_her_baseline_ls512_s{seed}_1000000_ls512",
            "metrics_selected_10k.csv",
        ),
        "saber": (
            "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
            "sac_saber_critic_only_v4_ls512",
            lambda seed: f"fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s{seed}_1000000_ls512",
            "metrics_selected_10k.csv",
        ),
    }
    for method, (root, run_name_factory, metrics_name) in specs.items():
        for seed in SEEDS:
            run_root = f"{root}/{run_name_factory(seed)}"
            names = ["summary.json", "config.json", metrics_name]
            if method == "hgr":
                names.insert(2, "env_snapshot.json")
            for name in names:
                output[f"{run_root}/{name}"] = (method, seed)
    return output


def _verify_provenance_and_completion() -> None:
    input_manifest = _read_strict_json(ARTIFACT_ROOT / "input_manifest.json")
    completion = _read_strict_json(ARTIFACT_ROOT / "completion_validation.json")
    assert isinstance(input_manifest, dict) and isinstance(completion, dict)
    assert input_manifest["artifact"] == ARTIFACT_NAME
    assert input_manifest["source_commit"] == SOURCE_COMMIT_FULL
    assert completion["source_commit"] == SOURCE_COMMIT_FULL
    input_files = input_manifest["files"]
    expected_inputs = _expected_inputs()
    assert isinstance(input_files, list) and len(input_files) == len(expected_inputs) == 50
    assert input_manifest["file_count"] == 50
    seen_paths: set[str] = set()
    input_hashes: dict[str, str] = {}
    for entry in input_files:
        assert isinstance(entry, dict)
        logical_path = str(entry["logical_path"])
        assert logical_path in expected_inputs
        expected_method, expected_seed = expected_inputs[logical_path]
        assert entry["method_key"] == expected_method and int(entry["seed"]) == expected_seed
        assert not logical_path.startswith("/") and ".." not in Path(logical_path).parts
        digest = str(entry["sha256"])
        assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        assert int(entry["size_bytes"]) > 0
        assert logical_path not in seen_paths
        seen_paths.add(logical_path)
        input_hashes[logical_path] = digest
    assert seen_paths == set(expected_inputs)

    assert completion["ok"] is True
    runs = completion["runs"]
    assert isinstance(runs, list) and len(runs) == 5
    by_seed = {int(run["seed"]): run for run in runs}
    assert set(by_seed) == set(SEEDS)
    for seed in SEEDS:
        run = by_seed[seed]
        assert isinstance(run, dict)
        run_name = f"fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000"
        assert run["run_name"] == run_name and run["ok"] is True
        assert run["total_steps"] == 1_000_000
        assert run["episodes"] == 20_000
        assert run["total_updates"] == 995_001
        assert run["final_replay_size"] == 300_000
        assert run["eval_points"] == 40 and run["exact_eval_grid"] is True
        assert run["hgr_tables"] == 6_000 and run["hgr_pairs"] == 7_650_000
        source_summary = f"experiment_FPP/runs/sac_hgr_tdpriority_1m/{run_name}/summary.json"
        assert run["summary_sha256"] == input_hashes[source_summary]

    frozen_path = ARTIFACT_ROOT / "provenance/frozen_source.sha256"
    assert _sha256(frozen_path) == FROZEN_MANIFEST_SHA256
    frozen_lines = [line for line in frozen_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(frozen_lines) == 28
    for line in frozen_lines:
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        assert relative and not relative.startswith("/") and ".." not in Path(relative).parts


def _verify_readme_and_exclusions() -> None:
    readme = (ARTIFACT_ROOT / "README.md").read_text(encoding="utf-8")
    boundary = (
        "> This is an algorithmically faithful HGR pair-priority and importance-correction\n"
        "> implementation transferred to a matched SAC+HER backbone. It is not an exact\n"
        "> reproduction of the original HGR paper’s environment and training configuration."
    )
    assert boundary in readme
    assert f"Paper label: **{PAPER_LABEL}**" in readme
    assert SOURCE_COMMIT in readme
    assert "25k" in readme and "50k" in readme
    assert "canonical" not in readme.lower()
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.name not in FORBIDDEN_NAMES, path
        assert path.suffix not in {".pt", ".jsonl", ".pyc"}, path


def main() -> int:
    _verify_manifest()
    per_seed_rows = _verify_seed_payloads_and_results()
    _verify_aggregates(per_seed_rows)
    _verify_paired(per_seed_rows)
    _verify_provenance_and_completion()
    _verify_readme_and_exclusions()
    print("artifact verification passed")
    print("hgr_seed_payloads=5")
    print("primary_eval_grid_points=40")
    print("paired_common_grid_points=20")
    print("paired_comparison_rows=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
