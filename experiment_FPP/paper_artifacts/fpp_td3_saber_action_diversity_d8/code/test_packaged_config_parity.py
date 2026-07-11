#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parent
SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 1_000_001, 10_000))
METHOD_ROOTS = {
    "saber": ARTIFACT_ROOT / "data" / "saber",
    "diverse_d8": ARTIFACT_ROOT / "data" / "diverse_d8",
    "random_count_d8": ARTIFACT_ROOT / "data" / "random_count_d8",
}
METADATA_FIELDS = {"output_root", "run_name"}
DIVERSITY_FIELDS = {
    "support_assignment_mode",
    "support_diversity_probe_thresholds",
    "support_min_unique_action_bins",
}
SHARED_SETTING_GROUPS = {
    "environment": (
        "task_name",
        "env_id",
        "resolved_env_id",
        "reward_type",
        "obs_mode",
        "max_episode_steps",
        "success_terminates",
        "timeout_bootstrap",
        "terminal_convention",
    ),
    "td3": (
        "gamma",
        "tau",
        "noise_std",
        "td3_policy_delay",
        "td3_target_policy_noise",
        "td3_target_noise_clip",
        "start_steps",
        "warmup_steps",
        "buffer_capacity",
        "batch_size",
        "hidden_dim",
    ),
    "her": ("her_ratio",),
    "optimizer": (
        "actor_lr",
        "critic_lr",
        "eps",
        "l2_action_penalty",
        "updates_per_step",
    ),
    "evaluation": ("eval_freq", "eval_episodes"),
}


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def row_step(row: Mapping[str, str]) -> int | None:
    for key in ("step", "global_step", "env_step"):
        value = finite_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def row_success(row: Mapping[str, str]) -> float | None:
    primary = finite_float(row.get("eval/env_success_mean"))
    if primary is not None:
        return primary
    return finite_float(row.get("eval/success_mean"))


def load_exact_eval_steps(metrics_path: Path) -> tuple[int, ...]:
    values: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = row_step(row)
            success = row_success(row)
            if step is None or success is None:
                continue
            previous = values.get(step)
            if previous is not None and not math.isclose(
                previous, success, rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError(
                    f"{metrics_path}: evaluation rows disagree at step {step}"
                )
            values[step] = success
    return tuple(sorted(values))


def discover_runs(root: Path) -> dict[int, tuple[Path, dict[str, object]]]:
    if not root.is_dir():
        raise AssertionError(f"missing packaged run root: {root}")
    found: dict[int, tuple[Path, dict[str, object]]] = {}
    for config_path in sorted(root.glob("*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(config["seeds"]["env_seed"])
        if seed in found:
            raise AssertionError(f"{root}: duplicate seed {seed}")
        for name in ("summary.json", "metrics_selected_10k.csv"):
            if not (run_dir / name).is_file():
                raise AssertionError(f"{run_dir}: missing {name}")
        found[seed] = (run_dir, config)
    if tuple(sorted(found)) != SEEDS:
        raise AssertionError(
            f"{root}: expected seeds {list(SEEDS)}, found {sorted(found)}"
        )
    return found


def changed_keys(
    left: Mapping[str, object],
    right: Mapping[str, object],
    ignored: set[str] | None = None,
) -> set[str]:
    ignored = set() if ignored is None else ignored
    return {
        key
        for key in set(left) | set(right)
        if key not in ignored and left.get(key) != right.get(key)
    }


def assert_shared_groups(
    left: Mapping[str, object], right: Mapping[str, object], seed: int
) -> None:
    for group, fields in SHARED_SETTING_GROUPS.items():
        for field in fields:
            if field not in left or field not in right:
                raise AssertionError(f"seed {seed} {group}: missing {field}")
            if left[field] != right[field]:
                raise AssertionError(
                    f"seed {seed} {group}.{field}: "
                    f"{left[field]!r} != {right[field]!r}"
                )
    if left.get("seeds") != right.get("seeds"):
        raise AssertionError(f"seed {seed}: seed mappings differ")


def test_seed_coverage_steps_and_checkpoints() -> None:
    for method, root in METHOD_ROOTS.items():
        for seed, (run_dir, config) in discover_runs(root).items():
            assert int(config["total_steps"]) == 1_000_000, (method, seed)
            steps = load_exact_eval_steps(run_dir / "metrics_selected_10k.csv")
            assert steps == EVAL_STEPS, (method, seed, len(steps))


def test_saber_and_diverse_allowlisted_parity() -> None:
    saber = discover_runs(METHOD_ROOTS["saber"])
    diverse = discover_runs(METHOD_ROOTS["diverse_d8"])
    allowed = METADATA_FIELDS | DIVERSITY_FIELDS
    for seed in SEEDS:
        saber_config = saber[seed][1]
        diverse_config = diverse[seed][1]
        assert_shared_groups(saber_config, diverse_config, seed)
        differences = changed_keys(saber_config, diverse_config)
        assert differences <= allowed, (seed, sorted(differences - allowed))
        assert int(saber_config.get("support_min_unique_action_bins", 1)) == 1
        assert int(diverse_config["support_min_unique_action_bins"]) == 8
        assert diverse_config["support_assignment_mode"] == "support"


def test_diverse_and_random_count_assignment_only_parity() -> None:
    diverse = discover_runs(METHOD_ROOTS["diverse_d8"])
    random_count = discover_runs(METHOD_ROOTS["random_count_d8"])
    for seed in SEEDS:
        diverse_config = diverse[seed][1]
        random_config = random_count[seed][1]
        assert_shared_groups(diverse_config, random_config, seed)
        differences = changed_keys(
            diverse_config, random_config, ignored=METADATA_FIELDS
        )
        assert differences == {"support_assignment_mode"}, (
            seed,
            sorted(differences),
        )
        assert random_config["support_assignment_mode"] == "random_count"
        assert int(random_config["support_min_unique_action_bins"]) == 8


def main() -> int:
    test_seed_coverage_steps_and_checkpoints()
    test_saber_and_diverse_allowlisted_parity()
    test_diverse_and_random_count_assignment_only_parity()
    print("test_packaged_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
