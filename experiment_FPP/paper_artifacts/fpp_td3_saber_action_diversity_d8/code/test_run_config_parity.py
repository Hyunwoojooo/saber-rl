#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "td3_saber_critic_only_v4_ls512"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_threshold(path: Path, dmin: int = 4, success_used: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_dmin": dmin,
                "candidate_thresholds": [2, 4, 8],
                "window_start_step": 20_000,
                "window_end_step": 50_000,
                "success_metrics_used": success_used,
            }
        ),
        encoding="utf-8",
    )


def command_options(command: list[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise AssertionError(f"unexpected command token {token!r}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            out[token] = command[index + 1]
            index += 2
        else:
            out[token] = True
            index += 1
    return out


def test_snapshot_config_diff_is_allowlisted() -> None:
    baseline = json.loads((BASELINE / "config.json").read_text(encoding="utf-8"))
    current = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    changed = {
        key
        for key in set(baseline) | set(current)
        if baseline.get(key) != current.get(key)
    }
    allowed = {
        "folder",
        "output_root",
        "purpose",
        "reference",
        "run_slug",
        "source_spec",
        "action_bins_per_dim",
        "support_min_unique_action_bins",
        "support_assignment_mode",
        "support_diversity_probe_thresholds",
    }
    assert changed <= allowed, sorted(changed - allowed)


def test_probe_and_production_resolution() -> None:
    module = load_module("diversity_run", HERE / "run.py")
    base = module.load_config(HERE)
    probe = module.resolve_condition_config(base, "probe", None)
    assert probe["total_steps"] == 50_000
    assert probe["support_min_unique_action_bins"] == 1
    assert probe["support_diversity_probe_thresholds"] == [2, 4, 8]
    assert probe["support_predicate_early_exit"] is False
    assert probe["support_short_circuit"] is False

    with tempfile.TemporaryDirectory() as tmp:
        selected = Path(tmp) / "selected.json"
        selected_threshold(selected)
        diverse = module.resolve_condition_config(base, "diverse", selected)
        random_count = module.resolve_condition_config(base, "random_count", selected)

    assert diverse["total_steps"] == 1_000_000
    assert diverse["support_min_unique_action_bins"] == 4
    assert diverse["support_assignment_mode"] == "support"
    assert random_count["support_min_unique_action_bins"] == 4
    assert random_count["support_assignment_mode"] == "random_count"
    changed = {
        key
        for key in set(diverse) | set(random_count)
        if diverse.get(key) != random_count.get(key)
    }
    assert changed == {"run_slug", "output_root", "support_assignment_mode"}


def test_production_requires_outcome_blind_selection_file() -> None:
    module = load_module("diversity_run_required", HERE / "run.py")
    base = module.load_config(HERE)
    try:
        module.resolve_condition_config(base, "diverse", None)
    except ValueError as exc:
        assert "requires --selected-threshold" in str(exc)
    else:
        raise AssertionError("production run must require selected threshold JSON")

    with tempfile.TemporaryDirectory() as tmp:
        selected = Path(tmp) / "selected.json"
        selected_threshold(selected, success_used=True)
        try:
            module.resolve_condition_config(base, "diverse", selected)
        except ValueError as exc:
            assert "success_metrics_used=false" in str(exc)
        else:
            raise AssertionError("success-informed threshold selection must fail")


def test_probe_config_requires_exact_scan() -> None:
    source_root = HERE / "code" / "src"
    import sys

    sys.path.insert(0, str(source_root))
    try:
        from metacog_lite_v0.config import LiteConfig

        cfg = LiteConfig(
            support_diversity_probe_thresholds=(2, 4, 8),
            support_predicate_early_exit=False,
            support_short_circuit=True,
        )
        try:
            cfg.normalize()
        except ValueError as exc:
            assert "support_short_circuit=False" in str(exc)
        else:
            raise AssertionError("probe counters must reject short-circuit scans")
    finally:
        sys.path.remove(str(source_root))


def test_common_cli_matches_baseline() -> None:
    baseline_module = load_module("baseline_run", BASELINE / "run.py")
    module = load_module("diversity_run_cli", HERE / "run.py")
    baseline_module.repo_root_from = lambda _path: HERE.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        selected = Path(tmp) / "selected.json"
        selected_threshold(selected)
        baseline_command = baseline_module.build_command(
            seed=3,
            steps=1_000_000,
            device="cpu",
            output_root="/tmp/fpp-config-parity",
            smoke=False,
            validate_only=True,
        )
        diverse_command, _ = module.build_command(
            condition="diverse",
            seed=3,
            steps=1_000_000,
            device="cpu",
            output_root="/tmp/fpp-config-parity",
            selected_threshold_path=selected,
            smoke=False,
            validate_only=True,
        )

    baseline_options = command_options(baseline_command)
    diverse_options = command_options(diverse_command)
    expected_new = {
        "--action_bins_per_dim",
        "--support_min_unique_action_bins",
        "--support_assignment_mode",
    }
    assert set(diverse_options) - set(baseline_options) == expected_new
    for key, value in baseline_options.items():
        if key == "--run_name":
            continue
        assert diverse_options[key] == value, (key, value, diverse_options[key])
    assert diverse_options["--support_min_unique_action_bins"] == "4"
    assert diverse_options["--support_assignment_mode"] == "support"
    assert diverse_options["--weight_seed"] == "20003"
    assert diverse_options["--eval_seed"] == "10003"


def test_canonical_run_shape_is_enforced() -> None:
    module = load_module("diversity_run_shape", HERE / "run.py")
    module.validate_run_request(
        "probe", 1, 50_000, smoke=False, validate_only=False, dry_run=False
    )
    module.validate_run_request(
        "diverse", 5, None, smoke=False, validate_only=False, dry_run=False
    )
    for condition, seed, steps in (
        ("probe", 2, 50_000),
        ("diverse", 6, 1_000_000),
        ("random_count", 1, 500_000),
    ):
        try:
            module.validate_run_request(
                condition,
                seed,
                steps,
                smoke=False,
                validate_only=False,
                dry_run=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError((condition, seed, steps))
    module.validate_run_request(
        "diverse", 99, 1_000, smoke=True, validate_only=False, dry_run=False
    )


def test_numba_preflight() -> None:
    module = load_module("diversity_run_numba_preflight", HERE / "run.py")
    try:
        module.require_numba_environment("/bin/false", {})
    except RuntimeError as exc:
        assert "numba_support_scan=true" in str(exc)
    else:
        raise AssertionError("failed Numba import preflight must raise")


def main() -> int:
    test_snapshot_config_diff_is_allowlisted()
    test_probe_and_production_resolution()
    test_production_requires_outcome_blind_selection_file()
    test_probe_config_requires_exact_scan()
    test_common_cli_matches_baseline()
    test_canonical_run_shape_is_enforced()
    test_numba_preflight()
    print("test_run_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
