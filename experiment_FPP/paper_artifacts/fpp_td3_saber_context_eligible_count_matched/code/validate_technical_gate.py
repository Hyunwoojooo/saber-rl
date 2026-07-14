#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPECTED_STEPS = {
    "smoke": 5_000,
    "gate": 50_000,
}


def load_analyzer():
    path = HERE / "analyze_results.py"
    spec = importlib.util.spec_from_file_location(
        "count_matched_gate_analyzer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_finite(name: str, value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} is not finite: {parsed}")
    return parsed


def validate_run(run_dir: Path, kind: str) -> dict[str, object]:
    analyzer = load_analyzer()
    expected_steps = EXPECTED_STEPS[kind]
    required = {
        "config.json",
        "summary.json",
        "metrics.csv",
        "assignment_diagnostics_1000.csv",
        "checkpoint.pt",
        "run.log",
    }
    missing = sorted(
        name for name in required if not (run_dir / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing files {missing}")

    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    if int(config["total_steps"]) != expected_steps:
        raise ValueError(
            f"{run_dir}: total_steps={config['total_steps']}, "
            f"expected {expected_steps}"
        )
    if int(summary["total_steps"]) != expected_steps:
        raise ValueError(f"{run_dir}: summary total_steps mismatch")
    if config.get("support_query_mode") != "current_policy":
        raise ValueError(f"{run_dir}: support_query_mode is not current_policy")
    if (
        config.get("support_assignment_mode")
        != "context_eligible_count_matched"
    ):
        raise ValueError(f"{run_dir}: wrong support_assignment_mode")
    seeds = dict(config["seeds"])
    if int(seeds["env_seed"]) != 1 or int(seeds["assignment_seed"]) != 30_001:
        raise ValueError(f"{run_dir}: gate seed mapping is not seed 1")

    eval_freq = int(config["eval_freq"])
    exact_steps = tuple(range(eval_freq, expected_steps + 1, eval_freq))
    metrics_path = analyzer.metrics_path_for(run_dir)
    steps, values = analyzer.load_eval_curve(metrics_path, exact_steps)
    if steps.size != len(exact_steps):
        raise ValueError(f"{run_dir}: exact evaluation grid count mismatch")
    for index, value in enumerate(values):
        require_finite(f"eval[{index}]", value)

    for field, value in dict(summary["final_losses"]).items():
        require_finite(f"final_losses.{field}", value)
    for field in (
        "final_eval_return",
        "final_eval_success",
        "final_eval_goal_success",
        "final_eval_env_success",
        "eval_AUC_success",
        "runtime_total_sec",
        "final_steps_per_sec",
    ):
        require_finite(field, summary[field])

    total_updates = int(summary["total_updates"])
    diagnostics = analyzer.load_assignment_diagnostics(
        run_dir / "assignment_diagnostics_1000.csv",
        total_updates,
    )
    for field in (
        "exact_k_mismatch_count",
        "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch_count",
        "invariant_failure_count",
    ):
        if int(diagnostics[field]) != 0:
            raise ValueError(f"{run_dir}: {field} is nonzero")

    checkpoint = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["global_step"]) != expected_steps:
        raise ValueError(f"{run_dir}: checkpoint global_step mismatch")
    if int(checkpoint["total_updates"]) != total_updates:
        raise ValueError(f"{run_dir}: checkpoint total_updates mismatch")
    if "assignment_rng_state" not in checkpoint:
        raise ValueError(f"{run_dir}: checkpoint lacks assignment RNG state")
    checkpoint_eval_history = [
        (int(step), float(value))
        for step, value in checkpoint.get("eval_history_env", [])
    ]
    if [step for step, _ in checkpoint_eval_history] != list(exact_steps):
        raise ValueError(
            f"{run_dir}: checkpoint evaluation history is incomplete"
        )
    for step, value in checkpoint_eval_history:
        require_finite(f"checkpoint.eval_history[{step}]", value)
    checkpoint_last_eval = dict(checkpoint.get("last_eval_metrics", {}))
    for field in (
        "eval_return_mean",
        "eval_goal_success_mean",
        "eval_env_success_mean",
    ):
        require_finite(f"checkpoint.last_eval_metrics.{field}", checkpoint_last_eval[field])

    diagnostic_state = dict(checkpoint.get("assignment_diagnostics_state", {}))
    if int(diagnostic_state.get("last_update", -1)) != total_updates:
        raise ValueError(
            f"{run_dir}: checkpoint diagnostic last_update mismatch"
        )
    if int(diagnostic_state.get("count", -1)) != 0:
        raise ValueError(
            f"{run_dir}: final checkpoint has an unflushed diagnostic window"
        )

    log_text = (run_dir / "run.log").read_text(
        encoding="utf-8",
        errors="replace",
    )
    if "Traceback (most recent call last)" in log_text:
        raise ValueError(f"{run_dir}: traceback found in run.log")
    if "finished run at" not in log_text:
        raise ValueError(f"{run_dir}: completion marker missing from run.log")
    if kind == "smoke" and "resumed from" not in log_text:
        raise ValueError(
            f"{run_dir}: smoke checkpoint was not exercised via --resume"
        )

    return {
        "schema_version": 1,
        "kind": kind,
        "passed": True,
        "performance_used_for_gate_decision": False,
        "expected_environment_steps": expected_steps,
        "evaluation_steps": list(exact_steps),
        "evaluation_checkpoint_count": len(exact_steps),
        "total_updates": total_updates,
        "assignment_diagnostic_windows": int(
            diagnostics["window_count"]
        ),
        "invariant_failure_count": 0,
        "exact_k_mismatch_count": 0,
        "raw_histogram_mismatch_count": 0,
        "normalized_histogram_mismatch_count": 0,
        "rng_isolation_test_required": (
            "code/src/test_context_eligible_count_matched.py"
        ),
        "checkpoint_assignment_rng_state_present": True,
        "checkpoint_evaluation_state_present": True,
        "checkpoint_resume_exercised": kind == "smoke",
        "checkpoint_diagnostic_state_restored_contract": True,
        "crash_or_nan": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the count-matched technical smoke or gate run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(EXPECTED_STEPS), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = validate_run(args.run_dir.resolve(), args.kind)
    output = (
        args.output
        if args.output is not None
        else args.run_dir / f"{args.kind}_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
