#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


FATAL_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|\[failed\]|\bERROR\b",
    re.IGNORECASE,
)
PER_METRICS = (
    "per/priority_entropy",
    "per/priority_ESS",
    "per/weight_mean",
    "per/weight_max",
    "per/weight_entropy",
    "per/weight_ESS",
    "per/critic_weight_mean",
    "per/critic_weight_max",
    "per/actor_weight_mean",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_metric_gate(path: Path) -> tuple[list[int], list[dict[str, float]]]:
    eval_steps: list[int] = []
    per_rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(line.replace("\x00", "") for line in handle):
            eval_value = row.get("eval/goal_success_mean", "")
            if eval_value not in (None, ""):
                require(
                    math.isfinite(float(eval_value)),
                    f"non-finite eval value in {path}",
                )
                eval_steps.append(int(float(row["step"])))
            raw_weight_max = row.get("per/weight_max", "")
            if raw_weight_max in (None, ""):
                continue
            try:
                raw_weight_max_value = float(raw_weight_max)
            except ValueError:
                continue
            if not math.isfinite(raw_weight_max_value):
                continue
            parsed: dict[str, float] = {}
            for key in PER_METRICS:
                value = row.get(key, "")
                require(
                    value not in (None, ""),
                    f"missing {key} in a PER metric row: {path}",
                )
                number = float(value)
                require(math.isfinite(number), f"non-finite {key} in {path}")
                parsed[key] = number
            per_rows.append(parsed)
    return eval_steps, per_rows


def validate_one(output_root: Path, seed: int, steps: int) -> dict[str, object]:
    run_name = (
        "fetch_pick_and_place_sac_her_per_style_loss_weighting_ls512_"
        f"s{seed}_{steps}"
    )
    run_dir = output_root / run_name
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    required_files = (
        "config.json",
        "env_snapshot.json",
        "metrics.csv",
        "metrics.jsonl",
        "run.log",
        "checkpoint.pt",
        "summary.json",
    )
    for name in required_files:
        require((run_dir / name).is_file(), f"missing {run_dir / name}")

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    require(
        cfg.get("method") == "sac_her_per_style_loss_weighting",
        f"wrong method for seed {seed}",
    )
    require(cfg.get("her_ratio") == 0.8, f"wrong HER ratio for seed {seed}")
    require(cfg.get("updates_per_step") == 1, f"wrong UTD for seed {seed}")
    require(cfg.get("batch_size") == 256, f"wrong batch for seed {seed}")
    require(
        cfg.get("buffer_capacity") == 300_000,
        f"wrong replay capacity for seed {seed}",
    )
    require(
        cfg.get("sac_auto_alpha") is True,
        f"SAC auto-alpha disabled for seed {seed}",
    )
    require(
        cfg.get("sac_alpha") == 0.2,
        f"wrong SAC initial alpha for seed {seed}",
    )
    require(
        cfg.get("sac_target_entropy") is None,
        f"wrong SAC target entropy for seed {seed}",
    )
    require(cfg.get("per_alpha") == 0.6, f"wrong PER alpha for seed {seed}")
    require(cfg.get("eps_PER") == 1e-6, f"wrong PER epsilon for seed {seed}")
    require(cfg.get("U_PER_max") == 2.0, f"wrong PER cap for seed {seed}")
    require(
        cfg.get("weight_scope") == "critic_only",
        f"wrong weight scope for seed {seed}",
    )
    require(
        cfg.get("sac_per_priority_reduction") == "min_abs",
        f"wrong priority reduction for seed {seed}",
    )
    require(cfg.get("agent_kind") == "sac", f"wrong agent kind for seed {seed}")
    require(
        cfg.get("support_lookup") is False,
        f"support lookup enabled for seed {seed}",
    )
    require(
        cfg.get("replay_sampling") == "uniform_her",
        f"replay is not uniform HER for seed {seed}",
    )
    require(
        cfg.get("priority_storage") is False,
        f"priority storage enabled for seed {seed}",
    )
    require(
        cfg.get("importance_sampling_beta") is False,
        f"PER beta correction enabled for seed {seed}",
    )

    convention = cfg.get("per_style_convention", {})
    require(convention.get("enabled") is True, f"PER-style disabled for seed {seed}")
    require(convention.get("backbone") == "SAC", f"wrong backbone for seed {seed}")
    require(
        convention.get("replay_sampling")
        == "uniform HER; no priority storage or replay resampling",
        f"wrong replay convention for seed {seed}",
    )
    require(
        convention.get("twin_priority_reduction") == "min_abs",
        f"wrong convention reduction for seed {seed}",
    )
    require(
        convention.get("target_reused_for_update") is True,
        f"target is not reused for seed {seed}",
    )
    require(
        convention.get("actor_loss_weight") == "all_ones",
        f"actor is weighted for seed {seed}",
    )
    require(
        convention.get("temperature_loss_weight") == "all_ones",
        f"temperature is weighted for seed {seed}",
    )
    require(
        cfg.get("hgr_convention", {}).get("enabled") is False,
        f"HGR unexpectedly enabled for seed {seed}",
    )

    require(summary.get("total_steps") == steps, f"wrong total_steps for seed {seed}")
    expected_updates = max(0, steps - max(5_000, 256) + 1)
    require(
        summary.get("total_updates") == expected_updates,
        f"wrong update count for seed {seed}: {summary.get('total_updates')}",
    )
    require(
        summary.get("final_replay_size") == min(steps, 300_000),
        f"wrong replay size for seed {seed}",
    )
    require(
        math.isfinite(float(summary.get("runtime_total_sec", float("nan")))),
        f"non-finite runtime for seed {seed}",
    )
    final_lite = summary.get("final_lite_metrics", {})
    if expected_updates > 0:
        for key in PER_METRICS:
            require(key in final_lite, f"missing final {key} for seed {seed}")
            require(
                math.isfinite(float(final_lite[key])),
                f"non-finite final {key} for seed {seed}",
            )
        require(
            float(final_lite["per/weight_max"]) <= 2.0 + 1e-6,
            f"raw PER cap exceeded for seed {seed}",
        )
        require(
            np.isclose(
                float(final_lite["per/critic_weight_mean"]),
                1.0,
                atol=1e-6,
            ),
            f"critic weights are not mean-normalized for seed {seed}",
        )
        require(
            np.isclose(
                float(final_lite["per/actor_weight_mean"]),
                1.0,
                atol=1e-6,
            ),
            f"actor weights are not all-one for seed {seed}",
        )

    expected_eval_steps = list(range(25_000, steps + 1, 25_000))
    if not expected_eval_steps or expected_eval_steps[-1] != steps:
        expected_eval_steps.append(steps)
    actual_eval_steps, per_rows = load_metric_gate(run_dir / "metrics.csv")
    require(
        actual_eval_steps == expected_eval_steps,
        f"eval grid mismatch for seed {seed}: {actual_eval_steps}",
    )
    if expected_updates > 0:
        require(per_rows, f"no finite PER metric rows for seed {seed}")
        require(
            max(row["per/weight_max"] for row in per_rows) <= 2.0 + 1e-6,
            f"logged raw PER cap exceeded for seed {seed}",
        )

    log_text = (run_dir / "run.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require(not FATAL_PATTERN.search(log_text), f"fatal log pattern for seed {seed}")
    require("finished run at" in log_text, f"missing finished marker for seed {seed}")

    payload = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    require(payload.get("global_step") == steps, f"checkpoint step mismatch for seed {seed}")
    require(
        payload.get("total_updates") == expected_updates,
        f"checkpoint update mismatch for seed {seed}",
    )
    require("torch_rng_state" in payload, f"missing Torch RNG for seed {seed}")
    require("cuda_rng_state_all" in payload, f"missing CUDA RNG field for seed {seed}")
    for key in (
        "noise_rng_state",
        "sample_rng_state",
        "weight_rng_state",
        "diagnostic_rng_state",
    ):
        require(key in payload, f"missing {key} for seed {seed}")
    require(payload.get("hgr_sampler") is None, f"HGR state present for seed {seed}")

    replay_state = payload.get("replay", {})
    require(
        not any("priority" in str(key).lower() for key in replay_state),
        f"priority state present in uniform replay for seed {seed}",
    )
    agent = payload.get("agent", {})
    for key in (
        "actor",
        "critic1",
        "critic2",
        "critic1_target",
        "critic2_target",
        "actor_opt",
        "critic_opt",
    ):
        require(key in agent, f"missing SAC {key} for seed {seed}")
    require(
        agent.get("total_it") == expected_updates,
        f"SAC update state mismatch for seed {seed}",
    )
    require(
        agent.get("td_priority_reduction") == "min_abs",
        f"agent priority mismatch for seed {seed}",
    )
    require("log_alpha" in agent, f"missing SAC log_alpha for seed {seed}")
    require("alpha_opt" in agent, f"missing SAC alpha optimizer for seed {seed}")
    require(
        agent.get("auto_alpha") is True,
        f"SAC auto-alpha state mismatch for seed {seed}",
    )

    del payload
    gc.collect()
    return {
        "seed": seed,
        "steps": steps,
        "updates": expected_updates,
        "eval_points": len(actual_eval_steps),
        "per_metric_rows": len(per_rows),
        "max_raw_weight": (
            max(row["per/weight_max"] for row in per_rows)
            if per_rows
            else None
        ),
        "success_auc": summary.get("eval_AUC_success"),
        "final_success": summary.get("final_eval_success"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict SAC+HER+PER-style result gate"
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5]
    )
    args = parser.parse_args()
    rows = [
        validate_one(args.output_root, seed, args.steps)
        for seed in args.seeds
    ]
    print(json.dumps({"ok": True, "runs": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
