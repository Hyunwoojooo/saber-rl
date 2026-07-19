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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_eval_steps(path: Path) -> list[int]:
    steps: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(line.replace("\x00", "") for line in handle):
            value = row.get("eval/goal_success_mean", "")
            if value in (None, ""):
                continue
            require(math.isfinite(float(value)), f"non-finite eval value in {path}")
            steps.append(int(float(row["step"])))
    return steps


def validate_one(output_root: Path, seed: int, steps: int) -> dict[str, object]:
    run_name = f"fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_{steps}"
    run_dir = output_root / run_name
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    required = [
        "config.json",
        "env_snapshot.json",
        "metrics.csv",
        "metrics.jsonl",
        "run.log",
        "checkpoint.pt",
        "summary.json",
    ]
    for name in required:
        require((run_dir / name).is_file(), f"missing {run_dir / name}")

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    require(cfg.get("method") == "sac_hgr", f"wrong method for seed {seed}")
    require(cfg.get("her_ratio") == 0.8, f"wrong HER ratio for seed {seed}")
    require(cfg.get("sac_auto_alpha") is True, f"SAC auto-alpha disabled for seed {seed}")
    require(cfg.get("sac_alpha") == 0.2, f"wrong SAC initial alpha for seed {seed}")
    require(cfg.get("sac_target_entropy") is None, f"wrong SAC target entropy for seed {seed}")
    convention = cfg.get("hgr_convention", {})
    require(convention.get("enabled") is True, f"HGR disabled for seed {seed}")
    require(convention.get("backbone") == "SAC", f"wrong backbone for seed {seed}")
    require(
        convention.get("twin_priority_reduction") == "min_abs",
        f"wrong twin-priority reduction for seed {seed}",
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

    expected_eval_steps = list(range(25_000, steps + 1, 25_000))
    if not expected_eval_steps or expected_eval_steps[-1] != steps:
        expected_eval_steps.append(steps)
    actual_eval_steps = load_eval_steps(run_dir / "metrics.csv")
    require(
        actual_eval_steps == expected_eval_steps,
        f"eval grid mismatch for seed {seed}: {actual_eval_steps}",
    )
    log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    require(not FATAL_PATTERN.search(log_text), f"fatal log pattern for seed {seed}")
    require("finished run at" in log_text, f"missing finished marker for seed {seed}")

    payload = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    require(payload.get("global_step") == steps, f"checkpoint step mismatch for seed {seed}")
    require("torch_rng_state" in payload, f"missing Torch RNG for seed {seed}")
    require("cuda_rng_state_all" in payload, f"missing CUDA RNG field for seed {seed}")
    agent = payload.get("agent", {})
    for key in ("critic1", "critic2", "critic1_target", "critic2_target"):
        require(key in agent, f"missing SAC {key} for seed {seed}")
    require(agent.get("total_it") == expected_updates, f"SAC update state mismatch for seed {seed}")
    require(agent.get("td_priority_reduction") == "min_abs", f"agent priority mismatch for seed {seed}")
    require("log_alpha" in agent, f"missing SAC log_alpha for seed {seed}")
    require("alpha_opt" in agent, f"missing SAC alpha optimizer for seed {seed}")
    require(agent.get("auto_alpha") is True, f"SAC auto-alpha state mismatch for seed {seed}")

    hgr = payload.get("hgr_sampler", {})
    require(hgr.get("schema") == "hgr_episode_pair_priority_v1", f"wrong HGR schema for seed {seed}")
    require(hgr.get("critic_updates") == expected_updates, f"HGR update mismatch for seed {seed}")
    expected_tables = min(steps // 50, 300_000 // 50)
    tables = hgr.get("tables", {})
    require(len(tables) == expected_tables, f"HGR table count mismatch for seed {seed}")
    pair_count = sum(len(table["priorities"]) for table in tables.values())
    require(pair_count == expected_tables * 1_275, f"HGR pair count mismatch for seed {seed}")
    require(np.isfinite(float(hgr.get("max_priority", np.nan))), f"bad max priority for seed {seed}")

    del payload
    gc.collect()
    return {
        "seed": seed,
        "steps": steps,
        "updates": expected_updates,
        "eval_points": len(actual_eval_steps),
        "hgr_tables": expected_tables,
        "hgr_pairs": pair_count,
        "success_auc": summary.get("eval_AUC_success"),
        "final_success": summary.get("final_eval_success"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict SAC+HER+HGR result gate")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    rows = [validate_one(args.output_root, seed, args.steps) for seed in args.seeds]
    print(json.dumps({"ok": True, "runs": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
