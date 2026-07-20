#!/usr/bin/env python3
"""Analyze paired FPP DDPG Main and Replay-Action Query seeds 1--5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from query_ablation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    AnalysisError,
    analyze_query_ablation,
    find_repo_root,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-root",
        type=Path,
        default=repo / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512",
        help="Canonical Main SABER run root; exact 1M seeds 1-5 are selected by resolved config.",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=repo / "experiment_FPP/runs/ddpg_saber_query_replay_ls512",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "experiment_FPP/results/fpp_ddpg_saber_query_replay_ls512_1m",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bundle = analyze_query_ablation(
            args.main_root,
            args.replay_root,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_outputs(bundle, args.output_dir)
    except (AnalysisError, OSError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "main_root": str(args.main_root.resolve()),
                "replay_root": str(args.replay_root.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "seeds": list(bundle.seeds),
                "main_minus_replay_auc": bundle.inference_rows[0],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
