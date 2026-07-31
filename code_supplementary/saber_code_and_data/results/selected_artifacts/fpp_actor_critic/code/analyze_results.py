#!/usr/bin/env python3
"""Analyze the paired FPP DDPG+HER+SABER actor-critic scope ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scope_ablation import (
    AnalysisError,
    SEEDS,
    analysis_payload,
    analyze_scope_ablation,
    find_repo_root,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-root",
        type=Path,
        default=repo / "experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512",
        help="Canonical raw actor-critic run root.",
    )
    parser.add_argument(
        "--critic-root",
        type=Path,
        default=repo / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512",
        help="Canonical critic-only v4 LS512 run root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo
            / "experiment_FPP/paper_artifacts/fpp_ddpg_saber_actor_critic_v4_ls512_1m/analysis"
        ),
        help="Directory for deterministic CSV/JSON/Markdown outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bundle = analyze_scope_ablation(args.actor_root, args.critic_root, SEEDS)
        write_outputs(bundle, args.output_dir)
    except (AnalysisError, OSError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(analysis_payload(bundle), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
