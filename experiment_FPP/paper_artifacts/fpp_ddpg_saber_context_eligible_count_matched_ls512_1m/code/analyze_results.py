#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from strict_analysis import AnalysisError, analyze, find_repo_root, write_outputs


def main() -> int:
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description="Analyze FPP DDPG Main versus Strict count-matched control.")
    parser.add_argument("--main-root", type=Path, default=repo / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512")
    parser.add_argument("--control-root", type=Path, default=repo / "experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_ls512")
    parser.add_argument("--output-dir", type=Path, default=repo / "experiment_FPP/results/fpp_ddpg_saber_context_eligible_count_matched_ls512_1m")
    args = parser.parse_args()
    try:
        bundle = analyze(args.main_root, args.control_root)
        write_outputs(bundle, args.output_dir)
    except (AnalysisError, OSError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "passed", "seeds": [1, 2, 3, 4, 5],
        "output_dir": str(args.output_dir.resolve()),
        "main_minus_control_auc": bundle.inference_rows[0],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
