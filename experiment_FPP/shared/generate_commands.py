from __future__ import annotations

import argparse
import shlex

from experiment_registry import EXPERIMENTS
from run_experiment import build_command


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SABER FetchPickAndPlace run commands.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()
    names = args.experiment or list(EXPERIMENTS)
    for name in names:
        spec = EXPERIMENTS[name]
        for seed in spec.seeds:
            cmd = build_command(name, seed, args.steps, args.device, None, args.smoke)
            print(" ".join(shlex.quote(part) for part in cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
