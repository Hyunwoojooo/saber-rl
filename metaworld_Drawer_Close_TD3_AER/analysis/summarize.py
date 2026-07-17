#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed DrawerClose AER summaries.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results_2m",
    )
    return parser.parse_args()


def mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {sd:.4f}"


def main() -> int:
    args = parse_args()
    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(args.results.rglob("summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        rows[str(payload.get("method", "unknown"))].append(payload)

    print("| Method | n | AUC | Final success (20 ep) | Final fixed success (100 ep) | Steps/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for method, payloads in sorted(rows.items()):
        auc = [float(p["eval_AUC_success"]) for p in payloads]
        final = [float(p["final_eval_env_success"]) for p in payloads]
        fixed = [float(p["final_fixed_eval_env_success"]) for p in payloads]
        speed = [float(p["final_steps_per_sec"]) for p in payloads]
        print(
            f"| {method} | {len(payloads)} | {mean_sd(auc)} | {mean_sd(final)} | "
            f"{mean_sd(fixed)} | {mean_sd(speed)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
