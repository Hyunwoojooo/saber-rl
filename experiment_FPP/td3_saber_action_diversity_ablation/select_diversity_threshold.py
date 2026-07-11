from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


CANDIDATE_THRESHOLDS = (2, 4, 8)
WINDOW_START_STEP = 20_000
WINDOW_END_STEP = 50_000
RATIO_LOW = 0.4
RATIO_HIGH = 0.7
RATIO_TARGET = 0.55
MIN_POSITIVE_RATE = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _step(row: dict[str, str]) -> int | None:
    for key in ("step", "global_step", "env_step"):
        value = _finite_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def _snapshot(rows: Iterable[dict[str, str]], step: int) -> dict[str, float]:
    fields = [
        "support/probe_total_count",
        "support/probe_main_positive_count",
        *[
            f"support/probe_d{threshold}_positive_count"
            for threshold in CANDIDATE_THRESHOLDS
        ],
    ]
    candidates: list[dict[str, float]] = []
    for row in rows:
        if _step(row) != step:
            continue
        values: dict[str, float] = {}
        valid = True
        for field in fields:
            value = _finite_float(row.get(field))
            if value is None:
                valid = False
                break
            values[field] = value
        if valid:
            candidates.append(values)
    if not candidates:
        raise ValueError(f"no finite probe counter row found at exact step {step}")
    first = candidates[0]
    for candidate in candidates[1:]:
        if candidate != first:
            raise ValueError(f"probe counter rows disagree at step {step}")
    return first


def _distance_to_interval(value: float) -> float:
    if value < RATIO_LOW:
        return RATIO_LOW - value
    if value > RATIO_HIGH:
        return value - RATIO_HIGH
    return 0.0


def select_threshold(metrics_path: Path) -> dict[str, object]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    start = _snapshot(rows, WINDOW_START_STEP)
    end = _snapshot(rows, WINDOW_END_STEP)

    total = end["support/probe_total_count"] - start["support/probe_total_count"]
    main = (
        end["support/probe_main_positive_count"]
        - start["support/probe_main_positive_count"]
    )
    if total <= 0:
        raise ValueError("probe total count did not increase in the 20k-50k window")
    if main <= 0:
        raise ValueError("Dmin=1 had no positive rows in the 20k-50k window")

    rates: dict[str, dict[str, float]] = {
        "1": {
            "positive_count": main,
            "positive_rate": main / total,
            "ratio_to_d1": 1.0,
        }
    }
    eligible: list[tuple[int, float]] = []
    for threshold in CANDIDATE_THRESHOLDS:
        field = f"support/probe_d{threshold}_positive_count"
        count = end[field] - start[field]
        if count < 0 or count > main:
            raise ValueError(
                f"invalid Dmin={threshold} count delta {count}; expected 0 <= count <= {main}"
            )
        rate = count / total
        ratio = count / main
        rates[str(threshold)] = {
            "positive_count": count,
            "positive_rate": rate,
            "ratio_to_d1": ratio,
        }
        if rate >= MIN_POSITIVE_RATE:
            eligible.append((threshold, ratio))

    if not eligible:
        raise ValueError(
            f"no candidate has positive_rate >= {MIN_POSITIVE_RATE:.2f}"
        )
    inside = [
        (threshold, ratio)
        for threshold, ratio in eligible
        if RATIO_LOW <= ratio <= RATIO_HIGH
    ]
    pool = inside if inside else eligible
    selected, selected_ratio = min(
        pool,
        key=lambda item: (
            _distance_to_interval(item[1]),
            abs(item[1] - RATIO_TARGET),
            item[0],
        ),
    )

    return {
        "schema_version": "1.0",
        "selection_rule_version": "ratio_interval_midpoint_smallest_d_v1",
        "source_metrics": metrics_path.as_posix(),
        "source_metrics_sha256": sha256(metrics_path),
        "window_start_step": WINDOW_START_STEP,
        "window_end_step": WINDOW_END_STEP,
        "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
        "minimum_positive_rate": MIN_POSITIVE_RATE,
        "target_ratio_interval": [RATIO_LOW, RATIO_HIGH],
        "target_ratio_midpoint": RATIO_TARGET,
        "target_interval_hit": bool(inside),
        "fallback_used": not bool(inside),
        "tie_break": "smallest Dmin",
        "success_metrics_used": False,
        "window_sample_count": total,
        "d1_positive_count": main,
        "rates": rates,
        "selected_dmin": selected,
        "selected_ratio_to_d1": selected_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select Dmin from pre-registered support-diversity probe counters."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = select_threshold(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
