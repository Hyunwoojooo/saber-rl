#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module():
    path = HERE / "select_diversity_threshold.py"
    spec = importlib.util.spec_from_file_location("select_diversity_threshold", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_probe_csv(path: Path, *, disagree: bool = False) -> None:
    fields = [
        "step",
        "eval/env_success_mean",
        "support/probe_total_count",
        "support/probe_main_positive_count",
        "support/probe_d2_positive_count",
        "support/probe_d4_positive_count",
        "support/probe_d8_positive_count",
    ]
    rows = [
        [20_000, 0.99, 2_000, 1_000, 800, 500, 100],
        [20_000, 0.01, 2_000, 1_000, 800, 500, 100],
        [50_000, 0.00, 3_000, 1_500, 1_200, 775, 120],
        [50_000, 1.00, 3_000, 1_500, 1_200, 776 if disagree else 775, 120],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_selects_d4_without_success_metrics() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "metrics.csv"
        write_probe_csv(metrics)
        result = module.select_threshold(metrics)
    assert result["selected_dmin"] == 4
    assert result["success_metrics_used"] is False
    assert result["window_sample_count"] == 1_000
    assert result["d1_positive_count"] == 500
    assert result["rates"]["4"]["positive_rate"] == 0.275
    assert result["rates"]["4"]["ratio_to_d1"] == 0.55
    assert result["rates"]["8"]["positive_rate"] == 0.02


def test_duplicate_counter_disagreement_raises() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "metrics.csv"
        write_probe_csv(metrics, disagree=True)
        try:
            module.select_threshold(metrics)
        except ValueError as exc:
            assert "disagree" in str(exc)
        else:
            raise AssertionError("disagreeing duplicate rows must fail")


def test_missing_exact_checkpoint_raises() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "metrics.csv"
        write_probe_csv(metrics)
        text = metrics.read_text(encoding="utf-8").replace("50000", "49999")
        metrics.write_text(text, encoding="utf-8")
        try:
            module.select_threshold(metrics)
        except ValueError as exc:
            assert "exact step 50000" in str(exc)
        else:
            raise AssertionError("missing exact 50k checkpoint must fail")


def main() -> int:
    test_selects_d4_without_success_metrics()
    test_duplicate_counter_disagreement_raises()
    test_missing_exact_checkpoint_raises()
    print("test_select_diversity_threshold.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
