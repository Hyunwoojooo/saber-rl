#!/usr/bin/env python3
from __future__ import annotations

import run
import validate_config_parity


def main() -> int:
    report = validate_config_parity.build_report()
    assert report["passed"] is True
    assert report["nonallowed_differences"] == []
    assert all(row["passed"] for row in report["seeds"])
    for seed in run.MODES["main2m"].seeds:
        args = run.train_args(run.MODES["main2m"], seed)
        assert "context_eligible_count_matched" in args
        assert str(30_000 + seed) in args
    print("test_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
