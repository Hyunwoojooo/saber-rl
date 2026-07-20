#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import run
import validate_config_parity
import validate_completion
import validate_source_parity


def main() -> int:
    report = validate_config_parity.build_report()
    assert report["passed"] is True
    assert report["nonallowed_differences"] == []
    assert all(row["passed"] for row in report["seeds"])
    for seed in run.MODES["main2m"].seeds:
        args = run.train_args(run.MODES["main2m"], seed)
        assert "context_eligible_count_matched" in args
        assert str(30_000 + seed) in args
    source_report = validate_source_parity.build_report()
    assert source_report["passed"] is True
    with tempfile.TemporaryDirectory(
        prefix="drawer_close_strict_parity_"
    ) as temp:
        root = Path(temp)
        config_path = root / "config_parity.json"
        source_path = root / "source_parity.json"
        config_path.write_text(
            json.dumps(report, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_path.write_text(
            json.dumps(source_report, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for seed in run.MODES["main2m"].seeds:
            validated = validate_completion.validate_config_parity(
                config_path,
                seed,
            )
            assert validated["passed"] is True
        validated_source = validate_completion.validate_source_parity(
            source_path
        )
        assert validated_source["passed"] is True
    print("test_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
