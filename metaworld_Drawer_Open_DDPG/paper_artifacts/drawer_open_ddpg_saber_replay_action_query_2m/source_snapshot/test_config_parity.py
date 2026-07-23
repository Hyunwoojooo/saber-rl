#!/usr/bin/env python3
from __future__ import annotations

import run
import validate_config_parity


def _value_after(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


def main() -> int:
    report = validate_config_parity.build_report()
    assert report["passed"] is True
    assert report["nonallowed_differences"] == []
    assert report["allowed_paths"] == [
        "output_root",
        "run_name",
        "support_query_mode",
    ]
    assert all(row["passed"] for row in report["seeds"])
    for row in report["seeds"]:
        assert {item["path"] for item in row["differences"]} == {
            "output_root",
            "run_name",
            "support_query_mode",
        }
    for seed in run.MODES["main2m"].seeds:
        args = run.train_args(run.MODES["main2m"], seed)
        assert _value_after(args, "--support_query_mode") == "replay_action"
        assert _value_after(args, "--support_assignment_mode") == "support"
        assert _value_after(args, "--weight_scope") == "critic_only"
        assert "--assignment_seed" not in args
        assert "--no_save_step_checkpoints" in args
        assert str(30_000 + seed) not in args
    assert run.parse_args(["--mode", "main2m"]).max_parallel == 1
    print("test_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
