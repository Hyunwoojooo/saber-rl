#!/usr/bin/env python3
from __future__ import annotations

import validate_context_count_only_config_parity as parity


def main() -> int:
    report = parity.build_report()
    assert report["passed"] is True, report
    assert report["nonallowed_option_differences"] == []
    assert all(report["source_checks"].values())
    assert all(report["identity_checks"].values())
    assert report["description"] == (
        "Context eligibility only (count-unmatched descriptive control)"
    )
    assert report["option_differences"] == [
        {
            "option": "--support_query_mode",
            "main": "current_policy",
            "control": "context_count_only",
            "allowed": True,
        }
    ]

    control_source = parity.CONTROL_LAUNCHER.read_text(encoding="utf-8")
    control_options = parity.launcher_options(control_source, main=False)
    assert control_options["--M_B"] == "16"
    assert control_options["--M_Q_min"] == "4"
    assert control_options["--N_local_min"] == "4"
    assert control_options["--local_search_max"] == "512"
    assert control_options["--action_bins_per_dim"] == "5"
    assert control_options["--lambda_support"] == "1.0"
    assert control_options["--weighting_mode"] == "batch_norm"
    assert control_options["--support_scan_mode"] == "deterministic_tail_v1"
    assert control_options["--support_query_mode"] == "context_count_only"
    assert control_options["--weight_scope"] == "critic_only"
    assert control_options["--method"] == "b3_support_runtime_opt"
    assert control_options["--weight_seed"] == "$((20000 + seed))"
    assert control_options["--eval_seed"] == "$((10000 + seed))"
    print("test_context_count_only_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
