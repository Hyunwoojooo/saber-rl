#!/usr/bin/env python3
from __future__ import annotations

import analyze_context_count_only_results as analysis


def main() -> int:
    steps = [10_000, 20_000, 30_000]
    assert analysis.normalized_auc(steps, [0.0, 0.5, 1.0]) == 0.5
    differences = [0.1, 0.2, -0.1, 0.0, 0.3]
    first = analysis.paired_bootstrap(differences)
    second = analysis.paired_bootstrap(differences)
    assert first == second
    assert first[2] == 10_000
    p_value, count = analysis.exact_two_sided_sign_flip(differences)
    assert 0.0 <= p_value <= 1.0 and count == 32
    print("test_analyze_context_count_only_results.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
