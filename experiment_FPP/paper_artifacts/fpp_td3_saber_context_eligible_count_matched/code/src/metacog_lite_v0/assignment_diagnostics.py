from __future__ import annotations

from typing import Dict

import numpy as np


TRACKED_METRICS = {
    "context_eligible_count": "assignment/context_eligible_count",
    "main_positive_count": "assignment/main_positive_count",
    "assigned_positive_count": "assignment/assigned_positive_count",
    "context_eligible_rate": "assignment/context_eligible_rate",
    "main_positive_rate": "assignment/main_positive_rate",
    "assigned_positive_rate": "assignment/assigned_positive_rate",
    "overlap_count": "assignment/overlap_count",
    "overlap_over_k": "assignment/overlap_over_k",
    "jaccard": "assignment/jaccard",
    "reference_raw_weight_mean": "assignment/reference_raw_weight_mean",
    "reference_raw_weight_std": "assignment/reference_raw_weight_std",
    "reference_raw_weight_ESS": "assignment/reference_raw_weight_ESS",
    "assigned_raw_weight_mean": "assignment/assigned_raw_weight_mean",
    "assigned_raw_weight_std": "assignment/assigned_raw_weight_std",
    "assigned_raw_weight_ESS": "assignment/assigned_raw_weight_ESS",
    "reference_normalized_weight_mean": (
        "assignment/reference_normalized_weight_mean"
    ),
    "reference_normalized_weight_std": (
        "assignment/reference_normalized_weight_std"
    ),
    "reference_normalized_weight_ESS": (
        "assignment/reference_normalized_weight_ESS"
    ),
    "assigned_normalized_weight_mean": (
        "assignment/assigned_normalized_weight_mean"
    ),
    "assigned_normalized_weight_std": (
        "assignment/assigned_normalized_weight_std"
    ),
    "assigned_normalized_weight_ESS": (
        "assignment/assigned_normalized_weight_ESS"
    ),
    "reference_normalized_weight_sum": (
        "assignment/reference_normalized_weight_sum"
    ),
    "assigned_normalized_weight_sum": (
        "assignment/assigned_normalized_weight_sum"
    ),
    "normalized_weight_sum_abs_diff": (
        "assignment/normalized_weight_sum_abs_diff"
    ),
}

COUNT_METRICS = {
    "raw_histogram_mismatch_count": (
        "assignment/raw_histogram_mismatch_count"
    ),
    "normalized_histogram_mismatch_count": (
        "assignment/normalized_histogram_mismatch_count"
    ),
    "invariant_failure_count": "assignment/invariant_failure_count",
}

ASSIGNMENT_DIAGNOSTIC_FIELDS = [
    "window_index",
    "update_start",
    "update_end",
    "updates",
    "complete_window",
]
for _name in TRACKED_METRICS:
    ASSIGNMENT_DIAGNOSTIC_FIELDS.append(f"{_name}_mean")
for _name in (
    "context_eligible_count",
    "main_positive_count",
    "assigned_positive_count",
):
    ASSIGNMENT_DIAGNOSTIC_FIELDS.extend([f"{_name}_min", f"{_name}_max"])
ASSIGNMENT_DIAGNOSTIC_FIELDS.extend(
    [
        "overlap_count_sum",
        "k_zero_update_count",
        "exact_k_mismatch_count",
        *COUNT_METRICS.keys(),
        "normalized_weight_sum_abs_diff_max",
    ]
)


class AssignmentDiagnosticsWindow:
    """Aggregate non-overlapping assignment diagnostics without using RNG."""

    def __init__(self, window_size: int = 1_000) -> None:
        self.window_size = max(1, int(window_size))
        self.window_index = 0
        self.last_update = 0
        self._reset_window()

    def _reset_window(self) -> None:
        self.update_start = 0
        self.update_end = 0
        self.count = 0
        self.sums = {name: 0.0 for name in TRACKED_METRICS}
        self.minimums = {
            name: float("inf")
            for name in (
                "context_eligible_count",
                "main_positive_count",
                "assigned_positive_count",
            )
        }
        self.maximums = {
            name: float("-inf")
            for name in (
                "context_eligible_count",
                "main_positive_count",
                "assigned_positive_count",
            )
        }
        self.count_sums = {name: 0 for name in COUNT_METRICS}
        self.k_zero_update_count = 0
        self.exact_k_mismatch_count = 0
        self.normalized_weight_sum_abs_diff_max = 0.0

    @staticmethod
    def _required_value(metrics: Dict[str, float], key: str) -> float:
        if key not in metrics:
            raise KeyError(f"missing assignment diagnostic metric {key!r}")
        value = float(metrics[key])
        if not np.isfinite(value):
            raise RuntimeError(
                f"non-finite assignment diagnostic metric {key!r}: {value}"
            )
        return value

    @staticmethod
    def _assert_equal(name: str, left: float, right: float) -> None:
        if not np.isclose(left, right, rtol=0.0, atol=1e-7):
            raise RuntimeError(
                f"assignment diagnostic invariant failed for {name}: "
                f"{left} != {right}"
            )

    def update(
        self,
        update_index: int,
        metrics: Dict[str, float],
    ) -> Dict[str, float] | None:
        index = int(update_index)
        if index <= 0:
            raise ValueError("update_index must be positive")
        if self.last_update and index != self.last_update + 1:
            raise RuntimeError(
                f"assignment diagnostics are not contiguous: "
                f"last={self.last_update}, next={index}"
            )

        values = {
            name: self._required_value(metrics, key)
            for name, key in TRACKED_METRICS.items()
        }
        counts = {
            name: int(round(self._required_value(metrics, key)))
            for name, key in COUNT_METRICS.items()
        }

        eligible = int(round(values["context_eligible_count"]))
        main = int(round(values["main_positive_count"]))
        assigned = int(round(values["assigned_positive_count"]))
        exact_k_mismatch = int(main != assigned)
        if main > eligible:
            raise RuntimeError(
                f"assignment diagnostic invariant failed: k={main} exceeds |E|={eligible}"
            )
        if exact_k_mismatch:
            raise RuntimeError(
                f"assignment diagnostic invariant failed: assigned={assigned}, main={main}"
            )
        if any(value != 0 for value in counts.values()):
            raise RuntimeError(
                f"assignment diagnostics reported an invariant failure: {counts}"
            )

        for family in ("raw_weight", "normalized_weight"):
            for statistic in ("mean", "std", "ESS"):
                self._assert_equal(
                    f"{family}_{statistic}",
                    values[f"reference_{family}_{statistic}"],
                    values[f"assigned_{family}_{statistic}"],
                )
        self._assert_equal(
            "normalized_weight_sum",
            values["reference_normalized_weight_sum"],
            values["assigned_normalized_weight_sum"],
        )

        if self.count == 0:
            self.update_start = index
        self.update_end = index
        self.last_update = index
        self.count += 1
        for name, value in values.items():
            self.sums[name] += value
        for name in self.minimums:
            self.minimums[name] = min(self.minimums[name], values[name])
            self.maximums[name] = max(self.maximums[name], values[name])
        for name, value in counts.items():
            self.count_sums[name] += value
        self.k_zero_update_count += int(main == 0)
        self.exact_k_mismatch_count += exact_k_mismatch
        self.normalized_weight_sum_abs_diff_max = max(
            self.normalized_weight_sum_abs_diff_max,
            values["normalized_weight_sum_abs_diff"],
        )

        if self.count == self.window_size:
            return self._emit(complete=True)
        return None

    def _emit(self, complete: bool) -> Dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("cannot emit an empty assignment diagnostic window")
        self.window_index += 1
        denominator = float(self.count)
        row: Dict[str, float] = {
            "window_index": float(self.window_index),
            "update_start": float(self.update_start),
            "update_end": float(self.update_end),
            "updates": float(self.count),
            "complete_window": float(bool(complete)),
        }
        for name, total in self.sums.items():
            row[f"{name}_mean"] = float(total / denominator)
        for name in self.minimums:
            row[f"{name}_min"] = float(self.minimums[name])
            row[f"{name}_max"] = float(self.maximums[name])
        row["overlap_count_sum"] = float(self.sums["overlap_count"])
        row["k_zero_update_count"] = float(self.k_zero_update_count)
        row["exact_k_mismatch_count"] = float(self.exact_k_mismatch_count)
        for name, value in self.count_sums.items():
            row[name] = float(value)
        row["normalized_weight_sum_abs_diff_max"] = float(
            self.normalized_weight_sum_abs_diff_max
        )
        self._reset_window()
        return row

    def flush_partial(self) -> Dict[str, float] | None:
        if self.count == 0:
            return None
        return self._emit(complete=False)

    def state_dict(self) -> Dict[str, object]:
        return {
            "window_size": self.window_size,
            "window_index": self.window_index,
            "last_update": self.last_update,
            "update_start": self.update_start,
            "update_end": self.update_end,
            "count": self.count,
            "sums": dict(self.sums),
            "minimums": dict(self.minimums),
            "maximums": dict(self.maximums),
            "count_sums": dict(self.count_sums),
            "k_zero_update_count": self.k_zero_update_count,
            "exact_k_mismatch_count": self.exact_k_mismatch_count,
            "normalized_weight_sum_abs_diff_max": (
                self.normalized_weight_sum_abs_diff_max
            ),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if int(state.get("window_size", -1)) != self.window_size:
            raise RuntimeError(
                "assignment diagnostic window size differs from checkpoint"
            )
        self.window_index = int(state["window_index"])
        self.last_update = int(state["last_update"])
        self.update_start = int(state["update_start"])
        self.update_end = int(state["update_end"])
        self.count = int(state["count"])
        self.sums = {
            name: float(dict(state["sums"])[name])
            for name in TRACKED_METRICS
        }
        self.minimums = {
            name: float(dict(state["minimums"])[name])
            for name in self.minimums
        }
        self.maximums = {
            name: float(dict(state["maximums"])[name])
            for name in self.maximums
        }
        self.count_sums = {
            name: int(dict(state["count_sums"])[name])
            for name in COUNT_METRICS
        }
        self.k_zero_update_count = int(state["k_zero_update_count"])
        self.exact_k_mismatch_count = int(state["exact_k_mismatch_count"])
        self.normalized_weight_sum_abs_diff_max = float(
            state["normalized_weight_sum_abs_diff_max"]
        )

