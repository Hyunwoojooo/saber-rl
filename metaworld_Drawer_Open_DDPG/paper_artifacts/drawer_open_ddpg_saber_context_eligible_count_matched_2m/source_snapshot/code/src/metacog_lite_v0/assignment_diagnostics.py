from __future__ import annotations

from typing import Dict

import numpy as np


TRACKED_METRICS = {
    "context_eligible_count": "strict/context_eligible_count",
    "main_positive_count": "strict/main_positive_count",
    "control_positive_count": "strict/control_positive_count",
    "context_eligible_rate": "strict/context_eligible_rate",
    "main_positive_rate": "strict/main_positive_rate",
    "control_positive_rate": "strict/control_positive_rate",
    "main_control_overlap_count": "strict/main_control_overlap_count",
    "main_control_overlap_rate": "strict/main_control_overlap_rate",
    "raw_weight_mean": "strict/raw_weight_mean",
    "normalized_weight_mean": "strict/normalized_weight_mean",
    "normalized_weight_min": "strict/normalized_weight_min",
    "normalized_weight_max": "strict/normalized_weight_max",
    "support_lookup_sec_per_update": "runtime/support_lookup_sec_per_update",
    "strict_assignment_sec_per_update": "runtime/strict_assignment_sec_per_update",
}

COUNT_METRICS = {
    "control_selected_outside_eligible": (
        "strict/control_selected_outside_eligible"
    ),
    "count_equality_violation": "strict/count_equality_violation",
    "raw_histogram_mismatch": "strict/raw_histogram_mismatch",
    "fallback_sampling_used": "strict/fallback_sampling_used",
    "replacement_sampling_used": "strict/replacement_sampling_used",
}

CORE_COUNT_METRICS = (
    "context_eligible_count",
    "main_positive_count",
    "control_positive_count",
)

EXTREMA_METRICS = (
    *CORE_COUNT_METRICS,
    "raw_weight_mean",
    "normalized_weight_mean",
    "normalized_weight_min",
    "normalized_weight_max",
)

STRICT_DIAGNOSTIC_FIELDS = [
    "window_index",
    "update_start",
    "update_end",
    "updates",
    "complete_window",
]
for _name in TRACKED_METRICS:
    STRICT_DIAGNOSTIC_FIELDS.append(f"{_name}_mean")
for _name in EXTREMA_METRICS:
    STRICT_DIAGNOSTIC_FIELDS.extend([f"{_name}_min", f"{_name}_max"])
STRICT_DIAGNOSTIC_FIELDS.extend(
    [
        "main_control_overlap_count_sum",
        "k_zero_update_count",
        *[f"{name}_sum" for name in COUNT_METRICS],
        "normalized_weight_mean_abs_error_max",
    ]
)


class StrictAssignmentDiagnosticsWindow:
    """Checkpointable non-overlapping strict-control diagnostic windows."""

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
        self.minimums = {name: float("inf") for name in EXTREMA_METRICS}
        self.maximums = {name: float("-inf") for name in EXTREMA_METRICS}
        self.count_sums = {name: 0 for name in COUNT_METRICS}
        self.k_zero_update_count = 0
        self.normalized_weight_mean_abs_error_max = 0.0

    @staticmethod
    def _required_value(metrics: Dict[str, float], key: str) -> float:
        if key not in metrics:
            raise KeyError(f"missing strict diagnostic metric {key!r}")
        value = float(metrics[key])
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite strict diagnostic metric {key!r}: {value}")
        return value

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
                "strict diagnostics are not contiguous: "
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
        control = int(round(values["control_positive_count"]))

        if main > eligible:
            raise RuntimeError(
                f"strict diagnostic invariant failed: |M|={main} exceeds |E|={eligible}"
            )
        if control > eligible:
            raise RuntimeError(
                f"strict diagnostic invariant failed: |C|={control} exceeds |E|={eligible}"
            )
        if main != control:
            raise RuntimeError(
                f"strict diagnostic invariant failed: |M|={main}, |C|={control}"
            )
        if any(value != 0 for value in counts.values()):
            raise RuntimeError(
                f"strict diagnostics reported an invariant failure: {counts}"
            )
        if values["normalized_weight_min"] <= 0.0:
            raise RuntimeError("strict diagnostic invariant failed: weight is not positive")
        normalized_error = abs(values["normalized_weight_mean"] - 1.0)
        if normalized_error > 2e-6:
            raise RuntimeError(
                "strict diagnostic invariant failed: normalized weight mean "
                f"is {values['normalized_weight_mean']}"
            )
        if values["support_lookup_sec_per_update"] < 0.0:
            raise RuntimeError("strict diagnostic support lookup time is negative")
        if values["strict_assignment_sec_per_update"] < 0.0:
            raise RuntimeError("strict diagnostic assignment time is negative")

        if self.count == 0:
            self.update_start = index
        self.update_end = index
        self.last_update = index
        self.count += 1
        for name, value in values.items():
            self.sums[name] += value
        for name in EXTREMA_METRICS:
            self.minimums[name] = min(self.minimums[name], values[name])
            self.maximums[name] = max(self.maximums[name], values[name])
        for name, value in counts.items():
            self.count_sums[name] += value
        self.k_zero_update_count += int(main == 0)
        self.normalized_weight_mean_abs_error_max = max(
            self.normalized_weight_mean_abs_error_max,
            normalized_error,
        )

        if self.count == self.window_size:
            return self._emit(complete=True)
        return None

    def _emit(self, complete: bool) -> Dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("cannot emit an empty strict diagnostic window")
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
        for name in EXTREMA_METRICS:
            row[f"{name}_min"] = float(self.minimums[name])
            row[f"{name}_max"] = float(self.maximums[name])
        row["main_control_overlap_count_sum"] = float(
            self.sums["main_control_overlap_count"]
        )
        row["k_zero_update_count"] = float(self.k_zero_update_count)
        for name, value in self.count_sums.items():
            row[f"{name}_sum"] = float(value)
        row["normalized_weight_mean_abs_error_max"] = float(
            self.normalized_weight_mean_abs_error_max
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
            "normalized_weight_mean_abs_error_max": (
                self.normalized_weight_mean_abs_error_max
            ),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if int(state.get("window_size", -1)) != self.window_size:
            raise RuntimeError("strict diagnostic window size differs from checkpoint")
        self.window_index = int(state["window_index"])
        self.last_update = int(state["last_update"])
        self.update_start = int(state["update_start"])
        self.update_end = int(state["update_end"])
        self.count = int(state["count"])
        self.sums = {
            name: float(dict(state["sums"])[name]) for name in TRACKED_METRICS
        }
        self.minimums = {
            name: float(dict(state["minimums"])[name])
            for name in EXTREMA_METRICS
        }
        self.maximums = {
            name: float(dict(state["maximums"])[name])
            for name in EXTREMA_METRICS
        }
        self.count_sums = {
            name: int(dict(state["count_sums"])[name]) for name in COUNT_METRICS
        }
        self.k_zero_update_count = int(state["k_zero_update_count"])
        self.normalized_weight_mean_abs_error_max = float(
            state["normalized_weight_mean_abs_error_max"]
        )


# Compatibility aliases keep the validated FPP trainer transplant minimal.
AssignmentDiagnosticsWindow = StrictAssignmentDiagnosticsWindow
ASSIGNMENT_DIAGNOSTIC_FIELDS = STRICT_DIAGNOSTIC_FIELDS
