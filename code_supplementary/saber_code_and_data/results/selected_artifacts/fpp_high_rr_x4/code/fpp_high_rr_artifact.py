#!/usr/bin/env python3
"""Shared, fail-closed helpers for the FPP DDPG/TD3 High-RR artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_ID = "fpp_ddpg_td3_her_high_rr_x4_1m"
SCHEMA_VERSION = 1
SEEDS = tuple(range(1, 6))
TOTAL_STEPS = 1_000_000
FULL_GRID = tuple(range(10_000, TOTAL_STEPS + 1, 10_000))
TABLE_GRID = tuple(
    step for step in range(50_000, TOTAL_STEPS + 1, 50_000)
    if step != 800_000
)
FINAL_QUARTER_GRID = tuple(step for step in TABLE_GRID if step >= 750_000)
SUCCESS_FIELD = "eval/env_success_mean"
RUNTIME_FIELD = "runtime/total_sec"


class ArtifactError(RuntimeError):
    """Raised when a raw run or packaged artifact fails a required gate."""


@dataclass(frozen=True)
class RuntimeSegment:
    index: int
    first_row: int
    last_row: int
    first_step: int
    last_step: int
    first_runtime_seconds: float
    active_seconds: float


@dataclass(frozen=True)
class RawMetrics:
    curve: dict[int, float]
    segments: tuple[RuntimeSegment, ...]
    row_count: int
    evaluation_row_count: int
    nul_padding_bytes: int
    nul_padding_rows: int


def finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ArtifactError(f"{path}: non-standard JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path}: expected a JSON object")
    return payload


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in fields
                }
            )


def parse_raw_metrics(path: Path) -> RawMetrics:
    if not path.is_file():
        raise ArtifactError(f"missing raw metrics: {path}")
    curve: dict[int, float] = {}
    row_count = 0
    evaluation_row_count = 0
    nul_padding_bytes = 0
    nul_padding_rows = 0
    segments: list[RuntimeSegment] = []
    segment_index = 1
    segment_first_row: int | None = None
    segment_last_row: int | None = None
    segment_first_step: int | None = None
    segment_last_step: int | None = None
    segment_first_runtime: float | None = None
    segment_max_runtime: float | None = None
    previous_runtime: float | None = None

    def close_segment() -> None:
        nonlocal segment_index
        if (
            segment_first_row is None
            or segment_last_row is None
            or segment_first_step is None
            or segment_last_step is None
            or segment_first_runtime is None
            or segment_max_runtime is None
        ):
            return
        segments.append(
            RuntimeSegment(
                index=segment_index,
                first_row=segment_first_row,
                last_row=segment_last_row,
                first_step=segment_first_step,
                last_step=segment_last_step,
                first_runtime_seconds=segment_first_runtime,
                active_seconds=segment_max_runtime,
            )
        )
        segment_index += 1

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", SUCCESS_FIELD, RUNTIME_FIELD}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ArtifactError(f"{path}: missing columns {sorted(missing)}")
        for row_count, row in enumerate(reader, start=1):
            row_nul_bytes = sum(
                value.count("\0")
                for value in row.values()
                if isinstance(value, str)
            )
            if row_nul_bytes:
                non_step_nul = [
                    key
                    for key, value in row.items()
                    if key != "step"
                    and isinstance(value, str)
                    and "\0" in value
                ]
                if non_step_nul:
                    raise ArtifactError(
                        f"{path}: NUL padding outside step field at row "
                        f"{row_count}: {non_step_nul}"
                    )
                nul_padding_bytes += row_nul_bytes
                nul_padding_rows += 1
            raw_step = row["step"]
            step_text = raw_step.lstrip("\0")
            if "\0" in step_text:
                raise ArtifactError(
                    f"{path}: NUL bytes are not leading step padding at "
                    f"row {row_count}"
                )
            if not step_text:
                raise ArtifactError(
                    f"{path}: empty step after NUL removal at row {row_count}"
                )
            step = int(float(step_text))
            success = finite(row.get(SUCCESS_FIELD))
            if success is not None:
                if not 0.0 <= success <= 1.0:
                    raise ArtifactError(
                        f"{path}: invalid success={success} at step={step}"
                    )
                if step in curve and curve[step] != success:
                    raise ArtifactError(
                        f"{path}: conflicting evaluation at step={step}"
                    )
                if step not in curve:
                    evaluation_row_count += 1
                curve[step] = success

            runtime = finite(row.get(RUNTIME_FIELD))
            if runtime is None:
                continue
            if runtime < 0.0:
                raise ArtifactError(f"{path}: negative runtime at row {row_count}")
            if (
                previous_runtime is not None
                and runtime + 1.0 < previous_runtime
            ):
                close_segment()
                segment_first_row = row_count
                segment_first_step = step
                segment_first_runtime = runtime
                segment_max_runtime = runtime
            elif segment_first_row is None:
                segment_first_row = row_count
                segment_first_step = step
                segment_first_runtime = runtime
                segment_max_runtime = runtime
            else:
                segment_max_runtime = max(
                    float(segment_max_runtime), runtime
                )
            segment_last_row = row_count
            segment_last_step = step
            previous_runtime = runtime

    close_segment()
    if not curve:
        raise ArtifactError(f"{path}: no finite evaluations")
    if not segments:
        raise ArtifactError(f"{path}: no finite runtime records")
    return RawMetrics(
        curve=curve,
        segments=tuple(segments),
        row_count=row_count,
        evaluation_row_count=evaluation_row_count,
        nul_padding_bytes=nul_padding_bytes,
        nul_padding_rows=nul_padding_rows,
    )


def read_compact_curve(path: Path) -> dict[int, float]:
    curve: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", SUCCESS_FIELD}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ArtifactError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            value = finite(row.get(SUCCESS_FIELD))
            if value is None:
                continue
            step = int(float(row["step"]))
            if step in curve:
                raise ArtifactError(f"{path}: duplicate evaluation at {step}")
            if not 0.0 <= value <= 1.0:
                raise ArtifactError(f"{path}: invalid success at {step}")
            curve[step] = value
    if not curve:
        raise ArtifactError(f"{path}: no finite evaluations")
    return curve


def normalized_auc(curve: Mapping[int, float], grid: Sequence[int]) -> float:
    if len(grid) < 2:
        raise ArtifactError("AUC grid must have at least two points")
    missing = [step for step in grid if step not in curve]
    if missing:
        raise ArtifactError(f"AUC grid is missing exact steps {missing[:8]}")
    area = sum(
        0.5 * (curve[left] + curve[right]) * (right - left)
        for left, right in zip(grid, grid[1:])
    )
    return area / (grid[-1] - grid[0])


def first_hit(curve: Mapping[int, float], grid: Sequence[int]) -> int | None:
    return next((step for step in grid if curve[step] >= 0.9), None)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ArtifactError("cannot aggregate an empty value sequence")
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def runtime_total(segments: Sequence[RuntimeSegment]) -> float:
    total = sum(segment.active_seconds for segment in segments)
    if total <= 0.0:
        raise ArtifactError("active runtime total must be positive")
    return total


def expected_native_grid(backbone: str, seed: int) -> tuple[int, ...]:
    if backbone == "DDPG":
        return FULL_GRID
    if backbone == "TD3" and seed in {1, 3, 4}:
        return tuple(step for step in FULL_GRID if step != 800_000)
    if backbone == "TD3":
        return FULL_GRID
    raise ArtifactError(f"unknown backbone {backbone!r}")


def sanitize_json(
    value: Any, replacements: Sequence[tuple[str, str]]
) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_json(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json(item, replacements) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        for before, after in replacements:
            if before:
                value = value.replace(before, after)
    return value


def file_record(
    path: Path,
    logical_path: str,
    *,
    role: str,
    packaged_path: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "logical_path": logical_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if packaged_path is not None:
        record["packaged_path"] = packaged_path
    return record
