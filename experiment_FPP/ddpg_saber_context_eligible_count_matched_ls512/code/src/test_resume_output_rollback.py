#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import torch

from metacog_lite_v0.assignment_diagnostics import (
    ASSIGNMENT_DIAGNOSTIC_FIELDS,
)
from metacog_lite_v0.io_utils import MetricsCSVWriter, MetricsJSONLWriter
from metacog_lite_v0.trainer import (
    _required_int,
    canonicalize_resume_checkpoint,
    load_checkpoint,
    rollback_resume_outputs,
)


METRIC_FIELDS = [
    "step",
    "episode",
    "eval/return_mean",
    "eval/env_success_mean",
]


def _metric_row(
    step: int,
    episode: int,
    *,
    eval_return: float | str = "",
    eval_success: float | str = "",
) -> dict[str, object]:
    return {
        "step": step,
        "episode": episode,
        "eval/return_mean": eval_return,
        "eval/env_success_mean": eval_success,
    }


def _assignment_row(
    index: int,
    update_start: int,
    update_end: int,
) -> dict[str, object]:
    updates = update_end - update_start + 1
    row: dict[str, object] = {field: "" for field in ASSIGNMENT_DIAGNOSTIC_FIELDS}
    row.update(
        {
            "window_index": index,
            "update_start": update_start,
            "update_end": update_end,
            "updates": updates,
            "complete_window": int(updates == 1_000),
            "exact_k_mismatch_count": 0,
            "raw_histogram_mismatch_count": 0,
            "normalized_histogram_mismatch_count": 0,
            "invariant_failure_count": 0,
        }
    )
    return row


def _write_metrics(run_dir: Path) -> None:
    csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", METRIC_FIELDS)
    jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")
    records = [
        ("episode", _metric_row(50, 1)),
        ("episode", _metric_row(100, 2)),
        (
            "eval",
            _metric_row(100, 2, eval_return=-1.0, eval_success=0.5),
        ),
        (
            "eval",
            _metric_row(200, 4, eval_return=-0.5, eval_success=0.75),
        ),
        # Simulate a prior broken resume that appended an older future row.
        # Ordering outside the retained checkpoint prefix is irrelevant.
        ("episode", _metric_row(150, 3)),
    ]
    for record_type, row in records:
        csv_writer.write(row)
        jsonl_writer.write({"type": record_type, **row})
    csv_writer.close()
    jsonl_writer.close()
    with (run_dir / "metrics.csv").open("ab") as handle:
        handle.write(b"malformed,trailing,partial")
    with (run_dir / "metrics.jsonl").open("ab") as handle:
        handle.write(b'{"type":"episode"')


def _write_assignment_diagnostics(run_dir: Path) -> None:
    writer = MetricsCSVWriter(
        run_dir / "assignment_diagnostics_1000.csv",
        ASSIGNMENT_DIAGNOSTIC_FIELDS,
    )
    integer_like_fields = (
        "window_index",
        "update_start",
        "update_end",
        "updates",
        "complete_window",
        "exact_k_mismatch_count",
        "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch_count",
        "invariant_failure_count",
    )
    for row in (
        _assignment_row(1, 1, 1_000),
        _assignment_row(2, 1_001, 2_000),
    ):
        # Production diagnostics are aggregated through floating-point arrays,
        # so integer-like CSV cells are emitted as e.g. "1.0" and "1000.0".
        for field in integer_like_fields:
            row[field] = float(row[field])
        writer.write(row)
    writer.close()
    with (run_dir / "assignment_diagnostics_1000.csv").open("ab") as handle:
        handle.write(b"malformed,trailing,partial")


def _checkpoint_payload() -> dict[str, object]:
    return {
        "global_step": 100,
        "episode_idx": 2,
        "eval_history_env": [(100, 0.5)],
        "assignment_diagnostics_state": {
            "window_index": 1,
            "last_update": 1_500,
            "count": 500,
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_advanced_outputs_rollback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        _write_metrics(run_dir)
        _write_assignment_diagnostics(run_dir)
        run_log = run_dir / "run.log"
        run_log.write_text("preserve this append-only log\n", encoding="utf-8")

        (run_dir / "best_checkpoint.json").write_text(
            json.dumps({"step": 200}) + "\n",
            encoding="utf-8",
        )
        torch.save({"global_step": 200}, run_dir / "best_checkpoint.pt")
        torch.save({"global_step": 100}, run_dir / "checkpoint_step_100.pt")
        torch.save({"global_step": 200}, run_dir / "checkpoint_step_200.pt")

        report = rollback_resume_outputs(
            run_dir,
            _checkpoint_payload(),
            METRIC_FIELDS,
            strict_assignment=True,
            assignment_window_size=1_000,
        )
        assert report["metrics_rows"] == 3
        assert report["eval_rows"] == 1
        assert report["assignment_rows"] == 1
        assert report["assignment_emitted_update_boundary"] == 1_000
        assert report["dropped_csv_partial"] == 1
        assert report["dropped_jsonl_partial"] == 1
        assert report["dropped_assignment_partial"] == 1
        assert len(report["quarantined_checkpoint_evidence"]) == 3
        assert run_log.read_text(encoding="utf-8") == (
            "preserve this append-only log\n"
        )
        assert not (run_dir / "best_checkpoint.json").exists()
        assert not (run_dir / "best_checkpoint.pt").exists()
        assert (run_dir / "checkpoint_step_100.pt").exists()
        assert not (run_dir / "checkpoint_step_200.pt").exists()

        metric_rows = _read_csv(run_dir / "metrics.csv")
        assert [(row["step"], row["episode"]) for row in metric_rows] == [
            ("50", "1"),
            ("100", "2"),
            ("100", "2"),
        ]
        jsonl_rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["type"] for row in jsonl_rows] == [
            "episode",
            "episode",
            "eval",
        ]
        diagnostics_path = run_dir / "assignment_diagnostics_1000.csv"
        assert [
            _required_int(
                row["window_index"], path=diagnostics_path,
                field="window_index", row=index,
            )
            for index, row in enumerate(_read_csv(diagnostics_path), start=1)
        ] == [1]

        # Reopening the production append writers after rollback must continue
        # the validated prefixes without retaining duplicated future records.
        csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", METRIC_FIELDS)
        jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")
        assignment_writer = MetricsCSVWriter(
            run_dir / "assignment_diagnostics_1000.csv",
            ASSIGNMENT_DIAGNOSTIC_FIELDS,
        )
        replacement_episode = _metric_row(150, 3)
        replacement_eval = _metric_row(
            200,
            3,
            eval_return=-0.25,
            eval_success=0.6,
        )
        csv_writer.write(replacement_episode)
        jsonl_writer.write({"type": "episode", **replacement_episode})
        csv_writer.write(replacement_eval)
        jsonl_writer.write({"type": "eval", **replacement_eval})
        assignment_writer.write(_assignment_row(2, 1_001, 2_000))
        csv_writer.close()
        jsonl_writer.close()
        assignment_writer.close()

        metric_rows = _read_csv(run_dir / "metrics.csv")
        identities = [
            (
                "eval" if row["eval/return_mean"] else "episode",
                int(row["step"]),
                int(row["episode"]),
            )
            for row in metric_rows
        ]
        assert identities == [
            ("episode", 50, 1),
            ("episode", 100, 2),
            ("eval", 100, 2),
            ("episode", 150, 3),
            ("eval", 200, 3),
        ]
        assert len(identities) == len(set(identities))
        assignment_indices = [
            _required_int(
                row["window_index"], path=diagnostics_path,
                field="window_index", row=index,
            )
            for index, row in enumerate(_read_csv(diagnostics_path), start=1)
        ]
        assert assignment_indices == [1, 2]


def _assert_complete_malformed_record_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", METRIC_FIELDS)
        jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")
        row = _metric_row(50, 1)
        csv_writer.write(row)
        jsonl_writer.write({"type": "episode", **row})
        csv_writer.close()
        jsonl_writer.close()
        # This complete record occurs inside the would-be retained prefix. A
        # resume must stop instead of guessing how to repair it.
        with (run_dir / "metrics.csv").open("ab") as handle:
            handle.write(b"not-an-int,2,,\n")
        payload = {
            "global_step": 100,
            "episode_idx": 1,
            "eval_history_env": [],
        }
        try:
            rollback_resume_outputs(
                run_dir,
                payload,
                METRIC_FIELDS,
                strict_assignment=False,
                assignment_window_size=1_000,
            )
        except RuntimeError as exc:
            assert "invalid integer step" in str(exc)
        else:
            raise AssertionError("complete malformed CSV record was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        csv_writer = MetricsCSVWriter(run_dir / "metrics.csv", METRIC_FIELDS)
        jsonl_writer = MetricsJSONLWriter(run_dir / "metrics.jsonl")
        row = _metric_row(50, 1)
        csv_writer.write(row)
        jsonl_writer.write({"type": "episode", **row})
        csv_writer.close()
        jsonl_writer.close()
        with (run_dir / "metrics.jsonl").open("ab") as handle:
            handle.write(b'{"type":"episode"\n')
        try:
            rollback_resume_outputs(
                run_dir,
                {
                    "global_step": 100,
                    "episode_idx": 1,
                    "eval_history_env": [],
                },
                METRIC_FIELDS,
                strict_assignment=False,
                assignment_window_size=1_000,
            )
        except RuntimeError as exc:
            assert "malformed complete JSON record" in str(exc)
        else:
            raise AssertionError("complete malformed JSONL record was accepted")


def _assert_integer_like_numeric_parser() -> None:
    path = Path("assignment_diagnostics_1000.csv")
    assert _required_int("1.0", path=path, field="window_index", row=1) == 1
    assert _required_int("1000.000", path=path, field="updates", row=1) == 1_000
    assert _required_int("1e3", path=path, field="update_end", row=1) == 1_000
    for value in ("1.5", "nan", "NaN", "inf", "-inf"):
        try:
            _required_int(value, path=path, field="updates", row=1)
        except RuntimeError as exc:
            assert "invalid integer" in str(exc)
        else:
            raise AssertionError(f"non-integral/non-finite value accepted: {value}")


def _assert_custom_resume_becomes_canonical() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        selected_path = run_dir / "checkpoint_step_100.pt"
        canonical_path = run_dir / "checkpoint.pt"
        selected_payload = {
            "global_step": 100,
            "marker": "selected-authoritative-state",
        }
        torch.save(selected_payload, selected_path)
        torch.save(
            {"global_step": 200, "marker": "stale-future-state"},
            canonical_path,
        )

        report = canonicalize_resume_checkpoint(
            run_dir,
            selected_path,
            selected_payload,
        )
        assert report["canonical_checkpoint_rewritten"] == 1
        quarantined = Path(report["quarantined_canonical_checkpoint"])
        assert quarantined.is_file()
        assert load_checkpoint(quarantined)["global_step"] == 200

        # If the process stops before another periodic save, default resume is
        # still anchored to the explicitly selected checkpoint state.
        canonical_payload = load_checkpoint(canonical_path)
        assert canonical_payload["global_step"] == 100
        assert canonical_payload["marker"] == "selected-authoritative-state"
        assert selected_path.is_file()


def main() -> int:
    _assert_integer_like_numeric_parser()
    _assert_advanced_outputs_rollback()
    _assert_complete_malformed_record_fails_closed()
    _assert_custom_resume_becomes_canonical()

    # The checkpoint contract remains exact at the requested 1M endpoint.
    assert 45_001 + (1_000_000 - 50_000) == 995_001
    print("test_resume_output_rollback.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
