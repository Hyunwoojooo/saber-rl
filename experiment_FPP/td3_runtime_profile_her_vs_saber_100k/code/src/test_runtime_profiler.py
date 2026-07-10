from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from metacog_lite_v0.runtime_profiler import RuntimeProfiler


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="td3-runtime-profiler-") as tmp:
        run_dir = Path(tmp)
        profiler = RuntimeProfiler(
            enabled=True,
            run_dir=run_dir,
            method="td3_her",
            seed=1,
            device=torch.device("cpu"),
            total_steps=20,
            profile_start_step=0,
            profile_end_step=20,
            profile_sample_every_updates=5,
            profile_sync_cuda=True,
            profile_events_jsonl=True,
        )
        for step in range(1, 21):
            profiler.record_raw("action_selection", 0.001, step=step)
            profiler.record_group("Interaction", 0.001, step=step)
            profiler.note_update(step=step, actor_updated=step % 2 == 0, sampled=step % 5 == 0)
            profiler.note_env_step(step=step, training_wall_sec=0.002)
        profiler.add_eval(0.1, step=10)
        profiler.add_logging(0.01, step=10)
        summary = profiler.finalize(total_wall_sec=0.2, total_updates=20)

        assert summary["profiled_env_steps"] == 20
        assert summary["profiled_updates"] == 20
        assert summary["profiled_actor_updates"] == 10
        assert summary["actor_updates"] == 10
        assert summary["unaccounted_other_sec"] >= 0.0
        assert 0.95 <= summary["component_accounting_ratio"] <= 1.05
        assert summary["components"]["Eval/logging amortized"]["total_sec"] == 0.11
        assert all("Other/unaccounted" in window["components"] for window in summary["subwindows"].values())

        event_path = run_dir / "runtime_profile_events.jsonl"
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
        assert len(events) == 8
        assert {int(event["step"]) for event in events} == {5, 10, 15, 20}

        disabled_dir = run_dir / "disabled"
        disabled = RuntimeProfiler(
            enabled=False,
            run_dir=disabled_dir,
            method="td3_her",
            seed=1,
            device=torch.device("cpu"),
            total_steps=20,
            profile_start_step=0,
            profile_end_step=20,
            profile_sample_every_updates=5,
            profile_sync_cuda=True,
            profile_events_jsonl=True,
        )
        disabled.record_raw("action_selection", 0.001, step=5)
        assert disabled.finalize(total_wall_sec=0.0, total_updates=0) == {}
        assert not (disabled_dir / "runtime_profile_events.jsonl").exists()

    print("runtime profiler test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
