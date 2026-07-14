from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch


PROFILE_SCHEMA_VERSION = "1.0"

PAPER_GROUPS = (
    "Interaction",
    "Replay + HER",
    "Batch preparation",
    "Support and weighting",
    "Critic update",
    "Actor + target update",
    "Eval/logging amortized",
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _hardware_metadata(device: torch.device) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": props.name,
                "cuda_total_memory_bytes": int(props.total_memory),
                "cuda_capability": [int(props.major), int(props.minor)],
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return metadata


class SampledUpdateTimer:
    """Times one sampled optimizer update without affecting ordinary updates."""

    def __init__(self, device: torch.device, sync_cuda: bool) -> None:
        self.device = device
        self.use_cuda_events = bool(device.type == "cuda" and sync_cuda and torch.cuda.is_available())
        self._cuda_sections: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_sections: Dict[str, float] = {}
        self._administrative_sec = 0.0

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if self.use_cuda_events:
            t_admin = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._administrative_sec += time.perf_counter() - t_admin
            try:
                yield
            finally:
                t_admin = time.perf_counter()
                end.record()
                self._cuda_sections.append((str(name), start, end))
                self._administrative_sec += time.perf_counter() - t_admin
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_sections[str(name)] = self._cpu_sections.get(str(name), 0.0) + (
                time.perf_counter() - started
            )

    def finish(self) -> tuple[Dict[str, float], Dict[str, float]]:
        if not self.use_cuda_events:
            return dict(self._cpu_sections), {
                "profiler_administrative_sec": 0.0,
                "profiled_cuda_sync_wait_sec": 0.0,
            }

        sync_started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        sync_wait_sec = time.perf_counter() - sync_started
        query_started = time.perf_counter()
        timings: Dict[str, float] = {}
        for name, start, end in self._cuda_sections:
            timings[name] = timings.get(name, 0.0) + float(start.elapsed_time(end)) / 1_000.0
        query_sec = time.perf_counter() - query_started

        # This second synchronization has no queued work and estimates the call overhead itself.
        noop_sync_started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        noop_sync_sec = time.perf_counter() - noop_sync_started
        return timings, {
            "profiler_administrative_sec": self._administrative_sec + query_sec + noop_sync_sec,
            "profiled_cuda_sync_wait_sec": sync_wait_sec,
        }


class RuntimeProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        run_dir: Path,
        method: str,
        seed: int,
        device: torch.device,
        total_steps: int,
        profile_start_step: int,
        profile_end_step: int,
        profile_sample_every_updates: int,
        profile_sync_cuda: bool,
        profile_events_jsonl: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.run_dir = Path(run_dir)
        self.method = str(method)
        self.seed = int(seed)
        self.device = device
        self.total_steps = int(total_steps)
        self.profile_start_step = max(0, int(profile_start_step))
        self.profile_end_step = min(int(total_steps), max(self.profile_start_step, int(profile_end_step)))
        self.profile_sample_every_updates = max(1, int(profile_sample_every_updates))
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self.profile_events_jsonl = bool(profile_events_jsonl)

        self.raw_samples: Dict[str, list[float]] = defaultdict(list)
        self.group_samples: Dict[str, list[float]] = defaultdict(list)
        self.raw_totals: Dict[str, float] = defaultdict(float)
        self.group_totals: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, list[float]] = defaultdict(list)
        self.training_wall_sec = 0.0
        self.eval_sec = 0.0
        self.logging_sec = 0.0
        self.checkpoint_sec = 0.0
        self.profiler_event_logging_sec = 0.0
        self.profiler_administrative_sec = 0.0
        self.profiled_cuda_sync_wait_sec = 0.0
        self.profiled_env_steps = 0
        self.profiled_updates = 0
        self.profiled_critic_updates = 0
        self.profiled_actor_updates = 0
        self.sampled_gpu_updates = 0
        self.sampled_actor_updates = 0

        self.subwindows = self._make_subwindows()
        self.window_data: Dict[str, Dict[str, Any]] = {
            name: {
                "start_step": start,
                "end_step": end,
                "training_wall_sec": 0.0,
                "eval_sec": 0.0,
                "logging_sec": 0.0,
                "checkpoint_sec": 0.0,
                "profiled_env_steps": 0,
                "profiled_updates": 0,
                "profiled_actor_updates": 0,
                "groups": defaultdict(list),
            }
            for name, start, end in self.subwindows
        }
        self._events_file = None
        if self.enabled and self.profile_events_jsonl:
            path = self.run_dir / "runtime_profile_events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._events_file = path.open("w", encoding="utf-8")

    def _make_subwindows(self) -> list[tuple[str, int, int]]:
        if self.profile_start_step == 20_000 and self.profile_end_step == 100_000:
            bounds = (20_000, 40_000, 70_000, 100_000)
        else:
            width = max(1, self.profile_end_step - self.profile_start_step)
            bounds = (
                self.profile_start_step,
                self.profile_start_step + width // 3,
                self.profile_start_step + (2 * width) // 3,
                self.profile_end_step,
            )
        return [(f"{bounds[i]}-{bounds[i + 1]}", bounds[i], bounds[i + 1]) for i in range(3)]

    def active(self, step: int) -> bool:
        # Treat profile_start_step as the completed warm-up boundary and include profile_end_step.
        return self.enabled and self.profile_start_step < int(step) <= self.profile_end_step

    def sampled_update(self, update_index: int, step: int) -> bool:
        return self.active(step) and int(update_index) % self.profile_sample_every_updates == 0

    def _window_name(self, step: int) -> Optional[str]:
        for name, start, end in self.subwindows:
            if start < int(step) <= end:
                return name
        return None

    def _write_event(self, payload: Dict[str, Any]) -> None:
        if self._events_file is None:
            return
        # Aggregates retain every measured CPU event. The JSONL is diagnostic
        # evidence only, so keep sampled updates/steps to avoid a multi-GB file.
        if not bool(payload.get("sampled", False)):
            sample_index = payload.get("update_index")
            if sample_index is None:
                sample_index = payload.get("step")
            if sample_index is not None and int(sample_index) % self.profile_sample_every_updates != 0:
                return
        started = time.perf_counter()
        self._events_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self.profiler_event_logging_sec += time.perf_counter() - started

    def record_raw(
        self,
        component: str,
        seconds: float,
        *,
        step: int,
        update_index: Optional[int] = None,
        sampled: bool = False,
    ) -> None:
        if not self.active(step):
            return
        value = max(0.0, float(seconds))
        self.raw_samples[str(component)].append(value)
        self.raw_totals[str(component)] += value
        self._write_event(
            {
                "type": "raw_component",
                "component": str(component),
                "seconds": value,
                "step": int(step),
                "update_index": None if update_index is None else int(update_index),
                "sampled": bool(sampled),
            }
        )

    def record_group(
        self,
        group: str,
        seconds: float,
        *,
        step: int,
        update_index: Optional[int] = None,
        sampled: bool = False,
    ) -> None:
        if not self.active(step):
            return
        value = max(0.0, float(seconds))
        self.group_samples[str(group)].append(value)
        self.group_totals[str(group)] += value
        window = self._window_name(step)
        if window is not None:
            self.window_data[window]["groups"][str(group)].append(value)
        self._write_event(
            {
                "type": "paper_group",
                "component": str(group),
                "seconds": value,
                "step": int(step),
                "update_index": None if update_index is None else int(update_index),
                "sampled": bool(sampled),
            }
        )

    @contextmanager
    def cpu_section(
        self,
        component: str,
        *,
        step: int,
        update_index: Optional[int] = None,
    ) -> Iterator[None]:
        if not self.active(step):
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_raw(component, time.perf_counter() - started, step=step, update_index=update_index)

    def note_env_step(self, *, step: int, training_wall_sec: float) -> None:
        if not self.active(step):
            return
        value = max(0.0, float(training_wall_sec))
        self.profiled_env_steps += 1
        self.training_wall_sec += value
        window = self._window_name(step)
        if window is not None:
            self.window_data[window]["profiled_env_steps"] += 1
            self.window_data[window]["training_wall_sec"] += value

    def note_update(self, *, step: int, actor_updated: bool, sampled: bool) -> None:
        if not self.active(step):
            return
        self.profiled_updates += 1
        self.profiled_critic_updates += 1
        self.profiled_actor_updates += int(bool(actor_updated))
        self.sampled_gpu_updates += int(bool(sampled))
        self.sampled_actor_updates += int(bool(sampled and actor_updated))
        window = self._window_name(step)
        if window is not None:
            self.window_data[window]["profiled_updates"] += 1
            self.window_data[window]["profiled_actor_updates"] += int(bool(actor_updated))

    def record_gauge(self, name: str, value: Any, *, step: int) -> None:
        if not self.active(step):
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            self.gauges[str(name)].append(numeric)

    def add_eval(self, seconds: float, *, step: int) -> None:
        if self.active(step):
            value = max(0.0, float(seconds))
            self.eval_sec += value
            window = self._window_name(step)
            if window is not None:
                self.window_data[window]["eval_sec"] += value

    def add_logging(self, seconds: float, *, step: int) -> None:
        if self.active(step):
            value = max(0.0, float(seconds))
            self.logging_sec += value
            window = self._window_name(step)
            if window is not None:
                self.window_data[window]["logging_sec"] += value

    def add_checkpoint(self, seconds: float, *, step: int) -> None:
        if self.active(step):
            value = max(0.0, float(seconds))
            self.checkpoint_sec += value
            window = self._window_name(step)
            if window is not None:
                self.window_data[window]["checkpoint_sec"] += value

    def add_update_profiler_overhead(self, diagnostics: Dict[str, float]) -> None:
        self.profiler_administrative_sec += max(0.0, float(diagnostics.get("profiler_administrative_sec", 0.0)))
        self.profiled_cuda_sync_wait_sec += max(0.0, float(diagnostics.get("profiled_cuda_sync_wait_sec", 0.0)))

    def _scaled_group_total(self, group: str, samples: list[float], counts: Dict[str, int]) -> tuple[float, int]:
        if not samples:
            return 0.0, 0
        if group == "Critic update":
            count = int(counts["profiled_updates"])
            return statistics.fmean(samples) * count, count
        if group == "Actor + target update":
            count = int(counts["profiled_actor_updates"])
            return statistics.fmean(samples) * count, count
        return float(sum(samples)), len(samples)

    def _component_stats(
        self,
        *,
        group: str,
        samples: list[float],
        total_sec: float,
        count: int,
        env_steps: int,
        updates: int,
        training_sec: float,
    ) -> Dict[str, Any]:
        return {
            "total_sec": float(total_sec),
            "count": int(count),
            "sample_count": int(len(samples)),
            "mean_ms": 1_000.0 * (statistics.fmean(samples) if samples else 0.0),
            "median_ms": 1_000.0 * (statistics.median(samples) if samples else 0.0),
            "p90_ms": 1_000.0 * _percentile(samples, 0.90),
            "ms_per_env_step": 1_000.0 * float(total_sec) / max(1, int(env_steps)),
            "ms_per_update": 1_000.0 * float(total_sec) / max(1, int(updates)),
            "training_share": float(total_sec) / max(1e-12, float(training_sec)),
            "estimated_from_sampled_updates": group in {"Critic update", "Actor + target update"},
        }

    def _summarize_window(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        counts = {
            "profiled_updates": int(data["profiled_updates"]),
            "profiled_actor_updates": int(data["profiled_actor_updates"]),
        }
        training = float(data["training_wall_sec"])
        components: Dict[str, Any] = {}
        accounted = 0.0
        for group in PAPER_GROUPS[:-1]:
            samples = list(data["groups"].get(group, []))
            total, count = self._scaled_group_total(group, samples, counts)
            accounted += total
            components[group] = self._component_stats(
                group=group,
                samples=samples,
                total_sec=total,
                count=count,
                env_steps=int(data["profiled_env_steps"]),
                updates=int(data["profiled_updates"]),
                training_sec=training,
            )
        eval_logging_total = (
            float(data["eval_sec"])
            + float(data["logging_sec"])
            + float(data["checkpoint_sec"])
        )
        eval_samples = list(data["groups"].get("Eval/logging amortized", []))
        components["Eval/logging amortized"] = self._component_stats(
            group="Eval/logging amortized",
            samples=eval_samples,
            total_sec=eval_logging_total,
            count=len(eval_samples),
            env_steps=int(data["profiled_env_steps"]),
            updates=int(data["profiled_updates"]),
            training_sec=training,
        )
        other = max(0.0, training - accounted)
        components["Other/unaccounted"] = self._component_stats(
            group="Other/unaccounted",
            samples=[other / max(1, int(data["profiled_env_steps"]))]
            if int(data["profiled_env_steps"])
            else [],
            total_sec=other,
            count=int(data["profiled_env_steps"]),
            env_steps=int(data["profiled_env_steps"]),
            updates=int(data["profiled_updates"]),
            training_sec=training,
        )
        return {
            "start_step": int(data["start_step"]),
            "end_step": int(data["end_step"]),
            "profiled_env_steps": int(data["profiled_env_steps"]),
            "profiled_updates": int(data["profiled_updates"]),
            "profiled_actor_updates": int(data["profiled_actor_updates"]),
            "training_excluding_eval_sec": training,
            "eval_sec": float(data["eval_sec"]),
            "logging_sec": float(data["logging_sec"]),
            "checkpoint_sec": float(data["checkpoint_sec"]),
            "profile_elapsed_sec": training + eval_logging_total,
            "accounted_component_sec": accounted,
            "other_sec": other,
            "accounting_ratio": accounted / max(1e-12, training),
            "accounting_ratio_with_other": (accounted + other) / max(1e-12, training),
            "components": components,
        }

    def finalize(self, *, total_wall_sec: float, total_updates: int) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if self._events_file is not None:
            self._events_file.flush()

        counts = {
            "profiled_updates": self.profiled_updates,
            "profiled_actor_updates": self.profiled_actor_updates,
        }
        components: Dict[str, Any] = {}
        accounted = 0.0
        for group in PAPER_GROUPS[:-1]:
            samples = list(self.group_samples.get(group, []))
            total, count = self._scaled_group_total(group, samples, counts)
            accounted += total
            components[group] = self._component_stats(
                group=group,
                samples=samples,
                total_sec=total,
                count=count,
                env_steps=self.profiled_env_steps,
                updates=self.profiled_updates,
                training_sec=self.training_wall_sec,
            )

        eval_logging_total = self.eval_sec + self.logging_sec + self.checkpoint_sec
        eval_samples = list(self.group_samples.get("Eval/logging amortized", []))
        components["Eval/logging amortized"] = self._component_stats(
            group="Eval/logging amortized",
            samples=eval_samples,
            total_sec=eval_logging_total,
            count=len(eval_samples),
            env_steps=self.profiled_env_steps,
            updates=self.profiled_updates,
            training_sec=self.training_wall_sec,
        )
        other = max(0.0, self.training_wall_sec - accounted)
        components["Other/unaccounted"] = self._component_stats(
            group="Other/unaccounted",
            samples=[other / max(1, self.profiled_env_steps)] if self.profiled_env_steps else [],
            total_sec=other,
            count=self.profiled_env_steps,
            env_steps=self.profiled_env_steps,
            updates=self.profiled_updates,
            training_sec=self.training_wall_sec,
        )

        raw_components: Dict[str, Any] = {}
        for name in sorted(self.raw_samples):
            samples = self.raw_samples[name]
            total = float(self.raw_totals[name])
            raw_components[name] = {
                "total_sec": total,
                "count": len(samples),
                "mean_ms": 1_000.0 * statistics.fmean(samples) if samples else 0.0,
                "median_ms": 1_000.0 * statistics.median(samples) if samples else 0.0,
                "p90_ms": 1_000.0 * _percentile(samples, 0.90),
            }

        profile_elapsed = self.training_wall_sec + eval_logging_total
        summary: Dict[str, Any] = {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "method": self.method,
            "seed": self.seed,
            "device": str(self.device),
            "hardware_metadata": _hardware_metadata(self.device),
            "total_steps": self.total_steps,
            "profile_start_step": self.profile_start_step,
            "profile_end_step": self.profile_end_step,
            "profile_sample_every_updates": self.profile_sample_every_updates,
            "profile_sync_cuda": self.profile_sync_cuda,
            "profile_events_jsonl": self.profile_events_jsonl,
            "profiled_env_steps": self.profiled_env_steps,
            "total_updates": int(total_updates),
            "profiled_updates": self.profiled_updates,
            "profiled_critic_updates": self.profiled_critic_updates,
            "profiled_actor_updates": self.profiled_actor_updates,
            "actor_updates": self.profiled_actor_updates,
            "sampled_gpu_updates": self.sampled_gpu_updates,
            "sampled_actor_updates": self.sampled_actor_updates,
            "total_wall_sec": float(total_wall_sec),
            "training_excluding_eval_sec": self.training_wall_sec,
            "eval_sec": self.eval_sec,
            "logging_sec": self.logging_sec,
            "checkpoint_sec": self.checkpoint_sec,
            "profile_elapsed_sec": profile_elapsed,
            "steps_per_sec_excluding_eval": self.profiled_env_steps / max(1e-12, self.training_wall_sec),
            "steps_per_sec_including_eval": self.profiled_env_steps / max(1e-12, profile_elapsed),
            "accounted_component_sec": accounted,
            "unaccounted_other_sec": other,
            "measured_component_accounting_ratio": accounted / max(1e-12, self.training_wall_sec),
            "component_accounting_ratio": (accounted + other) / max(1e-12, self.training_wall_sec),
            "component_accounting_ratio_with_other": (accounted + other) / max(1e-12, self.training_wall_sec),
            "profiler_event_logging_sec": self.profiler_event_logging_sec,
            "profiler_administrative_sec": self.profiler_administrative_sec,
            "profiled_cuda_sync_wait_sec": self.profiled_cuda_sync_wait_sec,
            "components": components,
            "raw_components": raw_components,
            "subwindows": {
                name: self._summarize_window(name, data) for name, data in self.window_data.items()
            },
            "support_cache_hit_rate": statistics.fmean(self.gauges["support_cache_hit_rate"])
            if self.gauges["support_cache_hit_rate"]
            else 0.0,
            "local_scan_cache_hit_rate": statistics.fmean(self.gauges["local_scan_cache_hit_rate"])
            if self.gauges["local_scan_cache_hit_rate"]
            else 0.0,
            "numba_support_scan_enabled": bool(
                statistics.fmean(self.gauges["numba_support_scan_enabled"])
                if self.gauges["numba_support_scan_enabled"]
                else False
            ),
        }
        self._write_outputs(summary)
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None
        return summary

    def _write_outputs(self, summary: Dict[str, Any]) -> None:
        summary_path = self.run_dir / "runtime_profile_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        csv_path = self.run_dir / "runtime_profile_components.csv"
        fields = (
            "component",
            "total_sec",
            "count",
            "sample_count",
            "mean_ms",
            "median_ms",
            "p90_ms",
            "ms_per_env_step",
            "ms_per_update",
            "training_share",
            "estimated_from_sampled_updates",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for component, values in summary["components"].items():
                writer.writerow({"component": component, **{key: values.get(key, "") for key in fields[1:]}})
