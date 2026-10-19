from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


class MetricsCSVWriter:
    def __init__(self, path: Path, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class MetricsJSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        self._file.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def make_run_dir(output_root: Path, run_name: str, resume: bool) -> Path:
    if resume:
        return output_root / run_name
    plain = output_root / run_name
    if not plain.exists():
        return plain
    return output_root / f"{run_name}_{utc_timestamp()}"


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"metacog_lite_v0_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def select_device(device_flag: str) -> torch.device:
    d = device_flag.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def set_all_seeds(torch_seed: int, numpy_seed: int) -> None:
    np.random.seed(int(numpy_seed))
    torch.manual_seed(int(torch_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(torch_seed))


def env_thread_hint() -> Dict[str, str]:
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
    }
