"""Experiment-local imports and a fixed, CPU-only single-thread runtime."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code" / "src"))

