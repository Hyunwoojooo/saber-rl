from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FPP_DDPG = ROOT / "source/fpp_ddpg"


def add_fpp_ddpg() -> Path:
    value = str(FPP_DDPG)
    if value not in sys.path:
        sys.path.insert(0, value)
    return FPP_DDPG
