from __future__ import annotations

import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED))

from run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(default_experiment=Path(__file__).resolve().parent.name))
