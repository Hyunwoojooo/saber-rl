#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "context_count_only_source_manifest.json"
FILES = (
    "DDPG-HER-B3-Context-Count-Only-2M.sh",
    "DDPG-HER-B3-Critic-Only-2M.sh",
    "README.md",
    "validate_context_count_only_config_parity.py",
    "test_context_count_only_config_parity.py",
    "validate_context_count_only_run.py",
    "freeze_context_count_only_source.py",
    "run_context_count_only_correctness_5k.sh",
    "analyze_context_count_only_results.py",
    "test_analyze_context_count_only_results.py",
    "build_context_count_only_artifact.py",
    "verify_context_count_only_artifact.py",
    "code/src/train_drawer_open_lite_v0.py",
    "code/src/test_context_count_only_drawer_open.py",
    "code/src/metacog_lite_v0/__init__.py",
    "code/src/test_context_count_only_ddpg_loss_scope.py",
    "code/src/metacog_lite_v0/agent.py",
    "code/src/metacog_lite_v0/config.py",
    "code/src/metacog_lite_v0/envs.py",
    "code/src/metacog_lite_v0/envs_maniskill.py",
    "code/src/metacog_lite_v0/io_utils.py",
    "code/src/metacog_lite_v0/networks.py",
    "code/src/metacog_lite_v0/numba_support_scan_drawer_open.py",
    "code/src/metacog_lite_v0/replay.py",
    "code/src/metacog_lite_v0/task_drawer_open.py",
    "code/src/metacog_lite_v0/task_fetch_pick_and_place.py",
    "code/src/metacog_lite_v0/task_fetch_push.py",
    "code/src/metacog_lite_v0/task_fetch_slide.py",
    "code/src/metacog_lite_v0/task_pick_cube.py",
    "code/src/metacog_lite_v0/task_push_t.py",
    "code/src/metacog_lite_v0/task_unitree_apple.py",
    "code/src/metacog_lite_v0/trainer.py",
    "code/src/metacog_lite_v0/weights.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current() -> dict:
    entries = []
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    return {"schema_version": 2, "source_fingerprint": fingerprint, "files": entries}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    observed = current()
    if args.write:
        MANIFEST.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST} ({len(observed['files'])} files)")
        return 0
    expected = json.loads(MANIFEST.read_text())
    if observed != expected:
        raise SystemExit("context-count-only source manifest mismatch")
    print(f"verified {MANIFEST} ({len(observed['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
