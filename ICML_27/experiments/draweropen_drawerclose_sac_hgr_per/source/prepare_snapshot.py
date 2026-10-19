"""One-time copy of existing scientific sources; never modifies the originals."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PARENT = ROOT.parent / "2026-10-12_drawer_sac_fpp_v2_recovery_s1_5"


def main():
    if (ROOT / "code").exists() or (ROOT / "source_manifest.json").exists():
        raise RuntimeError("Snapshot already prepared; no overwrite")
    rows = []

    def snapshot(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append({"path": str(destination.relative_to(ROOT)),
                     "origin": str(source.relative_to(REPO)),
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})

    for source in sorted((PARENT / "code").rglob("*.py")):
        rel = source.relative_to(PARENT)
        snapshot(source, ROOT / "upstream/current" / rel)
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in sorted((PARENT / "upstream").rglob("*.py")):
        snapshot(source, ROOT / source.relative_to(PARENT))
    for name in ("protocol.json", "protocol.py", "manifest.py", "gates.py", "train.py",
                 "launch.py", "analyze.py", "validate.py", "gpu_validate.py", "scheduled_run.py"):
        snapshot(PARENT / name, ROOT / "upstream/current" / name)
        shutil.copy2(PARENT / name, ROOT / name)
    source = PARENT / "tests/test_contract.py"
    snapshot(source, ROOT / "upstream/current/tests/test_contract.py")
    (ROOT / "tests").mkdir()
    shutil.copy2(source, ROOT / "tests/test_contract.py")
    for family, folder in (("hgr", "sac_hgr_tdpriority_1m"), ("per", "sac_her_per_full_ls512")):
        base = REPO / "experiment_FPP" / folder
        for source in sorted((base / "code/src").rglob("*.py")):
            snapshot(source, ROOT / "upstream" / family / source.relative_to(base))
        snapshot(base / "README.md", ROOT / "upstream" / family / "README.md")
    # PER extends the same replay class; its uniform-HER branch will be parity-tested.
    shutil.copy2(ROOT / "upstream/per/code/src/metacog_lite_v0/replay.py",
                 ROOT / "code/src/metacog_lite_v0/replay.py")
    shutil.copy2(ROOT / "upstream/hgr/code/src/metacog_lite_v0/hgr.py",
                 ROOT / "code/src/metacog_lite_v0/hgr.py")
    (ROOT / "upstream_manifest.json").write_text(json.dumps({"files": rows}, indent=2) + "\n")
    print(f"Copied {len(rows)} immutable reference files; current experiment untouched")


if __name__ == "__main__":
    main()
