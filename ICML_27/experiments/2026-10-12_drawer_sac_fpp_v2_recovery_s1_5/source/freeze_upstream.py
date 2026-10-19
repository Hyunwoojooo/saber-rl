"""One-time, byte-preserving capture. Never regenerate an existing snapshot."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FPP = REPO / "worktrees/code-supplementary-final/experiment_FPP/sac_saber_critic_only_v4_ls512"
DRAWER = REPO / "metaworld_Drawer_Close_DDPG"


def main():
    upstream = ROOT / "upstream"
    upstream.mkdir(exist_ok=False)
    entries = []
    for label, source in (("fpp", FPP), ("drawer", DRAWER)):
        files = sorted((source / "code/src/metacog_lite_v0").glob("*.py"))
        files += [source / "config.json"] if label == "fpp" else [
            source / "DDPG-HER-B3-Critic-Only-2M.sh"]
        for original in files:
            dest = upstream / label / original.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
            entries.append({"path": str(dest.relative_to(ROOT)),
                            "origin": str(original.relative_to(REPO)),
                            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    active = ROOT / "code/src/metacog_lite_v0"
    shutil.copytree(upstream / "fpp/code/src/metacog_lite_v0", active)
    for name in ("envs.py", "weights.py", "task_drawer_open.py", "task_drawer_close.py",
                 "numba_support_scan_drawer_open.py"):
        shutil.copy2(upstream / "drawer/code/src/metacog_lite_v0" / name, active / name)
    (ROOT / "upstream_manifest.json").write_text(json.dumps({
        "snapshot_id": "drawer_sac_fpp_v1", "files": entries,
        "note": "Working-file SHA256 snapshots, not a claim about historical execution commits."
    }, indent=2) + "\n")
    print(f"Captured {len(entries)} upstream files; active copy ready for integration.")


if __name__ == "__main__":
    main()
