"""Create or verify the independent snapshot; never updates an existing manifest."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGIN = "experiments_post_review/2026-09-25_draweropen_ddpg_assignment"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    files = [p for p in ROOT.glob("*.py")] + list(ROOT.glob("*.md"))
    for name in ("code", "tests", "upstream"):
        files += list((ROOT / name).rglob("*.py"))
    files += list((ROOT / "upstream").glob("*.json"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def verify():
    manifest = ROOT / "source_manifest.json"
    data = json.loads(manifest.read_text())
    current = inventory()
    if current != data["sha256"]:
        bad = sorted(k for k in current.keys() | data["sha256"].keys()
                     if current.get(k) != data["sha256"].get(k))
        raise ValueError(f"Source snapshot changed: {bad}")
    return digest(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    args = ap.parse_args()
    if args.create:
        payload = {"snapshot_id": "draweropen_retrieval_v1", "origin": ORIGIN,
                   "sha256": inventory()}
        with (ROOT / "source_manifest.json").open("x") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    print(verify())


if __name__ == "__main__":
    main()
