"""Freeze and verify source inputs separately from generated validation/results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files():
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md"))
    paths += [ROOT / "protocol.json", ROOT / "schedule_spec.json", ROOT / "upstream_manifest.json"]
    paths += list((ROOT / "code").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def verify_upstream(check_origins=False):
    rows = json.loads((ROOT / "upstream_manifest.json").read_text())["files"]
    for row in rows:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Changed upstream snapshot: {row['path']}")
        origin = ROOT.parents[1] / row["origin"]
        if check_origins and origin.exists() and sha(origin) != row["sha256"]:
            raise RuntimeError(f"Original file changed: {row['origin']}")
    return len(rows)


def verify():
    verify_upstream()
    path = ROOT / "source_manifest.json"
    expected = json.loads(path.read_text())["sha256"]
    current = source_files()
    if expected != current:
        changed = [p for p in expected.keys() | current.keys() if expected.get(p) != current.get(p)]
        raise RuntimeError(f"Source manifest mismatch: {changed}")
    return sha(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        if any((ROOT / p).exists() for p in ("runs", ".production_claims", "automation/status.json")):
            raise RuntimeError("Production has been reserved; make a new experiment instead of refreezing")
        verify_upstream()
        (ROOT / "source_manifest.json").write_text(json.dumps({
            "snapshot_id": "drawer_sac_fpp_v1", "sha256": source_files()}, indent=2) + "\n")
    print(verify())
