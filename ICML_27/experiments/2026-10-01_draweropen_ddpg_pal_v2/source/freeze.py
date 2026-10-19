"""Seal source with SHA256. This utility never imports or runs training/tests."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md"))
    files += list(ROOT.glob("*.txt")) + [ROOT / ".gitignore"]
    for directory in ("code", "upstream", "tests", "third_party"):
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".nbc", ".nbi"}]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def verify():
    saved = json.loads((ROOT / "source_manifest.json").read_text())["sha256"]
    current = inventory()
    if saved != current:
        bad = sorted(k for k in saved.keys() | current.keys() if saved.get(k) != current.get(k))
        raise ValueError(f"Source changed after freeze: {bad}")
    return digest(ROOT / "source_manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    args = ap.parse_args()
    if args.create:
        payload = {"snapshot_id": "draweropen_ddpg_pal_v2", "sha256": inventory(),
                   "execution_validation": "not_run; requires separate tests and smoke gate",
                   "parent_pal_manifest_sha256": "86617e1a2b4710848ee2cd96e35856b01cc6cb39d6fbe54b96d05eefaacd2f0c",
                   "origin": "../2026-09-25_draweropen_ddpg_assignment/code/src"}
        with (ROOT / "source_manifest.json").open("x") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    print(verify())


if __name__ == "__main__":
    main()
