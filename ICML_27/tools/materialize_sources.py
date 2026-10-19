"""Restore original sibling source layout into a NEW directory, without training."""
import argparse
from pathlib import Path
import shutil

from verify_archive import ROOT, load, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    dest = args.destination.resolve()
    if dest.exists() or ROOT in dest.parents:
        parser.error("Use a new destination outside this archive; existing paths are never overwritten")
    verify()
    dest.mkdir(parents=True, exist_ok=False)
    for entry in load(ROOT / "experiments/index.json")["experiments"]:
        name = entry["directory"]
        shutil.copytree(ROOT / "experiments" / name / "source", dest / "experiments_post_review" / name)
    print(f"Sources only: {dest / 'experiments_post_review'}")
    print("No results, gates or environments restored. No training or queue launched.")


if __name__ == "__main__":
    main()
