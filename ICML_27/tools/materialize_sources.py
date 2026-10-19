"""Restore original sibling source layout into a NEW directory, without training."""
import argparse
from pathlib import Path, PurePosixPath
import shutil

from verify_archive import ROOT, load, verify


def source_layout(root):
    layout = []
    names = set()
    for entry in load(root / "experiments/index.json")["experiments"]:
        folder = root / "experiments" / entry["directory"]
        original = PurePosixPath(load(folder / "provenance.json")["original_directory"])
        if (len(original.parts) != 2 or original.parts[0] != "experiments_post_review"
                or original.name in {".", ".."} or original.name in names):
            raise ValueError(f"Invalid or duplicate original execution directory: {original}")
        names.add(original.name)
        layout.append((folder / "source", original.name))
    return layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    dest = args.destination.resolve()
    if dest.exists() or ROOT in dest.parents:
        parser.error("Use a new destination outside this archive; existing paths are never overwritten")
    verify()
    layout = source_layout(ROOT)
    dest.mkdir(parents=True, exist_ok=False)
    for source, original_name in layout:
        shutil.copytree(source, dest / "experiments_post_review" / original_name)
    print(f"Sources only: {dest / 'experiments_post_review'}")
    print("No results, gates or environments restored. No training or queue launched.")


if __name__ == "__main__":
    main()
