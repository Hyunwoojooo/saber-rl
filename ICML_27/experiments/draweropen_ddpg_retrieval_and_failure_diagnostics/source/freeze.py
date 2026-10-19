"""Immutable local source, including byte-identical upstream implementations."""
import argparse
from common import ROOT, ASSIGNMENT_SHA, digest, read_json, write_json


def inventory():
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + [ROOT / ".gitignore"]
    for directory in ("upstream", "tests"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".nbc", ".nbi"})
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def verify_upstream():
    path = ROOT / "upstream/assignment_source_manifest.json"
    if digest(path) != ASSIGNMENT_SHA:
        raise ValueError("Upstream manifest changed")
    source = read_json(path)["sha256"]
    copied = [p for p in (ROOT / "upstream/code").rglob("*") if p.is_file()
              and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".nbc", ".nbi"}]
    expected = {name: sha for name, sha in source.items() if name.startswith("code/")}
    actual = {str(p.relative_to(ROOT / "upstream")): digest(p) for p in copied}
    if actual != expected:
        raise ValueError("Copied implementation is not the frozen assignment source")


def verify():
    path = ROOT / "source_manifest.json"
    if read_json(path)["sha256"] != inventory():
        raise ValueError("Diagnostic source changed after freeze; execution blocked")
    verify_upstream()
    return digest(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    if args.create:
        verify_upstream()
        write_json(ROOT / "source_manifest.json",
                   {"snapshot_id": "draweropen_retrieval_failure_diagnostics_v2",
                    "upstream_manifest_sha256": ASSIGNMENT_SHA, "sha256": inventory()}, exclusive=True)
    print(verify())


if __name__ == "__main__":
    main()
