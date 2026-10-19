"""AST and file hashes only. Does not import NumPy, Torch, simulator, or tests."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    files = [p for p in ROOT.rglob("*.py") if "outputs" not in p.relative_to(ROOT).parts
             and "__pycache__" not in p.parts]
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    manifest_path = ROOT / "upstream/source_manifest.json"
    expected_manifest = "d281263084807ec6010986dc762301dce420b004e022c88985e51dcc50ccf496"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest:
        raise ValueError("Copied upstream manifest changed")
    manifest = json.loads(manifest_path.read_text())["sha256"]
    copied = list((ROOT / "upstream/code").rglob("*.py"))
    expected_files = {rel for rel in manifest if rel.startswith("code/") and rel.endswith(".py")}
    if {str(p.relative_to(ROOT / "upstream")) for p in copied} != expected_files:
        raise ValueError("Incomplete upstream source copy")
    for path in copied + [ROOT / "upstream/protocol.py", ROOT / "upstream/PROTOCOL.md"]:
        rel = str(path.relative_to(ROOT / "upstream"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[rel]:
            raise ValueError(f"Copied upstream source changed: {rel}")
    for name in ("replay.py", "envs.py", "networks.py", "task_drawer_open.py", "io_utils.py"):
        a, b = (ROOT / "code/src/metacog_lite_v0" / name,
                ROOT / "upstream/code/src/metacog_lite_v0" / name)
        if a.read_bytes() != b.read_bytes():
            raise ValueError(f"Unapproved change to shared training component: {name}")
    origin = ROOT.parent / "2026-09-25_draweropen_ddpg_assignment"
    origin_checked = False
    if origin.is_dir():
        if (origin / "source_manifest.json").read_bytes() != manifest_path.read_bytes():
            raise ValueError("Running experiment manifest changed")
        for rel, expected in manifest.items():
            if hashlib.sha256((origin / rel).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Running experiment source changed: {rel}")
        origin_checked = True
    report = {"python_files_parsed": len(files), "upstream_python_files_hash_checked": len(copied),
              "static_check_passed": True, "runtime_tests_run": False,
              "running_experiment_source_unchanged": origin_checked,
              "smoke_run": False, "training_started": False,
              "scope": "syntax and immutable-copy checks only; not runtime validation"}
    destination = ROOT / "validation/static_check.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
