import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from bootstrap import ROOT


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    packages = {p: importlib.metadata.version(p) for p in
                ("torch", "numpy", "numba", "mujoco", "metaworld", "gymnasium", "gymnasium-robotics")}
    return dict(python=platform.python_version(), architecture=platform.machine(),
                packages=packages, device="cpu", torch_threads=1,
                threads={k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")})


def source_files():
    candidates = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md"))
    candidates += [ROOT / "config_template.json", ROOT / "provenance.json"]
    candidates += list((ROOT / "code").rglob("*.py"))
    candidates += list((ROOT / "reference").rglob("*.py"))
    candidates += list((ROOT / "tests").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(candidates)}


def verify_source():
    data = json.loads((ROOT / "source_manifest.json").read_text())
    if data["files"] != source_files():
        raise RuntimeError("Source changed after freeze; validation must be repeated")
    for item in json.loads((ROOT / "provenance.json").read_text())["reference_files"]:
        if sha(ROOT / "reference" / item["path"]) != item["sha256"]:
            raise RuntimeError("Frozen reference source changed")
    return sha(ROOT / "source_manifest.json")


def verify_gate(source, env):
    gate = json.loads((ROOT / "validation/gate.json").read_text())
    if not (gate.get("passed") is True and gate.get("source_manifest_sha256") == source
            and gate.get("environment") == env):
        raise RuntimeError("Pilot gate is absent or stale")
    report = Path(gate["report_path"])
    if not report.is_file() or sha(report) != gate["report_sha256"]:
        raise RuntimeError("Validation evidence is missing or changed")
    result = json.loads(report.read_text())
    if not (result.get("passed") is True and result.get("source_manifest_sha256") == source
            and result.get("environment") == env):
        raise RuntimeError("Validation report does not match this release")
    return gate


class JsonLines:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8")

    def write(self, row):
        self.file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

