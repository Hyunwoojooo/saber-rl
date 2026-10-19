"""Read-only prerequisite and reference-result audit, with no Torch imports."""
import csv
import hashlib
import json
import math
import importlib.metadata
import sys
from pathlib import Path

from protocol import ROOT, SCHEDULE, make_config


def alive(pid, expected_script=None):
    try:
        state = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0]
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return state != "Z" and (expected_script is None or str(expected_script) in command)
    except (OSError, ValueError, TypeError):
        return False


def check_manifest(root, name, expected_digest, key="sha256"):
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise RuntimeError(f"Predecessor {name} changed")
    for rel, digest in json.loads(path.read_text())[key].items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Predecessor scientific source changed: {rel}")


def predecessor_state(prior=None, expected_digest=None):
    prior = Path(prior) if prior is not None else (ROOT / SCHEDULE["predecessor"]).resolve()
    expected_digest = expected_digest or SCHEDULE["predecessor_source_manifest_sha256"]
    state = json.loads((prior / "recovery_state/status.json").read_text())
    if state.get("source_manifest_sha256") != expected_digest or state.get(
            "recovery_manifest_sha256") != SCHEDULE["predecessor_recovery_manifest_sha256"]:
        raise RuntimeError("Unexpected predecessor scientific/recovery snapshot")
    script = prior / "recovery/manage.py"
    if state.get("status") == "running":
        if not alive(state.get("pid"), script):
            raise RuntimeError("Predecessor reports running but its process is absent; inspect manually")
        return {"state": "waiting", "reason": "reference SAC experiment active", "stage": state.get("stage")}
    if state.get("status") != "complete":
        raise RuntimeError(f"Predecessor did not complete successfully: {state.get('status')}")
    if [s.get("name") for s in state.get("stages", [])] != SCHEDULE["predecessor_required_stages"] or any(
            s.get("returncode") != 0 for s in state["stages"]):
        raise RuntimeError("Predecessor training and analyses did not all pass")
    check_manifest(prior, "source_manifest.json", expected_digest)
    check_manifest(prior, "recovery_manifest.json", SCHEDULE["predecessor_recovery_manifest_sha256"])
    read_references(prior)
    if alive(state.get("pid"), script):
        return {"state": "waiting", "reason": "predecessor shutting down"}
    return {"state": "ready", "reason": "all 20 reference runs and final analysis validated"}


def reference_config_parity(recorded, task, method, seed):
    expected = json.loads(json.dumps(make_config(task, method, seed).as_dict()))
    ignored = {"output_root", "experiment_id", "thread_hint", "terminal_convention",
               "sac_target_entropy_effective"}
    for key, value in recorded.items():
        if key not in ignored and expected.get(key) != value:
            raise ValueError(f"Reference configuration mismatch: {task}/{method}/{seed}/{key}")
    for key in ("total_steps", "sac_auto_alpha", "sac_target_entropy", "environment_reset_mode", "seeds"):
        if recorded.get(key) != expected[key]:
            raise ValueError(f"Missing/mismatched critical reference setting: {key}")


def read_references(prior=None):
    prior = Path(prior) if prior is not None else (ROOT / SCHEDULE["predecessor"]).resolve()
    report = json.loads((prior / "analysis/final.json").read_text())
    expected = {(t, m, s) for t in ("drawer_open", "drawer_close") for m in ("her", "saber") for s in range(1, 6)}
    rows = report.get("runs", [])
    if report.get("status") != "completed" or report.get("incomplete") != [] or len(rows) != 20 or {
            (r["task"], r["method"], r["seed"]) for r in rows} != expected:
        raise RuntimeError("Reference cohort is incomplete; never select only successful seeds")
    from analyze import auc
    for row in rows:
        folder = Path(row["run_dir"]).resolve()
        allowed = {prior / "runs", prior.parent / "2026-10-08_drawer_sac_fpp_v2_s1_5/runs"}
        if folder.parent not in allowed:
            raise RuntimeError("Unapproved reference-result root")
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        reference_config_parity(config, row["task"], row["method"], row["seed"])
        claim = json.loads((folder.parent.parent / ".production_claims" / folder.name / "status.json").read_text())
        if claim["status"] != "completed" or claim["source_manifest_sha256"] != SCHEDULE["predecessor_source_manifest_sha256"]:
            raise RuntimeError("Reference execution claim is invalid")
        if summary["total_steps"] != 2000000 or summary["total_updates"] != 1995001:
            raise RuntimeError("Reference training budget is incomplete")
        with (folder / "metrics.csv").open(newline="") as stream:
            curve = [(int(r["step"]), float(r["eval/env_success_mean"]))
                     for r in csv.DictReader(stream) if r.get("eval/env_success_mean", "") != ""]
        if [x for x, _ in curve] != list(range(10000, 2000001, 10000)) or any(
                not math.isfinite(y) or not 0 <= y <= 1 for _, y in curve):
            raise RuntimeError("Reference evaluation grid invalid")
        values = {"auc": auc(curve), "fq_auc": auc([p for p in curve if p[0] >= 1500000]),
                  "final": curve[-1][1], "hit90": next((s for s, y in curve if y >= .9), None)}
        for key, value in values.items():
            if (value is None and row[key] is not None) or (value is not None and not math.isclose(value, row[key], abs_tol=1e-12)):
                raise RuntimeError(f"Reference result disagrees with raw CSV: {key}")
    return rows


def audit_reference_environment():
    names = ("torch", "numpy", "scipy", "mujoco", "metaworld", "gymnasium", "gymnasium-robotics")
    current = {name: importlib.metadata.version(name) for name in names}
    rows = read_references()
    for row in rows:
        folder = Path(row["run_dir"])
        path = folder.parent.parent / ".production_claims" / folder.name / "environment.json"
        env = json.loads(path.read_text())
        packages = {k.lower().replace("_", "-"): v for k, v in env["packages"].items()}
        if env["python"].split()[0] != sys.version.split()[0] or any(packages.get(k) != v for k, v in current.items()):
            raise RuntimeError(f"Reference/library mismatch: {folder.name}; do not silently change environment")
    return {"status": "passed", "reference_runs": len(rows), "python": sys.version.split()[0], "packages": current}
