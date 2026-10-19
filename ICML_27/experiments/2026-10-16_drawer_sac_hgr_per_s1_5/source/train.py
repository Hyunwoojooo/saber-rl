"""Dry-run by default. Production requires a matching successful validation report."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import signal
import sys
import traceback

from protocol import ROOT, SPEC, make_config, make_gpu_validation_config, validate_parity
from manifest import verify
from gates import require_cpu_gate, require_production_gates, write_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["drawer_open", "drawer_close"], required=True)
    parser.add_argument("--method", choices=SPEC["methods"], required=True)
    parser.add_argument("--seed", type=int, choices=SPEC["seeds"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-validation", choices=["gpu_smoke", "stability"])
    parser.add_argument("--output-root", help="Validation-only output location")
    args = parser.parse_args()
    validation = args.smoke or bool(args.gpu_validation)
    if args.smoke and args.gpu_validation:
        parser.error("CPU smoke and GPU validation are separate profiles")
    if args.output_root and not validation:
        parser.error("Production output is fixed inside the experiment")
    if args.execute and not args.smoke and args.device != "cuda":
        parser.error("Production is CUDA-only; CPU is reserved for the validation preset")
    if args.gpu_validation:
        if not args.output_root or args.device != "cuda":
            parser.error("GPU validation requires an explicit output root and CUDA")
        cfg = make_gpu_validation_config(args.task, args.method, args.seed, args.gpu_validation, args.output_root)
    else:
        cfg = make_config(args.task, args.method, args.seed, args.device, args.smoke, args.output_root)
    validate_parity()
    if not args.execute:
        print(json.dumps(cfg.as_dict(), indent=2))
        return
    digest = verify()
    if args.gpu_validation:
        require_cpu_gate(digest)
    elif not args.smoke:
        require_production_gates(digest)
    claim_root = Path(cfg.output_root) / ".claims" if validation else ROOT / ".production_claims"
    claim = claim_root / cfg.run_name
    claim.mkdir(parents=True, exist_ok=False)

    def record(status, **extra):
        payload = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "source_manifest_sha256": digest, "config": cfg.as_dict(), **extra}
        write_atomic(claim / "status.json", payload)

    def interrupted(signum, _frame):
        raise InterruptedError(f"Signal {signum}; no automatic resume or replacement")

    signal.signal(signal.SIGTERM, interrupted)
    record("starting", pid=os.getpid())
    try:
        # No package installation or modifications to the shared environment.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[name] = "1"
        os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        from metacog_lite_v0.trainer import run_training
        packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
        (claim / "environment.json").write_text(json.dumps({
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": cfg.device, "gpu": torch.cuda.get_device_name(0) if cfg.device == "cuda" else None,
            "packages": packages}, indent=2) + "\n")
        record("running", pid=os.getpid())
        run_dir = run_training(cfg, ROOT)
        from analyze import validate_run
        metrics = validate_run(run_dir, cfg)
        record("completed", run_dir=str(run_dir), metrics=metrics)
    except BaseException as exc:
        nonfinite = isinstance(exc, FloatingPointError) or (
            isinstance(exc, RuntimeError) and "non-finite" in str(exc).lower())
        category = "non_finite" if nonfinite else "interrupted" if isinstance(
            exc, (KeyboardInterrupt, InterruptedError)) else "execution_error"
        record("failed", failure_category=category, error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
