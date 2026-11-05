"""One fresh run. Pilot execution requires an identical validated release."""
import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from bootstrap import ROOT
from common import environment, utc, verify_gate, verify_source, write_json
from protocol import ARMS, SEEDS, PROFILES, make_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--profile",choices=PROFILES,default="smoke")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--diagnostics",choices=("on","off"),default="on")
    args = parser.parse_args()
    source = verify_source()
    env = environment()
    if args.profile == "pilot":
        verify_gate(source, env)
        if args.diagnostics != "on":
            raise RuntimeError("Pilot requires enabled diagnostics")
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    state = dict(status="running",arm=args.arm,seed=args.seed,profile=args.profile,
        diagnostics=args.diagnostics,pid=os.getpid(),source_manifest_sha256=source,
        checkpoint_continuation=False,started_utc=utc())
    write_json(root/"status.json",state)
    observer = None
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupt)
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        write_json(root/"environment.json",env)
        cfg = make_config(args.arm,args.seed,root,args.profile)
        write_json(root/"requested_config.json",cfg.as_dict())
        if args.diagnostics == "on":
            from diagnostics import Observer
            observer = Observer(args.arm,args.seed,args.profile,root)
        from metacog_lite_v0.trainer import run_training
        run_training(cfg,ROOT,observer=observer)
        if observer:
            observer.close()
        from analyze import audit_attempt
        metrics = audit_attempt(root)
        write_json(root/"completion_audit.json",metrics)
        state.update(status="complete",metrics=metrics)
    except FloatingPointError as exc:
        state.update(status="failed_nonfinite",error=str(exc))
        raise
    except KeyboardInterrupt as exc:
        state.update(status="interrupted",error=str(exc))
        raise
    except BaseException as exc:
        state.update(status="failed",error=str(exc))
        raise
    finally:
        state["finished_utc"] = utc()
        if state["status"] != "complete":
            (root/"failure_traceback.txt").write_text(traceback.format_exc())
        try:
            if observer:
                observer.close()
        finally:
            write_json(root/"status.json",state)


if __name__ == "__main__":
    main()

