"""Deferred scientific, CUDA and checkpoint checks, never training production."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
from datetime import datetime, timezone

from freeze import verify
from protocol import ROOT, SEEDS, REFERENCE_MANIFEST_SHA256, make_config
from records import accepted_attempt, local_output, read_json, write_json

CHECKS = ["source", "reference", "cpu_tests", "cuda_loss", "cuda_smoke", "checkpoint_reload", "analysis"]


def load_test_helpers():
    spec = importlib.util.spec_from_file_location("pal_validation_helpers", ROOT / "tests/test_pal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_audit():
    import analyze
    from after_current import REFERENCE, inspect_reference, reference_processes
    from train import environment

    if inspect_reference()["complete"] != 30 or reference_processes():
        raise ValueError("All 30 reference runs must finish and release their processes")
    ref = analyze.audit_reference_source(REFERENCE)
    rows, accepted, problems = analyze.collect_cohort(
        ref / "outputs/production", "support", REFERENCE_MANIFEST_SHA256, reference=True)
    if problems or set(accepted) != set(SEEDS):
        raise ValueError(f"SABER reuse audit failed: {problems}")
    observed = environment()
    for seed, record in accepted.items():
        analyze.audit_environment(record["environment"], observed)
        attempt = accepted_attempt(ref / "outputs/production/support" / f"seed_{seed:02d}")
        retained = read_json(attempt / "status.json")["metrics"]
        for key in ("auc", "fq_auc", "final_success"):
            if not math.isclose(retained[key], record["metrics"][key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Reference metric mismatch: seed {seed}, {key}")
    return {"reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "seeds": list(SEEDS), "environment": observed, "runs": rows,
            "read_only_reference": True, "runtime_comparison_valid": False}


def cuda_loss_audit():
    import torch
    from metacog_lite_v0.pal import prioritized_approximation_loss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA validation cannot fall back to CPU")
    helpers = load_test_helpers()
    cases = 0
    max_loss_difference = max_gradient_difference = 0.0
    for dtype in (torch.float32, torch.float64):
        values = [-1000, -10, -1.00001, -1, -0.99999, -0.001, 0, 0.001, 0.99999, 1, 1.00001, 10, 1000]
        x = torch.tensor(values, dtype=dtype, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        actual = prioritized_approximation_loss(x, 0.4, 1)
        expected, normalizer, _ = helpers.official_loss(y, 0.4, 1)
        tolerance = 1e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(actual.loss, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual.normalizer, normalizer, rtol=tolerance, atol=tolerance)
        if actual.normalizer.requires_grad:
            raise ValueError("PAL normalizer unexpectedly requires gradients")
        actual.loss.backward()
        expected.backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=tolerance, atol=tolerance)
        if not torch.isfinite(x.grad).all():
            raise FloatingPointError("Non-finite PAL gradient")
        max_loss_difference = max(max_loss_difference, abs(actual.loss.item() - expected.item()))
        max_gradient_difference = max(max_gradient_difference, (x.grad - y.grad).abs().max().item())
        cases += len(values)
    torch.cuda.synchronize()
    return {"device": "cuda", "residual_cases": cases, "alpha": 0.4, "threshold": 1,
            "max_loss_difference": max_loss_difference,
            "max_gradient_difference": max_gradient_difference}


def checkpoint_audit(smoke_root):
    import numpy as np
    import torch
    import analyze
    from metacog_lite_v0.agent import DDPGAgent
    from metacog_lite_v0 import trainer

    attempt = accepted_attempt(local_output(smoke_root) / "smoke/pal/seed_01")
    metrics = analyze.validate_run(attempt, "smoke")
    cfg = make_config("pal", 1, attempt, "smoke", "cuda")
    checkpoint = attempt / "run/checkpoint.pt"
    # Only the checkpoint just created locally by the verified smoke is loaded.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    analyze.compare_config(payload["config"], cfg.as_dict())
    if (payload["global_step"] != 6000 or payload["total_updates"] != 1001
            or payload.get("critic_objective") != "pal"
            or payload.get("checkpoint_continuation_permitted") is not False
            or payload.get("pal_alpha") != 0.4 or payload.get("pal_min_priority") != 1):
        raise ValueError("Checkpoint identity/objective/update count mismatch")
    if payload["replay"]["size"] != 6000:
        raise ValueError("Smoke checkpoint replay is incomplete")

    def finite_tree(value):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite checkpoint tensor")
        if isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                finite_tree(child)

    finite_tree(payload["agent"])
    env = trainer.make_env(cfg, cfg.seeds.eval_seed, False)
    try:
        agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                          hidden_dim=cfg.hidden_dim, actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr,
                          gamma=cfg.gamma, tau=cfg.tau, action_low=env.action_low,
                          action_high=env.action_high, device=torch.device("cuda"),
                          l2_action_penalty=cfg.l2_action_penalty, critic_objective="pal",
                          pal_alpha=cfg.pal_alpha, pal_min_priority=cfg.pal_min_priority)
        agent.load_state_dict(payload["agent"])
        if agent.total_it != 1001:
            raise ValueError("Optimizer/model update state did not reload")
        result = trainer.evaluate_agent(agent, env, cfg.seeds.eval_seed + 6000 * 10, cfg.eval_episodes)
        summary = read_json(attempt / "run/summary.json")
        for actual, retained in (("eval_env_success_mean", "final_eval_env_success"),
                                 ("eval_goal_success_mean", "final_eval_goal_success"),
                                 ("eval_return_mean", "final_eval_return")):
            if not math.isclose(result[actual], summary[retained], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Reloaded evaluation differs: {actual}")
        with (attempt / "run/metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        last = rows[-1]
        for key in ("q_policy_mean", "q_policy_std"):
            value = float(last["critic/" + key])
            if not math.isfinite(value) or value != payload["last_loss"][key]:
                raise ValueError(f"Policy-Q checkpoint/CSV mismatch: {key}")
        if not all(np.isfinite(float(last[k])) for k in ("loss/critic", "loss/actor", "pal/normalizer")):
            raise FloatingPointError("Non-finite final smoke measurements")
    finally:
        env.close()
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "updates": 1001, "reloaded_evaluation_success": result["eval_env_success_mean"],
            "success_threshold_for_passing": None, "smoke_metrics": metrics}


def analysis_audit():
    import numpy as np
    import analyze

    x = np.arange(10000, 2000001, 10000)
    y = np.linspace(0.1, 0.9, len(x))
    actual = analyze.curve_metrics(x, y, 1_500_000)
    auc = sum((float(y[i]) + float(y[i+1])) / 2 * (int(x[i+1]) - int(x[i]))
              for i in range(len(x)-1)) / 1_990_000
    if not math.isclose(actual["auc"], auc, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Independent AUC calculation disagrees")
    pairs = np.column_stack((np.linspace(.2, .8, 10), np.linspace(.1, .9, 10)))
    observed = analyze.paired_bootstrap(pairs)
    indices = np.random.default_rng(20260930).integers(0, 10, size=(10000, 10))
    draws = np.array([sum(float(pairs[j, 0] - pairs[j, 1]) for j in row) / 10 for row in indices])
    np.testing.assert_allclose(observed["ci95"], np.quantile(draws, [.025, .975]), rtol=0, atol=1e-12)
    return {"synthetic_only": True, "auc_denominator": 1_990_000,
            "bootstrap_resamples": 10000, "ci_quantiles": [.025, .975]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("reference", "cuda_loss", "readiness"), required=True)
    parser.add_argument("--smoke-root")
    args = parser.parse_args()
    if args.stage == "readiness" and not args.smoke_root:
        parser.error("readiness requires --smoke-root")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import torch
    torch.set_num_threads(1)
    snapshot = verify()
    output = ROOT / "validation" / ("final_readiness.json" if args.stage == "readiness"
                                     else args.stage + "_audit.json")
    report = {"passed": False, "stage": args.stage, "source_manifest_sha256": snapshot,
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output, report)
    try:
        if args.stage == "reference":
            report.update(reference_audit())
        elif args.stage == "cuda_loss":
            report.update(cuda_loss_audit())
        else:
            from train import environment
            import analyze
            gate = read_json(ROOT / "validation/gate_cuda.json")
            tests = read_json(ROOT / "validation/tests.json")
            for item in (gate, tests, read_json(ROOT / "validation/cuda_loss_audit.json")):
                if item.get("passed") is not True or item.get("source_manifest_sha256") != snapshot:
                    raise ValueError("Incomplete or stale validation evidence")
            if gate.get("device") != "cuda" or gate.get("tests_passed") is not True:
                raise ValueError("No GPU smoke certification")
            report["reference"] = reference_audit()
            report["checkpoint_reload"] = checkpoint_audit(args.smoke_root)
            report["analysis"] = analysis_audit()
            observed = environment()
            analyze.audit_environment(gate["environment"], observed)
            report.update(environment=observed, checks=CHECKS,
                          reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
                          performance_based_gate=False, production_started=False)
        report["passed"] = True
    except BaseException as exc:
        report["error"] = str(exc)
        write_json(output, report)
        raise
    write_json(output, report)
    print(f"PASS {args.stage}: {output}", flush=True)


if __name__ == "__main__":
    main()
