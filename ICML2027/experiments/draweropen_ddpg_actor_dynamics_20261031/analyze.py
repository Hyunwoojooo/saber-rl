"""Strict completion checks and descriptive pilot summaries; no CI hunting."""
import argparse
import csv
import json
from pathlib import Path
import bootstrap  # Set CPU thread limits before loading NumPy/PyTorch.
import numpy as np
import torch
from common import write_json, sha
from protocol import ARMS, SEEDS, PROFILES, checkpoints, probe_due, make_config


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def checkpoint_check(path, cfg, step):
    x = torch.load(path,map_location="cpu",weights_only=False)
    n = max(0,step-cfg["warmup_steps"]+1)
    check(x["global_step"] == step and x["total_updates"] == n,"Checkpoint step/update mismatch")
    check(x["agent"]["total_it"] == n,"Agent update mismatch")
    check(x["episode_state"]["at_episode_boundary"],"Checkpoint is not at episode boundary")
    check(x["checkpoint_continuation_permitted"] is False,"Unexpected resume permission")
    check(x["assignment_rule"] == cfg["assignment_rule"],"Wrong assignment rule")
    check(x["replay"]["size"] == min(step,cfg["buffer_capacity"]),"Replay size mismatch")
    check(x["replay"]["ptr"] == step % cfg["buffer_capacity"],"Replay pointer mismatch")
    check("action_index" in x["replay"],"Missing replay index")
    for key in ("python_random_state","numpy_global_rng_state","torch_cpu_rng_state", "noise_rng_state",
                "sample_rng_state","assignment_rng_state","td_tie_rng_state","weight_rng_state"):
        check(key in x,"Missing RNG "+key)
    for module in ("actor","actor_target","critic","critic_target"):
        check(all(torch.isfinite(t).all() for t in x["agent"][module].values()),"Nonfinite checkpoint parameters")
    for optimizer in ("actor_opt","critic_opt"):
        states = x["agent"][optimizer]["state"]
        check(bool(states) == (n > 0),"Missing/unexpected Adam state")
        for state in states.values():
            check(int(state["step"].item()) == n,"Adam step mismatch")
            check(all(torch.isfinite(state[k]).all() for k in ("exp_avg","exp_avg_sq")),"Nonfinite Adam state")
    recorded = dict(x["config"])
    recorded.pop("thread_hint",None); recorded.pop("terminal_convention",None)
    check(recorded == cfg,"Checkpoint config mismatch")
    return dict(step=step,updates=n,replay_size=x["replay"]["size"])


def audit_attempt(root, require_complete=False):
    root = Path(root)
    meta = json.loads((root/"status.json").read_text())
    if require_complete:
        check(meta["status"] == "complete","Attempt is not complete")
    cfg = json.loads((root/"requested_config.json").read_text())
    expected = make_config(meta["arm"],meta["seed"],root,meta["profile"]).as_dict()
    check(cfg == expected,"Requested config differs from protocol")
    actual = json.loads((root/"run/config.json").read_text())
    actual.pop("thread_hint",None); actual.pop("terminal_convention",None)
    check(actual == cfg,"Requested/effective config mismatch")
    total = PROFILES[meta["profile"]]
    eval_steps = sorted(set(range(cfg["eval_freq"],total+1,cfg["eval_freq"])) | {total})
    evaluations = [r for r in read_jsonl(root/"run/metrics.jsonl") if r.get("type") == "eval"]
    steps = [int(r["step"]) for r in evaluations]
    check(steps == eval_steps,"Missing, duplicate or extra evaluation checkpoint")
    success = np.asarray([r["eval/env_success_mean"] for r in evaluations],dtype=float)
    check(np.isfinite(success).all() and ((0 <= success)&(success <= 1)).all(),"Invalid success data")
    cp = checkpoint_check(root/"run/checkpoint.pt",cfg,total)
    metrics = dict(arm=meta["arm"],seed=meta["seed"],profile=meta["profile"],
        auc=float(np.trapezoid(success,steps)/(steps[-1]-steps[0])), auc_start=steps[0],auc_end=steps[-1],
        final=float(success[-1]),hit90=next((s for s,v in zip(steps,success) if v >= .9),None),
        zero_evaluation_run=bool(np.all(success == 0)), updates=cp["updates"],
        eligible_for_pilot_summary=meta["profile"] == "pilot")
    if meta["diagnostics"] == "off":
        return metrics
    records = read_jsonl(root/"diagnostics/checkpoints.jsonl")
    check([r["step"] for r in records] == checkpoints(total),"Missing/extra scheduled checkpoints")
    for r in records:
        p = root/r["file"]
        check(sha(p) == r["sha256"],"Scheduled checkpoint hash mismatch")
        checkpoint_check(p,cfg,r["step"])
    windows = read_jsonl(root/"diagnostics/update_windows.jsonl")
    expected_update = 1
    for row in windows:
        check(row["start_update"] == expected_update,"Update windows overlap or skip updates")
        check(row["end_update"]-row["start_update"]+1 == row["n_updates"],"Invalid window count")
        expected_update = row["end_update"]+1
        check(row["start_step"] == row["start_update"]+4999,"Update/window step mismatch")
        check(row["end_step"] == row["end_update"]+4999,"Update/window end mismatch")
        if meta["arm"] == "her":
            check(row["means"]["actual_high_weight_fraction"] == 0,"HER had high weights")
            check(row["means"]["critic_coefficient_variance"] == 0,"HER had nonuniform weights")
    check(expected_update-1 == metrics["updates"],"Incorrect total observed updates")
    transitions = read_jsonl(root/"diagnostics/training_windows.jsonl")
    check(sum(r["n_transitions"] for r in transitions) == total,"Lost transition observations")
    check([r["step"] for r in read_jsonl(root/"diagnostics/evaluation.jsonl")] == eval_steps,"Evaluation observer mismatch")
    probes = read_jsonl(root/"diagnostics/probes.jsonl")
    expected_probes = [s for s in range(total+1) if probe_due(s,total)]
    for kind in ("fixed_original","fixed_her"):
        rows = [r for r in probes if r["query_source"] == kind]
        check([r["step"] for r in rows] == expected_probes,"Fixed probe schedule mismatch")
        matches = [r for r in rows if r["tanh_derivative_exact_zero_fraction"] == 1]
        metrics["first_observed_all_tanh_zero_"+kind] = matches[0]["step"] if matches else None
    metrics["zero_gradient_censoring_step"] = total
    events = json.loads((root/"diagnostics/summary.json").read_text())
    metrics["first_training_goal_success_step"] = events["first_training_goal_success_step"]
    metrics["first_training_env_success_step"] = events["first_training_env_success_step"]
    saturated = [r for r in windows if r["means"]["assignment_saturated_nonempty"] > 0]
    for end in ("start", "end"):
        metrics["first_support_saturation_window_"+end] = saturated[0][end+"_step"] if saturated else None
    metrics["probe_rows"] = len(probes)
    metrics["scheduled_checkpoints"] = len(records)
    return metrics


def analyze_cohort(root):
    root = Path(root)
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            p = root/"runs"/arm/f"seed_{seed}"
            row = audit_attempt(p,require_complete=True)
            check(row["profile"] == "pilot" and (row["arm"],row["seed"]) == (arm,seed),"Wrong cohort cell")
            rows.append(row)
    out = root/"analysis"
    out.mkdir(exist_ok=False)
    with (out/"seed_metrics.csv").open("x",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        summary[arm] = dict(auc_mean=float(np.mean([r["auc"] for r in subset])),
            auc_sd=float(np.std([r["auc"] for r in subset],ddof=1)),
            final_mean=float(np.mean([r["final"] for r in subset])),
            zero_evaluation_runs=sum(r["zero_evaluation_run"] for r in subset))
    report = dict(purpose="A1 exploratory 500k event-tracking pilot",n_seeds=3,summary=summary,
        confidence_intervals=None,limitation="Observed probe time is not exact onset. Non-events are censored at 500k. No causal inference from temporal order alone.")
    write_json(out/"report.json",report)
    text = ["# A1 초기 학습 추적 pilot", "", "3-seed 탐색 결과. 전체 seed와 zero-success 실행을 포함한다.", "",
            "| 방법 | AUC 평균 ± SD | 마지막 성공률 평균 | 평가 전부 0 |", "|---|---:|---:|---:|"]
    for arm,s in summary.items():
        text.append(f"| {arm} | {s['auc_mean']:.6f} ± {s['auc_sd']:.6f} | {s['final_mean']:.3f} | {s['zero_evaluation_runs']}/3 |")
    text += ["", "| 방법 | Seed | 고정 original probe 전체 tanh 포화 최초 관측 | 고정 HER probe 전체 tanh 포화 최초 관측 | 최초 학습 env success | 최초 support 포화 window |",
             "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        def shown(key):
            return str(r[key]) if r[key] is not None else "500k까지 미관측"
        window = (f"{r['first_support_saturation_window_start']}–{r['first_support_saturation_window_end']}"
                  if r['first_support_saturation_window_start'] is not None else "500k까지 미관측")
        text.append(f"| {r['arm']} | {r['seed']} | {shown('first_observed_all_tanh_zero_fixed_original')} | {shown('first_observed_all_tanh_zero_fixed_her')} | {shown('first_training_env_success_step')} | {window} |")
    text += ["", "AUC는 10k–500k 구간이다. 2M 결과와 합치지 않는다. 고정 probe에서 포화가 처음 관측된 시점은 실제 최초 발생 시점과 다를 수 있다.",
             "CI나 우월성 판정을 생성하지 않는다. Probe Q는 참조 return으로 평가한 정확도가 아니다."]
    (out/"REPORT_ko.md").write_text("\n".join(text)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--attempt",type=Path)
    group.add_argument("--cohort",type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_attempt(args.attempt,True) if args.attempt else analyze_cohort(args.cohort),indent=2))

