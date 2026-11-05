"""Unit tests + five 10,200-step smoke runs, never the 500k cohort."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from bootstrap import ROOT
from common import environment, sha, utc, verify_source, write_json
from protocol import ARMS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=verify_source()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,source_manifest_sha256=source,environment=environment(),started_utc=utc(),runs=[])
    try:
        with (out/'unit_tests.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'tests/test_a1.py'),'-v'],stdout=log,stderr=subprocess.STDOUT,check=True)
        # The reference agent/selector parity is covered by real optimizer unit
        # tests; the on/off pair additionally compares entire live trajectories.
        for arm,diagnostics in [(a,'on') for a in ARMS]+[('saber','off')]:
            dest=out/'runs'/f'{arm}_{diagnostics}'
            print('SMOKE_START',arm,diagnostics,flush=True)
            with (out/f'{arm}_{diagnostics}.log').open('x') as log:
                subprocess.run([sys.executable,str(ROOT/'train.py'),'--arm',arm,'--seed','101',
                    '--profile','smoke','--diagnostics',diagnostics,'--output',str(dest)],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            report['runs'].append(dict(arm=arm,diagnostics=diagnostics,path=str(dest)))
            print('SMOKE_COMPLETE',arm,diagnostics,flush=True)
        import numpy as np
        import torch
        spec=importlib.util.spec_from_file_location('a1_test_helpers',ROOT/'tests/test_a1.py')
        helpers=importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
        load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
        left=load(out/'runs/saber_on/run/checkpoint.pt')
        right=load(out/'runs/saber_off/run/checkpoint.pt')
        parity_keys=['agent','replay','gap_stats','episode_state','eval_history_env','last_loss',
            'noise_rng_state','sample_rng_state','weight_rng_state','assignment_rng_state','td_tie_rng_state',
            'python_random_state','numpy_global_rng_state','torch_cpu_rng_state','torch_cuda_rng_state_all',
            'total_updates','global_step','assignment_diagnostics_state']
        for key in parity_keys:
            if not helpers.equal_tree(left[key],right[key]):
                raise AssertionError('Observer changed live training: '+key)
        if sha(out/'runs/saber_on/run/assignment_windows.csv') != sha(out/'runs/saber_off/run/assignment_windows.csv'):
            raise AssertionError('Observer changed assignment history')
        del left,right
        baseline=load(out/'runs/her_on/scheduled_checkpoints/step_0004800.pt')
        with np.load(out/'runs/her_on/diagnostics/fixed_probe_bank.npz') as bank:
            fixed={k:bank[k].copy() for k in bank.files}
        for arm in ARMS[1:]:
            other=load(out/f'runs/{arm}_on/scheduled_checkpoints/step_0004800.pt')
            for key in ('agent','replay','noise_rng_state','sample_rng_state','python_random_state',
                        'numpy_global_rng_state','torch_cpu_rng_state'):
                if not helpers.equal_tree(baseline[key],other[key]):
                    raise AssertionError('Pre-learning state differs across arms: '+key)
            with np.load(out/f'runs/{arm}_on/diagnostics/fixed_probe_bank.npz') as bank:
                for key in bank.files:
                    if not np.array_equal(fixed[key],bank[key]):
                        raise AssertionError('Fixed probe bank differs across arms')
            del other
        if source != verify_source():
            raise AssertionError('Source changed during validation')
        report.update(passed=True,observer_parity_keys=parity_keys,prelearning_checkpoint_parity=True,
            fixed_probe_bank_parity=True,smoke_steps_per_run=10200,smoke_training_steps_total=51000,
            new_pilot_runs=0,checked_learning_updates_per_smoke=5201)
    finally:
        report['finished_utc']=utc()
        write_json(out/'report.json',report)
    if report['passed']:
        (ROOT/'validation').mkdir(exist_ok=True)
        write_json(ROOT/'validation/gate.json',dict(passed=True,source_manifest_sha256=source,
            environment=report['environment'],report_path=str(out/'report.json'),
            report_sha256=sha(out/'report.json'),validated_utc=utc()))
        print('VALIDATION_PASSED',str(out/'report.json'),flush=True)


if __name__=='__main__':
    main()
