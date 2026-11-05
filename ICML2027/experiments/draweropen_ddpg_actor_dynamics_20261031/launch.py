"""Explicit fresh pilot launch; planning and status commands never train."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from bootstrap import ROOT
from common import environment, utc, verify_gate, verify_source, write_json
from protocol import schedule, make_config


def resource_check(workers, output):
    if workers not in (1,2,3):
        raise ValueError("Use one to three CPU workers")
    if hasattr(os,"sched_getaffinity") and len(os.sched_getaffinity(0)) < workers:
        raise RuntimeError("Insufficient CPU affinity for requested workers")
    meminfo = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(meminfo['MemAvailable'].split()[0])*1024
    if available < (8+2*workers)*1024**3:
        raise RuntimeError("Insufficient available memory for the conservative pilot allocation")
    if shutil.disk_usage(output).free < 64*1024**3:
        raise RuntimeError("Reserve at least 64 GiB for dense checkpoints before pilot launch")


def plan(output):
    return dict(profile="pilot",total_runs=12,total_environment_steps=6000000,
        cpu_threads_per_run=1,default_workers=2,
        rows=[dict(r,config=make_config(r['arm'],r['seed'],Path(output)/'runs'/r['arm']/f"seed_{r['seed']}").as_dict())
              for r in schedule()])


def run(output,workers):
    source=verify_source()
    verify_gate(source, environment())
    output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    resource_check(workers,output.parent)
    output.mkdir(exist_ok=False)
    declaration=plan(output)
    write_json(output/'plan.json',declaration)
    state=dict(status='running',started_utc=utc(),pid=os.getpid(),workers=workers,
               source_manifest_sha256=source,jobs=[],new_jobs_stopped=False)
    write_json(output/'status.json',state)
    active=[]
    def stop(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        for seed in sorted({r['seed'] for r in declaration['rows']}):
            pending=[r for r in declaration['rows'] if r['seed']==seed]
            while pending or active:
                while pending and len(active)<workers and not state['new_jobs_stopped']:
                    resource_check(workers,output)
                    row=pending.pop(0)
                    destination=output/'runs'/row['arm']/f"seed_{row['seed']}"
                    log=(output/f"{row['arm']}_s{row['seed']}.log").open('x')
                    command=[sys.executable,str(ROOT/'train.py'),'--profile','pilot','--arm',row['arm'],
                             '--seed',str(row['seed']),'--output',str(destination)]
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    record=dict(arm=row['arm'],seed=row['seed'],position=row['position'],pid=process.pid,
                                status='running',output=str(destination),started_utc=utc())
                    state['jobs'].append(record); active.append((process,log,record))
                    write_json(output/'status.json',state)
                for item in list(active):
                    process,log,record=item
                    rc=process.poll()
                    if rc is None: continue
                    log.close(); active.remove(item)
                    record.update(returncode=rc,finished_utc=utc(),status='complete' if rc==0 else 'failed')
                    if rc!=0: state['new_jobs_stopped']=True
                    write_json(output/'status.json',state)
                if state['new_jobs_stopped'] and not active:
                    raise RuntimeError('A worker failed; remaining jobs were not started. No automatic retry.')
                if active: time.sleep(2)
        from analyze import analyze_cohort
        analyze_cohort(output)
        state['status']='complete'
    except KeyboardInterrupt:
        state['status']='interrupted'
        raise
    except BaseException as exc:
        state.update(status='failed',error=str(exc))
        raise
    finally:
        for process,log,record in active:
            os.killpg(process.pid,signal.SIGTERM)
        for process,log,record in active:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            log.close(); record['status']='interrupted_by_launcher'
        state['finished_utc']=utc()
        write_json(output/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('plan','run','status'))
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/pilot_01')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    args=parser.parse_args()
    if args.command=='plan': print(json.dumps(plan(args.output),indent=2))
    elif args.command=='status': print((args.output/'status.json').read_text())
    else: run(args.output,args.workers)

