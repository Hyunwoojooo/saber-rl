# Quick Start

## 1. Create the environment

Python 3.12 is recommended. On the NVIDIA GB10 systems used for validation,
PyTorch 2.12.1+cu130 was provided by the NVIDIA software stack.

```bash
bash environment/install.sh .venv
source .venv/bin/activate
```

If a compatible CUDA-enabled PyTorch is already installed:

```bash
bash environment/install.sh .venv --skip-torch
source .venv/bin/activate
```

The exact validated versions are in `environment/requirements-lock.txt` and
`environment/verified_stack.txt`.

## 2. Verify without training

```bash
python reviewer_tools/verify_package.py
python reviewer_tools/reproduce_main_results.py --check
python reviewer_tools/reproduce_supplement_results.py --check
bash reviewer_tools/run_smoke_tests.sh
```

All commands are package-relative and do not require the original repository.

## 3. Inspect a paper configuration

```bash
python reviewer_tools/launch_experiment.py --list
python reviewer_tools/launch_experiment.py \
  --preset drawer_open_ddpg --seed 1 --dry-run
```

## 4. Validate an environment

This creates the environment and performs the task-specific setup validation
without a production training run:

```bash
python reviewer_tools/launch_experiment.py \
  --preset fpp_ddpg --seed 1 --validate-only --device cpu \
  --execute
```

MuJoCo may require `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` on a headless host.

## 5. Launch a full primary run

```bash
python reviewer_tools/launch_experiment.py \
  --preset fpp_td3 --seed 1 --device cuda \
  --output-root reviewer_runs --execute
```

To run a short integration smoke instead of a paper-budget run:

```bash
python reviewer_tools/launch_experiment.py \
  --preset drawer_close_ddpg --seed 1 --device cpu \
  --steps 5000 --smoke --output-root reviewer_runs --execute
```

`--steps` is an explicit reviewer override. Without it, the fixed paper budget
from the selected preset is used.
