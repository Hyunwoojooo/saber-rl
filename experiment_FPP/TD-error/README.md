# TD-error

Purpose: Critic-only TD-error loss-weighting baseline.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/TD-error/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/TD-error/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/TD-error/run_all_seeds.sh --device auto
```
