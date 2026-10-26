# SABER: Post-AAAI Experiments for ICML 2027

Executed-experiment archive following the September 25 AAAI reviews.
The submitted AAAI paper and its artifacts are not modified here.

## Experiment Index

The experiment READMEs contain results or a timestamped progress snapshot.
They are not live dashboards. See [index.json](experiments/index.json) for the exact capture time and counts.

| Experiment | Scope | Status |
| --- | --- | --- |
| [SABER / Random / TD-error Top-k](experiments/draweropen_ddpg_saber_vs_random_vs_td_topk/) | DrawerOpen DDPG; 3 x 10 seeds x 2M | Completed, 30 fresh runs |
| [SABER / PAL](experiments/draweropen_ddpg_saber_vs_pal/) | DrawerOpen DDPG; 10 new PAL seeds x 2M | Completed; reuses 10 SABER references |
| [Offline retrieval and failure diagnostics](experiments/draweropen_ddpg_retrieval_and_failure_diagnostics/) | DrawerOpen DDPG; 10 checkpoints and 30 training logs | Completed; no new training |
| [SABER Tail / Random-512 retrieval](experiments/draweropen_ddpg_saber_tail_vs_random512/) | DrawerOpen DDPG; 10 new Random-512 runs | Completed; reuses 10 Tail references; includes final retrieval audits |
| [HER / SABER](experiments/draweropen_drawerclose_sac_her_vs_saber/) | DrawerOpen/DrawerClose SAC; 2 methods x 5 seeds x 2M per task | Completed, 20 unique runs |
| [HGR / PER](experiments/draweropen_drawerclose_sac_hgr_per/) | DrawerOpen/DrawerClose SAC; 2 methods x 5 seeds x 2M per task | Completed, 20 new runs; 20 HER/SABER references reused |

Only these six executed experiment/analysis families are published: five training
comparisons and one completed offline analysis. Unexecuted preparation versions,
the superseded diagnostic pilot, and a duplicate SAC source directory are excluded.
Folder names identify the benchmark, backbone and comparison or diagnostic.
They do not contain dates, version suffixes or seed-range abbreviations. Each
experiment has one source folder; seed sets are in its README and protocol, and
original execution paths remain in provenance records.

The SAC cohort includes three DrawerOpen HER runs completed before the power loss.
All 78 files of their original exported source are byte-identical to files already
retained in the final SAC folder. Removing the duplicate directory does not remove
those results or replace their implementation. The final report preserves the
interruption/reuse history and all unfavorable completed results.

The old [October 6 report](2026-10-06_post_review_experiment_status_ko.md) remains a historical record.
Its old local-workspace links and then-current progress are not the current archive layout or current status.

Completed SAC compact results are organized by task alongside the repository's
other task/backbone folders: [DrawerOpen SAC](../metaworld_Drawer_Open_SAC/) and
[DrawerClose SAC](../metaworld_Drawer_Close_SAC/). Together they contain 40 unique
2M-step runs, not 60: the HGR/PER analysis reuses the same HER/SABER reference runs.
The six archived families now total 90 completed new training runs. This count
does not include subsequently launched DDPG seed-extension experiments.

The later [DrawerOpen DDPG HER/SABER seed-extension artifact](../metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_her_saber_seed_extension_2m/README.md)
now reports 10 completed new executions (seeds 6-10 for each of HER and SABER)
alongside 10 historical references. It separately records the still-incomplete
eight-condition queue. These executions are not included in the six-family count
above, and are not substituted for the earlier assignment experiment.

```bash
python3 ICML_27/tools/ddpg_seed_extension_results.py verify
```

## What Is Included

Each experiment directory has a current `README.md`, frozen `source/`, compact `results/`
where available, and `provenance.json`. Source, tests, upstream snapshots, original
protocols, third-party notices, and source SHA256 manifests are copied without edits.
The original `source/README.md` is deliberately preserved, including historical
pending-status text and local-machine instructions. The outer README is the status reference.

Exported reports replace the originating workspace path with `<WORKSPACE>` and
serialize nonfinite JSON values as `null`; provenance records both original and
exported hashes. Full training metrics, checkpoints, replay buffers, process locks and validation
certificates are not included. The task-specific SAC artifacts additionally retain
every evaluation checkpoint, configs, summaries and recorded execution environments.
Missing other artifacts must not be
mistaken for passing a production/reuse validation gate.

## Verify or Prepare the Code

From the repository root, using Python 3.12 (verification needs only the standard library):

```bash
python3 ICML_27/tools/verify_archive.py
python3 ICML_27/tools/materialize_sources.py /tmp/saber-post-review-sources
```

The second command uses `provenance.json` to copy the six retained sources into a
NEW directory with their original sibling names under `experiments_post_review/`.
This preserves frozen source references without using historical names for the
GitHub folders. It does not recreate the
excluded preparation or duplicate directories. It does not install dependencies,
copy results into production paths, launch a queue, or train a model. Do not point it
at a live experiment workspace. It refuses to overwrite an existing destination.

Inspect each materialized protocol before executing anything. The training code needs
PyTorch, NumPy, Gymnasium, Gymnasium Robotics, MetaWorld, MuJoCo and Numba as specified
by the historical experiment environment. This archive does not promise a newly
validated cross-platform environment. Original launchers that require historical
raw records, checkpoints, validation gates, or repo-relative source origins will
remain blocked until those dependencies are provided and validated. Compact reports
support inspection of reported results, not full recomputation of all diagnostics.
Historical audits against an excluded preparation snapshot (for example SAC's v1
parity test) are not standalone tests of this reduced archive. The archive verifier
checks the retained source hashes, results and links without those historical inputs.

The compact SAC artifacts support full reanalysis of performance metrics and paired
AUC intervals from included evaluation CSVs, without historical raw directories:

```bash
python3 ICML_27/tools/sac_results.py verify
```

This command requires NumPy (recorded analysis version 2.3.5), uses CPU only, and
does not import a trainer or start training. Frozen source and original records are
not edited during verification.

## Interpretation and Open Work

- DDPG follow-ups concern DrawerOpen, not every task/backbone. Higher sample means with CIs crossing zero do not establish superiority or equivalence.
- Assignment Random chooses high-weight minibatch positions; Random-512 retrieval chooses replay candidates. These are different interventions.
- Failure/saturation associations and final-checkpoint search agreement are descriptive, not causal proof.
- The SAC observation retains original-goal coordinates while HER relabels a separate goal input. The inherited representation is shared across compared SAC methods; its causal effect is unresolved. Executed code is not silently corrected in this archive.
- Runtime includes instrumentation. These experiments do not supply a controlled, matched-instrumentation wall-clock superiority claim.
- Critic-branch/policy-action diagnostics, broader hyperparameter sensitivity, context transfer, and SAC seed 6-10 expansion are not completed experiments in this archive.

## License

Original SABER code is distributed under [MIT](LICENSE), consistent with the existing
code supplement. Preserved PAL third-party code retains its own MIT copyright and
license in each PAL snapshot's `source/third_party/lap_pal/`. Dependency packages
and simulator assets are not redistributed; their licenses remain applicable.

## Maintaining This Archive

Do not edit the live experiment or its frozen source to update a README. Export into
a separate workspace, review the selected diff, and publish only the scoped archive.
`tools/export_local_experiments.py` exports only the six executed families from the
original workspace into a new destination; it never launches training or republishes
preparation versions. Tests and upstream references used by the executed snapshots'
validation are retained inside those snapshots, not listed as extra experiments. Generated
reports are snapshots and require an explicit later update. SHA256 verification
checks bytes, not scientific validity or reproducibility of omitted raw artifacts.
