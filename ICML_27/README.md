# SABER: Post-AAAI Experiments for ICML 2027

Research archive of the experiments following the September 25 AAAI reviews.
The submitted AAAI paper and its artifacts are not modified here.

## Experiment Index

The experiment READMEs contain results or a timestamped progress snapshot.
They are not live dashboards. See [index.json](experiments/index.json) for the exact capture time and counts.

| Experiment | Scope | Status |
| --- | --- | --- |
| [Support / Random / TD-error Top-k](experiments/2026-09-25_draweropen_ddpg_assignment/) | DrawerOpen DDPG; 3 x 10 seeds x 2M | Completed, 30 fresh runs |
| [HER + PAL](experiments/2026-10-01_draweropen_ddpg_pal_v2/) | DrawerOpen DDPG; 10 seeds x 2M | Completed; reuses 10 SABER references |
| [Offline retrieval and failure diagnostics](experiments/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/) | 10 checkpoints and 30 training logs | Completed; no new training |
| [Online Tail / Random-512 retrieval](experiments/2026-10-04_draweropen_ddpg_retrieval_v1/) | DrawerOpen DDPG; 10 new Random-512 runs | Completed; reuses 10 Tail references; includes final retrieval audits |
| [Drawer SAC HER / SABER](experiments/2026-10-12_drawer_sac_fpp_v2_recovery_s1_5/) | Open/Close x 2 methods x 5 seeds x 2M | Completed, 20 unique runs |
| [Drawer SAC HGR / PER](experiments/2026-10-16_drawer_sac_hgr_per_s1_5/) | Open/Close x 2 methods x 5 seeds x 2M | In progress; see timestamped per-seed table |

Historical source versions are retained, not counted as extra completed experiments:

- [PAL v1 preparation](experiments/2026-09-30_draweropen_ddpg_pal/): superseded before production.
- [Offline diagnostics v1](experiments/2026-10-02_draweropen_retrieval_failure_diagnostics/): blocked pilot, corrected in v2.
- [SAC v1 preparation](experiments/2026-10-08_drawer_sac_fpp_v1/): stopped waiting queue, no production.
- [SAC original v2 execution](experiments/2026-10-08_drawer_sac_fpp_v2_s1_5/): interrupted by power loss; three completed runs retained, five interrupted attempts rerun from scratch in the recovery cohort.

The old [October 6 report](2026-10-06_post_review_experiment_status_ko.md) remains a historical record.
Its old local-workspace links and then-current progress are not the current archive layout or current status.

## What Is Included

Each dated directory has a current `README.md`, frozen `source/`, compact `results/`
where available, and `provenance.json`. Source, tests, upstream snapshots, original
protocols, third-party notices, and source SHA256 manifests are copied without edits.
The original `source/README.md` is deliberately preserved, including historical
pending-status text and local-machine instructions. The outer README is the status reference.

Exported reports replace the originating workspace path with `<WORKSPACE>` and
serialize nonfinite JSON values as `null`; provenance records both original and
exported hashes. Raw metrics, checkpoints, replay buffers, process locks, validation
certificates, and environments are not included. Missing these artifacts must not be
mistaken for passing a production/reuse validation gate.

## Verify or Prepare the Code

From the repository root, using Python 3.12 (verification needs only the standard library):

```bash
python3 ICML_27/tools/verify_archive.py
python3 ICML_27/tools/materialize_sources.py /tmp/saber-post-review-sources
```

The second command copies sources into a NEW directory with the original sibling
experiment names under `experiments_post_review/`. It does not install dependencies,
copy results into production paths, launch a queue, or train a model. Do not point it
at a live experiment workspace. It refuses to overwrite an existing destination.

Inspect each materialized protocol before executing anything. The training code needs
PyTorch, NumPy, Gymnasium, Gymnasium Robotics, MetaWorld, MuJoCo and Numba as specified
by the historical experiment environment. This archive does not promise a newly
validated cross-platform environment. Original launchers that require historical
raw records, checkpoints, validation gates, or repo-relative source origins will
remain blocked until those dependencies are provided and validated. Compact reports
support inspection of reported results, not full recomputation of all diagnostics.

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
`tools/export_local_experiments.py` performs a one-time read-only export from the
original workspace into a new destination; it never launches training. Generated
reports are snapshots and require an explicit later update. SHA256 verification
checks bytes, not scientific validity or reproducibility of omitted raw artifacts.
