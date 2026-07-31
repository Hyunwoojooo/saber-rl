# DrawerClose REDQ-HER scaling reference

This compact artifact validates and summarizes REDQ-HER TQB seeds 1--5 on DrawerClose. Every seed completed 2M environment steps, 7,980,004 updates, and the exact 200-point 10k--2M evaluation grid. AUC and final-quarter AUC are recomputed without smoothing, interpolation, extrapolation, or forward filling.

REDQ-HER uses UTD 4, ten critics, a random two-critic target subset, automatic entropy adjustment, and target-Q bounding to [-50,0]. The DDPG-HER and SABER rows are imported from the verified primary seed artifact for context. This is a scaling and architecture reference, not a resource-matched post-HER allocation control.

Raw run outputs are not copied here. source_manifest.json binds their canonical repository-relative paths and SHA256 digests; artifact_manifest.json binds this compact package.
