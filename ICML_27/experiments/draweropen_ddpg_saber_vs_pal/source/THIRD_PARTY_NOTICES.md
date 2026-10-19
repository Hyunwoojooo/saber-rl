# PAL source provenance

`third_party/lap_pal/PAL_TD3.py` and `third_party/lap_pal/LICENSE` are unmodified
copies from Scott Fujimoto's LAP-PAL repository, MIT License, copyright 2020
Scott Fujimoto. The license text is retained verbatim.

- Repository: https://github.com/sfujim/LAP-PAL
- Revision: a649b1b977e979ed0e49d67335d3c79b555e9cfb
- Code: https://raw.githubusercontent.com/sfujim/LAP-PAL/a649b1b977e979ed0e49d67335d3c79b555e9cfb/continuous/PAL_TD3.py
- License: https://raw.githubusercontent.com/sfujim/LAP-PAL/a649b1b977e979ed0e49d67335d3c79b555e9cfb/LICENSE
- Retrieved: 2026-09-30 UTC

The local `pal.py` adapts the loss and minibatch normalization to a single DDPG
critic. Tests extract the original PAL function from the vendored AST, without
instantiating its TD3 networks or selecting a GPU. Full file SHA256s are stored
in source_manifest.json. The original source is never used as a training entry
point in this experiment.

The SABER/HER/DDPG source is copied from the independent 2026-09-25 assignment
experiment. Its immutable copy and original SHA256 manifest are in upstream/.
This folder does not relicense external simulators or bundle their installations.
