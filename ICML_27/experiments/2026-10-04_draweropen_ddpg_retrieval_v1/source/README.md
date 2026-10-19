# Online tail versus random retrieval

Independent, frozen copy of the completed DrawerOpen DDPG assignment experiment.
No original source or retained result is modified. See PROTOCOL.md for the fixed
scientific question, candidate sampling unit, statistical plan and failure policy.

Entry points: freeze.py, audit_reuse.py, tests/test_retrieval.py, launch.py,
validate.py, after_training.py and analyze.py. Production requires the same-source
regression, historical-reuse, upstream-tail trajectory and two-arm smoke gates.

Historical Tail seeds 1--10 remain at the source location recorded by the reuse
catalog. New Random-512 seeds 1--10 are under outputs/production/random512.
Raw results and source provenance are local research artifacts, not anonymized
submission material. No new source may be edited after production begins.
