# SABER Technical Supplement

Regenerate the derived analyses and manifests from the repository root:

    .venv/bin/python paper/analysis/build_supplementary_results.py
    .venv/bin/python paper/analysis/build_technical_supplement_manifest.py

Build from the repository root with the AAAI-27 author-kit style on the TeX
search path:

    cd paper
    TEXINPUTS=../AuthorKit27: pdflatex -interaction=nonstopmode technical_supplement.tex
    TEXINPUTS=../AuthorKit27: pdflatex -interaction=nonstopmode technical_supplement.tex

The current working draft intentionally contains two visible pending result
slots:

1. FPP DDPG Actor+Critic.
2. DrawerOpen DDPG fixed-target diagnostics.

Do not submit the Technical Supplement until both slots are replaced and the
working-draft completion paragraph is removed. Existing DrawerOpen live-target
diagnostic values must not be substituted into the fixed-target rows.
Both analysis commands fail rather than silently mixing a partial five-seed
cohort into a completed table.
