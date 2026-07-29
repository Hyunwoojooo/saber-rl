# SABER Technical Supplement

Regenerate the derived analyses and manifests from the repository root:

    .venv/bin/python paper/analysis/build_supplementary_results.py
    .venv/bin/python paper/analysis/build_technical_supplement_manifest.py

Build with Tectonic:

    cd paper
    tectonic technical_supplement.tex

Alternatively, use two pdfLaTeX passes to resolve references:

    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error technical_supplement.tex
    pdflatex -interaction=nonstopmode -halt-on-error technical_supplement.tex

The current working draft intentionally contains two visible pending result
slots:

1. FPP DDPG Actor+Critic.
2. DrawerOpen DDPG fixed-target diagnostics.

Do not submit the Technical Supplement until both slots are replaced and the
working-draft completion paragraph is removed. Existing DrawerOpen live-target
diagnostic values must not be substituted into the fixed-target rows.
Both analysis commands fail rather than silently mixing a partial five-seed
cohort into a completed table.
