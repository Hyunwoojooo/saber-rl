# SABER Technical Supplement

Regenerate the derived analyses and manifests from the repository root:

    .venv/bin/python paper/analysis/build_supplementary_results.py
    .venv/bin/python paper/analysis/build_draweropen_fixed_target_summary.py \
        --primary-root <completed-seeds-1-5-results-root> \
        --output-root <empty-output-root>
    .venv/bin/python paper/analysis/build_technical_supplement_manifest.py

Build with Tectonic:

    cd paper
    tectonic technical_supplement.tex

Alternatively, use two pdfLaTeX passes to resolve references:

    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error technical_supplement.tex
    pdflatex -interaction=nonstopmode -halt-on-error technical_supplement.tex

The current working draft intentionally contains one visible pending result
slot:

1. FPP DDPG Actor+Critic.

Do not submit the Technical Supplement until this slot is replaced and the
working-draft completion paragraph is removed. DrawerOpen fixed-target tables
are bound to the validated all-ten-seed artifact; the earlier live-target
diagnostic is not used. Analysis commands fail rather than silently mixing a
partial five-seed cohort into a completed table.
