# SABER Technical Supplement

Regenerate the derived analyses and manifests from the repository root:

    .venv/bin/python paper/analysis/build_supplementary_results.py
    .venv/bin/python paper/analysis/build_mechanism_and_phase_supplement.py
    .venv/bin/python paper/analysis/build_update_reference_tables.py
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

The actor-scope analysis is bound to complete paired five-seed artifacts for
FPP, DrawerOpen, and DrawerClose. DrawerOpen fixed-target tables are bound to
the validated all-ten-seed artifact; the earlier live-target diagnostic is not
used. The inclusive-future table reports its original seeds 1--5 cohort and a
separate seeds 6--10 extension. Analysis commands fail rather than silently
mixing a partial cohort into a completed table.
