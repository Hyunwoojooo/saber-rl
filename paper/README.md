# SABER AAAI-27 LaTeX Sources

The long review draft and the page-limited submission skeleton are maintained separately.

## Files

- `saber_aaai27_draft_kr.tex`: full Korean review draft with appendix and stress evidence.
- `saber_aaai27_submission_kr.tex`: compact English submission skeleton; critical claims and controls stay in the main body.
- `saber_refs.bib`: shared bibliography.
- `aaai2027.sty`: AAAI-27 style file copied from `AuthorKit27/`.
- `aaai2027.bst`: AAAI-27 bibliography style copied from `AuthorKit27/`.
- `build_pdf.sh`: builds either source and copies the stable result to `output/pdf/`.

## Build

```bash
cd paper
./build_pdf.sh saber_aaai27_draft_kr.tex
./build_pdf.sh saber_aaai27_submission_kr.tex
```

The generated PDFs are:

- `output/pdf/saber_aaai27_draft_kr_latest.pdf`
- `output/pdf/saber_aaai27_submission_kr_latest.pdf`

Numerical tables are checked against `analysis/result_pipeline/output/`. FPP SAC comparisons use the exact 50k common-grid artifact under `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/`; native metrics remain preserved for audit.

## Important

Neither source is final-submission ready. The long review draft loads `kotex`; the English submission skeleton does not. For the final AAAI-27 submission:

1. Translate the paper body to English.
2. Remove `\usepackage{kotex}` from the translated long-draft/supplement sources.
3. Keep `\usepackage[submission]{aaai2027}`.
4. Compile with PDFLaTeX.
5. Include the anonymized reproducibility package and only figures used by the submission source.

The final AAAI technical-content limit is 7 pages, excluding references. The long review draft must not be submitted as the main paper.
