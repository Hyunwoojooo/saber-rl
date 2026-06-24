# SABER AAAI-27 LaTeX Draft

This folder contains the LaTeX drafting source for the SABER AAAI-27 paper.

## Files

- `saber_aaai27_draft_kr.tex`: Korean drafting version in AAAI-style structure.
- `saber_refs.bib`: Initial bibliography entries.
- `aaai2027.sty`: AAAI-27 style file copied from `AuthorKit27/`.
- `aaai2027.bst`: AAAI-27 bibliography style copied from `AuthorKit27/`.

## Important

`saber_aaai27_draft_kr.tex` is not final-submission ready. It contains Korean prose and uses `kotex` for drafting. For AAAI-27 submission:

1. Translate the paper body to English.
2. Remove `\usepackage{kotex}`.
3. Keep `\usepackage[submission]{aaai2027}`.
4. Compile with PDFLaTeX.
5. Submit a single `.tex` source file plus `.bib` and only the figures actually used.

The final AAAI technical content limit is 7 pages, excluding references.
