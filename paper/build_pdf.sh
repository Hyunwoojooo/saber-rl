#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
tex_file="${1:-saber_aaai27_draft_kr.tex}"
job_name="${tex_file%.tex}"

cd "$script_dir"

pdflatex -interaction=nonstopmode -halt-on-error "$tex_file"
bibtex "$job_name"
pdflatex -interaction=nonstopmode -halt-on-error "$tex_file"
pdflatex -interaction=nonstopmode -halt-on-error "$tex_file"

mkdir -p "$repo_root/output/pdf"
cp "$job_name.pdf" "$repo_root/output/pdf/${job_name}_latest.pdf"

echo "Built $repo_root/output/pdf/${job_name}_latest.pdf"
