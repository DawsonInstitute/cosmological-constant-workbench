#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"

cd "${REPO_DIR}"

PYTHONPATH=src python papers/lqg_cc_constraints/generate_scan_data.py

cd "${ROOT_DIR}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error lqg_cc_constraints.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error lqg_cc_constraints.tex
  bibtex lqg_cc_constraints
  pdflatex -interaction=nonstopmode -halt-on-error lqg_cc_constraints.tex
  pdflatex -interaction=nonstopmode -halt-on-error lqg_cc_constraints.tex
fi

echo "Built: ${ROOT_DIR}/lqg_cc_constraints.pdf"
