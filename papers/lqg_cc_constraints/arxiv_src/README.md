# LQG CC Constraints (Short Paper)

This folder contains a short manuscript and a reproducible data/figure pipeline supporting an empirical constraint rememberable as a "no-go style" bound for the current LQG CC predictor formulation in a bounded parameter domain.

## Key Result

Within a bounded 108-point scan over natural parameter ranges (polymer coefficients, scaling factors, volume cutoffs), the predictor yields vacuum energy densities that exceed the observed dark-energy density by at least ~150 orders of magnitude.

## Reproduce the scan data

From the repo root:

```bash
PYTHONPATH=src python papers/lqg_cc_constraints/generate_scan_data.py
```

This writes:

- `papers/lqg_cc_constraints/data/scan_results.tsv`

## Build the PDF

```bash
bash papers/lqg_cc_constraints/build_paper.sh
```

This regenerates the TSV and compiles `lqg_cc_constraints.tex` using `latexmk` (preferred) or `pdflatex` + `bibtex`.

## Files

- `lqg_cc_constraints.tex`: manuscript (pgfplots histogram reads `data/scan_results.tsv`)
- `lqg_cc_constraints.bib`: bibliography
- `generate_scan_data.py`: bounded parameter scan → stable TSV export
- `data/scan_results.tsv`: scan outputs (108 points)
- `build_paper.sh`: one-command reproducible build
