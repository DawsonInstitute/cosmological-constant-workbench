# Cosmological Constant Workbench

A reproducible, test-first workbench for the cosmological constant problem:

- Reproduce the baseline discrepancy between naive QFT vacuum energy estimates and the observed dark-energy density.
- Provide small, explicit calculators (with units) and a CLI that outputs JSON.
- Create a place to evaluate candidate mechanisms *by their assumptions* and *their observable consequences*, not by curve-fitting.

## Quick start

```bash
cd cosmological-constant-workbench
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Baseline: observed ρ_Λ, Λ, and naive QFT cutoffs
ccw-baseline --h0 67.4 --omega-lambda 0.6889 --cutoff planck
ccw-baseline --cutoff electroweak
ccw-baseline --cutoff 1TeV
```

## What this repo is (and isn’t)

- This is **not** a claim of a solved cosmological constant problem.
- It is a **rigorous, incremental** environment to (1) reproduce the problem, (2) encode assumptions in code, and (3) compare mechanisms against constraints.

## Repository layout

- `src/ccw/` — library code
- `scripts/` — runnable scripts (thin wrappers)
- `tests/` — unit tests (pytest)
- `docs/` — roadmap, notes, and provenance

## Next steps

See `docs/TASKS.md` for the working task list.
