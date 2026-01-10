# Baseline: the discrepancy in one page

This repo starts by reproducing the cosmological constant problem as a **scaling mismatch**.

## Observed quantities (ΛCDM bookkeeping)

Let $H_0$ be the Hubble constant and $\Omega_\Lambda$ be the fractional dark-energy density today.

Critical density (mass):

$$\rho_c = \frac{3 H_0^2}{8\pi G}$$

Convert to energy density:

$$\rho_{c,E} = \rho_c c^2$$

Observed dark-energy density:

$$\rho_\Lambda = \Omega_\Lambda\,\rho_{c,E}$$

And the cosmological constant parameter is

$$\Omega_\Lambda = \frac{\Lambda c^2}{3 H_0^2}\quad\Rightarrow\quad\Lambda = \frac{3\,\Omega_\Lambda\,H_0^2}{c^2}.$$

## Naive QFT vacuum energy scaling

A common “demonstration integral” for the vacuum energy density uses a sharp UV cutoff.
For a massless field (one degree of freedom), a compact form is

$$\rho_{\rm naive} \approx \frac{E_{\rm cut}^4}{16\pi^2\,\hbar^3 c^3}.$$

The key point is the scaling $\rho_{\rm naive} \propto E_{\rm cut}^4$.

- Electroweak scale cutoffs ($\sim 10^2\,\rm GeV$) already overshoot by tens of orders of magnitude.
- Planck scale cutoffs ($\sim 10^{19}\,\rm GeV$) overshoot by roughly $\sim 10^{120}$.

This “one-line” estimate is **not a physical prediction** (renormalization and gravity’s coupling to the vacuum are the hard part); it’s the baseline mismatch the rest of the workbench is built to evaluate.
