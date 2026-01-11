# CCW report

## Observed
- H0: 67.4 km/s/Mpc
- ΩΛ: 0.6889
- ρΛ,0: 5.283e-10 J/m³

## Naive QFT cutoffs
| cutoff | E_cut (J) | rho_naive (J/m³) | ratio |
| --- | --- | --- | --- |
| electroweak | 1.602e-08 | 1.320e+43 | 2.499e+52 |
| 1TeV | 1.602e-07 | 1.320e+47 | 2.499e+56 |
| planck | 1.956e+09 | 2.934e+111 | 5.553e+120 |

## Mechanism
- name: scalar_field_quintessence

| mechanism | z | rho_de_j_m3 | w_de | params |
| --- | --- | --- | --- | --- |
| scalar_field_quintessence | 0.0 | 5.283138117252511e-10 | -1.0 | {"lam": 2.0, "n_eval": 300, "potential": "exponential", "x0": 0.0, "z_max": 2.0} |
| scalar_field_quintessence | 1.0 | 5.1967714026684475e-09 | 0.9659865609062751 | {"lam": 2.0, "n_eval": 300, "potential": "exponential", "x0": 0.0, "z_max": 2.0} |
| scalar_field_quintessence | 2.0 | 5.861722173025258e-08 | 0.9995024532131547 | {"lam": 2.0, "n_eval": 300, "potential": "exponential", "x0": 0.0, "z_max": 2.0} |

## Constraints
**Params**: {"lam": 2.0, "n_eval": 300, "potential": "exponential", "x0": 0.0, "z_max": 2.0}

- swampland: ✓ PASS — Passes swampland gradient bound: λ=2 >= c_min=1.15
- holographic (z=0.0): ✓ PASS — Passes holographic bound: ρ_DE=5.283e-10 <= ρ_bound=7.669e-10
- holographic (z=1.0): ✓ PASS — Passes holographic bound: ρ_DE=5.197e-09 <= ρ_bound=7.105e-09
- holographic (z=2.0): ✓ PASS — Passes holographic bound: ρ_DE=5.862e-08 <= ρ_bound=6.506e-08

