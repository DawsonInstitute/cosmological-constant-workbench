(* verify_claims.wl

Mathematica verification script for the bounded 108-point scan in
papers/lqg_cc_constraints.

Goals:
- Recompute \[Rho]_{\[CapitalLambda],0} from (H0, \[CapitalOmega]_\[CapitalLambda]) in SI.
- Reimplement the (as-implemented) external predictor vacuum-energy sum used by CCW:
  - sinc(x) = sin(\[Pi] x)/(\[Pi] x)
  - golden-ratio modulation in \[Mu]_eff(k)
  - SU(2) 3nj correction via Hypergeometric2F1[-2 k, 1/2; 1; -\[Rho]_k] with \[Rho]_k \[TildeTilde] k
  - bounded sum over k = 1/2, 1, 3/2, ... up to k_max = j_max
  - scale enhancement via \[CapitalLambda]_eff(\[ScriptL]) factor consistent with the predictor's wiring

Run:
  wolfram -script papers/lqg_cc_constraints/verify_claims.wl

Tested with Wolfram Engine 14.x.
*)

ClearAll["Global`*"];

(* --- SI constants (numeric, CODATA) --- *)
(* Using explicit numbers avoids slow unit interpretation in batch runs. *)
c = 299792458.0; (* m/s *)
ħ = 1.054571817*10^-34; (* J*s *)
G = 6.67430*10^-11; (* m^3/(kg*s^2) *)
Mpc = 3.0856775814913673*10^22; (* m *)

(* Planck length (correct) and prefactor *)
lPl = Sqrt[ħ G/c^3];
pref = (ħ c)/(8 Pi lPl^4);

(* Observed rho_Lambda0 from (H0, OmegaLambda) *)
H0 = (67.4*1000)/Mpc; (* s^-1 *)
ΩΛ = 0.6889;
ρcritMass = 3 H0^2/(8 Pi G);
ρcritEnergy = ρcritMass c^2;
ρobs = ΩΛ ρcritEnergy;

(* --- Predictor-inspired knobs --- *)
φ = GoldenRatio;
IMMIRZI0 = 0.2375;
βln = 0.05;
βback = 1.9443254780147017;
δ3nj = 0.1; (* matches predictor default SU2_3NJ_DELTA *)

sinc[x_?NumericQ] := If[Abs[x] < 10^-10, 1 - (Pi x)^2/6, Sin[Pi x]/(Pi x)];

muScale[μ0_?NumericQ, α0_?NumericQ, ℓ_?NumericQ] := Module[{r, ln, α},
  r = ℓ/lPl;
  ln = If[r <= 1, 0, Log[r]];
  α = α0/(1 + βln ln);
  μ0*r^(-α)
];

volumeEigenvalue[jmax_?NumericQ] := Module[{js, s},
  js = Range[1/2, jmax, 1/2];
  s = Total[Sqrt[js (js + 1)]];
  Sqrt[IMMIRZI0]*s
];

gammaScale[α0_?NumericQ, μ0_?NumericQ, jmax_?NumericQ, ℓ_?NumericQ] := Module[
  {μ, r, lnCorr, sincCorr, volFactor},
  μ = muScale[μ0, α0, ℓ];
  r = ℓ/lPl;
  lnCorr = If[r > 1, 1 + βln Log[r], 1.0];
  sincCorr = 1 + 0.1 (lPl/ℓ)^2 sinc[μ]^2;
  volFactor = Sqrt[volumeEigenvalue[jmax]];
  IMMIRZI0*lnCorr*sincCorr*volFactor
];

lambdaEnhancement[γcoeff_?NumericQ, α0_?NumericQ, μ0_?NumericQ, jmax_?NumericQ, ℓ_?NumericQ] := Module[
  {μ, g, sc, goldenEnh},
  μ = muScale[μ0, α0, ℓ];
  g = gammaScale[α0, μ0, jmax, ℓ];
  sc = γcoeff*g*(lPl/ℓ)^2*sinc[μ]^2;
  goldenEnh = 1 + 0.1/φ;
  (1 + sc)*goldenEnh
];

goldenModulation[k_?NumericQ] := 1 + (φ - 1)/φ Cos[2 Pi k/φ];
energyEnhancement[E_?NumericQ] := 1 + 0.2 Exp[-((E - 5.5)/3)^2];

su2Factor[k_?NumericQ] := 1 + δ3nj Hypergeometric2F1[-2 k, 1/2, 1, -k];

rhoPred[μ0_?NumericQ, γcoeff_?NumericQ, α0_?NumericQ, jmax_?NumericQ, ℓ_?NumericQ] := Module[
  {ks, sum, enh, back, μeff, arg, summand},
  ks = Range[1/2, jmax, 1/2];
  sum = Total @ Table[
     μeff = μ0*goldenModulation[k]*energyEnhancement[k];
     arg = μeff*Sqrt[k (k + 1)];
     summand = (2 k + 1) * sinc[arg]^2 * (k (k + 1))^(α0/2) * su2Factor[k];
     summand,
     {k, ks}
  ];
  back = 1 + βback μ0^2;
  enh = lambdaEnhancement[γcoeff, α0, μ0, jmax, ℓ];
  pref*sum*back*enh
];

(* --- Scan grid --- *)
μVals = {0.05, 0.10, 0.15, 0.20};
γVals = {0.3, 1.0, 3.0};
αVals = {0.05, 0.10, 0.20};
jVals = {5, 10, 20};
ℓ0 = 10^-15;

scan = Flatten[
  Table[
    Module[{ρ = rhoPred[μ, γ, α, j, ℓ0], Δ},
      Δ = Log10[ρ/ρobs];
      <|
        "mu_polymer" -> μ,
        "gamma_coefficient" -> γ,
        "alpha_scaling" -> α,
        "volume_eigenvalue_cutoff" -> j,
        "rho_pred_j_m3" -> ρ,
        "rho_obs_j_m3" -> ρobs,
        "log10_ratio" -> Δ
      |>
    ],
    {μ, μVals}, {γ, γVals}, {α, αVals}, {j, jVals}
  ],
  3
];

minAbs = MinimalBy[scan, Abs[#"log10_ratio"] &][[1]];

Print["l_Pl (m) = ", ScientificForm[lPl, 6]];
Print["pref (J/m^3) = ", ScientificForm[pref, 6]];
Print["rho_obs (J/m^3) = ", ScientificForm[ρobs, 6]];
Print["Evaluations = ", Length[scan]];
Print["min |log10(rho_pred/rho_obs)| = ", N[Abs[minAbs["log10_ratio"]], 6]];
Print["argmin params = ", KeyTake[minAbs, {"mu_polymer", "gamma_coefficient", "alpha_scaling", "volume_eigenvalue_cutoff"}]];
Print["argmin rho_pred = ", ScientificForm[minAbs["rho_pred_j_m3"], 6]];

(* Hypergeometric sanity checks *)
Print["_2F1 samples (rho_k=k):"]; 
Print@Table[{k, N[Hypergeometric2F1[-2 k, 1/2, 1, -k], 10]}, {k, {1/2, 1, 3/2, 2}}];

(* Asymptotic behavior of the core summand (without scale enhancement) *)
(* Note: this treats oscillatory sinc^2 in a rough analytic way via Asymptotic. *)
k =.;
summandExpr = (2 k + 1) * (Sin[Pi μ0 Sqrt[k (k + 1)]]/(Pi μ0 Sqrt[k (k + 1)]))^2 * (k (k + 1))^(α0/2);
Print["Asymptotic(summand without SU2/energy mods) as k->∞:"];
Print[Asymptotic[summandExpr, {k, Infinity, 2}]];
