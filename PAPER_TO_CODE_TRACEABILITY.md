# Paper-to-code traceability

Every numerical claim in the paper is checked by `verify_paper_numbers.py`, which calls a
check defined in `src/verifier_core.py` (the `CHECKS` list). Each check recomputes the value
from a committed file under `results/` and compares it to the value reported in the paper.
Run the audit via `ICDM2026_GNAVAR_pipeline.ipynb` (Colab) or `python verify_paper_numbers.py`.

| Paper section | Claim | Check kind | Source artifact |
|---|---|---|---|
| §VI-A | Synthetic recovery seeds @ T=1k | exact | `results/experiment1/` |
| §VI-A | Synthetic recovery seeds @ T=5k | exact | `results/experiment1/` |
| §VI-A | Synthetic recovery seeds @ T=25k | exact | `results/experiment1/` |
| §VI-A | Synthetic recovery seeds @ T=100k | exact | `results/experiment1/` |
| §VI-A | Replications per cell = 5 | exact | `results/experiment1/` |
| §VI-A | Gate err g123 @ T=1k (approx 0.31) | tol | `results/experiment1/` |
| §VI-A | Gate err g123 @ T=100k (approx 0.27) | tol | `results/experiment1/` |
| §VI-A | Gate err g145 @ T=1k (approx 0.29) | tol | `results/experiment1/` |
| §VI-A | Gate err g145 @ T=100k (approx 0.15) | tol | `results/experiment1/` |
| §VI-B | r_eff at rho=0 (approx 3.00) | tol | `results/experiment2/` |
| §VI-B | r_eff at rho=1 (approx 1.54) | tol | `results/experiment2/` |
| §VI-B | r_eff strictly decreasing in rho | exact | `results/experiment2/` |
| §VI-B | Both-modulator recovery @ rho=0.99 = 0/10 | exact | `results/experiment2/` |
| §VI-B | Both-modulator recovery @ rho=1.0 = 0/10 | exact | `results/experiment2/` |
| §VI-C | Additive NAVAR MSE @ T=100k (approx 0.17) | tol | `results/experiment_gating_value/` |
| §VI-C | G-NAVAR MSE @ T=100k (approx 0.0106) | tol | `results/experiment_gating_value/` |
| §VI-C | GA2M MSE @ T=100k (approx 0.0106) | tol | `results/experiment_gating_value/` |
| §VI-C | Black-box MLP MSE @ T=100k (approx 0.0106) | tol | `results/experiment_gating_value/` |
| §VI-C | GA2M recovers true pair (15/15) | exact | `results/experiment_gating_value/` |
| §VI-C | G-NAVAR recovers true pair (12/15) | exact | `results/experiment_gating_value/` |
| §VI-C | Total runs = 15 | exact | `results/experiment_gating_value/` |
| §VI-C | G-NAVAR params = 2065 | exact | `results/experiment_gating_value/` |
| §VI-C | GA2M params >= G-NAVAR (2251) | exact | `results/experiment_gating_value/` |
| §VI-C | MLP params >= G-NAVAR (2833) | exact | `results/experiment_gating_value/` |
| §VII | Beijing Setup A r_eff all > 4 | range | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | Beijing Setup B r_eff all < 3.02 | range | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | Beijing Setup A: TEMP rank-1 at 4/4 sites | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | Beijing Setup B: expected mod rank-1 at 0/4 sites | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | Beijing Setup A margin range approx 2.0x-59.3x | range | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | Beijing G-NAVAR MSE wins 6/8 | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | WDI r_eff approx 4.47 (rich) | tol | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | WDI two seeds DISAGREE on top modulator | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | WDI G-NAVAR MSE approx 1.26 (no gain over additive) | tol | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | WDI additive MSE approx 1.19 | tol | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | WDI additive <= G-NAVAR (no interaction gain) | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV r_eff < 2 at all targets | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV top modulator differs across all 4 targets | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV number of targets = 4 | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV SPX ranks #1 in 5/12 peer edges | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV mean seed agreement approx 0.44 | tol | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV within-fit margin range approx 1.9x-5.8x | range | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |
| §VII | RV G-NAVAR MSE wins 1/4 | exact | `results/experiment_beijing/, experiment_rv/, experiment_wdi_resource_curse/` |

**Total: 42 checks.** A correct run prints `42/42 checks passed`.
