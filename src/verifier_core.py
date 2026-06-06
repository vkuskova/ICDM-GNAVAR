"""
verifier_core.py
================
Defines CHECKS: one entry per numerical claim in the paper. Each check recomputes
the quantity from a committed results artifact and compares it to the value stated
in the paper. The runner (verify_paper_numbers.py) executes these and prints PASS/FAIL.

Check kinds
-----------
  exact      : computed value must equal expected exactly (integer counts, recovery fractions)
  tol        : |computed - expected| <= tol (means/scores that are seed-fixed and reproducible)
  range      : expected is (lo, hi); computed must lie in [lo, hi] (seed-sensitive quantities
               reported in the paper as a range, e.g. cross-fit margins, MSE ratios)
  membership : computed (a set/flag) must satisfy a stated structural property
               (e.g. "the recovered top modulator differs across all four targets",
                "two seeds disagree") -- used where the *finding itself* is instability,
               so a point value would be the wrong thing to assert.

Design note on honesty
-----------------------
Quantities that are stable under reseeding are checked exactly or with a tight tolerance.
Quantities that are seed- or hardware-sensitive are checked as ranges or as structural
properties, NOT as point values, because asserting a brittle point value would be a
false claim of reproducibility. The WDI checks assert *disagreement* between seeds,
because non-recovery is the reported result.
"""

import json
import os
import pandas as pd
import numpy as np


def _csv(root, rel):
    return pd.read_csv(os.path.join(root, rel))


# ----------------------------------------------------------------------------
# Compute functions: each takes the repository root and returns a scalar / set.
# ----------------------------------------------------------------------------

# --- Synthetic recovery (experiment1) ---------------------------------------
def syn_recovery(root, T):
    df = _csv(root, "results/experiment1/results.csv")
    return int(df[df["T"] == T].modulator_correct.sum())

def syn_recovery_n(root, T):
    df = _csv(root, "results/experiment1/results.csv")
    return int((df["T"] == T).sum())

def syn_gate_err(root, T, col):
    df = _csv(root, "results/experiment1/results.csv")
    return float(df[df["T"] == T][col].mean())

# --- Support-collapse rho-sweep (experiment2) ----------------------------
def rho_reff(root, rho):
    df = _csv(root, "results/experiment2/results.csv")
    return float(df[np.isclose(df.rho, rho)].r_eff_x3_x5.mean())

def rho_recovery(root, rho):
    df = _csv(root, "results/experiment2/results.csv")
    return int(df[np.isclose(df.rho, rho)].modulator_correct_both.sum())

def rho_reff_monotone(root):
    df = _csv(root, "results/experiment2/results.csv")
    m = df.groupby("rho").r_eff_x3_x5.mean()
    return bool(np.all(np.diff(m.values) < 0))  # strictly decreasing in rho

# --- Gating-value (experiment_gating_value) ---------------------------------
def gv_mse(root, T, col):
    df = _csv(root, "results/experiment_gating_value/results.csv")
    return float(df[df["T"] == T][col].mean())

def gv_recovers(root, col):
    df = _csv(root, "results/experiment_gating_value/results.csv")
    return int(df[col].sum())

def gv_n(root):
    df = _csv(root, "results/experiment_gating_value/results.csv")
    return int(len(df))

def gv_param(root, col):
    df = _csv(root, "results/experiment_gating_value/results.csv")
    return int(df[col].iloc[0])

# --- Beijing (experiment_beijing) -------------------------------------------
def bj_reff_range(root, setup):
    df = _csv(root, "results/experiment_beijing/results.csv")
    s = df[df.setup == setup].r_eff
    return float(s.min()), float(s.max())

def bj_expected_rank1_count(root, setup):
    tr = _csv(root, "results/experiment_beijing/triv_score_rankings.csv")
    s = tr[tr.setup == setup]
    return int((s.expected_rank == 1).sum())

def bj_margin_minmax(root, setup):
    tr = _csv(root, "results/experiment_beijing/triv_score_rankings.csv")
    s = tr[tr.setup == setup].top_margin_over_2nd
    return float(s.min()), float(s.max())

def bj_mse_wins(root):
    df = _csv(root, "results/experiment_beijing/results.csv")
    return int((df.mse_ratio_pw_to_gn > 1.0).sum())

# --- Realized volatility (experiment_rv) ------------------------------------
def rv_reff_all_below(root, thresh):
    df = _csv(root, "results/experiment_rv/results.csv")
    return bool(np.all(df.r_eff < thresh))

def rv_distinct_top_modulators(root):
    df = _csv(root, "results/experiment_rv/results.csv")
    return int(df.most_common_top_modulator.nunique())

def rv_n_targets(root):
    df = _csv(root, "results/experiment_rv/results.csv")
    return int(len(df))

def rv_spx_rank1(root):
    df = _csv(root, "results/experiment_rv/results.csv")
    return int(df.spx_rank1_count.sum()), int(df.n_peer_edges.sum())

def rv_mean_seed_agreement(root):
    df = _csv(root, "results/experiment_rv/results.csv")
    return float(df.seed_agreement.mean())

def rv_margin_range(root):
    df = _csv(root, "results/experiment_rv/results.csv")
    return float(df.mean_top_margin.min()), float(df.mean_top_margin.max())

def rv_mse_wins(root):
    df = _csv(root, "results/experiment_rv/results.csv")
    return int((df.mse_ratio_pw_to_gn > 1.0).sum())

# --- WDI (experiment_wdi_resource_curse) ------------------------------------
def wdi_reff(root):
    df = _csv(root, "results/experiment_wdi_resource_curse/results.csv")
    return float(df.r_eff.iloc[0])

def wdi_seed_agree(root):
    df = _csv(root, "results/experiment_wdi_resource_curse/results.csv")
    return bool(df.seed_agree.iloc[0])

def wdi_mse(root, col):
    df = _csv(root, "results/experiment_wdi_resource_curse/results.csv")
    return float(df[col].iloc[0])


# ----------------------------------------------------------------------------
# CHECKS: (section, quantity, kind, expected, fn, [tol])
# 'expected' is the value AS STATED IN THE PAPER. fn recomputes it from artifacts.
# ----------------------------------------------------------------------------

CHECKS = [
    # ---- Synthetic recovery, Sec. VI-A / Table (recovery vs sample size) ----
    ("VI-A", "Synthetic recovery seeds @ T=1k",  "exact", 0, lambda r: syn_recovery(r, 1000)),
    ("VI-A", "Synthetic recovery seeds @ T=5k",  "exact", 4, lambda r: syn_recovery(r, 5000)),
    ("VI-A", "Synthetic recovery seeds @ T=25k", "exact", 5, lambda r: syn_recovery(r, 25000)),
    ("VI-A", "Synthetic recovery seeds @ T=100k","exact", 5, lambda r: syn_recovery(r, 100000)),
    ("VI-A", "Replications per cell = 5",        "exact", 5, lambda r: syn_recovery_n(r, 25000)),
    ("VI-A", "Gate err g123 @ T=1k (approx 0.31)",  "tol", 0.31, lambda r: syn_gate_err(r, 1000, "err_g123"), 0.02),
    ("VI-A", "Gate err g123 @ T=100k (approx 0.27)","tol", 0.27, lambda r: syn_gate_err(r, 100000, "err_g123"), 0.02),
    ("VI-A", "Gate err g145 @ T=1k (approx 0.29)",  "tol", 0.29, lambda r: syn_gate_err(r, 1000, "err_g145"), 0.02),
    ("VI-A", "Gate err g145 @ T=100k (approx 0.15)","tol", 0.15, lambda r: syn_gate_err(r, 100000, "err_g145"), 0.02),

    # ---- Support-collapse transition, Sec. VI-B / Fig. (eff. rank) ----------
    ("VI-B", "r_eff at rho=0 (approx 3.00)",  "tol", 3.00, lambda r: rho_reff(r, 0.00), 0.05),
    ("VI-B", "r_eff at rho=1 (approx 1.54)",  "tol", 1.54, lambda r: rho_reff(r, 1.00), 0.05),
    ("VI-B", "r_eff strictly decreasing in rho", "exact", True, lambda r: rho_reff_monotone(r)),
    ("VI-B", "Both-modulator recovery @ rho=0.99 = 0/10", "exact", 0, lambda r: rho_recovery(r, 0.99)),
    ("VI-B", "Both-modulator recovery @ rho=1.0 = 0/10",  "exact", 0, lambda r: rho_recovery(r, 1.00)),

    # ---- Gating-value comparison, Sec. VI-C / Table II ----------------------
    ("VI-C", "Additive NAVAR MSE @ T=100k (approx 0.17)", "tol", 0.172, lambda r: gv_mse(r, 100000, "additive_mse"), 0.02),
    ("VI-C", "G-NAVAR MSE @ T=100k (approx 0.0106)",      "tol", 0.0106, lambda r: gv_mse(r, 100000, "gnavar_mse"), 0.002),
    ("VI-C", "GA2M MSE @ T=100k (approx 0.0106)",         "tol", 0.0106, lambda r: gv_mse(r, 100000, "addpair_mse"), 0.002),
    ("VI-C", "Black-box MLP MSE @ T=100k (approx 0.0106)","tol", 0.0106, lambda r: gv_mse(r, 100000, "mlp_mse"), 0.002),
    ("VI-C", "GA2M recovers true pair (15/15)",   "exact", 15, lambda r: gv_recovers(r, "addpair_recovers")),
    ("VI-C", "G-NAVAR recovers true pair (12/15)", "exact", 12, lambda r: gv_recovers(r, "gnavar_recovers")),
    ("VI-C", "Total runs = 15",                    "exact", 15, lambda r: gv_n(r)),
    ("VI-C", "G-NAVAR params = 2065",              "exact", 2065, lambda r: gv_param(r, "gnavar_params")),
    ("VI-C", "GA2M params >= G-NAVAR (2251)",      "exact", 2251, lambda r: gv_param(r, "addpair_params")),
    ("VI-C", "MLP params >= G-NAVAR (2833)",       "exact", 2833, lambda r: gv_param(r, "mlp_params")),

    # ---- Beijing (rich support -> recoverable), Sec. VII / Table ------------
    ("VII", "Beijing Setup A r_eff all > 4",  "range", (4.0, 4.25), lambda r: bj_reff_range(r, "A_O3_photochem")),
    ("VII", "Beijing Setup B r_eff all < 3.02","range", (2.6, 3.02), lambda r: bj_reff_range(r, "B_PM10_dispersion")),
    ("VII", "Beijing Setup A: TEMP rank-1 at 4/4 sites", "exact", 4, lambda r: bj_expected_rank1_count(r, "A_O3_photochem")),
    ("VII", "Beijing Setup B: expected mod rank-1 at 0/4 sites", "exact", 0, lambda r: bj_expected_rank1_count(r, "B_PM10_dispersion")),
    ("VII", "Beijing Setup A margin range approx 2.0x-59.3x", "range", (1.9, 60.0), lambda r: bj_margin_minmax(r, "A_O3_photochem")),
    ("VII", "Beijing G-NAVAR MSE wins 6/8", "exact", 6, lambda r: bj_mse_wins(r)),

    # ---- WDI (rich support, weak structure -> not recoverable), Sec. VII ----
    ("VII", "WDI r_eff approx 4.47 (rich)", "tol", 4.47, lambda r: wdi_reff(r), 0.05),
    ("VII", "WDI two seeds DISAGREE on top modulator", "exact", False, lambda r: wdi_seed_agree(r)),
    ("VII", "WDI G-NAVAR MSE approx 1.26 (no gain over additive)", "tol", 1.259, lambda r: wdi_mse(r, "mse_gnavar"), 0.02),
    ("VII", "WDI additive MSE approx 1.19", "tol", 1.194, lambda r: wdi_mse(r, "mse_additive"), 0.02),
    ("VII", "WDI additive <= G-NAVAR (no interaction gain)", "exact", True,
            lambda r: wdi_mse(r, "mse_additive") <= wdi_mse(r, "mse_gnavar")),

    # ---- Realized volatility (collapsed support), Sec. VII / Table III ------
    ("VII", "RV r_eff < 2 at all targets", "exact", True, lambda r: rv_reff_all_below(r, 2.0)),
    ("VII", "RV top modulator differs across all 4 targets", "exact", 4, lambda r: rv_distinct_top_modulators(r)),
    ("VII", "RV number of targets = 4", "exact", 4, lambda r: rv_n_targets(r)),
    ("VII", "RV SPX ranks #1 in 5/12 peer edges", "exact", (5, 12), lambda r: rv_spx_rank1(r)),
    ("VII", "RV mean seed agreement approx 0.44", "tol", 0.44, lambda r: rv_mean_seed_agreement(r), 0.02),
    ("VII", "RV within-fit margin range approx 1.9x-5.8x", "range", (1.8, 6.0), lambda r: rv_margin_range(r)),
    ("VII", "RV G-NAVAR MSE wins 1/4", "exact", 1, lambda r: rv_mse_wins(r)),
]
