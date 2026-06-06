# ICDM 2026 Submission — Reproducibility Repository

Reproducibility bundle for the ICDM 2026 submission "When Are Neural Interaction Discoveries Real? Identifiability, Recoverability, and a Pre-Fit Diagnostic."

The paper studies when a neural model's discovered pairwise interactions are
identifiable and recoverable, introduces a pre-fit effective-rank diagnostic and a
two-seed stability check, and demonstrates the framework on synthetic data and three
real domains (Beijing air quality, realized volatility, WDI development indicators).

---

## What's in this repository

```
.
├── README.md                            ← this file (landing page)
├── ICDM2026_GNAVAR_pipeline.ipynb       ← Colab notebook: upload the tarball, Run all
├── verify_paper_numbers.py              ← audit script (Path A entry point)
├── verifier_core.py                     ← the CHECKS list (surfaced for direct viewing)
├── requirements.txt                     ← dependencies for the audit (numpy, pandas)
├── PAPER_TO_CODE_TRACEABILITY.md        ← every paper claim → check → source artifact
├── BUNDLE_README.md                     ← detailed README from inside the bundle
└── gnavar-icdm-reproducibility.tar.gz   ← the full reproducibility bundle (~0.5 MB)
```

The canonical artifact is the **bundle** (`gnavar-icdm-reproducibility.tar.gz`).
Everything else at the repository root is either a copy of a file from inside the
bundle, surfaced for direct GitHub viewing, or this landing README.

---
This bundle is designed to run in Colab to allow reviewers take advantage of free compute that Colab offers. 

*Prefer a terminal? After extracting the tarball:*
```bash
tar xzf gnavar-icdm-reproducibility.tar.gz
pip install -r requirements.txt      # numpy + pandas
python verify_paper_numbers.py       # prints 42/42 checks passed
```
## Reproduce every number in the paper (Colab, no setup)

The fastest path. **No training, no GPU, no datasets, ~30 seconds.** It recomputes every
reported number from the committed result files and asserts each matches the paper.

1. Download **`gnavar-icdm-reproducibility.tar.gz`** from this repository.
2. Open the notebook **`ICDM2026_GNAVAR_pipeline.ipynb`** in
   [Google Colab](https://colab.research.google.com) (**File > Upload notebook**).
3. If you plan to re-run the entire experiment (not just reproduce numbers), connect to GPU.
4. Run cell 1 in path A; upload the tarball when prompted. Run cell 2 to reproduce paper numbers.
The notebook unpacks the tarball and runs the audit. Expected output: `42/42 checks passed`.

### What the 42 checks cover

Every numerical claim in the paper, organized by section:

- §VI-A synthetic recovery vs. sample size, and gate L2-error decay
- §VI-B support-collapse transition (effective rank and recovery vs. coupling)
- §VI-C capacity-matched baseline comparison (G-NAVAR vs. GA2M vs. MLP vs. additive)
- §VII the three-domain taxonomy: Beijing (recoverable), WDI (rich support but not
  recoverable), realized volatility (support collapse) — effective rank, modulator
  rankings, seed agreement, and forecasting comparisons

Quantities that are stable under reseeding (recovery counts, parameter counts, rank
orderings) are checked exactly or within a tight tolerance; seed- or hardware-sensitive
quantities (held-out MSEs, cross-fit margins, seed-agreement fractions) are checked as
ranges or as structural properties. For the WDI domain the check asserts disagreement
between independent seeds, since non-recovery is the reported result.

### Reproducible results table

Every row is checked by `verify_paper_numbers.py` against the cited artifact (the listed
paths are inside `gnavar-icdm-reproducibility.tar.gz`). A per-claim cross-reference to the
exact check is in [`PAPER_TO_CODE_TRACEABILITY.md`](PAPER_TO_CODE_TRACEABILITY.md).

| Paper claim | Value | Check | Source artifact |
|---|---|---|---|
| **Synthetic recovery vs. sample size** (§VI-A) | 0/5, 4/5, 5/5, 5/5 at T = 1k/5k/25k/100k | exact | `results/experiment1/results.csv` |
| Gate L² error shrinks with T | g123 0.31→0.27, g145 0.29→0.15 | tol | `results/experiment1/results.csv` |
| **Support-collapse transition** (§VI-B) | effective rank 3.00 → 1.54 as ρ→1 | tol + monotone | `results/experiment2/results.csv` |
| Recovery vanishes at collapse | 0/10 seeds at ρ ∈ {0.99, 1.0} | exact | `results/experiment2/results.csv` |
| **No interpretability tax** (§VI-C, Table II) | G-NAVAR = GA²M = MLP MSE ≈ 0.0106; additive ≈ 0.17 | tol | `results/experiment_gating_value/results.csv` |
| Recovery under rich support | GA²M 15/15, G-NAVAR 12/15 | exact | `results/experiment_gating_value/results.csv` |
| Competitors capacity-matched ≥ G-NAVAR | params 2065 / 2251 / 2833 | exact | `results/experiment_gating_value/results.csv` |
| **Beijing — recoverable** (§VII) | Setup A r_eff > 4; TEMP rank-1 at 4/4 sites | range + exact | `results/experiment_beijing/` |
| Beijing Setup B (low support) | r_eff < 3.02; expected modulator rank-1 at 0/4 | range + exact | `results/experiment_beijing/` |
| Beijing forecasting | G-NAVAR wins MSE 6/8 | exact | `results/experiment_beijing/results.csv` |
| **WDI — rich support, not recoverable** (§VII) | r_eff ≈ 4.47; two seeds **disagree** on top modulator | tol + structural | `results/experiment_wdi_resource_curse/results.csv` |
| WDI no predictive gain | additive MSE ≤ G-NAVAR MSE (1.19 vs 1.26) | structural + tol | `results/experiment_wdi_resource_curse/results.csv` |
| **Realized volatility — support collapse** (§VII, Table III) | r_eff < 2 at all 4 targets | exact | `results/experiment_rv/results.csv` |
| RV confidently-arbitrary signature | 4 distinct top modulators; SPX #1 in 5/12 edges; seed agreement ≈ 0.44 | exact + tol | `results/experiment_rv/results.csv` |
| RV margins deceptively large | 1.9× – 5.8× | range | `results/experiment_rv/results.csv` |

The three real domains realize the three logical states the theory permits: recoverable
(Beijing), rich support but not recoverable (WDI), and support collapse (realized
volatility).

---

## Regenerate the result files from scratch (optional, GPU)

For verifying that the pipeline reproduces, not just that the numbers match the saved
artifacts, continue with path B in the notebook. If you haven't set the GPU for path A, do so now,
but all artifacts in memory will be removed and you will need to re-upload the tarball.

**Runtime** Approximately 22 minutes on Colab G4 GPU.

The bundle's `notebooks/` folder has one notebook per experiment; each
regenerates one `results/` subfolder on a GPU. The three synthetic experiments need no
external data; the three real-data experiments need their dataset (provided in the \data folder). 

Re-running on different hardware reproduces the exact-checked quantities (recovery counts,
parameter counts, rank orderings); seed- and hardware-sensitive quantities differ slightly
and are verified as ranges. Each `results/<folder>/metadata.json` records the original
run environment.

---

## Software requirements

Audit path: Python 3.10+ with `numpy` and `pandas` (pinned in `requirements.txt`).
Regeneration path: additionally `torch` and `scikit-learn`.

## License

- **Code**: MIT (see `LICENSE` inside the bundle).
- **Datasets**: publicly available from their original sources, cited in the paper. Provided here for reproducibility purposes; should not be reused and reproduced without citing the original dataset sources.
- **Paper**: ICDM submission; all rights reserved by the authors.

## Contact

This repository is anonymized for double-blind review. Please direct questions through the
review system for this submission.
