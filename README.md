# When Are Neural Interaction Discoveries Real? — Reproducibility Bundle

Reproducibility bundle for the ICDM 2026 submission *"When Are Neural Interaction
Discoveries Real? Identifiability, Recoverability, and a Pre-Fit Diagnostic."*

There are two ways to reproduce the results. **Path A verifies every number in the
paper in about 30 seconds on any laptop (no GPU).** Path B regenerates the underlying
result files from scratch on a GPU. Most reviewers will only need Path A.

---

## Path A — verify every paper number (no GPU, ~30 seconds)

This recomputes each numerical claim in the paper from the committed result files in
`results/` and checks it against the value stated in the paper.

**Step 1.** Download this repository. On the Anonymous GitHub page, click **Download**
(or **ZIP**) at the top right, and unzip it. You now have a folder containing
`verify_paper_numbers.py`, `src/`, `results/`, etc.

**Step 2.** Open a terminal in that folder and run:

```bash
pip install -r requirements.txt      # installs numpy + pandas only
python verify_paper_numbers.py
```

**Step 3.** Read the output. The script prints one `[PASS]`/`[FAIL]` line per claim and
ends with a summary. A correct run prints:

```
RESULT: 42/42 checks passed.
All paper numbers reproduce from the committed artifacts.
```

and exits with code 0. If any check fails, the script exits non-zero and lists the
failures. Nothing else is required — no GPU, no datasets, no training.

---

## Path B — regenerate the result files from scratch (GPU)

Path B re-runs the experiments that produced the files in `results/`. It needs a GPU.
The three synthetic experiments need no external data; the three real-data experiments
need their dataset placed in `data/` (sources listed in `data/README.md`; the raw data
are not redistributed here).

**Option B1 — Google Colab (free GPU, recommended for Path B):**

1. Download and unzip this repository (as in Path A, Step 1).
2. Go to [colab.research.google.com](https://colab.research.google.com), choose
   **File > Upload notebook**, and upload one notebook from the `notebooks/` folder
   (e.g. `experiment1.ipynb`).
3. Also upload the supporting files when prompted by the first cell, **or** the simplest
   route: zip the unzipped repo folder, upload it to your Colab session
   (`Files` pane > upload), unzip it in a cell with `!unzip yourbundle.zip`, then
   `cd` into it. The first ("bootstrap") cell of each notebook resolves paths
   automatically once the repo folder is present in the session.
4. Set the runtime to GPU (**Runtime > Change runtime type > GPU**) and choose
   **Runtime > Run all**.
5. The notebook writes its output to `results/<that experiment>/`. To confirm the
   regenerated files still match the paper, re-run Path A (`python verify_paper_numbers.py`).

**Option B2 — local machine with a GPU:**

```bash
pip install -r requirements.txt
pip install torch scikit-learn jupyter     # Path B extras
jupyter notebook                            # open and run any notebooks/<experiment>.ipynb
python verify_paper_numbers.py              # re-verify after regenerating
```

Each notebook maps to exactly one `results/` subfolder; see `notebooks/README.md` for the
table and which dataset each real-data notebook needs.

> **Note on Path B reproducibility.** Re-running will not bit-for-bit reproduce the committed
> CSVs: seed- and hardware-sensitive quantities (MSEs, cross-fit margins, seed-agreement
> fractions) shift slightly and are checked by Path A as ranges, while exact-checked
> quantities (recovery counts, parameter counts, rank orderings) match. Each
> `results/<folder>/metadata.json` records the original environment.

---

## Headline results

Every row is checked by `verify_paper_numbers.py` against the cited artifact. Quantities
that are stable under reseeding are checked exactly or with a tight tolerance; quantities
that are seed- or hardware-sensitive are checked as **ranges** or **structural properties**
(noted), because asserting a brittle point value would overstate reproducibility.

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

The three real domains realize the three logical states the theory permits — recoverable
(Beijing), unrecoverable despite rich support (WDI), and support collapse (RV) — which is
the paper's empirical taxonomy (Table in §VII).

---

## Repository layout

```
.
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt                # Path A: numpy, pandas. Path B adds torch, sklearn.
├── verify_paper_numbers.py         # Path A runner: recompute + check every paper number
├── src/
│   ├── verifier_core.py            # CHECKS list: one entry per numerical claim
│   └── gnavar_core.py              # model, synthetic generator, fit/eval utilities (Path B)
├── results/                        # committed artifacts (one folder per experiment)
│   ├── experiment1/                # synthetic recovery vs. sample size      (§VI-A)
│   ├── experiment2/             # support-collapse ρ-sweep                 (§VI-B)
│   ├── experiment_gating_value/    # capacity-matched baseline comparison     (§VI-C)
│   ├── experiment_beijing/         # Beijing air quality                      (§VII)
│   ├── experiment_rv/              # realized volatility                      (§VII)
│   └── experiment_wdi_resource_curse/  # WDI development indicators           (§VII)
├── data/
│   └── README.md                   # data dictionary + sources (raw data not redistributed)
└── notebooks/
    └── reproduce.ipynb             # convenience runner (Path A; pointers to Path B)
```

Each `results/<experiment>/` folder contains the canonical `results.csv` (plus
`recovery.csv` / `triv_score_rankings.csv` where relevant) and a `metadata.json`
recording the code SHA-256, data SHA-256, library versions, and hardware, so each
artifact is traceable to the exact run that produced it.

---

## Reproducibility notes (honest)

- **Exact vs. range checks are deliberate.** Recovery counts, parameter counts, rank
  orderings, and SPX-edge counts are stable and checked exactly. MSE values, cross-fit
  margins, and seed-agreement fractions vary with seed and hardware and are checked as
  ranges or tolerances. The WDI checks assert that two seeds **disagree** — non-recovery
  is the reported finding, so a point value would be the wrong thing to verify.
- **Seed sensitivity is a result, not noise.** The realized-volatility and WDI domains
  are reported precisely because their recovered structure is *not* stable across seeds;
  the two-seed stability check in the paper is what detects this.
- **Path B will not bit-reproduce Path A.** Regenerating artifacts on different
  hardware/library versions shifts the seed-sensitive quantities slightly (within the
  checked ranges); the exact-checked quantities should match. See each `metadata.json`
  for the original environment.
- **Raw third-party data are not redistributed**; see `data/README.md` for public sources.

## Citation

See `paper/` for the submission. (Citation details omitted for anonymous review.)
