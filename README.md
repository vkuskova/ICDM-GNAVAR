# ICDM 2026 Submission — Reproducibility Repository

This repository contains the reproducibility package for:

> **When Are Neural Interaction Discoveries Real? Identifiability, Recoverability,
> and a Pre-Fit Diagnostic** — Anonymous (ICDM 2026 submission)

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

## Reproduce every number in the paper (Colab, no setup)

The fastest path. **No training, no GPU, no datasets, ~30 seconds.** It recomputes every
reported number from the committed result files and asserts each matches the paper.

1. Download **`gnavar-icdm-reproducibility.tar.gz`** from this repository.
2. Open the notebook **`ICDM2026_GNAVAR_pipeline.ipynb`** in
   [Google Colab](https://colab.research.google.com) (**File > Upload notebook**).
3. Upload the tarball into the Colab session: **Files** panel (folder icon, left) **> Upload**.
4. **Runtime > Run all.**

The notebook unpacks the tarball and runs the audit. Expected output: `42/42 checks passed`.

*Prefer a terminal? After extracting the tarball:*

```bash
tar xzf gnavar-icdm-reproducibility.tar.gz
pip install -r requirements.txt      # numpy + pandas
python verify_paper_numbers.py       # prints 42/42 checks passed
```

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
ranges. For the WDI domain the check asserts disagreement between independent seeds, since
non-recovery is the reported result. A per-claim cross-reference is in
[`PAPER_TO_CODE_TRACEABILITY.md`](PAPER_TO_CODE_TRACEABILITY.md).

---

## Regenerate the result files from scratch (optional, GPU)

For verifying that the pipeline reproduces, not just that the numbers match the saved
artifacts. The bundle's `notebooks/` folder has one notebook per experiment; each
regenerates one `results/` subfolder on a GPU. The three synthetic experiments need no
external data; the three real-data experiments need their dataset (public sources in
`data/README.md`; the raw data are not redistributed). After regenerating, re-run the audit
to re-check. See `BUNDLE_README.md` for step-by-step instructions and the
notebook-to-folder map.

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
- **Datasets**: publicly available from their original sources, cited in the paper and in
  `data/README.md`; raw data are not redistributed here.
- **Paper**: ICDM submission; all rights reserved by the authors.

## Contact

This repository is anonymized for double-blind review. Please direct questions through the
review system for this submission.
