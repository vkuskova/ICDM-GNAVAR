# Experiment notebooks (Path B: regenerate artifacts)

Each notebook regenerates exactly one `results/` subfolder, then `verify_paper_numbers.py`
(Path A) checks the paper's numbers against it. **Colab-first:** open any notebook with the
badge in the top-level README and choose `Runtime > Run all`. The first cell ("Reproducibility
bootstrap") resolves all paths automatically for three settings: the authors' Google Drive,
a fresh Colab (clones this repo), or a local clone. No edits required.

| Notebook | Regenerates | Raw data needed |
|---|---|---|
| `experiment1.ipynb` | `results/experiment1/` — synthetic recovery vs. sample size (§VI-A) | none (synthetic) |
| `experiment2.ipynb` | `results/experiment2/` — support-collapse ρ-sweep (§VI-B) | none (synthetic) |
| `experiment_gating_value.ipynb` | `results/experiment_gating_value/` — capacity-matched baselines (§VI-C) | none (synthetic) |
| `experiment_beijing.ipynb` | `results/experiment_beijing/` — Beijing air quality (§VII) | `data/beijing_multisite.csv` |
| `experiment_rv.ipynb` | `results/experiment_rv/` — realized volatility (§VII) | `data/rv_dataset.csv` |
| `experiment_wdi_resource_curse.ipynb` | `results/experiment_wdi_resource_curse/` — WDI (§VII) | `data/wdi_reversal_panel.csv` |

## Notes
- **These are the notebooks as run to produce the committed artifacts.** They require a GPU
  (Colab T4/A100 is sufficient) and, for the three real-data experiments, the raw dataset placed
  in `data/` (not redistributed here — see `data/README.md` for public sources). The three
  synthetic notebooks need no external data and run end to end on a free Colab GPU.
- **Re-running will not bit-reproduce the committed CSVs.** Seed- and hardware-sensitive
  quantities (MSEs, cross-fit margins, seed-agreement fractions) shift slightly; the verifier
  checks those as ranges. Exact-checked quantities (recovery counts, parameter counts, rank
  orderings) should match. Each `results/<folder>/metadata.json` records the original environment.
- **Path A does not require any of this.** To simply confirm the paper's numbers match the
  committed artifacts, run `python verify_paper_numbers.py` (CPU, ~seconds).
