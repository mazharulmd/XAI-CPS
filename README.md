# Deadline-aware, control-loop-structured explainable AI for intrusion detection in cyber-physical systems

Research code, data configurations, results and manuscript for:

> **Explanations That Arrive in Time: Deadline-Aware, Control-Loop-Structured Explainable AI for Intrusion Detection in Cyber-Physical Systems** (manuscript in `paper/`).

The question this project asks: when an ML anomaly detector raises an alarm in a cyber-physical system, **does the operator receive a *correct* indication of the compromised control loop before the physical process reaches an unsafe state?**

The manuscript combines three parallel lines of work in this repository: the deadline-aware evaluation, LX and TEA (this branch); the independent HAI ground-truth transcription and novelty audit (`codex/xai-cps-research-full-20261031`); and the PI-CAX physics-informed explainer (`feat/pi-cax-q1-manuscript`), re-evaluated here unchanged on real data. See `docs/branch_integration_review.md` for what was taken from each and why.

## What is in this repository

| Path | Content |
|---|---|
| `XAI_IDS_CPS_Papers_Final.xlsx`, `Research Papers/` | Curated literature corpus (33 papers; 28 PDFs present). |
| `docs/01_literature_review_and_gaps.md` | Per-paper matrix, cross-cutting findings F1–F8, novelty check, gaps G1–G5. |
| `docs/02_research_design.md` | Ranked research directions, RQs, hypotheses, threat model, formulation, protocol (fixed before the main runs). |
| `docs/citation_provenance.md` | How each bibliography entry was verified. |
| `configs/hai/hai2103_attacks.csv` | **Loop- and tag-level ground truth for all 50 HAI 21.03 attacks** (transcribed from the HAI technical documentation, matched to the label runs). |
| `configs/hai/hai2103_loops.yaml` | Control-loop grouping of the 79 HAI 21.03 tags. |
| `src/xaicps/sim.py` | **PCL-Sim**: closed-loop process simulator (5 PI loops) with HAI-style attacks and physics-based hazard deadlines. |
| `src/xaicps/data.py` | Leakage-safe loaders (chronological splits, train-only scaling). |
| `src/xaicps/detectors.py` | Linear / MLP / 1D-CNN / GRU forecasting detectors. |
| `src/xaicps/explain.py` | **LX** (loop-exact Shapley), **TEA** (temporal evidence accumulation), H-LX tag refinement, and baselines: RE, saliency, IG, KernelSHAP, LIME, ShaTS-sensor, Fung et al. ensemble, cluster/random-partition controls. |
| `src/xaicps/picax.py` | **PI-CAX** physics-informed loop attribution, ported unchanged (all constants) from `feat/pi-cax-q1-manuscript`; adaptations to our interface are listed in its docstring. |
| `configs/hai/independent/` | Independent transcription of the HAI attack table (from `codex/xai-cps-research-full-20261031`), used to cross-check the ground truth. |
| `src/xaicps/metrics.py`, `src/xaicps/stats.py` | Detection, localization, stability, deletion fidelity, DCL; Friedman / Wilcoxon–Holm / bootstrap. |
| `scripts/` | `run_localization.py` (main experiment), `run_picax.py` (PI-CAX on the same incidents/grids; `--ablation`, `--frozen-stats`), `analyze_integration.py` (ground-truth cross-check, chance, strata, PI-CAX ablation and tests), `run_benchmark.py` (latency, ShaTS), `run_leakage.py` (random vs blocked split), `analyze.py`, `make_figures.py`, `make_numbers.py`, `run_all.sh`. |
| `results/tables/` | Every table/number used in the paper (CSV/JSON). Raw per-attack pickles are in `results/raw/` (regenerable). |
| `paper/` | Computers & Security submission: `main.tex` (elsarticle; title page, line numbers, declarations), `supplementary.tex`, `highlights.txt`, `submission_checklist.md`, `sections/`, `refs.bib`, `figures/`, `generated/`. |
| `tests/` | Unit tests for the Shapley machinery, TEA and PI-CAX. |

## Reproducing

```bash
pip install -r requirements.txt
```

Get HAI 21.03 (public, ~200 MB) into `data/raw/hai-21.03/`:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/icsdataset/hai data/raw/hai-src
```

```bash
git -C data/raw/hai-src sparse-checkout set hai-21.03 && ln -s hai-src/hai-21.03 data/raw/hai-21.03
```

Run the unit tests:

```bash
python3 -m pytest -q
```

Run every experiment (≈4–6 h on 4 CPU cores; PCL-Sim data is generated deterministically). Phase 3 runs PI-CAX on the incidents, alarms and grids saved by phases 1–2 (`bash scripts/run_all.sh 3` alone takes a few minutes):

```bash
bash scripts/run_all.sh
```

Sensor-noise robustness with the threshold recalibrated on equally noisy validation data (the runs without `--recal` keep the clean threshold, which saturates the detector):

```bash
bash scripts/run_queue.sh scripts/queue_recal.txt 4
```

Phase-2 analysis (reference gap, noise, decoy; writes `results/tables/phase2.csv` and `phase2_tests.csv`):

```bash
python3 scripts/analyze_phase2.py
python3 scripts/analyze_integration.py   # ground-truth cross-check (configs/hai/independent/), chance reference, SP/CV strata
```

Edge-hardware sensitivity (every measured latency multiplied by an assumed factor):

```bash
python3 scripts/analyze.py --datasets sim hai --slowdown 10 && python3 scripts/analyze.py --datasets sim hai --slowdown 100
```

Latency benchmark (run on an idle machine) and ShaTS accuracy:

```bash
python3 scripts/run_benchmark.py --dataset sim && python3 scripts/run_benchmark.py --dataset hai
```

Regenerate tables, figures, paper numbers, and the PDF:

```bash
python3 scripts/analyze.py --datasets sim hai sim-decoy && python3 scripts/make_figures.py && python3 scripts/make_numbers.py
```

```bash
cd paper && latexmk -pdf main.tex && latexmk -pdf supplementary.tex
```

## Leakage controls (summary)

- Detectors are trained on normal data only.
- Validation is the chronologically last 20 % of each normal file.
- Scaling comes from the training portion; residual scales and alarm thresholds come from validation only.
- No test label, test file or test seed is used for fitting or for any hyperparameter choice. All hyperparameters live in `scripts/run_localization.py:CFG` and were fixed before the evaluation runs.
- PCL-Sim train/validation seeds (0–11) are disjoint from test seeds (100–103).

## Known limitations

See Section "Limitations" of the manuscript:
- one real testbed;
- simulated hazard deadlines;
- CPU-only latency (no embedded hardware measured);
- loop maps require engineering metadata;
- a novelty search limited by blocked bibliographic APIs, so the check should be repeated in Scopus/WoS before submission.
