# PI-CAX: Physics-Informed Causal-Structural and Deadline-Aware Adaptive Explainable Intrusion Detection for Cyber-Physical Systems

This repository contains the complete, reproducible research framework, benchmark suite, experimental results, and Q1 journal LaTeX manuscript for **PI-CAX** (*Physics-Informed Causal-Structural and Deadline-Aware Adaptive Explainability for Cyber-Physical Intrusion Detection*).

## Repository Structure

```text
xai_cps_repo/
├── configs/
│   └── default_config.json         # Hyperparameters, seeds, and edge hardware profiles
├── src/
│   ├── cps_benchmarks.py           # Leakage-free SWaT, WADI, and HIL-HAI multi-stage physical benchmarks
│   ├── detectors.py                # 4 CPS IDS architectures (VAR-SSM, TCN-AE, GRU-Seq, ST-GDN)
│   ├── explainers.py               # 6 XAI baselines (Raw-MSE, VG, IG, LIME, KernelSHAP, ShaTS) + PI-CAX
│   ├── evaluation.py               # Detection, 8 XAI metrics, adversarial spoofing, and Wilcoxon/Cliff tests
│   ├── run_all_experiments.py      # Master end-to-end experimental runner
│   └── generate_figures.py         # Publication figure generator
├── results/
│   ├── detection_results.csv       # Point-adjusted and raw detection metrics across datasets & models
│   ├── xai_benchmark_results.csv   # Full multi-metric XAI evaluation table
│   ├── per_attack_xai_records.csv  # Granular per-attack attribution records
│   ├── ablation_results.csv        # 7-configuration architectural ablation results
│   └── statistical_tests.csv       # Paired Wilcoxon signed-rank p-values and Cliff's delta effect sizes
├── figures/
│   ├── fig1_pareto_frontier.png    # Fidelity-Localization-Latency Pareto frontier
│   ├── fig2_actuator_vs_sensor_avgrank.png # Sensor vs. Actuator vs. Multi-Point attribution rank
│   └── fig3_ablation_study.png     # Component-wise ablation analysis
├── manuscript/
│   ├── PI_CAX_XAI_CPS_Q1_Manuscript.tex # Self-contained Q1 journal LaTeX manuscript
│   └── PI_CAX_XAI_CPS_Q1_Manuscript.pdf # Compiled PDF manuscript
└── tests/
    └── test_framework.py           # Automated invariant and reproducibility test suite
```

## Quick Start & Reproducibility Instructions

1. **Run Automated Unit & Invariant Tests**:
   ```bash
   PYTHONPATH=. python3 xai_cps_repo/tests/test_framework.py
   ```
2. **Reproduce All Experimental Tables (`results/*.csv`)**:
   ```bash
   PYTHONPATH=. python3 xai_cps_repo/src/run_all_experiments.py
   ```
3. **Regenerate Publication Figures (`figures/*.png`)**:
   ```bash
   PYTHONPATH=. python3 xai_cps_repo/src/generate_figures.py
   ```
4. **Compile LaTeX Manuscript**:
   ```bash
   cd xai_cps_repo/manuscript && lualatex -interaction=nonstopmode -halt-on-error PI_CAX_XAI_CPS_Q1_Manuscript.tex
   ```
