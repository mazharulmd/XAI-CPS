# PI-CAX: Physics-Informed Causal-Structural and Deadline-Aware Adaptive Explainable Intrusion Detection for Cyber-Physical Systems

This branch (`feat/pi-cax-q1-manuscript`) contains the Q1 research framework, benchmark suite (SWaT, WADI, HIL-HAI), 4 detector architectures (`VAR-SSM`, `TCN-AE`, `GRU-Seq`, `ST-GDN`), 6 XAI baselines (`Raw-MSE`, `VG`, `IG`, `LIME`, `KernelSHAP`, `ShaTS`), the proposed **PI-CAX** explainer, 7-configuration ablation studies, Wilcoxon signed-rank statistical tests, and the self-contained LaTeX manuscript (`manuscript/PI_CAX_XAI_CPS_Q1_Manuscript.tex`).

## Key Empirical Highlights (Mean Across SWaT, WADI, and HIL-HAI on ST-GDN)
- **Top-1 / Top-3 Root-Cause Localization**: 75.88% / 87.89% (vs. 12.19%–34.65% Top-1 and 25.79%–53.51% Top-3 for baselines)
- **Discrete Actuator Attribution Rank Error (`ActRank`)**: 0.0000 (resolving actuator attribution collapse)
- **Counterfactual Fidelity (`Comp`) & Stability (`Stab`)**: 0.7571 & 0.9968
- **Adversarial Attribution-Spoofing Robustness (`AdvRob`)**: 84.47%
- **Real-Time Edge Latency (ARM Cortex-A72 Profile)**: 6.78 ms (12.13x faster than ShaTS, 15.54x faster than KernelSHAP), 93.25% hard 10 ms deadline compliance (`2.88 ms` and `100%` compliance in Tier-1 urgent mode)
- **False-Positive Rate Reduction**: -33.6% on SWaT, -25.4% on WADI, -18.9% on HIL-HAI
