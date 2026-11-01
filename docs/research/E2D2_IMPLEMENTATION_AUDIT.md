# E2D2 implementation and evaluation audit

**Audit date:** 1 November 2026  
**Purpose:** assess methodological overlap and benchmark compatibility; this is not an independent reproduction.

## Source inspected

- Paper: Beny Nugraha and Thomas Bauschert, “E2D2: A Modular Framework for Explanation-Based Drift Detection in Unsupervised Network Intrusion Detection,” IFIP Networking 2026, DOI [10.23919/ifipnetworking70592.2026.11578962](https://doi.org/10.23919/ifipnetworking70592.2026.11578962). The publisher landing page was not readable in this audit.
- Author-maintained implementation and documentation: [DLTeamTUC/E2D2](https://github.com/DLTeamTUC/E2D2).
- The `main` branch notebook `e2d2.ipynb` was fetched through GitHub on 1 November 2026. Its Git blob SHA was `e44cafdf0a6b66934ddc8d978b64131dbae3b270`. The notebook contains saved execution outputs; this audit inspects source and those embedded outputs but does not rerun the notebook or independently validate its reported numbers.
- IEEE CSR 2026's official accepted-paper program separately lists the same authors' DARE paper, “A Framework for Drift-Aware Robust Explanations in Unsupervised Intrusion Detection.” A DOI, full text, and implementation were not located; no details beyond the title/authors are asserted.

## What E2D2 evaluates

The author repository describes a VAE/AE reconstruction-error detector and explanation-derived drift monitoring. It evaluates four attribution methods (raw gradient, input-times-gradient, integrated gradients, SmoothGrad), three population views, and two aggregation strategies. With two detector backbones this gives 48 configurations per dataset/scenario. The notebook reports five network-traffic datasets (three CIC-DDoS2019 subsets, CICIoT2023, and 5GNIDD) and four drift scenarios, with 960 E2D2 configurations and 160 baseline runs in the saved execution output.

Its time-series detection ground truth is a synthetic drift-onset index. Explanation alignment is a separate alarm-window overlap calculation: predicted top-ranked features are compared by Jaccard to `drift_features_names` and `shift_features_names` stored in scenario metadata. The saved output reports mean Jaccard 0.193 and mean ground-truth feature recall 0.200 over 15 dataset/scenario combinations with a non-empty explanation-alignment result. These are embedded notebook outputs, not results reproduced by this project. This endpoint checks agreement with injected drift-feature metadata; the inspected code does not connect those features to independently documented CPS attack targets, manipulated registers, or process tags.

E2D2 is therefore a direct prior work for explanation-space drift detection and drift-explanation summaries. It is not a matched baseline for first-alarm CPS target localization: its data are network-flow features, its event target is synthetic feature-shift metadata, and its outputs do not establish register-to-process-tag agreement. DARE's accepted-paper title strengthens the overlap warning for generic drift-aware robustness, but its method and endpoint remain unknown until the full paper is available.

## Preprocessing and leakage audit

The notebook's cell 0.2 calls `select_features_xai(X_df, y, ...)` before constructing the chronological benign reference and streaming pool. Inside that function:

1. An XGBoost classifier is fitted on a random stratified 80% subset.
2. `cross_val_score` is computed on the full `(X_df, y)` table.
3. ANOVA feature scores are computed from all rows and labels using `f_classif(X_df.values, y)`.
4. TreeSHAP values are evaluated on all rows, and the selected top-20 feature list is derived before the chronological reference/stream split.

The later MinMax scaler is fitted only on the benign reference data, but the selected feature set has already used attack labels from the full timeline, including the streaming pool. Under a label-free online or strict train/reference-only evaluation, this is outcome-informed feature selection and contaminates the held-out stream. If the intended estimand allows offline labeled feature selection, that must be stated as a different, label-assisted setting; the current output should not be treated as evidence for a fully unsupervised, leakage-free pipeline. This audit finding concerns the inspected implementation and does not infer author intent.

## Implications for this study

- Do not claim drift-aware explanation monitoring, global explanation-drift summaries, generic IDS explanation robustness, or feature-set overlap with injected drift as new.
- Keep the candidate endpoint limited to independently sourced CPS attack-target agreement at the first alarm, stratified by evidence provenance and resolution. The difference from E2D2 is a candidate boundary, not a novelty certification; compare the complete E2D2 and DARE papers before submission.
- Do not use E2D2's published metrics as matched performance baselines for HAI, Data4Cyber, or BATADAL.
- For any adapted reproduction, define the label-access regime before fitting. Fit feature selection, scaling, detector thresholds, and explanation baselines using only permitted training/reference data. Report a label-assisted variant separately if scientifically useful.

