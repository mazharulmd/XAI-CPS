# Data4Cyber and IPAL baseline compatibility audit

**Audit date:** 31 October 2026  
**Status:** source and input compatibility audit; no new detector result is claimed.

## Why this audit is needed

The Data4Cyber v3 record says baseline results from twelve IPAL detector implementations are reported in the companion publication. The local corpus contains no copy of that companion paper. The IEEE CSR program verifies the paper title and authors, and the v3 archive is the data release used here, but this audit has not verified the paper's exact detector list, split protocol, thresholding, or output values from its full text. The current PCA/RobustZ results therefore are not comparisons against those published results.

The archive itself contains one IPAL-compatible `state.jsonl.gz` record per process-table row. Each record includes the measurements under `state`, the activity label under `malicious`/`attack_active`, and missing-variable names under `missing_state_variables`. In v3 the state logs now contain usable values, while 7,569 measurement cells remain missing in the tabular release. These facts are from the archive's own README and change log; the extraction and checksums are recorded in `DATA4CYBER_SCHEMA_AUDIT.md`.

## What the public IPAL implementation supports

The upstream `ipal_ids_framework` documents two input modalities: packet/message records and process-state records. Its current detector catalogue includes several state methods (including GeCo, PASAD, and sequence models) as well as methods that operate on messages or either modality. Its Random Forest, SVM, and BLSTM implementations consume the `malicious` annotation during training. They are supervised classifiers and cannot be placed in the study's normal-only training condition without changing the learning problem or introducing attack-label supervision from other scenarios.

GeCo is the most directly compatible published state-based reference for the current normal-only protocol. The pinned upstream source is [IPAL `geco.py` at commit `0d71866b777deb13bd2ee038d5058c2425ec4568`](https://github.com/ipal-ids/ipal_ids_framework/blob/0d71866b777deb13bd2ee038d5058c2425ec4568/ids/geco/geco.py); its framework overview is the upstream [README](https://github.com/ipal-ids/ipal_ids_framework/tree/0d71866b777deb13bd2ee038d5058c2425ec4568). GeCo fits cross-sensor equations on state streams, then uses prediction residuals and CUSUM thresholds to emit alerts. It is not directly plug-and-play on Data4Cyber v3:

1. The state records omit unavailable measurements from `state`; upstream GeCo indexes every fitted sensor in every record and does not impute missing values. A training-only imputer is required, or the candidate sensor set must be fixed using training data alone.
2. The upstream training path requests Python's `fork` multiprocessing context. That start method is unavailable in the current Windows runtime. The command-line interface loads the GeCo configuration, but its fitting path needs a serial adapter or a compatible Linux runtime.
3. With 145 input variables, the upstream default `max_formel_length=3` enumerates 147,385,540 target/equation/input-set fits. Setting the maximum length to one reduces this to 42,340 fits. This is a material compute/algorithm setting change and must be disclosed; it is not an exact reproduction of the default model.
4. Threshold transfer must match the present family-held-out estimand. GeCo's default CUSUM threshold is fit internally. Any adjustment to a nominal 1% or 5% normal false-positive operating point must use only the separate, normal-only calibration data from the outer fold, never the held-out attack or target annotations.
5. State streams must be processed in chronological order, with temporal state reset at scenario boundaries. Remove activity/phase labels, timestamps, and clock-like telemetry from the detector inputs before writing IPAL records. No detector or explanation may see the attack-phase files, event log, endpoint map, or response-target annotations.

## Implemented constrained reference and results

Do not label either a conventional Isolation Forest run or a modified GeCo run as a reproduction of the paper's twelve-baseline table. The shared Data4Cyber runner now includes a **normal-only Isolation Forest detector-family reference** with the same family-held-out folds, threshold protocol, and grouped first-alarm ablation audit as PCA/RobustZ. It is a standard added comparator, not one of the paper's reproduced IPAL systems. We also implemented an **adapted GeCo-AR1 reference** in `src/xai_cps/geco_reference.py` and `scripts/run_data4cyber_geco_reference.py`, with the following protocol:

- pin the upstream IPAL source revision above and record all dependency versions;
- use only IPAL state inputs and training-normal rows in each existing six-family outer fold, grouping S1 with S1-alt;
- fit median imputation using training-normal rows only, drop all-missing training columns, and exclude labels, time fields, and explicit real-time clock features;
- restrict the upstream equation search to each channel's own one-step linear SUM equation (equivalent to the upstream `max_formel_length=0` case); this is a univariate persistence-residual baseline and omits GeCo's cross-sensor search;
- choose the alert threshold factor on the fold's normal-only calibration partition for each predeclared FPR target;
- reset model stream state between each held-out scenario, process timestamp order, and score point metrics, event recall, target evidence tiers, latency, and memory using the existing evaluation code;
- identify this as a constrained GeCo-AR1 benchmark, not full GeCo. Report it separately from the published IPAL results and alongside PCA/RobustZ until the companion paper's full experimental protocol and dataset version are independently verified.

The full six-family GeCo run is saved at `results/summaries/data4cyber_geco_ar1_reference.json`. It used 145 features and completed all 14 scenario/FPR records. Equal-weight family-mean average precision was 0.620 for GeCo-AR1, versus 0.728 for PCA residual and 0.681 for RobustZ. At nominal 1% normal FPR, GeCo-AR1 mean family F1/recall/test benign FPR were 0.121/0.087/0.014; at nominal 5%, they were 0.174/0.119/0.074. Paired family-level comparisons across six folds did not establish superiority over PCA or RobustZ after Holm correction over 14 comparisons (all adjusted $p\geq0.4375$). GeCo-AR1 detected five of eight eligible target events at nominal 1% and all eight at nominal 5%. Its top-five localization at 5% was 8/8, but the documented-device-group tier alone has a 0.979 random top-five intersection probability; this is descriptive and does not validate explanations. The exact-register tier has one event only.

The 500-tree Isolation Forest run is saved with PCA/RobustZ in `results/summaries/data4cyber_loco_baselines.json`. It reached family-mean AP 0.663, below PCA (0.728) and slightly below RobustZ (0.681); no paired AP contrast remained significant after Holm correction across 18 detector/metric/threshold comparisons. Its 1%-threshold group-ablation Hit@5 was 5/8 against random-ranking expectation 0.645; at 5%, it was 6/8 against 0.714. No native attribution is claimed. These outputs do not close the compatibility gap with the twelve published IPAL systems.

This adapter was exercised through all six folds under the leakage controls above. The package tests for the new scorer pass. The experiment supports a limited process-detector comparison; it does not reproduce or supersede the paper's twelve IPAL detector results.

This experiment can strengthen the process-detector comparison, but it cannot resolve the small number of target-eligible Data4Cyber events or certify generalization to other plants. The current HAI results remain the larger exact-tag anchor; Data4Cyber remains a distinct, limited replication testbed.

## Evidence sources

- Data4Cyber v3 primary archive record: <https://zenodo.org/records/22953576>.
- Data4Cyber release files and checksum-verified local audit: `DATA4CYBER_SCHEMA_AUDIT.md`.
- IEEE CSR 2026 conference program: <https://www.ieee-csr.org/2026-conference-program/>.
- IPAL detector catalogue and state/message modality description: upstream repository README linked above.
- Upstream GeCo algorithm and configuration: pinned source file linked above.

