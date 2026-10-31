# Data4Cyber v3 preliminary detector and explanation audit

**Status:** exploratory first pass; not a final benchmark or a Q1-ready result.
**Data:** Data4Cyber v3, 14,354 rows, six attack-family folds (S1 and S1-alt grouped).
**Run:** `python scripts/run_data4cyber_baseline.py --data-dir data/raw/data4cyber/full_v3/data4cyber_dataset --compact-output --output results/summaries/data4cyber_loco_baselines.json`

## Protocol actually run

- Outer evaluation holds out one attack family at a time. When S1 is held out, both S1 runs are excluded from fitting and calibration. S0 benign rows are retained as normal-only context.
- Within each non-held-out normal scenario, the first 80% of benign rows by timestamp fit the model and training-only median imputer; the final 20% calibrates thresholds. Normal-difference features reset at timestamp gaps.
- Detector features exclude timestamps, `attack_active`, `attack_phase`, `attack_phase_all`, and `*.realtime` columns. Target annotations and phase labels are post-hoc only.
- The matched unsupervised baselines are RobustZ, PCA residual (95% variance, up to 48 components), and a 500-tree Isolation Forest on robust-scaled state and first-difference features (`max_samples=512`, seed 42). Forest fitting is normal-only; its threshold is selected on held-out normal calibration rows. It has no native per-channel attribution, so only grouped black-box score ablation is reported for its first-alarm ranking; `native_*` attribution fields are intentionally null. A separate constrained GeCo-AR1 reference was added later; its deviations from full GeCo are documented in [`DATA4CYBER_IPAL_BASELINE_COMPATIBILITY.md`](DATA4CYBER_IPAL_BASELINE_COMPATIBILITY.md). All models use a model-specific calibration threshold at nominal normal FPR 1% and 5%.
- Process-group explanations compare native detector contributions averaged within each semantic prefix against component-wise deletion ablation. The target map was frozen in `data/annotations/data4cyber_response_targets.csv` before examining these explanation results. Detector misses count as target misses; reconnaissance-only events are excluded from target localization and retained for phase/detection reporting.
- The runner now verifies all 16 attack/benign split CSVs consumed against the embedded SHA-256 manifest. It reports an analytic random-ranking reference: for a detector hit, the probability a uniformly random top-k group set intersects the target set, given the available group universe; misses have reference hit probability zero. This is a descriptive chance baseline, not an independent inferential replicate.
- Exact paired sign-flip tests use six family-level paired differences; Holm correction covers the 18 pairwise detector/metric/threshold contrasts among PCA, RobustZ, and Isolation Forest. These tests are low-powered and exploratory.

## Detection results

Equal-weight family mean average precision was 0.728 for PCA, 0.681 for RobustZ, and 0.663 for Isolation Forest. The Isolation Forest's paired AP difference against PCA was -0.0646 (exact two-sided sign-flip $p=0.03125$, Holm-adjusted $p=0.5625$ across 18 comparisons); against RobustZ it was -0.0180 ($p=0.28125$, adjusted $p=1.0$). The Forest does not improve on the existing process baselines, and the family-level evidence does not establish a statistically significant difference after correction.

| Held-out family | PCA AP | RobustZ AP | Isolation Forest AP | PCA F1 at nominal 1% FPR | RobustZ F1 at nominal 1% FPR | IF F1 at nominal 1% FPR |
|---|---:|---:|---:|---:|---:|---:|
| S1 (S1 and S1-alt grouped) | 0.721 | 0.634 | 0.654 | 0.019 | 0.019 | 0.020 |
| S2 | 0.603 | 0.558 | 0.529 | 0.000 | 0.006 | 0.009 |
| S3 | 0.554 | 0.485 | 0.473 | 0.045 | 0.000 | 0.000 |
| S4 | 0.805 | 0.764 | 0.764 | 0.020 | 0.020 | 0.005 |
| S5 | 0.835 | 0.820 | 0.819 | 0.032 | 0.015 | 0.012 |
| S6 | 0.848 | 0.825 | 0.740 | 0.021 | 0.006 | 0.006 |

With a nominal 1% normal-only threshold, held-out-file FPR ranged from 0.0% to 1.5% for PCA, 3.6% to 5.9% for RobustZ, and 0.6% to 1.3% for Isolation Forest. At nominal 5%, the ranges were 2.7%–32.3% for PCA, 3.6%–31.1% for RobustZ, and 3.3%–7.8% for Isolation Forest. This demonstrates substantial normal-regime transfer error for some held-out families. At nominal 1%, equal-weight family-mean point recall was low (about 1.2% PCA, 0.6% RobustZ, and 0.4% Isolation Forest), despite nonzero event recall in several files. Point and event metrics therefore tell different stories; no point-adjustment was used.

At nominal 5%, equal-weight family-mean F1/recall/realized benign FPR were 0.0698/0.0385/0.0436 for Isolation Forest, 0.2494/0.1939/0.1040 for PCA, and 0.1477/0.1279/0.0863 for RobustZ. Higher attack coverage for PCA came with substantial threshold-transfer error; all outcomes remain exploratory across six families from one testbed.

## Isolation Forest first-alarm explanation reference

At nominal 1%, the Forest detected seven of eight eligible target events; at nominal 5%, it detected all eight. Unconditional grouped-ablation Hit@5 was 5/8 and 6/8, respectively, compared with event-matched random-ranking expectations of 0.645 and 0.714. At 1%, Hit@1/3/5 was 0.125/0.500/0.625; at 5% it was 0.125/0.500/0.750. The exact-tag tier contains one event and the coarse device-group tier has a high random top-five intersection probability, so these values provide no persuasive target-localization evidence. Isolation Forest supplies no native feature attribution; the reported ranks are solely grouped score-deletion effects. Across the six family folds, mean fit time was 2.50 s, sampled single-row score p95 was 70.4 ms, and mean event explanation time was about 0.6 s (p95 0.69 s) on the recorded Windows/Intel host. Fitted tree arrays occupied 4.53–4.65 MB, excluding Python/runtime overhead. These timings and sizes are host-specific, exclude rendering and alert handling, and do not establish edge deployment. The compact machine-readable summary retains detector metrics and all target-eligible first-alarm rankings; omitting `--compact-output` recreates the full event trace.

## Constrained GeCo-AR1 reference

The adapter applies the one-step, self-lag linear SUM subset of GeCo, with training-normal-only median imputation and separate normal-only calibration. It uses no attack labels or cross-sensor equations. The 145-feature six-family run produced equal-weight mean AP 0.620, below PCA (0.728) and RobustZ (0.681). At nominal 1% FPR, family-mean F1/recall/test benign FPR were 0.121/0.087/0.014; at nominal 5%, they were 0.174/0.119/0.074. Six-family paired comparisons with PCA and RobustZ did not establish a performance difference after Holm correction across 14 comparisons (all adjusted $p\geq0.4375$).

For first-alarm target localization, five of eight eligible events were detected at nominal 1%; all eight were detected at nominal 5%. Unconditional Hit@5 at the latter operating point was 8/8, but three of eight events are in the documented-device-group tier, whose target sets alone imply a 0.979 random top-five hit probability. The exact-register tier has only one event. These results are exploratory and do not validate full GeCo, explainability superiority, or independent generalization. Machine-readable results are in `results/summaries/data4cyber_geco_ar1_reference.json`.

The current host-only timing pass measured per-transition scoring p95 between 0.008 and 0.042 ms across held-out scenarios, with batch-amortized scoring of 8.8–42.8 microseconds per row and 4,640 bytes of fitted NumPy parameter arrays. These are measurements of the constrained reference implementation on this research host; they exclude explanation rendering and alert handling, and do not establish edge-device or deployment feasibility.

## First-alarm localization and explanation properties

The localization sample is tiny: S3 supplies one exact register-to-tag impact event; S4 supplies one device-group event; S5 supplies two; S1/S1-alt together supply two asset-response events; S2 and S6 each supply one mapped-response/topic event. These event counts are not independent replicates, and evidence tiers are not pooled.

At nominal 1% FPR:

- **Exact S3 evidence (n=1):** at nominal 1%, PCA detected the impact event; native grouped contribution ranked the BSS meter within top-5, while group deletion did not. RobustZ missed. At nominal 5%, both detectors detected; PCA group deletion still missed the target in top-5, while RobustZ ranked it in top-3. One event cannot establish a method advantage.
- **S4 documented meter groups (n=1):** both detectors and both explanation methods hit a documented group at top-1 at nominal 1%. Three of nine available groups are accepted as targets, so a random top-5 ranking has a 0.952 hit probability; top-5 agreement is therefore weak evidence here.
- **S5 documented meter groups (n=2):** at nominal 1%, each detector missed one eligible event and unconditional top-1 target hit was 0.5. At nominal 5%, PCA group deletion hit both at top-1. The target set contains four documented groups among nine available, making random top-5 hit probability close to one; top-5 is uninformative for this tier.
- **Indirect response groups (S1/S2):** localization refers to telemetry associated with the controlled component, not the manipulated control register itself. At nominal 1%, native contributions and group deletion differ by family/model; misses count as zero. These are not exact target-attribution labels.
- **S6 topic-to-process-name mapping (n=1):** evidence is a post-hoc semantic link, not an endpoint-map or packet-level join. At nominal 1%, PCA detected and ranked the topic-related process group at top-1; RobustZ missed. At nominal 5%, both detected and ranked it at top-1. This is one event, not evidence of robust generalization.

Across detected first alarms, top-five explanation rankings were usually unchanged under 1% and 5% independent Gaussian perturbations scaled by training-normal channel scale (five repeats per level). This is a narrow local measurement-noise check, not adaptive robustness or process-valid perturbation. Component deletion p95 was approximately 0.35–0.85 ms per explanation; single-row detector score p95 averaged about 0.024 ms on the recorded Windows/Intel runtime. These are local Python timings on one machine, not end-to-end edge or control-loop feasibility measurements. The result JSON records fit/scoring time and runtime metadata.

## Interpretation and next experiments

The dataset supports an evidence-tiered explanation audit, but not uniform root-target ground truth. Current detector transfer is imperfect, and target counts are too small for strong explanation claims. The Forest reference is a conventional normal-only comparator, not an IPAL detector reproduction. The constrained GeCo-AR1 reference is an adapted subset, not a result against the paper's 12 IPAL baselines; those published results were not reproduced. The exact companion-paper protocol has not been verified from full text. See [`DATA4CYBER_IPAL_BASELINE_COMPATIBILITY.md`](DATA4CYBER_IPAL_BASELINE_COMPATIBILITY.md).

Before manuscript claims, obtain and compare the full GRIDGUARD-AI preprint and supplement, resolve S4/S5 endpoint ambiguity, conduct seeded repeats and scenario-level uncertainty analysis, measure p50/p95 overhead on a disclosed edge-class device, and compare against the exact Data4Cyber companion-paper protocol if its full text and versioned artifacts become available. The random-ranking reference shows why top-K must be interpreted relative to target-set size and available groups; it does not rescue the tiny, dependent event sample. Expand testing with a second testbed/dataset if comparable ground-truth granularity is available. No causal, process-safety, or physical-transition claim is supported by this run.

