# Data4Cyber first-alarm explanation-to-target study specification

**Version:** 0.1, 31 October 2026
**Status:** preregistration-style design draft informed by a pilot run. The current sample and baseline set do not yet support confirmatory claims.

## 1. Scope and novelty boundary

Study whether an explanation emitted at a detector's first alarm agrees with independently documented attack-target evidence, and whether that agreement can be estimated at different evidence resolutions. The contribution is an evaluation protocol and audit dataset, not a new IDS or explainer. It does not claim causal attribution, physical feasibility, attacker reachability, operator benefit, or a deployment-ready assurance policy.

The nearest published/posted overlaps to address directly are NDSS attribution evaluation on ICS anomaly detectors and GRIDGUARD-AI's Data4Cyber scenario-held-out explainable assurance. The primary GRIDGUARD-AI text has been inspected: it reports only a single reconstructed anonymous-feature TreeSHAP case and no per-event target-to-register localization, but it occupies the same benchmark and broad assurance framing. The paper must explain that alarm-time target validity with audited evidence tiers is a distinct outcome from general attribution quality or risk-escalation policy, and demonstrate value empirically beyond both.

## 2. Research questions and falsifiable hypotheses

- **RQ1 — Detection transfer:** How do normal-only detector thresholds and detection quality transfer when an attack family is held out? Report point metrics, event recall, detection delay, and realized false-positive rate separately.
- **RQ2 — Target validity:** At the first alarm in each eligible attack event, how often do explanation rankings include the independently documented target at each evidence tier: exact register-to-process tag, documented device group, or coarse asset/phase?
- **RQ3 — Explanation properties:** Do group-ablation explanations identify evidence-backed targets more often than matched native detector contributions or a within-fold random-group reference, and how do fidelity, local stability, runtime, and missing/ambiguous evidence affect that comparison?

**H1 (baseline target signal):** across attack-family units with eligible targets, the detector-conditioned explanation's unconditional Hit@K is greater than a matched within-fold random-group baseline. Detector misses count as zero; report evidence tiers separately. The test is not meaningful until the number of independent family units is large enough for a nontrivial exact test.

**H2 (method comparison):** grouped component ablation and native detector contributions have different target Hit@K. This is two-sided: the current pilot does not justify a directional superiority claim.

**H3 (transfer):** a threshold set on benign training/validation scenarios will not necessarily achieve its nominal false-positive rate on a held-out attack-family scenario. Test this as calibration transfer; do not use attack labels to select thresholds.

**H4 (evidence resolution):** observed Hit@K depends on evidence granularity and target coverage. Test/report tier-specific outcomes; do not treat the tier effect as an explainer effect or pool targets into one score.

## 3. Threat model and observability

The evaluator is a passive defender who receives 1 Hz process telemetry and, in a later extension only, timestamp-aligned protocol observations. The released attacks manipulate documented Modbus registers/measurements, conduct ARP/MitM operations, or publish MQTT price-signal events. The attacker may cause the recorded labeled observations; no online adaptive attacker, evasion budget, insider privilege model, or cryptographic compromise is simulated by this study. The detector and explainer do not receive attack labels, phase labels, event logs, endpoint maps, scenario names, or target annotations during fitting, calibration, scoring, or explanation generation. Those sources are used only by the offline evaluation layer to form evidence sets.

A detector miss is an operational failure and remains a target-localization miss in unconditional metrics. The analysis distinguishes exact target evidence from documented device groups and coarse asset/phase evidence. Ambiguous, absent, or only post-hoc semantic mappings are marked unavailable or indirect, not silently promoted to exact truth. In particular, S6 packet capture does not contain MQTT packets; S1/S2 control registers are not direct telemetry-tag labels; S4/S5 require documented device context to disambiguate endpoint candidates.

## 4. System architecture

```text
Timestamped process rows
        │
        ├── label/time separation and train-normal-only preprocessing
        │
        ├── unsupervised detector ──► score s(t) ──► normal-only threshold q
        │                                          │
        │                                          └── first alarm τ(e)
        │                                                  │
        └──────────────────────────────────────────────────┴──► group explainer
                                                                    │
                                                                    ▼
                                                        ranked channel groups π(e)

Offline evaluation only: phase log + event log + endpoint/register maps
       └──► evidence ledger E(e, tier) ──► tier-specific Hit@K / coverage / abstention
```

The evaluator is kept outside the detector path. Target annotations must not participate in feature selection, preprocessing, calibration, explanation, or threshold selection.

## 5. Formal definitions

Let event `e` occupy annotated attack interval `[a_e,b_e)` with input history `X_{\le t}` and binary event label `y_t`. A detector produces anomaly score `s_\theta(X_{\le t})`. For a fold-specific threshold `q_\alpha` calibrated on normal-only validation scores at nominal false-positive level `\alpha`, define its first alarm

\[
\tau_e = \inf\{t\in[a_e,b_e): s_\theta(X_{\le t})>q_\alpha\}.
\]

If the set is empty, `e` is a detector miss and has no alarm explanation. An explainer returns group scores `\phi_g(X_{\le\tau_e},s_\theta)` for fixed, label-independent semantic groups `g\in\mathcal G`; sort them into ranking `\pi_e`. The evidence ledger stores a set of target groups `E_{e,r}\subseteq\mathcal G` for evidence tier `r`. Only targets supported by independent source records are included. If no valid target exists at tier `r`, the event is excluded from that tier denominator and its exclusion reason is counted.

For eligible events `\mathcal A_r` and `k\in\{1,3,5\}`, unconditional target hit is

\[
\operatorname{Hit@k}_{r}=\frac{1}{|\mathcal A_r|}\sum_{e\in\mathcal A_r}
\mathbf 1[\tau_e\text{ exists}]\,\mathbf 1[\operatorname{TopK}(\pi_e)\cap E_{e,r}\ne\varnothing].
\]

Also report conditional Hit@K over detected events, but never in place of the unconditional value. Report the eligible count, detected count, evidence coverage, and miss count beside each estimate. Evidence tiers are separate estimands; there is no pooled “root-cause accuracy.”

A grouped-deletion score change at first alarm is

\[
\Delta_{e,k}=s_\theta(X_{\le\tau_e})-s_\theta(\widetilde X_{\le\tau_e}^{\operatorname{TopK}(\pi_e)}),
\]

where `\widetilde X` replaces selected groups using a declared, train-normal reference. Compare paired `\Delta_{e,k}` to identically sized random-group deletions. This is local detector-score faithfulness under a specified reference intervention; it is not causal effect, physical feasibility, or proof of target correctness.

Local rank stability uses a predeclared perturbation distribution fit only to training-normal scales. Report perturbation semantics and top-K Jaccard; iid Gaussian noise is a narrow sensor-noise sensitivity test, not adaptive robustness or process-validity evidence. Detector latency and explainer latency are measured separately.

## 6. Leakage-free validation and statistical unit

- Outer fold: hold out one primary attack family. Keep S1 and S1-alt together. S0 contributes only benign context.
- Fit feature imputation, scaling, detector parameters, and calibration using training-normal observations from non-held-out scenarios. Reserve the final chronological segment of training-normal data for calibration; do not expose the held-out family or its labels to model selection.
- Exclude timestamps, attack activity/phase fields, file/scenario identifiers, and clock/realtime channels from features. Reset temporal differences across gaps.
- Use family/scenario as the unit for paired detector comparisons and uncertainty; row and phase counts are not independent replicates. With six families, exact paired tests have a minimum two-sided p-value of 0.03125 and low power. Report effects and intervals; Holm-adjust across a predeclared comparison family.
- Select thresholds from normal validation only. Report the realized held-out benign FPR; nominal calibration is not assumed to transfer.
- Freeze the evidence-tier annotation table and provenance before explaining test events. Keep unresolved targets as abstentions, not negatives or exact labels.

## 7. Baselines and required evaluation matrix

1. Detector baselines: RobustZ and PCA residual as reproducible pilot references; reproduce an appropriate subset of the Data4Cyber paper's 12 IPAL detector implementations once the exact input representation, code, and operating-point protocol are established. Network-only baselines must not be compared as if they observe the same modality as process-only detectors.
2. Explanation baselines: native detector contribution, grouped component deletion, matched random groups, and at least one matched temporal attribution method where compatible with the detector. Do not call metric adaptation to AR-Pro/JoCE a reproduction of their methods.
3. Detection: average precision, point precision/recall/F1 at normal-calibrated thresholds, event recall, first-detection delay, and per-scenario realized normal FPR. No point adjustment.
4. Explanation: tier-specific unconditional and conditional Hit@1/3/5, target coverage, phase alignment where independently supported, grouped score-deletion fidelity, stability, p50/p95 explanation latency, and peak memory where measurable.
5. Robustness: only add process-valid or adaptive attack perturbations when their threat model and valid perturbation set are independently specified. Keep iid measurement noise results labeled as local sensitivity.
6. Ablations: no temporal difference, no group aggregation, no normal-only imputation (diagnostic only), detector-native versus group ablation, and calibration regime. Predeclare comparisons and multiplicity handling.

## 8. Pilot status and decision gate

The current Data4Cyber pilot has six grouped family folds, two unsupervised process-only detectors, and first-alarm explanations. Mean family AP was 0.727 for PCA residual and 0.674 for RobustZ; the paired mean advantage was not significant after Holm correction. Nominal 5% FPR thresholds transferred poorly in some held-out files (maximum observed FPR 32.3%). Exact S3 first-alarm evidence has one eligible event; device-group and coarse evidence are also sparse. These values are exploratory and do not establish that explanations beat random baselines or native contributions in a population sense.

**Go/no-go for a target-validity paper:** (i) verify GRIDGUARD-AI code and supplementary package, (ii) establish a matched empirical comparison and precise relation/difference to its assurance outputs, (iii) reproduce matched released detector baselines, (iv) acquire more independently annotated events or another compatible testbed, (v) run seeded/uncertainty analyses at the scenario level, and (vi) compile and visually inspect the final manuscript. If target counts remain at the current level, reframe as a dataset audit/benchmark note rather than claiming a high-impact general XAI-IDS contribution.
