# Evidence-aware first-alarm explanation evaluation

**Status:** working study design; novelty remains provisional. The HAI and Data4Cyber runs are separate datasets, not pooled replications. This design supersedes a Data4Cyber-only framing as the recommended paper scope because the exact-target sample in Data4Cyber is too small for the primary localization estimand.

## Research questions

- **RQ1:** At a detector's first alarm in a documented CPS attack, how often do feature-group explanations rank an independently documented target within top-*k*, counting detector misses as localization misses?
- **RQ2:** How do target agreement and its evaluable coverage change when ground truth is exact-tag, indirect component-response, documented device-group, or coarse phase evidence?
- **RQ3:** Do explanations that agree with documented targets also faithfully summarize the detector score, and are their rankings stable under a declared measurement-noise perturbation?
- **RQ4:** What are the explanation latency and computational costs at the alarm point, separately from detector inference?

## Hypotheses and confirmatory boundary

- **H1 (primary, exploratory until powered):** grouped explanations outperform a group-size-matched random-ranking reference on unconditional exact-target Hit@K in HAI, with a paired attack-event difference and family/file-cluster uncertainty reported.
- **H2 (descriptive across evidence tiers):** Hit@K estimates and their chance baselines depend on target-set size and evidence resolution; pooled “root-cause accuracy” is invalid.
- **H3 (secondary):** target agreement is not implied by local score faithfulness; report both quantities independently and test their association without causal interpretation.
- **H4 (operational):** p50/p95 explanation cost is measurable independently of detector cost on the disclosed hardware; no real-time claim follows unless end-to-end load and hardware are evaluated.

The supplied HAI run summaries and Data4Cyber pilot do not justify declaring H1 supported. The HAI mapping has 29 complete exact-tag intervals among 50; repeated seeds reuse those events. Data4Cyber has a handful of eligible target events across six primary families. Data4Cyber tier comparisons therefore remain descriptive. Pre-register a minimum number of independent attack families/events before upgrading H1 to confirmatory.

## Threat model and claim boundary

The evaluated adversary can compromise an OT-reachable component and manipulate sensor reports, actuator/setpoint values, or traffic supporting a documented attack scenario, with the aim of evading or triggering a process anomaly detector or producing a harmful process response. The study does not assume control over the detector, historian, evaluation labels, or explanation code. The datasets provide bounded scripted attack episodes, not adaptive red-team trials. The work evaluates explanations of a detector's score against independent attack records; it does not infer attacker intent, prove causal root cause, establish process safety, prescribe remediation, or claim the documented target is the only affected component.

## Proposed evaluation architecture

```text
Dataset-specific time series ─► leakage-safe preprocessing ─► detector score ─► normal-only threshold
                                                                           │
                                                                           └─► first alarm per event
                                                                                └─► native / grouped / random explanations

Frozen independent evidence ledger ─► exact/indirect/coarse tier ─► unconditional Hit@K + coverage/abstention
                                                                  └─► separate score-faithfulness, stability, latency
```

Annotations, phase labels, attack IDs, scenario names, and target maps remain outside detector fitting, feature selection, threshold calibration, and explanation generation. HAI and Data4Cyber are processed and reported separately. BATADAL remains a descriptive external audit because its current component targets and interval timing are approximate.

## Mathematical estimands

For event `e` with attack interval `[a_e,b_e)`, detector score `s_theta(x_{<=t})`, and threshold `q_alpha` selected only from normal validation data, define the first alarm

\[
\tau_e=\inf\{t\in[a_e,b_e):s_\theta(x_{\le t})>q_\alpha\}.
\]

Let `pi_e` be the descending ranking over fixed semantic groups at `tau_e`; let `E_{e,r}` be the independently documented target-group set at evidence tier `r`. For eligible events `A_r`, the primary unconditional estimand is

\[
\mathrm{Hit@k}_r=\frac{1}{|A_r|}\sum_{e\in A_r}\mathbf 1[\tau_e\text{ exists}]\,\mathbf 1[\mathrm{TopK}(\pi_e)\cap E_{e,r}\ne\varnothing].
\]

Report detected-event-conditional Hit@K only beside the unconditional result, along with eligible count, misses, evidence coverage, and abstentions. A matched chance reference samples the same number of groups from the same eligible group universe; for fixed-size target set `m` among `G` groups and `k<=G`, its hit probability is `1 - choose(G-m,k)/choose(G,k)`. If no alarm occurs, unconditional hit is zero. Keep score deletion

\[
\Delta_{e,k}=s_\theta(x_{\le\tau_e})-s_\theta(\tilde{x}^{\mathrm{TopK}}_{\le\tau_e})
\]

separate from target agreement. This measures score sensitivity under the stated reference intervention, not causality or physical validity.

## Experimental protocol

1. Freeze and version each dataset-specific target ledger before inspecting explanation ranks. HAI exact matches require the documented mapping checks; incomplete target lists are excluded from exact-tag metrics and counted in coverage. Data4Cyber retains its register-to-tag, indirect response, device-group, and coarse/post-hoc semantic tiers separately.
2. Split by whole attack event and file/scenario/family; never randomly split overlapping windows. Fit imputation, scaling, detector parameters, and thresholds on training-normal data only. Reset temporal differences across gaps and exclude labels, timestamps, scenario identifiers, and clock channels from features.
3. Retain current statistical detector anchors (PCA residual and RobustZ) and native detector ranking, grouped deletion, matched random ranking, and score-faithfulness controls. Add a matched learned temporal baseline and attribution method only after version, representation, window and threshold alignment are verified. Do not call an adapted metric a reproduction of JoCE, AR-Pro, or GRIDGUARD-AI.
4. Report AP, point precision/recall/F1 without point adjustment, event recall, first-alarm delay, calibration/held-out normal FPR, target Hit@1/3/5, coverage, score deletion, stability, and p50/p95 detector and explanation latency. Report each dataset and evidence tier separately.
5. Treat attack event as the localization unit and scenario/family/file as the independent cluster where possible. Average repeated seeds within event before paired inference. Use paired cluster bootstrap intervals and exact paired tests where the number of independent clusters permits; apply Holm correction to predeclared comparison families. With six Data4Cyber families, emphasize effect sizes and uncertainty, not significance claims.
6. Ablate semantic grouping, temporal differences, explainer type, threshold calibration, and evidence tier. Include group-size-aware random rankings because coarse target sets can make top-5 hit nearly certain by chance.
7. Measure on a disclosed edge-class target before making a real-time claim. Current Windows/Intel Python timing is local diagnostic profiling only.

## Current evidence and reviewer gates

The current HAI mapping supports 29 exact-target intervals; the five-seed explanation evaluations reuse these intervals, and three tag-name mismatches remain an explicit caveat. Data4Cyber provides a separately sourced dataset and six grouped scenario-family folds, but its first-alarm target sample is too small for broad inferential claims. The updated normal-only comparison found family AP 0.728 for PCA, 0.681 for RobustZ, and 0.663 for Isolation Forest. No PCA/RobustZ/Forest detector contrast remained significant after Holm correction across 18 comparisons (the PCA-versus-RobustZ AP raw sign-flip p=0.03125, adjusted p=0.5625). At a nominal 5% false-positive target, PCA held-out-file FPR reached 32.3%; the Forest maximum was 7.8%. The sparse target counts and transfer errors argue against claiming a validated XAI method or operational system.

Before a submission claim: verify the GRIDGUARD-AI code and supplement and state the overlap; reproduce matched released baselines with modality-aligned inputs; complete target-ledger review; add an independent testbed with enough exact target events; run cluster-aware uncertainty; measure edge-class latency; and compile/render the manuscript. If these gates cannot be met, the honest output is a scoped empirical audit or dataset note rather than a Q1-ready general method paper.

