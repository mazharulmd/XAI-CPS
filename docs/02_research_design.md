# Research design

## 1. Candidate directions, ranked

Scores are 1–5 (5 = best) and reflect our judgement after the gap analysis in `01_literature_review_and_gaps.md`.

| Rank | Direction | Gaps | Originality | Practical importance | Feasibility here | Q1 potential | Total |
|---|---|---|---|---|---|---|---|
| **1** | **Deadline-aware, control-loop-structured attribution for ICS intrusion localization** (*explain correctly before the process safety time expires*) | G1, G2, G3, G4 | 4 | 5 | 5 | 4–5 | **18–19** |
| 2 | Explanation-misdirection ("blame-shifting") attacks on ICS attribution and certified/consensus defences | G5 | 5 | 4 | 3 | 4–5 | 16–17 |
| 3 | Leakage audit and reproducibility study of XAI-IDS-CPS claims | G4 | 3 | 4 | 4 | 3 | 14 |

**Why direction 1 ranks first**
- **Originality.** It is the only direction that targets the failure Fung et al. documented: attributions are unreliable at the moment of detection. It also adds a dimension no paper in the corpus measures: correct localization *by a deadline* that the physical process sets.
- **Feasibility.** It is fully feasible on public data. HAI 21.03 documents 50 attacks with target loops. Physics-grounded deadlines come from an open closed-loop simulator that we release.
- **Absorbs the other two.** It subsumes the strongest elements of directions 2 and 3: a decoy/blame-shifting robustness experiment, and a leakage experiment. Each becomes one evaluation axis instead of a separate paper.

**Why direction 2 ranks second.** It is very original, but credible results need closed-loop physical consistency of the adversarial manipulations. Recorded datasets cannot provide that, so it would rest almost entirely on simulation. We keep a reduced form as RQ5.

**Why direction 3 ranks third.** It has high practical value, but reviewers often view a pure audit as incremental. We keep it as a secondary result (H6).

## 2. Selected direction — positioning

| | Fung et al. NDSS'24 | ShaTS FGCS'26 | Oinonen EPSR'26 | XAI-CPS ComNet'26 | **This work** |
|---|---|---|---|---|---|
| Attribution unit | tag | tag / 6 stages | pixel | tag | **control loop (+ tag refinement)** |
| Ground-truth localization | yes (SWaT, WADI, TEP) | case-based | no | expert agreement | **yes, 50 HAI attacks + simulator** |
| Evaluated at operational timing | yes (finds failure) | no | n/a | no | **yes, as the primary setting** |
| Uses evidence across alarm windows | no | no | no | no | **yes (temporal evidence accumulation)** |
| Latency | not measured | per window vs 1 s (GPU) | per explanation (GPU) | 12 ms (A100) | **CPU, single-thread; evaluation counts** |
| Physical deadline | no | no | no | no | **process-safety-time-conditioned metrics** |
| Leakage-free protocol stated | yes (normal-only training) | yes (padding) | — | 5-fold CV | **chronological + val-only thresholds** |
| Statistical tests across attacks | ANOVA/Pearson on properties | no | no | mean ± std | **Friedman + Wilcoxon/Holm + bootstrap CI** |

## 3. Research questions and hypotheses

- **RQ1 (structure).** Does attributing to *control loops* by exact Shapley values improve localization of the attacked loop at operational timing, compared with tag-level attribution aggregated post hoc?
  - **H1:** LX (loop-exact Shapley) achieves higher HR@1 and lower normalised rank than RE, SM, IG, KernelSHAP, LIME, ShaTS-sensor and the Fung ensemble.
  - **H4:** The gain is due to *control* structure, not grouping per se. LX beats exact Shapley over data-driven correlation clusters with the same number of groups.
- **RQ2 (time).** Does accumulating attribution evidence over successive alarm windows improve accuracy and temporal stability?
  - **H2:** TEA increases HR@k at a fixed delivery time and increases top-k Jaccard stability between consecutive explanations.
- **RQ3 (deadline).** Which explainers deliver a correct localization before a deadline once compute latency is accounted for?
  - **H3:** LX+TEA has a deterministic evaluation budget (2^L detector evaluations), lower latency than sampling explainers at equal or higher accuracy, and the highest deadline-conditioned localization DCL@k(D) and deadline-met rate.
- **RQ4 (robustness).** How do localization and stability degrade under sensor noise?
  - **H5a:** LX+TEA degrades less than tag-level methods.
- **RQ5 (adversarial).** How do they degrade under a decoy manipulation of an innocent loop?
  - **H5b:** The decoy captures top-1 less often under LX+TEA than under tag-level methods.
- **RQ6 (rigour, secondary).** Does a random window split inflate detection performance relative to a chronological split?
  - **H6:** Yes, materially.

Every hypothesis is tested with paired tests over attacks (Section 7). A hypothesis is reported as *not supported* if the corrected p-value ≥ 0.05 or the effect has the wrong sign.

## 4. Threat model

- **System.** Process-level ICS data (SP, PV, CV/feedback tags) sampled at 1 Hz from a historian or SCADA feed. The IDS host and its data feed are trusted, but that feed carries whatever the controllers report.
- **Adversary.** The adversary has Level-1/2 access (controller / HMI network) and can, over an interval:
  - change a setpoint (SP), hidden from the HMI;
  - override an actuator command (CV);
  - spoof a sensor value (PV) by bias, freeze or replay, as seen by both controller and historian;
  - combine these (e.g. CV attack + PV replay to mask it) on one or two loops.

  This matches HAI's attack taxonomy.
- **Goal.** Drive the process toward an unsafe state (overflow, dry-run, over-temperature, over-pressure) or disrupt production.
- **Knowledge.** Grey-box in RQ1–4: the adversary does not know the detector or explainer. In RQ5 (decoy) the adversary knows an explainer is used and adds a small manipulation to an innocent loop to shift blame.
- **Out of scope.** Network-layer attacks without process effect, DoS of the historian, compromise of the IDS host, and poisoning of training data.
- **Defender's goal.** Name the compromised loop(s), ideally with the tags, early enough that operators can act before the process safety time is consumed.

## 5. Formulation

**Data and loops.**
- Tags x_t ∈ ℝ^d are partitioned into control loops 𝒢 = {g_1,…,g_L}. For HAI, L = 10 (`configs/hai/hai2103_loops.yaml`).

**Detector.**
- A forecaster f maps a window W_t = x_{t-w:t-1} to x̂_t.
- Residual: r_t = (x_t − x̂_t) ⊘ σ, where σ is the per-tag residual standard deviation on validation normal data.
- Score: e_t = (1/d)‖r_t‖²; s_t = EWMA_α(e_t).
- Alarm: s_t > θ, where θ is the q-quantile of s on chronologically held-out *normal* validation data. No test labels are used.

**Value function.**
- At an incident with alarm time t_a, fix a *last-known-good* reference time t_ref = t_a − Δ_ref (the last time s ≤ θ before the alarm run, minus a guard Δ_g).
- For a coalition S ⊆ 𝒢, build the segment Z_t^S: tags in loops of S take observed values over [t−w, t]; all other tags take the time-aligned values from [t_ref−w, t_ref].
- v_t(S) = e(Z_t^S) − e(Z_t^∅), so that v_t(∅) = 0 and v_t(𝒢) = e_t − e_ref.

**Loop-exact Shapley (LX).**
- φ_t(g) = Σ_{S⊆𝒢∖{g}} |S|!(L−|S|−1)!/L! · [v_t(S∪{g}) − v_t(S)].
- It is computed exactly from 2^L batched detector evaluations, a fixed and data-independent cost (for L = 10, 1,024 evaluations).
- Efficiency gives Σ_g φ_t(g) = e_t − e_ref.

**Temporal evidence accumulation (TEA).**
- Normalised positive attribution: p_t(g) = φ_t(g)₊ / Σ_h φ_t(h)₊.
- Accumulated evidence: A_T(g) = Σ_{t=t_a}^{T} ω_t p_t(g), with ω_t = log(s_t/θ)₊ + ε.
- The ranking published at T sorts loops by A_T.
- Confidence: c_T = (A_T^{(1)} − A_T^{(2)}) / Σ_g A_T(g).
- Optional stopping rule: publish once c_T ≥ γ.

**Tag refinement (H-LX).**
- Within the top-ranked loop, exact Shapley over its tags is computed with the same value function, other loops held at observed values.
- Cost: 2^{|g|} evaluations.

**Delivery and deadlines.**
- For an attack with onset t_on and explainer latency ℓ, the explanation available at t_on + D is the one computed on data up to T = t_on + D − ℓ. It exists only if t_a ≤ T.
- DCL@k(D) = (1/N) Σ_i 𝟙[t_a,i ≤ T_i ∧ 𝒢_i* ∩ top_k(A_{T_i}) ≠ ∅], where 𝒢_i* are the target loops.
- TTCL@k = inf{D : the above indicator holds for attack i}.
- Simulator only: the hazard deadline D_i^haz is the first time the *true* physical state crosses a safety limit with no operator response. An attack is *explained in time* if TTCL@k ≤ D_i^haz − t_act, where t_act is the operator action time.
- On HAI, physical limits are not available, so DCL is reported over a deadline grid D ∈ {10, 30, 60, 120, 300} s.

## 6. Experimental protocol

**Datasets**
- **HAI 21.03 (real HIL testbed).**
  - 3 normal training files and 5 test files with 50 documented attacks (25 single, 25 combined).
  - Splits: train = first 80 % of each training file, val = last 20 % (chronological). Test files are untouched until final evaluation.
- **PCL-Sim (ours, released).**
  - A closed-loop two-tank + heater + pump process with five PI loops (LC1, FC1, LC2, TC1, PC1), actuator dynamics, sensor noise, operator setpoint schedules and disturbances.
  - Attack primitives mirror HAI's: SP, CV, PV bias/freeze/replay, CV+PV mask, short pulses.
  - Each attack episode is simulated twice, observed and true physics, so the hazard deadline is known.
  - Training: normal runs from seeds 0–9. Validation: seeds 10–11. Test attacks: seeds ≥ 100.

**Detectors.** Ridge-VAR (linear), MLP, 1D-CNN and GRU forecasters, each trained on normal data with 3 seeds.

**Explainers.**
- Baselines:
  - RE (raw error; Fung's MSE ranking);
  - SM (|∂e/∂x|);
  - IG (from the reference);
  - KernelSHAP (tag players, budget B);
  - LIME (tag players, B);
  - ShaTS-sensor (feature-grouped sampled Shapley, m = 20·d);
  - Fung ensemble (RE + SM + LIME with β = 2.5 for actuators; LEMNA replaced by LIME);
  - Cluster-exact (exact Shapley over L correlation clusters).
- Tag-level methods are aggregated to loops by Σ_{j∈g}|φ_j|.

**Ablations**
1. Loop vs cluster vs stage grouping.
2. Exact vs sampled Shapley at equal budget.
3. Last-known-good reference vs training-mean vs random background.
4. TEA vs single-shot; weighting ω (uniform vs log-excess).
5. EWMA α.
6. Raw-residual aggregated per loop (RE-loop), to show what Shapley adds beyond grouping.

**Metrics**
- *Detection:* point-wise P/R/F1, event recall, detection delay, false alarms per hour on attack-free test segments.
- *Localization:* HR@1, HR@3, normalised rank (best target rank − 1)/(L − 1), at practical timing (t_a) and over time.
- *Faithfulness:* deletion AUC — restore the top-ranked loops to the reference and track the score drop.
- *Stability:* top-k Jaccard between consecutive explanations; under input noise σ_n.
- *Cost:* wall-clock latency (single thread, CPU), number of detector evaluations, peak memory.
- *Deadline:* DCL@k(D), TTCL, deadline-met rate.

## 7. Statistics and leakage control

- **Unit of analysis.** The attack. Seeds are averaged per attack before testing.
- **Omnibus test.** Friedman across methods. Ours vs each baseline: Wilcoxon signed-rank with Holm correction, plus a matched-pairs rank-biserial effect size.
- **Confidence intervals.** 95 % bootstrap CIs (10,000 resamples over attacks).
- **Leakage controls.**
  - Normalisation, σ and θ are fit on train/val normal data only.
  - No hyperparameter is selected on test data. Defaults are fixed in `configs/`, and simulator settings are chosen on dev seeds 50–59 only.
  - Test files are read only by `scripts/run_*` evaluation stages.
- **Secondary leakage experiment (H6).** A supervised window classifier evaluated under a random window split vs a chronological split.
