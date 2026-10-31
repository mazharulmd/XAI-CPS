# HAI 21.03 exploratory protocol

**Status:** pilot protocol; not a preregistered confirmatory study. Freeze and version this protocol before any confirmatory reruns.

## Scope and claims

This experiment asks whether compact process-level anomaly baselines can detect HAI 21.03 attack events, and whether their explanations identify the HAI process-stage labels. It does **not** establish sensor-level root cause, network intrusion detection, causal explanation, plant safety, or operational actionability. HAI 21.03 has overall attack labels and three process labels (P1–P3); stage labels are not exact manipulated-tag annotations.

## Data and split

- Dataset: official HAI 21.03 CSV release, pinned by `data/raw/hai-21.03/manifest.json` SHA-256 values.
- Fit data: `train1.csv.gz` and `train2.csv.gz`; normal-only. At most 150,000 rows are sampled deterministically with the run seed.
- Threshold calibration: `train3.csv.gz`; normal-only validation. Threshold is the empirical 99th percentile of the detector score, corresponding to the configured 1% validation false-positive target.
- Test: the complete `test1.csv.gz`–`test5.csv.gz` files, with 50 labeled attack intervals. No test label enters fitting, normalization, threshold selection, or reference construction.
- Files are processed independently. One-step state and first-difference representations reset at every file boundary; no windows cross boundaries.
- Any feature grouping uses tag-name prefix metadata and is fixed before explanation scoring. Process labels are excluded from detector fitting, thresholding, capability declaration, and explanation generation; they are used only for post-hoc stage-localization evaluation.
- HAI's official technical manual documents attack IDs, start times, target controllers, and target points. A source-pinned transcription is aligned post hoc to the 50 CSV intervals by timestamp, then cross-checked against P1–P3 controller labels. Exact tag localization uses only intervals where every documented target point is an exact CSV column name; unresolved schema aliases are excluded, never inferred from explanations. See `HAI_ATTACK_TARGET_MAPPING.md` and the generated mapping audit. This mapping does not enter detector or explainer generation.

## Secondary dataset: BATADAL partial-label development evaluation

- Data: official BATADAL Dataset03 (normal-only) and Dataset04 (partially labeled attack development file), downloaded by `scripts/download_batadal.py`; checksums are in `data/raw/batadal/manifest.json`.
- Chronological split: first 75% of Dataset03 for normal-only fitting; final 25% for threshold and support calibration. Dataset04 remains held out from fitting and threshold selection.
- Dataset04 has 4,177 hourly rows: 219 `ATT_FLAG=1` labeled positive rows, 3,958 `ATT_FLAG=-999` unknown rows, and no known negative rows. The current annotated positives form five contiguous labeled intervals. Labels are approximate.
- Evaluate known-positive point recall and labeled-interval event recall/delay only. Do not report test precision, F1, or FPR because the remaining rows are unknown, not confirmed normal. Validation FPR is reported separately.
- Tag-prefix groups (`L`, `F`, `S`, `P`) denote tag types in this adapter, not documented attack stages or plant process zones. BATADAL's official FAQ provides post-hoc pump-component targets for five approximate positive intervals, but not a complete process-stage or exact manipulated-channel annotation.
- The official BATADAL FAQ documents pump-component targets for each of the five approximate Dataset04 positive intervals. These are evaluated post hoc using `data/annotations/batadal_attack_target_map.csv`; target labels do not enter fitting or explanation generation. Attributions over `F_PU*`/`S_PU*` tags are summed to pump components. This is component-target agreement, not exact manipulated-tag or causal localization. Dates are day-level and the Dataset04 interval boundaries remain approximate.
- The official BATADAL competition test dataset is unlabeled and is not used for detection-quality metrics.

## Detectors

1. **RobustZ-TEMPORAL:** robust-scaled per-tag level and first-difference residual score.
2. **PCA-STATE:** PCA reconstruction residual on robust-scaled current process state.
3. **PCA-TEMPORAL:** PCA reconstruction residual on robust-scaled state plus first differences.

These are transparent, reproducible baselines, not state-of-the-art deep IDS models. No claim of detector superiority is intended. Detector fitting uses normal data only.

## Explanation prototypes

- **Native residual ranking:** detector's per-tag residual/contribution score.
- **Grouped ablation:** replace one feature's level and difference by the training-normal reference and measure the score decrease, then sum scores by process-prefix group.
- **Greedy score-flipping recourse:** edit top-ranked channels to training medians/modes until the score falls below the validation threshold. It is explicitly an input-space score recourse, not a plant action or causal intervention.
- **Capability/support-constrained recourse prototype:** use an explicitly hypothetical attacker who can overwrite up to four current-sample process-prefix groups and 16 sensor readings. Rank groups and tags using grouped ablation; search ranked prefixes and a bounded set of nearest training-normal donor rows for candidate measurement levels. For temporal detectors, recompute each candidate first difference from the candidate level and the observed previous level, enforcing the data identity $\Delta x_t=x_t-x_{t-1}$; PCA-STATE continues to receive its state-only zero-difference representation. The candidate must flip the frozen score and lie within the validation-normal q95 nearest-reference distance. The profile groups and budgets do not encode network reachability, actuator control, or HAI attack identity. Report sensitivity at group budgets 1/2/4 and tag budgets 1/2/4/8/16.
- **Stability:** top-5 grouped-ablation Jaccard in a crossed audit. Input perturbation semantics (independent state/difference noise versus identity-preserving state jitter) are crossed with ablation semantics (legacy independent-channel edit versus identity-preserving channel edit). Input contrasts hold the attribution baseline fixed. PCA-STATE has no temporal-input contrast; it receives zero difference. Coherent current-state jitter is scaled to 2% and 5% of robust training feature dispersion. This measures local rank sensitivity only and is not adversarial robustness or distribution-shift robustness.

Explanation quality on attack intervals is evaluated at the **first threshold crossing inside each detected attack interval**, matching the actual alert point. Missed attack events remain in detection metrics but are excluded from attack-explanation denominators because no alarm exists to explain.

For false-alarm analysis, predicted-positive episodes with no overlap with any HAI attack label are treated as pure false alarms using the same event definition as `event_metrics`. The runner samples up to 20 such episodes per test file, deterministically by seed, model, and file, and explains each episode's first alert point with the same frozen detector and candidate generator. Labels are used only to choose and audit this post-hoc sample. The output reports sampled fraction, score-flip/support success, edit size, and latency; HAI provides no correctness label for these explanations, so these measures are behavior and operator-load diagnostics, not explanation-quality scores.

The candidate's first-difference identity is an algebraic consistency condition, not a plant transition model. The accepted-candidate distance check is only an empirical support heuristic. Neither is a causal constraint, actuator safety check, nor proof that the edited trajectory is physically feasible. The current approximate search checks ranked prefixes rather than all channel/tag subsets and has no optimality guarantee. Post-hoc HAI labels are used only to evaluate the generated edit groups.

The group/tag budgets are an analyst-declared stress grid, not capabilities inferred from HAI attack descriptions. The resulting recourse should be described as capability-budgeted under an explicit hypothetical profile. Validation against documented attacker reachability and actuator semantics is still required.

The additional transition-screen ablation fits a normal-only one-step ridge predictor on the 150,000 sampled training rows (79 tags). It scores the top-five robust standardized residuals for the observed previous state and candidate next state; the acceptance threshold is the 95th percentile measured on normal-only train3 validation transitions. This constraint is statistical, not a plant model, process invariant, or safety certificate. It is not operating-regime conditioned. The filter is evaluated alongside, and in addition to, detector score-flip and empirical support.

## Outcomes

- Detection: point precision/recall/F1, average precision, point FPR, event recall, event detection delay, false alarm episodes per normal hour, per-file and pooled results.
- HAI stage localization: hit@1 and hit@3 for documented P1–P3 labels on labeled attack events. Exact-tag localization is separately limited to its 29 complete target intervals.
- BATADAL component localization: pump-level hit@K for the five approximate positive intervals; aggregate F_PU*/S_PU* channel contributions to pump IDs, and count detection misses as localization misses. Do not generalize from five intervals or interpret pump-component agreement as exact attacked-tag attribution.
- Explanation behavior: score-flip success for unconstrained and capability/support-constrained recourse, minimum group budget, number and normalized magnitude of edited tags, local top-5 stability, and distance to the nearest train-normal reference. The reference-distance threshold is the 95th percentile of distances on a sampled normal validation subset. This is a weak statistical proxy; it does not prove physical/process plausibility.
- False-alarm explanation behavior: per-file episode counts and a deterministic sample of first-alarm explanations, with score-flip/support pass rates and latency. No target-accuracy claim is possible for these unlabeled false alarms.
- Cost: fit time, point scoring latency, batch throughput and per-event explanation latency on the recorded host. These measurements are hardware-specific and do not include an actual PLC/edge deployment or queueing load.
- Uncertainty: event-level percentile bootstrap intervals. Paired detector comparison uses matching attack intervals. Because there are only 50 attack events and attacks within files may share scenarios, intervals are descriptive; file/scenario cluster bootstrap should be added for confirmatory analysis.

## Leakage and interpretation controls

- No random split of overlapping windows; no test-time threshold tuning.
- Report all five test files and the pooled score. Pooled FPR may hide per-file distribution shift.
- Do not use point-adjusted F1 as a headline metric.
- Do not call the process group a root cause. Evaluate P1–P3 only as stage localization, and report unlabeled/ambiguous events distinctly.
- Do not call a nearest-normal distance a physical plausibility test. Plant constraints, actuator discreteness, temporal dynamics, and transition feasibility remain unmodeled.
- No multiplicity-adjusted confirmatory significance claims are made from this pilot.

## Confirmatory changes required

Before treating any result as a manuscript claim: add file/scenario cluster uncertainty; report per-file thresholds and calibration shift; add a matched temporal-counterfactual baseline (AR-Pro and/or JoCE) plus attribution baselines; establish that the HAI version, schema, windowing, and operating points match before making direct comparisons; replace the nearest-neighbor support proxy with validated process constraints or explicitly remove the physical-plausibility claim; and compare at common operating points in addition to a single validation quantile. The published AR-Pro HAI setup (86 features, 100-step windows) has not been shown to match the 79-tag HAI 21.03 data used here. BATADAL pump-target evaluation is limited to five approximate intervals, day-level target timing, and component-level target metadata; it is descriptive and cannot support confirmatory generalization. False-alarm explanations now have a descriptive deterministic sample, but they lack explanation-correctness labels and require expert or incident-grounded validity assessment before operational conclusions.

## Corrected capability and transition-screen results

This section uses the identity-consistent five-seed run set. Temporal grouped ablations and greedy references recompute first differences from the fixed preceding state; state-only PCA continues to use zero differences. The one-step learned transition screen is a normal-data statistical filter, not a validated plant model, safety constraint, or physical-feasibility certificate. Current run IDs are `hai21_seed42_20261031T051335_879603Z`, `hai21_seed43_20261031T051709_959445Z`, `hai21_seed44_20261031T052039_198005Z`, `hai21_seed45_20261031T052411_161777Z`, and `hai21_seed46_20261031T052743_620105Z`.

| Detector | Support-only recourse pass rate on detected attacks | Transition-screen pass rate (mean; seed range) | Screened stage hit@1 on successes | Added search p95 (ms) | Screened pass rate on 100 sampled pure false alarms/run |
|---|---:|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 0.222 (0.111–0.333) | 0.067 (0.000–0.111) | 1.000* | 45.1 | 0.522 |
| PCA-STATE | 0.947 (0.933–0.957) | 0.544 (0.511–0.587) | 0.411 | 155.8 | 0.898 |
| PCA-TEMPORAL | 0.614 (0.545–0.659) | 0.345 (0.318–0.364) | 1.000 | 130.2 | 0.010 |

*RobustZ's conditional screen hit is based on very few successes and is unstable. False-alarm behavior varies by detector: the screen still passes 89.8% of sampled PCA-STATE false-alarm candidates and 52.2% for RobustZ, while the PCA-TEMPORAL sampled pass rate is 1.0%. This is not evidence of general attack/false-alarm discrimination; it is an exploratory 100-episode sample per model and seed.

The exact-target-complete HAI subset has 29 events. Among successful PCA-TEMPORAL counterfactuals, documented-target intersection was 0.275 for support-only recourse (mean 18.2 successes/seed) and 0.385 with the transition screen (mean 11.4 successes/seed). Counting all 29 eligible events and treating misses and failed candidates as no hit, the support-only hit count was 5 in each seed; the screened hit counts were 5, 4, 5, 4, and 4. The paired unconditional difference (screened minus support-only) was -0.021 (event-bootstrap 95% CI [-0.062, 0.000], paired sign-flip p=1.000; Holm-adjusted p=1.000 across three detector comparisons). PCA-STATE and RobustZ produced zero exact-target hits under either generator. This does not show that the screen improves documented-target localization. The 29 scenario IDs recur across seeds, and five CSV files are too few for reliable file-cluster inference.

Current end-to-end explanation p95 means were 99.3 ms for RobustZ-TEMPORAL, 279.3 ms for PCA-STATE, and 219.9 ms for PCA-TEMPORAL on the recorded host. Extra transition-search p95 was 45.1, 155.8, and 130.2 ms, respectively. These host-specific timings do not establish hard real-time suitability; the architecture keeps explanations asynchronous to the detector/control loop. The current aggregate and event results are in `results/summaries/hai21_identity_consistent_seed42_46.json` and `results/summaries/hai21_identity_consistent_target_localization_seed42_46.json`.

Reproduce the corrected runs with `python scripts/run_hai.py --seed N` for N=42,...,46, aggregate with `scripts/summarize_explanation_runs.py <five-run-directories> --output results/summaries/hai21_identity_consistent_seed42_46.json`, and evaluate exact targets with `scripts/evaluate_hai_target_localization.py <five-run-directories> --randomization-draws 20000 --randomization-seed 161803 --output results/summaries/hai21_identity_consistent_target_localization_seed42_46.json`. The evaluator fixes each event's target set and detector hit/miss outcomes, uses the exact hypergeometric random Hit@K probability for a 79-tag universe, and samples 20,000 conditional random-ranking replicates. It reports event-ID and five-file-cluster bootstrap intervals for excess unconditional Hit@K, and Holm-adjusted Monte Carlo p-values across the 18 model/explainer/K comparisons. The randomization test is conditional on this benchmark; it does not establish transfer to new attack families. Then run the stability and fidelity commands in their sections below. Use `configs/hai_21_03.json`; do not combine these runs with legacy candidate-generation outputs.

## Documented action-type-stratified target localization

The HAI 21.03 Appendix A lists each attack scenario's manipulated variable type (SP, PV, or CV), target point, and attack operation. The project transcribes the variable types for AP01–AP25 to `data/annotations/hai21_scenario_action_types.csv`. These scenario labels are used only for post-hoc strata; they are never supplied to the detector or explainer. Exact-target evaluation is limited to the 29 attack IDs with complete target names in the raw CSV schema. The five seeds are averaged within attack ID before descriptive stratum means, and no subgroup tests are reported because several strata have only two or three events.

For PCA-TEMPORAL, exact-target hit@5 was 0.556 for CV-only events (n=9), 0.333 for CV+PV (n=3), 1.000 for CV+PV+SP (n=2), 0.250 for CV+SP (n=4), 0.750 for PV+SP (n=4), and 0.000 for SP-only events (n=7). Grouped-ablation hit@5 was 0.333, 0.000, 0.500, 0.250, 1.000, and 0.000 in those strata. Unconditional support-only counterfactual target-hit means were 0.333, 0, 0, 0.250, 0.250, and 0; screened means were 0.333, 0, 0, 0.100, 0.250, and 0. These small, post-hoc strata are descriptive leads only, not evidence of general action-type differences or causal validity. The per-event machine-readable output is `results/summaries/hai21_identity_consistent_target_localization_by_action_seed42_46.json`.

The HAI paper also gives a one-day normal setpoint schedule, but no executable or independently validated HAI 21.03 process model. The local `P1_B2016` tag is identified as a pressure setpoint in Appendix A, yet its local train1 values (0.9473–2.0103) do not directly match the paper table's pressure values (0.03–0.1); the paper does not provide a tag-level unit conversion resolving this gap. Do not apply those schedule numbers as hard counterfactual bounds until exact units and mapping are independently verified. See `docs/research/PROCESS_CONSTRAINT_EVIDENCE_AUDIT.md` for source and measurement details.

Reproduce with `python scripts/evaluate_hai_target_localization_by_action.py <five-identity-consistent-run-directories> --output results/summaries/hai21_identity_consistent_target_localization_by_action_seed42_46.json`.

## Factorial explanation-stability audit

The audit crosses input perturbation semantics (independently perturb current state and first difference versus jitter current state and recompute the difference from the observed preceding state) with grouped-ablation semantics (legacy independent-channel edits versus identity-preserving channel edits). Within each explainer method, input-design contrasts hold the baseline attribution fixed. PCA-STATE receives zero difference and is excluded from temporal input-design contrasts. The same saved model seeds and alarm rows are reused; the script does not rerun detection or threshold calibration. Perturbations use 2% and 5% robust training dispersion, with five draws per event and design. This is a local ranking-sensitivity audit, not adaptive robustness or distribution-shift testing.

The superseding factorial output is `results/summaries/hai21_explanation_stability_factorial_seed42_46.json`, based on runs `hai21_seed42_20261031T051335_879603Z`, `hai21_seed43_20261031T051709_959445Z`, `hai21_seed44_20261031T052039_198005Z`, `hai21_seed45_20261031T052411_161777Z`, and `hai21_seed46_20261031T052743_620105Z`. Run the evaluator once per run directory because each process reloads the data:

```powershell
python scripts/evaluate_hai_explanation_stability.py <run-42> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results/summaries/hai21_stability_factorial_seed42.json
python scripts/evaluate_hai_explanation_stability.py <run-43> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results/summaries/hai21_stability_factorial_seed43.json
python scripts/evaluate_hai_explanation_stability.py <run-44> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results/summaries/hai21_stability_factorial_seed44.json
python scripts/evaluate_hai_explanation_stability.py <run-45> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results/summaries/hai21_stability_factorial_seed45.json
python scripts/evaluate_hai_explanation_stability.py <run-46> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results/summaries/hai21_stability_factorial_seed46.json
python scripts/aggregate_hai_explanation_stability.py results/summaries/hai21_stability_factorial_seed42.json results/summaries/hai21_stability_factorial_seed43.json results/summaries/hai21_stability_factorial_seed44.json results/summaries/hai21_stability_factorial_seed45.json results/summaries/hai21_stability_factorial_seed46.json --bootstrap-draws 5000 --output results/summaries/hai21_explanation_stability_factorial_seed42_46.json
```

The aggregate averages seeds within repeated event IDs before event bootstrap/sign-flip inference. For PCA-TEMPORAL at 2%, legacy ablation gave Jaccard 0.989 under independent input noise and 0.717 under identity-preserving input jitter; identity-preserving ablation gave 0.701 under either input design. With legacy ablation fixed, the independent-minus-identity-preserving input contrast was +0.272 (95% CI [0.194, 0.350], Holm-adjusted $p=0.00180$); with identity-preserving ablation fixed, the contrast was zero. Their difference-in-differences was +0.272 (95% CI [0.197, 0.351], Holm-adjusted $p=0.00180$ across nine comparisons). Thus the input-design effect depends on ablation semantics; the data do not support a general claim that invalid inputs alone inflate stability. The earlier `hai21_identity_consistent_stability_seed42_46.json` compared mismatched baseline attributions and is retained only as a superseded provenance artifact. Neither result measures adversarial robustness, shift robustness, causal validity, or explanation correctness.
## Deletion-based detector-score faithfulness

At each saved first-alarm point, top-ranked, bottom-ranked, and uniformly random sets of $k\in\{1,3,5,10\}$ sensor values were replaced with training-normal reference values. Every temporal replacement recomputed its difference from the observed preceding state; the PCA-STATE input remained zero-difference. The primary quantity is whether the top-$k$ deletion lowers the detector score to or below its threshold. A secondary paired test compares the fraction of the alarm margin removed by top-$k$ deletion against 50 random sets per event and $k$, with event IDs bootstrapped after averaging repeated seeds. Holm correction covers 12 model-by-$k$ tests.

At $k=5$, top-ranked deletion crossed the alarm threshold in 97.8% of PCA-STATE, 65.9% of PCA-TEMPORAL, and 100% of RobustZ-TEMPORAL explanation events; the corresponding random-set crossing rates were 7.3%, 1.9%, and 9.7%. For the paired margin-removed comparison, the event-bootstrap 95% CIs were [7.36, 77.58] for PCA-STATE, [6.23, 56.32] for PCA-TEMPORAL, and [14.60, 47.23] for RobustZ; Holm-adjusted sign-flip $p$ values were 0.0024, 0.0024, and 0.0120. The numbers of repeated-seed-averaged alarm IDs were 46, 44, and 9, respectively; RobustZ's estimate is sparse. The margin ratio can be large when the initial alarm is close to threshold. These results show local score sufficiency under reference deletion relative to random feature sets, not explanation correctness, causal validity, documented-target localization, or physical feasibility. Random-control Monte Carlo uncertainty is not nested inside the event bootstrap.

Reproduce this audit with `python scripts/evaluate_hai_explanation_fidelity.py <five-run-directories> --random-sets 50 --bootstrap-draws 5000 --output results/summaries/hai21_identity_consistent_fidelity_seed42_46.json`. The per-event output preserves seed and event identifiers. The checked-in result is losslessly compressed as results/summaries/hai21_identity_consistent_fidelity_seed42_46.json.gz; use gzip -dk to restore the plain JSON.


## HAI threshold-sensitivity audit

`evaluate_hai_threshold_sensitivity.py` reconstructs each of five seed-specific normal-only detector fits and calibrates thresholds at validation quantiles 0.95, 0.975, 0.99, 0.995, and 0.999 using train3 only. It verifies that each 0.99 threshold reproduces its source run before the report is emitted. It reports pooled and per-file point/event results and false-alarm episodes per normal hour. Test labels are used only after threshold selection to compute test metrics. The analysis leaves explanations fixed. Means and seed ranges summarize repeated model fits on the same five held-out files and are not confidence intervals or independent replication.

Run each seed separately so each process releases its data/model memory, then aggregate the per-seed JSON reports:

```powershell
$env:PYTHONPATH = "src"
python scripts\evaluate_hai_threshold_sensitivity.py results\runs\hai21_seed42_20261031T051335_879603Z --output results\summaries\hai21_threshold_sensitivity_seed42.json
python scripts\evaluate_hai_threshold_sensitivity.py results\runs\hai21_seed43_20261031T051709_959445Z --output results\summaries\hai21_threshold_sensitivity_seed43.json
python scripts\evaluate_hai_threshold_sensitivity.py results\runs\hai21_seed44_20261031T052039_198005Z --output results\summaries\hai21_threshold_sensitivity_seed44.json
python scripts\evaluate_hai_threshold_sensitivity.py results\runs\hai21_seed45_20261031T052411_161777Z --output results\summaries\hai21_threshold_sensitivity_seed45.json
python scripts\evaluate_hai_threshold_sensitivity.py results\runs\hai21_seed46_20261031T052743_620105Z --output results\summaries\hai21_threshold_sensitivity_seed46.json
python scripts\aggregate_hai_threshold_sensitivity.py `
  results\summaries\hai21_threshold_sensitivity_seed42.json `
  results\summaries\hai21_threshold_sensitivity_seed43.json `
  results\summaries\hai21_threshold_sensitivity_seed44.json `
  results\summaries\hai21_threshold_sensitivity_seed45.json `
  results\summaries\hai21_threshold_sensitivity_seed46.json `
  --output results\summaries\hai21_threshold_sensitivity_seed42_46.json
```

The full aggregate preserves each run's validation threshold, per-file metrics, seed ranges, and source run IDs. Do not interpret the repeated fits as new independent test samples.
