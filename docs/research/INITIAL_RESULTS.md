# Initial HAI 21.03 pilot results

**Status: exploratory diagnostics only.** These results are not confirmatory, publication-ready, or evidence that the proposed constrained-counterfactual direction works. See `EXPERIMENT_PROTOCOL.md` for scope, split, and limitations.

**Current explanation results use identity-consistent single-tag ablations.** Earlier summaries used independent state and first-difference replacements for temporal attribution and greedy recourse. Those explanation rankings, counterfactual results, and stability comparisons are historical and are superseded by the corrected five-seed runs and audit at the end of this file. Detector fits, thresholds, and detection metrics were not changed by the correction.

The first development run (`hai21_20261031T021101Z`) computed explanations at attack-interval peaks, including missed events. It is excluded from explanation claims. The current primary evaluation uses the false-alarm-enabled runs listed below, with transition-consistent candidate generation. Earlier runs are retained for provenance only and are labeled where superseded.

## Run provenance

- Earlier corrected donor-paired runs: `hai21_seed42_20261031T024747_898999Z`, `hai21_seed43_20261031T024946_195322Z`, `hai21_seed44_20261031T025230_178332Z`, `hai21_seed45_20261031T025421_830799Z`, and `hai21_seed46_20261031T025614_604839Z`. Retained for provenance; their temporal candidates are superseded because donor levels and differences could contradict each other.
- Dataset: HAI 21.03 official release; 150,000 sampled train rows, train3 normal validation, test1–test5 (402,005 rows; 50 labeled attack intervals).
- All eight downloaded data files have SHA-256 hashes recorded in the run summary and local `data/raw/hai-21.03/manifest.json`. Raw data are excluded from Git.
- Superseded explanation runs: `hai21_seed42_20261031T034423_664547Z`, `hai21_seed43_20261031T034617_521452Z`, `hai21_seed44_20261031T034754_974592Z`, `hai21_seed45_20261031T034930_955869Z`, and `hai21_seed46_20261031T035106_659812Z`. Their detection metrics remain descriptive, but their independent state/difference ablations are not current explanation results.
- Current identity-consistent five-seed runs: `hai21_seed42_20261031T051335_879603Z`, `hai21_seed43_20261031T051709_959445Z`, `hai21_seed44_20261031T052039_198005Z`, `hai21_seed45_20261031T052411_161777Z`, and `hai21_seed46_20261031T052743_620105Z`. Current aggregates are `results/summaries/hai21_identity_consistent_seed42_46.json` and `results/summaries/hai21_identity_consistent_target_localization_seed42_46.json`.

## Pooled detection and explanation diagnostics

| Detector | Point F1 (5-seed mean; range) | AP (mean; range) | Test point FPR (mean; range) | Event recall (mean; range) | False alarm episodes / normal hour (mean; range) |
|---|---:|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 0.0108 (0.0093–0.0130) | 0.0244 (0.0243–0.0245) | 0.0084 (0.0084–0.0084) | 0.18 (0.18–0.18) | 4.26 (4.24–4.27) |
| PCA-STATE | 0.3971 (0.3936–0.4013) | 0.3458 (0.3429–0.3489) | 0.0118 (0.0117–0.0118) | 0.912 (0.90–0.92) | 17.82 (17.22–18.34) |
| PCA-TEMPORAL | 0.1987 (0.1985–0.1990) | 0.1704 (0.1669–0.1743) | 0.0095 (0.0095–0.0095) | 0.88 (0.88–0.88) | 25.62 (25.57–25.67) |

The validation target was 1% point FPR on a normal file. The test point FPRs are near that target in aggregate, but this does not mean the detectors have acceptable operating behavior: event-level false alarms remain high, and the per-file results vary substantially. This split therefore exposes a calibration/operational problem that a headline F1 alone would obscure. In seed 42, PCA-STATE and PCA-TEMPORAL event-recall intervals overlap; a paired bootstrap found a temporal-minus-state difference of -0.02 (95% CI [-0.12, 0.08], two-sided p=0.844). This event bootstrap is conditional on one sampled seed and is not a confirmatory across-seed or scenario-clustered test.

Per-file event recall averaged across the same five seeds is also uneven:

| Detector | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 |
|---|---:|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 0.00 | 0.15 | 0.38 | 0.00 | 0.25 |
| PCA-STATE | 0.80 | 0.82 | 1.00 | 1.00 | 1.00 |
| PCA-TEMPORAL | 0.60 | 0.85 | 1.00 | 1.00 | 0.92 |

## Corrected alarm-point explanation diagnostics

The five corrected runs explain the **first threshold crossing** within each detected attack event. Missed events are not included in explanation-quality denominators. Mean grouped-ablation stage hit@1 was 0.78 for RobustZ-TEMPORAL, 0.22 for PCA-STATE, and 0.68 for PCA-TEMPORAL. Mean explanation p95 across runs was 44.3, 79.0, and 79.2 ms respectively on the recorded Windows host. Five seeds describe sensitivity to training-sample and donor-bank sampling, but do not address scenario/file uncertainty; timing is host/load-sensitive. Full streamed predictions and event-level explanation rows are saved for all five runs.

## Explanation cautions

- Stage localization is not attacked-tag/root-cause localization. A group can match any active stage label.
- The originally reported top-5 Jaccard stability used independent state/difference perturbations and is superseded for temporal models by the representation-consistent audit below. Neither result establishes robustness to adaptive attack or distribution shift.
- The earlier oracle process-constrained recourse used HAI test labels to select editable groups. It has been removed from the current inference pipeline; those oracle values are not current results.
- The validation-reference distance is only a statistical proxy. It does not establish plant feasibility or safe actionability.

## Interpretation

This pilot does **not** support a positive claim for the proposed method. It provides three useful design signals: (i) preserve event-level false-alarm burden and per-file behavior in the primary evaluation; (ii) distinguish state and temporal representations because they trade off stage localization and false alarms; and (iii) a score-flipping counterfactual needs explicit attacker constraints and validated process dynamics before it can count as a meaningful explanation. Continue only with a narrowed protocol and stronger baselines; report these pilot values as development diagnostics if retained at all.

## Superseded interim recourse summaries

The intermediate recourse summaries `hai21_false_alarm_explanations_seed42_46.json`, `hai21_transition_consistent_seed42_46.json`, and `hai21_transition_dynamics_seed42_46.json` predate the final identity-consistent reruns. They are retained as run provenance only. Temporal candidate state levels and first differences were not consistently coupled in those analyses, so their recourse, target, false-alarm, and timing metrics must not be treated as current results. Current results are reported in “Current identity-consistent explanation reruns” below and in `results/summaries/hai21_identity_consistent_seed42_46.json`.

## False-alarm explanation audit (five seeds)

For this audit, a predicted-positive episode with no overlap with an HAI attack label is a pure false alarm under the dataset annotations. The runner sampled 100 first-alert explanations per detector per run (20 per test file), using a deterministic model/file/seed sample. This is about 3.6% of PCA-TEMPORAL false-alarm episodes, 5.1% of PCA-STATE episodes, and 21.5% of RobustZ episodes. The exact episode counts are over all five test files; sampled explanation metrics are descriptive.

| Detector | Pure false-alarm episodes, mean (range across seeds) | Greedy score-flip rate in sample | Support-only pass rate (seed range) | Screened pass rate | Mean sample p95 explanation latency (ms) |
|---|---:|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 465 (463–466) | 1.000 (1.00–1.00) | 0.582 (0.48–0.67) | 0.522 | 89.5 |
| PCA-STATE | 1,945 (1,880–2,002) | 1.000 (1.00–1.00) | 0.988 (0.98–1.00) | 0.898 | 309.9 |
| PCA-TEMPORAL | 2,797 (2,792–2,803) | 0.028 (0.00–0.07) | 0.014 (0.00–0.03) | 0.010 | 118.4 |

For PCA-STATE and RobustZ, support-only recourse passed more often on the sampled false alarms (0.988 and 0.582) than on detected attacks (0.947 and 0.222). For PCA-TEMPORAL, the false-alarm support-only pass rate was 0.014 versus 0.614 on detected attacks. The screened false-alarm pass rates were 0.522, 0.898, and 0.010 for RobustZ, PCA-STATE, and PCA-TEMPORAL. These model-dependent outcomes do not establish a generally reliable attack/false-alarm discriminator. HAI supplies no explanation correctness labels for false alarms, and sampled episodes are not matched to attack events. The 100 episodes per model and seed are a deterministic sample, not a census. Current aggregate rates and attack/false-alarm sample latencies are recorded in `results/summaries/hai21_identity_consistent_seed42_46.json`.

## Exact attack-target localization (five transition-consistent seeds)

The HAI maintainer technical manual provides attack IDs, timestamps, controllers, and target points. The local source-pinned mapping aligns all 50 entries to unique CSV attack intervals; all documented controller stages match the P1–P3 CSV labels. Twenty-one intervals include target names not present in the CSV schema, so the primary tag-localization analysis uses the 29 intervals where every manual-listed target is an exact CSV column name. Candidate schema aliases are not treated as gold labels. A209 also has a 40-second manual-versus-label-span discrepancy; the timestamp mapping is unaffected and the discrepancy is retained in the audit. Full provenance is in `docs/research/HAI_ATTACK_TARGET_MAPPING.md` and `data/annotations/hai21_mapping_audit.json`.

| Detector | Eligible complete targets detected / 29 | Native target hit@1 / hit@5 | Grouped-ablation target hit@1 / hit@5 | Supported counterfactual intersects a target |
|---|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 5 / 29 | 0.000 / 0.200 | 0.000 / 0.200 | 0.000 (mean 2.0 successful cases) |
| PCA-STATE | 27 / 29 | 0.000 / 0.074 | 0.000 / 0.185 | 0.000 (mean 24.6 successful cases) |
| PCA-TEMPORAL | 29 / 29 | 0.172 / 0.414 | 0.172 / 0.310 | 0.275 (mean 18.2 successful cases) |

Hit@k is conditional on the detector producing an explanation in that attack event. The counterfactual metric is conditional on score-flip and empirical-support success and records whether any edited tag overlaps the documented target set. It is not causal validation. The table reports the corrected identity-consistent runs (`hai21_seed42_20261031T051335_879603Z` through `hai21_seed46_20261031T052743_620105Z`); machine-readable values are in `results/summaries/hai21_identity_consistent_target_localization_seed42_46.json`. Five seeds reuse the same attack scenarios, so the run range is not independent scenario replication. Despite high PCA-STATE score-flip/support rates, its counterfactuals intersected no exact target tags; PCA-TEMPORAL support-only candidates intersected targets in 27.5% of successful cases but only 5/29 events unconditionally in every seed. The transition-screened conditional fraction was 38.5%, while unconditional hits were 5, 4, 5, 4, and 4 out of 29. The paired screened-minus-support-only estimate was -0.021 (95% event-bootstrap CI [-0.062, 0.000], sign-flip p=1.000; Holm-adjusted p=1.000 across three detectors). This gives no evidence that the screen improves target localization.

An exploratory paired analysis counts missed detections as localization misses and averages the five seeds within each attack ID. For PCA-STATE, the unconditional grouped-ablation minus native hit@5 difference is +0.103 (attack-event 95% bootstrap CI [0, 0.241], five-file cluster CI [0.036, 0.267], sign-flip p=0.243); for PCA-TEMPORAL it is -0.103 (attack-event CI [-0.241, 0.034], file-cluster CI [-0.235, -0.038], p=0.369). The event-level interval includes zero; the file-cluster interval is unstable with only five clusters. Neither supports a reliable localization improvement. The earlier interim target table and test (which reported 0.448 grouped-ablation hit@5 and a +0.034 paired difference for PCA-TEMPORAL) are superseded by the identity-consistent run summary cited above.

Ranges above are only the five run-specific values, not confidence intervals. Reproduce the target analysis with `python scripts/evaluate_hai_target_localization.py results/runs/hai21_seed42_20261031T051335_879603Z results/runs/hai21_seed43_20261031T051709_959445Z results/runs/hai21_seed44_20261031T052039_198005Z results/runs/hai21_seed45_20261031T052411_161777Z results/runs/hai21_seed46_20261031T052743_620105Z --randomization-draws 20000 --randomization-seed 161803 --output results/summaries/hai21_identity_consistent_target_localization_seed42_46.json`.

The updated evaluator also benchmarks native and grouped-ablation rankings against a uniform random top-$k$ null, conditioning on each event's target-set size and the observed detector hit/miss pattern. For PCA-TEMPORAL, unconditional native Hit@5 was 0.414 versus 0.100 expected under random ranking (excess 0.314; event-cluster bootstrap 95% CI [0.143, 0.488]; five-file cluster CI [0.239, 0.401]; Holm-adjusted conditional randomization $p=0.00090$). Grouped ablation was 0.310 versus 0.100 (excess 0.210; event CI [0.048, 0.380]; file CI [0.058, 0.283]; adjusted $p=0.00090$). This supports target-set ranking information on the fixed HAI benchmark, not causal validity or transfer to new plants. Native PCA-STATE and RobustZ excess intervals cross zero under event and file resampling. Some grouped-ablation conditional randomization tests are significant while their resampling intervals cross zero; interpret those p-values as conditional tests on these fixed cases, not population-level evidence. Holm correction covers all 18 model/explainer/K comparisons. The exact output records the seed, draw count, event outcomes, and both uncertainty summaries.

An earlier group-wide version (`hai21_seed42_20261031T022914_768500Z`) had 95–100% support-passing score flips but edited entire process groups. It was superseded because its median edit sizes (12–45 tags) were not operationally informative. The first tag-sparse runs (`hai21_seed42_20261031T023239_775201Z`, `hai21_seed43_20261031T023502_732488Z`) are retained for provenance but their PCA-STATE support figures are superseded by the corrected rerun above.

## BATADAL secondary partial-label evaluation (two donor seeds)

The official [BATADAL dataset page](https://www.batadal.net/data.html) describes Dataset03 as normal-only and Dataset04 as partially labeled attack data; the official competition test dataset is unlabeled. Local checksums and source URLs are saved in `data/raw/batadal/manifest.json`. The adapter uses 6,570 hourly rows from the first 75% of Dataset03 for fitting and its last 2,191 rows for normal-only calibration. Dataset04 contains 219 approximate positive rows and 3,958 unknown (`ATT_FLAG=-999`) rows, forming five annotated positive intervals. There are no confirmed normal rows in this evaluation period.

Runs: `batadal_seed42_20261031T024143_311772Z` and `batadal_seed43_20261031T024338_302964Z`. The detection metrics are identical because detectors use the same full chronological fit/validation split; the seed changes the sampled normal donor bank used by explanations.

| Detector | Validation FPR | Recall on 219 labeled positive rows | Recall on 5 approximate labeled intervals | Mean delay from approximate label start (hours) |
|---|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 0.0100 | 0.027 | 0.60 (3/5) | 25.7 |
| PCA-STATE | 0.0100 | 0.489 | 1.00 (5/5) | 4.8 |
| PCA-TEMPORAL | 0.0100 | 0.023 | 0.40 (2/5) | 16.0 |

Test precision, F1, and false-positive rate are intentionally omitted because unknown rows cannot be treated as negatives. These values are development evidence on approximately labeled intervals, not an independent test-set comparison. The tag-prefix groups (`L`, `F`, `S`, `P`) denote measurement/actuator tag types here; they are not attack-stage labels. No root-cause or stage-localization score is reported.

The run summaries retain full label masks, predictions, and per-interval delay/explanation records. Recreate the descriptive aggregation with `python scripts/summarize_batadal_runs.py results/runs/batadal_seed42_20261031T024143_311772Z results/runs/batadal_seed43_20261031T024338_302964Z --output results/summaries/batadal_summary_seed42_43.json`. These two seeds do not change the detector fit or threshold because Dataset03 is not subsampled; they perturb only the sampled normal donor bank.

## Superseded pre-rerun transition-screen analysis

The `hai21_transition_dynamics_seed42_46.json` and `hai21_transition_dynamics_target_localization_seed42_46.json` summaries use the intermediate run IDs `hai21_seed42_20261031T035956_662007Z` through `hai21_seed46_20261031T040800_776942Z`. They are retained for provenance but excluded from current interpretation because the final identity-preserving temporal explanation rerun changed the candidate and attribution inputs. Use the corrected results in the “Current identity-consistent explanation reruns” section; do not compare these pre-rerun numbers as if they were the same experiment.

## Factorial explanation-stability audit

The initial HAI stability analysis crossed two factors that had previously been conflated: input perturbation semantics (independent current-state/first-difference noise versus current-state jitter with first differences recomputed from the fixed prior state) and ablation semantics (legacy independent-channel edits versus identity-preserving channel edits). Within each ablation method, the input contrast holds its baseline attribution fixed. PCA-STATE receives zero difference and is not included in the temporal input-design contrast. The same saved models and alarm points were reused; detector thresholds and event detection were not rerun. Five Gaussian perturbations per event and design were evaluated at 2% and 5% of robust training dispersion.

| Detector | Legacy ablation + independent input, 2% | Legacy ablation + identity-preserving input, 2% | Identity-preserving ablation + independent input, 2% | Identity-preserving ablation + identity-preserving input, 2% | Identity-preserving ablation, 5% | Unique events |
|---|---:|---:|---:|---:|---:|---:|
| PCA-STATE | 0.995 | n/a | n/a | 0.995 | 0.989 | 36 |
| PCA-TEMPORAL | 0.989 | 0.717 | 0.701 | 0.701 | 0.558 | 35 |
| RobustZ-TEMPORAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6 |

For PCA-TEMPORAL, with legacy ablation fixed, independent minus identity-preserving input yielded +0.272 (95% event-bootstrap CI [0.194, 0.350], paired sign-flip $p=0.00020$, Holm-adjusted $p=0.00180$). With identity-preserving ablation fixed, the input-design difference was 0.000 (95% CI [0, 0], $p=1.0$). The difference-in-differences was +0.272 (95% CI [0.197, 0.351], sign-flip $p=0.00020$, Holm-adjusted $p=0.00180$ across nine comparisons). This is an interaction between perturbation and explainer semantics; the earlier interpretation that input validity alone explained the difference is withdrawn. Identity-preserving ablation stability declined by 0.142 from 2% to 5% jitter (95% CI [-0.173, -0.113], adjusted $p=0.00180$). PCA-STATE identity-preserving stability was 0.995 at 2% and 0.989 at 5%. RobustZ values use only six events. Seeds are averaged within repeated event IDs before event resampling; only five test files are available, so file-cluster uncertainty remains unresolved. Synthetic noise results concern local ranking sensitivity, not adversarial robustness, distribution shift, explanation correctness, or causal validity.

The superseding aggregate and per-event rows are in `results/summaries/hai21_explanation_stability_factorial_seed42_46.json`. Reproduce with five separate per-seed evaluator runs followed by `scripts/aggregate_hai_explanation_stability.py`, as specified in `docs/research/EXPERIMENT_PROTOCOL.md`. The older `hai21_identity_consistent_stability_seed42_46.json` is retained only as a superseded historical artifact because it used mismatched baseline attributions for the comparison.
## Current identity-consistent explanation reruns

The corrected method changes a one-tag reference ablation by replacing the state and recomputing its first difference against the observed preceding state. The greedy reference counterfactual uses the same identity; state-only PCA keeps a zero difference. Thus every temporal explanation query used in these runs satisfies $\Delta x_t=x_t-x_{t-1}$. The detector fits, thresholds, and attack alarm points are held constant relative to matching legacy-seed runs, but explanation ranks and recourse outputs are recomputed. The runs and aggregate files are identified near the top of this document.

Support-only score-flip/support pass rates were 0.222 (RobustZ-TEMPORAL), 0.947 (PCA-STATE), and 0.614 (PCA-TEMPORAL). With the one-step screen they were 0.067, 0.544, and 0.345. Screened stage hit@1 among successful edits was 1.000, 0.411, and 1.000; the RobustZ result is based on only a few successes. In the 100-event sampled false-alarm sets, screened pass rates were 0.522, 0.898, and 0.010. Thus the screen's false-alarm behavior differs by model and does not establish a generally reliable attack/false-alarm discriminator. Current end-to-end explanation p95 means were 99.3 ms (RobustZ), 279.3 ms (PCA-STATE), and 219.9 ms (PCA-TEMPORAL). These host-specific timings do not demonstrate hard real-time feasibility.

On the 29 HAI attack intervals with exact documented target tags, the corrected PCA-TEMPORAL grouped-ablation hit@5 was 0.310, versus 0.414 for native contribution ranking. The paired unconditional grouped-ablation-minus-native difference was -0.103 (event-bootstrap 95% CI [-0.241, 0.034], sign-flip $p=0.369$); five-file cluster bootstrap gave [-0.235, -0.038], which is unstable with only five files and should be treated as exploratory. For support-only counterfactuals, target intersection was 0.275 among successful candidates (mean 18.2 successes per seed) but only 5/29 events unconditionally in every seed. For the transition-screened method, the conditional target intersection was 0.385 among 11–12 successes per seed, while unconditional target hits were 5, 4, 5, 4, and 4 (mean 4.4/29). The paired screened-minus-support-only unconditional difference was -0.021 (event-bootstrap 95% CI [-0.062, 0.000], sign-flip $p=1.000$, Holm-adjusted $p=1.000$ across three detector comparisons). PCA-STATE and RobustZ had zero unconditional target hits under either method. This confirms no localization gain in the current pilot; conditional fractions should not be read as evidence that filtering improves target explanations.

## Normal-only Isolation Forest target-localization reference

To address the direct HAI/Isolation Forest overlap with RAID, a standard 500-tree Isolation Forest was fit to robust-scaled state and first-difference features from train1/train2 only; each seed's 1% threshold was calibrated on train3 only. The input uses the same 79 numeric tags as the other HAI pilot models, not RAID's regime-aware, physics-engineered 330-feature pipeline. Across seeds 42–46, mean point AP/F1/recall/test FPR were 0.242/0.298/0.229/0.0070. It detected an average of 22.2/29 exact-target events. Identity-preserving per-tag deletion rankings yielded unconditional Hit@1/3/5 of 0.041/0.172/0.297 over the 29 unique target-complete events. Against the conditional uniform-ranking reference, Hit@5 exceeded the event-matched expectation of 0.083 by 0.213 (event-cluster bootstrap 95% CI [0.096, 0.340], conditional random-ranking Monte Carlo $p<0.0001$, Holm-adjusted $p=0.0003$ over three Hit@K tests). This shows target-set ranking information above a uniform ranking on these fixed HAI cases; it does not imply causal validity or plant transfer.

Against the existing PCA-TEMPORAL grouped-ablation result averaged over the same five seeds, the paired event-level Hit@1/3/5 differences were -0.131/-0.103/-0.014; none survived Holm adjustment (adjusted $p=0.255/0.574/0.945$). The Isolation Forest therefore does not establish superior target localization over the process PCA baseline. Its single-row score p95 latency ranged from 118.6 to 138.6 ms across seeds, and fitted tree arrays occupied about 6.34–6.58 MB. These are host-only measurements and the scorer's p95 latency is not an edge-device result. Full seed-, file-, and event-level output is in `results/summaries/hai21_isolation_forest_target_reference.json`; reproduce with `python scripts/run_hai_isolation_forest_reference.py --seeds 42 43 44 45 46` after `python -m pip install -e ".[baselines]"`.

## Deletion-based detector-score faithfulness audit

At $k=5$, replacing the highest-ranked values with training-normal references crossed the alarm threshold in 97.8% of PCA-STATE, 65.9% of PCA-TEMPORAL, and 100% of RobustZ-TEMPORAL explanation events, compared with 7.3%, 1.9%, and 9.7% for uniformly random five-tag sets. Paired tests on alarm-margin removal favored top-ranked over random edits after Holm correction across 12 model-by-$k$ tests (adjusted $p=0.0024$, $0.0024$, and $0.0120$ at $k=5$). This is deletion sufficiency for the fitted score under reference replacement; it does not validate a causal explanation or demonstrate physical plausibility. RobustZ has only nine unique attack-event IDs in this analysis. Full results are in `results/summaries/hai21_identity_consistent_fidelity_seed42_46.json.gz`.

The current vectorized grouped-ablation implementation was numerically checked against an identity-preserving feature-loop implementation on the same HAI alarm points. The maximum attribution difference was below $5\times10^{-13}$. Across 20 event points for each PCA model and nine for RobustZ, median grouped-ablation latency was 10.7–10.9 times faster with vectorization. This microbenchmark covers grouped ablation only; it is not end-to-end explanation latency. Results are in `results/summaries/hai21_grouped_ablation_latency_ab_seed42.json`.

## BATADAL pump-component localization extension

The BATADAL FAQ maps its five approximate Dataset04 positive date ranges to pump components. The post-hoc target audit sums nonnegative contributions from the F_PU*/S_PU* measurements by pump, evaluates the first alarm, and counts missed events as misses. Across five seeds that reuse the same five intervals, PCA-STATE native component hit@1 was 0.80 and hit@3 was 1.00; grouped-ablation hit@5 was 0.80. RobustZ-TEMPORAL native hit@5 was 0.60 and PCA-TEMPORAL native hit@5 was 0.40. PCA-STATE detected all five labeled intervals; RobustZ detected three and PCA-TEMPORAL two. These results describe component-target agreement with approximate, day-level annotations; they do not establish exact manipulated tags, causality, or physical feasibility. Five unique events do not support confirmatory inference. Full results are in `docs/research/BATADAL_ATTACK_TARGET_AUDIT.md` and `results/summaries/batadal_target_localization_seed42_46.json`.


## Validation-threshold sensitivity (five seeds; 31 October 2026)

To check whether detector outcomes depend on the main 99th-percentile train3 operating point, each of the five seed-specific normal-only models was refit on its exact configured training subsample. Thresholds were recalibrated on train3 at q = 0.95, 0.975, 0.99, 0.995, and 0.999, then applied to unchanged saved test scores from the matching run. Every q=.99 threshold exactly reproduced its source run. Test labels were used only after threshold selection; explanation outputs were not recomputed. Values below are means over five fits on the same held-out test files, not confidence intervals or independent test replication.

| Detector | q=.95 event recall / false alarms per normal hour | q=.975 | q=.99 | q=.995 | q=.999 |
|---|---:|---:|---:|---:|---:|
| RobustZ-TEMPORAL | 0.224 / 3.34 | 0.180 / 3.41 | 0.180 / 4.26 | 0.180 / 3.67 | 0.180 / 2.52 |
| PCA-STATE | 0.940 / 36.21 | 0.932 / 26.56 | 0.912 / 17.82 | 0.892 / 13.21 | 0.700 / 2.99 |
| PCA-TEMPORAL | 0.980 / 81.91 | 0.940 / 47.08 | 0.880 / 25.62 | 0.840 / 15.43 | 0.680 / 3.10 |

PCA detector event recall decreases as the threshold becomes stricter, alongside a lower false-alarm episode rate. RobustZ event recall is flat from q=.975 onward, with a non-monotonic episode rate because contiguous alarm episodes merge or split as the threshold changes. At q=.99, seed ranges were 0.90–0.92 event recall for PCA-STATE and 17.22–18.34 false alarms/hour; PCA-TEMPORAL event recall was 0.88 across all seeds with 25.57–25.67 false alarms/hour; RobustZ event recall was 0.18 with 4.24–4.27 false alarms/hour. These threshold results do not identify a universally preferred operating point. They show that operating-point choice materially affects event coverage and alert burden, and make explicit that the single q=.99 setting does not tell the full detector trade-off. Full validation thresholds, seed ranges, and per-file metrics are in `results/summaries/hai21_threshold_sensitivity_seed42_46.json.gz`; individual seed reports and reproduction commands are documented in `docs/research/EXPERIMENT_PROTOCOL.md`.

