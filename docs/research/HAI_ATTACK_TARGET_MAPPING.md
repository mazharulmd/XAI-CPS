# HAI 21.03 attack-target mapping audit

## Source and procedure

The HAI maintainers' technical manual lists 50 HAI 21.03 attacks with attack IDs, minute-resolution start times, durations, target controllers, scenarios, and target points (printed pages 36–38; PDF pages 37–39). The source was downloaded from the maintainers' repository at pinned commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`; its SHA-256 is `0668345c4e80331b918fe17c81f8f363b13bd22886831d286e761bc62b71a556`. The manual is linked at [the pinned official PDF](https://github.com/icsdataset/hai/blob/2a814cebc9a66b06c9e5cd545e2d72e65d383737/hai_dataset_technical_details.pdf), and the HAI dataset paper is [Shin et al. (CSET 2021)](https://doi.org/10.1145/3474718.3474719).

The manually transcribed source records are in `data/annotations/hai21_attack_scenarios.csv`. `scripts/build_hai_target_annotations.py` aligns each record to a unique test-file attack interval using the manual's minute-resolution time and the CSV's one-second timestamps. All 50 IDs map uniquely across test1–test5 with counts 5, 20, 8, 5, and 12. The documented target-controller process stages match the CSV's P1–P3 labels for all 50 intervals. The script writes `data/annotations/hai21_event_target_map.csv` and `data/annotations/hai21_mapping_audit.json`.

This mapping is post-hoc evaluation metadata. It is not passed to detector fitting, threshold calibration, feature ranking, or counterfactual generation. The evaluator uses exact target-point names present in the CSV and excludes incomplete target lists from its exact tag-localization denominator.

## Annotation discrepancies and restrictions

- **29/50 intervals** have every manual-listed target point represented by an exact CSV column name and can enter exact target-tag localization.
- **21/50 intervals** contain at least one target point absent from the CSV schema. The manual's `P2_AutoSD` and `P2_SCO` have plausible candidate aliases `P2_ASD` and `P2_CO_rpm`; the mapping tool records these only as unverified candidates. The manual lists `P2_LCV01D` for controller `P3-LC-CO2`; `P3_LCV01D` exists in the CSV but remains an unverified candidate because the source itself is inconsistent. None of these aliases count as ground truth.
- For attack A209, the manual duration is 159 seconds while the CSV attack-label interval spans 199 rows, a 40-second discrepancy. The unique start-time mapping is retained, but the discrepancy is visible in the audit and the duration is not silently reconciled.
- The manual timestamps have minute precision. Alignment accepts a unique interval whose observed start is within 60 seconds of that minute; all matches are unique and stage-consistent. `observed_start` and `observed_end_exclusive` retain the exact CSV values.
- A documented target point is not automatically a causal root cause. Exact target localization should be reported as agreement with the scenario's documented manipulated/target point, not as causal explanation or plant-safe recourse.

## Reproduction

With the local HAI 21.03 CSV files available:

```powershell
python scripts\build_hai_target_annotations.py
python scripts\evaluate_hai_target_localization.py <run-dir-1> <run-dir-2> --output results\summaries\hai21_target_localization.json
```

The target evaluator expects event-level top-tag rankings saved by the HAI runner and uses only fully exact target lists. Its seed ranges are descriptive, not confidence intervals. Before reporting a headline target-localization result, resolve the three tag-name discrepancies with the dataset maintainers or a version-specific authoritative crosswalk and preserve the current exact-match analysis as a sensitivity baseline.

## Initial exact-target evaluation (five seeds)

The corrected identity-consistent five-seed alarm-point runs with persisted per-tag rankings evaluated all 29 complete target lists. The detectors explained 27/29 PCA-STATE, 29/29 PCA-TEMPORAL, and 5/29 RobustZ target-complete events. Among explained events:

| Detector | Native hit@1 / hit@5 | Grouped-ablation hit@1 / hit@5 | Supported counterfactual intersects any target tag |
|---|---:|---:|---:|
| RobustZ-TEMPORAL | 0.000 / 0.200 | 0.000 / 0.200 | 0.000 (mean 2.0 successful cases per run) |
| PCA-STATE | 0.000 / 0.074 | 0.000 / 0.185 | 0.000 (mean 24.6 successful cases) |
| PCA-TEMPORAL | 0.172 / 0.414 | 0.172 / 0.310 | 0.275 (mean 18.2 successful cases) |

The table uses runs `hai21_seed42_20261031T051335_879603Z` through `hai21_seed46_20261031T052743_620105Z`; its machine-readable result is `results/summaries/hai21_identity_consistent_target_localization_seed42_46.json`. The counterfactual column is conditional on a score-flipping and empirical-support-passing candidate; it measures whether the edited tag set intersects the documented target set. It is not proof that the edit recovers a cause or attack action. The low exact-target hit rates, despite score-flip/support pass rates for PCA, are evidence that detector recourse and attacker-target explanation are different objectives. Five runs reuse the same 50 test scenarios, so they are not independent scenario replications and seed ranges are descriptive only.

The aggregate output also reports an analytic uniform-ranking reference for the 79 numeric HAI tags. For each eligible event and $k\in\{1,3,5\}$, it computes the hypergeometric probability that a size-$k$ random ranking intersects the documented target set; detector misses contribute zero to the unconditional reference. At hit@5 this reference is 0.096 for PCA-STATE, 0.100 for PCA-TEMPORAL, and 0.017 for RobustZ-TEMPORAL. Conditional randomization tests and event/file resampling for excess unconditional hit rates are reported below; conditional-on-detection Hit@K summaries must not be compared directly with these unconditional reference rates.

The evaluator now also conditions a uniform random-ranking null on the observed detector hit/miss pattern and reports the excess unconditional Hit@K over its exact event-specific expected value. At $k=5$, PCA-TEMPORAL native attribution was 0.414 against 0.100 expected random, an excess of 0.314 (event-cluster bootstrap 95% CI [0.143, 0.488]; five-file cluster CI [0.239, 0.401]; Monte Carlo conditional randomization $p<0.00005$, Holm-adjusted $p=0.00090$ across 18 model/explainer/$k$ comparisons). Grouped ablation was 0.310 against 0.100 (excess 0.210; event CI [0.048, 0.380]; file CI [0.058, 0.283]; adjusted $p=0.00090$). These tests use the fixed 29-event benchmark, with five seeds averaged within event ID; the randomization $p$ tests the conditional ranking null on those cases, while the event/file bootstrap intervals describe sensitivity to resampling the small event/file sample. Neither establishes transport to new plants or attack families. Other models and $k$ values are reported in the machine-readable summary.

An exploratory paired analysis averages seed-level outcomes within attack ID, counts a missed detection as a localization miss, and compares grouped-ablation with native hit@k. At hit@5, the unconditional paired difference was +0.103 for PCA-STATE (attack-event bootstrap 95% CI [0, 0.241]; file-cluster bootstrap CI [0.036, 0.267]; paired sign-flip p=0.243) and -0.103 for PCA-TEMPORAL (attack-event CI [-0.241, 0.034]; file-cluster CI [-0.235, -0.038]; p=0.369). The event-level interval includes zero. The file-cluster interval is unstable with only five clusters, and the same 29 attack IDs recur across all five seeds. These analyses do not establish a statistically reliable improvement. The earlier interim result of PCA-TEMPORAL grouped-ablation hit@5=0.448 and paired difference +0.034 predates these final identity-consistent runs and is superseded.
