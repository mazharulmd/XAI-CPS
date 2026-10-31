# BATADAL pump-component target audit

**Audit date:** 31 October 2026
**Purpose:** add independently documented, post-hoc target-component evaluation for the five approximate positive intervals in BATADAL Dataset04.

## Annotation source and alignment

The BATADAL FAQ identifies the pump components associated with the approximate positive date ranges in Dataset04: PU10/PU11 (14–15 September 2016), PU2 (9–11 October), PU2 (30 October–1 November), PU7 (27 November), and PU6/PU7 (6–9 December). It warns that other undiscovered Dataset04 attacks may exist. The FAQ also states that the C-Town EPANET demand patterns differ from those used to generate BATADAL telemetry and that the generating demand values are unavailable. It therefore supports target-component annotation, but not exact replay or numeric transition validation. See the [official BATADAL FAQ](https://www.batadal.net/faq.html) and [official data page](https://www.batadal.net/data.html).

The source transcription is data/annotations/batadal_attack_target_map.csv. Its five date ranges align one-to-one with the five ATT_FLAG=1 contiguous intervals in local Dataset04. Dataset04 has 219 approximate positive rows and 3,958 unknown rows. The annotation timestamps are day-level; the narrower CSV spans are approximate and are not represented as precise attack onset/offset labels. The target granularity is the pump component (PU number), not a precise attacked sensor or actuator channel.

## Evaluation method

scripts/run_batadal.py now saves per-tag native detector contributions and grouped-ablation score decreases at the first alarm in each detected interval. These run outputs do not contain target annotations. scripts/evaluate_batadal_target_localization.py joins run event indices to the date-level source map only after the runs finish.

For tags such as F_PU10 and S_PU10, nonnegative tag scores are summed to a pump-component score. Components are ranked by this aggregate. Hit@K counts an interval as localized when at least one FAQ-listed target pump appears in the top K components. Missed detections are zero hits. Counterfactual target intersection is counted only when the candidate passes the frozen detector score-flip and normal-reference support checks; it is also reported unconditionally over all five intervals. It is not interpreted as causal evidence or safe repair.

The five seeds reuse the same five intervals. They are repeated runs, not 25 independent attack events. This audit reports descriptive means across seeds and events, with no subgroup significance tests.

## Results, seeds 42–46

All rows below use the five unique approximate intervals and count missed detections as localization misses. The seed runs reproduce the same rankings for these deterministic detectors.

| Detector | Event recall | Native component hit@1 | Native hit@3 | Native hit@5 | Grouped-ablation hit@5 | Recourse support-pass rate | Recourse target hit, unconditional |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA-STATE | 1.00 | 0.80 | 1.00 | 1.00 | 0.80 | 0.80 | 0.40 |
| RobustZ-TEMPORAL | 0.60 | 0.00 | 0.20 | 0.60 | 0.20 | 0.48 | 0.20 |
| PCA-TEMPORAL | 0.40 | 0.00 | 0.40 | 0.40 | 0.40 | 0.20 | 0.20 |

PCA-STATE’s native residual ranking intersects a documented pump target at rank 1 for four intervals and within rank 3 for all five. This is a small secondary benchmark result; the five events are not an inferential sample. RobustZ-TEMPORAL and PCA-TEMPORAL have lower detection event recall, and localization scores count their misses as zero. Neither native nor grouped-ablation target agreement establishes causal responsibility.

The machine-readable event- and seed-level results are in results/summaries/batadal_target_localization_seed42_46.json. Run IDs are recorded there. The secondary BATADAL detector results remain subject to partial positive labels: test precision, F1, and false-positive rate are not estimable from Dataset04.

## Reproduction

With local Dataset03/Dataset04 files present, set PYTHONPATH to src, run scripts/run_batadal.py with seeds 42 through 46, then pass those five run directories to scripts/evaluate_batadal_target_localization.py with --output results/summaries/batadal_target_localization_seed42_46.json.

The official FAQ documents target timing only by calendar date. The score measures agreement with those published component annotations; it does not certify exact manipulation times, tag-level attack attribution, physical feasibility, or causal explanation.
