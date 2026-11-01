# TEP IDS and explanation-interposition pilot

**Status:** Exploratory mechanism check only. This is not a confirmatory benchmark, novelty claim, or physical feasibility result.

## Question

Can a single IDS observation edit change the first-alarm channel ranking while leaving the detector above its alarm threshold, under a simple normal-only CPS detector? The pilot also checks whether a persistent single-channel replacement behaves differently from a one-observation edit.

The TEP simulator's IDV(6) disturbance is an **A-feed-loss process disturbance**, not a cyberattack. We overlay a separately defined IDS-input manipulation on telemetry from that process event. The controller and plant are not re-run under the manipulated IDS input. The result therefore does not demonstrate a closed-loop sensor compromise.

## Reproduction

The public simulator is pinned to version 2.5.0, commit `9a6c8e5fcef4a2850778704e7793c87b0a187005`. Install it and run the pilot from the repository root:

```powershell
python -m pip install "git+https://github.com/jkitchin/tennessee-eastman-profbraatz.git@9a6c8e5fcef4a2850778704e7793c87b0a187005"
python scripts/run_tep_interposition_pilot.py --config configs/tep_interposition_pilot.json --output results/summaries/tep_interposition_pilot.json
```

The configuration and output are [`configs/tep_interposition_pilot.json`](../../configs/tep_interposition_pilot.json) and [`results/summaries/tep_interposition_pilot.json`](../../results/summaries/tep_interposition_pilot.json). The runner uses the simulator's Python backend, one-hour runs, 1-second recording, and the repository's `RobustZDetector` and identity-consistent single-channel ablation.

## Data and detector protocol

- Train a single RobustZ detector on the complete normal TEP run with seed 42.
- Select its threshold as the higher empirical 99.5th percentile of scores on a separate, complete normal run with seed 43. Observed validation FPR was 0.004999; threshold was 5.022276.
- Evaluate separate complete IDV(6) runs with seeds 46 and 47. The disturbance is scheduled at 0.5 simulated hours. Seeds 44 and 45 were inspected during disturbance screening and were excluded from the reported test runs.
- Each test has 3,601 recorded samples. Event labels are derived from configured simulator onset time, not from test scores. No random windows are split across partitions; scaling and references are learned from training only.
- The predeclared localization target is XMEAS(1), A Feed Flow, which corresponds to the A-feed-loss disturbance. The simulator source/API documents IDV(6) as A Feed Loss and XMEAS(1) as A Feed Flow ([pinned TEP API](https://github.com/jkitchin/tennessee-eastman-profbraatz/blob/9a6c8e5fcef4a2850778704e7793c87b0a187005/docs/api.md)). This is intervention-defined target agreement, not a causal explanation ground truth.
- The IDS consumes 41 measured channels and their first differences. For every attack edit, the first difference is recomputed against the preceding **observed IDS input**. Channel deletion uses training references and recomputes the paired difference, following the repository's identity-consistent explainer semantics.

## Results

| Test seed | First alarm delay | Event alarm rate | Point AP | Point F1 | Pre-event FPR | XMEAS(1) rank at first baseline alarm |
|---:|---:|---:|---:|---:|---:|---:|
| 46 | 2 s | 0.6006 | 0.9587 | 0.7491 | 0.00278 | 1 |
| 47 | 2 s | 0.5828 | 0.9857 | 0.7351 | 0.00278 | 1 |

**One-observation edit.** At the first post-onset alarm, the runner evaluates 410 single-channel offsets (41 channels × 10 signed multipliers from 0.1 to 1.0 robust training scales). All candidates retained the alarm at that sample, but the best candidate left XMEAS(1) at rank 1 on both seeds. Within this small one-observation budget, the pilot did not displace the target ranking.

**Persistent edit.** The runner evaluates 451 candidates per test (41 channels × one hold-at-training-reference edit plus 10 persistent additive biases). It selects only candidates that keep the detector score above threshold at the *same sample as the clean run's first alarm*. On both seeds, exactly one candidate demoted XMEAS(1): hold XMEAS(1) at its training reference from event onset through the end of the run. Its first-alarm rank changed from 1 to 5, with the alarm score still at 24.9495 (seed 46) and 24.5913 (seed 47), both above the 5.0223 threshold. The realized hold corrections were 10.7365 and 10.7241 robust training scales, respectively. The next four highest-ranked channels replaced XMEAS(1) at the first alarm.

The persistent hold reduced the event alarm rate from 0.6006 to 0.5439 on seed 46 and from 0.5828 to 0.5100 on seed 47. The initial alarm was preserved at the same sample, while later point alarms declined. Thus this is **pointwise first-alarm preservation**, not preservation of the full alarm sequence. It is also a large oracle-selected correction, not a small stealthy perturbation. The per-seed search found no target demotion among the 450 remaining persistent bias/hold candidates.

## Interpretation and limits

The preliminary evidence is mixed. Small one-sample offsets did not move the event target. A persistent replacement of the target channel with its normal training reference moved the ranking on both held-out seeds while retaining the first alarm, but required a correction around 10.7 robust scales and reduced later alarm frequency. This exposes a possible tension between target masking and sustained detection; it does not establish that a realistic attacker can meet this objective.

The exhaustive search is a white-box upper bound: it knows the fitted detector, validation threshold, event onset, and intervention-defined target. The reference-hold mode is not clipped to sensor limits, instrument noise, a rate-of-change constraint, or an attacker-accessible range. The objective tests one simple channel-deletion explainer and one detector. It does not test alternative explanation methods, adaptive attack algorithms, controller feedback, alarm latch policies, process safety, or the inaccessible SSRN overlap. Only two test seeds and one disturbance type are included; no significance test or generalization claim is warranted.

The next experiment must add independently justified sensor bounds and benign-noise/process-disturbance controls, compare matched detector and explainer baselines, evaluate both the full alarm sequence and latched-alert semantics, and use additional untouched seeds and disturbance types. If reference replacement exceeds plausible sensor capabilities, this attack mode must be excluded from any physical-threat claim.
