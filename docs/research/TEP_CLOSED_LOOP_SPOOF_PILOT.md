# TEP shared-view closed-loop spoof pilot

**Status:** exploratory simulator mechanism study. It is not a confirmatory experiment, a field-validated threat model, or a novelty claim.

## Purpose

The earlier TEP explanation-interposition pilot changed only the IDS observation after the simulator run. It did not change the controller's sensor input or the resulting plant trajectory. This follow-up closes that specific modeling gap: from the configured onset onward, one spoofed measurement is passed to the TEP controller and the exact same observation trace is replayed into the passive IDS. The simulator separately records the true process measurement and resulting state/control trajectories.

The scenario combines the simulator's IDV(6) A-feed-loss process disturbance with a measurement replacement on XMEAS(1), A Feed Flow. The process disturbance is not itself a cyberattack. The replacement is a deliberately simple sensor-path manipulation intended to test shared-view plumbing and its effect on control, detection, and first-alarm channel ranking.

## Reproduction

Use the pinned TEP Python implementation and run:

```powershell
python -m pip install "git+https://github.com/jkitchin/tennessee-eastman-profbraatz.git@9a6c8e5fcef4a2850778704e7793c87b0a187005"
python scripts/run_tep_closed_loop_spoof_pilot.py --config configs/tep_closed_loop_spoof_pilot.json --output results/summaries/tep_closed_loop_spoof_pilot.json
```

The identity interposition check uses seed 42 for 0.1 hours and exactly matches the simulator-native state, measurement, and manipulated-variable arrays. In every attacked run, the callback's pre-spoof raw measurements exactly match the simulator's time-aligned measurement record. The script separately copies the post-spoof observation for the controller and IDS trace and verifies exact equality.

## Protocol

- Simulator: TEP Python package 2.5.0, pinned commit `9a6c8e5fcef4a2850778704e7793c87b0a187005`, Python backend, 1-second integration/recording.
- Detector: normal-only RobustZ over 41 measured channels and their first differences. The IDS is passive: its score is replayed offline from the exact observation stream used by the controller and does not feed back into the process.
- Data separation: complete normal seed 42 for fitting; complete normal seed 43 for a 99.5th-percentile validation threshold; IDV(6) test runs use seeds 48–52. Earlier exploration seeds 44–47 are excluded from this pilot's test set.
- Attack: beginning at 0.5 h, replace the controller/IDS view of XMEAS(1) with its training-normal reference. The selected channel and event onset are fixed in configuration; the run does not search test candidates or use test labels to fit detector parameters.
- Explanation: identity-consistent grouped channel ablation at the clean run's first post-onset alarm sample and, separately, at the spoofed run's first alarm. Target rank is the rank of XMEAS(1) among 41 channel contributions.
- Labels: all samples at or after the configured onset are treated as the event interval. These are process-disturbance windows with a sensor spoof, not independently labeled real attacks.

The validation threshold was 5.066265 and the validation false-positive rate was 0.004722. Training references, robust scales, and the threshold use only the normal training/validation runs.

## Results

| Seed | Clean event-window point recall | Spoofed event-window point recall | Clean first alarm | Spoofed first alarm | XMEAS(1) rank at clean first-alarm sample under spoof | Spoof score at that sample |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.4067 | 0.4017 | 1802 | 1800 | 5 | 20.9302 |
| 49 | 0.4939 | 0.2633 | 1802 | 1802 | 6 | 26.3178 |
| 50 | 0.4017 | 0.3528 | 1802 | 1800 | 6 | 26.0135 |
| 51 | 0.4167 | 0.2689 | 1802 | 1802 | 6 | 16.0380 |
| 52 | 0.5528 | 0.2917 | 1802 | 1802 | 6 | 22.7821 |

The clean first alarm occurred 2 seconds after onset in all five seeds. Under the spoof, the score remained above threshold at that same clean first-alarm sample in all five seeds, and XMEAS(1)'s rank moved from 1 to 5 or 6 in all five. At the spoofed stream's own first alarm, however, the target ranked 1 on seeds 48 and 50 and 6 on seeds 49, 51, and 52; two spoofed first alarms occurred at the onset sample. This timing distinction changes the interpretation: displacement is consistent at the matched clean alarm point in this pilot, but not at each run's earliest spoofed alarm.

Mean event-window point recall over the post-onset interval fell from 0.4543 on the clean disturbance runs to 0.3157 with spoofing (paired mean change −0.1387; per-seed changes ranged from −0.0050 to −0.2611). These are five repeated simulator seeds for one disturbance, so the values are descriptive; no inferential test is claimed. The spoof's 95th-percentile absolute edit was approximately 10.78 robust training scales. Across the same-seed run pairs, the maximum absolute state difference ranged from 25.6202 to 25.6286 simulator state units and the maximum manipulated-variable difference from 25.7444 to 25.8594 percentage points.

## Interpretation and limitations

This result is stronger than the IDS-only pilot as evidence that the measurement wrapper feeds both consumers and changes closed-loop behavior. It is still a mechanism check. The intervention holds a channel at a training reference without independent sensor range, noise, slew-rate, or attacker-reachability bounds. The required measurement discrepancy grows to about 10.78 robust scales. It causes a measurable change in controller outputs and process-state trajectories, and it reduces subsequent IDS event alarm rates. Thus the experiment does not establish a stealthy or physically plausible attack, sustained alarm preservation, or operator-safe explanation.

The process state is simulator ground truth and is not available to a deployed monitor by default. No state-estimator integrity defense was evaluated. The study includes one detector, one channel, one process disturbance, five seeds, and one simulator implementation. It has no statistical power for generalization across disturbance families or plants and does not resolve the SSRN prior-art overlap. It should not be described as a new XAI method.

Next experiments require independently documented measurement bounds and rate limits, bounded/gradual spoof profiles selected before test evaluation, benign sensor-fault controls, additional disturbances and unseen seeds, matched explanation and detector baselines, and an independently implementable state/measurement consistency detector. Results should keep alarm-at-matched-sample, full event alarm rate, target-rank shift, and physical trajectory deviation as separate endpoints.

Machine-readable output: [`results/summaries/tep_closed_loop_spoof_pilot.json`](../../results/summaries/tep_closed_loop_spoof_pilot.json). Configuration and runner: [`configs/tep_closed_loop_spoof_pilot.json`](../../configs/tep_closed_loop_spoof_pilot.json), [`scripts/run_tep_closed_loop_spoof_pilot.py`](../../scripts/run_tep_closed_loop_spoof_pilot.py).
