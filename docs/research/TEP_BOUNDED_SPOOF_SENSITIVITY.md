# TEP closed-loop spoof profile sensitivity

**Status:** exploratory, paired simulator sensitivity study. The rate limits are statistical stress settings, not field-validated sensor constraints or attacker capabilities.

## Question

The earlier closed-loop reference-hold pilot moved the explanation rank at the matched clean alarm point, but needed a correction above ten robust training scales. This follow-up asks whether smaller, gradual additive edits to the same channel have a similar effect when the edited signal drives both the controller and the IDS observation stream.

The experiment compares a fixed unbounded hold at the training reference with positive additive-bias ramps capped at 0.5 and 1.0 RobustZ training scales. Each ramp starts at the IDV(6) onset and increases at 0.02 training scales per second, reaching its cap after 25 or 50 seconds. These caps and slew rates are declared stress values, not instrument specifications. The IDV(6) A-feed-loss process disturbance is not itself a cyberattack.

## Reproduction

```powershell
python -m pip install "git+https://github.com/jkitchin/tennessee-eastman-profbraatz.git@9a6c8e5fcef4a2850778704e7793c87b0a187005"
python scripts/run_tep_closed_loop_spoof_sensitivity.py --config configs/tep_closed_loop_spoof_sensitivity.json --output results/summaries/tep_closed_loop_spoof_sensitivity.json
```

The runner validates identity-wrapper equivalence, trains a normal-only RobustZ detector on seed 42, selects a threshold on normal seed 43, and compares the fixed profiles on the same IDV(6) seeds 53–57. Earlier exploration seeds 44–52 are excluded. For each attack profile, the same spoofed observation goes to the controller and is retained for offline passive IDS scoring; true process measurements, states, and manipulated variables remain separately recorded.

## Results

Threshold: 5.066265; validation false-positive rate: 0.004722. Clean and spoofed first-alarm samples and target ranks below are reported at the **clean run's first alarm sample**, which occurred at sample 1802 in all five seeds.

| Seed | Clean event-window point recall | Reference hold recall | 0.5-scale ramp recall | 1.0-scale ramp recall | Target rank: hold / 0.5 / 1.0 at matched sample |
|---:|---:|---:|---:|---:|---:|
| 53 | 0.1272 | 0.0056 | 0.1006 | 0.0761 | 6 / 1 / 1 |
| 54 | 0.5344 | 0.4339 | 0.5200 | 0.5094 | 6 / 1 / 1 |
| 55 | 0.3650 | 0.2028 | 0.3083 | 0.2778 | 6 / 1 / 1 |
| 56 | 0.5006 | 0.2694 | 0.4944 | 0.4911 | 5 / 1 / 1 |
| 57 | 0.5611 | 0.2156 | 0.5406 | 0.5233 | 6 / 1 / 1 |

The alarm score remained above threshold at the matched sample in all 15 profile/seed runs. The target moved from rank 1 to rank 5 or 6 under the reference hold on all five seeds, but remained rank 1 under both bounded ramps on all five. Mean event-window point recall was 0.4177 on clean disturbance runs, 0.2254 under reference hold, 0.3928 under the 0.5-scale ramp, and 0.3756 under the 1.0-scale ramp. Mean paired changes from clean were −0.1922, −0.0249, and −0.0421, respectively.

The ramp edits reached their configured caps; their 95th-percentile absolute edit sizes were exactly 0.5 and 1.0 robust scales. Maximum state difference from the clean same-seed trajectory ranged from 1.1893 to 1.1894 state units for the 0.5-scale profile and 2.3787 to 2.3787 for the 1.0-scale profile, versus approximately 25.6 for the reference hold. The lower-amplitude profiles caused smaller simulated trajectory changes and did not move the target attribution at the matched alarm point.

## Interpretation and limitations

Under these declared ramp settings, modest measurement edits changed detection rates slightly but did not steer the grouped-ablation explanation away from XMEAS(1). The large normal-reference hold did move the target rank, while reducing post-onset point recall substantially. This is a sensitivity result for one detector, one channel, one process disturbance, and five simulator seeds. No significance test is claimed because the seeds repeat one disturbance configuration and do not represent independent plants or attack families.

Robust-scale caps do not establish plausible physical sensor errors. The study has no sensor manufacturer limits, noise model, calibrated dynamics, benign sensor-fault controls, adaptive threat evaluation, or independently implementable state-estimation defense. Simulator states are ground truth, not deployment telemetry. These experiments do not establish causal explanation validity, safety, attacker reachability, novelty, or generalization.

Machine-readable output: [`results/summaries/tep_closed_loop_spoof_sensitivity.json`](../../results/summaries/tep_closed_loop_spoof_sensitivity.json). Configuration and runner: [`configs/tep_closed_loop_spoof_sensitivity.json`](../../configs/tep_closed_loop_spoof_sensitivity.json), [`scripts/run_tep_closed_loop_spoof_sensitivity.py`](../../scripts/run_tep_closed_loop_spoof_sensitivity.py).
