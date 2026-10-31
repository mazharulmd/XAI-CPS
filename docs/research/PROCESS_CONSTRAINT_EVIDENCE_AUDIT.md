# HAI 21.03 process-constraint and attack-action evidence audit

**Audit date:** 31 October 2026
**Purpose:** determine which claims and evaluations HAI 21.03 can support without treating a learned normal-data score as physical feasibility.

## Source evidence

The HAI 21.03 dataset paper describes a coupled hardware-in-the-loop testbed with boiler (P1), turbine (P2), water-treatment (P3), and simulator (P4) processes. It states that the six controller loops and additional simulator valve loops were adjusted during testbed reconfiguration, including PID gains and analog scale/bias values. It also says the data include setpoint (SP), process-variable (PV), and control-variable (CV) measurements to support analysis of control-loop relationships. The paper's normal-operation table gives one-day, time-indexed setpoint schedules for pressure, level, flow, and temperature. Its attack section describes user-defined attacks run through OPC and records target/time metadata. Appendix A identifies HAI 21.03 scenario targets and whether the manipulated variable is SP, PV, or CV, with a short attack operation description. See [Shin et al., CSET 2021, Sections 2–3 and Appendix A](https://cset21.isi.edu/papers/cset21-1.pdf) and the [pinned HAI maintainer technical manual](https://github.com/icsdataset/hai/blob/2a814cebc9a66b06c9e5cd545e2d72e65d383737/hai_dataset_technical_details.pdf).

This evidence is sufficient to define **post-hoc, documented attack-action strata** and to audit target agreement. It is not a plant model. The paper does not publish executable controller equations, calibrated actuator bounds, a verified state-transition relation, or a safety invariant for the HAI 21.03 release. It explicitly notes limitations in representing the real world through the simulator and describes changes to HIL signal scaling/bias and controller gains. Therefore a normal-only learned predictor or empirical data support cannot be described as a physical-feasibility or safety check.

## Schema and numerical mapping caveat

The local HAI 21.03 files have 84 columns: `time`, 79 numeric process tags, and four attack-label columns. The CSET paper has an internal count inconsistency: it describes 78 data points and 84 total columns, while also referring to competition columns C01–C79. This project follows the pinned raw files and reports the observed 79 numeric tags; it does not infer missing points from the paper's count.

The CSET Appendix A maps `P1_B2016` to the P1 pressure-controller setpoint target. In local `train1.csv.gz`, that tag ranges from 0.9473 to 2.0103, while the paper's daily schedule table lists pressure values from 0.03 to 0.1 without an explicit tag-level unit conversion. This mismatch prevents directly turning the published schedule numbers into an executable numeric bound without further source clarification. Other schedule values likewise require exact channel/unit mapping and validation across normal files before they can be used as constraints.

## Later-release topology source audit

The HAI maintainers' README says HAI/HAIEnd 23.05 adds internal boiler DCS points and NetworkX graph data for anomaly-detection analysis. I inspected the graph files at the source commit [`ebcd09b`](https://github.com/icsdataset/hai/commit/ebcd09bbb432a35be39dcfaf1d800083fd06777b), which introduced the HAI/HAIEnd 23.05 update. `graph/boiler/phy_boiler.json` contains 41 equipment/sensor/actuator nodes and 67 directed edges; nodes include human-readable descriptions plus `in_tags`/`out_tags`, and edges encode a coarse `dynamics` category. `graph/boiler/dcs_1001h.json` contains 31 DCS point nodes and 42 directed links labeled by signal names. The physical graph's exported `type`, `device`, and `dynamics` node attributes repeat tuple-like strings across nodes rather than supplying a clean per-node categorical value, so those attributes require source clarification before use.

This graph is a useful candidate source for **topological dependency checks and tag-map audit**, not an independently validated process model: it has no numeric transition equations, calibrated operating bounds, or safety invariants. It is also tied to HAI/HAIEnd 23.05, while the current quantitative study uses HAI 21.03. Until a version-specific, point-by-point mapping between the 23.05 graph tags and the 21.03 CSV schema is demonstrated, the graph must not be applied to the current 21.03 counterfactuals or described as validating their physical feasibility. The source inspection therefore does not resolve the process-constraint gap.

## BATADAL process-model limit

BATADAL provides a C-Town EPANET input file, but the official FAQ warns that its published demand patterns differ from those used to generate Dataset03/Dataset04 and that the actual generating demands are not released. The FAQ therefore says the provided EPANET model cannot be used to reconstruct the actual measurements and verify the released telemetry against a simulated trajectory. It does provide useful topology, actuator control logic, units, and day-level pump target descriptions. Those support the pump-component target audit in `docs/research/BATADAL_ATTACK_TARGET_AUDIT.md`, but they do not support exact numeric transition checks or physical certification of the current counterfactual edits. See the [official BATADAL FAQ](https://www.batadal.net/faq.html).

## Implemented action-type evaluation

`data/annotations/hai21_scenario_action_types.csv` transcribes the HAI 21.03 Appendix A variable types for scenarios AP01–AP25. `scripts/evaluate_hai_target_localization_by_action.py` maps each attack interval's documented scenario ID to the union of its action types and evaluates exact-tag target localization on the 29 intervals whose full target lists match CSV columns. The scenario action labels are used **only as post-hoc evaluation strata**; they are not supplied to the detector, attribution method, donor search, or candidate generator. Missed detections and failed counterfactuals count as misses. Five seeds are averaged within attack ID before the descriptive stratum mean, so repeated runs do not inflate the scenario denominator.

On the corrected five-seed set, exact-target-complete attack intervals divide into six action-type strata. PCA-TEMPORAL results are shown below; each group is small, and these are descriptive outcomes without subgroup significance tests.

| Documented manipulated-variable types | Attack IDs | n | Native hit@5 | Grouped-ablation hit@5 | Support-only CF target hit, unconditional | Screened CF target hit, unconditional |
|---|---|---:|---:|---:|---:|---:|
| CV | A103, A203, A214, A216, A218, A219, A303, A401, A508 | 9 | 0.556 | 0.333 | 0.333 | 0.333 |
| CV+PV | A206, A210, A213 | 3 | 0.333 | 0.000 | 0.000 | 0.000 |
| CV+PV+SP | A510, A512 | 2 | 1.000 | 0.500 | 0.000 | 0.000 |
| CV+SP | A301, A307, A501, A509 | 4 | 0.250 | 0.250 | 0.250 | 0.100 |
| PV+SP | A202, A204, A208, A302 | 4 | 0.750 | 1.000 | 0.250 | 0.250 |
| SP | A101, A102, A201, A205, A207, A209, A217 | 7 | 0.000 | 0.000 | 0.000 | 0.000 |

The action strata are derived from scenario IDs, while exact-target eligibility is based on complete exact CSV target names. The generated machine-readable analysis is `results/summaries/hai21_identity_consistent_target_localization_by_action_seed42_46.json`. The small strata and shared attack IDs across seeds do not support generalization or confirmatory inference. The SP-only zero hit rate (7 events) is a falsifiable lead for a larger, independent evaluation, not evidence of a general SP-specific explanation failure.

## Implications for the research design

1. Keep the capability-constrained counterfactual direction provisional. Use action class (SP/PV/CV) as a preregistered threat-model stratum; do not reveal per-event scenario IDs or target points to the explainer in the primary evaluation.
2. Treat exact per-scenario target-restricted recourse, if run, as an **oracle upper bound** only. It leaks target metadata by construction and must never be presented as deployable explanation performance.
3. Retain the one-step normal-data ridge model as a statistical transition screen. It is not independent engineering validation and the action-type target analysis shows no basis for treating score-recourse as recovered attack intent.
4. Before adding numerical setpoint constraints, resolve the tag, unit, and schedule mapping against the release documentation and demonstrate their coverage on every normal training/validation file. Treat the later HAI 23.05 graph and BATADAL's mismatched EPANET file as topology/control documentation only until version-matched dynamics and operating inputs are independently verified. If the current sources cannot supply that evidence, add a dataset with a specified process model or narrow the physical-feasibility claim.

## Reproduction

```powershell
python scripts\evaluate_hai_target_localization_by_action.py `
  results\runs\hai21_seed42_20261031T051335_879603Z `
  results\runs\hai21_seed43_20261031T051709_959445Z `
  results\runs\hai21_seed44_20261031T052039_198005Z `
  results\runs\hai21_seed45_20261031T052411_161777Z `
  results\runs\hai21_seed46_20261031T052743_620105Z `
  --output results\summaries\hai21_identity_consistent_target_localization_by_action_seed42_46.json
```
