# Data4Cyber cross-layer explanation evaluation (protocol draft)

**Status:** full-release integrity audit and exploratory process-only baseline run; not a final benchmark. The 164,148,991-byte Zenodo v3 archive passed its published MD5; all 196 extracted files were written, and all 195 data members listed in the embedded SHA-256 manifest were verified. The split CSVs reproduce 14,354 process rows: 7,586 active and 6,768 benign. The Zenodo record page retains the older 5,880/8,474 class-count summary; `CHANGES.md` explains that S3 was corrected to contain 888 benign and 912 attack rows. The verified archive README/CSV labels are the analysis source of truth. S1 and S1-alt are repeated PV attack runs from one testbed and are grouped as one family. An exploratory family-held-out run of RobustZ/PCA and first-alarm component explanations is documented in `docs/research/DATA4CYBER_PRELIMINARY_RESULTS.md` and `results/summaries/data4cyber_loco_baselines.json`; the released IPAL baselines were not reproduced.

## Schema audit findings and implications

- The time-series CSV schema contains `timestamp`, `attack_active`, `attack_phase`, and `attack_phase_all` alongside telemetry. All four are excluded from detector and explainer input. The split tables reproduce the README's 14,354 rows, 7,586 active rows, 6,768 benign rows, and 7,569 empty feature cells. Missing values remain untouched in the adapter; imputation is fit on outer-fold training-normal data only.
- S1 and S1-alt each provide 3 phase records and 520/516 Modbus records. Both impact annotations identify only `solar-power`; the event point `110.48.rrw.40242` resolves to PV inverter `p_set_percent`, while the endpoint map has no corresponding `dataset_variable`. They support device/register semantics, not direct alignment to an observed process feature.
- S2 has 3 phase records and 698 Modbus events. It names `battery-storage`; its event register resolves to a BSS inverter setpoint but not a direct process telemetry field.
- S3 has 2 intervals and 1,284 rewritten register records. Its impact interval names `192.168.1.61:502 unit 3`; device, unit, and address together resolve the manipulated measurement registers to BSS meter process variables. This is the strongest exact process-target case.
- S4 and S5 have 5/8 phase records and 1,731/1,940 register events. Their phase target is the generic string `mitm`, while the scenario READMEs describe affected meter groups (loads/PV in S4; loads/PV/BSS in S5). Address/unit joins produce multiple candidate telemetry variables unless the scenario-level device context is applied; do not score those as exact register-to-tag labels without a documented disambiguation.
- S6 has two MQTT phases and three event-log publications to `energy/price_signals`. The process table includes price-signal and dispatch context, but the released endpoint/register map does not directly connect that topic to a process column, and the dataset README states that its packet capture contains no MQTT messages. Do not claim packet-level detection for S6.
- There are six primary attack families from one testbed, plus the S1-alt repeat. Group S1/S1-alt in outer evaluation; six family-level units and one site still preclude broad population inference. A held-out scenario is not external validation.
- No complete PCAP feature pipeline or released IPAL detector reproduction has yet been verified. The frozen response-group annotation file is `data/annotations/data4cyber_response_targets.csv`; mappings are separated by evidence tier and indirect-response targets are marked. If exact mapping is sparse, report a tiered audit (exact register-to-tag, documented device-group, coarse asset/phase) with separate denominators; do not pool them into one localization score.
- Across the seven non-benign scenario folders, the implemented adapter identifies 145 numeric telemetry columns after excluding direct labels and clock-like fields. It preserves missing values; the training-normal-only median imputer is implemented and tested. The initial grouped-family leave-one-out process detector run is exploratory. Explanation baselines, repeated-seed sensitivity, edge-device latency, and independent testbed evaluation remain outstanding.

## Candidate research questions

The executable candidate is now scoped to process telemetry: (RQ1) how well do normal-only thresholds and detector scores transfer to a held-out attack family; (RQ2) at the first alarm, how often do process-group explanations recover the documented target separately at exact, device-group, and coarse evidence tiers; and (RQ3) do group-deletion rankings outperform native detector contributions and an analytic random-ranking reference, with detection misses counted as target misses?

Protocol-only and late-fusion experiments remain a future extension. No cross-modal input is currently implemented or evaluated, so this pilot cannot answer whether fusion improves target validity. Phase timing is recorded descriptively; no phase-localization claim is made unless the annotation protocol yields a separate, independently supported phase target.

This is an explanation-validity question, not causal inference, safe action planning, or plant-transition certification. Data4Cyber contains one testbed and a small number of attack scenarios. Treat scenario as the outer statistical unit; 1 Hz rows and multiple phases inside a scenario are not independent replicates.

## Target hypotheses (not results)

- **H1:** Across family units with eligible targets, first-alarm group explanations have higher unconditional Hit@K than a matched random-group reference. This is not estimable confirmatorily with the current six family units.
- **H2:** Group-ablation and native detector-contribution rankings have different target Hit@K; this is two-sided because the pilot provides no justified superiority direction.
- **H3:** A threshold calibrated on benign training scenarios does not necessarily achieve its nominal FPR on a held-out family. Attack labels may not be used to tune thresholds.
- **H4:** Target hit estimates vary with evidence tier and target-set size; tiers and coverage are reported separately, never pooled as root-cause accuracy.

These hypotheses must be revised or withdrawn if inspection finds that target mappings are descriptive only, not tied to each attack phase/event, or if the relevant feature namespaces cannot be aligned without label-derived choices.

## Data, unit, and split

- Pin Zenodo v3 record `10.5281/zenodo.22953576`, its CC BY 4.0 license, archive checksum, README, `CHANGES.md`, and embedded `SHA256SUMS.json`. Keep raw archive and extracted data under ignored `data/raw/`; commit only the acquisition/validation code and provenance manifest.
- Candidate primary unit: the attack scenario, grouping the alternate S1 PV run with the PV Modbus scenario rather than counting it as an independent attack family. Six primary attack scenarios are not enough to support broad population claims.
- Use scenario-grouped leave-one-attack-scenario-out evaluation. Fit detectors and normalization on benign rows from the other scenarios only. Split these training benign rows chronologically into fit and calibration portions; choose each threshold using calibration rows only at predeclared false-positive targets. The held-out scenario's benign and attack periods are test data only.
- Do not expose `attack_active`, `attack_phase`, `attack_phase_all`, event/phase JSON/CSV annotations, packet `label`, or target mappings as predictive or explanatory features. Use these only after predictions and rankings are frozen.
- Audit repeated timestamps, missing/carry-forward flags, endpoint/register schemas, and phase interval inclusivity before constructing windows. Report scenario-specific missingness and alignment quality. Never impute using attack labels or future rows.
- If the companion paper's twelve IPAL detector implementations and preprocessing can be verified and run, reproduce the subset that can be matched exactly. Otherwise, report a faithful, explicit baseline set and do not compare numerically against the paper's published scores as if reproduced.

## Modalities and detector protocol

The implemented pilot uses process telemetry only. Packet/communication features are not yet implemented, and the archive's PCAP capture cannot support S6 MQTT detection because it contains no MQTT packets. Register/endpoint semantics are used only for post-hoc target grouping. A future cross-modal extension must independently define its packet features and time aggregation, exclude packet labels, align timestamp uncertainty, and compare process-only, protocol-only, and late-fusion variants at normal-calibrated operating points. Until that work is complete, this protocol's empirical claims are process-only.

For modality (m), a normal-only detector gives score (s_m(x)). Normalize using fit/calibration benign data only, then combine frozen scores using predeclared equal-weight late fusion:

\[
 z_m(x)=\frac{s_m(x)-\mu_m^{\mathrm{fit}}}{\sigma_m^{\mathrm{fit}}+\epsilon},\qquad
 s_F(x)=\frac{1}{|\mathcal M_x|}\sum_{m\in\mathcal M_x}z_m(x),
\]

where $\mathcal M_x$ contains available modalities and any missing-modality rule is fixed before test evaluation. Calibrate model- and fusion-specific thresholds on normal calibration rows at at least two predeclared operating points. Do not tune on attack labels. The initial detector suite should include the repository PCA/robust-Z baselines plus one released IPAL baseline if reproducible. Add stronger models only after the split and input compatibility are settled.

## Explanation and target alignment

At each detected event's first threshold crossing $t_e$, compute grouped score-deletion attribution. For semantic feature group $g$ and benign reference $r$,

\[
\phi_g(x_{t_e};r)=s(x_{t_e})-s(x_{t_e}^{(-g,r)}).
\]

Group protocol fields/registers and mapped process variables using the published endpoint/register mapping. For an attack event with documented target set $T_e$, compute unconditional target Hit@K, counting detector misses as misses and unresolved mappings according to a separately declared eligibility table. Keep exact register hit, mapped process-variable hit, and coarser component hit as distinct outcomes. For phase alignment, compare the timestamps of top-ranked evidence groups with the independently annotated phase interval; do not call temporal order causal. Report the fraction of first alarms occurring in each phase and the phase-alignment distribution separately from target Hit@K.

Counterfactual deletion fidelity, stability under representation-preserving perturbations, and latency are secondary explanation properties. Feature deletion must respect protocol types and process representation; use random and bottom-ranked matched group baselines. No single composite explanation score is proposed.

## Statistical analysis and ablations

- Report every held-out attack scenario, its phase/event count, target coverage, detection misses, target Hit@K, alarm delay, and normal false-alarm burden. Pooled event/row estimates are secondary.
- With only six primary scenarios, uncertainty and significance are low-powered. Average repeated fits within attack event, summarize scenario-level paired differences, and show every scenario. Use a scenario-cluster bootstrap only as descriptive sensitivity analysis; do not treat thousands of rows or windows as independent (n). Any exact paired test is exploratory and must disclose its resolution and multiplicity family.
- In the current process-only pilot, report the analytic uniform random-ranking hit probability conditional on group-universe and target-set sizes. Because coarse target sets can make random top-5 hit probability near one, interpret Hit@K alongside that probability. A future matched random-group permutation null may preserve feature-group frequencies, but it has not been run here. Protocol-only/fusion ablations and phase timing remain future work.
- Add input corruption, missing-modality, and timestamp-alignment checks only after actual alignment/missingness semantics are confirmed. These are not adversarial robustness claims.

## Stop/go criteria before implementation

Before the full manuscript experiment, verify: (1) phase interval and event identity handling in the complete adapter; (2) per-family target eligibility using the frozen evidence tiers; (3) temporal alignment between process rows and event/packet evidence; (4) no direct labels or post-hoc fields in model inputs; (5) adequate train/calibration/test roles under every family holdout; and (6) reproducible comparison with appropriate released IPAL baselines. If exact mapping remains limited, keep distinct denominators and downgrade claims to the evidence levels actually supported. Do not claim causal linkage or physical feasibility.
