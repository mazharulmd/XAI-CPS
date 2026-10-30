# Literature review and research-gap analysis — XAI for Intrusion Detection in CPS

Scope: the 30 MAIN + 3 EXTRA papers listed in `XAI_IDS_CPS_Papers_Final.xlsx`.
Of these, 28 PDFs are in `Research Papers/` (`1.pdf`…`31.pdf`, numbering = spreadsheet `#`; `31.pdf` = E1, Skrodelis & Romanovs).
**Not available locally:** #24 (Alani 2023), #28 (Arreche et al., E-XAI, 2024), #29 (Wei et al., xNIDS, 2023); for these only the spreadsheet abstract/notes were used.
E2 (Al Rawajbeh 2025) and E3 (Neupane 2022) are also not in the folder.

Evidence labels used below: **[PDF]** = checked in the full text; **[XLSX]** = from the curated spreadsheet only; **[WEB]** = web search result summary (abstract level, not full text).

---

## 1. Per-paper matrix (what is explained, how it is evaluated)

| # | Paper | Detector | XAI | Data (CPS?) | Explanation evaluated against ground truth? | Explanation cost reported? | Validation protocol (as stated) |
|---|---|---|---|---|---|---|---|
| 1 | Deng & Hooi, GDN, AAAI'21 | GNN forecaster | attention / per-sensor deviation | SWaT, WADI | qualitative case study [PDF] | no | train normal / test attack files |
| 2 | Huong et al., FedeX, Access'22 | VAE+SVDD, FL | Kernel SHAP | SCADA liquid storage, SWaT | no | training/edge cost; recommends periodic SHAP [XLSX] | — |
| 3 | Tang et al., GRN, C&S'23 | GNN+GRU | attention | SWaT, WADI | qualitative root-cause | no | — |
| 4 | Ha et al., IFAC'22 | LSTM-AE+OCSVM | Gradient SHAP | gas pipeline | no (visual) | no | **80/20 split** [PDF] |
| 5 | Yazdinejad et al., TCSS'25 | FL + DP + PHE | SHAP | SWaT, gas pipeline, UNSW-NB15 | no | crypto latency only | **70/30 split** [PDF] |
| 6 | Birihanu & Lendák, Frontiers'24 | correlation LSTM-AE | SHAP | SWaT, HAI, Modbus | partial | memory/exec time per window [PDF] | **80/20 split** [PDF] |
| 7 | Abudurexiti et al., C&S'25 | CNN-attn + TCN/KAN-VAE | SHAP | SWaT, SKAB, … | no (plots) | no | — |
| 8 | **Fung, Zeng, Bauer, NDSS'24** | AR, PASAD, CNN, GRU, LSTM | SM, SG, IG, EG, LIME, SHAP, LEMNA, CF + ensemble | SWaT, WADI, TEP (synthetic) | **yes — rank of manipulated feature (AvgRank), 156 manipulations** [PDF] | **no** | detector trained on normal data |
| 9 | Trivedi et al., Energy Rep.'25 | ML/DL classifiers | SHAP | simulated CIGRE microgrid | no | no | LOOCV / CV [PDF] |
| 10 | Mathuros et al., WaXAI, CPSS'24 | ECOD, DeepSVDD | Kernel SHAP, LIME, ALE, IG | SWaT | **yes — IoU@5/10 vs attacked devices, 100 sampled records** [PDF] | yes (XAI compute time) | — |
| 11 | Eltomy & Lalouani, ICPS'24 | MHA + AE | attention | water ICS network traffic | no | no | — |
| 12 | Elgarhy et al., IoT-J'25 | FL false-data detector | XAI used for defence | smart-grid consumption | n/a (robustness of XAI-FL) | no | — |
| 13 | Xu et al., ALRL, TII'24 | distilled logic NN | intrinsic rules | SWaT, WADI | rule quality | no | **five-fold CV, 80 %** [PDF] |
| 14 | **Franco de la Peña et al., ShaTS, FGCS'26** | supervised BiLSTM | a-priori grouped Shapley (temporal / sensor / 6 SWaT stages), exact or sampled (m = 20·\|G\|) | SWaT | process/sensor identification (case-based) | **yes — time per window vs 1 s, RTX 4080** [PDF/XLSX] | padded segment split to avoid leakage [PDF] |
| 15 | Kenmogne & Mocanu, NetSoft'24 | ML detector | XAI | process-oriented attacks | — | — | — |
| 16 | Rasheed et al., XAI-CPS, ComNet'26 | TAC-LSTM-GNN | domain-aware LRP | SWaT, WADI, BATADAL, IIoT | **expert agreement 89.3 %** | 12.3 ms (A100 GPU) [PDF] | **5-fold CV, 70/15/15** [PDF] |
| 17 | **Oinonen & Morsi, EPSR'26** | CNN | 7 XAI incl. SHAP, LIME, Grad-CAM, RISE | IEC 61850 substation | sparsity, TV, stability (perturbation MSE), deletion AUC, fairness [PDF] | **yes (SHAP ≈ 8 s, LIME ≈ 7 s, gradients 0.05 s)** [XLSX] | — |
| 18 | Panesar et al., TICPS'26 | CNN on scalograms | MMD-Critic (examples) | OPAL-RT + SIPROTEC (HIL) | no | real-time HIL | **70/30 split** [PDF] |
| 19 | Kenmogne & Mocanu, IntelliSecAI'25 | ML | SHAP, LIME | simulated process | qualitative | — | — |
| 20 | Oseni et al., T-ITS'23 | DL | SHAP | ToN_IoT (network) | no | — | — |
| 21 | Khan et al., IoT-J'22 | CNN/RNN AE | intrinsic | IIoT | no | — | — |
| 22 | Zolanvari et al., TRUST, IoT-J'23 | any | TRUST (statistical surrogate) | IIoT (3 sets) | surrogate success rate | "a few ms" [PDF] | — |
| 23 | Shoukat et al., C&S'25 | LSTM-AE + AGRU | SHAP | N-BaIoT, Edge-IIoTset, CIC-IDS2017 | no | — | **70/30 split** [PDF] |
| 24 | Alani, CEE'23 | ML | SHAP | WUSTL-IIoT | no | detection time only [XLSX] | 99.97 % acc. — split unclear [XLSX] |
| 25 | Attique et al., IoT-J'24 | BiLSTM-attn | SHAP | CICIDS2017, X-IIoTID | no | — | **70/30 split** [PDF] |
| 26 | Ahakonye et al., IoT-M'24 | several ML | LIME, SHAP | IIoT | decision/confidence impact ratios | model compute time | — |
| 27 | Saiyed et al., E-SHIELD, IEEE Netw.'26 | CNN-LSTM | SHAP + dashboard | WUSTL-IIoT, CICIoT2023 | no | inference time | **stratified 5-fold CV** [PDF] |
| 28 | Arreche et al., E-XAI, Access'24 | 7 models | SHAP, LIME | network IDS | 6 metrics incl. stability, efficiency | yes (hours) [XLSX] | — |
| 29 | Wei et al., xNIDS, USENIX Sec'23 | DL NIDS | xNIDS | network IDS | fidelity, sparsity, completeness, stability [XLSX] | — | — |
| 30 | Ericson et al., WFCS'25 | CNN/LSTM/GCN (TFLite) | SHAP | UNSW-NB15 (network) on RPi 4 | no | per-prediction ms on RPi 4 [XLSX] | — |
| E1 | Skrodelis & Romanovs, review'26 | — | — | 40 studies | — | latency numbers borrowed [PDF] | — |

(“—” = not stated in the parts we read; it is **not** a claim that the paper lacks it.)

## 2. Cross-cutting findings

**F1 — Explanations are rarely checked against ground truth.** Most of the corpus shows SHAP/LIME/attention plots and argues that they are plausible. Quantitative localization against the documented attack target appears only in Fung et al. (#8), WaXAI (#10) and, through expert agreement, XAI-CPS (#16). In ICS the target is the manipulated sensor or actuator, or the compromised control loop, and public datasets document it.

**F2 — Attributions computed at the alarm instant are unreliable (#8).** Fung et al. show this across SWaT, WADI and TEP. Detection points fall either immediately after the attack start or well after it, and attributions computed at those "practical" timings are inaccurate. ML attributions improve when computed at the right time, while raw-error ranking is best early on. Fung et al. explicitly suggest separate designs for sensors and actuators and for their interdependencies, and they suggest that attribution may belong in *non-real-time* workflows. **No follow-up proposes a method that makes attribution reliable at operational timing.**

**F3 — Explanation cost is measured, but never against the physical process.** ShaTS (#14), Oinonen (#17), WaXAI (#10), Ericson (#30), E2 and Arreche (#28) report explanation time in seconds or milliseconds. Each compares it to a fixed budget (e.g. 1 s per window) or to nothing at all. None relates the time *to a correct localization* (detection delay + compute + attribution accuracy) to how long the process can tolerate the attack before an unsafe state. Process-safety engineering formalises that interval as the *process safety time*.

**F4 — Grouping helps, but the grouping is not the operator's unit.** ShaTS groups by SWaT stage (6 groups) or by sensor (with sampling). Owen/Partition SHAP groups by data-driven clusters. HAI attacks, and operators' mental models, are organised by **process control loop** (SP–PV–CV of one PID). Loop grouping also directly encodes the sensor–actuator coupling that Fung et al. identify as the failure cause. With ≈10 loops, exact Shapley values become affordable and deterministic.

**F5 — Validation protocols often permit temporal leakage.** At least eight papers report random proportional splits or k-fold CV on time-series data without stating that the folds are chronological or gapped (#4, #5, #6, #13, #16, #18, #23, #25, #27). Explanations are then computed for models whose detection quality may be inflated. ShaTS (#14) is a positive exception, using padded segments.

**F6 — Explanation stability is reported per input perturbation, not over time.** Oinonen (#17) and E-XAI (#28) measure stability under input noise. An operator watching an incident instead sees a *sequence* of explanations, one per second. Whether the top-k list flickers between consecutive alarm windows (temporal stability) is not measured in the corpus.

**F7 — Robustness of ICS explanations to an explanation-aware adversary is open.** Elgarhy et al. (#12) study adversarial robustness of XAI-based FL for smart-grid data, and Fung et al. (#8, App. D) note that multi-point or stealthier manipulations degrade attributions. No study measures *blame shifting* — an attacker adding a small decoy manipulation on an innocent loop — on a closed-loop process.

**F8 — Evidence on CPS datasets with documented targets is thin.** Most papers use SWaT/WADI, which require a data request, or network-traffic datasets that are not process-level (#20, #23–#25, #27, #30). HAI 21.03 is public and documents 50 attacks with target controller and points. In our search it was not used for quantitative attribution evaluation; it appears only as an extra detection dataset in #6.

## 3. External novelty check (beyond the spreadsheet)

| Query | Result | Consequence |
|---|---|---|
| Temporal aggregation of attributions / RCA in time-series AD [WEB] | arXiv 2604.17616 (Apr 2026, preprint) — "Conditional Attribution for Root Cause Analysis in Time-Series Anomaly Detection": sensor-time attributions with on-manifold counterfactuals, SWaT and MSDS. | Must be cited and distinguished. It aggregates *within* a window and uses no loop structure, deadlines or latency; we accumulate *across* alarm windows with a stopping rule. Full text not accessible here (arXiv blocked), so it is compared at abstract level only. |
| Explanation latency vs. safety deadline [WEB, Consensus] | Network-IDS papers report explanation latency of tens of ms (e.g. Alnsour 2026; Prabhu 2026). GiBy (arXiv 2504.20906) explains via per-sensor safety bounds within ~1 ms on SWaT. Dash et al. 2026 decompose *detection* latency. Puccetti et al. 2025 relate FPR to detection latency. | Timeliness of *detection* is studied. Timeliness of a *correct explanation* relative to a process deadline is not, in the results we found. |
| Owen value / hierarchical Shapley [WEB] | SHAP PartitionExplainer (Owen values over a clustering tree); O-Shap (arXiv 2602.17107) shows group definition is critical. | Hierarchical Shapley itself is **not** novel. Our claim is restricted to loop-structured exact attribution + temporal accumulation + deadline-conditioned evaluation in ICS. |
| HAI attribution [WEB] | No attribution benchmark on HAI found. | Our HAI 21.03 loop-level ground-truth table is a contribution, with documented name corrections. |

Limitations of this check: OpenAlex, Crossref, arXiv and Semantic Scholar APIs were blocked from this environment, and Consensus quota was exhausted after one query. The check therefore relied on general web search. It shows that we did not *find* the contribution; it does not prove absence. **Before submission, re-run the check in Scopus/WoS** with the queries in `docs/novelty_queries.md`.

## 4. Consolidated research gaps

- **G1 (timeliness).** No metric or method targets *a correct localization delivered before the process deadline*. This combines detection delay, explanation compute and attribution correctness over time (F2 + F3).
- **G2 (structure).** Attribution units do not match the ICS control structure, which hurts accuracy at detection time and makes cost scale with the number of tags (F2 + F4).
- **G3 (temporal reliability).** Explanations are single-shot. Evidence that accrues over consecutive alarm windows is ignored, and temporal stability is not measured (F2 + F6).
- **G4 (rigour).** Leakage-prone validation; few quantitative ground-truth evaluations; no statistical testing across attacks (F1 + F5 + F8).
- **G5 (adversarial).** Robustness of localization to decoy or blame-shifting manipulations on closed-loop processes is unmeasured (F7).
