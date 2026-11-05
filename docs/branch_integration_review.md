# Review of parallel research branches (5 Nov 2026)

Three other branches in this repository contain independent attempts at the same research brief. This note records what was checked, what was integrated into the manuscript on `claude/gallant-fermat-wfcigs`, and what was not, and why.

## `codex/xai-cps-research-full-20261031` (supersedes `codex/research-xai-cps-20261031`)

An evidence-audit-oriented branch: HAI 21.03, BATADAL, Data4Cyber and TEP pilots with RobustZ/PCA/Isolation-Forest detectors and grouped-ablation explanations, plus a detailed literature/novelty audit. The branch itself states that it is exploratory and not submission-ready. Its claims are carefully hedged; we found no integrity problems.

**Integrated**
- *Independent HAI ground truth.* `configs/hai/independent/` holds its machine-readable transcription of the HAI technical manual attack table (copied unchanged). `scripts/analyze_integration.py` compares it with ours: target loops and points agree for 49/50 attacks. The disagreement (A401) is a table row spanning the page break between printed pp. 37–38 of the manual (SHA-256 `0668345c…a556`); we read the continuation line (AP12, carrying the year of the date) as part of A401, consistent with the two-scenario A4xx series. Excluding A401 changes no reported HR@1 by more than 0.005.
- *Action-type stratification* (SP / CV / +PV spoof), an idea from the branch's process-constraint audit; we derive the types from the target-controller codes of the 21.03 table (all 50 agree with the branch's scenario table) and apply them to our own results (`tab:strata`).
- *Hidden setpoints.* Prompted by the stratification, we checked all 36 (attack, setpoint-tag) targets: 30 are exactly constant during the attack and only 2 vary beyond their pre-attack range.
- *Random-ranking reference* for HR@1/HR@3 (the branch's event-matched null), computed analytically on our units.
- *Related work.* AR-Pro, JoCE and explanation-manipulation attacks (Mia & Pritom), each re-confirmed by web search. Two further works (Cho 2026; Ma et al. 2026) could not be confirmed here and are listed in `docs/citation_provenance.md` for checking.

**Not integrated (and why)**
- *BATADAL*: five day-level, pump-component events with partial labels and hourly sampling; too few and too coarse for loop-level timing claims, and the host was unreachable for verified download from this environment.
- *Data4Cyber*: eight target-eligible events with mixed evidence (branch's own assessment).
- *TEP explanation-interposition / closed-loop spoof pilots*: informative (a large reference-hold spoof demoted the target in 5/5 seeds while keeping the alarm; bounded 0.5/1.0-scale ramps did not), but they use a different detector and explainer, n = 5 seeds, so they say nothing direct about LX. They motivate the adaptive-adversary experiment listed as future work.

## `feat/pi-cax-q1-manuscript`

A complete-looking manuscript ("PI-CAX") with strong headline numbers. **None of its results were integrated**, because:
- All three "benchmarks" are synthetic. `src/cps_benchmarks.py::build_cps_benchmark()` generates data from hand-written equations with SWaT/WADI/HAI tag names, but the abstract and results tables present them as "SWaT, WADI and HIL-HAI" without stating that they are synthetic surrogates. SWaT/WADI are request-only datasets and are not loaded anywhere in the code.
- The "ARM Cortex-A72 edge" latency (6.78 ms) is not measured: `src/evaluation.py` multiplies desktop time by a constant 6.4.
- The method contains hand-set constants (e.g. clipping to [0.10, 1.35], weights 0.20 + 0.95·coherence) tuned on the same synthetic data on which it is evaluated.

Reporting these numbers as results on SWaT/WADI/HAI would be misleading. Its conceptual ideas (physical safety bounds, sensor-vs-actuator breakdown) are covered in our manuscript by the deadline metrics and the action-type stratification.
