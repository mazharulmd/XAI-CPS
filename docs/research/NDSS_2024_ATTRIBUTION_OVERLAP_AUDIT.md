# NDSS 2024 ICS attribution overlap audit

**Audit date:** 1 November 2026  
**Primary source:** Fung, Zeng, and Bauer, *Attributions for ML-based ICS Anomaly Detection: From Theory to Practice*, NDSS 2024, [paper page](https://www.ndss-symposium.org/ndss-paper/attributions-for-ml-based-ics-anomaly-detection-from-theory-to-practice/) and [official full text](https://dev.ndss-symposium.org/wp-content/uploads/2024-216-paper.pdf).  
**Status:** full-text source review; this is not an independent reproduction of their experiments.

## What the paper already establishes

The paper explicitly frames attribution as identifying the feature manipulated in an ICS attack. Its SWaT and WADI annotations provide attack start/end times and manipulated sensor/actuator locations. The real-attack corpus contains 32 SWaT attacks and 15 WADI attacks, with multiple manipulations counted independently for 67 total feature manipulations. The study also adds physically realizable Tennessee Eastman Process manipulations produced through an open-source simulator.

For each detected attack, the practical evaluation finds the first detection point, uses the preceding 50 time steps as the attribution input, ranks features, and computes AvgRank for the ground-truth manipulated feature(s). It compares CNN, GRU, and LSTM anomaly detectors and attribution methods including raw prediction error, saliency, SHAP, and LEMNA. It also analyzes a “best-guess” timing that aligns the input with attack onset, detection timing, manipulation properties, and a weighted attribution ensemble. Their results report that attribution quality changes with timing and attack properties, including low-magnitude and categorical actuator attacks; the paper recommends diverse, complex attacks and warns that practical alarm-time attribution can be inaccurate.

These details are visible in the official paper's problem formulation and dataset description (PDF lines 395–429), practical-timing method (lines 934–942), timing analysis (lines 1022–1148), ensemble analysis (lines 1524–1593), and discussion/recommendations (lines 1832–1875). Line numbers refer to the searchable official PDF text returned by the publisher's PDF service and may change with document rendering.

## Overlap with this repository

The repository's current first-alarm target Hit@K endpoint evaluates whether a ranked list contains a documented manipulated tag/component at the first alarm. Fung et al. already evaluate the same underlying target-attribution question at practical detection time against manipulated-feature ground truth. Our HAI, BATADAL, and Data4Cyber datasets, evidence provenance tiers, threshold-free ranks, detector families, or coverage audits may provide a useful replication and extension, but dataset substitution and a coarser target taxonomy do not by themselves make first-alarm localization novel.

The existing GRIDGUARD-AI audit remains a separate comparison: that artifact's inspected supplement and notebook show no event-level target attribution. However, the absence of target localization in GRIDGUARD-AI does not close the overlap with Fung et al. The novelty assessment must compare both papers.

## Consequence for research direction

1. Do not claim first-alarm target localization, attack-grounded attribution evaluation, or evidence-tier reporting alone as a novel method.
2. Retain current HAI/BATADAL/Data4Cyber results as replication, data-quality, and supporting evidence; report their event counts and annotation limits.
3. A possible evaluation-only contribution needs a distinct validated endpoint, such as calibrated abstention under independently audited target ambiguity, and must demonstrate added practical or statistical value over matched random-ranking and attribution baselines. Current event counts do not yet support that claim.
4. The highest-impact method hypothesis is attacker-capability-constrained counterfactual explanation evaluated against an executable, version-matched process model. It remains conditional: HAI lacks a validated numeric transition model in this repository, the present ridge transition screen is only a statistical proxy, and AR-Pro, JoCE, and physics-aware methods create substantial overlap. No physical-feasibility result is established.

## Reproducibility boundary

This audit is a source-based literature finding, not a reproduction. It makes no claim that the authors' artifact is rerun here, and it does not treat their numerical results as results from this project. The full bibliographic citation is in `paper/references.bib` under `fung2024`.
