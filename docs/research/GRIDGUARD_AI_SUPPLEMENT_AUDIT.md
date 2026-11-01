# GRIDGUARD-AI supplementary artifact audit

**Audit date:** 1 November 2026
**Purpose:** verify the closest Data4Cyber/XAI overlap from the authors' primary supplementary archive.
**Status:** Zenodo record and bundled notebook inspected; not an independent reproduction of GRIDGUARD-AI.

## Provenance and integrity

The official [Zenodo record 23512250](https://doi.org/10.5281/zenodo.23512250) identifies Stephen Kamran and Farah Saman as creators and describes the archive as supplementary material for their Research Square manuscript, *GRIDGUARD-AI: Scenario-Held-Out Cyberattack Detection with Explainable Assurance and Bounded Agentic Investigation*. The 4,250,625-byte archive contains an author-version manuscript, an executed Kaggle notebook, supplement S1, a metrics/scope CSV, two figures, upstream dataset documentation, and integrity metadata. The Zenodo API's MD5 (`287adabe200d634eda34f789c61d5f5d`) matches the downloaded archive. All 13 paths in the internal SHA-256 manifest match their files. The archive is CC BY 4.0; the underlying Data4Cyber data remains governed by its separate upstream terms and is not redistributed here.

The archive's own README says the notebook has 99 code cells and outputs for 58 cells, is not a complete clean rerun, and depends on Kaggle data/workspace state. It says no independent end-to-end replay was performed and that the routing thresholds were refined after inspection of held-out results. These are author-reported limitations, corroborated by the notebook's saved code/output and the supplementary scope table; they preclude treating the notebook as an independently reproduced or untouched test-set evaluation.

## Explanation-specific evidence

Notebook cell 83 (“RF SHAP Explanations + HITL Audit”) reconstructs a single scenario-held-out Random Forest model for global index 248, checks that the reconstructed probability equals the saved result, and emits a top-10 TreeSHAP table. The saved table labels its predictors `feature_139`, `feature_136`, and so on; it does not map them to named telemetry tags. The code explains the RF baseline only, not the TCN or the generative agent. The saved audit reports human review as `PENDING` and grid control action as `NONE`. The supplementary `metrics_and_scope.csv` independently summarizes this as “RF TreeSHAP, 1 reconstructed case” and explicitly says it is not general XAI verification.

The inspected artifact reports no evaluation of explanation rankings against event-specific, independently documented register, telemetry-tag, or component targets; no explanation Hit@K denominator; and no target-mapping coverage audit. This is a bounded finding about the inspected version 1.0.0 artifact, not a claim that no later or private materials exist.

## Relationship to this project

GRIDGUARD-AI already occupies Data4Cyber scenario-held-out detection, temporal windows, RF/TCN/autoencoder comparisons, deterministic assurance escalation, and agent-assisted investigation. Those elements cannot be claimed as new here. Its one-case, anonymous-feature RF TreeSHAP illustration does not establish event-level agreement with documented attack targets. The project's possible distinction remains a leakage-controlled, first-alarm explanation-to-target audit with evidence-tier-specific denominators, explicit abstention for unresolved mappings, and separate measurement of detector-score faithfulness. That distinction is a **candidate gap**, not a certified novelty claim; it still requires a fair comparison to the released baselines and independent target evidence.

The GRIDGUARD-AI archive says it evaluates 14,202 overlapping windows across eight operational scenarios from Data4Cyber v2 (Zenodo DOI 10.5281/zenodo.20004612). Our primary Data4Cyber analysis uses the corrected v3 archive and does not pool its rows, labels, or metrics with the earlier study. Both studies have small numbers of independent scenario/family units despite large window counts.

## Reproduction pointer

Download the cited Zenodo archive and verify its MD5 before extracting it. The local research copy used for this audit is intentionally stored under ignored `data/raw/`; it is not committed. See `LITERATURE_REVIEW.md` and the manuscript for the source citation and scoped overlap statement.
