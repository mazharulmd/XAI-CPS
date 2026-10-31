# XAI-CPS: explainable intrusion detection for cyber-physical systems

This repository contains the supplied literature corpus, a pilot HAI 21.03 detector/explanation evaluation, and an evolving research framework for CPS/ICS intrusion detection.

## Research direction

The strongest current candidate is an **evidence-provenance- and resolution-aware first-alarm explanation-to-target evaluation**, with HAI 21.03 as the larger exact-tag anchor and Data4Cyber v3 as a distinct-testbed replication. HAI has 29 fully mapped target intervals after strict manual-to-column matching; Data4Cyber contributes eight scenario folders and 14,354 rows but only a handful of target-eligible attack events, with evidence ranging from one register-to-telemetry join to coarse device groups. Keep dataset metrics separate, count detector misses as localization misses, and report coverage/abstention rather than pooling exact and coarse targets. The Data4Cyber process-only pilot shows cross-family threshold-transfer problems and tiny target-event counts, so its findings remain exploratory. BATADAL has only five approximate component events. Physical-transition-constrained counterfactuals remain conditional on acquiring a version-matched, independently validated process model. No direction has a novelty certification; the existing counterfactual is a diagnostic score-flipping prototype. A close 2026 preprint, GRIDGUARD-AI, also uses Data4Cyber with scenario-held-out explainable assurance, so any Data4Cyber contribution remains narrowly about separately audited first-alarm target validity; the primary article is reviewed; code/supplement comparison remains pending. See [`docs/research/EVIDENCE_AWARE_STUDY_DESIGN.md`](docs/research/EVIDENCE_AWARE_STUDY_DESIGN.md), [`docs/research/DATA4CYBER_SCHEMA_AUDIT.md`](docs/research/DATA4CYBER_SCHEMA_AUDIT.md), [`docs/research/DATA4CYBER_PRELIMINARY_RESULTS.md`](docs/research/DATA4CYBER_PRELIMINARY_RESULTS.md), [`docs/research/DATA4CYBER_EVALUATION_PROTOCOL_DRAFT.md`](docs/research/DATA4CYBER_EVALUATION_PROTOCOL_DRAFT.md), [`docs/research/DATA4CYBER_STUDY_SPEC.md`](docs/research/DATA4CYBER_STUDY_SPEC.md), [`docs/research/RESEARCH_DIRECTION_ASSESSMENT.md`](docs/research/RESEARCH_DIRECTION_ASSESSMENT.md), [`docs/research/LITERATURE_REVIEW.md`](docs/research/LITERATURE_REVIEW.md), [`docs/research/EXPERIMENT_PROTOCOL.md`](docs/research/EXPERIMENT_PROTOCOL.md), and [`docs/research/INITIAL_RESULTS.md`](docs/research/INITIAL_RESULTS.md).

## Repository contents

- `XAI_IDS_CPS_Papers_Final.xlsx`, `Research Papers/`: source corpus supplied with the project.
- `docs/research/`: initial gap assessment, experiment protocol, pilot results, literature inventory, and a HAI process-constraint evidence audit.
- `data/`: local dataset location only; raw datasets are excluded from version control.
- `configs/`: experiment configurations, to be pinned before evaluation.
- `src/xai_cps/`: HAI/BATADAL loaders, a label-separated Data4Cyber schema adapter with training-normal-only median imputation, RobustZ/PCA/Isolation Forest and constrained GeCo-AR1 detectors, grouped ablation, a capability/support-constrained recourse prototype, and event metrics.
- `paper/`: LaTeX manuscript and verified bibliography as the study matures.
- `results/`: machine-readable exploratory runs and reproducible summary aggregations.

## Current status

The current HAI evaluation uses identity-consistent temporal ablation and counterfactual edits. It includes five-seed support-only and one-step-screen comparisons, exact target analysis for the 29 scenarios with verifiable CSV target tags, a post-hoc SP/PV/CV action-type stratification, a crossed perturbation/ablation stability audit, and deletion-based detector-score fidelity. The factorial audit shows that the observed stability effect depends on both input perturbation and ablation semantics. The learned transition screen remains a statistical filter, not a validated plant model or safety certificate; unconditional target localization did not improve. A new BATADAL audit maps its five approximate labeled intervals to FAQ-documented pump components and evaluates first-alarm component localization on seeds 42–46. BATADAL still has unknown labels, so precision/F1/test FPR are not reported. Per-file variability and false-alarm burden remain concerns. Current analyses are in `docs/research/INITIAL_RESULTS.md`, `docs/research/EXPERIMENT_PROTOCOL.md`, `docs/research/PROCESS_CONSTRAINT_EVIDENCE_AUDIT.md`, and `docs/research/BATADAL_ATTACK_TARGET_AUDIT.md`, with machine-readable results under `results/summaries/`.

The HAI/HAIEnd 23.05 repository contains a later-release boiler topology graph. It may support a future topology-aware study, but it has not been mapped to HAI 21.03 and contains no independently validated numeric plant dynamics or safety invariants; it is not used to claim physical feasibility in the current experiments.

## Data handling

Do not commit dataset files. Put authorized local copies under `data/raw/` or set a local path in a config; retain the exact upstream release and checksums in experiment metadata. Follow dataset-specific terms and citations. HAI dataset versions differ in channels and labels; never combine versions silently.

## Reproducibility principles

- Split by chronology and complete attack identity; do not randomly split overlapping windows.
- Fit normalization, thresholds, feature groups, and process-feasibility models on training/validation data only.
- Report attack detection, root/process localization, explanation fidelity, plausibility, stability, latency, and overhead as distinct outcomes.
- Keep detector inference and explanation-generation time separate; explanations must not delay the detection path.
- Save exact configuration, software versions, random seeds, data release/checksums, hardware, and per-attack predictions with each run.
- Do not report experiments before execution or present a plausible explanation as a causal finding or safe action.

## Setup and execution

Use Python 3.11 or later in a clean environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts\download_hai_21_03.py
python scripts\run_hai.py --seed 42
python scripts\download_batadal.py
python scripts\run_batadal.py --seed 42
```

Data4Cyber v3 is a conditional new benchmark candidate. Its full archive is about 164 MB; downloading it is optional and is not required for reproducing the existing HAI/BATADAL pilot. To obtain and verify the release, use `python scripts\download_data4cyber.py`. The helper checks the Zenodo archive size/MD5, rejects unsafe ZIP paths, and verifies extracted members against the embedded SHA-256 manifest. Then audit the extracted schema with `python scripts\audit_data4cyber_metadata.py data\raw\data4cyber\data4cyber_dataset\data4cyber_dataset --output results\data4cyber_metadata_audit.json` and targets with `python scripts\audit_data4cyber_targets.py data\raw\data4cyber\data4cyber_dataset\data4cyber_dataset --output results\data4cyber_target_coverage.json`. The family-held-out baseline runner is `python scripts\run_data4cyber_baseline.py --output results\summaries\data4cyber_loco_baselines.json`; it uses the default data path in `configs/data4cyber_v3.json`. For the selective local audit subset, pass `--data-dir data/raw/data4cyber/audit_extract`. The archive and extracted data remain under ignored `data/raw/` and must not be committed. The initial baseline is exploratory; see [`docs/research/DATA4CYBER_PRELIMINARY_RESULTS.md`](docs/research/DATA4CYBER_PRELIMINARY_RESULTS.md).

The Data4Cyber family-held-out runner includes a 500-tree normal-only Isolation Forest alongside PCA residual and RobustZ. It uses robust-scaled process levels and first differences, and reports first-alarm target ranks from grouped black-box ablation; the forest has no native attribution. Reproduce the compact report with `python -m pip install -e ".[baselines]"` followed by `python scripts/run_data4cyber_baseline.py --data-dir data/raw/data4cyber/full_v3/data4cyber_dataset --compact-output --output results/summaries/data4cyber_loco_baselines.json`. The report retains per-file metrics, paired family-level statistics, and every target-eligible first-alarm ranking. Omit `--compact-output` to retain reconnaissance and all intermediate per-event score details. All folds group S1 with S1-alt and keep the held-out family's normal and attack rows out of fitting and calibration.

The separate constrained GeCo-AR1 reference can be reproduced with `python scripts/run_data4cyber_geco_reference.py --output results/summaries/data4cyber_geco_ar1_reference.json`. It uses the same Data4Cyber archive path and split manifest as the PCA/RobustZ runner. This model is restricted to one-step self-lag residuals, imputes from training-normal data, and calibrates only on normal holdout data; it is not full GeCo and does not reproduce the twelve published IPAL baselines. See [`docs/research/DATA4CYBER_IPAL_BASELINE_COMPATIBILITY.md`](docs/research/DATA4CYBER_IPAL_BASELINE_COMPATIBILITY.md) for the exact protocol and caveats.

The HAI Isolation Forest target-localization reference requires the optional baseline dependency (`python -m pip install -e ".[baselines]"`) and can be reproduced with `python scripts/run_hai_isolation_forest_reference.py --seeds 42 43 44 45 46`. It fits only the normal training files, selects its threshold on the normal validation file, and evaluates grouped-ablation rankings against the 29 exact-target attack intervals. This is a detector-family comparator, not a reproduction of RAID's regime-aware feature pipeline or language/RAG stages; see `results/summaries/hai21_isolation_forest_target_reference.json` after the run completes.

After generating five HAI runs, aggregate detection and explanation outcomes and reproduce localization, stability, and fidelity analyses with the same run directories:

```powershell
python scripts\summarize_explanation_runs.py <run-42> <run-43> <run-44> <run-45> <run-46> `
  --output results\summaries\hai21_identity_consistent_seed42_46.json
python scripts\evaluate_hai_target_localization.py <run-42> <run-43> <run-44> <run-45> <run-46> `
  --randomization-draws 20000 --randomization-seed 161803 `
  --output results\summaries\hai21_identity_consistent_target_localization_seed42_46.json
python scripts\evaluate_hai_explanation_stability.py <run-42> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results\summaries\hai21_stability_factorial_seed42.json
python scripts\evaluate_hai_explanation_stability.py <run-43> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results\summaries\hai21_stability_factorial_seed43.json
python scripts\evaluate_hai_explanation_stability.py <run-44> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results\summaries\hai21_stability_factorial_seed44.json
python scripts\evaluate_hai_explanation_stability.py <run-45> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results\summaries\hai21_stability_factorial_seed45.json
python scripts\evaluate_hai_explanation_stability.py <run-46> --sigmas 0.02 0.05 --perturbations 5 --bootstrap-draws 100 --output results\summaries\hai21_stability_factorial_seed46.json
python scripts\aggregate_hai_explanation_stability.py results\summaries\hai21_stability_factorial_seed42.json results\summaries\hai21_stability_factorial_seed43.json results\summaries\hai21_stability_factorial_seed44.json results\summaries\hai21_stability_factorial_seed45.json results\summaries\hai21_stability_factorial_seed46.json --bootstrap-draws 5000 --output results\summaries\hai21_explanation_stability_factorial_seed42_46.json
python scripts\evaluate_hai_explanation_fidelity.py `
  <run-42> <run-43> <run-44> <run-45> <run-46> `
  --random-sets 50 --bootstrap-draws 5000 `
  --output results\summaries\hai21_identity_consistent_fidelity_seed42_46.json
python scripts\evaluate_hai_target_localization_by_action.py `
  <run-42> <run-43> <run-44> <run-45> <run-46> `
  --output results\summaries\hai21_identity_consistent_target_localization_by_action_seed42_46.json
```

Recompute threshold sensitivity across the five seed-42–46 runs (normal train3 quantiles; saved test scores):

```powershell
$env:PYTHONPATH = "src"
python scripts\aggregate_hai_threshold_sensitivity.py `
  results\summaries\hai21_threshold_sensitivity_seed42.json `
  results\summaries\hai21_threshold_sensitivity_seed43.json `
  results\summaries\hai21_threshold_sensitivity_seed44.json `
  results\summaries\hai21_threshold_sensitivity_seed45.json `
  results\summaries\hai21_threshold_sensitivity_seed46.json `
  --output results\summaries\hai21_threshold_sensitivity_seed42_46.json
```

The vectorized grouped-ablation scorer was checked against an identity-preserving loop baseline on 20 events per PCA model and all nine sampled RobustZ events; it preserved attribution values to within $5\times10^{-13}$ and reduced median grouped-ablation latency by about 10.5–10.9×. This microbenchmark covers grouped ablation alone, not end-to-end explanation generation.

The commands read `configs/hai_21_03.json` or `configs/batadal.json` and write timestamped runs under `results/runs/`. The checked-in HAI fidelity and threshold-sensitivity aggregates are losslessly gzip-compressed (.json.gz) to keep the research branch compact; reproduction commands emit plain JSON. Raw datasets and generated per-sample predictions are ignored by Git; do not commit them. See the protocol and `docs/research/BATADAL_ATTACK_TARGET_AUDIT.md` before interpreting BATADAL's partial-label and pump-target metrics. The five BATADAL target-audit runs recorded Python 3.12.14, NumPy 2.3.5, and Pandas 3.0.1; a clean-environment run on another host remains to be checked.

