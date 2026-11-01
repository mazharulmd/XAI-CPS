# Data4Cyber v3 Reproduction Audit

**Date:** 1 November 2026
**Purpose:** Verify the repository's family-held-out Data4Cyber v3 pilot from the released archive and compare it with the previously saved result.

## Data acquisition and integrity

- Source: [Data4Cyber v3, Zenodo record 22953576](https://doi.org/10.5281/zenodo.22953576), release license CC BY 4.0.
- Archive: `data4cyber_dataset.zip`, 164,148,991 bytes; expected MD5 `20d81c975e5c37f453103cdadb052c9e`.
- The repository downloader verified the archive MD5, rejected unsafe ZIP paths and non-regular members, extracted 196 files, and verified the dataset's embedded SHA-256 manifest.
- `scripts/audit_data4cyber_metadata.py` and `scripts/audit_data4cyber_targets.py` completed against the extracted v3 tree. The metadata audit reports 14,354 rows, 7,586 attack-active rows, 6,768 benign rows, and the previously documented disagreement between Zenodo landing-page counts and the archive's corrected README/change log. See `results/data4cyber_metadata_audit.json` and `results/data4cyber_target_coverage.json`.
- Raw archive and extracted records remain under ignored `data/raw/`; they are not committed.

## Reproduction

The full family-held-out process-only run was executed with the checked-in configuration and code:

```powershell
python scripts\download_data4cyber.py --workers 8
python scripts\audit_data4cyber_metadata.py data\raw\data4cyber\data4cyber_dataset\data4cyber_dataset --output results\data4cyber_metadata_audit.json
python scripts\audit_data4cyber_targets.py data\raw\data4cyber\data4cyber_dataset\data4cyber_dataset --output results\data4cyber_target_coverage.json
python scripts\run_data4cyber_baseline.py --compact-output --output results\summaries\data4cyber_loco_baselines_reproduction_20261101.json
```

The run evaluated six held-out attack families (S1 and S1-alt grouped), three normal-only detector baselines, and two nominal validation false-positive targets. It emitted 42 scenario/model/threshold records and 48 target-eligible first-alarm explanation records. The compact report is `results/summaries/data4cyber_loco_baselines_reproduction_20261101.json`.

## Comparison with the saved pilot

The new run was compared with `results/summaries/data4cyber_loco_baselines.json` after aligning each record by held-out family, scenario, model, and operating point.

- All 42 records matched on point-level detection outcomes and metrics, including average precision, precision, recall, F1, specificity, false-positive rate, and confusion counts.
- The 48 first-alarm target rankings matched on target sets, alarm status and time, top-five groups, Hit@1/3/5, random-ranking references, and stability values. The only non-identical explanation value was the top-group score reduction, with maximum absolute difference below `1e-11`, consistent with floating-point round-off.
- Validation thresholds differed by at most `1.6e-11`; the family-level paired comparison statistics and adjusted p-values were unchanged.
- Wall-clock and latency fields are hardware/run dependent and are not expected to match. The reproduction completed in 251.75 seconds on the recorded host.

This establishes computational reproducibility for the checked-in pipeline on the same Data4Cyber v3 release; it is not an independent-site replication or evidence of generalization beyond the six benchmark families. Target coverage remains sparse and mixed-resolution, and detector threshold performance varies sharply by held-out family. The results remain exploratory.

## Runtime

- Python 3.12.10
- NumPy 2.5.4
- pandas 3.1.0
- scikit-learn 1.9.1
- Windows 11, x86-64; Intel Core family 6/model 154 host

See the compact JSON for the full runtime metadata, model configuration, per-family results, target rankings, and paired tests. No test-set labels enter fitting, imputation, or threshold selection; attack and target annotations are joined for evaluation after each fold's model and threshold are frozen.
