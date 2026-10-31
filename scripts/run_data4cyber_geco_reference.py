from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from xai_cps.data import contiguous_events
from xai_cps.data4cyber import (
    fit_normal_median_imputer,
    load_data4cyber_benign_file,
    load_data4cyber_scenario,
)
from xai_cps.geco_reference import GeCoAR1Reference
from xai_cps.metrics import binary_metrics, event_metrics
from xai_cps.explainers import channel_groups
from scripts.run_data4cyber_baseline import (
    combine_test,
    exact_sign_flip_p,
    holm,
    read_benign,
    split_normal_rows,
    threshold_at_fpr,
    verify_split_inputs,
)


def first_alarm_events(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    channel_scores: np.ndarray,
    timestamps: np.ndarray,
    phases: np.ndarray,
    phases_all: np.ndarray,
    feature_names: list[str],
    targets: list[str],
    tier: str,
    status: str,
) -> list[dict[str, object]]:
    groups = channel_groups(feature_names)
    unique_groups = list(dict.fromkeys(groups))
    result = []
    for event_index, (start, end) in enumerate(contiguous_events(y)):
        phase = str(phases_all[start]) or str(phases[start])
        eligible = "recon" not in (phase + " " + str(phases[start])).lower()
        alarms = np.flatnonzero(scores[start:end] > threshold)
        alarm_index = int(start + alarms[0]) if alarms.size else None
        ranked_groups: list[str] = []
        group_scores: dict[str, float] = {}
        if alarm_index is not None:
            per_channel = channel_scores[alarm_index]
            group_scores = {
                group: float(np.max(per_channel[np.asarray(groups) == group]))
                for group in unique_groups
            }
            ranked_groups = sorted(group_scores, key=lambda g: (-group_scores[g], g))
        valid_targets = sorted(set(targets) & set(unique_groups)) if eligible else []
        result.append({
            "event_index": event_index,
            "start_utc": str(timestamps[start]),
            "end_utc": str(timestamps[end - 1]),
            "phase_at_start": str(phases[start]),
            "phase_all_at_start": phase,
            "eligible": eligible,
            "evidence_tier": tier if eligible else "phase_only_reconnaissance",
            "target_status": status if eligible else "not_applicable",
            "target_groups": valid_targets,
            "detected": alarm_index is not None,
            "first_alarm_utc": str(timestamps[alarm_index]) if alarm_index is not None else None,
            "delay_seconds": float((timestamps[alarm_index] - timestamps[start]) / np.timedelta64(1, "s")) if alarm_index is not None else None,
            "group_scores": group_scores,
            "ranked_groups": ranked_groups,
            "target_hit_at_1": bool(alarm_index is not None and valid_targets and set(ranked_groups[:1]) & set(valid_targets)),
            "target_hit_at_3": bool(alarm_index is not None and valid_targets and set(ranked_groups[:3]) & set(valid_targets)),
            "target_hit_at_5": bool(alarm_index is not None and valid_targets and set(ranked_groups[:5]) & set(valid_targets)),
            "random_hit_probability_at_1": 1.0 - math.comb(len(unique_groups) - len(valid_targets), 1) / math.comb(len(unique_groups), 1) if alarm_index is not None and valid_targets else 0.0,
            "random_hit_probability_at_3": 1.0 - math.comb(len(unique_groups) - len(valid_targets), min(3, len(unique_groups))) / math.comb(len(unique_groups), min(3, len(unique_groups))) if alarm_index is not None and valid_targets else 0.0,
            "random_hit_probability_at_5": 1.0 - math.comb(len(unique_groups) - len(valid_targets), min(5, len(unique_groups))) / math.comb(len(unique_groups), min(5, len(unique_groups))) if alarm_index is not None and valid_targets else 0.0,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Family-held-out constrained GeCo state baseline for Data4Cyber v3")
    parser.add_argument("--config", type=Path, default=Path("configs/data4cyber_v3.json"))
    parser.add_argument("--output", type=Path, default=Path("results/summaries/data4cyber_geco_ar1_reference.json"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--baseline-results", type=Path, default=Path("results/summaries/data4cyber_loco_baselines.json"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    repo_root = args.config.resolve().parent.parent
    data_dir = args.data_dir if args.data_dir is not None else Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    audit_path = Path(config.get("metadata_audit_report", "results/data4cyber_metadata_audit.json"))
    if not audit_path.is_absolute():
        audit_path = repo_root / audit_path
    metadata_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    families: dict[str, list[str]] = config["attack_family_folds"]
    folders = sorted({folder for group in families.values() for folder in group} | {"S0_benign_baseline"})
    integrity = verify_split_inputs(data_dir, folders)
    target_path = Path(config["response_target_annotations"])
    if not target_path.is_absolute():
        target_path = repo_root / target_path
    target_annotations = pd.read_csv(target_path)
    normal_splits = {"S0_benign_baseline": read_benign(data_dir / config["normal_only_baseline"])}
    attack_splits = {}
    for folder in folders:
        if folder == "S0_benign_baseline":
            continue
        attack_splits[folder] = load_data4cyber_scenario(data_dir / folder)
        normal_splits[folder] = attack_splits[folder][1]

    targets_fpr = [float(x) for x in config["false_positive_rate_targets"]]
    fraction = float(config["train_fraction_within_normal_scenarios"])
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for heldout_family, heldout_folders in families.items():
        training_folders = [
            folder for folder in normal_splits
            if folder == "S0_benign_baseline" or folder not in heldout_folders
        ]
        fit_frames, cal_frames = [], []
        fit_times, cal_times = [], []
        for folder in training_folders:
            fit, cal, fit_ts, cal_ts = split_normal_rows(normal_splits[folder], fraction)
            fit_frames.append(fit)
            cal_frames.append(cal)
            fit_times.append(fit_ts)
            cal_times.append(cal_ts)
        imputer = fit_normal_median_imputer(pd.concat(fit_frames, ignore_index=True))
        fit_sequences = [imputer.transform(frame) for frame in fit_frames]
        fit_started = time.perf_counter()
        model = GeCoAR1Reference(fit_fraction=0.8).fit(fit_sequences)
        fit_seconds = time.perf_counter() - fit_started
        cal_sequences = [imputer.transform(frame) for frame in cal_frames]
        calibration_scores = np.concatenate([model.score_sequence(x)[1] for x in cal_sequences])
        thresholds = {fpr: threshold_at_fpr(calibration_scores, fpr) for fpr in targets_fpr}

        for folder in heldout_folders:
            attack, benign = attack_splits[folder]
            test_frame, y, timestamps, phases, phases_all = combine_test(attack, benign)
            x_test = imputer.transform(test_frame)
            scoring_started = time.perf_counter()
            channel_scores, scores, per_row_latency_ms = model.score_sequence(x_test)
            scoring_seconds = time.perf_counter() - scoring_started
            parameter_bytes = sum(
                value.nbytes for value in (
                    model.coefficients_, model.intercepts_, model.drift_,
                    model.training_cusum_limit_,
                )
            )
            rows = target_annotations[target_annotations["scenario_folder"] == folder]
            if rows.empty:
                raise ValueError(f"No frozen target annotation for {folder}")
            target_groups = sorted(rows["target_group"].astype(str).unique())
            tier = str(rows["evidence_tier"].iloc[0])
            status = str(rows["target_status"].iloc[0])
            for fpr, threshold in thresholds.items():
                point = binary_metrics(y, scores, threshold)
                events = event_metrics(y, scores > threshold, timestamps)
                explanation_events = first_alarm_events(
                    y, scores, threshold, channel_scores, timestamps, phases,
                    phases_all, imputer.feature_names, target_groups, tier, status,
                )
                records.append({
                    "heldout_family": heldout_family,
                    "heldout_scenario": folder,
                    "training_scenarios": training_folders,
                    "model": model.name,
                    "normal_fpr_target": fpr,
                    "calibration_observed_fpr": float(np.mean(calibration_scores > threshold)),
                    "threshold": threshold,
                    "n_features": len(imputer.feature_names),
                    "train_transitions": int(sum(max(0, len(x) - 1) for x in fit_sequences)),
                    "test_rows": int(len(y)),
                    "attack_rows": int(y.sum()),
                    "point_metrics": point,
                    "event_metrics": events,
                    "explanation_events": explanation_events,
                    "fit_seconds": fit_seconds,
                    "score_seconds": scoring_seconds,
                    "batch_amortized_score_us_per_row": scoring_seconds * 1e6 / max(1, len(y)),
                    "single_row_score_latency_p50_ms": float(np.percentile(per_row_latency_ms, 50)) if len(per_row_latency_ms) else 0.0,
                    "single_row_score_latency_p95_ms": float(np.percentile(per_row_latency_ms, 95)) if len(per_row_latency_ms) else 0.0,
                    "model_parameter_bytes": int(parameter_bytes),
                    "equation_search": "one-step self-lag linear SUM only; no cross-sensor inputs",
                })

    summary = []
    for fpr in targets_fpr:
        for family in families:
            rows = [r for r in records if r["heldout_family"] == family and r["normal_fpr_target"] == fpr]
            if not rows:
                continue
            eligible = [e for r in rows for e in r["explanation_events"] if e["eligible"]]
            result = {
                "heldout_family": family,
                "normal_fpr_target": fpr,
                "scenarios": [r["heldout_scenario"] for r in rows],
                "family_mean_average_precision": float(np.mean([r["point_metrics"]["average_precision"] for r in rows])),
                "family_mean_f1": float(np.mean([r["point_metrics"]["f1"] for r in rows])),
                "family_mean_recall": float(np.mean([r["point_metrics"]["recall"] for r in rows])),
                "family_mean_test_benign_fpr": float(np.mean([r["point_metrics"]["false_positive_rate"] for r in rows])),
                "eligible_target_events": len(eligible),
                "detected_target_events": sum(e["detected"] for e in eligible),
                "unconditional_target_hits_at_1": float(np.mean([e["target_hit_at_1"] for e in eligible])) if eligible else None,
                "unconditional_target_hits_at_3": float(np.mean([e["target_hit_at_3"] for e in eligible])) if eligible else None,
                "unconditional_target_hits_at_5": float(np.mean([e["target_hit_at_5"] for e in eligible])) if eligible else None,
            }
            summary.append(result)

    baseline_path = args.baseline_results
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path
    paired_comparisons = []
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison_specs = [("average_precision", targets_fpr[0])]
        comparison_specs.extend((metric, fpr) for fpr in targets_fpr
                                for metric in ("f1", "recall", "false_positive_rate"))
        for reference_model in ("PCA-95", "RobustZ"):
            for metric, fpr in comparison_specs:
                family_differences = []
                family_rows = []
                for family in families:
                    ours = [r for r in records if r["heldout_family"] == family
                            and r["normal_fpr_target"] == fpr]
                    other = [r for r in baseline["heldout_family_results"]
                             if r["heldout_family"] == family and r["model"] == reference_model
                             and r["normal_fpr_target"] == fpr]
                    if not ours or not other:
                        continue
                    gecko_value = float(np.mean([r["point_metrics"][metric] for r in ours]))
                    reference_value = float(np.mean([r["point_metrics"][metric] for r in other]))
                    difference = gecko_value - reference_value
                    family_differences.append(difference)
                    family_rows.append({"family": family, "geco_minus_reference": difference,
                                        "geco": gecko_value, "reference": reference_value})
                paired_comparisons.append({
                    "reference_model": reference_model,
                    "metric": metric,
                    "normal_fpr_target": fpr,
                    "n_families": len(family_differences),
                    "mean_paired_difference": float(np.mean(family_differences)) if family_differences else None,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_p(family_differences),
                    "family_differences": family_rows,
                })
        adjusted = holm([r["exact_two_sided_sign_flip_p"] for r in paired_comparisons])
        for row, p_adjusted in zip(paired_comparisons, adjusted):
            row["holm_adjusted_p_within_14_comparisons"] = p_adjusted

    output = {
        "study": "Data4Cyber v3 constrained GeCo state-stream reference",
        "status": "exploratory adapted reference; not reproduction of the companion paper's twelve-detector results",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "repository": "https://github.com/ipal-ids/ipal_ids_framework",
            "commit": "0d71866b777deb13bd2ee038d5058c2425ec4568",
            "method_source": "ids/geco/geco.py",
            "adaptations": [
                "restrict equation search to each channel's own one-step lag (upstream max_formel_length=0)",
                "replace upstream fork-only worker pool with a serial NumPy implementation of its length-zero SUM case",
                "median-impute using training-normal data because upstream GeCo indexes missing state keys",
                "calibrate a global threshold on normal-only calibration sequences for each nominal FPR target",
            ],
        },
        "runtime": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                    "numpy": np.__version__, "pandas": pd.__version__},
        "config": config,
        "dataset_integrity": integrity,
        "archive_aggregate": metadata_audit.get("archive_aggregate") if metadata_audit else None,
        "heldout_family_results": records,
        "family_summary": summary,
        "paired_family_comparisons_against_existing_pilots": paired_comparisons,
        "wall_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a constrained GeCo-style self-lag model, not full cross-sensor GeCo or an exact reproduction of published IPAL baselines.",
            "Six attack-family folds are from one testbed; all family-level inference is exploratory.",
            "Target evidence coverage and resolution remain uneven and sparse.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} family/scenario/threshold records to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

